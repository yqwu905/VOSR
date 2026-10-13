"""Model-free cleaning probes for text HR candidates, and the routing they drive.

Every probe works on the stored pixels at native resolution: resizing before
measuring would hide exactly the upscaling, blur and compression it looks for.
Large images are measured on a centre crop. Thresholds live in the routing
rules; the probes only measure.
"""

import io
import math

import cv2
import numpy as np
from PIL import Image, JpegImagePlugin


# ITU-T T.81 Annex K tables in natural (row-major) order, as Pillow exposes them.
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
WATERMARK_WORDS = ("shutterstock", "alamy", "gettyimages", "getty images", "istock", "dreamstime",
                   "123rf", "depositphotos", "adobe stock", "vcg", "视觉中国", "图虫", "千图网", "摄图网",
                   "包图网", "昵图网", "水印")
MAX_PROBE_EDGE = 2048


def ijg_table(quality):
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((IJG_LUMA * scale + 50) / 100), 1, 255)


def jpeg_quality(table):
    """IJG-equivalent quality of a luminance table and the fit residual.

    Camera and Photoshop tables are not IJG-scaled; a large residual says the
    number is only an equivalent, not the encoder setting.
    """
    table = np.asarray(table, dtype=float)
    errors = [np.abs(ijg_table(q) - table).mean() for q in range(1, 101)]
    best = int(np.argmin(errors))
    return best + 1, float(errors[best])


def probe_container(data):
    """Container facts that pixel decoding silently discards."""
    with Image.open(io.BytesIO(data)) as image:
        result = {"format": image.format, "mode": image.mode, "size": list(image.size),
                  "frames": getattr(image, "n_frames", 1), "file_bytes": len(data),
                  "has_icc": bool(image.info.get("icc_profile")),
                  "exif_orientation": int(image.getexif().get(0x0112, 1))}
        if image.format == "JPEG":
            quality, residual = jpeg_quality(image.quantization[0])
            result.update(jpeg_quality=quality, jpeg_table_residual=residual,
                          jpeg_mean_luma_q=float(np.mean(image.quantization[0])),
                          jpeg_subsampling=SUBSAMPLING.get(JpegImagePlugin.get_sampling(image), "n/a"))
        if image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = np.asarray(image.convert("RGBA"))
            alpha = rgba[..., 3]
            transparent = alpha < 250
            result["alpha_fraction"] = float(transparent.mean())
            if transparent.any():
                # convert("RGB") keeps the colour under alpha=0; usually black, so dark
                # text on a transparent canvas becomes invisible after decoding.
                hidden = rgba[..., :3][alpha == 0]
                result["alpha_hidden_luma_std"] = float(hidden.std()) if hidden.size else 0.0
        result["bit_depth"] = 16 if image.mode in {"I;16", "I;16B", "I", "F"} else 8
    return result


def to_gray(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def center_crop(array, edge=MAX_PROBE_EDGE):
    h, w = array.shape[:2]
    y, x = max(0, (h - edge) // 2), max(0, (w - edge) // 2)
    return array[y:y + edge, x:x + edge]


def padding_fraction(gray, tolerance=2.0):
    """Area of constant-colour borders (letterboxing, square padding)."""
    h, w = gray.shape
    edges = {}
    for side, lines in (("top", gray), ("bottom", gray[::-1]), ("left", gray.T), ("right", gray.T[:, ::-1])):
        reference = np.median(lines[0])
        flat = (np.abs(lines - reference).max(axis=1) <= tolerance)
        edges[side] = int(np.argmin(flat)) if not flat.all() else len(flat)
    rows = min(h, edges["top"] + edges["bottom"])
    cols = min(w, edges["left"] + edges["right"])
    return float(1 - (h - rows) * (w - cols) / (h * w)), edges


def dct_matrix(n=8):
    k = np.arange(n)
    matrix = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    matrix[0] /= np.sqrt(2)
    return matrix


DCT8 = dct_matrix()
# Low-frequency AC positions (natural order) whose quantizers identify the table;
# their IJG steps lie within a factor 1.6 of each other at any quality.
HISTORY_POSITIONS = (1, 8, 9, 2, 16, 10, 17, 3, 24)


def lattice_scores(values, largest=48, minimum=32):
    """Per candidate step: how tightly the nonzero coefficients sit on its multiples."""
    steps = np.arange(2, largest + 1, dtype=np.float32)
    ratio = values[:, None] / steps[None, :]
    counted = np.abs(ratio) >= 0.75
    counts = counted.sum(axis=0)
    score = (np.cos(2 * np.pi * (ratio - np.round(ratio))) * counted).sum(axis=0) / np.maximum(counts, 1)
    return steps, np.where(counts >= minimum, score, 0.0)


def quantizer_peak(values, threshold=0.11):
    """(step, peak contrast) of the lattice score, or (None, contrast).

    A requantized coefficient scores high at its table step and low at the
    neighbouring steps; unquantized content scores smoothly across steps.
    Clipped or noisy blocks lower the whole curve but do not create a peak.
    Multiples of the step peak too, so the largest near-best step is taken.
    """
    steps, score = lattice_scores(values)
    padded = np.concatenate([[0.0], score, [0.0]])
    contrast = score - np.maximum(padded[:-2], padded[2:])
    best = float(contrast.max())
    if best < threshold:
        return None, best
    return int(steps[np.flatnonzero(contrast >= 0.8 * best)[-1]]), best


def grid_coefficients(y, dy, dx, max_blocks):
    h, w = ((y.shape[0] - dy) // 8) * 8, ((y.shape[1] - dx) // 8) * 8
    if h < 64 or w < 64:
        return None
    blocks = y[dy:dy + h, dx:dx + w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    blocks = blocks[blocks.reshape(len(blocks), 64).std(axis=1) > 1]  # flat background has no evidence
    if len(blocks) < 48:
        return None
    blocks = blocks[:: max(1, len(blocks) // max_blocks)]
    return (DCT8 @ blocks @ DCT8.T).reshape(len(blocks), 64)


def busiest_window(gray, size):
    """Top-left corner of the size x size window with the most gradient energy."""
    h, w = gray.shape
    if h <= size and w <= size:
        return 0, 0
    cell = 32
    energy = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))[:h // cell * cell, :w // cell * cell]
    energy = energy.reshape(h // cell, cell, -1, cell).sum(axis=(1, 3))
    span_y, span_x = min(size, h) // cell, min(size, w) // cell
    if not span_y or not span_x:
        return 0, 0
    sums = cv2.boxFilter(energy, -1, (span_x, span_y), normalize=False, anchor=(0, 0),
                         borderType=cv2.BORDER_CONSTANT)
    sums = sums[:energy.shape[0] - span_y + 1, :energy.shape[1] - span_x + 1]
    cy, cx = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(cy * cell), int(cx * cell)


def jpeg_history(rgb, max_blocks=4096, required=4, max_step_ratio=2.0):
    """Detect and grade past JPEG coding from decoded pixels (e.g. PNG re-encodes).

    Requantized DCT coefficients stay on multiples of the table steps, which
    clean renders and photos do not. The grid offset is searched over all 64
    phases on a subsample, so crops are found too; resampling after compression
    destroys the evidence. Returns the IJG-equivalent quality, or None when
    fewer than ``required`` low-frequency positions show a quantizer peak or
    their steps disagree: repeated identical glyphs in renders can make a few
    chance peaks, but not a consistent low-frequency table.
    """
    y = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32) - 128
    # Bilevel and palette images (fax scans, GIFs) have discrete coefficients
    # that mimic a lattice, and a decoded JPEG never has so few levels.
    if len(np.unique(np.round(y[::4, ::4]))) < 32:
        return None

    def evidence(luma, dy, dx, blocks, positions=HISTORY_POSITIONS):
        coefficients = grid_coefficients(luma, dy, dx, blocks)
        if coefficients is None:
            return {}
        return {p: quantizer_peak(coefficients[:, p]) for p in positions}

    def strength(found):
        return float(np.mean([contrast for _, contrast in found.values()])) if found else 0.0

    best = (0, 0)
    if sum(step is not None for step, _ in evidence(y, 0, 0, 1024).values()) < required:
        # Cropped JPEGs: rank the 64 phases on the most textured window.
        y0, x0 = busiest_window(y, 512)
        window = y[y0:y0 + 512, x0:x0 + 512]
        phase = max(((dy, dx) for dy in range(8) for dx in range(8)),
                    key=lambda d: strength(evidence(window, *d, 512, HISTORY_POSITIONS[:2])))
        best = ((y0 + phase[0]) % 8, (x0 + phase[1]) % 8)
    found = {p: step for p, (step, _) in evidence(y, *best, max_blocks).items() if step is not None}
    if len(found) < required or max(found.values()) > max_step_ratio * min(found.values()):
        return None
    positions = list(found)
    observed = np.array([found[p] for p in positions], dtype=float)
    errors = [np.abs(ijg_table(q)[positions] - observed).mean() for q in range(1, 101)]
    return {"quality": int(np.argmin(errors)) + 1, "grid_offset": list(best),
            "steps": {str(p): int(q) for p, q in found.items()}}


def edge_width(gray, window=9, mask=None):
    """Median step width at the strongest edges, in pixels (contrast / slope).

    A step blurred by a Gaussian of sigma s measures about 2.5 s; anti-aliased
    renders and sharp photos measure 1-2 px. Unlike global sharpness scores it
    does not depend on how much of the image is flat background. With a mask
    the steepest 1% inside it are used (10% for small regions).
    """
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    kernel = np.ones((window, window), np.uint8)
    contrast = cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)
    candidates = magnitude if mask is None else magnitude[mask]
    if candidates.size < 32:
        return None
    strong = magnitude >= max(np.percentile(candidates, 99 if candidates.size >= 3200 else 90), 4.0)
    if mask is not None:
        strong &= mask
    if strong.sum() < 32:
        return None
    return float(np.median(contrast[strong] / magnitude[strong]))


def noise_sigma(gray, mask=None, percentile=10):
    """Immerkaer 1996 noise estimate on the flattest pixels, in 0..255 units.

    Flatness is judged on a smoothed copy so the noise itself does not bias the
    selection, and only the flattest 10% are used: at 50% foliage, fur and
    engraving hatching read as sigma 3-14 on clean photos, at 10% the median
    clean photo reads 0.9 while added noise is still measured exactly.
    """
    mask_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = np.abs(cv2.filter2D(gray, -1, mask_kernel))[1:-1, 1:-1]
    smooth = cv2.GaussianBlur(gray, (0, 0), 2)
    gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    inside = np.ones_like(gradient, bool) if mask is None else mask[1:-1, 1:-1]
    if inside.sum() < 640:
        return None
    flat = inside & (gradient <= np.percentile(gradient[inside], percentile))
    return float(math.sqrt(math.pi / 2) * response[flat].mean() / 6)


def resample_residual(gray, scale):
    """Fraction of AC energy lost by an area-down, cubic-up round trip.

    Content truly band-limited to 1/scale (an upscaled image) survives the round
    trip, so the fraction collapses towards 0.
    """
    h, w = gray.shape
    m = max(4, math.ceil(2 * scale))  # resampling borders are not content
    if min(h, w) < 2 * m + 4 * scale:
        return None  # too small to measure at this factor (e.g. word crops)
    small = cv2.resize(gray, (max(1, round(w / scale)), max(1, round(h / scale))), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    centre = gray[m:-m, m:-m]
    energy = ((centre - centre.mean()) ** 2).mean()
    return float(((centre - back[m:-m, m:-m]) ** 2).mean() / max(energy, 1e-6))


def effective_upscale(gray, ladder=(1.5, 2, 3, 4), ratio=0.12, cap=0.05):
    """Largest factor at which the image looks like a past upscale.

    The loss at ``scale`` is compared with the loss at twice that factor, so a
    smooth photo is not mistaken for an upscale merely for having little detail.
    """
    residuals = {f"{s:g}": resample_residual(gray, s) for s in sorted({*ladder, *(2 * x for x in ladder)})}
    result = 1.0
    for scale in ladder:
        lost, reference = residuals[f"{scale:g}"], residuals[f"{2 * scale:g}"]
        if lost is not None and reference is not None and lost <= cap and lost <= ratio * reference:
            result = float(scale)
    return result, residuals


def periodic_peak(gray, size=512):
    """Share of off-axis high-frequency energy in its strongest peak.

    Halftone screens and screen moire concentrate energy in a few off-axis
    peaks; text and photos spread it. Text lines and page grids put their
    harmonics on the axes, which are excluded.
    """
    tile = center_crop(gray, size)
    if min(tile.shape) < 64:
        return 0.0
    tile = tile - cv2.GaussianBlur(tile, (0, 0), 4)
    window = np.outer(np.hanning(tile.shape[0]), np.hanning(tile.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(tile * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(tile.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(tile.shape[1]))[None, :]
    band = (np.hypot(fx, fy) > 0.08) & (np.abs(fx) > 0.02) & (np.abs(fy) > 0.02)
    total = power[band].sum()
    if total <= 0:
        return 0.0
    peaks = cv2.boxFilter(power, -1, (3, 3), normalize=False)
    return float(peaks[band].max() / total)


def chroma_stats(rgb):
    signed = rgb.astype(np.int16)
    spread = np.abs(signed[..., 0] - signed[..., 1]) + np.abs(signed[..., 1] - signed[..., 2])
    return float(spread.mean())


def phash(rgb):
    """64-bit DCT perceptual hash as 16 hex digits."""
    small = cv2.resize(to_gray(rgb), (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def text_geometry(items, size):
    """Line height = short side of each region's minimum-area rectangle."""
    if not items:
        return {"text_regions": 0}
    heights, area = [], 0.0
    for item in items:
        points = np.asarray(item["polygon"], dtype=np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(points)
        heights.append(min(w, h))
        area += abs(cv2.contourArea(points))
    heights = np.asarray(heights)
    return {"text_regions": len(items), "text_height_min": float(heights.min()),
            "text_height_p10": float(np.percentile(heights, 10)),
            "text_height_median": float(np.median(heights)),
            "text_coverage": float(min(1.0, area / (size[0] * size[1])))}


def text_mask(items, shape, margin=4):
    mask = np.zeros(shape, np.uint8)
    for item in items:
        points = np.round(np.asarray(item["polygon"], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)


def watermark_hits(texts):
    found = set()
    for text in texts:
        lowered = (text or "").lower()
        found.update(word for word in WATERMARK_WORDS if word in lowered)
    return sorted(found)


def probe_pixels(rgb):
    rgb = center_crop(np.asarray(rgb))
    history = jpeg_history(rgb)  # before trimming, so the grid offset is the stored one
    padding, padding_edges = padding_fraction(to_gray(rgb))
    h, w = rgb.shape[:2]
    # Measure the content, not letterbox bars or flat page margins.
    inner = rgb[padding_edges["top"]:h - padding_edges["bottom"], padding_edges["left"]:w - padding_edges["right"]]
    if min(inner.shape[:2]) >= 32:
        rgb = inner
    gray = to_gray(rgb)
    low, high = np.percentile(gray, [1, 99])
    scale, residuals = effective_upscale(gray)
    sample = rgb.reshape(-1, 3)[:: max(1, rgb.shape[0] * rgb.shape[1] // 200000)]
    unique = len(np.unique(sample.view(np.dtype((np.void, 3)))))
    return {"probe_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "padding_fraction": padding, "padding_edges": padding_edges,
            "luma_std": float(gray.std()), "luma_range": float(high - low),
            "chroma_mean": chroma_stats(rgb), "unique_colors_sampled": int(unique),
            "jpeg_history": history, "edge_width": edge_width(gray), "noise_sigma": noise_sigma(gray),
            "effective_upscale": scale, "resample_residual": residuals,
            "periodic_peak": periodic_peak(gray)}


def probe_image(data, annotations=None):
    """All probes for one encoded image; ``annotations`` are canonical regions."""
    container = probe_container(data)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"))
    result = {**container, **probe_pixels(rgb), "phash": phash(rgb)}
    result.update(text_geometry(annotations or [], container["size"]))
    if annotations:
        # Text sharpness is what the HR has to teach; background bokeh is fine.
        gray, mask = to_gray(rgb), text_mask(annotations, rgb.shape[:2])
        result["text_edge_width"] = edge_width(gray, mask=mask)
        result["text_noise_sigma"] = noise_sigma(gray, mask=mask)
    result["watermark_words"] = watermark_hits(x.get("text") for x in annotations or [])
    return result


DEFAULT_RULES = {
    "min_short_edge": 256, "max_aspect": 8.0, "min_luma_range": 24.0, "max_padding_fraction": 0.3,
    "require_text": True, "min_text_height": 16.0, "max_downscale": 4.0,
    "max_edge_width": 2.2, "edge_width_per_downscale": 1.6, "max_noise_sigma": 1.5,
    "max_periodic_peak": 0.03, "near_duplicate_bits": 6,
    # Downscale that brings JPEG residuals on text below ~40 dB PSNR, by quality floor.
    "jpeg_downscale": [[90, 1.0], [75, 1.5], [50, 2.0], [40, 3.0]],
}
LADDER = (1.0, 1.5, 2.0, 3.0, 4.0)


def jpeg_quality_of(stats):
    """The worst of container and pixel-recovered quality.

    A q100 re-save keeps the lattice of an earlier, harsher compression: 15 of
    24 sampled SROIE receipts carry q62-q86 history under a higher container quality.
    """
    container, history = stats.get("jpeg_quality"), (stats.get("jpeg_history") or {}).get("quality")
    if container is not None and history is not None and history >= container - 3:
        return container  # the same compression; the pixel estimate is good to +-1
    return min((q for q in (container, history) if q is not None), default=None)


def route(stats, rules=None):
    """Decide direct HR / clean by downscaling / teacher pseudo-GT / reject.

    Area downscaling by k averages away noise, blur, past upscaling and JPEG
    residue at the price of k-times smaller text, so it is chosen only while
    the median text line keeps ``min_text_height`` pixels. Otherwise the
    generative teacher is the remaining option, and the six-stage pipeline
    verifies its glyphs. Factors come from calibration (docs/text_data_strategy.md).
    """
    rules = {**DEFAULT_RULES, **(rules or {})}
    reject, fixes, causes = [], [], {}
    w, h = stats["size"]
    if stats.get("frames", 1) != 1:
        reject.append("multi_frame")
    if min(w, h) < rules["min_short_edge"]:
        reject.append("too_small")
    if max(w, h) / min(w, h) > rules["max_aspect"]:
        reject.append("extreme_aspect")
    if stats["luma_range"] < rules["min_luma_range"]:
        reject.append("blank_or_low_contrast")
    if stats.get("watermark_words"):
        reject.append("watermark")
    height = stats.get("text_height_median")
    if rules["require_text"] and not stats.get("text_regions"):
        reject.append("no_text")
    elif height is not None and height < rules["min_text_height"]:
        reject.append("text_too_small")
    if stats.get("alpha_fraction", 0) > 0:
        fixes.append("composite_alpha")
    if stats.get("exif_orientation", 1) != 1:
        fixes.append("exif_orientation")
    if stats["padding_fraction"] > rules["max_padding_fraction"]:
        fixes.append("trim_padding")
    if stats.get("bit_depth", 8) != 8:
        fixes.append("bit_depth")
    # Each degradation implies the smallest downscale that removes it.
    quality = jpeg_quality_of(stats)
    if quality is not None:
        causes["jpeg"] = next((k for floor, k in rules["jpeg_downscale"] if quality >= floor), math.inf)
    if stats["effective_upscale"] > 1:
        causes["upscaled"] = stats["effective_upscale"]
    width = stats.get("text_edge_width") or stats.get("edge_width")
    if width is not None and width > rules["max_edge_width"]:
        causes["blur"] = width / rules["edge_width_per_downscale"]
    noise = stats.get("text_noise_sigma")
    noise = stats["noise_sigma"] if noise is None else noise
    if noise is not None and noise > rules["max_noise_sigma"]:
        causes["noise"] = noise / rules["max_noise_sigma"]
    if stats["periodic_peak"] > rules["max_periodic_peak"]:
        causes["periodic_pattern"] = 2.0
    causes = {key: value for key, value in causes.items() if value > 1}
    needed = max(causes.values(), default=1.0)
    k = next((x for x in LADDER if x >= needed - 1e-6), math.inf)
    if reject:
        decision = "reject"
    elif k == 1.0:
        decision = "direct"
    elif k <= rules["max_downscale"] and (height is None or height / k >= rules["min_text_height"]):
        decision = "downscale"
    else:
        decision = "teacher"
    # null marks a degradation no downscale removes (JSON has no infinity).
    causes = {key: (value if math.isfinite(value) else None) for key, value in causes.items()}
    return {"decision": decision, "downscale": k if decision == "downscale" else 1.0,
            "reasons": reject, "fixes": fixes, "degradations": causes}
