"""Text-aware LR synthesis: scenario profiles over pixel-aligned operators.

Generic Real-ESRGAN chains model a camera photo of a natural scene. Text
images also arrive as resized screenshots, recompressed web reposts, printed
and scanned pages, and phone photos of screens, whose dominant artifacts
(nearest-neighbour jaggies, 4:2:0 colour bleeding around coloured strokes,
halftone rosettes, ink spread, screen moire, demosaicing zippers, denoise
smearing and sharpening halos) a natural-image chain never produces.

Every operator keeps the LR pixel grid aligned with the HR: there is no
geometric warp after the HR is fixed, and the final size is exactly
round(HR / scale). Parameters are drawn from ``[mild, severe]`` ranges at a
per-sample severity, so one seed reproduces one LR and its full record.
"""

import io
import math

import cv2
import numpy as np
from PIL import Image


INTERPOLATION = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
                 "nearest": cv2.INTER_NEAREST, "lanczos": cv2.INTER_LANCZOS4}
SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


class Sampler:
    """Draws parameters around one severity; a range is written [mild, severe]."""

    def __init__(self, rng, severity):
        self.rng = rng
        self.severity = severity

    def level(self):
        if self.severity is None:
            return float(self.rng.uniform())
        return float(np.clip(self.severity + self.rng.uniform(-0.2, 0.2), 0, 1))

    def value(self, bounds, log=False):
        mild, severe = bounds
        t = self.level()
        if log:
            return float(math.exp(math.log(mild) + t * (math.log(severe) - math.log(mild))))
        return float(mild + t * (severe - mild))

    def integer(self, bounds):
        return int(round(self.value(bounds)))

    def choice(self, options):
        return options[int(self.rng.integers(len(options)))]

    def chance(self, probability):
        return bool(self.rng.uniform() < probability)


# ---------------------------------------------------------------- kernels


def gaussian_kernel(sigma_x, sigma_y, theta, beta=2.0, size=None):
    """(Generalized) anisotropic Gaussian; beta=2 is Gaussian, <2 heavy-tailed, >2 plateau."""
    size = size or 2 * math.ceil(3 * max(sigma_x, sigma_y)) + 1
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    c, s = math.cos(theta), math.sin(theta)
    u, v = c * x + s * y, -s * x + c * y
    kernel = np.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2) ** (beta / 2))
    return (kernel / kernel.sum()).astype(np.float32)


def disk_kernel(radius, supersample=8):
    """Anti-aliased defocus disc."""
    size = 2 * math.ceil(radius) + 1
    fine = np.mgrid[0:size * supersample, 0:size * supersample].astype(np.float64)
    center = (size * supersample - 1) / 2
    inside = np.hypot(fine[0] - center, fine[1] - center) <= radius * supersample
    kernel = inside.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return (kernel / kernel.sum()).astype(np.float32)


def motion_kernel(length, angle, supersample=8):
    """Anti-aliased straight-line shake of ``length`` pixels."""
    size = 2 * math.ceil(length / 2) + 1
    canvas = np.zeros((size * supersample, size * supersample), np.float32)
    center = (size * supersample - 1) / 2
    dx, dy = math.cos(angle) * length * supersample / 2, math.sin(angle) * length * supersample / 2
    cv2.line(canvas, (round(center - dx), round(center - dy)), (round(center + dx), round(center + dy)),
             1.0, thickness=supersample)
    kernel = canvas.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return kernel / kernel.sum()


def sinc_kernel(cutoff, size=21):
    """Circular low-pass (jinc) kernel; ringing like over-sharpened resampling."""
    from scipy.special import j1
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    r = np.hypot(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = cutoff * j1(cutoff * r) / (2 * np.pi * r)
    kernel[radius, radius] = cutoff ** 2 / (4 * np.pi)
    return (kernel / kernel.sum()).astype(np.float32)


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- operators
# Each takes float32 RGB in [0, 1], a Sampler and options, and returns
# (image, record). The record holds every drawn value.


def op_blur(image, p, kinds=("gaussian", "aniso", "defocus", "motion"), sigma=(0.3, 2.0),
            sigma_scale=1.0):
    kind = p.choice(list(kinds))
    record = {"kind": kind}
    if kind == "gaussian":
        s = p.value(sigma) * sigma_scale
        kernel = gaussian_kernel(s, s, 0.0)
        record["sigma"] = s
    elif kind == "aniso":
        sx, sy = p.value(sigma) * sigma_scale, p.value(sigma) * sigma_scale * p.value((1.0, 0.4))
        theta = float(p.rng.uniform(0, math.pi))
        beta = p.choice([1.0, 2.0, 2.0, 4.0])
        kernel = gaussian_kernel(sx, max(0.2, sy), theta, beta)
        record.update(sigma_x=sx, sigma_y=max(0.2, sy), theta=theta, beta=beta)
    elif kind == "defocus":
        radius = p.value((0.6, 3.0)) * sigma_scale
        kernel = disk_kernel(radius)
        record["radius"] = radius
    elif kind == "motion":
        length, angle = p.value((1.5, 7.0)) * sigma_scale, float(p.rng.uniform(0, math.pi))
        kernel = motion_kernel(length, angle)
        record.update(length=length, angle=angle)
    else:
        cutoff = p.value((math.pi * 0.9, math.pi / 4))
        kernel = sinc_kernel(cutoff)
        record["cutoff"] = cutoff
    return convolve(image, kernel), record


def resize_to(image, size, mode):
    if mode == "pil_bicubic":  # Pillow widens the filter support when shrinking
        pixels = np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)
        resized = Image.fromarray(pixels).resize(size, Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / 255
    return cv2.resize(image, size, interpolation=INTERPOLATION[mode])


def op_resize(image, p, target, modes=("area", "bilinear", "bicubic"), factor=None):
    """``factor`` (relative random resize) or the exact LR ``target`` size."""
    mode = p.choice(list(modes))
    if factor is not None:
        f = p.value(factor)
        size = (max(8, round(image.shape[1] * f)), max(8, round(image.shape[0] * f)))
    else:
        size = target
    return resize_to(image, size, mode), {"mode": mode, "size": list(size)}


def op_gaussian_noise(image, p, sigma=(1.0, 15.0), gray_probability=0.4):
    s = p.value(sigma) / 255
    gray = p.chance(gray_probability)
    shape = image.shape[:2] + ((1,) if gray else (3,))
    return image + p.rng.normal(0, s, shape).astype(np.float32), {"sigma_255": s * 255, "gray": gray}


def op_sensor_noise(image, p, shot=(0.0005, 0.012), read=(0.5, 6.0), chroma=(0.1, 0.5)):
    """Poisson-Gaussian noise in linear light, as a sensor before the ISP gamma.

    ISPs suppress chroma noise far more than luma noise, so only a ``chroma``
    fraction of the noise is independent per channel.
    """
    a, b, c = p.value(shot, log=True), p.value(read) / 255, p.value(chroma)
    linear = np.clip(image, 0, 1) ** 2.2
    shared = p.rng.normal(0, 1, image.shape[:2] + (1,)).astype(np.float32)
    independent = p.rng.normal(0, 1, image.shape).astype(np.float32)
    unit = (shared * math.sqrt(1 - c ** 2) + independent * c)
    noisy = linear + unit * np.sqrt(a * linear + b ** 2)
    return np.clip(noisy, 0, 1) ** (1 / 2.2), {"shot": a, "read_255": b * 255, "chroma_fraction": c}


def op_jpeg(image, p, quality=(92, 35), subsampling=("4:2:0",)):
    q = p.integer(quality)
    mode = p.choice(list(subsampling))
    buffer = io.BytesIO()
    Image.fromarray(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)).save(
        buffer, format="JPEG", quality=q, subsampling=SUBSAMPLING[mode])
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result = np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255
    return result, {"quality": q, "subsampling": mode}


def op_webp(image, p, quality=(90, 30)):
    q = p.integer(quality)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)).save(buffer, format="WEBP", quality=q)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255, {"quality": q}


def op_sharpen(image, p, sigma=(0.6, 2.0), amount=(0.3, 2.0)):
    """Unsharp masking: the halos phone ISPs and resizers leave around strokes."""
    s, k = p.value(sigma), p.value(amount)
    blurred = cv2.GaussianBlur(image, (0, 0), s)
    return image + k * (image - blurred), {"sigma": s, "amount": k}


def op_denoise(image, p, strength=(0.04, 0.25)):
    """Edge-preserving smoothing of an ISP noise reducer: waxy flats, thinned strokes."""
    s = p.value(strength)
    clipped = np.clip(image, 0, 1).astype(np.float32)
    return cv2.bilateralFilter(clipped, 5, s, 3), {"sigma_color": s}


def op_demosaic(image, p, strength=(0.3, 1.0), chroma_sigma=(2.0, 1.0)):
    """RGGB mosaic, bilinear demosaicing and ISP false-colour suppression.

    On strokes one or two LR pixels wide, raw bilinear demosaicing alone turns
    an 8 px text line unreadable (OCR CER 0.05 -> 0.68); phone ISPs low-pass the
    chroma to suppress that false colour (-> 0.31) and demosaic better than
    bilinear, modelled by blending towards the input. OpenCV's edge-aware
    variant damaged text more (0.52 with suppression) and VNG shifts the image,
    so neither is used.
    """
    pixels = np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8)
    h, w = pixels.shape[:2]
    mosaic = np.empty((h, w), np.uint8)
    mosaic[0::2, 0::2] = pixels[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = pixels[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = pixels[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = pixels[1::2, 1::2, 2]
    # OpenCV names Bayer codes after the 2x2 block starting at the second row and
    # column; for an RGGB sensor that block is BG, as checked in the test suite.
    restored = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB)
    sigma, alpha = p.value(chroma_sigma), p.value(strength)
    ycc = cv2.cvtColor(restored, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycc[..., 1:] = cv2.GaussianBlur(ycc[..., 1:], (0, 0), sigma)
    restored = cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    result = alpha * (restored.astype(np.float32) / 255) + (1 - alpha) * image
    return result, {"pattern": "RGGB", "method": "bilinear", "chroma_sigma": sigma, "strength": alpha}


def op_quantize(image, p, colors=(256, 16), dither_probability=0.5):
    """Palette images (GIF/PNG-8) with optional Floyd-Steinberg dithering."""
    n = p.integer(colors)
    dither = p.chance(dither_probability)
    pixels = Image.fromarray(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8))
    quantized = pixels.quantize(colors=n, method=Image.Quantize.MEDIANCUT,
                                dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE)
    return np.asarray(quantized.convert("RGB"), dtype=np.float32) / 255, {"colors": n, "dither": dither}


def screen(shape, period, angle, phase):
    y, x = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    a = math.radians(angle)
    u = (x * math.cos(a) + y * math.sin(a)) / period + phase[0]
    v = (-x * math.sin(a) + y * math.cos(a)) / period + phase[1]
    return (np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v) + 1) / 2


def op_halftone(image, p, period=(6.0, 3.0), softness=(0.15, 0.05)):
    """Clustered-dot CMY screens at 15/75/0 degrees: print rosettes on scans."""
    pitch, soft = p.value(period), p.value(softness)
    result = np.empty_like(image)
    for channel, angle in enumerate((15, 75, 0)):
        ink = 1 - image[..., channel]
        threshold = screen(image.shape[:2], pitch, angle, p.rng.uniform(0, 1, 2))
        result[..., channel] = 1 - 1 / (1 + np.exp(-(ink - threshold) / soft))
    return result, {"period": pitch, "softness": soft, "angles": [15, 75, 0]}


def op_ink_bleed(image, p, radius=(1, 3), strength=(0.3, 1.0)):
    """Dark ink spreading into paper: strokes thicken and counters fill."""
    r, s = p.integer(radius), p.value(strength)
    spread = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    spread = cv2.GaussianBlur(spread, (0, 0), max(0.5, r / 2))
    return image * (1 - s) + np.minimum(image, spread) * s, {"radius": r, "strength": s}


def low_frequency_field(shape, rng, cells):
    field = rng.normal(0, 1, (cells, cells)).astype(np.float32)
    field = cv2.resize(field, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return field / max(np.abs(field).max(), 1e-6)


def op_paper(image, p, strength=(0.02, 0.12)):
    """Uneven paper tone and fibre texture, multiplicative."""
    s = p.value(strength)
    tone = 1 + s * low_frequency_field(image.shape, p.rng, 4)
    fibre = 1 + s * 0.3 * cv2.GaussianBlur(p.rng.normal(0, 1, image.shape[:2]).astype(np.float32), (0, 0), 1.2)
    return image * (tone * fibre)[..., None], {"strength": s}


def op_binarize(image, p, offset=(0.0, 0.12), speckle=(0.0, 0.004)):
    """Fax / 1-bit scan: Otsu threshold with drift, plus salt-and-pepper speckles."""
    gray = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2GRAY)
    otsu, _ = cv2.threshold((gray * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = otsu / 255 + p.value(offset) * (1 if p.chance(0.5) else -1)
    rate = p.value(speckle)
    binary = (gray > t).astype(np.float32)
    flips = p.rng.uniform(size=binary.shape) < rate
    binary[flips] = 1 - binary[flips]
    return np.repeat(binary[..., None], 3, axis=2), {"threshold": float(t), "speckle": rate}


def op_screen_grid(image, p, scale, contrast=(0.15, 0.6), period=(0.75, 1.3), angle=(0.5, 4.0)):
    """Display sub-pixel stripes photographed near the sensor's Nyquist: moire.

    The RGB stripe grid is laid at the HR resolution with a pitch close to one
    LR pixel, so the later downsampling aliases it into colour fringes as a
    camera does. Mean brightness is preserved.
    """
    pitch = scale * p.value(period)
    tilt = p.value(angle) * (1 if p.chance(0.5) else -1)
    c = p.value(contrast)
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(tilt)
    u = (x * math.cos(a) + y * math.sin(a)) / pitch
    rows = (-x * math.sin(a) + y * math.cos(a)) / pitch
    mask = np.empty_like(image)
    for channel in range(3):  # R, G, B stripes one third of a pixel apart
        stripe = 0.5 + 0.5 * np.cos(2 * np.pi * (u - channel / 3))
        mask[..., channel] = stripe * (0.75 + 0.25 * np.cos(2 * np.pi * rows))
    mask = 1 + c * (mask / mask.mean() - 1)
    return image * mask, {"pitch_hr": pitch, "tilt_degrees": tilt, "contrast": c}


def op_illumination(image, p, strength=(0.03, 0.2)):
    """Vignetting and uneven light; low frequency only.

    The mean gain is 1, as auto-exposure keeps the overall level: a pure
    darkening shifted ScreenSR-like LR by -4 grey levels against ~0 in the real pairs.
    """
    s = p.value(strength)
    h, w = image.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = p.rng.uniform(0.2, 0.8) * w, p.rng.uniform(0.2, 0.8) * h
    falloff = np.clip(1 - s * (((x - cx) / w) ** 2 + ((y - cy) / h) ** 2) * 2, 0.3, 1.2)
    return image * (falloff / falloff.mean())[..., None], {"strength": s, "center": [float(cx), float(cy)]}


def op_color_shift(image, p, gamma=(1.0, 1.15), contrast=(1.0, 0.88), white_balance=(0.0, 0.04)):
    """Tone and white-balance drift of a re-photographed display.

    Kept mild: paired data is colour-aligned and VOSR aligns colour at
    inference, so strong drift would teach colour correction instead of SR.
    """
    g = p.value(gamma) ** (1 if p.chance(0.5) else -1)
    k = p.value(contrast)
    wb = p.value(white_balance)
    gains = 1 + p.rng.uniform(-wb, wb, 3).astype(np.float32)
    result = np.clip(image, 0, 1) ** g
    result = (result - result.mean()) * k + result.mean()
    return result * gains, {"gamma": g, "contrast": k, "gains": gains.tolist()}


# ---------------------------------------------------------------- profiles
# A step is (operator, probability, options). "resize" without a factor
# lands on the exact LR size; operators listed after it run at LR resolution.

PROFILES = {
    "camera": [
        ("blur", 1.0, {"kinds": ("gaussian", "aniso", "defocus", "motion")}),
        ("resize", 1.0, {"modes": ("area", "bilinear", "bicubic")}),
        ("sensor_noise", 0.8, {}),
        ("demosaic", 0.3, {}),
        ("denoise", 0.4, {}),
        ("sharpen", 0.5, {}),
        ("jpeg", 1.0, {"quality": (92, 35), "subsampling": ("4:2:0",)}),
    ],
    "screenshot": [
        ("resize", 1.0, {"modes": ("area", "bilinear", "bicubic", "nearest", "lanczos", "pil_bicubic")}),
        ("quantize", 0.15, {}),
        ("jpeg", 0.75, {"quality": (90, 20), "subsampling": ("4:2:0", "4:2:0", "4:4:4")}),
        ("webp", 0.2, {}),
    ],
    "web_chain": [
        ("resize", 1.0, {"factor": (0.8, 0.4), "modes": ("area", "bilinear", "bicubic")}),
        ("jpeg", 1.0, {"quality": (90, 45), "subsampling": ("4:2:0",)}),
        ("resize", 1.0, {"modes": ("area", "bilinear", "bicubic")}),
        ("sharpen", 0.3, {"amount": (0.2, 1.0)}),
        ("jpeg", 1.0, {"quality": (88, 30), "subsampling": ("4:2:0",)}),
    ],
    "scan": [
        ("halftone", 0.35, {}),
        ("ink_bleed", 0.4, {}),
        ("paper", 0.5, {}),
        ("blur", 1.0, {"kinds": ("gaussian", "aniso"), "sigma": (0.4, 1.6)}),
        ("resize", 1.0, {"modes": ("area", "bilinear")}),
        ("gaussian_noise", 0.5, {"sigma": (0.5, 6.0), "gray_probability": 1.0}),
        ("binarize", 0.1, {}),
        ("jpeg", 0.7, {"quality": (95, 50), "subsampling": ("4:4:4", "4:2:0")}),
    ],
    "screen_photo": [
        # Rare on purpose: on ScreenSR's diagnosis pairs (0-39) dropping both moved
        # coverage from 5.8 to 3.6 SD; aligned paired data rarely shows moire or drift.
        ("color_shift", 0.2, {}),
        ("screen_grid", 0.3, {}),
        ("blur", 1.0, {"kinds": ("gaussian", "defocus", "aniso"), "sigma": (0.5, 2.0)}),
        ("illumination", 0.4, {}),
        # A sensor integrates over each pixel; point sampling would draw the grid itself.
        ("resize", 1.0, {"modes": ("area",)}),
        ("sensor_noise", 0.9, {}),
        ("denoise", 0.6, {}),
        ("sharpen", 0.5, {}),
        ("jpeg", 1.0, {"quality": (92, 50), "subsampling": ("4:2:0",)}),
    ],
    "realesrgan": [
        ("blur", 1.0, {"kinds": ("gaussian", "aniso", "sinc"), "sigma": (0.2, 3.0)}),
        ("resize", 1.0, {"factor": (1.0, 0.3), "modes": ("area", "bilinear", "bicubic")}),
        ("gaussian_noise", 0.5, {"sigma": (1.0, 30.0)}),
        ("sensor_noise", 0.5, {}),
        ("jpeg", 1.0, {"quality": (95, 30), "subsampling": ("4:2:0",)}),
        ("blur", 0.8, {"kinds": ("gaussian", "aniso"), "sigma": (0.2, 1.5)}),
        ("resize", 1.0, {"modes": ("area", "bilinear", "bicubic")}),
        ("gaussian_noise", 0.5, {"sigma": (1.0, 25.0)}),
        ("blur", 0.4, {"kinds": ("sinc",)}),
        ("jpeg", 1.0, {"quality": (95, 30), "subsampling": ("4:2:0",)}),
    ],
}
OPERATORS = {"blur": op_blur, "resize": op_resize, "gaussian_noise": op_gaussian_noise,
             "sensor_noise": op_sensor_noise, "jpeg": op_jpeg, "webp": op_webp, "sharpen": op_sharpen,
             "denoise": op_denoise, "demosaic": op_demosaic, "quantize": op_quantize, "halftone": op_halftone,
             "ink_bleed": op_ink_bleed, "paper": op_paper, "binarize": op_binarize,
             "screen_grid": op_screen_grid, "illumination": op_illumination, "color_shift": op_color_shift}
DEFAULT_WEIGHTS = {"camera": 0.25, "screenshot": 0.2, "web_chain": 0.15, "scan": 0.15,
                   "screen_photo": 0.15, "realesrgan": 0.1}


def synthesize(hr, seed, scale=4, profile=None, weights=None, severity=None):
    """One LR for one HR: returns (LR PIL image, metadata).

    ``profile`` fixes the scenario; otherwise one is drawn by ``weights``.
    ``severity`` in [0, 1] centres every parameter range (None: uniform), which
    lets the same code serve training (broad) and diagnostics (fixed level).
    """
    rng = np.random.default_rng(seed)
    p = Sampler(rng, severity)
    weights = weights or DEFAULT_WEIGHTS
    if profile is None:
        names = sorted(weights)
        probabilities = np.array([weights[name] for name in names], dtype=float)
        profile = names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]
    image = np.asarray(hr.convert("RGB"), dtype=np.float32) / 255
    target = (max(1, round(image.shape[1] / scale)), max(1, round(image.shape[0] / scale)))
    steps = []
    for name, probability, options in PROFILES[profile]:
        if not p.chance(probability):
            continue
        options = dict(options)
        if name == "resize":
            options["target"] = target
        if name == "screen_grid":
            options["scale"] = image.shape[1] / target[0]
        if name == "blur" and (image.shape[1], image.shape[0]) == target:
            options["sigma_scale"] = 1 / scale  # optics after the LR grid is reached
        image, record = OPERATORS[name](image, p, **options)
        image = np.clip(image, 0, 1).astype(np.float32)
        steps.append({"op": name, **record})
    if (image.shape[1], image.shape[0]) != target:
        image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        steps.append({"op": "resize", "mode": "area", "size": list(target)})
    lr = Image.fromarray(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8))
    return lr, {"profile": profile, "seed": int(seed), "scale": scale, "severity": severity,
                "hr_size": [hr.width, hr.height], "lr_size": list(target), "steps": steps}
