"""Text-aware degradation profiles: determinism, alignment and measurable effects."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from data_pipeline import degradations as D
from data_pipeline.augmentation import degrade
from data_pipeline.config import validate_config
from data_pipeline.probes import edge_width, jpeg_history, noise_sigma, periodic_peak, to_gray


def text_image(width=384, height=160, color=(20, 20, 30)):
    image = Image.new("RGB", (width, height), (247, 244, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=34)
    draw.text((12, 14), "Invoice 4471-B", fill=color, font=font)
    draw.text((12, 84), "Total $1,284.50", fill=(190, 25, 25), font=font)
    return image


def dense_text(width=768, height=512):
    image = Image.new("RGB", (width, height), (247, 244, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    words = ["Invoice", "4471-B", "Total", "$1,284.50", "Exit", "Platform", "2026", "fox"]
    for row, y in enumerate(range(10, height - 30, 40)):
        draw.text((10, y), " ".join(words[(row + i) % len(words)] for i in range(6)), fill=(20, 20, 30), font=font)
    return image


def as_float(image):
    return np.asarray(image, dtype=np.float32) / 255


def psnr(a, b):
    error = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(error, 1e-12))


@pytest.mark.parametrize("profile", sorted(D.PROFILES))
@pytest.mark.parametrize("size,scale", [((384, 160), 4), ((257, 131), 4), ((257, 131), 3), ((100, 60), 2)])
def test_every_profile_lands_on_the_exact_lr_grid(profile, size, scale):
    hr = text_image().resize(size)
    lr, metadata = D.synthesize(hr, seed=11, scale=scale, profile=profile)
    assert lr.size == (round(size[0] / scale), round(size[1] / scale))
    assert metadata["lr_size"] == list(lr.size) and metadata["profile"] == profile
    assert lr.mode == "RGB"


def test_same_seed_reproduces_lr_and_record():
    hr = text_image()
    first = D.synthesize(hr, seed=5)
    second = D.synthesize(hr, seed=5)
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0])) and first[1] == second[1]
    assert first[1] != D.synthesize(hr, seed=6)[1]


def test_weights_restrict_profiles():
    hr = text_image()
    drawn = {D.synthesize(hr, seed, weights={"scan": 1, "screenshot": 0})[1]["profile"] for seed in range(12)}
    assert drawn == {"scan"}


@pytest.mark.parametrize("profile", sorted(D.PROFILES))
def test_severity_orders_fidelity(profile):
    hr = text_image()
    reference = np.asarray(hr)

    def fidelity(severity):
        values = []
        for seed in range(6):
            lr, _ = D.synthesize(hr, seed, scale=4, profile=profile, severity=severity)
            values.append(psnr(lr.resize(hr.size, Image.Resampling.BICUBIC), reference))
        return np.mean(values)

    assert fidelity(0.1) > fidelity(0.9)


def test_rggb_mosaic_matches_opencv_bayer_bg_code():
    rng = np.random.default_rng(0)
    smooth = cv2.GaussianBlur(rng.uniform(0, 1, (64, 64, 3)).astype(np.float32), (0, 0), 4)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    restored, record = D.op_demosaic(smooth, D.Sampler(rng, None), strength=(1.0, 1.0), chroma_sigma=(0.5, 0.5))
    assert record["pattern"] == "RGGB"
    assert np.abs(restored - smooth)[4:-4, 4:-4].mean() < 0.01


def test_jpeg_operator_is_measured_by_the_history_probe():
    image = as_float(dense_text())
    p = D.Sampler(np.random.default_rng(0), None)
    compressed, record = D.op_jpeg(image, p, quality=(55, 55))
    history = jpeg_history(np.clip(compressed * 255 + 0.5, 0, 255).astype(np.uint8))
    assert record["quality"] == 55 and abs(history["quality"] - 55) <= 3


def test_chroma_subsampling_bleeds_coloured_strokes():
    image = as_float(text_image(512, 256))
    p = D.Sampler(np.random.default_rng(0), None)
    # At q95 quantization is mild, so the difference isolates chroma subsampling.
    full, _ = D.op_jpeg(image, p, quality=(95, 95), subsampling=("4:4:4",))
    sub, _ = D.op_jpeg(image, p, quality=(95, 95), subsampling=("4:2:0",))

    def chroma_error(x):
        ycc = lambda v: cv2.cvtColor(np.round(v * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(float)
        return np.abs(ycc(x)[80:135, ..., 1:] - ycc(image)[80:135, ..., 1:]).mean()  # the red line

    assert chroma_error(sub) > 2 * chroma_error(full)


def test_noise_operators_match_requested_strength():
    flat = np.full((256, 256, 3), 0.5, np.float32)
    p = D.Sampler(np.random.default_rng(1), None)
    noisy, record = D.op_gaussian_noise(flat, p, sigma=(6, 6), gray_probability=1.0)
    assert abs(noise_sigma(to_gray(np.clip(noisy * 255, 0, 255))) - 6) < 0.6
    shot, record = D.op_sensor_noise(flat, p, shot=(0.004, 0.004), read=(2, 2), chroma=(0.0, 0.0))
    channels = shot - shot.mean(axis=(0, 1))
    assert np.allclose(channels[..., 0], channels[..., 1], atol=1e-6)  # luma-only when chroma=0


def test_blur_widens_edges_in_proportion():
    image = as_float(text_image(512, 256))
    p = D.Sampler(np.random.default_rng(0), None)
    widths = []
    for sigma in (0.5, 1.5, 3.0):
        blurred, _ = D.op_blur(image, p, kinds=("gaussian",), sigma=(sigma, sigma))
        widths.append(edge_width(to_gray(np.clip(blurred * 255, 0, 255))))
    assert widths[0] < widths[1] < widths[2]


def test_nearest_downscale_aliases_where_area_averages():
    stripes = np.zeros((256, 256, 3), np.float32)
    stripes[:, ::2] = 1.0  # one-pixel stripes: above the LR Nyquist
    size = (70, 70)  # a non-integer factor, so nearest samples drift across phases
    std = {mode: D.resize_to(stripes, size, mode).std() for mode in ("area", "pil_bicubic", "bilinear", "bicubic", "nearest")}
    # Area and Pillow (support widened when shrinking) average the stripes away;
    # OpenCV's bilinear/bicubic keep a fixed support and alias, nearest the most.
    assert std["area"] < 0.05 and std["pil_bicubic"] < 0.05
    assert min(std["bilinear"], std["bicubic"]) > 0.2 and std["nearest"] > 0.4


def test_halftone_and_screen_grid_leave_their_patterns():
    rng = np.random.default_rng(2)
    p = D.Sampler(rng, 0.5)
    tone = np.repeat(np.linspace(0.2, 0.8, 512, dtype=np.float32)[None, :, None], 512, axis=0).repeat(3, axis=2)
    printed, _ = D.op_halftone(tone, p)
    assert periodic_peak(to_gray(printed * 255)) > 0.03 > periodic_peak(to_gray(tone * 255))
    gray = np.full((512, 512, 3), 0.6, np.float32)
    shot, record = D.op_screen_grid(gray, p, scale=4, contrast=(0.5, 0.5), period=(1.15, 1.15))
    lr = D.resize_to(shot, (128, 128), "area")
    assert np.abs(lr - lr.mean(axis=2, keepdims=True)).mean() > 0.002  # colour moire survives (0 without)
    assert abs(float(shot.mean()) - 0.6) < 0.02  # brightness preserved


def test_binarize_follows_the_page_not_a_fixed_level():
    dark_page = as_float(text_image(color=(20, 20, 30))) * 0.55
    out, record = D.op_binarize(dark_page, D.Sampler(np.random.default_rng(0), 0.0), speckle=(0, 0))
    assert 0.05 < out.mean() < 0.98  # text and paper survive on a dim scan


def test_pipeline_profiles_transform_boxes_and_keep_legacy_default():
    hr = text_image()
    boxes = [{"bbox": [12, 14, 260, 50], "polygon": [[12, 14], [260, 14], [260, 50], [12, 50]], "text": "x"}]
    base = {"backends": {"teacher": {"type": "identity"}, "ocr": {"type": "sidecar_ocr"},
                         "iqa": {"type": "sidecar_iqa"}, "glyph": {"type": "identity_source"}},
            "quality": {"ocr_min_confidence": 0.9, "original_min_confidence": 0.95, "mode": "audit"}}
    legacy = validate_config(base)["augmentation"]
    _, _, metadata = degrade(hr, boxes, legacy, seed=3)
    assert metadata["operations"][0] == "gaussian_blur"
    config = validate_config({**base, "augmentation": {"scales": [4], "profiles": {"camera": 1}}})["augmentation"]
    lr, lr_boxes, metadata = degrade(hr, boxes, config, seed=3)
    assert metadata["profile"] == "camera" and lr.size == (96, 40)
    assert lr_boxes[0]["bbox"] == pytest.approx([3, 3.5, 65, 12.5])


def test_illumination_keeps_the_mean_level():
    image = as_float(text_image())
    lit, record = D.op_illumination(image, D.Sampler(np.random.default_rng(4), 1.0))
    assert abs(float(lit.mean()) - float(image.mean())) < 0.01 and record["strength"] > 0.1
