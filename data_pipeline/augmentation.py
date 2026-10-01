"""Deterministic LR degradation. The cleaned HR and its geometry stay intact."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageFilter

from .geometry import scale_matrix, transform


def sample_seed(seed, key, variant):
    payload = f"{seed}:{key}:{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def degrade(hr, ocr, config, seed):
    rng = np.random.default_rng(seed)
    scale = int(rng.choice(config["scales"]))
    sigma = float(rng.uniform(*config["blur_sigma"]))
    noise = float(rng.uniform(*config["noise_std"]))
    quality = int(rng.integers(config["jpeg_quality"][0], config["jpeg_quality"][1] + 1))
    size = (max(1, round(hr.width / scale)), max(1, round(hr.height / scale)))
    lr = hr.filter(ImageFilter.GaussianBlur(sigma)).resize(size, Image.Resampling.BICUBIC)
    pixels = np.asarray(lr, dtype=np.float32)
    pixels += rng.normal(0, noise, pixels.shape).astype(np.float32)
    lr = Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    lr.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        lr = compressed.convert("RGB")
    matrix = scale_matrix(hr.size, lr.size)
    metadata = {"seed": seed, "scale": scale, "blur_sigma": sigma,
                "noise_std_255": noise, "jpeg_quality": quality,
                "jpeg_subsampling": 2, "hr_to_lr": matrix.tolist(),
                "operations": ["gaussian_blur", "bicubic_downsample", "gaussian_noise", "jpeg"]}
    return lr, transform(ocr, matrix), metadata
