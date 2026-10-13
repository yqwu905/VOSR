"""Cleaning probes, routing, deduplication, materialization and source adapters."""

import io
import json
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from data_pipeline.cleaning import duplicate_pairs, probe_manifest
from data_pipeline.datasets import convert_annotations
from data_pipeline.geometry import regions
from data_pipeline.pipeline import Pipeline, json_text
from data_pipeline.config import validate_config
from data_pipeline.probes import (edge_width, effective_upscale, hamming, jpeg_history, noise_sigma,
                                  periodic_peak, probe_container, probe_image, route, to_gray)
from data_pipeline.sources import ADAPTERS, anyword


def render(width=640, height=480, size=22, seed=0):
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), (246, 243, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=size)
    boxes, y = [], 12
    while y < height - 2 * size:
        text = " ".join(rng.choice(["Invoice", "TOTAL", "No.4471", "Exit B", "$1,284.50", "Platform",
                                    "Quick", "brown", "fox", "2026"]) for _ in range(5))
        draw.text((14, y), text, fill=(18, 18, 28), font=font)
        boxes.append({"bbox": list(draw.textbbox((14, y), text, font=font)), "text": text})
        y += int(size * 1.7)
    return image, boxes


def encode(image, fmt="PNG", **options):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def decoded(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


@pytest.mark.parametrize("quality", [30, 60, 90])
def test_container_quality_matches_encoder_setting(quality):
    image, _ = render()
    assert probe_container(encode(image, "JPEG", quality=quality))["jpeg_quality"] == quality


def test_jpeg_history_survives_png_reencode_and_crop():
    image, _ = render()
    recompressed = decoded(encode(image, "JPEG", quality=50)).crop((3, 5, 600, 470))
    history = jpeg_history(np.asarray(decoded(encode(recompressed))))
    assert history is not None and abs(history["quality"] - 50) <= 3
    assert history["grid_offset"] == [3, 5]


def test_jpeg_history_absent_without_compression():
    image, _ = render()
    assert jpeg_history(np.asarray(image)) is None
    upscaled = image.resize((320, 240), Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)
    assert jpeg_history(np.asarray(upscaled)) is None


def test_effective_upscale_recovers_the_factor():
    image, _ = render(960, 720, size=26)
    assert effective_upscale(to_gray(np.asarray(image)))[0] == 1.0
    for factor in (2, 3):
        small = image.resize((960 // factor, 720 // factor), Image.Resampling.LANCZOS)
        upscaled = small.resize(image.size, Image.Resampling.BICUBIC)
        assert effective_upscale(to_gray(np.asarray(upscaled)))[0] == factor


def test_noise_and_edge_width_track_injected_degradation():
    image, _ = render()
    gray = to_gray(np.asarray(image))
    assert noise_sigma(gray) < 0.2
    noisy = gray + np.random.default_rng(0).normal(0, 4, gray.shape).astype(np.float32)
    assert abs(noise_sigma(noisy) - 4) < 0.8
    sharp = edge_width(gray)
    import cv2
    assert edge_width(cv2.GaussianBlur(gray, (0, 0), 1.0)) > sharp + 0.7
    assert edge_width(cv2.GaussianBlur(gray, (0, 0), 2.0)) > sharp + 2.0


def test_periodic_peak_separates_halftone_from_text():
    image, _ = render()
    gray = to_gray(np.asarray(image))
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    tone = (x / 255) * 0.8 + 0.1
    screen = (np.cos(2 * np.pi * x / 5) * np.cos(2 * np.pi * y / 5) + 1) / 2
    halftone = (tone > screen).astype(np.float32) * 255
    assert periodic_peak(gray) < 0.03 < periodic_peak(halftone)


def test_alpha_hidden_text_is_flagged():
    rgba = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).text((20, 120), "HIDDEN", fill=(0, 0, 0, 255), font=ImageFont.load_default(size=40))
    stats = probe_container(encode(rgba))
    assert stats["alpha_fraction"] > 0.5
    # Pillow keeps the colour under alpha=0, so a plain RGB decode is all black here.
    assert np.asarray(rgba.convert("RGB")).max() == 0


BASE = {"size": [1000, 800], "frames": 1, "luma_range": 200.0, "padding_fraction": 0.0,
        "effective_upscale": 1.0, "edge_width": 1.6, "noise_sigma": 0.3, "periodic_peak": 0.005,
        "text_regions": 5, "text_height_median": 48.0}


@pytest.mark.parametrize("changes,decision,downscale", [
    ({}, "direct", 1.0),
    ({"effective_upscale": 2.0}, "downscale", 2.0),
    ({"jpeg_quality": 60}, "downscale", 2.0),
    ({"jpeg_history": {"quality": 80}}, "downscale", 1.5),
    ({"noise_sigma": 4.0}, "downscale", 3.0),
    ({"jpeg_quality": 30}, "teacher", 1.0),
    ({"effective_upscale": 3.0, "text_height_median": 30.0}, "teacher", 1.0),
    ({"text_height_median": 9.0}, "reject", 1.0),
    ({"text_regions": 0, "text_height_median": None}, "reject", 1.0),
    ({"size": [200, 800]}, "reject", 1.0),
])
def test_route_decisions(changes, decision, downscale):
    result = route({**BASE, **changes})
    assert result["decision"] == decision
    assert result["downscale"] == downscale


def test_route_fixes_do_not_reject():
    result = route({**BASE, "alpha_fraction": 0.4, "padding_fraction": 0.5})
    assert result["decision"] == "direct"
    assert set(result["fixes"]) == {"composite_alpha", "trim_padding"}


def test_banded_duplicate_search_matches_brute_force():
    rng = random.Random(3)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(60)]
    for i in range(0, 20, 2):  # plant near duplicates at distance <= 6
        value = int(hashes[i], 16)
        for bit in rng.sample(range(64), rng.randint(0, 6)):
            value ^= 1 << bit
        hashes[i + 1] = f"{value:016x}"
    expected = [(a, b) for a in range(60) for b in range(a + 1, 60) if hamming(hashes[a], hashes[b]) <= 6]
    assert duplicate_pairs(hashes, 6) == expected


def write_manifest(path, records):
    path.write_text("".join(json_text(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_probe_manifest_routes_dedupes_and_materializes(tmp_path):
    image, boxes = render(960, 720, size=30)
    image.save(tmp_path / "clean.png")
    image.resize((954, 716), Image.Resampling.BICUBIC).save(tmp_path / "near_copy.png")
    small = image.resize((480, 360), Image.Resampling.LANCZOS)
    small.resize(image.size, Image.Resampling.BICUBIC).save(tmp_path / "upscaled.png")
    other, other_boxes = render(960, 720, size=30, seed=9)
    padded = Image.new("RGB", (960, 1200), (0, 0, 0))  # square-ish letterboxing, 40% of the area
    padded.paste(other, (0, 240))
    padded.save(tmp_path / "padded.png")
    padded_boxes = [{"bbox": [b["bbox"][0], b["bbox"][1] + 240, b["bbox"][2], b["bbox"][3] + 240]}
                    for b in other_boxes]
    manifest = write_manifest(tmp_path / "input.jsonl", [
        {"id": "clean", "source": "render", "image_path": "clean.png", "annotations": boxes},
        {"id": "near", "source": "render", "image_path": "near_copy.png", "annotations": boxes},
        {"id": "upscaled", "source": "web", "image_path": "upscaled.png", "annotations": boxes},
        {"id": "padded", "source": "web", "image_path": "padded.png", "annotations": padded_boxes},
    ])
    summary = probe_manifest(manifest, tmp_path / "out", materialize=True)
    rows = {r["id"]: r for r in map(json.loads, (tmp_path / "out" / "probes.jsonl").read_text().splitlines())}
    assert rows["clean"]["route"]["decision"] == "direct"
    assert "near_duplicate" in rows["near"]["route"]["reasons"]
    assert rows["near"]["route"]["duplicate_of"] == "clean"
    assert "near_duplicate" in rows["upscaled"]["route"]["reasons"]
    assert rows["padded"]["route"]["decision"] == "direct"
    assert "trim_padding" in rows["padded"]["route"]["fixes"]
    assert summary["materialized"] == {"clean": 2, "rejected": 2}
    clean = {r["id"]: r for r in map(json.loads, (tmp_path / "out" / "clean.jsonl").read_text().splitlines())}
    trimmed = Image.open(clean["padded"]["image_path"])
    assert trimmed.size == (960, 720)
    assert clean["padded"]["annotations"][0]["bbox"][1] == pytest.approx(other_boxes[0]["bbox"][1])


def test_materialized_downscale_scales_annotations(tmp_path):
    image, boxes = render(960, 720, size=40)
    small = image.resize((480, 360), Image.Resampling.LANCZOS)
    small.resize(image.size, Image.Resampling.BICUBIC).save(tmp_path / "upscaled.png")
    manifest = write_manifest(tmp_path / "input.jsonl", [
        {"id": "upscaled", "source": "web", "image_path": "upscaled.png", "annotations": boxes}])
    probe_manifest(manifest, tmp_path / "out", materialize=True)
    record = json.loads((tmp_path / "out" / "clean.jsonl").read_text())
    assert record["cleaning"]["route"]["decision"] == "downscale"
    assert Image.open(record["image_path"]).size == (480, 360)
    assert record["annotations"][0]["bbox"][2] == pytest.approx(boxes[0]["bbox"][2] / 2)


def test_benchmark_overlap_is_rejected(tmp_path):
    image, boxes = render(800, 600, size=30)
    image.save(tmp_path / "train.png")
    image.save(tmp_path / "bench.png")
    write_manifest(tmp_path / "bench.jsonl", [{"id": "b0", "source": "screensr", "image_path": "bench.png"}])
    manifest = write_manifest(tmp_path / "input.jsonl", [
        {"id": "t0", "source": "web", "image_path": "train.png", "annotations": boxes}])
    probe_manifest(manifest, tmp_path / "out", exclude=[tmp_path / "bench.jsonl"])
    row = json.loads((tmp_path / "out" / "probes.jsonl").read_text())
    assert row["route"]["decision"] == "reject" and row["route"]["benchmark_id"] == "b0"


def test_source_adapters_put_boxes_in_pixel_coordinates():
    size = (762, 1000)
    funsd = ADAPTERS["funsd"]({"words": ["TO:"], "bboxes": [[29, 87, 52, 98]]}, size)
    assert funsd[0]["bbox"] == pytest.approx([29 * 0.762, 87, 52 * 0.762, 98])
    sroie = ADAPTERS["sroie"]({"words": ["TOTAL"], "bboxes": [[72, 25, 326, 64]]}, size)
    assert sroie[0]["bbox"] == [72, 25, 326, 64]
    info = {"ocr_info": [{"word": "GRME", "bounding_box": {"width": 0.05, "height": 0.04,
                                                             "top_left_x": 0.3, "top_left_y": 0.4}}]}
    box = ADAPTERS["ocr_info"](info, (640, 500))[0]
    assert box["bbox"] == pytest.approx([192, 200, 224, 220]) and box["annotation_origin"] == "dataset_ocr"
    quad = {"x1": 1, "y1": 2, "x2": 30, "y2": 2, "x3": 30, "y3": 12, "x4": 1, "y4": 12}
    cord = ADAPTERS["cord"]({"ground_truth": json.dumps({"valid_line": [{"words": [{"quad": quad, "text": "Nasi"}]}]})},
                            size)
    assert regions(cord, size)[0]["bbox"] == [1, 2, 30, 12]


def test_null_anyword_annotations_are_recorded_not_fatal():
    assert convert_annotations("anyword3m", {"annotations": None}, (512, 512)) == ([], [])
    assert anyword({"annotations": None}, (512, 512)) == [{"invalid": {"reason": "annotations_null"}}]


def pipeline_config(**augmentation):
    return validate_config({
        "long_edge": 1024, "upscale_to_long_edge": False,
        "crop": {"width": 400, "height": 120, "max_crops": 2},
        "quality": {"ocr_min_confidence": 0.9, "original_min_confidence": 0.95,
                    "iqa_thresholds": {"clarity": {"threshold": 0.5, "direction": "higher"},
                                       "naturalness": {"threshold": 0.5, "direction": "higher"},
                                       "artifacts": {"threshold": 0.5, "direction": "lower"}}},
        "backends": {"teacher": {"type": "identity"}, "ocr": {"type": "sidecar_ocr"},
                     "iqa": {"type": "sidecar_iqa"}, "glyph": {"type": "identity_source"}},
        "augmentation": {"variants": 2, "scales": [4], **augmentation},
    })


def test_identity_teacher_keeps_native_size_and_uses_text_profiles(tmp_path):
    image = Image.new("RGB", (400, 120), (246, 243, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((20, 36), "Invoice 4471", fill=(18, 18, 28), font=font)
    boxes = [{"bbox": list(draw.textbbox((20, 36), "Invoice 4471", font=font)), "text": "Invoice 4471"}]
    image.save(tmp_path / "page.png")
    crop = image.crop((0, 0, 400, 120))
    sample = {"id": "page", "source": "render", "image_path": "page.png",
              "annotations": [dict(boxes[0], trusted=True)],
              "ocr": {"final": {"crop_0000": {"size": list(crop.size),
                                              "regions": [dict(boxes[0], confidence=0.99)]}}},
              "iqa": {"crop_0000": {"size": list(crop.size),
                                    "scores": {"clarity": 0.9, "naturalness": 0.9, "artifacts": 0.1}}}}
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json_text(sample) + "\n", encoding="utf-8")
    config = pipeline_config(profiles={"screenshot": 1, "scan": 1}, severity=0.5)
    Pipeline(config, manifest, tmp_path / "out").run()
    accepted = [json.loads(x) for x in (tmp_path / "out" / "accepted.jsonl").read_text().splitlines()]
    assert len(accepted) == 1 and accepted[0]["full_hr_size"] == [400, 120]
    variants = accepted[0]["variants"]
    assert {v["degradation"]["profile"] for v in variants} <= {"screenshot", "scan"}
    assert all(v["lr_size"] == [100, 30] for v in variants)
    lr_box = variants[0]["ocr_lr"][0]["bbox"]
    assert lr_box[2] == pytest.approx(boxes[0]["bbox"][2] / 4)


def test_config_rejects_unknown_profile_and_misused_identity_glyph():
    with pytest.raises(ValueError, match="profiles"):
        pipeline_config(profiles={"fax_machine": 1})
    config = pipeline_config()
    config["backends"]["teacher"] = {"type": "precomputed", "kwargs": {"model": "vosr2"}}
    with pytest.raises(ValueError, match="identity"):
        validate_config(config)


def test_probe_image_measures_text_regions_only():
    image, boxes = render(800, 600, size=30)
    import cv2
    pixels = np.asarray(image).copy()
    pixels[300:, :] = cv2.GaussianBlur(pixels[300:, :], (0, 0), 2.0)  # blur the lower half
    sharp = probe_image(encode(Image.fromarray(pixels)), regions([b for b in boxes if b["bbox"][3] < 290], (800, 600)))
    blurred = probe_image(encode(Image.fromarray(pixels)), regions([b for b in boxes if b["bbox"][1] > 310], (800, 600)))
    assert sharp["edge_width"] == blurred["edge_width"]
    assert blurred["text_edge_width"] > sharp["text_edge_width"] + 2


def test_word_crops_too_small_for_large_factors_do_not_break_the_run(tmp_path):
    crop = Image.new("RGB", (72, 31), (240, 240, 240))
    ImageDraw.Draw(crop).text((4, 6), "Lube", fill=(10, 10, 10), font=ImageFont.load_default(size=18))
    crop.save(tmp_path / "word.png")
    manifest = write_manifest(tmp_path / "input.jsonl", [{"id": "w", "source": "mjsynth", "image_path": "word.png"}])
    probe_manifest(manifest, tmp_path / "out")
    row = json.loads((tmp_path / "out" / "probes.jsonl").read_text())
    assert row["route"]["decision"] == "reject" and "too_small" in row["route"]["reasons"]
    assert row["stats"]["resample_residual"]["8"] is None


@pytest.mark.parametrize("size", [(512, 1296), (640, 512), (520, 700)])
def test_jpeg_grid_search_handles_window_sized_edges(size):
    image, _ = render(*size, size=24)
    shifted = decoded(encode(decoded(encode(image, "JPEG", quality=60)).crop((5, 2, size[0], size[1]))))
    history = jpeg_history(np.asarray(shifted))
    assert history is not None and history["grid_offset"] == [6, 3]


def test_double_compression_is_judged_by_the_earlier_quality():
    image, boxes = render(800, 600, size=40)
    resaved = encode(decoded(encode(image, "JPEG", quality=60)), "JPEG", quality=100)
    stats = probe_image(resaved, regions(boxes, image.size))
    assert stats["jpeg_quality"] == 100 and abs(stats["jpeg_history"]["quality"] - 60) <= 3
    assert route(stats)["degradations"]["jpeg"] == 2.0
