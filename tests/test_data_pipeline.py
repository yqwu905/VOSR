"""Geometry, quality decisions and real artifact-backed pipeline contracts."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from data_pipeline.__main__ import create_demo
from data_pipeline.augmentation import degrade, sample_seed
from data_pipeline.backends import TesseractOCR, pixel_digest, read_rgb
from data_pipeline.config import validate_config
from data_pipeline.geometry import (contains, crop_regions, normalize_1k, regions,
                                    text_crops, transform)
from data_pipeline.pipeline import InputIntegrityError, Pipeline, json_text
from data_pipeline.quality import assess, reference_check


@pytest.fixture
def config():
    return validate_config({
        "long_edge": 64,
        "crop": {"width": 64, "height": 32},
        "quality": {"ocr_min_confidence": 0.9, "original_min_confidence": 0.95,
                    "iqa_thresholds": {"clarity": {"threshold": 0.7, "direction": "higher"},
                                       "naturalness": {"threshold": 0.6, "direction": "higher"},
                                       "artifacts": {"threshold": 0.2, "direction": "lower"}}},
        "backends": {"teacher": {"type": "precomputed", "kwargs": {"model": "vosr2"}},
                     "ocr": {"type": "sidecar_ocr"}, "iqa": {"type": "sidecar_iqa"},
                     "glyph": {"type": "human"}},
        "augmentation": {"variants": 2},
    })


def make_record(root, name="one", color="white", group=None):
    image = Image.new("RGB", (64, 32), color)
    ImageDraw.Draw(image).rectangle((10, 8, 50, 24), fill="black")
    source_path, hr_path = root / (name + ".png"), root / (name + "_hr.png")
    image.save(source_path)
    hr = image.resize((128, 64), Image.Resampling.NEAREST)
    hr.save(hr_path)
    crop, _ = normalize_1k(hr, 64)
    item = {"bbox": [10, 8, 50, 24], "text": "SR 2026", "confidence": 0.99}
    sample = {"id": name, "source": "magazine", "image_path": source_path.name,
              "hr_path": hr_path.name, "teacher": {"model": "vosr2", "checkpoint": "test-fixture"},
              "annotations": [dict(item, trusted=True)],
              "ocr": {"original": {"size": [64, 32], "regions": [dict(item, reliable=True)]},
                      "final": {"crop_0000": {"size": [64, 32], "image_sha256": pixel_digest(crop),
                                               "regions": [item]}}},
              "iqa": {"crop_0000": {"size": [64, 32], "scores": {"clarity": 0.9,
                      "naturalness": 0.8, "artifacts": 0.05}}},
              "glyph_checks": {"crop_0000": {"size": [64, 32], "image_sha256": pixel_digest(crop),
                               "original_crop_sha256": pixel_digest(image), "reviewer": "fixture-reviewer",
                               "status": "verified", "evidence": "fixture glyphs compared with original"}}}
    if group is not None:
        sample["group_id"] = group
    return sample


def write_manifest(root, samples):
    path = root / "input.jsonl"
    path.write_text("".join(json_text(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def records(directory, status):
    return [json.loads(line) for line in (directory / (status + ".jsonl")).read_text().splitlines()]


def test_1k_keeps_panorama_aspect_ratio():
    resized, matrix = normalize_1k(Image.new("RGB", (2000, 500)))
    assert resized.size == (1024, 256)
    assert matrix[0, 0] == matrix[1, 1] == 0.512


def test_text_center_crop_expands_for_complete_neighboring_lines():
    items = regions([{"bbox": [45, 45, 65, 55]}, {"bbox": [62, 40, 110, 60]},
                     {"bbox": [108, 35, 170, 65]}], (200, 100))
    crops = text_crops(items, (200, 100), width=40, height=20, context=0, max_crops=8)
    first = crops[0]
    assert contains(first, items[0]["bbox"])
    assert contains(first, items[1]["bbox"])
    assert contains(first, items[2]["bbox"])
    assert len(crop_regions(items, first)) == 3


def test_polygon_coordinates_roundtrip_after_resize_and_crop():
    items = regions([{"polygon": [[1, 2], [12, 3], [11, 9], [2, 10]], "text": "abc"}], (20, 12))
    matrix = np.array([[2.5, 0, -1], [0, 3.5, -2], [0, 0, 1]])
    transformed = transform(items, matrix)
    returned = transform(transformed, np.linalg.inv(matrix))
    np.testing.assert_allclose(returned[0]["polygon"], items[0]["polygon"])
    assert returned[0]["text"] == "abc"


@pytest.mark.parametrize("item", [
    {"bbox": [-1, 0, 2, 3]}, {"bbox": [1, 1, 1, 4]},
    {"bbox": [0, 0, 11, 3]}, {"bbox": [0, 0, float("nan"), 2]},
    {"bbox": [0, 0, 2, 3], "confidence": float("nan")},
    {"bbox": [0, 0, 2, 3], "confidence": 1.1},
    {"bbox": [0, 0, 2, 3], "confidence": True},
    {"bbox": [0, 0, 2, 3], "trusted": "false"},
])
def test_invalid_regions_are_not_silently_clipped(item):
    with pytest.raises(ValueError):
        regions([item], (10, 10))


def test_ocr_consistency_alone_is_not_accepted(config):
    prediction = regions([{"bbox": [0, 0, 10, 10], "text": "text", "confidence": 1}], (20, 20))
    reference = reference_check(prediction, [], prediction, config["quality"])
    assert reference["status"] == "unverifiable"
    decision = assess(prediction, reference, {"status": "unverifiable"},
                      {"clarity": 1, "naturalness": 1, "artifacts": 0}, config["quality"])
    assert decision["status"] == "review"
    assert not decision["eligible_for_training"]


def test_trusted_text_match_still_requires_glyph_check(config):
    prediction = regions([{"bbox": [0, 0, 10, 10], "text": "ABC", "confidence": 1}], (20, 20))
    trusted = [dict(prediction[0], trusted=True)]
    reference = reference_check(prediction, trusted, [], config["quality"])
    assert reference["status"] == "matched"
    decision = assess(prediction, reference, {}, {"clarity": 1, "naturalness": 1, "artifacts": 0}, config["quality"])
    assert decision["status"] == "review"
    assert decision["reasons"] == ["glyph_unverifiable"]


def test_reference_conflict_and_unmatched_text_go_to_review(config):
    predictions = regions([{"bbox": [0, 0, 10, 10], "text": "ABC", "confidence": 1},
                           {"bbox": [10, 10, 20, 20], "text": "new", "confidence": 1}], (20, 20))
    trusted = [dict(predictions[0], text="abc", trusted=True)]
    reference = reference_check(predictions, trusted, [], config["quality"])
    assert reference["status"] == "conflict"  # Case is semantically meaningful.
    assert reference["comparisons"][1]["status"] == "no_reliable_reference"


def test_unreadable_original_does_not_force_mechanical_consistency(config):
    prediction = regions([{"bbox": [0, 0, 10, 10], "text": "right", "confidence": 1}], (20, 20))
    original = [dict(prediction[0], text="wrong", confidence=0.4, reliable=True)]
    reference = reference_check(prediction, [], original, config["quality"])
    assert reference["status"] == "unverifiable"
    assert all(x["status"] != "conflict" for x in reference["comparisons"])


def test_quality_respects_iqa_direction_and_low_confidence(config):
    predictions = [{"text": "ok", "confidence": 0.5}]
    decision = assess(predictions, {"status": "matched"}, {"status": "verified", "evidence": "human"},
                      {"clarity": 1, "naturalness": 1, "artifacts": 0.3}, config["quality"])
    assert decision["status"] == "rejected"
    assert "low_ocr_confidence" in decision["reasons"]
    assert "low_iqa:artifacts" in decision["reasons"]


def test_missing_iqa_dimension_requires_review(config):
    decision = assess([{"text": "ok", "confidence": 1}], {"status": "matched"},
                      {"status": "verified", "evidence": "human"},
                      {"clarity": 1, "naturalness": 1}, config["quality"])
    assert decision["status"] == "review"
    assert decision["iqa_labels"]["artifacts"] == "missing"


def test_degradation_is_repeatable_and_updates_lr_coordinates(config):
    hr = Image.new("RGB", (65, 33), "white")
    ocr = regions([{"bbox": [5, 3, 60, 30], "text": "SR", "confidence": 1}], hr.size)
    seed = sample_seed(42, "one", 0)
    first = degrade(hr, ocr, config["augmentation"], seed)
    second = degrade(hr, ocr, config["augmentation"], seed)
    assert first[0].tobytes() == second[0].tobytes()
    assert first[1:] == second[1:]
    assert sample_seed(42, "two", 0) != seed
    np.testing.assert_allclose(first[1][0]["bbox"], [5 * first[0].width / 65, 3 * first[0].height / 33,
                                                   60 * first[0].width / 65, 30 * first[0].height / 33])
    assert ocr[0]["bbox"] == [5, 3, 60, 30]


def test_config_requires_calibrated_thresholds():
    production = json.loads((Path(__file__).parents[1] /
                             "configs/data_pipeline/production.json").read_text())
    with pytest.raises(ValueError, match="calibrate"):
        validate_config(production)


@pytest.mark.parametrize("field,value", [("jpeg_quality", [95, 65]), ("blur_sigma", [-1, 1]),
                                         ("noise_std", [0, float("inf")])])
def test_bad_augmentation_config_fails_before_processing(config, field, value):
    config["augmentation"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_artifact_pipeline_exports_only_accepted_and_resumes(tmp_path, config):
    sample = make_record(tmp_path)
    input_path = write_manifest(tmp_path, [sample])
    directory = tmp_path / "output"
    result = Pipeline(config, input_path, directory).run()
    assert result["crops"] == {"accepted": 1}
    accepted = records(directory, "accepted")[0]
    assert accepted["is_pseudo_gt"] and accepted["eligible_for_training"]
    assert accepted["reference_check"]["status"] == "matched"
    assert len(accepted["variants"]) == 2
    assert read_rgb(accepted["full_hr_path"]).size == (64, 32)
    assert (directory / (accepted["split"] + "_hq.txt")).read_text().strip() == accepted["hr_path"]
    before = (directory / "accepted.jsonl").read_bytes()
    hr_time = Path(accepted["hr_path"]).stat().st_mtime_ns
    Pipeline(config, input_path, directory, resume=True).run()
    assert (directory / "accepted.jsonl").read_bytes() == before
    assert Path(accepted["hr_path"]).stat().st_mtime_ns == hr_time
    # Missing outputs are repaired rather than skipped on resume.
    Path(accepted["variants"][0]["lr_path"]).unlink()
    Pipeline(config, input_path, directory, resume=True).run()
    assert Path(accepted["variants"][0]["lr_path"]).is_file()


def test_corrupt_image_is_recorded_and_other_samples_finish(tmp_path, config):
    good = make_record(tmp_path)
    (tmp_path / "bad.png").write_bytes(b"broken")
    bad = {"id": "bad", "source": "anyword3m", "image_path": "bad.png"}
    directory = tmp_path / "output"
    result = Pipeline(config, write_manifest(tmp_path, [bad, good]), directory).run()
    assert result["crops"] == {"accepted": 1, "error": 1}
    assert records(directory, "errors")[0]["source"] == "anyword3m"


def test_model_runtime_error_isolated_per_sample(tmp_path, config, monkeypatch):
    from data_pipeline import pipeline as module
    first = make_record(tmp_path, "first", "white")
    second = make_record(tmp_path, "second", "yellow")
    original_loader = module.load_backend

    def loader(spec):
        backend = original_loader(spec)
        if spec["type"] == "precomputed":
            restore = backend.restore

            def conditional(image, context):
                if context["sample"]["id"] == "first":
                    raise RuntimeError("model sample failure")
                return restore(image, context)
            backend.restore = conditional
        return backend

    monkeypatch.setattr(module, "load_backend", loader)
    result = Pipeline(config, write_manifest(tmp_path, [first, second]), tmp_path / "output").run()
    assert result["crops"] == {"error": 1, "accepted": 1}


def test_exact_duplicates_do_not_add_training_images(tmp_path, config):
    first = make_record(tmp_path)
    second = dict(first, id="duplicate", source="easytext")
    directory = tmp_path / "output"
    result = Pipeline(config, write_manifest(tmp_path, [first, second]), directory).run()
    assert result["crops"] == {"accepted": 1, "rejected": 1}
    assert records(directory, "rejected")[0]["reasons"] == ["duplicate_image"]


def test_resume_repairs_canonical_sample_when_duplicate_also_exists(tmp_path, config):
    sample = make_record(tmp_path)
    duplicate = dict(sample, id="duplicate", source="easytext")
    input_path = write_manifest(tmp_path, [sample, duplicate])
    directory = tmp_path / "output"
    Pipeline(config, input_path, directory).run()
    accepted = records(directory, "accepted")[0]
    Path(accepted["hr_path"]).unlink()
    result = Pipeline(config, input_path, directory, resume=True).run()
    assert result["crops"] == {"accepted": 1, "rejected": 1}
    assert Path(accepted["hr_path"]).is_file()


def test_missing_glyph_review_and_iqa_keep_images_out_of_training(tmp_path, config):
    sample = make_record(tmp_path)
    sample.pop("glyph_checks")
    sample.pop("iqa")
    directory = tmp_path / "output"
    result = Pipeline(config, write_manifest(tmp_path, [sample]), directory).run()
    assert result["crops"] == {"review": 1}
    assert not (directory / "train_hq.txt").read_text()
    assert not (directory / "validation_hq.txt").read_text()
    assert not records(directory, "review")[0]["variants"]


def test_stale_glyph_and_ocr_artifacts_fail_closed(tmp_path, config):
    sample = make_record(tmp_path)
    sample["glyph_checks"]["crop_0000"]["image_sha256"] = "stale"
    result = Pipeline(config, write_manifest(tmp_path, [sample]), tmp_path / "output").run()
    assert result["crops"] == {"error": 1}


def test_shared_group_stays_in_one_split(tmp_path, config):
    samples = [make_record(tmp_path, "first", "white", "magazine-issue-1"),
               make_record(tmp_path, "second", "yellow", "magazine-issue-1")]
    directory = tmp_path / "output"
    Pipeline(config, write_manifest(tmp_path, samples), directory).run()
    accepted = records(directory, "accepted")
    assert len(accepted) == 2
    assert len({x["split"] for x in accepted}) == 1


def test_changed_source_pixels_do_not_overwrite_committed_output(tmp_path, config):
    sample = make_record(tmp_path)
    input_path = write_manifest(tmp_path, [sample])
    directory = tmp_path / "output"
    Pipeline(config, input_path, directory).run()
    before = (directory / "accepted.jsonl").read_bytes()
    Image.new("RGB", (64, 32), "red").save(tmp_path / sample["image_path"])
    with pytest.raises(InputIntegrityError, match="pixels"):
        Pipeline(config, input_path, directory, resume=True).run()
    assert (directory / "accepted.jsonl").read_bytes() == before


def test_missing_committed_source_does_not_destroy_previous_result(tmp_path, config):
    sample = make_record(tmp_path)
    input_path = write_manifest(tmp_path, [sample])
    directory = tmp_path / "output"
    Pipeline(config, input_path, directory).run()
    before = (directory / "accepted.jsonl").read_bytes()
    (tmp_path / sample["image_path"]).unlink()
    with pytest.raises(InputIntegrityError, match="cannot verify"):
        Pipeline(config, input_path, directory, resume=True).run()
    assert (directory / "accepted.jsonl").read_bytes() == before


def test_exported_hq_list_loads_with_upstream_vosr_dataset(tmp_path, config):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from dataloaders.realsr_dataset import TxtPairDataset
    sample = make_record(tmp_path)
    directory = tmp_path / "output"
    Pipeline(config, write_manifest(tmp_path, [sample]), directory).run()
    accepted = records(directory, "accepted")[0]
    arguments = SimpleNamespace(resolution=16,
                                train_dataset_txt_paths_list=[str(directory / (accepted["split"] + "_hq.txt"))],
                                train_dataset_prob_paths_list=[1])
    dataset = TxtPairDataset(split="train", args=arguments)
    assert len(dataset) == 1
    item = dataset[0]
    assert set(item) == {"hq"}
    assert item["hq"].shape == (3, 16, 16)
    assert item["hq"].dtype == torch.float32


def test_duplicate_input_ids_fail_without_overwriting_first(tmp_path, config):
    sample = make_record(tmp_path)
    input_path = write_manifest(tmp_path, [sample, sample])
    with pytest.raises(InputIntegrityError, match="duplicate"):
        Pipeline(config, input_path, tmp_path / "output").run()


def test_demo_does_not_export_training_hq_or_pairs(tmp_path):
    result = create_demo(tmp_path)
    assert result["crops"] == {"review": 1}
    review = records(tmp_path / "output", "review")[0]
    assert "demo_backend" in review["reasons"]
    assert len(review["variants"]) == 2
    assert not (tmp_path / "output/train_hq.txt").read_text()
    assert not (tmp_path / "output/pairs_train.jsonl").read_text()


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract executable unavailable")
def test_real_tesseract_line_detection_smoke():
    image = Image.new("RGB", (640, 160), "white")
    ImageDraw.Draw(image).text((30, 45), "TEXT SR 2026", font=ImageFont.load_default(size=52), fill="black")
    output = regions(TesseractOCR(psm=6).recognize(image, {}), image.size)
    assert len(output) == 1
    assert "2026" in output[0]["text"]
    assert 0 < output[0]["confidence"] <= 1
