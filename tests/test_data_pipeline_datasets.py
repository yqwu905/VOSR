import io
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from data_pipeline.backends import TesseractOCR
from data_pipeline.datasets import convert_annotations, sample_parquet, reject_known_misalignment, import_anyword_json
from data_pipeline.geometry import regions
from data_pipeline.vosr_teacher import VOSR2Teacher
from data_pipeline.visualize import annotation_errors


def test_easytext_caption_is_not_a_transcript():
    row = {"text": "A sign displaying <sks1>", "position": "[[[[1,2],[19,2],[19,9],[1,9]], [[0,0],[90,20]]]]"}
    annotations, invalid = convert_annotations("easytext", row, (20, 10), trust_annotations=True)
    assert not invalid
    assert len(annotations) == 1
    assert "text" not in annotations[0]
    assert not annotations[0]["trusted"]


def test_anyword_invalid_polygon_is_audited_without_clipping():
    row = {"annotations": [{"polygon": [[-1,0],[10,0],[10,9],[0,9]], "text": "bad"},
                           {"polygon": [[1,1],[8,1],[8,8],[1,8]], "text": "ok", "rec_score": .99},
                           {"polygon": [[1,1],[8,1],[8,8],[1,8]], "text": "###", "illegibility": True}]}
    annotations, invalid = convert_annotations("anyword3m", row, (10, 10))
    assert len(invalid) == 2
    assert invalid[0]["annotation"]["polygon"][0][0] == -1
    assert len(annotations) == 1
    assert annotations[0]["dataset_rec_score"] == .99
    assert not annotations[0]["trusted"]


def test_known_bad_anyword_art_mirror_fails_before_import():
    with pytest.raises(ValueError, match="mismatched"):
        reject_known_misalignment("stzhao/AnyWord-3M", "54f67ddf34eb4a5aac46539c2a5b511dd0ca1fc4", ["OCR_Art"])


@pytest.mark.parametrize("size,cap", [((257,193), None), ((1,1), None), ((513,127), 1024)])
def test_vosr_padding_preserves_target_size_and_rng(size, cap):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    teacher = VOSR2Teacher.__new__(VOSR2Teacher)
    teacher.args = SimpleNamespace(patch_size=2, tile_size=512, align_method="nofix")
    teacher.device, teacher.upscale = "cpu", 2
    teacher.precision, teacher.dtype = "fp32", torch.float32
    teacher.max_output_edge = cap
    teacher.dit = teacher.vae = teacher.venc = None
    def inference(dit, vae, venc, tensor, args, device):
        assert tensor.shape[-1] % 16 == tensor.shape[-2] % 16 == 0
        return tensor
    teacher.inference = SimpleNamespace(tiled_latent_inference=inference)
    image = Image.new("RGB", size, "white")
    before = torch.random.get_rng_state().clone()
    restored = teacher.restore(image, {"seed": 123})
    assert torch.equal(before, torch.random.get_rng_state())
    factor = 2 if cap is None else max(1, min(2, cap / max(size)))
    assert restored.size == tuple(round(value * factor) for value in size)
    np.testing.assert_array_equal(np.asarray(restored), 255)


def test_parquet_sampling_reads_binary_images_across_shards(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
    rows = [{"filename": f"{i}.png", "image": buffer.getvalue(), "text": "caption <sks1>",
             "position": "[[[[1,1],[18,1],[18,8],[1,8]], [[0,0],[30,20]]]]"} for i in range(8)]
    files = [tmp_path / "one.parquet", tmp_path / "two.parquet"]
    pq.write_table(pa.Table.from_pylist(rows[:4]), files[0])
    pq.write_table(pa.Table.from_pylist(rows[4:]), files[1])
    first = sample_parquet("easytext", files, tmp_path / "a", count=6, seed=42)
    second = sample_parquet("easytext", files, tmp_path / "b", count=6, seed=42)
    assert first["indices"] == second["indices"]
    assert len({path for path, index in first["indices"]}) == 2
    records = [json.loads(line) for line in (tmp_path / "a/easytext.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert all("text" not in record["annotations"][0] for record in records)


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract unavailable")
def test_region_ocr_reads_pixels_instead_of_annotation_text():
    image = Image.new("RGB", (460, 100), "white")
    ImageDraw.Draw(image).text((20, 15), "TEXT SR 2026", fill="black", font=ImageFont.load_default(size=48))
    annotations = regions([{"bbox": [10,10,440,85], "text": "invented reference", "trusted": True}], image.size)
    output = TesseractOCR(region_mode=True).recognize(image, {"regions": annotations})
    assert len(output) == 1
    assert "2026" in output[0]["text"]
    assert "invented" not in output[0]["text"]
    assert 0 < output[0]["confidence"] <= 1


def test_transcript_cer_counts_missing_text():
    annotation = {"bbox": [0,0,10,10], "text": "ABCD"}
    result = annotation_errors([], [annotation])
    assert result["cer"] == 1
    result = annotation_errors([dict(annotation, text="ABXD")], [annotation])
    assert result["cer"] == .25


def test_official_json_sampling_is_reproducible_and_keeps_provenance(tmp_path):
    pytest.importorskip("ijson")
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for index in range(8):
        name = f"{index}.png"
        Image.new("RGB", (20,10), (index,0,0)).save(images / name)
        rows.append({"img_name": name, "annotations": [{"bbox": [1,1,19,9], "text": str(index)}]})
    annotations = tmp_path / "data.json"
    annotations.write_text(json.dumps({"data_root": "ignored", "data_list": rows}))
    first = import_anyword_json(annotations, images, tmp_path / "a.jsonl", True, count=4, seed=42)
    second = import_anyword_json(annotations, images, tmp_path / "b.jsonl", True, count=4, seed=42)
    assert first["indices"] == second["indices"]
    records = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert all(record["annotations"][0]["trusted"] for record in records)
    assert all(record["dataset"]["annotation_sha256"] == first["annotation_sha256"] for record in records)
