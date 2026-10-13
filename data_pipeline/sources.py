"""Catalog of surveyed text-image sources and a parquet sampler for original bytes.

The HF dataset viewer serves every image as a JPEG re-encode (verified: FUNSD
PNGs come back as JPEG with pixel errors up to 46/255), so SR candidates are
read from the auto-converted parquet instead: a random file, row group and row
via HTTP range requests, keeping the stored bytes and their container format.
"""

import hashlib
import json
import random
from pathlib import Path

from .datasets import convert_annotations, structured
from .geometry import regions
from .pipeline import atomic_json, json_text


def quad(points):
    return [[float(x), float(y)] for x, y in points]


def ocr_info(row, size):
    """TextVQA-style Rosetta OCR: normalized axis-aligned word boxes, untrusted."""
    output = []
    for item in structured(row.get("ocr_info") or []):
        box = item["bounding_box"]
        x0, y0 = box["top_left_x"] * size[0], box["top_left_y"] * size[1]
        output.append({"bbox": [x0, y0, x0 + box["width"] * size[0], y0 + box["height"] * size[1]],
                       "text": item.get("word", ""), "annotation_origin": "dataset_ocr"})
    return output


def cord(row, size):
    output = []
    for line in structured(row["ground_truth"]).get("valid_line", []):
        for word in line.get("words", []):
            q = word["quad"]
            output.append({"polygon": quad([(q["x1"], q["y1"]), (q["x2"], q["y2"]),
                                            (q["x3"], q["y3"]), (q["x4"], q["y4"])]),
                           "text": word.get("text", ""), "annotation_origin": "dataset_annotation"})
    return output


def words_bboxes(scale):
    def adapter(row, size):
        sx, sy = (size[0] / 1000, size[1] / 1000) if scale == "per_mille" else (1, 1)
        return [{"bbox": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], "text": text,
                 "annotation_origin": "dataset_annotation"} for text, b in zip(row["words"], row["bboxes"])]
    return adapter


def anyword(row, size):
    if row.get("annotations") is None:
        return [{"invalid": {"reason": "annotations_null"}}]
    valid, invalid = convert_annotations("anyword3m", row, size)
    return valid + [{"invalid": item} for item in invalid]


ADAPTERS = {"anyword": anyword, "ocr_info": ocr_info, "cord": cord, "none": lambda row, size: [],
            "funsd": words_bboxes("per_mille"), "sroie": words_bboxes("pixels")}
# Only these columns are fetched: captions, HTML or condition images can dwarf the image.
ADAPTER_COLUMNS = {"anyword": ["annotations"], "ocr_info": ["ocr_info"], "cord": ["ground_truth"],
                   "none": [], "funsd": ["words", "bboxes"], "sroie": ["words", "bboxes"]}

# category: scene / document / born_digital / synthetic / generative / handwriting / word_crop.
CATALOG = {
    "anyword_lsvt": dict(repo="stzhao/AnyWord-3M", config="OCR_LSVT", adapter="anyword", category="scene",
                         language="zh", license="apache-2.0 (mirror); LSVT upstream terms"),
    "anyword_rects": dict(repo="stzhao/AnyWord-3M", config="OCR_ReCTS", adapter="anyword", category="scene",
                          language="zh", license="apache-2.0 (mirror); ReCTS upstream terms"),
    "anyword_mtwi": dict(repo="stzhao/AnyWord-3M", config="OCR_MTWI2018", adapter="anyword", category="scene",
                         language="zh", license="apache-2.0 (mirror); MTWI upstream terms"),
    "anyword_cocotext": dict(repo="stzhao/AnyWord-3M", config="OCR_COCO_Text", adapter="anyword",
                             category="scene", language="en", license="apache-2.0 (mirror); COCO CC-BY"),
    "anyword_mlt": dict(repo="stzhao/AnyWord-3M", config="OCR_mlt2019", adapter="anyword", category="scene",
                        language="multi", license="apache-2.0 (mirror); MLT research-only"),
    "anyword_laion": dict(repo="stzhao/AnyWord-3M", config="laion", adapter="anyword", category="web",
                          language="en", license="apache-2.0 (mirror); LAION web images, per-image rights"),
    "anyword_wukong": dict(repo="stzhao/AnyWord-3M", config="wukong_1of5", adapter="anyword", category="web",
                           language="zh", license="apache-2.0 (mirror); Wukong web images, per-image rights"),
    "cocotext": dict(repo="howard-hou/COCO-Text", adapter="ocr_info", category="scene", language="en",
                     license="none tagged; COCO CC-BY"),
    "icdar2015": dict(repo="dlxjj/ICDAR2015", adapter="none", category="scene", language="en",
                      license="none tagged; ICDAR research-only"),
    "totaltext": dict(repo="Voxel51/Total-Text-Dataset", adapter="none", category="scene", language="en",
                      license="bsd-3-clause"),
    "textvqa": dict(repo="lmms-lab-encoder/textvqa", adapter="none", category="scene", language="en",
                    license="none tagged; OpenImages CC-BY 2.0", dedupe="image_id"),
    "cord": dict(repo="naver-clova-ix/cord-v2", adapter="cord", category="document", language="id/en",
                 license="cc-by-4.0"),
    "funsd": dict(repo="nielsr/funsd", adapter="funsd", category="document", language="en",
                  license="none tagged; FUNSD research-only"),
    "sroie": dict(repo="jsdnrs/ICDAR2019-SROIE", adapter="sroie", category="document", language="en",
                  license="cc-by-4.0"),
    "docvqa": dict(repo="HuggingFaceM4/DocumentVQA", adapter="none", category="document", language="en",
                   license="none tagged; UCSF IDL", dedupe="docId"),
    "ocrvqa": dict(repo="howard-hou/OCR-VQA", adapter="ocr_info", category="born_digital", language="en",
                   license="none tagged; Amazon covers"),
    "websight": dict(repo="HuggingFaceM4/WebSight", config="v0.2", adapter="none", category="born_digital",
                     language="en", license="cc-by-4.0"),
    "chartqa": dict(repo="HuggingFaceM4/ChartQA", adapter="none", category="born_digital", language="en",
                    license="gpl-3.0", dedupe="image_sha256"),
    "textatlas_ppt": dict(repo="CSU-JPG/TextAtlas5M", config="PPT2Structured", adapter="none",
                          category="born_digital", language="en", license="mit"),
    "textatlas_paper": dict(repo="CSU-JPG/TextAtlas5M", config="Paper2Text", adapter="none",
                            category="born_digital", language="en", license="mit"),
    "textatlas_cover": dict(repo="CSU-JPG/TextAtlas5M", config="CoverBook", adapter="none",
                            category="born_digital", language="en", license="mit"),
    "textatlas_scenes": dict(repo="CSU-JPG/TextAtlas5M", config="TextScenesHQ", adapter="none",
                             category="scene", language="en", license="mit"),
    "synthdog_zh": dict(repo="naver-clova-ix/synthdog-zh", adapter="none", category="synthetic",
                        language="zh", license="none tagged; Donut MIT"),
    "synthdog_en": dict(repo="naver-clova-ix/synthdog-en", adapter="none", category="synthetic",
                        language="en", license="none tagged; Donut MIT"),
    "easytext": dict(repo="lllrrnn/EasyText", adapter="none", category="generative", language="multi",
                     license="mit"),
    "iam_line": dict(repo="Teklia/IAM-line", adapter="none", category="handwriting", language="en",
                     license="mit"),
    "mjsynth": dict(repo="priyank-m/MJSynth_text_recognition", adapter="none", category="word_crop",
                    language="en", license="none tagged"),
}


def parquet_files(repo, config, split):
    """Auto-converted parquet shards and the pinned conversion commit."""
    import requests
    from huggingface_hub import HfApi
    listing = requests.get("https://datasets-server.huggingface.co/parquet", params={"dataset": repo},
                           timeout=60).json()["parquet_files"]
    files = [f for f in listing if f["split"] == split and (config is None or f["config"] == config)]
    if not files:
        raise ValueError(f"{repo}: no parquet files for config={config} split={split}")
    revision = next(ref.target_commit for ref in HfApi().list_repo_refs(repo, repo_type="dataset").converts
                    if ref.name == "parquet")
    prefix = "refs%2Fconvert%2Fparquet/"
    paths = [f"datasets/{repo}@{revision}/{f['url'].split(prefix, 1)[1]}" for f in files]
    return paths, revision


def sample_source(name, output, count=24, seed=42, split="train", spec=None, rows_per_group=4):
    """Sample ``count`` distinct images from random shards and row groups.

    A row group is the smallest unit a range request can fetch, so up to
    ``rows_per_group`` random rows are taken from each one fetched. Shards are
    chosen uniformly, so rows of small shards are slightly favoured; exact
    positions are recorded for reproduction.
    """
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image
    spec = spec or CATALOG[name]
    output = Path(output).resolve()
    manifest = output / f"{name}.jsonl"
    if manifest.exists():
        raise ValueError(f"sample already exists: {manifest}")
    paths, revision = parquet_files(spec["repo"], spec.get("config"), split)
    rng = random.Random(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16))
    fs, seen, records, groups = HfFileSystem(), set(), [], 0
    adapter = ADAPTERS[spec["adapter"]]
    while len(records) < count:
        groups += 1
        if groups > count * 4:
            raise ValueError(f"{name}: could not find {count} distinct images")
        path = rng.choice(paths)
        parquet = pq.ParquetFile(fs.open(path, "rb", block_size=4 * 1024 * 1024))
        names = parquet.schema_arrow.names
        column = next(c for c in ("image", "jpg", "png") if c in names)
        wanted = [column] + ADAPTER_COLUMNS[spec["adapter"]]
        if spec.get("dedupe") in names:
            wanted.append(spec["dedupe"])
        group = rng.randrange(parquet.num_row_groups)
        table = parquet.read_row_group(group, columns=wanted)
        for index in sorted(rng.sample(range(table.num_rows), min(rows_per_group, table.num_rows))):
            if len(records) >= count:
                break
            row = table.slice(index, 1).to_pylist()[0]
            data = row[column]["bytes"] if isinstance(row[column], dict) else row[column]
            digest = hashlib.sha256(data).hexdigest()
            key = str(row.get(spec.get("dedupe"), digest))
            if digest in seen or key in seen:
                continue
            seen.update({digest, key})
            with Image.open(io.BytesIO(data)) as image:
                size, fmt = image.size, (image.format or "bin").lower()
            raw = adapter(row, size)
            invalid = [item["invalid"] for item in raw if "invalid" in item]
            annotations = []
            for item in (item for item in raw if "invalid" not in item):
                try:
                    annotations.extend(regions([item], size))
                except (ValueError, KeyError, TypeError) as error:
                    invalid.append({"reason": str(error), "annotation": item})
            target = output / "images" / name / f"{digest[:24]}.{'jpg' if fmt == 'jpeg' else fmt}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records.append({"id": f"{name}/{digest[:24]}", "source": name, "image_path": str(target),
                            "group_id": f"{name}/{key}", "annotations": annotations,
                            "dataset": {"repo": spec["repo"], "config": spec.get("config"), "split": split,
                                        "revision": revision, "parquet": path, "row_group": group, "row": index,
                                        "stored_format": fmt, "stored_bytes_sha256": digest,
                                        "original_size": list(size), "category": spec["category"],
                                        "license": spec["license"], "invalid_annotations": invalid,
                                        "asset_transport": "parquet_original_bytes"}})
    output.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json_text(r) + "\n" for r in records), encoding="utf-8")
    temporary.replace(manifest)
    summary = {"source": name, "repo": spec["repo"], "config": spec.get("config"), "revision": revision,
               "seed": seed, "count": len(records), "row_groups_read": groups, "manifest": str(manifest),
               "sampling": f"uniform shard and row group, up to {rows_per_group} random rows per group"}
    atomic_json(output / f"{name}_sampling.json", summary)
    return summary


def catalog_json():
    return json.dumps(CATALOG, ensure_ascii=False, indent=2)
