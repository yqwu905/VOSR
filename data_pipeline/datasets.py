"""Import native text annotations and reproducible, bounded HF viewer samples.

The viewer's /rows API avoids reading multi-GB Parquet row groups for a small
experiment. Sample indices are uniform within each selected subset, rather than
sampling only the first shard. Cached image assets must match a pinned revision.
"""

import ast
import hashlib
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .geometry import regions
from .pipeline import atomic_json, json_text, save_image


SOURCES = {
    "anyword3m": {"repo": "stzhao/AnyWord-3M", "revision": "54f67ddf34eb4a5aac46539c2a5b511dd0ca1fc4",
                  "subsets": ["laion", "OCR_LSVT"]},
    "easytext": {"repo": "lllrrnn/EasyText", "revision": "6997bd59c9645f9a0dbf024db3c3052a2dd8074c",
                 "subsets": ["default"]},
}


def reject_known_misalignment(repo, revision, subsets):
    # Independently reproduced with gt_886.jpg and other sampled rows. The mirror
    # also records this defect at https://huggingface.co/datasets/stzhao/AnyWord-3M/discussions/2.
    if (repo == "stzhao/AnyWord-3M" and revision == SOURCES["anyword3m"]["revision"]
            and "OCR_Art" in subsets):
        raise ValueError("this AnyWord-3M HF OCR_Art revision has mismatched images/annotations; "
                         "use official ModelScope ocr_data/Art with import-anyword instead")


def structured(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def convert_annotations(source, row, size, trust_annotations=False):
    """Descriptions are never transcripts; invalid polygons remain in the audit."""
    valid, invalid = [], []
    if source == "anyword3m":
        raw = structured(row.get("annotations", []))
    elif source == "easytext":
        raw = [{"polygon": position[0], "condition_bbox": position[1]}
               for position in structured(row.get("position", []))]
    else:
        raise ValueError(f"unsupported dataset: {source}")
    for index, item in enumerate(raw):
        if item.get("valid") is False or item.get("illegibility") is True:
            invalid.append({"index": index, "reason": "dataset_invalid", "annotation": item})
            continue
        candidate = {key: item[key] for key in ("polygon", "bbox", "text", "language") if key in item}
        candidate["trusted"] = bool(trust_annotations and candidate.get("text"))
        candidate["annotation_origin"] = "dataset_annotation" if trust_annotations else "dataset_localization"
        if "rec_score" in item:
            candidate["dataset_rec_score"] = item["rec_score"]
        if "condition_bbox" in item:
            candidate["condition_bbox"] = item["condition_bbox"]
        try:
            valid.extend(regions([candidate], size))
        except (ValueError, KeyError, TypeError) as error:
            invalid.append({"index": index, "reason": str(error), "annotation": item})
    return valid, invalid


def _request(url, params=None):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))))
        result = session.get(url, params=params, timeout=(15, 90))
        result.raise_for_status()
        return result


def sample_dataset(source, output, count=12, seed=42, subsets=None, revision=None, workers=4,
                   trust_annotations=False):
    if source not in SOURCES:
        raise ValueError(f"unsupported dataset: {source}")
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    spec = SOURCES[source]
    revision = revision or spec["revision"]
    subsets = subsets or spec["subsets"]
    reject_known_misalignment(spec["repo"], revision, subsets)
    if len(set(subsets)) != len(subsets) or not subsets:
        raise ValueError("subsets must be unique and nonempty")
    output = Path(output).resolve()
    manifest = output / f"{source}.jsonl"
    if manifest.exists() or (output / f"{source}_sampling.json").exists():
        raise ValueError(f"dataset sample already exists: {manifest}")
    output.mkdir(parents=True, exist_ok=True)
    selected, subset_sizes = [], {}
    for position, subset in enumerate(subsets):
        params = {"dataset": spec["repo"], "config": subset, "split": "train", "offset": 0, "length": 1}
        total = _request("https://datasets-server.huggingface.co/rows", params).json()["num_rows_total"]
        subset_sizes[subset] = total
        quota = count // len(subsets) + (position < count % len(subsets))
        if quota > total:
            raise ValueError(f"requested {quota} samples from {subset}, which only has {total}")
        subset_seed = int(hashlib.sha256(f"{seed}:{source}:{subset}".encode()).hexdigest()[:16], 16)
        selected.extend((subset, index) for index in sorted(random.Random(subset_seed).sample(range(total), quota)))

    def fetch(selection):
        subset, index = selection
        params = {"dataset": spec["repo"], "config": subset, "split": "train", "offset": index, "length": 1}
        result = _request("https://datasets-server.huggingface.co/rows", params).json()
        row = result["rows"][0]["row"]
        image_asset = row["image"]
        # The viewer follows main. Refuse silent drift instead of mislabelling a sample.
        if f"/--/{revision}/--/" not in image_asset["src"]:
            raise ValueError("HF viewer revision changed; use the matching pinned revision or local data")
        content = _request(image_asset["src"]).content
        with Image.open(io.BytesIO(content)) as decoded:
            if getattr(decoded, "n_frames", 1) != 1:
                raise ValueError("multi-frame dataset image")
            image = decoded.convert("RGB")
        annotations, invalid = convert_annotations(source, row, image.size, trust_annotations)
        name = str(row.get("img_name", row.get("filename", index)))
        identity = f"{subset}/{index}/{name}"
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        path = output / "images" / source / subset / f"{key}.png"
        save_image(image, path)
        return {"id": identity, "source": source, "group_id": f"{source}/{subset}/{name}",
                "image_path": str(path), "annotations": annotations,
                "dataset": {"repo": spec["repo"], "revision": revision, "subset": subset, "split": "train",
                            "row_index": index, "filename": name, "asset_transport": "hf_viewer_rgb",
                            "download_sha256": hashlib.sha256(content).hexdigest(),
                            "original_size": list(image.size), "invalid_annotations": invalid,
                            "caption": row.get("caption", row.get("text", ""))}}

    temporary = manifest.with_suffix(".jsonl.tmp")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, temporary.open("w", encoding="utf-8") as handle:
            for record in pool.map(fetch, selected):
                handle.write(json_text(record) + "\n")
        temporary.replace(manifest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    summary = {"source": source, "repo": spec["repo"], "revision": revision, "seed": seed,
               "count": len(selected), "subset_sizes": subset_sizes, "indices": selected,
               "sampling": "uniform_without_replacement_per_subset", "manifest": str(manifest)}
    atomic_json(output / f"{source}_sampling.json", summary)
    return summary


def sample_parquet(source, files, output, count=12, seed=42, trust_annotations=False,
                   repo=None, revision=None, subset="local"):
    """Uniform sampling over ALL rows in supplied native Parquet files, bounded RAM."""
    import pyarrow.parquet as pq
    from .pipeline import file_digest
    reject_known_misalignment(repo, revision, [subset])
    files = sorted({Path(path).resolve() for path in files})
    if not files or count < 1:
        raise ValueError("provide Parquet files and a positive sample count")
    counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
    total = sum(counts)
    if count > total:
        raise ValueError("sample count exceeds supplied Parquet rows")
    indices = sorted(random.Random(seed).sample(range(total), count))
    output = Path(output).resolve()
    manifest = output / f"{source}.jsonl"
    if manifest.exists() or (output / f"{source}_sampling.json").exists():
        raise ValueError("dataset sample already exists")
    output.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.tmp")
    offset, selected = 0, []
    with temporary.open("w", encoding="utf-8") as handle:
        for path, rows in zip(files, counts):
            wanted = {index - offset for index in indices if offset <= index < offset + rows}
            offset += rows
            if not wanted:
                continue
            digest = file_digest(path)
            row_offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for local in sorted(i - row_offset for i in wanted if row_offset <= i < row_offset + len(batch)):
                    row = batch.slice(local, 1).to_pylist()[0]
                    asset = row["image"]
                    content = asset["bytes"] if isinstance(asset, dict) else asset
                    if content is None:
                        raise ValueError("native Parquet must contain embedded image bytes")
                    with Image.open(io.BytesIO(content)) as decoded:
                        if getattr(decoded, "n_frames", 1) != 1:
                            raise ValueError("multi-frame dataset image")
                        image = decoded.convert("RGB")
                    annotations, invalid = convert_annotations(source, row, image.size, trust_annotations)
                    name = str(row.get("img_name", row.get("filename", row_offset + local)))
                    identity = f"{subset}/{path.name}/{row_offset + local}/{name}"
                    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
                    target = output / "images" / source / subset / f"{key}.png"
                    save_image(image, target)
                    record = {"id": identity, "source": source, "image_path": str(target),
                              "group_id": f"{source}/{subset}/{name}", "annotations": annotations,
                              "dataset": {"repo": repo, "revision": revision, "subset": subset,
                                          "parquet_path": str(path), "parquet_sha256": digest,
                                          "row_index": row_offset + local, "filename": name,
                                          "original_size": list(image.size), "asset_transport": "native_parquet",
                                          "invalid_annotations": invalid,
                                          "caption": row.get("caption", row.get("text", ""))}}
                    handle.write(json_text(record) + "\n")
                    selected.append([str(path), row_offset + local])
                row_offset += len(batch)
                if row_offset > max(wanted):
                    break
    if len(selected) != count:
        temporary.unlink()
        raise ValueError("Parquet sample count mismatch")
    temporary.replace(manifest)
    summary = {"source": source, "repo": repo, "revision": revision, "seed": seed, "count": count,
               "subset": subset, "total_rows": total, "files": [str(path) for path in files],
               "indices": selected, "sampling": "uniform_without_replacement_over_supplied_parquet",
               "manifest": str(manifest)}
    atomic_json(output / f"{source}_sampling.json", summary)
    return summary


def import_anyword_json(annotation_file, image_root, output, trust_annotations=False,
                       count=None, seed=42):
    """Stream the official AnyWord data_list JSON after extracting its image archives."""
    import ijson
    from .backends import read_rgb
    from .pipeline import file_digest
    image_root = Path(image_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("import output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    selected = None
    if count is not None:
        if count < 1:
            raise ValueError("sample count must be positive")
        selected, rng, total = [], random.Random(seed), 0
        with Path(annotation_file).open("rb") as source:
            for index, row in enumerate(ijson.items(source, "data_list.item", use_float=True)):
                total += 1
                if len(selected) < count:
                    selected.append((index, row))
                else:
                    slot = rng.randrange(total)
                    if slot < count:
                        selected[slot] = (index, row)
        if len(selected) < count:
            raise ValueError("sample count exceeds AnyWord records")
        selected.sort(key=lambda pair: pair[0])
    annotation_digest = file_digest(annotation_file)
    indexed = 0
    with Path(annotation_file).open("rb") as source, temporary.open("w", encoding="utf-8") as handle:
        stream = selected if selected is not None else enumerate(ijson.items(source, "data_list.item", use_float=True))
        for index, row in stream:
            path = image_root / row["img_name"]
            size = read_rgb(path).size
            annotations, invalid = convert_annotations("anyword3m", row, size, trust_annotations)
            record = {"id": row["img_name"], "source": "anyword3m", "image_path": str(path),
                      "group_id": f"anyword3m/{row['img_name']}", "annotations": annotations,
                      "dataset": {"annotation_file": str(Path(annotation_file).resolve()),
                                  "annotation_sha256": annotation_digest, "row_index": index,
                                  "filename": row["img_name"], "caption": row.get("caption", ""),
                                  "original_size": list(size), "asset_transport": "official_native_json",
                                  "invalid_annotations": invalid}}
            handle.write(json_text(record) + "\n")
            indexed += 1
    if indexed == 0:
        temporary.unlink()
        raise ValueError("AnyWord JSON has no data_list records")
    temporary.replace(output)
    result = {"indexed": indexed, "manifest": str(output), "annotation_sha256": annotation_digest}
    if selected is not None:
        result.update(seed=seed, total_rows=total, indices=[index for index, row in selected],
                      sampling="reservoir_uniform_without_replacement")
        atomic_json(output.with_suffix(".sampling.json"), result)
    return result
