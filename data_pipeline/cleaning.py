"""Cleaning before the teacher: probe, deduplicate, route and materialize HR candidates.

``probe`` measures every manifest image with model-free probes, optionally
localizes text with a detector where a source has no boxes, removes near
duplicates (within the run and against benchmark manifests), and routes each
image to direct HR, HR by area downscaling, teacher pseudo-GT, or rejection.
``--materialize`` writes the cleaned images: ``clean.jsonl`` feeds the
six-stage pipeline with the identity teacher, ``teacher.jsonl`` with VOSR2.
"""

import hashlib
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import regions, scale_matrix, transform
from .pipeline import atomic_json, json_text, resolve_input, save_image
from .probes import hamming, phash, probe_image, route


class RapidOCRDetector:
    """PP-OCRv4 detection through rapidocr_onnxruntime (CPU), boxes only."""

    def __init__(self, box_thresh=0.5):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.box_thresh = box_thresh

    def detect(self, image):
        # A PIL input is converted RGB->BGR by RapidOCR; an ndarray would be taken as BGR.
        boxes, _ = self.engine(image, use_det=True, use_cls=False, use_rec=False, box_thresh=self.box_thresh)
        return [{"polygon": box, "annotation_origin": "detector"} for box in boxes or []]


def load_detector(name):
    if not name:
        return None
    if name == "rapidocr":
        return RapidOCRDetector()
    module, attribute = name.split(":", 1)
    return getattr(importlib.import_module(module), attribute)()


def duplicate_pairs(hashes, bits):
    """Pairs within ``bits`` Hamming distance. With 8 bands of 8 bits, any pair
    differing in at most 7 bits agrees on a whole band (pigeonhole)."""
    if bits > 7:
        raise ValueError("near-duplicate search supports at most 7 differing bits")
    buckets = defaultdict(list)
    for index, value in enumerate(hashes):
        for band in range(8):
            buckets[(band, value[2 * band:2 * band + 2])].append(index)
    pairs = set()
    for members in buckets.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if hamming(hashes[a], hashes[b]) <= bits:
                    pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def benchmark_hashes(paths):
    hashes = []
    for path in paths or []:
        path = Path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                with Image.open(resolve_input(record, path.parent, "image_path")) as image:
                    hashes.append((record.get("id"), phash(np.asarray(image.convert("RGB")))))
    return hashes


def probe_manifest(input_path, output, rules=None, detector=None, exclude=None, materialize=False):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "probes.jsonl").exists():
        raise ValueError("probe output already exists; choose a new directory")
    detector = load_detector(detector)
    rules = rules or {}
    rows = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        path = resolve_input(record, input_path.parent, "image_path")
        result = {"id": record["id"], "source": record["source"], "image_path": str(path),
                  "line_number": line_number}
        try:
            data = path.read_bytes()
            with Image.open(path) as image:
                size = image.size
                annotations = regions(record.get("annotations", []), size)
                origin = "manifest" if annotations else None
                if not annotations and detector is not None:
                    annotations = regions(detector.detect(image.convert("RGB")), size)
                    origin = "detector" if annotations else None
            stats = probe_image(data, annotations)
            result.update(stats=stats, annotation_origin=origin, route=route(stats, rules),
                          sha256=hashlib.sha256(data).hexdigest())
            if detector is not None and origin == "detector":
                result["detected_regions"] = annotations
            json_text(result)  # non-finite measurements fail inside this record's boundary
        except Exception as error:  # one bad file must not stop a survey
            result = {key: result[key] for key in ("id", "source", "image_path", "line_number")}
            result.update(route={"decision": "reject", "reasons": [f"error:{type(error).__name__}"],
                                 "fixes": [], "degradations": {}, "downscale": 1.0}, error=str(error))
        rows.append(result)
    bits = rules.get("near_duplicate_bits", 6)
    hashes = [row.get("stats", {}).get("phash") for row in rows]
    valid = [i for i, value in enumerate(hashes) if value]
    def reject(i, reason, **extra):
        if reason not in rows[i]["route"]["reasons"]:
            rows[i]["route"] = {**rows[i]["route"], "decision": "reject",
                                "reasons": rows[i]["route"]["reasons"] + [reason], **extra}

    for a, b in duplicate_pairs([hashes[i] for i in valid], bits):
        a, b = valid[a], valid[b]
        # Keep the larger image of a near-duplicate pair.
        keep, drop = sorted((a, b), key=lambda i: -np.prod(rows[i]["stats"]["size"]))
        reject(drop, "near_duplicate", duplicate_of=rows[keep]["id"])
    for benchmark_id, value in benchmark_hashes(exclude):
        for i in valid:
            if hamming(hashes[i], value) <= bits:
                reject(i, "benchmark_overlap", benchmark_id=benchmark_id)
    with (output / "probes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_text(row) + "\n")
    summary = summarize(rows)
    if materialize:
        summary["materialized"] = materialize_routes(input_path, rows, output)
    atomic_json(output / "summary.json", summary)
    return summary


METRICS = ("effective_upscale", "edge_width", "text_edge_width", "noise_sigma", "text_noise_sigma", "periodic_peak",
           "padding_fraction", "text_height_median", "text_coverage", "luma_range")


def summarize(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    summary = {}
    for source, items in sorted(by_source.items()):
        stats = [x["stats"] for x in items if "stats" in x]
        entry = {"images": len(items),
                 "decisions": dict(Counter(x["route"]["decision"] for x in items)),
                 "reasons": dict(Counter(r for x in items for r in x["route"]["reasons"])),
                 "fixes": dict(Counter(r for x in items for r in x["route"].get("fixes", []))),
                 "degradations": dict(Counter(r for x in items for r in x["route"].get("degradations", {}))),
                 "formats": dict(Counter(s.get("format") for s in stats)),
                 "jpeg_container_quality": quantiles([s.get("jpeg_quality") for s in stats]),
                 "jpeg_history_quality": quantiles([(s.get("jpeg_history") or {}).get("quality") for s in stats]),
                 "jpeg_any": sum(1 for s in stats if s.get("jpeg_quality") or s.get("jpeg_history")),
                 "short_edge": quantiles([min(s["size"]) for s in stats]),
                 "long_edge": quantiles([max(s["size"]) for s in stats])}
        for metric in METRICS:
            entry[metric] = quantiles([s.get(metric) for s in stats])
        summary[source] = entry
    return {"sources": summary, "images": len(rows),
            "decisions": dict(Counter(x["route"]["decision"] for x in rows))}


def quantiles(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"n": len(values), "p10": round(float(p10), 4), "median": round(float(p50), 4),
            "p90": round(float(p90), 4)}


def materialize_routes(input_path, rows, output):
    """Apply fixes and the chosen downscale; annotations follow every pixel change."""
    records = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("clean", "teacher", "rejected")}
    counts = Counter()
    try:
        for row in rows:
            decision = row["route"]["decision"]
            record = dict(records[row["id"]], cleaning={"route": row["route"], "probe_id": row["id"]})
            if decision in ("reject", "teacher") or "error" in row:
                name = "teacher" if decision == "teacher" else "rejected"
                handles[name].write(json_text(record) + "\n")
                counts[name] += 1
                continue
            if "exif_orientation" in row["route"]["fixes"]:
                # Annotations use stored axes; rotating silently would break them.
                record["cleaning"]["route"] = {**row["route"], "decision": "reject",
                                               "reasons": row["route"]["reasons"] + ["exif_orientation"]}
                handles["rejected"].write(json_text(record) + "\n")
                counts["rejected"] += 1
                continue
            with Image.open(row["image_path"]) as image:
                image.load()
                if "composite_alpha" in row["route"]["fixes"]:
                    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(canvas, image.convert("RGBA"))
                image = image.convert("RGB")
            annotations = row.get("detected_regions") or regions(record.get("annotations", []), image.size)
            matrix = np.eye(3)
            if "trim_padding" in row["route"]["fixes"]:
                edges = row["stats"]["padding_edges"]
                box = (edges["left"], edges["top"], image.width - edges["right"], image.height - edges["bottom"])
                inside = [a for a in annotations if a["bbox"][0] >= box[0] and a["bbox"][1] >= box[1]
                          and a["bbox"][2] <= box[2] and a["bbox"][3] <= box[3]]
                if len(inside) == len(annotations) and box[2] > box[0] and box[3] > box[1]:
                    image = image.crop(box)
                    matrix = np.array([[1, 0, -box[0]], [0, 1, -box[1]], [0, 0, 1]], dtype=float) @ matrix
            k = row["route"]["downscale"]
            if k > 1:
                size = (max(1, round(image.width / k)), max(1, round(image.height / k)))
                matrix = scale_matrix(image.size, size) @ matrix
                image = image.resize(size, Image.Resampling.BOX)
            key = hashlib.sha256(json_text([record["source"], record["id"]]).encode()).hexdigest()
            target = output / "clean_images" / key[:2] / f"{key}.png"
            save_image(image, target)
            record.update(image_path=str(target), annotations=transform(annotations, matrix))
            record["cleaning"].update(source_path=row["image_path"], source_to_clean=matrix.tolist())
            for field in ("hr_path", "teacher"):
                record.pop(field, None)
            handles["clean"].write(json_text(record) + "\n")
            counts["clean"] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return dict(counts)
