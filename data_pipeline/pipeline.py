"""Streaming, audited six-stage data construction with transactional resume."""

import hashlib
import json
import sys
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .augmentation import degrade, sample_seed
from .backends import load_backend, pixel_digest, read_rgb
from .config import validate_config
from .geometry import (crop_regions, normalize_1k, regions, scale_matrix,
                       text_crops, transform)
from .quality import assess, reference_check


REFERENCE = "https://chatgpt.com/space/page_6abcb03b0d108191b0cac50aab487148"


class InputIntegrityError(ValueError):
    """Do not replace a committed sample when an input ID is ambiguous."""


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def resolve_input(sample, root, key):
    path = Path(sample[key])
    return path if path.is_absolute() else root / path


def split_for(group, seed, fraction):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2 ** 64
    return "validation" if value < fraction else "train"


class Pipeline:
    def __init__(self, config, input_path, output_path, resume=False, progress=False):
        self.config = validate_config(config)
        self.input = Path(input_path).resolve()
        self.output = Path(output_path).resolve()
        self.root = self.input.parent
        self.output.mkdir(parents=True, exist_ok=True)
        self.backends = None
        self.backend_load_seconds = {}
        self.resume = resume
        self.progress = progress

    def initialize(self):
        metadata_path = self.output / "run.json"
        implementation = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            implementation.update(path.name.encode())
            implementation.update(path.read_bytes())
        dependency_files = {str(Path(path).resolve()): file_digest(path)
                            for spec in self.config["backends"].values()
                            for path in spec.get("fingerprint_files", [])}
        metadata = {"schema_version": 1, "config": self.config,
                    "input_path": str(self.input), "input_sha256": file_digest(self.input),
                    "implementation_sha256": implementation.hexdigest(),
                    "dependency_sha256": dependency_files, "reference": REFERENCE}
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous != metadata:
                raise ValueError("output belongs to different inputs/config/code; use a new output directory")
            if not self.resume:
                raise ValueError("output already initialized; pass --resume or choose a new directory")
        elif self.resume:
            raise ValueError("cannot resume: run.json is missing")
        else:
            # Avoid replacing any user files in a pre-existing output directory.
            if any(p.name != ".pipeline.lock" for p in self.output.iterdir()):
                raise ValueError("new run requires an empty output directory")
            atomic_json(metadata_path, metadata)
        self.db = sqlite3.connect(self.output / "state.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, fingerprint TEXT, "
                        "digest TEXT, state TEXT, payload TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS digest_index ON jobs(digest)")
        self.db.execute("CREATE TEMP TABLE input_ids (key TEXT PRIMARY KEY)")
        self.db.commit()
        # A completed resume should verify files without loading a 1.4B teacher.
        self.backends = {}

    def backend(self, name):
        if name not in self.backends:
            started = time.perf_counter()
            self.backends[name] = load_backend(self.config["backends"][name])
            self.backend_load_seconds[name] = time.perf_counter() - started
        return self.backends[name]

    def process(self, sample, image, digest, key):
        config = self.config
        started = time.perf_counter()
        timings = {}
        ocr = self.backend("ocr")
        annotations = regions(sample.get("annotations", []), image.size)
        context = {"sample": sample, "input_root": self.root, "stage": "original", "regions": annotations,
                   "seed": sample_seed(config["seed"], key, "teacher")}
        # Reject inputs with no text localization before loading/running the teacher.
        stage_started = time.perf_counter()
        original_ocr = regions(ocr.recognize(image, context), image.size)
        timings["original_ocr"] = time.perf_counter() - stage_started
        if not annotations and not original_ocr:
            return [{"sample_id": sample["id"], "source": sample["source"], "sample_key": key,
                     "source_path": str(resolve_input(sample, self.root, "image_path")),
                     "source_sha256": digest, "status": "rejected", "eligible_for_training": False,
                     "reasons": ["no_localization_text"], "timings_seconds": timings}]
        teacher = self.backend("teacher")
        # 01: exactly one teacher option; returned pixels are a pseudo GT.
        stage_started = time.perf_counter()
        restored = teacher.restore(image, context).convert("RGB")
        timings["teacher"] = time.perf_counter() - stage_started
        expected_height = image.height * restored.width / image.width
        if abs(restored.height - expected_height) > 1.01:
            raise ValueError("teacher must preserve aspect ratio and pixel alignment")
        if restored.width < image.width or restored.height < image.height:
            raise ValueError("teacher output cannot be smaller than its input")
        # 02: aspect-preserving normalization and full-line text-center crops.
        full_hr, teacher_to_full = normalize_1k(restored, config["long_edge"])
        original_to_full = teacher_to_full @ scale_matrix(image.size, restored.size)
        full_annotations = transform(annotations, original_to_full)
        full_original = transform(original_ocr, original_to_full)
        localization = full_annotations + full_original
        rects = text_crops(localization, full_hr.size, **config["crop"])
        teacher_metadata = {"backend": config["backends"]["teacher"], "seed": context["seed"]}
        if getattr(teacher, "metadata", None):
            teacher_metadata["resolved"] = teacher.metadata
        if config["backends"]["teacher"]["type"] == "precomputed":
            teacher_metadata.update(provenance=sample["teacher"],
                                    artifact_path=str(resolve_input(sample, self.root, "hr_path")),
                                    artifact_sha256=file_digest(resolve_input(sample, self.root, "hr_path")))
        base = {"sample_id": sample["id"], "source": sample["source"], "sample_key": key,
                "source_path": str(resolve_input(sample, self.root, "image_path")),
                "source_sha256": digest, "teacher": teacher_metadata,
                "is_pseudo_gt": True, "original_size": list(image.size),
                "teacher_size": list(restored.size), "full_hr_size": list(full_hr.size),
                "original_to_full_hr": original_to_full.tolist(),
                "original_ocr": original_ocr, "dataset": sample.get("dataset", {}),
                "timings_seconds": timings}
        if not rects:
            return [{**base, "status": "rejected", "reasons": ["no_localization_text"],
                     "eligible_for_training": False}]
        full_path = self.output / "full_hr" / key[:2] / f"{key}.png"
        if config["save_full_hr"]:
            save_image(full_hr, full_path)
            base["full_hr_sha256"] = pixel_digest(full_hr)
        group = sample.get("group_id") or digest
        split = split_for(group, config["seed"], config["split"]["validation_fraction"])
        iqa, glyph = self.backend("iqa"), self.backend("glyph")
        demo = any(getattr(backend, "demo", False) for backend in self.backends.values())
        output = []
        for index, rect in enumerate(rects):
            crop_id = f"crop_{index:04d}"
            crop = full_hr.crop(rect)
            trusted = crop_regions(full_annotations, rect)
            source_regions = crop_regions(full_original, rect)
            context.update({"stage": "final", "crop_id": crop_id, "regions": trusted,
                            "trusted_regions": trusted, "original_ocr": source_regions})
            # 03: formal OCR on processed HR, never reuse the pre-detection output.
            stage_started = time.perf_counter()
            predictions = regions(ocr.recognize(crop, context), crop.size)
            crop_timings = {"final_ocr": time.perf_counter() - stage_started}
            for prediction in predictions:
                if "text" not in prediction or "confidence" not in prediction:
                    raise ValueError("formal OCR requires text and confidence for every region")
            translation = np.array([[1, 0, -rect[0]], [0, 1, -rect[1]], [0, 0, 1]], dtype=float)
            original_to_crop = translation @ original_to_full
            inverse = np.linalg.inv(original_to_crop)
            original_rect = [rect[0] / original_to_full[0, 0], rect[1] / original_to_full[1, 1],
                             rect[2] / original_to_full[0, 0], rect[3] / original_to_full[1, 1]]
            original_crop = image.transform(crop.size, Image.Transform.AFFINE,
                                            tuple(inverse[:2].reshape(-1)), Image.Resampling.BICUBIC)
            reference = reference_check(predictions, trusted, source_regions, config["quality"])
            # 04: reject low confidence before expensive glyph and IQA evaluation.
            ocr_pass = bool(predictions) and all(x["text"].strip() and
                       x["confidence"] >= config["quality"]["ocr_min_confidence"] for x in predictions)
            # 05: character reference and a separate glyph/structure verifier.
            glyph_result = glyph.check(original_crop, crop, context) if ocr_pass else {
                "status": "unverifiable", "evidence": "skipped after OCR rejection"}
            # 06: calibrated scores do not substitute for textual correctness.
            scores = dict(iqa.score(crop, context)) if ocr_pass else {}
            decision = assess(predictions, reference, glyph_result, scores, config["quality"], demo)
            record = {**base, "crop_id": crop_id, "crop_rect_full_hr": list(rect),
                      "timings_seconds": {**timings, **crop_timings},
                      "crop_size": list(crop.size), "crop_sha256": pixel_digest(crop),
                      "original_crop_sha256": pixel_digest(original_crop),
                      "original_crop_rect": original_rect, "original_to_crop": original_to_crop.tolist(),
                      "crop_to_original": inverse.tolist(), "annotations": trusted, "ocr": predictions,
                      "reference_check": reference, "glyph_check": glyph_result,
                      "iqa_scores": scores, "group_id": group, "split": split, **decision,
                      "variants": []}
            if config["save_full_hr"]:
                record["full_hr_path"] = str(full_path)
            if decision["status"] != "rejected" or config["save_rejected"]:
                name = f"{key}_{crop_id}"
                hr_path = self.output / "hr" / key[:2] / f"{name}.png"
                source_path = self.output / "original_crops" / key[:2] / f"{name}.png"
                save_image(crop, hr_path)
                save_image(original_crop, source_path)
                record.update(hr_path=str(hr_path), original_crop_path=str(source_path))
                if decision["status"] == "accepted" or (decision["status"] == "review" and
                                                         config["augmentation"]["review_samples"]):
                    for variant in range(config["augmentation"]["variants"]):
                        seed = sample_seed(config["seed"], name, variant)
                        lr, lr_ocr, parameters = degrade(crop, predictions, config["augmentation"], seed)
                        lr_path = self.output / "lr" / key[:2] / f"{name}_v{variant:03d}.png"
                        save_image(lr, lr_path)
                        record["variants"].append({"lr_path": str(lr_path), "lr_size": list(lr.size),
                                                   "lr_sha256": pixel_digest(lr),
                                                   "ocr_lr": lr_ocr, "degradation": parameters})
            output.append(record)
        for record in output:
            record["timings_seconds"]["sample_total"] = time.perf_counter() - started
        return output

    def run(self):
        """Hold a single-writer lock; commit one source image at a time."""
        import fcntl
        with (self.output / ".pipeline.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another pipeline is writing this output directory") from error
            try:
                self.initialize()
                self.consume()
                return self.export()
            finally:
                if hasattr(self, "db"):
                    self.db.close()

    def consume(self):
        with self.input.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                sample, key, fingerprint, digest = {}, f"invalid_line_{line_number}", "", None
                previous = None
                verifying_input = False
                try:
                    sample = json.loads(line)
                    if not isinstance(sample, dict):
                        raise ValueError("manifest record must be an object")
                    for name in ("id", "source", "image_path"):
                        if not isinstance(sample.get(name), str) or not sample[name].strip():
                            raise ValueError(f"manifest requires nonempty string {name}")
                    if "group_id" in sample and (not isinstance(sample["group_id"], str) or not sample["group_id"]):
                        raise ValueError("group_id must be a nonempty string")
                    key = hashlib.sha256(json_text([sample["source"], sample["id"]]).encode()).hexdigest()
                    try:
                        self.db.execute("INSERT INTO input_ids VALUES (?)", (key,))
                    except sqlite3.IntegrityError as error:
                        raise InputIntegrityError("duplicate source/id in input manifest") from error
                    previous = self.db.execute("SELECT fingerprint, state FROM jobs WHERE key=?", (key,)).fetchone()
                    verifying_input = previous is not None and previous[1] != "error"
                    image = read_rgb(resolve_input(sample, self.root, "image_path"))
                    digest = pixel_digest(image)
                    fingerprint_parts = [json_text(sample), digest]
                    if "hr_path" in sample:
                        fingerprint_parts.append(file_digest(resolve_input(sample, self.root, "hr_path")))
                    fingerprint = hashlib.sha256(json_text(fingerprint_parts).encode()).hexdigest()
                    verifying_input = False
                    if previous and previous[1] != "error":
                        if previous[0] != fingerprint:
                            raise InputIntegrityError("input pixels or record changed for an existing ID; use a new output directory")
                        if not self.resume:
                            raise InputIntegrityError("duplicate source/id in input manifest")
                        payload = self.db.execute("SELECT payload FROM jobs WHERE key=?", (key,)).fetchone()[0]
                        if self.outputs_intact(json.loads(payload)):
                            continue
                    duplicate = self.db.execute("SELECT key FROM jobs WHERE digest=? AND state='complete' AND key!=?",
                                                (digest, key)).fetchone()
                    if duplicate:
                        output = [{"sample_id": sample["id"], "source": sample["source"], "sample_key": key,
                                   "status": "rejected", "eligible_for_training": False,
                                   "reasons": ["duplicate_image"], "duplicate_of": duplicate[0], "source_sha256": digest}]
                    else:
                        output = self.process(sample, image, digest, key)
                    # Fail malformed backend metadata inside this sample's error boundary.
                    json_text(output)
                    state = "duplicate" if duplicate else "complete"
                except InputIntegrityError:
                    # Resume/ID integrity violations must not overwrite committed work.
                    raise
                except Exception as error:
                    if verifying_input:
                        raise InputIntegrityError("cannot verify previously committed input; restore the input "
                                                  "or use a new output directory") from error
                    output = [{"sample_id": sample.get("id") if isinstance(sample, dict) else None,
                               "source": sample.get("source", "unknown") if isinstance(sample, dict) else "unknown",
                               "sample_key": key, "status": "error", "eligible_for_training": False,
                               "line_number": line_number, "reasons": [type(error).__name__], "error": str(error)}]
                    state = "error"
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?)",
                                    (key, fingerprint, digest, state, json_text(output)))
                if self.progress:
                    print(json_text({"line": line_number, "sample_id": sample.get("id") if isinstance(sample, dict) else None,
                                     "state": state, "decisions": dict(Counter(x["status"] for x in output))}),
                          file=sys.stderr, flush=True)

    def outputs_intact(self, records):
        for record in records:
            paths = [(record.get("hr_path"), record.get("crop_sha256")),
                     (record.get("original_crop_path"), record.get("original_crop_sha256")),
                     (record.get("full_hr_path"), record.get("full_hr_sha256"))]
            paths += [(variant["lr_path"], variant["lr_sha256"]) for variant in record.get("variants", [])]
            for path, digest in paths:
                if path is None:
                    continue
                try:
                    image = read_rgb(path)
                    if digest is not None and pixel_digest(image) != digest:
                        return False
                except (OSError, ValueError):
                    return False
        return True

    def export(self):
        """Rebuild manifests atomically from committed rows, including after resume."""
        names = ("accepted.jsonl", "review.jsonl", "rejected.jsonl", "errors.jsonl",
                 "pairs_train.jsonl", "pairs_validation.jsonl", "train_hq.txt", "validation_hq.txt")
        handles = {name: (self.output / (name + ".tmp")).open("w", encoding="utf-8") for name in names}
        counts, sources = Counter(), defaultdict(Counter)
        jobs, source_jobs, reason_counts = Counter(), defaultdict(Counter), defaultdict(Counter)
        teacher_seconds, teacher_count = 0.0, 0
        try:
            for state, payload in self.db.execute("SELECT state, payload FROM jobs ORDER BY key"):
                jobs[state] += 1
                records = json.loads(payload)
                if records:
                    source_jobs[records[0]["source"]][state] += 1
                    elapsed = records[0].get("timings_seconds", {}).get("teacher")
                    if elapsed is not None:
                        teacher_seconds += elapsed
                        teacher_count += 1
                for record in records:
                    status, source = record["status"], record["source"]
                    counts[status] += 1
                    sources[source][status] += 1
                    reason_counts[source].update(record.get("reasons", []))
                    target = "errors.jsonl" if status == "error" else status + ".jsonl"
                    handles[target].write(json_text(record) + "\n")
                    if status == "accepted":
                        handles[f"{record['split']}_hq.txt"].write(record["hr_path"] + "\n")
                        for index, variant in enumerate(record["variants"]):
                            pair = {"sample_key": record["sample_key"], "crop_id": record["crop_id"],
                                    "variant": index, "source": source, "group_id": record["group_id"],
                                    "split": record["split"], "hr_path": record["hr_path"],
                                    "eligible_for_training": True, **variant}
                            handles[f"pairs_{record['split']}.jsonl"].write(json_text(pair) + "\n")
        finally:
            for handle in handles.values():
                handle.close()
        for name in names:
            (self.output / (name + ".tmp")).replace(self.output / name)
        for split in ("train", "validation"):
            path = self.output / f"{split}_dataset.txt"
            temporary = path.with_suffix(".tmp")
            text = f"{self.output / (split + '_hq.txt')}, 1\n" if (self.output / (split + "_hq.txt")).stat().st_size else ""
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        summary = {"jobs": dict(jobs), "crops": dict(counts),
                   "sources": {key: dict(value) for key, value in sources.items()},
                   "source_jobs": {key: dict(value) for key, value in source_jobs.items()},
                   "reasons": {key: dict(value) for key, value in reason_counts.items()},
                   "teacher_timing": {"images": teacher_count, "total_seconds": teacher_seconds,
                                      "mean_seconds": teacher_seconds / teacher_count if teacher_count else None},
                   "reference": REFERENCE, "output": str(self.output),
                   "backend_load_seconds": self.backend_load_seconds}
        atomic_json(self.output / "summary.json", summary)
        return summary
