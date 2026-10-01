"""Built-in production adapters and explicit demo adapters.

Custom adapters are loaded as module:Class with JSON kwargs. Interfaces:
teacher.restore(image, context) -> PIL RGB image;
ocr.recognize(image, context) -> region dictionaries;
iqa.score(image, context) -> {dimension: float};
glyph.check(original_crop, hr_crop, context) -> {status, evidence}.
"""

import csv
import hashlib
import importlib
import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def pixel_digest(image):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def read_rgb(path):
    # Do not silently apply EXIF rotation: annotations use stored pixel axes.
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("multi-frame input is unsupported")
        image.load()
        return image.convert("RGB")


def validate_artifact(artifact, image):
    if artifact.get("size") != list(image.size):
        raise ValueError("sidecar size must match the actual processed image")
    digest = artifact.get("image_sha256")
    if digest is not None and digest != pixel_digest(image):
        raise ValueError("sidecar image_sha256 does not match processed pixels")


class PrecomputedTeacher:
    demo = False

    def __init__(self, model):
        if model not in {"tadisr", "vosr2"}:
            raise ValueError("select exactly one teacher: tadisr or vosr2")
        self.model = model

    def restore(self, image, context):
        sample = context["sample"]
        provenance = sample.get("teacher", {})
        if provenance.get("model") != self.model or not provenance.get("checkpoint"):
            raise ValueError("precomputed HR requires teacher.model and teacher.checkpoint provenance")
        path = Path(sample["hr_path"])
        if not path.is_absolute():
            path = context["input_root"] / path
        return read_rgb(path)


class SidecarOCR:
    demo = False

    def recognize(self, image, context):
        sample = context["sample"]
        if context["stage"] == "original":
            artifact = sample.get("ocr", {}).get("original")
            if artifact is None:
                return []
        else:
            artifact = sample["ocr"]["final"][context["crop_id"]]
        validate_artifact(artifact, image)
        return artifact["regions"]


class SidecarIQA:
    demo = False

    def score(self, image, context):
        artifact = context["sample"].get("iqa", {}).get(context["crop_id"])
        if artifact is None:
            return {}
        validate_artifact(artifact, image)
        return dict(artifact["scores"])


class HumanGlyphCheck:
    """A verified human decision must be bound to these exact HR pixels."""
    demo = False

    def check(self, original, hr, context):
        artifact = context["sample"].get("glyph_checks", {}).get(context["crop_id"])
        if artifact is None:
            return {"status": "unverifiable", "evidence": "no glyph review supplied"}
        validate_artifact(artifact, hr)
        if artifact.get("status") == "verified":
            if not artifact.get("image_sha256") or not artifact.get("reviewer"):
                raise ValueError("verified human glyph review requires image_sha256 and reviewer")
            if artifact.get("original_crop_sha256") != pixel_digest(original):
                raise ValueError("verified glyph review must match the original_crop_sha256")
        return dict(artifact)


class TesseractOCR:
    """No Python OCR dependency; TSV words are joined into complete text lines."""
    demo = False

    def __init__(self, language="eng", psm=11, executable="tesseract", timeout=120):
        self.language, self.psm = language, int(psm)
        self.executable, self.timeout = executable, timeout

    def recognize(self, image, context):
        with tempfile.TemporaryDirectory(prefix="text-sr-ocr-") as directory:
            path = Path(directory) / "input.png"
            image.save(path)
            result = subprocess.run([self.executable, str(path), "stdout", "-l", self.language,
                                     "--psm", str(self.psm), "tsv"], check=True,
                                    capture_output=True, text=True, timeout=self.timeout)
        lines = {}
        for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
            if row["level"] != "5" or not row["text"].strip() or float(row["conf"]) < 0:
                continue
            key = tuple(row[x] for x in ("page_num", "block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(row)
        output = []
        for words in lines.values():
            boxes = [(int(x["left"]), int(x["top"]), int(x["width"]), int(x["height"])) for x in words]
            output.append({"bbox": [min(x for x, y, w, h in boxes), min(y for x, y, w, h in boxes),
                                    max(x + w for x, y, w, h in boxes), max(y + h for x, y, w, h in boxes)],
                           "text": " ".join(x["text"] for x in words),
                           "confidence": min(float(x["conf"]) / 100 for x in words)})
        return output


class PyIQA:
    """Explicit dimension -> model mapping, with user-calibrated score direction."""
    demo = False

    def __init__(self, metrics, device="cpu", metric_options=None):
        import pyiqa
        self.device = device
        self.models = {dimension: pyiqa.create_metric(name, device=device,
                       **(metric_options or {}).get(dimension, {})) for dimension, name in metrics.items()}

    def score(self, image, context):
        import torch
        import numpy as np
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        with torch.inference_mode():
            return {key: float(model(tensor.to(self.device)).item()) for key, model in self.models.items()}


class DemoTeacher:
    demo = True
    model = "demo_bicubic"

    def restore(self, image, context):
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)


class DemoOCR:
    demo = True

    def recognize(self, image, context):
        # Annotation replay tests plumbing only; this does not perform OCR.
        return [dict(x, confidence=0.99) for x in context["regions"]]


class DemoIQA:
    demo = True

    def score(self, image, context):
        return {"clarity": 1.0, "naturalness": 1.0, "artifacts": 0.0}


BUILTINS = {"precomputed": PrecomputedTeacher, "sidecar_ocr": SidecarOCR,
            "sidecar_iqa": SidecarIQA, "human": HumanGlyphCheck,
            "tesseract": TesseractOCR, "pyiqa": PyIQA,
            "demo_teacher": DemoTeacher, "demo_ocr": DemoOCR, "demo_iqa": DemoIQA}


def load_backend(spec):
    name = spec["type"]
    if name == "vosr2":
        from .vosr_teacher import VOSR2Teacher
        constructor = VOSR2Teacher
    elif name in BUILTINS:
        constructor = BUILTINS[name]
    else:
        module, attribute = name.split(":", 1)
        constructor = getattr(importlib.import_module(module), attribute)
    return constructor(**spec.get("kwargs", {}))
