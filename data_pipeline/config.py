"""JSON configuration validation; production thresholds are never guessed."""

import copy
import math


DEFAULTS = {
    "seed": 42,
    "long_edge": 1024,
    "crop": {"width": 512, "height": 256, "context": 0.2, "max_crops": 64},
    "quality": {"ocr_min_confidence": None, "original_min_confidence": None,
                "match_iou": 0.5, "iqa_thresholds": {}},
    "augmentation": {"variants": 2, "scales": [2, 4], "blur_sigma": [0.2, 1.2],
                     "noise_std": [0.0, 5.0], "jpeg_quality": [65, 95], "review_samples": False},
    "split": {"validation_fraction": 0.05},
    "save_full_hr": True,
    "backends": {},
}


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {key}")
        # Backend specs and metric dimensions are extensible dictionaries.
        if key in {"backends", "iqa_thresholds"}:
            result[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(base[key], dict):
            result[key] = merge(base[key], value)
        else:
            result[key] = value
    return result


def number(value, name, minimum, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} requires a finite {'integer' if integer else 'number'}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} requires an integer")
    if value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} out of range")


def validate_config(updates):
    config = merge(DEFAULTS, updates)
    number(config["seed"], "seed", 0, integer=True)
    number(config["long_edge"], "long_edge", 1, integer=True)
    for key in ("width", "height", "max_crops"):
        number(config["crop"][key], f"crop.{key}", 1, integer=True)
    number(config["crop"]["context"], "crop.context", 0)
    for key in ("ocr_min_confidence", "original_min_confidence", "match_iou"):
        number(config["quality"][key], f"quality.{key} (calibrate explicitly)", 0, 1)
    if config["quality"]["match_iou"] == 0:
        raise ValueError("quality.match_iou must be positive")
    dimensions = config["quality"]["iqa_thresholds"]
    if not {"clarity", "naturalness", "artifacts"}.issubset(dimensions):
        raise ValueError("calibrated IQA thresholds must cover clarity, naturalness and artifacts")
    for dimension, rule in dimensions.items():
        if rule.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"IQA {dimension}: direction must be higher or lower")
        value = rule.get("threshold")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"IQA {dimension}: calibrate a finite threshold explicitly")
    augmentation = config["augmentation"]
    number(augmentation["variants"], "augmentation.variants", 0, integer=True)
    if not augmentation["scales"]:
        raise ValueError("augmentation.scales cannot be empty")
    for scale in augmentation["scales"]:
        number(scale, "augmentation.scale", 1, integer=True)
    for key in ("blur_sigma", "noise_std", "jpeg_quality"):
        values = augmentation[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"augmentation.{key} must be [min, max]")
        for value in values:
            number(value, f"augmentation.{key}", 1 if key == "jpeg_quality" else 0,
                   100 if key == "jpeg_quality" else None, integer=key == "jpeg_quality")
        if values[0] > values[1]:
            raise ValueError(f"augmentation.{key}: min exceeds max")
    number(config["split"]["validation_fraction"], "split.validation_fraction", 0, 1)
    for value in (config["save_full_hr"], augmentation["review_samples"]):
        if not isinstance(value, bool):
            raise ValueError("save_full_hr and review_samples must be booleans")
    if set(config["backends"]) != {"teacher", "ocr", "iqa", "glyph"}:
        raise ValueError("configure teacher, ocr, iqa and glyph backends")
    for name, spec in config["backends"].items():
        if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
            raise ValueError(f"invalid {name} backend")
        if not isinstance(spec.get("kwargs", {}), dict):
            raise ValueError(f"{name} backend kwargs must be an object")
        files = spec.get("fingerprint_files", [])
        if not isinstance(files, list) or any(not isinstance(path, str) for path in files):
            raise ValueError(f"{name} backend fingerprint_files must be a list of paths")
    return config
