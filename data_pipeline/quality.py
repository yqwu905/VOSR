"""Independent OCR, reference fidelity, glyph and IQA quality gates."""

import math
import unicodedata

from .geometry import iou


def normalized_text(text):
    # Preserve case and punctuation: changing them can change the ground truth.
    return " ".join(unicodedata.normalize("NFC", text).split())


def reference_check(predictions, trusted, original, config):
    """Reliable original OCR is evidence, never a proof of glyph authenticity."""
    evidence = [dict(x, evidence="trusted_annotation") for x in trusted if x.get("trusted") and x.get("text")]
    if not evidence:
        evidence = [dict(x, evidence="reliable_original_ocr") for x in original
                    if x.get("reliable") is True and x.get("confidence", 0) >= config["original_min_confidence"]
                    and x.get("text")]
    comparisons, used = [], set()
    conflict, uncertain = False, False
    for prediction in predictions:
        candidates = [(iou(prediction["bbox"], ref["bbox"]), i, ref)
                      for i, ref in enumerate(evidence) if i not in used]
        overlap, index, ref = max(candidates, key=lambda x: x[0], default=(0, -1, None))
        if overlap < config["match_iou"]:
            uncertain = True
            comparisons.append({"prediction": prediction["text"], "status": "no_reliable_reference"})
            continue
        used.add(index)
        equal = normalized_text(prediction["text"]) == normalized_text(ref["text"])
        conflict |= not equal
        comparisons.append({"prediction": prediction["text"], "reference": ref["text"],
                            "evidence": ref["evidence"], "iou": overlap,
                            "status": "match" if equal else "conflict"})
    missing = [ref["text"] for i, ref in enumerate(evidence) if i not in used]
    uncertain |= bool(missing) or not predictions
    status = "conflict" if conflict else "unverifiable" if uncertain else "matched"
    return {"status": status, "comparisons": comparisons, "missing_references": missing}


def assess(predictions, reference, glyph, scores, config, demo=False):
    rejected, review = [], []
    if config.get("mode", "strict") == "audit":
        review.append("audit_mode_uncalibrated")
    for name, value in scores.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"non-finite or invalid IQA score: {name}")
        scores[name] = float(value)
    if not predictions:
        rejected.append("no_text")
    elif any(not x.get("text", "").strip() or x.get("confidence", 0) < config["ocr_min_confidence"]
             for x in predictions):
        rejected.append("low_ocr_confidence")
    if reference["status"] == "conflict":
        review.append("text_reference_conflict")
    elif reference["status"] != "matched":
        review.append("text_reference_unverifiable")
    glyph_status = glyph.get("status", "unverifiable")
    if glyph_status not in {"verified", "conflict", "unverifiable"}:
        raise ValueError("glyph status must be verified, conflict or unverifiable")
    if glyph_status == "verified" and not glyph.get("evidence"):
        raise ValueError("verified glyph checks require evidence")
    if glyph_status != "verified":
        review.append("glyph_conflict" if glyph_status == "conflict" else "glyph_unverifiable")
    labels = {}
    for name, rule in config["iqa_thresholds"].items():
        score = scores.get(name)
        if score is None:
            labels[name] = "missing"
            review.append(f"missing_iqa:{name}")
            continue
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"non-finite IQA score: {name}")
        scores[name] = score
        passed = score >= rule["threshold"] if rule["direction"] == "higher" else score <= rule["threshold"]
        labels[name] = "pass" if passed else "fail"
        if not passed:
            rejected.append(f"low_iqa:{name}")
    if demo:
        review.append("demo_backend")
    status = "rejected" if rejected else "review" if review else "accepted"
    return {"status": status, "eligible_for_training": status == "accepted",
            "reasons": rejected + review, "iqa_labels": labels}
