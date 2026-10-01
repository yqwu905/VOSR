"""Continuous pixel-edge coordinates, in the decoded image's stored orientation."""

import math

import numpy as np
from PIL import Image


def regions(items, size):
    """Validate and canonicalize rectangle/polygon annotations without clipping."""
    result = []
    for item in items:
        item = dict(item)
        points = item.get("polygon")
        if points is None:
            x0, y0, x1, y1 = item["bbox"]
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("text polygon must contain >=3 finite (x, y) points")
        # Snap numerical roundoff at an image edge, without trimming real regions.
        p[np.abs(p) < 1e-7] = 0
        for axis in (0, 1):
            p[np.abs(p[:, axis] - size[axis]) < 1e-7, axis] = size[axis]
        if (p < 0).any() or (p[:, 0] > size[0]).any() or (p[:, 1] > size[1]).any():
            raise ValueError("text polygon lies outside image bounds")
        bbox = [float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max())]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("empty text region")
        if "confidence" in item:
            if isinstance(item["confidence"], bool):
                raise ValueError("OCR confidence must be numeric, not boolean")
            confidence = float(item["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("OCR confidence must be finite and in [0, 1]")
            item["confidence"] = confidence
        if "text" in item and not isinstance(item["text"], str):
            raise ValueError("recognized text must be a string")
        for flag in ("trusted", "reliable"):
            if flag in item and not isinstance(item[flag], bool):
                raise ValueError(f"{flag} must be a boolean")
        result.append({**item, "polygon": p.tolist(), "bbox": bbox})
    return result


def transform(items, matrix):
    result = []
    m = np.asarray(matrix, dtype=float)
    for item in items:
        p = np.asarray(item["polygon"])
        p = np.column_stack([p, np.ones(len(p))]) @ m.T
        p = p[:, :2] / p[:, 2:]
        result.append({**item, "polygon": p.tolist(),
                       "bbox": [float(p[:, 0].min()), float(p[:, 1].min()),
                                float(p[:, 0].max()), float(p[:, 1].max())]})
    return result


def scale_matrix(source_size, target_size):
    return np.diag([target_size[0] / source_size[0],
                    target_size[1] / source_size[1], 1.0])


def normalize_1k(image, edge=1024):
    factor = edge / max(image.size)
    size = tuple(max(1, round(x * factor)) for x in image.size)
    return image.resize(size, Image.Resampling.LANCZOS), scale_matrix(image.size, size)


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0.0


def text_crops(items, size, width=512, height=256, context=0.2, max_crops=64):
    """Center on each text region; expand to keep every intersected line whole.

    The initial size is a minimum. A long line or dense document can produce a
    larger crop, including the whole image. Never trim a glyph to fit a canvas.
    """
    seen = set()
    output = []
    for item in items:
        a = item["bbox"]
        w = min(size[0], max(width, math.ceil((a[2] - a[0]) * (1 + 2 * context))))
        h = min(size[1], max(height, math.ceil((a[3] - a[1]) * (1 + 2 * context))))
        x = min(max(0, math.floor((a[0] + a[2] - w) / 2)), size[0] - w)
        y = min(max(0, math.floor((a[1] + a[3] - h) / 2)), size[1] - h)
        rect = [x, y, x + w, y + h]
        while True:
            updated = list(rect)
            for neighbor in items:
                b = neighbor["bbox"]
                if intersects(rect, b):
                    updated = [min(updated[0], math.floor(b[0])), min(updated[1], math.floor(b[1])),
                               max(updated[2], math.ceil(b[2])), max(updated[3], math.ceil(b[3]))]
            if updated == rect:
                break
            rect = updated
        key = tuple(rect)
        if key not in seen:
            seen.add(key)
            output.append(key)
        if len(output) >= max_crops:
            break
    return output


def crop_regions(items, rect):
    """Return complete regions only, translated to crop coordinates."""
    matrix = np.array([[1, 0, -rect[0]], [0, 1, -rect[1]], [0, 0, 1]], dtype=float)
    return transform([x for x in items if contains(rect, x["bbox"])], matrix)
