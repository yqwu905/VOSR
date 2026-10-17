"""Render stored documents into HR pages with exact text boxes from the text layer.

Vector documents can be rendered at any resolution, so each page gets the DPI that puts
its median text-line height (character-weighted, visible lines only) at a target drawn
from ``text_height`` (pixels): the HR is never resampled. Pages are classified first:

  text      a usable text layer: line quads and transcripts become trusted annotations
  garbled   glyph boxes are exact but the text maps to junk (Founder fonts without
            ToUnicode, private-use, replacement or rare-character soup): render-only,
            untrusted boxes
  vector    no text layer but visible vector ink (glyphs converted to outlines): render-only
  scanned   a large image with little or only invisible (OCR-layer) text: dropped
  blank     nothing visible, dropped

Per line, invisible text (render mode 3, zero opacity, white-on-white or off-page ad
lines that leave no ink after rendering) is removed, lines containing .notdef glyphs
(missing from every allowed font; the converter rejects bitmap fallbacks) lose their
trust, symbol-font bullets are excluded, and labels keep full-width punctuation (NFKC
only for Kangxi radicals and compatibility ideographs). Pages drawn mostly with
non-embedded CJK fonts are flagged ``style_substituted`` (MuPDF draws them with its
fallback face). Slide placeholders and template-site watermarks drop the page, a deck
with more than 30% placeholder pages is dropped whole, and pHash near-duplicates are
removed across the run. An optional text detector flags text that the text layer does
not cover (text inside images, WordArt).
"""

import hashlib
import json
import random
import re
import statistics
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..pipeline import atomic_json, json_text
from .refill import PLACEHOLDER, WATERMARK
from .sniff import CONVERTIBLE, cjk_ratio

GARBLED_FONTS = re.compile(r"FzBookMaker|DLF-|DlFont", re.I)
# Bullet glyphs of symbol fonts sit in the private-use area; they are not text.
SYMBOL_FONTS = re.compile(r"OpenSymbol|Wingdings|Webdings|Symbol|ZapfDingbats|Dingbats|Marlett|MT ?Extra", re.I)
MAX_PIXELS = 25_000_000
# Control characters and line separators are not glyphs and break line-based JSONL readers.
CONTROL = re.compile("[%s]" % "".join(re.escape(chr(c)) for c in [*range(0, 32), *range(0x7F, 0xA0), 0x2028, 0x2029]))
SYMBOL_PUA = range(0xF020, 0xF100)
NFKC_RANGES = ((0x2E80, 0x2FDF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def font_name(name):
    """Subset prefix removed; names that a producer stored as Latin-1 bytes decoded."""
    name = name.split("+", 1)[1] if re.match(r"^[A-Z]{6}\+", name) else name
    if any(0x80 <= ord(c) <= 0xFF for c in name):
        for codec in ("utf-8", "gbk", "big5"):  # UTF-8 first: GBK would also accept its bytes
            try:
                return name.encode("latin-1").decode(codec)
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    return name


def label(text):
    """Transcript as drawn: no controls or symbol-font code points; NFKC only where it is safe."""
    out = []
    for c in CONTROL.sub("", text):
        code = ord(c)
        if code in SYMBOL_PUA:
            continue
        out.append(unicodedata.normalize("NFKC", c) if any(a <= code <= b for a, b in NFKC_RANGES) else c)
    return "".join(out)


def _is_cjk(c):
    code = ord(c)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF


def _common_share(text):
    """Share of CJK characters in GB2312 or Big5: broken ToUnicode maps give rare-character soup."""
    cjk = [c for c in text if _is_cjk(c)]
    if not cjk:
        return 1.0
    def common(c):
        for codec in ("gb2312", "big5"):
            try:
                c.encode(codec)
                return True
            except UnicodeEncodeError:
                pass
        return False
    return sum(common(c) for c in cjk) / len(cjk)


def _junk_share(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    def junk(c):
        code = ord(c)
        return 0xE000 <= code <= 0xF8FF or code == 0xFFFD or code < 32 or 0x7F <= code <= 0x9F or code >= 0xF0000
    return sum(junk(c) for c in chars) / len(chars)


def _targets(spec):
    low, _, high = str(spec).partition(",")
    return float(low), float(high or low)


def _page_seed(seed, digest, number):
    return int.from_bytes(hashlib.sha256(f"{seed}:{digest}:{number}".encode()).digest()[:8], "big")


def page_lines(page):
    """Text lines with quads (unrotated page space), labels, font, size and per-line flags."""
    import pymupdf
    fonts = page.get_fonts()
    type3 = {font_name(f[3]) for f in fonts if f[2] == "Type3"}
    not_embedded = {font_name(f[3]) for f in fonts if f[1] == "n/a"}
    invisible, notdef = [], []
    for span in page.get_texttrace():
        hidden = span["type"] == 3 or span["opacity"] == 0
        check_glyphs = font_name(span["font"]) not in type3 and not SYMBOL_FONTS.search(span["font"])
        for char in span["chars"]:
            x0, y0, x1, y1 = char[3]
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            if hidden:
                invisible.append(centre)
            if check_glyphs and char[1] == 0 and not chr(char[0]).isspace():
                notdef.append(centre)
    invisible, notdef = np.array(invisible).reshape(-1, 2), np.array(notdef).reshape(-1, 2)

    def inside(points, rect):
        return int(((points[:, 0] >= rect.x0) & (points[:, 0] <= rect.x1) &
                    (points[:, 1] >= rect.y0) & (points[:, 1] <= rect.y1)).sum()) if len(points) else 0

    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip() and not SYMBOL_FONTS.search(s["font"])]
            if not spans:
                continue
            quad = pymupdf.recover_line_quad(line, spans=spans)
            rect = quad.rect
            raw = "".join(s["text"] for s in spans)
            chars = len([c for c in raw if not c.isspace()])
            substituted = sum(1 for s in spans if font_name(s["font"]) in not_embedded for c in s["text"] if _is_cjk(c))
            lines.append({"quad": quad, "raw": raw, "text": label(raw), "font": font_name(spans[0]["font"]),
                          "size": statistics.median(s["size"] for s in spans), "chars": chars,
                          "height": min(rect.height, rect.width) if line["dir"][0] == 0 else rect.height,
                          "visible": inside(invisible, rect) * 2 < max(chars, 1),
                          "on_page": bool(rect & page.rect) and rect.width > 0,
                          "notdef": inside(notdef, rect), "substituted_cjk": substituted})
    return lines


def classify(page, lines, ink):
    import pymupdf
    visible = [line for line in lines if line.get("visible", True)]
    text = "".join(line.get("raw", line["text"]) for line in visible)
    chars = len([c for c in text if not c.isspace()])
    all_chars = sum(line.get("chars", len(line["text"])) for line in lines)
    area = abs(page.rect)
    image_cover = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) / area for info in page.get_image_info()) if area else 0
    if image_cover > 0.5 and (chars < 20 or all_chars > 2 * chars):  # scan, possibly with an OCR layer
        return "scanned", chars
    if chars < 10:
        return ("vector" if ink > 0.002 else "blank"), chars
    founder = any(GARBLED_FONTS.search(line["font"]) for line in visible)
    ascii_punct = sum(1 for c in text if c in PUNCTUATION) / max(chars, 1)
    cjk_chars = sum(1 for c in text if _is_cjk(c))
    if (_junk_share(text) > 0.02 or (founder and cjk_ratio(text) < 0.2) or (founder and cjk_chars == 0 and ascii_punct > 0.2)
            or (cjk_chars >= 20 and _common_share(text) < 0.9)):
        return "garbled", chars
    return "text", chars


def _has_ink(gray, polygon):
    """False for a line that leaves no visible mark (white-on-white, clipped, hidden)."""
    xs, ys = zip(*polygon)
    x0, x1 = max(0, int(min(xs))), min(gray.shape[1], int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(gray.shape[0], int(max(ys)) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    crop = gray[y0:y1, x0:x1].astype(np.int16)
    border = np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]))
    return float((np.abs(crop - border) > 60).mean()) > 0.002


def _unlabeled_share(detector, rgb, annotations, em):
    """Area share of detected text boxes that the text-layer quads do not cover."""
    import cv2
    from PIL import Image
    scale = min(1.0, 1600 / max(rgb.shape[:2]))
    small = np.ascontiguousarray(rgb if scale == 1 else cv2.resize(rgb, None, fx=scale, fy=scale,
                                                                   interpolation=cv2.INTER_AREA))
    mask = np.zeros(small.shape[:2], np.uint8)
    for item in annotations:
        cv2.fillPoly(mask, [np.round(np.array(item["polygon"]) * scale).astype(np.int32)], 1)
    grow = max(1, int(round(0.3 * em * scale)))
    mask = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    missing = 0.0
    for box in detector.detect(Image.fromarray(small)):
        points = np.array(box["polygon"], np.float32)
        x0, y0 = np.floor(points.min(axis=0)).astype(int).clip(0)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        region = mask[y0:y1, x0:x1]
        if region.size and region.mean() < 0.5:
            missing += region.size
    return missing / mask.size


def _render_document(task):
    try:
        return _render(*task)
    except Exception as error:  # one malformed file must not stop the run
        return {"sha256": task[0]["sha256"], "pages": [], "status": "error",
                "reason": f"{type(error).__name__}: {error}"[:200]}


_DETECTOR = None


def _render(entry, output, options):
    import os
    import pymupdf
    from ..probes import phash
    from .convert import ConversionError, fonts_conf, prepare_profile, to_pdf
    global _DETECTOR
    pymupdf.TOOLS.mupdf_display_errors(False)
    if options.get("detector") and _DETECTOR is None:
        from ..cleaning import load_detector
        _DETECTOR = load_detector(options["detector"])
    digest = entry["sha256"]
    rng_seed, (low, high) = options["seed"], _targets(options["text_height"])
    result = {"sha256": digest, "pages": [], "status": "ok"}
    path = Path(entry["path"])
    try:
        if entry["type"] in CONVERTIBLE:
            work = Path(options["work"])
            profile = prepare_profile(work / f"profile_{os.getpid()}")
            pdf_path = Path(output) / "pdf" / digest[:2] / f"{digest}.pdf"
            if not pdf_path.exists():
                with tempfile.TemporaryDirectory(dir=work) as tmp:
                    source = Path(tmp) / f"{digest}.{entry['type']}"
                    source.symlink_to(path.resolve())
                    pdf = to_pdf(source, Path(tmp) / "out", profile, fonts_conf(work))
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf.replace(pdf_path)
            path = pdf_path
        document = pymupdf.open(path)
    except (ConversionError, RuntimeError, ValueError) as error:
        result.update(status="error", reason=f"{type(error).__name__}: {error}"[:200])
        return result
    result["page_count"] = document.page_count
    fonts_used = Counter()
    for number in range(min(document.page_count, options["max_pages"])):
        page = document[number]
        try:
            lines = page_lines(page)
            thumb = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), alpha=False, annots=False)
        except RuntimeError as error:
            result["pages"].append({"page": number, "status": "error", "reason": str(error)[:120]})
            continue
        thumb_rgb = np.frombuffer(thumb.samples, np.uint8).reshape(thumb.height, thumb.width, thumb.n)[..., :3]
        ink = float((thumb_rgb.mean(axis=2) < 160).mean())
        page_class, chars = classify(page, lines, ink)
        used = [line for line in lines if line["visible"] and line["on_page"] and line["size"] >= 3]
        text = "".join(line["text"] for line in used)
        record = {"page": number, "class": page_class, "chars": chars,
                  "cjk": round(cjk_ratio(text), 3) if page_class != "vector" else None,
                  "size_pt": [round(page.rect.width, 1), round(page.rect.height, 1)]}
        if page_class in ("scanned", "blank"):
            result["pages"].append(dict(record, status="dropped_" + page_class))
            continue
        if page_class == "text" and record["cjk"] < options["min_cjk"]:
            result["pages"].append(dict(record, status="dropped_language"))
            continue
        if any(PLACEHOLDER.search(line["text"]) or WATERMARK.search(line["text"]) for line in used):
            result["pages"].append(dict(record, status="dropped_template"))
            continue
        heights = [line["height"] for line in used for _ in range(max(1, line["chars"])) if line["height"] > 1]
        target = random.Random(_page_seed(rng_seed, digest, number)).uniform(low, high)
        zoom = target / statistics.median(heights) if heights else options["default_dpi"] / 72
        zoom = max(min(zoom, options["max_dpi"] / 72, (MAX_PIXELS / max(abs(page.rect), 1)) ** 0.5), 1.0)
        matrix = page.rotation_matrix * pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False,
                                 annots=False)
        rgb = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3)
        gray = rgb.mean(axis=2)
        annotations, hidden, notdef, substituted, cjk_total = [], 0, 0, 0, 0
        for line in used:
            q = line["quad"] * matrix
            polygon = [[round(p.x, 2), round(p.y, 2)] for p in (q.ul, q.ur, q.lr, q.ll)]
            if not _has_ink(gray, polygon):
                hidden += 1
                continue
            notdef += line["notdef"]
            substituted += line["substituted_cjk"]
            cjk_total += sum(1 for c in line["text"] if _is_cjk(c))
            annotations.append({"polygon": polygon, "text": line["text"], "font": line["font"],
                                "size_px": round(line["size"] * zoom, 2), "annotation_origin": "pdf_text_layer",
                                "trusted": page_class == "text" and line["notdef"] == 0})
            fonts_used[line["font"]] += 1
        flags = {"hidden_lines": hidden, "notdef_glyphs": notdef,
                 "style_substituted": bool(cjk_total and substituted / cjk_total >= 0.1)}
        if _DETECTOR is not None:
            em = statistics.median([a["size_px"] for a in annotations]) if annotations else 20
            flags["unlabeled_share"] = round(_unlabeled_share(_DETECTOR, rgb, annotations, em), 4)
        image_path = Path(output) / "pages" / digest[:2] / f"{digest}_p{number:04d}.png"
        status = "kept"
        if flags.get("unlabeled_share", 0) > options["max_unlabeled"]:
            status = "dropped_unlabeled"
        else:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap.save(image_path)
        record.update(status=status, image_path=str(image_path), dpi=round(72 * zoom, 1),
                      size=[pixmap.width, pixmap.height],
                      median_text_px=round(statistics.median(heights) * zoom, 1) if heights else None,
                      annotations=annotations, flags=flags, phash=phash(thumb_rgb))
        result["pages"].append(record)
    templates = sum(1 for p in result["pages"] if p["status"] == "dropped_template")
    if templates > 0.3 * max(1, len(result["pages"])):  # a template deck: drop it whole
        for page in result["pages"]:
            if page["status"] == "kept":
                page["status"] = "dropped_template_deck"
                Path(page["image_path"]).unlink(missing_ok=True)
    result["fonts"] = dict(fonts_used.most_common(12))
    return result


def _entries(store):
    seen, entries = set(), []
    for line in (Path(store) / "fetch_log.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line) if line.strip() else None
        if not entry or entry.get("status") != "ok" or entry["sha256"] in seen:
            continue
        seen.add(entry["sha256"])
        entries.append(entry)
    return entries


def _expand_archives(entries, output):
    """Replace ZIP/RAR entries (template sites) by their document members."""
    from .sniff import RENDERABLE, sniff
    expanded = []
    for entry in entries:
        if entry["type"] not in ("zip", "rar"):
            expanded.append(entry)
            continue
        from .unpack import members
        for name, data in members(entry["path"]):
            kind = sniff(data)
            if kind not in RENDERABLE:
                continue
            digest = hashlib.sha256(data).hexdigest()
            path = Path(output) / "unpacked" / digest[:2] / f"{digest}.{kind}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            expanded.append(dict(entry, sha256=digest, type=kind, path=str(path), archive_member=name,
                                 archive_sha256=entry["sha256"]))
    return expanded


def render_store(store, output, text_height="32,64", min_cjk=0.3, max_pages=200, limit=None, workers=2, seed=42,
                 max_dpi=600, default_dpi=200, near_duplicate_bits=4, detector=None, max_unlabeled=0.05, log=print):
    """Render every downloaded document; writes pages.jsonl (all pages) and clean.jsonl."""
    from ..cleaning import duplicate_pairs
    from .sniff import RENDERABLE
    output = Path(output).resolve()
    if (output / "clean.jsonl").exists():
        raise ValueError("render output already exists; choose a new directory")
    (output / "work").mkdir(parents=True, exist_ok=True)
    entries = _expand_archives(_entries(store), output)
    entries = [e for e in entries if e["type"] in RENDERABLE][:limit]
    options = {"seed": seed, "text_height": text_height, "min_cjk": min_cjk, "max_pages": max_pages,
               "max_dpi": max_dpi, "default_dpi": default_dpi, "work": str(output / "work"), "detector": detector,
               "max_unlabeled": max_unlabeled}
    tasks = [(e, str(output), options) for e in entries]
    by_sha = {e["sha256"]: e for e in entries}
    pages, documents = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, result in enumerate(pool.map(_render_document, tasks), 1):
            entry = by_sha[result["sha256"]]
            documents.append({k: entry.get(k) for k in ("sha256", "type", "source", "terms", "url", "title", "license")}
                             | {"status": result["status"], "reason": result.get("reason"),
                                "page_count": result.get("page_count"), "fonts": result.get("fonts")})
            for page in result["pages"]:
                pages.append((entry, page))
            if log and done % 10 == 0:
                log(f"{done}/{len(tasks)} documents rendered")
    kept = [i for i, (_, p) in enumerate(pages) if p["status"] == "kept"]
    for a, b in duplicate_pairs([pages[i][1]["phash"] for i in kept], near_duplicate_bits):
        later = pages[max(kept[a], kept[b])][1]
        if later["status"] == "kept":
            later["status"] = "dropped_duplicate"
            Path(later["image_path"]).unlink(missing_ok=True)
    counts = Counter(p["status"] for _, p in pages)
    kept_pages = [p for _, p in pages if p["status"] == "kept"]
    with (output / "pages.jsonl").open("w", encoding="utf-8") as all_pages, \
            (output / "clean.jsonl.tmp").open("w", encoding="utf-8") as clean:
        for entry, page in pages:
            record = {"id": f"{entry['sha256'][:16]}_p{page['page']:04d}", "source": entry.get("source"),
                      "document": {k: entry.get(k) for k in ("sha256", "type", "url", "title", "license", "terms",
                                                               "archive_member")} | {"page": page["page"]},
                      **{k: v for k, v in page.items() if k != "page"}}
            all_pages.write(json_text({k: v for k, v in record.items() if k != "annotations"}) + "\n")
            if page["status"] == "kept":
                clean.write(json_text(record) + "\n")
    (output / "clean.jsonl.tmp").replace(output / "clean.jsonl")
    with (output / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json_text(document) + "\n")
    summary = {"documents": len(documents), "document_status": dict(Counter(d["status"] for d in documents)),
               "pages": dict(counts), "kept_classes": dict(Counter(p["class"] for p in kept_pages)),
               "kept_flags": {"style_substituted": sum(p["flags"]["style_substituted"] for p in kept_pages),
                              "with_notdef": sum(p["flags"]["notdef_glyphs"] > 0 for p in kept_pages),
                              "with_hidden_lines": sum(p["flags"]["hidden_lines"] > 0 for p in kept_pages)},
               "clean": str(output / "clean.jsonl")}
    atomic_json(output / "summary.json", summary)
    return summary
