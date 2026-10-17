"""Render stored documents into HR pages with exact text boxes from the text layer.

Vector documents can be rendered at any resolution, so each page gets the DPI that puts
its median text-line height at a target drawn from ``text_height`` (pixels): the HR is
never resampled. Pages are classified before anything is kept:

  text      a usable text layer: line quads and transcripts become trusted annotations
  garbled   glyph boxes are exact but the text maps to junk (Founder fonts without
            ToUnicode, private-use or replacement characters): render-only, untrusted boxes
  vector    no text layer but visible vector ink (glyphs converted to outlines): render-only
  scanned   one large image and almost no text: not born-digital, dropped
  blank     nothing visible, dropped

Slide template placeholders (单击此处添加标题 ...) and template-site watermarks drop the
page; near-duplicate pages (repeated template slides, re-uploaded decks) are removed by
pHash across the whole run.
"""

import hashlib
import json
import random
import re
import statistics
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..pipeline import atomic_json, json_text
from .sniff import CONVERTIBLE, cjk_ratio

PLACEHOLDER = re.compile(r"单击此处|点击此处|点击添加|单击添加|请替换|请在此|此处输入|输入标题|标题文本|添加标题|添加文本|"
                         r"添加文字|在此处添加|您的标题|副标题|click (?:here )?to add|add your (?:title|text)|your title|"
                         r"lorem ipsum|xxx{2,}", re.I)
WATERMARK = re.compile(r"1ppt\.com|ypppt\.com|51pptmoban|第一PPT|优品PPT|pptstore|docer\.com", re.I)
GARBLED_FONTS = re.compile(r"FzBookMaker|DLF-|DlFont", re.I)
# Bullet glyphs of symbol fonts sit in the private-use area; they are not text.
SYMBOL_FONTS = re.compile(r"OpenSymbol|Wingdings|Webdings|Symbol|ZapfDingbats|Dingbats|Marlett|MT Extra", re.I)


def font_name(name):
    """Subset prefix removed; GBK names that a producer stored as Latin-1 bytes decoded."""
    name = name.split("+", 1)[1] if re.match(r"^[A-Z]{6}\+", name) else name
    if any("\x80" <= c <= "\xff" for c in name):
        for codec in ("gbk", "utf-8"):
            try:
                return name.encode("latin-1").decode(codec)
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    return name
MAX_PIXELS = 36_000_000


def _targets(spec):
    low, _, high = str(spec).partition(",")
    return float(low), float(high or low)


def _page_seed(seed, digest, number):
    return int.from_bytes(hashlib.sha256(f"{seed}:{digest}:{number}".encode()).digest()[:8], "big")


def _junk_share(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    junk = sum(1 for c in chars if "" <= c <= "" or c == "�" or ord(c) < 32
               or 0xF0000 <= ord(c) <= 0x10FFFF)
    return junk / len(chars)


def page_lines(page):
    """Text lines with quads (unrotated page space), text, font and size."""
    import pymupdf
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip() and not SYMBOL_FONTS.search(s["font"])]
            if not spans:
                continue
            quad = pymupdf.recover_line_quad(line, spans=spans)
            text = "".join(s["text"] for s in spans)
            size = statistics.median(s["size"] for s in spans)
            lines.append({"quad": quad, "text": text, "font": font_name(spans[0]["font"]), "size": size,
                          "height": min(quad.rect.height, quad.rect.width) if line["dir"][0] == 0 else quad.rect.height})
    return lines


def classify(page, lines, ink):
    import pymupdf
    text = "".join(line["text"] for line in lines)
    chars = len([c for c in text if not c.isspace()])
    image_cover = 0.0
    area = abs(page.rect)
    for info in page.get_image_info():
        image_cover += abs(pymupdf.Rect(info["bbox"]) & page.rect) / area if area else 0
    fonts = {line["font"] for line in lines}
    if chars < 10:
        if image_cover > 0.5:
            return "scanned", chars
        return ("vector" if ink > 0.002 else "blank"), chars
    if image_cover > 0.5 and chars < 20:
        return "scanned", chars
    junk = _junk_share(text)
    founder = any(GARBLED_FONTS.search(f) for f in fonts)
    ascii_punct = sum(1 for c in text if c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") / max(chars, 1)
    if junk > 0.02 or (founder and cjk_ratio(text) < 0.2) or (cjk_ratio(text) == 0 and ascii_punct > 0.2 and founder):
        return "garbled", chars
    return "text", chars


def _render_document(task):
    try:
        return _render(*task)
    except Exception as error:  # one malformed file must not stop the run
        return {"sha256": task[0]["sha256"], "pages": [], "status": "error",
                "reason": f"{type(error).__name__}: {error}"[:200]}


def _render(entry, output, options):
    import os
    import pymupdf
    from ..probes import phash
    from .convert import ConversionError, fonts_conf, prepare_profile, to_pdf
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
                    pdf = to_pdf(path, tmp, profile, fonts_conf(work))
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
        gray = np.frombuffer(thumb.samples, np.uint8).reshape(thumb.height, thumb.width, thumb.n).mean(axis=2)
        ink = float((gray < 160).mean())
        page_class, chars = classify(page, lines, ink)
        text = "".join(line["text"] for line in lines)
        record = {"page": number, "class": page_class, "chars": chars,
                  "cjk": round(cjk_ratio(text), 3) if page_class != "vector" else None,
                  "size_pt": [round(page.rect.width, 1), round(page.rect.height, 1)]}
        placeholder = sum(len(m.group()) for line in lines for m in PLACEHOLDER.finditer(line["text"]))
        if page_class in ("scanned", "blank"):
            result["pages"].append(dict(record, status="dropped_" + page_class))
            continue
        if page_class == "text" and record["cjk"] < options["min_cjk"]:
            result["pages"].append(dict(record, status="dropped_language"))
            continue
        if placeholder or any(WATERMARK.search(line["text"]) for line in lines):
            result["pages"].append(dict(record, status="dropped_template"))
            continue
        heights = [line["height"] for line in lines if line["height"] > 1]
        target = random.Random(_page_seed(rng_seed, digest, number)).uniform(low, high)
        zoom = target / statistics.median(heights) if heights else options["default_dpi"] / 72
        zoom = min(zoom, options["max_dpi"] / 72, (MAX_PIXELS / max(abs(page.rect), 1)) ** 0.5)
        zoom = max(zoom, 1.0)
        matrix = page.rotation_matrix * pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, annots=False)
        image_path = Path(output) / "pages" / digest[:2] / f"{digest}_p{number:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(image_path)
        annotations = []
        for line in lines:
            q = line["quad"] * matrix
            annotations.append({"polygon": [[round(p.x, 2), round(p.y, 2)] for p in (q.ul, q.ur, q.lr, q.ll)],
                                "text": line["text"], "font": line["font"], "size_px": round(line["size"] * zoom, 2),
                                "annotation_origin": "pdf_text_layer", "trusted": page_class == "text"})
            fonts_used[line["font"]] += 1
        thumb_rgb = np.frombuffer(thumb.samples, np.uint8).reshape(thumb.height, thumb.width, thumb.n)[..., :3]
        record.update(status="kept", image_path=str(image_path), dpi=round(72 * zoom, 1),
                      size=[pixmap.width, pixmap.height], median_text_px=round(statistics.median(heights) * zoom, 1)
                      if heights else None, annotations=annotations, phash=phash(thumb_rgb))
        result["pages"].append(record)
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
                 max_dpi=600, default_dpi=200, near_duplicate_bits=4, log=print):
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
               "max_dpi": max_dpi, "default_dpi": default_dpi, "work": str(output / "work")}
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
    hashes = [pages[i][1]["phash"] for i in kept]
    for a, b in duplicate_pairs(hashes, near_duplicate_bits):
        later = max(kept[a], kept[b])
        pages[later][1]["status"] = "dropped_duplicate"
    counts = Counter(p["status"] for _, p in pages)
    classes = Counter(p["class"] for _, p in pages if p["status"] == "kept")
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
               "pages": dict(counts), "kept_classes": dict(classes), "clean": str(output / "clean.jsonl")}
    atomic_json(output / "summary.json", summary)
    return summary
