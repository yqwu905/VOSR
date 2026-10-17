"""Render stored documents into HR pages with exact text boxes from the text layer.

Vector documents can be rendered at any resolution, so each page gets the DPI that puts
its median text-line height (character-weighted, visible lines only, measured across the
line so slanted text counts its glyph height) at a target drawn from ``text_height``
(pixels): the HR is never resampled. Pages are classified first:

  text      a usable text layer: line quads and transcripts become trusted annotations
  garbled   glyph boxes are exact but the text maps to junk (Founder fonts without
            ToUnicode, private-use, replacement or rare-character soup): render-only,
            untrusted boxes
  vector    no text layer but visible vector ink (glyphs converted to outlines): render-only
  scanned   a large image with little or only invisible (OCR-layer) text: dropped
  blank     nothing visible, dropped

Visibility is decided per character, and a line keeps only runs of drawn characters, each
with its own quad and transcript. Text-layer characters are matched to the drawing trace:
render mode 3, zero opacity and annotation or form-field text (pages render without
annotations) are removed; after rendering, a character must leave ink in its glyph box
(white-on-white, background-coloured, off-page) and, where an image, fill or shading is
drawn over it later, must change the pixels of the page rendered without text (text under
a photo or a scan; a page whose text all lies under its image is reclassified as scanned).
Glyph boxes come from the text layer unless they miss the drawn glyph (negative widths in
LibreOffice vertical text, vertical writing mode), glyphs stacked in a column become one
vertical line (a tall box read top to bottom), overlapping copies of a line (shadow, fake
bold) keep the topmost, and fonts with bogus vertical metrics (Founder FzBookMaker:
ascender 0.001) are measured as size-high. Lines running past the page edge lose the
clipped glyphs (``offpage_lines``), quads that stay degenerate are dropped
(``degenerate_lines``) and the rest are clamped to the image. Lines with .notdef glyphs
(missing from every allowed font; the converter rejects bitmap fallbacks) or with
characters a transcript cannot carry (controls such as Founder digits, private use,
U+FFFD) lose their trust, symbol-font bullets and Founder newspaper frame pieces are
excluded, and labels keep full-width punctuation (NFKC only for Kangxi radicals and
compatibility ideographs). Pages drawn mostly with non-embedded CJK fonts are flagged
``style_substituted`` (MuPDF draws them with its fallback face). Slide placeholders (in
presentations and landscape PDF pages) and template-site watermarks drop the page, a deck
with more than 30% placeholder pages is dropped whole, and pHash near-duplicates are
removed across the run. An optional text detector flags text that the text layer does not
cover (text inside images, WordArt).
"""

import functools
import hashlib
import json
import random
import re
import statistics
import tempfile
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path

import numpy as np

from ..geometry import iou
from ..pipeline import atomic_json, json_text
from .refill import PLACEHOLDER, WATERMARK
from .sniff import CONVERTIBLE, cjk_ratio

GARBLED_FONTS = re.compile(r"FzBookMaker|DLF-|DlFont", re.I)
# Bullet glyphs of symbol fonts sit in the private-use area; they are not text.
SYMBOL_FONTS = re.compile(r"OpenSymbol|Wingdings|Webdings|Symbol|ZapfDingbats|Dingbats|Marlett|MT ?Extra", re.I)
# Founder newspaper fonts (People's Daily) draw dashed caption frames with private-use glyphs.
DECORATION_FONTS = re.compile(r"^(?:[A-Z]{6}\+)?NEU-")
PRESENTATIONS = {"pptx", "ppt", "odp"}
COVERS = {"fill-image", "fill-imgmask", "fill-path", "fill-shade"}
MAX_PIXELS = 25_000_000
# Control characters and line separators are not glyphs and break line-based JSONL readers.
CONTROL = re.compile("[%s]" % "".join(re.escape(chr(c)) for c in [*range(0, 32), *range(0x7F, 0xA0), 0x2028, 0x2029]))
SYMBOL_PUA = range(0xF020, 0xF100)
# Code points a label cannot carry for a drawn glyph: controls other than \t\n\v\f\r and U+0085 (U+001C-001F
# included: Founder digits), private use, U+FFFD and the unknown-character U+FFFE/FFFF.
UNTRANSCRIBED = re.compile("[%s]" % "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a, b in (
    (0x00, 0x08), (0x0E, 0x1F), (0x7F, 0x84), (0x86, 0x9F), (0xE000, 0xF8FF), (0xFFFD, 0xFFFF), (0xF0000, 0x10FFFF))))
NFKC_RANGES = ((0x2E80, 0x2FDF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))
# Writing direction -> glyph corners [ul, ur, lr, ll] of a text-layer box (x0, y0, x1, y1).
UPRIGHT = {(1, 0): lambda x0, y0, x1, y1: ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
           (-1, 0): lambda x0, y0, x1, y1: ((x1, y1), (x0, y1), (x0, y0), (x1, y0)),
           (0, 1): lambda x0, y0, x1, y1: ((x1, y0), (x1, y1), (x0, y1), (x0, y0)),
           (0, -1): lambda x0, y0, x1, y1: ((x0, y1), (x0, y0), (x1, y0), (x1, y1))}
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@functools.lru_cache(maxsize=4096)
def font_name(name):
    """Subset prefix removed; names that a producer stored as Latin-1 bytes decoded.

    GBK accepts most Big5 byte strings too (標楷體 reads as 夹发砰), so when both decode, a
    known font name or style keyword decides, then characters from the common sets (GB2312,
    Big5 level 1), then GBK."""
    name = name.split("+", 1)[1] if re.match(r"^[A-Z]{6}\+", name) else name
    if not any(0x80 <= ord(c) <= 0xFF for c in name):
        return name
    try:
        data = name.encode("latin-1")
        return data.decode("utf-8")  # UTF-8 first: GBK would also accept its bytes
    except UnicodeEncodeError:
        return name
    except UnicodeDecodeError:
        pass
    decoded = {}
    for codec in ("gbk", "big5"):
        try:
            decoded[codec] = data.decode(codec)
        except UnicodeDecodeError:
            pass
    if len(decoded) < 2:
        return next(iter(decoded.values()), name)
    from .convert import FONT_ALIASES, FONT_KEYWORDS
    known = {family for families in FONT_ALIASES.values() for family in families}
    keywords = [keyword for _, words in FONT_KEYWORDS for keyword in words if not keyword.isascii()]

    def common(c, codec):
        if not _is_cjk(c):
            return True
        if codec == "gbk":
            try:
                return bool(c.encode("gb2312"))
            except UnicodeEncodeError:
                return False
        return 0xA440 <= int.from_bytes(c.encode("big5"), "big") <= 0xC67E

    for test in (lambda text, codec: text in known or any(k in text for k in keywords),
                 lambda text, codec: all(common(c, codec) for c in text)):
        for codec, text in decoded.items():
            if test(text, codec):
                return text
    return decoded["gbk"]


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


def _private(c):
    code = ord(c)
    return 0xE000 <= code <= 0xF8FF or code >= 0xF0000


def _blank(c):
    """Whitespace, without the controls U+001C-001F that str.isspace() also accepts (Founder digits)."""
    return c.isspace() and not 0x1C <= ord(c) <= 0x1F


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


def _content_text(page):
    """rawdict of the page contents only (annotations and form fields are not rendered), in
    unrotated page space like get_text()."""
    import pymupdf
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    try:
        textpage = pymupdf.TextPage(page.get_displaylist(annots=False).get_textpage(pymupdf.TEXTFLAGS_RAWDICT))
    finally:
        if rotation:
            page.set_rotation(rotation)
    textpage.parent = page
    return page.get_text("rawdict", textpage=textpage)


def _box_quad(direction, span, box):
    """The glyph quad inside a text-layer box, as corners [ul, ur, lr, ll]: the box itself for
    upright and quarter-turned text, else PyMuPDF's recover_bbox_quad on plain floats."""
    cos, sin = direction
    x0, y0, x1, y1 = box
    if direction in UPRIGHT:
        return np.array(UPRIGHT[direction](x0, y0, x1, y1), float)
    height = span["size"] * (span["ascender"] - span["descender"])
    hs, hc = height * sin, height * cos
    if hc >= 0 and hs <= 0:
        corners = ((x0, y1 - hc), (x1 + hs, y0), (x1, y0 + hc), (x0 - hs, y1))
    elif hc <= 0 and hs <= 0:
        corners = ((x1 + hs, y1), (x0, y0 - hc), (x0 - hs, y0), (x1, y1 + hc))
    elif hc <= 0 and hs >= 0:
        corners = ((x1, y0 - hc), (x0 + hs, y1), (x0, y1 + hc), (x1 - hs, y0))
    else:
        corners = ((x0 + hs, y0), (x1, y1 - hc), (x1 - hs, y1), (x0, y0 + hc))
    return np.array(corners)


def _rect_corners(x0, y0, x1, y1):
    return np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), float)


def _glyph_quad(direction, span, box, drawn):
    """The text layer's glyph quad; where its box misses the drawn glyph's box (LibreOffice vertical
    text: negative /Widths put it one advance away), the drawn advance across the text layer's
    height; None if that cannot be rebuilt (not axis-aligned)."""
    (bx0, by0, bx1, by1), (dx0, dy0, dx1, dy1) = box, drawn
    area, drawn_area = (bx1 - bx0) * (by1 - by0), (dx1 - dx0) * (dy1 - dy0)
    common = max(0, min(bx1, dx1) - max(bx0, dx0)) * max(0, min(by1, dy1) - max(by0, dy0))
    if bx1 <= bx0 or by1 <= by0 or dx1 <= dx0 or dy1 <= dy0 or common >= 0.5 * min(area, drawn_area):
        return _box_quad(direction, span, box)
    if direction in ((1, 0), (-1, 0)):
        return _box_quad(direction, span, (dx0, by0, dx1, by1))
    if direction in ((0, 1), (0, -1)):
        return _box_quad(direction, span, (bx0, dy0, bx1, dy1))
    return None


def _run_quad(quads, vertical=False):
    """Corners around consecutive glyphs, aligned with their writing direction (glyphs on shifted
    baselines included); a vertical column is the rectangle around its glyphs."""
    if len(quads) == 1:
        return quads[0]
    points = np.concatenate(quads)
    direction = next((q[1] - q[0] for q in quads if abs(q[1] - q[0]).sum() > 1e-3), quads[-1][2] - quads[0][3])
    length = np.hypot(*direction)
    u = direction / length if length > 1e-3 and not vertical else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])
    a, b = points @ u, points @ v
    return np.array([a.min() * u + b.min() * v, a.max() * u + b.min() * v,
                     a.max() * u + b.max() * v, a.min() * u + b.max() * v])


def _text_line(glyphs, visible=True, vertical=False):
    """A line from consecutive glyphs, spaces at either end dropped; None if nothing is left."""
    import pymupdf
    while glyphs and _blank(glyphs[0]["c"]):
        glyphs = glyphs[1:]
    while glyphs and _blank(glyphs[-1]["c"]):
        glyphs = glyphs[:-1]
    if not glyphs:
        return None
    ul, ur, lr, ll = _run_quad([glyph["quad"] for glyph in glyphs], vertical)
    raw = "".join(glyph["c"] for glyph in glyphs)
    marks = [glyph for glyph in glyphs if not _blank(glyph["c"])]
    return {"quad": pymupdf.Quad(tuple(ul), tuple(ur), tuple(ll), tuple(lr)), "raw": raw, "text": label(raw),
            "font": glyphs[0]["font"], "size": statistics.median(glyph["size"] for glyph in marks),
            "chars": len(marks), "height": float(ur[0] - ul[0] if vertical else np.hypot(*(ll - ul))),
            "visible": visible, "vertical": vertical, "notdef": sum(glyph["notdef"] for glyph in glyphs),
            "substituted_cjk": sum(glyph["substituted"] for glyph in glyphs),
            "junk": bool(UNTRANSCRIBED.search(raw)), "seqno": max(glyph["seqno"] for glyph in glyphs),
            "glyphs": glyphs}


def _covered(boxes, seqnos, log):
    """Per glyph, the part of its box under the largest image, fill or shading drawn after it
    (when at least half is), else NaN. Only candidates: clipping and transparency are unknown
    here, the renderer checks pixels."""
    covers = np.array([[i, *rect] for i, (kind, rect) in enumerate(log) if kind in COVERS], float).reshape(-1, 5)
    parts = np.full((len(boxes), 4), np.nan)
    covers = covers[covers[:, 0] > seqnos.min()] if len(boxes) else covers[:0]
    if not len(covers):
        return parts
    half = 0.5 * (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    step = max(1, 2_000_000 // len(covers))
    for a in range(0, len(boxes), step):
        box = boxes[a:a + step, None]
        w = np.minimum(box[..., 2], covers[:, 3]) - np.maximum(box[..., 0], covers[:, 1])
        h = np.minimum(box[..., 3], covers[:, 4]) - np.maximum(box[..., 1], covers[:, 2])
        area = np.where((covers[:, 0] > seqnos[a:a + step, None]) & (w > 0) & (h > 0), w * h, 0)
        best = area.argmax(axis=1)
        for i in np.nonzero(area[np.arange(len(best)), best] >= half[a:a + step])[0]:
            cover, glyph = covers[best[i], 1:], boxes[a + i]
            parts[a + i] = np.maximum(glyph[:2], cover[:2]).tolist() + np.minimum(glyph[2:], cover[2:]).tolist()
    return parts


def _overlap(a, b):
    """IoU of two quads in the frame of the first one's baseline (exact for parallel lines)."""
    import pymupdf
    if abs(a.lr - a.ll) < 1e-3:
        return iou(tuple(a.rect), tuple(b.rect))
    matrix = pymupdf.planish_line(a.ll, a.lr)
    return iou(tuple((a * matrix).rect), tuple((b * matrix).rect))


def _stacked(above, below):
    """Upright glyphs (single-glyph lines or columns) one under the other: vertical writing."""
    if not all(line["vertical"] or len(line["glyphs"]) == 1 for line in (above, below)):
        return False
    a, b = above["glyphs"][-1]["quad"], below["glyphs"][0]["quad"]
    (ax0, ay0), (ax1, _), (bx0, by0), (bx1, _) = a.min(axis=0), a.max(axis=0), b.min(axis=0), b.max(axis=0)
    size = below["size"]
    return (abs(above["size"] - size) <= 0.1 * size and abs(a[0, 1] - a[1, 1]) < 0.01 and abs(b[0, 1] - b[1, 1]) < 0.01
            and abs(ax0 - bx0) <= 0.25 * size and abs(ax1 - bx1) <= 0.25 * size and 0.5 * size <= by0 - ay0 <= 1.6 * size)


def page_lines(page):
    """Text runs with quads (unrotated page space), labels, font, size, glyphs and per-line flags.

    Runs of invisible text (OCR layers) are returned with visible=False; glyphs under a later
    image or fill carry ``cover`` for the renderer to check."""
    fonts = page.get_fonts()
    type3 = {font_name(f[3]) for f in fonts if f[2] == "Type3"}
    not_embedded = {font_name(f[3]) for f in fonts if f[1] == "n/a"}
    traced, columns = {}, {}
    for span in page.get_texttrace():
        for char in span["chars"]:
            x, y = round(char[2][0], 1), round(char[2][1], 1)
            traced.setdefault((x, y), []).append((char, span))
            if span["wmode"] == 1:
                columns.setdefault((x, chr(char[0])), []).append((y, (char, span)))
    lines = []
    for block in _content_text(page)["blocks"]:
        for line in block.get("lines", []):
            glyphs, match = [], None
            for span in line["spans"]:
                text = "".join(char["c"] for char in span["chars"])
                if SYMBOL_FONTS.search(span["font"]) or (DECORATION_FONTS.search(span["font"]) and text.strip() and
                                                         all(_private(c) for c in text if not _blank(c))):
                    glyphs.append(None)  # bullets and frame pieces are not text: they split the line
                    continue
                if not 0.5 <= span["ascender"] - span["descender"] <= 3:  # bogus metrics: MuPDF boxes are size-high
                    span = dict(span, ascender=1.0, descender=0.0)
                font = font_name(span["font"])
                for char in span["chars"]:
                    x, y = (round(v, 1) for v in char["origin"])
                    found = traced.get((x, y), [])  # another character there is the next one in vertical writing
                    found = [m for m in found if chr(m[0][0]) == char["c"]] or [m for m in found if not m[1]["wmode"]]
                    if not found:  # vertical writing (wmode 1): the trace's origin is near the top of the cell
                        top = char["bbox"][1]
                        near = [(abs(ty - top), m) for ty, m in columns.get((x, char["c"]), [])
                                if top - 0.5 * span["size"] <= ty < y]
                        found = [min(near, key=lambda item: item[0])[1]] if near else None
                    if found:  # unmatched characters (ligature parts) inherit from the glyph before them
                        match = found[0]
                    drawn = match is not None and match[1]["type"] != 3 and match[1]["opacity"] > 0
                    # Vertical writing: the text layer's cell sits one advance below the drawn glyph.
                    upright = match is not None and match[1]["wmode"] == 1
                    box = char["bbox"]
                    quad = (_rect_corners(box[0], match[0][3][1], box[2], match[0][3][3]) if upright and found else
                            _rect_corners(*box) if upright else
                            _glyph_quad(line["dir"], span, box, match[0][3]) if found and drawn else None)
                    drawn = drawn and (quad is not None or not found)
                    glyphs.append({"c": char["c"], "quad": _box_quad(line["dir"], span, box) if quad is None else quad,
                                   "font": font, "size": span["size"], "seqno": match[1]["seqno"] if match else -1,
                                   "drawn": drawn, "upright": upright, "cover": None,
                                   "substituted": font in not_embedded and _is_cjk(char["c"]),
                                   "notdef": bool(found) and match[0][1] == 0 and font not in type3
                                   and not _blank(char["c"])})
            for key, run in groupby(glyphs, key=lambda glyph: glyph and (glyph["drawn"], glyph["upright"])):
                made = _text_line(list(run), visible=key[0], vertical=key[1]) if key else None
                if made:
                    lines.append(made)
    shown = [glyph for line in lines if line["visible"] for glyph in line["glyphs"]]
    if shown:
        corners = np.array([glyph["quad"] for glyph in shown])
        parts = _covered(np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1),
                         np.array([glyph["seqno"] for glyph in shown]), page.get_bboxlog())
        for glyph, part in zip(shown, parts):
            glyph["cover"] = None if np.isnan(part[0]) else tuple(part)
    seen, copies = {}, set()
    for line in sorted((line for line in lines if line["visible"]), key=lambda line: -line["seqno"]):
        others = seen.setdefault(line["text"], [])  # shadow, fake bold, overprint: keep the topmost copy
        rect = tuple(line["quad"].rect)
        if any(iou(rect, box) > 0 and _overlap(line["quad"], quad) > 0.5 for box, quad in others):
            copies.add(id(line))
        others.append((rect, line["quad"]))
    merged = []
    for line in lines:
        if id(line) in copies:
            continue
        if merged and line["visible"] and merged[-1]["visible"] and _stacked(merged[-1], line):
            line = _text_line(merged.pop()["glyphs"] + line["glyphs"], vertical=True)
        merged.append(line)
    return merged


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


def _has_ink(rgb, polygon, glyphs, reference=None, shared=None):
    """Which glyphs of a line leave a visible mark (white-on-white, covered, clipped, hidden: none).

    ``glyphs`` are (corners, part) in pixels; a glyph with a ``part`` under a later image or fill
    must show there (a sliver peeking past a panel does not count). Pixels that other lines'
    boxes also cover (``shared`` > 1) are left out where the glyph has others. A glyph's pixels
    must differ from ``reference`` (the page rendered without text) or, without one, from the
    line's background (border median). Colour channels are compared separately: white text on
    a cyan shape differs from it by only ~40 grey levels but by >60 in red."""
    import cv2
    xs, ys = zip(*polygon)
    x0, x1 = max(0, int(min(xs))), min(rgb.shape[1], int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(rgb.shape[0], int(max(ys)) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2 or not glyphs:
        return [False] * len(glyphs)
    crop = rgb[y0:y1, x0:x1].astype(np.int16)
    base = (reference[y0:y1, x0:x1] if reference is not None
            else np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]), axis=0))
    mask = (np.abs(crop - base).max(axis=2) > 60).astype(np.uint8)
    own = (shared[y0:y1, x0:x1] <= 1).astype(np.uint8) if shared is not None else np.ones_like(mask)
    # Upright glyph boxes: pixel counts from summed-area tables; other glyphs are filled one by one.
    sums = [cv2.integral(m) for m in (mask, own, mask & own)]
    pixels = np.round(np.array([corners for corners, _ in glyphs], float) - (x0, y0)).astype(int)
    parts = np.array([part if part is not None else (-1e9, -1e9, 1e9, 1e9) for _, part in glyphs], float)
    size = np.array(mask.shape[::-1])
    low = np.clip(np.maximum(pixels.min(axis=1), np.floor(parts[:, :2] - (x0, y0))), 0, size).astype(int)
    high = np.clip(np.minimum(pixels.max(axis=1) + 1, np.ceil(parts[:, 2:] - (x0, y0))), low, size).astype(int)
    count = [table[high[:, 1], high[:, 0]] - table[low[:, 1], high[:, 0]] - table[high[:, 1], low[:, 0]]
             + table[low[:, 1], low[:, 0]] for table in sums]
    area, ink = np.where(count[1] > 0, count[1], np.prod(high - low, axis=1)), np.where(count[1] > 0, count[2], count[0])
    upright = ((pixels[:, 0, 1] == pixels[:, 1, 1]) & (pixels[:, 2, 1] == pixels[:, 3, 1])
               & (pixels[:, 0, 0] == pixels[:, 3, 0]) & (pixels[:, 1, 0] == pixels[:, 2, 0]))
    for i in np.nonzero(~upright)[0]:
        (gx0, gy0), (gx1, gy1) = pixels[i].min(axis=0).clip(0), (pixels[i].max(axis=0) + 1).clip(0)
        fill = np.zeros_like(mask[gy0:gy1, gx0:gx1])
        if fill.size:
            cv2.fillConvexPoly(fill, pixels[i] - (gx0, gy0), 1)
            fill[:, :max(0, low[i, 0] - gx0)], fill[:, max(0, high[i, 0] - gx0):] = 0, 0
            fill[:max(0, low[i, 1] - gy0)], fill[max(0, high[i, 1] - gy0):] = 0, 0
            alone = fill & own[gy0:gy1, gx0:gx1]
            fill = alone if alone.any() else fill
        area[i], ink[i] = int(fill.sum()), int((fill & mask[gy0:gy1, gx0:gx1]).sum())
    return list((area > 0) & (ink > np.maximum(1.0, 0.002 * area)))


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


def _corners(quad):
    return [[p.x, p.y] for p in (quad.ul, quad.ur, quad.lr, quad.ll)]


def _render(entry, output, options):
    import os
    import cv2
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
    fonts_used, scratch = Counter(), None
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
        used = [line for line in lines if line["visible"] and line["size"] >= 3
                and (line["quad"] * page.rotation_matrix).rect.intersects(page.rect)]
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
        # Placeholder wording (单击添加, 副标题, 20XX) is also ordinary text in manuals: slides only.
        slides = entry["type"] in PRESENTATIONS or page.rect.width > page.rect.height
        if any((slides and PLACEHOLDER.search(line["text"])) or WATERMARK.search(line["text"]) for line in used):
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
        width, height, reference = pixmap.width, pixmap.height, None
        if any(glyph["cover"] is not None for line in used for glyph in line["glyphs"]):
            try:  # text under a later image or fill: compare with the page rendered without text
                scratch = scratch or pymupdf.open(path)
                bare = scratch[number]
                bare.add_redact_annot(pymupdf.Rect(-1e5, -1e5, 1e5, 1e5), fill=False, cross_out=False)
                bare.apply_redactions(images=0, graphics=0, text=0)
                bare = bare.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False,
                                       annots=False)
                if (bare.width, bare.height) == (width, height):
                    reference = np.frombuffer(bare.samples, np.uint8).reshape(height, width, 3).astype(np.int16)
            except (RuntimeError, ValueError):
                pass  # covered glyphs then count as hidden
        shared = np.zeros((height, width), np.uint8)  # pixels inside two or more line boxes
        for line in used:
            points = np.round(_corners(line["quad"] * matrix)).astype(np.int32)
            (lx0, ly0), (lx1, ly1) = points.min(axis=0).clip(0), points.max(axis=0).clip(0) + 1
            box = np.zeros((max(0, min(ly1, height) - ly0), max(0, min(lx1, width) - lx0)), np.uint8)
            if box.size:
                cv2.fillConvexPoly(box, points - (lx0, ly0), 1)
                shared[ly0:ly0 + box.shape[0], lx0:lx0 + box.shape[1]] += box
        pieces, removed, hidden, offpage = [], 0, 0, 0
        linear, shift = np.array(tuple(matrix)[:4]).reshape(2, 2), np.array(tuple(matrix)[4:])
        for line in used:
            corners = np.array([glyph["quad"] for glyph in line["glyphs"]]) @ linear + shift
            blank = np.array([_blank(glyph["c"]) for glyph in line["glyphs"]])
            covered = np.array([glyph["cover"] is not None for glyph in line["glyphs"]])
            inside = ((corners.min(axis=1) >= -1).all(axis=1) & (corners[..., 0].max(axis=1) <= width + 1)
                      & (corners[..., 1].max(axis=1) <= height + 1))
            parts = [_rect_corners(*glyph["cover"]) @ linear + shift if glyph["cover"] is not None else None
                     for glyph in line["glyphs"]]
            inked = blank.copy()
            inked[~blank] = _has_ink(rgb, _corners(line["quad"] * matrix), [
                (corners[i], None if parts[i] is None else (*parts[i].min(axis=0), *parts[i].max(axis=0)))
                for i in np.nonzero(~blank)[0]], reference, shared)
            keep = blank | (inked & inside & ((reference is not None) | ~covered))
            outside = bool((~inside & ~blank).any())
            if keep.all():
                parts = [line]
            else:
                parts = [_text_line([glyph for glyph, _ in run], vertical=line["vertical"])
                         for kept, run in groupby(zip(line["glyphs"], keep), key=lambda item: item[1]) if kept]
                parts = [part for part in parts if part]
                removed += line["chars"] - sum(part["chars"] for part in parts)
            offpage += outside
            hidden += not parts and not outside
            pieces.extend(parts)
        if removed:  # text found hidden only now (under an image, off the page): judge the page again
            shown = {id(line) for line in used}
            page_class, chars = classify(page, [line for line in lines if id(line) not in shown] + pieces + [
                {"text": "", "font": "", "chars": removed, "visible": False}], ink)
            record.update({"class": page_class, "chars": chars})
            if page_class in ("scanned", "blank"):
                result["pages"].append(dict(record, status="dropped_" + page_class))
                continue
        annotations, degenerate, notdef, substituted, cjk_total = [], 0, 0, 0, 0
        for line in pieces:
            q = line["quad"] * matrix
            polygon = [[round(min(max(p.x, 0), width), 2), round(min(max(p.y, 0), height), 2)]
                       for p in (q.ul, q.ur, q.lr, q.ll)]
            xs, ys = zip(*polygon)
            if not (abs(q.ll - q.ul) >= 0.25 * line["size"] * zoom and abs(q.ur - q.ul) >= 1
                    and max(xs) > min(xs) and max(ys) > min(ys)):  # flat or collapsed quad: not a label
                degenerate += 1
                continue
            notdef += line["notdef"]
            substituted += line["substituted_cjk"]
            cjk_total += sum(1 for c in line["text"] if _is_cjk(c))
            annotations.append({"polygon": polygon, "text": line["text"], "font": line["font"],
                                "size_px": round(line["size"] * zoom, 2), "annotation_origin": "pdf_text_layer",
                                "trusted": page_class == "text" and line["notdef"] == 0 and not line["junk"]})
            fonts_used[line["font"]] += 1
        flags = {"hidden_lines": hidden, "offpage_lines": offpage, "degenerate_lines": degenerate,
                 "notdef_glyphs": notdef, "style_substituted": bool(cjk_total and substituted / cjk_total >= 0.1)}
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
