"""Render stored documents into HR pages with exact text boxes from the text layer.

Vector documents can be rendered at any resolution, so each page gets the DPI that puts
its median text-line height (character-weighted, visible lines only, measured across the
line so slanted text counts its glyph height) at a target drawn from ``text_height``
(pixels): the HR is never resampled. The DPI stays within 72 and ``max_dpi`` and a page
within MAX_PIXELS (plans and posters past 25 MP at 72 DPI go below it). Pages are
classified first:

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
Characters rendered under 8 px, or under 0.35 of the smaller of the page's median text
height and their line's largest glyph (anti-copy text of a few points inside body lines),
count as hidden, while small print on its own line (page numbers, captions) stays; lines
that lose characters as hidden count as ``hidden_lines``. The other
letters of a ligature glyph (empty boxes in the text layer) share its box and fate when they
are Latin letters, or the glyph's punctuation mark repeated ("——" drawn as one two-em dash),
as wide as the glyph (fi, ffi, ft: at least 0.8 of their width in MuPDF's Helvetica); other
characters a ToUnicode entry adds to a glyph (a CJK glyph or a blank read as several
characters) make the run untrusted. Glyph boxes come from the text layer unless they
miss the drawn glyph (negative widths in LibreOffice vertical text, vertical writing mode),
glyphs stacked in a column become one vertical line (a tall box read top to bottom),
overlapping copies of a line (IoU > 0.5: shadow, fake bold, overprint; not repeated
characters placed one by one) keep the topmost while the same text further off (a far drop
shadow, the same cell in the next table row) keeps its own label, and fonts with bogus
vertical metrics get glyph boxes of a standard height (Founder FzBookMaker: ascender 0.001,
one em; over 1.5 em, Founder E-BZ: at most 1 em above and 0.25 em below the baseline).
Lines running past the page edge lose the clipped glyphs (``offpage_lines``), quads that stay
degenerate or tower over their glyph size are dropped (``degenerate_lines``) and the rest are
clamped to the image. Lines with .notdef glyphs (missing from every allowed font; the
converter rejects bitmap fallbacks; a page where they are over a tenth of the labelled
characters, e.g. a PDF embedding an empty font subset, is dropped) or with characters a transcript cannot carry (controls,
also \\v \\f \\r: Founder digits, TeX delimiters; private use, U+FFFD; glyphs MuPDF finds no
Unicode for, which the text layer reads as their code; blanks in upright text whose glyph
has an outline, where MuPDF reports one, that leaves ink) lose their trust, symbol-font bullets and Founder
newspaper frame pieces are excluded, and labels keep full-width punctuation (NFKC only for
Kangxi radicals and compatibility ideographs). Pages drawn mostly with non-embedded CJK fonts are flagged
``style_substituted`` (MuPDF draws them with its fallback face). Slide placeholders (in
presentations and landscape PDF pages) and template-site watermarks drop the page, a deck
with more than 30% placeholder pages is dropped whole, and pHash near-duplicates whose
transcripts are near-identical character sequences (a common subsequence of 90% of their
length; under 20 characters, identical text) are removed across the run. A file reached twice
(another URL, inside an archive) is rendered once, with the other provenances under
``also``. A document whose worker dies (crash, OOM kill, ``timeout``) becomes an error and
the run goes on. An optional text detector flags text that the text layer does not cover
(text inside images, WordArt).
"""

import functools
import hashlib
import json
import random
import re
import signal
import statistics
import tempfile
import time
import unicodedata
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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
# Code points a label cannot carry for a drawn glyph: controls and line separators (Founder digits are drawn
# from U+000B-001F, TeX delimiters from U+000C-000D), private use, U+FFFD and the unknown-character U+FFFE/FFFF.
UNTRANSCRIBED = re.compile("[%s]" % "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a, b in (
    (0x00, 0x1F), (0x7F, 0x9F), (0x2028, 0x2029), (0xE000, 0xF8FF), (0xFFFD, 0xFFFF), (0xF0000, 0x10FFFF))))
NFKC_RANGES = ((0x2E80, 0x2FDF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))
# Writing direction -> glyph corners [ul, ur, lr, ll] of a text-layer box (x0, y0, x1, y1).
UPRIGHT = {(1, 0): lambda x0, y0, x1, y1: ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
           (-1, 0): lambda x0, y0, x1, y1: ((x1, y1), (x0, y1), (x0, y0), (x1, y0)),
           (0, 1): lambda x0, y0, x1, y1: ((x1, y0), (x1, y1), (x0, y1), (x0, y0)),
           (0, -1): lambda x0, y0, x1, y1: ((x0, y1), (x0, y0), (x1, y0), (x1, y1))}
PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
LINE_SEPARATORS = {c: "\\u%04x" % c for c in (0x85, 0x2028, 0x2029)}
PROVENANCE = ("source", "url", "title", "license", "terms", "archive_member", "archive_sha256")


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
    """A space; not the controls and line separators that str.isspace() also accepts (Founder fonts draw
    digits from \\v, \\f and U+001C-001F)."""
    return unicodedata.category(c) == "Zs"


@functools.lru_cache(maxsize=1)
def _helvetica():
    import pymupdf
    return pymupdf.Font("helv")


def _ligature(text, width):
    """Whether one glyph ``width`` em wide can draw ``text`` (its ToUnicode characters): a typographic
    ligature of Latin letters (fi, ffl, ft, st) or one punctuation mark repeated (the two-em dash of "——"
    in Noto CJK), about as wide as the characters apart. Measured against MuPDF's Helvetica: the Latin
    ligatures of 27 installed fonts mostly keep 0.8 of its width or more (the narrowest: FreeSans st
    0.66, FreeMono ffi 0.77, which then count as untrusted), a glyph given extra characters (an 'o'
    read as 'of', an em dash as "——") far less."""
    font = _helvetica()
    latin = all(unicodedata.category(c)[0] == "L" and unicodedata.name(c, "").startswith("LATIN ") for c in text)
    if not (latin or len(set(text)) == 1 and unicodedata.category(text[0])[0] == "P") or not all(
            font.has_glyph(ord(c)) for c in text):
        return False
    return width >= 0.8 * sum(font.glyph_advance(ord(c)) for c in text)


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


def _content_text(page, flags=0):
    """rawdict of the page contents only (annotations and form fields are not rendered), in
    unrotated page space like get_text(); ``flags`` are added to TEXTFLAGS_RAWDICT."""
    import pymupdf
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    try:
        textpage = pymupdf.TextPage(page.get_displaylist(annots=False).get_textpage(pymupdf.TEXTFLAGS_RAWDICT | flags))
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


def _refit(quad, origin, direction, size, ascender, descender):
    """A glyph quad rebuilt across its baseline: the same advance, ``ascender`` and ``descender``
    (em) above and below the origin."""
    up = np.array([direction[1], -direction[0]]) * size
    ul, ll = np.add(origin, ascender * up), np.add(origin, descender * up)
    return np.array([ul, ul + quad[1] - quad[0], ll + quad[1] - quad[0], ll])


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
            "junk": bool(UNTRANSCRIBED.search(raw)) or any(glyph.get("suspect") for glyph in glyphs),
            "seqno": max(glyph["seqno"] for glyph in glyphs),
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


def _copy(a, b):
    """Line b a copy of line a with the same text (shadow, fake bold, overprint): their quads overlap
    (IoU > 0.5 in a's baseline frame, exact for parallel lines). The same text further off (a far drop
    shadow, the same cell in the next row of a tight table, the separately placed digits of "11", the
    stacked pieces of a tall bracket) is ink of its own."""
    import pymupdf
    qa, qb = a["quad"], b["quad"]
    if abs(qa.lr - qa.ll) >= 1e-3:
        matrix = pymupdf.planish_line(qa.ll, qa.lr)
        qa, qb = qa * matrix, qb * matrix
    ra, rb = qa.rect, qb.rect
    common = abs(ra & rb)
    return common > 0.5 * (abs(ra) + abs(rb) - common)


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
    image or fill carry ``cover``, blanks drawn with an outline ``outline`` for the renderer to check."""
    import pymupdf
    fonts = page.get_fonts()
    type3 = {font_name(f[3]) for f in fonts if f[2] == "Type3"}
    not_embedded = {font_name(f[3]) for f in fonts if f[1] == "n/a"}
    traced, columns, letters = {}, {}, {}
    for span in page.get_texttrace():
        base = None
        for char in span["chars"]:
            if char[1] < 0 and base:  # no glyph: another letter of the one before (ToUnicode one-to-many)
                letters[id(base)] = letters.get(id(base), "") + chr(char[0])
                continue
            base = char
            x, y = round(char[2][0], 1), round(char[2][1], 1)
            traced.setdefault((x, y), []).append((char, span))
            if span["wmode"] == 1:
                columns.setdefault((x, chr(char[0])), []).append((y, (char, span)))
    lines, blanks = [], {}
    for block in _content_text(page)["blocks"]:
        for line in block.get("lines", []):
            glyphs, match, pending = [], None, ""
            for span in line["spans"]:
                text = "".join(char["c"] for char in span["chars"])
                if SYMBOL_FONTS.search(span["font"]) or (DECORATION_FONTS.search(span["font"]) and text.strip() and
                                                         all(_private(c) for c in text if not _blank(c))):
                    glyphs.append(None)  # bullets and frame pieces are not text: they split the line
                    continue
                # Bogus metrics: glyphs are refitted one em high (Founder FzBookMaker: ascender 0.001) or cut
                # to at most 1 em above and 0.25 em below the baseline (Founder E-BZ: 2.4 em; one comma in
                # such a font would make its line's quad cover the lines around it).
                spread, ascender = span["ascender"] - span["descender"], span["ascender"]
                fit = (None if 0.5 <= spread <= 1.5 else (0.9, -0.1) if spread < 0.5 else
                       (min(ascender, 1.0) if ascender >= 0.5 else 0.9, max(min(span["descender"], -0.1), -0.25)))
                if spread < 1:  # PyMuPDF's boxes are then one em high (JM_char_quad)
                    spread = max(spread, 1e-3)
                    span = dict(span, ascender=span["ascender"] / spread, descender=span["descender"] / spread)
                font = font_name(span["font"])
                for char in span["chars"]:
                    if pending[:1] == char["c"] and glyphs and glyphs[-1]:  # a ligature's next letter, without advance:
                        ul, ur = _box_quad(line["dir"], span, char["bbox"])[:2]
                        if np.hypot(*(ur - ul)) < 0.01 * span["size"]:  # its glyph's box and visibility
                            pending = pending[1:]
                            glyphs.append(dict(glyphs[-1], c=char["c"], joined=True, notdef=False))
                            continue
                    x, y = (round(v, 1) for v in char["origin"])
                    found = traced.get((x, y), [])  # another character there is the next one in vertical writing
                    found = [m for m in found if chr(m[0][0]) == char["c"]] or [m for m in found if not m[1]["wmode"]]
                    if not found:  # vertical writing (wmode 1): the trace's origin is near the top of the cell
                        top = char["bbox"][1]
                        near = [(abs(ty - top), m) for ty, m in columns.get((x, char["c"]), [])
                                if top - 0.5 * span["size"] <= ty < y]
                        found = [min(near, key=lambda item: item[0])[1]] if near else None
                    pending = letters.get(id(found[0][0]), "") if found else ""
                    if found:  # unmatched characters inherit from the glyph before them
                        match = found[0]
                    drawn = match is not None and match[1]["type"] != 3 and match[1]["opacity"] > 0
                    # Vertical writing: the text layer's cell sits one advance below the drawn glyph.
                    upright = match is not None and match[1]["wmode"] == 1
                    box = char["bbox"]
                    quad = (_rect_corners(box[0], match[0][3][1], box[2], match[0][3][3]) if upright and found else
                            _rect_corners(*box) if upright else
                            _glyph_quad(line["dir"], span, box, match[0][3]) if found and drawn else None)
                    drawn = drawn and (quad is not None or not found)
                    quad = _box_quad(line["dir"], span, box) if quad is None else quad
                    if fit and not upright:
                        quad = _refit(quad, char["origin"], line["dir"], span["size"], *fit)
                    # Text that is not the glyph's: letters a ToUnicode entry adds to a glyph that is no
                    # ligature of them, or the code shown for a glyph MuPDF finds no Unicode for.
                    suspect = bool(found) and (match[0][0] == 0xFFFD or bool(pending) and not _ligature(
                        char["c"] + pending, np.hypot(*(quad[1] - quad[0])) / span["size"]))
                    glyphs.append({"c": char["c"], "quad": quad,
                                   "font": font, "size": span["size"], "seqno": match[1]["seqno"] if match else -1,
                                   "drawn": drawn, "upright": upright, "cover": None, "suspect": suspect,
                                   "substituted": font in not_embedded and _is_cjk(char["c"]),
                                   "notdef": bool(found) and match[0][1] == 0 and font not in type3
                                   and not _blank(char["c"])})
                    if found and drawn and _blank(char["c"]) and line["dir"] in UPRIGHT:  # a glyph: see below
                        blanks.setdefault((x, y, char["c"]), []).append((glyphs[-1], box, line["dir"][0] != 0))
            for key, run in groupby(glyphs, key=lambda glyph: glyph and (glyph["drawn"], glyph["upright"])):
                made = _text_line(list(run), visible=key[0], vertical=key[1]) if key else None
                if made:
                    lines.append(made)
    if blanks:  # outline boxes (upright text): a real space has none; where MuPDF has none to give, the cell
        for block in _content_text(page, pymupdf.TEXT_ACCURATE_BBOXES)["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    for char in span["chars"]:
                        (x0, y0, x1, y1), size = char["bbox"], span["size"]
                        key = (round(char["origin"][0], 1), round(char["origin"][1], 1), char["c"])
                        for glyph, cell, horizontal in blanks.get(key, ()):
                            # The renderer checks the ink of an outline across the line, the cell's advance along it.
                            if ((y1 - y0 if horizontal else x1 - x0) >= 0.05 * size
                                    and not np.allclose(char["bbox"], cell, atol=0.02 * size)):
                                glyph["outline"] = ((cell[0], y0, cell[2], y1) if horizontal else
                                                    (x0, cell[1], x1, cell[3]))
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
        near = [other for box, other in others if iou(rect, box) > 0]
        if any(_copy(line, other) for other in near):
            copies.add(id(line))
        elif near:  # the same text close by (a far shadow, the next table row): twins share their boxes' pixels
            line["twin"] = near[0].setdefault("twin", id(near[0]))
        others.append((rect, line))
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


def _blank_ink(rgb, polygon, corners, reference=None, shared=None):
    """Whether a blank's outline box (pixel corners) holds ink: measured like _has_ink against its line's
    background, without pixels of other lines' boxes and without straight lines across the whole box
    (underlines and table rules pass through spaces)."""
    xs, ys = zip(*polygon)
    lx0, lx1 = max(0, int(min(xs))), min(rgb.shape[1], int(max(xs)) + 1)
    ly0, ly1 = max(0, int(min(ys))), min(rgb.shape[0], int(max(ys)) + 1)
    (x0, y0), (x1, y1) = np.round(corners.min(axis=0)).astype(int), np.round(corners.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, lx0), max(y0, ly0), min(x1, lx1), min(y1, ly1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    crop = rgb[ly0:ly1, lx0:lx1].astype(np.int16)
    base = (reference[y0:y1, x0:x1] if reference is not None
            else np.median(np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]), axis=0))
    mask = np.abs(rgb[y0:y1, x0:x1].astype(np.int16) - base).max(axis=2) > 60
    mask[mask.all(axis=1)] = False
    mask[:, mask.all(axis=0)] = False
    if shared is not None and (shared[y0:y1, x0:x1] <= 1).any():
        mask &= shared[y0:y1, x0:x1] <= 1
    return mask.sum() > max(1.0, 0.01 * mask.size)


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
    """_render in a worker process; past ``timeout`` seconds the alarm's default action kills the
    worker (a MuPDF page that never finishes cannot be interrupted from Python)."""
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(task[2]["timeout"])
    try:
        return _render(*task)
    except Exception as error:  # one malformed file must not stop the run
        return {"sha256": task[0]["sha256"], "pages": [], "status": "error",
                "reason": f"{type(error).__name__}: {error}"[:200]}
    finally:
        signal.alarm(0)


def _worker_died(task, elapsed):
    """The result of a document whose worker died; the pages it had written are removed."""
    entry, output, options = task
    for path in (Path(output) / "pages" / entry["sha256"][:2]).glob(entry["sha256"] + "_p*.png"):
        path.unlink()
    timed_out = options["timeout"] and elapsed >= options["timeout"]
    return {"sha256": entry["sha256"], "pages": [], "status": "error",
            "reason": "timeout" if timed_out else "worker_died"}


def _render_all(tasks, workers, log=None):
    """_render_document results in task order, at most ``workers`` documents at a time. A worker
    that dies (native crash, OOM kill, timeout) breaks the pool: it is replaced, and the documents
    it had in flight run again one at a time; one that kills its worker alone is an error."""
    results, alone, done = [None] * len(tasks), [], 0
    for width, todo in ((max(1, workers), list(range(len(tasks)))), (1, alone)):
        while todo:
            running, failed = {}, []
            with ProcessPoolExecutor(max_workers=width) as pool:
                while running or (todo and not failed):
                    while todo and not failed and len(running) < width:
                        try:
                            running[pool.submit(_render_document, tasks[todo[0]])] = todo[0], time.monotonic()
                        except BrokenProcessPool:  # broken before any future said so
                            failed.append(None)
                            break
                        todo.pop(0)
                    finished, _ = wait(running, return_when=FIRST_COMPLETED) if running else ((), ())
                    for future in finished:
                        index, start = running.pop(future)
                        if isinstance(future.exception(), BrokenProcessPool):
                            failed.append((index, time.monotonic() - start))
                            continue
                        results[index], done = future.result(), done + 1
                        if log and done % 10 == 0:
                            log(f"{done}/{len(tasks)} documents rendered")
            failed = [item for item in failed if item]
            if len(failed) == 1:
                index, elapsed = failed[0]
                results[index] = _worker_died(tasks[index], elapsed)
                if log:
                    log(f"{tasks[index][0]['sha256'][:16]}: {results[index]['reason']}")
            else:
                alone.extend(index for index, _ in failed)
    return results


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
            scale = min(0.5, (MAX_PIXELS / 10 / max(abs(page.rect), 1)) ** 0.5)  # A0 and smaller: 0.5
            thumb = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, annots=False)
        except RuntimeError as error:
            result["pages"].append({"page": number, "status": "error", "reason": str(error)[:120]})
            continue
        thumb_rgb = np.frombuffer(thumb.samples, np.uint8).reshape(thumb.height, thumb.width, thumb.n)[..., :3]
        ink = float((thumb_rgb.mean(axis=2) < 160).mean())
        page_class, chars = classify(page, lines, ink)
        used = [line for line in lines if line["visible"] and max(glyph["size"] for glyph in line["glyphs"]) >= 3
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
        zoom = min(max(min(zoom, options["max_dpi"] / 72), 1.0), (MAX_PIXELS / max(abs(page.rect), 1)) ** 0.5)
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
        shared = np.zeros((height, width), np.uint8)  # pixels inside two or more line boxes (twins count once)
        twins = {}
        for line in used:
            points = np.round(_corners(line["quad"] * matrix)).astype(np.int32)
            twins.setdefault(line.get("twin", id(line)), []).append(points)
        for polygons in twins.values():
            points = np.concatenate(polygons)
            (lx0, ly0), (lx1, ly1) = points.min(axis=0).clip(0), points.max(axis=0).clip(0) + 1
            box = np.zeros((max(0, min(ly1, height) - ly0), max(0, min(lx1, width) - lx0)), np.uint8)
            if box.size:
                for polygon in polygons:
                    cv2.fillConvexPoly(box, polygon - (lx0, ly0), 1)
                shared[ly0:ly0 + box.shape[0], lx0:lx0 + box.shape[1]] += box
        pieces, removed, hidden, offpage = [], 0, 0, 0
        linear, shift = np.array(tuple(matrix)[:4]).reshape(2, 2), np.array(tuple(matrix)[4:])
        # Characters rendered far smaller than both the page's text and their own line's largest glyph (a few
        # points of anti-copy text inside body lines) cannot be read; small print on its own line can.
        page_px = statistics.median(heights) * zoom if heights else float("inf")
        for line in used:
            corners = np.array([glyph["quad"] for glyph in line["glyphs"]]) @ linear + shift
            blank = np.array([_blank(glyph["c"]) for glyph in line["glyphs"]])
            joined = np.array([glyph.get("joined", False) for glyph in line["glyphs"]])  # ligature letters
            across = corners[:, 1] - corners[:, 0] if line["vertical"] else corners[:, 3] - corners[:, 0]
            size = np.hypot(across[:, 0], across[:, 1])
            largest = size[~blank].max() if (~blank).any() else 0.0
            legible = size >= max(8.0, 0.35 * min(page_px, largest))
            covered = np.array([glyph["cover"] is not None for glyph in line["glyphs"]])
            inside = ((corners.min(axis=1) >= -1).all(axis=1) & (corners[..., 0].max(axis=1) <= width + 1)
                      & (corners[..., 1].max(axis=1) <= height + 1))
            parts = [_rect_corners(*glyph["cover"]) @ linear + shift if glyph["cover"] is not None else None
                     for glyph in line["glyphs"]]
            inked = blank.copy()
            inked[~blank & ~joined] = _has_ink(rgb, _corners(line["quad"] * matrix), [
                (corners[i], None if parts[i] is None else (*parts[i].min(axis=0), *parts[i].max(axis=0)))
                for i in np.nonzero(~blank & ~joined)[0]], reference, shared)
            keep = blank | (inked & inside & legible & ((reference is not None) | ~covered))
            for i in np.nonzero(joined)[0]:
                keep[i] = i > 0 and keep[i - 1]
            for i in np.nonzero(blank)[0]:  # a blank drawn with a glyph that leaves ink is another character
                outline = line["glyphs"][i].get("outline")
                if outline and _blank_ink(rgb, _corners(line["quad"] * matrix),
                                          _rect_corners(*outline) @ linear + shift, reference, shared):
                    line["glyphs"][i]["suspect"] = line["junk"] = True
            outside = bool((~inside & ~blank).any())
            if keep.all():
                parts = [line]
            else:
                parts = [_text_line([glyph for glyph, _ in run], vertical=line["vertical"])
                         for kept, run in groupby(zip(line["glyphs"], keep), key=lambda item: item[1]) if kept]
                parts = [part for part in parts if part]
                removed += line["chars"] - sum(part["chars"] for part in parts)
            offpage += outside
            hidden += bool((~keep & inside & ~blank).any())  # lines with characters hidden on the page
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
                    and line["height"] <= 3 * max(glyph["size"] for glyph in line["glyphs"])
                    and max(xs) > min(xs) and max(ys) > min(ys)):  # flat, collapsed or towering quad: not a label
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
        elif notdef > 0.1 * max(1, sum(len(a["text"].replace(" ", "")) for a in annotations)):
            status = "dropped_notdef"  # boxes where text should be (an empty embedded font subset): not an HR page
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
    """ok records of the fetch log. Only a line feed ends one (titles may hold U+2028 or U+0085,
    which str.splitlines() also splits on); a last record cut short by a crash is skipped."""
    complete, _, tail = (Path(store) / "fetch_log.jsonl").read_bytes().rpartition(b"\n")
    entries = [json.loads(line) for line in complete.decode("utf-8").split("\n") if line.strip()]
    try:
        entries += [json.loads(tail)] if tail.strip() else []
    except ValueError:  # invalid JSON or UTF-8
        pass
    return [entry for entry in entries if entry.get("status") == "ok"]


def _unique(entries):
    """The first entry per sha256 (a file downloaded twice, or also inside an archive: one
    render, one image path); later copies add their provenance to its ``also`` list."""
    first = {}
    for entry in entries:
        kept = first.setdefault(entry["sha256"], entry)
        if kept is not entry:
            kept.setdefault("also", []).append({k: entry[k] for k in PROVENANCE if entry.get(k) is not None})
    return list(first.values())


def _lcs(a, b):
    """Length of the longest common subsequence of two strings, bit-parallel (Allison-Dix, Hyyrö): one
    big-integer step per character of ``a``."""
    masks = {}
    for i, c in enumerate(b):
        masks[c] = masks.get(c, 0) | 1 << i
    row = full = (1 << len(b)) - 1
    for c in a:
        match = row & masks.get(c, 0)
        row = (row + match) | (row - match)
    return len(b) - (row & full).bit_count()


def _alike(a, b):
    """Two pages' transcripts near-identical as character sequences: a common subsequence of at least 90%
    of their mean length (the quantity difflib's ratio approximates; their character counts bound it first);
    transcripts both under 20 characters must be identical. pHash alone confuses distinct pages of one
    layout (dense text, a themed deck, tables with other numbers, divider slides titled by one word or
    year)."""
    if len(a) < 20 and len(b) < 20:
        return a == b
    total = len(a) + len(b)
    return 2 * (Counter(a) & Counter(b)).total() >= 0.9 * total and 2 * _lcs(a, b) >= 0.9 * total


def _jsonl(record):
    """One JSONL line; U+0085, U+2028 and U+2029 (titles) are escaped: str.splitlines() splits on them."""
    return json_text(record).translate(LINE_SEPARATORS) + "\n"


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
                 max_dpi=600, default_dpi=200, near_duplicate_bits=4, detector=None, max_unlabeled=0.05, timeout=3600,
                 log=print):
    """Render every downloaded document; writes pages.jsonl (all pages) and clean.jsonl. A document
    whose worker dies or that runs past ``timeout`` seconds is an error; the run goes on."""
    from ..cleaning import duplicate_pairs
    from .sniff import RENDERABLE
    output = Path(output).resolve()
    if (output / "clean.jsonl").exists():
        raise ValueError("render output already exists; choose a new directory")
    (output / "work").mkdir(parents=True, exist_ok=True)
    entries = _unique(_expand_archives(_entries(store), output))
    entries = [e for e in entries if e["type"] in RENDERABLE][:limit]
    options = {"seed": seed, "text_height": text_height, "min_cjk": min_cjk, "max_pages": max_pages,
               "max_dpi": max_dpi, "default_dpi": default_dpi, "work": str(output / "work"), "detector": detector,
               "max_unlabeled": max_unlabeled, "timeout": int(timeout or 0)}
    tasks = [(e, str(output), options) for e in entries]
    pages, documents = [], []
    for entry, result in zip(entries, _render_all(tasks, workers, log)):
        documents.append({k: entry.get(k) for k in ("sha256", "type", "source", "terms", "url", "title", "license",
                                                     "also")}
                         | {"status": result["status"], "reason": result.get("reason"),
                            "page_count": result.get("page_count"), "fonts": result.get("fonts")})
        for page in result["pages"]:
            pages.append((entry, page))
    kept = [i for i, (_, p) in enumerate(pages) if p["status"] == "kept"]
    transcript = functools.cache(lambda i: "".join(item["text"] for item in pages[kept[i]][1]["annotations"]))
    for a, b in duplicate_pairs([pages[i][1]["phash"] for i in kept], near_duplicate_bits):
        later = pages[max(kept[a], kept[b])][1]
        if later["status"] == "kept" and _alike(transcript(a), transcript(b)):
            later["status"] = "dropped_duplicate"
    shown = {p["image_path"] for _, p in pages if p["status"] == "kept"}
    for _, page in pages:  # an image a kept page shows stays
        if page["status"] == "dropped_duplicate" and page["image_path"] not in shown:
            Path(page["image_path"]).unlink(missing_ok=True)
    counts = Counter(p["status"] for _, p in pages)
    kept_pages = [p for _, p in pages if p["status"] == "kept"]
    with (output / "pages.jsonl").open("w", encoding="utf-8") as all_pages, \
            (output / "clean.jsonl.tmp").open("w", encoding="utf-8") as clean:
        for entry, page in pages:
            record = {"id": f"{entry['sha256'][:16]}_p{page['page']:04d}", "source": entry.get("source"),
                      "document": {k: entry.get(k) for k in ("sha256", "type", "url", "title", "license", "terms",
                                                               "archive_member", "also")} | {"page": page["page"]},
                      **{k: v for k, v in page.items() if k != "page"}}
            all_pages.write(_jsonl({k: v for k, v in record.items() if k != "annotations"}))
            if page["status"] == "kept":
                clean.write(_jsonl(record))
    (output / "clean.jsonl.tmp").replace(output / "clean.jsonl")
    with (output / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(_jsonl(document))
    summary = {"documents": len(documents), "document_status": dict(Counter(d["status"] for d in documents)),
               "pages": dict(counts), "kept_classes": dict(Counter(p["class"] for p in kept_pages)),
               "kept_flags": {"style_substituted": sum(p["flags"]["style_substituted"] for p in kept_pages),
                              "with_notdef": sum(p["flags"]["notdef_glyphs"] > 0 for p in kept_pages),
                              "with_hidden_lines": sum(p["flags"]["hidden_lines"] > 0 for p in kept_pages)},
               "clean": str(output / "clean.jsonl")}
    atomic_json(output / "summary.json", summary)
    return summary
