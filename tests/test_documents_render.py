"""Document conversion and rendering: font aliases, macro refusal, page classes, DPI targeting, boxes."""

import io
import json
import math
import os
import shutil
import subprocess
import zipfile

import numpy as np
import pytest

pymupdf = pytest.importorskip("pymupdf")

from data_pipeline.documents.convert import ConversionError, fonts_conf, has_macros, prepare_profile, to_pdf
from data_pipeline.documents.fetch import Store
from data_pipeline.documents.render import classify, font_name, page_lines, render_store
from data_pipeline.geometry import regions

ZH = "关于进一步优化营商环境的若干措施"


def installed(family):
    out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True).stdout if shutil.which("fc-list") else ""
    return family in out


@pytest.mark.skipif(not installed("Noto Serif CJK SC") or not installed("AR PL UKai CN"), reason="CJK fonts missing")
def test_font_aliases_keep_song_kai_and_hei_styles(tmp_path):
    env = dict(os.environ, FONTCONFIG_FILE=str(fonts_conf(tmp_path)))
    match = lambda family: subprocess.run(["fc-match", family, "family"], capture_output=True, text=True,
                                          env=env).stdout.strip()
    assert match("微软雅黑") == "Noto Sans CJK SC"
    assert match("仿宋_GB2312") == "Noto Serif CJK SC"
    assert match("楷体") == "AR PL UKai CN"
    assert match("汉仪中黑简") == "Noto Sans CJK SC"  # unknown family, matched by name fragment
    assert match("方正小标宋_GBK") == "Noto Serif CJK SC"
    assert match("Arial") != "Noto Sans CJK SC"  # Latin fonts are untouched


def test_macro_documents_are_refused(tmp_path):
    path = tmp_path / "deck.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", "x")
        archive.writestr("ppt/vbaProject.bin", "x")
    assert has_macros(path)
    with pytest.raises(ConversionError, match="macros"):
        to_pdf(path, tmp_path / "out", tmp_path / "profile", tmp_path / "fonts.conf")


def test_font_name_cleanup():
    assert font_name("BAAAAA+NotoSerifCJKsc-Regular") == "NotoSerifCJKsc-Regular"
    assert font_name("宋体".encode("gbk").decode("latin-1")) == "宋体"
    # Big5 names also decode as GBK (標楷體 -> 夹发砰): a known family or style keyword decides.
    assert font_name("BAAAAA+" + "標楷體".encode("big5").decode("latin-1")) == "標楷體"
    assert font_name("華康中黑體".encode("big5").decode("latin-1")) == "華康中黑體"
    assert font_name("仿宋_GB2312".encode("gbk").decode("latin-1")) == "仿宋_GB2312"


def pdf_with(pages):
    document = pymupdf.open()
    for build in pages:
        page = document.new_page(width=595, height=842)
        build(page)
    return document


def text_page(page, size=12, text=ZH):
    for i in range(12):
        page.insert_text((60, 80 + i * 2.2 * size), text, fontsize=size, fontname="china-s")


def test_page_classes():
    def scanned(page):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 280), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)

    def vector(page):
        for i in range(20):
            page.draw_rect(pymupdf.Rect(60, 60 + i * 30, 400, 75 + i * 30), color=(0, 0, 0), fill=(0, 0, 0))

    document = pdf_with([text_page, scanned, vector, lambda page: None])
    found = []
    for page in document:
        thumb = page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5), alpha=False)
        gray = np.frombuffer(thumb.samples, np.uint8).reshape(thumb.height, thumb.width, 3).mean(axis=2)
        found.append(classify(page, page_lines(page), float((gray < 160).mean()))[0])
    assert found == ["text", "scanned", "vector", "blank"]
    blank = document[3]
    private_use = [{"text": "\ue001\ue002\ue003 第一章 总则 \ue004\ue005\ue006", "font": "SimSun"}]
    founder = [{"text": "!\"#$%&'()*+,-./:;<=>? ;:!", "font": "FzBookMaker0DlFont005368"}]
    assert classify(blank, private_use, 0.05)[0] == "garbled" == classify(blank, founder, 0.05)[0]


def store_with_pdf(tmp_path, document, source="test", key="k"):
    store = Store(tmp_path / "store")
    digest, kind, path = store.put(document.tobytes())
    store.record({"key": key, "source": source, "status": "ok", "sha256": digest, "type": kind, "path": str(path),
                  "terms": "official", "title": "t"})
    return digest


def rendered(tmp_path, **options):
    """render_store over the test store: (pages.jsonl rows, clean.jsonl rows)."""
    render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None, **options)
    read = lambda name: [json.loads(line) for line in (tmp_path / "out" / name).read_text().splitlines()]
    return read("pages.jsonl"), read("clean.jsonl")


def cjk_font(name, flat_metrics=False, controls=False, decoration=""):
    """MuPDF's built-in CJK face, renamed; optionally with Founder defects: ascender 0.001, and
    digits reachable only from code points 0x10+d, so the text layer reads controls (9 is drawn
    from glyph 31 and reads U+001F, which str.isspace() accepts). ``decoration`` glyphs are only
    reachable from U+E000+i."""
    ttlib = pytest.importorskip("fontTools.ttLib")
    from fontTools import subset
    font = ttlib.TTFont(io.BytesIO(pymupdf.Font("cjk").buffer))
    subsetter = subset.Subsetter(subset.Options(layout_features=[], notdef_outline=True, retain_gids=True))
    subsetter.populate(text=ZH + "（）—年第条0123456789>" + decoration)
    subsetter.subset(font)
    if flat_metrics:
        ascent = max(1, round(0.001 * font["head"].unitsPerEm))
        font["hhea"].ascent, font["hhea"].descent = ascent, 0
        os2 = font["OS/2"]
        os2.sTypoAscender, os2.sTypoDescender, os2.usWinAscent, os2.usWinDescent = ascent, 0, ascent, 0
    glyphs = dict(font.getBestCmap())
    if controls:
        font["glyf"]["greater"], font["hmtx"]["greater"] = font["glyf"]["nine"], font["hmtx"]["nine"]
    for table in font["cmap"].tables:
        if table.isUnicode():
            for d in "0123456789>" if controls else "":
                table.cmap.pop(ord(d))
            for d in "012345678" if controls else "":
                table.cmap[0x10 + int(d)] = glyphs[ord(d)]
            if controls:
                table.cmap[0x1F] = "greater"
            for i, c in enumerate(decoration):
                table.cmap[0xE000 + i] = table.cmap.pop(ord(c))
    for record in font["name"].names:
        if record.nameID in (1, 4, 6):
            record.string = name
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


def test_render_targets_text_height_and_boxes_cover_ink(tmp_path):
    def rotated(page):
        text_page(page, 10)
        page.set_rotation(90)

    def placeholder(page):  # a slide: placeholders only count on presentations and landscape pages
        page.set_mediabox(pymupdf.Rect(0, 0, 842, 595))
        page.insert_text((60, 80), "单击此处添加标题", fontsize=24, fontname="china-s")
        text_page(page, 12)

    document = pdf_with([lambda p: text_page(p, 12), lambda p: text_page(p, 9), rotated, placeholder,
                         lambda p: text_page(p, 12)])
    store_with_pdf(tmp_path, document)
    summary = render_store(tmp_path / "store", tmp_path / "out", text_height="40", workers=1, log=None)
    rows = [json.loads(line) for line in (tmp_path / "out" / "clean.jsonl").read_text().splitlines()]
    pages = {json.loads(line)["document"]["page"]: json.loads(line)
             for line in (tmp_path / "out" / "pages.jsonl").read_text().splitlines()}
    assert pages[3]["status"] == "dropped_template" and pages[4]["status"] == "dropped_duplicate"
    assert summary["pages"]["kept"] == 3 == len(rows)
    for row in rows:
        assert abs(row["median_text_px"] - 40) < 2
        from PIL import Image
        image = np.asarray(Image.open(row["image_path"]).convert("L"))
        assert list(image.shape[::-1]) == row["size"]
        regions(row["annotations"], row["size"])  # the manifest contract: polygons inside the image
        for item in row["annotations"]:
            xs, ys = zip(*item["polygon"])
            box = image[int(min(ys)):int(max(ys)) + 1, int(min(xs)):int(max(xs)) + 1]
            assert item["trusted"] and item["text"] == ZH and box.size and (box < 128).mean() > 0.05


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice missing")
def test_word_document_is_converted_and_labelled(tmp_path):
    docx = pytest.importorskip("docx")
    from docx.oxml.ns import qn
    document = docx.Document()
    for _ in range(6):
        run = document.add_paragraph().add_run(ZH)
        run.font.name = "仿宋_GB2312"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "仿宋_GB2312")
    path = tmp_path / "notice.docx"
    document.save(path)
    store = Store(tmp_path / "store")
    digest, kind, stored = store.put(path.read_bytes())
    store.record({"key": "k", "source": "t", "status": "ok", "sha256": digest, "type": kind, "path": str(stored),
                  "terms": "official"})
    render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None)
    row = json.loads((tmp_path / "out" / "clean.jsonl").read_text().splitlines()[0])
    assert row["class"] == "text" and row["annotations"][0]["text"] == ZH
    assert "NotoSerifCJK" in row["annotations"][0]["font"]  # 仿宋 rendered in a Song-style face


def test_control_characters_never_reach_the_manifest(tmp_path):
    document = pdf_with([lambda p: text_page(p, 12)])
    store_with_pdf(tmp_path, document)
    import data_pipeline.documents.render as render
    original = render.page_lines

    def with_controls(page):
        lines = original(page)
        for line in lines:
            line["raw"] = line["text"] + "\x85\u2028"
            line["text"] = render.CONTROL.sub("", line["raw"])
        return lines

    render.page_lines = with_controls
    try:
        render.render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None)
    finally:
        render.page_lines = original
    text = (tmp_path / "out" / "clean.jsonl").read_text(encoding="utf-8")
    assert len(text.splitlines()) == len(text.split("\n")) - 1 == 1


def test_labels_keep_fullwidth_punctuation_and_fix_radicals():
    from data_pipeline.documents.render import label
    kangxi_fang = chr(0x2F45)  # KANGXI RADICAL SQUARE, drawn like 方
    assert label("第一条，（总则）：" + kangxi_fang + "案") == "第一条，（总则）：方案"
    assert label("要点" + chr(0xF0B7)) == "要点"  # symbol-font bullet code point dropped
    assert font_name("宋体".encode("utf-8").decode("latin-1")) == "宋体"


def test_rare_character_soup_is_garbled():
    blank = pdf_with([lambda page: None])[0]
    soup = "".join(chr(c) for c in range(0x9F00, 0x9F40))  # valid CJK, but outside GB2312/Big5
    assert classify(blank, [{"text": soup, "font": "SimSun"}], 0.05)[0] == "garbled"


def test_hidden_notdef_and_ocr_layer_pages(tmp_path):
    def hidden_line(page):
        text_page(page, 12)
        page.insert_text((60, 700), "隐藏广告链接 www.example.com", fontsize=12, fontname="china-s", color=(1, 1, 1))
        # Coloured text as bright as its background (grey difference 0) is visible and keeps its box.
        page.draw_rect(pymupdf.Rect(50, 730, 400, 780), color=None, fill=(0.9, 0.3, 0.6))
        page.insert_text((60, 762), "等亮度彩色文字", fontsize=20, fontname="china-s", color=(0.3, 0.75, 0.75))

    def ocr_layer(page):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 280), False)
        pix.clear_with(230)
        page.insert_image(page.rect, pixmap=pix)
        for i in range(10):
            page.insert_text((60, 80 + 30 * i), ZH, fontsize=12, fontname="china-s", render_mode=3)

    store_with_pdf(tmp_path, pdf_with([hidden_line, ocr_layer]))
    render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None)
    pages = [json.loads(line) for line in (tmp_path / "out" / "pages.jsonl").read_text().splitlines()]
    assert pages[0]["status"] == "kept" and pages[0]["flags"]["hidden_lines"] == 1
    assert pages[1]["status"] == "dropped_scanned"
    rows = [json.loads(line) for line in (tmp_path / "out" / "clean.jsonl").read_text().splitlines()]
    assert all("隐藏" not in a["text"] for a in rows[0]["annotations"])
    assert any(a["text"] == "等亮度彩色文字" for a in rows[0]["annotations"])


def test_template_deck_is_dropped_whole(tmp_path):
    def slide(text):
        return lambda page: (page.set_mediabox(pymupdf.Rect(0, 0, 842, 595)),
                             page.insert_text((60, 80), text, fontsize=24, fontname="china-s"), text_page(page, 12))

    pages = [slide("单击此处添加标题")] * 2 + [slide("第" + str(i) + "章 正文内容") for i in range(3)]
    store_with_pdf(tmp_path, pdf_with(pages))
    summary = render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None)
    assert summary["pages"].get("kept", 0) == 0 and summary["pages"]["dropped_template_deck"] == 3


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice missing")
def test_glyphs_missing_from_every_allowed_font_are_detected(tmp_path):
    docx = pytest.importorskip("docx")
    tibetan = "".join(chr(c) for c in (0x0F40, 0x0F41, 0x0F42, 0x0F44))  # no allowed font covers Tibetan
    document = docx.Document()
    for _ in range(5):
        document.add_paragraph(ZH)
    document.add_paragraph("藏文" + tibetan)
    path = tmp_path / "rare.docx"
    document.save(path)
    store = Store(tmp_path / "store")
    digest, kind, stored = store.put(path.read_bytes())
    store.record({"key": "k", "source": "t", "status": "ok", "sha256": digest, "type": kind, "path": str(stored),
                  "terms": "official"})
    render_store(tmp_path / "store", tmp_path / "out", workers=1, log=None)
    row = json.loads((tmp_path / "out" / "clean.jsonl").read_text().splitlines()[0])
    assert row["flags"]["notdef_glyphs"] >= 1
    rare = [a for a in row["annotations"] if "藏文" in a["text"]]
    assert rare and not rare[0]["trusted"] and all(a["trusted"] for a in row["annotations"] if a["text"] == ZH)


def test_vertical_text_boxes_follow_the_drawn_glyphs(tmp_path):
    # LibreOffice writes tb-rl text glyph by glyph with negative /Widths: the text layer puts each
    # box one advance left of its glyph (onto the next column) and makes every glyph a line.
    def glyph_by_glyph(page):
        for x in (500, 470):
            for i, c in enumerate(ZH):
                page.insert_text((x, 80 + 16 * i), c, fontsize=16, fontname="china-s")

    # Vertical writing mode (Identity-V, newspapers): the text layer's cells sit one advance low.
    def writing_mode(page):
        page.insert_font(fontname="cjk", fontbuffer=pymupdf.Font("cjk").buffer)
        for x in (500, 460):
            page.insert_text((x, 60), ZH, fontsize=20, fontname="cjk")

    document = pdf_with([glyph_by_glyph, writing_mode])
    descendant = document.xref_get_key(document[0].get_fonts()[0][0], "DescendantFonts")[1]
    document.xref_set_key(int(descendant.strip("[]").split()[0]), "DW", "-1000")
    document.xref_set_key(document[1].get_fonts()[0][0], "Encoding", "/Identity-V")
    reopened = pymupdf.open("pdf", document.tobytes())
    for page, size in zip(reopened, (16, 20)):
        lines = page_lines(page)
        assert [line["text"] for line in lines] == [ZH, ZH] and all(line["vertical"] for line in lines)
        assert all(abs(line["height"] - size) < 0.5 for line in lines)  # the glyph size drives the DPI
    store_with_pdf(tmp_path, document)
    pages, rows = rendered(tmp_path, text_height="32")
    from PIL import Image
    for row in rows:
        ink = np.asarray(Image.open(row["image_path"]).convert("L")) < 128
        assert row["flags"]["hidden_lines"] == 0 and len(row["annotations"]) == 2
        for item in row["annotations"]:
            xs, ys = zip(*item["polygon"])
            column = ink[:, int(min(xs)):int(max(xs)) + 1]
            rows_with_ink = np.nonzero(column.any(axis=1))[0]
            assert item["text"] == ZH and min(ys) - 1 <= rows_with_ink.min() and rows_with_ink.max() <= max(ys) + 1
            assert column.any(axis=0).mean() > 0.75  # not on blank paper beside the glyphs


def test_annotation_and_covered_text_is_not_labelled(tmp_path):
    def annotated(page):  # pages render without annotations, so their text is not drawn
        text_page(page, 12)
        page.add_freetext_annot(pymupdf.Rect(60, 200, 400, 230), "批注文字", fontsize=14, fontname="china-s",
                                text_color=(1, 0, 0), fill_color=None).update()
        widget = pymupdf.Widget()
        widget.field_type, widget.field_name, widget.field_value = pymupdf.PDF_WIDGET_TYPE_TEXT, "name", "表单内容"
        widget.rect, widget.text_font, widget.text_fontsize = pymupdf.Rect(60, 480, 400, 505), "china-s", 14
        page.add_widget(widget)

    def photo_on_top(page):  # a picture placed over a line hides it
        text_page(page, 14)
        page.insert_text((60, 700), "这一行被图片遮住", fontsize=16, fontname="china-s")
        noise = np.random.default_rng(0).integers(0, 255, (120, 500, 3), dtype=np.uint8)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, 500, 120, noise.tobytes(), False)
        page.insert_image(pymupdf.Rect(40, 660, 560, 760), pixmap=pixmap, keep_proportion=False)

    def veil_on_top(page):  # a translucent picture over the text leaves it readable
        text_page(page, 10)
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), True)
        pixmap.set_rect(pixmap.irect, (90, 140, 200, 40))
        page.insert_image(page.rect, pixmap=pixmap)

    def sandwich(page):  # OCR text drawn normally, then the scan on top of it
        for i in range(15):
            page.insert_text((60, 100 + i * 40), ZH.replace("营商", "菅商"), fontsize=16, fontname="china-s")
        scan = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 280), False)
        scan.clear_with(225)
        page.insert_image(page.rect, pixmap=scan)

    store_with_pdf(tmp_path, pdf_with([annotated, photo_on_top, veil_on_top, sandwich]))
    pages, rows = rendered(tmp_path)
    assert [page["status"] for page in pages] == ["kept", "kept", "kept", "dropped_scanned"]
    assert [[a["text"] for a in row["annotations"]] for row in rows] == [[ZH] * 12] * 3
    assert all(a["trusted"] for row in rows for a in row["annotations"])
    assert rows[1]["flags"]["hidden_lines"] == 1


def test_hidden_characters_split_lines_and_copies_collapse(tmp_path):
    def interference(page):  # anti-copy characters inside a line: render mode 3 or white
        for i in range(12):
            y, hidden = 100 + i * 30, {"render_mode": 3} if i % 2 else {"color": (1, 1, 1)}
            page.insert_text((60, y), "关于进一步", fontsize=14, fontname="china-s")
            page.insert_text((130, y), "干扰", fontsize=14, fontname="china-s", **hidden)
            page.insert_text((158, y), "优化营商环境", fontsize=14, fontname="china-s")

    def shadowed(page):  # shadow, fake bold and overprint copies under each line
        for i in range(12):
            dx, dy = ((1.5, 1.5), (0.3, 0), (0, 0))[i % 3]
            page.insert_text((60 + dx, 80 + i * 40 + dy), ZH, fontsize=16, fontname="china-s", color=(0.6, 0.6, 0.6))
            page.insert_text((60, 80 + i * 40), ZH, fontsize=16, fontname="china-s")

    store_with_pdf(tmp_path, pdf_with([interference, shadowed]))
    pages, rows = rendered(tmp_path)
    assert [a["text"] for a in rows[0]["annotations"]] == ["关于进一步", "优化营商环境"] * 12
    for a, b in zip(rows[0]["annotations"][::2], rows[0]["annotations"][1::2]):
        assert b["polygon"][0][0] - a["polygon"][1][0] > 1.5 * a["size_px"]  # the gap of the hidden pair
    assert [a["text"] for a in rows[1]["annotations"]] == [ZH] * 12
    assert all(a["trusted"] for row in rows for a in row["annotations"])


def test_slanted_lines_target_their_glyph_height(tmp_path):
    def slanted(page):
        for i in range(8):
            origin = pymupdf.Point(60 + i * 40, 700)
            page.insert_text(origin, ZH, fontsize=14, fontname="china-s", morph=(origin, pymupdf.Matrix(30)))

    store_with_pdf(tmp_path, pdf_with([slanted]))
    pages, rows = rendered(tmp_path, text_height="32")
    assert abs(rows[0]["median_text_px"] - 32) < 2 and len(rows[0]["annotations"]) == 8
    for item in rows[0]["annotations"]:
        ul, ur, lr, ll = item["polygon"]
        assert abs(math.dist(ul, ll) - 32) < 2 and math.dist(ul, ur) > 10 * 32


def test_flat_font_metrics_and_untranscribed_glyphs(tmp_path):
    # Founder FzBookMaker fonts (State Council gazette): ascender 0.001 and descender 0 give flat
    # quads, and digits drawn from control code points make "（2021—2035年）" read "（—年）".
    font = cjk_font("FzBookMaker1DlFont1", flat_metrics=True, controls=True)
    digits = lambda text: "".join(chr(0x1F if d == "9" else 0x10 + int(d)) for d in text)

    def gazette(page):
        page.insert_font(fontname="fz", fontbuffer=font)
        for i in range(30):
            page.insert_text((60, 60 + 24 * i), ZH, fontsize=12, fontname="fz")
        year = "（" + digits("2019") + "—" + digits("2035") + "年）"
        page.insert_text((60, 790), year, fontsize=12, fontname="fz")
        page.insert_text((60, 814), "第" + digits("9") + "条", fontsize=12, fontname="fz")

    store_with_pdf(tmp_path, pdf_with([gazette]))
    pages, rows = rendered(tmp_path)
    assert rows[0]["class"] == "text" and len(rows[0]["annotations"]) == 32
    for item in rows[0]["annotations"]:
        ul, ur, lr, ll = item["polygon"]
        assert math.dist(ul, ll) >= 0.9 * item["size_px"] and abs(ul[1] - ur[1]) < 1
    assert [a["trusted"] for a in rows[0]["annotations"] if a["text"] in ("（—年）", "第条")] == [False, False]
    assert all(a["trusted"] for a in rows[0]["annotations"] if a["text"] == ZH)


def test_text_past_the_page_edge_is_cut_and_boxes_stay_inside(tmp_path):
    def slide(page):
        page.set_mediabox(pymupdf.Rect(0, 0, 842, 595))
        text_page(page, 12)
        page.insert_text((700, 500), ZH, fontsize=20, fontname="china-s")  # the 8th glyph crosses the edge

    store_with_pdf(tmp_path, pdf_with([slide]))
    pages, rows = rendered(tmp_path)
    row = rows[0]
    regions(row["annotations"], row["size"])  # every polygon lies within the image
    assert row["flags"]["offpage_lines"] == 1 and [a["text"] for a in row["annotations"]] == [ZH] * 12 + [ZH[:7]]


def test_placeholder_wording_only_drops_slides(tmp_path):
    def manual(page):  # UI instructions in a manual use the same words as slide placeholders
        page.insert_text((60, 50), "单击添加按钮，选择要安装的软件包", fontsize=12, fontname="china-s")
        text_page(page, 12)

    def slide(page):
        page.set_mediabox(pymupdf.Rect(0, 0, 842, 595))
        manual(page)

    def advert(page):
        page.insert_text((60, 50), "更多模板请访问 www.1ppt.com", fontsize=12, fontname="china-s")
        text_page(page, 12)

    digests = [store_with_pdf(tmp_path, pdf_with([build]), key=name)
               for name, build in (("manual", manual), ("slide", slide), ("advert", advert))]
    pages, rows = rendered(tmp_path)
    status = {page["document"]["sha256"]: page["status"] for page in pages}
    assert [status[digest] for digest in digests] == ["kept", "dropped_template", "dropped_template"]


def test_newspaper_frame_glyphs_are_not_text(tmp_path):
    # People's Daily draws dashed caption frames with private-use glyphs of a Founder NEU font.
    frame = cjk_font("NEU-BZ-S92", decoration="—")  # the dash glyph, reachable only from U+E000

    def newspaper(page):
        page.insert_font(fontname="neu", fontbuffer=frame)
        text_page(page, 12)
        for i in range(12):
            page.insert_text((60, 70 + i * 26.4), chr(0xE000) * 3, fontsize=12, fontname="neu")
            page.insert_text((300, 70 + i * 26.4), chr(0xE000) * 3, fontsize=12, fontname="neu")

    store_with_pdf(tmp_path, pdf_with([newspaper]))
    pages, rows = rendered(tmp_path)
    assert rows[0]["class"] == "text" and [a["text"] for a in rows[0]["annotations"]] == [ZH] * 12
