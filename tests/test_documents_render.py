"""Document conversion and rendering: font aliases, macro refusal, page classes, DPI targeting, boxes."""

import json
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


def store_with_pdf(tmp_path, document, source="test"):
    store = Store(tmp_path / "store")
    digest, kind, path = store.put(document.tobytes())
    store.record({"key": "k", "source": source, "status": "ok", "sha256": digest, "type": kind, "path": str(path),
                  "terms": "official", "title": "t"})
    return store


def test_render_targets_text_height_and_boxes_cover_ink(tmp_path):
    def rotated(page):
        text_page(page, 10)
        page.set_rotation(90)

    def placeholder(page):
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


def test_template_deck_is_dropped_whole(tmp_path):
    def slide(text):
        return lambda page: (page.insert_text((60, 80), text, fontsize=24, fontname="china-s"), text_page(page, 12))

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
