"""Newspaper e-paper adapters (offline): 人民日报 page PDFs in both layouts, 经济日报."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from data_pipeline.documents import news
from data_pipeline.documents.fetch import FetchError

RMRB = "http://paper.people.com.cn/rmrb/"


class FakeFetcher:
    """Canned pages by URL; a value may be a FetchError reason to raise."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, headers=None, **kwargs):
        self.calls.append(url)
        page = self.pages.get(url, "http_404")
        if page.startswith("http_"):
            raise FetchError(page, url)
        return SimpleNamespace(url=url, status_code=200, headers={}), page.encode("utf-8")


def node_page(number, sections, pdf):
    """A 2024-12- pc/layout node page: the own PDF and every page of the issue."""
    links = "".join(f'<div class="swiper-slide">\n<a id="pageLink" href="node_{n:02d}.html">{n:02d}版：{s}</a>\n</div>'
                    for n, s in enumerate(sections, 1))
    return (f'<a href="http://paper.people.com.cn/rmrb/paperindex.htm" target="_blank">人民日报</a>\n'
            f'<div class="paper-bot"><p class="left ban">\n第{number:02d}版：{sections[number - 1]}\n</p>\n'
            f'<p class="right btn">\n<a href="../../../attachement/{pdf}" download="rmrb.pdf">PDF下载</a></p></div>\n'
            f'<div class="swiper-container">{links}</div>'
            f'<a href="../../../content/202610/17/content_30186187.html"> 统筹高质量发展 </a>')


def old_page(day, sections, pdf=None):
    """A 2023-2024 html/ layout page (BOM, unquoted attributes); the PDF name is predictable."""
    ym, d, ymd = day[:7], day[8:], day.replace("-", "")
    pdf = pdf or f"../../../images/{ym}/{d}/01/rmrb{ymd}01.pdf"
    links = "".join(f"<a id=pageLink href={'./' if n == 1 else ''}nbs.D110000renmrb_{n:02d}.htm>{n:02d}版：{s}</a>\n"
                    for n, s in enumerate(sections, 1))
    return (chr(0xFEFF) + f'<p class="left ban">第01版:{sections[0]}</p>\n'
            f'<p class="right btn"> <a href={pdf}>PDF下载</a></p>\n{links}')


def test_rmrb_new_layout_reads_each_node_page():
    base = RMRB + "pc/layout/202610/17/"
    sections = ["要闻", "科技·新知", "假日生活"]
    fake = FakeFetcher({base + "node_01.html": node_page(1, sections, "202610/17/71e0.pdf"),
                        base + "node_02.html": node_page(2, sections, "202610/17/03b3.pdf")})
    log = []
    found = list(news.rmrb(fake, {"start": "2026-10-17", "days": 1, "log": log.append}))
    assert [c["key"] for c in found] == ["rmrb:20261017:01", "rmrb:20261017:02"]
    assert found[0]["url"] == RMRB + "pc/attachement/202610/17/71e0.pdf"
    assert found[1]["url"] == RMRB + "pc/attachement/202610/17/03b3.pdf"
    assert found[1]["title"] == "人民日报 2026-10-17 第02版：科技·新知" and found[1]["type_hint"] == "pdf"
    assert found[1]["headers"] == {"Referer": base + "node_02.html"}
    assert found[1]["meta"] == {"paper": "人民日报", "date": "2026-10-17", "page": "02", "section": "科技·新知",
                                "page_url": base + "node_02.html", "issue_url": base + "node_01.html",
                                "host": "paper.people.com.cn"}
    assert found[0]["license"].startswith("人民日报")
    # node_01 is read once; node_03 is missing and only logged.
    assert fake.calls == [base + "node_01.html", base + "node_02.html", base + "node_03.html"]
    assert log == [f"skip {base}node_03.html: http_404"]
    # A refused node page ends the walk: the site is turning us away.
    fake = FakeFetcher({base + "node_01.html": node_page(1, sections, "a.pdf"), base + "node_02.html": "http_403"})
    found = list(news.rmrb(fake, {"start": "2026-10-17", "days": 2}))
    assert [c["key"] for c in found] == ["rmrb:20261017:01"] and len(fake.calls) == 2


def test_rmrb_old_layout_predicts_pdf_names_from_one_request():
    page = RMRB + "html/2024-06/03/nbs.D110000renmrb_01.htm"
    fake = FakeFetcher({page: old_page("2024-06-03", ["要闻", "要闻", "学术"])})
    found = list(news.rmrb(fake, {"start": "2024-06-03", "end": "2024-06-03"}))
    assert fake.calls == [page]
    assert [c["url"] for c in found] == [RMRB + f"images/2024-06/03/{n:02d}/rmrb202406030{n}.pdf" for n in (1, 2, 3)]
    assert found[2]["title"] == "人民日报 2024-06-03 第03版：学术"
    assert found[2]["meta"]["page_url"] == RMRB + "html/2024-06/03/nbs.D110000renmrb_03.htm"

    # A PDF link that does not follow the template: every node page is read instead.
    other = old_page("2024-06-03", ["要闻", "学术"], pdf="../../../pdf/a.pdf")
    second = RMRB + "html/2024-06/03/nbs.D110000renmrb_02.htm"
    log = []
    fake = FakeFetcher({page: other, second: old_page("2024-06-03", ["要闻", "学术"], pdf="../../../pdf/b.pdf")})
    found = list(news.rmrb(fake, {"start": "2024-06-03", "days": 1, "log": log.append}))
    assert [c["url"] for c in found] == [RMRB + "pdf/a.pdf", RMRB + "pdf/b.pdf"]
    assert fake.calls == [page, second] and "does not follow the template" in log[0]


def test_walk_switches_layouts_and_stops_at_refusal_or_missing_run():
    new = RMRB + "pc/layout/{}/node_01.html"
    old = RMRB + "html/{}/nbs.D110000renmrb_01.htm"
    pages = {new.format("202412/01"): node_page(1, ["要闻"], "202412/01/x.pdf"),
             old.format("2024-11/30"): old_page("2024-11-30", ["要闻"]),
             old.format("2024-11/29"): "http_403"}
    fake, log = FakeFetcher(pages), []
    found = list(news.rmrb(fake, {"start": "2024-12-02", "log": log.append}))
    assert [c["key"] for c in found] == ["rmrb:20241201:01", "rmrb:20241130:01"]
    # 2024-12-02 is missing (not published yet); the layout follows the date; 403 ends the walk.
    assert fake.calls == [new.format("202412/02"), new.format("202412/01"), old.format("2024-11/30"),
                          old.format("2024-11/29")]
    assert log[0] == "no issue 2024-12-02 (http_404)" and "access refused; stopping at 2024-11-29" in log[-1]

    fake, log = FakeFetcher({}), []
    assert list(news.rmrb(fake, {"start": "2026-10-17", "max_missing": 3, "log": log.append})) == []
    assert len(fake.calls) == 3 and log[-1] == "3 dates without an issue; stopping at 2026-10-15"
    # A single missing date does not end the walk; max_items stops early.
    fake = FakeFetcher({new.format("202412/01"): node_page(1, ["要闻"], "a.pdf"),
                        old.format("2024-11/29"): old_page("2024-11-29", ["要闻", "要闻"])})
    found = list(news.rmrb(fake, {"start": "2024-12-01", "max_missing": 2, "max_items": 2}))
    assert [c["key"] for c in found] == ["rmrb:20241201:01", "rmrb:20241129:01"] and len(fake.calls) == 3


JJRB = "http://paper.ce.cn/pc/"


def jjrb_page(day, sections):
    ym, d = day[:7].replace("-", ""), day[8:]
    items = "".join(f'<li class="posRelative">\n<a href="node_{n:02d}.html">第{n:02d}版：{s}</a>\n'
                    f'<input type="hidden" value="../../../attachment/{ym}/{d}/{n}{n}.pdf">\n</li>'
                    for n, s in enumerate(sections, 1))
    return (f'<span id="layout">第01版：<span class="font-weight">{sections[0]}</span></span>\n'
            f'<a class="nextArt" href="node_02.html">下一版</a>\n'
            f'<a href="../../../attachment/{ym}/{d}/11.pdf" download><img src="09.png"></a>\n'
            f'<p id="pdfUrl" style="display:none">../../../attachment/{ym}/{d}/11.pdf</p>\n{items}')


def test_jjrb_lists_every_pdf_on_the_first_page():
    pages = {JJRB + "layout/202610/17/node_01.html": jjrb_page("2026-10-17", ["要闻", "要闻", "综合"]),
             JJRB + "layout/202001/01/node_01.html": " ",  # holidays answer an empty page
             JJRB + "layout/201912/31/node_01.html": jjrb_page("2019-12-31", ["要闻"])}
    fake = FakeFetcher(pages)
    found = list(news.jjrb(fake, {"start": "2026-10-17", "days": 1}))
    assert fake.calls == [JJRB + "layout/202610/17/node_01.html"]
    assert [c["url"] for c in found] == [JJRB + f"attachment/202610/17/{n}{n}.pdf" for n in (1, 2, 3)]
    assert found[2]["key"] == "jjrb:20261017:03" and found[2]["title"] == "经济日报 2026-10-17 第03版：综合"
    assert found[2]["meta"]["page_url"] == JJRB + "layout/202610/17/node_03.html"
    assert found[0]["license"].startswith("经济日报")
    found = list(news.epaper(FakeFetcher(pages), {"site": "jjrb", "start": "2020-01-01", "end": "2019-12-31"}))
    assert [c["key"] for c in found] == ["jjrb:20191231:01"]


def test_dates_and_page_parsing():
    assert list(news._dates({"start": "2024-03-01", "end": "2024-02-28"})) == \
        [date(2024, 3, 1), date(2024, 2, 29), date(2024, 2, 28)]
    assert list(news._dates({"start": "2024-02-28", "end": "2024-03-01"}))[0] == date(2024, 3, 1)
    assert list(news._dates({"start": "2024-03-01", "days": 3, "step": 10})) == \
        [date(2024, 3, 1), date(2024, 2, 20), date(2024, 2, 10)]
    beijing = timezone(timedelta(hours=8))
    before = datetime.now(beijing).date()
    first = next(news._dates({}))
    assert first in (before, datetime.now(beijing).date())
    text = '<a href="node_01.html">第01版：要闻</a><a href="node_01.html">01版：要闻</a><a href="node_A2.html">A2版: 广告</a>'
    assert news.pages(text, JJRB + "layout/202610/17/node_01.html") == [
        ("01", "要闻", JJRB + "layout/202610/17/node_01.html", None),
        ("A2", "广告", JJRB + "layout/202610/17/node_A2.html", None)]
    assert news.own_pdf('<a href=../../x/y.PDF>PDF</a>', JJRB + "layout/a/b.html") == JJRB + "x/y.PDF"
    assert news.own_pdf('<a href="y.pdf.html">no</a>', JJRB) is None
