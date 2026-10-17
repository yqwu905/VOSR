"""Procurement notice adapter (offline): ccgp-beijing listings, search windows and notice attachments."""

import copy
import json
from types import SimpleNamespace

import pytest

from data_pipeline.documents import procurement
from data_pipeline.documents.fetch import FetchError

HOST = "http://www.ccgp-beijing.gov.cn"


class FakeFetcher:
    """Canned pages by URL; POSTs to the search API are answered by ``search(form)``."""

    def __init__(self, pages, search=None):
        self.pages, self.search, self.calls, self.forms = pages, search, [], []

    def get(self, url, headers=None, method="GET", **kwargs):
        self.calls.append((url, dict(headers or {})))
        if method == "POST":
            self.forms.append(copy.deepcopy(kwargs["json"]))
            answer = json.dumps(self.search(kwargs["json"]), ensure_ascii=False)
            return SimpleNamespace(url=url, status_code=200, headers={}), answer.encode()
        page = self.pages.get(url)
        if page is None:
            raise FetchError("http_404", url)
        return SimpleNamespace(url=url, status_code=200, headers={}), page.encode()

    def json(self, url, method="GET", **kwargs):
        return json.loads(self.get(url, method=method, **kwargs)[1])

    def urls(self):
        return [url for url, _ in self.calls]


def listing(total, *rows):
    items = "".join(f'<li><a href="//www.ccgp-beijing.gov.cn{href}" target="_parent">{title}</a>'
                    f'<span class="datetime" style="float: right">{day}</span></li>' for href, title, day in rows)
    return f'<ul class="inner-ul">\n<script>function gotoPage(selObj){{}}</script>\n{items}\n</ul>' \
           f'<li><a href=\'//www.ccgp-beijing.gov.cn/xxgg/sjxxgg/zbgg/A002004001001index_2.htm\'>下一页</a></li>' \
           f'<input id="TotalPage" type="hidden" value="{total}"/>'


def notice(title, day, *links):
    anchors = "".join(f'<p><a href="{href}" target="_blank" style="color:blue;cursor:pointer;">{name}</a></p>'
                      for href, name in links)
    return f"""<div class="xl-box-t"><h1><p>{title}</p></h1>
      <span>{day}</span><dl class="fr" id="fontSize"><a href='javascript:changeSize(18)'>大</a></dl></div>
      <div id="mainText"><p>联系方式：010-00000000</p>{anchors}</div>
      <a href="//www.ccgp-beijing.gov.cn/xxgg/sjxxgg/zbgg/A002004001001index_1.htm">招标公告</a>"""


ZB = "/xxgg/sjxxgg/zbgg/2026/10/"
CF = "//www.ccgp-beijing.gov.cn/content_file/xxgg/sjxxgg/zbgg/2026/10/"
A, B, C = "2c52860be20646c391f8f63c79011986", "2e743c74382940e7958b8a4e975273cc", "04b5870d3db9421a8057ecc14e31b716"
QJ = "cca229f91075479bb1a760246ed7c864"


def site():
    return {
        procurement.listing_url("city", "zbgg", 1): listing(
            2, (ZB + A + ".htm", "[公开]北京天坛医院大型医疗设备维保项目...", "2026-10-16"),
            (ZB + B + ".htm", "[公开]科研设备购置项目公开招标公告", "2026-10-15")),
        procurement.listing_url("city", "zbgg", 2): listing(
            2, (ZB + C + ".htm", "[公开]儿童福利院物业管理服务招标公告", "2026-10-01")),
        procurement.listing_url("district", "zbgg", 1): listing(
            1, (f"/xxgg/qjxxgg/qjzbgg/2026/10/{QJ}.htm", "[通州区]锅炉房托管运营服务公开招标公告", "2026-10-16")),
        HOST + ZB + A + ".htm": notice(
            "[公开]北京天坛医院大型医疗设备维保项目（05、08包重招）公开招标公告", "2026-10-16",
            (CF + "68463eb1423b48c6bff52041a2020ebe.docx", "招标公告（5、8包重招）-北京天坛医院大型医疗设备维保项目.docx"),
            (CF + "ed7bde476ac246e39dc11515aef90e75.PDF", "采购需求-05、08包.pdf"),
            (CF + "ed7bde476ac246e39dc11515aef90e75.PDF#page=2", "采购需求-05、08包.pdf"),
            (CF + "1b0d7d2f0c3b4f7e9e3c6b1d5a7f9e01.pdf", "中小企业声明函-某公司.pdf"),
            (CF + "2b0d7d2f0c3b4f7e9e3c6b1d5a7f9e02.pdf", "单一来源论证意见及证件.pdf"),
            (CF + "3b0d7d2f0c3b4f7e9e3c6b1d5a7f9e03.xlsx", "报名表.xlsx")),
        HOST + ZB + B + ".htm": notice("[公开]科研设备购置项目公开招标公告", "2026-10-15"),
        HOST + ZB + C + ".htm": notice("[公开]北京市儿童福利院物业管理服务招标公告", "2026-10-01",
                                       ("/wwprotect/P0202610/P020261001/P020261001666041173264.doc", "招标文件.doc"),
                                       ("/files/attachment.wps", "附件")),
        HOST + f"/xxgg/qjxxgg/qjzbgg/2026/10/{QJ}.htm": notice(
            "[通州区]马驹桥镇锅炉房托管运营服务公开招标公告", "2026-10-16",
            (CF + "68463eb1423b48c6bff52041a2020ebe.docx", "招标公告.docx"),  # same file as on another notice
            ("//www.ccgp-beijing.gov.cn/content_file/xxgg/qjxxgg/qjzbgg/2026/10/9b47720456fc412bad580076c3e2dd64.docx",
             "采购需求.docx")),
    }


def test_listing_urls_for_levels_and_types():
    assert procurement.listing_url("city", "zbgg", 1) == HOST + "/xxgg/sjxxgg/zbgg/A002004001001index_1.htm"
    assert procurement.listing_url("district", "gzgg", 7) == HOST + "/xxgg/qjxxgg/qjgzgg/A002004002004index_7.htm"
    assert procurement.listing_url("city", "qtgg", 143) == HOST + "/xxgg/sjxxgg/qtgg/A002004001007index_143.htm"
    assert procurement._kind("招标公告") == "zbgg" and procurement._level("区级") == "district"
    with pytest.raises(ValueError):
        procurement._kind("xxgg")


def test_listing_walk_yields_attachments_with_keys_titles_and_filters():
    fake = FakeFetcher(site())
    found = list(procurement.ccgp_beijing(fake, {"types": "zbgg"}))
    assert [c["key"] for c in found] == ["ccgp_bj:68463eb1423b48c6bff52041a2020ebe",
                                         "ccgp_bj:ed7bde476ac246e39dc11515aef90e75",
                                         "ccgp_bj:P020261001666041173264", f"ccgp_bj:{C}:1",
                                         "ccgp_bj:9b47720456fc412bad580076c3e2dd64"]
    first, pdf, doc, wps = found[:4]
    assert first["url"] == "http:" + CF + "68463eb1423b48c6bff52041a2020ebe.docx" and first["type_hint"] == "docx"
    # The full title comes from the notice page (listings truncate it).
    assert first["title"] == ("[公开]北京天坛医院大型医疗设备维保项目（05、08包重招）公开招标公告 / "
                              "招标公告（5、8包重招）-北京天坛医院大型医疗设备维保项目.docx")
    assert first["headers"] == {"Referer": HOST + ZB + A + ".htm"}
    assert first["meta"] == dict(type="zbgg", type_name="招标公告", level="city", published="2026-10-16",
                                 notice=HOST + ZB + A + ".htm",
                                 notice_title="[公开]北京天坛医院大型医疗设备维保项目（05、08包重招）公开招标公告",
                                 attachment="招标公告（5、8包重招）-北京天坛医院大型医疗设备维保项目.docx",
                                 host="www.ccgp-beijing.gov.cn")
    assert pdf["type_hint"] == "pdf"
    assert doc["url"] == HOST + "/wwprotect/P0202610/P020261001/P020261001666041173264.doc"
    assert wps["type_hint"] == "doc" and wps["title"] == "[公开]北京市儿童福利院物业管理服务招标公告 / 附件"
    assert found[4]["meta"]["level"] == "district" and found[4]["meta"]["published"] == "2026-10-16"
    # City pages 1-2 (TotalPage=2), then the district listing; declarations, expert opinions, xlsx and repeats skipped.
    assert [url for url in fake.urls() if "index_" in url] == [procurement.listing_url("city", "zbgg", 1),
                                                              procurement.listing_url("city", "zbgg", 2),
                                                              procurement.listing_url("district", "zbgg", 1)]


def test_listing_since_pages_levels_and_max_items():
    fake = FakeFetcher(site())
    recent = list(procurement.ccgp_beijing(fake, {"types": ["招标公告"], "levels": "市级", "since": "2026-10-10"}))
    assert [c["meta"]["notice"].rsplit("/", 1)[1] for c in recent] == [A + ".htm", A + ".htm"]
    # Page 2 is read to find that it is older; its notice is not opened and the district is not listed.
    assert HOST + ZB + B + ".htm" in fake.urls() and HOST + ZB + C + ".htm" not in fake.urls()
    assert not any("qjxxgg" in url for url in fake.urls())
    fake = FakeFetcher(site())
    assert len(list(procurement.ccgp_beijing(fake, {"types": "zbgg", "levels": "city", "pages": 1}))) == 2
    assert procurement.listing_url("city", "zbgg", 2) not in fake.urls()
    assert len(list(procurement.ccgp_beijing(FakeFetcher(site()), {"types": "zbgg", "max_items": 1}))) == 1
    # Default types skip 合同公告 and 废标公告; missing listings are simply empty.
    fake = FakeFetcher(site())
    list(procurement.ccgp_beijing(fake, {"levels": "city"}))
    listed = [url for url in fake.urls() if "index_" in url]
    assert not any("/htgg/" in url or "/fbgg/" in url for url in listed)
    assert {url.split("/")[5] for url in listed} == {"zbgg", "zbggs", "gzgg", "dygg", "qtgg"}


def test_notice_with_personal_title_and_failed_pages_are_skipped():
    pages = site()
    pages[HOST + ZB + B + ".htm"] = notice("[公开]专家名单公示", "2026-10-15", (CF + "aa.pdf", "附件.pdf"))
    del pages[HOST + ZB + A + ".htm"]
    keys = [c["key"] for c in procurement.ccgp_beijing(FakeFetcher(pages), {"types": "zbgg", "levels": "city"})]
    assert keys == ["ccgp_bj:P020261001666041173264", f"ccgp_bj:{C}:1"]


def search_row(channel, path, title, published):
    return dict(channel=channel, path=path, title=f'{title}<span style="color: red">公告</span>', publishedAt=published,
                text="项目联系人：某某")


def test_search_windows_filter_channels_and_query_history():
    web = "//www.ccgp-beijing.gov.cn/xxgg/"
    rows = {("A002", "2017-09-10"): [
        search_row("A002004001001", f"//www.ccgp-beijing.gov.cn{ZB}{A}.htm", "招标", "2017-09-12T10:00:00"),
        search_row("A002004001003", web + "sjxxgg/htgg/2024/12/x.htm", "合同", "2017-09-12T09:00:00"),
        search_row("A002004002001", web + f"qjxxgg/qjzbgg/2026/10/{QJ}.htm", "区级招标", "2017-09-11T09:00:00")],
        ("A003", "2017-09-10"): [search_row("A003003", f"{ZB}{C}.htm", "历史招标", "2017-09-10T08:00:00")]}

    def search(form):
        found = rows.get((form["site"], form["range"]["after"][:10]), [])
        if form["site"] == "A003":
            assert form["indexes"] == [procurement.HISTORY["zbgg"]]
        batch = found[(form["page"] - 1) * 2:form["page"] * 2]
        assert form["size"] == 100 and form["keywords"]
        return {"code": 0, "data": {"total": len(found) if len(found) <= 2 else 150, "results": batch}}

    fake = FakeFetcher(site(), search)
    found = list(procurement.ccgp_beijing(fake, {"search": True, "types": "zbgg", "levels": "city", "history": True,
                                                 "since": "2017-09-10", "until": "2017-09-20", "window_days": 7}))
    windows = [(f["site"], f["range"]["after"][:10], f["range"]["before"][:10], f["page"]) for f in fake.forms]
    # Newest window first; A003 only for windows starting before 2017-09-16; 3 rows of total 150 = 2 pages.
    assert windows == [("A002", "2017-09-14", "2017-09-21", 1), ("A003", "2017-09-14", "2017-09-21", 1),
                       ("A002", "2017-09-10", "2017-09-14", 1), ("A002", "2017-09-10", "2017-09-14", 2),
                       ("A003", "2017-09-10", "2017-09-14", 1)]
    # Only city 招标公告 rows of A002 (contract and district rows dropped) plus the A003 row.
    assert [c["meta"]["level"] for c in found] == ["city", "city", "history", "history"]
    assert found[0]["meta"]["notice"] == HOST + ZB + A + ".htm"
    assert [headers for url, headers in fake.calls if url == procurement.API] == [procurement.SEARCH_PAGE] * 5
    with pytest.raises(ValueError):
        list(procurement.ccgp_beijing(FakeFetcher({}), {"search": True}))


def test_search_errors_end_the_window():
    calls = []

    def search(form):
        calls.append(form["range"]["after"][:10])
        return {"code": -1, "data": None, "msg": "参数 [keywords] 的值 [] 不正确"}

    messages = []
    found = list(procurement.ccgp_beijing(FakeFetcher({}, search), {"search": True, "since": "2026-10-01",
                                                                     "until": "2026-10-14", "log": messages.append}))
    assert found == [] and calls == ["2026-10-08", "2026-10-01"] and "不正确" in messages[0]
