"""Government document adapters (offline): MOJ regulations, gazette issues, TRS attachments."""

import json
from types import SimpleNamespace

import pytest

from data_pipeline.documents import catalog, gov
from data_pipeline.documents.fetch import FetchError


class FakeFetcher:
    """Canned pages by URL; a value may be (final_url, body) to simulate a redirect."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, dict(headers or {})))
        page = self.pages.get(url)
        if page is None:
            raise FetchError("http_404", url)
        final, body = page if isinstance(page, tuple) else (url, page)
        return SimpleNamespace(url=final, status_code=200, headers={}), body.encode() if isinstance(body, str) else body

    def json(self, url, **kwargs):
        return json.loads(self.get(url, **kwargs)[1])

    def urls(self):
        return [url for url, _ in self.calls]


def xzfg_item(law_id, title, published, history=()):
    versions = "".join(f'<div class="incident-record" data-time="{date}">\n <a class="" target="_blank" '
                       f'href="https://xzfg.moj.gov.cn/front/law/detail?LawID={old}">{title}</a></div>'
                       for date, old in history)
    return f"""<li class="list-item">
      <div class="list-item-body"><div class="col-left"><div class="title">
        <span><a target="_blank" href="https://xzfg.moj.gov.cn/front/law/detail?LawID={law_id}">{title}</a></span>
      </div><ul class="info-list"><li class="publish-date">
            {published}公布
      </li><li class="implemented-date"> 2026-12-01施行 </li></ul></div>
      <a href="https://xzfg.moj.gov.cn/law/download?LawID={law_id}" download="doc_name">文字版</a>
      <a href="https://xzfg.moj.gov.cn/law/download?LawID={law_id}&type=pdf" download="pdf_name">图片版</a>
      </div><div class="fold-con">{versions}
      <div class="incident-record" data-time="{published}"><a class="on" target="_blank"
         href="https://xzfg.moj.gov.cn/front/law/detail?LawID={law_id}">{title}</a></div></div></li>"""


def xzfg_page(total, *items):
    return f'<input id="pageIndex" type="hidden" value="1" />\n<input id="law-total" type="hidden" value="{total}" />' \
           f'<ul class="searching-results-list">{"".join(items)}</ul>'


XZFG = "https://xzfg.moj.gov.cn/search2.html?PageIndex="


def test_moj_xzfg_walks_pages_formats_history_and_dedupes():
    pages = {XZFG + "1": xzfg_page(12, xzfg_item(1843, "中华人民共和国审计法实施条例", "2026-09-19",
                                                 [("1997-10-21", 1195), ("2010-02-11", 492)]),
                                   xzfg_item(1842, "市场监督管理所条例", "2026-08-01")),
             XZFG + "2": xzfg_page(12, xzfg_item(1842, "市场监督管理所条例", "2026-08-01"),
                                   xzfg_item(15, "中华人民共和国打捞沉船管理办法", "1957-10-11"))}
    fake = FakeFetcher(pages)
    found = list(gov.moj_xzfg(fake, {}))
    assert [c["key"] for c in found] == ["moj_xzfg:1843:docx", "moj_xzfg:1843:pdf", "moj_xzfg:1842:docx",
                                         "moj_xzfg:1842:pdf", "moj_xzfg:15:docx", "moj_xzfg:15:pdf"]
    assert found[0]["url"] == "https://xzfg.moj.gov.cn/law/download?LawID=1843"
    assert found[1]["url"].endswith("LawID=1843&type=pdf") and found[1]["type_hint"] == "pdf"
    assert found[0]["title"] == "中华人民共和国审计法实施条例" and found[0]["license"].startswith("PRC Copyright Law")
    assert found[0]["meta"]["published"] == "2026-09-19" and found[0]["meta"]["effective"] == "2026-12-01"
    assert fake.urls() == [XZFG + "1", XZFG + "2"]  # 12 regulations = 2 pages, then stop

    history = list(gov.moj_xzfg(FakeFetcher(pages), {"formats": "pdf", "history": True, "max_pages": 1}))
    assert [c["key"] for c in history] == ["moj_xzfg:1843:pdf", "moj_xzfg:1195:pdf", "moj_xzfg:492:pdf",
                                           "moj_xzfg:1842:pdf"]
    assert history[1]["meta"] == dict(history[1]["meta"], current=False, version_date="1997-10-21", current_law_id=1843)
    assert len(list(gov.moj_xzfg(FakeFetcher(pages), {"max_items": 3}))) == 3


def test_moj_xzfg_failed_pages():
    assert list(gov.moj_xzfg(FakeFetcher({}), {})) == []  # first page down: nothing known, nothing yielded
    pages = {XZFG + "1": xzfg_page(30, xzfg_item(9, "条例甲", "2001-01-01")),
             XZFG + "3": xzfg_page(30, xzfg_item(7, "条例乙", "2001-01-01"))}
    assert [c["meta"]["law_id"] for c in gov.moj_xzfg(FakeFetcher(pages), {"formats": ["docx"]})] == [9, 7]


GB = "https://www.gov.cn/gongbao/"


def gb_issue(href):
    return f'<ul class="qrcode-list"><li class="gbpdf"><a href="{href}" target="_blank"><img src="x.png" /></a></li>'


def test_gazette_issue_pdfs_years_and_empty_links():
    index = [{"values": {
        "2026年": {"第28号": {"gname": GB + "2026/issue_13026/", "issue": "第28号"},
                  "第27号": {"gname": GB + "2026/issue_13006/", "issue": "第27号"}},
        "2021年": {"第35号": {"gname": GB + "2021/issue_9486/", "issue": "第35号"}},
        "2020年": {"第8号": {"gname": GB + "2020/issue_7346/", "issue": "第8号"},
                  "第7号": {"gname": GB + "2020/issue_7326/", "issue": "第7号"},
                  "第6号": {"gname": GB + "2020/issue_7306/", "issue": "第6号"},
                  "第5号": {"gname": None, "issue": "第5号"}}}}]
    hexes = "d08e7a70beb64cf9bb1b0610b0d9c104/files/b73a63946add4dff9aff03d9ce0cef98"
    pages = {GB + "gbgl.json": json.dumps(index, ensure_ascii=False),
             GB + "2026/issue_13026/": gb_issue("material/gwygb202628.pdf"),
             GB + "2026/issue_13006/": gb_issue(f"https://www.gov.cn/gbgl/{hexes}.pdf"),
             GB + "2021/issue_9486/": gb_issue(f"https://www.gov.cn/gbgl/{hexes}.pdf"),
             GB + "2020/issue_7346/": gb_issue(""),  # gbIssue.js hides the icon: no PDF
             GB + "2020/issue_7326/": f'<p>下载</p><a href="https://www.gov.cn/gbgl/{hexes}.pdf">PDF</a>'}
    found = list(gov.gazette(FakeFetcher(pages), {"years": "2020-2026"}))
    assert [c["key"] for c in found] == ["gov_gazette:2026:28", "gov_gazette:2026:27", "gov_gazette:2021:35",
                                         "gov_gazette:2020:7"]
    assert found[0]["url"] == GB + "2026/issue_13026/material/gwygb202628.pdf"
    assert found[0]["title"] == "国务院公报2026年第28号" and found[0]["type_hint"] == "pdf"
    assert found[2]["meta"] == {"year": 2021, "issue": 35, "issue_url": GB + "2021/issue_9486/", "host": "www.gov.cn"}
    assert [c["key"] for c in gov.gazette(FakeFetcher(pages), {"years": [2021]})] == ["gov_gazette:2021:35"]
    newest = gov.gazette(FakeFetcher(pages), {"years": 2026, "max_items": 1})
    assert [c["key"] for c in newest] == ["gov_gazette:2026:28"]
    assert list(gov._years(None))[0] == 2021 and list(gov._years("2022..2023")) == [2022, 2023]


def test_listing_page_urls_for_each_pagination_style():
    mee = "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/"
    assert gov._page_urls(mee, '//createPageHTML(34, 0, "index", "shtml");\nvar countPage = 34') == \
        [mee] + [f"{mee}index_{i}.shtml" for i in range(1, 34)]
    mof = "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/"
    assert gov._page_urls(mof, 'createPageHTML(3, 0, "index", "htm")')[1:] == [mof + "index_1.htm", mof + "index_2.htm"]
    assert gov._page_urls("https://a.gov.cn/x/index.html", "var countPage = 3//共多少页")[2] == \
        "https://a.gov.cn/x/index_2.html"
    sh = "https://www.shanghai.gov.cn/nw12344/index.html"
    urls = gov._page_urls(sh, "$('.pagination').pagination({ totalPage: 357, showPageNum: 5", offset=1)
    assert len(urls) == 357 and urls[1].endswith("/index_2.html") and urls[-1].endswith("/index_357.html")
    most = "https://www.most.gov.cn/tztg/"
    assert gov._page_urls(most, "pageNumber: '0',\n total: '25',// 总页数")[-1] == most + "index_24.html"
    npc = "http://www.npc.gov.cn/npc/c2/c12435/c12491/"
    links = "<a href='index_1.html'>2</a><a href='index_1.html'>下一页</a><a href='index_36.html'>尾页</a>"
    assert gov._page_urls(npc, links)[-1] == npc + "index_36.html" and len(gov._page_urls(npc, links)) == 37
    webplus = "https://www.x.gov.cn/2045/list.htm"
    assert gov._page_urls(webplus, '<em class="all_pages">3</em>') == [webplus, webplus[:-8] + "list2.htm",
                                                                       webplus[:-8] + "list3.htm"]
    assert gov._page_urls("https://www.beijing.gov.cn/zhengce/zhengcefagui/", "<ul></ul>") == \
        ["https://www.beijing.gov.cn/zhengce/zhengcefagui/"]


SITE = "https://www.most.gov.cn/tztg/"
LISTING = """<ul>
  <li><a href="./202610/t20261016_197521.html" target="_blank" title="关于印发《国家重点研发计划管理办法》的通知">关于印发…</a></li>
  <li><a href="./202610/t20261014_197499.html">关于公布2026年度公开招聘拟录用人员名单的通知</a></li>
  <li><a href="./202610/t20261012_197473.html">科技部关于开展2026年度科技评估工作的通知</a></li>
  <li><a href="http://szs.mof.gov.cn/zt/201706/t20170620_2626928.htm">外站文件</a></li>
  <li><a href="./202609/P020260930111111111111.pdf">附件：直接挂在列表上的文件.pdf</a></li>
  <li><a href="javascript:void(0)">更多</a></li>
</ul>
<script>var countPage = 2//共多少页</script>"""
DETAIL_1 = """<meta name="ArticleTitle" content="关于印发《国家重点研发计划管理办法》的通知" />
<meta name="PubDate" content="2026-10-16 02:43" />
<a href="./P020261016642044882404.docx" OLDSRC="/protect/P0202610/P020261016/P020261016642044882404.docx">
  附件1：国家重点研发计划管理办法.docx</a>
<a href="/protect/P0202610/P020261016/P020261016642044882404.docx">镜像</a>
<a href="./P020261016642044882405.pdf">附件2：申报指南.pdf</a>
<a href="./P020261016642044882405.pdf#page=2">附件2：申报指南.pdf</a>
<a href="./P020261016642044882406.xls">附件3：预算表.xls</a>
<a href="./P020261016642044882407.pdf">附件4：专家名单.pdf</a>
<a href="./P020261016642044882408.wps">附件5：说明.wps</a>"""
DETAIL_3 = """<meta name="ArticleTitle" content="科技部关于开展2026年度科技评估工作的通知" />
<a href="./P020261016642044882405.pdf">申报指南</a>
<a href="https://www.most.gov.cn/tztg/202610/W020261012000000000001.PPTX">培训课件</a>"""


def most_pages():
    return {SITE: LISTING,
            SITE + "202610/t20261016_197521.html": DETAIL_1,
            SITE + "202610/t20261012_197473.html": ("https://www.most.gov.cn/tztg/202610/t20261012_197473.html",
                                                     DETAIL_3.encode("gb18030")),
            SITE + "index_1.html": '<a href="./202609/t20260901_197000.html">第二页的通知</a>',
            SITE + "202609/t20260901_197000.html": '<a href="./P020260901000000000001.doc">附件</a>'}


def test_trs_walk_finds_attachments_follows_pages_and_filters():
    fake = FakeFetcher(most_pages())
    found = list(gov.trs_attachments(fake, {"listing": SITE}))
    urls = [c["url"] for c in found]
    assert urls == [SITE + "202609/P020260930111111111111.pdf",
                    SITE + "202610/P020261016642044882404.docx", SITE + "202610/P020261016642044882405.pdf",
                    SITE + "202610/P020261016642044882408.wps",
                    SITE + "202610/W020261012000000000001.PPTX",
                    SITE + "202609/P020260901000000000001.doc"]
    # Out-of-scope pages, personal-data lists, /protect/ mirrors, spreadsheets and repeats are skipped.
    assert SITE + "202610/t20261014_197499.html" not in fake.urls()
    assert not any("mof.gov.cn" in url for url in fake.urls())
    docx = found[1]
    assert docx["key"] == "www.most.gov.cn/tztg/202610/P020261016642044882404.docx"
    assert docx["title"] == "附件1：国家重点研发计划管理办法.docx" and docx["type_hint"] == "docx"
    assert docx["headers"] == {"Referer": SITE + "202610/t20261016_197521.html"}
    assert docx["meta"]["page_title"] == "关于印发《国家重点研发计划管理办法》的通知"
    assert docx["meta"]["published"] == "2026-10-16 02:43" and docx["meta"]["host"] == "www.most.gov.cn"
    assert found[3]["type_hint"] == "doc" and found[4]["type_hint"] == "pptx"
    assert found[4]["meta"]["page_title"] == "科技部关于开展2026年度科技评估工作的通知"  # GBK page decoded
    assert found[5]["meta"]["published"] == "2026-09-01" and found[5]["title"] == "第二页的通知"
    assert found[0]["meta"]["published"] == "2026-09-30"  # from the P0YYYYMMDD file name
    # The listing link to the first detail page carried its full title in the title attribute.
    assert (SITE + "202610/t20261016_197521.html", {"Referer": SITE}) in fake.calls

    assert [c["type_hint"] for c in gov.trs_attachments(FakeFetcher(most_pages()),
                                                        {"listing": SITE, "extensions": "pptx,ppt"})] == ["pptx"]
    assert len(list(gov.trs_attachments(FakeFetcher(most_pages()), {"listing": SITE, "max_items": 2}))) == 2
    flat = FakeFetcher(most_pages())
    assert [c["url"] for c in gov.trs_attachments(flat, {"listing": [SITE], "details": False, "max_pages": 1})] == \
        [SITE + "202609/P020260930111111111111.pdf"]
    assert flat.urls() == [SITE]


def test_trs_meta_refresh_failed_pages_and_presets():
    stub = '<meta http-equiv="refresh" content=" 0;url=./shjzlbz/ " />'
    leaf = "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/shjzlbz/"
    pages = {"https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/": stub,
             leaf: '//createPageHTML(9, 0, "index", "shtml");\nvar countPage = 9\n'
                   '<a href="./202001/t20200101_1.shtml" target="_blank">地表水环境质量标准 GB 3838-2002</a>',
             leaf + "202001/t20200101_1.shtml": '<a href="./W020200101000000000001.pdf">GB 3838-2002</a>'}
    fake = FakeFetcher(pages)
    found = list(gov.trs_attachments(fake, {"preset": "mee_standards",
                                            "listing": ["https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/"]}))
    assert [c["url"] for c in found] == [leaf + "202001/W020200101000000000001.pdf"]
    assert found[0]["title"] == "地表水环境质量标准 GB 3838-2002"  # link text without CJK falls back to the page
    assert found[0]["license"].startswith("standard text") and found[0]["meta"]["preset"] == "mee_standards"
    assert fake.urls().count(leaf + "index_1.shtml") == 1 and leaf + "index_3.shtml" not in fake.urls()

    with pytest.raises(ValueError):
        next(gov.trs_attachments(FakeFetcher({}), {}))
    for name, preset in gov.PRESETS.items():
        assert preset["listing"] and all(url.startswith(("http://", "https://")) for url in preset["listing"]), name
        assert not any(catalog.blocked(url) for url in preset["listing"]), name
    assert len(set(gov.PRESETS["mee_standards"]["listing"])) == len(gov.MEE_LEAVES)


def test_catalog_runs_gov_sources_with_official_terms():
    pages = {XZFG + "1": xzfg_page(1, xzfg_item(1843, "中华人民共和国审计法实施条例", "2026-09-19"))}
    found = list(catalog.discover("moj_xzfg", FakeFetcher(pages)))
    assert [(c["source"], c["terms"]) for c in found] == [("moj_xzfg", "official")] * 2
    shanghai = {"https://www.shanghai.gov.cn/nw12344/index.html":
                '<a href="/nw12344/20260907/ceeb20cafc704f44892a90ec5d0d9a8b.html" target="_blank">市政府文件</a>'
                '<a href="/nw12344/20260908/notadetail.html">新闻</a>',
                "https://www.shanghai.gov.cn/nw12344/20260907/ceeb20cafc704f44892a90ec5d0d9a8b.html":
                '<meta name="ArticleTitle" content="上海市人民政府办公厅关于印发《三年行动计划》的通知">'
                '<a class="Gover-file0301" href="/cmsres/18/1884/6eee.pdf" title="hfbf2624+A.pdf"> </a>'
                '<a href="https://www.doc88.com/p-1.pdf">转载</a>'}
    found = list(catalog.discover("gov_attachments", FakeFetcher(shanghai), {"preset": "shanghai", "max_pages": 1}))
    assert [c["url"] for c in found] == ["https://www.shanghai.gov.cn/cmsres/18/1884/6eee.pdf"]
    assert found[0]["title"] == "上海市人民政府办公厅关于印发《三年行动计划》的通知"
    assert found[0]["source"] == "gov_attachments" and found[0]["terms"] == "official"


def test_trs_options_cannot_label_non_government_or_mee_standards_files_official():
    leaf = gov.MEE + "shjbh/swrwpfbz/"
    pages = {leaf: '<a href="./W020190101.pdf">HJ 1-2019 标准文本</a>'}
    for listing in (leaf, "http://mee.gov.cn/ywgz/fgbz/bz/bzwb/trhj/", "https://www.mee.gov.cn/ywgz/fgbz/bz/x/../bzwb/",
                    "https://www.mee.gov.cn//YWGZ/fgbz/bz/%62zwb/trhj/", "https://www.ypppt.com/moban/",
                    "https://evilgov.cn/", "https://www.gov.cn.example.com/", "https://evil.example\\@www.gov.cn/",
                    "https://www.gov.cn@evil.example/", "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb;jsessionid=1/trhj/",
                    leaf.replace("www.", "www" + chr(0x3002)), leaf.replace("mee", chr(0xFF4D) + "ee")):
        fake = FakeFetcher(pages)
        with pytest.raises(PermissionError):
            next(catalog.discover("gov_attachments", fake, {"listing": listing, "details": False}))
        with pytest.raises(PermissionError):
            next(catalog.discover("gov_attachments", fake, {"preset": "most", "listing": [SITE, listing]}))
        assert fake.calls == []
    for name, preset in gov.PRESETS.items():
        assert all(gov._official(url, name) for url in preset["listing"]), name
    for preset in (["mee_standards"], "mee"):
        fake = FakeFetcher(pages)
        with pytest.raises(ValueError, match="unknown preset"):
            next(catalog.discover("gov_attachments", fake, {"preset": preset, "listing": leaf}))
        assert fake.calls == []
    # gov_standards (terms public_web) still walks the preset and MEE listings of its own.
    for options in ({"details": False}, {"listing": leaf, "details": False}):
        found = list(catalog.discover("gov_standards", FakeFetcher(pages), options, accepted={"public_web"}))
        assert [(c["url"], c["terms"]) for c in found] == [(leaf + "W020190101.pdf", "public_web")]
    # A government listing cannot lead the walk to other hosts or into the MEE standards tree.
    mee = "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/trhj/202001/"
    pages = {SITE: '<a href="https://www.ypppt.com/moban/t20260101_1.html">模板下载页面</a>'
                   '<a href="https://down.ypppt.com/x/1.pptx">模板下载文件</a>'
                   f'<a href="{mee}W020200101000000000001.pdf">土壤环境质量标准文本</a>'
                   f'<a href="{mee}t20200101_1.shtml">土壤环境质量标准</a>'
                   '<a href="./202610/P020261016000000000001.pdf">科技部通知附件</a>',
             "https://www.ypppt.com/moban/t20260101_1.html": '<a href="https://down.ypppt.com/x/2.pptx">更多模板下载</a>',
             mee + "t20200101_1.shtml": '<a href="./W020200101000000000002.pdf">土壤环境质量标准文本</a>'}
    fake = FakeFetcher(pages)
    found = list(catalog.discover("gov_attachments", fake, {"listing": SITE, "scope": ".*"}))
    assert [c["url"] for c in found] == [SITE + "202610/P020261016000000000001.pdf"] and fake.urls() == [SITE]
    messages = []
    moved = FakeFetcher({SITE: ("https://www.ypppt.com/moban/", '<a href="./1.pdf">模板下载文件</a>')})
    assert list(gov.trs_attachments(moved, {"listing": SITE, "log": messages.append})) == []
    assert any("not a government listing" in m for m in messages)


class FakeHTTP:
    """Transport for the real Fetcher: canned (status, body) by URL, 404 otherwise."""

    def __init__(self, routes):
        self.routes, self.headers = routes, {}

    def request(self, method, url, **kwargs):
        status, body = self.routes.get(url, (404, b""))
        body = body.encode() if isinstance(body, str) else body
        return SimpleNamespace(status_code=status, headers={}, url=url, is_redirect=False, text=body.decode(),
                               iter_content=lambda size: iter([body]), close=lambda: None)


def test_real_fetcher_robots_refusal_is_logged_and_skipped():
    from data_pipeline.documents.fetch import Fetcher
    routes = {"https://zrzy.jl.gov.cn/robots.txt": (200, "User-agent: *\nDisallow: /zwgk/fgwj/zcjd/202406/\n"),
              "https://zrzy.jl.gov.cn/zwgk/fgwj/zcjd/": (200, 'var countPage = 1\n'
                                                            '<a href="./202406/t20240613_8920591.html">PPT解读</a>'
                                                            '<a href="./202405/t20240515_8900000.html">政策解读</a>'),
              "https://zrzy.jl.gov.cn/zwgk/fgwj/zcjd/202405/t20240515_8900000.html":
                  (200, '<a href="./P020241107578361694167.pptx">吉林省矿产资源管理改革实施意见解读.pptx</a>')}
    messages = []
    fetcher = Fetcher(http=FakeHTTP(routes), interval=0, sleep=lambda s: None)
    found = list(gov.trs_attachments(fetcher, {"preset": "jilin_zcjd", "log": messages.append}))
    assert [c["url"] for c in found] == ["https://zrzy.jl.gov.cn/zwgk/fgwj/zcjd/202405/P020241107578361694167.pptx"]
    assert found[0]["type_hint"] == "pptx" and found[0]["meta"]["published"] == "2024-05-15"  # the page date wins
    assert any("robots_disallow" in m and "t20240613_8920591" in m for m in messages)
