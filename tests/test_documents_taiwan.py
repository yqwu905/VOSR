"""Executive Yuan Gazette adapter (offline): Open Data archives, personal-data filter, attachments."""

import io
import zipfile
from types import SimpleNamespace

from data_pipeline.documents import taiwan
from data_pipeline.documents.fetch import FetchError

SITE = "https://gazette.nat.gov.tw/egFront/"
PUB = "https://gazette.nat.gov.tw/EG_FileManager/eguploadpub/"


class FakeFetcher:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, headers=None, **kwargs):
        self.calls.append(url)
        page = self.pages.get(url)
        if page is None:
            raise FetchError("http_404", url)
        return SimpleNamespace(url=url, status_code=200, headers={}), page.encode() if isinstance(page, str) else page


def record(metaid, title, path, style="法規", substyle="中央法規標準法第3條之7種命令", chapter="內政篇",
           agency="內政部", attachments="", scheme="https"):
    return f"""<Record>
<MetaId><![CDATA[{metaid}]]></MetaId>
<Doc_Style_LName><![CDATA[{style}]]></Doc_Style_LName>
<Doc_Style_SName><![CDATA[{substyle}]]></Doc_Style_SName>
<PreviewStageURL/>
<Chapter><![CDATA[{chapter}]]></Chapter>
<PubGov><![CDATA[{agency}]]></PubGov>
<PubGovName><![CDATA[{agency}]]></PubGovName>
<Officer_name><![CDATA[部長 某某]]></Officer_name>
<Title><![CDATA[{title}]]></Title>
<GazetteHTML><![CDATA[{scheme}://gazette.nat.gov.tw/EG_FileManager/eguploadpub/{path}/Eg.htm]]></GazetteHTML>
<HTMLContent><![CDATA[<div class="content"><p>主旨</p>
<!-- ATTACH FILE LIST START -->{attachments}<!-- ATTACH FILE LIST END --></div>]]></HTMLContent>
</Record>"""


def attach(label, name):
    return f'<p class="attachment">{label}（<a href="images/{name}">請參見PDF</a>）</p>\n'


def archive(days):
    """ZIP like the Open Data ones: <day>/<day>.xml and <day>/<metaid>.pdf."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for day, records in days.items():
            handle.writestr(f"{day}/", b"")
            for metaid in [r.split("<MetaId><![CDATA[")[1].split("]")[0] for r in records]:
                handle.writestr(f"{day}/{metaid}.pdf", b"%PDF-1.4 " + metaid.encode())
            handle.writestr(f"{day}/{day}.xml", '<?xml version="1.0" encoding="UTF-8"?>\n<Gazette>\n'
                            + "\n".join(records) + "\n</Gazette>")
    return buffer.getvalue()


def link(kind, token, name):
    href = f"fileView.do?fileType={kind}&amp;fileName={token}&amp;attached&amp;log=x&amp;zipfile={name}.zip"
    return f'<a href="{href}">{name}.zip</a>'


def zip_url(kind, token, name):
    return f"{SITE}fileView.do?fileType={kind}&fileName={token}&attached&log=x&zipfile={name}.zip"


PREVIEW = "法規命令訂定、修正草案及廢止案之預告"
OCT16 = [record(168928, "內政部國家公園署公告：預告「太魯閣國家公園奇萊山屋使用規費收費標準」草案",
                "eg032192/ch02/type3/gov10/num1", style="公告及送達", substyle=PREVIEW,
                attachments=attach("收費標準草案總說明及逐條說明", "Eg01.pdf") + attach("其他事項說明", "OO.pdf")),
         record(168943, "勞動部公告：修正「最低工資」，自116年1月1日生效", "eg032192/ch08/type1/gov82/num28",
                chapter="衛生勞動篇", agency="勞動部", attachments=attach("其他事項說明", "OO.pdf"))]
OCT15 = [record(168900, "內政部移民署對某甲等251人廢止居留許可並註銷外僑居留證處分書公示送達",
                "eg032191/ch02/type3/gov10/num5", style="公告及送達", substyle="公示送達"),
         record(168901, "經濟部公告：公告進口貨物稅則", "eg032191/ch04/type3/gov31/num1", style="公告及送達",
                substyle="依法規應公告之事項", chapter="財政經濟篇"),
         record(168902, "財政部對某公司處分", "eg032191/ch04/type5/gov30/num2", style="處分", substyle="法規規定應刊登之處分"),
         record(168903, "某部函：前職員某某違法案，經公務員懲戒委員會議決記過1次", "eg032191/ch01/type7/gov04/num3",
                style="其他", substyle="更正"),
         record(168904, "考選部令：修正「公務員懲戒法施行細則」", "eg032191/ch01/type1/gov04/num4", chapter="綜合行政篇")]
SEP30 = [record(168618, "財政部令：修正「海關管理保稅運貨工具辦法」", "eg032181/ch04/type1/gov30/num2",
                chapter="財政經濟篇", agency="財政部",
                attachments=attach("海關管理保稅運貨工具辦法修正總說明及對照表", "AA.pdf")
                + attach("其他事項說明", "OO.pdf")),
         record(168610, "環境部公告：排放管道中重金屬檢測方法", "eg032181/ch07/type1/gov60/num17", chapter="農業環保篇",
                attachments=attach("排放管道中重金屬檢測方法（NIEA A302.74C）", "Eg01.pdf")
                + attach("排放管道中重金屬檢測方法總說明及逐項說明", "AA.pdf"))]
SEP29 = [record(168928, "重複的項目", "eg032180/ch02/type1/gov10/num1")]
WORD = "mso-bidi-font-family:&quot;Times New Roman&quot;;" * 40  # Word HTML of 2016: long style attributes
JUN16 = [record(84123, "經濟部令：修正「創業投資事業輔導辦法」第三條、第七條、第八條", "eg022101/ch04/type1/gov31/num1",
                chapter="財政經濟篇", scheme="http",
                attachments=f'<p class="013" style="{WORD}"><span style="{WORD}"><b>創業投資事業輔導辦法第三條、第七條、'
                            '第八條</b></span><span><b>總說明及對照表</b>&nbsp;（<a href="images/Eg01.pdf">請參見PDF</a>）<br />'
                            '<b>附件</b>（<a href="images/Eg02.pdf">請參見PDF</a>）</span></p>'
                            '<p>附表（<a href="images/Eg03.htm">網頁</a>）</p>')]


def site():
    month = link("zipPath", "aa16", "115-10-16") + link("zipPath", "aa15", "115-10-15")
    history115 = link("historyPath", "h0904", "115-09_04") + link("historyPath", "h0903", "115-09_03") \
        + link("historyPath", "h0902", "115-09_02") + link("historyPath", "h0801", "115-08_01") \
        + link("historyPath", "h1001", "115-10_01")
    return {SITE + "openData03.do": f"<div>民國115年10月 {month}</div>",
            SITE + "openData04.do?action=historyYearDaily&year=115": history115,
            SITE + "openData04.do?action=historyYearDaily&year=105": link("historyPath", "h105", "105-06_01"),
            zip_url("zipPath", "aa16", "115-10-16"): archive({"115-10-16": OCT16}),
            zip_url("zipPath", "aa15", "115-10-15"): archive({"115-10-15": OCT15}),
            zip_url("historyPath", "h0904", "115-09_04"): archive({"115-09-30": SEP30}),
            zip_url("historyPath", "h0903", "115-09_03"): archive({"115-09-28": [], "115-09-29": SEP29}),
            zip_url("historyPath", "h105", "105-06_01"): archive({"105-06-01": JUN16})}


def test_open_data_archives_newest_first_with_filters_and_attachments():
    fake = FakeFetcher(site())
    found = list(taiwan.ey_gazette(fake, {"start": "2026-09-29", "end": "2026-10-16"}))
    assert [c["key"] for c in found] == ["twgazette:168928", "twgazette:168943", "twgazette:168904",
                                         "twgazette:168618", "twgazette:168618:AA", "twgazette:168610",
                                         "twgazette:168610:AA"]
    main = found[0]
    assert main["url"] == PUB + "eg032192/ch02/type3/gov10/num1/Eg.pdf" and main["type_hint"] == "pdf"
    assert main["title"].startswith("內政部國家公園署公告：預告") and main["license"].startswith("Taiwan Copyright Act")
    assert main["meta"] == dict(main["meta"], metaid=168928, date="2026-10-16", volume=32, issue=192, chapter="內政篇",
                                chapter_no=2, style="公告及送達", substyle=PREVIEW, agency="內政部", part="main",
                                archive="115-10-16.zip", archive_member="115-10-16/168928.pdf",
                                archive_url=zip_url("zipPath", "aa16", "115-10-16"),
                                detail=SITE + "detail.do?metaid=168928")
    assert "某某" not in str(found)  # agency heads' names stay out of the metadata
    aa = found[4]
    assert aa["url"] == PUB + "eg032181/ch04/type1/gov30/num2/images/AA.pdf"
    assert aa["title"] == "海關管理保稅運貨工具辦法修正總說明及對照表"
    assert aa["meta"]["part"] == "attachment" and aa["meta"]["item_title"] == "財政部令：修正「海關管理保稅運貨工具辦法」"
    # Days outside the range and packs of other months are not downloaded; the 10-15 day ZIP is.
    assert zip_url("historyPath", "h0801", "115-08_01") not in fake.calls
    assert zip_url("historyPath", "h1001", "115-10_01") not in fake.calls  # its month is listed day by day
    assert fake.calls[:2] == [SITE + "openData03.do", SITE + "openData04.do?action=historyYearDaily&year=115"]
    assert fake.calls.count(zip_url("historyPath", "h0903", "115-09_03")) == 1


def test_personal_rule_and_title_filter():
    def wanted(style, substyle, title):
        return not taiwan._personal(dict(style=style, substyle=substyle, title=title))

    assert wanted("公告及送達", "法規命令草案之預告", "預告「某某辦法」草案")  # 2005 name of the preview type
    assert not wanted("公告及送達", "公示送達", "勞工保險局對某君老年給付公文書公示送達")
    assert not wanted("處分", "法規規定應刊登之處分", "核准某公司")
    assert not wanted("行政規則", "", "公告115年度合格人員名單")
    assert not wanted("其他", "更正", "某監理所對某甲等324件汽車燃料使用費催繳通知書")
    assert wanted("法規", "", "修正「公務員懲戒法施行細則」") and wanted("特載", "", "中華民國115年國慶　總統講話")
    found = list(taiwan.ey_gazette(FakeFetcher(site()), {"start": "2026-10-15", "end": "2026-10-15",
                                                         "skip_personal": False, "attachments": False}))
    assert [c["meta"]["metaid"] for c in found] == [168900, 168901, 168902, 168903, 168904]


def test_options_chapters_styles_limits_and_dates():
    options = {"start": "2026-09-01", "end": "2026-10-16"}
    by_number = taiwan.ey_gazette(FakeFetcher(site()), dict(options, chapters=[8, "7"]))
    assert [c["key"] for c in by_number] == ["twgazette:168943", "twgazette:168610", "twgazette:168610:AA"]
    assert [c["key"] for c in taiwan.ey_gazette(FakeFetcher(site()), dict(options, chapters=8))] == ["twgazette:168943"]
    by_name = taiwan.ey_gazette(FakeFetcher(site()), dict(options, chapters="財政經濟", attachments=False))
    assert [c["key"] for c in by_name] == ["twgazette:168618"]
    styles = taiwan.ey_gazette(FakeFetcher(site()), dict(options, styles=["公告及送達"]))
    assert [c["key"] for c in styles] == ["twgazette:168928"]
    assert len(list(taiwan.ey_gazette(FakeFetcher(site()), dict(options, max_items=3)))) == 3
    fake = FakeFetcher(site())
    current = list(taiwan.ey_gazette(fake, {"start": "2026-10-16", "end": "2026-10-31"}))
    assert [c["key"] for c in current] == ["twgazette:168928", "twgazette:168943"]
    assert fake.calls == [SITE + "openData03.do", zip_url("zipPath", "aa16", "115-10-16")]  # no history pages
    fake = FakeFetcher(site())
    last_day = taiwan.ey_gazette(fake, {"start": "2026-09-30", "end": "2026-09-30", "attachments": False})
    assert [c["key"] for c in last_day] == ["twgazette:168618", "twgazette:168610"]
    # Packs come newest first: once one ends before the start, older ones are not downloaded.
    assert zip_url("historyPath", "h0903", "115-09_03") in fake.calls
    assert zip_url("historyPath", "h0902", "115-09_02") not in fake.calls
    assert taiwan._roc("中華民國115年10月16日").isoformat() == "2026-10-16" and taiwan._roc("115-02-30") is None


def test_older_archives_http_links_and_order_summaries():
    fake = FakeFetcher(site())
    found = list(taiwan.ey_gazette(fake, {"start": "2016-06-01", "end": "2016-06-30"}))
    assert [c["key"] for c in found] == ["twgazette:84123", "twgazette:84123:Eg01"]
    assert found[0]["url"] == PUB + "eg022101/ch04/type1/gov31/num1/Eg.pdf"  # http:// upgraded
    assert found[1]["title"] == "創業投資事業輔導辦法第三條、第七條、第八條總說明及對照表"
    assert found[0]["meta"]["volume"] == 22 and found[0]["meta"]["issue"] == 101
    years = [url for url in fake.calls if "historyYearDaily" in url]
    assert years == [SITE + f"openData04.do?action=historyYearDaily&year={y}" for y in (105,)]


def test_latest_xml_and_failures_are_logged():
    xml = '<?xml version="1.0"?>\n<Gazette>\n' + OCT16[1].replace(
        "<Title>", "<Date_Published><![CDATA[中華民國115年10月16日]]></Date_Published>\n<Title>") + "</Gazette>"
    found = list(taiwan.ey_gazette(FakeFetcher({SITE + "OpenData/downloadXML.jsp": xml}), {"latest": True}))
    assert [c["key"] for c in found] == ["twgazette:168943"]
    assert found[0]["meta"]["date"] == "2026-10-16" and found[0]["meta"]["archive_member"] is None
    pages = site()
    pages[zip_url("zipPath", "aa16", "115-10-16")] = b"<html>Just a moment...</html>"
    del pages[zip_url("zipPath", "aa15", "115-10-15")]
    messages = []
    found = list(taiwan.ey_gazette(FakeFetcher(pages), {"start": "2026-10-01", "log": messages.append}))
    assert found == []
    assert any(m.startswith("bad archive 115-10-16.zip") for m in messages)
    assert any("http_404" in m and "115-10-15.zip" in m for m in messages)
    assert list(taiwan.ey_gazette(FakeFetcher({}), {"start": "2026-10-01"})) == []


DETAIL = """<meta property="og:title" content="財政部令：修正「海關管理保稅運貨工具辦法」">
<li class="active">第032卷第181期(2026-09-30)</li>
<a title="附表PDF" class="btn" href="/EG_FileManager/eguploadpub/eg032181/ch04/type1/gov30/num2/Eg.pdf">PDF</a>
<div class="Side_Links"><div class="Item"><span class="Tag">法規</span>
<h2>財政經濟篇</h2>
<h2>財政部</h2>
<div class="Inner_wrap"><dl><dt>類型：</dt>
<dd>中央法規標準法第3條之7種命令</dd>
<dt>出刊日期：</dt>
<dd>2026-09-30</dd></dl></div></div></div>"""


def test_metaids_use_detail_pages_and_eg_htm():
    base = PUB + "eg032181/ch04/type1/gov30/num2/"
    pages = {SITE + "detail.do?metaid=168618&log=detailLog": DETAIL,
             base + "Eg.htm": ('<meta charset="big-5">' + attach("對照表", "AA.pdf") + attach("其他事項說明", "OO.pdf")
                               + attach("修正總說明", "Eg01.pdf")).encode("cp950"),
             SITE + "detail.do?metaid=168921&log=detailLog": DETAIL.replace("法規</span>", "公告及送達</span>")
             .replace("中央法規標準法第3條之7種命令", "公示送達"),
             SITE + "detail.do?metaid=5&log=detailLog": "<html>查無資料</html>"}
    fake, messages = FakeFetcher(pages), []
    found = list(taiwan.ey_gazette(fake, {"metaids": [168618, "168921", 5, 6], "log": messages.append}))
    assert [c["key"] for c in found] == ["twgazette:168618", "twgazette:168618:AA", "twgazette:168618:Eg01"]
    assert found[0]["url"] == base + "Eg.pdf" and found[0]["title"] == "財政部令：修正「海關管理保稅運貨工具辦法」"
    assert found[0]["meta"] == dict(found[0]["meta"], date="2026-09-30", chapter="財政經濟篇", agency="財政部",
                                    style="法規", volume=32, issue=181, archive=None)
    assert found[1]["title"] == "財政部令：修正「海關管理保稅運貨工具辦法」（對照表）"
    assert base + "Eg.htm" in fake.calls and fake.calls.count(base + "Eg.htm") == 1  # not for the 公示送達 item
    assert any("no PDF" in m and "5" in m for m in messages) and any("http_404" in m for m in messages)
    for metaids in ("168618", 168618):  # --opt metaids=168618 arrives as a JSON number
        assert [c["key"] for c in taiwan.ey_gazette(FakeFetcher(pages), {"metaids": metaids, "attachments": False})] \
            == ["twgazette:168618"]


class FakeHTTP:
    """Transport for the real Fetcher: canned bodies by URL, 404 otherwise (robots.txt too)."""

    def __init__(self, routes):
        self.routes, self.headers = routes, {}

    def request(self, method, url, **kwargs):
        body = self.routes.get(url)
        status, body = (404, b"") if body is None else (200, body.encode() if isinstance(body, str) else body)
        return SimpleNamespace(status_code=status, headers={}, url=url, is_redirect=False, text=body.decode("latin-1"),
                               iter_content=lambda size: iter([body]), close=lambda: None)


def test_real_fetcher_without_robots_rules():
    from data_pipeline.documents.fetch import Fetcher
    fetcher = Fetcher(http=FakeHTTP(site()), interval=0, sleep=lambda s: None)
    found = list(taiwan.ey_gazette(fetcher, {"start": "2026-10-16", "end": "2026-10-16"}))
    assert [c["key"] for c in found] == ["twgazette:168928", "twgazette:168943"]
