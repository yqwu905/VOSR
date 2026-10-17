"""cninfo adapters against an in-memory imitation of hisAnnouncement/query (offline)."""

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from data_pipeline.documents import catalog, cninfo
from data_pipeline.documents.fetch import Fetcher, FetchError

CST = timezone(timedelta(hours=8))
PLATES = {"sz": {"SZZB", "SZZX", "SZCY"}, "szmb": {"SZZB", "SZZX"}, "szcy": {"SZCY"}, "sh": {"SHZB", "SHKCB"},
          "shmb": {"SHZB"}, "shkcp": {"SHKCB"}, "bj": {"BJS"}, "hkmb": {"HKZB"}, "hkcy": {"HKCY"}}


def row(aid, day, title, kind="PDF", code="000001", tab="fulltext", column="szse", plate="SZZB", category="",
        hour=12):
    when = datetime.fromisoformat(day).replace(hour=hour, tzinfo=CST)
    ext = kind.strip(".").upper()
    return {"announcementId": str(aid), "secCode": code, "secName": f"公司{code}", "orgId": f"gssz0{code}",
            "announcementTitle": title, "announcementTime": int(when.timestamp() * 1000) + aid % 1000,
            "adjunctUrl": f"finalpage/{day}/{aid}.{ext}", "adjunctSize": 321, "adjunctType": kind, "pageColumn": plate,
            "_day": day, "_tab": tab, "_column": column, "_category": category}


class FakeCninfo:
    """Filters by the form, sorts newest first, serves 30-row pages and wraps page 101+ to page 1, like the site."""

    def __init__(self, rows, fail=None, wrap_after=100):
        self.rows, self.fail, self.wrap_after, self.forms = rows, fail, wrap_after, []

    def get(self, url, method="GET", data=None, headers=None, **kwargs):
        assert url == cninfo.API and method == "POST" and headers["Referer"].startswith("http://www.cninfo.com.cn/")
        self.forms.append(dict(data))
        start, end = data["seDate"].split("~")
        if self.fail and (reason := self.fail(data, date.fromisoformat(start), date.fromisoformat(end))):
            raise FetchError(reason, url)
        hits = [r for r in self.rows if start <= r["_day"] <= end and r["_tab"] == data["tabName"]
                and r["_column"] == data["column"] and (not data["plate"] or r["pageColumn"] in PLATES[data["plate"]])
                and (not data["category"] or r["_category"] == data["category"])
                and data["searchkey"] in r["announcementTitle"]]
        hits.sort(key=lambda r: -r["announcementTime"])
        page = data["pageNum"] if data["pageNum"] <= self.wrap_after else 1
        shown = [{k: v for k, v in r.items() if not k.startswith("_")} for r in hits[(page - 1) * 30:page * 30]]
        key = data["searchkey"]
        for r in shown:
            if key:
                r["announcementTitle"] = r["announcementTitle"].replace(key, f"<em>{key}</em>")
        body = {"totalAnnouncement": len(hits), "totalpages": len(hits) // 30, "hasMore": page * 30 < len(hits),
                "announcements": shown or None}
        return None, json.dumps(body, ensure_ascii=False).encode()


def days(start, count):
    return [(date.fromisoformat(start) + timedelta(days=i)).isoformat() for i in range(count)]


def test_ir_office_keeps_native_office_files_with_metadata():
    rows = [row(1, "2015-03-02", "2015年3月1日投资者关系活动记录表", "DOC", tab="relation"),
            row(2, "2015-03-03", "2015年3月2日调研活动附件之演示文稿", "PPTX", tab="relation", plate="SZCY"),
            row(3, "2015-05-04", "投资者关系活动记录表", "PDF", tab="relation"),
            row(4, "2015-05-05", "调研活动附件之参会人员名单", "DOCX", tab="relation"),
            row(5, "2015-06-01", "盈利预测表", ".XLS", tab="relation"),
            row(9, "2015-06-01", "2015年5月29日调研活动附件之投资者接待活动签到表", "DOC", tab="relation"),
            row(10, "2015-06-01", "调研活动附件之来访登记表及承诺书扫描件", "DOCX", tab="relation"),
            row(6, "2015-06-02", "2015年6月1日投资者关系活动记录表", ".DOCX", tab="relation"),
            row(7, "2015-06-03", "年度报告", "DOC"),  # fulltext tab: not an IR record
            row(8, "2021-01-05", "2021年投资者关系活动记录表", "DOC", tab="relation")]
    api = FakeCninfo(rows)
    found = list(cninfo.ir_office(api, {"years": 2015}))
    assert [c["key"] for c in found] == ["cninfo:2:2.PPTX", "cninfo:1:1.DOC", "cninfo:6:6.DOCX"]  # newest first
    deck = found[0]
    assert deck["url"] == "http://static.cninfo.com.cn/finalpage/2015-03-03/2.PPTX" and deck["type_hint"] == "pptx"
    assert (deck["meta"]["date"], deck["meta"]["plate"], deck["meta"]["tab"]) == ("2015-03-03", "SZCY", "relation")
    assert deck["license"] == cninfo.LICENSE and deck["headers"]["Referer"] == cninfo.SEARCH
    assert "source" not in deck and "terms" not in deck
    # One query per month window: Jan-Dec 2015, all on the relation tab of column szse.
    assert [f["seDate"] for f in api.forms][:2] == ["2015-01-01~2015-01-31", "2015-02-01~2015-02-28"]
    assert len(api.forms) == 12 and {(f["tabName"], f["column"]) for f in api.forms} == {("relation", "szse")}
    pdfs = cninfo.ir_office(FakeCninfo(rows), {"years": "2015", "types": "pdf"})
    assert [c["key"] for c in pdfs] == ["cninfo:3:3.PDF"]


def test_sharding_halves_to_one_day_then_splits_plates():
    # 3,500 rows on 30 April (over the 3,000-row window even for one day; sz alone holds 3,150, sh 350).
    plates = ["SZZB"] * 1800 + ["SZZX"] * 300 + ["SZCY"] * 1050 + ["SHZB"] * 200 + ["SHKCB"] * 150
    rows = [row(10000 + i, "2024-04-30", "2023年年度报告", plate=p, code=f"{i:06d}", category="category_ndbg_szsh")
            for i, p in enumerate(plates)]
    rows += [row(20000 + i, d, "2023年年度报告", code=f"9{i:05d}", category="category_ndbg_szsh")
             for i, d in enumerate(days("2024-04-01", 29) * 20)]
    api = FakeCninfo(rows)
    found = list(cninfo.reports(api, {"start": "2024-04-01", "end": "2024-04-30", "one_per_company_per_year": False}))
    assert len(found) == len(rows) and len({c["key"] for c in found}) == len(rows)
    assert max(f["pageNum"] for f in api.forms) <= 100
    day_plates = [f["plate"] for f in api.forms if f["seDate"] == "2024-04-30~2024-04-30" and f["pageNum"] == 1]
    assert day_plates == ["", "sz", "szmb", "szcy", "sh", "bj"]
    assert all(f["category"] == "category_ndbg_szsh" and f["tabName"] == "fulltext" for f in api.forms)


def test_unsplittable_overflow_stops_at_page_100_and_wrap_guard_stops_early():
    rows = [row(30000 + i, "2024-04-30", "年度报告", plate="SZZB", code=f"{i:06d}") for i in range(3100)]
    messages = []
    api = FakeCninfo(rows)
    found = list(cninfo.reports(api, {"start": "2024-04-30", "end": "2024-04-30", "plate": "szmb",
                                      "one_per_company_per_year": False, "category": "", "log": messages.append}))
    assert len(found) == 3000 and max(f["pageNum"] for f in api.forms) == 100
    assert any("only 3000 are reachable" in m for m in messages)
    # A server that wraps early (page 3 returns page 1) must not produce silent repeats.
    messages.clear()
    api = FakeCninfo(rows[:200], wrap_after=2)
    found = list(cninfo.reports(api, {"start": "2024-04-30", "end": "2024-04-30", "category": "",
                                      "one_per_company_per_year": False, "log": messages.append}))
    assert len(found) == 60 and [f["pageNum"] for f in api.forms] == [1, 2, 3]
    assert any("repeats page 1" in m for m in messages)


def test_timeouts_halve_the_window_and_hard_errors_stop():
    rows = [row(40000 + i, d, "投资者关系活动记录表", "DOC", tab="relation") for i, d in enumerate(days("2016-01-01", 60))]
    api = FakeCninfo(rows, fail=lambda form, start, end: "http_504" if (end - start).days > 9 else None)
    found = list(cninfo.ir_office(api, {"start": "2016-01-01", "end": "2016-02-29", "window_months": 0}))
    assert len(found) == 60
    assert any(f["seDate"] == "2016-01-01~2016-02-29" for f in api.forms)  # tried whole, then halved
    messages = []
    api = FakeCninfo(rows, fail=lambda form, start, end: "network:Reset" if start <= date(2016, 1, 3) <= end else None)
    found = list(cninfo.ir_office(api, {"start": "2016-01-01", "end": "2016-01-31", "log": messages.append}))
    assert len(found) == 30 and any("skipped 2016-01-03~2016-01-03" in m for m in messages)
    with pytest.raises(FetchError):
        list(cninfo.ir_office(FakeCninfo(rows, fail=lambda *a: "http_403"), {"years": 2016}))
    # An outage is not a slow window: give up after a few consecutive failures instead of splitting forever.
    api = FakeCninfo(rows, fail=lambda *a: "http_503")
    with pytest.raises(FetchError) as error:
        list(cninfo.ir_office(api, {"years": 2016}))
    assert error.value.reason == "api_down:http_503" and len(api.forms) == 6

    class Challenge:
        def get(self, url, **kwargs):
            return None, b"<html><script>var arg1='challenge'</script></html>"

    with pytest.raises(FetchError) as error:
        list(cninfo.ir_office(Challenge(), {"years": 2016}))
    assert error.value.reason == "not_json"


def test_max_queries_bounds_the_work():
    rows = [row(50000 + i, d, "记录表", "DOC", tab="relation") for i, d in enumerate(days("2016-01-01", 300) * 3)]
    api = FakeCninfo(rows)
    list(cninfo.ir_office(api, {"years": 2016, "max_queries": 5}))
    assert len(api.forms) == 5


def test_slides_title_filters_types_dedupe_and_native_attachments():
    rows = [row(1, "2024-05-10", "2024年5月10日投资者关系活动记录表附件之演示文稿", tab="relation"),
            row(2, "2024-05-11", "业绩说明会材料之演示文稿", tab="relation"),  # two keywords, one candidate
            row(3, "2025-04-22", "2024年度业绩说明会PPT"),
            row(4, "2025-04-20", "关于召开2024年度业绩说明会的通知（附PPT）"),
            row(5, "2025-04-23", "演示文稿（英文版）", tab="relation"),
            row(6, "2025-04-24", "业绩发布会演示文稿", "DOC", tab="relation"),
            row(7, "2025-08-29", "中国某某：中期业绩推介材料", code="601601", plate="SHZB"),
            row(8, "2025-03-27", "年度业绩发布会材料"),
            row(9, "2015-03-24", "2015年3月24日调研活动附件之公司简介", "PPTX", tab="relation"),
            row(10, "2015-03-25", "2015年3月25日调研活动附件之调研记录", "DOC", tab="relation"),
            row(11, "2016-06-01", "2016年6月1日路演活动附件之演示文稿（英文）", "PPT", tab="relation"),
            row(12, "2014-05-20", "2014年5月15日路演活动附件之公司介绍", ".PPT", tab="relation"),
            row(13, "2025-03-20", "年度业绩演示", column="hke", plate="HKZB", code="00700")]
    api = FakeCninfo(rows)
    found = list(cninfo.slides(api, {"start": "2012-01-01", "end": "2025-12-31"}))
    ids = [c["meta"]["announcement_id"] for c in found]
    assert sorted(ids, key=int) == ["1", "2", "3", "7", "8", "9", "12"]
    assert len(ids) == len(set(ids))
    by_id = {c["meta"]["announcement_id"]: c for c in found}
    assert by_id["1"]["title"] == "2024年5月10日投资者关系活动记录表附件之演示文稿"  # <em> highlighting removed
    assert by_id["12"]["type_hint"] == "ppt" and by_id["12"]["meta"]["searchkey"] == "附件"
    assert {f["column"] for f in api.forms} == {"szse"}
    # Whole-range keyword windows; the attachment pass covers 2012-2020 only.
    assert any(f["seDate"] == "2012-01-01~2025-12-31" for f in api.forms)
    assert {f["seDate"] for f in api.forms if f["searchkey"] == "附件"} == {"2012-01-01~2020-12-31"}
    hk = list(cninfo.slides(FakeCninfo(rows), {"start": "2025-01-01", "end": "2025-12-31", "include_hk": True,
                                               "keywords": [], "attachments": False}))
    assert [c["meta"]["announcement_id"] for c in hk] == ["13"] and hk[0]["meta"]["column"] == "hke"


def test_reports_one_per_company_per_year_and_title_filters():
    ndbg = "category_ndbg_szsh"
    rows = [row(1, "2025-04-25", "2024年年度报告", category=ndbg),
            row(2, "2025-04-25", "2024年年度报告摘要", category=ndbg),
            row(3, "2025-04-25", "2024年年度报告（英文版）", category=ndbg),
            row(4, "2025-04-28", "2024年年度报告（更正后）", category=ndbg),
            row(5, "2025-03-15", "2024年年度报告", code="600000", plate="SHZB", category=ndbg),
            row(6, "2025-08-20", "2025年半年度报告", category="category_bndbg_szsh"),
            row(7, "2025-08-21", "2025年半年度报告摘要", category="category_bndbg_szsh"),
            row(8, "2025-02-10", "关于股东名册的公告", code="000002", category=ndbg),
            row(9, "2024-04-20", "2023年年度报告", category=ndbg),
            row(10, "2025-01-18", "2022年年度报告全文（更正后）", category=ndbg),  # an old report, corrected
            row(11, "2025-09-01", "2024年年度报告（更新后）", category=ndbg)]
    found = list(cninfo.reports(FakeCninfo(rows), {"years": 2025}))
    # One per company and report year: the corrected April report wins over the original of the same month
    # and blocks a later update; a corrected FY2022 report published in 2025 does not block FY2024.
    assert [c["meta"]["announcement_id"] for c in found] == ["10", "5", "4"]
    assert found[0]["meta"]["category"] == ndbg and found[0]["type_hint"] == "pdf"
    both = list(cninfo.reports(FakeCninfo(rows), {"years": [2024, 2025], "category": [ndbg, "category_bndbg_szsh"]}))
    assert [c["meta"]["announcement_id"] for c in both] == ["9", "10", "5", "4", "6"]
    options = {"years": "2025", "one_per_company_per_year": False, "exclude": "英文"}
    every = list(cninfo.reports(FakeCninfo(rows), options))
    assert sorted(c["meta"]["announcement_id"] for c in every) == ["1", "10", "11", "2", "4", "5"]
    api = FakeCninfo(rows)
    list(cninfo.reports(api, {}))
    last = date.today().year - 1
    assert api.forms[0]["seDate"] == f"{last}-01-01~{last}-01-31" and api.forms[-1]["seDate"].endswith(f"{last}-12-31")


def test_windows_and_ranges():
    assert list(cninfo._windows(date(2024, 1, 15), date(2024, 3, 10), 1)) == [
        (date(2024, 1, 15), date(2024, 1, 31)), (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 10))]
    yearly = list(cninfo._windows(date(2023, 11, 1), date(2024, 12, 31), 12))
    assert yearly == [(date(2023, 11, 1), date(2024, 10, 31)), (date(2024, 11, 1), date(2024, 12, 31))]
    assert cninfo._range({"years": "2013-2016"}, "2000-01-01", "2000-01-01") == (date(2013, 1, 1), date(2016, 12, 31))
    assert cninfo._range({"years": [2013], "end": "2013-06-30"}, None, None) == (date(2013, 1, 1), date(2013, 6, 30))
    assert cninfo._range({}, "2020-01-01", "2999-01-01")[1] <= date.today() + timedelta(days=1)


class Response:
    def __init__(self, status, body=b""):
        self.status_code, self._body, self.headers, self.url, self.is_redirect = status, body, {}, "", False

    def iter_content(self, size):
        yield self._body

    def close(self):
        pass


class Transport:
    """requests.Session stand-in that hands POSTs to the imitation API."""
    headers = {}

    def __init__(self, api):
        self.api, self.calls = api, []

    def request(self, method, url, data=None, headers=None, **kwargs):
        self.calls.append((method, url))
        if url.endswith("/robots.txt"):
            return Response(404)
        return Response(200, self.api.get(url, method=method, data=data, headers=headers)[1])


def test_catalog_gates_terms_and_requests_go_through_the_fetcher():
    rows = [row(1, "2013-07-24", "2013年7月22日媒体采访附件之演示文稿", "PPTX", tab="relation")]
    transport = Transport(FakeCninfo(rows))
    fetcher = Fetcher(http=transport, interval=0, sleep=lambda s: None)
    with pytest.raises(PermissionError):
        next(catalog.discover("cninfo_ir_office", fetcher, {"years": 2013}))
    found = list(catalog.discover("cninfo_ir_office", fetcher, {"years": 2013}, accepted={"disclosure"}))
    assert [(c["source"], c["terms"], c["type_hint"]) for c in found] == [("cninfo_ir_office", "disclosure", "pptx")]
    assert transport.calls[0] == ("GET", "http://www.cninfo.com.cn/robots.txt")
    assert transport.calls.count(("POST", cninfo.API)) == 12
    assert catalog.load("cninfo_slides") is cninfo.slides and catalog.load("cninfo_reports") is cninfo.reports
