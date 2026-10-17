"""Academic adapters (offline): DOAJ, OJS over OAI-PMH, the OpenAlex parquet snapshot, NTU theses."""

import io
import json
import os
import random
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.documents import academic, catalog
from data_pipeline.documents.fetch import Fetcher, FetchError


class FakeFetcher:
    """Canned bodies by URL; a value may be (final_url, body), a FetchError or a callable(url, headers)."""

    def __init__(self, routes=None):
        self.routes, self.calls = dict(routes or {}), []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, dict(headers or {})))
        route = self.routes.get(url)
        if callable(route):
            route = route(url, headers or {})
        if route is None:
            raise FetchError("http_404", url)
        if isinstance(route, FetchError):
            raise route
        final, body = route if isinstance(route, tuple) else (url, route)
        headers = {}
        if isinstance(body, SimpleNamespace):
            body, headers = body.body, body.headers
        body = body.encode() if isinstance(body, str) else body
        return SimpleNamespace(url=final, status_code=200, headers=headers), body

    def json(self, url, **kwargs):
        return json.loads(self.get(url, **kwargs)[1])

    def urls(self):
        return [url for url, _ in self.calls]


def page(*metas, body=""):
    tags = "".join(f'<meta name="{name}" content="{content}" />\n' for name, content in metas)
    return f"<!DOCTYPE html><html><head><title>t</title>\n{tags}</head><body>{body}</body></html>"


# --- DOAJ ---

def journal(issn, title="Zhongguo xuebao", licence="CC BY-NC-ND", languages=("ZH",)):
    licences = [{"type": licence, "url": "https://creativecommons.org/licenses/"}]
    return {"id": "j" + issn, "bibjson": {"title": title, "alternative_title": title + "（中文）", "pissn": issn,
                                         "language": list(languages), "ref": {"journal": "http://journal.example.cn/"},
                                         "license": licences}}


def article(aid, issn, year, link, title="An English title"):
    return {"id": aid, "_issn": issn, "bibjson": {"title": title, "year": str(year), "link": [{"type": "fulltext",
            "url": link}], "identifier": [{"type": "pissn", "id": issn}, {"type": "doi", "id": f"10.1/{aid}"}]}}


class FakeDoaj(FakeFetcher):
    """The DOAJ search API: issn and bibjson.year range queries, 400 beyond 1000 results."""

    def __init__(self, journals, articles, routes=None, fail=None):
        super().__init__(routes)
        self.journals, self.articles, self.fail, self.searches = journals, articles, fail, []

    def get(self, url, headers=None, **kwargs):
        if not url.startswith(academic.DOAJ):
            return super().get(url, headers, **kwargs)
        self.calls.append((url, dict(headers or {})))
        parts = urlsplit(url)
        kind, query = parts.path.split("/api/search/")[1].split("/", 1)
        if self.fail and kind == "articles":
            raise FetchError(self.fail, url)
        params = {k: int(v[0]) for k, v in parse_qs(parts.query).items()}
        if params["page"] * params["pageSize"] > 1000:
            raise FetchError("http_400", url)
        query = unquote(query)
        if kind == "journals":
            assert query == "bibjson.language:ZH"
            hits = self.journals
        else:
            self.searches.append((query, params["page"]))
            m = re.fullmatch(r"issn:(\S+)(?: AND bibjson\.year:\[(\d+) TO (\d+)\])?", query)
            hits = [a for a in self.articles if a["_issn"] == m[1]
                    and (not m[2] or int(m[2]) <= int(a["bibjson"]["year"]) <= int(m[3]))]
        size, number = params["pageSize"], params["page"]
        shown = [{k: v for k, v in h.items() if not k.startswith("_")} for h in hits[(number - 1) * size:number * size]]
        return SimpleNamespace(url=url, status_code=200, headers={}), json.dumps(
            {"total": len(hits), "page": number, "results": shown}, ensure_ascii=False).encode()


def test_doaj_resolves_fulltext_links_and_skips_spas_portals_and_other_languages():
    issn = "1000-0001"
    renhe = "https://pmt.ustb.edu.cn/article/doi/10.19591/a2"
    links = {"a1": "https://www.qk.sjtu.edu.cn/jscp/fileup/1007-9610/PDF/1659371792062.pdf",
             "a2": renhe,
             "a3": "http://www.zhenciyanjiu.cn/thesisDetails#10.13702/j.1000-0607.2008.03.012",
             "a4": "https://rlfde.cbpt.cnki.net/portal/journal/portal/client/paper/3d52",
             "a5": "https://xhyxzz.pumch.cn/en/article/doi/10.12290/a5",
             "a6": "https://doi.org/10.3785/a6",
             "a7": "https://pmt.ustb.edu.cn/article/doi/10.19591/a7",
             "a8": "https://pmt.ustb.edu.cn/article/doi/10.19591/a8",
             "a9": "https://www.hjs.sinap.ac.cn/zh/article/a9/",
             "b1": "https://journal.example.cn/b1.PDF"}
    articles = [article(k, "1000-0002" if k == "b1" else issn, 2025, v) for k, v in links.items()]
    articles[6]["bibjson"]["title"] = "2024年拟录用人员名单"
    routes = {renhe: page(("citation_title", "粉末冶金技术研究"), ("citation_language", "zh"),
                          ("citation_pdf_url", "/cn/article/pdf/preview/10.19591/a2.pdf")),
              links["a5"]: page(("citation_language", "en"), ("citation_pdf_url", "/en/article/pdf/a5.pdf")),
              links["a6"]: ("https://www.academax.com/a6.pdf", b"%PDF-1.5 body"),
              links["a8"]: page(("citation_title", "无全文")),
              links["a9"]: page(("citation_pdf_url", "https://www.sciengine.com/doi/pdf/a9"))}
    journals = [journal(issn), journal("1000-0002", "Bilingual", "CC BY", ("EN", "ZH")),
                journal("1000-0003", "Own", "Publisher's own license")]
    api = FakeDoaj(journals, articles, routes)
    found = list(academic.doaj(api, {"probe": 0, "licences": "CC BY-NC-ND,CC BY"}))
    by_key = {c["key"]: c for c in found}
    assert list(by_key) == ["doaj:a1", "doaj:a2", "doaj:a6", "doaj:b1"]
    assert by_key["doaj:a1"]["url"] == links["a1"] and by_key["doaj:a1"]["headers"] == {}
    renhe_pdf = by_key["doaj:a2"]
    assert renhe_pdf["url"] == "https://pmt.ustb.edu.cn/cn/article/pdf/preview/10.19591/a2.pdf"
    assert renhe_pdf["headers"] == {"Referer": renhe} and renhe_pdf["title"] == "粉末冶金技术研究"
    assert renhe_pdf["license"] == "CC BY-NC-ND" and renhe_pdf["type_hint"] == "pdf"
    assert renhe_pdf["meta"]["doi"] == "10.1/a2" and renhe_pdf["meta"]["language"] == "zh"
    assert renhe_pdf["meta"]["host"] == "pmt.ustb.edu.cn" and renhe_pdf["meta"]["issn"] == issn
    assert by_key["doaj:a6"]["url"] == links["a6"]  # the link served the PDF itself
    assert by_key["doaj:b1"]["license"] == "CC BY" and "source" not in renhe_pdf and "terms" not in renhe_pdf
    fetched = api.urls()
    for skipped in ("a1", "a3", "a4", "a7", "b1"):  # PDFs, SPAs, CNKI portals and personal lists are not fetched
        assert links[skipped] not in fetched
    assert not any("1000-0003" in url for url in fetched)  # licence filter


def test_doaj_splits_by_year_ranges_under_the_1000_cap():
    issn = "1000-0607"
    articles = [article(f"y{year}-{i}", issn, year, f"https://j.example.cn/{year}/{i}.pdf")
                for year in range(2001, 2026) for i in range(100)]
    articles += [article(f"big-{i}", issn, 2026, f"https://j.example.cn/2026/{i}.pdf") for i in range(1100)]
    messages = []
    api = FakeDoaj([journal(issn)], articles)
    found = list(academic.doaj(api, {"log": messages.append}))
    assert len(found) == 2500 + 1000 and len({c["key"] for c in found}) == len(found)
    assert max(page for _, page in api.searches) <= 10
    assert any("only 1000 are reachable" in m for m in messages)
    year = time.localtime().tm_year
    assert api.searches[:3] == [(f"issn:{issn}", 1), (f"issn:{issn} AND bibjson.year:[1900 TO {year}]", 1),
                                (f"issn:{issn} AND bibjson.year:[{(1900 + year) // 2 + 1} TO {year}]", 1)]
    # A years option queries the range directly.
    api = FakeDoaj([journal(issn)], articles)
    found = list(academic.doaj(api, {"years": "2024-2025"}))
    assert len(found) == 200 and api.searches[0] == (f"issn:{issn} AND bibjson.year:[2024 TO 2025]", 1)


def test_doaj_drops_journals_without_pdfs_and_bounds_queries():
    spa = [article(f"s{i}", "1000-0001", 2020, f"http://x.example.cn/thesisDetails?columnId={i}") for i in range(250)]
    good = [article(f"g{i}", "1000-0002", 2020, f"https://y.example.cn/{i}.pdf") for i in range(3)]
    messages = []
    api = FakeDoaj([journal("1000-0001"), journal("1000-0002")], spa + good)
    found = list(academic.doaj(api, {"probe": 5, "log": messages.append}))
    assert [c["key"] for c in found] == ["doaj:g0", "doaj:g1", "doaj:g2"]
    assert [s for s in api.searches if "1000-0001" in s[0]] == [("issn:1000-0001", 1)]  # no further pages
    assert any("no PDF among the first 5 articles" in m for m in messages)
    api = FakeDoaj([journal("1000-0001"), journal("1000-0002")], spa + good)
    assert list(academic.doaj(api, {"max_queries": 2, "probe": 0})) == []  # journal list + one search
    assert len(api.calls) == 2
    assert len(list(academic.doaj(FakeDoaj([journal("1000-0002")], good), {"max_items": 2}))) == 2
    assert [c["key"] for c in academic.doaj(FakeDoaj([journal("1000-0001"), journal("1000-0002")], spa + good),
                                            {"journals": "1000-0002", "max_articles": 1})] == ["doaj:g0"]


def test_doaj_api_outage_and_block_pages_stop_the_run():
    api = FakeDoaj([journal(f"1000-000{i}") for i in range(8)], [], fail="http_503")
    with pytest.raises(FetchError) as error:
        list(academic.doaj(api, {}))
    assert error.value.reason == "api_down:http_503" and len(api.searches) == 0 and len(api.calls) == 1 + 6
    unlisted = FakeFetcher({})
    with pytest.raises(FetchError):  # no journal list, no run
        list(academic.doaj(unlisted, {}))
    listing = f"{academic.DOAJ}journals/bibjson.language%3AZH?page=1&pageSize=100"
    blocked = FakeFetcher({listing: "<html>blocked</html>"})
    with pytest.raises(FetchError) as error:
        list(academic.doaj(blocked, {}))
    assert error.value.reason == "not_json"


# --- OJS ---

BASE = academic.OJS
OAI_NS = 'xmlns="http://www.openarchives.org/OAI/2.0/"'


def oai(verb, inner, token=None):
    tail = f'<resumptionToken completeListSize="9" cursor="0">{token}</resumptionToken>' if token else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<OAI-PMH {OAI_NS}><responseDate>2026-10-17T00:00:00Z'
            f'</responseDate><request verb="{verb}">{BASE}</request><{verb}>{inner}{tail}</{verb}></OAI-PMH>')


def oai_error(code):
    return f'<?xml version="1.0"?><OAI-PMH {OAI_NS}><error code="{code}">none</error></OAI-PMH>'


def oai_set(spec, name):
    return f"<set><setSpec>{spec}</setSpec><setName>{name}</setName></set>"


def record(number, title, journal="eccs", galleys=(None,), formats=("application/pdf",), deleted=False,
           rights=("Copyright (c) 2019 作者", "https://creativecommons.org/licenses/by-nc/4.0")):
    view = f"http://ojs.omniscient.sg/index.php/{journal}/article/view/{number}"
    status = ' status="deleted"' if deleted else ""
    header = (f"<header{status}><identifier>oai:ojs.omniscient.sg:article/{number}</identifier>"
              f"<datestamp>2019-12-26T05:35:12Z</datestamp><setSpec>{journal}:ART</setSpec></header>")
    if deleted:
        return f"<record>{header}</record>"
    fields = [("title", title), ("date", "2019-12-26"), ("identifier", view), ("language", "eng"),
              ("source", "电子通信与计算机科学; Vol 1 No 1 (2019); 9"), ("source", "2717-5170"),
              ("identifier", f"10.37155/2717-5170-{number}")]
    fields += [("format", f) for f in formats] + [("rights", r) for r in rights]
    fields += [("relation", f"{view}/{g if g is not None else number + 1000}") for g in galleys]
    dc = "".join(f"<dc:{tag}>{value}</dc:{tag}>" for tag, value in fields)
    return (f'<record>{header}<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" '
            f'xmlns:dc="http://purl.org/dc/elements/1.1/">{dc}</oai_dc:dc></metadata></record>')


def q(**params):
    return f"{BASE}?{urlencode(params)}"


def records_of(spec):
    return q(verb="ListRecords", metadataPrefix="oai_dc", set=spec)


def ojs_routes():
    article = "http://ojs.omniscient.sg/index.php/eccs/article/view/"
    return {
        q(verb="ListSets"): oai("ListSets", oai_set("eng", "English Journal") + oai_set("eng:ART", "Articles")
                                + oai_set("eccs", "电子通信与计算机科学"), token="s1"),
        q(verb="ListSets", resumptionToken="s1"): oai("ListSets", oai_set("eccs:CACT", "计算机") + oai_set("kong", "空刊")),
        records_of("eccs"): oai("ListRecords", "".join([
            record(108, "5G+AI的综合应用对经济和社会影响", galleys=(99,)),
            record(200, "听障学生教学管理系统设计分析", galleys=(1, 2), formats=("application/pdf", "text/html")),
            record(201, "无关联的中文记录", galleys=()),
            record(202, "", deleted=True),
            record(203, "An English article in a Chinese journal"),
            record(204, "2020年优秀作者名单"),
        ]), token="r1"),
        q(verb="ListRecords", resumptionToken="r1"): oai("ListRecords", "".join([
            record(205, "只有网页版的文章", galleys=(7,), formats=("text/html",)),
            record(108, "5G+AI的综合应用对经济和社会影响", galleys=(99,)),
        ])),
        records_of("kong"): oai_error("noRecordsMatch"),
        records_of("eng"): oai("ListRecords", record(300, "Teaching English", journal="eng"),
                                                         token="e1"),
        article + "200": page(("citation_pdf_url", article.replace("view", "download") + "200/1")),
        article + "201": page(("citation_language", "en"),
                              ("citation_pdf_url", "/index.php/eccs/article/download/201/5")),
        article + "205": page(("citation_title", "x")),
    }


def test_ojs_harvests_chinese_titles_and_resolves_galleys():
    messages = []
    api = FakeFetcher(ojs_routes())
    found = list(academic.ojs(api, {"log": messages.append}))
    assert [c["key"] for c in found] == ["ojs:ojs.omniscient.sg:108", "ojs:ojs.omniscient.sg:200",
                                         "ojs:ojs.omniscient.sg:201"]
    first = found[0]
    assert first["url"] == "http://ojs.omniscient.sg/index.php/eccs/article/download/108/99"
    assert first["license"] == "CC BY-NC 4.0" and first["title"] == "5G+AI的综合应用对经济和社会影响"
    assert first["headers"] == {"Referer": "http://ojs.omniscient.sg/index.php/eccs/article/view/108"}
    meta = first["meta"]
    assert (meta["issn"], meta["journal"], meta["galley_id"], meta["date"]) == ("2717-5170", "电子通信与计算机科学", "99",
                                                                                 "2019-12-26")
    assert meta["doi"] == "10.37155/2717-5170-108" and meta["oai_id"] == "oai:ojs.omniscient.sg:article/108"
    # Two galleys or none: the article page's citation_pdf_url decides (relative URLs resolved).
    assert found[1]["url"].endswith("/eccs/article/download/200/1") and found[1]["meta"]["galley_id"] == "1"
    assert found[2]["url"] == "http://ojs.omniscient.sg/index.php/eccs/article/download/201/5"
    urls = api.urls()
    assert "http://ojs.omniscient.sg/index.php/eccs/article/view/108" not in urls  # dc:relation was enough
    assert "http://ojs.omniscient.sg/index.php/eccs/article/view/205" in urls  # HTML-only galley: page has no PDF
    assert any("no PDF galley for oai:ojs.omniscient.sg:article/205" in m for m in messages)
    # Chinese-named sets first; the English journal's first page has no Chinese title, so page 2 is never read.
    sets = [parse_qs(urlsplit(u).query).get("set", [None])[0] for u in urls if "ListRecords" in u and "set=" in u]
    assert sets == ["eccs", "kong", "eng"] and q(verb="ListRecords", resumptionToken="e1") not in urls
    assert any("skipping it" in m for m in messages)


def test_ojs_options_sets_article_page_and_record_budget():
    api = FakeFetcher(ojs_routes())
    found = list(academic.ojs(api, {"sets": "eccs", "max_records": 3}))
    assert [c["key"] for c in found] == ["ojs:ojs.omniscient.sg:108", "ojs:ojs.omniscient.sg:200",
                                         "ojs:ojs.omniscient.sg:201"]
    assert q(verb="ListSets") not in api.urls() and q(verb="ListRecords", resumptionToken="r1") not in api.urls()
    routes = ojs_routes()
    routes["http://ojs.omniscient.sg/index.php/eccs/article/view/108"] = page(
        ("citation_pdf_url", "http://ojs.omniscient.sg/index.php/eccs/article/download/108/98"))
    found = list(academic.ojs(FakeFetcher(routes), {"sets": ["eccs"], "article_page": True, "max_items": 1}))
    assert found[0]["url"].endswith("/download/108/98")
    broken = FakeFetcher({records_of("eccs"): "<html>not xml"})
    assert list(academic.ojs(broken, {"sets": "eccs"})) == []


class EndlessOai(FakeFetcher):
    """An OAI endpoint answering every request through respond(verb, request count); fails past a budget."""

    def __init__(self, respond, budget=300):
        super().__init__()
        self.respond, self.budget = respond, budget

    def get(self, url, headers=None, **kwargs):
        assert len(self.calls) < self.budget, f"still paging after {len(self.calls)} requests"
        self.calls.append((url, dict(headers or {})))
        verb = parse_qs(urlsplit(url).query)["verb"][0]
        return SimpleNamespace(url=url, status_code=200, headers={}), self.respond(verb, len(self.calls)).encode()


def test_oai_paging_stops_on_repeated_tokens_and_is_bounded():
    def sets(token):
        return lambda verb, n: (oai("ListSets", oai_set("j1", "期刊"), token=token(n)) if verb == "ListSets"
                                else oai_error("noRecordsMatch"))
    messages = []
    api = EndlessOai(sets(lambda n: "same"))  # the endpoint hands back the same token for ever
    assert list(academic.ojs(api, {"log": messages.append})) == []
    assert sum("ListSets" in u for u in api.urls()) == 2 and any("resumptionToken repeats" in m for m in messages)
    assert sum("ListRecords" in u for u in api.urls()) == 1  # j1 listed on both pages, harvested once
    for options, pages in (({}, 100), ({"max_set_pages": 4}, 4)):  # a fresh token on every page
        api = EndlessOai(sets(lambda n: f"t{n}"))
        list(academic.ojs(api, options))
        assert sum("ListSets" in u for u in api.urls()) == pages
    api = EndlessOai(lambda verb, n: oai("ListRecords", record(108, "中文标题的文章"), token="again"))
    assert [c["key"] for c in academic.ojs(api, {"sets": "eccs"})] == ["ojs:ojs.omniscient.sg:108"]
    assert len(api.calls) == 2
    api = EndlessOai(lambda verb, n: oai("ListRecords", "", token=f"t{n}"))  # empty pages, fresh tokens
    assert list(academic.ojs(api, {"sets": "eccs", "max_records": 5})) == [] and len(api.calls) == 5


# --- OpenAlex ---

def works_parquet(rows, group=4):
    table = pa.table({
        "id": [f"https://openalex.org/{r[0]}" for r in rows],
        "title": [r[5] for r in rows],
        "publication_year": pa.array([2025] * len(rows), pa.int32()),
        "language": [r[1] for r in rows],
        "open_access": [{"is_oa": r[2], "oa_status": "gold"} for r in rows],
        "best_oa_location": [{"pdf_url": r[3], "license": r[4], "landing_page_url": "https://doi.org/10.1/x"}
                             for r in rows],
        "abstract_inverted_index": [os.urandom(20000).hex() for _ in rows],
    })
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=group)
    return buffer.getvalue()


ROWS = [("W1", "zh", True, "http://ojs.omniscient.sg/index.php/eccs/article/download/1/2", "cc-by-nc", "中文论文一"),
        ("W2", "zh", True, None, "cc-by", "无PDF"),
        ("W3", "en", True, "https://a.example.org/3.pdf", "cc-by", "English"),
        ("W4", "zh", False, "https://a.example.cn/4.pdf", None, "非开放"),
        ("W5", "en", True, "https://a.example.org/5.pdf", "cc-by", "E5"),
        ("W6", "ja", True, "https://j.jp/6.pdf", None, "日本"),
        ("W7", None, True, "https://a.example.org/7.pdf", None, "E7"), ("W8", "en", False, None, None, "E8"),
        ("W9", "zh", True, "https://journals.example.com/9.pdf", None, "中文论文九"),
        ("W10", "zh", True, "https://sci-open.example.net/10.pdf", "cc-by", "中文论文十"),
        ("W11", "zh", True, "https://a.example.cn/11.pdf", "cc-by-nc-nd", "2023年录取名单"),
        ("W12", "zh", True, "http://ojs.omniscient.sg/index.php/eccs/article/download/1/2", "cc-by-nc", "重复"),
        ("W13", "zh", True, "https://kumadai.repo.nii.ac.jp/record/1/files/a.pdf", None, "商標と表現の自由"),
        ("W14", "zh", True, "http://hdl.handle.net/2433/51591", None, "中文"),
        ("W15", "zh", True, "https://x.example.org/15.pdf", "cc-by", "商標と表現")]


class FakeS3(FakeFetcher):
    """Serves byte ranges of in-memory files and remembers every range read."""

    def __init__(self, files, routes=None):
        super().__init__(routes)
        self.files, self.ranges = files, []

    def get(self, url, headers=None, **kwargs):
        if url not in self.files:
            return super().get(url, headers, **kwargs)
        self.calls.append((url, dict(headers or {})))
        data = self.files[url]
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        self.ranges.append((url, start, end))
        headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
        return SimpleNamespace(url=url, status_code=206, headers=headers), data[start:end + 1]

    def transferred(self, url):
        return sum(end - start + 1 for u, start, end in self.ranges if u == url)


def chunk_ranges(data, column):
    meta = pq.ParquetFile(io.BytesIO(data)).metadata
    spans = []
    for group in range(meta.num_row_groups):
        for index in range(meta.num_columns):
            chunk = meta.row_group(group).column(index)
            if chunk.path_in_schema == column:
                start = chunk.dictionary_page_offset or chunk.data_page_offset
                spans.append((group, start, start + chunk.total_compressed_size - 1))
    return spans


def manifest(*names):
    return json.dumps({"files": [{"url": f"s3://openalex/data/parquet/works/{n}", "meta": {"content_length": size}}
                                 for n, size in names]})


def test_openalex_keeps_chinese_oa_pdfs_with_lean_range_reads():
    data = works_parquet(ROWS)
    url = academic.S3 + "data/parquet/works/updated_date=2026-10-14/part_0001.parquet"
    messages = []
    api = FakeS3({url: data}, {academic.MANIFEST: manifest(("updated_date=2026-10-14/part_0001.parquet", len(data)))})
    found = list(academic.openalex(api, {"log": messages.append}))
    assert [c["key"] for c in found] == ["openalex:W1", "openalex:W9", "openalex:W10", "openalex:W11", "openalex:W15"]
    assert [c["license"] for c in found] == ["CC BY-NC", "unknown (no licence in OpenAlex)", "CC BY", "CC BY-NC-ND",
                                             "CC BY"]
    first = found[0]
    assert first["url"] == ROWS[0][3] and first["title"] is None and first["type_hint"] == "pdf"
    assert first["meta"] == {"openalex_id": "https://openalex.org/W1", "language": "zh", "licence_id": "cc-by-nc",
                             "file": url, "host": "ojs.omniscient.sg", "publication_year": 2025}
    # Exact ranges only: past the speculative 64 KB footer read, the abstract column is never touched, nor the
    # id column of the row group without matches.
    reads = [(start, end) for _, start, end in api.ranges[1:]]
    assert api.ranges[0][2] == len(data) - 1 and api.ranges[0][2] - api.ranges[0][1] < 1 << 16
    for _, start, end in chunk_ranges(data, "abstract_inverted_index"):
        assert not any(s <= end and start <= e for s, e in reads)
    group_one_ids = [(s, e) for g, s, e in chunk_ranges(data, "id") if g == 1]
    assert group_one_ids and not any(s <= e2 and s2 <= e for s, e in group_one_ids for s2, e2 in reads)
    assert api.transferred(url) < len(data) / 4 and any("MB read" in m for m in messages)
    titled = list(academic.openalex(FakeS3({url: data}, api.routes), {"columns": "id,title"}))
    assert [c["key"] for c in titled] == ["openalex:W1", "openalex:W9", "openalex:W10"]  # 名单 and kana dropped
    assert titled[0]["title"] == "中文论文一" and "publication_year" not in titled[0]["meta"]
    licensed = list(academic.openalex(FakeS3({url: data}, api.routes), {"licences": "cc-by,none"}))
    assert [c["key"] for c in licensed] == ["openalex:W9", "openalex:W10", "openalex:W15"]
    every_host = list(academic.openalex(FakeS3({url: data}, api.routes), {"exclude_hosts": ""}))
    assert [c["key"] for c in every_host][-3:] == ["openalex:W13", "openalex:W14", "openalex:W15"]
    polite = FakeS3({url: data}, api.routes)
    polite.robots_for = lambda target: SimpleNamespace(allowed=lambda u: "omniscient" not in u)
    assert [c["key"] for c in academic.openalex(polite, {})] == ["openalex:W9", "openalex:W10", "openalex:W11",
                                                                 "openalex:W15"]
    assert [c["key"] for c in academic.openalex(polite, {"robots": False})][0] == "openalex:W1"


def test_openalex_file_selection_failures_and_explicit_files():
    data = works_parquet(ROWS[:4])
    names = [f"updated_date=2026-10-{d:02d}/part_0000.parquet" for d in range(1, 7)]
    urls = [academic.S3 + "data/parquet/works/" + n for n in names]
    routes = {academic.MANIFEST: manifest(*[(n, len(data)) for n in names])}

    def read_files(options):
        api = FakeS3(dict.fromkeys(urls, data), routes)
        list(academic.openalex(api, options))
        return list(dict.fromkeys(u for u, _, _ in api.ranges))

    assert read_files({"max_files": 2}) == urls[::-1][:2]  # newest partitions first
    assert read_files({"max_files": 2, "offset": 2}) == urls[::-1][2:4]
    shuffled = list(reversed(urls))
    random.Random(7).shuffle(shuffled)
    assert read_files({"max_files": 3, "seed": 7}) == shuffled[:3]
    messages = []
    files = dict.fromkeys(urls, data)
    files.pop(urls[-1])
    api = FakeS3(files, dict(routes, **{urls[-1]: FetchError("http_403", urls[-1])}))
    found = list(academic.openalex(api, {"max_files": 2, "log": messages.append}))
    assert [c["meta"]["file"] for c in found] == [urls[-2]]
    assert any("skipped" in m and urls[-1] in m for m in messages)
    api = FakeS3({urls[0]: data})
    found = list(academic.openalex(api, {"files": f"s3://openalex/data/parquet/works/{names[0]}"}))
    assert [c["key"] for c in found] == ["openalex:W1"] and academic.MANIFEST not in api.urls()
    assert api.ranges[0][1:] == (0, 0)  # the size comes from a one-byte range read


def test_range_file_reads_exact_ranges():
    api = FakeS3({"https://s3.example/f": bytes(range(256)) * 4})
    handle = academic._RangeFile(api, "https://s3.example/f", 1024)
    handle.seek(-10, 2)
    assert handle.read(100) == (bytes(range(256)) * 4)[-10:] and handle.read(5) == b""
    handle.seek(3)
    assert handle.read(4) == bytes([3, 4, 5, 6]) and handle.tell() == 7
    assert api.ranges == [("https://s3.example/f", 1014, 1023), ("https://s3.example/f", 3, 6)]
    short = academic._RangeFile(api, "https://s3.example/f", 2048)  # claims more bytes than exist
    short.seek(1020)
    with pytest.raises(FetchError):
        short.read(10)


# --- NTU theses ---

NTU = academic.NTU


def ntu_item(number, title, rights="同意授權(全球公開)", language="zh_TW", pdf=True, degree="碩士", tail="", sequence=1):
    metas = [("DC.title", title), ("DC.rights", rights), ("DC.language", language), ("DC.description", degree),
             ("DCTERMS.issued", "2025"), ("DC.identifier", f"10.6342/NTU2025{number}"), ("citation_title", title),
             ("citation_language", language), ("citation_date", "2025-08-18")]
    if pdf:
        bitstream = f"http://tdr.lib.ntu.edu.tw/jspui/bitstream/123456789/{number}/{sequence}/ntu-113-2.pdf"
        metas.append(("citation_pdf_url", bitstream))
    return page(*metas, body=tail)


def ntu_map(*numbers):
    return "<ul>" + "".join(f'<li><a href="http://tdr.lib.ntu.edu.tw/jspui/handle/123456789/{n}">{n}</a></li>'
                            for n in numbers) + "</ul>"


def ntu_routes():
    index = "".join(f'<a href="http://tdr.lib.ntu.edu.tw/jspui/htmlmap?map={n}">{n}</a>' for n in range(3))
    item = NTU + "handle/123456789/{}?mode=full"
    return {NTU + "htmlmap": index,
            NTU + "htmlmap?map=0": ntu_map(100, 101),
            NTU + "htmlmap?map=1": ntu_map(200, 201, 202),
            NTU + "htmlmap?map=2": ntu_map(300, 301, 302, 302),
            item.format(100): ntu_item(100, "早期論文"),
            item.format(101): ntu_item(101, "檔案尚未公開的論文", sequence=-1),
            item.format(200): ntu_item(200, "研究生名單與身分"),
            item.format(201): ntu_item(201, "沒有全文的論文", pdf=False),
            item.format(202): ntu_item(202, "臺灣白線斑蚊的地景基因體學", degree="博士"),
            item.format(300): ntu_item(300, "An English thesis", language="en"),
            item.format(301): ntu_item(301, "校內公開的論文", rights="同意授權(限校園內公開)"),
            item.format(302): ntu_item(302, "使用全基因組單核苷酸多型性探討台灣白線斑蚊")}


def test_ntu_theses_keeps_worldwide_public_chinese_items_newest_first():
    api = FakeFetcher(ntu_routes())
    found = list(academic.ntu_theses(api, {}))
    assert [c["key"] for c in found] == ["ntu_theses:123456789/302", "ntu_theses:123456789/202",
                                         "ntu_theses:123456789/100"]
    thesis = found[0]
    assert thesis["url"] == "http://tdr.lib.ntu.edu.tw/jspui/bitstream/123456789/302/1/ntu-113-2.pdf"
    assert thesis["title"] == "使用全基因組單核苷酸多型性探討台灣白線斑蚊" and thesis["type_hint"] == "pdf"
    assert thesis["headers"] == {"Referer": NTU + "handle/123456789/302?mode=full"}
    assert "全球公開" in thesis["license"]
    meta = thesis["meta"]
    assert (meta["degree"], meta["year"], meta["doi"]) == ("碩士", "2025", "10.6342/NTU2025302")
    assert found[1]["meta"]["degree"] == "博士" and found[1]["meta"]["map"] == 1
    assert api.urls().count(NTU + "handle/123456789/302?mode=full") == 1  # duplicate handle on the map page
    oldest = list(academic.ntu_theses(FakeFetcher(ntu_routes()), {"maps": "1-2", "order": "oldest", "language": ""}))
    assert [c["meta"]["handle"] for c in oldest] == ["123456789/202", "123456789/300", "123456789/302"]
    api = FakeFetcher(ntu_routes())
    assert len(list(academic.ntu_theses(api, {"maps": 2, "max_items": 1}))) == 1
    assert NTU + "htmlmap" not in api.urls() and NTU + "handle/123456789/301?mode=full" not in api.urls()


class Response:
    def __init__(self, status, body=b"", url=""):
        self.status_code, self._body, self.headers, self.url, self.is_redirect = status, body, {}, url, False
        self.text, self.chunks = body.decode("utf-8", "replace"), 0

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            self.chunks += 1
            yield self._body[i:i + size]

    def close(self):
        pass


class Transport:
    """requests.Session stand-in over the canned NTU pages, with the site's robots.txt."""
    headers = {}

    def __init__(self, routes, robots):
        self.routes, self.robots, self.responses = routes, robots, []

    def request(self, method, url, **kwargs):
        if url.endswith("/robots.txt"):
            return Response(200, self.robots.encode(), url)
        body = self.routes.get(url)
        response = Response(200, body.encode(), url) if body is not None else Response(404, b"", url)
        self.responses.append(response)
        return response


def test_ntu_through_the_fetcher_waits_the_crawl_delay_and_reads_heads_only():
    routes = ntu_routes()
    routes[NTU + "handle/123456789/302?mode=full"] = ntu_item(302, "長篇論文", tail="正文" * 100000)
    transport = Transport(routes, "User-agent: *\ncrawl-delay: 10\nDisallow: /discover\n")
    slept = []
    fetcher = Fetcher(http=transport, interval=2.0, clock=lambda: sum(slept), sleep=slept.append)
    with pytest.raises(PermissionError):
        next(catalog.discover("ntu_theses", fetcher, {"maps": 2}))
    found = list(catalog.discover("ntu_theses", fetcher, {"maps": 2}, accepted={"public_web"}))
    assert [(c["source"], c["terms"], c["title"]) for c in found] == [("ntu_theses", "public_web", "長篇論文")]
    assert sum(slept) >= 10 * 3  # map page and three item pages, each in its own 10 s slot
    long_page = next(r for r in transport.responses if r.url.endswith("/302?mode=full"))
    assert long_page.chunks == 1  # stopped at </head>, not 600 KB later


def test_catalog_wires_the_academic_adapters_and_their_terms():
    assert catalog.load("doaj_zh") is academic.doaj and catalog.load("ojs_zh") is academic.ojs
    assert catalog.load("openalex_zh") is academic.openalex and catalog.load("ntu_theses") is academic.ntu_theses
    assert {catalog.SOURCES[s]["terms"] for s in ("doaj_zh", "ojs_zh", "openalex_zh")} == {"open_nc"}
    api = FakeDoaj([journal("1000-0002")], [article("g0", "1000-0002", 2020, "https://cnki.net.example.cn/g0.pdf"),
                                            article("g1", "1000-0002", 2020, "https://www.sciengine.com/g1.pdf")])
    with pytest.raises(PermissionError):
        next(catalog.discover("doaj_zh", api, {}))
    found = list(catalog.discover("doaj_zh", api, {}, accepted={"open_nc"}))
    assert [(c["key"], c["terms"], c["source"]) for c in found] == [("doaj:g0", "open_nc", "doaj_zh")]


def test_meta_parsing_handles_attribute_order_entities_and_charsets():
    text = ('<meta data-hid="citation_title" name="citation_title" content="惠州&amp;汉族">'
            "<meta content='/a.pdf' name='citation_pdf_url'/><META NAME=\"DC.rights\" CONTENT=\"同意授權(全球公開)\">")
    metas = academic._metas(text)
    assert metas == {"citation_title": ["惠州&汉族"], "citation_pdf_url": ["/a.pdf"], "dc.rights": ["同意授權(全球公開)"]}
    gbk = '<meta charset="gbk"><title>中文</title>'.encode("gbk")
    assert "中文" in academic._decode(gbk)
    assert academic._cc(["Copyright (c) 2020", "http://creativecommons.org/licenses/by/4.0/"]) == "CC BY 4.0"
    assert academic._cc(["https://creativecommons.org/publicdomain/zero/1.0/"]) == "CC0"
    assert academic._chinese("中文标题") and not academic._chinese("日本語のタイトル") and not academic._chinese("A 中")
