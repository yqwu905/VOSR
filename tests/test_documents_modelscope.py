"""ModelScope dataset-repository lister against an in-memory imitation of the /api/v1 endpoints (offline)."""

import json
from urllib.parse import parse_qs, quote_plus, urlsplit

import pytest

from data_pipeline.documents import modelscope
from data_pipeline.documents.fetch import Fetcher, FetchError, Store, download

FINGLM_FILES = ["allpdf/2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
                "allpdf/2021-03-30__贵州茅台酒股份有限公司__600519__贵州茅台__2020年__年度报告.pdf",
                "allpdf/2022-04-28__科大讯飞股份有限公司__002230__科大讯飞__2021年__年度报告.PDF"]
REPORT_FILES = ["2023/000001_平安银行2023年一季度报告.pdf", "2023/000001_平安银行2023年半年度报告摘要.pdf",
                "2023/000002_万科A2023年半年度报告(英文版).pdf", "2023/600109_国金证券二〇二三年第三季度报告.pdf",
                "2023/688981_中芯国际2023年年度报告.pdf", "2023/000004_ST国华2023年半年度财务报告.pdf"]


def entry(path, size=1000, kind="blob"):
    return {"Name": path.rsplit("/", 1)[-1], "Path": path, "Type": kind, "Size": size, "IsLFS": kind == "blob",
            "Sha256": "ab" * 32 if kind == "blob" else ""}


class FakeModelScope:
    """Serves dataset info and recursive tree pages; ``cap`` imitates the server's silent PageSize cap."""

    def __init__(self, datasets, cap=None, repeat_after=None, licence="Apache License 2.0"):
        self.datasets, self.cap, self.repeat_after, self.licence, self.urls = datasets, cap, repeat_after, licence, []

    def get(self, url, **kwargs):
        self.urls.append(url)
        parts = urlsplit(url)
        assert url.startswith(modelscope.API) and parts.scheme == "https"
        name = parts.path[len(urlsplit(modelscope.API).path):]
        dataset = name.removesuffix("/repo/tree")
        if dataset not in self.datasets:
            raise FetchError("http_404", url)
        if not name.endswith("/repo/tree"):
            body = {"Code": 200, "Data": {"License": self.licence, "StorageSize": 74437149633, "Name": dataset}}
            return None, json.dumps(body).encode()
        query = {k: v[0] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
        assert query["Recursive"] == "True" and query["Revision"] == "master" and "Root" in query
        size, page = int(query["PageSize"]), int(query["PageNumber"])
        if self.repeat_after and page > self.repeat_after:
            page = 1
        start, served = (page - 1) * size, min(size, self.cap or size)
        rows = [e for e in self.datasets[dataset] if e["Path"].startswith(query["Root"])]
        body = {"Code": 200, "Message": "success", "Data": {"Files": rows[start:start + served], "TotalCount": len(rows)}}
        return None, json.dumps(body, ensure_ascii=False).encode()


def repo(paths, extra=()):
    return [entry("allpdf", kind="tree"), entry(".gitattributes", 87), entry("README.md", 900),
            *[entry(p, 1000 + i) for i, p in enumerate(paths)], *extra]


def test_files_yields_one_candidate_per_document_with_parsed_metadata():
    api = FakeModelScope({modelscope.FINGLM: repo(FINGLM_FILES, [entry("train.csv", 50)])})
    messages = []
    found = list(modelscope.finglm(api, {"log": messages.append}))
    assert [c["meta"]["code"] for c in found] == ["000001", "600519", "002230"]
    first = found[0]
    assert first["key"] == f"modelscope:{modelscope.FINGLM}:{FINGLM_FILES[0]}"
    assert first["url"] == (f"{modelscope.API}{modelscope.FINGLM}/repo?Revision=master&FilePath="
                            + quote_plus(FINGLM_FILES[0]))
    assert first["title"] == "平安银行股份有限公司2019年年度报告" and first["type_hint"] == "pdf"
    assert first["meta"] == dict(dataset=modelscope.FINGLM, revision="master", path=FINGLM_FILES[0], size=1000,
                                 sha256="ab" * 32, lfs=True, date="2020-02-14", company="平安银行股份有限公司",
                                 code="000001", short_name="平安银行", year=2019, report_type="年度报告")
    assert found[2]["type_hint"] == "pdf"  # upper-case extension
    assert first["license"].startswith("issuer copyright") and "Apache License 2.0" in first["license"]
    assert "source" not in first and "terms" not in first and "headers" not in first
    assert api.urls[0] == modelscope.API + modelscope.FINGLM  # licence lookup first
    assert any("'kept': 3" in m and "'type': 3" in m for m in messages)  # .gitattributes, README.md, train.csv


def test_filters_types_exclude_glob_regex_and_size():
    api = FakeModelScope({modelscope.REPORTS_2023: repo(REPORT_FILES, [entry("2023/000005_ST星源路演.pptx", 10 ** 9)])})

    def paths(**options):
        return [c["meta"]["path"].split("/")[-1] for c in modelscope.reports_2023(api, options)]

    assert paths() == ["000001_平安银行2023年一季度报告.pdf", "600109_国金证券二〇二三年第三季度报告.pdf",
                       "688981_中芯国际2023年年度报告.pdf", "000004_ST国华2023年半年度财务报告.pdf",
                       "000005_ST星源路演.pptx"]  # summaries and English versions excluded by default
    assert len(paths(exclude="")) == 7
    assert paths(types="pptx") == ["000005_ST星源路演.pptx"]
    assert paths(max_size="1MB") == paths()[:-1] and paths(max_size=1003) == paths()[:2]
    assert paths(pattern="2023/6*") == ["600109_国金证券二〇二三年第三季度报告.pdf", "688981_中芯国际2023年年度报告.pdf"]
    assert paths(regex="季度", exclude="三季") == ["000001_平安银行2023年一季度报告.pdf"]
    assert paths(max_items=2) == paths()[:2]
    with pytest.raises(ValueError):
        paths(max_size="huge")


def test_max_size_defaults_to_the_fetchers_limit():
    api = FakeModelScope({modelscope.REPORTS_2023: repo(REPORT_FILES)})
    api.max_bytes = 1002
    assert [c["meta"]["size"] for c in modelscope.reports_2023(api, {"exclude": ""})] == [1000, 1001, 1002]


def test_parse_name_patterns():
    assert modelscope.parse_name("2023/600109_国金证券国金证券股份有限公司二〇二三年第一季度报告.pdf") == dict(
        code="600109", short_name="国金证券国金证券股份有限公司", year=2023, report_type="第一季度报告",
        title="国金证券国金证券股份有限公司二〇二三年第一季度报告")
    assert modelscope.parse_name("2023/000002_万科A2023年半年度报告摘要.pdf")["report_type"] == "半年度报告"
    assert modelscope.parse_name("2023/600100_同方股份2023第三季度报告.pdf")["year"] == 2023
    assert "report_type" not in modelscope.parse_name("2023/000004_ST国华2023年度财务决算报告.pdf")
    assert modelscope.parse_name("docs/某某白皮书.pdf") == {"title": "某某白皮书"}
    assert modelscope.parse_name("2023/000001_年度报告.pdf") == dict(code="000001", report_type="年度报告", title="年度报告")


def test_paging_runs_to_total_count_through_a_page_size_cap():
    paths = [f"2023/{i:06d}_公司{i}2023年年度报告.pdf" for i in range(2300)]
    # The server serves at most 700 rows per page whatever PageSize says: a short page is not the end.
    api = FakeModelScope({modelscope.REPORTS_2023: repo(paths)}, cap=700)
    assert len(list(modelscope.reports_2023(api, {"page_size": 700}))) == 2300
    pages = [parse_qs(urlsplit(u).query) for u in api.urls[1:]]
    assert [p["PageNumber"][0] for p in pages] == ["1", "2", "3", "4"] and {p["PageSize"][0] for p in pages} == {"700"}
    # Asking for more than the measured cap is clamped to 2,000.
    api = FakeModelScope({modelscope.REPORTS_2023: repo(paths)})
    assert len(list(modelscope.reports_2023(api, {"page_size": 5000}))) == 2300
    assert [parse_qs(urlsplit(u).query)["PageSize"][0] for u in api.urls[1:]] == ["2000", "2000"]
    # A server that wraps back to page 1 stops the listing with a log line instead of repeating rows.
    messages = []
    api = FakeModelScope({modelscope.REPORTS_2023: repo(paths)}, repeat_after=1)
    found = list(modelscope.reports_2023(api, {"page_size": 1000, "log": messages.append}))
    assert len(found) == 997 and any("stopped at 1000 of 2303" in m for m in messages)


def test_root_lists_one_directory():
    api = FakeModelScope({"someone/mixed": repo(["a/x.pdf", "b/y.docx", "b/z.pptx"])})
    found = list(modelscope.files(api, {"dataset": "someone/mixed", "root": "b"}))
    assert [c["type_hint"] for c in found] == ["docx", "pptx"]
    assert found[0]["license"] == "as declared by the repository: Apache License 2.0"
    assert parse_qs(urlsplit(api.urls[1]).query)["Root"] == ["b"]


def test_sample_is_deterministic_and_independent_of_listing_order():
    paths = [f"2023/{i:06d}_公司2023年年度报告.pdf" for i in range(4000)]

    def sampled(rows, **options):
        api = FakeModelScope({"x/y": rows})
        return {c["meta"]["path"] for c in modelscope.files(api, dict(options, dataset="x/y"))}

    tenth = sampled(repo(paths), sample=0.1)
    assert 300 < len(tenth) < 500
    assert sampled(repo(paths[::-1]), sample=0.1) == tenth
    assert sampled(repo(paths[:2000]), sample=0.1) == {p for p in tenth if p in set(paths[:2000])}
    assert sampled(repo(paths), sample=10) == tenth  # one in ten
    assert sampled(repo(paths), sample=0.1, seed=1) != tenth
    assert sampled(repo(paths), sample=1.0) == set(paths)


def test_errors_propagate():
    with pytest.raises(ValueError):
        list(modelscope.files(FakeModelScope({}), {}))
    with pytest.raises(ValueError):
        list(modelscope.files(FakeModelScope({}), {"dataset": "no-slash"}))
    with pytest.raises(FetchError) as error:
        list(modelscope.files(FakeModelScope({}), {"dataset": "nobody/missing"}))
    assert error.value.reason == "http_404"

    class Refusing:
        def __init__(self, body):
            self.body = body

        def get(self, url, **kwargs):
            return None, self.body

    for body, reason in ((b"<html>challenge</html>", "not_json"), (b'{"Code": 10010205001, "Message": "x"}',
                                                                      "api_10010205001")):
        with pytest.raises(FetchError) as error:
            list(modelscope.files(Refusing(body), {"dataset": "a/b"}))
        assert error.value.reason == reason


class Response:
    def __init__(self, status, body=b"", headers=None, url=""):
        self.status_code, self._body, self.headers, self.url = status, body, dict(headers or {}), url
        self.text = body.decode("utf-8", "replace")

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, size):
        yield self._body

    def close(self):
        pass


class Transport:
    """requests.Session stand-in: www serves robots.txt, the API and small files; LFS files redirect to a
    signed-URL CDN that refuses every unsigned path, robots.txt included (as measured)."""
    headers = {}
    CDN = "https://cdn-lfs-cn-1.modelscope.cn"

    def __init__(self, api):
        self.api, self.calls = api, []

    def request(self, method, url, **kwargs):
        self.calls.append(url)
        parts = urlsplit(url)
        if url == "https://www.modelscope.cn/robots.txt":
            return Response(200, b"User-agent: *\nDisallow: /datasets/*/*/files\nDisallow: /datasets/*/*/file/*\n")
        if url.startswith(self.CDN):
            if "auth_key=" not in parts.query:
                return Response(403, b"<html>403 Forbidden</html>",
                                {"X-Tengine-Error": "denied by req auth: no url arg auth_key"})
            return Response(200, b"%PDF-1.5\n%%EOF\n", url=url)
        if parts.path.endswith("/repo") and "FilePath=" in parts.query:
            path = parse_qs(parts.query)["FilePath"][0]
            if path.endswith(".md"):
                return Response(200, b"# readme", url=url)
            return Response(302, headers={"Location": f"{self.CDN}/prod/lfs-objects/ab/ab/x?filename=f&auth_key=1-2-0-3"})
        return Response(200, self.api.get(url)[1], url=url)


def test_requests_go_through_the_fetcher_and_lfs_downloads_stop_at_the_cdn_robots(tmp_path):
    transport = Transport(FakeModelScope({modelscope.FINGLM: repo(FINGLM_FILES[:1])}))
    fetcher = Fetcher(http=transport, interval=0, sleep=lambda s: None)
    found = list(modelscope.finglm(fetcher, {"types": "pdf,md", "exclude": ""}))
    assert transport.calls[0] == "https://www.modelscope.cn/robots.txt"
    assert [c["type_hint"] for c in found] == ["md", "pdf"]
    store = Store(tmp_path)
    counts = download(found, store, fetcher)
    # The README is served by www itself; the PDF's signed CDN URL is refused because the CDN's robots.txt
    # answers 403, which the Fetcher reads as disallow (see the module docstring).
    assert counts == {"ok": 1, "robots_disallow": 1}
    assert transport.calls.count(Transport.CDN + "/robots.txt") == 1
    assert not any("auth_key" in c for c in transport.calls)
