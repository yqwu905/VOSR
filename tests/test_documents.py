"""Document collection: robots.txt rules, content sniffing and the polite fetcher (offline)."""

import io
import zipfile

import pytest

from data_pipeline.documents.fetch import Fetcher, FetchError, Store, download
from data_pipeline.documents.robots import Robots
from data_pipeline.documents.sniff import cjk_ratio, sniff

AGENT = "VOSR-research-crawler/0.1"


def test_robots_wildcards_end_anchor_and_longest_match():
    rules = Robots("User-agent: *\nDisallow: /*.pdf$\nDisallow: /*?\nAllow: /open/*.pdf$\nDisallow: /private\n", AGENT)
    assert not rules.allowed("https://a.cn/docs/file.pdf")
    assert rules.allowed("https://a.cn/docs/file.pdf.html")  # $ anchors the end
    assert not rules.allowed("https://a.cn/list?page=2")
    assert rules.allowed("https://a.cn/open/notice.pdf")  # longer Allow beats shorter Disallow
    assert not rules.allowed("https://a.cn/private/x")
    assert rules.allowed("https://a.cn/")


def test_robots_group_selection_merging_and_crawl_delay():
    text = ("User-agent: Googlebot\nDisallow: /\n\n"
            "User-agent: vosr-research-crawler\nUser-agent: other\nDisallow: /tmp\nCrawl-delay: 5\n\n"
            "User-agent: *\nDisallow: /\n")
    rules = Robots(text, AGENT)
    assert rules.allowed("https://a.cn/doc.pdf") and not rules.allowed("https://a.cn/tmp/x")
    assert rules.crawl_delay == 5
    assert not Robots(text, "SomeoneElse/1.0").allowed("https://a.cn/doc.pdf")
    assert Robots("User-agent: *\nDisallow:\n", AGENT).allowed("https://a.cn/anything")


def test_robots_percent_encoding_and_allow_tie():
    rules = Robots("User-agent: *\nDisallow: /%E6%96%87%E4%BB%B6/\nAllow: /a\nDisallow: /a\n", AGENT)
    assert not rules.allowed("https://a.cn/文件/x.pdf")
    assert rules.allowed("https://a.cn/a/b")  # equal length: allow wins


def zip_bytes(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in names.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_sniff_by_content():
    assert sniff(b"%PDF-1.7\n...") == "pdf"
    assert sniff(zip_bytes({"[Content_Types].xml": "x", "ppt/presentation.xml": "x"})) == "pptx"
    assert sniff(zip_bytes({"[Content_Types].xml": "x", "word/document.xml": "x"})) == "docx"
    assert sniff(zip_bytes({"OFD.xml": "x"})) == "ofd"
    assert sniff(zip_bytes({"mimetype": "application/vnd.oasis.opendocument.presentation"})) == "odp"
    assert sniff(zip_bytes({"template/readme.txt": "x"})) == "zip"
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100
    assert sniff(ole + "PowerPoint Document".encode("utf-16-le")) == "ppt"
    assert sniff(ole + "WordDocument".encode("utf-16-le")) == "doc"
    assert sniff(b"\xef\xbb\xbf<!DOCTYPE html><html>") == "html"
    assert sniff(b"PK\x03\x04garbage") == "zip_broken"


def test_cjk_ratio():
    assert cjk_ratio("中华人民共和国 law") == pytest.approx(7 / 10)
    assert cjk_ratio("   ") == 0.0


class FakeResponse:
    def __init__(self, status, body=b"", headers=None, url=""):
        self.status_code, self._body, self.headers, self.url = status, body, headers or {}, url
        self.text = body.decode("utf-8", "replace")

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []
        self.headers = {}

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, **kwargs):
        self.calls.append(url)
        responses = self.routes.get(url, [FakeResponse(404)])
        return responses.pop(0) if len(responses) > 1 else responses[0]


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def fetcher(routes, **kwargs):
    clock = Clock()
    return Fetcher(http=FakeHTTP(routes), clock=clock, sleep=clock.sleep, **kwargs), clock


def test_fetcher_obeys_robots_paces_hosts_and_backs_off():
    routes = {"https://a.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nDisallow: /secret\nCrawl-delay: 3\n")],
              "https://a.cn/doc.pdf": [FakeResponse(503), FakeResponse(200, b"%PDF-1.4 x", url="https://a.cn/doc.pdf")],
              "https://a.cn/2.pdf": [FakeResponse(200, b"%PDF-1.4 y")]}
    f, clock = fetcher(routes, interval=1.0)
    _, body = f.get("https://a.cn/doc.pdf")
    assert body.startswith(b"%PDF")
    with pytest.raises(FetchError) as error:
        f.get("https://a.cn/secret/x.pdf")
    assert error.value.reason == "robots_disallow"
    f.get("https://a.cn/2.pdf")
    assert clock.now >= 3 * 3  # robots, retry and second file all wait for the Crawl-delay slot
    assert f.http.calls.count("https://a.cn/robots.txt") == 1


def test_robots_server_error_means_disallow_and_404_means_allow():
    f, _ = fetcher({"https://b.cn/robots.txt": [FakeResponse(500)], "https://c.cn/robots.txt": [FakeResponse(404)],
                    "https://c.cn/x.pdf": [FakeResponse(200, b"%PDF-1.4")]})
    with pytest.raises(FetchError):
        f.get("https://b.cn/x.pdf")
    assert f.get("https://c.cn/x.pdf")[1] == b"%PDF-1.4"


def test_redirects_are_checked_against_the_target_hosts_robots():
    routes = {"https://a.cn/robots.txt": [FakeResponse(404)],
              "https://a.cn/go": [FakeResponse(302, headers={"Location": "https://d.cn/private/x.pdf"})],
              "https://d.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nDisallow: /private\n")]}
    f, _ = fetcher(routes)
    with pytest.raises(FetchError) as error:
        f.get("https://a.cn/go")
    assert error.value.reason == "robots_disallow" and "https://d.cn/private/x.pdf" not in f.http.calls


def test_size_cap_and_forbidden_are_not_retried(tmp_path):
    routes = {"https://a.cn/robots.txt": [FakeResponse(404)],
              "https://a.cn/big.pdf": [FakeResponse(200, b"%PDF" + b"0" * 5000)],
              "https://a.cn/login.pdf": [FakeResponse(403)],
              "https://a.cn/ok.pdf": [FakeResponse(200, b"%PDF-1.4 ok")],
              "https://a.cn/page.pdf": [FakeResponse(200, b"<html>blocked</html>")]}
    f, _ = fetcher(routes, max_bytes=1000)
    store = Store(tmp_path)
    candidates = [{"url": "https://a.cn/" + n, "source": "t"} for n in ("big.pdf", "login.pdf", "ok.pdf", "page.pdf")]
    counts = download(candidates, store, f, accept={"pdf"}, log=None)
    assert counts == {"too_large": 1, "http_403": 1, "ok": 1, "unwanted_type": 1}
    assert f.http.calls.count("https://a.cn/login.pdf") == 1
    # Resume: a second pass skips every logged key.
    assert download(candidates, Store(tmp_path), f, log=None) == {}


def test_ai_crawler_optout_binds_us_and_content_signal():
    text = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: CCBot\nDisallow: /papers/\n\nUser-agent: *\nAllow: /\n"
    rules = Robots(text, AGENT)
    assert not rules.allowed("https://a.cn/x.pdf") and rules.reason("https://a.cn/x.pdf") == "ai_optout"
    partial = Robots("User-agent: CCBot\nDisallow: /papers/\n\nUser-agent: *\nAllow: /\n", AGENT)
    assert partial.allowed("https://a.cn/news/1.pdf") and not partial.allowed("https://a.cn/papers/1.pdf")
    # Sites that list AI bots with the same rules as everyone else stay open.
    same = Robots("User-agent: GPTBot\nUser-agent: *\nDisallow: /sd\n", AGENT)
    assert same.allowed("https://a.cn/doc.pdf") and not same.allowed("https://a.cn/sd/x")
    signal = Robots("User-agent: *\nContent-signal: search=yes, ai-train=no\nAllow: /\n", AGENT)
    assert not signal.allowed("https://a.cn/doc.pdf")


def test_robots_forbidden_disallows_and_transient_failure_is_rechecked():
    routes = {"https://w.cn/robots.txt": [FakeResponse(403)],
              "https://t.cn/robots.txt": [FakeResponse(503), FakeResponse(503), FakeResponse(503), FakeResponse(503),
                                          FakeResponse(404)],
              "https://t.cn/x.pdf": [FakeResponse(200, b"%PDF-1.4")]}
    f, clock = fetcher(routes, interval=1.0)
    with pytest.raises(FetchError):
        f.get("https://w.cn/x.pdf")
    with pytest.raises(FetchError):
        f.get("https://t.cn/x.pdf")  # four 503s: disallowed for now
    clock.now += 601
    assert f.get("https://t.cn/x.pdf")[1] == b"%PDF-1.4"  # rechecked after the hold


def test_declared_oversize_is_refused_before_reading_and_pdf_wrappers_are_stripped(tmp_path):
    routes = {"https://a.cn/robots.txt": [FakeResponse(404)],
              "https://a.cn/huge.pdf": [FakeResponse(200, b"%PDF-1.4", headers={"Content-Length": "999999"})]}
    f, _ = fetcher(routes, max_bytes=1000)
    with pytest.raises(FetchError) as error:
        f.get("https://a.cn/huge.pdf")
    assert error.value.reason == "too_large"
    wrapped = b"------abc\r\nContent-Disposition: form-data\r\n\r\n%PDF-1.5 body %%EOF\r\n------abc--\r\n"
    digest, kind, path = Store(tmp_path).put(wrapped)
    assert kind == "pdf" and path.read_bytes() == b"%PDF-1.5 body %%EOF"


class BrokenResponse(FakeResponse):
    def iter_content(self, size):
        yield self._body[:4]
        raise ConnectionError("IncompleteRead")


def test_connection_broken_mid_body_is_retried():
    routes = {"https://a.cn/robots.txt": [FakeResponse(404)],
              "https://a.cn/deck.pptx": [BrokenResponse(200, b"PK\x03\x04 data"), FakeResponse(200, b"PK\x03\x04 data")]}
    f, _ = fetcher(routes)
    assert f.get("https://a.cn/deck.pptx")[1] == b"PK\x03\x04 data"


def test_cc_data_host_needs_an_explicit_decision(tmp_path):
    routes = {"https://a.cn/robots.txt": [FakeResponse(404)],
              "https://a.cn/x.pdf": [FakeResponse(200, b"%PDF-1.4 origin")]}
    f, _ = fetcher(routes)
    warc = {"filename": "crawl-data/CC-MAIN-2024-33/x.warc.gz", "offset": 0, "length": 10}
    store = Store(tmp_path)
    counts = download([{"key": "a", "url": "https://a.cn/x.pdf", "warc": warc, "terms": "web_mixed"},
                       {"key": "b", "warc": warc, "terms": "web_mixed"}], store, f, log=None)
    assert counts == {"ok": 1, "cc_data_host_disabled": 1}
    assert store.seen["a"]["via"] == "origin"
    assert not any("commoncrawl" in url for url in f.http.calls)


def test_parquet_blob_candidates_read_each_row_group_once(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"id": [1, 2, 3], "blobs": [[b"en", b"%PDF-1.4 one"], [b"en", b"%PDF-1.4 two"], [b"en", None]]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=3)
    data = buffer.getvalue()

    class RangeHTTP(FakeHTTP):
        def request(self, method, url, headers=None, **kwargs):
            self.calls.append(url)
            if url.endswith("robots.txt"):
                return FakeResponse(404)
            start, end = (int(v) for v in headers["Range"].split("=")[1].split("-"))
            return FakeResponse(206, data[start:end + 1], {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, url)

    clock = Clock()
    f = Fetcher(http=RangeHTTP({}), clock=clock, sleep=clock.sleep)
    url = "https://hf.example/data.parquet"
    candidates = [{"key": str(i), "parquet": {"url": url, "row_group": 0, "row": i, "column": "blobs", "item": 1},
                   "terms": "official"} for i in range(3)]
    counts = download(candidates, Store(tmp_path), f, log=None)
    assert counts == {"ok": 2, "empty_cell": 1}
    footer_and_column_reads = len([c for c in f.http.calls if c == url])
    assert footer_and_column_reads <= 6  # one row-group read shared by all three candidates
