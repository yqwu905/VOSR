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
