"""Document collection: robots.txt rules, content sniffing and the polite fetcher (offline)."""

import errno
import gzip
import io
import json
import logging
import os
import socket
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_pipeline.documents import fetch
from data_pipeline.documents.fetch import Fetcher, FetchError, Store, download, parquet_blob, session
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


def test_robots_byte_order_mark_and_wildcards_at_either_end():
    assert not Robots(chr(0xFEFF) + "User-agent: *\nDisallow: /\n", AGENT).allowed("https://a.cn/x.pdf")
    cases = {"/*": (["/"], []), "/a*": (["/a", "/abc"], ["/b"]),
             "/*.pdf$": (["/.pdf", "/d/x.pdf"], ["/x.pdf?y", "/x.pdfx"]),
             "/a*a$": (["/aa", "/axa"], ["/a", "/aab"]), "/a*b*$": (["/ab", "/axbyz"], ["/a"]),
             "/*a*a": (["/aa", "/xaya"], ["/a"]), "/a$b": (["/a$b"], ["/a"]), "/a$$": (["/a$"], ["/a"]),
             "/%61b$": (["/ab", "/%61b"], ["/abc"])}
    for pattern, (hits, misses) in cases.items():
        rules = Robots(f"User-agent: *\nDisallow: {pattern}\n", AGENT)
        assert [rules.allowed("https://a.cn" + p) for p in hits + misses] == [False] * len(hits) + [True] * len(misses)


def test_robots_wildcards_match_in_linear_time():
    rules = Robots("User-agent: *\nDisallow: /" + "*a" * 7 + "*b\n", AGENT)
    start = time.perf_counter()
    assert rules.allowed("https://a.cn/" + "a" * 60 + ".pdf")
    assert time.perf_counter() - start < 0.5  # a backtracking regex took ~20 s


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


def test_sniff_inflates_at_most_the_head_of_a_member():
    import tracemalloc
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        with archive.open("mimetype", "w") as member:  # 64 MiB in a 64 kB download
            member.write(b"application/vnd.oasis.opendocument.text")
            for _ in range(4):
                member.write(bytes(16 << 20))
    data = buffer.getvalue()
    tracemalloc.start()
    try:
        assert sniff(data) == "odt"
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(data) < 1 << 20 and peak < 4 << 20


def test_containers_are_sniffed_before_an_embedded_pdf_header(tmp_path):
    buffer = io.BytesIO()
    chapters = [b"%PDF-1.4 chapter one %%EOF", b"%PDF-1.4 chapter two %%EOF"]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for number, chapter in enumerate(chapters):
            archive.writestr(f"{number}.pdf", chapter)
    _, kind, path = Store(tmp_path).put(buffer.getvalue())
    assert kind == "zip" and path.read_bytes() == buffer.getvalue()  # kept whole for expansion, not spliced
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 500 + b"%PDF-1.4 embedded" + "WordDocument".encode("utf-16-le")
    assert sniff(ole) == "doc"
    wrapped = b"--boundary\r\nContent-Type: application/pdf\r\n\r\n%PDF-1.5 body %%EOF\r\n--boundary--\r\n"
    assert sniff(wrapped) == "pdf"


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


def public_dns(host, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def fetcher(routes, **kwargs):
    clock = Clock()
    kwargs.setdefault("resolve", public_dns)  # offline: every name resolves to a public address
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


def test_robots_txt_is_read_as_utf8_and_only_its_first_500_kib():
    raw = b"\xef\xbb\xbfUser-agent: *\nDisallow: /\n"
    bom = FakeResponse(200, raw, {"Content-Type": "text/plain"})
    bom.text = raw.decode("latin-1")  # what requests makes of text/plain without a charset
    filler = b"# " + b"x" * 1021 + b"\n"
    big = raw + filler * 499 + b"Allow: /cut" + b"t" * 2000 + b"\nAllow: /late\n" + filler * 1500
    served = []

    class Counted(FakeResponse):
        def iter_content(self, size):
            for chunk in super().iter_content(size):
                served.append(len(chunk))
                yield chunk

    f, _ = fetcher({"https://a.cn/robots.txt": [bom], "https://b.cn/robots.txt": [Counted(200, big)]})
    assert f.robots_for("https://a.cn/x.pdf").reason("https://a.cn/x.pdf") == "robots_disallow"
    rules = f.robots_for("https://b.cn/x.pdf")
    assert not rules.allowed("https://b.cn/late") and not rules.allowed("https://b.cn/cut" + "t" * 1000)
    assert sum(served) < (500 << 10) + (64 << 10)  # the rest of the 2 MB file is never read


def test_robots_txt_cut_keeps_cr_only_lines_and_a_declared_charset():
    cr = b"User-agent: *\rDisallow: /\r" + (b"# " + b"x" * 1021 + b"\r") * 700  # RFC 9309 allows CR alone
    rules = "User-agent: *\nDisallow: /文件/\n"
    routes = {"https://cr.cn/robots.txt": [FakeResponse(200, cr)],
              "https://gbk.cn/robots.txt": [FakeResponse(200, rules.encode("gbk"),
                                                         {"Content-Type": "text/plain; charset=gbk"})],
              "https://utf8.cn/robots.txt": [FakeResponse(200, rules.encode(),
                                                          {"Content-Type": "text/plain; charset=ISO-8859-1"})]}
    f, _ = fetcher(routes)
    assert not f.robots_for("https://cr.cn/doc.pdf").allowed("https://cr.cn/doc.pdf")
    for host in ("gbk.cn", "utf8.cn"):  # GBK as declared; valid UTF-8 stays UTF-8 whatever the header says
        url = f"https://{host}/%E6%96%87%E4%BB%B6/x.pdf"
        assert f.robots_for(url).reason(url) == "robots_disallow" and f.robots_for(url).allowed(f"https://{host}/x")


def test_robots_429_disallows_for_a_while_and_is_rechecked():
    routes = {"https://r.cn/robots.txt": [FakeResponse(429)] * 4 + [FakeResponse(200, b"User-agent: *\nDisallow: /\n")],
              "https://r.cn/x.pdf": [FakeResponse(200, b"%PDF-1.4")]}
    f, clock = fetcher(routes, interval=1.0)
    for reason in ("robots_unavailable", "robots_disallow"):  # a temporary verdict, then the real rules
        with pytest.raises(FetchError) as error:
            f.get("https://r.cn/x.pdf")
        assert error.value.reason == reason
        clock.now += 601
    assert f.http.calls.count("https://r.cn/robots.txt") == 5 and "https://r.cn/x.pdf" not in f.http.calls


def test_signed_url_cdn_refusing_robots_txt_means_unavailable(caplog):
    cdn = "https://cdn-lfs-cn-1.modelscope.cn"
    signed = cdn + "/prod/lfs-objects/ab/cd/ef?filename=a.pdf&auth_key=1-0-0-x"
    api = "https://www.modelscope.cn/api/v1/datasets/x/y/repo?FilePath=a.pdf"
    routes = {api: [FakeResponse(302, headers={"Location": signed})], signed: [FakeResponse(200, b"%PDF-1.5")],
              cdn + "/robots.txt": [FakeResponse(403, b"<html>403</html>",
                                                 {"X-Tengine-Error": "denied by req auth: no url arg auth_key"})],
              "https://waf.cn/robots.txt": [FakeResponse(403, b"<html>403</html>",
                                                         {"X-Tengine-Error": "denied by ACL"})]}
    f, _ = fetcher(routes)
    with caplog.at_level(logging.WARNING, logger="data_pipeline.documents.fetch"):
        assert f.get(api)[1] == b"%PDF-1.5"
    assert "signed-URL CDN" in caplog.text and cdn in caplog.text
    with pytest.raises(FetchError) as error:
        f.get("https://waf.cn/x.pdf")  # any other 401/403 on robots.txt still disallows
    assert error.value.reason == "robots_disallow"


def test_every_hop_is_refused_on_blocked_hosts_private_addresses_and_other_schemes(tmp_path, monkeypatch):
    names = {"intranet.example.cn": "10.1.2.3", "www.example.cn": "93.184.216.34"}

    def resolve(host, port):
        if host not in names:
            raise socket.gaierror(socket.EAI_NONAME, "unknown")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (names[host], 0))]

    hop = {"cnki": "https://kns.cnki.net/kcms/download.pdf", "meta": "http://169.254.169.254/latest/meta-data/",
           "file": "file:///etc/passwd", "intranet": "http://intranet.example.cn/x.pdf",
           "cnki_dot": "https://kns.cnki.net./kcms/download.pdf",  # the root dot
           "backslash": "http://kns.cnki.net\\@www.example.cn/kcms/download.pdf"}  # urllib3 would go to cnki
    routes = {f"https://www.example.cn/{name}": [FakeResponse(302, headers={"Location": target})]
              for name, target in hop.items()}
    routes.update({"https://www.example.cn/ok.pdf": [FakeResponse(200, b"%PDF-1.4")],
                   "https://unresolved.gov.cn/ok.pdf": [FakeResponse(200, b"%PDF-1.4")],
                   "https://moved.cn/robots.txt": [FakeResponse(301, headers={"Location": "http://10.0.0.1/r.txt"})]})
    f, _ = fetcher(routes, resolve=resolve)
    expected = {"https://www.example.cn/cnki": "blocked_host", "https://www.example.cn/meta": "private_address",
                "https://www.example.cn/file": "unsupported_scheme",
                "https://www.example.cn/intranet": "private_address",
                "http://127.0.0.1:8080/admin": "private_address", "http://[::ffff:127.0.0.1]/": "private_address",
                "http://2130706433/": "private_address", "http://localhost/x": "private_address",
                "http://metadata.google.internal/": "private_address", "http://224.0.0.1/": "private_address",
                "ftp://www.example.cn/x.pdf": "unsupported_scheme", "https://moved.cn/x.pdf": "robots_disallow",
                "https://www.example.cn/cnki_dot": "blocked_host", "https://www.example.cn/backslash": "bad_url",
                "http://127.0.0.1:8080\\@www.example.cn/admin": "bad_url", "http://x.kns%2ecnki.net/a.pdf": "bad_url",
                "https://ｋｎｓ。cnki。net/a.pdf": "blocked_host"}
    store = Store(tmp_path)
    counts = download([{"key": url, "url": url} for url in expected], store, f, log=None)
    assert {url: store.seen[url]["reason"] for url in expected} == expected and sum(counts.values()) == len(expected)
    # Unresolvable names that look public pass (the egress proxy resolves them); nothing else was contacted.
    assert f.get("https://www.example.cn/ok.pdf")[1] == f.get("https://unresolved.gov.cn/ok.pdf")[1] == b"%PDF-1.4"
    assert all(url.startswith(("https://www.example.cn/", "https://unresolved.gov.cn/", "https://moved.cn/robots.txt"))
               for url in f.http.calls)
    # The curl fallback transport gets the same checks; local servers need allow_private (tests only).
    sent = []
    monkeypatch.setattr(fetch, "curl_request",
                        lambda method, url, **kwargs: sent.append(url) or FakeResponse(200, b"%PDF"))
    f.curl_hosts.add("127.0.0.1:9000")
    with pytest.raises(FetchError, match="private_address"):
        f.get("http://127.0.0.1:9000/x.pdf")
    local, _ = fetcher({"http://127.0.0.1:9000/x.pdf": [FakeResponse(200, b"%PDF")]}, allow_private=True)
    local.curl_hosts.add("127.0.0.1:9000")
    assert local.get("http://127.0.0.1:9000/x.pdf")[1] == b"%PDF" and sent == ["http://127.0.0.1:9000/robots.txt",
                                                                             "http://127.0.0.1:9000/x.pdf"]


def warc_bytes(url, payload):
    from warcio.statusandheaders import StatusAndHeaders
    from warcio.warcwriter import WARCWriter
    out = io.BytesIO()
    writer = WARCWriter(out, gzip=True)
    headers = StatusAndHeaders("200 OK", [("Content-Type", "application/pdf")], protocol="HTTP/1.1")
    writer.write_record(writer.create_warc_record(url, "response", payload=io.BytesIO(payload), http_headers=headers))
    return out.getvalue()


def test_warc_reads_obey_the_base_and_origin_robots_and_the_cc_host_gate(tmp_path):
    hf, mirror, cc = ("https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/", "https://mirror.example.org/",
                      "http://DATA.commoncrawl.org:80/")
    name = "crawl-data/x.warc.gz"
    records = {url: warc_bytes(url, b"%PDF-1.4 archived") for url in ("https://news.cn/a.pdf", "https://open.cn/a.pdf")}
    optout = b"User-agent: GPTBot\nUser-agent: CCBot\nDisallow: /\n"
    routes = {"https://news.cn/robots.txt": [FakeResponse(200, optout)],
              "https://open.cn/robots.txt": [FakeResponse(404)],
              "https://open.cn/a.pdf": [FakeResponse(200, b"%PDF-1.4 live")],
              mirror + "robots.txt": [FakeResponse(200, b"User-agent: *\nDisallow: /\n")]}
    for base, origin in ((hf, "https://open.cn/a.pdf"), (hf + "news/", "https://news.cn/a.pdf"),
                         (mirror, "https://open.cn/a.pdf")):
        size = len(records[origin])
        routes[base + name] = [FakeResponse(206, records[origin], {"Content-Range": f"bytes 0-{size - 1}/{size}"})]
    f, _ = fetcher(routes)

    def warc(base, url="https://open.cn/a.pdf"):
        return {"filename": name, "offset": 0, "length": len(records[url]), "base": base}

    candidates = [{"key": "open", "url": "https://open.cn/a.pdf", "warc": warc(hf)},
                  {"key": "optout", "url": "https://news.cn/a.pdf",
                   "warc": warc(hf + "news/", "https://news.cn/a.pdf")},
                  {"key": "no_url", "warc": warc(hf + "news/", "https://news.cn/a.pdf")},
                  {"key": "mirror", "url": "https://open.cn/a.pdf", "warc": warc(mirror)},
                  {"key": "cc", "url": "https://open.cn/a.pdf", "warc": warc(cc)}]
    store = Store(tmp_path)
    download(candidates, store, f, log=None)
    assert {key: entry.get("reason", entry["status"]) for key, entry in store.seen.items()} == {
        "open": "ok", "optout": "ai_optout", "no_url": "ai_optout", "mirror": "robots_disallow", "cc": "ok"}
    assert store.seen["cc"]["via"] == "origin" and not any("data.commoncrawl" in url.lower() for url in f.http.calls)
    assert f.http.calls.count(hf + "news/" + name) == 1  # only the url-less candidate was read, then refused
    assert mirror + name not in f.http.calls and "https://huggingface.co/robots.txt" in f.http.calls


FAKE_CURL = """#!{python}
import json, pathlib, sys
here = pathlib.Path(__file__).parent
args = sys.argv[1:]
with open(here / "calls.jsonl", "a") as calls:
    calls.write(json.dumps(args) + "\\n")
status, segments = json.loads((here / "routes.json").read_text()).get(args[-1], ["404 Not Found", []])
with open(args[args.index("-D") + 1], "w") as head:
    head.write("HTTP/1.1 " + status + "\\r\\n\\r\\n")
out = open(args[args.index("-o") + 1], "wb") if "-o" in args else sys.stdout.buffer
for text, repeats in segments:
    for _ in range(repeats):
        out.write(text.encode())
        out.flush()
        with open(here / "written", "a") as written:
            written.write(str(len(text)) + "\\n")
"""


def fake_curl(tmp_path, monkeypatch, routes):
    """A curl on PATH serving {url: [status line, [[text, repeats], ...]]}; returns a reader of its argument lists."""
    script = tmp_path / "curl"
    script.write_text(FAKE_CURL.format(python=sys.executable))
    script.chmod(0o755)
    (tmp_path / "routes.json").write_text(json.dumps(routes))
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return lambda: [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_curl_transport_never_retries_by_itself(tmp_path, monkeypatch):
    commands = fake_curl(tmp_path, monkeypatch, {"https://slow.cn/x.pdf": ["503 Busy", []]})
    f, clock = fetcher({}, interval=5.0, retries=1)
    f.curl_hosts.add("slow.cn")
    with pytest.raises(FetchError, match="http_503"):
        f.get("https://slow.cn/x.pdf")
    assert len(commands()) == 3 and clock.now >= 2 * 5.0  # robots.txt, then one paced retry of the document
    for command in commands():
        assert command[command.index("--retry") + 1] == "0" and "--retry-all-errors" not in command
        assert command[command.index("--proto") + 1] == "=http,https"


def test_curl_transport_stops_reading_robots_txt_at_500_kib(tmp_path, monkeypatch):
    filler = ["#" * 1023 + "\n", 2000]  # 2 MB of comments
    commands = fake_curl(tmp_path, monkeypatch, {
        "https://big.cn/robots.txt": ["200 OK", [["User-agent: *\nDisallow: /private/\n", 1], filler,
                                                 ["Disallow: /late/\n", 1]]],
        "https://big.cn/big.pdf": ["200 OK", [["%PDF-1.4\n", 1], filler]]})
    f, _ = fetcher({}, max_bytes=1 << 20)  # robots.txt larger than max_bytes is still read up to the cut
    f.curl_hosts.add("big.cn")
    rules = f.robots_for("https://big.cn/x.pdf")
    assert not rules.allowed("https://big.cn/private/x.pdf") and rules.allowed("https://big.cn/late/x.pdf")
    assert "--max-filesize" not in commands()[0]
    written = len((tmp_path / "written").read_text().split())
    assert written < 700  # curl was stopped after 500 KiB (plus a pipe buffer), not left to send 2 MB
    with pytest.raises(FetchError, match="too_large"):
        f.get("https://big.cn/big.pdf")
    assert len((tmp_path / "written").read_text().split()) - written < 1300


def test_retry_after_is_capped_and_bodies_have_a_deadline():
    class Drip(FakeResponse):
        def iter_content(self, size):
            for i in range(len(self._body)):
                clock.now += 1.5
                yield self._body[i:i + 1]

    class Steady(FakeResponse):  # 64 KiB every 8 s: slow, but faster than the 4 KiB/s floor
        def iter_content(self, size):
            for chunk in super().iter_content(size):
                clock.now += 8
                yield chunk

    routes = {"https://a.cn/year.pdf": [FakeResponse(503, headers={"Retry-After": "31536000"})],
              "https://a.cn/soon.pdf": [FakeResponse(429, headers={"Retry-After": "30"}),
                                        FakeResponse(200, b"%PDF-1.4")],
              "https://a.cn/drip.pdf": [Drip(200, b"%PDF-1.4" * 4)],
              "https://a.cn/big.pdf": [Steady(200, b"%PDF" * (1 << 19))]}
    f, clock = fetcher(routes, timeout=2)
    with pytest.raises(FetchError, match="http_503"):
        f.get("https://a.cn/year.pdf")
    assert f.http.calls.count("https://a.cn/year.pdf") == 1 and max(clock.slept, default=0) <= 600
    assert f.get("https://a.cn/soon.pdf")[1] == b"%PDF-1.4" and 30 in clock.slept
    start = clock.now
    with pytest.raises(FetchError, match="too_slow"):
        f.get("https://a.cn/drip.pdf")
    assert clock.now - start <= 2 + 2 * 1.5 and f.http.calls.count("https://a.cn/drip.pdf") == 1  # not retried
    assert len(f.get("https://a.cn/big.pdf")[1]) == 2 << 20


def test_http_date_retry_after_crawl_delay_and_backoff_are_capped():
    pdf = FakeResponse(200, b"%PDF-1.4")
    routes = {"https://a.cn/2099.pdf": [FakeResponse(503, headers={"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})],
              "https://a.cn/2015.pdf": [FakeResponse(503, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
                                        pdf],
              "https://year.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nCrawl-delay: 31536000\n")],
              "https://inf.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nCrawl-delay: inf\n")],
              "https://nan.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nCrawl-delay: nan\n")],
              "https://nan.cn/x.pdf": [pdf],
              "https://busy.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nCrawl-delay: 500\n")],
              "https://busy.cn/x.pdf": [FakeResponse(503)] * 3 + [pdf]}
    f, clock = fetcher(routes, interval=1.0)
    with pytest.raises(FetchError, match="http_503"):  # an HTTP-date beyond the cap, like delay-seconds
        f.get("https://a.cn/2099.pdf")
    assert f.http.calls.count("https://a.cn/2099.pdf") == 1 and max(clock.slept) <= 1
    assert f.get("https://a.cn/2015.pdf")[1] == b"%PDF-1.4" and 0.0 in clock.slept
    for host in ("year", "inf"):
        with pytest.raises(FetchError, match="crawl_delay_too_long"):
            f.get(f"https://{host}.cn/x.pdf")
        assert f"https://{host}.cn/x.pdf" not in f.http.calls
    assert f.get("https://nan.cn/x.pdf")[1] == b"%PDF-1.4" and max(clock.slept) <= 1
    assert f.get("https://busy.cn/x.pdf")[1] == b"%PDF-1.4"
    assert max(clock.slept) <= fetch.MAX_WAIT and clock.slept.count(fetch.MAX_WAIT) == 3  # backoff capped


class LocalServer(BaseHTTPRequestHandler):
    """404 for everything; with ``server.drip`` the status line and headers come one byte every 50 ms."""
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.hits.append(self.path)
        if not self.server.drip:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 26\r\nX-Pad: " + b"p" * 40 + b"\r\n\r\n":
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"User-agent: *\nDisallow: /\n")
        except OSError:
            pass
        self.close_connection = True

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalServer)
    server.hits, server.drip = [], False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_session_bounds_the_status_line_and_header_phase(local_server):
    local_server.drip = True
    http = session()
    http.trust_env = False
    f = Fetcher(http=http, interval=0, timeout=1, retries=0, allow_private=True)
    for path, reason in (("/x.pdf", "too_slow"), ("/y.pdf", "robots_unavailable")):  # not the Disallow read late
        start = time.monotonic()
        with pytest.raises(FetchError, match=reason):
            f.get(f"http://127.0.0.1:{local_server.server_port}{path}", check_robots=path == "/y.pdf")
        assert time.monotonic() - start < 2.5  # the headers take 3.5 s to drip in


def test_session_refuses_a_name_that_resolves_to_loopback_at_connect_time(local_server, monkeypatch):
    real = socket.getaddrinfo  # the name passed the check (public_dns), then resolves to 127.0.0.1: DNS rebinding
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kwargs: real(
        "127.0.0.1" if host == "rebind.example.com" else host, *args, **kwargs))
    f = Fetcher(http=session(), interval=0, timeout=2, retries=0, resolve=public_dns)
    f.http.trust_env = False
    with pytest.raises(FetchError, match="private_address"):
        f.get(f"http://rebind.example.com:{local_server.server_port}/admin", check_robots=False)
    assert local_server.hits == []


def test_slow_drip_over_a_real_socket_hits_the_deadline():
    text = b"%PDF-1.4 " * 4000

    class Server(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/robots.txt":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/gz":  # chunked and gzipped: decoded as it arrives
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                packed = gzip.compress(text)
                for i in range(0, len(packed), 1000):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(packed[i:i + 1000]), packed[i:i + 1000]))
                self.wfile.write(b"0\r\n\r\n")
                return
            self.send_response(200)
            self.send_header("Content-Length", "20")
            self.end_headers()
            try:
                for _ in range(20):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.5)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    http = session()
    http.trust_env = False
    f = Fetcher(http=http, interval=0, timeout=1, retries=0, allow_private=True)
    try:
        assert f.get(f"http://127.0.0.1:{server.server_port}/gz")[1] == text
        start = time.monotonic()
        with pytest.raises(FetchError, match="too_slow"):
            f.get(f"http://127.0.0.1:{server.server_port}/drip")
        assert time.monotonic() - start < 4  # the whole drip takes 10 s
    finally:
        server.shutdown()


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


class RangeServer(FakeHTTP):
    """One parquet file served with Range support; reads are logged as (start, end)."""

    def __init__(self, data):
        super().__init__({})
        self.data, self.reads = data, []

    def request(self, method, url, headers=None, **kwargs):
        if url.endswith("robots.txt"):
            return FakeResponse(404)
        start, end = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        self.reads.append((start, end))
        body = self.data[start:end + 1]
        return FakeResponse(206, body, {"Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(self.data)}"}, url)


def blob_file(rows, **options):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"id": list(range(len(rows))), "blobs": pa.array(rows, pa.list_(pa.large_binary(), 3)),
                      "note": ["x"] * len(rows)})
    buffer = io.BytesIO()
    pq.write_table(table, buffer, **dict(dict(row_group_size=20, write_page_index=True, data_page_size=3000,
                                              write_batch_size=3, dictionary_pagesize_limit=5000), **options))
    return buffer.getvalue()


def read_cells(data, wanted, item=1, max_bytes=200 << 20):
    clock = Clock()
    server = RangeServer(data)
    f = Fetcher(http=server, clock=clock, sleep=clock.sleep, max_bytes=max_bytes)
    cache, cells = {}, {}
    for row in wanted:
        reference = {"url": "https://hf.example/d.parquet", "row_group": row // 20, "row": row % 20, "column": "blobs",
                     "item": item, "rows": [r % 20 for r in wanted if r // 20 == row // 20]}
        try:
            cells[row] = parquet_blob(f, reference, cache)
        except FetchError as error:
            cells[row] = error.reason
    return cells, server.reads


def test_parquet_blob_reads_only_the_pages_of_wanted_rows():
    import random
    import pyarrow.parquet as pq
    rng = random.Random(0)
    rows = [[None if rng.random() < 0.25 else rng.randbytes(rng.choice([7, 900, 2500])) for _ in range(3)]
            for _ in range(40)]
    rows[1][1] = rows[0][1] = b"same blob"  # dictionary-encoded values at the start of the group
    rows[33] = [b"a", rng.randbytes(40000), b"c"]  # a page of its own
    expected = {r: rows[r][1] or "empty_cell" for r in range(40)}
    wanted = [0, 1, 5, 6, 21, 30, 33]
    variants = [{}, {"compression": "zstd"}, {"compression": "none", "use_dictionary": False}, {"version": "1.0"},
                {"write_page_index": False}, {"data_page_version": "2.0"}]  # the last two: whole-chunk reads
    for options in variants:
        data = blob_file(rows, **options)
        cells, reads = read_cells(data, wanted)
        assert cells == {r: expected[r] for r in wanted}, options
        metadata = pq.read_metadata(io.BytesIO(data))
        chunks = sum(metadata.row_group(g).column(1).total_compressed_size for g in range(2))
        body = sum(end - start + 1 for start, end in reads if 0 < end < len(data) - 1)  # not the footer
        assert (body < chunks * 0.6) == (options not in variants[-2:]), options
    data = blob_file(rows)
    _, reads = read_cells(data, [30])
    assert sum(end - start + 1 for start, end in reads if 0 < end < len(data) - 1) < chunks / 10
    assert read_cells(data, [0, 1], item=2)[0] == {0: rows[0][2] or "empty_cell", 1: rows[1][2] or "empty_cell"}
    # A page over the size limit fails its rows without being requested; the others are read.
    cells, reads = read_cells(data, wanted, max_bytes=30000)
    assert cells[33] == "too_large" and all(cells[r] == expected[r] for r in wanted if r != 33)
    assert not any(end - start > 30000 for start, end in reads)
    cells, reads = read_cells(blob_file(rows, write_page_index=False), wanted, max_bytes=30000)
    assert set(cells.values()) == {"too_large"} and not any(end - start > 30000 for start, end in reads)


def test_parquet_blob_turns_malformed_pages_into_fetch_errors():
    import random
    from data_pipeline.documents.fetch import _hybrid, _thrift

    def varint(n):
        out = bytearray()
        while True:
            out.append(n & 0x7F | (0x80 if n >> 7 else 0))
            n >>= 7
            if not n:
                return bytes(out)

    # Runs longer than the values a page still needs are refused before anything is allocated.
    for data in (varint((1 << 22) << 1 | 1) + b"\0" * 16, varint((1 << 45) << 1) + b"\1", varint(0) + b"\1"):
        with pytest.raises(ValueError):
            _hybrid(data, 0, 1, 1)
    assert _hybrid(varint(1 << 1 | 1) + b"\x05", 0, 1, 3) == [1, 0, 1]  # a padded last group is fine
    rng = random.Random(1)
    rows = [[rng.randbytes(rng.choice([7, 900])) for _ in range(3)] for _ in range(40)]
    data = bytearray(blob_file(rows, compression="none"))
    footer = _thrift(bytes(data[-8 - int.from_bytes(data[-8:-4], "little"):]))[0]
    chunk = footer[4][0][1][1]
    page = _thrift(bytes(data[chunk[4]:chunk[4] + chunk[5]]))[0][1][1]  # the second data page of group 0
    _, payload = _thrift(bytes(data), page[1])
    data[payload + 4:payload + 11] = varint((1 << 45) << 1)  # its repetition levels claim 2^45 values
    cells, _ = read_cells(bytes(data), [page[3]])
    assert cells == {page[3]: "parquet_ValueError"}
    # Random corruption anywhere before the footer: bytes or a FetchError, never another exception.
    clean = blob_file(rows, compression="snappy")
    end = len(clean) - 8 - int.from_bytes(clean[-8:-4], "little")
    for seed in range(150):
        broken = bytearray(clean)
        for _ in range(3):
            broken[rng.randrange(4, end)] = rng.choice([0, 0x7F, 0x80, 0xFF, rng.randrange(256)])
        cells, _ = read_cells(bytes(broken), rng.sample(range(40), 4), max_bytes=1 << 20)
        assert all(isinstance(v, (bytes, str)) for v in cells.values())


def test_fetch_log_splits_only_on_newlines_and_survives_a_cut_last_record(tmp_path):
    store = Store(tmp_path)
    title = "关于印发" + chr(0x2028) + "通知" + chr(0x85) + "续" + chr(0x2029)
    store.record({"key": "a", "status": "ok", "title": title})
    assert Store(tmp_path).seen["a"]["title"] == title
    with store.log_path.open("ab") as handle:
        handle.write('{"key": "b", "status": "ok", "title": "截'.encode()[:-1])  # a crash mid-record
    reopened = Store(tmp_path)
    assert set(reopened.seen) == {"a"} and reopened.dropped == 1
    reopened.record({"key": "c", "status": "ok"})
    assert set(Store(tmp_path).seen) == {"a", "c"} and Store(tmp_path).dropped == 0


def test_one_candidates_unexpected_failure_does_not_end_the_run(tmp_path):
    base = "https://huggingface.co/buckets/commoncrawl/commoncrawl/resolve/"
    junk = b"<html>temporarily unavailable</html>"
    routes = {base + "crawl-data/bad.warc.gz": [FakeResponse(206, junk, {"Content-Range": "bytes 0-35/9999"})],
              "https://a.cn/good.pdf": [FakeResponse(200, b"%PDF-1.4 ok")]}
    f, _ = fetcher(routes)
    candidates = [{"key": "bad", "url": "https://a.cn/x.pdf",
                   "warc": {"filename": "crawl-data/bad.warc.gz", "offset": 0, "length": len(junk), "base": base}},
                  {"key": "good", "url": "https://a.cn/good.pdf"}]
    store, messages = Store(tmp_path), []
    assert download(candidates, store, f, log=messages.append) == {"exception:ArchiveLoadFailed": 1, "ok": 1}
    assert store.seen["bad"]["status"] == "error" and any("ArchiveLoadFailed" in m for m in messages)

    class Interrupted(FakeHTTP):
        def request(self, method, url, **kwargs):
            raise KeyboardInterrupt

    stopped, _ = fetcher({})
    stopped.http = Interrupted({})
    with pytest.raises(KeyboardInterrupt):
        download([{"key": "x", "url": "https://b.cn/x.pdf"}], Store(tmp_path / "2"), stopped, log=None)


def test_a_store_that_cannot_write_ends_the_run_and_keyless_candidates_are_skipped(tmp_path):
    routes = {f"https://a.cn/{i}.pdf": [FakeResponse(200, b"%%PDF-1.4 %d" % i)] for i in range(3)}
    candidates = [{"key": str(i), "url": f"https://a.cn/{i}.pdf"} for i in range(3)]
    f, _ = fetcher(routes)

    class Full(Store):
        def put(self, data):
            raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(OSError):  # not "exception:OSError" logged for every candidate
        download(candidates, Full(tmp_path), f, log=None)
    messages = []
    malformed = [{"title": "no key, no url"}, {"key": ["a list"], "url": "https://a.cn/0.pdf"}, "not a dict"]
    counts = download(malformed + candidates, Store(tmp_path), f, log=messages.append)
    assert counts == {"invalid_candidate": 3, "ok": 3}
    assert sum("without a key" in message for message in messages) == 3


def test_resume_retries_transient_failures_only(tmp_path):
    class Flaky(FakeHTTP):
        def request(self, method, url, **kwargs):
            if url == "https://net.cn/x.pdf" and self.down:
                self.calls.append(url)
                raise ConnectionError("reset")
            return super().request(method, url, **kwargs)

    class Drip(FakeResponse):
        def iter_content(self, size):
            for i in range(len(self._body)):
                clock.now += 100
                yield self._body[i:i + 1]

    pdf = [FakeResponse(200, b"%PDF-1.4 ok")]
    routes = {"https://down.cn/robots.txt": [FakeResponse(503)] * 4 + [FakeResponse(404)],
              "https://drip.cn/x.pdf": [Drip(200, b"%PDF-1.4 ok")] + pdf,
              "https://busy.cn/x.pdf": [FakeResponse(503)] * 4 + pdf,
              "https://slow.cn/x.pdf": [FakeResponse(429)] * 4 + pdf,
              "https://shut.cn/robots.txt": [FakeResponse(200, b"User-agent: *\nDisallow: /\n")],
              "https://ai.cn/robots.txt": [FakeResponse(200, b"User-agent: GPTBot\nDisallow: /\n")],
              "https://down.cn/x.pdf": pdf, "https://net.cn/x.pdf": pdf, "https://gone.cn/x.pdf": [FakeResponse(404)]}
    f, clock = fetcher(routes, retries=3)
    f.http = Flaky(routes)
    f.http.down = True
    hosts = ("down", "busy", "slow", "net", "drip", "shut", "ai", "gone")
    candidates = [{"key": host, "url": f"https://{host}.cn/x.pdf"} for host in hosts]
    first = download(candidates, Store(tmp_path), f, log=None)
    assert first == {"robots_unavailable": 1, "http_503": 1, "http_429": 1, "network:ConnectionError": 1,
                     "too_slow": 1, "robots_disallow": 1, "ai_optout": 1, "http_404": 1}
    f.http.down = False
    again, _ = fetcher({})
    again.http = f.http
    calls = len(f.http.calls)
    assert download(candidates, Store(tmp_path), again, log=None) == {"ok": 5}
    retried = {url.split("/")[2].split(".")[0] for url in f.http.calls[calls:] if url.endswith("/x.pdf")}
    assert retried == {"down", "busy", "slow", "net", "drip"}
    assert download(candidates, Store(tmp_path), again, log=None) == {}


def test_zip_member_candidates_fetch_each_archive_once_and_fall_back_to_their_url(tmp_path):
    def pack(members, compression=zipfile.ZIP_DEFLATED):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression) as archive:
            for name, body in members.items():
                archive.writestr(name, body)
        return buffer.getvalue()

    one, two = b"%PDF-1.4 one " + b"x" * 100, b"%PDF-1.4 two " + b"y" * 100
    pack_url, broken_url, gone_url = "https://g.tw/pack.zip", "https://g.tw/broken.zip", "https://g.tw/gone.zip"
    routes = {pack_url: [FakeResponse(200, pack({"d/1.pdf": one, "d/2.pdf": two, "d/big.pdf": b"%PDF" + bytes(5000)}))],
              broken_url: [FakeResponse(200, b"<html>Just a moment...</html>")],
              "https://g.tw/robots.txt": [FakeResponse(200, b"User-agent: *\nDisallow: /private/\n")]}
    for name in ("3", "big", "4", "5"):
        routes[f"https://g.tw/{name}/Eg.pdf"] = [FakeResponse(200, b"%PDF-1.4 live " + name.encode())]
    f, _ = fetcher(routes, max_bytes=1000)

    def item(name, url=pack_url, member=None, live=True):
        candidate = {"key": name, "zip": {"url": url, "member": member or f"d/{name}.pdf"}}
        return dict(candidate, url=f"https://g.tw/{name}/Eg.pdf") if live else candidate

    candidates = [item("1"), item("2"), item("3"), item("big"), item("7", live=False),
                  dict(item("8"), url="https://g.tw/private/Eg.pdf"), item("4", broken_url), item("5", broken_url),
                  item("6", gone_url, live=False)]
    store = Store(tmp_path)
    counts = download(candidates, store, f, log=None)
    assert counts == {"ok": 6, "http_404": 1, "zip_member_missing": 1, "robots_disallow": 1}
    seen = store.seen
    assert [seen[k].get("via") for k in ("1", "2", "3", "big", "4")] == ["zip", "zip", None, None, None]
    assert open(seen["1"]["path"], "rb").read() == one and open(seen["2"]["path"], "rb").read() == two
    assert {k: seen[k].get("zip_error") for k in ("3", "big", "4")} == {"3": "zip_member_missing", "big": "too_large",
                                                                        "4": "zip_BadZipFile"}
    assert all("zip" not in entry for entry in seen.values())
    assert f.http.calls.count(pack_url) == f.http.calls.count(broken_url) == f.http.calls.count(gone_url) == 1
    assert "https://g.tw/1/Eg.pdf" not in f.http.calls and "https://g.tw/private/Eg.pdf" not in f.http.calls
