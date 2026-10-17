"""Crawl corpora (offline): FinePDFs row-group sampling and the Common Crawl columnar index, read
through the real Fetcher from a fake server that honours Range requests."""

import gzip
import io
import json
import re
import socket

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.documents import catalog, commoncrawl
from data_pipeline.documents.fetch import CC_BASE, Fetcher, FetchError


class Response:
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


class Server:
    """Files with Range support, plain pages, redirects (a list: one Location per request, the last
    one kept) and refused URLs; every request is logged as (url, Range)."""

    def __init__(self, files=None, pages=None, redirects=None):
        self.files, self.pages, self.redirects = files or {}, pages or {}, redirects or {}
        self.refused, self.calls, self.headers = set(), [], {}

    def request(self, method, url, headers=None, **kwargs):
        span = (headers or {}).get("Range")
        self.calls.append((url, span))
        if url in self.refused:
            return Response(403, url=url)
        if url in self.redirects:
            targets = self.redirects[url]
            return Response(302, headers={"Location": targets.pop(0) if len(targets) > 1 else targets[0]})
        if url in self.files:
            data = self.files[url]
            found = re.fullmatch(r"bytes=(\d*)-(\d*)", span or "")
            if not found:
                return Response(200, data, url=url)
            if not found.group(1):
                start, end = max(0, len(data) - int(found.group(2))), len(data)
            else:
                start = int(found.group(1))
                end = min(len(data), int(found.group(2)) + 1) if found.group(2) else len(data)
            return Response(206, data[start:end], {"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"}, url)
        if url in self.pages:
            page = self.pages[url]
            body, extra = page if isinstance(page, tuple) else (page, {})
            return Response(200, body.encode() if isinstance(body, str) else body, extra, url)
        return Response(404, url=url)

    def ranges(self, url):
        return [span for called, span in self.calls if called == url]

    def urls(self):
        return [url for url, _ in self.calls]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def fetcher(server):
    clock = Clock()
    public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
    return Fetcher(http=server, interval=1.0, clock=clock, sleep=clock.sleep, resolve=lambda host, port: public)


def parquet_bytes(columns, row_group_size, schema=None):
    buffer = io.BytesIO()
    pq.write_table(pa.table(columns, schema=schema), buffer, row_group_size=row_group_size, compression="zstd")
    return buffer.getvalue()


# --- Common Crawl columnar index ---

CRAWL = "CC-MAIN-2026-39"
TABLE = f"cc-index/table/cc-main/warc/crawl={CRAWL}/subset=warc/"
PDF, DOC, WPS = "application/pdf", "application/msword", "application/vnd.ms-works"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
OLE, HTML = "application/x-tika-msoffice", "application/xhtml+xml"
INDEX_SCHEMA = pa.schema([("url_surtkey", pa.string()), ("url", pa.string()), ("url_host_name", pa.string()),
                          ("content_digest", pa.string()), ("content_mime_type", pa.string()),
                          ("content_mime_detected", pa.string()), ("content_languages", pa.string()),
                          ("content_truncated", pa.string()), ("warc_filename", pa.string()),
                          ("warc_record_offset", pa.int32()), ("warc_record_length", pa.int32())])


def index_part(rows, row_group_size):
    """Index rows (surtkey, url, host, mime, truncated, languages) sorted by surtkey, with filler columns."""
    rows = sorted(rows)
    columns = {"url_surtkey": [r[0] for r in rows], "url": [r[1] for r in rows], "url_host_name": [r[2] for r in rows],
               "content_digest": [f"{i:032d}" * 8 for i in range(len(rows))],
               "content_mime_type": [r[3] for r in rows], "content_mime_detected": [r[3] for r in rows],
               "content_truncated": [r[4] for r in rows], "content_languages": [r[5] for r in rows],
               "warc_filename": [f"crawl-data/{CRAWL}/segments/1/warc/w-{i % 3:05d}.warc.gz" for i in range(len(rows))],
               "warc_record_offset": [1000 * i for i in range(len(rows))],
               "warc_record_length": [100 + i for i in range(len(rows))]}
    return parquet_bytes(columns, row_group_size, INDEX_SCHEMA)


def row(host, path, mime=PDF, truncated=None, languages=None):
    labels = host.split(".")
    labels = labels[1:] if labels[0] == "www" else labels
    return (",".join(reversed(labels)) + ")" + path.lower(), f"http://{host}{path}", host, mime, truncated, languages)


GBK_NAME = "%B9%D8%D3%DA%CD%A8%D6%AA.doc"  # 关于通知.doc in GBK escapes
COM_ROWS = [row(f"site{i}.com", f"/a{i}.pdf") for i in range(8)]  # a part far from cn
# Six rows per row group in SURT order: cn,com,shop*; cn,edu,pku .. cn,gov,moe 录取名单; plan.wps .. zzz.org.cn.
CN_ROWS = ([row(f"shop{i}.com.cn", f"/p{i}.pdf") for i in range(6)]
           + [row("www.pku.edu.cn", "/files/%E8%AF%BE%E4%BB%B6.pptx", PPTX),
              row("jwc.pku.edu.cn", "/x/report.pdf", PDF, "length"),
              row("jwc.pku.edu.cn", "/x/grades.xls", OLE),
              row("jwc.pku.edu.cn", "/download.jsp?id=7", OLE),
              row("www.moe.gov.cn", "/files/" + GBK_NAME, DOC),
              row("www.moe.gov.cn", "/files/plan.wps", WPS),
              row("www.moe.gov.cn", "/files/%E5%BD%95%E5%8F%96%E5%90%8D%E5%8D%95.pdf", PDF),  # 录取名单
              row("www.moe.gov.cn", "/files/table.xlsx", XLSX),
              row("www.moe.gov.cn", "/index.html", HTML),
              row("zzz.org.cn", "/out.docx", DOCX)])


def index_server(base=CC_BASE, empty_group=False):
    parts = {"part-00000-x.c000.zstd.parquet": index_part(COM_ROWS, 4),
             "part-00001-x.c000.zstd.parquet": index_part(CN_ROWS, 6)}
    if empty_group:  # a part whose stats overlap cn,gov, but which holds no documents
        parts["part-00002-x.c000.zstd.parquet"] = index_part(
            [row("www.beijing.gov.cn", f"/news/{i}.html", HTML) for i in range(5)], 5)
    listing = "\n".join([f"cc-index/table/cc-main/warc/crawl={CRAWL}/subset=crawldiagnostics/part-00000.parquet"]
                        + [TABLE + name for name in reversed(list(parts))])
    pages = {base + f"crawl-data/{CRAWL}/cc-index-table.paths.gz": gzip.compress(listing.encode()),
             "https://data.commoncrawl.org/robots.txt": "User-agent: *\nDisallow: /\n"}
    return Server(files={base + TABLE + name: data for name, data in parts.items()}, pages=pages)


def run_index(server, tmp_path, **options):
    options = dict(dict(cache=str(tmp_path / "cache"), crawl=CRAWL, base=CC_BASE, cc_data_host=True), **options)
    return list(commoncrawl.index(fetcher(server), options))


def test_index_reads_only_overlapping_row_groups_and_keeps_documents(tmp_path):
    server = index_server(empty_group=True)
    found = run_index(server, tmp_path, prefixes="cn,edu, cn,gov,")
    assert [c["url"] for c in found] == [
        "http://www.pku.edu.cn/files/%E8%AF%BE%E4%BB%B6.pptx", "http://jwc.pku.edu.cn/download.jsp?id=7",
        "http://jwc.pku.edu.cn/x/report.pdf", "http://www.moe.gov.cn/files/" + GBK_NAME,
        "http://www.moe.gov.cn/files/plan.wps"]
    deck, handler, report, notice, wps = found
    assert deck["title"] == "课件" and deck["type_hint"] == "pptx" and deck["meta"]["truncated"] is None
    assert notice["title"] == "关于通知" and notice["type_hint"] == "doc" and wps["type_hint"] == "doc"
    assert handler["type_hint"] == "doc" and handler["meta"]["mime"] == OLE  # generic OLE without a sheet extension
    assert report["meta"]["truncated"] == "length"
    assert report["warc"] == {"filename": f"crawl-data/{CRAWL}/segments/1/warc/w-00000.warc.gz", "offset": 9000,
                              "length": 109}  # the tenth row in SURT order
    assert report["key"] == f"cc:{report['warc']['filename']}@{report['warc']['offset']}"
    assert report["meta"]["part"] == "part-00001-x.c000.zstd.parquet" and report["meta"]["row_group"] == 1
    assert "https://data.commoncrawl.org/robots.txt" not in server.urls()  # dataset reads skip robots.txt
    com = CC_BASE + TABLE + "part-00000-x.c000.zstd.parquet"
    assert [span[:7] for span in server.ranges(com)] == ["bytes=-"]  # footer only
    # The overlapping part without documents gets its host/MIME columns read, never its urls.
    empty = CC_BASE + TABLE + "part-00002-x.c000.zstd.parquet"
    metadata = pq.read_metadata(io.BytesIO(server.files[empty]))
    url_start = metadata.row_group(0).column(1).data_page_offset
    assert all(not span.startswith(f"bytes={url_start}-") for span in server.ranges(empty))
    assert len(server.ranges(empty)) == 2


def test_index_caches_footers_and_listing(tmp_path):
    server = index_server()
    first = run_index(server, tmp_path)
    calls = len(server.calls)
    second = run_index(server, tmp_path)
    assert first == second and len(first) == 6 + 4 + 2
    later = [span for _, span in server.calls[calls:]]
    assert None not in later and not any(span.startswith("bytes=-") for span in later)  # no listing, no footer
    first_part, second_part = (span for url, span in server.calls[:calls] if span and span.startswith("bytes=-"))
    assert first_part == f"bytes=-{commoncrawl.TAIL}" and int(second_part[7:]) < commoncrawl.TAIL  # sized by the first
    assert (tmp_path / "cache" / CRAWL / "paths.txt").read_text().count("subset=warc") == 2


def test_index_truncated_modes_regex_mime_names_and_limits(tmp_path):
    server = index_server()
    live = run_index(server, tmp_path, prefixes=["cn,edu,"], live_truncated=True)
    report = next(c for c in live if c["url"].endswith("report.pdf"))
    assert "warc" not in report and report["key"] == "jwc.pku.edu.cn/x/report.pdf"
    assert not any(c["url"].endswith("report.pdf") for c in run_index(server, tmp_path, prefixes="cn,edu,",
                                                                       skip_truncated=True))
    decks = run_index(server, tmp_path, mimes="pptx,ole", url_regex=r"\.(ppt|pptx|pps|ppsx|dps)$")
    assert [c["type_hint"] for c in decks] == ["pptx"]
    assert [c["url"] for c in run_index(server, tmp_path, mimes=["wps"])] == ["http://www.moe.gov.cn/files/plan.wps"]
    assert len(run_index(server, tmp_path, max_row_groups=1)) == 6  # first group in SURT order: cn,com,...
    assert len(run_index(server, tmp_path, max_items=2)) == 2
    assert run_index(server, tmp_path, prefixes="cn,edu,pku)/files") == run_index(server, tmp_path)[6:7]
    seeded = run_index(server, tmp_path, seed=3)
    groups = [c["meta"]["row_group"] for c in seeded]
    assert sorted(groups) == [0] * 6 + [1] * 4 + [2] * 2 and groups != sorted(groups)
    shops = [c["url"] for c in seeded if c["meta"]["row_group"] == 0]
    assert shops != sorted(shops) and seeded == run_index(server, tmp_path, seed=3)  # shuffled within a group too
    assert [c["meta"]["host"] for c in run_index(server, tmp_path, prefixes="cn,edu,", per_host=1)] == [
        "www.pku.edu.cn", "jwc.pku.edu.cn"]


def test_index_hugging_face_bucket_resolves_once_and_reresolves_expired_urls(tmp_path):
    bucket = commoncrawl.HF_BUCKET
    server = index_server(base=bucket)
    part = bucket + TABLE + "part-00001-x.c000.zstd.parquet"
    other = bucket + TABLE + "part-00000-x.c000.zstd.parquet"
    signed = "https://cdn.example.org/part1?sig="
    server.files[signed + "1"] = server.files[signed + "2"] = server.files.pop(part)
    server.files[signed + "other"] = server.files.pop(other)
    server.redirects = {part: [signed + "1", signed + "2"], other: [signed + "other"]}
    server.pages["https://huggingface.co/robots.txt"] = "User-agent: *\nAllow: /\n"
    found = run_index(server, tmp_path, prefixes="cn,gov,", base=bucket)
    assert [c["warc"]["base"] for c in found] == [bucket, bucket]
    assert server.urls().count(part) == 1 and "https://huggingface.co/robots.txt" in server.urls()
    # A signed URL that expires is resolved again.
    remote = commoncrawl._Remote(fetcher(server), part)
    assert remote.read_at(0, 4) == b"PAR1" and remote.url == signed + "2"
    server.refused.add(signed + "2")
    server.redirects[part] = [signed + "1"]
    assert remote.read_at(4, 8) == server.files[signed + "1"][4:8] and remote.url == signed + "1"


def test_index_skips_parts_without_a_footer_and_catalog_gates_terms(tmp_path):
    server = index_server()
    server.files.pop(CC_BASE + TABLE + "part-00001-x.c000.zstd.parquet")
    logged = []
    assert run_index(server, tmp_path, log=logged.append) == []
    assert any("no footer" in line and "http_404" in line for line in logged)
    with pytest.raises(PermissionError):
        next(catalog.discover("cc_office_zh", fetcher(index_server()), {}))
    bucket = commoncrawl.HF_BUCKET  # the default base of 2026+ crawls
    server = index_server(base=bucket)
    server.pages["https://huggingface.co/robots.txt"] = "User-agent: *\nAllow: /\n"
    found = list(catalog.discover("cc_office_zh", fetcher(server), {"cache": str(tmp_path / "c2")},
                                  accepted={"web_mixed"}))
    assert found and {c["source"] for c in found} == {"cc_office_zh"} and {c["terms"] for c in found} == {"web_mixed"}
    assert {c["warc"]["base"] for c in found} == {bucket}


def test_remote_refuses_servers_that_ignore_ranges():
    server = Server(pages={"https://huggingface.co/robots.txt": "", "https://huggingface.co/x.parquet": b"PAR1" * 10})
    with pytest.raises(FetchError) as error:
        commoncrawl._Remote(fetcher(server), "https://huggingface.co/x.parquet").footer()
    assert error.value.reason == "range_ignored"


def test_index_zho_share_keeps_hosts_whose_html_is_mostly_chinese(tmp_path):
    server = index_server()
    pages = [row("www.school.edu.tw", f"/news/{i}.html", HTML, None, "zho,eng") for i in range(5)]
    pages += [row("www.school.edu.tw", "/en.html", HTML, None, "eng"), row("www.school.edu.tw", "/a.pdf")]
    pages += [row("shop.com.tw", f"/{i}.html", HTML, None, "eng,zho") for i in range(6)]
    pages += [row("shop.com.tw", "/b.pdf")]
    pages += [row("tiny.org.tw", f"/{i}.html", HTML, None, "zho") for i in range(2)] + [row("tiny.org.tw", "/c.pdf")]
    server.files[CC_BASE + TABLE + "part-00003-x.c000.zstd.parquet"] = index_part(pages, 20)
    listing = CC_BASE + f"crawl-data/{CRAWL}/cc-index-table.paths.gz"
    extra = "\n" + TABLE + "part-00003-x.c000.zstd.parquet"
    server.pages[listing] = gzip.compress(gzip.decompress(server.pages[listing]) + extra.encode())
    found = run_index(server, tmp_path, prefixes="tw,", zho_share=0.5)
    assert [(c["url"], c["meta"]["host_zho_share"]) for c in found] == [("http://www.school.edu.tw/a.pdf", 0.833)]
    assert len(run_index(server, tmp_path, prefixes="tw,", zho_share=0.5, zho_pages=2)) == 2
    assert len(run_index(server, tmp_path, prefixes="tw,")) == 3


def test_names_from_paths_and_download_handlers():
    weihai = ("https://www.weihai.gov.cn/module/download/downfile.jsp?filename=185d7be3.doc&classid=0"
              "&showname=%E5%A8%81%E6%B5%B7%E5%B8%82%E7%87%83%E6%B0%94%E7%AE%A1%E7%90%86%E5%8A%9E%E6%B3%95.doc")
    assert commoncrawl._name(weihai)[0] == "威海市燃气管理办法.doc"
    assert commoncrawl._name("http://a.cn/dl.jsp?path=/files/x/report.xls")[0] == "report.xls"
    assert commoncrawl._name("http://a.cn/f/61de9751.docx?name=%E9%80%9A%E7%9F%A5.pdf")[0] == "通知.pdf"
    assert commoncrawl._name("http://a.cn/f/61de9751.docx?v=2")[0] == "61de9751.docx"
    assert commoncrawl._name("http://a.cn/download?siteId=28&id=1")[0] == "download"
    assert commoncrawl._name("http://a.cn/f/" + GBK_NAME)[0] == "关于通知.doc"


def test_surt_helpers():
    assert commoncrawl._surt_host("WWW2.Moe.Gov.CN") == "cn,gov,moe)"
    assert commoncrawl._surt_host("www.cn") == "cn)" and commoncrawl._surt_host(None) == ")"
    assert commoncrawl._overlaps("cn,get17)/a", "cn,gov,shanghai)/b", "cn,gov,")
    assert not commoncrawl._overlaps("cn,com,a)/", "cn,com,z)/", "cn,gov,")
    assert not commoncrawl._overlaps("cn,gp)/", "cn,zz)/", "cn,gov,")
    assert commoncrawl._prefixes("cn,edu, | cn,gov,;tw,") == ["cn,edu,", "cn,gov,", "tw,"]
    assert commoncrawl._mimes("doc") == {DOC, WPS}


# --- FinePDFs ---

FP_SCHEMA = pa.schema([("text", pa.string()), ("id", pa.string()), ("dump", pa.string()), ("url", pa.string()),
                       ("file_path", pa.string()), ("offset", pa.int64()), ("is_truncated", pa.bool_()),
                       ("extractor", pa.string()), ("page_ends", pa.list_(pa.int64())), ("token_count", pa.int64())])


def finepdfs_shard(shard, count, group=10):
    rows = []
    for i in range(count):
        truncated = i % 5 == 4
        extractor = "rolmOCR" if truncated or i % 5 == 3 else "docling"
        host = ("www.moe.gov.cn", "www.ntu.edu.tw", "example.com")[i % 3]
        name = "%E5%85%A5%E5%AD%A6%E5%90%8D%E5%8D%95.pdf" if i == 7 else f"doc{shard}_{i}.pdf"  # 入学名单
        text = "".join(chr(0x4E00 + (i * 7919 + k * 104729) % 20000) for k in range(3000))  # incompressible
        rows.append(dict(text=text, id=f"<urn:uuid:{shard}-{i}>", dump="CC-MAIN-2024-10",
                         url=f"https://{host}/files/{name}", file_path=f"crawl-data/CC-MAIN-2024-10/w{shard}.warc.gz",
                         offset=1000 * i, is_truncated=truncated, extractor=extractor,
                         page_ends=list(range(100, 100 * (i % 4 + 1) + 1, 100)), token_count=10))
    rows[-3]["url"] = rows[0]["url"]  # the same URL again (docling, untruncated): dropped as a duplicate
    columns = {name: [r[name] for r in rows] for name in FP_SCHEMA.names}
    return parquet_bytes(columns, group, FP_SCHEMA)


def finepdfs_server(shards=(("000_00000", 40), ("000_00001", 30))):
    tree = commoncrawl.FINEPDFS_TREE + "data/cmn_Hani/train"
    files, redirects, listing = {}, {}, []
    for name, count in shards:
        path = f"data/cmn_Hani/train/{name}.parquet"
        data = finepdfs_shard(name, count)
        files[f"https://cdn.example.org/{name}?sig=1"] = data
        redirects[commoncrawl.FINEPDFS + path] = [f"https://cdn.example.org/{name}?sig=1"]
        listing.append({"type": "file", "path": path, "size": len(data), "lfs": {"oid": name * 4}})
    first_page = json.dumps(listing[:1] + [{"type": "directory", "path": "x"}])
    pages = {"https://huggingface.co/robots.txt": "User-agent: *\nAllow: /\n",
             tree: (first_page, {"Link": f'<{tree}?cursor=2>; rel="next"'}),
             tree + "?cursor=2": json.dumps(listing[1:])}
    return Server(files=files, pages=pages, redirects=redirects)


def run_finepdfs(server, tmp_path, **options):
    return list(commoncrawl.finepdfs(fetcher(server), dict(cache=str(tmp_path / "fp"), **options)))


def test_finepdfs_samples_row_groups_and_keeps_untruncated_docling(tmp_path):
    server = finepdfs_server()
    found = run_finepdfs(server, tmp_path, max_row_groups=7, seed=1)
    groups = {(c["meta"]["shard"], c["meta"]["row_group"]) for c in found}
    assert len(groups) == 7 and len({shard for shard, _ in groups}) == 2
    assert all(c["meta"]["extractor"] == "docling" and not c["meta"]["truncated"] for c in found)
    assert len({c["url"] for c in found}) == len(found) and not any("%E5%90%8D%E5%8D%95" in c["url"] for c in found)
    first = next(c for c in found if c["url"].endswith("doc000_00000_1.pdf"))
    assert first["warc"] == {"filename": "crawl-data/CC-MAIN-2024-10/w000_00000.warc.gz", "offset": 1000}
    assert first["key"] == "cc:crawl-data/CC-MAIN-2024-10/w000_00000.warc.gz@1000"
    assert first["meta"]["pages"] == 2 and first["title"] == "doc000_00000_1" and first["type_hint"] == "pdf"
    assert run_finepdfs(server, tmp_path, max_row_groups=7, seed=1) == found  # reproducible, cached footers
    # Each shard is resolved once per run; exact reads never touch the text column.
    assert server.urls().count(commoncrawl.FINEPDFS + "data/cmn_Hani/train/000_00000.parquet") == 2

    def drawn(seed):
        found = run_finepdfs(server, tmp_path, max_row_groups=3, seed=seed)
        return list(dict.fromkeys((c["meta"]["shard"], c["meta"]["row_group"]) for c in found))
    assert drawn(2) == drawn(2) != drawn(1)
    shard = server.files["https://cdn.example.org/000_00000?sig=1"]
    metadata = pq.read_metadata(io.BytesIO(shard))
    texts = [(c.data_page_offset if not c.has_dictionary_page else c.dictionary_page_offset, c.total_compressed_size)
             for c in (metadata.row_group(g).column(0) for g in range(metadata.num_row_groups))]
    reads = [tuple(map(int, span[6:].split("-"))) for span in server.ranges("https://cdn.example.org/000_00000?sig=1")
             if not span.startswith("bytes=-")]
    assert reads and all(end < start or begin >= start + size for begin, end in reads for start, size in texts)


def test_finepdfs_options_truncated_pages_hosts_and_per_group(tmp_path):
    server = finepdfs_server()
    live = run_finepdfs(server, tmp_path, max_row_groups=3, include_truncated=True, extractors="all")
    truncated = [c for c in live if c["meta"]["truncated"]]
    assert truncated and all("warc" not in c and c["key"] == c["url"].split("://")[1] for c in truncated)
    assert {c["meta"]["extractor"] for c in live} == {"docling", "rolmOCR"}
    only_truncated = run_finepdfs(server, tmp_path, max_row_groups=3, include_truncated=True, extractors="none")
    assert only_truncated and all(c["meta"]["truncated"] for c in only_truncated)
    picked = run_finepdfs(server, tmp_path, max_row_groups=4, min_pages=2, max_pages=3, hosts=r"\.(cn|tw)$")
    assert picked and all(2 <= c["meta"]["pages"] <= 3 and c["meta"]["host"] != "example.com" for c in picked)
    assert len(run_finepdfs(server, tmp_path, max_row_groups=4, per_row_group=1)) == 4
    assert len(run_finepdfs(server, tmp_path, max_row_groups=None, per_row_group=None)) == 22 + 16  # both shards
    assert len(run_finepdfs(server, tmp_path, max_row_groups=None, per_row_group=2)) == 2 * 7
    based = run_finepdfs(server, tmp_path, max_row_groups=1, warc_base=commoncrawl.HF_BUCKET)
    assert based and all(c["warc"]["base"] == commoncrawl.HF_BUCKET for c in based)


def test_finepdfs_long_footer_is_read_in_two_steps_and_bad_shards_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(commoncrawl, "TAIL", 256)
    server = finepdfs_server()
    server.redirects[commoncrawl.FINEPDFS + "data/cmn_Hani/train/000_00001.parquet"] = ["https://cdn.example.org/gone"]
    logged = []
    found = run_finepdfs(server, tmp_path, max_row_groups=3, log=logged.append)
    assert found and {c["meta"]["shard"] for c in found} == {"data/cmn_Hani/train/000_00000.parquet"}
    assert any("skipped data/cmn_Hani/train/000_00001.parquet: http_404" in line for line in logged)
    spans = server.ranges("https://cdn.example.org/000_00000?sig=1")
    assert spans[0] == "bytes=-256" and re.fullmatch(r"bytes=\d+-\d+", spans[1])
    assert len(list((tmp_path / "fp" / "cmn_Hani").glob("000_00000.*.footer"))) == 1


def test_index_on_the_cc_data_host_needs_the_operator_flag(tmp_path):
    with pytest.raises(PermissionError, match="--cc-data-host"):
        next(commoncrawl.index(None, dict(crawl="CC-MAIN-2024-33", cache=str(tmp_path))))
    for base in ("http://data.commoncrawl.org/", "https://DATA.CommonCrawl.org:443/"):  # the host decides
        with pytest.raises(PermissionError, match="--cc-data-host"):
            next(commoncrawl.index(None, dict(crawl=CRAWL, base=base, cache=str(tmp_path))))
