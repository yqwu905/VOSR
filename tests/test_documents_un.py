"""UN documents (offline): UPRPRC parquet dumps on a fake Hugging Face server. Discovery reads only
the small columns; downloads read only the data pages of the rows with a Chinese file."""

import io
import json
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.documents import fetch, un
from data_pipeline.documents.fetch import Fetcher, Store, download

HF = "https://huggingface.co/"
SHA = "0123456789abcdef0123456789abcdef01234567"
SCHEMA = pa.schema([("id", pa.string()), ("symbol", pa.string()), ("symbols", pa.list_(pa.string(), 3)),
                    ("publication_date", pa.string()), ("area", pa.string()), ("distribution", pa.string()),
                    ("job_numbers", pa.list_(pa.string(), 7)), ("sizes", pa.list_(pa.int64(), 21)),
                    ("title", pa.string()), ("blobs", pa.list_(pa.large_binary(), 7)),
                    ("crawl_res", pa.list_(pa.string(), 7))])


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
    """Hugging Face API pages, resolve URLs that redirect to a CDN, and CDN files with Range support."""

    def __init__(self):
        self.files, self.pages, self.redirects, self.calls, self.headers = {}, {}, {}, [], {}

    def request(self, method, url, headers=None, **kwargs):
        span = (headers or {}).get("Range")
        self.calls.append((url, span))
        if url in self.redirects:
            return Response(302, headers={"Location": self.redirects[url]})
        if url in self.files:
            data = self.files[url]
            found = re.fullmatch(r"bytes=(\d*)-(\d*)", span or "")
            if not found.group(1):
                start, end = max(0, len(data) - int(found.group(2))), len(data)
            else:
                start, end = int(found.group(1)), min(len(data), int(found.group(2)) + 1)
            return Response(206, data[start:end], {"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"}, url)
        if url in self.pages:
            return Response(200, self.pages[url].encode(), url=url)
        return Response(404, url=url)

    def ranges(self, url):
        return [tuple(map(int, span[6:].split("-"))) for called, span in self.calls
                if called == url and span and not span.startswith("bytes=-")]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def fetcher(server):
    clock = Clock()
    return Fetcher(http=server, interval=1.0, clock=clock, sleep=clock.sleep)


def blob(symbol, language, size=12000):
    return f"%PDF-1.7 {symbol} {language} ".encode() + random.Random(f"{symbol}{language}").randbytes(size)


def shard_rows(start, specs):
    """specs: (year, zh extension, title, other languages) per row."""
    rows = []
    for i, (year, zh, title, others) in enumerate(specs, start):
        symbol = f"A/RES/{i}"
        extensions = ["pdf" if code in others else "" for code in un.LANGUAGES]
        extensions[1] = zh
        blobs = [blob(symbol, code) if ext else None for code, ext in zip(un.LANGUAGES, extensions)]
        sizes = [-1] * 21
        for language, data in enumerate(blobs):
            if data:
                sizes[3 * language + un.SIZE_SLOT.get(extensions[language], 2)] = len(data)
        rows.append(dict(id=f"{symbol}_{1000 + i}", symbol=symbol, symbols=[symbol, "", ""],
                         publication_date=f"{year}-05-0{i % 9 + 1}T00:00:00Z", area="UNDOC", distribution="GEN",
                         job_numbers=[f"N{i}{k}" if ext else "" for k, ext in enumerate(extensions)], sizes=sizes,
                         title=title, blobs=blobs, crawl_res=extensions))
    return rows


def parquet_bytes(rows, group=6):
    buffer = io.BytesIO()
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    # Like the HF shards: an offset index, small record-aligned pages, the first values in a dictionary page.
    pq.write_table(table, buffer, row_group_size=group, compression="snappy", write_page_index=True,
                   data_page_size=100000, write_batch_size=7, dictionary_pagesize_limit=2500)
    return buffer.getvalue()


ALL = ["ar", "en", "fr", "ru", "es"]
PDF_SPECS = [  # shard 0: two groups of six rows
    (2018, "pdf", "Report of the Secretary-General", ALL), (2018, "", "Letter dated 1 May", ALL),
    (2018, "pdf", "Draft resolution", ["en"]), (2018, "wpf", "Old WordPerfect file", ALL),
    (2018, "", "Agenda", ALL), (2018, "pdf", "List of participants", ALL),
    (1975, "pdf", "Summary record of the 5th meeting", ALL), (1975, "", "Note verbale", ALL),
    (1975, "", "Note", ALL), (1976, "", "Corrigendum", ALL), (1976, "pdf", "Petition from a resident", ALL),
    (1976, "pdf", "Resolution 3379", ALL)]
DOC_SPECS = [(2020, "doc", "Programme budget", ALL), (2020, "docx", "Annual report", ALL), (2020, "", "Note", ALL)]


def hf_server():
    server = Server()
    server.pages[HF + "robots.txt"] = "User-agent: *\nAllow: /\n"
    shards = {"pdf": [PDF_SPECS, [(2019, "pdf", "Statement", ALL), (2019, "", "Statement", ALL)]], "doc": [DOC_SPECS]}
    start = 0
    for kind, specs in shards.items():
        dataset, listing = un.DATASETS[kind], []
        for number, spec in enumerate(specs):
            path = f"data/train-{number:05d}-of-{len(specs):05d}.parquet"
            data = parquet_bytes(shard_rows(start, spec))
            start += len(spec)
            cdn = f"https://cdn.example.org/{kind}{number}?sig=1"
            server.files[cdn] = data
            server.redirects[f"{HF}datasets/{dataset}/resolve/{SHA}/{path}"] = cdn
            listing.append({"type": "file", "path": path, "size": len(data), "lfs": {"oid": f"{kind}{number}" * 8}})
        listing.append({"type": "file", "path": "data/README.md", "size": 10})
        server.pages[f"{HF}api/datasets/{dataset}"] = json.dumps({"sha": SHA})
        server.pages[f"{HF}api/datasets/{dataset}/tree/{SHA}/data"] = json.dumps(listing[::-1])
    return server


def run(server, tmp_path, **options):
    return list(un.uprprc(fetcher(server), dict(cache=str(tmp_path / "footers"), **options)))


def chunk_spans(data, column="blobs"):
    """(start, end) of each row group's column chunk."""
    metadata = pq.read_metadata(io.BytesIO(data))
    spans = []
    for group in range(metadata.num_row_groups):
        for j in range(metadata.num_columns):
            chunk = metadata.row_group(group).column(j)
            if chunk.path_in_schema.split(".")[0] == column:
                start = chunk.dictionary_page_offset if chunk.has_dictionary_page else chunk.data_page_offset
                spans.append((start, start + chunk.total_compressed_size))
    return spans


def test_discovery_keeps_chinese_rows_and_reads_only_small_columns(tmp_path):
    server = hf_server()
    found = run(server, tmp_path)
    assert [c["meta"]["symbol"] for c in found] == ["A/RES/0", "A/RES/2", "A/RES/6", "A/RES/11", "A/RES/12"]
    first, draft = found[0], found[1]
    url = f"{HF}datasets/{un.DATASETS['pdf']}/resolve/{SHA}/data/train-00000-of-00002.parquet"
    assert first["key"] == "un:A/RES/0:pdf" and first["type_hint"] == "pdf"
    assert first["title"] == "Report of the Secretary-General" and "public domain" in first["license"]
    assert first["parquet"] == {"url": url, "row_group": 0, "row": 0, "column": "blobs", "item": 1, "rows": [0, 2]}
    assert found[2]["parquet"]["rows"] == [0, 5] and found[3]["parquet"]["row"] == 5  # second group
    meta = draft["meta"]
    assert meta["languages"] == ["zh", "en"] and meta["date"] == "2018-05-03" and meta["job_number"] == "N21"
    assert meta["size"] == len(blob("A/RES/2", "zh")) and meta["revision"] == SHA and meta["row_group"] == 0
    assert meta["shard"] == "data/train-00000-of-00002.parquet" and meta["dataset"] == un.DATASETS["pdf"]
    # Discovery never touches a blobs column chunk; a second run reuses the cached footers.
    cdn = "https://cdn.example.org/pdf0?sig=1"
    blobs = chunk_spans(server.files[cdn])
    assert len(server.ranges(cdn)) == 4 and all(end < low or start >= high
                                                for start, end in server.ranges(cdn) for low, high in blobs)
    calls = len(server.calls)
    assert run(server, tmp_path) == found
    assert not any(span and span.startswith("bytes=-") for _, span in server.calls[calls:])


def test_download_reads_only_the_pages_of_chinese_rows(tmp_path):
    server = hf_server()
    found = run(server, tmp_path, shards="0")
    store = Store(tmp_path / "store")
    counts = download(found, store, fetcher(server), log=None)
    assert counts == {"ok": 4}
    for candidate in found:
        entry = store.seen[candidate["key"]]
        assert entry["via"] == "parquet" and entry["type"] == "pdf"
        assert open(entry["path"], "rb").read() == blob(candidate["meta"]["symbol"], "zh")
    # The offset index of each blobs chunk tells which data pages hold which rows: only pages
    # with a Chinese row were fetched (each in full), the others never.
    cdn = "https://cdn.example.org/pdf0?sig=1"
    data, read = server.files[cdn], server.ranges(cdn)
    footer = fetch._thrift(data[-8 - int.from_bytes(data[-8:-4], "little"):])[0]
    leaf = [b".".join(c[3][3]) for c in footer[4][0][1]].index(b"blobs.list.element")
    skipped = 0
    for group, zh_rows in enumerate([{0, 2}, {0, 5}]):
        chunk = footer[4][group][1][leaf]
        pages = fetch._thrift(data[chunk[4]:chunk[4] + chunk[5]])[0][1]
        for page, end in zip(pages, [page[3] for page in pages[1:]] + [6]):
            covered = any(s <= page[1] and page[1] + page[2] - 1 <= e for s, e in read)
            assert covered == any(page[3] <= row < end for row in zh_rows)
            skipped += not covered
    assert skipped >= 4
    assert sum(e - s + 1 for s, e in read) < sum(end - start for start, end in chunk_spans(data))


def test_options_filter_years_kinds_types_and_shards(tmp_path):
    server = hf_server()
    recent = run(server, tmp_path, since=2000)
    assert [c["meta"]["symbol"] for c in recent] == ["A/RES/0", "A/RES/2", "A/RES/12"]
    cdn = "https://cdn.example.org/pdf0?sig=1"
    second = chunk_spans(server.files[cdn], "id")[1][0]
    assert not any(start >= second for start, _ in server.ranges(cdn))  # the 1975-76 group was not read
    old = run(hf_server(), tmp_path, until=1980, shards=[0, 1])
    assert [c["meta"]["date"][:4] for c in old] == ["1975", "1976"]
    both = run(hf_server(), tmp_path, kind="both", since=2019)
    assert [(c["key"], c["type_hint"]) for c in both] == [("un:A/RES/12:pdf", "pdf"), ("un:A/RES/14:doc", "doc"),
                                                          ("un:A/RES/15:doc", "docx")]
    docs = run(hf_server(), tmp_path, kind="doc", types=["docx"])
    assert [c["meta"]["size"] for c in docs] == [len(blob("A/RES/15", "zh"))]
    assert len(run(hf_server(), tmp_path, max_items=2)) == 2
    shuffled = run(hf_server(), tmp_path, seed=1)
    assert sorted(c["key"] for c in shuffled) == sorted(c["key"] for c in run(hf_server(), tmp_path))
    pinned = run(hf_server(), tmp_path, revision=SHA, shards="1")
    assert [c["meta"]["symbol"] for c in pinned] == ["A/RES/12"]


def test_shard_lists():
    assert un._shards(None, 3) == [0, 1, 2]
    assert un._shards("0-2, 5,9", 8) == [0, 1, 2, 5]
    assert un._shards([4, "1-2", 1], 10) == [4, 1, 2] and un._shards(7, 10) == [7]
