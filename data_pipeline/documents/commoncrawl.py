"""Crawl corpora: FinePDFs (Chinese PDFs with Common Crawl WARC pointers) and Common Crawl's
columnar URL index.

finepdfs: HuggingFaceFW/finepdfs data/<config>/train (cmn_Hani: 17 shards, 54.6 GB, 4.91M rows in
1,000-row row groups; yue_Hani and lzh_Hani one shard each) is listed by
https://huggingface.co/api/datasets/HuggingFaceFW/finepdfs/tree/main/data/<config>/train; a file's
/resolve/main/<path> answers 302 with a signed CDN URL, which is reused until it is refused. Only
the provenance columns are read (~0.2-1.3 MB per row group, never the text); a large shard's footer
is ~2 MB and is cached. Row groups are internally clustered (docling share 38-100%, few dumps), so
they are sampled at random across shards. extractor 'docling' marks text-layer PDFs (6/6
born-digital in the survey); rolmOCR marks scans and every truncated capture (21.6% of rows), whose
WARC payload stops at 1 MiB (5 MiB from CC-MAIN-2025-13) and needs a live refetch. ~30% of URLs
are .tw and ~10% .hk. FinePDFs gives the record offset only; fetch.warc_record then reads an
open-ended range up to the end of the record's gzip member.

index: crawl-data/<crawl>/cc-index-table.paths.gz lists the 300 subset=warc parquet parts of
cc-index/table/cc-main/warc/crawl=<crawl>/ (CC-MAIN-2026-39: 183 GB, 1,497 row groups of up to ~2M
rows). Each part holds a contiguous url_surtkey range, but parts are numbered in random SURT order,
so every footer is read once per crawl (one 64 KB suffix read each, ~15 MB) and cached; row groups
are then chosen by their url_surtkey min/max (.cn sits in 11 of 1,497). A chosen group is read in
two steps: host and MIME columns (~0.3 MB), then, only if documents remain, url and the WARC
pointers (~15-20 MB). DuckDB is not used: its httpfs reads a 256 KB tail of every part (77 MB per
crawl). content_languages (CLD2) is NULL for documents, so the host prefix is the Chinese signal;
outside .cn the share of a host's HTML pages tagged zho in the same SURT-sorted row group can stand
in (zho_share option: 145k of 273k tw/hk/mo/sg document captures sit on hosts whose HTML is mostly
zho, 93 of 96k in other row groups). content_mime_detected (Tika) beats URL extensions (.jsp/.do
handlers; .wps shows up as vnd.ms-works or msword; generic x-tika-msoffice/ooxml also cover
spreadsheets). CC-MAIN-2026-39 .cn: 115k PDF, 1.6k DOC, 950 DOCX, 440 WPS, 16 PPT/PPTX/PPSX; 9.8%
of PDFs truncated (37% in 2024-33). data.commoncrawl.org's robots.txt says Disallow: / while Common
Crawl documents that host as the HTTPS download endpoint of the dataset, so dataset reads there
skip robots.txt (as fetch.warc_record does) but keep the fetcher's pacing. 2026+ crawls are
mirrored byte for byte in the Hugging Face bucket ``HF_BUCKET`` (robots.txt allows; base option).
"""

import gzip
import itertools
import json
import random
import re
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

from .fetch import CC_BASE, FetchError

HF = "https://huggingface.co/"
FINEPDFS = HF + "datasets/HuggingFaceFW/finepdfs/resolve/main/"
FINEPDFS_TREE = HF + "api/datasets/HuggingFaceFW/finepdfs/tree/main/"
HF_BUCKET = HF + "buckets/commoncrawl/commoncrawl/resolve/"
CRAWL = "CC-MAIN-2026-39"
CACHE = Path("~/.cache/vosr")
TAIL = 64 << 10  # first footer read: CC footers are 19-95 KB (28 KB median lately), FinePDFs' ~2 MB
GAP = 256 << 10  # column chunks closer than this are fetched in one range read

PERSONAL = re.compile("名[单單]|名[册冊]|身份[证證]")
LICENSE = "copyright of the original publisher (Common Crawl capture; rights vary per document)"
FINEPDFS_LICENSE = LICENSE + "; FinePDFs metadata ODC-By 1.0"
FINEPDFS_COLUMNS = ["id", "dump", "url", "file_path", "offset", "is_truncated", "extractor", "page_ends"]
HOST_COLUMNS = ["url_host_name", "content_mime_type", "content_mime_detected", "content_truncated"]
HTML = ["text/html", "application/xhtml+xml"]
POINTER_COLUMNS = ["url", "warc_filename", "warc_record_offset", "warc_record_length"]

# Detected MIME -> type hint; None: a generic OLE2/OOXML container whose URL tells the type.
MIMES = {
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.ms-works": "doc",  # Kingsoft .wps
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow": "pptx",
    "application/rtf": "rtf",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/x-tika-msoffice": None,
    "application/x-tika-ooxml": None,
}
MIME_NAMES = {"wps": ["application/vnd.ms-works"], "ole": ["application/x-tika-msoffice"],
              "ooxml": ["application/x-tika-ooxml"]}
EXTENSIONS = {"pdf": "pdf", "doc": "doc", "dot": "doc", "wps": "doc", "docx": "docx", "ppt": "ppt", "pps": "ppt",
              "dps": "ppt", "pptx": "pptx", "ppsx": "pptx", "rtf": "rtf", "odt": "odt", "odp": "odp"}
NOT_DOCUMENTS = {"xls", "xlsx", "xlsm", "xlt", "xltx", "et", "ett", "csv", "msi", "msg", "vsd", "vsdx", "pub", "mpp"}


def _quiet(message):
    pass


def _list(value):
    if value is None:
        return []
    return value.replace(",", " ").split() if isinstance(value, str) else list(value)


def _bare(url):
    return url.split("://", 1)[-1]


def _decoded(text):
    raw = unquote_to_bytes(text)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", "replace")  # GBK escapes are common on Chinese sites


def _name(url):
    """Percent-decoded last path segment and the decoded path with query (for the personal-data check)."""
    parts = urlsplit(url)
    path = _decoded(parts.path + ("?" + parts.query if parts.query else ""))
    return _decoded(parts.path.rsplit("/", 1)[-1]), path


def _stem(name):
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name).strip()


def _get(fetcher, url, **kwargs):
    # data.commoncrawl.org's robots.txt (Disallow: /) targets crawlers of the site, not downloads
    # of the dataset it is documented to serve; pacing still applies.
    return fetcher.get(url, check_robots=urlsplit(url).hostname != "data.commoncrawl.org", **kwargs)


class _Remote:
    """Read-only file over HTTP range reads with no read-ahead: a read inside a span fetched by
    ``prefetch`` is served from memory, any other read is one exact range request. A redirect
    (Hugging Face resolve URLs) is followed once and the signed URL kept until it is refused."""

    def __init__(self, fetcher, url, size=None):
        self.fetcher, self.origin, self.url, self.size = fetcher, url, url, size
        self.position = self.transferred = self.requests = 0
        self.spans, self.closed = {}, False

    def _range(self, spec, limit):
        for retry in (False, True):
            try:
                response, body = _get(self.fetcher, self.url, headers={"Range": "bytes=" + spec}, stream_limit=limit)
                break
            except FetchError as error:
                if retry or self.url == self.origin or error.reason not in ("http_401", "http_403"):
                    raise
                self.url = self.origin  # the signed URL expired: resolve again
        if response.status_code != 206:
            raise FetchError("range_ignored", self.origin)
        self.url = getattr(response, "url", None) or self.url
        self.requests, self.transferred = self.requests + 1, self.transferred + len(body)
        return response, body

    def footer(self, guess=None):
        """The parquet footer (metadata, its length, PAR1) in one or two reads; sets ``size``."""
        guess = guess or TAIL
        response, tail = self._range(f"-{guess}", guess)
        total = re.search(r"/(\d+)\s*$", response.headers.get("Content-Range", ""))
        if not total or tail[-4:] != b"PAR1":
            raise FetchError("not_parquet", self.origin)
        self.size = int(total.group(1))
        length = int.from_bytes(tail[-8:-4], "little") + 8
        if length > len(tail):
            tail = self.read_at(self.size - length, self.size - len(tail)) + tail
        return tail[-length:]

    def read_at(self, start, end):
        for (low, high), data in self.spans.items():
            if low <= start and end <= high:
                return data[start - low:end - low]
        _, body = self._range(f"{start}-{end - 1}", end - start)
        if len(body) != end - start:
            raise FetchError("short_range", self.origin)
        return body

    def prefetch(self, spans):
        merged = []
        for start, end in sorted(spans):
            if merged and start - merged[-1][1] <= GAP:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.spans = {}
        self.spans = {(start, end): self.read_at(start, end) for start, end in merged}

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset, self.position + offset, self.size + offset)[whence]
        return self.position

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        data = self.read_at(self.position, end) if end > self.position else b""
        self.position = end
        return data

    def close(self):
        self.closed = True


def _parquet(fetcher, url, cache_file, guess=None):
    """(remote file, ParquetFile) with the footer from ``cache_file`` or read once and cached there."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    remote = _Remote(fetcher, url)
    if cache_file.exists():
        raw = cache_file.read_bytes()
        remote.size, tail = int.from_bytes(raw[:8], "little"), raw[8:]
    else:
        tail = remote.footer(guess)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        partial = cache_file.with_name(cache_file.name + ".part")
        partial.write_bytes(remote.size.to_bytes(8, "little") + tail)
        partial.replace(cache_file)
    metadata = pq.read_metadata(pa.BufferReader(tail))
    return remote, pq.ParquetFile(pa.PythonFile(remote, mode="r"), metadata=metadata, pre_buffer=False)


def _row_group(remote, parquet, group, columns):
    """Some columns of one row group, fetching exactly their column chunks."""
    row_group, spans = parquet.metadata.row_group(group), []
    for j in range(row_group.num_columns):
        chunk = row_group.column(j)
        if chunk.path_in_schema.split(".")[0] in columns:
            start = chunk.data_page_offset
            if chunk.has_dictionary_page and 0 < chunk.dictionary_page_offset < start:
                start = chunk.dictionary_page_offset
            spans.append((start, start + chunk.total_compressed_size))
    try:
        remote.prefetch(spans)
        return parquet.read_row_group(group, columns=columns, use_threads=False)
    finally:
        remote.spans = {}


def _errors():
    import pyarrow as pa
    return FetchError, OSError, ValueError, pa.ArrowException


# --- FinePDFs ---

def finepdfs(fetcher, options):
    """PDFs of HuggingFaceFW/finepdfs as Common Crawl WARC candidates (no length: open-ended read).

    Options: config (cmn_Hani, yue_Hani or lzh_Hani; default cmn_Hani), split (default train), seed
    (int, default 0: which row groups are drawn), max_row_groups (int, default 20; ~1,000 rows each,
    ~60% docling; null for all), per_row_group (int, candidates kept per row group, default 50 so
    that a sample spans many groups; null for all), extractors (list or comma string, default
    docling; "all" keeps rolmOCR rows too), include_truncated (bool, default False: truncated
    captures, all rolmOCR, become url-only candidates for a live fetch whatever the extractor
    filter), min_pages / max_pages (int, from page_ends), hosts (regex searched in the URL host,
    e.g. "\\.(cn|tw)$"), warc_base (default data.commoncrawl.org), cache (footer cache directory,
    default ~/.cache/vosr/finepdfs_footers), max_items (int), log (callable)."""
    yield from itertools.islice(_finepdfs(fetcher, options), options.get("max_items"))


def _shards(fetcher, config, split):
    url, shards = f"{FINEPDFS_TREE}data/{config}/{split}", []
    while url:
        response, body = fetcher.get(url)
        shards += [e for e in json.loads(body) if e.get("type") == "file" and e["path"].endswith(".parquet")]
        following = re.search(r'<([^>]+)>\s*;\s*rel="next"', response.headers.get("Link", ""))
        url = following and following.group(1)
    return shards


def _finepdfs(fetcher, options):
    log = options.get("log") or _quiet
    config, split = options.get("config", "cmn_Hani"), options.get("split", "train")
    cache = Path(options.get("cache") or CACHE / "finepdfs_footers").expanduser() / config
    shards = {e["path"]: e for e in _shards(fetcher, config, split)}
    weights = {path: e.get("size") or 1 for path, e in shards.items()}
    rng, opened, left = random.Random(options.get("seed", 0)), {}, {}
    budget, seen = options.get("max_row_groups", 20), set()
    for _ in itertools.count() if budget is None else range(int(budget)):
        if not weights:
            break
        # Shards are drawn by size (about their row-group count), groups uniformly within a shard.
        path = rng.choices(list(weights), weights=list(weights.values()))[0]
        if path not in opened:
            oid = ((shards[path].get("lfs") or {}).get("oid") or shards[path].get("oid") or "")[:16]
            try:
                opened[path] = _parquet(fetcher, FINEPDFS + path, cache / f"{Path(path).stem}.{oid}.footer")
            except _errors() as error:
                log(f"finepdfs: skipped {path}: {getattr(error, 'reason', error)}")
                del weights[path]
                continue
            left[path] = list(range(opened[path][1].metadata.num_row_groups))
        groups = left[path]
        group = groups.pop(rng.randrange(len(groups)))
        weights[path] = (shards[path].get("size") or 1) * len(groups) / opened[path][1].metadata.num_row_groups
        if not groups:
            del weights[path]
        try:
            table = _row_group(*opened[path], group, FINEPDFS_COLUMNS)
        except _errors() as error:
            log(f"finepdfs: skipped {path} row group {group}: {getattr(error, 'reason', error)}")
            continue
        kept = _finepdfs_rows(table.to_pylist(), options, seen, dict(config=config, shard=path, row_group=group))
        yield from itertools.islice(kept, options.get("per_row_group", 50))
        log(f"finepdfs: {path} row group {group}; {opened[path][0].transferred / 1e6:.1f} MB read from the shard")


def _finepdfs_rows(rows, options, seen, where):
    extractors = {e.lower() for e in _list(options.get("extractors", "docling"))}
    hosts = re.compile(options["hosts"], re.I) if options.get("hosts") else None
    low, high = options.get("min_pages"), options.get("max_pages")
    base = options.get("warc_base") or CC_BASE
    for row in rows:
        url, pages = row["url"] or "", len(row["page_ends"] or [])
        live = bool(row["is_truncated"])
        if live and not options.get("include_truncated"):
            continue
        if not live and "all" not in extractors and (row["extractor"] or "").lower() not in extractors:
            continue
        host = urlsplit(url).hostname or ""
        name, path = _name(url)
        if (not url.startswith(("http://", "https://")) or url in seen or PERSONAL.search(path)
                or (hosts and not hosts.search(host)) or (low and pages < low) or (high and pages > high)):
            continue
        seen.add(url)
        meta = dict(dump=row["dump"], pages=pages, extractor=row["extractor"], truncated=live, host=host,
                    finepdfs_id=row["id"], **where)
        candidate = dict(url=url, title=_stem(name) or host, type_hint="pdf", license=FINEPDFS_LICENSE, meta=meta)
        if live or not row["file_path"]:
            yield dict(key=_bare(url), **candidate)
            continue
        warc = dict(filename=row["file_path"], offset=int(row["offset"]))
        if base != CC_BASE:
            warc["base"] = base
        yield dict(key=f"cc:{warc['filename']}@{warc['offset']}", warc=warc, **candidate)


# --- Common Crawl columnar index ---

def index(fetcher, options):
    """Office and PDF captures of one crawl under SURT host prefixes, from the columnar URL index.

    Options: crawl (default CC-MAIN-2026-39), prefixes (SURT prefixes, list or a string split on
    spaces, | or ;, default ["cn,"]; e.g. "cn,edu, cn,gov," or "tw,edu,ntu)"; www is ignored as in
    url_surtkey), mimes (detected MIME types or the names pdf, doc, docx, ppt, pptx, wps, rtf, odt,
    odp, ole, ooxml; default all of these, never spreadsheets), url_regex (searched in the URL,
    case-insensitive, e.g. "\\.(ppt|pptx|pps|ppsx|dps)$"), max_row_groups (int, default 3; 0 or null
    reads every overlapping group, ~15-20 MB each), seed (int: random group order; default SURT
    order), zho_share (float, e.g. 0.5: keep only hosts with at least this share of their HTML pages
    in the row group tagged zho first; costs ~0.3 MB more per group; default off), zho_pages (int,
    default 5: HTML pages a host needs for that test), live_truncated (bool, default False:
    truncated captures stay WARC candidates flagged meta.truncated; True makes them url-only
    candidates fetched live), skip_truncated (bool, default False), base (default
    https://data.commoncrawl.org/, or ``HF_BUCKET`` for 2026+ crawls; WARC candidates use it too),
    cache (footer cache directory, default ~/.cache/vosr/cc_footers), max_items (int), log
    (callable). The first run for a crawl reads all ~300 footers (~15 MB; ~10 min at the default 2 s
    interval)."""
    yield from itertools.islice(_index(fetcher, options), options.get("max_items"))


def _prefixes(value):
    items = re.split(r"[\s|;]+", value) if isinstance(value, str) else list(value)
    return [p.strip().lower() for p in items if p and p.strip()]


def _mimes(value):
    wanted = set()
    for item in _list(value) or MIMES:
        item = item.lower()
        named = {mime for mime, kind in MIMES.items() if kind == item} | set(MIME_NAMES.get(item, ()))
        wanted |= {item} if "/" in item else named
    return wanted


def _surt_host(host):
    labels = (host or "").lower().strip(".").split(".")
    if len(labels) > 1 and re.fullmatch(r"www\d*", labels[0]):
        labels = labels[1:]
    return ",".join(reversed(labels)) + ")"


def _overlaps(low, high, prefix):
    return high >= prefix and low[:len(prefix)] <= prefix


def _parts(fetcher, base, crawl, cache, log):
    listing = cache / "paths.txt"
    if listing.exists():
        paths = listing.read_text(encoding="utf-8").split()
    else:
        _, body = _get(fetcher, f"{base}crawl-data/{crawl}/cc-index-table.paths.gz")
        text = (gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body).decode("utf-8")
        paths = [p for p in text.split() if "/subset=warc/" in p]
        if paths:
            cache.mkdir(parents=True, exist_ok=True)
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    fetched = transferred = 0
    guess = None
    for path in paths:
        cache_file = cache / (path.rsplit("/", 1)[-1] + ".footer")
        fresh = not cache_file.exists()
        try:
            remote, parquet = _parquet(fetcher, base + path, cache_file, guess)
        except _errors() as error:
            log(f"cc index: no footer for {path}: {getattr(error, 'reason', error)}")
            continue
        guess = cache_file.stat().st_size * 5 // 4  # footers of one crawl are of similar size
        if fresh:
            fetched, transferred = fetched + 1, transferred + remote.transferred
            if fetched % 25 == 0:
                log(f"cc index: {fetched} footers read ({transferred / 1e6:.1f} MB)")
        yield path, remote, parquet


def _index(fetcher, options):
    log = options.get("log") or _quiet
    crawl, base = options.get("crawl", CRAWL), options.get("base") or CC_BASE
    prefixes, mimes = _prefixes(options.get("prefixes", "cn,")), _mimes(options.get("mimes"))
    pattern = re.compile(options["url_regex"], re.I) if options.get("url_regex") else None
    zho = options.get("zho_share") and (float(options["zho_share"]), int(options.get("zho_pages", 5)))
    cache = Path(options.get("cache") or CACHE / "cc_footers").expanduser() / crawl
    groups = []
    for path, remote, parquet in _parts(fetcher, base, crawl, cache, log):
        metadata = parquet.metadata
        surt = metadata.schema.names.index("url_surtkey")
        for group in range(metadata.num_row_groups):
            stats = metadata.row_group(group).column(surt).statistics
            low, high = (stats.min, stats.max) if stats is not None and stats.has_min_max else ("", "\U0010ffff")
            if any(_overlaps(low, high, prefix) for prefix in prefixes):
                groups.append((low, path, group, remote, parquet))
    groups.sort(key=lambda g: (g[0], g[1], g[2]))
    if options.get("seed") is not None:
        random.Random(options["seed"]).shuffle(groups)
    groups = groups[:options.get("max_row_groups", 3) or None]
    log(f"cc index: row groups overlapping {prefixes}: reading {len(groups)}")
    seen = set()
    for _, path, group, remote, parquet in groups:
        before = remote.transferred
        try:
            rows = _documents(remote, parquet, group, prefixes, mimes, zho)
        except _errors() as error:
            log(f"cc index: skipped {path} row group {group}: {getattr(error, 'reason', error)}")
            continue
        log(f"cc index: {path} row group {group}: {len(rows)} documents, "
            f"{(remote.transferred - before) / 1e6:.1f} MB read")
        for row in rows:
            if row["url"] in seen or (pattern and not pattern.search(row["url"])):
                continue
            candidate = _index_candidate(row, crawl, base, options, dict(part=path.rsplit("/", 1)[-1], row_group=group))
            if candidate:
                seen.add(row["url"])
                yield candidate


def _documents(remote, parquet, group, prefixes, mimes, zho=None):
    """Rows of one row group with a wanted MIME under a prefix: host and MIME columns first, the
    url and WARC pointer columns only when some rows remain."""
    import pyarrow as pa
    import pyarrow.compute as pc
    first = _row_group(remote, parquet, group, HOST_COLUMNS + ["content_languages"] * bool(zho))
    wanted = pc.fill_null(pc.is_in(first["content_mime_detected"], value_set=pa.array(sorted(mimes))), False)
    rows = pc.indices_nonzero(wanted).to_pylist()
    names = first["url_host_name"].take(rows).to_pylist()
    picked = [(r, _surt_host(n), n) for r, n in zip(rows, names)]
    picked = [(r, h, n) for r, h, n in picked if any(h.startswith(p[:len(h)]) for p in prefixes)]
    shares = _zho_shares(first, {n for _, _, n in picked}, *zho) if zho and picked else None
    if shares is not None:
        picked = [(r, h, n) for r, h, n in picked if n in shares]
    if not picked:
        return []
    indices = pa.array([r for r, _, _ in picked], pa.int64())
    table = first.take(indices)
    second = _row_group(remote, parquet, group, POINTER_COLUMNS).take(indices)
    for name in POINTER_COLUMNS:
        table = table.append_column(name, second[name])
    documents = []
    for row, (_, host, name) in zip(table.to_pylist(), picked):
        parts = urlsplit(row["url"])
        key = host + (parts.path + ("?" + parts.query if parts.query else "")).lower()
        if any(key.startswith(p) for p in prefixes):  # prefixes that go past the host
            row["host_zho_share"] = shares[name] if shares is not None else None
            documents.append(row)
    return documents


def _zho_shares(table, hosts, share, least):
    """{host: share of its HTML pages tagged zho first} for hosts with ``least`` HTML pages in the row
    group and at least ``share`` of them Chinese."""
    import pyarrow as pa
    import pyarrow.compute as pc
    table = table.filter(pc.is_in(table["url_host_name"], value_set=pa.array(sorted(hosts))))
    html = pc.fill_null(pc.is_in(table["content_mime_detected"], value_set=pa.array(HTML)), False)
    zho = pc.fill_null(pc.and_(html, pc.starts_with(table["content_languages"], "zho")), False)
    counts = pa.table({"host": table["url_host_name"], "html": pc.cast(html, pa.int64()),
                       "zho": pc.cast(zho, pa.int64())}).group_by("host").aggregate([("html", "sum"), ("zho", "sum")])
    names, pages, chinese = (counts[c].to_pylist() for c in ("host", "html_sum", "zho_sum"))
    return {h: round(z / n, 3) for h, n, z in zip(names, pages, chinese) if n >= least and z >= share * n}


def _index_candidate(row, crawl, base, options, where):
    url, mime = row["url"], row["content_mime_detected"]
    name, path = _name(url)
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    kind = MIMES.get(mime)
    if kind is None:
        if extension in NOT_DOCUMENTS:
            return None
        kind = EXTENSIONS.get(extension) or ("docx" if mime.endswith("ooxml") else "doc")
    truncated = row["content_truncated"]
    if PERSONAL.search(path) or (truncated and options.get("skip_truncated")):
        return None
    host = urlsplit(url).hostname or ""
    meta = dict(crawl=crawl, mime=mime, declared_mime=row["content_mime_type"], truncated=truncated, host=host,
                **where)
    if row.get("host_zho_share") is not None:
        meta["host_zho_share"] = row["host_zho_share"]
    candidate = dict(url=url, title=_stem(name) or host, type_hint=kind, license=LICENSE, meta=meta)
    if truncated and options.get("live_truncated"):
        return dict(key=_bare(url), **candidate)
    warc = dict(filename=row["warc_filename"], offset=int(row["warc_record_offset"]),
                length=int(row["warc_record_length"]))
    if base != CC_BASE:
        warc["base"] = base
    return dict(key=f"cc:{warc['filename']}@{warc['offset']}", warc=warc, **candidate)
