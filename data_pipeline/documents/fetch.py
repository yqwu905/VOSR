"""Polite, resumable downloads into a content-addressed store.

Every live request checks robots.txt (RFC 9309: 404-style answers mean no rules; 5xx or a
network failure means disallow everything; we also treat 401/403 on robots.txt as
disallow, since that is how WAFs answer unwanted crawlers), waits for the host's slot
(the larger of our interval and any Crawl-delay), streams at most ``max_bytes`` and
backs off on 429/5xx. Logins, CAPTCHAs and anti-bot challenges are never handled: a
401/403 or an HTML page where a document was expected is recorded and not retried.

Some Chinese hosts reset Python's TLS handshake through the egress proxy while curl gets
through; such hosts switch to a curl transport (same pacing, same robots rules).
"""

import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from ..pipeline import json_text
from .robots import ALLOW_ALL, DISALLOW_ALL, Robots
from .sniff import sniff

AGENT = "VOSR-research-crawler/0.1 (text super-resolution research)"
CC_BASE = "https://data.commoncrawl.org/"


def session(agent=AGENT, contact=None):
    import requests
    s = requests.Session()
    # The contact is whatever the operator configures (e.g. a project URL); none by default.
    s.headers["User-Agent"] = agent + (f" +{contact}" if contact else "")
    s.headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.5"
    return s


class Fetcher:
    def __init__(self, http=None, interval=2.0, max_bytes=200 << 20, timeout=60, retries=3, agent=AGENT,
                 clock=time.monotonic, sleep=time.sleep):
        self.http = http or session(agent)
        self.agent, self.interval, self.max_bytes = agent, interval, max_bytes
        self.timeout, self.retries = timeout, retries
        self.clock, self.sleep = clock, sleep
        self.robots, self.next_slot = {}, {}
        self.curl_hosts = set()

    def _request(self, host, method, url, **kwargs):
        if host in self.curl_hosts:
            return curl_request(method, url, headers=dict(self.http.headers, **kwargs.pop("headers", {})),
                                max_bytes=self.max_bytes, **kwargs)
        try:
            return self.http.request(method, url, **kwargs)
        except Exception as error:
            if not _tls_failure(error) or not hasattr(self.http, "headers"):
                raise
            self.curl_hosts.add(host)
            return self._request(host, method, url, **kwargs)

    def _wait(self, host, delay):
        now = self.clock()
        start = max(now, self.next_slot.get(host, now))
        if start > now:
            self.sleep(start - now)
        self.next_slot[host] = start + delay

    def robots_for(self, url, recheck=600):
        """Rules for the URL's origin. A server error or unreachable robots.txt means
        disallow, but only for ``recheck`` seconds: a transient reset must not ban a host."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        cached = self.robots.get(origin)
        if cached and (cached[1] is None or self.clock() < cached[1]):
            return cached[0]
        rules, expires = DISALLOW_ALL, self.clock() + recheck
        for attempt in range(self.retries + 1):
            self._wait(parts.netloc, self.interval)
            try:
                response = self._request(parts.netloc, "GET", origin + "/robots.txt", timeout=self.timeout,
                                         stream=False, allow_redirects=True)
            except Exception:
                self.sleep(2 ** attempt * self.interval)
                continue
            if response.status_code >= 500:
                self.sleep(2 ** attempt * self.interval)
                continue
            expires = None
            if response.status_code in (401, 403):
                rules = DISALLOW_ALL
            elif response.status_code >= 400:
                rules = ALLOW_ALL
            else:
                rules = Robots(response.text, self.agent)
            break
        self.robots[origin] = (rules, expires)
        return rules

    def get(self, url, method="GET", check_robots=True, stream_limit=None, max_redirects=5, stop=None, **kwargs):
        """One polite request; returns (response, body bytes) or raises FetchError.

        Redirects are followed by hand so every hop passes its own host's robots.txt.
        ``stop(chunk)`` returning True ends the body early (open-ended range reads)."""
        for _ in range(max_redirects + 1):
            response, body = self._get_once(url, method, check_robots, stream_limit, stop=stop, **kwargs)
            if response.is_redirect and response.headers.get("Location"):
                url = urljoin(url, response.headers["Location"])
                if response.status_code in (301, 302, 303):
                    method, kwargs = "GET", {k: v for k, v in kwargs.items() if k not in ("data", "json")}
                continue
            return response, body
        raise FetchError("too_many_redirects", url)

    def _get_once(self, url, method, check_robots, stream_limit, stop=None, **kwargs):
        host = urlsplit(url).netloc
        rules = self.robots_for(url) if check_robots else ALLOW_ALL
        if not rules.allowed(url):
            raise FetchError(rules.reason(url), url)
        delay = max(self.interval, rules.crawl_delay or 0)
        limit = stream_limit or self.max_bytes
        for attempt in range(self.retries + 1):
            self._wait(host, delay)
            try:
                response = self._request(host, method, url, timeout=self.timeout, stream=True,
                                         allow_redirects=False, **kwargs)
            except Exception as error:
                if attempt == self.retries:
                    raise FetchError("network:" + type(error).__name__, url) from error
                self.sleep(2 ** attempt * delay)
                continue
            if response.is_redirect:
                response.close()
                return response, b""
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After", "")
                response.close()
                if attempt == self.retries:
                    raise FetchError(f"http_{response.status_code}", url)
                self.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1) * delay)
                continue
            if response.status_code >= 400:
                response.close()
                raise FetchError(f"http_{response.status_code}", url)
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > limit and "Range" not in kwargs.get("headers", {}):
                response.close()
                raise FetchError("too_large", url)
            body = io.BytesIO()
            try:
                for chunk in response.iter_content(1 << 16):
                    body.write(chunk)
                    if stop is not None and stop(chunk):
                        response.close()
                        break
                    if body.tell() > limit:
                        response.close()
                        raise FetchError("too_large", url)
            except FetchError:
                raise
            except Exception as error:  # connection broken mid-body (IncompleteRead, ChunkedEncodingError)
                response.close()
                if attempt == self.retries:
                    raise FetchError("network:" + type(error).__name__, url) from error
                self.sleep(2 ** attempt * delay)
                continue
            return response, body.getvalue()

    def json(self, url, method="GET", **kwargs):
        return json.loads(self.get(url, method=method, **kwargs)[1])


def _tls_failure(error):
    text = f"{type(error).__name__} {error}"
    return any(k in text for k in ("SSLError", "SSLEOFError", "UNEXPECTED_EOF", "HANDSHAKE_FAILURE", "ProxyError"))


class CurlResponse:
    def __init__(self, status, headers, body, url):
        self.status_code, self.headers, self._body, self.url = status, headers, body, url
        self.text = body.decode("utf-8", "replace")

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


def curl_request(method, url, headers=None, data=None, json=None, timeout=60, max_bytes=200 << 20,
                 allow_redirects=False, stream=True, retries=3):
    """The same request through the curl binary (which honours the proxy environment)."""
    import requests.structures
    from urllib.parse import urlencode
    with tempfile.TemporaryDirectory() as tmp:
        head, body = os.path.join(tmp, "h"), os.path.join(tmp, "b")
        command = ["curl", "-sS", "-X", method, "-D", head, "-o", body, "--max-time", str(timeout),
                   "--max-filesize", str(max_bytes), "--retry", str(retries), "--retry-all-errors",
                   "--retry-delay", "2", "--compressed"]
        if allow_redirects:
            command.append("-L")
        for key, value in (headers or {}).items():
            command += ["-H", f"{key}: {value}"]
        payload = None
        if json is not None:
            payload = __import__("json").dumps(json).encode()
            command += ["-H", "Content-Type: application/json"]
        elif data is not None:
            payload = (urlencode(data) if isinstance(data, dict) else data)
            payload = payload.encode() if isinstance(payload, str) else payload
            command += ["-H", "Content-Type: application/x-www-form-urlencoded"]
        if payload is not None:
            command += ["--data-binary", "@-"]
        result = subprocess.run(command + [url], input=payload, capture_output=True, timeout=timeout * (retries + 2))
        if result.returncode == 63:
            raise FetchError("too_large", url)
        if result.returncode != 0:
            raise ConnectionError(f"curl exit {result.returncode}: {result.stderr.decode(errors='replace')[:200]}")
        blocks = [b for b in open(head, "rb").read().decode("latin-1").split("\r\n\r\n") if b.strip()]
        lines = blocks[-1].split("\r\n")  # the proxy CONNECT reply comes first
        fields = requests.structures.CaseInsensitiveDict()
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        return CurlResponse(int(lines[0].split()[1]), fields, open(body, "rb").read(), url)


class FetchError(Exception):
    def __init__(self, reason, url):
        super().__init__(f"{reason}: {url}")
        self.reason, self.url = reason, url


def warc_record(fetcher, filename, offset, length=None, base=CC_BASE):
    """Payload of one Common Crawl WARC record via an HTTP range read (no live crawl).

    Without a length (FinePDFs gives offsets only) the range is open-ended and reading
    stops at the end of the record's own gzip member. ``base`` may also be the Hugging
    Face mirror of 2026+ crawls. data.commoncrawl.org is Common Crawl's documented HTTPS
    download endpoint for the dataset; its robots.txt (Disallow: /) targets crawlers of
    the site, so dataset reads skip the robots check.
    """
    import zlib
    from warcio.archiveiterator import ArchiveIterator
    if length:
        end = int(offset) + int(length) - 1
        _, raw = fetcher.get(base + filename, check_robots=False, headers={"Range": f"bytes={offset}-{end}"})
    else:
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        plain = io.BytesIO()

        def complete(chunk):
            if not member.eof:
                plain.write(member.decompress(chunk))
            return member.eof

        fetcher.get(base + filename, check_robots=False, headers={"Range": f"bytes={offset}-"}, stop=complete)
        if not member.eof:
            raise FetchError("warc_incomplete", filename)
        raw = plain.getvalue()
    for record in ArchiveIterator(io.BytesIO(raw)):
        if record.rec_type == "response":
            truncated = record.rec_headers.get_header("WARC-Truncated")
            status = record.http_headers.get_statuscode() if record.http_headers else None
            return record.content_stream().read(), {"warc_truncated": truncated, "http_status": status}
    raise FetchError("warc_no_response", filename)


class RangeFile(io.RawIOBase):
    """Read-only remote file over exact HTTP Range requests (for pyarrow/zip footers).

    The first read resolves redirects (e.g. Hugging Face resolve -> signed CDN URL) and
    the size from Content-Range; later reads go to the resolved URL. There is no
    read-ahead: a buffered wrapper with a large buffer turns every seek into megabytes.
    """

    def __init__(self, fetcher, url, check_robots=True):
        super().__init__()
        self.fetcher, self.url, self.check_robots = fetcher, url, check_robots
        self.position, self.size, self.requests, self.bytes = 0, None, 0, 0

    def _range(self, start, end):
        response, body = self.fetcher.get(self.url, check_robots=self.check_robots,
                                          headers={"Range": f"bytes={start}-{end}"})
        self.requests, self.bytes = self.requests + 1, self.bytes + len(body)
        if response.status_code != 206:
            raise FetchError("range_unsupported", self.url)
        self.url = response.url or self.url
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if total.isdigit():
            self.size = int(total)
        return body

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_END and self.size is None:
            self._range(0, 0)
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        if self.size is None:
            self._range(0, 0)
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        body = self._range(self.position, end - 1)
        self.position += len(body)
        return body

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


PAGE_SPAN = 64 << 20  # adjacent data pages are fetched in one range read up to this size
PAGE_VALUES = 1 << 20  # pages declaring more values are left to pyarrow
PAGE_CODECS = {"UNCOMPRESSED", "SNAPPY", "GZIP", "BROTLI", "ZSTD", "LZ4_RAW"}


def parquet_blob(fetcher, reference, cache):
    """Bytes stored in a parquet cell: {url, row_group, row, column, item?, rows?}.

    ``cache`` keeps the last file's footer and the cells read from the last row group, so
    candidates should come grouped. A byte-array column chunk with an offset index (page
    index) is read by data page: only the pages holding ``rows`` (the rows of the group
    that will be asked for; default ``row``), plus the dictionary page when they use it,
    and only ``item`` of each list is kept. Otherwise the whole column chunk is read with
    pyarrow, i.e. every list item of every row of the group. A failed read fails all the
    rows it was for, so a chunk over the size limit is not downloaded once per row. The
    file is untrusted: sizes and counts it declares are bounded before anything is
    allocated, and whatever goes wrong while decoding it becomes a FetchError."""
    url, row = reference["url"], reference["row"]
    key = (url, reference["row_group"], reference["column"], reference.get("item"))
    if cache.get("key") != key:
        cache.update(key=key, cells={})
    cells = cache["cells"]
    if row not in cells:
        wanted = set(reference.get("rows") or ()) - set(cells) | {row}
        try:
            handle = RangeFile(fetcher, url)
            if cache.get("url") != url:
                cache.update(url=url, footer=_footer(handle))
            cells.update(_parquet_cells(handle, *cache["footer"], reference, wanted))
        except FetchError as error:
            cells.update(dict.fromkeys(wanted, error))
        except Exception as error:  # malformed file: pyarrow errors, bad offsets, MemoryError...
            cells.update(dict.fromkeys(wanted, FetchError("parquet_" + type(error).__name__, url)))
    value = cells.get(row)
    if isinstance(value, FetchError):
        raise value
    if not value:
        raise FetchError("empty_cell", url)
    return value


def _footer(handle):
    """(pyarrow metadata, row groups as Thrift dicts or None) from the footer, in 2-3 reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    size = handle.seek(0, io.SEEK_END)
    handle.seek(max(0, size - min(64 << 10, handle.fetcher.max_bytes)))
    tail = handle.read()
    length = int.from_bytes(tail[-8:-4], "little") + 8
    if tail[-4:] != b"PAR1" or length > size:
        raise FetchError("not_parquet", handle.url)
    if length > handle.fetcher.max_bytes:
        raise FetchError("too_large", handle.url)
    if length > len(tail):
        handle.seek(size - length)
        tail = handle.read(length - len(tail)) + tail
    try:
        groups = _thrift(tail[-length:])[0][4]
    except (ValueError, IndexError, KeyError, RecursionError):
        groups = None
    return pq.read_metadata(pa.BufferReader(tail[-length:])), groups


def _parquet_cells(handle, metadata, groups, reference, wanted):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    group, item, name = reference["row_group"], reference.get("item"), reference["column"]
    leaves = [j for j in range(metadata.num_columns) if metadata.schema.column(j).path.split(".")[0] == name]
    if len(leaves) == 1 and groups:
        leaf, chunk = metadata.schema.column(leaves[0]), metadata.row_group(group).column(leaves[0])
        found = groups[group][1][leaves[0]]  # ColumnChunk: 4, 5 = offset index offset, length
        if (4 in found and leaf.physical_type == "BYTE_ARRAY" and chunk.compression in PAGE_CODECS
                and leaf.max_repetition_level == (0 if item is None else 1)):
            try:
                return _paged(handle, metadata.row_group(group).num_rows, chunk, found, leaf, wanted, item)
            except NotImplementedError:
                pass
    if sum(metadata.row_group(group).column(j).total_compressed_size for j in leaves) > handle.fetcher.max_bytes:
        raise FetchError("too_large", handle.url)
    table = pq.ParquetFile(pa.PythonFile(handle, mode="r"), metadata=metadata).read_row_group(group, columns=[name])
    if item is None:
        return dict(enumerate(table.column(name).to_pylist()))
    return {row: value[0] if value else None
            for row, value in enumerate(pc.list_slice(table.column(name), item, item + 1).to_pylist())}


def _paged(handle, rows, chunk, found, leaf, wanted, item):
    def read(start, end):
        if not 0 <= start < end:
            raise ValueError("bad page location")
        if end - start > handle.fetcher.max_bytes:
            raise FetchError("too_large", handle.url)
        handle.seek(start)
        return handle.read(end - start)

    def unpack(data, position, header):
        if not 0 <= header[2] <= handle.fetcher.max_bytes:
            raise FetchError("too_large", handle.url)
        return _decompress(data[position:position + header[3]], header[2], chunk.compression)

    pages = _thrift(read(found[4], found[4] + found[5]))[0][1]  # PageLocation: offset, size, first row
    spans = []  # [start, end, first row, end row, [(page offset, first row)]]
    for page, end in zip(pages, [page[3] for page in pages[1:]] + [rows]):
        start, stop, first = page[1], page[1] + page[2], page[3]
        if not any(first <= row < end for row in wanted):
            continue
        if spans and spans[-1][1] == start and stop - spans[-1][0] <= min(PAGE_SPAN, handle.fetcher.max_bytes):
            spans[-1][1], spans[-1][3] = stop, end
            spans[-1][4].append((start, first))
        else:
            spans.append([start, stop, first, end, [(start, first)]])
    dictionary = []

    def values():
        if not dictionary:
            start = chunk.dictionary_page_offset if chunk.has_dictionary_page else chunk.data_page_offset
            data = read(start, pages[0][1])
            header, position = _thrift(data)
            if header[1] != 2:
                raise NotImplementedError("no dictionary page")
            page, position, offsets = unpack(data, position, header), 0, []
            for _ in range(header[7][1]):
                end = position + 4 + int.from_bytes(page[position:position + 4], "little")
                if end > len(page):
                    raise ValueError("dictionary page overrun")
                offsets.append((position + 4, end))
                position = end
            dictionary.append((page, offsets))
        return dictionary[0]

    cells, levels = {}, (leaf.max_repetition_level, leaf.max_definition_level)
    for start, stop, first, end, members in spans:
        try:
            data = read(start, stop)
            for offset, row in members:
                cells.update(_data_page(data, offset - start, row, levels, item, wanted, unpack, values))
        except FetchError as error:  # e.g. a page or the dictionary over the size limit
            cells.update({row: error for row in wanted if first <= row < end and row not in cells})
    return cells


def _data_page(data, position, row, levels, item, wanted, unpack, dictionary):
    """Cells of the wanted rows in one v1 data page of a byte-array column (PLAIN or dictionary)."""
    header, position = _thrift(data, position)
    if header[1] != 0 or header[5].get(3, 3) != 3 or header[5].get(4, 3) != 3:
        raise NotImplementedError("page type or level encoding")
    count, encoding = header[5][1], header[5][2]
    if not 0 <= count <= PAGE_VALUES:
        raise NotImplementedError("page value count")
    page = unpack(data, position, header)
    (max_rep, max_def), position = levels, 0
    reps, defs = [0] * count, [max_def] * count
    if max_rep:
        reps, position = _hybrid(page, 4, max_rep.bit_length(), count), 4 + int.from_bytes(page[:4], "little")
    if max_def:
        defs = _hybrid(page, position + 4, max_def.bit_length(), count)
        position += 4 + int.from_bytes(page[position:position + 4], "little")
    if encoding in (2, 8):
        values, offsets = dictionary()
        indices = iter(_hybrid(page, position + 1, page[position], defs.count(max_def)))
    elif encoding:
        raise NotImplementedError(f"encoding {encoding}")
    cells, row, slot = {}, row - 1, 0
    for rep, level in zip(reps, defs):
        row, slot = (row + 1, 0) if rep == 0 else (row, slot + 1)
        keep = row in wanted and slot == (item or 0)
        if level < max_def:
            if keep:
                cells[row] = None
        elif encoding:
            start, end = offsets[next(indices)]
            if keep:
                cells[row] = values[start:end]
        else:
            end = position + 4 + int.from_bytes(page[position:position + 4], "little")
            if end > len(page):
                raise ValueError("data page overrun")
            if keep:
                cells[row] = page[position + 4:end]
            position = end
    return cells


def _decompress(data, size, codec):
    import pyarrow as pa
    return bytes(data) if codec == "UNCOMPRESSED" else pa.decompress(data, size, codec=codec.lower(), asbytes=True)


def _hybrid(data, position, width, count):
    """``count`` values in parquet's RLE/bit-packed hybrid encoding (levels, dictionary indices).
    A run longer than the values still needed (bit-packed: rounded up to a group of 8) is
    malformed and refused before anything is allocated for it."""
    values, mask = [], (1 << width) - 1
    while len(values) < count:
        head, position = _varint(data, position)
        run, need = head >> 1, count - len(values)
        if not run or (run * 8 >= need + 8 if head & 1 else run > need):
            raise ValueError("parquet run past the page's values")
        if head & 1:
            size = run * width
            bits = int.from_bytes(data[position:position + size], "little")
            values += [bits >> (k * width) & mask for k in range(min(run * 8, need))]
        else:
            size = (width + 7) // 8
            values += [int.from_bytes(data[position:position + size], "little")] * run
        position += size
    return values


def _varint(data, position):
    value = shift = 0
    while True:
        byte, position = data[position], position + 1
        value, shift = value | (byte & 0x7F) << shift, shift + 7
        if byte < 0x80:
            return value, position


def _thrift(data, position=0):
    """One Thrift compact-protocol struct (footer, offset index, page header) as
    ({field id: value}, end); lists become lists and structs dicts."""
    fields, field = {}, 0
    while data[position]:
        head, position = data[position], position + 1
        if head >> 4:
            field += head >> 4
        else:
            field, position = _varint(data, position)
            field = field >> 1 ^ -(field & 1)
        fields[field], position = _thrift_value(data, position, head & 15)
    return fields, position + 1


def _thrift_value(data, position, kind):
    if kind in (1, 2):
        return kind == 1, position
    if kind == 3:
        return data[position], position + 1
    if kind in (4, 5, 6):
        value, position = _varint(data, position)
        return value >> 1 ^ -(value & 1), position
    if kind in (7, 8):
        size, position = (8, position) if kind == 7 else _varint(data, position)
        if position + size > len(data):
            raise ValueError("thrift value past the end")
        return data[position:position + size], position + size
    if kind in (9, 10):
        size, element, position = data[position] >> 4, data[position] & 15, position + 1
        if size == 15:
            size, position = _varint(data, position)
        items = []
        for _ in range(size):
            value, position = (data[position] == 1, position + 1) if element in (1, 2) else \
                _thrift_value(data, position, element)
            items.append(value)
        return items, position
    if kind == 12:
        return _thrift(data, position)
    raise ValueError(f"thrift type {kind}")


class Store:
    """files/<sha[:2]>/<sha>.<type> plus an append-only fetch log for resuming."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "files").mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "fetch_log.jsonl"
        self.seen = {}
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.seen[entry["key"]] = entry

    def put(self, data):
        kind = sniff(data)
        if kind == "pdf" and not data.startswith(b"%PDF-"):
            # Some repositories store the upload with its multipart wrapper around the PDF.
            start, end = data.find(b"%PDF-"), data.rfind(b"%%EOF")
            data = data[start:end + 5] if end > start else data[start:]
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "files" / digest[:2] / f"{digest}.{kind}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(path)
        return digest, kind, path

    def record(self, entry):
        self.seen[entry["key"]] = entry
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json_text(entry) + "\n")


def download(candidates, store, fetcher, accept=None, limit=None, log=print, refetch_truncated=False,
             cc_data_host=False):
    """Fetch candidate dicts ({key, url | warc, source, license, ...}); returns counts by status.

    A WARC payload cut at the crawl's size limit is useless; with ``refetch_truncated`` the
    original URL is fetched live instead (robots.txt and pacing apply as usual).
    WARC records on data.commoncrawl.org are read only with ``cc_data_host``: that host's
    robots.txt disallows everything although Common Crawl documents it as the HTTPS
    download endpoint, so using it is an explicit operator decision. Without it such
    candidates fall back to their original URL."""
    counts = {}
    done = 0
    parquet_cache = {}
    for candidate in candidates:
        key = candidate.get("key") or candidate["url"]
        if key in store.seen:
            continue
        entry = {k: v for k, v in candidate.items() if k not in ("warc", "parquet")}
        entry["key"] = key
        try:
            w = candidate.get("warc")
            if w and w.get("base", CC_BASE) == CC_BASE and not cc_data_host:
                if not candidate.get("url"):
                    raise FetchError("cc_data_host_disabled", w["filename"])
                w = None
                entry["via"] = "origin"
            if candidate.get("parquet"):
                data = parquet_blob(fetcher, candidate["parquet"], parquet_cache)
                entry["via"] = "parquet"
            elif w:
                data, meta = warc_record(fetcher, w["filename"], w["offset"], w.get("length"),
                                         base=w.get("base", CC_BASE))
                entry.update(meta)
                if meta["warc_truncated"]:
                    if not (refetch_truncated and candidate.get("url")):
                        raise FetchError("warc_truncated", candidate.get("url") or w["filename"])
                    response, data = fetcher.get(candidate["url"], headers=candidate.get("headers") or {})
                    entry.update(final_url=response.url, refetched=True)
            else:
                response, data = fetcher.get(candidate["url"], headers=candidate.get("headers") or {})
                entry.update(final_url=response.url, content_type=response.headers.get("Content-Type"))
            digest, kind, path = store.put(data)
            entry.update(status="ok" if accept is None or kind in accept else "unwanted_type",
                         sha256=digest, type=kind, bytes=len(data), path=str(path))
        except FetchError as error:
            entry.update(status="error", reason=error.reason)
        entry["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        store.record(entry)
        status = entry["status"] if entry["status"] != "error" else entry["reason"]
        counts[status] = counts.get(status, 0) + 1
        done += 1
        if log and done % 10 == 0:
            log(f"{done} fetched: {counts}")
        if limit and counts.get("ok", 0) >= limit:
            break
    return counts
