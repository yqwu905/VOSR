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
            for chunk in response.iter_content(1 << 16):
                body.write(chunk)
                if stop is not None and stop(chunk):
                    response.close()
                    break
                if body.tell() > limit:
                    response.close()
                    raise FetchError("too_large", url)
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


def download(candidates, store, fetcher, accept=None, limit=None, log=print, refetch_truncated=False):
    """Fetch candidate dicts ({key, url | warc, source, license, ...}); returns counts by status.

    A WARC payload cut at the crawl's size limit is useless; with ``refetch_truncated`` the
    original URL is fetched live instead (robots.txt and pacing apply as usual)."""
    counts = {}
    done = 0
    for candidate in candidates:
        key = candidate.get("key") or candidate["url"]
        if key in store.seen:
            continue
        entry = {k: v for k, v in candidate.items() if k not in ("warc",)}
        entry["key"] = key
        try:
            if candidate.get("warc"):
                w = candidate["warc"]
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
