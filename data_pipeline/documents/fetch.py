"""Polite, resumable downloads into a content-addressed store.

Every live request checks robots.txt (RFC 9309: 4xx means no rules, 5xx or a network
failure means disallow everything), waits for the host's slot (the larger of our
interval and any Crawl-delay), streams at most ``max_bytes`` and backs off on 429/5xx.
Logins, CAPTCHAs and anti-bot challenges are never handled: a 401/403 or an HTML page
where a document was expected is recorded and the URL is not retried.
"""

import hashlib
import io
import json
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

    def _wait(self, host, delay):
        now = self.clock()
        start = max(now, self.next_slot.get(host, now))
        if start > now:
            self.sleep(start - now)
        self.next_slot[host] = start + delay

    def robots_for(self, url):
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            self._wait(parts.netloc, self.interval)
            try:
                response = self.http.get(origin + "/robots.txt", timeout=self.timeout)
                if response.status_code >= 500:
                    rules = DISALLOW_ALL
                elif response.status_code >= 400:
                    rules = ALLOW_ALL
                else:
                    rules = Robots(response.text, self.agent)
            except Exception:  # unreachable robots.txt: assume complete disallow
                rules = DISALLOW_ALL
            self.robots[origin] = rules
        return self.robots[origin]

    def get(self, url, method="GET", check_robots=True, stream_limit=None, max_redirects=5, **kwargs):
        """One polite request; returns (response, body bytes) or raises FetchError.

        Redirects are followed by hand so every hop passes its own host's robots.txt."""
        for _ in range(max_redirects + 1):
            response, body = self._get_once(url, method, check_robots, stream_limit, **kwargs)
            if response.is_redirect and response.headers.get("Location"):
                url = urljoin(url, response.headers["Location"])
                if response.status_code in (301, 302, 303):
                    method, kwargs = "GET", {k: v for k, v in kwargs.items() if k not in ("data", "json")}
                continue
            return response, body
        raise FetchError("too_many_redirects", url)

    def _get_once(self, url, method, check_robots, stream_limit, **kwargs):
        host = urlsplit(url).netloc
        rules = self.robots_for(url) if check_robots else ALLOW_ALL
        if not rules.allowed(url):
            raise FetchError("robots_disallow", url)
        delay = max(self.interval, rules.crawl_delay or 0)
        limit = stream_limit or self.max_bytes
        for attempt in range(self.retries + 1):
            self._wait(host, delay)
            try:
                response = self.http.request(method, url, timeout=self.timeout, stream=True,
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
            body = io.BytesIO()
            for chunk in response.iter_content(1 << 16):
                body.write(chunk)
                if body.tell() > limit:
                    response.close()
                    raise FetchError("too_large", url)
            return response, body.getvalue()

    def json(self, url, method="GET", **kwargs):
        return json.loads(self.get(url, method=method, **kwargs)[1])


class FetchError(Exception):
    def __init__(self, reason, url):
        super().__init__(f"{reason}: {url}")
        self.reason, self.url = reason, url


def warc_record(fetcher, filename, offset, length):
    """Payload of one Common Crawl WARC record via an HTTP range read (no live crawl)."""
    from warcio.archiveiterator import ArchiveIterator
    end = int(offset) + int(length) - 1
    _, raw = fetcher.get(CC_BASE + filename, check_robots=False, headers={"Range": f"bytes={offset}-{end}"})
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
        digest = hashlib.sha256(data).hexdigest()
        kind = sniff(data)
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


def download(candidates, store, fetcher, accept=None, limit=None, log=print):
    """Fetch candidate dicts ({key, url | warc, source, license, ...}); returns counts by status."""
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
                data, meta = warc_record(fetcher, w["filename"], w["offset"], w["length"])
                entry.update(meta)
            else:
                response, data = fetcher.get(candidate["url"])
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
