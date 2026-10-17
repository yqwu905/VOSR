"""Polite, resumable downloads into a content-addressed store.

Every live request checks robots.txt (RFC 9309: 404-style answers mean no rules; 429, 5xx
or a network failure means disallow everything for a while, refused as robots_unavailable;
we also treat 401/403 on robots.txt as disallow, since that is how WAFs answer unwanted
crawlers), waits for the host's slot (the larger of our interval and any Crawl-delay),
streams at most ``max_bytes`` and backs off on 429/5xx. Logins, CAPTCHAs and anti-bot
challenges are never handled: a 401/403 or an HTML page where a document was expected is
recorded and not retried.

robots.txt is read as UTF-8 (or, when it is not valid UTF-8, in the charset the server
declares, e.g. GBK), only its first 500 KiB (RFC 9309 2.5, on the curl transport too), with at most five
redirects followed by hand. Of the line that limit cuts, what precedes a "#" is kept (a cut comment
does not change the rule) and so is a Disallow path followed by whitespace; otherwise the line is
dropped (its value may be cut short, and an Allow or User-agent is not kept on whitespace alone: the
conservative reading). One narrow exception to the 401/403 rule: a signed-URL CDN
refuses every URL that lacks its signature, robots.txt included, so a 401/403 carrying
Alibaba Cloud CDN's ``X-Tengine-Error: denied by req auth...`` (ModelScope's LFS host)
means robots.txt is unavailable, i.e. allow-all as RFC 9309 2.3.1.3 has it for 4xx; the
decision is logged as a warning.

Every hop, redirects and robots.txt included, must be http(s), off ``catalog.BLOCKED_HOSTS``
and on a public address: hosts that are, or resolve to, loopback, link-local, private,
multicast, reserved or unspecified addresses are refused, as are special-use names
(localhost, .local, .internal, ...). The host checked is the one the transports contact (no root
dot): a non-ASCII host name (which requests maps by IDNA 2008 and curl by UTS 46, to different hosts;
no source uses one), a backslash in the authority or a %-escape in the host (which urlsplit,
urllib3 and curl read differently) and a malformed port are refused as bad_url. A name that does
not resolve locally passes when it looks public, since the egress proxy resolves names itself.
Without a proxy (a name may resolve differently a moment after the check), ``session()``
connections also check the address actually connected to, and the curl transport resolves the name
again, checks every address and pins curl to them (also through a socks4:// or socks5:// proxy,
which leaves names to curl); any other proxy resolves names itself. ``allow_private`` is for tests.

No single wait exceeds ``MAX_WAIT`` seconds: a longer Retry-After (seconds or an HTTP-date)
fails the request for this run, a host whose Crawl-delay is longer is refused
(crawl_delay_too_long) and backoff sleeps are capped. On ``session()`` connections the status
line and headers must arrive within ``timeout`` seconds; a body within ``timeout`` seconds plus
one second per ``MIN_RATE`` bytes received (it may not average below 4 KiB/s once past the first
``timeout`` seconds). Otherwise the request fails as too_slow (not retried in this run; a resume
tries it again). Both deadlines, robots.txt reads included, shut the socket down when they pass,
so no loop below the Fetcher (interim responses, trailer lines, a decoder fed empty deflate
blocks) outlives them.

Some Chinese hosts reset Python's TLS handshake through the egress proxy while curl gets
through; a host whose request fails in TLS (an SSL error in the exception's cause chain)
switches to a curl transport (same pacing, robots rules and address checks; curl itself never
retries, reads no .curlrc, takes the proxy requests would use for the URL, and its --max-time
bounds the whole request).
"""

import contextlib
import email.utils
import hashlib
import io
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import zipfile
from datetime import timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from ..pipeline import json_text
from .catalog import blocked
from .robots import ALLOW_ALL, DISALLOW_ALL, Robots
from .sniff import sniff

AGENT = "VOSR-research-crawler/0.1 (text super-resolution research)"
CC_BASE = "https://data.commoncrawl.org/"
ROBOTS_LIMIT = 500 << 10
MAX_WAIT = 600
MIN_RATE = 4 << 10  # bytes/s a body must average once past the first `timeout` seconds
SPECIAL_NAMES = ("localhost", "localdomain", "local", "internal", "home.arpa")
LOG = logging.getLogger(__name__)
UNAVAILABLE = Robots("User-agent: *\nDisallow: /", "-")  # robots.txt unreachable for now


def session(agent=AGENT, contact=None):
    import requests
    s = requests.Session()
    # The contact is whatever the operator configures (e.g. a project URL); none by default.
    s.headers["User-Agent"] = agent + (f" +{contact}" if contact else "")
    s.headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.5"
    adapter = _adapter()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _adapter():
    """A requests adapter whose connections refuse a direct connection to a non-global address (a
    name that passed Fetcher._guard may resolve elsewhere a moment later; a proxy resolves names
    itself, so proxied connections are left to it) and give up when the status line and headers
    take longer than the read timeout, which http.client applies to each socket read only."""
    import requests.adapters
    from urllib3 import connection, connectionpool, poolmanager

    class Guarded:
        def _new_conn(self):
            sock = super()._new_conn()
            peer = ipaddress.ip_address(sock.getpeername()[0].split("%")[0])
            if not (getattr(self, "proxy", None) or adapter.allow_private or _global(peer)):
                sock.close()
                raise FetchError("private_address", f"{self.host}:{self.port}")
            return sock

        def getresponse(self, *args, **kwargs):
            sock = self.sock  # http.client drops it here when the response ends the connection
            if not isinstance(self.timeout, (int, float)):
                return super().getresponse(*args, **kwargs)
            end = time.monotonic() + self.timeout
            with _deadline(sock, lambda: end - time.monotonic()) as expired:
                try:
                    response = super().getresponse(*args, **kwargs)
                except Exception:
                    if not expired.is_set():
                        raise
            if expired.is_set():  # cut short: whatever http.client made of it must not be used
                raise FetchError("too_slow", f"{self.host}:{self.port}")
            response.guarded_sock = sock  # for Fetcher._read's body deadline
            return response

    pools = {scheme: type(pool.__name__, (pool,), {"ConnectionCls": type(conn.__name__, (Guarded, conn), {})})
             for scheme, pool, conn in (("http", connectionpool.HTTPConnectionPool, connection.HTTPConnection),
                                        ("https", connectionpool.HTTPSConnectionPool, connection.HTTPSConnection))}

    class Adapter(requests.adapters.HTTPAdapter):
        allow_private = False

        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = pools

        def proxy_manager_for(self, *args, **kwargs):
            manager = super().proxy_manager_for(*args, **kwargs)
            if manager.pool_classes_by_scheme is poolmanager.pool_classes_by_scheme:  # not SOCKS
                manager.pool_classes_by_scheme = pools
            return manager

    adapter = Adapter()
    return adapter


@contextlib.contextmanager
def _deadline(sock, remaining):
    """Shut ``sock`` down once ``remaining()`` seconds are used up (asked again whenever they run out,
    as a body's allowance grows with its bytes), so that no read on it outlives the deadline however
    http.client or a decoder loops over it (interim responses, trailer lines, empty deflate blocks).
    Yields an Event that is set when that happened; without a socket to shut down it only yields."""
    expired, finished = threading.Event(), threading.Event()
    shutdown = getattr(sock, "shutdown", None)

    def watch():
        while not finished.wait(max(remaining(), 0)):
            if remaining() <= 0:
                expired.set()
                with contextlib.suppress(OSError):
                    shutdown(socket.SHUT_RDWR)
                return

    watcher = threading.Thread(target=watch, daemon=True) if shutdown else None
    if watcher:
        watcher.start()
    try:
        yield expired
    finally:
        finished.set()
        if watcher:
            watcher.join()


class Fetcher:
    def __init__(self, http=None, interval=2.0, max_bytes=200 << 20, timeout=60, retries=3, agent=AGENT,
                 clock=time.monotonic, sleep=time.sleep, allow_private=False, resolve=socket.getaddrinfo):
        self.http = http or session(agent)
        self.agent, self.interval, self.max_bytes = agent, interval, max_bytes
        self.timeout, self.retries = timeout, retries
        self.clock, self.sleep = clock, sleep
        self.allow_private, self.resolve, self.public = allow_private, resolve, {}
        self.robots, self.next_slot = {}, {}
        self.curl_hosts = set()
        for adapter in getattr(self.http, "adapters", {}).values():  # session()'s connect-time address check
            if hasattr(adapter, "allow_private"):
                adapter.allow_private = allow_private

    def _request(self, host, method, url, limit=None, truncate=False, **kwargs):
        """``limit`` and ``truncate`` matter to curl only, which is read as it runs (see curl_request)."""
        if host in self.curl_hosts:
            return curl_request(method, url, headers=dict(self.http.headers, **kwargs.pop("headers", {})),
                                max_bytes=limit or self.max_bytes, truncate=truncate, resolve=self.resolve,
                                allow_private=self.allow_private, **kwargs)
        try:
            return self.http.request(method, url, **kwargs)
        except Exception as error:
            if not _tls_failure(error) or not hasattr(self.http, "headers"):
                raise
            self.curl_hosts.add(host)
            return self._request(host, method, url, limit, truncate, **kwargs)

    def _wait(self, host, delay):
        now = self.clock()
        start = max(now, self.next_slot.get(host, now))
        if start > now:
            self.sleep(start - now)
        self.next_slot[host] = start + delay

    def _guard(self, url):
        """Refuse a hop that is not http(s), is on a blocked host or is not on a public address."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise FetchError("unsupported_scheme", url)
        try:  # the name requests and curl look up, no root dot (the codec refuses empty or overlong labels)
            # netloc, not the lower-cased hostname: KELVIN SIGN lower-cases to an ASCII "k"
            host = parts.hostname.encode("idna").decode("ascii").rstrip(".") if parts.netloc.isascii() else ""
            if parts.port == 0:  # parts.port raises ValueError unless the port is a number in range
                host = ""
        except (UnicodeError, ValueError):
            host = ""
        # requests (IDNA 2008) and curl (UTS 46) map some non-ASCII names to different ASCII hosts, urllib3
        # ends the authority at a backslash where urlsplit and curl read user info, and requests and curl
        # decode %XX in a host name: refuse rather than guess which host is meant (no source needs these).
        if not host or "\\" in parts.netloc or "%" in host:
            raise FetchError("bad_url", url)
        if blocked("//" + host):
            raise FetchError("blocked_host", url)
        if not self.allow_private:
            if host not in self.public:
                self.public[host] = _public(host, self.resolve)
            if not self.public[host]:
                raise FetchError("private_address", url)

    def check(self, url, check_robots=True):
        """The robots rules for a URL we may request; FetchError with the reason a request would get otherwise."""
        self._guard(url)
        rules = self.robots_for(url) if check_robots else ALLOW_ALL
        if not rules.allowed(url):
            raise FetchError("robots_unavailable" if rules is UNAVAILABLE else rules.reason(url), url)
        if (rules.crawl_delay or 0) > MAX_WAIT:  # a pace we will not keep within a run
            raise FetchError("crawl_delay_too_long", url)
        return rules

    def robots_for(self, url, recheck=600):
        """Rules for the URL's origin. A 429, a server error or an unreachable robots.txt means
        disallow (UNAVAILABLE), but only for ``recheck`` seconds: a transient reset must not ban a host."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        cached = self.robots.get(origin)
        if cached and (cached[1] is None or self.clock() < cached[1]):
            return cached[0]
        rules, expires = UNAVAILABLE, self.clock() + recheck
        for attempt in range(self.retries + 1):
            try:
                response, body = self._robots_txt(origin + "/robots.txt")
            except FetchError as error:  # a refused hop, a redirect loop or a body too slow
                rules = UNAVAILABLE if error.reason == "too_slow" else DISALLOW_ALL
                break
            except Exception:
                self.sleep(2 ** attempt * self.interval)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                self.sleep(2 ** attempt * self.interval)
                continue
            expires = None
            if response.status_code in (401, 403):
                rules, signature = DISALLOW_ALL, response.headers.get("X-Tengine-Error", "")
                if signature.startswith("denied by req auth"):  # a signed-URL CDN, not a WAF
                    rules = ALLOW_ALL
                    LOG.warning("%s/robots.txt: %d with X-Tengine-Error %r (signed-URL CDN): robots.txt "
                                "unavailable, all paths allowed", origin, response.status_code, signature)
            elif response.status_code >= 400:
                rules = ALLOW_ALL
            else:
                rules = Robots(_robots_text(body, response.headers.get("Content-Type") or ""), self.agent)
            break
        self.robots[origin] = (rules, expires)
        return rules

    def _robots_txt(self, url):
        """(response, body) of robots.txt; the body is cut to ROBOTS_LIMIT as ``_robots_cut`` says."""
        for _ in range(6):
            self._guard(url)
            host = urlsplit(url).netloc
            self._wait(host, self.interval)
            start = self.clock()
            response = self._request(host, "GET", url, limit=ROBOTS_LIMIT, truncate=True, timeout=self.timeout,
                                     stream=True, allow_redirects=False)
            if not response.is_redirect:
                if response.status_code >= 400:
                    response.close()
                    return response, b""
                body = self._read(response, url, ROBOTS_LIMIT, start, truncate=True)
                return response, body if len(body) < ROBOTS_LIMIT else _robots_cut(body)
            response.close()
            url = urljoin(url, response.headers["Location"])
        raise FetchError("too_many_redirects", url)

    def _read(self, response, url, limit, start, stop=None, truncate=False):
        """At most ``limit`` bytes of the body (more is too_large, or cut with ``truncate``) within the
        deadline of the module docstring; ``stop(chunk)`` returning True ends it early. Between chunks the
        deadline is checked here; on a session() connection the socket is also shut down when it passes."""
        body = io.BytesIO()

        def allowance():  # seconds left
            return start + self.timeout + body.tell() / MIN_RATE - self.clock()

        with _deadline(getattr(getattr(response, "raw", None), "guarded_sock", None), allowance) as expired:
            try:
                for chunk in _chunks(response):
                    body.write(chunk)
                    if (stop is not None and stop(chunk)) or (truncate and body.tell() >= limit):
                        response.close()
                        break
                    if body.tell() > limit:
                        response.close()
                        raise FetchError("too_large", url)
                    if allowance() < 0:
                        response.close()
                        raise FetchError("too_slow", url)
            except Exception:
                if not expired.is_set():
                    raise
        if expired.is_set():  # shut down mid-body: what came in is cut short and must not be used
            response.close()
            raise FetchError("too_slow", url)
        return body.getvalue()[:limit] if truncate else body.getvalue()

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
        rules = self.check(url, check_robots)
        delay = max(self.interval, rules.crawl_delay or 0)
        limit = stream_limit or self.max_bytes
        for attempt in range(self.retries + 1):
            self._wait(host, delay)
            start = self.clock()
            try:
                response = self._request(host, method, url, limit=limit, timeout=self.timeout, stream=True,
                                         allow_redirects=False, **kwargs)
            except FetchError as error:  # refused at connect time, headers too slow, curl's too_large
                raise FetchError(error.reason, url) from error
            except Exception as error:
                if attempt == self.retries:
                    raise FetchError("network:" + type(error).__name__, url) from error
                self.sleep(min(2 ** attempt * delay, MAX_WAIT))
                continue
            if response.is_redirect:
                response.close()
                return response, b""
            if response.status_code == 429 or response.status_code >= 500:
                wait = _retry_after(response.headers.get("Retry-After") or "")
                response.close()
                if attempt == self.retries or (wait or 0) > MAX_WAIT:
                    raise FetchError(f"http_{response.status_code}", url)
                self.sleep(min(2 ** (attempt + 1) * delay, MAX_WAIT) if wait is None else wait)
                continue
            if response.status_code >= 400:
                response.close()
                raise FetchError(f"http_{response.status_code}", url)
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > limit and "Range" not in kwargs.get("headers", {}):
                response.close()
                raise FetchError("too_large", url)
            try:
                body = self._read(response, url, limit, start, stop)
            except FetchError:
                raise
            except Exception as error:  # connection broken mid-body (IncompleteRead, ChunkedEncodingError)
                response.close()
                if attempt == self.retries:
                    raise FetchError("network:" + type(error).__name__, url) from error
                self.sleep(min(2 ** attempt * delay, MAX_WAIT))
                continue
            return response, body

    def json(self, url, method="GET", **kwargs):
        return json.loads(self.get(url, method=method, **kwargs)[1])


def _chunks(response):
    # urllib3's read1 returns whatever one socket read brings, so the deadline is checked while a
    # server drips; iter_content would block until a whole chunk is in.
    raw = getattr(response, "raw", None)
    if not hasattr(raw, "read1"):
        yield from response.iter_content(1 << 16)
        return
    while chunk := raw.read1(1 << 16, decode_content=True):
        yield chunk


def _retry_after(value):
    """Seconds a Retry-After asks for (delay-seconds or an HTTP-date, RFC 9110 10.2.3); None if unreadable."""
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        date = email.utils.parsedate_to_datetime(value)
        return max(0.0, (date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _robots_cut(body):
    """robots.txt cut at ROBOTS_LIMIT bytes: its full lines, and of the line the cut fell in as much as
    is known to mean what it would in full. Everything before a "#" does (the cut fell in a comment). A
    value followed by whitespace is complete too (RFC 9309 paths and product tokens hold none), but this
    is read conservatively: a Disallow is kept up to that whitespace, while an Allow or a User-agent
    (which Robots would read with the rest of the line, so as matching nothing) is dropped, as is a line
    whose value the cut may have shortened."""
    end = max(body.rfind(b"\n"), body.rfind(b"\r")) + 1  # CR, LF and CRLF all end lines
    last = body[end:]
    if b"#" in last:
        return body[:end] + last.split(b"#", 1)[0]
    disallow = re.match(rb"\s*disallow\s*:\s*\S+(?=\s)", last, re.I)
    return body[:end] + (disallow.group() if disallow else b"")


def _robots_text(body, content_type):
    """UTF-8 as RFC 9309 requires; a body that is not valid UTF-8 is read in the charset the server
    declares (Chinese sites serve GBK robots.txt as text/plain; charset=gbk)."""
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        pass
    charset = re.search(r"charset\s*=\s*[\"']?([\w.:-]+)", content_type, re.I)
    try:
        return body.decode(charset.group(1) if charset else "utf-8", "replace")
    except LookupError:
        return body.decode("utf-8", "replace")


def _public(host, resolve):
    """False for special-use names and for hosts that are, or resolve to, non-global addresses."""
    name = host.rstrip(".").lower()
    if any(name == s or name.endswith("." + s) for s in SPECIAL_NAMES):
        return False
    try:
        addresses = [ipaddress.ip_address(name)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in resolve(name, None)]
        except (OSError, UnicodeError, ValueError):  # unresolved here; the egress proxy resolves names itself
            tld = name.rpartition(".")[2]
            return "." in name and (tld.isalpha() or tld.startswith("xn--"))
    return bool(addresses) and all(_global(address) for address in addresses)


def _global(address):
    address = getattr(address, "ipv4_mapped", None) or address
    return address.is_global and not (address.is_multicast or address.is_reserved
                                      or getattr(address, "is_site_local", False))


def on_cc_data_host(url):
    """Whether a URL is on data.commoncrawl.org (any scheme, port or letter case)."""
    return (urlsplit(url).hostname or "").rstrip(".") == "data.commoncrawl.org"


def _tls_failure(error):
    """Whether a request failed in TLS (to the host or to the proxy), judged by the exception types along
    its ``__cause__``/``__context__`` chain (requests' error, urllib3's MaxRetryError, its ``reason``
    such as an SSLError or a ProxyError, then ssl.SSLError), never by the message, which can quote the URL."""
    import ssl
    import requests.exceptions
    import urllib3.exceptions
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if isinstance(error, (ssl.SSLError, urllib3.exceptions.SSLError, requests.exceptions.SSLError)):
            return True
        if isinstance(error, BaseException) and id(error) not in seen:
            seen.add(id(error))
            pending += [error.__cause__, error.__context__, getattr(error, "reason", None)]
    return False


class CurlResponse:
    def __init__(self, status, headers, body, url):
        self.status_code, self.headers, self._body, self.url = status, headers, body, url

    @property
    def text(self):
        return self._body.decode("utf-8", "replace")

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


def _curl_pin(url, resolve, allow_private):
    """--resolve arguments that make curl connect to the URL host's addresses as resolved now, all global
    (a name that passed Fetcher._guard may resolve elsewhere a moment later)."""
    parts = urlsplit(url)
    name, port = parts.hostname, parts.port or {"http": 80, "https": 443}[parts.scheme]
    try:
        addresses, literal = [ipaddress.ip_address(name)], True
    except ValueError:
        addresses = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in resolve(name.rstrip("."), port)]
        literal = False
    if not addresses or not (allow_private or all(_global(address) for address in addresses)):
        raise FetchError("private_address", url)
    listed = ",".join(f"[{a}]" if a.version == 6 else str(a) for a in dict.fromkeys(addresses))
    # curl looks a name up as the URL spells it (any letter case, root dot kept) and consults --resolve
    # even for an address literal; a specific entry wins over the wildcard, which catches any other spelling.
    return ([] if literal else ["--resolve", f"{name}:{port}:{listed}"]) + ["--resolve", f"*:{port}:{listed}"]


def curl_request(method, url, headers=None, data=None, json=None, timeout=60, max_bytes=200 << 20,
                 allow_redirects=False, stream=True, truncate=False, resolve=socket.getaddrinfo, allow_private=False):
    """The same request through the curl binary. curl never retries by itself (the Fetcher's retries
    keep its pacing), speaks only http(s) and reads no .curlrc. Its output is read as it arrives: past
    ``max_bytes`` curl is stopped and the request is too_large, or, with ``truncate``, the first
    ``max_bytes`` are kept (robots.txt).

    curl takes the proxy requests finds for the URL in the environment (http_proxy, HTTPS_PROXY,
    ALL_PROXY, NO_PROXY, ... as requests reads them), handed over as its only proxy variable, or none.
    A proxy resolves the host itself, except a socks4:// or socks5:// one, which leaves names to curl:
    then, as without a proxy, the host is resolved here, refused as private_address unless every
    address is global, and pinned with --resolve, so curl connects to the addresses checked."""
    import requests.structures
    import requests.utils
    from urllib.parse import urlencode
    proxy = requests.utils.select_proxy(url, requests.utils.get_environ_proxies(url))
    env = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    if proxy:
        env[urlsplit(url).scheme + "_proxy"] = proxy  # lower case: curl reads http_proxy in no other
    pin = [] if proxy and not proxy.lower().startswith(("socks4://", "socks5://")) else \
        _curl_pin(url, resolve, allow_private)
    with tempfile.TemporaryDirectory() as tmp:
        head, err, sent = os.path.join(tmp, "h"), os.path.join(tmp, "e"), os.path.join(tmp, "d")
        command = ["curl", "-q", "-sS", "-X", method, "-D", head, "--max-time", str(timeout), "--retry", "0",
                   "--proto", "=http,https", "--proto-redir", "=http,https", "--compressed", *pin]
        if not truncate:
            command += ["--max-filesize", str(max_bytes)]
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
            Path(sent).write_bytes(payload)
            command += ["--data-binary", "@" + sent]
        with open(err, "wb") as stderr:
            process = subprocess.Popen(command + [url], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=stderr,
                                       env=env)
        watchdog = threading.Timer(timeout * 2, process.kill)  # in case --max-time does not end it
        watchdog.start()
        try:
            body = process.stdout.read(max_bytes + 1)
            if len(body) > max_bytes:
                process.kill()
        except BaseException:
            process.kill()
            raise
        finally:
            watchdog.cancel()
            process.stdout.close()
            code = process.wait()
        if len(body) > max_bytes:
            if not truncate:
                raise FetchError("too_large", url)
            body = body[:max_bytes]
        elif code == 63:
            raise FetchError("too_large", url)
        elif code != 0:
            raise ConnectionError(f"curl exit {code}: {Path(err).read_bytes().decode(errors='replace')[:200]}")
        blocks = [b for b in open(head, "rb").read().decode("latin-1").split("\r\n\r\n") if b.strip()]
        lines = blocks[-1].split("\r\n")  # the proxy CONNECT reply comes first
        fields = requests.structures.CaseInsensitiveDict()
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        return CurlResponse(int(lines[0].split()[1]), fields, body, url)


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
    the site, so reads there skip the robots check; any other base is checked as usual.
    The meta returned holds the record's target URI (download checks that origin's robots.txt).
    """
    import zlib
    from warcio.archiveiterator import ArchiveIterator
    check = not on_cc_data_host(base)
    if length:
        end = int(offset) + int(length) - 1
        _, raw = fetcher.get(base + filename, check_robots=check, headers={"Range": f"bytes={offset}-{end}"})
    else:
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        plain = io.BytesIO()

        def complete(chunk):
            if not member.eof:
                plain.write(member.decompress(chunk))
            return member.eof

        fetcher.get(base + filename, check_robots=check, headers={"Range": f"bytes={offset}-"}, stop=complete)
        if not member.eof:
            raise FetchError("warc_incomplete", filename)
        raw = plain.getvalue()
    for record in ArchiveIterator(io.BytesIO(raw)):
        if record.rec_type == "response":
            truncated = record.rec_headers.get_header("WARC-Truncated")
            status = record.http_headers.get_statuscode() if record.http_headers else None
            return record.content_stream().read(), {"warc_truncated": truncated, "http_status": status,
                                                    "target_uri": record.rec_headers.get_header("WARC-Target-URI")}
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
    """files/<sha[:2]>/<sha>.<type> plus an append-only fetch log for resuming.

    Only a line feed ends a log record (titles may hold U+2028, U+0085 and the like, which
    str.splitlines() also splits on). A last record cut short by a crash is dropped from the
    log, counted in ``dropped``, so the next record starts on a line of its own."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "files").mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "fetch_log.jsonl"
        self.seen, self.dropped = {}, 0
        if self.log_path.exists():
            raw = self.log_path.read_bytes()
            complete, _, tail = raw.rpartition(b"\n")
            for line in complete.decode("utf-8").split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    self.seen[entry["key"]] = entry
            if tail.strip():
                self.dropped += 1
                os.truncate(self.log_path, len(raw) - len(tail))
                LOG.warning("%s: dropped an unfinished last record", self.log_path)

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


TRANSIENT = ("robots_unavailable", "http_429", "too_slow")  # plus network:* and http_5xx


def _transient(entry):
    reason = entry.get("reason") or ""
    return entry.get("status") == "error" and (reason in TRANSIENT or reason.startswith(("network:", "http_5")))


def zip_member(fetcher, reference, cache):
    """Bytes of {url, member}: the ZIP is fetched whole through the fetcher and ``cache`` keeps
    the last one (or its failure), so candidates should come grouped. A member declared or read
    over ``max_bytes`` is refused; whatever goes wrong becomes a FetchError."""
    url = reference["url"]
    if cache.get("url") != url:
        try:
            archive = zipfile.ZipFile(io.BytesIO(fetcher.get(url)[1]))
        except FetchError as error:
            archive = error
        except Exception as error:  # not a ZIP, or a damaged one
            archive = FetchError("zip_" + type(error).__name__, url)
        cache.update(url=url, archive=archive)
    archive = cache["archive"]
    if isinstance(archive, FetchError):
        raise archive
    try:
        info = archive.getinfo(reference["member"])
        if info.file_size > fetcher.max_bytes:
            raise FetchError("too_large", url)
        with archive.open(info) as member:
            data = member.read(fetcher.max_bytes + 1)
    except FetchError:
        raise
    except KeyError:
        raise FetchError("zip_member_missing", url) from None
    except Exception as error:  # encrypted, unsupported or damaged member
        raise FetchError("zip_" + type(error).__name__, url) from error
    if len(data) > fetcher.max_bytes:
        raise FetchError("too_large", url)
    return data


def _failed(entry, error, log):
    entry.update(status="error", reason="exception:" + type(error).__name__)
    if log:
        log(f"{entry['key']}: {type(error).__name__}: {error}")


def download(candidates, store, fetcher, accept=None, limit=None, log=print, refetch_truncated=False,
             cc_data_host=False):
    """Fetch candidate dicts ({key, url | warc | parquet | zip, source, license, ...}); returns counts by status.

    Keys already in the store's log are skipped, except those whose last attempt failed for a
    reason that may pass (robots_unavailable, network:*, http_429, http_5xx, too_slow): they
    are tried again. Fetching or storing one candidate may fail in any way: that is logged for
    the candidate (an unexpected exception as ``exception:<type>``) and the run goes on. A
    failing write to the store (an OSError such as a full disk) or a MemoryError ends the run
    instead, so a resume retries the rest. A candidate without a key or url to log it under, or
    with a field the log cannot hold (a NaN, a lone surrogate), is only counted
    (invalid_candidate), before any request.

    A WARC payload cut at the crawl's size limit is useless; with ``refetch_truncated`` the
    original URL is fetched live instead (robots.txt and pacing apply as usual).
    WARC records on data.commoncrawl.org are read only with ``cc_data_host``: that host's
    robots.txt disallows everything although Common Crawl documents it as the HTTPS
    download endpoint, so using it is an explicit operator decision. Without it such
    candidates fall back to their original URL. An archived copy is used only where a live
    fetch of its origin URL would pass ``Fetcher.check`` (robots.txt with the AI-crawler
    opt-outs, blocked hosts, public addresses), and is refused with the same reason.

    ``zip`` ({url, member}) names the same file inside a ZIP the source publishes (e.g. the
    Taiwan gazette's monthly Open Data packs): the ZIP is fetched once for all its members (see
    ``zip_member``); if it or the member cannot be used, the candidate's url is fetched."""
    counts = {}
    done = 0
    parquet_cache, zip_cache, tried = {}, {}, set()
    for candidate in candidates:
        key = (candidate.get("key") or candidate.get("url")) if isinstance(candidate, dict) else None
        if not isinstance(key, str):
            counts["invalid_candidate"] = counts.get("invalid_candidate", 0) + 1
            if log:
                log(f"skipped a candidate without a key or url: {str(candidate)[:200]}")
            continue
        entry = {k: v for k, v in candidate.items() if k not in ("warc", "parquet", "zip")}
        entry["key"] = key
        try:  # as store.record writes it: a NaN, a lone surrogate or a set could never be logged
            json_text(entry).encode("utf-8")
        except (ValueError, TypeError, RecursionError) as error:  # UnicodeEncodeError is a ValueError
            counts["invalid_candidate"] = counts.get("invalid_candidate", 0) + 1
            if log:
                log(f"skipped a candidate the fetch log cannot hold ({type(error).__name__}): {str(candidate)[:200]}")
            continue
        if key in tried or (key in store.seen and not _transient(store.seen[key])):
            continue
        tried.add(key)
        try:
            data = None
            w = candidate.get("warc")
            if w and on_cc_data_host(w.get("base", CC_BASE)) and not cc_data_host:
                if not candidate.get("url"):
                    raise FetchError("cc_data_host_disabled", w["filename"])
                w = None
                entry["via"] = "origin"
            if candidate.get("parquet"):
                data = parquet_blob(fetcher, candidate["parquet"], parquet_cache)
                entry["via"] = "parquet"
            elif w:
                origin = candidate.get("url")
                if origin:  # the origin's robots.txt, AI opt-outs included, binds its archived copies too
                    fetcher.check(origin)
                data, meta = warc_record(fetcher, w["filename"], w["offset"], w.get("length"),
                                         base=w.get("base", CC_BASE))
                target = meta.pop("target_uri")
                if not origin:
                    fetcher.check(target or "")
                entry.update(meta)
                if meta["warc_truncated"]:
                    if not (refetch_truncated and candidate.get("url")):
                        raise FetchError("warc_truncated", candidate.get("url") or w["filename"])
                    response, data = fetcher.get(candidate["url"], headers=candidate.get("headers") or {})
                    entry.update(final_url=response.url, refetched=True)
            elif candidate.get("zip"):
                if candidate.get("url"):  # as for WARC copies, the origin's robots.txt binds the archived one
                    fetcher.check(candidate["url"])
                try:
                    data = zip_member(fetcher, candidate["zip"], zip_cache)
                    entry["via"] = "zip"
                except FetchError as error:
                    if not candidate.get("url"):
                        raise
                    entry["zip_error"] = error.reason
            if data is None:
                response, data = fetcher.get(candidate["url"], headers=candidate.get("headers") or {})
                entry.update(final_url=response.url, content_type=response.headers.get("Content-Type"))
        except FetchError as error:
            entry.update(status="error", reason=error.reason)
        except MemoryError:
            raise
        except Exception as error:  # e.g. a WARC range that holds no WARC record: this candidate fails, not the run
            _failed(entry, error, log)
        else:
            try:
                digest, kind, path = store.put(data)
            except (OSError, MemoryError):  # a store that cannot write (a full disk) ends the run
                raise
            except Exception as error:  # content the store cannot handle: this candidate fails, not the run
                _failed(entry, error, log)
            else:
                entry.update(status="ok" if accept is None or kind in accept else "unwanted_type",
                             sha256=digest, type=kind, bytes=len(data), path=str(path))
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
