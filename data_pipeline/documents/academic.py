"""Academic sources: DOAJ journals in Chinese, Chinese OJS journals over OAI-PMH, Chinese OA works
in the OpenAlex snapshot, and National Taiwan University theses.

doaj: https://doaj.org/api/search/journals/bibjson.language%3AZH?page=N&pageSize=100 lists the
~331 journals that list Chinese; /api/search/articles/issn%3A{ISSN}?page=N&pageSize=100 their
articles. A query reaches at most 1000 results (page 11 of 100 is a 400), sorting returns 400 and
the API allows 2 requests/s, so journals over 1000 articles are split by bibjson.year ranges.
Fulltext links ending in .pdf (Magtech /fileup/.../PDF/...) are candidates as they are; other
landing pages (Magtech /CN/..., Renhe /article/doi/... and /article/id/..., Qinyun's
server-rendered /zh/article/..., academax behind doi.org) carry <meta name="citation_pdf_url">,
read from the page head only. Qinyun single-page apps (/thesisDetails..., about half of all links)
render nothing without JavaScript and *.cnki.net portals are excluded, so a journal whose first
few articles give no PDF is dropped. Some PDF hosts answer with a JavaScript guard page
(journal.yykxjz.cn: /_guard/html.js?js=delay_jump_html); the downloader records it as HTML.

ojs: OAI-PMH ListSets / ListRecords (oai_dc, resumptionToken) at
http://ojs.omniscient.sg/index.php/index/oai (plain http; 84,879 records in ~256 journal and
section sets). Its Chinese journals declare dc:language 'eng', so records are kept by CJK in
dc:title. dc:relation .../article/view/{article}/{galley} names the galley whose file is
.../article/download/{article}/{galley} (the article page's citation_pdf_url); without exactly
one galley the article page is read. Licence from dc:rights (CC BY-NC 4.0 there). Paging stops
when a resumptionToken repeats; ListSets pages are capped and every ListRecords page costs at
least one record of the max_records budget, so a misbehaving endpoint cannot loop forever.

openalex: https://openalex.s3.amazonaws.com/data/parquet/works/manifest.json lists ~1,900
parquet files (744 GB; robots.txt is 404). Files are read with column projection through exact
HTTP range reads: the filter columns cost ~0.3 MB per 40k-row row group, extra columns are read
only for row groups with matches (title alone is ~2.4 MB per row group, so it is opt-in): ~5.7 MB
per 300k-row file with id and publication_year. A buffered reader's read-ahead once turned one
file into 181 MB, hence one exact range per read. Most zh pdf_urls point to CAS DSpace
repositories whose robots.txt refuses us, to blocked hosts (SciEngine, opticsjournal) or to
Japanese repositories (kanji titles tagged zh); OJS hosts such as ojs.omniscient.sg give the PDFs.
api.openalex.org needs a personal key once the shared per-IP budget is spent, so it is not used.

ntu_theses: https://tdr.lib.ntu.edu.tw/jspui/htmlmap links ?map=0..236, pages of ~383 handles
(higher maps list more recently added items); an item's head (?mode=full) holds DC.rights
(同意授權(全球公開) = worldwide public access; also 限校園內公開 campus only and 有償授權 paid via a
vendor), citation_language (zh_TW) and citation_pdf_url (an http bitstream, sequence -1 while no
readable file exists). Sampled older maps (50, 100, 150, 200) are almost all 有償授權; in the two
newest maps 7 of ~240 items had a readable public file and 21 more a -1 sequence. robots.txt sets
Crawl-delay 10, which the Fetcher waits, so expect ~10 candidates an hour. /discover and
/simple-search are disallowed and OAI-PMH answers noRecordsMatch, so items are read one by one.
Bitstreams can carry a multipart wrapper, which Store.put strips.
"""

import html
import itertools
import json
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode, urljoin, urlsplit

from .catalog import blocked
from .fetch import FetchError

PERSONAL = re.compile("名[单單]|名[册冊]|身份[证證]")
KANA = re.compile("[぀-ヿ]")
META = re.compile(r"<meta\b([^>]*)>", re.I)
ATTR = re.compile(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
CC = re.compile(r"creativecommons\.org/(?:licenses/([a-z-]+)(?:/([\d.]+))?|publicdomain/(zero|mark))", re.I)
ZH_CODES = ("zh", "chi", "zho", "cn", "chinese", "中文")
HEAD_BYTES = 2 << 20


def _quiet(message):
    pass


def _list(value):
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    return [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)


def _span(value):
    """(first, last) from 2015, "2010-2020" or [2010, 2020]; None when unset."""
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value, value
    if isinstance(value, str):
        first, _, last = value.partition("-")
        return int(first), int(last or first)
    return int(value[0]), int(value[-1])


def _cjk(text, least=1):
    return sum("一" <= c <= "鿿" for c in text or "") >= least


def _chinese(text):
    return _cjk(text, 2) and not KANA.search(text)


def _decode(body):
    if re.search(rb"""charset\s*=\s*["']?(?:gb2312|gbk|gb18030)""", body[:4096], re.I):
        return body.decode("gb18030", "replace")
    return body.decode("utf-8", "replace")


def _metas(text):
    """Lower-case meta name -> list of contents (citation_*, DC.* and DCTERMS.* tags)."""
    found = {}
    for tag in META.finditer(text):
        attrs = {m.group(1).lower(): m.group(2) if m.group(2) is not None else m.group(3)
                 for m in ATTR.finditer(tag.group(1))}
        if attrs.get("name") and attrs.get("content") is not None:
            found.setdefault(attrs["name"].lower(), []).append(html.unescape(attrs["content"]).strip())
    return found


def _first(metas, *names):
    return next((v for name in names for v in metas.get(name, []) if v), "")


def _page(fetcher, url, log):
    try:
        return _decode(fetcher.get(url)[1])
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None


def _head(fetcher, url, log, headers=None):
    """(final URL, metas of the page head), (final URL, None) when the URL serves a PDF itself,
    or None when it cannot be fetched. Reading stops at </head>."""
    state = {"first": True, "tail": b"", "size": 0}

    def stop(chunk):
        if state["first"]:
            state["first"] = False
            if b"%PDF-" in chunk[:1024]:
                return True
        window = state["tail"] + chunk.lower()
        state["tail"], state["size"] = window[-8:], state["size"] + len(chunk)
        return b"</head" in window or state["size"] > HEAD_BYTES

    try:
        response, body = fetcher.get(url, headers=headers or {}, stop=stop)
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None
    final = getattr(response, "url", None) or url
    return final, None if b"%PDF-" in body[:1024] else _metas(_decode(body))


def _cc(texts):
    """'CC BY-NC 4.0' style name of the first Creative Commons URL among the texts."""
    for text in texts:
        found = CC.search(text or "")
        if found and found.group(1):
            return " ".join(filter(None, ("CC", found.group(1).upper(), found.group(2))))
        if found:
            return "CC0" if found.group(3).lower() == "zero" else "Public Domain Mark"
    return None


# --- DOAJ ---

DOAJ = "https://doaj.org/api/search/"
PAGE, CAP, FIRST_YEAR = 100, 1000, 1900
SPA = re.compile(r"thesisDetails|#/")


def doaj(fetcher, options):
    """PDFs of articles in DOAJ journals that list Chinese, one candidate per DOAJ article.

    Options: licences (list or comma string of DOAJ licence types to keep, e.g. "CC BY,CC BY-SA";
    default all), zh_only (bool, default False: also journals listing other languages, whose
    landing pages must not declare a non-Chinese citation_language), journals (ISSNs to keep),
    years (2020, "2015-2026" or [2015, 2026]; default all), seed (int: shuffle the journal order;
    default the API's order), max_journals (int), max_articles (int, articles examined per
    journal), probe (int, default 5: drop a journal whose first N articles give no PDF; 0 never
    drops), max_queries (int DOAJ API requests, default 5000), max_failures (consecutive failed API
    requests before giving up, default 6), max_items (int), log (callable)."""
    yield from itertools.islice(_doaj(fetcher, options), options.get("max_items"))


class _Api:
    """DOAJ JSON requests with a budget; a run of failures means the API is down."""

    def __init__(self, fetcher, options, log):
        self.fetcher, self.log = fetcher, log
        self.left, self.failures = int(options.get("max_queries", 5000)), 0
        self.patience = int(options.get("max_failures", 6))

    def get(self, url, required=False):
        if self.left <= 0:
            return None
        self.left -= 1
        if not self.left:
            self.log("doaj: max_queries reached; stopping after this request")
        try:
            _, body = self.fetcher.get(url)
        except FetchError as error:
            self.failures += 1
            if required:
                raise
            if self.failures >= self.patience:
                raise FetchError("api_down:" + error.reason, url) from error
            self.log(f"doaj: skipped {url}: {error.reason}")
            return None
        self.failures = 0
        try:
            return json.loads(body)
        except ValueError:
            raise FetchError("not_json", url) from None  # a block page: stop, never work around it

    def search(self, query, page):
        return self.get(f"{DOAJ}articles/{quote(query, safe='')}?page={page}&pageSize={PAGE}")


def _doaj(fetcher, options):
    log = options.get("log") or _quiet
    api = _Api(fetcher, options, log)
    seen = set()
    for journal in _doaj_journals(api, options):
        yield from _doaj_journal(fetcher, api, journal, options, seen, log)


def _doaj_journals(api, options):
    rows, page = [], 1
    while reply := api.get(f"{DOAJ}journals/{quote('bibjson.language:ZH', safe='')}?page={page}&pageSize={PAGE}",
                           required=True):
        rows += reply.get("results") or []
        if not reply.get("results") or page * PAGE >= (reply.get("total") or 0):
            break
        page += 1
    keep = {k.upper() for k in _list(options.get("licences"))}
    issns = set(_list(options.get("journals")))
    journals = []
    for row in rows:
        bib = row.get("bibjson") or {}
        ids = [i for i in (bib.get("eissn"), bib.get("pissn")) if i]
        licences = [item.get("type") for item in bib.get("license") or [] if item.get("type")]
        languages = [x.upper() for x in bib.get("language") or []]
        if not ids or (issns and not issns & set(ids)) or (keep and not keep & {x.upper() for x in licences}):
            continue
        if options.get("zh_only") and languages != ["ZH"]:
            continue
        journals.append(dict(issn=ids[0], doaj_journal=row.get("id"), title=bib.get("title"),
                             alternative_title=bib.get("alternative_title"), languages=languages,
                             licence="; ".join(licences) or "unknown (no licence in DOAJ)",
                             licence_urls=[item.get("url") for item in bib.get("license") or [] if item.get("url")],
                             home=(bib.get("ref") or {}).get("journal")))
    if options.get("seed") is not None:
        random.Random(options["seed"]).shuffle(journals)
    return journals[:options.get("max_journals")]


def _doaj_journal(fetcher, api, journal, options, seen, log):
    probe, limit = int(options.get("probe", 5)), options.get("max_articles")
    examined = found = 0
    for article in _doaj_articles(api, journal["issn"], _span(options.get("years")), log):
        if article.get("id") in seen:
            continue
        if (limit and examined >= limit) or (probe and examined >= probe and not found):
            if not found:
                log(f"doaj: no PDF among the first {examined} articles of {journal['title']} ({journal['issn']})")
            return
        seen.add(article.get("id"))
        examined += 1
        if candidate := _doaj_candidate(fetcher, article, journal, log):
            found += 1
            yield candidate


def _doaj_articles(api, issn, years, log):
    """Article records of one journal: one query when it fits the cap, else year ranges, newest first."""
    if years:
        yield from _doaj_years(api, issn, *years, log)
        return
    first = api.search(f"issn:{issn}", 1)
    if first is None:
        return
    total = first.get("total") or 0
    yield from first.get("results") or []  # repeats from the year split are dropped by the caller
    if total <= CAP:
        yield from _doaj_pages(api, f"issn:{issn}", total)
    else:
        yield from _doaj_years(api, issn, FIRST_YEAR, time.localtime().tm_year, log)


def _doaj_years(api, issn, low, high, log):
    query = f"issn:{issn} AND bibjson.year:[{low} TO {high}]"
    first = api.search(query, 1)
    if first is None:
        return
    total = first.get("total") or 0
    if total > CAP and low < high:
        middle = (low + high) // 2
        yield from _doaj_years(api, issn, middle + 1, high, log)
        yield from _doaj_years(api, issn, low, middle, log)
        return
    if total > CAP:
        log(f"doaj: {total} articles of {issn} in {low}; only {CAP} are reachable")
    yield from first.get("results") or []
    yield from _doaj_pages(api, query, total)


def _doaj_pages(api, query, total):
    for page in range(2, math.ceil(min(total, CAP) / PAGE) + 1):
        reply = api.search(query, page)
        if reply is None:
            continue
        yield from reply.get("results") or []


def _doaj_candidate(fetcher, article, journal, log):
    bib = article.get("bibjson") or {}
    title = (bib.get("title") or "").strip()
    links = [link["url"] for link in bib.get("link") or [] if link.get("type") == "fulltext" and link.get("url")]
    if PERSONAL.search(title):
        return None
    doi = next((i.get("id") for i in bib.get("identifier") or [] if (i.get("type") or "").lower() == "doi"), None)
    meta = dict(doaj_id=article.get("id"), journal=journal["title"], journal_alt=journal["alternative_title"],
                issn=journal["issn"], journal_languages=journal["languages"], year=bib.get("year"), doi=doi,
                licence_urls=journal["licence_urls"])
    for link in links:
        if SPA.search(link) or blocked(link):
            continue
        url, headers = link, {}
        if not urlsplit(link).path.lower().endswith(".pdf"):
            got = _head(fetcher, link, log)
            if got is None:
                continue
            landing, metas = got
            if metas is not None:  # an HTML landing page rather than the PDF itself
                pdf, language = _first(metas, "citation_pdf_url"), _first(metas, "citation_language").lower()
                if not pdf or (language and not language.startswith(ZH_CODES)):
                    continue
                url, headers = urljoin(landing, pdf), {"Referer": landing}
                title = next((t for t in metas.get("citation_title", []) if _cjk(t)), title)
                meta.update(landing=landing, language=language or None)
        if blocked(url) or PERSONAL.search(title):
            return None
        meta.update(fulltext=link, host=urlsplit(url).hostname)
        return dict(key=f"doaj:{article.get('id')}", url=url, title=title, type_hint="pdf", license=journal["licence"],
                    meta=meta, headers=headers)
    return None


# --- OJS over OAI-PMH ---

OJS = "http://ojs.omniscient.sg/index.php/index/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "dc": "http://purl.org/dc/elements/1.1/"}
GALLEY = re.compile(r"/article/view/(\d+)/(\d+)/?$")
ISSN = re.compile(r"^\d{4}-\d{3}[\dXx]$")


def ojs(fetcher, options):
    """Galley PDFs of Chinese-titled records of an OJS site, harvested over OAI-PMH.

    Options: base (OAI endpoint, default ``OJS``), sets (list or comma string of setSpecs; default
    every journal set, those with Chinese names first), probe (bool, default True: skip a set whose
    first page has no Chinese title), article_page (bool, default False: always read the galley from
    the article page's citation_pdf_url instead of dc:relation), max_records (int OAI records
    examined, default 100000; an empty page counts as one), max_set_pages (int ListSets pages,
    default 100), max_items (int), log (callable)."""
    yield from itertools.islice(_ojs(fetcher, options), options.get("max_items"))


def _oai(fetcher, base, log, **params):
    url = f"{base}?{urlencode(params)}"
    try:
        root = ET.fromstring(fetcher.get(url)[1])
    except FetchError as error:
        log(f"ojs: skipped {url}: {error.reason}")
        return None
    except ET.ParseError:
        log(f"ojs: not XML: {url}")
        return None
    error = root.find("oai:error", NS)
    if error is not None:
        if error.get("code") != "noRecordsMatch":
            log(f"ojs: {error.get('code')} for {url}: {(error.text or '').strip()}")
        return None
    return root


def _oai_pages(fetcher, base, verb, log, max_pages=None, **params):
    """The <verb> element of each response, following resumptionTokens until one repeats."""
    query, tokens = dict(verb=verb, **params), set()
    while (root := _oai(fetcher, base, log, **query)) is not None:
        part = root.find(f"oai:{verb}", NS)
        if part is None:
            return
        yield part
        token = (part.findtext("oai:resumptionToken", "", NS) or "").strip()
        if not token:
            return
        if token in tokens or (max_pages and len(tokens) + 1 >= max_pages):
            log(f"ojs: {verb} stopped after {len(tokens) + 1} pages: "
                + ("resumptionToken repeats" if token in tokens else "max pages reached"))
            return
        tokens.add(token)
        query = dict(verb=verb, resumptionToken=token)


def _ojs(fetcher, options):
    log = options.get("log") or _quiet
    base = options.get("base") or OJS
    sets = _list(options.get("sets"))
    if not sets:
        found = [(s.findtext("oai:setSpec", "", NS).strip(), s.findtext("oai:setName", "", NS).strip())
                 for part in _oai_pages(fetcher, base, "ListSets", log, int(options.get("max_set_pages", 100)))
                 for s in part.findall("oai:set", NS)]
        sets = list(dict.fromkeys(spec for spec, name in sorted(found, key=lambda s: not _cjk(s[1]))
                                  if spec and ":" not in spec))
    left, seen = [int(options.get("max_records", 100000))], set()
    for spec in sets:
        if left[0] <= 0:
            log("ojs: max_records reached")
            return
        yield from _ojs_set(fetcher, base, spec, options, left, seen, log)


def _ojs_set(fetcher, base, spec, options, left, seen, log):
    for page, part in enumerate(_oai_pages(fetcher, base, "ListRecords", log, metadataPrefix="oai_dc", set=spec)):
        records = part.findall("oai:record", NS)
        left[0] -= max(1, len(records))  # empty pages with fresh tokens must still use up the budget
        rows = [row for row in map(_dc, records) if row]
        chinese = [row for row in rows if _chinese(row["title"])]
        if page == 0 and rows and not chinese and options.get("probe", True):
            log(f"ojs: no Chinese title on the first page of set {spec}; skipping it")
            return
        for row in chinese:
            if candidate := _ojs_candidate(fetcher, row, options, seen, log):
                yield candidate
        if left[0] <= 0:
            return


def _dc(record):
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    metadata = record.find("oai:metadata", NS)

    def values(tag):
        elements = metadata.iter(f"{{{NS['dc']}}}{tag}") if metadata is not None else ()
        return [(e.text or "").strip() for e in elements if (e.text or "").strip()]

    titles = values("title")
    return dict(oai_id=header.findtext("oai:identifier", "", NS).strip(), set=header.findtext("oai:setSpec", "", NS),
                title=next((t for t in titles if _chinese(t)), titles[0] if titles else ""),
                identifiers=values("identifier"), relations=values("relation"), rights=values("rights"),
                dates=values("date"), sources=values("source"), formats=values("format"))


def _ojs_candidate(fetcher, row, options, seen, log):
    view = next((i for i in row["identifiers"] if re.search(r"/article/view/\d+", i)), None)
    if row["oai_id"] in seen or PERSONAL.search(row["title"]):
        return None
    seen.add(row["oai_id"])
    galleys = [g for g in row["relations"] if GALLEY.search(g)]
    pdf_only = not row["formats"] or "application/pdf" in row["formats"]
    url = None
    if len(galleys) == 1 and pdf_only and not options.get("article_page"):
        url = galleys[0].replace("/article/view/", "/article/download/")
    elif view:
        got = _head(fetcher, view, log)
        if got and got[1] is not None and (pdf := _first(got[1], "citation_pdf_url")):
            url = urljoin(got[0], pdf)
    if not url:
        log(f"ojs: no PDF galley for {row['oai_id']}")
        return None
    article = re.search(r"/article/(?:view|download)/(\d+)(?:/(\d+))?", url)
    host = urlsplit(url).hostname
    meta = dict(oai_id=row["oai_id"], set=row["set"], article_page=view, host=host, date=next(iter(row["dates"]), None),
                journal=next(iter(row["sources"]), "").split(";")[0].strip() or None,
                issn=next((s for s in row["sources"] if ISSN.match(s)), None),
                doi=next((i for i in row["identifiers"] if re.match(r"(https?://(dx\.)?doi\.org/)?10\.\d+/", i)), None),
                galley_id=article.group(2) if article else None, rights=row["rights"])
    key = f"ojs:{host}:{article.group(1)}" if article else f"ojs:{row['oai_id']}"
    return dict(key=key, url=url, title=row["title"], type_hint="pdf", meta=meta,
                license=_cc(row["rights"]) or "; ".join(row["rights"]) or "unknown (no dc:rights)",
                headers={"Referer": view} if view else {})


# --- OpenAlex snapshot ---

S3 = "https://openalex.s3.amazonaws.com/"
MANIFEST = S3 + "data/parquet/works/manifest.json"
FILTER = ("language", "open_access.is_oa", "best_oa_location.pdf_url", "best_oa_location.license")
EXCLUDE_HOSTS = r"\.jp$|^hdl\.handle\.net$"


def openalex(fetcher, options):
    """Best-OA PDF URLs of Chinese open-access works in the OpenAlex works snapshot (parquet on S3).

    Options: max_files (int, default 5), offset (int, default 0) and seed (int: shuffle the file
    list reproducibly; default newest updated_date partitions first) choose the files; files
    (explicit s3://, https:// or updated_date=.../part_NNNN.parquet paths instead), manifest (URL),
    languages (default zh), licences (OpenAlex licence ids to keep, e.g. "cc-by,cc-by-nc", "none" for
    works without one; default all), columns (extra columns read for row groups with matches,
    default id,publication_year; title costs ~2.4 MB per row group, doi ~0.5 MB; titles with kana
    are dropped), exclude_hosts (regex on the URL host, default ``EXCLUDE_HOSTS``: Japanese
    repositories whose kanji titles OpenAlex tags zh, and Handle redirects to HTML landing pages),
    robots (bool, default True: drop URLs whose robots.txt refuses us, e.g. the CAS DSpace
    repositories and J-STAGE that many pdf_urls point to; an unreachable robots.txt counts as a
    refusal), max_items (int), log (callable). With the default 2 s interval one file takes ~50 range reads; S3 has no
    robots.txt, so a smaller --interval is reasonable for this source."""
    yield from itertools.islice(_openalex(fetcher, options), options.get("max_items"))


class _RangeFile:
    """Read-only file over HTTP: every read is one exact range request, no read-ahead."""

    def __init__(self, fetcher, url, size):
        self.fetcher, self.url, self.size = fetcher, url, size
        self.position = self.transferred = 0
        self.closed = False

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
        if end <= self.position:
            return b""
        wanted = end - self.position
        _, body = self.fetcher.get(self.url, headers={"Range": f"bytes={self.position}-{end - 1}"}, stream_limit=wanted)
        if len(body) != wanted:
            raise FetchError("short_range", self.url)
        self.position, self.transferred = end, self.transferred + wanted
        return body

    def close(self):
        self.closed = True


def _https(path):
    if path.startswith("s3://openalex/"):
        return S3 + path[len("s3://openalex/"):]
    return path if path.startswith(("http://", "https://")) else S3 + "data/parquet/works/" + path.lstrip("/")


def _size(fetcher, url):
    response, _ = fetcher.get(url, headers={"Range": "bytes=0-0"})
    return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def _openalex_files(fetcher, options):
    if options.get("files"):
        yield from ((url, None) for url in map(_https, _list(options["files"])))
        return
    entries = fetcher.json(options.get("manifest") or MANIFEST)["files"]
    files = [(_https(e["url"]), (e.get("meta") or {}).get("content_length")) for e in reversed(entries)]
    if options.get("seed") is not None:
        random.Random(options["seed"]).shuffle(files)
    offset = int(options.get("offset", 0))
    yield from files[offset:offset + int(options.get("max_files", 5))]


def _flat(table):
    import pyarrow as pa
    while any(pa.types.is_struct(field.type) for field in table.schema):
        table = table.flatten()
    return table


def _openalex(fetcher, options):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    log = options.get("log") or _quiet
    languages = pa.array([str(x) for x in _list(options.get("languages", "zh"))])
    licences = {str(x).lower() for x in _list(options.get("licences"))}
    extra = [c for c in _list(options.get("columns", "id,publication_year")) if c not in FILTER]
    robots = options.get("robots", True) and getattr(fetcher, "robots_for", None)
    exclude = re.compile(options.get("exclude_hosts", EXCLUDE_HOSTS) or r"(?!)", re.I)
    seen = set()
    for url, size in _openalex_files(fetcher, options):
        try:
            source = _RangeFile(fetcher, url, size or _size(fetcher, url))
            parquet = pq.ParquetFile(pa.PythonFile(source, mode="r"), pre_buffer=False)
            groups = range(parquet.metadata.num_row_groups)
        except (FetchError, OSError, pa.ArrowException) as error:
            log(f"openalex: skipped {url}: {error}")
            continue
        for group in groups:
            try:
                table = _flat(parquet.read_row_group(group, columns=list(FILTER), use_threads=False))
                mask = pc.and_kleene(pc.is_in(table["language"], value_set=languages), table["open_access.is_oa"])
                mask = pc.fill_null(pc.and_kleene(mask, pc.is_valid(table["best_oa_location.pdf_url"])), False)
                if not pc.any(mask).as_py():
                    continue
                rows = table.filter(mask)
                if extra:
                    more = _flat(parquet.read_row_group(group, columns=extra, use_threads=False)).filter(mask)
                    for name in more.column_names:
                        rows = rows.append_column(name, more[name])
            except (FetchError, OSError, pa.ArrowException) as error:
                log(f"openalex: stopped reading {url} at row group {group}: {error}")
                break
            for row in rows.to_pylist():
                candidate = _openalex_candidate(row, url, licences, seen)
                if candidate and exclude.search(candidate["meta"]["host"] or ""):
                    continue
                if candidate and robots and not robots(candidate["url"]).allowed(candidate["url"]):
                    log(f"openalex: robots.txt refuses {candidate['url']}")
                elif candidate:
                    yield candidate
        log(f"openalex: {url}: {source.transferred / 1e6:.1f} MB read")


def _openalex_candidate(row, file_url, licences, seen):
    url, licence = (row.pop("best_oa_location.pdf_url") or "").strip(), row.pop("best_oa_location.license")
    titles = row.pop("title", None), row.pop("display_name", None)
    title = titles[0] or titles[1]
    work = row.pop("id", None)
    if not url.startswith(("http://", "https://")) or url in seen or work in seen:
        return None
    if (licences and (licence or "none").lower() not in licences) or (title and PERSONAL.search(title)):
        return None
    if title and KANA.search(title):
        return None
    seen.update({url, work} - {None})
    meta = dict(openalex_id=work, language=row.pop("language"), licence_id=licence, file=file_url,
                host=urlsplit(url).hostname)
    row.pop("open_access.is_oa", None)
    meta.update({name.rsplit(".", 1)[-1]: value for name, value in row.items()})
    name = licence and ("CC " + licence[3:].upper() if licence.startswith("cc-") else licence.replace("-", " "))
    return dict(key=f"openalex:{work.rsplit('/', 1)[-1]}" if work else f"openalex:{url}", url=url, title=title,
                type_hint="pdf", license=name or "unknown (no licence in OpenAlex)", meta=meta)


# --- NTU theses ---

NTU = "https://tdr.lib.ntu.edu.tw/jspui/"


def ntu_theses(fetcher, options):
    """Theses of tdr.lib.ntu.edu.tw whose authors authorised worldwide public access.

    Options: maps (htmlmap page numbers: int, list or "200-236"; default every page of the index),
    order ("newest" (default) or "oldest"), max_maps (int), rights (substring DC.rights must
    contain, default 全球公開), language (regex on citation_language, default ^zh; "" keeps all),
    max_items (int), log (callable). Every request waits robots.txt's Crawl-delay of 10 s."""
    yield from itertools.islice(_ntu(fetcher, options), options.get("max_items"))


def _ntu_maps(fetcher, options, log):
    value = options.get("maps")
    if isinstance(value, str) and "-" in value:
        first, last = _span(value)
        numbers = list(range(first, last + 1))
    elif value not in (None, ""):
        numbers = [int(x) for x in _list(value)]
    else:
        index = _page(fetcher, NTU + "htmlmap", log) or ""
        numbers = sorted({int(n) for n in re.findall(r"htmlmap\?map=(\d+)", index)})
    numbers = sorted(numbers, reverse=options.get("order", "newest") == "newest")
    return numbers[:options.get("max_maps")]


def _ntu(fetcher, options):
    log = options.get("log") or _quiet
    newest = options.get("order", "newest") == "newest"
    rights, language = options.get("rights", "全球公開"), re.compile(options.get("language", "^zh"), re.I)
    seen = set()
    for number in _ntu_maps(fetcher, options, log):
        text = _page(fetcher, f"{NTU}htmlmap?map={number}", log)
        handles = list(dict.fromkeys(re.findall(r"/handle/(\d+/\d+)", text or "")))
        for handle in reversed(handles) if newest else handles:
            if handle in seen:
                continue
            seen.add(handle)
            item = f"{NTU}handle/{handle}?mode=full"
            got = _head(fetcher, item, log)
            if not got or got[1] is None:
                continue
            metas = got[1]
            pdf, title = _first(metas, "citation_pdf_url"), _first(metas, "citation_title", "dc.title")
            granted = next((r for r in metas.get("dc.rights", []) if rights in r), None)
            code = _first(metas, "citation_language", "dc.language")
            # DSpace writes bitstream sequence -1 when no readable file exists (embargo not yet lifted); it 404s.
            if not pdf or "/-1/" in pdf or not granted or not language.search(code) or PERSONAL.search(title):
                continue
            degree = next((d for d in metas.get("dc.description", []) if d in ("碩士", "博士")), None)
            meta = dict(handle=handle, item=got[0], map=number, rights=granted, language=code, degree=degree,
                        year=_first(metas, "dcterms.issued") or None, date=_first(metas, "citation_date") or None,
                        doi=next((i for i in metas.get("dc.identifier", []) if i.startswith("10.")), None),
                        host=urlsplit(urljoin(got[0], pdf)).hostname)
            yield dict(key=f"ntu_theses:{handle}", url=urljoin(got[0], pdf), title=title, type_hint="pdf",
                       license=f"author authorisation {granted}: worldwide public access, copyright with the author",
                       meta=meta, headers={"Referer": got[0]})
