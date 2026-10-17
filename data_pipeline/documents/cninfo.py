"""巨潮资讯 cninfo: SZSE investor-relations Office files, investor slide decks, periodic reports.

The site's own search page uses one undocumented JSON API: POST
http://www.cninfo.com.cn/new/hisAnnouncement/query (form fields as in ``FORM``, with the search
page as Referer); files are at http://static.cninfo.com.cn/<adjunctUrl>. robots.txt is 404 on
both hosts and https is flaky through the proxy, so everything uses plain http.

Gotchas (measured): pageSize is fixed at 30 and pageNum is capped at 100 (page 101+ silently
returns page 1), so a query reaches at most 3,000 rows; queries are split by halving the seDate
window down to one day, then by plate, until totalAnnouncement <= 3,000, and paging stops if a
page repeats page 1. ``totalpages`` is floor(total/30), not the page count. Unfiltered windows
of a year can 504 after ~12 s while month windows never did (keyword searches over all years
take ~2 s). column=szse covers all A-shares (SSE, SZSE, BSE); tabName=relation holds SZSE
investor-relations records, ~87% DOC/DOCX in 2012-2020 and PDF only from 2021.
"""

import html
import json
import math
import re
from datetime import date, datetime, timedelta, timezone

from .fetch import FetchError

API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC = "http://static.cninfo.com.cn/"
SEARCH = "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
HEADERS = {"Referer": SEARCH, "X-Requested-With": "XMLHttpRequest"}
FORM = dict(pageNum=1, pageSize=30, column="szse", tabName="fulltext", plate="", stock="", searchkey="", secid="",
            category="", trade="", seDate="", sortName="", sortType="", isHLtitle="true")
PAGE, PAGES = 30, 100
# Disjoint plate splits (measured): sz = szmb (incl. the former SME board) + szcy, sh = shmb + shkcp, hke = hkmb + hkcy.
PLATES = {("szse", ""): ("sz", "sh", "bj"), ("szse", "sz"): ("szmb", "szcy"), ("szse", "sh"): ("shmb", "shkcp"),
          ("hke", ""): ("hkmb", "hkcy")}
LICENSE = "issuer copyright (statutory disclosure published on cninfo)"
# Titles of personal-data lists (IR attachments include visitor sign-in sheets, registration forms and signed pledges).
PERSONAL = re.compile("名单|名册|花名册|身份证|签到|登记表|承诺书|人员姓名|通讯录")
OFFICE = ("doc", "docx", "ppt", "pptx")
KEYWORDS = ("演示文稿", "PPT", "演示材料", "推介材料", "路演材料", "业绩说明会材料", "业绩演示", "投资者推介", "交流材料",
            "业绩发布会")
DECK = r"演示|PPT|推介|路演材料|说明会材料|交流材料|发布会材料"
NOT_DECK = r"召开|举办|举行|延期|变更|取消|通知|预告|英文|English"
CST = timezone(timedelta(hours=8))


def ir_office(fetcher, options):
    """SZSE investor-relations records (tabName=relation) published as native Word/PowerPoint files.

    Options: years ("2012-2020", 2015 or [2012, 2020]) or start/end ("YYYY-MM-DD"; default 2012-01-01
    to 2020-12-31, after which every record is a PDF); types (adjunctType list or comma string,
    default doc,docx,ppt,pptx); category ("" for all, or category_dyhd_szdy 调研 / category_lyhd_szdy
    路演 / category_mdcf_szdy 媒体采访); plate; window_months (first seDate window, default 1); form
    (extra form fields, e.g. {"stock": "000001,gssz0000001"}); max_queries (API requests, default
    20000); max_failures (consecutive failed requests before giving up, default 6); log (callable
    taking a message)."""
    query = _Query(fetcher, options)
    types = _types(options, OFFICE)
    form = _form(options, tabName="relation", category=options.get("category", ""))
    for row in query.run(form, *_range(options, "2012-01-01", "2020-12-31"), options.get("window_months", 1)):
        if _kind(row) in types and (candidate := query.candidate(row, form)):
            yield candidate


def slides(fetcher, options):
    """Investor presentation decks: title keyword searches plus native PPT/PPTX IR attachments.

    Options: keywords (list or comma string, default ``KEYWORDS``) searched on tabs (default
    relation,fulltext) of column szse; include_hk (default False: column=hke mirrors HKEXnews,
    whose terms forbid scraping and AI training) searches hk_keywords (default 演示,推介) on the
    hke fulltext tab; include / exclude (title regexes, default ``DECK`` / ``NOT_DECK``); types
    (default pdf,ppt,pptx); years or start/end (default 2010-01-01 to today); attachments (default
    True) also lists searchkey 附件 on the relation tab over attachment_years (default 2012-2020) and
    keeps its PPT/PPTX files whatever the title; window_months (default 0: one window, halved on
    overflow or 5xx); form; max_queries; max_failures; log."""
    query = _Query(fetcher, options)
    include, exclude = (re.compile(options.get(k, d), re.I) for k, d in (("include", DECK), ("exclude", NOT_DECK)))
    types = _types(options, ("pdf", "ppt", "pptx"))
    start, end = _range(options, "2010-01-01", date.today().isoformat())
    months = options.get("window_months", 0)
    searches = [("szse", tab, k) for tab in _list(options.get("tabs", ("relation", "fulltext")))
                for k in _list(options.get("keywords", KEYWORDS))]
    if options.get("include_hk", False):
        searches += [("hke", "fulltext", k) for k in _list(options.get("hk_keywords", ("演示", "推介")))]
    for column, tab, keyword in searches:
        form = _form(options, column=column, tabName=tab, searchkey=keyword)
        for row in query.run(form, start, end, months):
            title = _title(row)
            if _kind(row) in types and include.search(title) and not exclude.search(title):
                if candidate := query.candidate(row, form):
                    yield candidate
    if options.get("attachments", True):
        form = _form(options, tabName="relation", searchkey="附件")
        span = _range({"years": options.get("attachment_years", "2012-2020")}, "2012-01-01", "2020-12-31")
        for row in query.run(form, *span, months):
            if _kind(row) in {"ppt", "pptx"} & types and not exclude.search(_title(row)):
                if candidate := query.candidate(row, form):
                    yield candidate


def reports(fetcher, options):
    """A-share periodic reports from the fulltext tab, one announcement category at a time.

    Options: category (one or a list, default category_ndbg_szsh annual reports; also
    category_bndbg_szsh half-year, category_yjdbg_szsh Q1, category_sjdbg_szsh Q3); years or
    start/end (default the previous calendar year); one_per_company_per_year (default True: the
    first file seen per category, company and report year, i.e. the year in the title or else the
    announcement year; rows come newest first, so a corrected report usually wins over the
    original); exclude (title regex, default 摘要|英文|English); include (optional title regex);
    types (default pdf); plate ("" for all; sz, sh, bj, szmb, szcy, shmb, shkcp); window_months
    (default 1); form; max_queries; max_failures; log."""
    query = _Query(fetcher, options)
    last = date.today().year - 1
    start, end = _range(options, f"{last}-01-01", f"{last}-12-31")
    exclude = re.compile(options.get("exclude", "摘要|英文|English"), re.I)
    include = re.compile(options.get("include") or "", re.I)
    one, types, taken = options.get("one_per_company_per_year", True), _types(options, ("pdf",)), set()
    categories = options.get("category", "category_ndbg_szsh")
    for category in [categories] if isinstance(categories, str) else categories:
        form = _form(options, category=category)
        for row in query.run(form, start, end, options.get("window_months", 1)):
            title = _title(row)
            if _kind(row) not in types or exclude.search(title) or not include.search(title):
                continue
            year = re.search(r"(?<!\d)((?:19|20)\d\d)\s*年", title)
            company_year = (category, row.get("orgId") or row.get("secCode"), year[1] if year else _day(row)[:4])
            if one and company_year in taken:
                continue
            if candidate := query.candidate(row, form):
                taken.add(company_year)
                yield candidate


class _Query:
    """Sharded, paged listings with a request budget and a run-wide (announcementId, adjunctUrl) dedupe."""

    def __init__(self, fetcher, options):
        self.fetcher, self.left = fetcher, int(options.get("max_queries", 20000))
        self.patience, self.failures = int(options.get("max_failures", 6)), 0
        self.log = options.get("log") or (lambda message: None)
        self.seen = set()

    def post(self, form):
        if self.left <= 0:
            return None
        self.left -= 1
        if not self.left:
            self.log("cninfo: max_queries reached; stopping after this request")
        try:
            _, body = self.fetcher.get(API, method="POST", data=form, headers=HEADERS)
        except FetchError as error:
            self.failures += 1
            if self.failures >= self.patience:  # the API is down, not one slow window: stop splitting
                raise FetchError("api_down:" + error.reason, API) from error
            raise
        self.failures = 0
        try:
            return json.loads(body)
        except ValueError:
            raise FetchError("not_json", API) from None  # a block or challenge page: stop, never work around it

    def run(self, form, start, end, months):
        for low, high in _windows(start, end, months):
            yield from self.listing(form, low, high)

    def listing(self, form, start, end):
        """Rows matching ``form`` within [start, end], split until each query fits the 3,000-row window."""
        window = dict(form, seDate=f"{start}~{end}", pageNum=1)
        try:
            first = self.post(window)
        except FetchError as error:
            if not _transient(error):
                raise
            if start < end:  # long unfiltered windows time out (504): halve them
                yield from self._halves(form, start, end)
            else:
                self.log(f"cninfo: skipped {window['seDate']} {form['plate']}: {error.reason}")
            return
        if first is None:
            return
        total = first.get("totalAnnouncement") or 0
        if total > PAGE * PAGES:
            if start < end:
                yield from self._halves(form, start, end)
                return
            if plates := PLATES.get((form["column"], form["plate"])):
                for plate in plates:
                    yield from self.listing(dict(form, plate=plate), start, end)
                return
            self.log(f"cninfo: {total} rows on {start} plate {form['plate']!r}; only {PAGE * PAGES} are reachable")
        yield from self._pages(window, first, total)

    def _halves(self, form, start, end):
        middle = start + (end - start) // 2
        yield from self.listing(form, start, middle)
        yield from self.listing(form, middle + timedelta(days=1), end)

    def _pages(self, window, first, total):
        rows = first.get("announcements") or []
        if not rows:
            return
        head = rows[0].get("announcementId")
        yield from rows
        for page in range(2, min(PAGES, math.ceil(total / PAGE)) + 1):
            try:
                reply = self.post(dict(window, pageNum=page))
            except FetchError as error:
                if not _transient(error):
                    raise
                self.log(f"cninfo: skipped page {page} of {window['seDate']}: {error.reason}")
                continue
            if reply is None:
                return
            rows = reply.get("announcements") or []
            if not rows:
                return
            if rows[0].get("announcementId") == head:
                self.log(f"cninfo: page {page} of {window['seDate']} repeats page 1; stopping")
                return
            yield from rows

    def candidate(self, row, form):
        """The candidate for one row, or None for a repeat, a row without a file or a personal-data list."""
        path, title = row.get("adjunctUrl"), _title(row)
        ident = (row.get("announcementId"), path)
        if not path or ident in self.seen or PERSONAL.search(title):
            return None
        self.seen.add(ident)
        meta = dict(host="static.cninfo.com.cn", announcement_id=row.get("announcementId"), date=_day(row),
                    sec_code=row.get("secCode"), sec_name=row.get("secName"), org_id=row.get("orgId"),
                    plate=row.get("pageColumn"), size_kb=row.get("adjunctSize"), adjunct_type=row.get("adjunctType"),
                    column=form["column"], tab=form["tabName"])
        meta.update({k: form[k] for k in ("category", "searchkey") if form[k]})
        return dict(key=f"cninfo:{row.get('announcementId')}:{path.rsplit('/', 1)[-1]}", url=STATIC + path.lstrip("/"),
                    title=title, type_hint=_kind(row), license=LICENSE, meta=meta, headers={"Referer": SEARCH})


def _transient(error):
    return error.reason.startswith(("http_5", "network:")) or error.reason == "http_429"


def _windows(start, end, months):
    """[start, end] in calendar chunks of ``months`` months (one chunk when 0)."""
    while start <= end:
        stop = end
        if months:
            year, month = divmod(start.month - 1 + int(months), 12)
            stop = min(end, date(start.year + year, month + 1, 1) - timedelta(days=1))
        yield start, stop
        start = stop + timedelta(days=1)


def _range(options, start, end):
    if years := options.get("years"):
        first, last = (years[0], years[-1]) if isinstance(years, (list, tuple)) else str(years).partition("-")[::2]
        start, end = f"{first}-01-01", f"{last or first}-12-31"
    start, end = (date.fromisoformat(str(options.get(k) or d)) for k, d in (("start", start), ("end", end)))
    return start, min(end, datetime.now(CST).date())


def _form(options, **fields):
    form = dict(FORM, plate=options.get("plate", ""), **fields)
    form.update(options.get("form") or {})
    return form


def _list(value):
    return [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)


def _types(options, default):
    return {t.lower().lstrip(".") for t in _list(options.get("types", default))}


def _kind(row):
    kind = (row.get("adjunctType") or "").strip(" .").lower()
    return kind or (row.get("adjunctUrl") or "").rpartition(".")[2].lower()


def _title(row):
    return html.unescape(re.sub(r"</?em>", "", row.get("announcementTitle") or "")).strip()


def _day(row):
    if row.get("announcementTime"):
        return datetime.fromtimestamp(row["announcementTime"] / 1000, CST).date().isoformat()
    parts = (row.get("adjunctUrl") or "").split("/")
    return parts[1] if len(parts) > 2 else ""
