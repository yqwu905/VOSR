"""Government procurement notices: attachments of 北京市政府采购网 (www.ccgp-beijing.gov.cn).

Listings (measured 2026-10): the 信息公告 page only fills an iframe. Each notice type has static
listings /xxgg/{sjxxgg|qjxxgg}/{type|qj<type>}/A002004{001|002}<NNN>index_N.htm (市级|区级), 14 notices
per page, newest first (<li><a href=notice>title…</a><span class="datetime">YYYY-MM-DD</span>), N up to
the hidden #TotalPage, which every listing caps at 143: a listing holds only the newest 2,002 notices
(招标公告 back to 2026-07, 更正公告 to 2025-01, 其他公告 to 2016-11). 集采成交 is empty.

Older notices come from the site's own search API: POST /api/search/query with a JSON form (``FORM``).
Keywords must not be empty (the OR query in ``FORM`` matches 656k notices in site A002, index 信息公告,
2010-); ``page`` pages (``current`` does not), size 100 works and offset 12,000 still answered;
range {after, before} takes ISO times. Rows carry channel A002004<level><type>, path and publishedAt,
but their text is a ~200-character snippet, so every notice page is opened. Site A003 (政府采购-历史公告)
holds the previous system's notices up to 2017-09 (招标公告 index: 83k, from the 2000s), one index per
type. Migrated notices carry /2024/12/ in their URL whatever their date.

Notice pages: <h1><p>title</p></h1><span>YYYY-MM-DD</span>; attachments are links to
/content_file/xxgg/<level>/<type>/YYYY/M/<32hex>.(docx|pdf) (current system) or
/wwprotect/P0YYYYMM/P0YYYYMMDD/P0<digits>.(doc|pdf) (migrated and A003 notices). robots.txt is 404.
Measured attachments: tender notices and 采购需求 as born-digital DOCX/PDF (招标公告), full 招标文件终稿
PDFs (中标公告), framework agreements (其他公告), 单一来源 公示 DOCX. 中小企业声明函, 专家论证意见 (signed
scans with 证件) and lists of people are skipped by name; 合同公告 attachments are copier scans of signed
contracts with supplier personal data, so htgg is opt-in. Tender documents name the agencies' contact
staff, as every procurement notice must.

Measured yield (notices with document attachments): listing page 1 of the 10 default listings 64/140; search
2021-06-08 招标公告 38/75 (DOCX, PDF, DOC under /wwprotect/); 2025-06-10 区级 招标/中标 32/79; 2015-06-09
(A003) 1/66, as tender documents were then sold on site. 48 sampled files rendered to 534 kept pages
(first 30 pages each), all text-layer pages but one 1-page scan, CJK share median 0.78; 招标文件 run
70-200 pages.
"""

import html
import itertools
import math
import re
from datetime import date, timedelta
from urllib.parse import urljoin, urlsplit

from .fetch import FetchError

HOST = "http://www.ccgp-beijing.gov.cn"
API = HOST + "/api/search/query"
SEARCH_PAGE = {"Referer": HOST + "/webapp/index/search.html"}
LICENSE = "procurement notice attachment disclosed under 政府采购法; rights stay with the purchaser or agency"
TYPES = {"zbgg": ("001", "招标公告"), "zbggs": ("002", "中标公告"), "htgg": ("003", "合同公告"),
         "gzgg": ("004", "更正公告"), "fbgg": ("005", "废标公告"), "dygg": ("006", "单一公告"), "qtgg": ("007", "其他公告")}
DEFAULT_TYPES = ("zbgg", "zbggs", "gzgg", "dygg", "qtgg")
LEVELS = {"city": ("sjxxgg", "", "001"), "district": ("qjxxgg", "qj", "002")}
LEVEL_NAMES = {"市级": "city", "shi": "city", "区级": "district", "qu": "district"}
NOTICES = "1331802009359089664"  # A002 index 信息公告
# A003 index per type; that system ended in September 2017.
HISTORY = {"zbgg": "1284183408037068800", "zbggs": "1284183284426735616", "htgg": "1284183509795078144",
           "dygg": "1284183588090150912", "qtgg": "1284183371668258816"}
HISTORY_END = date(2017, 9, 16)
FORM = dict(site="A002", indexes=[NOTICES], keywords="公告 采购 合同 公示", fields=None, range=None, orderBy="TIME",
            page=1, size=100, requrl="www.ccgp-beijing.gov.cn")
EXTENSIONS = ("pdf", "doc", "docx", "wps", "rtf", "ppt", "pptx")
TYPE_HINT = {"wps": "doc"}
# Attachment names of personal data, signed or sealed scans and supplier paperwork; result and candidate
# notices restate the page, which names the experts and project leaders; scanner file names; company names.
SKIP = re.compile(r"名单|名册|身份证|证件|证书|证明|专家|论证|签字|签名|签到|声明函|承诺|营业执照|执照|资质|授权|委托书|"
                  r"简历|推荐|扫描|盖章|签章|报价|评审|评标|验收|业绩|人员|投标文件|响应文件|候选人|中标公告|成交公告|"
                  r"结果公告|结果确认|^[\d\s_.()（）-]+\.\w+$|公司\.\w+$|^(?:IMG|DSC|scan)", re.I)
PERSONAL_NOTICE = re.compile(r"名单|名册|身份证")

ROW = re.compile(r"""<li>\s*<a\s+href=["']([^"']+?\.htm)["'][^>]*>(.*?)</a>\s*<span\s+class=["']datetime["'][^>]*>"""
                 r"""\s*(\d{4}-\d{2}-\d{2})""", re.S)
TOTAL = re.compile(r"""id=["']TotalPage["'][^>]*value=["'](\d+)""")
HEAD = re.compile(r"<h1>\s*(?:<p>)?(.*?)(?:</p>)?\s*</h1>\s*<span>\s*(\d{4}-\d{2}-\d{2})", re.S)
ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.S | re.I)
HREF = re.compile(r"""\bhref\s*=\s*["']([^"']*)["']""", re.I)
FILE_ID = re.compile(r"/([0-9a-f]{32}|P0\d{10,})\.\w+$", re.I)
NOTICE_ID = re.compile(r"/([0-9a-f]{32})\.htm$")


def _quiet(message):
    pass


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _list(value):
    return value.split(",") if isinstance(value, str) else list(value)


def _page(fetcher, url, log, headers=None):
    try:
        response, body = fetcher.get(url, headers=headers or {})
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None
    return getattr(response, "url", None) or url, body.decode("utf-8", "replace")


def _kind(value):
    names = {name: key for key, (_, name) in TYPES.items()}
    kind = names.get(value, value)
    if kind not in TYPES:
        raise ValueError(f"unknown notice type {value!r}; one of {', '.join(TYPES)}")
    return kind


def _level(value):
    level = LEVEL_NAMES.get(value, value)
    if level not in LEVELS:
        raise ValueError(f"unknown level {value!r}; city (市级) or district (区级)")
    return level


def listing_url(level, kind, page):
    directory, prefix, code = LEVELS[level]
    return f"{HOST}/xxgg/{directory}/{prefix}{kind}/A002004{code}{TYPES[kind][0]}index_{page}.htm"


def ccgp_beijing(fetcher, options):
    """Attachments (PDF, Word, WPS, PowerPoint) of Beijing procurement notices, newest first.

    Options: types (list or comma string of zbgg 招标公告, zbggs 中标公告, gzgg 更正公告, fbgg 废标公告,
    dygg 单一公告, qtgg 其他公告, htgg 合同公告; default all but fbgg and htgg), levels (city/市级 and/or
    district/区级, default both), pages (listing pages per type and level, default all, at most 143),
    since ("YYYY-MM-DD": stop at older notices), search (bool, default False: list notices through the
    site search between since and until instead of the capped listings; needs since), until
    ("YYYY-MM-DD", default today), window_days (search window, default 7), history (bool, with search:
    also the pre-2017-09 archive A003, which has no level), extensions (default pdf doc docx wps rtf
    ppt pptx), max_items (int), log (callable)."""
    yield from itertools.islice(_candidates(fetcher, options), options.get("max_items"))


def _candidates(fetcher, options):
    log = options.get("log") or _quiet
    kinds = [_kind(t) for t in _list(options.get("types") or DEFAULT_TYPES)]
    levels = [_level(level) for level in _list(options.get("levels") or LEVELS)]
    extensions = [e.lower().lstrip(".") for e in _list(options.get("extensions") or EXTENSIONS)]
    notices = (_searched if options.get("search") else _listed)(fetcher, kinds, levels, options, log)
    opened, files = set(), set()
    for notice in notices:
        if notice["url"] in opened or PERSONAL_NOTICE.search(notice["title"]):
            continue
        opened.add(notice["url"])
        got = _page(fetcher, notice["url"], log)
        if got is None:
            continue
        page_url, text = got
        head = HEAD.search(text)
        title, published = (_text(head.group(1)), head.group(2)) if head else (notice["title"], notice["date"])
        if PERSONAL_NOTICE.search(title):
            continue
        hex_id = NOTICE_ID.search(urlsplit(page_url).path)
        for number, (url, name) in enumerate(_attachments(page_url, text, extensions)):
            if url in files or SKIP.search(name):
                continue
            files.add(url)
            extension = urlsplit(url).path.rsplit(".", 1)[-1].lower()
            file_id = FILE_ID.search(urlsplit(url).path)
            key = file_id.group(1) if file_id else f"{hex_id.group(1) if hex_id else _bare(page_url)}:{number}"
            yield dict(key=f"ccgp_bj:{key}", url=url, title=f"{title} / {name}" if name else title,
                       type_hint=TYPE_HINT.get(extension, extension), license=LICENSE, headers={"Referer": page_url},
                       meta=dict(type=notice["type"], type_name=TYPES[notice["type"]][1], level=notice["level"],
                                 published=published or notice["date"], notice=page_url, notice_title=title,
                                 attachment=name, host=urlsplit(url).hostname))


def _bare(url):
    return url.split("://", 1)[-1]


def _attachments(page_url, text, extensions):
    for match in ANCHOR.finditer(text):
        href = HREF.search(match.group(1))
        if not href:
            continue
        url = urljoin(page_url, html.unescape(href.group(1).strip())).split("#")[0]
        if url.startswith(("http://", "https://")) and urlsplit(url).path.lower().rsplit(".", 1)[-1] in extensions:
            yield url, _text(match.group(2))


def _listed(fetcher, kinds, levels, options, log):
    since, limit = options.get("since"), min(int(options.get("pages") or 143), 143)
    for level, kind in itertools.product(levels, kinds):
        page, last, failures = 1, 1, 0
        while page <= min(last, limit) and failures < 2:
            got = _page(fetcher, listing_url(level, kind, page), log)
            page += 1
            if got is None:  # a failed first page ends the listing (last stays 1)
                failures += 1
                continue
            total = TOTAL.search(got[1])
            failures, last = 0, int(total.group(1)) if total else last
            rows = ROW.findall(got[1])
            fresh = [row for row in rows if not since or row[2] >= since]
            for href, title, day in fresh:
                yield dict(url=urljoin(got[0], html.unescape(href)), title=_text(title), date=day, type=kind,
                           level=level)
            if len(fresh) < len(rows):
                break  # newest first: the rest is older than since


def _searched(fetcher, kinds, levels, options, log):
    if not options.get("since"):
        raise ValueError("ccgp_beijing search needs --opt since=YYYY-MM-DD")
    since = date.fromisoformat(options["since"])
    end = date.fromisoformat(options["until"]) + timedelta(days=1) if options.get("until") else \
        date.today() + timedelta(days=1)
    step = timedelta(days=int(options.get("window_days", 7)))
    channels = {f"A002004{LEVELS[level][2]}{TYPES[kind][0]}": (level, kind) for level in levels for kind in kinds}
    while end > since:
        start = max(since, end - step)
        for row in _query(fetcher, dict(FORM), start, end, log):
            if row.get("channel") in channels:
                yield _row(row, *channels[row["channel"]])
        if options.get("history") and start < HISTORY_END:
            for kind in kinds:
                if kind in HISTORY:
                    for row in _query(fetcher, dict(FORM, site="A003", indexes=[HISTORY[kind]]), start, end, log):
                        yield _row(row, "history", kind)
        end = start


def _row(row, level, kind):
    return dict(url=urljoin(HOST + "/", row["path"]), title=_text(row.get("title") or ""),
                date=(row.get("publishedAt") or "")[:10], type=kind, level=level)


def _query(fetcher, form, start, end, log):
    """Every row of one search window, page by page."""
    form["range"] = {"after": f"{start.isoformat()}T00:00:00.000Z", "before": f"{end.isoformat()}T00:00:00.000Z"}
    page, pages = 1, 1
    while page <= pages:
        form["page"] = page
        try:
            answer = fetcher.json(API, method="POST", json=form, headers=SEARCH_PAGE)
        except (FetchError, ValueError) as error:
            log(f"search {form['site']} {start}..{end} page {page} failed: {error}")
            return
        data = answer.get("data") if isinstance(answer, dict) else None
        if not isinstance(data, dict) or answer.get("code") != 0:
            log(f"search {form['site']} {start}..{end}: {answer.get('msg') if isinstance(answer, dict) else answer}")
            return
        rows = data.get("results") or []
        if not rows:
            return
        yield from rows
        pages, page = math.ceil(int(data.get("total") or 0) / form["size"]), page + 1
