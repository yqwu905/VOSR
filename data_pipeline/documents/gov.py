"""Government document sources (PRC Copyright Law Art. 5): MOJ regulations, State Council
gazette issues and the attachments of TRS/WebPlus CMS pages.

moj_xzfg: https://xzfg.moj.gov.cn/search2.html?PageIndex=N lists the ~606 current
administrative regulations, 10 per page (total in the hidden #law-total input). Each LawID
downloads as /law/download?LawID=N (DOCX, no file extension) and &type=pdf (labelled
图片版 but a text PDF); superseded versions (历史沿革) download the same way. robots.txt is 404.

gazette: https://www.gov.cn/gongbao/gbgl.json maps 年 -> 第N号 -> issue page, whose li.gbpdf
anchor holds the issue PDF: .../issue_N/material/gwygbYYYYNN.pdf (late 2023-, a few
material/YYYYNN.pdf) or /gbgl/<32hex>/files/<32hex>.pdf (2020-2023); some issues have an empty
href (no PDF). Issues weigh 13-216 MB (doc-fetch --max-mb caps them) and many pages are
outlined glyphs without a text layer. www.gov.cn resets connections after ~60 requests at 1 s
spacing, so keep the default 2 s interval.

trs_attachments: TRS listings page as index.html, index_1.html, ... (createPageHTML(count, 0,
'index', 'html'), `countPage = N` or a page-count variable); Shanghai's page 2 is index_2.html;
WebPlus lists as list.htm, list2.htm, ... Detail pages (tYYYYMMDD_N.html) link attachments next
to themselves (./P0....pdf, ./W0....docx) or under /cmsres/ (Shanghai); MOST mirrors them under
/protect/ (dropped). npc.gov.cn fails TLS through the proxy and MOF department subdomains are
http-only, so those presets use http://. Links whose title suggests a list of people are skipped.
Listings, detail pages and attachments must be on gov.cn hosts and, except for the mee_standards
preset (the gov_standards source: recommended standards are protected works), outside the MEE
standards tree; other listing URLs raise PermissionError, other pages and files are skipped.
"""

import html
import itertools
import posixpath
import re
import time
from urllib.parse import unquote, urljoin, urlsplit

from .fetch import FetchError

LICENSE = "PRC Copyright Law Art. 5(1): official document of a state organ"
PERSONAL = re.compile(r"名单|名册|身份证|拟录用|拟聘用|面试人员|考生")
EXTENSIONS = ("pdf", "doc", "docx", "wps", "ppt", "pptx")
TYPE_HINT = {"wps": "doc", "dps": "ppt"}

ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.S | re.I)
HREF = re.compile(r"""\bhref\s*=\s*["']([^"']*)["']""", re.I)
TITLE_ATTR = re.compile(r"""\btitle\s*=\s*["']([^"']*)["']""", re.I)
REFRESH = re.compile(r"""http-equiv\s*=\s*["']?refresh["']?[^>]*?url\s*=\s*([^"'\s>]+)""", re.I)
PAGE_COUNT = (r"createPageHTML\(\s*(\d+)", r"\bcountPage\s*=\s*(\d+)", r"\btotalPage\s*:\s*(\d+)",
              r"""\btotal\s*:\s*["']?(\d+)""", r"""class\s*=\s*["']all_pages["'][^>]*>\s*(\d+)""")
DETAIL = r"/t\d{8}_\d+\.s?html?$|/c\d+a\d+/page\.(?:htm|psp)$"

XZFG = "https://xzfg.moj.gov.cn/"
GAZETTE = "https://www.gov.cn/gongbao/"
MEE = "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/"
# Leaf categories of 标准文本; the parent categories only meta-refresh to their first child.
MEE_LEAVES = ("shjbh/swrwpfbz dqhjbh/dqgdwrywrwpfbz dqhjbh/dqydywrwpfbz shjbh/shjzlbz dqhjbh/dqhjzlbz "
              "wlhj/hjzspfbz wlhj/shjzlbz trhj gthw/gtfwwrkzbz gthw/wxfwjbffbz gthw/qtxgbz hxxhj/dcfsbz "
              "hxxhj/fsxhjbz hxxhj/xgjcffbz hxxhj/xgbz shjbh/xgbzh dqhjbh/xgbz jcffbz hp pwxk stzl kxxjszn "
              "wrfzjszc other/qjscbz other/hbcpjsyq other/hbysjsgf other/hjbhgc other/hjbz other/pjjsdz "
              "other/xxbz other/qt").split()
MOST = "https://www.most.gov.cn/"

PRESETS = {
    # 首都之窗 政策文件: one page with ~1k detail links. The 公报PDF attachments use Founder
    # fonts whose text layer is garbled although the pages render cleanly.
    "beijing": dict(listing=["https://www.beijing.gov.cn/zhengce/zhengcefagui/"]),
    # 市政府文件: ~357 pages; page 2 is index_2.html. Press-conference pages have no files.
    "shanghai": dict(listing=["https://www.shanghai.gov.cn/nw12344/index.html"], page_offset=1,
                     detail=r"/nw12344/\d{8}/[0-9a-f]{32}\.html$"),
    # MOST notices, normative and other documents; many notices are recruitment lists.
    "most": dict(listing=[MOST + "tztg/", MOST + "xxgk/xinxifenlei/fdzdgknr/fgzc/gfxwj/",
                          MOST + "xxgk/xinxifenlei/fdzdgknr/qtwj/"]),
    # MOF 政策发布 links detail pages on department subdomains (http only; 502s happen).
    "mof": dict(listing=["https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/"], scope=r"[a-z]+\.mof\.gov\.cn/"),
    # MEE standard full texts (GB discharge standards, HJ methods and guidelines).
    "mee_standards": dict(listing=[MEE + leaf + "/" for leaf in MEE_LEAVES], extensions=["pdf"],
                          license="standard text: mandatory GB treated as regulation-like (NCAC 1999 reply); "
                                  "recommended HJ/GB/T standards may be protected, internal research use"),
    # NPC 报告: report PDFs are linked from the first listing pages themselves; detail pages are HTML.
    "npc_reports": dict(listing=["http://www.npc.gov.cn/npc/c2/c12435/c12491/"], details=False, max_pages=3),
    # 吉林省自然资源厅 政策解读: some items are PPT图文解读 decks (./P0....pptx).
    "jilin_zcjd": dict(listing=["https://zrzy.jl.gov.cn/zwgk/fgwj/zcjd/"]),
}
PRESET_HOSTS = {urlsplit(url).hostname for preset in PRESETS.values() for url in preset["listing"]}


def _quiet(message):
    pass


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _decode(body):
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("gb18030", "replace")  # older TRS sites still serve GBK


def _page(fetcher, url, log, headers=None):
    """(final URL, HTML text), or None when the page cannot be fetched."""
    try:
        response, body = fetcher.get(url, headers=headers or {})
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None
    return getattr(response, "url", None) or url, _decode(body)


def _anchors(page_url, text):
    """(absolute URL without fragment, label) of each http(s) <a href>; the label is the link
    text, or the title attribute where listings truncate the text and keep the full title there."""
    for match in ANCHOR.finditer(text):
        href = HREF.search(match.group(1))
        if not href:
            continue
        url = urljoin(page_url, html.unescape(href.group(1).strip())).split("#")[0]
        if url.startswith(("http://", "https://")):
            label, title = _text(match.group(2)), TITLE_ATTR.search(match.group(1))
            title = html.unescape(title.group(1)).strip() if title else ""
            yield url, title if len(title) > len(label) and _cjk(title) else label


def _list(value):
    return value.split(",") if isinstance(value, str) else list(value)


def _bare(url):
    return url.split("://", 1)[-1]


def _cjk(text, least=1):
    return sum("一" <= c <= "鿿" for c in text) >= least


def _official(url, preset=None):
    """On a government host, and outside the MEE standards tree unless walking the mee_standards preset."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if "@" in parts.netloc or "\\" in url or not (host == "gov.cn" or host.endswith(".gov.cn") or host in PRESET_HOSTS):
        return False  # userinfo and backslashes make requests and urlsplit disagree on the host
    path = "/" + posixpath.normpath(unquote(parts.path) or "/").lstrip("/").lower() + "/"
    mee = host in ("www.mee.gov.cn", "mee.gov.cn") and path.startswith(urlsplit(MEE).path)
    return preset == "mee_standards" or not mee


# --- 国家行政法规库 ---

def moj_xzfg(fetcher, options):
    """Administrative regulations of xzfg.moj.gov.cn, one candidate per LawID and format.

    Options: formats (list or comma string of "docx"/"pdf", default both), history (bool,
    default False: also the superseded versions under 历史沿革), start_page (int, default 1),
    max_pages (int, default all ~61), max_items (int), log (callable)."""
    formats = _list(options.get("formats", ["docx", "pdf"]))
    yield from itertools.islice(_xzfg(fetcher, formats, options), options.get("max_items"))


def _xzfg(fetcher, formats, options):
    log = options.get("log") or _quiet
    page, last, seen = int(options.get("start_page", 1)), None, set()
    end = page + int(options["max_pages"]) if options.get("max_pages") else None
    while (last is None or page <= last) and (end is None or page < end):
        listing = f"{XZFG}search2.html?PageIndex={page}"
        got = _page(fetcher, listing, log)
        if got is None and last is None:
            return
        if got is not None:
            total = re.search(r"""id=["']law-total["'][^>]*value=["'](\d+)""", got[1])
            last = -(-int(total.group(1)) // 10) if total else page
            for law in _xzfg_items(got[1], options.get("history", False)):
                if law["law_id"] in seen:
                    continue
                seen.add(law["law_id"])
                title = law.pop("title")
                for fmt in formats:
                    yield dict(key=f"moj_xzfg:{law['law_id']}:{fmt}",
                               url=f"{XZFG}law/download?LawID={law['law_id']}" + ("&type=pdf" if fmt == "pdf" else ""),
                               title=title, type_hint=fmt, license=LICENSE,
                               meta=dict(law, page=page, host="xzfg.moj.gov.cn",
                                         detail=f"{XZFG}front/law/detail?LawID={law['law_id']}"))
        page += 1


def _xzfg_items(text, history):
    for block in re.split(r"""<li\s+class=["']list-item["']""", text)[1:]:
        main = re.search(r"""class=["']title["'][^>]*>\s*<span>\s*<a[^>]*LawID=(\d+)[^>]*>(.*?)</a>""", block, re.S)
        if not main:
            continue
        published = re.search(r"(\d{4}-\d{2}-\d{2})\s*公布", block)
        effective = re.search(r"(\d{4}-\d{2}-\d{2})\s*施行", block)
        yield dict(law_id=int(main.group(1)), title=_text(main.group(2)), current=True,
                   published=published and published.group(1), effective=effective and effective.group(1))
        if history:
            versions = r"""data-time=["']([\d-]+)["']>\s*<a[^>]*LawID=(\d+)[^>]*>(.*?)</a>"""
            for date, law_id, title in re.findall(versions, block, re.S):
                if int(law_id) != int(main.group(1)):
                    yield dict(law_id=int(law_id), title=_text(title), current=False, version_date=date,
                               current_law_id=int(main.group(1)))


# --- 国务院公报 ---

def gazette(fetcher, options):
    """State Council Gazette issue PDFs from gbgl.json, newest first, one candidate per issue.

    Options: years (int, list of ints or "2021-2026"; default 2021 to the current year),
    max_items (int), log (callable)."""
    yield from itertools.islice(_gazette(fetcher, options), options.get("max_items"))


def _years(value):
    if value is None:
        return range(2021, time.localtime().tm_year + 1)
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        first, _, last = value.replace("..", "-").partition("-")
        return range(int(first), int(last or first) + 1)
    return [int(year) for year in value]


def _gazette(fetcher, options):
    log = options.get("log") or _quiet
    volumes = fetcher.json(GAZETTE + "gbgl.json")[0]["values"]
    for year in sorted(_years(options.get("years")), reverse=True):
        for name, issue in volumes.get(f"{year}年", {}).items():
            got = issue.get("gname") and _page(fetcher, issue["gname"], log)
            if not got:
                continue
            page_url, text = got
            href = re.search(r"""class=["']gbpdf["'][^>]*>\s*<a[^>]*href=["']([^"']*)""", text)
            href = href.group(1).strip() if href else ""
            if not href:  # older layouts: any link to the issue PDF
                found = re.search(r"""href=["']([^"']*(?:gwygb\d+|/gbgl/[0-9a-f]{32}/files/[0-9a-f]{32})\.pdf)""", text)
                href = found.group(1) if found else ""
            if not href:
                log(f"no PDF for {year} {name}")
                continue
            number = re.search(r"\d+", name)
            number = int(number.group()) if number else name
            url = urljoin(page_url, href)
            yield dict(key=f"gov_gazette:{year}:{number}", url=url, title=f"国务院公报{year}年{name}",
                       type_hint="pdf", license=LICENSE,
                       meta=dict(year=year, issue=number, issue_url=page_url, host=urlsplit(url).hostname))


# --- TRS / WebPlus attachments ---

def trs_attachments(fetcher, options):
    """Attachments (PDF, Word, WPS, PowerPoint) of government CMS pages, walked from listings.

    Options: preset (one of PRESETS) and/or listing (URL or list of listing URLs; replaces the
    preset's), max_pages (int listing pages per listing, default the preset's or 50),
    max_items (int), extensions (list, default pdf doc docx wps ppt pptx), details (bool,
    default True: follow detail pages; False takes only files linked from listings), detail
    (regex on the detail URL path, default TRS tYYYYMMDD_N.html and WebPlus .../page.htm),
    scope (regex matched against the scheme-less detail URL, default the listing's
    directory), page_offset (int, 1 where page 2 is index_2.html), license (str), log.
    Listings must be government pages outside the MEE standards tree (PermissionError otherwise)
    unless the preset is mee_standards; detail pages and attachments elsewhere are skipped."""
    settings = dict(PRESETS[options["preset"]]) if options.get("preset") else {}
    settings.update(options)
    if not settings.get("listing"):
        raise ValueError(f"trs_attachments needs --opt preset=<{'|'.join(PRESETS)}> or --opt listing=<url>")
    listing = settings["listing"]
    settings["listing"] = [listing] if isinstance(listing, str) else list(listing)
    if refused := [url for url in settings["listing"] if not _official(str(url), settings.get("preset"))]:
        raise PermissionError(f"trs_attachments: {refused} are not government listings (gov.cn), or are MEE standards "
                              "listings, which belong to the gov_standards source (terms public_web)")
    yield from itertools.islice(_walk(fetcher, settings), settings.get("max_items"))


def _page_urls(url, text, offset=0):
    """Every page of a listing, from its first page's pagination script or links."""
    path = urlsplit(url).path
    count = next((int(m.group(1)) for p in PAGE_COUNT for m in [re.search(p, text)] if m), None)
    webplus = re.search(r"/list\d*\.(\w+)$", path)
    if webplus:
        return [url] + [urljoin(url, f"list{i + 1}.{webplus.group(1)}") for i in range(1, count or 1)]
    named = re.search(r"""createPageHTML\(\s*\d+\s*,\s*\d+\s*,\s*["']([\w-]+)["']\s*,\s*["'](\w+)["']""", text)
    stem, ext = named.groups() if named else ("index", (re.search(r"/index\.(\w+)$", path) or [None, "html"])[1])
    if count is None:
        numbers = [int(n) for n in re.findall(rf"\b{stem}_(\d+)\.{ext}\b", text)]
        count = max(numbers) + 1 - offset if numbers else 1
    return [url] + [urljoin(url, f"{stem}_{i + offset}.{ext}") for i in range(1, count)]


def _meta(text, name):
    found = re.search(rf"""<meta\s+name=["']{name}["']\s+content=["']([^"']*)""", text, re.I)
    return html.unescape(found.group(1)).strip() if found else ""


def _date(url):
    """Publication date in TRS names: tYYYYMMDD_N.html pages, P0YYYYMMDD... / W0YYYYMMDD... files."""
    found = re.search(r"/(?:t|[PW]0)(\d{4})(\d{2})(\d{2})(?:_|\d{6,}\.)", url)
    return "-".join(found.groups()) if found else None


def _attachment(url, extensions):
    path = urlsplit(url).path.lower()
    if "/protect/" in path:
        return None
    return next((e for e in extensions if path.endswith("." + e)), None)


def _walk(fetcher, settings):
    log = settings.get("log") or _quiet
    listings, preset = settings["listing"], settings.get("preset")
    extensions = [e.lower().lstrip(".") for e in _list(settings.get("extensions") or EXTENSIONS)]
    detail = re.compile(settings.get("detail") or DETAIL)
    max_pages, offset = int(settings.get("max_pages") or 50), int(settings.get("page_offset", 0))
    files, pages = set(), set()

    def attachments(anchors, page_url, page_title="", published=None):
        for url, label in anchors:
            extension = _attachment(url, extensions)
            if not extension or _bare(url) in files or PERSONAL.search(label) or not _official(url, preset):
                continue
            files.add(_bare(url))
            # Link texts like 附件 or 下载 say nothing; the page title does.
            yield dict(key=_bare(url), url=url, title=label if _cjk(label, 4) else page_title or label,
                       type_hint=TYPE_HINT.get(extension, extension), license=settings.get("license", LICENSE),
                       headers={"Referer": page_url},
                       meta=dict(page=page_url, page_title=page_title, link_text=label,
                                 published=published or _date(url), host=urlsplit(url).hostname,
                                 preset=preset))

    for listing in listings:
        got = _page(fetcher, listing, log)
        refresh = got and REFRESH.search(got[1])
        if refresh:  # a category stub that redirects to its first sub-listing
            got = _page(fetcher, urljoin(got[0], refresh.group(1)), log)
        if got is None:
            continue
        first_url, first_text = got
        if not _official(first_url, preset):
            log(f"skip {listing}: it leads to {first_url}, not a government listing for this source")
            continue
        scope = re.compile(settings.get("scope") or re.escape(_bare(first_url).rsplit("/", 1)[0] + "/"))
        failures = 0
        for index, page_url in enumerate(_page_urls(first_url, first_text, offset)[:max_pages]):
            got = (first_url, first_text) if index == 0 else _page(fetcher, page_url, log)
            if got is None:
                failures += 1
                if failures >= 2:  # past the real last page, or the site is refusing us
                    break
                continue
            failures, details, anchors = 0, [], list(_anchors(*got))
            yield from attachments(anchors, got[0])
            for url, label in anchors:
                if (settings.get("details", True) and detail.search(urlsplit(url).path) and scope.match(_bare(url))
                        and _bare(url) not in pages and not PERSONAL.search(label) and _official(url, preset)):
                    pages.add(_bare(url))
                    details.append((url, label))
            for url, label in details:
                page = _page(fetcher, url, log, headers={"Referer": got[0]})
                if page is None:
                    continue
                title = _meta(page[1], "ArticleTitle") or label
                if PERSONAL.search(title):
                    continue
                yield from attachments(_anchors(*page), page[0], title, _meta(page[1], "PubDate") or _date(url))
