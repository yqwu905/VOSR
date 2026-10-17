"""Slide decks: university courseware on WebPlus/Sudy CMS sites, VALSE webinar slides, openly
licensed Chinese decks on Zenodo and 优品PPT (ypppt) courseware and templates.

webplus: WebPlus/Sudy pages (/<site>/YYYY/MMDD/c<col>a<id>/page.htm; column listings list.htm,
list2.htm, ... with the page count in em.all_pages) link attachments as
/_upload/article/files/xx/yy/<hash>/<uuid>.<ext>. The stored name is a UUID; the real title is
in sudyfile-attr="{'title':'第一章&nbsp;….ppt'}" (embedded PDF players carry it on a div with
pdfsrc=). www.htu.edu.cn works over https, pub.ouc.edu.cn only over http; both robots.txt are
404. Decks are 10-30 MB and image-heavy, many of them legacy .ppt.

valse: https://valser.org/webinar/slide/index.php/Home/Index/index.html lists ~281 directories,
newest first (275 named YYYYMMDD, the rest 短教程… with URL-encoded names); a directory page
links http://valser.org/webinar/slide/slides/<dir>/<file> with raw spaces and Chinese in the
name (quoted here and fetched over https). ~1.5 PDFs per directory, fonts embedded, only ~25%
mainly Chinese. The open upload form has let in junk: files at the slides root (e.g.
software_cracking.pdf) and Cyrillic-named PDFs in 2024-2025 directories, so both are skipped.

zenodo_cjk: the Zenodo10K metadata parquet (10,448 PPTX rows: filename, size, url, license,
title, created, doi, checksum) has 20 rows with CJK in the title or file name: 6 are Japanese
(kana) and 9 are files of one 四川大学PPT模板 record (up to 223 MB each). Files come from
https://zenodo.org/records/<id>/files/<name>?download=1. Zenodo's robots.txt disallows /api and
/search (so no search API) and sets Crawl-delay 10, which the fetcher honours. More records can
be named by id: /records/<id>/export/json (allowed) holds their licence and file list.

ypppt: https://www.ypppt.com/kejian/yuwen/ lists ~763 语文 lessons under grade headings
(/special/<pinyin>/, a few straight /article/YYYY/<id>.html); a lesson page's ul.posts lists its
decks. Template listings are /moban/, /moban/list-N.html (also per category, /moban/lunwen/).
The download page /p/d.php?aid=<id> shows 比例/页数/格式/大小 and links the file on
down.ypppt.com (a ZIP or RAR with one PPTX, GBK member names; sometimes a bare PPTX), fetched
with the d.php page as Referer. robots.txt disallows only /h*u/ and /about/, so the copyright
page is unread; the terms are taken to be study-only like those of comparable sites.
"""

import heapq
import html
import io
import itertools
import re
from collections import deque
from urllib.parse import quote, unquote, unquote_plus, urljoin, urlsplit

from .fetch import FetchError

PERSONAL = re.compile(r"名单|名册|花名册|身份证")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KANA = re.compile(r"[\u3040-\u30ff]")
TYPE_HINT = {"pps": "ppt", "ppsx": "pptx", "dps": "ppt", "pot": "ppt", "potx": "pptx"}
ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.S | re.I)

WEBPLUS_PAGES = ["https://www.htu.edu.cn/szb/2020/0701/c9141a168693/page.htm",
                 "http://pub.ouc.edu.cn/2019/0729/c15638a254307/page.htm"]
WEBPLUS_LICENSE = "all rights reserved (course material published on the university's website)"
PLAYER = re.compile(r"<(?:div|iframe|embed|object)\b([^>]*\bpdfsrc\s*=[^>]*)>", re.I)
LISTING = re.compile(r"/list\d*\.(?:htm|html|psp)$")
ARTICLE = re.compile(r"/c\d+a\d+/page\.(?:htm|html|psp)$")
# Crawl order: links that name slides, then course material, then other listings, then articles.
HINTS = (re.compile(r"课件|讲义|教案|幻灯|演示文稿|PPT", re.I), re.compile(r"课程|教学|配套|资料|下载|精品课"))

VALSE = "https://valser.org/webinar/slide/"
VALSE_ROOT = VALSE + "index.php/Home/Index/index.html"
VALSE_DIR = re.compile(r"""href=["'](/webinar/slide/index\.php/Home/Index/index/dir/([^"'/]+)\.html)["']""")
VALSE_FILE = re.compile(r"""href=["']https?://valser\.org/webinar/slide/slides/([^"']+)["']""")
VALSE_LICENSE = "speaker's copyright; slides posted publicly by VALSE (no licence stated)"
# Names in scripts other than Latin and CJK (e.g. Cyrillic) are not talk slides.
VALSE_ODD = re.compile(r"[^\x00-\u024f\u2000-\u206f\u3000-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]")

ZENODO10K = "https://huggingface.co/datasets/Forceless/Zenodo10K/resolve/main/data/pptx-00000-of-00001.parquet"
ZENODO = "https://zenodo.org/records/"
# cc0-1.0 is how current Zenodo records name the licence Zenodo10K calls cc-zero.
OPEN_LICENSES = ("cc-by-4.0", "cc-by-sa-4.0", "cc-zero", "cc0-1.0")

YPPPT = "https://www.ypppt.com"
YPPPT_SECTIONS = {"kejian": "/kejian/yuwen/", "moban": "/moban/"}
YPPPT_LICENSE = "ypppt.com free download, study/research use only (assumed as on comparable template sites)"
YPPPT_FIELDS = {"分类": "category", "比例": "ratio", "页数": "pages", "格式": "format", "大小": "size", "效果": "effect"}


def _quiet(message):
    pass


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _page(fetcher, url, log, headers=None):
    """(final URL, HTML text), or None when the page cannot be fetched."""
    try:
        response, body = fetcher.get(url, headers=headers or {})
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = body.decode("gb18030", "replace")
    return getattr(response, "url", None) or url, text


def _list(value):
    if isinstance(value, (int, float)):
        return [value]
    return [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)


def _bare(url):
    return url.split("://", 1)[-1]


def _extension(name):
    name = name.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _stem(name):
    return re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name).strip()


def _attr(attrs, name):
    found = re.search(rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""", attrs, re.I)
    if not found:
        return None
    return html.unescape(found.group(1) if found.group(1) is not None else found.group(2)).strip()


def _years(value):
    """None (all years) or a set of years from 2020, [2019, 2020] or "2018-2026"."""
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        first, _, last = value.replace("..", "-").partition("-")
        return set(range(int(first), int(last or first) + 1))
    return {int(year) for year in value}


# --- WebPlus / Sudy university pages ---

def webplus(fetcher, options):
    """Slide attachments of university pages on WebPlus/Sudy CMS sites.

    Options: pages (URL or list of seed pages; default the htu.edu.cn and pub.ouc.edu.cn course
    pages), max_pages (int, HTML pages fetched per seed including the seed, default 1; more
    crawls same-host column listings list*.htm with their later pages and article pages
    c<col>a<id>/page.htm, links naming 课件/PPT/讲义 first, then 课程/教学/下载), prefix (path
    the crawl stays under, default the seed's sub-site such as /szb/, else /), extensions (list
    or comma string, default ppt,pptx,pdf), max_items (int), log (callable)."""
    yield from itertools.islice(_webplus(fetcher, options), options.get("max_items"))


def _subsite(path):
    """/szb/ for /szb/2020/0701/c9141a168693/page.htm or /szb/jxkj/list.htm, else /."""
    found = re.match(r"(/[^/]+)/(?:\d{4}/\d{4}/c\d+a\d+/page|[^/]+/list\d*)\.\w+$", path)
    return found.group(1) + "/" if found else "/"


def _webplus_files(page_url, text, extensions):
    """(url, title, link text) of each WebPlus attachment linked or embedded on the page."""
    for match in sorted(itertools.chain(ANCHOR.finditer(text), PLAYER.finditer(text)), key=lambda m: m.start()):
        attrs = match.group(1)
        href = _attr(attrs, "href") or _attr(attrs, "pdfsrc")
        if not href:
            continue
        url = urljoin(page_url, href).split("#")[0]
        sudy = _attr(attrs, "sudyfile-attr")
        path = urlsplit(url).path
        if not url.startswith(("http://", "https://")) or _extension(path) not in extensions:
            continue
        if not sudy and "/_upload/article/files/" not in path:
            continue
        named = re.search(r"""['"]title['"]\s*:\s*['"](.*?)['"]\s*[,}]""", sudy or "", re.S)
        label = _text(match.group(2)) if match.re is ANCHOR else ""
        title = re.sub(r"\s+", " ", named.group(1)).strip() if named else _attr(attrs, "title") or label
        yield url, _stem(title), label


def _same_scheme(url, base):
    """Same-host links keep the scheme the page was served over (some hosts answer on only one)."""
    parts, home = urlsplit(url), urlsplit(base)
    return home.scheme + url[len(parts.scheme):] if parts.netloc == home.netloc else url


def _webplus(fetcher, options):
    log = options.get("log") or _quiet
    extensions = {e.lower().lstrip(".") for e in _list(options.get("extensions") or ("ppt", "pptx", "pdf"))}
    seeds = options.get("pages") or WEBPLUS_PAGES
    budget = int(options.get("max_pages") or 1)
    visited, files = set(), set()
    for seed in [seeds] if isinstance(seeds, str) else seeds:
        host, prefix = urlsplit(seed).netloc, options.get("prefix") or _subsite(urlsplit(seed).path)
        order = itertools.count()
        queue, fetched = [(0, next(order), seed, "")], 0
        while queue and fetched < budget:
            priority, _, url, label = heapq.heappop(queue)
            if _bare(url) in visited:
                continue
            visited.add(_bare(url))
            got = _page(fetcher, url, log)
            fetched += 1
            if got is None:
                continue
            page_url, text = got
            title = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
            page_title = _text(title.group(1)) if title else label
            if PERSONAL.search(page_title):
                continue
            date = re.search(r"/(\d{4})/(\d{2})(\d{2})/c\d+a\d+/", urlsplit(page_url).path)
            for file_url, name, link in _webplus_files(page_url, text, extensions):
                file_url = _same_scheme(file_url, page_url)
                if _bare(file_url) in files or PERSONAL.search(name):
                    continue
                files.add(_bare(file_url))
                extension = _extension(urlsplit(file_url).path)
                yield dict(key=_bare(file_url), url=file_url, title=name or page_title,
                           type_hint=TYPE_HINT.get(extension, extension), license=WEBPLUS_LICENSE,
                           headers={"Referer": page_url},
                           meta=dict(page=page_url, page_title=page_title, link_text=link,
                                     published="-".join(date.groups()) if date else None,
                                     host=urlsplit(file_url).hostname))
            if fetched >= budget:
                continue
            links = []
            for match in ANCHOR.finditer(text):
                href = _attr(match.group(1), "href")
                if href and not href.lower().startswith(("javascript:", "mailto:")):
                    links.append((_same_scheme(urljoin(page_url, href).split("#")[0], page_url),
                                  _attr(match.group(1), "title") or _text(match.group(2)), None))
            pages = re.search(r"""class=["']all_pages["'][^>]*>\s*(\d+)""", text)
            if LISTING.search(urlsplit(page_url).path) and pages:  # later pages rank like the first
                links += [(re.sub(r"list\d*(\.\w+)$", rf"list{n}\1", page_url), label, priority)
                          for n in range(2, int(pages.group(1)) + 1)]
            for link_url, link_label, rank in links:
                parts = urlsplit(link_url)
                kind = 1 if LISTING.search(parts.path) else 2 if ARTICLE.search(parts.path) else None
                if (kind is None or parts.netloc != host or not parts.path.startswith(prefix)
                        or _bare(link_url) in visited or PERSONAL.search(link_label)):
                    continue
                if rank is None:
                    rank = next((i for i, hint in enumerate(HINTS) if hint.search(link_label)), len(HINTS) + kind - 1)
                heapq.heappush(queue, (rank, next(order), link_url, link_label))


# --- VALSE webinar slides ---

def valse(fetcher, options):
    """VALSE webinar slide files, newest directory first.

    Options: years (int, list or "2020-2026"; default all), undated (bool, default True: also
    the 短教程 and other non-date directories), extensions (default pdf), cjk_names (bool,
    default False; True keeps only files with Chinese in the name), max_dirs (int directory
    pages), max_items (int), log (callable)."""
    yield from itertools.islice(_valse(fetcher, options), options.get("max_items"))


def _valse(fetcher, options):
    log = options.get("log") or _quiet
    extensions = {e.lower().lstrip(".") for e in _list(options.get("extensions") or ["pdf"])}
    years, undated = _years(options.get("years")), options.get("undated", True)
    max_dirs = options.get("max_dirs")
    seen_dirs, seen_files = set(), set()

    def wanted(directory):
        date = re.match(r"(20\d\d)(\d\d)(\d\d)", directory)
        return undated if not date else years is None or int(date.group(1)) in years

    def files(page_url, text):
        for href in VALSE_FILE.findall(text):
            relative = re.sub(r"/{2,}", "/", unquote(html.unescape(href))).strip("/")
            name = relative.rsplit("/", 1)[-1]
            folder = relative.rsplit("/", 1)[0] if "/" in relative else ""
            if (not folder or relative in seen_files or _extension(name) not in extensions or not wanted(folder)
                    or VALSE_ODD.search(name) or (options.get("cjk_names") and not CJK.search(name))):
                continue
            seen_files.add(relative)
            date = re.match(r"(20\d\d)(\d\d)(\d\d)", folder)
            extension = _extension(name)
            yield dict(key="valse:" + relative, url=VALSE + "slides/" + quote(relative), title=_stem(name),
                       type_hint=TYPE_HINT.get(extension, extension), license=VALSE_LICENSE,
                       meta=dict(directory=folder, date="-".join(date.groups()) if date else None,
                                 filename=name, page=page_url, host="valser.org"))

    got = _page(fetcher, VALSE_ROOT, log)
    if got is None:
        return
    queue, fetched = deque(VALSE_DIR.findall(got[1])), 0
    while queue and (max_dirs is None or fetched < int(max_dirs)):
        path, encoded = queue.popleft()
        directory = unquote_plus(encoded)
        if directory in seen_dirs or not wanted(directory):
            continue
        seen_dirs.add(directory)
        page = _page(fetcher, urljoin(VALSE_ROOT, path), log)
        fetched += 1
        if page is not None:
            yield from files(page[0], page[1])
            queue.extend(VALSE_DIR.findall(page[1]))


# --- Zenodo ---

def zenodo_cjk(fetcher, options):
    """Zenodo presentations with Chinese in the title or file name and an open licence.

    Options: metadata (URL or local path of the Zenodo10K parquet; default the Hugging Face
    copy), licenses (list, default cc-by-4.0, cc-by-sa-4.0, cc-zero; rows under other licences
    are skipped), kana (bool, default False: titles or names with kana are Japanese and
    skipped), max_mb (float, default 200: larger files are skipped), records (Zenodo record ids
    whose files are listed via /records/<id>/export/json whatever their language, if the record
    licence is open), extensions (for those records, default pptx,ppt,odp,pdf), max_items (int),
    log (callable)."""
    yield from itertools.islice(_zenodo(fetcher, options), options.get("max_items"))


def _zenodo_rows(fetcher, source):
    import pyarrow.parquet as pq
    columns = ["filename", "size", "url", "license", "title", "created", "doi", "checksum"]
    if re.match(r"https?://", source):
        return pq.read_table(io.BytesIO(fetcher.get(source)[1]), columns=columns).to_pylist()
    return pq.read_table(source, columns=columns).to_pylist()


def _zenodo_candidate(record, filename, license, size, title, meta):
    extension = _extension(filename)
    return dict(key=f"zenodo:{record}/{filename}", url=f"{ZENODO}{record}/files/{quote(filename)}?download=1",
                title=title if CJK.search(title or "") else _stem(filename),
                type_hint=TYPE_HINT.get(extension, extension), license=license,
                meta=dict(meta, record=int(record), filename=filename, size=size, record_title=title,
                          host="zenodo.org"))


def _zenodo(fetcher, options):
    log = options.get("log") or _quiet
    licenses = {name.lower() for name in _list(options.get("licenses") or OPEN_LICENSES)}
    max_bytes = float(options.get("max_mb", 200)) * (1 << 20)
    seen = set()
    for row in _zenodo_rows(fetcher, options.get("metadata") or ZENODO10K):
        names = f"{row['title'] or ''} {row['filename'] or ''}"
        record = re.search(r"/records/(\d+)/", row["url"] or "") or re.search(r"zenodo\.(\d+)$", row["doi"] or "")
        if not CJK.search(names) or (KANA.search(names) and not options.get("kana")) or not record:
            continue
        if (row["license"] or "").lower() not in licenses or int(row["size"] or 0) > max_bytes:
            log(f"skip zenodo {record.group(1)} {row['filename']}: {row['license']}, {row['size']} bytes")
            continue
        candidate = _zenodo_candidate(record.group(1), row["filename"], row["license"], int(row["size"] or 0),
                                      row["title"], dict(doi=row["doi"], created=row["created"],
                                                         checksum=row["checksum"], listed_in="Zenodo10K"))
        if candidate["key"] not in seen:
            seen.add(candidate["key"])
            yield candidate
    extensions = {e.lower().lstrip(".") for e in _list(options.get("extensions") or ("pptx", "ppt", "odp", "pdf"))}
    for record in _list(options.get("records") or []):
        try:
            data = fetcher.json(f"{ZENODO}{record}/export/json")
        except (FetchError, ValueError) as error:
            log(f"skip zenodo record {record}: {getattr(error, 'reason', error)}")
            continue
        metadata = data.get("metadata") or {}
        rights = [right.get("id", "").lower() for right in metadata.get("rights") or []]
        license = next((right for right in rights if right in licenses), None)
        if license is None:
            log(f"skip zenodo record {record}: licence {rights}")
            continue
        meta = dict(doi=(data.get("pids") or {}).get("doi", {}).get("identifier"), created=data.get("created"),
                    languages=[language.get("id") for language in metadata.get("languages") or []])
        for name, entry in ((data.get("files") or {}).get("entries") or {}).items():
            if _extension(name) in extensions and int(entry.get("size") or 0) <= max_bytes:
                candidate = _zenodo_candidate(record, name, license, int(entry.get("size") or 0), metadata.get("title"),
                                              dict(meta, checksum=entry.get("checksum")))
                if candidate["key"] not in seen:
                    seen.add(candidate["key"])
                    yield candidate


# --- 优品PPT ---

def ypppt(fetcher, options):
    """优品PPT decks, one candidate per article id with its down.ypppt.com file.

    Options: sections (list or comma string of kejian (语文 courseware, default), moban
    (templates) or listing paths such as /moban/lunwen/), grades (substrings of the courseware
    grade headings, e.g. 七年级; default all), lessons (lesson slugs, e.g. beiying; default all),
    max_lessons (int lesson pages per run), max_list_pages (int listing pages per template
    section, default 5), max_items (int), log (callable)."""
    yield from itertools.islice(_ypppt(fetcher, options), options.get("max_items"))


def _posts(text):
    """(article id, article path, title) of the ul.posts entries of a lesson or listing page."""
    block = re.search(r"""<ul\s+class=["']posts[^"']*["']>(.*?)</ul>""", text, re.S | re.I)
    for match in ANCHOR.finditer(block.group(1) if block else ""):
        href = _attr(match.group(1), "href") or ""
        found = re.match(r"(?:https?://www\.ypppt\.com)?(/article/\d{4}/(\d+)\.html)$", href)
        if found and "p-title" in match.group(1):
            yield int(found.group(2)), found.group(1), _text(match.group(2))


def _ypppt_courseware(fetcher, options, log):
    got = _page(fetcher, YPPPT + YPPPT_SECTIONS["kejian"], log)
    if got is None:
        return
    menus = "".join(re.findall(r"""<div\s+class=["']b-menu[^"']*["']>(.*?)</div>""", got[1], re.S | re.I))
    grade, lessons = None, []
    for heading, attrs, label in re.findall(r"<h3>(.*?)</h3>|<a\b([^>]*)>(.*?)</a>", menus, re.S):
        if heading:
            grade = _text(heading)
        elif _attr(attrs, "href"):
            lessons.append((grade, _attr(attrs, "href"), _text(label)))
    grades, slugs = _list(options.get("grades") or []), set(_list(options.get("lessons") or []))
    lessons = [(g, href, label) for g, href, label in lessons
               if (not grades or any(want in (g or "") for want in grades))
               and (not slugs or href.strip("/").rsplit("/", 1)[-1] in slugs)]
    limit = int(options["max_lessons"]) if options.get("max_lessons") else None
    for grade, href, label in lessons[:limit]:
        context = dict(section="kejian", grade=grade, lesson=label)
        single = re.match(r"/article/\d{4}/(\d+)\.html$", href)
        if single:  # a lesson with one deck links its article directly
            yield int(single.group(1)), label, dict(context, article=YPPPT + href)
            continue
        page = _page(fetcher, urljoin(YPPPT, href), log)
        for aid, path, title in _posts(page[1]) if page else ():
            yield aid, title, dict(context, lesson_page=page[0], article=YPPPT + path)


def _ypppt_listing(fetcher, path, options, log):
    seen = set()
    for number in range(1, int(options.get("max_list_pages") or 5) + 1):
        url = YPPPT + path + ("" if number == 1 else f"list-{number}.html")
        got = _page(fetcher, url, log)
        posts = [post for post in _posts(got[1]) if post[0] not in seen] if got else []
        if not posts:
            return
        for aid, article, title in posts:
            seen.add(aid)
            yield aid, title, dict(section=path, listing=url, article=YPPPT + article)


def _ypppt_download(fetcher, aid, title, context, log):
    page_url = f"{YPPPT}/p/d.php?aid={aid}"
    got = _page(fetcher, page_url, log, headers={"Referer": context["article"]})
    link = got and re.search(r"""href=["'](https?://down\.ypppt\.com/[^"']+)["']""", got[1])
    if not link:
        log(f"no download link for ypppt {aid}")
        return None
    url = html.unescape(link.group(1))
    heading = re.search(r"<h1>(.*?)</h1>", got[1], re.S)
    fields = {YPPPT_FIELDS[k]: _text(v) for k, v in re.findall(r"<i>\s*(\S+?)：\s*</i>(.*?)</li>", got[1], re.S)
              if k in YPPPT_FIELDS}
    if re.match(r"\d+", fields.get("pages", "")):
        fields["pages"] = int(re.match(r"\d+", fields["pages"]).group())
    extension = _extension(urlsplit(url).path)
    name = re.sub(r"\s*-\s*下载页$", "", _text(heading.group(1))) if heading else title
    return dict(key=f"ypppt:{aid}", url=url, title=name, type_hint=TYPE_HINT.get(extension, extension),
                license=YPPPT_LICENSE, headers={"Referer": page_url},
                meta=dict(context, aid=aid, listed_title=title, download_page=page_url, host=urlsplit(url).hostname,
                          **fields))


def _ypppt(fetcher, options):
    log = options.get("log") or _quiet
    seen = set()
    for section in _list(options.get("sections") or ["kejian"]):
        path = YPPPT_SECTIONS.get(section, section)
        if not path.startswith("/"):
            raise ValueError(f"ypppt sections are {sorted(YPPPT_SECTIONS)} or listing paths like /moban/lunwen/")
        if path == YPPPT_SECTIONS["kejian"]:
            articles = _ypppt_courseware(fetcher, options, log)
        else:
            articles = _ypppt_listing(fetcher, path if path.endswith("/") else path + "/", options, log)
        for aid, title, context in articles:
            if aid in seen or PERSONAL.search(title):
                continue
            seen.add(aid)
            candidate = _ypppt_download(fetcher, aid, title, context, log)
            if candidate:
                yield candidate
