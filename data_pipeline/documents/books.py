"""Openly licensed Chinese books and manuals: CTAN package documentation, and a curated list of
books and manual sets whose licence statement is read from the source at discovery time.

ctan_chinese: https://ctan.org/topic/chinese lists 81 packages and /topic/chinese-doc 12 more
(lshort-zh-cn, ctex-faq, the Asymptote manuals, ...) as <div class="dt"><a href="/pkg/<name>">.
A package page /pkg/<name> has a table.entry that links its licences as /license/<id> and its
documentation on https://mirrors.ctan.org/<path>, a link optionally tagged
<img class="doc-lang" src="/assets/lang/zh.svg" alt="[Chinese]">. ctan.org's robots.txt
disallows /json/, /xml/ and /search, so there is no API. mirrors.ctan.org redirects to a random
mirror (one failed TLS) and mirrors.tuna.tsinghua.edu.cn disallows /CTAN, so documentation
links are rewritten to one mirror, by default https://mirrors.mit.edu/CTAN/ (robots.txt only
disallows /bulk.resource.org/). Licence ids are those that CTAN's /license page lists under
"Free Licenses" (LPPL, GPL, LGPL, FDL, MIT, BSD, Apache, CC BY/BY-SA/CC0, OFL, public domain,
other-free, ...); a package with any other id (nocommercial, nosell, cc-by-nc*, cc-by-nd*,
other-nonfree, noinfo, ...) is skipped. Measured 2026-10: 93 packages, all free; 105 PDFs taken
(85 tagged Chinese or named -cn/-zh, 20 untagged fallbacks), about 1 MB each.

open_books (licence re-read from the named page each run; an item whose statement is gone is
skipped):
  d2l-zh       https://zh.d2l.ai/d2l-zh.pdf (MXNet edition, 32.7 MB, 813 pages, xdvipdfmx, fonts
               embedded) and d2l-zh-pytorch.pdf (30.9 MB); zh-v2.d2l.ai serves the same files and
               robots.txt answers the HTML homepage. Apache-2.0 per the repository LICENSE on
               raw.githubusercontent.com (robots.txt 404).
  debian-*     www.debian.org (robots.txt 404): 参考手册 zh-cn/zh-tw (GPL, any DFSG-compatible
               version), 安装手册 amd64 zh-cn (GPL; no zh-tw build), 新维护者手册 zh-cn/zh-tw
               (GPL-2+), debmake-doc zh-cn (MIT); the licence is in each index.zh-cn.html.
  freebsd      autoindexes https://download.freebsd.org/doc/zh-cn/ and /zh-tw/ (robots.txt 404;
               docs.freebsd.org disallows /doc/): 23 PDFs, handbook 29 MB (zh-cn) and 23.5 MB
               (zh-tw), FAQ, porters/developers/arch handbooks, articles. FreeBSD Documentation
               License (BSD-2-Clause style) at www.freebsd.org/copyright/freebsd-doc-license/.
  python-docs  docs.python.org/zh-cn|zh-tw/<version>/download.html: prebuilt A4 PDF archives (a
               ZIP of the manuals, 30-34 MB) are no longer updated but still served for 3.12-3.14;
               3.15 has none. robots.txt disallows versions up to 3.10. The page footer states
               the PSF License v2 (code also 0BSD).
  suse         https://documentation.suse.com/zh-cn/<product>/<version>/ lists div.articleRow
               blocks (data-article-link to the HTML, a *_zh_cn.pdf); each HTML legal notice grants
               GFDL 1.2 or later. 81 PDFs over sles 16.0/15-SP7, sle-micro 6.1, sle-ha 15-SP7,
               sled 15-SP7 and suse-edge 3.3; most other products have no zh-cn PDFs.
tug.org disallows AI crawlers, so TeX Live's own guide translations are not fetched there.
"""

import html
import itertools
import re
from urllib.parse import quote, urljoin, urlsplit

from .fetch import FetchError

CTAN = "https://ctan.org"
CTAN_TOPICS = ("chinese-doc", "chinese")
CTAN_MIRROR = "https://mirrors.mit.edu/CTAN/"
REDIRECTOR = re.compile(r"^https?://mirrors?\.ctan\.org/")
# CTAN /license, section "Free Licenses" (2026-10).
FREE_LICENSES = frozenset(
    "0bsd agpl3 apache2 artistic2 bsd bsd2 bsd3 bsd4 cc-by-1 cc-by-2 cc-by-3 cc-by-4 cc-by-sa-1 cc-by-sa-2 "
    "cc-by-sa-3 cc-by-sa-4 cc0 eupl fdl gfl gfsl gpl gpl1 gpl1+ gpl2 gpl2+ gpl3 gpl3+ gpl3+fe isc knuth lgpl "
    "lgpl2.1 lgpl3 lppl lppl1 lppl1.1 lppl1.2 lppl1.3 lppl1.3a lppl1.3b lppl1.3c mit ofl opl other-free pd x11".split())
CHINESE_NAME = re.compile(r"(?:^|[-_.])(?:zh|cn|sc|tc|chs|cht|zhcn|zh[-_](?:cn|tw|hk|hans|hant)|chinese)(?=[-_.]|$)",
                          re.I)
TOPIC_PACKAGE = re.compile(r"""<div class=["']dt["']>\s*<a href=["']/pkg/([^"'/?#]+)["']""")
ANCHOR = re.compile(r"<a\b([^>]*)>(.*?)</a\s*>", re.S | re.I)
DOC_LANG = re.compile(r"/assets/lang/([A-Za-z-]+)\.svg")

RAW = "https://raw.githubusercontent.com/"
DEBIAN = "https://www.debian.org/"
FREEBSD = "https://download.freebsd.org/doc/"
PYTHON = "https://docs.python.org/"
SUSE = "https://documentation.suse.com/zh-cn/"
PYTHON_VERSIONS = ("3.13",)
SUSE_PRODUCTS = ("sles/16.0", "sles/15-SP7", "sle-micro/6.1", "sle-ha/15-SP7", "sled/15-SP7", "suse-edge/3.3")

# Each book: title, licenses ((name, regex on the page text), ...) and where to find the files:
# files ((url, title), ...) with license_url; index (autoindex roots, PDFs below) with license_url;
# pages + link (listing pages whose own footer states the licence); or pages + articles (SUSE
# product pages, licence read from each article's HTML).
BOOKS = {
    "d2l-zh": dict(title="动手学深度学习", licenses=(("Apache-2.0", r"Apache License\s+Version 2\.0"),),
                   license_url=RAW + "d2l-ai/d2l-zh/master/LICENSE",
                   files=(("https://zh.d2l.ai/d2l-zh.pdf", "动手学深度学习 (MXNet)"),
                          ("https://zh.d2l.ai/d2l-zh-pytorch.pdf", "动手学深度学习 (PyTorch)"))),
    "debian-reference": dict(title="Debian 参考手册",
                             licenses=(("GPL (any DFSG-compatible version)", r"GNU 通用公共许可证的条款下重新分发和修改"),),
                             license_url=DEBIAN + "doc/manuals/debian-reference/index.zh-cn.html",
                             files=tuple(DEBIAN + f"doc/manuals/debian-reference/debian-reference.{lang}.pdf"
                                         for lang in ("zh-cn", "zh-tw"))),
    "debian-installer": dict(title="Debian GNU/Linux 安装手册 (amd64)",
                             licenses=(("GPL", r"在 GNU 通用公共授权（GPL）下重新发布和（或）修改"),),
                             license_url=DEBIAN + "releases/stable/amd64/index.zh-cn.html",
                             files=(DEBIAN + "releases/stable/amd64/install.zh-cn.pdf",)),
    "debian-maint-guide": dict(title="Debian 新维护者手册",
                               licenses=(("GPL-2.0-or-later", r"GNU 通用公共许可证第二版或更高版本"),),
                               license_url=DEBIAN + "doc/manuals/maint-guide/index.zh-cn.html",
                               files=tuple(DEBIAN + f"doc/manuals/maint-guide/maint-guide.{lang}.pdf"
                                           for lang in ("zh-cn", "zh-tw"))),
    "debmake-doc": dict(title="Debian 维护者指南", licenses=(("MIT", r"Permission is hereby granted, free of charge"),),
                        license_url=DEBIAN + "doc/manuals/debmake-doc/index.zh-cn.html",
                        files=(DEBIAN + "doc/manuals/debmake-doc/debmake-doc.zh-cn.pdf",)),
    "freebsd": dict(title="FreeBSD",
                    licenses=(("FreeBSD Documentation License (BSD-2-Clause style)",
                               r"Redistribution and use in source \(AsciiDoc\) and 'compiled' forms"),),
                    license_url="https://www.freebsd.org/copyright/freebsd-doc-license/",
                    index=(FREEBSD + "zh-cn/", FREEBSD + "zh-tw/")),
    "python-docs": dict(title="Python 文档",
                        licenses=(("PSF-2.0", r"Python 软件基金会许可证第 2 版|Python 軟體基金會授權條款第 2 版"),),
                        pages=tuple(f"{PYTHON}{lang}/{version}/download.html"
                                    for version in PYTHON_VERSIONS for lang in ("zh-cn", "zh-tw")),
                        link=r"""href=["'](archives/python-[\d.]+-docs-pdf-a4\.zip)["']"""),
    "suse": dict(title="SUSE 文档",
                 licenses=(("GFDL-1.2-or-later", r"根据 ?GNU ?自由文档许可证.{0,80}?授予您复制、分发和/或修改本文档的权限"),),
                 pages=tuple(SUSE + product + "/" for product in SUSE_PRODUCTS), articles=True),
}


def _quiet(message):
    pass


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _list(value):
    return [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)


def _attr(attrs, name):
    found = re.search(rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""", attrs, re.I)
    if not found:
        return None
    return html.unescape(found.group(1) if found.group(1) is not None else found.group(2)).strip()


def _bare(url):
    return url.split("://", 1)[-1]


def _extension(name):
    name = name.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _stem(url):
    return re.sub(r"\.[A-Za-z0-9]{2,5}$", "", urlsplit(url).path.rsplit("/", 1)[-1])


def _page(fetcher, url, log):
    """(final URL, text), or None when the page cannot be fetched."""
    try:
        response, body = fetcher.get(url)
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None
    return getattr(response, "url", None) or url, body.decode("utf-8", "replace")


# --- CTAN ---

def ctan_chinese(fetcher, options):
    """PDF documentation of CTAN packages under Chinese topics, with each package's licence.

    Options: topics (default chinese-doc, chinese), packages (extra package names), mirror (CTAN
    root that replaces the mirrors.ctan.org redirector, default mirrors.mit.edu/CTAN), licenses
    (allowed CTAN licence ids, default FREE_LICENSES), all_docs (bool, default False: only PDFs
    tagged Chinese or named -cn/-zh/-sc/..., or the untagged ones when a package has none),
    max_items (int), log (callable)."""
    yield from itertools.islice(_ctan(fetcher, options), options.get("max_items"))


def ctan_package(text):
    """Title, version, licence ids, topics and PDF documentation links of a /pkg/<name> page."""
    table = re.search(r"<table class=[\"']entry[\"']>(.*?)</table>", text, re.S)
    table = table.group(1) if table else ""
    version = re.search(r"<td>Version</td>\s*<td>(.*?)</td>", table, re.S)
    title = re.search(r"<h1\b[^>]*>(.*?)</h1>", text, re.S)
    docs = []
    for attrs, label in ANCHOR.findall(table):
        href = _attr(attrs, "href")
        if href and (_attr(attrs, "class") or "").startswith("doc-") and _extension(urlsplit(href).path) == "pdf":
            lang = DOC_LANG.search(label)
            docs.append(dict(href=href, label=_text(label), lang=lang.group(1).lower() if lang else None))
    return dict(title=_text(title.group(1)) if title else None, version=_text(version.group(1)) if version else None,
                licenses=list(dict.fromkeys(re.findall(r"""href=["']/license/([^"'/?#]+)["']""", table))),
                topics=re.findall(r"""href=["']/topic/([^"'/?#]+)["']""", table), docs=docs)


def _chinese(doc):
    return (doc["lang"] or "").startswith("zh") or bool(CHINESE_NAME.search(_stem(doc["href"])))


def _ctan(fetcher, options):
    log = options.get("log") or _quiet
    mirror = (options.get("mirror") or CTAN_MIRROR).rstrip("/") + "/"
    allowed = {name.lower() for name in _list(options.get("licenses") or FREE_LICENSES)}
    names = {}
    for topic in _list(options.get("topics") or CTAN_TOPICS):
        page = _page(fetcher, f"{CTAN}/topic/{quote(topic)}", log)
        for name in TOPIC_PACKAGE.findall(page[1]) if page else ():
            names.setdefault(name, topic)
    for name in _list(options.get("packages") or []):
        names.setdefault(name, None)
    seen = set()
    for name, topic in names.items():
        source = f"{CTAN}/pkg/{quote(name)}"
        page = _page(fetcher, source, log)
        if not page:
            continue
        package = ctan_package(page[1])
        refused = [license for license in package["licenses"] if license.lower() not in allowed]
        if not package["licenses"] or refused:
            log(f"skip ctan {name}: licences {package['licenses'] or 'none listed'}")
            continue
        docs = package["docs"]
        if not options.get("all_docs"):
            docs = [doc for doc in docs if _chinese(doc)] or [doc for doc in docs if doc["lang"] is None]
        for doc in docs:
            path = REDIRECTOR.sub("", doc["href"]) if REDIRECTOR.match(doc["href"]) else None
            key = "ctan:" + (path or _bare(doc["href"]))
            if key in seen:
                continue
            seen.add(key)
            url = mirror + path if path else doc["href"]
            yield dict(key=key, url=url, title=f"{name}: {doc['label']}" if doc["label"] else name, type_hint="pdf",
                       license=", ".join(package["licenses"]),
                       meta=dict(package=name, topic=topic, version=package["version"], licenses=package["licenses"],
                                 license_source=source, lang=doc["lang"], chinese_doc=_chinese(doc),
                                 doc_label=doc["label"], ctan_path=path, host=urlsplit(url).hostname))


# --- curated books ---

def open_books(fetcher, options):
    """Curated openly licensed Chinese books and manuals (BOOKS), each licence statement re-read.

    Options: books (ids, default all of BOOKS), verify (bool, default True: fetch each licence
    page and drop items whose statement is missing), python_versions (default 3.13),
    suse_products (default SUSE_PRODUCTS, e.g. "sles/15-SP7"), max_items (int), log (callable)."""
    yield from itertools.islice(_open_books(fetcher, options), options.get("max_items"))


def _lang(url):
    found = re.search(r"zh[-_](cn|tw)", url, re.I)
    return f"zh-{found.group(1).lower()}" if found else None


def _autoindex(fetcher, root, log, depth=4):
    """PDF links below an autoindex directory (sub-directories followed up to ``depth``)."""
    pending = [(root, 0)]
    while pending:
        url, level = pending.pop(0)
        page = _page(fetcher, url, log)
        if not page:
            continue
        for href in re.findall(r"""href=["']([^"'?#]+)["']""", page[1]):
            if href.startswith(("/", "..")) or "://" in href:
                continue
            target = urljoin(page[0], href)
            if href.endswith("/") and level < depth:
                pending.append((target, level + 1))
            elif _extension(href) == "pdf":
                yield target


def _suse_articles(text, base):
    """(PDF URL, title, article HTML URL) per div.articleRow block with a Chinese PDF."""
    for block in re.split(r"""<div class=["']articleRow["']""", text)[1:]:
        pdf = re.search(r"""href=["']([^"']+_zh_cn\.pdf)["']""", block)
        article = re.search(r"""data-article-link=["']([^"']+)["']""", block)
        title = re.search(r"""class=["']articleTitle["'][^>]*>(.*?)</a>""", block, re.S)
        if pdf and article:
            yield urljoin(base, pdf.group(1)), _text(title.group(1)) if title else None, urljoin(base, article.group(1))


def _open_books(fetcher, options):
    log = options.get("log") or _quiet
    verify = options.get("verify", True)
    wanted = _list(options.get("books") or BOOKS)
    texts = {}

    def text(url):
        if url not in texts:
            page = _page(fetcher, url, log)
            texts[url] = page[1] if page else None
        return texts[url]

    def license_of(book, url):
        if not verify:
            return book["licenses"][0][0]
        plain = _text(text(url) or "")
        return next((name for name, marker in book["licenses"] if re.search(marker, plain)), None)

    seen = set()
    for book_id in wanted:
        book = BOOKS.get(book_id)
        if book is None:
            log(f"unknown book {book_id}")
            continue
        items = []  # (url, title, licence page)
        if book_id == "python-docs" and options.get("python_versions"):
            book = dict(book, pages=tuple(f"{PYTHON}{lang}/{version}/download.html"
                                          for version in _list(options["python_versions"])
                                          for lang in ("zh-cn", "zh-tw")))
        if book_id == "suse" and options.get("suse_products"):
            book = dict(book, pages=tuple(SUSE + p.strip("/") + "/" for p in _list(options["suse_products"])))
        for entry in book.get("files", ()):
            url, title = entry if isinstance(entry, tuple) else (entry, None)
            items.append((url, title or f"{book['title']} ({_lang(url) or 'zh'})", book["license_url"]))
        for root in book.get("index", ()):
            items += [(url, f"{book['title']} {_stem(url)}", book["license_url"]) for url in _autoindex(fetcher, root, log)]
        for page_url in book.get("pages", ()):
            page = text(page_url)
            if page is None:
                continue
            if book.get("articles"):
                items += [(url, f"{book['title']}: {title}" if title else _stem(url), article)
                          for url, title, article in _suse_articles(page, page_url)]
            else:
                items += [(urljoin(page_url, link), f"{book['title']} {_stem(link)} ({_lang(page_url) or 'zh'})",
                           page_url) for link in re.findall(book["link"], page)]
        for url, title, license_url in items:
            if url in seen:
                continue
            seen.add(url)
            license = license_of(book, license_url)
            if license is None:
                log(f"skip {url}: no licence statement on {license_url}")
                continue
            yield dict(key=f"book:{_bare(url)}", url=url, title=title, type_hint=_extension(urlsplit(url).path),
                       license=license, meta=dict(book=book_id, license_source=license_url, verified=bool(verify),
                                                  lang=_lang(url), host=urlsplit(url).hostname))
