"""Open book sources (offline): CTAN Chinese package documentation and the curated open books."""

from types import SimpleNamespace

from data_pipeline.documents import books
from data_pipeline.documents.fetch import FetchError


class FakeFetcher:
    """Canned pages by URL; a missing URL answers 404."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, headers=None, **kwargs):
        self.calls.append(url)
        page = self.pages.get(url)
        if page is None:
            raise FetchError("http_404", url)
        return SimpleNamespace(url=url, status_code=200, headers={}), page.encode()


# --- CTAN ---

CTAN, MIRRORS = "https://ctan.org", "https://mirrors.ctan.org/"
ZH = ' <img class="doc-lang" src="/assets/lang/zh.svg" alt="[Chinese]" title="[Chinese]"/>'


def topic(*names):
    rows = "".join(f'<div class="dt">\n  <a href="/pkg/{name}">{name}</a>\n</div>\n<div class="dd">x</div>'
                   for name in names)
    return f'<main class="content"><h1>Chinese</h1><div class="pkg-cols">{rows}</div></main>'


def package(licenses, docs, version="1.0 2026-01-01"):
    """A /pkg/<name> page: docs are (href, label, lang tag or None); README links are not PDFs."""
    links = '<div class="d-icon-readme"><a href="https://mirrors.ctan.org/x/README.md" class="doc-readme">README.md</a></div>'
    for href, label, lang in docs:
        tag = ZH if lang == "zh" else ""
        links += f'<div class="d-icon-pdf" title="PDF"><a href="{href}" class="doc-filetype.pdf">{label}{tag}</a></div>'
    licence_links = "".join(f'<a href="/license/{licence}">Licence {licence}</a><br />' for licence in licenses)
    return f"""<main class="content"><h1 lang="none">Package &ndash; <span class="t-logo">T<span class="e">e</span>X</span></h1>
    <table class="entry">
      <tr><td>Sources</td><td><a href="/tex-archive/x"><code>/x</code></a></td></tr>
      <tr><td>Documentation</td><td>{links}</td></tr>
      <tr><td>Version</td><td>{version} </td></tr>
      <tr><td>Licenses</td><td>{licence_links}</td></tr>
      <tr><td>Maintainer</td><td><a href="/author/someone">A. Maintainer</a><br /></td></tr>
      <tr><td>Topics</td><td><a href="/topic/class">Class</a><br /><a href="/topic/chinese">Chinese</a><br /></td></tr>
    </table>
    <p class="download"><a href="https://mirrors.ctan.org/x.zip">Download</a></p>
    <div class="right"><a href="/pkg/other">Suggestion</a></div></main>"""


CTAN_PAGES = {
    f"{CTAN}/topic/chinese-doc": topic("lshort-zh-cn", "shared"),
    f"{CTAN}/topic/chinese": topic("elegantbook", "shared", "luatex-cn", "context-notes-zh-cn", "nonfree", "mixed",
                                   "gone", "readme", "external", "unlicensed"),
    f"{CTAN}/pkg/lshort-zh-cn": package(["fdl"], [(MIRRORS + "info/lshort/chinese/lshort-zh-cn.pdf", "The document itself", "zh")]),
    f"{CTAN}/pkg/shared": package(["mit"], [(MIRRORS + "macros/shared/shared-doc.pdf", "Package documentation", None)]),
    f"{CTAN}/pkg/elegantbook": package(["lppl1.3c"], [
        (MIRRORS + "macros/latex/contrib/elegantbook/elegantbook-cn.pdf", "Package documentation (Chinese)", "zh"),
        (MIRRORS + "macros/latex/contrib/elegantbook/elegantbook-en.pdf", "Package documentation (English)", None)]),
    f"{CTAN}/pkg/luatex-cn": package(["apache2"], [
        (MIRRORS + "language/chinese/luatex-cn/doc/luatex-cn-wiki-en.pdf", "Package documentation (English)", None),
        (MIRRORS + "language/chinese/luatex-cn/doc/luatex-cn-wiki-zh.pdf", "Package documentation (Chinese)", "zh")]),
    f"{CTAN}/pkg/context-notes-zh-cn": package(["gpl3", "cc-by-sa-4"], [
        (MIRRORS + "info/context-notes-zh-cn/context-notes-zh-cn.pdf", "The document itself", None),
        (MIRRORS + "info/context-notes-zh-cn/extra.pdf", "Examples", None)]),
    f"{CTAN}/pkg/nonfree": package(["nocommercial"], [(MIRRORS + "a/nonfree-zh.pdf", "Doc", "zh")]),
    f"{CTAN}/pkg/mixed": package(["lppl1.3c", "other-nonfree"], [(MIRRORS + "a/mixed-cn.pdf", "Doc", "zh")]),
    f"{CTAN}/pkg/readme": package(["lppl1.3c"], []),
    f"{CTAN}/pkg/external": package(["cc-by-4"], [("https://example.org/files/manual-zh.pdf", "Manual", None)]),
    f"{CTAN}/pkg/unlicensed": package([], [(MIRRORS + "a/unlicensed-zh.pdf", "Doc", "zh")]),
    f"{CTAN}/pkg/extra": package(["lppl1.3c"], [(MIRRORS + "macros/extra/extra-sc.pdf", "Doc", None)]),
}


def test_ctan_package_page_parse():
    parsed = books.ctan_package(CTAN_PAGES[f"{CTAN}/pkg/elegantbook"])
    assert parsed["title"] == "Package – TeX" and parsed["version"] == "1.0 2026-01-01"
    assert parsed["licenses"] == ["lppl1.3c"] and parsed["topics"] == ["class", "chinese"]
    assert [(d["href"].rsplit("/", 1)[1], d["label"], d["lang"]) for d in parsed["docs"]] == [
        ("elegantbook-cn.pdf", "Package documentation (Chinese)", "zh"),
        ("elegantbook-en.pdf", "Package documentation (English)", None)]
    assert books.ctan_package("<html>gone</html>") == dict(title=None, version=None, licenses=[], topics=[], docs=[])


def test_ctan_topics_licences_language_choice_and_mirror():
    logs = []
    fake = FakeFetcher(CTAN_PAGES)
    found = list(books.ctan_chinese(fake, {"log": logs.append}))
    assert [c["key"] for c in found] == [
        "ctan:info/lshort/chinese/lshort-zh-cn.pdf", "ctan:macros/shared/shared-doc.pdf",
        "ctan:macros/latex/contrib/elegantbook/elegantbook-cn.pdf",
        "ctan:language/chinese/luatex-cn/doc/luatex-cn-wiki-zh.pdf",
        "ctan:info/context-notes-zh-cn/context-notes-zh-cn.pdf", "ctan:example.org/files/manual-zh.pdf"]
    lshort, shared, elegant, luatex, notes, external = found
    assert lshort["url"] == "https://mirrors.mit.edu/CTAN/info/lshort/chinese/lshort-zh-cn.pdf"
    assert lshort["license"] == "fdl" and lshort["type_hint"] == "pdf"
    assert lshort["title"] == "lshort-zh-cn: The document itself"
    assert lshort["meta"] == dict(package="lshort-zh-cn", topic="chinese-doc", version="1.0 2026-01-01", licenses=["fdl"],
                                  license_source=f"{CTAN}/pkg/lshort-zh-cn", lang="zh", chinese_doc=True,
                                  doc_label="The document itself", ctan_path="info/lshort/chinese/lshort-zh-cn.pdf",
                                  host="mirrors.mit.edu")
    assert shared["meta"]["topic"] == "chinese-doc" and shared["meta"]["chinese_doc"] is False  # untagged fallback
    assert elegant["license"] == "lppl1.3c" and luatex["license"] == "apache2"
    assert notes["license"] == "gpl3, cc-by-sa-4" and notes["meta"]["chinese_doc"] and notes["meta"]["lang"] is None
    assert external["url"] == "https://example.org/files/manual-zh.pdf" and external["meta"]["ctan_path"] is None
    assert any("nonfree" in line and "nocommercial" in line for line in logs)
    assert any("mixed" in line and "other-nonfree" in line for line in logs)
    assert any("unlicensed" in line and "none listed" in line for line in logs)
    assert any("gone" in line and "http_404" in line for line in logs)
    assert fake.calls.count(f"{CTAN}/pkg/shared") == 1
    assert all(url.startswith((f"{CTAN}/topic/", f"{CTAN}/pkg/")) for url in fake.calls)  # no mirror, /search or /json


def test_ctan_options_all_docs_mirror_packages_licences_and_cap():
    every = list(books.ctan_chinese(FakeFetcher(CTAN_PAGES), {"topics": "chinese", "all_docs": True,
                                                              "mirror": "https://mirror.example/ctan"}))
    keys = [c["key"].rsplit("/", 1)[1] for c in every]
    assert keys[:4] == ["elegantbook-cn.pdf", "elegantbook-en.pdf", "shared-doc.pdf", "luatex-cn-wiki-en.pdf"]
    assert "extra.pdf" in keys and every[0]["meta"]["topic"] == "chinese"
    assert every[0]["url"] == "https://mirror.example/ctan/macros/latex/contrib/elegantbook/elegantbook-cn.pdf"
    fake = FakeFetcher(CTAN_PAGES)
    only = list(books.ctan_chinese(fake, {"topics": [], "packages": "extra, lshort-zh-cn", "licenses": ["lppl1.3c"]}))
    assert [c["key"] for c in only] == ["ctan:macros/extra/extra-sc.pdf"]  # name hint -sc; fdl not allowed here
    assert only[0]["meta"]["topic"] is None and only[0]["meta"]["chinese_doc"]
    assert not any("/topic/" in url for url in fake.calls)
    capped = FakeFetcher(CTAN_PAGES)
    assert len(list(books.ctan_chinese(capped, {"max_items": 2}))) == 2
    assert capped.calls[-1] == f"{CTAN}/pkg/shared"


def test_free_licence_list_excludes_non_free_ids():
    assert {"lppl1.3c", "gpl3", "fdl", "mit", "cc-by-sa-4", "cc0", "pd", "ofl"} <= books.FREE_LICENSES
    assert not {"nocommercial", "nosell", "cc-by-nc-4", "cc-by-nd-4", "cc-by-nc-sa4", "other-nonfree", "noinfo",
                "shareware", "artistic", "nosource"} & books.FREE_LICENSES


# --- curated books ---

B = books.BOOKS
SUSE16, SUSE15 = books.SUSE + "sles/16.0/", books.SUSE + "sles/15-SP7/"
LEGAL = ("<p>版权所有 © 2006–2026 SUSE LLC 和贡献者。保留所有权利。</p><p>根据 GNU 自由文档许可证 (GNU Free Documentation "
         "License) 版本 1.2 或（根据您的选择）版本 1.3 中的条款，在此授予您复制、分发和/或修改本文档的权限；</p>")


def article_row(html, pdf, title):
    return (f'<div class="articleRow" data-topics="[]" data-article-link="{html}" data-lang="Chinese">'
            f'<div class="articleText"><a class="articleTitle" href="{html}">\n  {title}\n</a></div>'
            f'<div class="format-dropdown-menu"><a href="{pdf}" class="dropdown-item">PDF</a></div></div>\n')


def autoindex(*names):
    rows = "".join(f'<tr><td class="link"><a href="{name}" title="{name}">{name}</a></td></tr>' for name in names)
    return f'<table id="list"><tr><td><a href="?C=N&amp;O=A">File Name</a></td></tr><tr><td><a href="../">Parent</a></td></tr>{rows}</table>'


PY_CN, PY_TW = books.PYTHON + "zh-cn/3.13/download.html", books.PYTHON + "zh-tw/3.13/download.html"
BOOK_PAGES = {
    B["d2l-zh"]["license_url"]: "                                 Apache License\n                           Version 2.0, January 2004\n",
    B["debian-reference"]["license_url"]: ("<p>这本书是自由的；你可以在与 Debian 自由软件指导方针（DFSG）兼容的任意版本的 "
                                           "GNU 通用公共许可证的条款下重新分发和修改本书。</p>"),
    B["debian-installer"]["license_url"]: "<p>本手册的构建版本：20250803</p>",
    B["freebsd"]["license_url"]: ("<p>Redistribution and use in source (AsciiDoc) and &#39;compiled&#39; forms (HTML, PDF, "
                                  "EPUB and so forth) with or without modification, are permitted</p>"),
    books.FREEBSD + "zh-cn/": autoindex("books/", "README.txt"),
    books.FREEBSD + "zh-cn/books/": autoindex("handbook/", "faq/"),
    books.FREEBSD + "zh-cn/books/handbook/": autoindex("handbook_zh-cn.pdf", "handbook_zh-cn.epub"),
    books.FREEBSD + "zh-cn/books/faq/": autoindex("faq_zh-cn.pdf"),
    PY_CN: ('<a href="archives/python-3.13-docs-html.zip">下载</a><a href="archives/python-3.13-docs-pdf-a4.zip">PDF</a>'
            '<a href="archives/python-3.13-docs-pdf-a4.tar.bz2">PDF</a><div class="footer">本页面采用 Python 软件基金会许可证'
            '第 2 版授权。</div>'),
    PY_TW: '<a href="archives/python-3.13-docs-pdf-a4.zip">PDF</a><div class="footer">版權所有</div>',
    SUSE16: (article_row("/zh-cn/sles/16.0/html/SLES-cockpit/", "/zh-cn/sles/16.0/pdf/SLES-cockpit_zh_cn.pdf",
                         "使用 Cockpit 管理 SUSE Linux Enterprise Server")
             + article_row("/appliance/kiwi-9/html/kiwi/index.html", "/appliance/kiwi-9/pdf/kiwi_en.pdf", "KIWI")
             + article_row("/zh-cn/sles/16.0/html/SLES-other/", "/zh-cn/sles/16.0/pdf/SLES-other_zh_cn.pdf", "其他")),
    SUSE15: (article_row("/zh-cn/sles/16.0/html/SLES-cockpit/", "/zh-cn/sles/16.0/pdf/SLES-cockpit_zh_cn.pdf", "重复")
             + article_row("/zh-cn/sles/15-SP7/html/SLES-all/book-security.html",
                           "/zh-cn/sles/15-SP7/pdf/book-security_zh_cn.pdf", "安全和强化指南")),
    "https://documentation.suse.com/zh-cn/sles/16.0/html/SLES-cockpit/": LEGAL,
    "https://documentation.suse.com/zh-cn/sles/15-SP7/html/SLES-all/book-security.html": "<p>目录</p>" + LEGAL,
    "https://documentation.suse.com/zh-cn/sles/16.0/html/SLES-other/": "<p>B GNU licenses B1 GNU Free Documentation License</p>",
}


def test_open_books_files_with_verified_licences():
    logs = []
    fake = FakeFetcher(BOOK_PAGES)
    found = list(books.open_books(fake, {"books": "d2l-zh,debian-reference,debian-installer,nothing", "log": logs.append}))
    assert [c["url"] for c in found] == ["https://zh.d2l.ai/d2l-zh.pdf", "https://zh.d2l.ai/d2l-zh-pytorch.pdf",
                                         B["debian-reference"]["files"][0], B["debian-reference"]["files"][1]]
    d2l, _, reference, reference_tw = found
    assert d2l["key"] == "book:zh.d2l.ai/d2l-zh.pdf" and d2l["license"] == "Apache-2.0" and d2l["type_hint"] == "pdf"
    assert d2l["title"] == "动手学深度学习 (MXNet)"
    assert d2l["meta"] == dict(book="d2l-zh", license_source=B["d2l-zh"]["license_url"], verified=True, lang=None,
                               host="zh.d2l.ai")
    assert reference["license"] == "GPL (any DFSG-compatible version)" and reference["title"] == "Debian 参考手册 (zh-cn)"
    assert reference["meta"]["lang"] == "zh-cn" and reference_tw["meta"]["lang"] == "zh-tw"
    assert any("unknown book nothing" in line for line in logs)
    assert any("install.zh-cn.pdf" in line and "no licence statement" in line for line in logs)
    assert fake.calls.count(B["debian-reference"]["license_url"]) == 1


def test_open_books_autoindex_listing_pages_and_suse_articles():
    logs = []
    fake = FakeFetcher(BOOK_PAGES)
    found = list(books.open_books(fake, {"books": ["freebsd", "python-docs", "suse"], "log": logs.append,
                                         "suse_products": ["sles/16.0", "/sles/15-SP7/"]}))
    assert [c["key"] for c in found] == [
        "book:download.freebsd.org/doc/zh-cn/books/handbook/handbook_zh-cn.pdf",
        "book:download.freebsd.org/doc/zh-cn/books/faq/faq_zh-cn.pdf",
        "book:docs.python.org/zh-cn/3.13/archives/python-3.13-docs-pdf-a4.zip",
        "book:documentation.suse.com/zh-cn/sles/16.0/pdf/SLES-cockpit_zh_cn.pdf",
        "book:documentation.suse.com/zh-cn/sles/15-SP7/pdf/book-security_zh_cn.pdf"]
    handbook, _, python, cockpit, security = found
    assert handbook["license"].startswith("FreeBSD Documentation License") and handbook["title"] == "FreeBSD handbook_zh-cn"
    assert handbook["meta"]["lang"] == "zh-cn"
    assert python["type_hint"] == "zip" and python["license"] == "PSF-2.0" and python["meta"]["license_source"] == PY_CN
    assert python["url"] == books.PYTHON + "zh-cn/3.13/archives/python-3.13-docs-pdf-a4.zip"
    assert cockpit["title"] == "SUSE 文档: 使用 Cockpit 管理 SUSE Linux Enterprise Server"
    assert cockpit["license"] == "GFDL-1.2-or-later"
    assert cockpit["meta"]["license_source"] == "https://documentation.suse.com/zh-cn/sles/16.0/html/SLES-cockpit/"
    assert security["meta"]["license_source"].endswith("/html/SLES-all/book-security.html")
    assert any("zh-tw/" in line and "http_404" in line for line in logs)  # FreeBSD zh-tw root missing here
    assert any("python-3.13-docs-pdf-a4.zip" in line and PY_TW in line for line in logs)  # no footer licence
    assert any("SLES-other_zh_cn.pdf" in line for line in logs)  # only a ToC mention of the GFDL
    assert not any("kiwi" in url for url in fake.calls)
    assert fake.calls.count(PY_CN) == 1


def test_open_books_without_verification_and_version_override():
    fake = FakeFetcher({books.PYTHON + "zh-cn/3.12/download.html": '<a href="archives/python-3.12-docs-pdf-a4.zip">x</a>'})
    found = list(books.open_books(fake, {"books": ["python-docs", "d2l-zh"], "verify": False,
                                         "python_versions": "3.12", "max_items": 2}))
    assert [c["url"].rsplit("/", 1)[1] for c in found] == ["python-3.12-docs-pdf-a4.zip", "d2l-zh.pdf"]
    assert [c["license"] for c in found] == ["PSF-2.0", "Apache-2.0"]
    assert found[0]["meta"]["verified"] is False
    assert fake.calls == [books.PYTHON + "zh-cn/3.12/download.html", books.PYTHON + "zh-tw/3.12/download.html"]
