"""Newspaper e-paper page PDFs from Founder (方正) digital-paper sites: 人民日报 and 经济日报.

rmrb: http://paper.people.com.cn/rmrb/ (plain http, robots.txt 404) has two layouts.
2024-12-01 onwards: pc/layout/YYYYMM/DD/node_NN.html. A node page names its 版 (p.ban
"第01版：要闻"), lists every page of the issue (a#pageLink "02版：要闻" -> node_02.html) and
links only its own PDF, ../../../attachement/YYYYMM/DD/<uuid>.pdf (sic), so an issue costs
one request per page. 2023-01-01 to 2024-11-30: html/YYYY-MM/DD/nbs.D110000renmrb_NN.htm
(UTF-8 with BOM, same a#pageLink list), PDFs at the predictable
images/YYYY-MM/DD/NN/rmrbYYYYMMDDNN.pdf: one request per issue. Each layout 404s outside its
range and dates before 2023 answer 403 (the paid 图文数据库 holds 1946-2021), which ends the
walk. An issue appears every day: 20 pages on weekdays, 8 at weekends and on holidays. The
site notice allows reading and study/research use only: no commercial use, databases or mirrors.
Page PDFs are single pages of 1017x1438 pt (front page 1017x1453), 0.5-2.9 MB (median 1.1),
from 飞腾创艺8.2 网络版 (2024-12-) or pdflib 网络版 (2023-2024), with embedded FZ*--GBK1-0 and
NEU-* TrueType fonts and an exact Unicode text layer. NEU-BZ draws dashed caption frames with
plane-16 private-use code points (U+100948), which push some pages over the renderer's junk
limit (2 of 31 sampled pages classed garbled); 广告 pages may hold placed ad PDFs whose
re-encoded Type1 fonts give ASCII soup (dropped by the language check).

jjrb: http://paper.ce.cn/pc/layout/YYYYMM/DD/node_01.html (robots.txt 404, same Founder
template) lists every page as <a href="node_NN.html">第NN版：…</a> followed by a hidden input
holding that page's ../../../attachment/YYYYMM/DD/<uuid>.pdf: one request per issue. Issues
from 2020-01-02 (2019 dates 404 or answer an empty page; holidays have no issue), 12 pages.
Founder PDF Library output, 1016x1439 pt, 0.8-2.4 MB, embedded CID fonts, clean text layer.

Not supported: 中国青年报 (zqb.cyol.com/pc/layout) and 南方日报 (epaper.southcn.com/nfdaily/html)
run the same system but link no PDFs; 光明日报 links none either; dzb.whb.cn (文汇报) disallows
crawling in robots.txt.
"""

import html
import itertools
import re
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit

from .fetch import FetchError

RMRB = "http://paper.people.com.cn/rmrb/"
SITES = {
    "rmrb": dict(paper="人民日报", license="人民日报（电子版）, all rights reserved (人民网): the site allows reading and "
                                           "study/research use only, no commercial use, databases or mirrors; internal "
                                           "research use",
                 # (first date, issue URL, PDF URL of page {page} relative to it or None: fetch each node page)
                 layouts=[(date(2024, 12, 1), RMRB + "pc/layout/{Y}{m}/{d}/node_01.html", None),
                          (None, RMRB + "html/{Y}-{m}/{d}/nbs.D110000renmrb_01.htm",
                           "../../../images/{Y}-{m}/{d}/{page}/rmrb{Y}{m}{d}{page}.pdf")]),
    "jjrb": dict(paper="经济日报", license="经济日报, all rights reserved (经济日报社/中国经济网); internal research use",
                 layouts=[(None, "http://paper.ce.cn/pc/layout/{Y}{m}/{d}/node_01.html", None)]),
}
REFUSED = ("http_401", "http_403")
PAGE_LINK = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']?([^"'\s>]+)["']?[^>]*>\s*(?:第\s*)?([A-Z]?\d+)\s*版\s*[：:]"""
                       r"""(.*?)</a\s*>(?:\s*<input\b[^>]*?\bvalue\s*=\s*["']([^"']+\.pdf)["'])?""", re.S | re.I)
OWN_PDF = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']?([^"'\s>]+\.pdf)(?=["'\s>])""", re.I)


def _quiet(message):
    pass


def _decode(body):
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return body.decode("gb18030", "replace")


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _day(value):
    if not value or isinstance(value, date):
        return value or None
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _dates(options):
    """Dates from ``start`` (default today in Beijing) back to ``end`` or for ``days`` dates."""
    start = _day(options.get("start")) or datetime.now(timezone(timedelta(hours=8))).date()
    end = _day(options.get("end"))
    if end and end > start:
        start, end = end, start
    step, count = timedelta(days=int(options.get("step", 1))), options.get("days")
    for n in itertools.count():
        day = start - n * step
        if (end and day < end) or (count is not None and n >= int(count)):
            return
        yield day


def pages(text, page_url):
    """[(page, section, node URL, PDF URL or None)] of the issue a node page lists, in order."""
    found = {}
    for href, number, section, pdf in PAGE_LINK.findall(text):
        number = number.zfill(2)
        found.setdefault(number, (number, _text(section), urljoin(page_url, html.unescape(href)),
                                  urljoin(page_url, html.unescape(pdf)) if pdf else None))
    return list(found.values())


def own_pdf(text, page_url):
    found = OWN_PDF.search(text)
    return urljoin(page_url, html.unescape(found.group(1))) if found else None


def _issue(fetcher, site, day, log):
    """(issue URL, pages with PDF URLs where the issue page or a template gives them, error)."""
    _, template, pdf_template = next(l for l in site["layouts"] if l[0] is None or l[0] <= day)
    fields = dict(Y=f"{day:%Y}", m=f"{day:%m}", d=f"{day:%d}")
    url = template.format(**fields)
    try:
        response, body = fetcher.get(url)
    except FetchError as error:
        return url, [], error.reason
    url = getattr(response, "url", None) or url
    text = _decode(body)
    own = own_pdf(text, url)
    listed = [p[:3] + (own,) if p[2] == url and not p[3] else p for p in pages(text, url)]
    if pdf_template and listed:
        def predict(number):
            return urljoin(url, pdf_template.format(page=number, **fields))
        if any(p[2] == url and p[3] == predict(p[0]) for p in listed):
            listed = [p[:3] + (predict(p[0]),) for p in listed]
        else:
            log(f"{url}: PDF link {own} does not follow the template; fetching each page")
    return url, listed, None if listed else "no_pages"


def epaper(fetcher, options):
    """Page PDFs of a Founder e-paper (site rmrb or jjrb), newest first, one candidate per page.

    Options: site (default rmrb), start (YYYY-MM-DD, default today in Beijing), end (YYYY-MM-DD)
    or days (number of dates), step (days between sampled dates, default 1), max_missing
    (consecutive dates without an issue that end the walk, default 10; a 401/403 ends it at
    once), max_items (int), log (callable)."""
    yield from itertools.islice(_walk(fetcher, options), options.get("max_items"))


def rmrb(fetcher, options):
    yield from epaper(fetcher, dict(options, site="rmrb"))


def jjrb(fetcher, options):
    yield from epaper(fetcher, dict(options, site="jjrb"))


def _walk(fetcher, options):
    name = options.get("site", "rmrb")
    site, log = SITES[name], options.get("log") or _quiet
    missing, max_missing = 0, int(options.get("max_missing", 10))
    for day in _dates(options):
        url, listed, error = _issue(fetcher, site, day, log)
        if error in REFUSED:
            log(f"{url}: {error}, access refused; stopping at {day}")
            return
        if error:
            missing += 1
            log(f"no issue {day} ({error})")
            if missing >= max_missing:
                log(f"{missing} dates without an issue; stopping at {day}")
                return
            continue
        missing = 0
        for number, section, node, pdf in listed:
            if not pdf:
                try:
                    pdf = own_pdf(_decode(fetcher.get(node)[1]), node)
                except FetchError as error:
                    log(f"skip {node}: {error.reason}")
                    if error.reason in REFUSED:
                        return
                    continue
                if not pdf:
                    log(f"no PDF link on {node}")
                    continue
            yield dict(key=f"{name}:{day:%Y%m%d}:{number}", url=pdf,
                       title=f"{site['paper']} {day.isoformat()} 第{number}版" + (f"：{section}" if section else ""),
                       type_hint="pdf", license=site["license"], headers={"Referer": node},
                       meta=dict(paper=site["paper"], date=day.isoformat(), page=number, section=section,
                                 page_url=node, issue_url=url, host=urlsplit(pdf).hostname))
