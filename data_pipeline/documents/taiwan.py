"""Executive Yuan Gazette (行政院公報資訊網, Taiwan): one candidate per gazette item PDF.

Terms: Taiwan Copyright Act Art. 9 excludes 憲法、法律、命令或公文 from copyright. The site's
Open Data is data.gov.tw dataset 5959 (國家發展委員會) under 政府資料開放授權條款-第1版 (CC BY
4.0 compatible; attribution required); its FAQ says 開放資料：授權民間業者利用公報原始資料檔進行
加值利用. robots.txt is 404 (no rules); the site sits behind Cloudflare without challenges.

Open Data (https://gazette.nat.gov.tw/egFront/openData01.do, read 2026-10-17): ZIP archives of
<ROC date>/<ROC date>.xml plus one <MetaId>.pdf per record, byte-identical to the item's Eg.pdf.
openData03.do links the current month's daily archives (115-10-16.zip: 2.7 MB, 10 records);
openData04.do?action=historyYearDaily&year=094..115 the earlier months in packs of one to six
issue days (115-09_04.zip 4.5 MB/20 records, 110-06_01.zip 17 MB/37, 100-06_01.zip 15 MB/156,
094-01_01.zip 10 MB/73; 41-169 packs a year); OpenData/downloadXML.jsp is the latest day's XML
alone (72 kB, no date parameter). Links are fileView.do?fileType=...&fileName=<token>&zipfile=
<name>; fileView ignores Range, so an archive is read whole to get at its XML.

A <Record> holds MetaId, Doc_Style_LName/SName (類型), Chapter (篇), PubGovName, Title,
GazetteHTML .../eguploadpub/eg<vol:3><issue:3>/ch<NN>/type<t>/gov<g>/num<n>/Eg.htm (http:// before
2021) with Eg.pdf next to it, and HTMLContent. Recent XML leaves out 處分 and every 公告及送達
except draft-rule previews (預告); older archives keep them (2011: 26 公示送達 and 26 處分 of 156
records, with people's names), so that rule is applied here, plus a title check for lists of
people. Attachments are linked from HTMLContent as images/*.pdf: EgNN.pdf are printed inside
Eg.pdf (checked 2011/2016/2021/2026) except an order's 總說明, which before AA.pdf existed (2016)
was only online; AA.pdf (總說明及對照表, ODF/Word-made) is online only; OO.pdf (其他事項說明)
names the contact officer and is skipped. Explicit metaids go through detail.do?metaid=N, which
links the Eg.pdf. PDFs: iText 2.1.7 (older Acrobat Distiller), born-digital, Traditional
Chinese; MingLiU is sometimes not embedded.
"""

import html
import io
import itertools
import re
import zipfile
import zlib
from datetime import date
from urllib.parse import urljoin

from .fetch import FetchError

SITE = "https://gazette.nat.gov.tw/egFront/"
LICENSE = ("Taiwan Copyright Act Art. 9: laws, orders and official documents are not copyrighted; Open Data "
           "under 政府資料開放授權條款-第1版 (attribution: 國家發展委員會 行政院公報)")
FIRST = date(2005, 1, 1)  # the Open Data history starts with 民國94年
ARCHIVE = re.compile(r"""href=["'](fileView\.do\?[^"']*?zipfile=((\d{3})-(\d{2})(?:-(\d{2})|_\d+))\.zip)["']""")
FIELD = re.compile(r"<(\w+)>(?:<!\[CDATA\[(.*?)\]\]>|([^<]*))</\1>", re.S)
PATH = re.compile(r"/eg(\d{3})(\d{3})/ch(\d+)/")
# Lists of people outside the 處分/公示送達 types; names of rules in 「」 are not looked at.
PERSONAL = re.compile(r"公示送達|處分書|名單|名冊|等\d+(?:人|名|件)|懲戒|判決|通緝|歸化|死亡宣告")
MAX_XML = 64 << 20


def _quiet(message):
    pass


def _list(value):
    return value.split(",") if isinstance(value, str) else list(value)


def _text(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _get(fetcher, url, log):
    try:
        return fetcher.get(url)[1]
    except FetchError as error:
        log(f"skip {url}: {error.reason}")
        return None


def _day(value, default):
    return default if value is None else value if isinstance(value, date) else date.fromisoformat(str(value))


def _roc(text):
    """Date of 115-10-16 or 中華民國115年10月16日."""
    found = re.search(r"(\d{2,3})\D(\d{1,2})\D(\d{1,2})", text or "")
    try:
        return found and date(int(found.group(1)) + 1911, int(found.group(2)), int(found.group(3)))
    except ValueError:
        return None


def ey_gazette(fetcher, options):
    """Gazette items (Eg.pdf) and their online-only attachments, newest first.

    Options: start, end (YYYY-MM-DD; default 2005-01-01 and today) pick issue days from the Open
    Data archives; latest (bool: only the latest issue day, from its XML alone); metaids (list of
    ints: these items via their detail pages instead); chapters (numbers 1-9 of the chNN path or
    name parts such as 內政); styles (類型 such as 法規, 行政規則); attachments (bool, default
    True); skip_personal (bool, default True); max_items (int); log (callable)."""
    yield from itertools.islice(_items(fetcher, options), options.get("max_items"))


def _items(fetcher, options):
    log = options.get("log") or _quiet
    attachments = options.get("attachments", True)
    if options.get("metaids"):
        records = (r for metaid in _list(options["metaids"]) for r in _detail(fetcher, int(metaid), log))
    elif options.get("latest"):
        got = _get(fetcher, SITE + "OpenData/downloadXML.jsp", log)
        records = _records(got.decode("utf-8", "replace"), None, "downloadXML.jsp", None, log) if got else ()
    else:
        records = _archives(fetcher, _day(options.get("start"), FIRST), _day(options.get("end"), date.today()), log)
    seen = set()
    for record in records:
        if record["metaid"] in seen or not _wanted(record, options):
            continue
        seen.add(record["metaid"])
        if record["html"] is None and attachments:
            got = _get(fetcher, record["base"] + "Eg.htm", log)
            record["html"] = got.decode("utf-8", "replace") if got else ""
        yield from _candidates(record, attachments)


def _links(fetcher, url, log):
    """(archive URL, name, year, month, day or None) of an Open Data listing."""
    got = _get(fetcher, url, log)
    found = {}
    for match in ARCHIVE.finditer(got.decode("utf-8", "replace") if got else ""):
        href, name, year, month, day = match.groups()
        found[name] = (urljoin(SITE, html.unescape(href)), name, int(year) + 1911, int(month), day and int(day))
    return list(found.values())


def _archives(fetcher, start, end, log):
    """Records of the archives covering start..end, newest first."""
    links = _links(fetcher, SITE + "openData03.do", log)
    daily = {(year, month) for _, _, year, month, _ in links}
    if (start.year, start.month) not in daily:
        for year in range(end.year, max(start.year, FIRST.year) - 1, -1):
            links += _links(fetcher, f"{SITE}openData04.do?action=historyYearDaily&year={year - 1911:03d}", log)
    for url, name, year, month, day in sorted(links, key=lambda link: link[1], reverse=True):
        if day:
            wanted = start <= date(year, month, day) <= end
        else:  # a pack holds some days of one month; its month may also be listed day by day
            wanted = (year, month) not in daily and (start.year, start.month) <= (year, month) <= (end.year, end.month)
        if not wanted:
            continue
        body = _get(fetcher, url, log)
        days = _unzip(body, name + ".zip", log) if body is not None else []
        if days and days[0][0] < start:
            break  # archives come newest first, so the rest are older still
        for day, folder, text in days:
            if start <= day <= end:
                yield from _records(text, day, name + ".zip", folder, log)


def _unzip(body, name, log):
    """[(day, folder, XML text)] of an archive, newest day first."""
    days = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
        for info in archive.infolist():
            folder, _, file = info.filename.rpartition("/")
            day = file.lower().endswith(".xml") and _roc(file)
            if day and info.file_size <= MAX_XML:
                days.append((day, folder, archive.read(info).decode("utf-8", "replace")))
    except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, OSError, EOFError) as error:
        log(f"bad archive {name}: {error}")
    return sorted(days, key=lambda item: item[0], reverse=True)


def _records(text, day, archive, folder, log):
    for block in re.findall(r"<Record>(.*?)</Record>", text, re.S):
        f = {m.group(1): m.group(2) if m.group(3) is None else html.unescape(m.group(3)) for m in FIELD.finditer(block)}
        page = (f.get("GazetteHTML") or "").strip().replace("http://", "https://", 1)
        if not f.get("MetaId", "").strip().isdigit() or not page.endswith("/Eg.htm"):
            log(f"{archive}: record without id or Eg.htm: {f.get('MetaId')}")
            continue
        metaid = int(f["MetaId"])
        published = day or _roc(f.get("Date_Published"))
        yield dict(metaid=metaid, title=_text(f.get("Title", "")), style=_text(f.get("Doc_Style_LName", "")),
                   substyle=_text(f.get("Doc_Style_SName", "")), chapter=_text(f.get("Chapter", "")),
                   agency=_text(f.get("PubGovName") or f.get("PubGov") or ""),
                   date=published and published.isoformat(), base=page[:-6], html=f.get("HTMLContent") or "",
                   archive=archive, member=None if folder is None else f"{folder}/{metaid}.pdf".lstrip("/"))


def _detail(fetcher, metaid, log):
    got = _get(fetcher, f"{SITE}detail.do?metaid={metaid}&log=detailLog", log)
    text = got.decode("utf-8", "replace") if got else ""
    pdf = re.search(r"""href=["']([^"']*/eguploadpub/[^"']+/)Eg\.pdf["']""", text)
    if not pdf:
        if got:
            log(f"no PDF on the detail page of {metaid}")
        return
    side = text[max(text.find("Side_Links"), 0):]
    title = re.search(r"""property=["']og:title["']\s+content=["']([^"']*)""", text)
    tag = re.search(r"""class=["']Tag["']>([^<]*)<""", side)
    names = re.findall(r"<h2>([^<]*)</h2>", side)

    def field(label):
        found = re.search(rf"<dt>{label}：</dt>\s*<dd>([^<]*)</dd>", side)
        return _text(found.group(1)) if found else ""

    yield dict(metaid=metaid, title=_text(title.group(1)) if title else "", style=_text(tag.group(1)) if tag else "",
               substyle=field("類型"), chapter=_text(names[0]) if names else "",
               agency=_text(names[1]) if len(names) > 1 else "", date=field("出刊日期") or None,
               base=urljoin(SITE, html.unescape(pdf.group(1))), html=None, archive=None, member=None)


def _personal(record):
    """處分 and 公告及送達 other than draft-rule previews (the Open Data's own rule), or a title
    that names people."""
    style = record["style"]
    if style.startswith("處分") or (style.startswith("公告及送達") and "預告" not in record["substyle"]):
        return True
    return bool(PERSONAL.search(re.sub(r"「[^」]*」", "", record["title"])))


def _wanted(record, options):
    if options.get("skip_personal", True) and _personal(record):
        return False
    if options.get("styles") and not any(s and s in record["style"] for s in _list(options["styles"])):
        return False
    if not options.get("chapters"):
        return True
    path = PATH.search(record["base"])
    number = int(path.group(3)) if path else None
    return any(int(c) == number if str(c).strip().isdigit() else str(c).strip() in record["chapter"]
               for c in _list(options["chapters"]))


def _attachments(content, preview):
    """(file name, label) of the PDF attachments that Eg.pdf does not already print."""
    seen = set()
    for match in re.finditer(r"""href=["']images/([\w.-]+\.pdf)["']""", content, re.I):
        name, end = match.group(1), content.rfind("<", 0, match.start())
        label = _text(content[max(content.rfind("<p", 0, end), end - 600, 0):end]).rstrip("（( ")
        if name in seen or name.upper() == "OO.PDF" or label.startswith("其他事項說明"):
            continue
        if re.match(r"Eg\d+\.pdf$", name, re.I) and ("總說明" not in label or preview):
            continue
        seen.add(name)
        yield name, label


def _candidates(record, attachments):
    path = PATH.search(record["base"])
    meta = dict(metaid=record["metaid"], date=record["date"], volume=path and int(path.group(1)),
                issue=path and int(path.group(2)), chapter=record["chapter"], chapter_no=path and int(path.group(3)),
                style=record["style"], substyle=record["substyle"], agency=record["agency"],
                detail=f"{SITE}detail.do?metaid={record['metaid']}", archive=record["archive"],
                host="gazette.nat.gov.tw")
    key = f"twgazette:{record['metaid']}"
    yield dict(key=key, url=record["base"] + "Eg.pdf", title=record["title"], type_hint="pdf", license=LICENSE,
               meta=dict(meta, part="main", archive_member=record["member"]))
    if not attachments:
        return
    for name, label in _attachments(record["html"] or "", "預告" in record["substyle"]):
        title = label if len(label) >= 6 else f"{record['title']}（{label or name}）"
        yield dict(key=f"{key}:{name[:-4]}", url=record["base"] + "images/" + name, title=title, type_hint="pdf",
                   license=LICENSE, meta=dict(meta, part="attachment", attachment=name, item_title=record["title"]))
