"""ModelScope dataset repositories that hold raw document files (PDF, Word, PowerPoint).

Endpoints are the ones the official SDK uses (modelscope_hub 0.4.6); robots.txt of
www.modelscope.cn disallows only the HTML file pages (/datasets/*/*/file, /files) and its
comments point programmatic clients to /api/v1:
  GET https://www.modelscope.cn/api/v1/datasets/<owner>/<name>  -> Data.{License, StorageSize, ...}
  GET .../repo/tree?Revision=&Root=&Recursive=True&PageNumber=&PageSize=
      -> Data.{TotalCount, Files[].{Path, Name, Type blob|tree, Size, IsLFS, Sha256}}
  GET .../repo?Revision=<rev>&FilePath=<quote_plus(path)>: small files 200 from www, LFS files 302
      to https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/<sha256 split 2/2/60>?filename=..&auth_key=
      (Alibaba CDN type-A URL signing, timestamp = issue time; the file's sha256 is the tree's Sha256).

Measured 2026-10-17: TotalCount counts blobs and trees; pages are disjoint and complete (11,599 and
13,509 unique paths over 24 and 28 pages of 500, ~2.2 s per page at a 1 s interval). PageSize is
honoured up to 2,000 while 5,000 returns 3,000 rows, so paging stops at TotalCount rather than at
a short page. An unknown dataset is a 404 on both endpoints. The /repo endpoint answers 400 to a
suffix Range (bytes=-N); explicit ranges pass through to the CDN (206). The CDN answers 403 to
every unsigned path, robots.txt included (X-Tengine-Error: denied by req auth: no url arg
auth_key); the Fetcher reads a 401/403 robots.txt as disallow except this signed-URL refusal,
which it counts as robots.txt unavailable (allow-all, RFC 9309 2.3.1.3) with a logged warning, so
LFS downloads follow the 302 to the signed CDN URL like any other hop.

Datasets (both declare Apache-2.0, which cannot relicense the issuers' reports: terms disclosure;
files are byte-identical to the static.cninfo.com.cn copies, so the Store dedupes them with cninfo):
  modelscope/chatglm_llm_fintech_raw_dataset (FinGLM, ChatGLM 金融赛道): 11,588 A-share annual
    reports FY2019-2021 of 4,366 stock codes, published 2020-01-21..2023-06-28; 74.4 GB, median
    6.3 MB, max 91 MB, median 189 pages; allpdf/<date>__<company>__<code>__<short>__<year>年__年度报告.pdf.
  baimuqier/stocks_financial_reports: 13,506 SSE/SZSE main-board and STAR reports of 2,699 codes
    (Q1, half-year and Q3 2023, ~1,000 FY2022/2023 annual reports, 3,775 summaries or English
    versions); 20.9 GB, median 0.5 MB; 2023/<code>_<short name><announcement title>.pdf.
Render survey (36 FinGLM and 27 baimuqier files, every third page): 3,284/3,284 pages class text,
no garbled, vector or scanned pages; producers MS Word 59, Acrobat Distiller 1, none 3; 88-97% of
pages per report type reach min_cjk 0.3 (tables and contents pages miss it). An earlier
180-page FinGLM sample had 59 garbled pages from PDF-XChange printer output (broken ToUnicode).
"""

import fnmatch
import hashlib
import itertools
import json
import re
from urllib.parse import quote_plus, urlencode

from .fetch import FetchError

API = "https://www.modelscope.cn/api/v1/datasets/"
FINGLM = "modelscope/chatglm_llm_fintech_raw_dataset"
REPORTS_2023 = "baimuqier/stocks_financial_reports"
DISCLOSURE = "issuer copyright (statutory disclosure re-hosted on ModelScope; repository declares {})"
LICENSES = {FINGLM: DISCLOSURE, REPORTS_2023: DISCLOSURE}
TYPES = ("pdf", "doc", "docx", "ppt", "pptx")
EXCLUDE = "摘要|英文|English"
PAGE, MAX_PAGE = 1000, 2000
FINGLM_NAME = re.compile(r"(\d{4}-\d\d-\d\d)__(.+?)__(\d{6})__(.+?)__((?:19|20)\d\d)年__(.+)$")
CODE_NAME = re.compile(r"(\d{6})_(.+)$")
YEAR = re.compile(r"(?:19|20)\d\d|二[〇零○][一二三四五六七八九〇零○]{2}")
NUMERALS = dict(zip("〇零○一二三四五六七八九", "000123456789"))
REPORTS = (("半年度报告|半年报|中期报告", "半年度报告"), ("一季度报告|一季报", "第一季度报告"),
           ("三季度报告|三季报", "第三季度报告"), ("年度报告|年报", "年度报告"))
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([KMG]?)I?B?", re.I)


def _quiet(message):
    pass


def _list(value):
    return [v.strip() for v in value.split(",") if v.strip()] if isinstance(value, str) else list(value)


def _bytes(value):
    if value is None or isinstance(value, (int, float)):
        return value
    if not (m := SIZE.fullmatch(value.strip())):
        raise ValueError(f"modelscope: max_size {value!r} is not a size like 50000000 or '50MB'")
    return int(float(m[1]) * 1024 ** " KMG".index(m[2].upper() or " "))


def _json(fetcher, url):
    try:
        reply = json.loads(fetcher.get(url)[1])
    except ValueError:
        raise FetchError("not_json", url) from None
    if reply.get("Code") not in (200, None) or reply.get("Success") is False:
        raise FetchError(f"api_{reply.get('Code')}", url)
    return reply.get("Data") or {}


def tree(fetcher, dataset, revision="master", root="", page_size=PAGE, log=_quiet):
    """Every entry of the repository tree under ``root``, paged until TotalCount."""
    size, seen, page, head, total = max(1, min(int(page_size), MAX_PAGE)), 0, 1, None, None
    while total is None or seen < total:
        query = urlencode(dict(Revision=revision, Root=root, Recursive="True", PageNumber=page, PageSize=size))
        data = _json(fetcher, f"{API}{dataset}/repo/tree?{query}")
        files, total = data.get("Files") or [], int(data.get("TotalCount") or 0)
        if not files or files[0].get("Path") == head:
            break
        head = head or files[0].get("Path")
        yield from files
        seen, page = seen + len(files), page + 1
    if total and seen < total:
        log(f"modelscope: {dataset} tree stopped at {seen} of {total} entries (page {page})")


def parse_name(path):
    """Date, company, code, short name, report year and type from FinGLM or code_title file names."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if m := FINGLM_NAME.match(stem):
        meta = dict(date=m[1], company=m[2], code=m[3], short_name=m[4], year=int(m[5]))
        return dict(meta, report_type=_report(m[6]) or m[6], title=f"{m[2]}{m[5]}年{m[6]}")
    if not (m := CODE_NAME.match(stem)):
        return {"title": stem}
    meta, rest = dict(code=m[1]), m[2]
    if year := YEAR.search(rest):
        meta.update(short_name=rest[:year.start()].strip() or None,
                    year=int("".join(NUMERALS.get(c, c) for c in year[0])))
    if report := _report(rest[year.start():] if year else rest):
        meta["report_type"] = report
    return dict(meta, title=rest)


def _report(title):
    for pattern, name in REPORTS:
        if re.search(pattern, title):
            return name
    return None


def _share(path, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{path}".encode()).digest()[:8], "big") / 2 ** 64


def files(fetcher, options):
    """One candidate per document file of one ModelScope dataset repository.

    Options: dataset ("owner/name", required); revision (default master); root (directory to
    list, default the whole repo); pattern (fnmatch glob on the path, e.g. "allpdf/*__2021年__*");
    regex (searched in the path); exclude (regex on the path, default ``EXCLUDE``: summaries and
    English versions; "" keeps all); types (extensions, default pdf,doc,docx,ppt,pptx); max_size
    (bytes or "50MB"; default the fetcher's max_bytes); sample (deterministic subsample by a
    seeded hash of the path: a fraction in (0, 1] or an integer k >= 2 for one in k; independent
    of listing order and stable when files are added); seed (default 0); license (overrides
    the licence text); page_size (default 1000, at most 2000); max_items; log."""
    dataset = options.get("dataset")
    if not dataset or dataset.count("/") != 1:
        raise ValueError("modelscope: option dataset='owner/name' is required")
    log = options.get("log") or _quiet
    revision = options.get("revision") or "master"
    info = _json(fetcher, API + dataset)
    licence = options.get("license") or LICENSES.get(dataset, "as declared by the repository: {}").format(
        info.get("License") or "none")
    log(f"modelscope: {dataset} licence {info.get('License')!r}, {(info.get('StorageSize') or 0) / 1e9:.1f} GB")
    yield from itertools.islice(_files(fetcher, dataset, revision, licence, options, log), options.get("max_items"))


def _files(fetcher, dataset, revision, licence, options, log):
    types = {t.lower().lstrip(".") for t in _list(options.get("types") or TYPES)}
    glob, regex = options.get("pattern"), re.compile(options["regex"]) if options.get("regex") else None
    exclude = options.get("exclude", EXCLUDE)
    exclude = re.compile(exclude, re.I) if exclude else None
    max_size = _bytes(options.get("max_size") or getattr(fetcher, "max_bytes", None))
    sample, seed = options.get("sample"), options.get("seed", 0)
    share = None if sample is None else 1 / sample if isinstance(sample, int) and sample >= 2 else float(sample)
    counts = dict(blobs=0, type=0, pattern=0, size=0, sample=0, kept=0)
    for entry in tree(fetcher, dataset, revision, options.get("root") or "", options.get("page_size", PAGE), log):
        path = entry.get("Path") or ""
        if entry.get("Type") != "blob":
            continue
        counts["blobs"] += 1
        kind = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        if kind not in types:
            counts["type"] += 1
        elif ((glob and not fnmatch.fnmatchcase(path, glob)) or (regex and not regex.search(path))
              or (exclude and exclude.search(path))):
            counts["pattern"] += 1
        elif max_size and (entry.get("Size") or 0) > max_size:
            counts["size"] += 1
        elif share is not None and _share(path, seed) >= share:
            counts["sample"] += 1
        else:
            counts["kept"] += 1
            parsed = parse_name(path)
            meta = dict(dataset=dataset, revision=revision, path=path, size=entry.get("Size"),
                        sha256=entry.get("Sha256") or None, lfs=entry.get("IsLFS"))
            meta.update({k: v for k, v in parsed.items() if k != "title" and v is not None})
            yield dict(key=f"modelscope:{dataset}:{path}", title=parsed["title"], type_hint=kind, license=licence,
                       url=f"{API}{dataset}/repo?Revision={quote_plus(revision)}&FilePath={quote_plus(path)}",
                       meta=meta)
    log(f"modelscope: {dataset} {counts}")


def finglm(fetcher, options):
    """FinGLM A-share annual reports FY2019-2021 (the dataset is fixed; see ``files`` for options)."""
    return files(fetcher, dict(options, dataset=FINGLM))


def reports_2023(fetcher, options):
    """2023 periodic reports of SSE/SZSE main-board and STAR companies (dataset fixed; see ``files``)."""
    return files(fetcher, dict(options, dataset=REPORTS_2023))
