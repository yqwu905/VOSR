"""UN official documents in Chinese from the bot-yaya/UPRPRC Hugging Face dumps of the UN
Official Document System.

documents.un.org's robots.txt disallows /api, /doc and /access, so the files come from the
dumps on Hugging Face (robots.txt: Allow: /): bot-yaya/UPRPRC_pdffiles_from_UN (451 shards,
860.9 GB) and bot-yaya/UPRPRC_docfiles_from_UN (179 shards, 247.7 GB), 594,160 rows each, one
per document symbol. https://huggingface.co/api/datasets/<name> gives the head commit and
/api/datasets/<name>/tree/<commit>/data lists the shards in one page; shards are read at that
commit so candidates keep pointing at the bytes they were found in. Per language (index 0-6:
ar, zh, en, fr, ru, es, de) blobs[i] holds the file and crawl_res[i] its extension or ''; sizes
has three entries per language (pdf, docx, doc). Discovery reads the footer (~50 KB, cached)
and per row group only the small columns (2 range reads, ~10 KB); row groups whose
publication_date statistics fall outside since/until are skipped unread.

Layout (all 630 footers read 2026-10-17): PDF shards (parquet-cpp-arrow 22.0.0) hold 9,471
row groups of 65 rows with an offset index. Their blobs column chunk (fixed_size_list of 7
large_binary) is 62 MB at the median, 169 MB p90, 647 MB p99 and up to 2.6 GB (618 groups
over 200 MB); the first values sit in a dictionary page of 1-21 MB, then come record-aligned
PLAIN data pages of 1-2 MB holding 2-4 rows (one page per row of large files, up to 67 MB
seen). fetch.parquet_blob therefore reads only the pages of the wanted rows (all 7 languages
of each, plus the dictionary page when used). DOC shards (arrow 19.0.1) hold 6,086 groups of
100 rows without a page index, most values in the dictionary page, so the whole chunk is
read: 23 MB median, 80 MB p90, 356 MB p99, 1.6 GB max; the 153 groups over the default 200 MB
limit fail as too_large without being read (doc-fetch --max-mb reads them; pyarrow then holds
a few times the chunk in memory). Each shard covers about one publication year (pdf 00150: 2008,
00260: 2018; doc 00015: 1997, 00120: 2022-23) with older documents mixed in (statistics
minimum 1944-1958), so since prunes groups by their maximum and until rarely prunes.

Measured: a zh file in 242/733 rows of 12 random PDF groups and 2,057/6,585 rows of 5 PDF
shards (31%); 370/1,000 rows of 10 random DOC groups and 3,292 of 2 DOC shards (50%; 731 of
them PDFs, and 2022 .doc entries are DOCX). Discovery reads 0.3-0.4 MB per PDF shard and 0.8
MB per DOC shard, 87 s and 138 s at the default 2 s interval. Download cost per zh file:
PDF 2.0 MB (cost model over the 12 groups; 4.0 MB if whole chunks were read; mean zh PDF 0.37
MB) and 1.77 MB live (116 PDFs from 4 groups, 205 MB in 40 requests, including a 601 MB
chunk read as 92 MB); DOC 0.62 MB (model; mean zh file 0.14 MB) and 0.57 MB live (100 files,
57 MB, 8 requests). Rendered (first 20 pages each, 201 distinct files): 1,253 text, 4
garbled, 5 blank pages; 1,086 kept (CJK median 0.80), 70 dropped by the CJK threshold (most
with no CJK at all), 100 as near-duplicates. PDFs embed SimSun/SimHei/KaiTi (Word output); converted
Word files use the Noto CJK/UKai aliases. No scans in that sample (1997-2022 files); an
earlier survey found 3 of 20 older index-1 PDFs to be scans, which the renderer drops. The
same zh file can sit under several symbols (15 of 216 downloads were byte-identical); the
store deduplicates them.

Copyright: UN official records and documents are in the public domain as a general rule
(ST/AI/189/Add.9/Rev.2); the dataset card says MIT for the packaging. Lists of participants
and petitions from individuals are skipped (personal names).
"""

import itertools
import random
import re
from pathlib import Path

from .commoncrawl import CACHE, _errors, _parquet, _row_group

HF = "https://huggingface.co/"
DATASETS = {"pdf": "bot-yaya/UPRPRC_pdffiles_from_UN", "doc": "bot-yaya/UPRPRC_docfiles_from_UN"}
LANGUAGES = ["ar", "zh", "en", "fr", "ru", "es", "de"]
ZH = 1
COLUMNS = ["id", "symbol", "publication_date", "area", "distribution", "job_numbers", "sizes", "title", "crawl_res"]
TYPES = {"pdf": "pdf", "doc": "doc", "docx": "docx", "rtf": "rtf"}  # WordPerfect (wpf) does not render
SIZE_SLOT = {"pdf": 0, "docx": 1}  # sizes[3 * language + slot]; doc and the rest use slot 2
LICENSE = ("UN official document: public domain as a general rule (UN ST/AI/189/Add.9/Rev.2); "
           "Hugging Face packaging bot-yaya/UPRPRC (MIT)")
PERSONAL = re.compile(r"\blist of (participants|delegations|delegates|speakers|candidates)\b|\bpetitions? from\b", re.I)


def _quiet(message):
    pass


def _shards(value, count):
    """Shard numbers from a list or a string such as "0-9,200"; all shards by default."""
    if value is None:
        return list(range(count))
    if isinstance(value, int):
        return [value]
    picked = []
    for part in re.split(r"[\s,]+", value) if isinstance(value, str) else value:
        low, _, high = str(part).partition("-")
        if low:
            picked += range(int(low), int(high or low) + 1)
    return [s for s in dict.fromkeys(picked) if 0 <= s < count]


def _year(value):
    return int(value[:4]) if value and value[:4].isdigit() else None


def _dated(row_group, column, since, until):
    """False when the group's publication_date statistics lie wholly outside [since, until]."""
    stats = row_group.column(column).statistics
    if (since is None and until is None) or stats is None or not stats.has_min_max:
        return True
    low, high = _year(stats.min), _year(stats.max)
    return not ((since and high and high < since) or (until and low and low > until))


def uprprc(fetcher, options):
    """Chinese files (language index 1) of the UPRPRC dumps as parquet-cell candidates.

    Options: kind ("pdf", "doc" or "both"; default pdf), shards (list or string like "0-9,200";
    default all, in order), seed (int: shuffles the shard order), since / until (publication
    years, inclusive), types (crawl_res extensions kept, default pdf, doc, docx, rtf),
    revision (dataset commit; default the current head), cache (footer cache directory,
    default ~/.cache/vosr/un_footers), max_items (int), log (callable). Candidates of a row
    group come together and name all of the group's wanted rows, which fetch.parquet_blob
    reads in one pass."""
    yield from itertools.islice(_uprprc(fetcher, options), options.get("max_items"))


def _uprprc(fetcher, options):
    log = options.get("log") or _quiet
    kinds = ["pdf", "doc"] if options.get("kind") == "both" else [options.get("kind", "pdf")]
    since, until = (int(options[k]) if options.get(k) else None for k in ("since", "until"))
    types = options.get("types") or TYPES
    types = set(re.split(r"[\s,]+", types) if isinstance(types, str) else types)
    cache = Path(options.get("cache") or CACHE / "un_footers").expanduser()
    for kind in kinds:
        dataset = DATASETS[kind]
        revision = options.get("revision") or fetcher.json(f"{HF}api/datasets/{dataset}")["sha"]
        listing = fetcher.json(f"{HF}api/datasets/{dataset}/tree/{revision}/data")
        files = sorted((e for e in listing if e["path"].endswith(".parquet")), key=lambda e: e["path"])
        shards = _shards(options.get("shards"), len(files))
        if options.get("seed") is not None:
            random.Random(options["seed"]).shuffle(shards)
        log(f"un {kind}: {dataset}@{revision[:10]}, {len(shards)} of {len(files)} shards")
        for shard in shards:
            entry = files[shard]
            url = f"{HF}datasets/{dataset}/resolve/{revision}/{entry['path']}"
            oid = ((entry.get("lfs") or {}).get("oid") or entry.get("oid") or "")[:16]
            try:
                remote, parquet = _parquet(fetcher, url, cache / kind / f"{Path(entry['path']).stem}.{oid}.footer")
            except _errors() as error:
                log(f"un {kind}: skipped {entry['path']}: {getattr(error, 'reason', error)}")
                continue
            metadata = parquet.metadata
            date = metadata.schema.names.index("publication_date")
            found = 0
            for group in range(metadata.num_row_groups):
                if not _dated(metadata.row_group(group), date, since, until):
                    continue
                try:
                    rows = _row_group(remote, parquet, group, COLUMNS).to_pylist()
                except _errors() as error:
                    log(f"un {kind}: skipped {entry['path']} row group {group}: {getattr(error, 'reason', error)}")
                    continue
                picked = [(i, row) for i, row in enumerate(rows) if _wanted(row, types, since, until)]
                where = dict(dataset=dataset, revision=revision, shard=entry["path"], row_group=group)
                for i, row in picked:
                    yield _candidate(row, kind, dict(url=url, row_group=group, row=i, column="blobs", item=ZH,
                                                     rows=[j for j, _ in picked]), where)
                found += len(picked)
            log(f"un {kind}: {entry['path']}: {found} zh files; {remote.transferred / 1e6:.2f} MB read")


def _wanted(row, types, since, until):
    year = _year(row["publication_date"])
    return ((row["crawl_res"] or [""] * 7)[ZH] in types and not PERSONAL.search(row["title"] or "")
            and not (since and (year is None or year < since)) and not (until and (year is None or year > until)))


def _candidate(row, kind, reference, where):
    extension = row["crawl_res"][ZH]
    size = (row["sizes"] or [-1] * 21)[3 * ZH + SIZE_SLOT.get(extension, 2)]
    meta = dict(id=row["id"], symbol=row["symbol"], date=(row["publication_date"] or "")[:10], area=row["area"],
                distribution=row["distribution"], job_number=(row["job_numbers"] or [""] * 7)[ZH] or None,
                size=size if size and size > 0 else None,
                languages=[code for code, ext in zip(LANGUAGES, row["crawl_res"]) if ext], **where)
    return dict(key=f"un:{row['symbol'] or row['id']}:{kind}", parquet=reference, title=row["title"] or row["symbol"],
                type_hint=TYPES.get(extension, extension), license=LICENSE, meta=meta)
