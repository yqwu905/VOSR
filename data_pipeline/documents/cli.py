"""doc-sources / doc-discover / doc-fetch / doc-render commands of python -m data_pipeline."""

import itertools
import json
import sys
from pathlib import Path

from ..pipeline import json_text
from . import catalog
from .fetch import AGENT, Fetcher, Store, download, session
from .sniff import RENDERABLE


def _options(pairs):
    options = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _fetcher(args):
    return Fetcher(http=session(AGENT, args.contact), interval=args.interval, max_bytes=int(args.max_mb * (1 << 20)))


def sources():
    return {name: {"terms": spec["terms"], "doc_types": spec["doc_types"], "content": spec["content"],
                   "runs_by_default": spec["terms"] in catalog.DEFAULT_TERMS}
            for name, spec in catalog.SOURCES.items()}


def discover(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("candidate list already exists; choose a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    fetcher = _fetcher(args)
    try:
        options = _options(args.opt)  # the flags, not --opt, decide logging and data.commoncrawl.org reads
        options["log"] = lambda message: print(message, file=sys.stderr)
        options["cc_data_host"] = args.cc_data_host
        stream = catalog.discover(args.source, fetcher, options, set(args.accept_terms or ()))
        count, seen = 0, set()
        with output.with_suffix(output.suffix + ".tmp").open("w", encoding="utf-8") as handle:
            try:
                for candidate in itertools.islice(stream, args.limit):
                    if candidate["key"] in seen:
                        continue
                    seen.add(candidate["key"])
                    handle.write(json_text(candidate) + "\n")
                    handle.flush()  # slow sources (Crawl-delay 10) must not lose what they found
                    count += 1
            except KeyboardInterrupt:
                print("interrupted: keeping the candidates found so far", file=sys.stderr)
    except PermissionError as error:
        raise ValueError(str(error)) from error
    output.with_suffix(output.suffix + ".tmp").replace(output)
    return {"source": args.source, "candidates": count, "output": str(output)}


def fetch(args):
    accepted = set(catalog.DEFAULT_TERMS) | set(args.accept_terms or ())
    candidates = []
    # Only "\n" ends a record: titles may hold U+2028, U+0085 and the like, which splitlines() also splits on.
    for number, line in enumerate(Path(args.candidates).read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.candidates} line {number}: not a JSON candidate ({error.msg})") from None
            if not isinstance(candidates[-1], dict):
                raise ValueError(f"{args.candidates} line {number}: not a JSON object")
    refused = [c for c in candidates if c.get("terms") not in accepted]
    if refused:
        raise ValueError(f"{len(refused)} candidates have terms {sorted({c.get('terms') for c in refused})}; "
                         "pass --accept-terms after reviewing them")
    # By default only documents count as downloads; HTML (login walls, JS guards) does not.
    accept = set(args.types.split(",")) if args.types else RENDERABLE | {"zip", "rar"}
    counts = download(candidates, Store(args.store), _fetcher(args), accept=accept, limit=args.limit,
                      refetch_truncated=args.refetch_truncated, cc_data_host=args.cc_data_host)
    return {"store": args.store, "counts": counts}


def render(args):
    from .render import render_store
    return render_store(args.store, args.output, text_height=args.text_height, min_cjk=args.min_cjk,
                        max_pages=args.max_pages, limit=args.limit, workers=args.workers, seed=args.seed,
                        detector=args.detector, max_unlabeled=args.max_unlabeled)


def add_parsers(commands):
    def network(parser):
        parser.add_argument("--interval", type=float, default=2.0, help="seconds between requests to one host")
        parser.add_argument("--max-mb", type=float, default=200, help="largest file to download")
        parser.add_argument("--contact", help="URL or address appended to the User-Agent (none by default)")
        parser.add_argument("--cc-data-host", action="store_true",
                            help="allow reads from data.commoncrawl.org (its robots.txt says Disallow: /; Common "
                                 "Crawl documents it as the HTTPS download endpoint). Default: avoid it")
        parser.add_argument("--accept-terms", action="append", metavar="CLASS",
                            help="allow a reviewed terms class: open_nc, disclosure, public_web, study_only, web_mixed")

    commands.add_parser("doc-sources", help="list Chinese document sources and their terms")
    disc = commands.add_parser("doc-discover", help="list download candidates of one document source")
    disc.add_argument("--source", required=True, choices=sorted(catalog.SOURCES))
    disc.add_argument("--output", required=True)
    disc.add_argument("--limit", type=int)
    disc.add_argument("--opt", action="append", metavar="KEY=VALUE", help="adapter option (JSON values allowed)")
    network(disc)
    get = commands.add_parser("doc-fetch", help="politely download candidates into a content-addressed store")
    get.add_argument("--candidates", required=True)
    get.add_argument("--store", required=True)
    get.add_argument("--limit", type=int, help="stop after this many successful downloads")
    get.add_argument("--types", help="comma-separated sniffed types to keep, e.g. pdf,pptx,ppt,docx,doc")
    get.add_argument("--refetch-truncated", action="store_true", help="fetch truncated WARC records live")
    network(get)
    draw = commands.add_parser("doc-render", help="convert and render stored documents into HR pages with text boxes")
    draw.add_argument("--store", required=True)
    draw.add_argument("--output", required=True)
    draw.add_argument("--text-height", default="32,64", help="target median text-line height in px, or a range")
    draw.add_argument("--min-cjk", type=float, default=0.3, help="minimum CJK share of a page's text")
    draw.add_argument("--max-pages", type=int, default=200, help="pages rendered per document")
    draw.add_argument("--limit", type=int, help="documents to render")
    draw.add_argument("--workers", type=int, default=2)
    draw.add_argument("--seed", type=int, default=42)
    draw.add_argument("--detector", help="rapidocr: flag text the text layer misses (about 0.7 s per page)")
    draw.add_argument("--max-unlabeled", type=float, default=0.05, help="drop pages above this unlabeled-text area")


def run(args):
    if args.command == "doc-sources":
        return sources()
    return {"doc-discover": discover, "doc-fetch": fetch, "doc-render": render}[args.command](args)
