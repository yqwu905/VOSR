"""doc-discover / doc-fetch command plumbing (offline): option precedence and the candidate reader."""

from types import SimpleNamespace

import pytest

from data_pipeline.documents import catalog, cli
from data_pipeline.pipeline import json_text


def network(**kwargs):
    return SimpleNamespace(accept_terms=None, cc_data_host=False, contact=None, interval=0, max_mb=1, limit=None,
                           **kwargs)


def test_discover_flags_win_over_opt(tmp_path, monkeypatch):
    seen = {}

    def discover(source, fetcher, options, accepted):
        seen.update(options)
        yield dict(key="k", url="https://www.gov.cn/a.pdf")

    monkeypatch.setattr(catalog, "discover", discover)
    args = network(source="moj_xzfg", output=str(tmp_path / "c.jsonl"), opt=["cc_data_host=true", "log=nope", "x=1"])
    assert cli.discover(args)["candidates"] == 1
    assert seen["cc_data_host"] is False and callable(seen["log"]) and seen["x"] == 1


def test_refused_or_misconfigured_source_leaves_no_partial_list(tmp_path, monkeypatch):
    output = tmp_path / "c.jsonl"
    for error, raised in ((PermissionError("refused"), ValueError), (ValueError("bad option"), ValueError)):
        def discover(source, fetcher, options, accepted):
            raise error
            yield

        monkeypatch.setattr(catalog, "discover", discover)
        with pytest.raises(raised, match=str(error)):
            cli.discover(network(source="moj_xzfg", output=str(output), opt=[]))
        assert list(tmp_path.iterdir()) == []


def test_fetch_splits_candidates_on_newlines_only_and_names_bad_lines(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(cli, "download", lambda candidates, *args, **kwargs: got.extend(candidates) or {})
    titles = ["关于印发" + chr(0x2028) + "通知", "标题" + chr(0x85) + "续", "第一段" + chr(0x2029) + "第二段"]
    path = tmp_path / "c.jsonl"
    path.write_text("".join(json_text(dict(key=str(i), url=f"https://www.gov.cn/{i}.pdf", title=t, terms="official"))
                            + "\n\n" for i, t in enumerate(titles)), encoding="utf-8")
    cli.fetch(network(candidates=str(path), store=str(tmp_path / "store"), types=None, refetch_truncated=False))
    assert [c["title"] for c in got] == titles
    for bad, message in (('{"key": "0", "terms": "official"}\n\n{"key": \n', "line 3: not a JSON candidate"),
                         ('{"key": "0", "terms": "official"}\n["a list"]\n', "line 2: not a JSON object")):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            cli.fetch(network(candidates=str(path), store=str(tmp_path / "store"), types=None, refetch_truncated=False))


def test_options_that_decide_which_files_count_as_open_are_locked():
    for source, options in (("open_books_zh", {"verify": False}), ("ctan_zh", {"licenses": ["nocommercial"]}),
                            ("zenodo_cjk", {"licenses": "cc-by-nc-4.0"})):
        with pytest.raises(PermissionError):
            next(catalog.discover(source, None, options))


def test_fetch_names_refused_terms_even_when_some_are_missing(tmp_path):
    candidates = tmp_path / "c.jsonl"
    candidates.write_text(json_text({"key": "a", "url": "https://a.cn/x.pdf", "terms": "public_web"}) + "\n"
                          + json_text({"key": "b", "url": "https://a.cn/y.pdf"}) + "\n", encoding="utf-8")
    args = network(candidates=str(candidates), store=str(tmp_path / "store"), types=None, refetch_truncated=False)
    with pytest.raises(ValueError, match="2 candidates have terms"):
        cli.fetch(args)
