"""Tests of the board: terminal table, best runs by local vs public score, escaped HTML."""

from __future__ import annotations

from pathlib import Path

from test_hk_lab_support import make_lab

from hackathon.lab.board import (
    best_local,
    best_public,
    format_duration,
    format_number,
    metrics_text,
    public_text,
    render_html,
    render_text,
    sha_text,
    write_board,
)
from hackathon.lab.registry import RunRecord


def rec(run_id: str, **kw: object) -> RunRecord:
    base: dict[str, object] = {
        "run_id": run_id,
        "group_id": run_id,
        "task": "unlearning",
        "idea": "an idea",
        "spec_id": "s",
        "variant": "v1",
        "sha": "abcdef0123456789",
        "where": "local",
        "status": "done",
        "created_at": "2026-11-21T10:00:00",
        "updated_at": "2026-11-21T10:00:00",
    }
    base.update(kw)
    return RunRecord(**base)  # type: ignore[arg-type]


def test_best_local_respects_the_direction_of_the_primary_metric() -> None:
    lower = [
        rec("a", primary_metric="kl", primary_value=0.5, higher_is_better=False),
        rec("b", primary_metric="kl", primary_value=0.2, higher_is_better=False),
    ]
    assert [b.run.run_id for b in best_local(lower)] == ["b"]
    higher = [
        rec("a", primary_metric="acc", primary_value=0.5),
        rec("b", primary_metric="acc", primary_value=0.9),
        rec("c", primary_metric="acc", primary_value=0.7),
    ]
    best = best_local(higher)
    assert [b.run.run_id for b in best] == ["b"] and best[0].value == 0.9


def test_best_local_ignores_failed_runs_and_ranks_each_metric_apart() -> None:
    records = [
        rec("fail", status="failed", primary_metric="acc", primary_value=0.99),
        rec("ok", primary_metric="acc", primary_value=0.5),
        rec("other", primary_metric="mcc", primary_value=0.1),
        rec("none", primary_metric="acc", primary_value=None),
    ]
    assert {b.label: b.run.run_id for b in best_local(records)} == {"acc": "ok", "mcc": "other"}


def test_best_public_is_separate_from_best_local() -> None:
    records = [
        rec(
            "local-winner",
            primary_metric="kl",
            primary_value=0.1,
            higher_is_better=False,
            public_score=0.2,
        ),
        rec(
            "public-winner",
            primary_metric="kl",
            primary_value=0.9,
            higher_is_better=False,
            public_score=0.8,
        ),
    ]
    assert best_local(records)[0].run.run_id == "local-winner"
    top = best_public(records)
    assert top is not None and top.run.run_id == "public-winner"
    assert best_public([rec("x")]) is None


def test_text_board_lists_runs_and_both_bests_with_the_guidance() -> None:
    records = [
        rec("r1", primary_metric="kl", primary_value=0.1234567, higher_is_better=False,
            metrics={"kl": 0.1234567, "z": 2.0}, public_score=0.55, duration_s=83, dirty=True),
        rec("r2", task="model_equality", idea="x" * 60, status="failed"),
    ]  # fmt: skip
    text = render_text(records)
    assert "r1" in text and "r2" in text and "1m23s" in text and "0.1235" in text
    assert "abcdef0*" in text, "a dirty tree is starred"
    assert "best local  (kl) : 0.1235  r1" in text
    assert "best public (info only)  : 0.55  r1" in text
    assert "30 %" in text and "information only" in text
    assert "x" * 29 not in text, "long ideas are truncated"
    assert "model_equality: 1 run(s)" in text


def test_text_board_filters_by_task_and_handles_empty() -> None:
    records = [rec("r1"), rec("r2", task="model_equality")]
    only = render_text(records, "model_equality")
    assert "r2" in only and "r1" not in only
    assert render_text([], None) == "no run in the registry yet."
    assert "for task 'wmloc'" in render_text(records, "wmloc")


def test_cells_formatting() -> None:
    assert format_duration(None) == "-" and format_duration(5) == "5s"
    assert format_duration(3725) == "1h02m" and format_duration(60) == "1m00s"
    assert format_number(0.123456789) == "0.1235" and format_number(1234567.0) == "1.235e+06"
    assert metrics_text(rec("r")) == "-"
    many = rec("r", primary_metric="p", metrics={"a": 1.0, "p": 2.0, "b": 3.0, "c": 4.0, "d": 5.0})
    assert metrics_text(many).startswith("p=2 a=1 b=3") and metrics_text(many).endswith("(+2)")
    assert (
        public_text(rec("r")) == "-"
        and public_text(rec("r", public_submitted_at="t")) == "rejected"
    )
    assert sha_text(rec("r", sha="unknown")) == "unknown"


def test_html_is_self_contained_and_escapes_every_value() -> None:
    evil = rec("r<script>alert(1)</script>", idea='<img src=x onerror="alert(1)">', note="n")
    page = render_html([evil], None)
    assert "<script" not in page and "onerror=" not in page.replace("onerror=&quot;", "")
    assert "&lt;script&gt;" in page and "&lt;img" in page
    assert page.startswith("<!doctype html>") and "prefers-color-scheme:dark" in page
    assert "http://" not in page and "https://" not in page, "no external resource"
    assert 'name="viewport"' in page


def test_html_shows_cards_and_an_empty_message() -> None:
    page = render_html([rec("r1", primary_metric="kl", primary_value=0.3, public_score=0.6)], None)
    assert "best local (kl): 0.3" in page and "best public (info only): 0.6" in page
    assert "No run in the registry yet." in render_html([], None)


def test_write_board_writes_the_page_under_runs(tmp_path: Path) -> None:
    harness = make_lab(tmp_path)
    harness.ctx.registry.append(rec("r1", metrics={"m": 1.0}))
    text, page = write_board(harness.ctx)
    assert page == harness.repo / "runs" / "board.html" and page.is_file()
    assert "r1" in text and "r1" in page.read_text(encoding="utf-8")
