"""Team next: one-flip probes change exactly one endpoint and never a family."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.model_equality.research.n_flip import (
    FlipError,
    isolate,
    main,
    main_model,
    read_submission,
    to_main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
F6 = REPO_ROOT / "reports/phase2/model_equality/team-c/data-3/f6_partition_v1.csv"
F7B = REPO_ROOT / "reports/phase2/model_equality/team-c/data-3/f7b_carve_33.csv"


def _rows() -> list[dict[str, str]]:
    return [
        {"endpoint": "ep-01", "family": "F1", "model": "F1M1"},
        {"endpoint": "ep-02", "family": "F1", "model": "F1M1"},
        {"endpoint": "ep-03", "family": "F1", "model": "F1M2"},
        {"endpoint": "ep-04", "family": "F2", "model": "F2M1"},
        {"endpoint": "ep-05", "family": "F2", "model": "F2M1"},
    ]


def test_isolate_gives_a_new_model_of_the_same_family_and_changes_one_row() -> None:
    rows = _rows()
    out = isolate(rows, "ep-02")
    assert out[1] == {"endpoint": "ep-02", "family": "F1", "model": "F1M3"}
    assert [a for a, b in zip(rows, out, strict=True) if a != b] == [rows[1]]
    assert rows[1]["model"] == "F1M1"  # the base is not modified
    with pytest.raises(FlipError, match="already isolated"):
        isolate(rows, "ep-03")
    with pytest.raises(FlipError, match="unknown endpoint"):
        isolate(rows, "ep-99")


def test_to_main_joins_the_largest_model_of_the_family() -> None:
    rows = _rows()
    assert main_model(rows, "F1") == "F1M1"
    out = to_main(rows, "ep-03")
    assert out[2]["model"] == "F1M1" and out[2]["family"] == "F1"
    with pytest.raises(FlipError, match="already in the main"):
        to_main(rows, "ep-01")
    tie = [*_rows()[:1], {"endpoint": "ep-02", "family": "F1", "model": "F1M2"}]
    with pytest.raises(FlipError, match="no single largest"):
        main_model(tie, "F1")


def test_isolating_ep33_from_f6_gives_the_partition_of_f7b(tmp_path: Path) -> None:
    out = tmp_path / "probe.csv"
    assert main([str(F6), str(out), "--isolate", "ep-33"]) == 0

    def groups(path: Path) -> set[frozenset[str]]:
        by_model: dict[str, set[str]] = {}
        for row in read_submission(path):
            by_model.setdefault(row["model"], set()).add(row["endpoint"])
        return {frozenset(members) for members in by_model.values()}

    assert groups(out) == groups(F7B)
    assert out.read_text(encoding="utf-8").startswith("endpoint,family,model\n")
