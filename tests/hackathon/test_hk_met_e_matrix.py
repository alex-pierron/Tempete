"""Endpoint x candidate matrix and identification of team E (synthetic score containers)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research import e_matrix as mx
from hackathon.model_equality.research.e_loglik import RowScores, save_scores
from hackathon.model_equality.store import Answer

TOKENS_PER_ROW = 4
ROWS_PER_ENDPOINT = 6
ENDPOINTS = ("ep-a", "ep-b")


def _answers() -> list[Answer]:
    return [
        Answer(endpoint, f"p{row}", 0, "prompt", "x" * 10)
        for endpoint in ENDPOINTS for row in range(ROWS_PER_ENDPOINT)
    ]


def _scores(per_row_logp: list[float], n_tokens: int = TOKENS_PER_ROW) -> RowScores:
    """Container with `per_row_logp[i]` spread evenly over the tokens of row i."""
    logp = np.concatenate([np.full(n_tokens, value / n_tokens) for value in per_row_logp])
    flat = np.zeros_like(logp)
    return RowScores(
        n_tokens=np.full(len(per_row_logp), n_tokens, dtype=np.int32),
        logp=logp.astype(np.float32), ent=flat.astype(np.float32), var=(flat + 1).astype(np.float32),
    )


def _header(answers: list[Answer]) -> dict[str, object]:
    return {
        "endpoint": [a.endpoint for a in answers],
        "prompt_id": [a.prompt_id for a in answers],
        "repeat": [a.repeat for a in answers],
        "meta": {},
    }


def _containers(answers: list[Answer], a_logp: list[float], b_logp: list[float]) -> dict[
    str, tuple[dict[str, object], dict[tuple[str, str], RowScores]]
]:
    header = _header(answers)
    return {
        "cand-a": (header, {("default", "base"): _scores(a_logp), ("nosys", "base"): _scores([v - 5 for v in a_logp])}),
        "cand-b": (header, {("default", "base"): _scores(b_logp)}),
    }


def test_row_sums_handles_empty_rows() -> None:
    scores = RowScores(
        n_tokens=np.array([2, 0, 3], dtype=np.int32),
        logp=np.array([1, 2, 10, 20, 30], dtype=np.float32),
        ent=np.zeros(5, dtype=np.float32), var=np.ones(5, dtype=np.float32),
    )
    assert mx.row_sums(scores, "logp").tolist() == [3.0, 0.0, 60.0]


def test_build_cells_takes_the_better_template_and_counts_chars() -> None:
    answers = _answers()
    # cand-a is better on ep-a (-2/row) than on ep-b (-9/row): the default template wins both times.
    a_logp = [-2.0] * ROWS_PER_ENDPOINT + [-9.0] * ROWS_PER_ENDPOINT
    b_logp = [-6.0] * ROWS_PER_ENDPOINT + [-4.0] * ROWS_PER_ENDPOINT
    cells = mx.build_cells(_containers(answers, a_logp, b_logp), answers, ENDPOINTS)
    cell = cells["ep-a"]["cand-a"]
    assert cell.template == "default"
    assert cell.total_ll == pytest.approx(-12.0)
    assert cell.per_char == pytest.approx(-12.0 / (10 * ROWS_PER_ENDPOINT))
    assert cell.n_tokens == TOKENS_PER_ROW * ROWS_PER_ENDPOINT
    assert len(cell.row_ll) == ROWS_PER_ENDPOINT


def test_identify_picks_the_likelier_candidate_with_margin_and_confidence() -> None:
    answers = _answers()
    a_logp = [-2.0] * ROWS_PER_ENDPOINT + [-9.0] * ROWS_PER_ENDPOINT
    b_logp = [-6.0] * ROWS_PER_ENDPOINT + [-4.0] * ROWS_PER_ENDPOINT
    cells = mx.build_cells(_containers(answers, a_logp, b_logp), answers, ENDPOINTS)
    families = {"cand-a": "fam-1", "cand-b": "fam-2"}
    first = mx.identify("ep-a", cells["ep-a"], families)
    assert first.best == "cand-a" and first.runner_up == "cand-b"
    assert first.margin_nats == pytest.approx(24.0)
    assert first.win_share == 1.0 and first.bootstrap_win == 1.0
    assert first.confidence == mx.CONFIDENCE_HIGH
    assert first.best_other_family[0] == "cand-b"
    assert first.best_other_family[1] == pytest.approx(24.0)
    second = mx.identify("ep-b", cells["ep-b"], families)
    assert second.best == "cand-b"


def test_identify_is_low_confidence_when_one_row_carries_the_gap() -> None:
    answers = _answers()
    a_logp = [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0] + [0.0] * ROWS_PER_ENDPOINT
    b_logp = [-2.5, -2.5, -2.5, -2.5, -2.5, -60.0] + [0.0] * ROWS_PER_ENDPOINT
    cells = mx.build_cells(_containers(answers, a_logp, b_logp), answers, ENDPOINTS)
    result = mx.identify("ep-a", cells["ep-a"], {"cand-a": "f", "cand-b": "f"})
    assert result.best == "cand-a"
    assert result.margin_nats > mx.CONFIDENCE_HIGH_NATS
    assert result.confidence != mx.CONFIDENCE_HIGH
    assert result.best_other_family[0] == ""


def test_identify_needs_two_candidates_and_same_rows() -> None:
    answers = _answers()
    cells = mx.build_cells(
        _containers(answers, [-1.0] * 12, [-2.0] * 12), answers, ENDPOINTS
    )
    with pytest.raises(mx.MatrixError, match="at least two"):
        mx.identify("ep-a", {"cand-a": cells["ep-a"]["cand-a"]}, {"cand-a": "f"})
    shorter = mx.Cell(-1.0, -0.1, 3, "default", np.zeros(3))
    with pytest.raises(mx.MatrixError, match="different rows"):
        mx.identify("ep-a", {"cand-a": cells["ep-a"]["cand-a"], "cand-b": shorter}, {"cand-a": "f", "cand-b": "g"})


@pytest.mark.parametrize(
    ("margin", "boot", "expected"),
    [(25.0, 1.0, "high"), (25.0, 0.95, "medium"), (8.0, 0.95, "medium"), (8.0, 0.5, "low"), (2.0, 1.0, "low")],
)
def test_classify_confidence(margin: float, boot: float, expected: str) -> None:
    assert mx.classify_confidence(margin, boot) == expected


def test_load_containers_and_csv_round_trip(tmp_path: Path) -> None:
    answers = _answers()
    containers = _containers(answers, [-2.0] * 12, [-3.0] * 12)
    for candidate, (_, per_variant) in containers.items():
        save_scores(tmp_path / f"ll_matrix_{candidate}.npz", answers, per_variant, {"candidate": candidate})
    loaded = mx.load_containers(tmp_path, "matrix")
    assert sorted(loaded) == ["cand-a", "cand-b"]
    cells = mx.build_cells(loaded, answers, ENDPOINTS)
    out = tmp_path / "matrix.csv"
    mx.write_matrix_csv(out, cells, ["cand-a", "cand-b"])
    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert rows[0][0] == "endpoint" and rows[1][0] == "ep-a"
    assert float(rows[1][1]) == pytest.approx(-12.0)
