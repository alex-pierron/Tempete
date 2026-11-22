"""Small-sample calibration of team E on a synthetic row table."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.research import e_calib as cal
from hackathon.model_equality.research.e_loglik import RowScores

GRID = (0.5, 1.5)
TOKENS = 20


def _scores(n_rows: int) -> RowScores:
    """Container where every token has logp -1, ent 1, var 1 at every temperature."""
    total = n_rows * TOKENS
    full = np.ones((2, total), dtype=np.float32)
    return RowScores(
        n_tokens=np.full(n_rows, TOKENS, dtype=np.int32), logp=-np.ones(total, dtype=np.float32),
        ent=np.ones(total, dtype=np.float32), var=np.ones(total, dtype=np.float32),
        temperatures=GRID, t_logp=-full, t_ent=full, t_var=full,
    )


def _table(endpoints: list[str], rows_each: int) -> cal.RowFeatureTable:
    keys = [(e, f"p{i}", 0) for e in endpoints for i in range(rows_each)]
    llr = {"w_tensor": np.arange(len(keys), dtype=np.float64) * 0.0 + 0.1}
    return cal.build_row_table(keys, _scores(len(keys)), llr)


def test_auc_extremes_ties_and_empty() -> None:
    assert cal.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cal.auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert cal.auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert np.isnan(cal.auc(np.array([]), np.array([1.0])))
    rng = np.random.default_rng(0)
    assert cal.auc(rng.normal(size=2000), rng.normal(size=2000)) == pytest.approx(0.5, abs=0.04)


def test_build_row_table_sums_per_row_and_sorts_temperatures() -> None:
    table = _table(["ref-01"], 3)
    assert table.temperatures == (0.5, 1.0, 1.5)
    assert table.lp.shape == (3, 3)
    np.testing.assert_allclose(table.lp, -TOKENS)
    np.testing.assert_allclose(table.ent, TOKENS)
    assert table.n_tokens.tolist() == [TOKENS] * 3
    np.testing.assert_allclose(table.llr["w_tensor"], 0.1)


def test_build_row_table_rejects_misaligned_inputs() -> None:
    with pytest.raises(cal.CalibrationError, match="LLR rows"):
        cal.build_row_table([("a", "p", 0)], _scores(1), {"v": np.zeros(2)})
    plain = RowScores(
        n_tokens=np.array([1], dtype=np.int32), logp=np.zeros(1, dtype=np.float32),
        ent=np.zeros(1, dtype=np.float32), var=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(cal.CalibrationError, match="tempered"):
        cal.build_row_table([("a", "p", 0)], plain, {})


def test_features_of_rows_values() -> None:
    table = _table(["ref-01"], 4)
    features = cal.features_of_rows(table, np.array([0, 1], dtype=np.int64))
    assert features[cal.MEAN_LL] == pytest.approx(-1.0)
    assert features["llr_w_tensor"] == pytest.approx(0.1 * 2 / (2 * TOKENS))
    # (sum logp + sum ent) = 0 at every temperature: z = 0 and the zero crossing is flat
    assert features[cal.Z_AT_ONE] == pytest.approx(0.0)


def test_draw_rows_respects_the_token_budget_and_the_endpoint() -> None:
    table = _table(["ref-01", "ref-02"], 10)
    rng = np.random.default_rng(1)
    candidates = np.array([i for i, k in enumerate(table.keys) if k[0] == "ref-02"], dtype=np.int64)
    rows = cal.draw_rows(table, candidates, 5 * TOKENS, rng)
    assert len(rows) == 5 and len(set(rows.tolist())) == 5
    assert all(table.keys[i][0] == "ref-02" for i in rows)
    assert len(cal.draw_rows(table, candidates, 10_000, rng)) == 10  # budget above what exists


def test_draw_features_unknown_endpoint_is_an_error() -> None:
    with pytest.raises(cal.CalibrationError):
        cal.draw_features(_table(["ref-01"], 3), "ref-99", 40, 2, np.random.default_rng(0))


def test_separation_table_covers_every_class_feature_and_budget() -> None:
    endpoints = sorted(cal.CLASS_OF)
    table = _table(endpoints, 12)
    rows = cal.separation_table(table, budgets=(2 * TOKENS, 6 * TOKENS), draws=8)
    features = {r["feature"] for r in rows}
    assert features == {cal.MLE_TEMPERATURE, cal.Z_AT_ONE, cal.MEAN_LL, "llr_w_tensor"}
    assert {r["target"] for r in rows} == set(cal.TARGET_CLASSES)
    assert {r["tokens"] for r in rows} == {2 * TOKENS, 6 * TOKENS}
    assert len(rows) == len(features) * len(cal.TARGET_CLASSES) * 2
