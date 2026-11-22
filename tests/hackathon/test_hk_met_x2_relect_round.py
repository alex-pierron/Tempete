"""Tests of the round-level paired reading (x2-met-relect)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.research.x2_relect_round import (
    WALD_THRESHOLD_NATS,
    EndpointPaired,
    RhoEstimate,
    build_table,
    in_round,
    rho_of,
    round_keys,
    verdict,
    wald_log_lr,
)


def _row(obs: float, mean_b: float, var_b: float, mean_f: float, n: int = 100) -> list[float]:
    return [obs, mean_b, var_b, mean_f, var_b, float(n)]


def test_rho_is_zero_at_base_and_one_at_simulated_fp8() -> None:
    at_base = rho_of(np.array([_row(-10.0, -10.0, 16.0, 10.0)]))
    at_fp8 = rho_of(np.array([_row(10.0, -10.0, 16.0, 10.0)]))
    assert at_base.rho == pytest.approx(0.0)
    assert at_fp8.rho == pytest.approx(1.0)
    assert at_base.rho_se == pytest.approx(4.0 / 20.0)


def test_rho_sums_rows_before_dividing() -> None:
    stats = np.array([_row(0.0, -1.0, 1.0, 1.0), _row(-9.0, -9.0, 9.0, 9.0)])
    estimate = rho_of(stats)
    assert estimate.gap == pytest.approx(20.0)
    assert estimate.rho == pytest.approx(1.0 / 20.0)
    assert estimate.rows == 2 and estimate.n_tokens == 200


def test_empty_subset_is_undecided_not_an_error() -> None:
    estimate = rho_of(np.zeros((0, 6)))
    assert math.isnan(estimate.rho)
    assert verdict(wald_log_lr(estimate, 0.0, 0.05)) == "undecided"


def test_wald_is_symmetric_around_the_midpoint() -> None:
    def make(rho: float) -> RhoEstimate:
        return RhoEstimate(1, 1, 1.0, rho, 0.1)

    assert wald_log_lr(make(0.225), 0.0, 0.0, shift=0.45) == pytest.approx(0.0)
    assert wald_log_lr(make(0.45), 0.0, 0.0, shift=0.45) == pytest.approx(-wald_log_lr(make(0.0), 0.0, 0.0, shift=0.45))
    assert verdict(WALD_THRESHOLD_NATS + 0.01) == "fp8"
    assert verdict(-WALD_THRESHOLD_NATS - 0.01) == "not_fp8"
    assert verdict(0.0) == "undecided"


def test_round_keys_wildcard_and_explicit(tmp_path: Path) -> None:
    path = tmp_path / "round.json"
    path.write_text(json.dumps({"prompts": [
        {"id": "Q1", "prompt": "x", "endpoints": None},
        {"id": "XC18", "prompt": "y", "endpoints": ["ep-01"]},
    ]}), encoding="utf-8")
    keys = round_keys(path)
    assert in_round(keys, "ep-33", "Q1")
    assert in_round(keys, "ep-01", "XC18")
    assert not in_round(keys, "ep-02", "XC18")


def test_build_table_splits_old_and_new_rows() -> None:
    stats = np.array([_row(-10.0, -10.0, 4.0, 10.0), _row(10.0, -10.0, 4.0, 10.0)])
    paired = {"ep-01": EndpointPaired("fam", stats, np.array(["old-q", "new-q"]))}
    table = build_table(paired, {("ep-01", "new-q")}, null_spread=0.0)
    row = table[0]
    assert float(row["rho_old"]) == pytest.approx(0.0)
    assert float(row["rho_new"]) == pytest.approx(1.0)
    assert float(row["rho_all"]) == pytest.approx(0.5)
    assert row["rows_new"] == "1"
