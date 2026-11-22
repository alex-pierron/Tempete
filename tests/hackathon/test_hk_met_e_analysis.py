"""Temperature profile and likelihood-ratio statistics of team E (synthetic arrays)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hackathon.model_equality.research import e_analysis as an
from hackathon.model_equality.research.e_loglik import RowScores

GRID = (0.5, 0.7, 1.5, 2.0)


def _scores_for_sampling_temperature(
    peak_t: float, cross_t: float | None = None, n_rows: int = 6, tokens: int = 50
) -> RowScores:
    """Synthetic container: log-likelihood peaks at `peak_t`, z crosses 0 at `cross_t`.

    logp_T = -0.5 - 0.25 (log T - log peak_t)^2 per token, and the entropy is built so that
    logp_T + ent_T = -0.2 (log T - log cross_t): z changes sign at `cross_t` (default `peak_t`).
    """
    cross_t = peak_t if cross_t is None else cross_t
    temperatures = np.array([*GRID, 1.0])
    n_total = n_rows * tokens

    def stats_at(temperature: float) -> tuple[np.ndarray, np.ndarray]:
        peak_shape = -0.25 * (math.log(temperature) - math.log(peak_t)) ** 2
        logp = -0.5 + peak_shape
        numerator = -0.2 * (math.log(temperature) - math.log(cross_t))
        return (
            np.full(n_total, logp, dtype=np.float32),
            np.full(n_total, numerator - logp, dtype=np.float32),
        )

    stacked_logp = np.vstack([stats_at(t)[0] for t in temperatures])
    stacked_ent = np.vstack([stats_at(t)[1] for t in temperatures])
    stacked_var = np.ones_like(stacked_logp)
    unit = list(temperatures).index(1.0)
    return RowScores(
        n_tokens=np.full(n_rows, tokens, dtype=np.int32),
        logp=stacked_logp[unit], ent=stacked_ent[unit], var=stacked_var[unit],
        temperatures=GRID, t_logp=np.delete(stacked_logp, unit, axis=0),
        t_ent=np.delete(stacked_ent, unit, axis=0), t_var=np.delete(stacked_var, unit, axis=0),
    )


@pytest.mark.parametrize("true_t", [0.7, 1.0, 1.5])
def test_temperature_profile_recovers_the_sampling_temperature(true_t: float) -> None:
    scores = _scores_for_sampling_temperature(true_t)
    profile = an.temperature_profile(scores, list(range(6)))
    assert profile.mle_temperature == pytest.approx(true_t, rel=0.05)
    assert profile.zero_crossing_temperature is not None
    assert profile.zero_crossing_temperature == pytest.approx(true_t, rel=0.05)
    assert profile.temperatures == tuple(sorted((*GRID, 1.0)))
    assert profile.n_tokens == 6 * 50
    assert profile.ll_gain_nats >= 0.0


def test_mle_and_zero_crossing_can_disagree() -> None:
    """A model mismatch shows as a gap between the two temperature estimates."""
    profile = an.temperature_profile(_scores_for_sampling_temperature(0.7, cross_t=1.5), [0, 1, 2])
    assert profile.mle_temperature == pytest.approx(0.7, rel=0.05)
    assert profile.zero_crossing_temperature == pytest.approx(1.5, rel=0.05)


def test_temperature_profile_boundary_is_flagged() -> None:
    scores = _scores_for_sampling_temperature(0.2)
    profile = an.temperature_profile(scores, [0, 1])
    assert profile.mle_at_boundary
    assert profile.mle_temperature == pytest.approx(0.5)


def test_temperature_profile_requires_tempered_arrays() -> None:
    plain = RowScores(
        n_tokens=np.array([2], dtype=np.int32), logp=np.zeros(2, dtype=np.float32),
        ent=np.zeros(2, dtype=np.float32), var=np.ones(2, dtype=np.float32),
    )
    with pytest.raises(an.AnalysisError):
        an.temperature_profile(plain, [0])


def test_token_indices_select_the_right_rows() -> None:
    scores = RowScores(
        n_tokens=np.array([2, 0, 3], dtype=np.int32), logp=np.zeros(5, dtype=np.float32),
        ent=np.zeros(5, dtype=np.float32), var=np.zeros(5, dtype=np.float32),
    )
    assert an.token_indices(scores, [2, 0]).tolist() == [2, 3, 4, 0, 1]
    assert an.token_indices(scores, []).size == 0


def test_llr_summary_matches_hand_computation() -> None:
    delta = np.array([0.2, 0.1, 0.3, 0.0], dtype=np.float64)
    tokens = np.array([10, 10, 10, 10], dtype=np.int64)
    summary = an.llr_summary(delta, tokens)
    assert summary.n_tokens == 40
    assert summary.mean_per_token == pytest.approx(0.15 / 10)
    residual = delta - tokens * summary.mean_per_token
    assert summary.sd_per_token == pytest.approx(math.sqrt(float(np.sum(residual**2)) / 40))
    assert summary.t_statistic == pytest.approx(summary.mean_per_token / summary.standard_error)


def test_llr_summary_degenerate_inputs() -> None:
    assert an.llr_summary(np.array([1.0]), np.array([5])).t_statistic == 0.0
    assert an.llr_summary(np.array([1.0, 2.0]), np.array([0, 0])).n_tokens == 0
    flat = an.llr_summary(np.array([1.0, 1.0, 1.0]), np.array([10, 10, 10]))
    assert flat.t_statistic == 0.0 or math.isfinite(flat.t_statistic)


def test_tokens_needed_formula_and_sign() -> None:
    assert an.tokens_needed(0.01, 0.1, z=2.0) == pytest.approx((2.0 * 0.1 / 0.01) ** 2)
    assert an.tokens_needed(0.0, 0.1) is None
    assert an.tokens_needed(-0.02, 0.1) is None
    # an effect twice as large needs four times fewer tokens
    ratio = an.tokens_needed(0.01, 0.1, z=2.0) / an.tokens_needed(0.02, 0.1, z=2.0)  # type: ignore[operator]
    assert ratio == pytest.approx(4.0)


def test_paired_comparison_matches_hand_computation_and_skips_unshared_rows() -> None:
    cells_a = [("p1", 0), ("p2", 0), ("p3", 0), ("only-a", 0)]
    cells_b = [("p3", 0), ("p1", 0), ("p2", 0), ("only-b", 0)]
    values_a = np.array([-10.0, -20.0, -30.0, -99.0])
    tokens_a = np.array([10, 10, 10, 10], dtype=np.int64)
    values_b = np.array([-31.0, -11.0, -22.0, -99.0])
    tokens_b = np.array([10, 10, 10, 10], dtype=np.int64)
    result = an.paired_per_token_comparison(cells_a, values_a, tokens_a, cells_b, values_b, tokens_b)
    # per-token differences: p1 0.1, p2 0.2, p3 0.1
    assert result.n_rows == 3
    assert result.mean_difference == pytest.approx((0.1 + 0.2 + 0.1) / 3)
    spread = np.std([0.1, 0.2, 0.1], ddof=1)
    assert result.standard_error == pytest.approx(spread / math.sqrt(3))
    assert result.t_statistic == pytest.approx(result.mean_difference / result.standard_error)


def test_paired_comparison_with_too_few_shared_rows() -> None:
    result = an.paired_per_token_comparison(
        [("a", 0)], np.array([-1.0]), np.array([4], dtype=np.int64),
        [("b", 0)], np.array([-1.0]), np.array([4], dtype=np.int64),
    )
    assert result.n_rows == 0 and result.t_statistic == 0.0
