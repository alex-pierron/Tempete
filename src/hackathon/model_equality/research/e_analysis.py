"""Temperature, calibration and fp8 likelihood-ratio statistics on top of `e_loglik` containers.

Three questions, all answered with the LOCAL copy of the identified base model:

1. Which sampling temperature explains the replies of an endpoint? (`temperature_profile`). The
   temperature is a decoding setting, not part of the model identity, so it is fitted and removed
   before any model difference is read.
2. How far is the endpoint from the model's own samples once the temperature is accounted for?
   (`calibration_z` at the fitted temperature, per-row residuals.)
3. Does a simulated fp8 copy of the base explain the replies better than the base?
   (`llr_summary`: mean and spread of the per-token log-likelihood ratio, and the number of
   tokens needed to decide at a given error rate.)

Pure numpy on arrays already computed on the GPU; no model is loaded here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_loglik import RowScores

FloatArray = npt.NDArray[np.float64]
UNIT_TEMPERATURE = 1.0
Z_FIVE_PERCENT = 1.645
Z_ONE_PERCENT = 2.326
MIN_TOKENS_FOR_STATISTICS = 2


class AnalysisError(HackathonError):
    """The container lacks the arrays a statistic needs (for example the tempered statistics)."""


@dataclass(frozen=True)
class TemperatureProfile:
    """Log-likelihood and calibration of a set of rows as a function of the temperature.

    Attributes:
        temperatures: Sorted temperatures, `1.0` included.
        total_logp: Summed token log-probabilities at each temperature.
        total_ent: Summed predictive entropies at each temperature.
        total_var: Summed predictive variances of log p at each temperature.
        z: Calibration statistic `(sum logp + sum ent) / sqrt(sum var)` at each temperature.
        mle_temperature: Temperature maximising the likelihood (quadratic fit in log T).
        mle_at_boundary: True when the maximum sits on the edge of the grid (estimate is a bound).
        zero_crossing_temperature: Temperature at which `z` crosses 0, or None without a crossing.
        ll_gain_nats: Likelihood gain of the grid maximum over temperature 1, in nats.
        n_tokens: Number of tokens in the selected rows.
    """

    temperatures: tuple[float, ...]
    total_logp: FloatArray
    total_ent: FloatArray
    total_var: FloatArray
    z: FloatArray
    mle_temperature: float
    mle_at_boundary: bool
    zero_crossing_temperature: float | None
    ll_gain_nats: float
    n_tokens: int


def token_indices(scores: RowScores, rows: Sequence[int]) -> npt.NDArray[np.int64]:
    """Positions in the flat per-token arrays of the tokens of the given rows."""
    offsets = scores.offsets()
    if not rows:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(offsets[row], offsets[row + 1]) for row in rows])


def calibration_z_from_sums(total_logp: float, total_ent: float, total_var: float) -> float:
    """`(sum logp + sum ent) / sqrt(sum var)`, 0 when there is no variance."""
    if total_var <= 0.0:
        return 0.0
    return (total_logp + total_ent) / math.sqrt(total_var)


def _quadratic_peak_log_temperature(log_t: FloatArray, values: FloatArray) -> tuple[float, bool]:
    """Peak of a quadratic through the best grid point and its neighbours (in log T).

    Returns `(log T, at_boundary)`. On the edge of the grid, or for a non-concave fit, the best
    grid point itself is returned.
    """
    best = int(np.argmax(values))
    if best in (0, len(values) - 1):
        return float(log_t[best]), True
    xs, ys = log_t[best - 1 : best + 2], values[best - 1 : best + 2]
    a, b, _ = np.polyfit(xs, ys, 2)
    if a >= 0.0:
        return float(log_t[best]), False
    peak = float(-b / (2.0 * a))
    return float(np.clip(peak, xs[0], xs[-1])), False


def _zero_crossing(log_t: FloatArray, z: FloatArray) -> float | None:
    """Temperature where `z` changes sign (linear in log T), or None."""
    for left in range(len(z) - 1):
        z0, z1 = z[left], z[left + 1]
        if z0 == 0.0:
            return math.exp(float(log_t[left]))
        if z0 * z1 < 0.0:
            weight = z0 / (z0 - z1)
            return math.exp(float(log_t[left] + weight * (log_t[left + 1] - log_t[left])))
    return None


def temperature_profile_from_sums(
    temperatures: Sequence[float], total_logp: FloatArray, total_ent: FloatArray,
    total_var: FloatArray, n_tokens: int,
) -> TemperatureProfile:
    """Temperature profile from per-temperature sums (any order of temperatures, `1.0` included).

    Raises:
        AnalysisError: Temperature 1.0 is missing or the arrays have different lengths.
    """
    grid = np.asarray(temperatures, dtype=np.float64)
    if not (len(grid) == len(total_logp) == len(total_ent) == len(total_var)):
        raise AnalysisError("temperature grid and sums must have the same length")
    if not np.any(np.isclose(grid, UNIT_TEMPERATURE)):
        raise AnalysisError("temperature 1.0 must be part of the grid")
    order = np.argsort(grid)
    grid = grid[order]
    logp, ent, var = (np.asarray(a, dtype=np.float64)[order] for a in (total_logp, total_ent, total_var))
    z = np.array([calibration_z_from_sums(a, b, c) for a, b, c in zip(logp, ent, var, strict=True)])
    log_t = np.log(grid)
    peak_log_t, at_boundary = _quadratic_peak_log_temperature(log_t, logp)
    unit_index = int(np.argmin(np.abs(grid - UNIT_TEMPERATURE)))
    return TemperatureProfile(
        temperatures=tuple(float(t) for t in grid),
        total_logp=logp, total_ent=ent, total_var=var, z=z,
        mle_temperature=math.exp(peak_log_t), mle_at_boundary=at_boundary,
        zero_crossing_temperature=_zero_crossing(log_t, z),
        ll_gain_nats=float(logp.max() - logp[unit_index]),
        n_tokens=n_tokens,
    )


def temperature_profile(scores: RowScores, rows: Sequence[int]) -> TemperatureProfile:
    """Likelihood and calibration of `rows` over the temperature grid (plus temperature 1).

    Raises:
        AnalysisError: The container has no tempered statistics.
    """
    if scores.t_logp is None or scores.t_ent is None or scores.t_var is None:
        raise AnalysisError("container has no tempered statistics (matrix stage, grid missing)")
    index = token_indices(scores, rows)
    temperatures = [*scores.temperatures, UNIT_TEMPERATURE]
    logp = np.vstack([scores.t_logp[:, index], scores.logp[index][None, :]]).astype(np.float64)
    ent = np.vstack([scores.t_ent[:, index], scores.ent[index][None, :]]).astype(np.float64)
    var = np.vstack([scores.t_var[:, index], scores.var[index][None, :]]).astype(np.float64)
    return temperature_profile_from_sums(
        temperatures, logp.sum(axis=1), ent.sum(axis=1), var.sum(axis=1), int(index.size)
    )


@dataclass(frozen=True)
class LlrSummary:
    """Per-token log-likelihood ratio of a model variant against the base on some rows.

    Attributes:
        n_tokens: Tokens in the rows.
        mean_per_token: `sum(delta) / n_tokens`, the average nats gained per token.
        sd_per_token: Per-token standard deviation of the per-row residual `delta - n * mean`.
        standard_error: Standard error of `mean_per_token`, from the per-row spread.
        t_statistic: `mean_per_token / standard_error` (0 without spread).
    """

    n_tokens: int
    mean_per_token: float
    sd_per_token: float
    standard_error: float
    t_statistic: float


def llr_summary(delta_per_row: FloatArray, tokens_per_row: npt.NDArray[np.int64]) -> LlrSummary:
    """Summarise per-row log-likelihood differences `delta` over rows with `tokens` tokens each.

    The spread is estimated from the rows (not from the tokens), because the tokens of one reply
    are not independent.
    """
    keep = tokens_per_row > 0
    delta, tokens = delta_per_row[keep], tokens_per_row[keep].astype(np.float64)
    total_tokens = float(tokens.sum())
    if len(delta) < MIN_TOKENS_FOR_STATISTICS or total_tokens <= 0.0:
        return LlrSummary(int(total_tokens), 0.0, 0.0, 0.0, 0.0)
    mean = float(delta.sum() / total_tokens)
    residual = delta - tokens * mean
    sd_per_token = float(math.sqrt(float(np.sum(residual**2)) / total_tokens))
    row_variance = float(np.sum(residual**2) / (len(delta) - 1))
    standard_error = float(math.sqrt(len(delta) * row_variance) / total_tokens)
    t_stat = 0.0 if standard_error == 0.0 else mean / standard_error
    return LlrSummary(int(total_tokens), mean, sd_per_token, standard_error, t_stat)


def tokens_needed(mean_per_token: float, sd_per_token: float, z: float = Z_ONE_PERCENT) -> float | None:
    """Tokens for a likelihood-ratio test to separate the two hypotheses at normal quantile `z`.

    Under the base the summed ratio has mean `-N mu` and under the variant `+N mu` (the two
    divergences are close for a small perturbation), spread `sqrt(N) sigma` in both: the test at
    threshold 0 errs with probability `Phi(-sqrt(N) mu / sigma)`, so `N = (z sigma / mu)^2`.
    Returns None when the effect has the wrong sign or is zero.
    """
    if mean_per_token <= 0.0:
        return None
    return (z * sd_per_token / mean_per_token) ** 2


@dataclass(frozen=True)
class PairedComparison:
    """Paired comparison of two endpoints on the rows they share.

    Attributes:
        n_rows: Number of shared `(prompt id, repeat)` cells.
        mean_difference: Mean over shared cells of `mean_a - mean_b` per-token values.
        standard_error: Standard error of that mean.
        t_statistic: `mean_difference / standard_error` (0 without spread).
    """

    n_rows: int
    mean_difference: float
    standard_error: float
    t_statistic: float


def paired_per_token_comparison(
    cells_a: Sequence[tuple[str, int]], values_a: FloatArray, tokens_a: npt.NDArray[np.int64],
    cells_b: Sequence[tuple[str, int]], values_b: FloatArray, tokens_b: npt.NDArray[np.int64],
) -> PairedComparison:
    """Paired t on per-token row values (`value / tokens`) of two endpoints over shared cells.

    Rows without tokens in either endpoint are skipped; cells are matched on `(id, repeat)`.
    """
    index_b = {cell: position for position, cell in enumerate(cells_b)}
    differences: list[float] = []
    for position_a, cell in enumerate(cells_a):
        position_b = index_b.get(cell)
        if position_b is None or tokens_a[position_a] == 0 or tokens_b[position_b] == 0:
            continue
        differences.append(
            values_a[position_a] / tokens_a[position_a] - values_b[position_b] / tokens_b[position_b]
        )
    n_rows = len(differences)
    if n_rows < MIN_TOKENS_FOR_STATISTICS:
        return PairedComparison(n_rows, 0.0, 0.0, 0.0)
    array = np.array(differences, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(n_rows))
    t_stat = 0.0 if standard_error == 0.0 else mean / standard_error
    return PairedComparison(n_rows, mean, standard_error, t_stat)
