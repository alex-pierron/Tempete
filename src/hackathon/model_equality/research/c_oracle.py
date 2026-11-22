"""The public-score oracle as a function of the 3 x 3 truth-by-submission cross-tabulation.

Why a cross-tabulation is enough. Both MCCs are functions of pair counts only:

* MCC_family counts pairs by (truth same-family?, submitted same-family?);
* MCC_model counts, among truth-same-family pairs, (truth same-model?, submitted same-model?).

Both are sums of the 9 cells `cell = 3 * truth_state + submitted_state` (states of `c_pairs`).
The public leaderboard scores a random ~30 % of the pairs. If every pair is public independently
with probability `p`, the public count of each cell is `Binomial(n_cell, p)` and the cells are
independent, so the whole distribution of the public score of a (truth, submission) couple is
available in closed form from 9 integers, with no need to materialise the 528-pair mask. That is
what makes the one-step value of information of hundreds of probes over thousands of hypotheses
cheap.

Conventions: a zero MCC denominator gives 0 (as `score_practice.py`; the server's convention for
that degenerate case is not published, see `scoring.py`). The server score may be quantised; the
step is a parameter (`None` = exact).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_pairs import STATE_COUNT, StateArray

FAMILY_WEIGHT = 0.1
MODEL_WEIGHT = 0.9
CELL_COUNT = STATE_COUNT * STATE_COUNT
DEFAULT_PUBLIC_FRACTION = 0.3
"""Share of the pairs on the public board ("~30 %" in the team's leaderboard snapshot)."""
NORMAL_APPROXIMATION_MIN = 40
"""Cells of at least this many pairs are thinned with the normal approximation."""

FloatArray = npt.NDArray[np.float64]


def cross_tabulate(truth_states: StateArray, submitted_states: StateArray) -> FloatArray:
    """Cell counts `[..., 9]` of truth states `[..., pairs]` against ONE submission `[pairs]`.

    Cell index = `3 * truth_state + submitted_state`.
    """
    truth = truth_states.astype(np.float32)
    cells = np.empty((*truth_states.shape[:-1], CELL_COUNT), dtype=np.float64)
    for submitted in range(STATE_COUNT):
        selected = (submitted_states == submitted).astype(np.float32)
        for state in range(STATE_COUNT):
            cells[..., STATE_COUNT * state + submitted] = (
                (truth == state).astype(np.float32) @ selected
            )
    return cells


def cross_tabulate_many(truth_states: StateArray, submissions: StateArray) -> FloatArray:
    """Cell counts `[truths, submissions, 9]` for `[truths, pairs]` against `[submissions, pairs]`.

    One matrix product per cell, so thousands of hypotheses against hundreds of submissions
    stay in BLAS.
    """
    truth_masks = [(truth_states == s).astype(np.float32) for s in range(STATE_COUNT)]
    sub_masks = [(submissions == s).astype(np.float32).T for s in range(STATE_COUNT)]
    cells = np.empty((truth_states.shape[0], submissions.shape[0], CELL_COUNT), dtype=np.float64)
    for state in range(STATE_COUNT):
        for submitted in range(STATE_COUNT):
            cells[:, :, STATE_COUNT * state + submitted] = truth_masks[state] @ sub_masks[submitted]
    return cells


def mcc_from_counts(
    tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray
) -> FloatArray:
    """Matthews correlation, element-wise; 0 where a marginal is empty (zero denominator)."""
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    numerator = tp * tn - fp * fn
    safe = np.where(denominator > 0, denominator, 1.0)
    return np.asarray(np.where(denominator > 0, numerator / safe, 0.0), dtype=np.float64)


def family_model_mcc(cells: FloatArray) -> tuple[FloatArray, FloatArray]:
    """`(MCC_family, MCC_model)` (unclipped) from cell counts `[..., 9]`."""
    grid = cells.reshape(*cells.shape[:-1], STATE_COUNT, STATE_COUNT)
    # truth axis then submitted axis; index 0 = different family, >= 1 = same family
    fam_tp = grid[..., 1:, 1:].sum(axis=(-1, -2))
    fam_fn = grid[..., 1:, 0].sum(axis=-1)
    fam_fp = grid[..., 0, 1:].sum(axis=-1)
    fam_tn = grid[..., 0, 0]
    mod_tp = grid[..., 2, 2]
    mod_fp = grid[..., 1, 2]
    mod_fn = grid[..., 2, :2].sum(axis=-1)
    mod_tn = grid[..., 1, :2].sum(axis=-1)
    return (
        mcc_from_counts(fam_tp, fam_fp, fam_fn, fam_tn),
        mcc_from_counts(mod_tp, mod_fp, mod_fn, mod_tn),
    )


def official_score_from_cells(cells: FloatArray) -> FloatArray:
    """`0.1 * max(0, MCC_family) + 0.9 * max(0, MCC_model)` from cell counts `[..., 9]`."""
    family, model = family_model_mcc(cells)
    return FAMILY_WEIGHT * np.maximum(family, 0.0) + MODEL_WEIGHT * np.maximum(model, 0.0)


def quantize(score: FloatArray, step: float | None) -> FloatArray:
    """Round scores to a multiple of `step` (`None` or 0 = unchanged)."""
    if not step:
        return score
    return np.asarray(np.round(score / step) * step, dtype=np.float64)


def thin_cells(
    cells: FloatArray, keep_probability: float, draws: int, rng: np.random.Generator
) -> FloatArray:
    """Public cell counts `[..., draws, 9]`: each pair kept independently with `keep_probability`.

    Exact binomial for small cells; for cells of at least `NORMAL_APPROXIMATION_MIN` pairs the
    normal approximation (rounded, clipped to `[0, n]`), which is 10x faster and accurate there.
    """
    integer = np.rint(cells).astype(np.int64)
    expanded = np.broadcast_to(integer[..., None, :], (*integer.shape[:-1], draws, CELL_COUNT))
    spread = keep_probability * (1.0 - keep_probability)
    out = np.clip(
        np.rint(expanded * keep_probability
                + np.sqrt(expanded * spread) * rng.standard_normal(expanded.shape)),
        0, expanded,
    )
    small = (expanded > 0) & (expanded < NORMAL_APPROXIMATION_MIN)
    if small.any():
        out[small] = rng.binomial(expanded[small], keep_probability)
    return np.asarray(out, dtype=np.float64)


def public_score_draws(
    cells: FloatArray,
    draws: int,
    rng: np.random.Generator,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION,
    step: float | None = None,
) -> FloatArray:
    """Simulated public scores `[..., draws]` of cell counts `[..., 9]` (random public subset)."""
    return quantize(official_score_from_cells(thin_cells(cells, public_fraction, draws, rng)), step)


def score_bins(step: float | None, resolution: float = 0.01) -> FloatArray:
    """Right-open bin edges of the observable score: `[0, 1]` cut every `step` (or `resolution`)."""
    width = step if step else resolution
    return np.arange(-width / 2.0, 1.0 + width, width)


def full_score(truth_states: StateArray, submitted_states: StateArray) -> FloatArray:
    """Exact score(s) over ALL pairs of truth states `[..., pairs]` against one submission."""
    return official_score_from_cells(cross_tabulate(truth_states, submitted_states))
