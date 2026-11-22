"""Model Equality Testing (Gao, Liang, Guestrin, ICLR 2025): Hamming-kernel MMD two-sample test.

Question answered: do two samplers (a local reference and a black-box API) produce the same
distribution over completions, prompt by prompt?

Statistic (identical to `model_equality_testing.tests.mmd_hamming` of the official code):
kernel `k(y, y') = sum_i 1{y_i = y'_i} / L` between completions padded to length `L` with
`PAD_TOKEN_ID` (two paddings match), zero between different prompts; each block mean is taken
over the same-prompt pairs only, diagonals excluded:
`MMD = mean(K_PP) - 2 mean(K_PQ) + mean(K_QQ)`.

Speed: the Hamming kernel is a sum over positions of "same token" indicators, so the sum of the
kernel over all pairs equals `sum_g count_P[g] * count_Q[g]` where `g = (prompt, position,
token)`. The statistic costs O(N L) per evaluation instead of O(N^2 L), which makes thousands
of permutations cheap. A brute-force implementation in the tests pins the equality.

p-values: permutation (labels shuffled; by default within each prompt, which is exact
conditionally on the prompt design) or simulation under H0 from a pool of reference samples
(Algorithm 1 of the paper). Both use `(1 + #{T_i >= T_obs}) / (B + 1)`, a valid p-value.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

PAD_TOKEN_ID: Final[int] = -1
"""Padding value for short completions; real token ids are non-negative."""

DEFAULT_COMPLETION_LEN: Final[int] = 50
"""Completion length L (tokens) used in the paper's main experiments."""

DEFAULT_N_RESAMPLES: Final[int] = 1000
"""Permutations / simulations per p-value, as in the paper."""

POWER_CURVE_N_RESAMPLES: Final[int] = 199
"""Permutations per trial inside `power_curve` (resolution 0.005, enough for alpha = 0.05)."""

DEFAULT_ALPHA: Final[float] = 0.05
TIE_TOLERANCE: Final[float] = 1e-12
"""Float slack when comparing statistics: ties are common because the kernel is discrete."""

PValueMethod = Literal["permutation", "simulation"]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class METError(ValueError):
    """Raised for invalid samples or arguments of the equality tests."""


@dataclass(frozen=True)
class Sample:
    """One draw `(x, y)`: a prompt id and the completion token ids sampled for it."""

    prompt_id: int
    completion_ids: tuple[int, ...]


@dataclass(frozen=True)
class METResult:
    """Outcome of a two-sample equality test."""

    statistic: float
    p_value: float
    n_p: int
    n_q: int


SampleDrawFn = Callable[[int, np.random.Generator], Sequence[Sample]]
"""`(budget, rng) -> samples`: draws a fresh sample; the callee defines what `budget` counts."""


def encode_samples(samples: Sequence[Sample], completion_len: int) -> tuple[IntArray, IntArray]:
    """Convert samples to arrays: prompt ids `[n]` and completions `[n, L]`.

    Completions longer than `completion_len` are truncated, shorter ones are right-padded
    with `PAD_TOKEN_ID`.

    Raises:
        METError: On an empty sample, a non-positive length or a negative token id.
    """
    if completion_len < 1:
        raise METError(f"completion_len must be >= 1, got {completion_len}")
    if not samples:
        raise METError("a sample must contain at least one completion")
    prompts = np.array([sample.prompt_id for sample in samples], dtype=np.int64)
    completions = np.full((len(samples), completion_len), PAD_TOKEN_ID, dtype=np.int64)
    for row, sample in enumerate(samples):
        kept = sample.completion_ids[:completion_len]
        if kept and min(kept) < 0:
            raise METError(f"token ids must be >= 0 (got {min(kept)}); {PAD_TOKEN_ID} is padding")
        completions[row, : len(kept)] = kept
    return prompts, completions


class _PooledSample:
    """Pooled rows with precomputed (prompt, position, token) group ids for fast MMD."""

    def __init__(self, prompts: IntArray, completions: IntArray) -> None:
        """Index the pooled rows once; labels are supplied per statistic evaluation."""
        self.length = completions.shape[1]
        _, self.prompt_index = np.unique(prompts, return_inverse=True)
        self.prompt_index = self.prompt_index.astype(np.int64)
        self.n_prompts = int(self.prompt_index.max()) + 1
        values, value_rank = np.unique(completions, return_inverse=True)
        value_rank = value_rank.reshape(completions.shape).astype(np.int64)
        positions = np.arange(self.length, dtype=np.int64)[None, :]
        keys = (self.prompt_index[:, None] * self.length + positions) * len(values) + value_rank
        _, group_ids = np.unique(keys, return_inverse=True)
        self.group_ids = group_ids.reshape(completions.shape).astype(np.int64)
        self.n_groups = int(self.group_ids.max()) + 1

    def statistic(self, is_p: BoolArray, rows: IntArray | None = None) -> float:
        """Hamming MMD of the rows labelled P versus the other rows.

        Args:
            is_p: Boolean label per selected row (True = sample P).
            rows: Row indices to use (all rows when None).
        """
        group_ids = self.group_ids if rows is None else self.group_ids[rows]
        prompt_index = self.prompt_index if rows is None else self.prompt_index[rows]
        flat = group_ids.ravel()
        weights_p = np.repeat(is_p.astype(np.float64), self.length)
        counts_p = np.bincount(flat, weights=weights_p, minlength=self.n_groups)
        counts_q = np.bincount(flat, weights=1.0 - weights_p, minlength=self.n_groups)
        per_prompt_p = np.bincount(prompt_index[is_p], minlength=self.n_prompts).astype(np.float64)
        per_prompt_q = np.bincount(prompt_index[~is_p], minlength=self.n_prompts).astype(np.float64)
        n_p, n_q = float(is_p.sum()), float((~is_p).sum())
        length = float(self.length)
        # Sum over ordered same-prompt pairs of matches / L; self-pairs contribute L / L = 1.
        sum_pp = float(counts_p @ counts_p) / length - n_p
        sum_qq = float(counts_q @ counts_q) / length - n_q
        sum_pq = float(counts_p @ counts_q) / length
        pairs_pp = float((per_prompt_p * (per_prompt_p - 1.0)).sum())
        pairs_qq = float((per_prompt_q * (per_prompt_q - 1.0)).sum())
        pairs_pq = float((per_prompt_p * per_prompt_q).sum())
        return (
            sum_pp / max(pairs_pp, 1.0)
            - 2.0 * sum_pq / max(pairs_pq, 1.0)
            + sum_qq / max(pairs_qq, 1.0)
        )


def _pool(
    samples_p: Sequence[Sample], samples_q: Sequence[Sample], completion_len: int
) -> tuple[_PooledSample, BoolArray]:
    """Pool both samples (P first) and return the index with the observed labels."""
    prompts_p, completions_p = encode_samples(samples_p, completion_len)
    prompts_q, completions_q = encode_samples(samples_q, completion_len)
    pooled = _PooledSample(
        np.concatenate([prompts_p, prompts_q]), np.concatenate([completions_p, completions_q])
    )
    labels = np.zeros(len(prompts_p) + len(prompts_q), dtype=np.bool_)
    labels[: len(prompts_p)] = True
    return pooled, labels


def mmd_hamming(
    samples_p: Sequence[Sample],
    samples_q: Sequence[Sample],
    *,
    completion_len: int = DEFAULT_COMPLETION_LEN,
) -> float:
    """Hamming-kernel MMD between two samples (same-prompt pairs only).

    Matches `model_equality_testing.tests.mmd_hamming` on identical data (a test compares the
    two when the official package is installed). Values near 0 mean equal distributions;
    the unbiased estimator can be slightly negative.

    Args:
        samples_p: Samples from the reference.
        samples_q: Samples from the API under test.
        completion_len: Completion length `L` (truncate / right-pad with -1).

    Returns:
        The MMD estimate.
    """
    pooled, labels = _pool(samples_p, samples_q, completion_len)
    return pooled.statistic(labels)


def _permutation_statistics(
    pooled: _PooledSample,
    labels: BoolArray,
    n_resamples: int,
    rng: np.random.Generator,
    *,
    stratify_by_prompt: bool,
) -> FloatArray:
    """Statistics under random relabelling of the pooled rows."""
    stats = np.empty(n_resamples, dtype=np.float64)
    if stratify_by_prompt:
        grouped_order = np.argsort(pooled.prompt_index, kind="stable")
        grouped_labels = labels[grouped_order]
    for draw in range(n_resamples):
        if stratify_by_prompt:
            # Sorting by (prompt, random key) shuffles rows inside each prompt block, so the
            # number of P labels per prompt is preserved.
            shuffled = np.lexsort((rng.random(len(labels)), pooled.prompt_index))
            permuted = np.empty_like(labels)
            permuted[shuffled] = grouped_labels
        else:
            permuted = rng.permutation(labels)
        stats[draw] = pooled.statistic(permuted)
    return stats


def _simulated_statistics(
    samples_p: Sequence[Sample],
    samples_q: Sequence[Sample],
    reference_pool: Sequence[Sample],
    completion_len: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Algorithm 1: statistics of two disjoint draws from the reference pool, per prompt."""
    pool_prompts, pool_completions = encode_samples(reference_pool, completion_len)
    pooled = _PooledSample(pool_prompts, pool_completions)
    rows_by_prompt: dict[int, IntArray] = {
        int(prompt): np.flatnonzero(pool_prompts == prompt) for prompt in np.unique(pool_prompts)
    }
    need_p: dict[int, int] = {}
    need_q: dict[int, int] = {}
    for sample in samples_p:
        need_p[sample.prompt_id] = need_p.get(sample.prompt_id, 0) + 1
    for sample in samples_q:
        need_q[sample.prompt_id] = need_q.get(sample.prompt_id, 0) + 1
    plan: list[tuple[IntArray, int, int]] = []
    for prompt in sorted(set(need_p) | set(need_q)):
        available = rows_by_prompt.get(prompt)
        wanted_p, wanted_q = need_p.get(prompt, 0), need_q.get(prompt, 0)
        if available is None or len(available) < wanted_p + wanted_q:
            have = 0 if available is None else len(available)
            raise METError(
                f"reference pool has {have} samples for prompt {prompt}, needs "
                f"{wanted_p + wanted_q} (n_p + n_q): provide a larger pool"
            )
        plan.append((available, wanted_p, wanted_q))
    stats = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        rows: list[IntArray] = []
        labels: list[BoolArray] = []
        for available, wanted_p, wanted_q in plan:
            chosen = rng.permutation(available)[: wanted_p + wanted_q]
            rows.append(chosen)
            labels.append(np.arange(wanted_p + wanted_q) < wanted_p)
        stats[draw] = pooled.statistic(np.concatenate(labels), np.concatenate(rows))
    return stats


def two_sample_test(
    samples_p: Sequence[Sample],
    samples_q: Sequence[Sample],
    *,
    completion_len: int = DEFAULT_COMPLETION_LEN,
    pvalue: PValueMethod = "permutation",
    n_resamples: int = DEFAULT_N_RESAMPLES,
    reference_pool: Sequence[Sample] | None = None,
    seed: int = 0,
    stratify_by_prompt: bool = True,
) -> METResult:
    """Test H0: both samples come from the same prompt-conditional distribution.

    Args:
        samples_p: Samples from the local reference.
        samples_q: Samples from the API under test.
        completion_len: Completion length `L`.
        pvalue: "permutation" (no extra sampling needed) or "simulation" (Algorithm 1; needs
            `reference_pool`, usually more powerful and the paper's default).
        n_resamples: Number of permutations or simulated null draws `B`.
        reference_pool: Pool of reference samples for "simulation"; for each prompt it must
            hold at least `n_p + n_q` samples (a larger pool approximates fresh draws better;
            it should not contain `samples_p` itself).
        seed: Seed of the resampling generator.
        stratify_by_prompt: Permute labels within each prompt (exact given the prompt design);
            False reproduces the official global permutation.

    Returns:
        `METResult(statistic, p_value, n_p, n_q)` with `p = (1 + #{T_i >= T_obs}) / (B + 1)`.

    Raises:
        METError: For `n_resamples < 1`, or "simulation" without a sufficient pool.
    """
    if n_resamples < 1:
        raise METError(f"n_resamples must be >= 1, got {n_resamples}")
    rng = np.random.default_rng(seed)
    pooled, labels = _pool(samples_p, samples_q, completion_len)
    observed = pooled.statistic(labels)
    if pvalue == "permutation":
        null = _permutation_statistics(
            pooled, labels, n_resamples, rng, stratify_by_prompt=stratify_by_prompt
        )
    elif pvalue == "simulation":
        if reference_pool is None:
            raise METError("pvalue='simulation' requires a reference_pool")
        null = _simulated_statistics(
            samples_p, samples_q, reference_pool, completion_len, n_resamples, rng
        )
    else:
        raise METError(f"unknown pvalue method {pvalue!r}; expected 'permutation' or 'simulation'")
    exceed = int(np.count_nonzero(null >= observed - TIE_TOLERANCE))
    return METResult(
        statistic=float(observed),
        p_value=(1.0 + exceed) / (n_resamples + 1.0),
        n_p=len(samples_p),
        n_q=len(samples_q),
    )


def power_curve(
    sample_p_fn: SampleDrawFn,
    sample_q_fn: SampleDrawFn,
    budgets: Sequence[int],
    *,
    n_trials: int,
    alpha: float = DEFAULT_ALPHA,
    seed: int,
    completion_len: int = DEFAULT_COMPLETION_LEN,
    n_resamples: int = POWER_CURVE_N_RESAMPLES,
    stratify_by_prompt: bool = True,
) -> list[tuple[int, float]]:
    """Monte Carlo rejection rate of the permutation test as a function of the budget.

    With `sample_q_fn` drawing from the same law as `sample_p_fn` the curve is the empirical
    type-I error (it should hover around `alpha`); otherwise it is the power.

    Args:
        sample_p_fn: `(budget, rng) -> samples` from the reference.
        sample_q_fn: `(budget, rng) -> samples` from the API / alternative.
        budgets: Values passed as `budget` to both draw functions; they define their meaning
            (total samples, or samples per prompt as in the demos).
        n_trials: Independent trials per budget.
        alpha: Significance level; a trial rejects when `p <= alpha`.
        seed: Master seed (each trial derives its own stream).
        completion_len: Completion length `L`.
        n_resamples: Permutations per trial.
        stratify_by_prompt: See `two_sample_test`.

    Returns:
        `[(budget, rejection_rate), ...]` in the order of `budgets`.
    """
    if n_trials < 1:
        raise METError(f"n_trials must be >= 1, got {n_trials}")
    curve: list[tuple[int, float]] = []
    for budget_index, budget in enumerate(budgets):
        rejections = 0
        for trial in range(n_trials):
            rng = np.random.default_rng([seed, budget_index, trial])
            result = two_sample_test(
                sample_p_fn(budget, rng),
                sample_q_fn(budget, rng),
                completion_len=completion_len,
                pvalue="permutation",
                n_resamples=n_resamples,
                seed=int(rng.integers(0, 2**31)),
                stratify_by_prompt=stratify_by_prompt,
            )
            rejections += int(result.p_value <= alpha)
        curve.append((int(budget), rejections / n_trials))
    return curve
