"""Budget -> separation quality curves measured on the `ref-*` endpoints.

Question: for a budget of N answers per endpoint, is it better to ask N distinct prompts once
(`diversity`) or N/2 prompts twice (`repeat`)? And how fast does the separation of T1 / T2 from
the disguised S endpoints improve with N? Everything is a resampling experiment over prompts:

1. Precompute the agreement tensor of every window of consecutive draws (`build_windows`).
2. A design `(k prompts, r repeats)` draws k prompts at random among those that have a complete
   window of r draws on every endpoint, takes one such window per prompt at random, averages the
   distances and scores the pooled distance matrix (`pair_auc` or `endpoint_pool_auc`).
3. Repeating step 2 `n_draws` times gives a mean and a 5-95 % interval of the AUC and the
   probability that the separation is perfect (every hard pair farther apart than every S-S pair,
   which is what a clean clustering needs).

The resampling is over prompts, with the ref endpoints fixed: the interval measures prompt-choice
and draw variability, not endpoint variability (there are only 9 endpoints).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from hackathon.model_equality.research.discrimination import (
    AgreementTensor,
    EndpointTruth,
    agreement_tensor,
    endpoint_pool_auc,
    endpoint_sets,
    group_answers,
    pair_auc,
    pair_sets,
)
from hackathon.model_equality.store import Answer

LOW_QUANTILE = 0.05
HIGH_QUANTILE = 0.95
PERFECT_AUC = 1.0
STATISTIC_PAIR = "pair"
STATISTIC_POOL = "pool"
DEFAULT_DRAWS = 300
RESAMPLING_SEED = 20261121
"""Constant seed: the same answers always give the same curves."""


@dataclass(frozen=True)
class DesignResult:
    """Separation quality of one design `(k prompts x r repeats)`.

    Attributes:
        prompts: Distinct prompts per endpoint (k).
        repeats: Draws per prompt per endpoint (r).
        budget: Answers per endpoint (k * r).
        auc_mean: Mean AUC of (T1|T2, S) pairs against (S, S) pairs.
        auc_low: 5 % quantile of the AUC.
        auc_high: 95 % quantile of the AUC.
        p_perfect: Fraction of resamples with AUC = 1.
        draws: Number of resamples used.
        pool: Prompts eligible for the design (when `prompts` is close to it, the resamples
            overlap and the interval understates the real prompt-choice variability).
    """

    prompts: int
    repeats: int
    budget: int
    auc_mean: float
    auc_low: float
    auc_high: float
    p_perfect: float
    draws: int
    pool: int


Windows = Mapping[int, Sequence[AgreementTensor]]
"""`windows[r][j]`: agreement tensor built from the draws `j .. j + r - 1` of every endpoint."""


def build_windows(
    answers: Sequence[Answer], prompts: Sequence[str], endpoints: Sequence[str], kernel: str,
    max_repeats: int,
) -> dict[int, list[AgreementTensor]]:
    """Agreement tensors of every window of consecutive draws, for r = 1 .. `max_repeats`."""
    windows: dict[int, list[AgreementTensor]] = {}
    for r in range(1, max_repeats + 1):
        windows[r] = [
            agreement_tensor(group_answers(answers, endpoints, repeats=range(j, j + r)),
                             prompts, endpoints, kernel)
            for j in range(max_repeats - r + 1)
        ]
    return windows


def evaluate_design(
    windows: Windows,
    truth: Mapping[str, EndpointTruth],
    prompts: int,
    repeats: int,
    role: str = "hard",
    n_draws: int = DEFAULT_DRAWS,
    seed: int = RESAMPLING_SEED,
    statistic: str = STATISTIC_PAIR,
) -> DesignResult | None:
    """Resample one design and score it; None when too few prompts are eligible.

    Args:
        windows: Tensors of windows of draws (see `build_windows`); must contain `repeats`.
        truth: Ground truth of the endpoints.
        prompts: k, distinct prompts per endpoint.
        repeats: r, draws per prompt and endpoint (the design costs k * r answers per endpoint).
        role: `hard` (T1, T2 against S) or `easy` (T3, T4 against S).
        n_draws: Number of resamples.
        seed: Resampling seed.
        statistic: `pair` (T-S pairs against S-S pairs) or `pool` (each endpoint against the
            pooled S reference, an endpoint-level AUC).
    """
    if statistic not in (STATISTIC_PAIR, STATISTIC_POOL):
        raise ValueError(f"unknown statistic {statistic!r}")
    if repeats < 1 or repeats not in windows:
        raise ValueError(f"repeats must be one of {sorted(windows)}, got {repeats}")
    tensors = windows[repeats]
    endpoints = tensors[0].endpoints
    sets = endpoint_sets(truth, endpoints)
    pairs = pair_sets(sets)
    distances = [t.distance() for t in tensors]
    complete = np.stack([(t.draws >= repeats).all(axis=1) for t in tensors])
    pool = np.flatnonzero(complete.any(axis=0))
    if len(pool) < prompts:
        return None
    rng = np.random.default_rng(seed + 1009 * prompts + repeats)
    aucs = np.empty(n_draws)
    for draw in range(n_draws):
        chosen = rng.choice(pool, size=prompts, replace=False)
        picked = []
        for p in chosen:
            options = np.flatnonzero(complete[:, p])
            picked.append(distances[int(rng.choice(options))][p])
        with np.errstate(all="ignore"):
            pooled = np.nanmean(np.stack(picked), axis=0)
        aucs[draw] = (endpoint_pool_auc(pooled, sets, role) if statistic == STATISTIC_POOL
                      else pair_auc(pooled, pairs[role], pairs["ss"]))
    finite = aucs[np.isfinite(aucs)]
    if len(finite) == 0:
        return None
    return DesignResult(
        prompts=prompts, repeats=repeats, budget=prompts * repeats,
        auc_mean=float(finite.mean()),
        auc_low=float(np.quantile(finite, LOW_QUANTILE)),
        auc_high=float(np.quantile(finite, HIGH_QUANTILE)),
        p_perfect=float((finite >= PERFECT_AUC - 1e-12).mean()),
        draws=len(finite),
        pool=len(pool),
    )


def budget_curve(
    windows: Windows,
    truth: Mapping[str, EndpointTruth],
    budgets: Sequence[int],
    repeats: int,
    role: str = "hard",
    n_draws: int = DEFAULT_DRAWS,
    statistic: str = STATISTIC_PAIR,
) -> list[DesignResult]:
    """Curve of one repeat count: for each budget N, the design `(N / repeats, repeats)`."""
    results: list[DesignResult] = []
    for budget in budgets:
        if budget % repeats:
            continue
        result = evaluate_design(windows, truth, budget // repeats, repeats, role, n_draws,
                                 statistic=statistic)
        if result is not None:
            results.append(result)
    return results
