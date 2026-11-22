"""Simulation of an adaptive spending rule on the `ref-*` endpoints (resampling over prompts).

The rule mirrors what can be done on `ep-*`: ask prompts in chunks, keep a distance matrix
between endpoints, and stop feeding an endpoint when every pair that involves it is *resolved*,
that is, when its mean distance is more than `KAPPA` standard errors away from the threshold
`tau`. The threshold is calibrated on prompts DISJOINT from the ones used to decide (half of the
prompts calibrate, the other half are spent), at the same number of prompts, as the kit does with
`--ref-like`.

Ground truth: a pair is "same model" iff both endpoints are S (disguises included). The
decision for a pair is `distance > tau` -> different. Everything is a resampling experiment over
prompts with the nine ref endpoints fixed, so intervals reflect prompt choice only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.discrimination import (
    BASE_MODEL,
    AgreementTensor,
    EndpointTruth,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
KAPPA = 2.0
"""A pair is resolved when its distance is KAPPA standard errors from the threshold."""
CHUNK_PROMPTS = 4
"""Prompts added per round (one answer per endpoint and prompt)."""
TRIALS = 300
ADAPTIVE_SEED = 20261123
MIN_PROMPTS_FOR_SE = 3


@dataclass(frozen=True)
class StepStats:
    """Aggregate over trials of the state after `prompts` prompts per endpoint."""

    prompts: int
    pair_error_uniform: float
    hard_pair_error_uniform: float
    resolved_fraction: float
    error_among_resolved: float


@dataclass(frozen=True)
class AdaptiveSummary:
    """Uniform spending against the adaptive rule, at comparable average spend."""

    steps: list[StepStats]
    adaptive_mean_spend: float
    adaptive_pair_error: float
    adaptive_hard_pair_error: float
    uniform_error_at_same_spend: float
    uniform_hard_error_at_same_spend: float
    cap_prompts: int
    trials: int


def _per_prompt_distance(tensor0: AgreementTensor, tensor1: AgreementTensor) -> FloatArray:
    """`[P, E, E]` distance of one draw per prompt: repeat 0, or repeat 1 when 0 is missing."""
    d0, d1 = tensor0.distance(), tensor1.distance()
    return np.where(np.isfinite(d0), d0, d1)


def _pair_masks(truth: Mapping[str, EndpointTruth], endpoints: Sequence[str]) -> tuple[
        BoolArray, BoolArray, BoolArray]:
    """Boolean `[E, E]` masks: upper-triangle pairs, same-model pairs, hard (T1|T2 x S) pairs."""
    n = len(endpoints)
    model = [truth[e].model for e in endpoints]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    same = np.array([[model[i] == model[j] == BASE_MODEL for j in range(n)] for i in range(n)])
    hard = np.array([[{model[i], model[j]} in ({"T1", BASE_MODEL}, {"T2", BASE_MODEL})
                      for j in range(n)] for i in range(n)])
    return upper, same & upper, hard & upper


def simulate(
    tensor0: AgreementTensor,
    tensor1: AgreementTensor,
    truth: Mapping[str, EndpointTruth],
    trials: int = TRIALS,
    seed: int = ADAPTIVE_SEED,
) -> AdaptiveSummary | None:
    """Run the uniform-vs-adaptive experiment; None when there are too few prompts."""
    per_prompt = _per_prompt_distance(tensor0, tensor1)
    usable = np.flatnonzero(np.isfinite(per_prompt).any(axis=(1, 2)))
    if len(usable) < 4 * CHUNK_PROMPTS:
        return None
    endpoints = tensor0.endpoints
    n_endpoints = len(endpoints)
    upper, same, hard = _pair_masks(truth, endpoints)
    rng = np.random.default_rng(seed)
    cap = (len(usable) // 2 // CHUNK_PROMPTS) * CHUNK_PROMPTS
    rounds = cap // CHUNK_PROMPTS
    err_all = np.zeros((trials, rounds))
    err_hard = np.zeros((trials, rounds))
    resolved_frac = np.zeros((trials, rounds))
    err_resolved = np.full((trials, rounds), np.nan)
    spend_adaptive = np.zeros(trials)
    adaptive_err = np.zeros(trials)
    adaptive_hard = np.zeros(trials)
    for trial in range(trials):
        order = rng.permutation(usable)
        calibration, spending = order[:len(usable) // 2], order[len(usable) // 2:][:cap]
        active = np.ones(n_endpoints, dtype=bool)
        spend = np.full(n_endpoints, float(cap))
        final_wrong = np.zeros((n_endpoints, n_endpoints), dtype=bool)
        decided = np.zeros((n_endpoints, n_endpoints), dtype=bool)
        for round_index in range(rounds):
            n_prompts = (round_index + 1) * CHUNK_PROMPTS
            cal_mean = np.nanmean(per_prompt[calibration[:n_prompts]], axis=0)
            tau = 0.5 * (np.nanmean(cal_mean[same]) + np.nanmean(cal_mean[hard]))
            chosen = per_prompt[spending[:n_prompts]]
            mean = np.nanmean(chosen, axis=0)
            with np.errstate(all="ignore"):
                se = np.nanstd(chosen, axis=0) / np.sqrt(
                    np.maximum(np.isfinite(chosen).sum(axis=0), 1))
            resolved = (np.abs(mean - tau) >= KAPPA * se) & (n_prompts >= MIN_PROMPTS_FOR_SE)
            wrong = (mean > tau) == same
            wrong = wrong & upper
            err_all[trial, round_index] = wrong[upper].mean()
            err_hard[trial, round_index] = wrong[same | hard].mean()
            resolved_frac[trial, round_index] = resolved[upper].mean()
            if resolved[upper].any():
                err_resolved[trial, round_index] = wrong[resolved & upper].mean()
            for endpoint in range(n_endpoints):
                if not active[endpoint]:
                    continue
                pairs = upper[endpoint] | upper[:, endpoint]
                partner_resolved = resolved[endpoint] | resolved[:, endpoint]
                if bool(partner_resolved[pairs].all()):
                    active[endpoint] = False
                    spend[endpoint] = float(n_prompts)
            frozen = ~(active[:, None] | active[None, :])
            newly = frozen & ~decided
            final_wrong[newly] = wrong[newly]
            decided |= newly
        final_wrong[~decided] = wrong[~decided]
        spend_adaptive[trial] = spend.mean()
        adaptive_err[trial] = final_wrong[upper].mean()
        adaptive_hard[trial] = final_wrong[same | hard].mean()
    steps = [
        StepStats(
            prompts=(i + 1) * CHUNK_PROMPTS,
            pair_error_uniform=float(err_all[:, i].mean()),
            hard_pair_error_uniform=float(err_hard[:, i].mean()),
            resolved_fraction=float(resolved_frac[:, i].mean()),
            error_among_resolved=float(np.nanmean(err_resolved[:, i]))
            if np.isfinite(err_resolved[:, i]).any() else float("nan"),
        )
        for i in range(rounds)
    ]
    mean_spend = float(spend_adaptive.mean())
    nearest = min(range(rounds), key=lambda i: abs((i + 1) * CHUNK_PROMPTS - mean_spend))
    return AdaptiveSummary(
        steps=steps, adaptive_mean_spend=mean_spend,
        adaptive_pair_error=float(adaptive_err.mean()),
        adaptive_hard_pair_error=float(adaptive_hard.mean()),
        uniform_error_at_same_spend=float(err_all[:, nearest].mean()),
        uniform_hard_error_at_same_spend=float(err_hard[:, nearest].mean()),
        cap_prompts=cap, trials=trials,
    )
