"""Team C's scoring model applied to team G's exports (what the director forwards to `r-unl-c`).

Everything here is a re-implementation of formulas published by team C (`ms_predict.py`,
`lab14_routing_ladder.py`), plus the two constraints the director added:

* `mean_std_score = 1 - clip((|mean - gold_mean| + |std - gold_std|) / d_orig)` per group, with the
  fitted gold (2.25, 3.0) on exact images and (3.1, 3.0, std ASSUMED) on strong crops; `d_orig` is
  the same distance for the original model;
* public ~ `LEVEL * ms_exact` (LEVEL = 0.116: prediction_score x retain, constant over the
  gold_soft family; team C fitted it on the exact group only, so `predicted_public` keeps that
  definition) and the empirical routing correction `0.123 * (1 + 0.5 * (fallback - 0.212)) *
  ms_exact`; `predicted_public_both_groups` uses the mean of the exact and strong-crop groups
  (the official score averages both, but its level was never fitted that way: a diagnostic);
* hard constraint: KL(orig || sub) at T = 2 on the exact images of the NON-saturated classes <= 1.4.

None of this predicts what a better distribution fit would earn (the factor LEVEL is assumed
constant): it ranks files on the quantities team C's model sees. Fidelity and drift come from
`g_probe`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.proxy_eval import tempered_kl
from hackathon.unlearning.research.g_probe import TAIL_K
from hackathon.unlearning.research.g_targets import BLUR_HIGH, BLUR_LOW, entropy_nats, tail_mass

GOLD_MEAN_EXACT = 2.25
GOLD_STD_EXACT = 3.0
GOLD_MEAN_ALTERED = 3.1
GOLD_STD_ALTERED = 3.0
LEVEL = 0.116
ROUTED_LEVEL = 0.123
ROUTING_SLOPE = 0.5
BASELINE_FALLBACK = 0.212
KL_CONSTRAINT_NONSAT = 1.4
ALTERED_VIEW = "strong_view"
EXACT_VIEW = "exact"


class ScoreError(HackathonError):
    """An export or a reference file cannot be scored (missing view, shape mismatch)."""


@dataclass(frozen=True)
class CandidateScore:
    """Team C's quantities for one (file, delta) pair."""

    name: str
    delta: float
    scale: float
    loss_mean: float
    loss_std: float
    ms_exact: float
    altered_loss_mean: float
    altered_loss_std: float
    ms_altered: float
    ms: float
    sibling_fallback: float
    kl_t2_nonsat: float
    kl_t2_sat: float
    blur_share: float
    sure_right_share: float
    sure_wrong_share: float
    entropy_t1: float
    entropy_t2: float
    tail_mass_t2: float
    confidence: float
    flips_nonsat: float
    predicted_public: float
    predicted_public_routed: float
    predicted_public_both_groups: float

    @property
    def respects_kl_constraint(self) -> bool:
        """KL(orig || sub) at T = 2 on non-saturated exact images stays within the director's bound."""
        return self.kl_t2_nonsat <= KL_CONSTRAINT_NONSAT


def loss_moments(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    """Mean and population std of the cross-entropy on the pseudo-labels."""
    loss = torch_f.cross_entropy(logits.double(), labels, reduction="none")
    return float(loss.mean()), float(loss.std(unbiased=False))


def mean_std_score(
    mean: float,
    std: float,
    original_mean: float,
    original_std: float,
    gold_mean: float,
    gold_std: float,
) -> float:
    """`1 - clip(d_sub / d_orig, 0, 1)` with `d = |mean - gold_mean| + |std - gold_std|`."""
    d_original = abs(original_mean - gold_mean) + abs(original_std - gold_std)
    if d_original <= 0.0:
        raise ScoreError("the original already sits on the gold moments: the score is undefined")
    d_sub = abs(mean - gold_mean) + abs(std - gold_std)
    return 1.0 - min(1.0, max(0.0, d_sub / d_original))


def apply_class_bias(logits: Tensor, classes: Sequence[int], delta: float) -> Tensor:
    """`logits - delta` on the listed classes (team C's post-hoc class-removal edit)."""
    edited = logits.clone()
    if classes:
        edited[:, list(classes)] -= delta
    return edited


def scale_logits(logits: Tensor, scale: float) -> Tensor:
    """`scale * logits`: a head scale (weight and bias), team C's sharpness knob (argmax-preserving)."""
    if scale <= 0.0:
        raise ScoreError(f"the head scale must be positive, got {scale}")
    return logits * scale


def sibling_fallback(
    logits: Tensor, labels: Tensor, saturated: Sequence[int], siblings: dict[int, list[int]]
) -> float:
    """Share of saturated-class images whose best non-saturated class is a present sibling."""
    rows = torch.isin(labels, torch.tensor(list(saturated), dtype=torch.long)).nonzero(
        as_tuple=True
    )[0]
    if len(rows) == 0:
        return 0.0
    masked = logits[rows].clone()
    masked[:, list(saturated)] = float("-inf")
    top = masked.argmax(dim=1)
    hits = [int(t) in siblings.get(int(labels[r]), []) for t, r in zip(top, rows, strict=True)]
    return float(np.mean(hits))


def load_views(path: Path) -> dict[str, Tensor]:
    """The `logits_<view>` arrays of an export (team C format) as float tensors."""
    if not path.is_file():
        raise ScoreError(f"missing export {path}")
    with np.load(path) as payload:
        return {
            key.removeprefix("logits_"): torch.from_numpy(payload[key]).float()
            for key in payload.files
            if key.startswith("logits_")
        }


def score_candidate(
    name: str,
    views: dict[str, Tensor],
    original_views: dict[str, Tensor],
    saturated: Sequence[int],
    siblings: dict[int, list[int]],
    delta: float = 0.0,
    scale: float = 1.0,
) -> CandidateScore:
    """Score one model (given its views) at class-bias `delta`; labels = argmax of the original.

    Raises:
        ScoreError: a needed view is missing from either side.
    """
    for view in (EXACT_VIEW, ALTERED_VIEW):
        if view not in views or view not in original_views:
            raise ScoreError(f"view '{view}' missing for {name}")
    labels = original_views[EXACT_VIEW].argmax(dim=1)
    exact = apply_class_bias(scale_logits(views[EXACT_VIEW], scale), saturated, delta)
    altered = apply_class_bias(scale_logits(views[ALTERED_VIEW], scale), saturated, delta)
    mean, std = loss_moments(exact, labels)
    altered_mean, altered_std = loss_moments(altered, labels)
    original_mean, original_std = loss_moments(original_views[EXACT_VIEW], labels)
    original_altered = loss_moments(original_views[ALTERED_VIEW], labels)
    ms_exact = mean_std_score(
        mean, std, original_mean, original_std, GOLD_MEAN_EXACT, GOLD_STD_EXACT
    )
    ms_altered = mean_std_score(
        altered_mean, altered_std, *original_altered, GOLD_MEAN_ALTERED, GOLD_STD_ALTERED
    )
    combined = 0.5 * (ms_exact + ms_altered)
    in_saturated = torch.isin(labels, torch.tensor(list(saturated), dtype=torch.long))
    divergence = tempered_kl(original_views[EXACT_VIEW], exact)
    probabilities = torch.softmax(exact.double(), dim=1)
    tempered = torch.softmax(exact.double() / 2.0, dim=1)
    fallback = sibling_fallback(exact, labels, saturated, siblings)
    flips = (exact.argmax(dim=1) != original_views[EXACT_VIEW].argmax(dim=1)).float()
    p_true = probabilities.gather(1, labels[:, None]).squeeze(1)
    return CandidateScore(
        name=name,
        delta=delta,
        scale=scale,
        loss_mean=mean,
        loss_std=std,
        ms_exact=ms_exact,
        altered_loss_mean=altered_mean,
        altered_loss_std=altered_std,
        ms_altered=ms_altered,
        ms=combined,
        sibling_fallback=fallback,
        kl_t2_nonsat=float(divergence[~in_saturated].mean()),
        kl_t2_sat=float(divergence[in_saturated].mean()) if in_saturated.any() else 0.0,
        blur_share=float(((p_true > BLUR_LOW) & (p_true < BLUR_HIGH)).float().mean()),
        sure_right_share=float((p_true >= BLUR_HIGH).float().mean()),
        sure_wrong_share=float((p_true <= BLUR_LOW).float().mean()),
        entropy_t1=float(entropy_nats(probabilities).mean()),
        entropy_t2=float(entropy_nats(tempered).mean()),
        tail_mass_t2=float(tail_mass(tempered, TAIL_K).mean()),
        confidence=float(probabilities.amax(dim=1).mean()),
        flips_nonsat=float(flips[~in_saturated].mean()),
        predicted_public=LEVEL * ms_exact,
        predicted_public_routed=ROUTED_LEVEL
        * (1.0 + ROUTING_SLOPE * (fallback - BASELINE_FALLBACK))
        * ms_exact,
        predicted_public_both_groups=LEVEL * combined,
    )


def delta_grid(
    name: str,
    views: dict[str, Tensor],
    original_views: dict[str, Tensor],
    saturated: Sequence[int],
    siblings: dict[int, list[int]],
    deltas: Sequence[float],
    scales: Sequence[float] = (1.0,),
) -> list[CandidateScore]:
    """`score_candidate` over class biases and head scales (the post-hoc axes of the alpha x delta grid)."""
    return [
        score_candidate(name, views, original_views, saturated, siblings, delta, scale)
        for scale in scales
        for delta in deltas
    ]
