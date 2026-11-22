"""Threshold calibration and uncertainty estimates for score-map localizers.

Score maps are computed once, then every threshold is a cheap comparison. A threshold tuned and
evaluated on the same images is optimistic, so `cross_validated_miou` fits it on one half and
scores the other half (and the reverse); that is the figure to quote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from hackathon.wmloc.constants import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    DEFAULT_SEED,
)
from hackathon.wmloc.metric import per_image_miou

DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))
"""Agreement thresholds tried, from chance (0.50) to 0.99."""
CV_FOLDS = 2


@dataclass(frozen=True)
class SweepResult:
    """Mean mIoU for every candidate threshold.

    Attributes:
        thresholds: Candidate cuts.
        scores: Mean mIoU at each cut (same order).
        best_threshold: The cut with the highest mean mIoU (ties: the lowest cut).
        best_score: Its mean mIoU.
    """

    thresholds: tuple[float, ...]
    scores: tuple[float, ...]
    best_threshold: float
    best_score: float


def per_image_at_threshold(
    score_maps: torch.Tensor, truth: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Per-image mIoU [N] when `score_maps` [N, G, G] are cut at `threshold`."""
    return per_image_miou((score_maps >= threshold).to(torch.uint8), truth)


def sweep_thresholds(
    score_maps: torch.Tensor,
    truth: torch.Tensor,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
) -> SweepResult:
    """Mean mIoU of `score_maps` against `truth` for each threshold, and the best one."""
    means = [float(per_image_at_threshold(score_maps, truth, t).mean().item()) for t in thresholds]
    best_index = max(range(len(means)), key=lambda i: (means[i], -i))
    return SweepResult(tuple(thresholds), tuple(means), thresholds[best_index], means[best_index])


def cross_validated_per_image(
    score_maps: torch.Tensor,
    truth: torch.Tensor,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    """Per-image mIoU [N] where each image is scored with a threshold fitted on the other fold.

    Images are split into `CV_FOLDS` folds by a seeded permutation (reproducible).
    """
    count = score_maps.shape[0]
    generator = torch.Generator().manual_seed(seed)
    folds = torch.randperm(count, generator=generator).chunk(CV_FOLDS)
    result = torch.zeros(count, dtype=torch.float64)
    for held_out in folds:
        fit_mask = torch.ones(count, dtype=torch.bool)
        fit_mask[held_out] = False
        best = sweep_thresholds(score_maps[fit_mask], truth[fit_mask], thresholds).best_threshold
        result[held_out] = per_image_at_threshold(score_maps[held_out], truth[held_out], best)
    return result


def bootstrap_mean_ci(
    values: torch.Tensor,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """(mean, low, high) of `values` with a percentile bootstrap over images."""
    count = values.numel()
    if count == 0:
        raise ValueError("cannot bootstrap an empty set")
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(0, count, (resamples, count), generator=generator)
    means = values.to(torch.float64)[picks].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    low, high = torch.quantile(means, torch.tensor([tail, 1.0 - tail], dtype=torch.float64))
    return float(values.mean().item()), float(low.item()), float(high.item())
