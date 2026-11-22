"""Vectorized scoring and k-fold calibration helpers (numpy only).

All thresholds are fitted on the training folds and applied to the held-out fold, so every
reported number is an out-of-fold estimate; the spread over folds is reported as a standard
deviation. The metric is the official per-image mIoU (classes 0 and 1, empty union = 1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from hackathon.wmloc.research.errors import ResearchError

DEFAULT_FOLDS = 5
DEFAULT_SEED = 2026
"""Seed of every fold assignment (named once, shared by all experiments)."""
THRESHOLD_GRID_POINTS = 41


class EvaluationError(ResearchError):
    """Inconsistent shapes or empty input in an evaluation helper."""


def miou_per_image(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Official per-image mIoU of binary masks [N, H, W] (or [H, W]), float64 [N].

    Raises:
        EvaluationError: shape mismatch.
    """
    pred = np.atleast_3d(prediction) if prediction.ndim == 2 else prediction
    gold = np.atleast_3d(truth) if truth.ndim == 2 else truth
    if pred.shape != gold.shape:
        raise EvaluationError(f"shape mismatch {pred.shape} vs {gold.shape}")
    pred_one = pred.reshape(len(pred), -1).astype(bool)
    gold_one = gold.reshape(len(gold), -1).astype(bool)
    total = np.zeros(len(pred_one), dtype=np.float64)
    for class_one in (True, False):
        pred_c = pred_one == class_one
        gold_c = gold_one == class_one
        union = (pred_c | gold_c).sum(axis=1)
        inter = (pred_c & gold_c).sum(axis=1)
        total += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return total / 2.0


def stratified_folds(strata: Sequence[str], folds: int, seed: int) -> np.ndarray:
    """Fold id per item, balanced inside each stratum (e.g. method x regime).

    Raises:
        EvaluationError: fewer than 2 folds.
    """
    if folds < 2:
        raise EvaluationError("need at least 2 folds")
    rng = np.random.default_rng(seed)
    assignment = np.zeros(len(strata), dtype=np.int64)
    keys = np.asarray(strata)
    for key in sorted(set(strata)):
        members = np.flatnonzero(keys == key)
        rng.shuffle(members)
        assignment[members] = np.arange(len(members)) % folds
    return assignment


@dataclass(frozen=True)
class CrossValResult:
    """Out-of-fold result of a calibrated rule.

    Attributes:
        per_image: Out-of-fold mIoU of every image, [N].
        fold_means: Mean mIoU of each held-out fold.
        parameters: Parameter fitted on each training fold.
    """

    per_image: np.ndarray
    fold_means: np.ndarray
    parameters: list[float]

    @property
    def mean(self) -> float:
        """Mean out-of-fold mIoU."""
        return float(self.per_image.mean())

    @property
    def fold_std(self) -> float:
        """Standard deviation of the held-out fold means."""
        return float(self.fold_means.std(ddof=1)) if len(self.fold_means) > 1 else 0.0


def cross_validate_parameter(
    make_prediction: Callable[[float, np.ndarray], np.ndarray],
    truth: np.ndarray,
    candidates: Sequence[float],
    fold_ids: np.ndarray,
) -> CrossValResult:
    """Fit one scalar parameter per training fold, score it on the held-out fold.

    Args:
        make_prediction: `(parameter, indices) -> masks [len(indices), H, W]` for those images.
        truth: Ground-truth masks [N, H, W].
        candidates: Parameter values tried.
        fold_ids: Fold id of each of the N images.
    """
    per_image = np.zeros(len(truth), dtype=np.float64)
    means: list[float] = []
    fitted: list[float] = []
    for fold in sorted(set(fold_ids.tolist())):
        held = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        best_value, best_score = candidates[0], -1.0
        for value in candidates:
            score = float(miou_per_image(make_prediction(value, train), truth[train]).mean())
            if score > best_score:
                best_value, best_score = value, score
        per_image[held] = miou_per_image(make_prediction(best_value, held), truth[held])
        means.append(float(per_image[held].mean()))
        fitted.append(float(best_value))
    return CrossValResult(per_image, np.array(means), fitted)


def threshold_grid(values: np.ndarray, points: int = THRESHOLD_GRID_POINTS) -> list[float]:
    """Evenly spaced candidate thresholds between the 1st and 99th percentile of `values`."""
    low, high = np.percentile(values, [1, 99])
    return [float(v) for v in np.linspace(low, high, points)]


def roc_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney form) of `scores` for the boolean `positives`."""
    positive = scores[positives]
    negative = scores[~positives]
    if len(positive) == 0 or len(negative) == 0:
        raise EvaluationError("AUC needs both classes")
    ranks = rankdata(np.concatenate([positive, negative]))
    rank_sum = ranks[: len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )
