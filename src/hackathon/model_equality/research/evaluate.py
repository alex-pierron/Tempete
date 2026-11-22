"""Evaluation of pair statistics on the reference endpoints: AUC, calibration, leave-one-out MCC.

Honesty rules of this module:

* The z matrices are unsupervised (they use no label), so they are computed once for all nine
  endpoints and sliced for each fold.
* Weights and the model cut are always fitted on the training endpoints only; the held-out
  endpoint is then clustered with them (`loeo_predictions`).
* A pair appears twice in the leave-one-endpoint-out predictions (once per held-out member);
  both are scored, so the MCC is over `2 x pairs` predictions and is not a clean i.i.d. estimate.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.clustering import (
    ClusteringError,
    build_linkage,
    calibrate_model_threshold,
    calibrate_weights,
    combine_channels,
    cut_dendrogram,
    prior_weights,
    to_distance,
)
from hackathon.model_equality.pairwise import Channel
from hackathon.model_equality.research.dataset import ReferenceTruth
from hackathon.model_equality.scoring import confusion

FloatMatrix = npt.NDArray[np.float64]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261122
CONFIDENCE_LOW = 2.5
CONFIDENCE_HIGH = 97.5
MIN_TRAINING_ENDPOINTS = 3

METHOD_PRIOR = "prior"
METHOD_CALIBRATED = "calibrated"
METHOD_UNIFORM = "uniform"


@dataclass(frozen=True)
class ChannelScore:
    """Separation of one channel on the reference pairs."""

    channel: Channel
    n_same: int
    n_different: int
    mean_same: float
    mean_different: float
    auc: float
    cohens_d: float


def pairs_of(truth: ReferenceTruth) -> list[tuple[int, int, bool]]:
    """All endpoint pairs `(i, j, same_model)`."""
    size = len(truth.endpoints)
    return [(i, j, truth.same_model(i, j)) for i, j in itertools.combinations(range(size), 2)]


def auc_score(different: Sequence[float], same: Sequence[float]) -> float:
    """P(different > same), ties counting half; NaN when a side is empty."""
    if not len(different) or not len(same):
        return float("nan")
    d = np.asarray(different, dtype=np.float64)[:, None]
    s = np.asarray(same, dtype=np.float64)[None, :]
    return float((d > s).mean() + 0.5 * (d == s).mean())


def score_channel(channel: Channel, matrix: FloatMatrix, truth: ReferenceTruth) -> ChannelScore:
    """Mean z of same / different pairs, AUC and Cohen's d (different vs same)."""
    same: list[float] = []
    different: list[float] = []
    for i, j, is_same in pairs_of(truth):
        value = float(matrix[i, j])
        if np.isfinite(value):
            (same if is_same else different).append(value)
    pooled = float(np.sqrt((np.var(same) + np.var(different)) / 2.0)) if same and different else 0.0
    mean_same = float(np.mean(same)) if same else float("nan")
    mean_different = float(np.mean(different)) if different else float("nan")
    cohens = (mean_different - mean_same) / pooled if pooled > 0 else float("nan")
    return ChannelScore(channel, len(same), len(different), mean_same, mean_different,
                        auc_score(different, same), cohens)


def mcc_of(truth_flags: Sequence[bool], guess_flags: Sequence[bool]) -> float:
    """Matthews correlation coefficient (0 on a degenerate marginal, as the kit does)."""
    return confusion(list(truth_flags), list(guess_flags)).mcc


def bootstrap_interval(
    truth_flags: Sequence[bool], guess_flags: Sequence[bool],
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the MCC, resampling pairs."""
    rng = np.random.default_rng(seed)
    t = np.asarray(truth_flags)
    g = np.asarray(guess_flags)
    values = []
    for _ in range(resamples):
        pick = rng.integers(0, len(t), len(t))
        values.append(mcc_of(t[pick].tolist(), g[pick].tolist()))
    low, high = np.percentile(values, [CONFIDENCE_LOW, CONFIDENCE_HIGH])
    return float(low), float(high)


def subset_channels(
    z_by_channel: Mapping[Channel, FloatMatrix], keep: Sequence[int]
) -> dict[Channel, FloatMatrix]:
    """Slice every channel matrix to the endpoint indices `keep`."""
    index = np.ix_(list(keep), list(keep))
    return {channel: matrix[index] for channel, matrix in z_by_channel.items()}


def fit_calibration(
    z_by_channel: Mapping[Channel, FloatMatrix], labels: Sequence[tuple[str, str]], method: str,
) -> tuple[dict[Channel, float], float]:
    """Channel weights and model cut fitted on endpoints of known labels.

    Args:
        z_by_channel: z matrices over the training endpoints.
        labels: `(family, model)` of each training endpoint.
        method: `prior` (kit prior weights, cut fitted) or `calibrated` (weights and cut fitted).

    Raises:
        ClusteringError: The training set lacks same-model or different-model pairs.
    """
    if method == METHOD_CALIBRATED:
        weights, _ = calibrate_weights(z_by_channel, labels)
    elif method == METHOD_UNIFORM:
        weights = {channel: 1.0 for channel in z_by_channel}
    else:
        weights = prior_weights(list(z_by_channel))
    combined = combine_channels(z_by_channel, weights)
    distance, _ = to_distance(combined)
    return weights, calibrate_model_threshold(distance, labels).tau_model


def loeo_predictions(
    z_by_channel: Mapping[Channel, FloatMatrix], truth: ReferenceTruth, method: str,
) -> list[tuple[int, int, bool, bool]]:
    """Leave-one-endpoint-out predictions `(i, j, truth_same, predicted_same)`.

    For each held-out endpoint e: fit weights and cut on the other endpoints, cluster ALL
    endpoints with them (average linkage), keep the pairs that involve e.

    Raises:
        ClusteringError: A training fold has no same-model or no different-model pair.
    """
    size = len(truth.endpoints)
    labels = truth.labels
    out: list[tuple[int, int, bool, bool]] = []
    for held in range(size):
        train = [k for k in range(size) if k != held]
        train_labels = [labels[k] for k in train]
        weights, tau = fit_calibration(subset_channels(z_by_channel, train), train_labels, method)
        combined = combine_channels(z_by_channel, weights)
        distance, _ = to_distance(combined)
        assigned = cut_dendrogram(build_linkage(distance), tau)
        for other in train:
            i, j = sorted((held, other))
            out.append((i, j, truth.same_model(i, j), bool(assigned[i] == assigned[j])))
    return out


@dataclass(frozen=True)
class LoeoResult:
    """Leave-one-endpoint-out outcome of one channel set and method."""

    method: str
    mcc: float
    ci_low: float
    ci_high: float
    false_same: int
    false_different: int
    wrong_pairs: list[tuple[int, int]]


def evaluate_loeo(
    z_by_channel: Mapping[Channel, FloatMatrix], truth: ReferenceTruth, method: str,
    with_interval: bool = True,
) -> LoeoResult:
    """MCC of `loeo_predictions` with a bootstrap interval (optional), and the wrong pairs."""
    try:
        rows = loeo_predictions(z_by_channel, truth, method)
    except ClusteringError:
        return LoeoResult(method, float("nan"), float("nan"), float("nan"), 0, 0, [])
    truths = [t for _, _, t, _ in rows]
    guesses = [g for _, _, _, g in rows]
    low, high = (bootstrap_interval(truths, guesses) if with_interval
                 else (float("nan"), float("nan")))
    wrong = sorted({(i, j) for i, j, t, g in rows if t != g})
    return LoeoResult(
        method, mcc_of(truths, guesses), low, high,
        sum(1 for _, _, t, g in rows if not t and g), sum(1 for _, _, t, g in rows if t and not g),
        wrong)


def pairwise_threshold_mcc(
    combined: FloatMatrix, truth: ReferenceTruth,
) -> tuple[float, float]:
    """In-sample best pairwise threshold on a combined z and its MCC (an upper bound).

    Returns:
        `(mcc, threshold)`; a pair is called "same" when its score is below the threshold.
    """
    pairs = pairs_of(truth)
    values = np.array([combined[i, j] for i, j, _ in pairs])
    flags = [same for _, _, same in pairs]
    finite = np.isfinite(values)
    candidates = np.unique(values[finite])
    best, best_tau = -1.0, float("nan")
    for tau in np.concatenate([[candidates.min() - 1.0], (candidates[:-1] + candidates[1:]) / 2.0,
                               [candidates.max() + 1.0]]):
        guess = [bool(v < tau) if np.isfinite(v) else False for v in values]
        score = mcc_of(flags, guess)
        if score > best:
            best, best_tau = score, float(tau)
    return best, best_tau
