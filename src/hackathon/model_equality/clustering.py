"""Channel weighting, threshold calibration and two-level (family, then model) clustering.

Pipeline on top of the per-channel z-score matrices of `pairwise.py`:

1. `combine_channels`: weighted Stouffer combination into one pair score Z (large = different).
2. `calibrate_weights`: on endpoints whose truth is known (`ref-*`), keep the channels that
   separate different models while NOT separating same-model disguises (system prompt, persona,
   watermark, temperature): weight = (mean Z different - 90th percentile Z same) / (variances), clipped at 0,
   then shrunk toward the prior weights because the reference model is a different, small model.
3. `calibrate_model_threshold`: the cut height of an average-linkage dendrogram that maximises
   the model MCC on the reference endpoints (plateau midpoint), plus the highest merge height of
   the reference (all one family) as a floor for the family cut.
4. `cut_two_levels`: one dendrogram, two cuts (family >= model), so the labels are nested and a
   submission is consistent by construction (same model label implies same family label).

Everything here is statistics on small samples: thresholds transfer between `ref-*` and `ep-*`
only to the extent that the reference model behaves like the scored ones. The analysis reports
the margins so the team can see how fragile a cut is.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hackathon.model_equality.pairwise import (
    KERNEL_HAMMING_CHAR,
    KERNEL_HAMMING_WORD,
    KERNEL_LENGTH,
    KERNEL_ONEHOT_FIRST,
    KERNEL_ONEHOT_KEY,
    KERNEL_SPECTRUM,
    Channel,
)
from hackathon.model_equality.scoring import confusion

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]

PRIOR_KERNEL_WEIGHTS: dict[str, float] = {
    KERNEL_HAMMING_CHAR: 1.0,
    KERNEL_HAMMING_WORD: 1.0,
    KERNEL_SPECTRUM: 1.0,
    KERNEL_ONEHOT_KEY: 1.0,
    KERNEL_ONEHOT_FIRST: 0.0,
    KERNEL_LENGTH: 0.0,
}
"""Prior (uncalibrated) weight per kernel. `first` and `length` are the signals a persona or a
system prompt changes the most, so they are measured but trusted only if calibration says so."""
CALIBRATION_SHRINKAGE = 0.3
"""Weight of the prior in the calibrated weight `norm * (1 - s + s * prior)`: the prior only
scales channels that calibration found useful (the ref model is not the scored one, so
calibration is trusted but not blindly), it never resurrects a channel judged useless."""
SAME_PAIR_QUANTILE = 0.9
"""Same-model pairs are judged by their worst typical case, not their mean: one disguise that a
channel cannot ignore (a persona shifting every word) must be enough to distrust that channel."""
VARIANCE_FLOOR = 0.25
"""Added to the pooled variance in the weight formula so a noiseless channel cannot dominate."""
Z_CHANNEL_CLIP = 10.0
"""One channel never contributes more than this |z|, so a single channel cannot decide alone."""
Z_DISTANCE_MAX = 100.0
LINKAGE_METHOD = "average"
"""Average linkage: robust to the noise of a single pair, unlike single or complete linkage."""
TIE_TOLERANCE = 1e-9
FAMILY_MIN_GAP_RATIO = 1.5
"""Smallest ratio between consecutive merge heights accepted as a family boundary."""
FAMILY_FALLBACK_FACTOR = 2.0
"""Without a clear gap, the family cut is this multiple of the floor."""
FAMILY_FLOOR_FRACTION = 0.5
"""The family cut stays above this fraction of the highest in-family merge seen on `ref-*`.
Only a fraction: that merge is the last one of an average-linkage tree dominated by the single
most extreme variant (LoRA 1.5), and a populated scored family is not as extreme on average."""
FAMILY_FLOOR_MINIMUM = 2.0
"""Floor used when no reference calibration exists (a same-model pair scores about N(0, 1))."""
MIN_ENDPOINTS_TO_CLUSTER = 2
MIN_ENDPOINTS_TO_CALIBRATE = 3
AMBIGUITY_BAND = 0.3
"""A pair is ambiguous when its Z is within this fraction of the model threshold."""


class ClusteringError(ValueError):
    """Not enough endpoints or an inconsistent truth for the requested operation."""


@dataclass(frozen=True)
class ChannelDiagnostic:
    """How well one channel separates different models from same-model disguises (on ref)."""

    channel: Channel
    n_same: int
    n_different: int
    mean_same: float
    mean_different: float
    auc: float
    weight: float


@dataclass(frozen=True)
class ThresholdCalibration:
    """Model threshold chosen on the reference endpoints."""

    tau_model: float
    mcc_model: float
    lower_margin: float
    upper_margin: float


@dataclass(frozen=True)
class TwoLevelClustering:
    """Nested labelling: `family_ids[i]` and `model_ids[i]` for endpoint index i."""

    family_ids: list[int]
    model_ids: list[int]
    tau_model: float
    tau_family: float


def prior_weights(channels: Sequence[Channel]) -> dict[Channel, float]:
    """Prior weights of `channels` from `PRIOR_KERNEL_WEIGHTS`."""
    return {channel: PRIOR_KERNEL_WEIGHTS.get(channel[1], 0.0) for channel in channels}


def combine_channels(
    z_by_channel: Mapping[Channel, FloatMatrix], weights: Mapping[Channel, float]
) -> FloatMatrix:
    """Weighted Stouffer combination `sum(w z) / sqrt(sum w^2)` over the defined channels.

    Returns:
        `[E, E]` matrix, NaN for pairs with no defined positively weighted channel.
    """
    channels = [channel for channel in z_by_channel if weights.get(channel, 0.0) > 0.0]
    if not channels:
        raise ClusteringError("no channel has a positive weight")
    stack = np.stack([np.clip(z_by_channel[channel], -Z_CHANNEL_CLIP, Z_CHANNEL_CLIP)
                      for channel in channels])
    w = np.array([weights[channel] for channel in channels], dtype=np.float64)[:, None, None]
    defined = np.isfinite(stack)
    numerator = np.where(defined, stack, 0.0) * w
    denominator = np.sqrt((defined * w**2).sum(axis=0))
    combined = np.full(stack.shape[1:], np.nan)
    positive = denominator > 0
    combined[positive] = numerator.sum(axis=0)[positive] / denominator[positive]
    return combined


def to_distance(combined: FloatMatrix) -> tuple[FloatMatrix, list[tuple[int, int]]]:
    """Turn the combined score into a symmetric non-negative distance matrix.

    Undefined pairs are filled with the mean of the defined ones and returned so the report can
    flag them; negative scores (pairs closer than chance) are clipped to 0.
    """
    size = combined.shape[0]
    finite = np.isfinite(combined)
    upper = np.triu(np.ones((size, size), dtype=bool), k=1)
    undefined = [(int(i), int(j)) for i, j in zip(*np.nonzero(upper & ~finite), strict=True)]
    fill = float(combined[upper & finite].mean()) if (upper & finite).any() else 0.0
    distance = np.where(finite, combined, fill)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, Z_DISTANCE_MAX)
    np.fill_diagonal(distance, 0.0)
    return distance, undefined


def _auc(different: FloatMatrix, same: FloatMatrix) -> float:
    """P(score of a different-model pair > score of a same-model pair), ties counting half."""
    greater = (different[:, None] > same[None, :]).mean()
    ties = (different[:, None] == same[None, :]).mean()
    return float(greater + 0.5 * ties)


def split_pairs(
    labels: Sequence[tuple[str, str]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Same-model and different-model pairs *within a true family* from `(family, model)` labels."""
    same: list[tuple[int, int]] = []
    different: list[tuple[int, int]] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            if labels[first][0] != labels[second][0]:
                continue
            (same if labels[first] == labels[second] else different).append((first, second))
    return same, different


def calibrate_weights(
    z_by_channel: Mapping[Channel, FloatMatrix],
    labels: Sequence[tuple[str, str]],
) -> tuple[dict[Channel, float], list[ChannelDiagnostic]]:
    """Calibrate channel weights on endpoints with known `(family, model)` labels.

    Raises:
        ClusteringError: No same-model or no different-model pair in `labels`.
    """
    same_pairs, different_pairs = split_pairs(labels)
    if not same_pairs or not different_pairs:
        raise ClusteringError("calibration needs both same-model and different-model pairs")
    channels = list(z_by_channel)
    prior = prior_weights(channels)
    raw: dict[Channel, float] = {}
    diagnostics: list[ChannelDiagnostic] = []

    def pick(matrix: FloatMatrix, pairs: Sequence[tuple[int, int]]) -> FloatMatrix:
        values = np.array([matrix[i, j] for i, j in pairs], dtype=np.float64)
        return np.clip(values[np.isfinite(values)], -Z_CHANNEL_CLIP, Z_CHANNEL_CLIP)

    for channel in channels:
        same = pick(z_by_channel[channel], same_pairs)
        different = pick(z_by_channel[channel], different_pairs)
        if len(same) == 0 or len(different) == 0:
            raw[channel] = 0.0
            diagnostics.append(ChannelDiagnostic(channel, len(same), len(different),
                                                 math.nan, math.nan, math.nan, 0.0))
            continue
        gap = float(different.mean() - np.quantile(same, SAME_PAIR_QUANTILE))
        spread = float(same.var() + different.var()) + VARIANCE_FLOOR
        raw[channel] = max(0.0, gap) / spread
        diagnostics.append(ChannelDiagnostic(
            channel, len(same), len(different), float(same.mean()), float(different.mean()),
            _auc(different, same), raw[channel]))
    top = max(raw.values())
    if top <= 0:
        # Nothing separates on the reference data: keep the priors rather than a zero vector.
        weights = dict(prior)
    else:
        weights = {
            channel: (raw[channel] / top) * ((1.0 - CALIBRATION_SHRINKAGE)
                                             + CALIBRATION_SHRINKAGE * prior[channel])
            for channel in channels
        }
    final = [ChannelDiagnostic(d.channel, d.n_same, d.n_different, d.mean_same, d.mean_different,
                               d.auc, weights[d.channel]) for d in diagnostics]
    return weights, final


def build_linkage(distance: FloatMatrix) -> FloatMatrix:
    """Average-linkage dendrogram of a distance matrix.

    Raises:
        ClusteringError: Fewer than 2 endpoints.
    """
    if distance.shape[0] < MIN_ENDPOINTS_TO_CLUSTER:
        raise ClusteringError("need at least 2 endpoints to cluster")
    return np.asarray(linkage(squareform(distance, checks=False), method=LINKAGE_METHOD),
                      dtype=np.float64)


def cut_dendrogram(tree: FloatMatrix, tau: float) -> IntVector:
    """Cluster ids (>= 0) of the leaves after cutting merges above height `tau`."""
    return np.asarray(fcluster(tree, t=tau, criterion="distance"), dtype=np.int64) - 1


def calibrate_model_threshold(
    distance: FloatMatrix, labels: Sequence[tuple[str, str]]
) -> ThresholdCalibration:
    """Pick the model cut height maximising the model MCC on endpoints of known truth.

    Among equally good heights the midpoint of the plateau is taken (most robust to noise).

    Raises:
        ClusteringError: Fewer than `MIN_ENDPOINTS_TO_CALIBRATE` endpoints, or a label mismatch.
    """
    size = distance.shape[0]
    if size < MIN_ENDPOINTS_TO_CALIBRATE or len(labels) != size:
        raise ClusteringError("calibration needs >= 3 endpoints and one label per endpoint")
    tree = build_linkage(distance)
    heights = np.unique(np.concatenate([[0.0], tree[:, 2]]))
    midpoints = (heights[:-1] + heights[1:]) / 2.0
    candidates = np.concatenate([[heights[0] - 1.0], midpoints, [heights[-1] + 1.0]])
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size) if labels[i][0] == labels[j][0]]
    truth = [labels[i] == labels[j] for i, j in pairs]
    scores: list[float] = []
    for tau in candidates:
        assigned = cut_dendrogram(tree, float(tau))
        guess = [bool(assigned[i] == assigned[j]) for i, j in pairs]
        scores.append(confusion(truth, guess).mcc)
    best = max(scores)
    plateau = [float(tau) for tau, score in zip(candidates, scores, strict=True)
               if score >= best - TIE_TOLERANCE]
    tau_model = float(np.median(plateau))
    below = [float(h) for h in heights if h <= tau_model]
    above = [float(h) for h in heights if h > tau_model]
    return ThresholdCalibration(
        tau_model=tau_model, mcc_model=best,
        lower_margin=tau_model - (max(below) if below else 0.0),
        upper_margin=(min(above) - tau_model) if above else math.inf,
    )


def family_floor_of(family_distance: FloatMatrix) -> float:
    """Highest merge height of endpoints that all belong to ONE family (e.g. every `ref-*`).

    It is the largest in-family distance seen, LoRA 1.5 and fp8 included, so the family cut of
    scored endpoints must stay above it.
    """
    return float(build_linkage(family_distance)[:, 2].max())


def choose_family_threshold(tree: FloatMatrix, floor: float) -> float:
    """Family cut height: the widest multiplicative gap between merge heights above the floor.

    Falls back to `FAMILY_FALLBACK_FACTOR * floor` when no gap reaches `FAMILY_MIN_GAP_RATIO`.
    The floor is at least `FAMILY_FLOOR_MINIMUM` so an uncalibrated run still has a scale.
    """
    lower_bound = max(floor * FAMILY_FLOOR_FRACTION, FAMILY_FLOOR_MINIMUM)
    heights = sorted(float(h) for h in tree[:, 2] if h > lower_bound)
    sequence = [lower_bound, *heights]
    best_ratio, best_cut = 0.0, None
    for low, high in itertools.pairwise(sequence):
        ratio = high / low
        if ratio > best_ratio:
            best_ratio, best_cut = ratio, math.sqrt(low * high)
    if best_cut is None or best_ratio < FAMILY_MIN_GAP_RATIO:
        return FAMILY_FALLBACK_FACTOR * lower_bound
    return best_cut


def cut_two_levels(
    model_distance: FloatMatrix,
    family_distance: FloatMatrix,
    tau_model: float,
    tau_family: float | None = None,
    family_floor: float = 0.0,
) -> TwoLevelClustering:
    """Families from one dendrogram, then models inside each family from another.

    The family question is about the base model: it is answered with a distance that pools every
    channel (a different base model shows up everywhere, so more evidence helps). The model
    question is answered inside a family with the calibrated, disguise-robust distance. Models
    are clustered only among the members of one family, so the labelling is nested by
    construction (same model label implies same family label).

    Args:
        model_distance: `[E, E]` calibrated distance (model question).
        family_distance: `[E, E]` all-channel distance (family question).
        tau_model: Merge height below which two endpoints of a family are the same model.
        tau_family: Family cut; chosen by `choose_family_threshold` when None.
        family_floor: Highest in-family merge height seen on reference endpoints.
    """
    family_tree = build_linkage(family_distance)
    family_cut = (choose_family_threshold(family_tree, family_floor)
                  if tau_family is None else tau_family)
    family_ids = [int(v) for v in cut_dendrogram(family_tree, family_cut)]
    model_ids = [-1] * len(family_ids)
    next_model = 0
    for family in sorted(set(family_ids)):
        members = [i for i, f in enumerate(family_ids) if f == family]
        if len(members) == 1:
            local = np.zeros(1, dtype=np.int64)
        else:
            sub = model_distance[np.ix_(members, members)]
            local = cut_dendrogram(build_linkage(sub), tau_model)
        for position, member in enumerate(members):
            model_ids[member] = next_model + int(local[position])
        next_model += int(local.max()) + 1
    return TwoLevelClustering(family_ids=family_ids, model_ids=model_ids,
                              tau_model=tau_model, tau_family=family_cut)


def ambiguous_pairs(
    combined: FloatMatrix, clustering: TwoLevelClustering, band: float = AMBIGUITY_BAND
) -> list[tuple[int, int, float]]:
    """Same-family pairs whose score is close to the model threshold (candidates for phase 3).

    Returns:
        `(i, j, z)` sorted by closeness to the threshold.
    """
    tau = clustering.tau_model
    found: list[tuple[int, int, float]] = []
    size = combined.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            same_family = clustering.family_ids[i] == clustering.family_ids[j]
            if same_family and np.isfinite(combined[i, j]) and abs(combined[i, j] - tau) <= band * tau:
                found.append((i, j, float(combined[i, j])))
    return sorted(found, key=lambda item: abs(item[2] - tau))
