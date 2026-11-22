"""Class structure of the forget set (team C): which classes a gold model has (almost) never seen.

Measured fact (see `reports/phase2/unlearning/team-c`): the 3500 forget images cover 45 of the 100
classes, 9 of them with 129 or 130 images (the maximum per class) and the others with 41 to 104.
If every class holds about `CLASS_SIZE` training images, a gold model retrained without the forget
set has seen NONE of the 9 saturated classes and only `CLASS_SIZE - count` images of the others.
Such a gold never predicts a missing class, whatever the image: that is a class-level effect a
bias correction on the classifier head reproduces without any training.

This module only derives class statistics from pseudo-labels (argmax of the original on the exact
forget images, which the original has memorised, so they are the true labels up to a handful of
images) and builds the per-class bias shift; applying it to a state dict is `c_families`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hackathon.common.errors import HackathonError

CLASS_COUNT = 100
CLASS_SIZE = 130
"""Training images per class, inferred from the saturation of the largest per-class forget counts."""
FULL_CLASS_TOLERANCE = 1
"""A class counts as fully forgotten from `CLASS_SIZE - FULL_CLASS_TOLERANCE` images (one image of
a class can be pseudo-labelled as a neighbour class by the original)."""

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class ClassStructureError(HackathonError):
    """The pseudo-labels do not describe a valid class histogram."""


def class_counts(pseudo_labels: Sequence[int] | IntArray, class_count: int = CLASS_COUNT) -> IntArray:
    """Number of forget images per class, shape [class_count].

    Raises:
        ClassStructureError: empty input or a label outside [0, class_count).
    """
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if labels.size == 0:
        raise ClassStructureError("no pseudo-label given")
    if labels.min() < 0 or labels.max() >= class_count:
        raise ClassStructureError(f"labels must lie in [0, {class_count})")
    return np.bincount(labels, minlength=class_count).astype(np.int64)


def forgotten_fraction(counts: IntArray, class_size: int = CLASS_SIZE) -> FloatArray:
    """Fraction of each class that sits in the forget set, clipped to 1."""
    return np.minimum(counts.astype(np.float64) / class_size, 1.0)


def fully_forgotten_classes(
    counts: IntArray, class_size: int = CLASS_SIZE, tolerance: int = FULL_CLASS_TOLERANCE
) -> list[int]:
    """Classes whose count reaches `class_size - tolerance`: a gold model never saw them."""
    threshold = class_size - tolerance
    return [int(index) for index in np.flatnonzero(counts >= threshold)]


def bias_shift_vector(
    counts: IntArray,
    full_shift: float,
    partial_shift: float = 0.0,
    class_size: int = CLASS_SIZE,
) -> FloatArray:
    """Per-class additive bias: `-full_shift` on fully forgotten classes, graded elsewhere.

    Partially forgotten classes get `-partial_shift * fraction` (fraction of the class that is in
    the forget set); a class absent from the forget set gets exactly 0. Fully forgotten classes
    get `-full_shift` whatever `partial_shift` is.

    Raises:
        ClassStructureError: negative magnitudes (a positive shift would raise the prior).
    """
    if full_shift < 0 or partial_shift < 0:
        raise ClassStructureError("shift magnitudes must be >= 0 (they are subtracted)")
    fraction = forgotten_fraction(counts, class_size)
    shift = -partial_shift * fraction
    for index in fully_forgotten_classes(counts, class_size):
        shift[index] = -full_shift
    return shift


def centred_mean_direction(
    saturated_features: NDArray[np.float64], all_features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Unit vector from the mean feature of all forget images to that of the saturated-class ones.

    Raises:
        ClassStructureError: empty input or identical means (no direction to remove).
    """
    if saturated_features.size == 0 or all_features.size == 0:
        raise ClassStructureError("features are empty")
    direction = saturated_features.mean(axis=0) - all_features.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ClassStructureError("the saturated-class mean equals the global mean")
    return np.asarray(direction / norm, dtype=np.float64)


GROUPED_CLASSES = 60
GROUP_COUNT = 14


def cosine_groups(
    weight: NDArray[np.float64], grouped_classes: int = GROUPED_CLASSES, group_count: int = GROUP_COUNT
) -> list[list[int]]:
    """Sibling groups: average-linkage clustering of the centred, normalised head rows.

    Only the first `grouped_classes` rows are clustered (the classes above are isolated in this
    task, per a measurement by a teammate). Returns `group_count` groups of class indices, sorted.

    Raises:
        ClassStructureError: fewer rows than `grouped_classes` or a non-positive group count.
    """
    if weight.shape[0] < grouped_classes or group_count < 1:
        raise ClassStructureError("weight must have at least `grouped_classes` rows and group_count >= 1")
    rows = weight[:grouped_classes] - weight.mean(axis=0, keepdims=True)
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    similarity = rows @ rows.T
    clusters = [[i] for i in range(grouped_classes)]
    while len(clusters) > group_count:
        best, pair = -np.inf, (0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                value = float(similarity[np.ix_(clusters[a], clusters[b])].mean())
                if value > best:
                    best, pair = value, (a, b)
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return sorted((sorted(cluster) for cluster in clusters), key=lambda cluster: cluster[0])
