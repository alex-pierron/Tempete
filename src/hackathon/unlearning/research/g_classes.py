"""Class structure of the forget set (team G): saturated classes, forgotten fractions, siblings.

Same facts as team C's `c_classes` (re-implemented here because that module lives on another
branch): the 3500 forget images cover 45 of the 100 classes; 9 of them hold (almost) every training
image of their class, so a model retrained without the forget set has never seen them. Siblings are
the classes whose classifier rows are close (average-linkage clustering of the centred, normalised
head rows): a gold that never saw class `c` sends its images to a sibling of `c`.

Everything is derived from pseudo-labels (argmax of the original on the exact forget images, which
it memorised) and from the original head weight; nothing here needs a GPU.
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
"""A class counts as fully forgotten from `CLASS_SIZE - FULL_CLASS_TOLERANCE` images (one image can
be pseudo-labelled as a neighbour class by the original)."""
GROUPED_CLASSES = 60
"""Only the first rows are clustered: the classes above are isolated in this task (measured by team C)."""
GROUP_COUNT = 14

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class ClassStructureError(HackathonError):
    """The pseudo-labels or the head weight do not describe a valid class structure."""


def class_counts(
    pseudo_labels: Sequence[int] | IntArray, class_count: int = CLASS_COUNT
) -> IntArray:
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


def cosine_groups(
    weight: FloatArray, grouped_classes: int = GROUPED_CLASSES, group_count: int = GROUP_COUNT
) -> list[list[int]]:
    """Sibling groups: average-linkage clustering of the centred, normalised head rows.

    Returns `group_count` groups of class indices, each sorted, groups sorted by first member.

    Raises:
        ClassStructureError: fewer rows than `grouped_classes` or a non-positive group count.
    """
    if weight.shape[0] < grouped_classes or group_count < 1:
        raise ClassStructureError(
            "weight needs at least `grouped_classes` rows and group_count >= 1"
        )
    rows = weight[:grouped_classes] - weight.mean(axis=0, keepdims=True)
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    similarity = rows @ rows.T
    clusters: list[list[int]] = [[index] for index in range(grouped_classes)]
    while len(clusters) > group_count:
        best, pair = -np.inf, (0, 1)
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                value = float(similarity[np.ix_(clusters[first], clusters[second])].mean())
                if value > best:
                    best, pair = value, (first, second)
        first, second = pair
        clusters[first] = clusters[first] + clusters[second]
        del clusters[second]
    return sorted((sorted(cluster) for cluster in clusters), key=lambda cluster: cluster[0])


def present_siblings(
    groups: Sequence[Sequence[int]], saturated: Sequence[int]
) -> dict[int, list[int]]:
    """For each saturated class that sits in a group: the other group members that are NOT saturated.

    A saturated class whose group has no present member (or that is in no group) is left out: the
    caller falls back to the original's own off-label mass for it.
    """
    saturated_set = set(saturated)
    siblings: dict[int, list[int]] = {}
    for members in groups:
        present = [member for member in members if member not in saturated_set]
        if not present:
            continue
        for member in members:
            if member in saturated_set:
                siblings[member] = list(present)
    return siblings
