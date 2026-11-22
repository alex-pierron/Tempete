"""Class-removed training targets and a class-aware proxy gold (team A3, phase 3).

Context (see `reports/phase2/unlearning/team-c` and `docs/hackathon/unlearning.md`): 9 of the 45
forgotten classes hold 129 or 130 forget images, the maximum per class. A gold model retrained
without the forget set therefore never saw them and never predicts them. Team C removes them
after training with a head-bias offset; here the removal is part of the TARGETS the student is
distilled towards, so the whole network (not only the bias) learns it, and per-image behaviour
(second-choice class, confidence) stays that of the tempered original.

Units. The student matches `target_probs` at temperature 1, and the targets are the original's
softmax divided by `target_temperature`: the student logits are the original logits shrunk by that
temperature. Lowering the target log-probability of a class by `delta` nats therefore lowers the
student LOGIT of that class by `delta`: the same unit as team C's `head.bias -= delta`.

Everything here is a pure function of tensors (no model, no device), so it is cheap to test.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.training import temper_probabilities

CLASS_COUNT = 100
CLASS_SIZE = 130
"""Training images per class, inferred by team C from the saturation of the per-class counts."""
FULL_CLASS_TOLERANCE = 1
"""A class counts as fully forgotten from `CLASS_SIZE - FULL_CLASS_TOLERANCE` forget images."""
REMOVED_GOLD_MARGIN = 10.0
"""Logit gap below the best remaining class given to a removed class by the proxy gold (team C)."""
PROBABILITY_FLOOR = 1e-12


class A3TargetError(HackathonError):
    """A target or proxy computation received an input it cannot honour."""


def class_counts(pseudo_labels: Tensor, class_count: int = CLASS_COUNT) -> Tensor:
    """Number of forget images per class, shape [class_count] (int64).

    Args:
        pseudo_labels: [N] integer labels (argmax of the original on the exact forget images).
        class_count: Size of the label space.

    Raises:
        A3TargetError: no label, or a label outside [0, class_count).
    """
    if pseudo_labels.numel() == 0:
        raise A3TargetError("no pseudo-label given")
    if int(pseudo_labels.min()) < 0 or int(pseudo_labels.max()) >= class_count:
        raise A3TargetError(f"labels must lie in [0, {class_count})")
    return torch.bincount(pseudo_labels.long(), minlength=class_count)


def fully_removed_classes(
    counts: Tensor, class_size: int = CLASS_SIZE, tolerance: int = FULL_CLASS_TOLERANCE
) -> tuple[int, ...]:
    """Classes whose forget count reaches `class_size - tolerance` (a gold never saw them)."""
    threshold = class_size - tolerance
    return tuple(int(index) for index in torch.nonzero(counts >= threshold).flatten())


def removal_offsets(
    counts: Tensor,
    removed_classes: Sequence[int],
    delta_full: float,
    delta_partial: float,
    class_size: int = CLASS_SIZE,
) -> Tensor:
    """Per-class log-probability offset (>= 0, nats) subtracted from the targets, shape [C].

    Fully removed classes get `delta_full`. Every other class present in the forget set gets
    `delta_partial * count / class_size` (a gold saw `class_size - count` images of it, so its
    prior is lower, in proportion to the share that was forgotten). A class absent from the
    forget set gets exactly 0.

    Raises:
        A3TargetError: negative magnitude, or a removed class outside the label space.
    """
    if delta_full < 0 or delta_partial < 0:
        raise A3TargetError("offsets are magnitudes and must be >= 0")
    class_count = len(counts)
    if any(not 0 <= index < class_count for index in removed_classes):
        raise A3TargetError(f"removed class outside [0, {class_count})")
    fraction = (counts.double() / class_size).clamp(max=1.0)
    offsets = delta_partial * fraction
    offsets[list(removed_classes)] = delta_full
    return offsets.float()


def apply_class_offsets(target: Tensor, offsets: Tensor) -> Tensor:
    """Multiply class c of every row by exp(-offsets[c]) and renormalise (rows still sum to 1).

    Equivalent to subtracting `offsets[c]` from the logit of class c: the ranking of the other
    classes, hence the "second choice" class of the original, is untouched.
    """
    if target.shape[1] != offsets.shape[0]:
        raise A3TargetError(f"target has {target.shape[1]} classes, offsets {offsets.shape[0]}")
    shifted = target * torch.exp(-offsets.to(target.dtype)).unsqueeze(0)
    return shifted / shifted.sum(dim=1, keepdim=True).clamp_min(PROBABILITY_FLOOR)


def finish_targets(
    base: Tensor,
    offsets: Tensor,
    flatten_rows: Tensor | None = None,
    flatten: float = 1.0,
) -> Tensor:
    """Class-removal offsets, then optional flattening, on any base targets [N, C].

    Args:
        base: Per-image target distributions (tempered view average, or the research trainer's
            calibrated targets).
        offsets: [C] output of `removal_offsets`.
        flatten_rows: Optional boolean [N]: images whose remaining-class distribution is flattened.
            A gold that never saw the class of an image has no class to put it in: it hesitates
            between several farther classes (team C: higher loss and entropy than the original's
            fallback), so these rows get an extra temperature `flatten` after the removal.
        flatten: Extra temperature (>= 1; 1 changes nothing).

    Raises:
        A3TargetError: `flatten` below 1 or a mask of the wrong length.
    """
    if flatten < 1.0:
        raise A3TargetError("flatten must be >= 1")
    targets = apply_class_offsets(base, offsets)
    if flatten_rows is None or flatten == 1.0:
        return targets
    if flatten_rows.shape != (targets.shape[0],):
        raise A3TargetError("flatten_rows must have one entry per image")
    flat = temper_probabilities(targets, flatten)
    return torch.where(flatten_rows.unsqueeze(1), flat, targets)


def build_removal_targets(
    view_probs: Tensor,
    temperature: float,
    offsets: Tensor,
    flatten_rows: Tensor | None = None,
    flatten: float = 1.0,
) -> Tensor:
    """Per-image targets [N, C]: tempered view average, then `finish_targets`.

    Args:
        view_probs: [N, K, C] softmax (T = 1) of the original on K strong views of every image
            (the view cache of the research runner).
        temperature: Target temperature (the gold_soft knob, 2.0 at the kit default).
        offsets: [C] output of `removal_offsets`.
        flatten_rows: See `finish_targets`.
        flatten: See `finish_targets`.

    Raises:
        A3TargetError: bad shape, non-positive temperature, or the errors of `finish_targets`.
    """
    if view_probs.ndim != 3:
        raise A3TargetError("view_probs must have shape [N, K, C]")
    if temperature <= 0:
        raise A3TargetError("temperature must be > 0")
    views = view_probs.float()
    tempered = torch.stack(
        [temper_probabilities(views[:, k], temperature) for k in range(views.shape[1])]
    ).mean(dim=0)
    return finish_targets(tempered, offsets, flatten_rows, flatten)


def removed_class_gold_logits(
    orig_logits: Tensor, removed_classes: Sequence[int], margin: float = REMOVED_GOLD_MARGIN
) -> Tensor:
    """Proxy gold logits [N, C]: the original with the removed classes `margin` below the best other.

    This is team C's H_C hypothesis (divided by a temperature `tau` by the caller). It encodes ONLY
    the class-level effect; it is a proxy of the gold, never the gold.
    """
    if margin < 0:
        raise A3TargetError("margin must be >= 0")
    mask = torch.zeros(orig_logits.shape[1], dtype=torch.bool)
    mask[list(removed_classes)] = True
    remaining_best = orig_logits.masked_fill(mask.to(orig_logits.device), float("-inf")).amax(dim=1)
    gold = orig_logits.clone()
    gold[:, mask] = (remaining_best - margin).unsqueeze(1)
    return gold


def removed_mask(labels: Tensor, removed_classes: Sequence[int]) -> Tensor:
    """Boolean [N]: the pseudo-label of the image is one of the removed classes."""
    return torch.isin(labels, torch.tensor(list(removed_classes), device=labels.device))


def class_aware_view_scale(
    views: Tensor,
    labels: Tensor,
    counts: Tensor,
    removed_classes: Sequence[int],
    strength: float,
    class_size: int = CLASS_SIZE,
) -> Tensor:
    """Scale the label-class probability of every view by `1 - strength * forgotten_share`.

    The research trainer's calibrated law ranks the images by how much their strong views still
    support the original class and makes the most fragile ones "wrong". Scaling the label column
    BEFORE that ranking pushes the images of heavily forgotten classes (a gold saw few images of
    them, so it is less accurate there) towards "wrong". Only the ranking is affected: the law
    masks the label column and renormalises the rest, so the ratios between the other classes,
    hence the "second choice", are unchanged. The fully removed classes get no scaling (their
    targets are handled by the removal offsets), neither do classes absent from the forget set.

    Args:
        views: [N, K, C] softmax of the original on K strong views of every image.
        labels: [N] pseudo-labels.
        counts: [C] forget images per class.
        removed_classes: Fully removed classes.
        strength: In [0, 1]; 0 returns the views unchanged.

    Raises:
        A3TargetError: strength outside [0, 1] or shapes that do not match.
    """
    if not 0.0 <= strength <= 1.0:
        raise A3TargetError("strength must lie in [0, 1]")
    if views.ndim != 3 or labels.shape[0] != views.shape[0] or counts.shape[0] != views.shape[2]:
        raise A3TargetError("views [N, K, C], labels [N] and counts [C] must agree")
    if strength == 0.0:
        return views
    share = (counts.double() / class_size).clamp(max=1.0)
    share[list(removed_classes)] = 0.0
    scale = (1.0 - strength * share[labels]).to(views.dtype)
    scaled = views.clone()
    index = torch.arange(views.shape[0])
    scaled[index, :, labels] = views[index, :, labels] * scale.unsqueeze(1)
    return scaled


def fallback_targets(
    exact_logits: Tensor, removed_classes: Sequence[int], temperature: float
) -> Tensor:
    """Softmax of the original's exact-image logits with the removed classes masked, shape [N, C].

    The "fallback" of an image of a removed class: what the ORIGINAL would answer if those classes
    did not exist. No view average, no crop (team C's suggestion, because the averaged and tempered
    targets scramble the fallback class). The removed classes get probability exactly 0.

    Raises:
        A3TargetError: non-positive temperature or a removed class outside the label space.
    """
    if temperature <= 0:
        raise A3TargetError("temperature must be > 0")
    class_count = exact_logits.shape[1]
    if any(not 0 <= index < class_count for index in removed_classes):
        raise A3TargetError(f"removed class outside [0, {class_count})")
    masked = exact_logits.float().clone()
    masked[:, list(removed_classes)] = float("-inf")
    return torch.softmax(masked / temperature, dim=1)


CLASS_GROUPS = ((0, 1, 2, 3, 4), (5, 6, 7, 8), (29, 30, 31, 32, 33), (34, 35, 36, 37, 38))
"""Groups of related classes (the organisers' class grouping, relayed by team C): they overlap the
forget set exactly. In each group some classes are forgotten entirely and the others are PRESENT in
the gold's training set: a gold that never saw class 6, 7 or 8 puts such an image on its only present
sibling, class 5; 29, 30, 31 and 33 on class 32; class 0 on 1 to 4; class 38 on 34 to 37."""


def present_siblings(
    removed_classes: Sequence[int], groups: Sequence[Sequence[int]] = CLASS_GROUPS
) -> dict[int, tuple[int, ...]]:
    """For each removed class, the other classes of its group that are NOT removed."""
    removed = set(removed_classes)
    siblings: dict[int, tuple[int, ...]] = {}
    for group in groups:
        alive = tuple(index for index in group if index not in removed)
        for index in group:
            if index in removed:
                siblings[index] = alive
    return siblings


def sibling_targets(
    exact_logits: Tensor,
    labels: Tensor,
    removed_classes: Sequence[int],
    sibling_mass: float,
    temperature: float,
    groups: Sequence[Sequence[int]] = CLASS_GROUPS,
) -> Tensor:
    """Targets [N, C] for the images of the removed classes: siblings plus the original fallback.

    `sibling_mass` of the probability goes to the PRESENT siblings of the image's class, spread by
    the ORIGINAL's exact-image logits restricted to them (so it keeps depending on the image; a
    class with a single present sibling puts all of it there); the rest is `fallback_targets`
    (the original with every removed class masked). A removed class with no present sibling uses
    the fallback alone. Rows of other classes are returned as the plain fallback and are meant to
    be ignored by the caller (it only replaces the rows of removed-class images).

    Raises:
        A3TargetError: mass outside [0, 1], non-positive temperature, or labels of the wrong length.
    """
    if not 0.0 <= sibling_mass <= 1.0:
        raise A3TargetError("sibling_mass must lie in [0, 1]")
    if labels.shape[0] != exact_logits.shape[0]:
        raise A3TargetError("labels must have one entry per image")
    fallback = fallback_targets(exact_logits, removed_classes, temperature)
    siblings = present_siblings(removed_classes, groups)
    out = fallback.clone()
    logits = exact_logits.float()
    for removed, alive in siblings.items():
        rows = torch.nonzero(labels == removed).flatten()
        if len(alive) == 0 or len(rows) == 0:
            continue
        spread = torch.softmax(logits[rows][:, list(alive)] / temperature, dim=1)
        onto = torch.zeros(len(rows), exact_logits.shape[1])
        onto[:, list(alive)] = spread
        out[rows] = sibling_mass * onto + (1.0 - sibling_mass) * fallback[rows]
    return out
