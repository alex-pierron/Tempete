"""Calibrated per-image targets (team G): the distribution a retrained model would output, image by image.

The official forget score compares, per image, the candidate's loss and full distribution with those
of a gold model retrained without the forget set. We cannot see the gold; we can only state what
we believe about it and build every image's target from that belief. Four beliefs, all explicit:

1. **Loss law (calibrated).** The gold's loss on the exact images has mean `loss_mean` and standard
   deviation `loss_std` (team C fitted 2.25 and 3.0 on 13 public scores). The budget of each image
   is its own target loss `ell_i = -log q_i(y_i)`: large for images of classes the gold never saw,
   small for the others, graded inside each group by the fragility of the image (how little the
   original's label survives on strong views) and, for partly forgotten classes, by the share of
   the class that was forgotten (the gold saw fewer images of those). The two group scales are
   solved so that the law has EXACTLY the requested mean and std (clipping included).
2. **Routing.** A gold never predicts a class it never saw. On images of a saturated class, the
   off-label mass goes to the PRESENT siblings of that class (`routing_share` of it); every
   saturated class is removed from the off-label mass of every other image too.
3. **Entropy.** The off-label shape is sharpened by one global exponent `gamma`, solved so that the
   mean entropy of the targets at the scoring temperature equals `entropy_target` (team F measured
   our files too flat: 3.1 nats against 1.6-1.9 for a plausible gold).
4. **Off-label ranking** comes from the original itself: its tempered mass on strong views and on
   the exact image (which class a model that did not memorise the image would pick).

`build_target_bank` is a pure function of cached tensors (no GPU); `bank_statistics` reports every
quantity the beliefs above constrain, so a bank can be audited before it is trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.research.g_classes import (
    CLASS_COUNT,
    forgotten_fraction,
    fully_forgotten_classes,
    present_siblings,
)
from hackathon.unlearning.training import temper_probabilities

PROBABILITY_FLOOR = 1e-12
BISECTION_ITERATIONS = 80
GAMMA_BOUNDS = (0.2, 12.0)
"""Search interval of the sharpening exponent of the off-label shape."""
SIBLING_UNIFORM_SHARE = 0.2
"""Share of the routed mass spread uniformly over the present siblings (the rest follows the original)."""
FRAGILITY_OFFSET = 0.5
"""Fragility shape = offset + rank in [0, 1]: a 3x spread between the most and least fragile image."""
EXACT_VIEW_MIX = 0.5
"""Weight of the original's own tempered off-label mass against its strong-view mass."""
BLUR_LOW = 0.1
BLUR_HIGH = 0.9
"""p(true class) inside (BLUR_LOW, BLUR_HIGH) is "blur"; above is sure-right, below is sure-wrong."""
FloatArray = NDArray[np.float64]


class TargetError(HackathonError):
    """The requested belief cannot be honoured (unreachable loss law, empty group, bad shapes)."""


class TargetSpec(BaseModel):
    """Every belief about the gold, with its default (JSON-friendly, frozen)."""

    model_config = ConfigDict(frozen=True)

    loss_mean: float = Field(default=2.25, gt=0.0)
    loss_std: float = Field(default=3.0, gt=0.0)
    loss_floor: float = Field(
        default=0.02, gt=0.0, description="Smallest target loss (label mass <= exp(-floor))."
    )
    loss_ceiling: float = Field(
        default=10.0, gt=0.0, description="Largest target loss (label mass >= exp(-ceiling))."
    )
    forgotten_exponent: float = Field(default=1.0, ge=0.0)
    routing_share: float = Field(default=0.7, ge=0.0, le=1.0)
    entropy_target: float | None = Field(default=1.75, gt=0.0)
    entropy_temperature: float = Field(default=SCORING_TEMPERATURE, gt=0.0)
    view_temperature: float = Field(default=2.0, gt=0.0)
    mask_saturated_offlabel: bool = True
    tail_k: int = Field(default=5, ge=1)
    calibrate_loss: bool = True
    kind: Literal["calibrated", "v2"] = Field(
        default="calibrated",
        description="'v2' = the plain tempered strong-view average (the gold_soft target).",
    )
    altered_loss_mean: float = Field(
        default=3.1, gt=0.0, description="Gold loss mean on strong crops."
    )
    altered_loss_std: float = Field(
        default=3.0, gt=0.0, description="ASSUMED equal to the exact-group std."
    )
    other_tail_fraction: float = Field(
        default=0.15,
        ge=0.0,
        lt=1.0,
        description="Share of the most fragile 'other' images the gold gets wrong.",
    )
    saturated_spread: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="Relative spread of the loss inside the saturated group (a class-level effect).",
    )
    other_shape_power: float = Field(
        default=1.0,
        gt=0.0,
        description="Power on the fragility shape of the 'other' group: >1 makes the loss law "
        "bimodal (most images near the floor, a few carrying the loss), i.e. fewer blur images.",
    )
    other_tail_boost: float = Field(
        default=8.0, ge=1.0, description="Loss multiplier of that tail."
    )


@dataclass(frozen=True)
class TargetBank:
    """Per-image targets [N, C] (float32, rows sum to 1) and what produced them.

    `targets` is the target of the exact image (and of mild alterations); `targets_altered` is the
    target of strong crops, drawn from the altered-group loss law (deeper than the exact one).
    """

    targets: Tensor
    targets_altered: Tensor
    loss_targets: Tensor
    gamma: float
    scale_saturated: float
    scale_other: float
    statistics: dict[str, float]


def _rank01(values: FloatArray) -> FloatArray:
    """Ranks of `values` scaled to [0, 1] (ties broken by order); a single value maps to 0."""
    count = len(values)
    if count <= 1:
        return np.zeros(count)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(count)
    ranks[order] = np.arange(count) / (count - 1)
    return ranks


def _unit_mean(shape: FloatArray) -> FloatArray:
    """`shape` rescaled to mean 1 (a group of identical weights becomes all ones)."""
    mean = float(shape.mean())
    if mean <= 0.0:
        raise TargetError("a shape with non-positive mean cannot carry a loss scale")
    return shape / mean


def _clipped_law(
    shape_saturated: FloatArray,
    shape_other: FloatArray,
    scale_saturated: float,
    scale_other: float,
    floor: float,
    ceiling: float,
) -> tuple[FloatArray, FloatArray]:
    return (
        np.clip(scale_saturated * shape_saturated, floor, ceiling),
        np.clip(scale_other * shape_other, floor, ceiling),
    )


def _law_moments(
    shape_saturated: FloatArray,
    shape_other: FloatArray,
    scale_saturated: float,
    scale_other: float,
    floor: float,
    ceiling: float,
) -> tuple[float, float]:
    saturated, other = _clipped_law(
        shape_saturated, shape_other, scale_saturated, scale_other, floor, ceiling
    )
    merged = np.concatenate([saturated, other])
    return float(merged.mean()), float(merged.std())


def solve_group_scales(
    shape_saturated: FloatArray,
    shape_other: FloatArray,
    mean: float,
    std: float,
    floor: float,
    ceiling: float,
) -> tuple[float, float]:
    """Scales (saturated, other) such that the clipped loss law has exactly `mean` and `std`.

    Nested bisection: for a given `other` scale the saturated scale is the one that hits the mean
    (the mean is increasing in it); the std then decreases as the `other` scale grows (the gap
    between the groups closes), so the outer bisection finds the std.

    Raises:
        TargetError: empty group, or a std the model cannot reach (too large: even a vanishing
            `other` scale leaves it short; too small: the groups cannot be brought together).
    """
    if len(shape_saturated) == 0 or len(shape_other) == 0:
        raise TargetError("both the saturated and the other group need at least one image")

    def saturated_for(scale_other: float) -> float:
        low, high = 0.0, ceiling / float(shape_saturated.min()) + 1.0
        for _ in range(BISECTION_ITERATIONS):
            middle = 0.5 * (low + high)
            value, _ = _law_moments(
                shape_saturated, shape_other, middle, scale_other, floor, ceiling
            )
            if value < mean:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def std_for(scale_other: float) -> float:
        scale_sat = saturated_for(scale_other)
        return _law_moments(shape_saturated, shape_other, scale_sat, scale_other, floor, ceiling)[1]

    low, high = floor / 10.0, mean
    if std_for(low) < std:
        raise TargetError(f"loss std {std} unreachable: the widest law has {std_for(low):.3f}")
    if std_for(high) > std:
        raise TargetError(f"loss std {std} unreachable: the narrowest law has {std_for(high):.3f}")
    for _ in range(BISECTION_ITERATIONS):
        middle = 0.5 * (low + high)
        if std_for(middle) > std:
            low = middle
        else:
            high = middle
    scale_other = 0.5 * (low + high)
    return saturated_for(scale_other), scale_other


def entropy_nats(probabilities: Tensor) -> Tensor:
    """Shannon entropy in nats per row of a [N, C] probability matrix."""
    clamped = probabilities.clamp_min(PROBABILITY_FLOOR)
    return -(probabilities * clamped.log()).sum(dim=1)


def tail_mass(probabilities: Tensor, top_k: int) -> Tensor:
    """Probability mass outside the `top_k` most likely classes, per row."""
    return 1.0 - probabilities.topk(top_k, dim=1).values.sum(dim=1)


def _mix_target(label_mass: Tensor, onehot: Tensor, off_label: Tensor, gamma: float) -> Tensor:
    """`rho * onehot + (1 - rho) * normalised(off_label ** gamma)` (rows sum to 1)."""
    sharpened = off_label.clamp_min(PROBABILITY_FLOOR).pow(gamma) * (off_label > 0)
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(PROBABILITY_FLOOR)
    return label_mass[:, None] * onehot + (1.0 - label_mass)[:, None] * sharpened


def _solve_gamma(label_mass: Tensor, onehot: Tensor, off_label: Tensor, spec: TargetSpec) -> float:
    """Sharpening exponent that brings the mean entropy at `entropy_temperature` to the target.

    Entropy decreases with gamma (a sharper off-label shape puts its mass on fewer classes); an
    unreachable target is clamped to the nearest bound of `GAMMA_BOUNDS` (the statistics show it).
    """
    assert spec.entropy_target is not None

    def mean_entropy(gamma: float) -> float:
        target = _mix_target(label_mass, onehot, off_label, gamma)
        tempered = temper_probabilities(target, spec.entropy_temperature)
        return float(entropy_nats(tempered).mean())

    low, high = GAMMA_BOUNDS
    if mean_entropy(low) < spec.entropy_target:
        return low
    if mean_entropy(high) > spec.entropy_target:
        return high
    for _ in range(BISECTION_ITERATIONS):
        middle = 0.5 * (low + high)
        if mean_entropy(middle) > spec.entropy_target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _off_label_mass(
    view_probs: Tensor, exact_logits: Tensor, labels: Tensor, spec: TargetSpec
) -> Tensor:
    """The original's own off-label ranking [N, C]: tempered strong-view and exact mass, label removed."""
    views = torch.stack(
        [
            temper_probabilities(view_probs[:, index], spec.view_temperature)
            for index in range(view_probs.shape[1])
        ]
    ).mean(dim=0)
    exact = temper_probabilities(torch.softmax(exact_logits.float(), dim=1), spec.view_temperature)
    mixed = (1.0 - EXACT_VIEW_MIX) * views + EXACT_VIEW_MIX * exact
    onehot = torch.nn.functional.one_hot(labels, mixed.shape[1]).bool()
    mixed = mixed.masked_fill(onehot, 0.0)
    return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(PROBABILITY_FLOOR)


def _route_to_siblings(
    off_label: Tensor,
    labels: Tensor,
    siblings: dict[int, list[int]],
    routing_share: float,
) -> Tensor:
    """Send `routing_share` of the off-label mass of saturated-class images to their present siblings."""
    if routing_share == 0.0 or not siblings:
        return off_label
    routed = off_label.clone()
    for class_index, present in siblings.items():
        rows = (labels == class_index).nonzero(as_tuple=True)[0]
        if len(rows) == 0:
            continue
        sibling_mass = torch.zeros_like(off_label[rows])
        sibling_mass[:, present] = off_label[rows][:, present]
        sibling_mass = sibling_mass / sibling_mass.sum(dim=1, keepdim=True).clamp_min(
            PROBABILITY_FLOOR
        )
        uniform = torch.zeros_like(sibling_mass)
        uniform[:, present] = 1.0 / len(present)
        followed = (1.0 - SIBLING_UNIFORM_SHARE) * sibling_mass + SIBLING_UNIFORM_SHARE * uniform
        routed[rows] = routing_share * followed + (1.0 - routing_share) * off_label[rows]
    return routed


def _loss_law(
    spec: TargetSpec,
    shape_saturated: FloatArray,
    shape_other: FloatArray,
    mean: float,
    std: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Per-image target losses (saturated group, other group) and the two solved scales."""
    if spec.calibrate_loss:
        scale_saturated, scale_other = solve_group_scales(
            shape_saturated, shape_other, mean, std, spec.loss_floor, spec.loss_ceiling
        )
    else:
        scale_saturated = scale_other = mean
    law_saturated, law_other = _clipped_law(
        shape_saturated, shape_other, scale_saturated, scale_other, spec.loss_floor,
        spec.loss_ceiling,
    )  # fmt: skip
    return law_saturated, law_other, scale_saturated, scale_other


def _plain_view_bank(
    view_probs: Tensor, exact_logits: Tensor, spec: TargetSpec
) -> tuple[Tensor, Tensor]:
    """The gold_soft target: strong-view mass tempered at `view_temperature`, averaged (no beliefs)."""
    tempered = torch.stack(
        [
            temper_probabilities(view_probs[:, index].float(), spec.view_temperature)
            for index in range(view_probs.shape[1])
        ]
    ).mean(dim=0)
    labels = exact_logits.argmax(dim=1)
    loss = -tempered.gather(1, labels[:, None]).squeeze(1).clamp_min(PROBABILITY_FLOOR).log()
    return tempered.float(), loss.float()


def build_target_bank(
    view_probs: Tensor,
    exact_logits: Tensor,
    counts: NDArray[np.int64],
    groups: Sequence[Sequence[int]],
    spec: TargetSpec,
) -> TargetBank:
    """Per-image calibrated targets from the original's cached behaviour (see the module docstring).

    Args:
        view_probs: [N, K, C] softmax (T = 1) of the original on K strong views of every image.
        exact_logits: [N, C] logits of the original on the exact images (argmax = pseudo-label).
        counts: [C] forget images per class (from the pseudo-labels).
        groups: Sibling groups (`g_classes.cosine_groups`).
        spec: The beliefs. `kind="v2"` returns the plain gold_soft target instead (the baseline).

    Raises:
        TargetError: inconsistent shapes, or a loss law that cannot be reached.
    """
    count, _, classes = view_probs.shape
    if exact_logits.shape != (count, classes) or classes != CLASS_COUNT:
        raise TargetError(
            f"shape mismatch: views {tuple(view_probs.shape)}, logits {tuple(exact_logits.shape)}"
        )
    if spec.kind == "v2":
        plain, plain_loss = _plain_view_bank(view_probs, exact_logits, spec)
        return TargetBank(
            plain, plain, plain_loss, 1.0, spec.loss_mean, spec.loss_mean,
            bank_statistics(plain, exact_logits, counts, groups, spec),
        )  # fmt: skip
    labels = exact_logits.argmax(dim=1)
    saturated = fully_forgotten_classes(counts)
    is_saturated = torch.isin(labels, torch.tensor(saturated, dtype=torch.long))
    fractions = torch.from_numpy(forgotten_fraction(counts)).double()[labels].numpy()
    support = view_probs.float().mean(dim=1).gather(1, labels[:, None]).squeeze(1).double().numpy()
    saturated_mask = is_saturated.numpy()
    fragility = np.zeros(count)
    for mask in (saturated_mask, ~saturated_mask):
        if mask.any():
            fragility[mask] = _rank01(1.0 - support[mask])
    shape_all = FRAGILITY_OFFSET + fragility
    shape_saturated = _unit_mean(
        1.0 + spec.saturated_spread * (2.0 * fragility[saturated_mask] - 1.0)
    )
    graded = (fractions[~saturated_mask] / max(float(fractions[~saturated_mask].mean()), 1e-9)) ** (
        spec.forgotten_exponent
    )
    tail = np.where(
        fragility[~saturated_mask] >= 1.0 - spec.other_tail_fraction, spec.other_tail_boost, 1.0
    )
    shape_other = _unit_mean((shape_all[~saturated_mask] ** spec.other_shape_power) * graded * tail)
    onehot = torch.nn.functional.one_hot(labels, classes).float()
    off_label = _off_label_mass(view_probs.float(), exact_logits, labels, spec)
    if spec.mask_saturated_offlabel and saturated:
        off_label[:, saturated] = 0.0
        off_label = off_label / off_label.sum(dim=1, keepdim=True).clamp_min(PROBABILITY_FLOOR)
    off_label = _route_to_siblings(
        off_label, labels, present_siblings(groups, saturated), spec.routing_share
    )
    banks: list[Tensor] = []
    losses: list[Tensor] = []
    scales: tuple[float, float] = (0.0, 0.0)
    gamma = 1.0
    for index, (mean, std) in enumerate(
        ((spec.loss_mean, spec.loss_std), (spec.altered_loss_mean, spec.altered_loss_std))
    ):
        law_saturated, law_other, scale_saturated, scale_other = _loss_law(
            spec, shape_saturated, shape_other, mean, std
        )
        loss = np.zeros(count)
        loss[saturated_mask], loss[~saturated_mask] = law_saturated, law_other
        label_mass = torch.from_numpy(np.exp(-loss)).float()
        if index == 0:
            scales = (scale_saturated, scale_other)
            if spec.entropy_target is not None:
                gamma = _solve_gamma(label_mass, onehot, off_label, spec)
        banks.append(_mix_target(label_mass, onehot, off_label, gamma).float())
        losses.append(torch.from_numpy(loss).float())
    return TargetBank(
        banks[0], banks[1], losses[0], gamma, scales[0], scales[1],
        {
            **bank_statistics(banks[0], exact_logits, counts, groups, spec),
            **{
                f"altered_{key}": value
                for key, value in bank_statistics(banks[1], exact_logits, counts, groups, spec).items()
            },
        },
    )  # fmt: skip


def bank_statistics(
    targets: Tensor,
    exact_logits: Tensor,
    counts: NDArray[np.int64],
    groups: Sequence[Sequence[int]],
    spec: TargetSpec,
) -> dict[str, float]:
    """Everything the beliefs constrain, plus the distance of the targets from the original."""
    labels = exact_logits.argmax(dim=1)
    saturated = fully_forgotten_classes(counts)
    is_saturated = torch.isin(labels, torch.tensor(saturated, dtype=torch.long))
    own_loss = -targets.gather(1, labels[:, None]).squeeze(1).clamp_min(PROBABILITY_FLOOR).log()
    tempered = temper_probabilities(targets, spec.entropy_temperature)
    original_t1 = torch.softmax(exact_logits.float(), dim=1)
    kl_to_original = (
        targets
        * (
            targets.clamp_min(PROBABILITY_FLOOR).log()
            - original_t1.clamp_min(PROBABILITY_FLOOR).log()
        )
    ).sum(dim=1)
    siblings = present_siblings(groups, saturated)
    fallback_hits: list[float] = []
    for row in is_saturated.nonzero(as_tuple=True)[0].tolist():
        masked = targets[row].clone()
        masked[saturated] = 0.0
        fallback_hits.append(float(int(masked.argmax()) in siblings.get(int(labels[row]), [])))
    other = ~is_saturated
    own_probability = (-own_loss).exp()
    return {
        "blur_share": float(
            ((own_probability > BLUR_LOW) & (own_probability < BLUR_HIGH)).float().mean()
        ),
        "sure_right_share": float((own_probability >= BLUR_HIGH).float().mean()),
        "sure_wrong_share": float((own_probability <= BLUR_LOW).float().mean()),
        "loss_mean": float(own_loss.mean()),
        "loss_std": float(own_loss.std(unbiased=False)),
        "loss_mean_saturated": float(own_loss[is_saturated].mean()) if is_saturated.any() else 0.0,
        "loss_mean_other": float(own_loss[other].mean()) if other.any() else 0.0,
        "entropy_t1": float(entropy_nats(targets).mean()),
        "entropy_t2": float(entropy_nats(tempered).mean()),
        "tail_mass_top_k_t2": float(tail_mass(tempered, spec.tail_k).mean()),
        "label_top1_share": float((targets.argmax(dim=1) == labels).float().mean()),
        "sibling_fallback_saturated": float(np.mean(fallback_hits)) if fallback_hits else 0.0,
        "kl_to_original_t1_mean": float(kl_to_original.mean()),
        "kl_to_original_t1_median": float(kl_to_original.median()),
        "saturated_image_count": float(is_saturated.sum()),
    }
