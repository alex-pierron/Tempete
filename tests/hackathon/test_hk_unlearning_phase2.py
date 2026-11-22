"""Tests of the phase 2 helpers of team A: calibrated targets and weight extrapolation."""

from __future__ import annotations

import pytest
import torch

from hackathon.unlearning.research.phase2 import Phase2Error, offset_head_bias
from hackathon.unlearning.research.runner import blend_state
from hackathon.unlearning.research.trainer import (
    NUM_CLASSES,
    RunSpec,
    _with_self_anchor,
    calibrated_targets,
    remove_classes_from_target,
)

IMAGE_COUNT = 200
VIEW_COUNT = 3
WRONG_FRACTION = 0.25
SEED = 0


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Exact logits with a clear label and views whose support for the label varies per image."""
    generator = torch.Generator().manual_seed(SEED)
    exact = torch.randn(IMAGE_COUNT, NUM_CLASSES, generator=generator)
    labels = torch.randint(NUM_CLASSES, (IMAGE_COUNT,), generator=generator)
    exact[torch.arange(IMAGE_COUNT), labels] += 12.0
    support = torch.linspace(0.02, 0.98, IMAGE_COUNT)
    noise = torch.rand(IMAGE_COUNT, VIEW_COUNT, NUM_CLASSES, generator=generator)
    noise = noise / noise.sum(dim=2, keepdim=True)
    views = (1 - support)[:, None, None] * noise
    views[torch.arange(IMAGE_COUNT), :, labels] += support[:, None]
    return views, exact


def test_calibrated_targets_are_distributions_with_the_requested_error_fraction() -> None:
    views, exact = _inputs()
    spec = RunSpec(name="t", method="calibrated", cal_wrong_fraction=WRONG_FRACTION)
    targets = calibrated_targets(views, exact, spec)
    labels = exact.argmax(dim=1)
    assert torch.allclose(targets.sum(dim=1), torch.ones(IMAGE_COUNT), atol=1e-5)
    assert (targets >= 0).all()
    accuracy = float((targets.argmax(dim=1) == labels).float().mean())
    assert accuracy == pytest.approx(1.0 - WRONG_FRACTION, abs=0.03)


def test_calibrated_targets_make_the_most_fragile_images_the_wrong_ones() -> None:
    views, exact = _inputs()
    spec = RunSpec(name="t", method="calibrated", cal_wrong_fraction=WRONG_FRACTION)
    targets = calibrated_targets(views, exact, spec)
    labels = exact.argmax(dim=1)
    label_mass = targets.gather(1, labels[:, None]).squeeze(1)
    fragile = slice(0, int(WRONG_FRACTION * IMAGE_COUNT) - 5)
    solid = slice(int(WRONG_FRACTION * IMAGE_COUNT) + 5, IMAGE_COUNT)
    assert float(label_mass[fragile].max()) < spec.cal_wrong_high + 1e-6
    assert float(label_mass[solid].min()) >= spec.cal_right_low - 1e-6


def test_self_anchor_replaces_the_requested_fraction_of_the_retain_batch() -> None:
    side = 224
    retain = torch.zeros(8, side, side, 3, dtype=torch.uint8)
    forget = torch.full((20, side, side, 3), 200, dtype=torch.uint8)
    generator = torch.Generator().manual_seed(SEED)
    unchanged = _with_self_anchor(retain, forget, RunSpec(name="t"), generator)
    assert torch.equal(unchanged, retain)
    spec = RunSpec(name="t", anchor_view_fraction=0.5)
    mixed = _with_self_anchor(retain, forget, spec, generator)
    assert mixed.shape == retain.shape
    assert torch.equal(mixed[:4], retain[:4])
    assert int(mixed[4:].float().mean()) > 0


def test_extrapolation_scales_the_weight_delta_and_keeps_integer_tensors() -> None:
    original = {"w": torch.tensor([1.0, 2.0]), "count": torch.tensor(3)}
    trained = {"w": torch.tensor([2.0, 0.0]), "count": torch.tensor(7)}
    doubled = blend_state(original, trained, 2.0)
    assert torch.allclose(doubled["w"], torch.tensor([3.0, -2.0]))
    assert int(doubled["count"]) == 7


def test_removed_classes_get_the_floor_and_rows_still_sum_to_one() -> None:
    generator = torch.Generator().manual_seed(SEED)
    target = torch.softmax(torch.randn(6, 12, generator=generator), dim=1)
    floor = 1e-3
    out = remove_classes_from_target(target, (2, 5), floor)
    assert torch.allclose(out.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.allclose(out[:, [2, 5]], torch.full((6, 2), floor))
    kept = [c for c in range(12) if c not in (2, 5)]
    ratio = out[:, kept] / target[:, kept]
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)
    assert torch.equal(remove_classes_from_target(target, (), floor), target)


def test_head_bias_offset_moves_only_the_requested_classes() -> None:
    state = {"head.bias": torch.zeros(10), "head.weight": torch.ones(10, 4)}
    edited = offset_head_bias(state, (1, 7), 5.0)
    assert edited["head.bias"][1] == -5.0 and edited["head.bias"][7] == -5.0
    assert float(edited["head.bias"].sum()) == -10.0
    assert torch.equal(edited["head.weight"], state["head.weight"])
    assert float(state["head.bias"].sum()) == 0.0
    with pytest.raises(Phase2Error):
        offset_head_bias(state, (10,), 1.0)
