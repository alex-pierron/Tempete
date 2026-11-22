"""Team G: class structure and calibrated per-image targets (pure functions, no model)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hackathon.unlearning.research.g_classes import (
    ClassStructureError,
    class_counts,
    cosine_groups,
    forgotten_fraction,
    fully_forgotten_classes,
    present_siblings,
)
from hackathon.unlearning.research.g_targets import (
    TargetError,
    TargetSpec,
    build_target_bank,
    entropy_nats,
    solve_group_scales,
    tail_mass,
)

CLASSES = 100
IMAGES = 400
SATURATED_CLASS = 0
SATURATED_IMAGES = 100
SIBLING_GROUP = [0, 1, 2, 3]
ENTROPY_TOLERANCE = 0.02
LOSS_TOLERANCE = 1e-3


def _spec(**changes: object) -> TargetSpec:
    """No fragile-tail on the synthetic data: its heavy tail alone would exceed the requested std."""
    return TargetSpec(**{"other_tail_fraction": 0.0, **changes})  # type: ignore[arg-type]


def _synthetic_bank_inputs(
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, list[list[int]]]:
    """400 images: 100 of a saturated class 0, 300 spread over classes 1..9 (siblings 1-3 present)."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.cat(
        [
            torch.full((SATURATED_IMAGES,), SATURATED_CLASS),
            torch.randint(1, 10, (IMAGES - SATURATED_IMAGES,), generator=generator),
        ]
    )
    logits = torch.randn(IMAGES, CLASSES, generator=generator)
    logits[torch.arange(IMAGES), labels] += 12.0
    views = torch.stack(
        [
            torch.softmax(logits + torch.randn(IMAGES, CLASSES, generator=generator) * 3, 1)
            for _ in range(3)
        ],
        dim=1,
    )
    counts = np.zeros(CLASSES, dtype=np.int64)
    counts[SATURATED_CLASS] = 130
    counts[1:10] = 60
    return views, logits, counts, [SIBLING_GROUP]


def test_class_counts_and_saturation_follow_the_pseudo_labels() -> None:
    counts = class_counts([0] * 130 + [1] * 40)
    assert counts[0] == 130 and counts[1] == 40 and counts.sum() == 170
    assert fully_forgotten_classes(counts) == [0]
    assert forgotten_fraction(counts)[1] == pytest.approx(40 / 130)


def test_class_counts_rejects_empty_and_out_of_range_labels() -> None:
    with pytest.raises(ClassStructureError):
        class_counts([])
    with pytest.raises(ClassStructureError):
        class_counts([CLASSES])


def test_cosine_groups_recovers_planted_clusters_and_siblings_exclude_saturated() -> None:
    generator = torch.Generator().manual_seed(3)
    centres = torch.randn(3, 16, generator=generator) * 5
    rows = torch.cat(
        [centres[i] + 0.05 * torch.randn(4, 16, generator=generator) for i in range(3)]
    )
    weight = torch.cat([rows, torch.randn(48, 16, generator=generator)]).double().numpy()
    groups = cosine_groups(weight, grouped_classes=12, group_count=3)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert present_siblings(groups, saturated=[0, 5]) == {0: [1, 2, 3], 5: [4, 6, 7]}
    assert present_siblings([[0, 1]], saturated=[0, 1]) == {}


def test_solve_group_scales_hits_the_requested_mean_and_std() -> None:
    saturated = np.linspace(0.8, 1.2, 100)
    other = np.linspace(0.5, 1.5, 300)
    other = other / other.mean()
    saturated = saturated / saturated.mean()
    scale_saturated, scale_other = solve_group_scales(saturated, other, 2.25, 3.0, 0.02, 10.0)
    merged = np.concatenate(
        [np.clip(scale_saturated * saturated, 0.02, 10.0), np.clip(scale_other * other, 0.02, 10.0)]
    )
    assert merged.mean() == pytest.approx(2.25, abs=LOSS_TOLERANCE)
    assert merged.std() == pytest.approx(3.0, abs=LOSS_TOLERANCE)
    assert scale_saturated > scale_other


def test_solve_group_scales_refuses_an_unreachable_std() -> None:
    ones = np.ones(50)
    with pytest.raises(TargetError):
        solve_group_scales(ones, ones, 2.25, 30.0, 0.02, 10.0)
    with pytest.raises(TargetError):
        solve_group_scales(np.array([]), ones, 2.25, 3.0, 0.02, 10.0)


def test_calibrated_bank_is_a_distribution_with_the_requested_law() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    bank = build_target_bank(views, logits, counts, groups, _spec())
    for targets in (bank.targets, bank.targets_altered):
        assert targets.shape == (IMAGES, CLASSES)
        assert torch.allclose(targets.sum(dim=1), torch.ones(IMAGES), atol=1e-5)
        assert float(targets.min()) >= 0.0
    assert bank.statistics["loss_mean"] == pytest.approx(2.25, abs=1e-2)
    assert bank.statistics["loss_std"] == pytest.approx(3.0, abs=1e-2)
    assert bank.statistics["altered_loss_mean"] == pytest.approx(3.1, abs=1e-2)
    assert bank.statistics["loss_mean_saturated"] > 4 * bank.statistics["loss_mean_other"]


def test_saturated_images_route_to_present_siblings_and_never_predict_their_own_class() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    bank = build_target_bank(views, logits, counts, groups, _spec())
    saturated_rows = bank.targets[:SATURATED_IMAGES]
    assert set(saturated_rows.argmax(dim=1).tolist()) <= {1, 2, 3}
    assert bank.statistics["sibling_fallback_saturated"] == 1.0
    assert float(saturated_rows[:, SATURATED_CLASS].max()) < 0.02
    no_routing = build_target_bank(views, logits, counts, groups, _spec(routing_share=0.0))
    assert no_routing.statistics["sibling_fallback_saturated"] < 1.0


def test_entropy_control_reaches_the_requested_level_at_the_scoring_temperature() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    free = build_target_bank(views, logits, counts, groups, _spec(entropy_target=None))
    controlled = build_target_bank(views, logits, counts, groups, _spec(entropy_target=1.2))
    assert controlled.statistics["entropy_t2"] == pytest.approx(1.2, abs=ENTROPY_TOLERANCE)
    assert controlled.statistics["entropy_t2"] < free.statistics["entropy_t2"]
    assert controlled.statistics["tail_mass_top_k_t2"] < free.statistics["tail_mass_top_k_t2"]


def test_plain_view_bank_is_the_averaged_tempered_views() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    bank = build_target_bank(views, logits, counts, groups, _spec(kind="v2"))
    assert torch.allclose(bank.targets.sum(dim=1), torch.ones(IMAGES), atol=1e-5)
    assert torch.equal(bank.targets, bank.targets_altered)


def test_bank_is_deterministic_and_rejects_bad_shapes() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    first = build_target_bank(views, logits, counts, groups, _spec())
    second = build_target_bank(views, logits, counts, groups, _spec())
    assert torch.equal(first.targets, second.targets)
    with pytest.raises(TargetError):
        build_target_bank(views, logits[:-1], counts, groups, _spec())


def test_entropy_and_tail_mass_helpers() -> None:
    uniform = torch.full((2, CLASSES), 1.0 / CLASSES)
    assert float(entropy_nats(uniform)[0]) == pytest.approx(float(np.log(CLASSES)), abs=1e-5)
    assert float(tail_mass(uniform, 5)[0]) == pytest.approx(0.95, abs=1e-6)
    onehot = torch.zeros(1, CLASSES)
    onehot[0, 7] = 1.0
    assert float(entropy_nats(onehot)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(tail_mass(onehot, 1)[0]) == pytest.approx(0.0, abs=1e-6)


def test_blur_shares_partition_the_images_and_a_higher_shape_power_is_more_bimodal() -> None:
    views, logits, counts, groups = _synthetic_bank_inputs()
    flat = build_target_bank(views, logits, counts, groups, _spec(other_shape_power=0.4))
    sharp = build_target_bank(views, logits, counts, groups, _spec(other_shape_power=6.0))
    for bank in (flat, sharp):
        shares = (
            bank.statistics["blur_share"]
            + bank.statistics["sure_right_share"]
            + bank.statistics["sure_wrong_share"]
        )
        assert shares == pytest.approx(1.0, abs=1e-6)
    assert sharp.statistics["blur_share"] < flat.statistics["blur_share"]
    assert sharp.statistics["loss_mean"] == pytest.approx(2.25, abs=1e-2)
