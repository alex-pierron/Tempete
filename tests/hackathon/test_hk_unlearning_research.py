"""Tests of team A's research helpers: surrogate scores, targets, NPO loss, weight blending."""

from __future__ import annotations

import pytest
import torch

from hackathon.unlearning.research.runner import blend_state
from hackathon.unlearning.research.surrogate import (
    GoldHypothesis,
    clipped_ratio_score,
    score_group,
)
from hackathon.unlearning.research.trainer import (
    NUM_CLASSES,
    ResearchError,
    RunSpec,
    build_target,
    npo_loss,
)

CLASSES = 10
IMAGES = 12


def _logits(seed: int, scale: float = 4.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(IMAGES, CLASSES, generator=generator) * scale


def test_score_is_zero_for_the_original_and_one_for_an_exact_gold_match() -> None:
    orig = _logits(0, scale=12.0)
    labels = orig.argmax(dim=1)
    gold_views = (orig * 0.2).unsqueeze(0)
    hypothesis = GoldHypothesis(area=0.35, temperature=1.0)
    as_orig = score_group(orig, orig, labels, gold_views, hypothesis)
    assert as_orig.mean_std_score == pytest.approx(0.0, abs=1e-9)
    assert as_orig.prediction_score == pytest.approx(0.0, abs=1e-9)
    as_gold = score_group(orig * 0.2, orig, labels, gold_views, hypothesis)
    assert as_gold.mean_std_score == pytest.approx(1.0, abs=1e-9)
    assert as_gold.prediction_score == pytest.approx(1.0, abs=1e-9)


def test_clipped_ratio_score_handles_a_zero_normaliser_and_overshoot() -> None:
    assert clipped_ratio_score(1.0, 0.0) == 0.0
    assert clipped_ratio_score(5.0, 1.0) == 0.0
    assert clipped_ratio_score(0.25, 1.0) == pytest.approx(0.75)


def test_npo_loss_is_bounded_below_and_its_floor_switches_samples_off() -> None:
    teacher = _logits(1, scale=10.0)
    labels = teacher.argmax(dim=1)
    same = npo_loss(teacher, teacher, labels, beta=0.1, floor=0.0)
    assert float(same) == pytest.approx(2.0 / 0.1 * torch.log(torch.tensor(2.0)).item(), rel=1e-5)
    pushed = teacher.clone()
    pushed[torch.arange(IMAGES), labels] -= 30.0
    assert float(npo_loss(pushed, teacher, labels, beta=0.1, floor=0.0)) < float(same)
    assert float(npo_loss(pushed, teacher, labels, beta=0.1, floor=0.5)) == 0.0


def test_blend_state_endpoints_and_non_float_tensors() -> None:
    original = {"w": torch.zeros(3), "n": torch.tensor([1])}
    trained = {"w": torch.ones(3), "n": torch.tensor([1])}
    assert torch.equal(blend_state(original, trained, 0.0)["w"], original["w"])
    assert torch.equal(blend_state(original, trained, 1.0)["w"], trained["w"])
    assert torch.allclose(blend_state(original, trained, 0.5)["w"], torch.full((3,), 0.5))
    assert torch.equal(blend_state(original, trained, 0.5)["n"], trained["n"])


def test_runspec_rejects_unknown_names() -> None:
    with pytest.raises(ResearchError):
        RunSpec(name="x", method="gradient_ascent")
    with pytest.raises(ResearchError):
        RunSpec(name="x", trainable="everything")


def test_fallback_target_is_a_distribution_with_mass_on_a_wrong_class() -> None:
    class Teacher(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.zeros(len(batch), NUM_CLASSES)

    spec = RunSpec(name="fb", method="fallback", fallback_kind="rank2", target_views=2)
    exact = torch.zeros(4, 224, 224, 3, dtype=torch.uint8)
    cached = torch.softmax(torch.randn(2, 4, NUM_CLASSES), dim=2)
    target = build_target(
        Teacher(), exact, spec, torch.Generator().manual_seed(0), torch.device("cpu"), cached
    )
    assert target.shape == (4, NUM_CLASSES)
    assert torch.allclose(target.sum(dim=1), torch.ones(4), atol=1e-5)
