"""Tests of the forget-only SSD (LFSSD) dampening of x-unl-papers."""

from __future__ import annotations

import pytest
import torch

from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.x_ssd import (
    DampeningError,
    batched,
    dampen_state,
    importance_ratio,
    output_norm_sensitivity,
)

TINY = {"embed_dim": 32, "depth": 1, "num_heads": 2}


def test_dampen_selects_only_ratio_above_alpha_and_applies_beta() -> None:
    state = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    exact = {"w": torch.tensor([1.0, 10.0, 3.0, 0.5])}
    view = {"w": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    new, stats = dampen_state(state, exact, view, alpha=2.5, lam=1.0)
    # ratios 1, 10, 3, 0.5 -> selected 10 and 3 ; beta = 1/10 and 1/3
    assert torch.allclose(new["w"], torch.tensor([1.0, 0.2, 1.0, 4.0]))
    assert stats["selected_fraction"] == pytest.approx(0.5)
    assert stats["mean_beta_selected"] == pytest.approx((0.1 + 1 / 3) / 2)


def test_beta_never_amplifies() -> None:
    state = {"w": torch.ones(3)}
    exact = {"w": torch.tensor([3.0, 3.0, 3.0])}
    view = {"w": torch.ones(3)}
    new, _ = dampen_state(state, exact, view, alpha=2.0, lam=10.0)
    assert torch.equal(new["w"], torch.ones(3))


def test_keys_without_importance_are_copied() -> None:
    state = {"w": torch.ones(2), "buffer": torch.zeros(2)}
    new, _ = dampen_state(state, {"w": torch.ones(2)}, {"w": torch.ones(2)}, alpha=1.5)
    assert torch.equal(new["buffer"], state["buffer"])
    assert new["buffer"] is not state["buffer"]


@pytest.mark.parametrize(("alpha", "lam"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
def test_invalid_hyperparameters_raise(alpha: float, lam: float) -> None:
    with pytest.raises(DampeningError):
        dampen_state({"w": torch.ones(1)}, {"w": torch.ones(1)}, {"w": torch.ones(1)}, alpha, lam)


def test_zero_view_importance_is_selected() -> None:
    ratio = importance_ratio(torch.tensor([1e-6]), torch.tensor([0.0]))
    assert float(ratio) > 1e10


def test_sensitivity_has_parameter_shapes_and_is_nonnegative() -> None:
    torch.manual_seed(0)
    model = build_model(num_classes=5, **TINY).eval()
    images = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    omega = output_norm_sensitivity(model, batched(images, 2), torch.device("cpu"))
    params = dict(model.named_parameters())
    assert set(omega) == set(params)
    for name, value in omega.items():
        assert value.shape == params[name].shape
        assert bool((value >= 0).all())
    assert any(float(v.sum()) > 0 for v in omega.values())


def test_sensitivity_without_batch_raises() -> None:
    model = build_model(num_classes=5, **TINY).eval()
    with pytest.raises(DampeningError):
        output_norm_sensitivity(model, [], torch.device("cpu"))
