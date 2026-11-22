"""Tests of the original/candidate blends (tiny ViT, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.retain_proxy.blend import (
    BlendError,
    blend_state_dicts,
    write_blend,
)


def _tiny() -> nn.Module:
    return build_model(embed_dim=32, depth=2, num_heads=2)


def _state(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {key: value.clone() for key, value in _tiny().state_dict().items()}


def test_blend_is_a_linear_interpolation_and_extrapolation() -> None:
    original, candidate = _state(0), _state(1)
    for alpha in (0.0, 0.5, 1.0, 1.25):
        blended = blend_state_dicts(original, candidate, alpha)
        for key, base in original.items():
            expected = base + alpha * (candidate[key] - base)
            torch.testing.assert_close(blended[key], expected, atol=1e-6, rtol=1e-5)
            assert blended[key].dtype == base.dtype


def test_blend_rejects_mismatched_state_dicts() -> None:
    original, candidate = _state(0), _state(1)
    missing = dict(candidate)
    missing.popitem()
    with pytest.raises(BlendError):
        blend_state_dicts(original, missing, 1.0)
    reshaped = dict(candidate)
    first = next(iter(reshaped))
    reshaped[first] = torch.zeros(3)
    with pytest.raises(BlendError):
        blend_state_dicts(original, reshaped, 1.0)


def test_write_blend_saves_a_validated_float32_file(tmp_path: Path) -> None:
    original_path, candidate_path = tmp_path / "o.pt", tmp_path / "c.pt"
    torch.save(_state(0), original_path)
    torch.save(_state(1), candidate_path)
    out = tmp_path / "out" / "blend_a1.10.pt"
    report = write_blend(original_path, candidate_path, 1.1, out, model_factory=_tiny)
    assert out.is_file() and report.size_bytes == out.stat().st_size
    loaded = torch.load(out, weights_only=True)
    assert all(
        value.dtype == torch.float32 for value in loaded.values() if value.is_floating_point()
    )
    assert not list(out.parent.glob("*.partial*"))
