"""Team G: the (alpha, scale, delta) recipe applied to state dicts (tiny ViT)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_factory, tiny_model

from hackathon.common.errors import SubmissionInvalidError
from hackathon.unlearning.model import cpu_float32_state_dict, predict
from hackathon.unlearning.research.g_candidate import (
    CandidateError,
    blend,
    build_candidate,
    edit_head,
    write_candidate,
)
from hackathon.unlearning.research.g_score import apply_class_bias, scale_logits

SATURATED = [0, 7]
TOLERANCE = 1e-4


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        cpu_float32_state_dict(tiny_model(seed=1)),
        cpu_float32_state_dict(tiny_model(seed=2)),
    )


def test_blend_endpoints_and_extrapolation() -> None:
    origin, base = _states()
    assert all(torch.allclose(blend(origin, base, 1.0)[k], base[k], atol=1e-6) for k in base)
    assert all(torch.allclose(blend(origin, base, 0.0)[k], origin[k], atol=1e-6) for k in base)
    key = "head.weight"
    assert torch.allclose(
        blend(origin, base, 1.5)[key], origin[key] + 1.5 * (base[key] - origin[key])
    )
    with pytest.raises(CandidateError):
        blend(origin, {k: v for k, v in base.items() if k != key}, 1.0)


def test_head_edit_matches_the_logit_level_recipe_of_the_scoring_grid() -> None:
    origin, base = _states()
    model = tiny_model(seed=2)
    images = random_images(3, seed=5)
    expected = apply_class_bias(scale_logits(predict(model, images), 1.5), SATURATED, 4.0)
    edited = tiny_model(seed=2)
    edited.load_state_dict(build_candidate(origin, base, 1.0, 1.5, 4.0, SATURATED))
    assert torch.allclose(predict(edited, images), expected, atol=TOLERANCE)


def test_edit_head_validates_its_arguments_and_leaves_the_input_untouched() -> None:
    _, base = _states()
    before = base["head.bias"].clone()
    edit_head(base, 1.25, 3.0, SATURATED)
    assert torch.equal(base["head.bias"], before)
    with pytest.raises(CandidateError):
        edit_head(base, 0.0, 0.0, SATURATED)
    with pytest.raises(CandidateError):
        edit_head(base, 1.0, -1.0, SATURATED)
    with pytest.raises(CandidateError):
        edit_head(base, 1.0, 1.0, [100])
    with pytest.raises(CandidateError):
        edit_head({"x": torch.zeros(1)}, 1.0, 1.0, SATURATED)


def test_write_candidate_validates_then_moves_into_place(tmp_path: Path) -> None:
    origin, base = _states()
    path = tmp_path / "g" / "cand.pt"
    report = write_candidate(
        build_candidate(origin, base, 1.1, 1.25, 4.0, SATURATED), path, tiny_factory
    )
    assert path.is_file() and report.size_bytes > 0
    assert not list(path.parent.glob("*.partial*"))
    broken = {k: v for k, v in base.items() if k != "head.bias"}
    with pytest.raises(SubmissionInvalidError):
        write_candidate(broken, tmp_path / "g" / "bad.pt", tiny_factory)
    assert not (tmp_path / "g" / "bad.pt").exists()
    assert not list(path.parent.glob("*.partial*"))
