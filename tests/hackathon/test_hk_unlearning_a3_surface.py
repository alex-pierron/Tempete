"""Tests of the closed-form delta of the post-hoc surface and of its argument parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f

from hackathon.unlearning.research.a3_generate import GroupLogits, shift_removed
from hackathon.unlearning.research.a3_surface import SurfaceError, parse_bases

CLASSES = 10
FEATURES = 5
REMOVED = (2, 7)
DELTA = 4.5


def _group(logits: torch.Tensor) -> GroupLogits:
    return GroupLogits(
        exact=logits, altered={"alt_crop": logits + 1.0}, retain=logits * 2, views=None
    )


def test_shift_removed_equals_a_head_bias_edit() -> None:
    """The claim the whole surface rests on: a bias shift IS a logit shift of that class."""
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(16, FEATURES, generator=generator)
    weight = torch.randn(CLASSES, FEATURES, generator=generator)
    bias = torch.randn(CLASSES, generator=generator)
    edited_bias = bias.clone()
    edited_bias[list(REMOVED)] -= DELTA
    via_bias = torch_f.linear(features, weight, edited_bias)
    via_shift = shift_removed(_group(torch_f.linear(features, weight, bias)), REMOVED, DELTA).exact
    assert torch.allclose(via_bias, via_shift, atol=1e-6)


def test_shift_removed_touches_only_the_removed_columns_on_every_group_and_copies() -> None:
    logits = torch.randn(6, CLASSES)
    group = _group(logits)
    shifted = shift_removed(group, REMOVED, DELTA)
    kept = [index for index in range(CLASSES) if index not in REMOVED]
    for before, after in (
        (group.exact, shifted.exact),
        (group.altered["alt_crop"], shifted.altered["alt_crop"]),
        (group.retain, shifted.retain),
    ):
        assert torch.equal(after[:, kept], before[:, kept])
        assert torch.allclose(after[:, list(REMOVED)], before[:, list(REMOVED)] - DELTA)
    assert torch.equal(group.exact, logits)  # the input is not modified
    assert shifted.views is None


def test_zero_delta_is_the_identity() -> None:
    group = _group(torch.randn(4, CLASSES))
    assert torch.equal(shift_removed(group, REMOVED, 0.0).exact, group.exact)


def test_parse_bases_checks_the_format_and_the_file(tmp_path: Path) -> None:
    state = tmp_path / "state.pt"
    state.write_bytes(b"x")
    assert parse_bases([f"gs={state}"]) == [("gs", state)]
    with pytest.raises(SurfaceError):
        parse_bases(["no-separator"])
    with pytest.raises(SurfaceError):
        parse_bases([f"gs={tmp_path / 'missing.pt'}"])
