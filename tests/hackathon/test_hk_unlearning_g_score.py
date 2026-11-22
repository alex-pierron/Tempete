"""Team G: team C's scoring model on exports (formulas checked on synthetic logits)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.research.g_score import (
    KL_CONSTRAINT_NONSAT,
    ScoreError,
    apply_class_bias,
    delta_grid,
    load_views,
    mean_std_score,
    scale_logits,
    score_candidate,
    sibling_fallback,
)

CLASSES = 100
IMAGES = 60
SATURATED = [0]
SIBLINGS = {0: [1, 2]}


def _views(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"exact": logits, "strong_view": logits}


def _original() -> torch.Tensor:
    logits = torch.zeros(IMAGES, CLASSES)
    labels = torch.arange(IMAGES) % 3
    logits[torch.arange(IMAGES), labels] = 30.0
    return logits


def test_mean_std_score_is_one_on_the_gold_zero_on_the_original_and_clipped() -> None:
    assert mean_std_score(2.25, 3.0, 0.0, 0.0, 2.25, 3.0) == pytest.approx(1.0)
    assert mean_std_score(0.0, 0.0, 0.0, 0.0, 2.25, 3.0) == pytest.approx(0.0)
    assert mean_std_score(20.0, 20.0, 0.0, 0.0, 2.25, 3.0) == 0.0
    with pytest.raises(ScoreError):
        mean_std_score(1.0, 1.0, 2.25, 3.0, 2.25, 3.0)


def test_class_bias_only_touches_the_listed_classes() -> None:
    logits = torch.zeros(2, CLASSES)
    edited = apply_class_bias(logits, [0, 5], 7.0)
    assert (
        float(edited[0, 0]) == -7.0 and float(edited[0, 5]) == -7.0 and float(edited[0, 1]) == 0.0
    )
    assert float(logits.abs().sum()) == 0.0


def test_the_original_scores_zero_and_a_bias_on_the_saturated_class_raises_the_loss() -> None:
    original = _original()
    plain = score_candidate("orig", _views(original), _views(original), SATURATED, SIBLINGS)
    assert plain.ms_exact == pytest.approx(0.0, abs=1e-9)
    assert plain.kl_t2_nonsat == pytest.approx(0.0, abs=1e-9)
    biased = score_candidate("orig", _views(original), _views(original), SATURATED, SIBLINGS, 8.0)
    assert biased.loss_mean > plain.loss_mean
    assert biased.kl_t2_sat > 0.0
    assert plain.respects_kl_constraint


def test_sibling_fallback_and_the_kl_constraint_flag() -> None:
    labels = torch.tensor([0, 0, 3])
    logits = torch.zeros(3, CLASSES)
    logits[0, 1] = 5.0
    logits[1, 9] = 5.0
    logits[2, 3] = 5.0
    assert sibling_fallback(logits, labels, SATURATED, SIBLINGS) == pytest.approx(0.5)
    original = _original()
    flat = torch.zeros(IMAGES, CLASSES)
    score = score_candidate("flat", _views(flat), _views(original), SATURATED, SIBLINGS)
    assert score.kl_t2_nonsat > KL_CONSTRAINT_NONSAT and not score.respects_kl_constraint
    assert score.entropy_t2 == pytest.approx(float(np.log(CLASSES)), abs=1e-4)


def test_delta_grid_returns_one_score_per_bias_and_missing_views_are_rejected(
    tmp_path: Path,
) -> None:
    original = _original()
    grid = delta_grid("o", _views(original), _views(original), SATURATED, SIBLINGS, (0.0, 4.0, 8.0))
    assert [row.delta for row in grid] == [0.0, 4.0, 8.0]
    with pytest.raises(ScoreError):
        score_candidate("bad", {"exact": original}, _views(original), SATURATED, SIBLINGS)
    with pytest.raises(ScoreError):
        load_views(tmp_path / "missing.npz")
    path = tmp_path / "views.npz"
    np.savez(path, logits_exact=original.numpy(), feats_exact=np.zeros((IMAGES, 4)))
    assert set(load_views(path)) == {"exact"}


def test_head_scale_sharpens_without_changing_the_argmax_and_is_validated() -> None:
    original = _original()
    soft = original / 30.0
    plain = score_candidate("soft", _views(soft), _views(original), SATURATED, SIBLINGS)
    sharp = score_candidate("soft", _views(soft), _views(original), SATURATED, SIBLINGS, scale=6.0)
    assert sharp.scale == 6.0 and plain.scale == 1.0
    assert sharp.entropy_t1 < plain.entropy_t1
    assert sharp.flips_nonsat == plain.flips_nonsat == 0.0
    assert 0.0 <= sharp.blur_share <= 1.0
    assert sharp.sure_right_share + sharp.blur_share + sharp.sure_wrong_share == pytest.approx(1.0)
    with pytest.raises(ScoreError):
        scale_logits(original, 0.0)
    grid = delta_grid("o", _views(soft), _views(original), SATURATED, SIBLINGS, (0.0,), (1.0, 2.0))
    assert [row.scale for row in grid] == [1.0, 2.0]
