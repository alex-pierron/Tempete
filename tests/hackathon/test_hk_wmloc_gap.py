"""Tests of team gap's pure functions: post-processing, nested rule choice, hole ranking."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.wmloc.research.gap_analysis import class_ious, minority_view, trivial_masks
from hackathon.wmloc.research.gap_cv import Rule, family_of, nested_choice, paired_interval
from hackathon.wmloc.research.gap_holes import evidence_pick, hole_auc, proposals, upsample
from hackathon.wmloc.research.gap_postprocess import (
    SHAPE_BLANK,
    SHAPE_BOX_ALL,
    SHAPE_LARGEST,
    GapPostprocessError,
    PostConfig,
    postprocess,
    predicted_full,
    shape_minority,
)

GRID = 128


def _rect_probability(value_inside: float, value_outside: float) -> np.ndarray:
    probability = np.full((GRID, GRID), value_outside, dtype=np.float32)
    probability[10:30, 20:60] = value_inside
    return probability


def test_regime_is_read_from_the_reference_mask() -> None:
    assert not predicted_full(_rect_probability(0.9, 0.1))
    assert predicted_full(_rect_probability(0.1, 0.9))


def test_threshold_never_flips_the_regime() -> None:
    # Outside sits at 0.3: a 0.2 threshold would mark everything, but the regime is fixed by the
    # reference mask (rect), so the mask only grows inside that regime.
    probability = _rect_probability(0.9, 0.1)
    probability[40:50, 40:50] = 0.3
    base = postprocess(probability, PostConfig())
    grown = postprocess(probability, PostConfig(threshold=0.2))
    assert base.sum() == 20 * 40
    assert grown.sum() == 20 * 40 + 100
    assert grown.mean() < 0.5


def test_full_regime_threshold_grows_the_hole() -> None:
    probability = _rect_probability(0.1, 0.95)
    probability[60:70, 60:70] = 0.7
    base = postprocess(probability, PostConfig())
    grown = postprocess(probability, PostConfig(threshold=0.8))
    assert (1 - base).sum() == 20 * 40
    assert (1 - grown).sum() == 20 * 40 + 100


def test_shapes_and_abstention() -> None:
    minority = np.zeros((GRID, GRID), dtype=np.uint8)
    minority[5:10, 5:10] = 1
    minority[50:52, 50:52] = 1
    assert shape_minority(minority, SHAPE_LARGEST).sum() == 25
    assert shape_minority(minority, SHAPE_BOX_ALL).sum() == 47 * 47
    assert shape_minority(minority, SHAPE_BLANK).sum() == 0
    with pytest.raises(GapPostprocessError):
        shape_minority(minority, "unknown")
    probability = _rect_probability(0.9, 0.1)
    assert postprocess(probability, PostConfig(min_area=0.5)).sum() == 0


def test_class_ious_and_minority_view() -> None:
    truth = np.zeros((2, GRID, GRID), dtype=np.uint8)
    truth[0, :64] = 1
    prediction = truth.copy()
    one, zero = class_ious(prediction, truth)
    assert one.tolist() == [1.0, 1.0]
    assert zero.tolist() == [1.0, 1.0]
    is_full = np.array([True, False])
    assert trivial_masks(is_full, GRID)[0].all()
    assert not trivial_masks(is_full, GRID)[1].any()
    assert minority_view(truth, is_full)[0, :64].sum() == 0


def test_nested_choice_never_uses_the_held_fold() -> None:
    # Rule 1 is better everywhere except on the images of fold 0, where it is catastrophic. The
    # choice for fold 0 is made on fold 1 only, so fold 0 still receives rule 1.
    scores = np.array([[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.9, 0.9]])
    folds = np.array([0, 0, 1, 1])
    groups = np.zeros(4, dtype=np.int64)
    allowed = np.array([True, True])
    nested, chosen = nested_choice(scores, allowed, folds, groups, reference=0)
    assert nested.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert chosen == [1, 0]


def test_families_and_interval() -> None:
    assert "threshold" in family_of(Rule("base", PostConfig(threshold=0.2)))
    assert "shape" not in family_of(Rule("base", PostConfig(threshold=0.2)))
    assert family_of(Rule("base", PostConfig())) >= {"ensemble", "tta", "threshold", "shape"}
    mean, low, high = paired_interval(np.full(50, 0.01))
    assert mean == pytest.approx(0.01)
    assert low == pytest.approx(0.01)
    assert high == pytest.approx(0.01)


def test_hole_ranking_and_proposals() -> None:
    hole = np.zeros((GRID, GRID), dtype=bool)
    hole[10:30, 20:60] = True
    assert hole_auc(hole.astype(np.float64), hole) == pytest.approx(1.0)
    assert hole_auc(-hole.astype(np.float64), hole) == pytest.approx(0.0)
    assert hole_auc(np.zeros((GRID, GRID)), np.zeros((GRID, GRID), dtype=bool)) is None
    assert upsample(np.arange(64, dtype=np.float32).reshape(8, 8)).shape == (GRID, GRID)
    probability = _rect_probability(0.05, 0.999)
    probability[100:110, 100:110] = 0.9
    regions = proposals(probability)
    assert np.array_equal(regions[0], hole)
    evidence = np.ones((GRID, GRID))
    evidence[100:110, 100:110] = -1.0
    assert evidence_pick(evidence, regions) == 1
