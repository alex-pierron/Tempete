"""Tests of the segment-proposal modules (pure numpy parts; no decoder, no data needed)."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.wmloc.research.proposal_cv import (
    CHANCE,
    ImageProposals,
    best_pick,
    proposal_features,
    target_masks,
)
from hackathon.wmloc.research.proposals import (
    STAT_COLUMNS,
    Rectangle,
    downsample_mask,
    largest_rectangle,
    proposal_cache_path,
)


def test_largest_rectangle_finds_the_block() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:12, 6:25] = 1
    mask[12, 6] = 1  # a bump that must not extend the rectangle
    assert largest_rectangle(mask) == Rectangle(4, 6, 12, 25)


def test_largest_rectangle_empty_mask_is_none() -> None:
    assert largest_rectangle(np.zeros((8, 8), dtype=np.uint8)) is None


def test_largest_rectangle_non_convex_object() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:10, 0:3] = 1  # tall bar (area 30)
    mask[7:10, 0:10] = 1  # wide bar (area 30 as well, shares a corner)
    rect = largest_rectangle(mask)
    assert rect is not None
    assert rect.height * rect.width >= 30


def test_downsample_mask_majority_vote() -> None:
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:4, :4] = 1  # one full 4x4 block
    mask[4, 4:6] = 1  # 2 of 16 pixels of another block: below the majority
    small = downsample_mask(mask)
    assert small.shape == (128, 128)
    assert small[0, 0] == 1 and small[1, 1] == 0 and small.sum() == 1


def test_rectangle_scaling() -> None:
    assert Rectangle(1, 2, 3, 5).scaled(4) == Rectangle(4, 8, 12, 20)


def test_cache_path_layout(tmp_path: object) -> None:
    path = proposal_cache_path(tmp_path, "train", "pixelseal", "abc")  # type: ignore[arg-type]
    assert path.name == "abc.props.npz" and path.parent.name == "pixelseal"


def _stats(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32).reshape(-1, len(STAT_COLUMNS))


def test_features_missing_inscribed_decode_is_chance_with_flag() -> None:
    nan = float("nan")
    # det_score, label, area, bbox_agree, bbox_margin, inscr_agree, inscr_margin, h, w
    stats = _stats(
        [
            [0.9, 1.0, 0.2, 0.95, 4.0, 0.9, 3.0, 100.0, 80.0],
            [0.8, 2.0, 0.1, 0.5, 0.0, nan, nan, 0.0, 0.0],
        ]
    )
    feats = proposal_features(stats)
    assert feats.shape == (2, 10)
    assert feats[1, 2] == pytest.approx(CHANCE) and feats[1, 4] == 0.0 and feats[0, 4] == 1.0
    assert feats[0, 7] == pytest.approx(0.0) and feats[1, 7] == pytest.approx(-0.45)


def test_features_empty_input() -> None:
    assert proposal_features(_stats([])).shape == (0, 10)


def test_target_masks_complement_for_full_regime() -> None:
    masks = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    assert target_masks(masks, full_regime=False).tolist() == masks.tolist()
    assert target_masks(masks, full_regime=True).tolist() == [[[0, 1], [1, 1]]]


def test_best_pick_without_model_or_proposals() -> None:
    empty = ImageProposals(np.zeros((0, 128, 128), np.uint8), np.zeros((0, 10), np.float32))
    assert best_pick(None, empty) == (-1, 0.0)


def test_evidence_columns_inside_outside_contrast() -> None:
    from hackathon.wmloc.research.proposal_cv import EVIDENCE_COLUMNS, evidence_columns

    evidence = np.zeros((4, 4), dtype=np.float32)
    evidence[:2, :] = 3.0
    top = np.zeros((4, 4), dtype=np.uint8)
    top[:2, :] = 1
    cols = evidence_columns(evidence, top[None])
    assert cols.shape == (1, EVIDENCE_COLUMNS)
    assert cols[0].tolist() == [3.0, 0.0, 3.0]


def test_evidence_columns_absent_evidence_is_zeros() -> None:
    from hackathon.wmloc.research.proposal_cv import evidence_columns

    masks = np.ones((2, 4, 4), dtype=np.uint8)
    assert not evidence_columns(None, masks).any()
