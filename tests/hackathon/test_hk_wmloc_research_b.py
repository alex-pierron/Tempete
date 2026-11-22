"""Tests of the learned-localization research code (team B): shapes, metric parity, folds, model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hackathon.wmloc.research.model import Localizer, residual_input
from hackathon.wmloc.research.shapes import (
    GRID_SIZE,
    IMAGE_SIZE,
    KINDS,
    bounding_box,
    downsample_mask,
    sample_region,
)
from hackathon.wmloc.research.train import (
    localization_loss,
    official_miou,
    stratified_folds,
)

SEED = 7
N_SHAPES = 60
KIND_MIX = {"rect": 0.4, "blob": 0.3, "polygon": 0.3}


def naive_miou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Reference implementation copied from the kit's `utils.miou_image` semantics."""
    total = 0.0
    for cls in (0, 1):
        union = np.logical_or(pred == cls, truth == cls).sum()
        inter = np.logical_and(pred == cls, truth == cls).sum()
        total += 1.0 if union == 0 else inter / union
    return total / 2.0


def test_official_miou_matches_reference() -> None:
    rng = np.random.default_rng(SEED)
    pred = rng.random((12, 16, 16)) > 0.6
    truth = rng.random((12, 16, 16)) > 0.6
    pred[0] = truth[0]
    pred[1] = False
    truth[1] = False
    expected = np.array([naive_miou(p.astype(int), t.astype(int)) for p, t in zip(pred, truth)])
    assert np.allclose(official_miou(pred, truth), expected)
    assert official_miou(pred, truth)[1] == pytest.approx(1.0)  # both all-zero: perfect agreement


def test_empty_union_counts_one() -> None:
    zeros = np.zeros((1, 8, 8), dtype=bool)
    assert official_miou(zeros, zeros)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_sample_region_is_valid(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(N_SHAPES):
        region = sample_region(rng, KIND_MIX, max_rect_aspect=2.0)
        assert region.kind in KINDS
        assert region.mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert region.mask.any()
        assert downsample_mask(region.mask).any()
        y0, y1, x0, x1 = bounding_box(region.mask)
        assert 0 <= y0 < y1 <= IMAGE_SIZE and 0 <= x0 < x1 <= IMAGE_SIZE


def test_rect_aspect_cap_respected() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        region = sample_region(rng, {"rect": 1.0}, max_rect_aspect=2.0)
        y0, y1, x0, x1 = region.bbox
        height, width = y1 - y0, x1 - x0
        assert max(height, width) / min(height, width) <= 2.05 or min(height, width) <= 8


def test_downsample_mask_rule() -> None:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[:4, :2] = True  # 8 of 16 pixels -> cell is 1 (>= 50 %)
    mask[:4, 4:5] = True  # 4 of 16 pixels -> cell is 0
    grid = downsample_mask(mask)
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert grid[0, 0] == 1 and grid[0, 1] == 0


def test_stratified_folds_partition_and_balance() -> None:
    full = np.array([True] * 20 + [False] * 20)
    folds = stratified_folds(full, 5, SEED)
    merged = np.sort(np.concatenate(folds))
    assert np.array_equal(merged, np.arange(40))
    assert all(full[f].sum() == 4 for f in folds)
    again = stratified_folds(full, 5, SEED)
    assert all(np.array_equal(a, b) for a, b in zip(folds, again))


@pytest.mark.parametrize("use_extras", [False, True])
def test_model_output_shape(use_extras: bool) -> None:
    model = Localizer(use_extras=use_extras).eval()
    images = torch.randint(0, 256, (2, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8)
    maps = torch.zeros(2, 2, 8, 8)
    scalars = torch.zeros(2, 2)
    out = model(images, maps, scalars)
    assert out.shape == (2, GRID_SIZE, GRID_SIZE)


def test_extras_model_requires_extras() -> None:
    model = Localizer(use_extras=True).eval()
    images = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8)
    with pytest.raises(ValueError):
        model(images, None, None)


def test_residual_input_flat_image_has_zero_highpass() -> None:
    flat = torch.full((1, IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=torch.uint8)
    x = residual_input(flat)
    assert x.shape == (1, 6, IMAGE_SIZE, IMAGE_SIZE)
    assert float(x[:, 3:].abs().max()) < 1e-5


def test_loss_is_lower_for_correct_prediction() -> None:
    truth = torch.zeros(2, 8, 8)
    truth[:, :4] = 1.0
    good = (truth * 2 - 1) * 6.0
    bad = -good
    assert localization_loss(good, truth) < localization_loss(bad, truth)
