"""Tests of the localization metric: formula, edge cases, downsampling rule, official parity."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from hackathon.common.config import load_settings
from hackathon.wmloc.constants import (
    GRID_SIZE,
    IMAGE_SIZE,
    RELEASE_SUBDIR,
    STARTER_KIT_SUBDIR,
    WMLOC_DATA_SUBDIR,
)
from hackathon.wmloc.errors import MetricInputError
from hackathon.wmloc.metric import downsample_mask_to_grid, miou_score, per_image_miou

RANDOM_SEED = 7
RANDOM_TRIALS = 50
MASK_PNG_THRESHOLD = 127


def grid(value: int) -> torch.Tensor:
    return torch.full((1, GRID_SIZE, GRID_SIZE), value, dtype=torch.uint8)


def single_cell() -> torch.Tensor:
    mask = torch.zeros(1, GRID_SIZE, GRID_SIZE, dtype=torch.uint8)
    mask[0, 5, 9] = 1
    return mask


def test_perfect_prediction_scores_one() -> None:
    truth = single_cell()
    assert miou_score(truth, truth) == 1.0


def test_both_all_ones_has_empty_clean_union_and_scores_one() -> None:
    # Class 0 union is empty on both sides -> IoU_0 = 1 by the task rule; IoU_1 = 1.
    assert miou_score(grid(1), grid(1)) == 1.0


def test_both_all_zeros_has_empty_marked_union_and_scores_one() -> None:
    assert miou_score(grid(0), grid(0)) == 1.0


def test_all_ones_against_all_zeros_scores_zero() -> None:
    assert miou_score(grid(1), grid(0)) == 0.0
    assert miou_score(grid(0), grid(1)) == 0.0


def test_all_ones_prediction_on_a_single_marked_cell() -> None:
    # IoU_1 = 1 / 16384 ; IoU_0 = 0 (nothing predicted clean) -> mean is half of IoU_1.
    expected = (1 / (GRID_SIZE * GRID_SIZE)) / 2
    assert miou_score(grid(1), single_cell()) == pytest.approx(expected)


def test_all_zeros_prediction_on_a_single_marked_cell() -> None:
    # IoU_1 = 0 ; IoU_0 = (N - 1) / N.
    total = GRID_SIZE * GRID_SIZE
    assert miou_score(grid(0), single_cell()) == pytest.approx(((total - 1) / total) / 2)


def test_hand_computed_half_overlap() -> None:
    truth = torch.zeros(1, GRID_SIZE, GRID_SIZE, dtype=torch.uint8)
    prediction = torch.zeros_like(truth)
    truth[0, :, :64] = 1  # left half marked
    prediction[0, :, 32:96] = 1  # shifted half
    # class 1: inter 32 cols, union 96 cols -> 1/3 ; class 0: inter 32 cols, union 96 cols -> 1/3
    assert miou_score(prediction, truth) == pytest.approx(1 / 3)


def test_images_weigh_the_same_whatever_their_edit_size() -> None:
    perfect = single_cell()
    wrong = single_cell()
    wrong[0, 5, 9] = 0
    wrong[0, 6, 9] = 1  # same size, disjoint -> IoU_1 = 0, IoU_0 = (N-2)/N
    batch_pred = torch.cat([perfect, wrong])
    batch_truth = torch.cat([single_cell(), single_cell()])
    per_image = per_image_miou(batch_pred, batch_truth)
    assert per_image[0].item() == 1.0
    assert miou_score(batch_pred, batch_truth) == pytest.approx(per_image.mean().item())


def test_random_masks_are_symmetric_bounded_and_one_only_when_equal() -> None:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    for _ in range(RANDOM_TRIALS):
        a = torch.randint(0, 2, (3, GRID_SIZE, GRID_SIZE), generator=generator, dtype=torch.uint8)
        b = torch.randint(0, 2, (3, GRID_SIZE, GRID_SIZE), generator=generator, dtype=torch.uint8)
        ab, ba = per_image_miou(a, b), per_image_miou(b, a)
        assert torch.allclose(ab, ba)
        assert bool(((ab >= 0) & (ab <= 1)).all())
        assert bool((per_image_miou(a, a) == 1.0).all())


def test_bool_and_float_inputs_are_accepted() -> None:
    truth = single_cell()
    assert miou_score(truth.bool(), truth.float()) == 1.0


def test_non_binary_values_are_rejected() -> None:
    bad = grid(1) * 2
    with pytest.raises(MetricInputError):
        miou_score(bad, grid(1))


def test_shape_mismatch_and_empty_are_rejected() -> None:
    with pytest.raises(MetricInputError):
        miou_score(torch.zeros(1, 8, 8), torch.zeros(1, 16, 16))
    with pytest.raises(MetricInputError):
        miou_score(torch.zeros(0, GRID_SIZE, GRID_SIZE), torch.zeros(0, GRID_SIZE, GRID_SIZE))


def test_downsample_cell_rule_is_exactly_fifty_percent() -> None:
    mask = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    mask[0:4, 0:2] = 1  # 8 of 16 pixels -> cell (0, 0) is 1 (>= 50 %)
    mask[0:4, 4:5] = 1  # 4 of 16 pixels -> cell (0, 1) is 0
    mask[4:8, 0:4][:, :] = 0
    mask[4:7, 0:2] = 1  # 6 of 16 -> cell (1, 0) is 0
    cells = downsample_mask_to_grid(mask)
    assert cells.shape == (GRID_SIZE, GRID_SIZE)
    assert cells.dtype == torch.uint8
    assert cells[0, 0].item() == 1
    assert cells[0, 1].item() == 0
    assert cells[1, 0].item() == 0


def test_downsample_handles_batches_and_extremes() -> None:
    ones = torch.ones(2, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    zeros = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    assert bool((downsample_mask_to_grid(ones) == 1).all())
    assert bool((downsample_mask_to_grid(zeros) == 0).all())
    assert downsample_mask_to_grid(ones).shape == (2, GRID_SIZE, GRID_SIZE)


def test_downsample_rejects_indivisible_size() -> None:
    with pytest.raises(MetricInputError):
        downsample_mask_to_grid(torch.zeros(100, 100, dtype=torch.uint8))


KIT_UTILS_RELATIVE = Path("wmloc") / "utils.py"


def _official_utils() -> ModuleType:
    """Load the starter kit's `wmloc/utils.py` BY PATH under a private module name.

    A path load avoids any clash between the kit's top-level `wmloc` package and our
    `hackathon.wmloc`, and the file only imports numpy and torch. The kit folder is
    $WMLOC_STARTER_KIT_DIR, else <data_dir>/wmloc/starter_kit.
    """
    configured = os.environ.get("WMLOC_STARTER_KIT_DIR")
    kit = (
        Path(configured)
        if configured
        else (load_settings().data_dir / WMLOC_DATA_SUBDIR / STARTER_KIT_SUBDIR)
    )
    path = kit / KIT_UTILS_RELATIVE
    if not path.is_file():
        pytest.skip(f"starter kit not found at {kit} (run `fetch`, or set WMLOC_STARTER_KIT_DIR)")
    spec = importlib.util.spec_from_file_location("_official_wmloc_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structured_masks(count: int) -> torch.Tensor:
    """Random 512 masks mixing blobs, exact-50 % cells and empty / full images."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    masks = (torch.rand(count, IMAGE_SIZE, IMAGE_SIZE, generator=generator) > 0.5).to(torch.uint8)
    masks[0] = 0
    masks[1] = 1
    masks[2, :, :] = 0
    masks[2, :, : IMAGE_SIZE // 2] = 1
    masks[3].view(GRID_SIZE, 4, GRID_SIZE, 4)[:, :2].fill_(1)  # every cell has exactly 8 of 16
    masks[3].view(GRID_SIZE, 4, GRID_SIZE, 4)[:, 2:].fill_(0)
    return masks


def test_downsampling_matches_the_official_one_including_exact_half_cells() -> None:
    official = _official_utils()
    for mask in _structured_masks(8):
        expected = torch.from_numpy(official.downsample_mask(mask.numpy()))
        assert torch.equal(downsample_mask_to_grid(mask), expected)


def test_per_image_miou_and_score_match_the_official_ones() -> None:
    official = _official_utils()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    truths = torch.randint(0, 2, (12, GRID_SIZE, GRID_SIZE), generator=generator, dtype=torch.uint8)
    preds = torch.randint(0, 2, (12, GRID_SIZE, GRID_SIZE), generator=generator, dtype=torch.uint8)
    truths[0], preds[0] = 0, 0  # empty marked union on both sides
    truths[1], preds[1] = 1, 1  # empty clean union on both sides
    truths[2], preds[2] = 1, 0
    ours = per_image_miou(preds, truths)
    for index in range(12):
        expected = official.miou_image(preds[index].numpy(), truths[index].numpy())
        assert ours[index].item() == pytest.approx(expected)
    assert miou_score(preds, truths) == pytest.approx(
        official.score(list(preds.numpy()), list(truths.numpy()))
    )


def test_downsampling_reproduces_the_official_train_masks_128() -> None:
    kit_data = load_settings().data_dir / WMLOC_DATA_SUBDIR / RELEASE_SUBDIR / "train"
    archive = kit_data / "masks_128.npz"
    if not archive.is_file():
        pytest.skip(f"train release not found at {kit_data}")
    pytest.importorskip("PIL")
    from PIL import Image

    npz = np.load(archive)
    for name, official_mask in list(zip(npz["names"].tolist(), npz["masks"], strict=True))[:40]:
        full = np.asarray(Image.open(kit_data / "masks" / f"{name}.png")) > MASK_PNG_THRESHOLD
        ours = downsample_mask_to_grid(torch.from_numpy(full.astype(np.uint8)))
        assert torch.equal(ours, torch.from_numpy(official_mask.astype(np.uint8)))
