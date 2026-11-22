"""The task metric: per-image mean IoU over the clean and watermarked classes on a 128x128 grid.

Formula (task PDF):
    IoU_c = |P_c & G_c| / |P_c | G_c|, equal to 1 when the union is empty;
    mIoU_image = (IoU_0 + IoU_1) / 2; score = mean over images (each image weighs the same).

This is our re-implementation. The official starter-kit function is authoritative: a test
compares both as soon as the kit is on disk (tests/hackathon/test_hk_wmloc_metric.py).
"""

from __future__ import annotations

import torch

from hackathon.wmloc.constants import (
    CELL_WATERMARK_FRACTION,
    CLASS_VALUES,
    GRID_SIZE,
    MASK_CLEAN,
    MASK_WATERMARKED,
)
from hackathon.wmloc.errors import MetricInputError

MASK_BATCH_NDIM = 3
MASK_SINGLE_NDIM = 2


def _as_batch(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Return `mask` as [N, H, W], accepting a single [H, W] mask."""
    if mask.ndim == MASK_SINGLE_NDIM:
        return mask.unsqueeze(0)
    if mask.ndim != MASK_BATCH_NDIM:
        raise MetricInputError(f"{name} must be [H, W] or [N, H, W], got {tuple(mask.shape)}")
    return mask


def _require_binary(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Return `mask` as a bool tensor, raising if any value is not 0 or 1."""
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() and bool(torch.isnan(mask).any()):
        raise MetricInputError(f"{name} contains NaN")
    is_binary = (mask == MASK_CLEAN) | (mask == MASK_WATERMARKED)
    if not bool(is_binary.all()):
        raise MetricInputError(f"{name} must only contain 0 and 1")
    return mask == MASK_WATERMARKED


def downsample_mask_to_grid(
    mask: torch.Tensor,
    grid_size: int = GRID_SIZE,
    min_fraction: float = CELL_WATERMARK_FRACTION,
) -> torch.Tensor:
    """Reduce a pixel mask to the evaluation grid: a cell is 1 if >= `min_fraction` of it is 1.

    Counting integer pixels (instead of interpolating floats) keeps the 50 % rule exact: a
    4x4 cell needs 8 of its 16 pixels, with no rounding ambiguity.

    Args:
        mask: [H, W] or [N, H, W], values in {0, 1} (bool or numeric); H and W must be
            multiples of `grid_size`.
        grid_size: Side of the output grid.
        min_fraction: Fraction of watermarked pixels from which a cell is watermarked.

    Returns:
        uint8 tensor [grid_size, grid_size] or [N, grid_size, grid_size].

    Raises:
        MetricInputError: bad shape, non-binary values, or size not divisible by the grid.
    """
    single = mask.ndim == MASK_SINGLE_NDIM
    batch = _require_binary(_as_batch(mask, "mask"), "mask")
    n_images, height, width = batch.shape
    if height % grid_size or width % grid_size:
        raise MetricInputError(f"mask {height}x{width} is not divisible by the grid {grid_size}")
    cell_h, cell_w = height // grid_size, width // grid_size
    blocks = batch.reshape(n_images, grid_size, cell_h, grid_size, cell_w)
    counts = blocks.sum(dim=(2, 4))
    grid = (counts >= min_fraction * cell_h * cell_w).to(torch.uint8)
    return grid[0] if single else grid


def per_image_miou(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-image mean IoU of the two classes, float64 [N].

    Args:
        prediction: [N, G, G] (or [G, G]) binary mask, 1 = watermarked.
        truth: Same shape, same convention.

    Raises:
        MetricInputError: shape mismatch or non-binary values.
    """
    pred = _require_binary(_as_batch(prediction, "prediction"), "prediction")
    gold = _require_binary(_as_batch(truth, "truth"), "truth")
    if pred.shape != gold.shape:
        raise MetricInputError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(gold.shape)}")
    ious: list[torch.Tensor] = []
    for class_value in CLASS_VALUES:
        pred_c = pred == bool(class_value)
        gold_c = gold == bool(class_value)
        intersection = (pred_c & gold_c).flatten(1).sum(dim=1).to(torch.float64)
        union = (pred_c | gold_c).flatten(1).sum(dim=1).to(torch.float64)
        # An empty union means both masks agree there is nothing of this class: IoU = 1.
        ious.append(
            torch.where(union == 0, torch.ones_like(union), intersection / union.clamp(min=1))
        )
    return torch.stack(ious).mean(dim=0)


def miou_score(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    """The leaderboard score: mean over images of the per-image mIoU, in [0, 1]."""
    per_image = per_image_miou(prediction, truth)
    if per_image.numel() == 0:
        raise MetricInputError("cannot score an empty set of masks")
    return float(per_image.mean().item())
