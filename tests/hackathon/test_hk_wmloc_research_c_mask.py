"""`Box.to_mask` must reproduce the official 4 x 4 majority downsampling of a pixel box."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE
from hackathon.wmloc.research.c_rect_search import Box

CELL_PX = IMAGE_SIZE // GRID_SIZE


def official_downsample(mask: np.ndarray) -> np.ndarray:
    """Official rule (kit `downsample_mask`): a cell is 1 if at least 50 % of its pixels are 1."""
    cells = mask.astype(np.float32).reshape(GRID_SIZE, CELL_PX, GRID_SIZE, CELL_PX)
    return (cells.mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


@pytest.mark.parametrize("top", [0, 1, 2, 3, 5, 6, 7, 10])
@pytest.mark.parametrize("bottom_extra", [8, 9, 10, 11, 13])
def test_box_mask_matches_the_official_downsampling_of_the_pixel_box(
    top: int, bottom_extra: int
) -> None:
    box = Box(top=top, left=top + 3, bottom=top + bottom_extra, right=top + 3 + bottom_extra + 2)
    pixels = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    pixels[box.top : box.bottom, box.left : box.right] = 1
    assert np.array_equal(box.to_mask(), official_downsample(pixels))
