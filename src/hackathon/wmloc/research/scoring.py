"""Cached logits plus the known message -> agreement and margin scores, and dense maps.

Two complementary readings of a decoder output against the KNOWN message (chance level in
brackets):

* hard bit agreement, the fraction of bits whose sign matches the message  [0.5];
* soft margin, the mean over bits of `logit x message sign` (a signed confidence) [0.0].

The hard agreement saturates at 1.0 on a wide plateau, the soft margin keeps growing with the
amount of watermarked content in the crop, hence is the better regressor of "how much of this
window is marked".
"""

from __future__ import annotations

import numpy as np

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.extract import (
    CELL_PX,
    PASS_GLOBAL,
    PASS_POSMAP,
    SHIFT_OFFSETS,
    SHIFT_STEP_PX,
    cache_path,
)

CHANCE_AGREEMENT = 0.5
POSMAP_SIDE = IMAGE_SIZE // CELL_PX
"""Cells per side of one (unshifted) per-position map."""
DENSE_SIDE = 2 * POSMAP_SIDE
"""Cells per side after stitching the 2x2 shifts (spacing SHIFT_STEP_PX)."""
GRID_CELL_PX = IMAGE_SIZE // GRID_SIZE


class MissingCacheError(ResearchError):
    """A cache file needed by the analysis has not been extracted yet."""


def message_signs(message: np.ndarray) -> np.ndarray:
    """0/1 bits -> -1/+1 signs, shape [n_bits]."""
    return message.astype(np.float32) * 2.0 - 1.0


def agreement(logits: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Hard bit agreement along the last axis, for logits [..., n_bits]."""
    signed = logits.astype(np.float32) * message_signs(message)
    return np.asarray((signed > 0).mean(axis=-1))


def margin(logits: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Soft margin along the last axis, for logits [..., n_bits]."""
    return np.asarray((logits.astype(np.float32) * message_signs(message)).mean(axis=-1))


def load_pass(split: str, method: str, name: str, pass_name: str) -> dict[str, np.ndarray]:
    """Arrays of one cached pass.

    Raises:
        MissingCacheError: the file does not exist.
    """
    path = cache_path(split, method, name, pass_name)
    if not path.is_file():
        raise MissingCacheError(f"missing {path}")
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def global_logits(split: str, method: str, name: str) -> np.ndarray:
    """Logits [n_bits] of the whole image."""
    return np.asarray(load_pass(split, method, name, PASS_GLOBAL)["logits"][0])


def dense_position_maps(maps: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Stitch the 2x2 shifted per-position maps into one margin grid [16, 16].

    `maps` is [4, n_bits, 8, 8] in `SHIFT_OFFSETS` order. Content shifted down/right by d moves
    the point (y) to y + d, so cell i of a shifted image looks at the original pixel
    CELL_PX * i + CELL_PX / 2 - d: shifts interleave at SHIFT_STEP_PX spacing.

    Returns:
        float32 [16, 16] of per-position soft margins; slot s sits at pixel
        SHIFT_STEP_PX * s (see `dense_cell_centres`).
    """
    per_shift = margin(np.moveaxis(maps, 1, -1), message)  # [4, 8, 8]
    dense = np.zeros((DENSE_SIDE, DENSE_SIDE), dtype=np.float32)
    for index, (row_steps, col_steps) in enumerate(SHIFT_OFFSETS):
        # Centre of cell i in original px: 64 i + 32 - 32 k = 32 (2 i + 1 - k), i.e. slot
        # 2 i + 1 - k; every slot 0..15 is hit by exactly one shift combination.
        rows = 2 * np.arange(POSMAP_SIDE) + 1 - row_steps
        cols = 2 * np.arange(POSMAP_SIDE) + 1 - col_steps
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if 0 <= r < DENSE_SIDE and 0 <= c < DENSE_SIDE:
                    dense[r, c] = per_shift[index, i, j]
    return dense


def dense_cell_centres() -> np.ndarray:
    """Pixel centre (at 512) of each dense slot: SHIFT_STEP_PX * slot."""
    return SHIFT_STEP_PX * (np.arange(DENSE_SIDE) + 0.0)


def dense_to_grid(dense: np.ndarray) -> np.ndarray:
    """Bilinear resample of a [16, 16] dense map to the 128 evaluation grid (float32)."""
    centres = dense_cell_centres()
    cell_centres = (np.arange(GRID_SIZE) + 0.5) * GRID_CELL_PX
    rows = _interp_matrix(centres, cell_centres)
    return np.asarray(rows @ dense @ rows.T, dtype=np.float32)


def _interp_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linear interpolation operator [len(target), len(source)] (edge values held)."""
    matrix = np.zeros((len(target), len(source)), dtype=np.float32)
    for index, value in enumerate(target):
        position = np.clip(np.interp(value, source, np.arange(len(source))), 0, len(source) - 1)
        low = int(np.floor(position))
        high = min(low + 1, len(source) - 1)
        weight = position - low
        matrix[index, low] += 1.0 - weight
        matrix[index, high] += weight
    return matrix


def posmap_score_grid(split: str, method: str, name: str, message: np.ndarray) -> np.ndarray:
    """[128, 128] soft-margin map from the cached per-position maps of one image."""
    maps = load_pass(split, method, name, PASS_POSMAP)["maps"]
    return dense_to_grid(dense_position_maps(maps, message))
