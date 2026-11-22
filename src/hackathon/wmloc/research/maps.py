"""Margin maps of the scale-space probe and the evidence map built from them.

For a probe at side L the extractor's per-position map has one cell per 32 network pixels, that
is `32 * 512 / L` pixels of the original image. Each map is turned into a z-score against its own
robust background (median and MAD), upsampled to the 128 grid and the best scale wins per cell:
a rectangle watermarked at its own scale lights up at the probe whose enlargement makes it span
the decoder resolution, and stays at background level elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.research.scoring import load_pass, margin

MAD_TO_SIGMA = 1.4826
"""Scale factor turning a median absolute deviation into a Gaussian standard deviation."""
MIN_SIGMA = 1e-3
"""Floor of the background deviation (a perfectly flat map must not blow the z-score up)."""
DEFAULT_SIDES: tuple[int, ...] = (384, 512, 768, 1024)


def pass_name_for_side(side: int) -> str:
    """Cache pass name of the scale-space probe at `side`."""
    return f"ss{side}"


def margin_map(split: str, method: str, name: str, message: np.ndarray, side: int) -> np.ndarray:
    """Soft-margin map [h, w] of one probe (mean over bits of logit x message sign)."""
    maps = load_pass(split, method, name, pass_name_for_side(side))["maps"]  # [bits, h, w]
    return margin(np.moveaxis(maps, 0, -1), message)


def robust_z(grid: np.ndarray) -> np.ndarray:
    """z-score of a map against its median / MAD background."""
    median = float(np.median(grid))
    sigma = max(MAD_TO_SIGMA * float(np.median(np.abs(grid - median))), MIN_SIGMA)
    return ((grid - median) / sigma).astype(np.float32)


def to_grid(small: np.ndarray) -> np.ndarray:
    """Bilinear resample of a square map to the 128 evaluation grid."""
    factor = GRID_SIZE / small.shape[0]
    return np.asarray(ndimage.zoom(small, factor, order=1, mode="nearest"), dtype=np.float32)


def evidence_grid(
    split: str, method: str, name: str, message: np.ndarray, sides: tuple[int, ...]
) -> np.ndarray:
    """Per-cell best z-score over the probes `sides`, [128, 128]."""
    best = None
    for side in sides:
        z = to_grid(robust_z(margin_map(split, method, name, message, side)))
        best = z if best is None else np.maximum(best, z)
    if best is None:
        raise ValueError("sides must not be empty")
    return best
