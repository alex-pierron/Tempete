"""Score map -> binary mask: threshold, connected components, shape priors, area matching.

Every function maps float [128, 128] evidence (higher = more likely watermarked) to a uint8
mask; the parameters are scalars so they can be calibrated by k-fold cross-validation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.research.errors import ResearchError

MODE_THRESHOLD = "threshold"
MODE_LARGEST = "largest"
MODE_BBOX_LARGEST = "bbox_largest"
MODE_BBOX_ALL = "bbox_all"
MODES: tuple[str, ...] = (MODE_THRESHOLD, MODE_LARGEST, MODE_BBOX_LARGEST, MODE_BBOX_ALL)
OPEN_ITERATIONS = 1


class PostprocessError(ResearchError):
    """Unknown post-processing mode."""


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the connected component with the most cells (8-connectivity); empty stays empty."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return np.asarray(labels == (int(np.argmax(sizes)) + 1), dtype=np.uint8)


def bounding_box_mask(mask: np.ndarray) -> np.ndarray:
    """Filled bounding box of the on-cells; empty stays empty."""
    cells = np.argwhere(mask > 0)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if len(cells) == 0:
        return out
    (top, left), (bottom, right) = cells.min(axis=0), cells.max(axis=0)
    out[top : bottom + 1, left : right + 1] = 1
    return out


def evidence_to_mask(
    evidence: np.ndarray, threshold: float, mode: str = MODE_THRESHOLD
) -> np.ndarray:
    """Threshold the evidence, then apply the shape prior `mode`.

    Raises:
        PostprocessError: unknown mode.
    """
    if mode not in MODES:
        raise PostprocessError(f"mode must be one of {MODES}, got {mode!r}")
    mask = (evidence >= threshold).astype(np.uint8)
    if mode == MODE_THRESHOLD:
        return mask
    if mode == MODE_LARGEST:
        return largest_component(mask)
    if mode == MODE_BBOX_LARGEST:
        return bounding_box_mask(largest_component(mask))
    return bounding_box_mask(mask)


def top_fraction_mask(evidence: np.ndarray, fraction: float) -> np.ndarray:
    """Mask of the `fraction` of cells with the highest evidence (area-matched cut)."""
    if fraction <= 0.0:
        return np.zeros(evidence.shape, dtype=np.uint8)
    count = min(evidence.size, max(1, round(fraction * evidence.size)))
    cut = np.partition(evidence.reshape(-1), -count)[-count]
    return np.asarray(evidence >= cut, dtype=np.uint8)


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill the enclosed holes of a mask (the restored objects of the rectangle regime)."""
    return np.asarray(ndimage.binary_fill_holes(mask), dtype=np.uint8)


def open_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological opening, removes isolated cells."""
    return np.asarray(ndimage.binary_opening(mask, iterations=OPEN_ITERATIONS), dtype=np.uint8)
