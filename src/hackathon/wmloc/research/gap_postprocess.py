"""Post-processing of head probabilities: threshold, shape prior, small-zone abstention.

Pure functions of one probability map. The shape priors act on the MINORITY zone (the hole when
the mask is mostly marked, the marked zone otherwise), which is the only part the metric is
sensitive to; the regime is read from the mask itself, so nothing here needs the train labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.postprocess import bounding_box_mask, largest_component

SHAPE_NONE = "none"
SHAPE_LARGEST = "largest"
SHAPE_BOX_LARGEST = "box_largest"
SHAPE_BOX_ALL = "box_all"
SHAPE_FILL = "fill"
SHAPE_OPEN = "open"
SHAPE_BLANK = "blank"
SHAPES: tuple[str, ...] = (
    SHAPE_NONE,
    SHAPE_LARGEST,
    SHAPE_BOX_LARGEST,
    SHAPE_BOX_ALL,
    SHAPE_FILL,
    SHAPE_OPEN,
    SHAPE_BLANK,
)
REGIME_SPLIT_FRACTION = 0.5
"""A mask marked on at least half of the image is read as the full regime."""
REFERENCE_THRESHOLD = 0.5
"""Threshold of the mask the regime is read from (the one the base candidate uses)."""


class GapPostprocessError(ResearchError):
    """Unknown shape prior."""


@dataclass(frozen=True)
class PostConfig:
    """One post-processing rule.

    Attributes:
        threshold: Probability threshold of the watermarked class.
        shape: Shape prior applied to the minority zone (one of `SHAPES`).
        min_area: Minority zones smaller than this image share are erased (abstention).
    """

    threshold: float = REFERENCE_THRESHOLD
    shape: str = SHAPE_NONE
    min_area: float = 0.0


def predicted_full(probability: NDArray[np.float32]) -> bool:
    """Regime read from the reference mask: True = mostly marked (find the hole)."""
    return float((probability >= REFERENCE_THRESHOLD).mean()) >= REGIME_SPLIT_FRACTION


def shape_minority(minority: NDArray[np.uint8], shape: str) -> NDArray[np.uint8]:
    """Apply the shape prior `shape` to a minority-zone mask.

    Raises:
        GapPostprocessError: unknown shape.
    """
    if shape == SHAPE_NONE:
        return minority
    if shape == SHAPE_LARGEST:
        return largest_component(minority)
    if shape == SHAPE_BOX_LARGEST:
        return bounding_box_mask(largest_component(minority))
    if shape == SHAPE_BOX_ALL:
        return bounding_box_mask(minority)
    if shape == SHAPE_FILL:
        return np.asarray(ndimage.binary_fill_holes(minority), dtype=np.uint8)
    if shape == SHAPE_OPEN:
        return np.asarray(ndimage.binary_opening(minority), dtype=np.uint8)
    if shape == SHAPE_BLANK:
        return np.zeros_like(minority)
    raise GapPostprocessError(f"shape must be one of {SHAPES}, got {shape!r}")


def postprocess(probability: NDArray[np.float32], config: PostConfig) -> NDArray[np.uint8]:
    """Binary mask [128, 128] of one probability map under `config`.

    The regime never flips: it is fixed by the reference mask, then the threshold and the shape
    prior only reshape the minority zone inside that regime.
    """
    full = predicted_full(probability)
    marked = probability >= config.threshold
    minority = (~marked if full else marked).astype(np.uint8)
    minority = shape_minority(minority, config.shape)
    if float(minority.mean()) < config.min_area:
        minority = np.zeros_like(minority)
    return (1 - minority if full else minority).astype(np.uint8)


def postprocess_batch(
    probabilities: NDArray[np.float32], configs: list[PostConfig]
) -> NDArray[np.uint8]:
    """Masks [n, 128, 128]; image `i` uses `configs[i]`."""
    return np.stack(
        [postprocess(p, c) for p, c in zip(probabilities, configs, strict=True)]
    ).astype(np.uint8)
