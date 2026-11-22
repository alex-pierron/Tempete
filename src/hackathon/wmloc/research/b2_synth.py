"""Synthetic training pairs with REAL mask shapes, built with the official embedders (team B2).

The 510 real masks are the only ground truth of what the organisers' edits look like (object
silhouettes, blobs, thin strips). Team B's phase-1 generator drew rectangles and ellipses, which the
head never saw at test time. Here every synthetic sample reuses a real shape, randomly flipped,
rotated, scaled and shifted, on a clean base image that is watermarked by the true embedder:

* FULL regime: the whole image is watermarked, then the hole (a real hole shape) is restored to the
  clean original. Label = 1 outside the hole.
* RECT regime: only the region is watermarked, embedded on the crop of its bounding box and pasted
  back inside the shape (the process the phase-1 probes measured on the real data).

Everything here is pure numpy so it can be tested without a GPU; the embedder is injected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from hackathon.wmloc.research.errors import ResearchError

GRID = 128
IMAGE = 512
UPSCALE = IMAGE // GRID
MIN_AREA_FRACTION = 0.004
MAX_AREA_FRACTION = 0.45
SCALE_RANGE: tuple[float, float] = (0.6, 1.5)
MAX_PLACEMENT_TRIES = 12
SMOOTH_SIGMA = 1.2
"""Gaussian sigma (in 512 px) used when the 128 grid shape is upsampled, to avoid 4 px staircases."""
MIN_CROP_SIDE = 16
EmbedFn = Callable[[NDArray[np.uint8], NDArray[np.uint8]], NDArray[np.uint8]]
"""`embed(image uint8 [h, w, 3], bits [n]) -> watermarked uint8 [h, w, 3]` (any size)."""


class SynthError(ResearchError):
    """A shape library is empty or a composite is inconsistent."""


@dataclass(frozen=True)
class ShapeLibrary:
    """Real shapes of the training masks, split by regime.

    Attributes:
        holes: Restored regions of the FULL regime, bool [n, 128, 128] (True = hole).
        regions: Watermarked regions of the RECT regime, bool [n, 128, 128] (True = marked).
        hole_folds: Cross-validation fold of the real image each hole was taken from.
        region_folds: Same for the regions. A synthetic sample inherits the fold of its shape, so a
            training fold can exclude the shapes of its own held-out images (no shape leakage).
    """

    holes: NDArray[np.bool_]
    regions: NDArray[np.bool_]
    hole_folds: NDArray[np.int64]
    region_folds: NDArray[np.int64]


def build_library(
    masks: NDArray[np.uint8],
    is_full: NDArray[np.bool_],
    folds: NDArray[np.int64] | None = None,
) -> ShapeLibrary:
    """Split real masks [n, 128, 128] (1 = marked) into hole and region shapes.

    Args:
        masks: Real train masks, 1 = marked.
        is_full: Regime flag of each mask.
        folds: Cross-validation fold of each mask (zeros when omitted).

    Raises:
        SynthError: one of the regimes has no usable shape.
    """
    holes = masks[is_full] == 0
    regions = masks[~is_full] == 1
    if len(holes) == 0 or len(regions) == 0:
        raise SynthError("need at least one full-regime and one rectangle-regime mask")
    fold_ids = np.zeros(len(masks), dtype=np.int64) if folds is None else folds.astype(np.int64)
    return ShapeLibrary(
        holes=holes.astype(np.bool_),
        regions=regions.astype(np.bool_),
        hole_folds=fold_ids[is_full],
        region_folds=fold_ids[~is_full],
    )


def augment_shape(shape: NDArray[np.bool_], rng: np.random.Generator) -> NDArray[np.bool_]:
    """Random symmetry, isotropic scale and shift of a 128 grid shape; area kept in range.

    Falls back to the unscaled symmetric copy when no placement keeps the area acceptable.
    """
    element = int(rng.integers(8))
    moved = np.rot90(shape, element % 4)
    moved = np.fliplr(moved) if element >= 4 else moved
    base_area = float(moved.mean())
    for _ in range(MAX_PLACEMENT_TRIES):
        scale = float(rng.uniform(*SCALE_RANGE))
        shift = rng.uniform(-0.3, 0.3, size=2) * GRID
        centre = (GRID - 1) / 2.0
        matrix = np.eye(2) / scale
        offset = np.array([centre, centre]) - matrix @ (np.array([centre, centre]) + shift)
        warped = ndimage.affine_transform(moved.astype(np.float32), matrix, offset=offset, order=1)
        candidate = warped > 0.5
        area = float(candidate.mean())
        if MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION and area > 0.25 * base_area:
            return np.asarray(candidate, dtype=np.bool_)
    return np.asarray(moved, dtype=np.bool_)


def upsample_mask(mask128: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """128 -> 512 with smooth edges (bilinear zoom of a smoothed mask, thresholded)."""
    smooth = ndimage.gaussian_filter(mask128.astype(np.float32), sigma=SMOOTH_SIGMA / UPSCALE)
    zoomed = ndimage.zoom(smooth, UPSCALE, order=1)
    return np.asarray(zoomed > 0.5, dtype=np.bool_)


def downsample_mask(mask512: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """512 -> 128 with the official rule: a cell is marked when at least half of it is."""
    blocks = mask512.reshape(GRID, UPSCALE, GRID, UPSCALE).sum(axis=(1, 3))
    return np.asarray(blocks >= (UPSCALE * UPSCALE) / 2, dtype=np.uint8)


def compose_full(
    clean: NDArray[np.uint8],
    bits: NDArray[np.uint8],
    hole128: NDArray[np.bool_],
    embed: EmbedFn,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """FULL regime sample: watermark everything, restore the hole. Returns (image, label128)."""
    marked = embed(clean, bits)
    hole = upsample_mask(hole128)[..., None]
    image = np.where(hole, clean, marked).astype(np.uint8)
    return image, downsample_mask(~hole[..., 0])


def compose_rect(
    clean: NDArray[np.uint8],
    bits: NDArray[np.uint8],
    region128: NDArray[np.bool_],
    embed: EmbedFn,
    max_aspect: float | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """RECT regime sample: watermark the bbox crop, paste it back inside the region shape.

    Args:
        max_aspect: Largest accepted long / short side ratio of the bounding box (TrustMark only
            watermarks the whole patch when the ratio is at most 2); None accepts any.

    Raises:
        SynthError: the region is empty, too thin to embed, or too elongated.
    """
    region = upsample_mask(region128)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        raise SynthError("empty region")
    top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    short, long = sorted((bottom - top, right - left))
    if short < MIN_CROP_SIDE:
        raise SynthError("region too thin to embed")
    if max_aspect is not None and long > max_aspect * short:
        raise SynthError("region too elongated for this embedder")
    marked_crop = embed(np.ascontiguousarray(clean[top:bottom, left:right]), bits)
    image = clean.copy()
    inside = region[top:bottom, left:right, None]
    image[top:bottom, left:right] = np.where(inside, marked_crop, clean[top:bottom, left:right])
    return image, downsample_mask(region)


def sample_pair(
    clean: NDArray[np.uint8],
    bits: NDArray[np.uint8],
    library: ShapeLibrary,
    embed: EmbedFn,
    rng: np.random.Generator,
    full_probability: float = 0.5,
    max_aspect: float | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], bool, int]:
    """One synthetic (image, label128, is_full, shape_fold) sample, retrying on bad regions."""
    for _ in range(MAX_PLACEMENT_TRIES):
        is_full = bool(rng.random() < full_probability)
        pool = library.holes if is_full else library.regions
        folds = library.hole_folds if is_full else library.region_folds
        source = int(rng.integers(len(pool)))
        shape = augment_shape(pool[source], rng)
        try:
            if is_full:
                image, label = compose_full(clean, bits, shape, embed)
            else:
                image, label = compose_rect(clean, bits, shape, embed, max_aspect)
        except (SynthError, RuntimeError, ValueError):
            # A degenerate region, or an embedder that refuses a very small / odd crop: redraw.
            continue
        return image, label, is_full, int(folds[source])
    raise SynthError("could not draw a valid sample")
