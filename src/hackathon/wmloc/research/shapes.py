"""Random region masks that imitate the real train masks (rectangles, blobs, silhouettes).

The real masks (see the team-B report, section "Donnees reelles") are NOT only rectangles: the
minority region is a rectangle (often a thin band), a smooth multi-component blob or an object
silhouette. Areas are log-uniform-like (median 4-6 %, range 0.3-50 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

IMAGE_SIZE = 512
GRID_SIZE = 128
CELL_SIZE = IMAGE_SIZE // GRID_SIZE
CELL_FRACTION = 0.5
"""Official rule: a 128-grid cell is 1 when at least this fraction of its pixels is 1."""

AREA_FRACTION_RANGE = (0.004, 0.45)
RECT_MIN_SIDE_PX = 8
RECT_ASPECT_NEAR_SQUARE_PROBABILITY = 0.55
RECT_NEAR_SQUARE_MAX_ASPECT = 2.0
BLOB_NOISE_SIDE = 64
BLOB_SIGMA_RANGE = (1.5, 7.0)
POLYGON_VERTEX_RANGE = (7, 18)
POLYGON_RADIUS_JITTER = (0.45, 1.25)
POLYGON_UNION_PARTS = (1, 3)
AREA_TOLERANCE = 0.25

KIND_RECT = "rect"
KIND_BLOB = "blob"
KIND_POLYGON = "polygon"
KINDS: tuple[str, ...] = (KIND_RECT, KIND_BLOB, KIND_POLYGON)

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class RegionSpec:
    """A sampled region.

    Attributes:
        kind: One of `KINDS`.
        mask: Boolean [512, 512], True inside the region.
        bbox: (y0, y1, x0, x1) of the region (exclusive upper bounds).
    """

    kind: str
    mask: BoolMask
    bbox: tuple[int, int, int, int]


def bounding_box(mask: BoolMask) -> tuple[int, int, int, int]:
    """Tight (y0, y1, x0, x1) box of a non-empty mask.

    Raises:
        ValueError: the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty mask has no bounding box")
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def downsample_mask(mask: BoolMask) -> NDArray[np.uint8]:
    """Official 512 -> 128 reduction (a cell is 1 when >= 50 % of its 4x4 pixels are 1)."""
    blocks = mask.reshape(GRID_SIZE, CELL_SIZE, GRID_SIZE, CELL_SIZE).mean(axis=(1, 3))
    return (blocks >= CELL_FRACTION).astype(np.uint8)


def sample_area_fraction(rng: np.random.Generator) -> float:
    """Log-uniform area fraction in `AREA_FRACTION_RANGE` (median close to the real 4-6 %)."""
    low, high = AREA_FRACTION_RANGE
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_rect(rng: np.random.Generator, area_fraction: float, max_aspect: float) -> RegionSpec:
    """Axis-aligned rectangle with a given area and an aspect ratio drawn in [1, max_aspect]."""
    if rng.random() < RECT_ASPECT_NEAR_SQUARE_PROBABILITY:
        aspect = float(rng.uniform(1.0, min(RECT_NEAR_SQUARE_MAX_ASPECT, max_aspect)))
    else:
        aspect = float(math.exp(rng.uniform(0.0, math.log(max_aspect))))
    area_px = area_fraction * IMAGE_SIZE**2
    short = math.sqrt(area_px / aspect)
    long_side = short * aspect
    if rng.random() < 0.5:
        height, width = long_side, short
    else:
        height, width = short, long_side
    h = int(np.clip(round(height), RECT_MIN_SIDE_PX, IMAGE_SIZE))
    w = int(np.clip(round(width), RECT_MIN_SIDE_PX, IMAGE_SIZE))
    y0 = int(rng.integers(0, IMAGE_SIZE - h + 1))
    x0 = int(rng.integers(0, IMAGE_SIZE - w + 1))
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = True
    return RegionSpec(KIND_RECT, mask, (y0, y0 + h, x0, x0 + w))


def sample_blob(rng: np.random.Generator, area_fraction: float) -> RegionSpec:
    """Smooth organic region: threshold a low-frequency noise field at the area quantile."""
    sigma = float(rng.uniform(*BLOB_SIGMA_RANGE))
    for _ in range(8):
        field = ndimage.gaussian_filter(
            rng.normal(size=(BLOB_NOISE_SIDE, BLOB_NOISE_SIDE)), sigma, mode="wrap"
        )
        big = ndimage.zoom(field, IMAGE_SIZE / BLOB_NOISE_SIDE, order=1)
        mask = big >= np.quantile(big, 1.0 - area_fraction)
        if mask.any():
            return RegionSpec(KIND_BLOB, mask, bounding_box(mask))
        sigma *= 0.7
    raise ValueError("blob sampling failed to produce a non-empty region")


def sample_polygon(rng: np.random.Generator, area_fraction: float) -> RegionSpec:
    """Silhouette-like region: union of 1-3 jittered star polygons rescaled to the area."""
    import cv2

    parts = int(rng.integers(POLYGON_UNION_PARTS[0], POLYGON_UNION_PARTS[1] + 1))
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    base_radius = math.sqrt(area_fraction * IMAGE_SIZE**2 / (math.pi * parts) / 0.7)
    centre = rng.uniform(0.15 * IMAGE_SIZE, 0.85 * IMAGE_SIZE, size=2)
    for _ in range(parts):
        n_vertices = int(rng.integers(POLYGON_VERTEX_RANGE[0], POLYGON_VERTEX_RANGE[1] + 1))
        angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, n_vertices))
        radii = base_radius * rng.uniform(*POLYGON_RADIUS_JITTER, size=n_vertices)
        squash = rng.uniform(0.5, 1.6)
        offset = rng.normal(0.0, base_radius * 0.9, size=2)
        points = np.stack(
            [
                centre[0] + offset[0] + radii * np.cos(angles) * squash,
                centre[1] + offset[1] + radii * np.sin(angles) / squash,
            ],
            axis=1,
        )
        cv2.fillPoly(canvas, [np.round(points).astype(np.int32)], 1)
    mask = canvas.astype(bool)
    if not mask.any():
        return sample_blob(rng, area_fraction)
    return RegionSpec(KIND_POLYGON, mask, bounding_box(mask))


def sample_region(
    rng: np.random.Generator, kind_probabilities: dict[str, float], max_rect_aspect: float
) -> RegionSpec:
    """Draw a region kind, then a region, retrying until its area is near the target.

    Args:
        rng: Random generator.
        kind_probabilities: Probability of each kind in `KINDS` (normalised internally).
        max_rect_aspect: Largest aspect ratio of a rectangle (16 PixelSeal/StegaStamp, 2 TrustMark).

    Returns:
        A region whose 128-grid mask is non-empty.
    """
    kinds = list(kind_probabilities)
    weights = np.array([kind_probabilities[k] for k in kinds], dtype=float)
    weights /= weights.sum()
    for _ in range(20):
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        area = sample_area_fraction(rng)
        if kind == KIND_RECT:
            region = sample_rect(rng, area, max_rect_aspect)
        elif kind == KIND_BLOB:
            region = sample_blob(rng, area)
        else:
            region = sample_polygon(rng, area)
        realised = float(region.mask.mean())
        if (
            abs(realised - area) <= AREA_TOLERANCE * area + 0.002
            and downsample_mask(region.mask).any()
        ):
            return region
    return sample_rect(rng, 0.05, max_rect_aspect)
