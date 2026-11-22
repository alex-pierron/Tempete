"""Occlusion maps against the known message (team C, full regime).

The full regime watermarks the whole image and then restores an object-shaped hole to the clean
original. The decoder's soft margin (mean of logit x message sign) of the whole image is, to a
first approximation, a sum of per-region contributions. Degrading one window of the image and
re-decoding measures that window's contribution: a window inside the hole has nothing to lose.

The degradation is a strong Gaussian blur of the window: a blur with sigma <= 2 px leaves
TrustMark's mid-frequency pattern intact (measured: no drop, AUC 0.48), a blur of 10 px removes
it (AUC 0.70 at 64 px windows, 8 TrustMark images).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from hackathon.wmloc.constants import IMAGE_SIZE
from hackathon.wmloc.research.c_batching import batched_logits
from hackathon.wmloc.research.c_rect_search import LogitDecoder, resize_batch
from hackathon.wmloc.research.errors import ResearchError

GAUSSIAN_TRUNCATE = 3.0
"""Kernel radius in standard deviations."""
DEFAULT_BLOCK_PX = 32
VARIANT_CHUNK = 256
"""Occluded images built and decoded per call (3 MB each); the decoder batches them further."""
DEFAULT_STRIDE_PX = 16


class OcclusionError(ResearchError):
    """Invalid occlusion geometry."""


@dataclass(frozen=True)
class OcclusionMap:
    """Margin drop of every occluded window of one image.

    Attributes:
        base_margin: Soft margin of the untouched image.
        drop: [ny, nx] margin of the untouched image minus margin with the window blurred.
        origins: [ny, nx, 2] top-left pixel of each window.
        block: Window side in pixels.
    """

    base_margin: float
    drop: np.ndarray
    origins: np.ndarray
    block: int


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a [3, H, W] image (reflect padding).

    Raises:
        OcclusionError: sigma <= 0.
    """
    if sigma <= 0:
        raise OcclusionError(f"sigma must be positive, got {sigma}")
    radius = max(1, int(GAUSSIAN_TRUNCATE * sigma))
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32, device=image.device)
    kernel = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    padded = functional.pad(image[None], (radius,) * 4, mode="reflect")
    horizontal = functional.conv2d(padded, kernel.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    vertical = functional.conv2d(horizontal, kernel.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return vertical[0]


def window_origins(block: int, stride: int) -> np.ndarray:
    """Top-left offsets along one axis, covering the image with the last window flush."""
    if block <= 0 or stride <= 0 or block > IMAGE_SIZE:
        raise OcclusionError(f"bad block/stride ({block}, {stride})")
    origins = list(range(0, IMAGE_SIZE - block + 1, stride))
    if origins[-1] != IMAGE_SIZE - block:
        origins.append(IMAGE_SIZE - block)
    return np.asarray(origins, dtype=np.int32)


def signed_margin(decoder: LogitDecoder, images: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Soft margin of each image of a [B, 3, 512, 512] batch, [B] (resized per batch)."""
    crops = resize_batch(images, decoder.method, decoder.resolution)
    return (batched_logits(decoder, crops) * signs).mean(dim=1)


def occlusion_map(
    decoder: LogitDecoder,
    image: torch.Tensor,
    message: np.ndarray,
    sigma: float,
    block: int = DEFAULT_BLOCK_PX,
    stride: int = DEFAULT_STRIDE_PX,
) -> OcclusionMap:
    """Margin drop for every `block` x `block` window (spacing `stride`) blurred with `sigma`."""
    signs = torch.as_tensor(message, dtype=torch.float32) * 2.0 - 1.0
    device = str(getattr(decoder, "device", "cpu"))
    # Antialiased resizing exists on CPU and CUDA only (not MPS): keep the images on the GPU there.
    image = image.to(device if device.startswith("cuda") else "cpu")
    base = float(signed_margin(decoder, image[None], signs)[0])
    blurred = gaussian_blur(image, sigma)
    axis = window_origins(block, stride)
    origins = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    flat = origins.reshape(-1, 2)
    drops = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), VARIANT_CHUNK):
        variants = []
        for top, left in flat[start : start + VARIANT_CHUNK].tolist():
            variant = image.clone()
            variant[:, top : top + block, left : left + block] = blurred[
                :, top : top + block, left : left + block
            ]
            variants.append(variant)
        margins = signed_margin(decoder, torch.stack(variants), signs)
        drops[start : start + len(variants)] = base - margins.numpy()
    return OcclusionMap(base, drops.reshape(len(axis), len(axis)), origins, block)


def to_grid(occlusion: OcclusionMap, grid: int = 128) -> np.ndarray:
    """Average the window drops into a [grid, grid] map (each window spreads over its cells).

    Cells covered by no window cannot occur (the last window is flush); a cell is the mean of
    the drops of the windows that contain it.
    """
    cell = IMAGE_SIZE // grid
    total = np.zeros((grid, grid), dtype=np.float64)
    count = np.zeros((grid, grid), dtype=np.float64)
    span = occlusion.block // cell
    for (top, left), value in zip(
        occlusion.origins.reshape(-1, 2).tolist(), occlusion.drop.reshape(-1).tolist(), strict=True
    ):
        r, c = top // cell, left // cell
        total[r : r + span, c : c + span] += value
        count[r : r + span, c : c + span] += 1.0
    return (total / np.maximum(count, 1.0)).astype(np.float32)
