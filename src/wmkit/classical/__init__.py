"""Training-free classical watermarking baselines and the machinery they share.

All three methods (`dct_qim`, `spread_spectrum`, `dwt_dct_svd`) follow the same recipe,
implemented once in `ReferenceLumaWatermarker`:

1. Take the luminance Y (BT.601, full range) of the image.
2. Resize Y to a fixed square *reference size* (256 by default), embed there, and
   resize the resulting luminance *residual* back to the native size. Extraction resizes
   the (possibly attacked) luminance to the same reference size before decoding, which
   is what makes decoding robust to rescaling: the decoder never needs to know the
   original size.
3. Add the residual to R, G and B alike, which leaves Cb and Cr untouched.

Computations run on the CPU in float64 whatever the `device` of the factory; outputs are
returned on the device of the input images, in float32.

Strength semantics differ per method (see each module) but all defaults target
`TARGET_PSNR_DB` before clipping.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as torch_f

TARGET_PSNR_DB: float = 40.0
"""PSNR the default strength of every method is calibrated for (before clipping)."""

REFERENCE_SIZE: int = 256
"""Side of the square luminance grid on which embedding and decoding happen."""

DEFAULT_N_BITS: int = 32
RGB_CHANNELS: int = 3
LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)
"""ITU-R BT.601 luma weights; Y' = Y + d shifts R, G and B by d and keeps Cb, Cr fixed."""

CLOSED_LOOP_ITERATIONS: int = 2
"""Feedback passes compensating clipping and resampling losses at embedding time."""
MIN_IMAGE_SIDE: int = 1
EPSILON: float = 1e-12
QIM_LATTICE_OFFSET_FRACTION: float = 0.5
"""Bit 1 uses the lattice shifted by half a step relative to bit 0."""


class ClassicalWatermarkError(ValueError):
    """Raised for invalid configuration or inputs of a classical watermarker."""


def psnr_to_rms_error(psnr_db: float) -> float:
    """RMS pixel error (images in [0, 1]) that corresponds to `psnr_db`."""
    return float(10.0 ** (-psnr_db / 20.0))


def luma(images: torch.Tensor) -> torch.Tensor:
    """BT.601 luminance of [B, 3, H, W] images, as float64 [B, H, W] on the CPU."""
    if images.ndim != 4 or images.shape[1] != RGB_CHANNELS:
        raise ClassicalWatermarkError(
            f"expected images of shape [B, 3, H, W], got {tuple(images.shape)}"
        )
    weights = torch.tensor(LUMA_WEIGHTS, dtype=torch.float64).view(1, RGB_CHANNELS, 1, 1)
    rgb = images.detach().to("cpu").to(torch.float64)
    luminance: torch.Tensor = (rgb * weights).sum(dim=1)
    return luminance


def resize_plane(planes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear (anti-aliased) resize of [B, H, W] planes; returns the input if unchanged."""
    if tuple(planes.shape[-2:]) == size:
        return planes
    resized: torch.Tensor = torch_f.interpolate(
        planes[:, None], size=size, mode="bilinear", align_corners=False, antialias=True
    )[:, 0]
    return resized


def dct_matrix(size: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix [size, size] (float64): `C @ x` is the DCT of a column x."""
    positions = torch.arange(size, dtype=torch.float64)
    frequencies = positions.view(-1, 1)
    matrix = torch.cos(math.pi * (2.0 * positions.view(1, -1) + 1.0) * frequencies / (2.0 * size))
    matrix = matrix * math.sqrt(2.0 / size)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix


def block_dct(planes: torch.Tensor, block: int) -> torch.Tensor:
    """Blockwise orthonormal 2-D DCT: [B, H, W] -> [B, H/block, W/block, block, block]."""
    batch, height, width = planes.shape
    if height % block or width % block:
        raise ClassicalWatermarkError(f"plane {height}x{width} is not a multiple of {block}")
    tiles = planes.reshape(batch, height // block, block, width // block, block)
    tiles = tiles.permute(0, 1, 3, 2, 4)
    matrix = dct_matrix(block)
    transformed: torch.Tensor = matrix @ tiles @ matrix.T
    return transformed


def block_idct(blocks: torch.Tensor) -> torch.Tensor:
    """Inverse of `block_dct`: [B, nbh, nbw, block, block] -> [B, H, W]."""
    batch, blocks_h, blocks_w, block, _ = blocks.shape
    matrix = dct_matrix(block)
    tiles = matrix.T @ blocks @ matrix
    plane: torch.Tensor = tiles.permute(0, 1, 3, 2, 4).reshape(
        batch, blocks_h * block, blocks_w * block
    )
    return plane


def qim_embed(values: torch.Tensor, bits: torch.Tensor, step: float) -> torch.Tensor:
    """Quantisation index modulation: snap `values` onto the lattice of their bit.

    Bit 0 lattice: k * step. Bit 1 lattice: (k + 1/2) * step. The embedding error is
    uniform in [-step/2, step/2], i.e. variance step^2 / 12.
    """
    offset = bits * (QIM_LATTICE_OFFSET_FRACTION * step)
    snapped: torch.Tensor = torch.round((values - offset) / step) * step + offset
    return snapped


def qim_margin(values: torch.Tensor, step: float) -> torch.Tensor:
    """Normalised evidence for bit 1 versus bit 0, in [-1, 1].

    +1: exactly on the bit-1 lattice; -1: exactly on the bit-0 lattice; 0: equidistant.
    """
    remainder = torch.remainder(values, step)
    distance_zero = torch.minimum(remainder, step - remainder)
    distance_one = (remainder - QIM_LATTICE_OFFSET_FRACTION * step).abs()
    margin: torch.Tensor = (distance_zero - distance_one) / (QIM_LATTICE_OFFSET_FRACTION * step)
    return margin


def balanced_assignment(n_units: int, n_bits: int, rng: np.random.Generator) -> torch.Tensor:
    """Keyed assignment of embedding units to payload bits, as balanced as possible.

    Spreading each bit over pseudo-random positions keeps a local defect (saturated
    area, cropped corner) from wiping out a whole bit.
    """
    if n_units < n_bits:
        raise ClassicalWatermarkError(
            f"{n_bits} bits do not fit in {n_units} embedding units; lower n_bits or raise "
            "the reference size"
        )
    permutation = rng.permutation(n_units)
    assignment = np.empty(n_units, dtype=np.int64)
    assignment[permutation] = np.arange(n_units) % n_bits
    return torch.from_numpy(assignment)


def pool_by_bit(values: torch.Tensor, assignment: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Average per-unit values [B, U] into per-bit values [B, n_bits]."""
    sums = torch.zeros(values.shape[0], n_bits, dtype=values.dtype)
    sums.index_add_(1, assignment, values)
    counts = torch.bincount(assignment, minlength=n_bits).to(values.dtype)
    pooled: torch.Tensor = sums / counts
    return pooled


class ReferenceLumaWatermarker(ABC):
    """Template of a luminance watermarker working on a fixed-size reference grid.

    Subclasses implement `_embed_reference` / `_extract_reference` on [B, R, R] float64
    planes. This class handles validation, colour handling and resizing.
    """

    def __init__(self, name: str, n_bits: int, reference_size: int = REFERENCE_SIZE) -> None:
        """Store identity and geometry.

        Args:
            name: Method name reported by the evaluation harness.
            n_bits: Payload size, > 0.
            reference_size: Side R of the square embedding grid.

        Raises:
            ClassicalWatermarkError: Non-positive sizes.
        """
        if n_bits <= 0:
            raise ClassicalWatermarkError(f"n_bits must be > 0, got {n_bits}")
        if reference_size <= 0:
            raise ClassicalWatermarkError(f"reference_size must be > 0, got {reference_size}")
        self.name = name
        self.n_bits = n_bits
        self.reference_size = reference_size

    @abstractmethod
    def _embed_reference(self, plane: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Return the luminance residual [B, R, R] that encodes `bits` [B, n_bits] in `plane`."""

    @abstractmethod
    def _extract_reference(self, plane: torch.Tensor) -> torch.Tensor:
        """Return soft bits [B, n_bits] in [0, 1] read from a [B, R, R] plane."""

    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Hide `messages` [B, n_bits] in `images` [B, 3, H, W]; returns [B, 3, H, W] in [0, 1]."""
        plane = luma(images)
        if messages.ndim != 2 or messages.shape != (images.shape[0], self.n_bits):
            raise ClassicalWatermarkError(
                f"expected messages of shape [{images.shape[0]}, {self.n_bits}], "
                f"got {tuple(messages.shape)}"
            )
        height, width = plane.shape[-2:]
        if min(height, width) < MIN_IMAGE_SIDE:
            raise ClassicalWatermarkError(f"empty image: {height}x{width}")
        reference = resize_plane(plane, (self.reference_size, self.reference_size))
        bits = messages.detach().to("cpu").to(torch.float64)
        ideal = self._embed_reference(reference, bits)
        host = images.detach().to("cpu").to(torch.float64)
        # Closed loop: clipping to [0, 1] and the resampling round trip make the residual
        # that a decoder will see differ from the ideal one; feed the shortfall back so the
        # clipped or attenuated part is made up elsewhere. A no-op when nothing is lost.
        residual = ideal
        for iteration in range(CLOSED_LOOP_ITERATIONS + 1):
            marked = (host + resize_plane(residual, (height, width))[:, None]).clamp(0.0, 1.0)
            if iteration == CLOSED_LOOP_ITERATIONS:
                break
            achieved = resize_plane(luma(marked), (self.reference_size, self.reference_size)) - reference
            residual = residual + (ideal - achieved)
        return marked.to(device=images.device, dtype=torch.float32)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Read soft bits [B, n_bits] in [0, 1] from images of any size [B, 3, H', W']."""
        plane = luma(images)
        reference = resize_plane(plane, (self.reference_size, self.reference_size))
        soft = self._extract_reference(reference)
        result: torch.Tensor = soft.to(device=images.device, dtype=torch.float32)
        return result
