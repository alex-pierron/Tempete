"""DWT-DCT-SVD watermarking (invisible-watermark style) on the luminance.

Embedding (on the 256x256 reference luminance): one-level Haar DWT; the LL band is cut
into 4x4 blocks (8x8 reference pixels), each block is DCT-transformed and decomposed by
SVD; the largest singular value is quantised (QIM) onto the lattice of the bit assigned
to the block. Decoding repeats the transform and measures the distance of that singular
value to the two lattices, averaging per bit.

Differences from the `invisible-watermark` package: the lattice is the minimum-distortion
QIM one (error uniform in +/- step/2, instead of the one-sided dither of the original),
the block-to-bit assignment is a keyed balanced permutation, and bits are soft.

Strength: the quantisation step on the largest singular value (its units are those of the
orthonormal block DCT of the Haar LL band, i.e. about 8x the mean luminance of a block).
Changing that value by d changes the 64 reference pixels of the block by energy d^2, so
MSE = step^2 / (12 * 64) and the default step targets `TARGET_PSNR_DB`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from wmkit.classical import (
    DEFAULT_N_BITS,
    REFERENCE_SIZE,
    TARGET_PSNR_DB,
    ClassicalWatermarkError,
    ReferenceLumaWatermarker,
    balanced_assignment,
    block_dct,
    block_idct,
    pool_by_bit,
    psnr_to_rms_error,
    qim_embed,
    qim_margin,
)
from wmkit.types import Watermarker

BLOCK_SIZE: int = 4
DWT_SUBSAMPLING: int = 2
"""Haar level 1 halves each side; one LL coefficient covers a 2x2 pixel square."""
PIXELS_PER_BLOCK: int = (BLOCK_SIZE * DWT_SUBSAMPLING) ** 2
SOFT_BIT_GAIN: float = 5.0
"""Slope of the sigmoid mapping the mean margin in [-1, 1] to a soft bit."""
HAAR_LL_NORMALISATION: float = 2.0
"""Orthonormal 2-D Haar: LL = (a + b + c + d) / 2."""


def default_step() -> float:
    """Quantisation step on the top singular value giving `TARGET_PSNR_DB` before clipping."""
    mse = psnr_to_rms_error(TARGET_PSNR_DB) ** 2
    return math.sqrt(mse * 12.0 * PIXELS_PER_BLOCK)


def haar_ll(plane: torch.Tensor) -> torch.Tensor:
    """Level-1 orthonormal Haar approximation band of [B, H, W] (H, W even)."""
    a = plane[:, 0::2, 0::2]
    b = plane[:, 0::2, 1::2]
    c = plane[:, 1::2, 0::2]
    d = plane[:, 1::2, 1::2]
    band: torch.Tensor = (a + b + c + d) / HAAR_LL_NORMALISATION
    return band


def inverse_haar_ll(band: torch.Tensor) -> torch.Tensor:
    """Pixel-domain image of an LL-only band (all detail bands zero): 2x2 replication / 2."""
    spread = band.repeat_interleave(DWT_SUBSAMPLING, dim=1).repeat_interleave(
        DWT_SUBSAMPLING, dim=2
    )
    restored: torch.Tensor = spread / HAAR_LL_NORMALISATION
    return restored


class DwtDctSvd(ReferenceLumaWatermarker):
    """DWT-DCT-SVD watermarker with soft decoding."""

    def __init__(
        self,
        n_bits: int = DEFAULT_N_BITS,
        strength: float | None = None,
        key: int = 0,
        reference_size: int = REFERENCE_SIZE,
    ) -> None:
        """Create the watermarker.

        Args:
            n_bits: Payload size.
            strength: Quantisation step on the largest singular value; None targets
                `TARGET_PSNR_DB`.
            key: Secret seed of the block-to-bit assignment and the dither.
            reference_size: Embedding grid side, a multiple of 8.

        Raises:
            ClassicalWatermarkError: Bad sizes or a non-positive step.
        """
        super().__init__("dwt_dct_svd", n_bits, reference_size)
        if reference_size % (BLOCK_SIZE * DWT_SUBSAMPLING):
            raise ClassicalWatermarkError(
                f"reference_size must be a multiple of {BLOCK_SIZE * DWT_SUBSAMPLING}"
            )
        self.step = default_step() if strength is None else float(strength)
        if self.step <= 0.0:
            raise ClassicalWatermarkError(f"strength (quantisation step) must be > 0, got {strength}")
        self._n_blocks = (reference_size // (BLOCK_SIZE * DWT_SUBSAMPLING)) ** 2
        rng = np.random.default_rng(key)
        self._assignment = balanced_assignment(self._n_blocks, n_bits, rng)
        self._dither = torch.from_numpy(rng.random(self._n_blocks)).to(torch.float64) * self.step

    def _top_singular(self, plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top singular triplet of every LL block: values [B, nb], left/right vectors."""
        blocks = block_dct(haar_ll(plane), BLOCK_SIZE)
        left, singular, right_h = torch.linalg.svd(blocks)
        batch = plane.shape[0]
        values = singular[..., 0].reshape(batch, self._n_blocks)
        return values, left[..., :, 0], right_h[..., 0, :]

    def _embed_reference(self, plane: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        values, left, right = self._top_singular(plane)
        unit_bits = bits[:, self._assignment]
        snapped = qim_embed(values - self._dither, unit_bits, self.step) + self._dither
        change = (snapped - values).reshape(left.shape[:3])
        # Only the top singular value moves, so the DCT-domain change is rank one.
        block_change = change[..., None, None] * left[..., :, None] * right[..., None, :]
        return inverse_haar_ll(block_idct(block_change))

    def _extract_reference(self, plane: torch.Tensor) -> torch.Tensor:
        values, _, _ = self._top_singular(plane)
        margins = qim_margin(values - self._dither, self.step)
        pooled = pool_by_bit(margins, self._assignment, self.n_bits)
        return torch.sigmoid(SOFT_BIT_GAIN * pooled)


def build(
    n_bits: int | None = None,
    device: str = "cpu",
    strength: float | None = None,
    key: int = 0,
    reference_size: int = REFERENCE_SIZE,
) -> Watermarker:
    """Factory used by the method registry.

    Args:
        n_bits: Payload size; None gives 32.
        device: Accepted for interface uniformity; computation always runs on the CPU and
            results are returned on the device of the input images.
        strength: Quantisation step of the top singular value; None targets 40 dB PSNR.
        key: Secret seed.
        reference_size: Reference grid side (multiple of 8).
    """
    del device
    return DwtDctSvd(DEFAULT_N_BITS if n_bits is None else n_bits, strength, key, reference_size)
