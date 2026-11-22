"""DCT-QIM: quantisation index modulation of mid-frequency 8x8 DCT coefficients of Y.

Embedding (on the 256x256 reference luminance): split into 8x8 blocks, take a few
low/mid-frequency AC coefficients per block, and quantise each onto the lattice of the
bit it carries (dithered by a keyed offset). Every bit is spread, through a keyed
balanced permutation, over many (block, coefficient) units. Decoding measures, for each
unit, how close the coefficient is to the bit-1 lattice versus the bit-0 lattice and
averages these margins per bit.

Strength: the quantisation step in intensity units ([0, 1] scale, DCT orthonormal). The
default is derived from `TARGET_PSNR_DB`: with K modified coefficients out of 64 per
block the MSE is (K / 64) * step^2 / 12.
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

BLOCK_SIZE: int = 8
QIM_COEFFICIENTS: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
"""(row, column) of the DCT coefficients carrying payload: low-to-mid frequencies, which
survive JPEG and mild resampling far better than high frequencies and are less visible
than DC-adjacent ones."""
SOFT_BIT_GAIN: float = 5.0
"""Slope of the sigmoid mapping the mean margin in [-1, 1] to a soft bit."""
COEFFICIENTS_PER_BLOCK: int = len(QIM_COEFFICIENTS)


def default_step() -> float:
    """Quantisation step giving `TARGET_PSNR_DB` before clipping."""
    mse = psnr_to_rms_error(TARGET_PSNR_DB) ** 2
    return math.sqrt(mse * 12.0 * BLOCK_SIZE**2 / COEFFICIENTS_PER_BLOCK)


class DctQim(ReferenceLumaWatermarker):
    """Blockwise DCT quantisation-index-modulation watermarker."""

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
            strength: Quantisation step; None calibrates for `TARGET_PSNR_DB`.
            key: Secret seed of the bit assignment and the dither.
            reference_size: Embedding grid side, a multiple of 8.

        Raises:
            ClassicalWatermarkError: Bad sizes or a non-positive step.
        """
        super().__init__("dct_qim", n_bits, reference_size)
        if reference_size % BLOCK_SIZE:
            raise ClassicalWatermarkError(f"reference_size must be a multiple of {BLOCK_SIZE}")
        self.step = default_step() if strength is None else float(strength)
        if self.step <= 0.0:
            raise ClassicalWatermarkError(f"strength (quantisation step) must be > 0, got {strength}")
        blocks_per_side = reference_size // BLOCK_SIZE
        self._n_blocks = blocks_per_side**2
        rng = np.random.default_rng(key)
        self._assignment = balanced_assignment(self._n_blocks * COEFFICIENTS_PER_BLOCK, n_bits, rng)
        self._dither = torch.from_numpy(
            rng.random((self._n_blocks, COEFFICIENTS_PER_BLOCK))
        ).to(torch.float64) * self.step
        self._rows = torch.tensor([row for row, _ in QIM_COEFFICIENTS])
        self._cols = torch.tensor([col for _, col in QIM_COEFFICIENTS])

    def _coefficients(self, blocks: torch.Tensor) -> torch.Tensor:
        """Select the payload coefficients: [B, nbh, nbw, 8, 8] -> [B, n_blocks, K]."""
        selected = blocks[..., self._rows, self._cols]
        return selected.reshape(blocks.shape[0], self._n_blocks, COEFFICIENTS_PER_BLOCK)

    def _embed_reference(self, plane: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        blocks = block_dct(plane, BLOCK_SIZE)
        coefficients = self._coefficients(blocks)
        unit_bits = bits[:, self._assignment].reshape(
            bits.shape[0], self._n_blocks, COEFFICIENTS_PER_BLOCK
        )
        snapped = qim_embed(coefficients - self._dither, unit_bits, self.step) + self._dither
        change = torch.zeros_like(blocks)
        change[..., self._rows, self._cols] = (snapped - coefficients).reshape(
            blocks.shape[0], *blocks.shape[1:3], COEFFICIENTS_PER_BLOCK
        )
        return block_idct(change)

    def _extract_reference(self, plane: torch.Tensor) -> torch.Tensor:
        coefficients = self._coefficients(block_dct(plane, BLOCK_SIZE))
        margins = qim_margin(coefficients - self._dither, self.step)
        pooled = pool_by_bit(margins.reshape(plane.shape[0], -1), self._assignment, self.n_bits)
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
        strength: Quantisation step in [0, 1] intensity units; None targets 40 dB PSNR.
        key: Secret seed.
        reference_size: Embedding grid side (multiple of 8).
    """
    del device
    return DctQim(DEFAULT_N_BITS if n_bits is None else n_bits, strength, key, reference_size)
