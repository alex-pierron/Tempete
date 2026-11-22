"""Additive spread spectrum: one band-limited pseudo-random pattern per bit.

Embedding (on the 256x256 reference luminance): add `sum_i (2 b_i - 1) * a * p_i` where
`p_i` are unit-variance, key-seeded, band-pass patterns (difference of Gaussians, so that
they live in the frequencies that survive JPEG and mild blur and avoid the energetic
low-frequency content of the host). Decoding correlates the received reference luminance
with each pattern; the correlations are turned into z-scores using extra *null* patterns
(same band, unused by any bit) which measure the host-plus-attack noise in exactly that
band, then squashed into soft bits.

Embedding is informed (improved spread spectrum): most of the host's correlation with each
pattern is subtracted, so that the decoder sees about +/- amplitude whatever the image.

Strength: RMS of the luminance residual, in [0, 1] intensity units (default 0.01, i.e.
40 dB PSNR before clipping); the residual is rescaled to exactly that RMS for every image.
It is shared by the n_bits patterns: each has nominal amplitude `strength / sqrt(n_bits)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as torch_f

from wmkit.classical import (
    DEFAULT_N_BITS,
    EPSILON,
    REFERENCE_SIZE,
    TARGET_PSNR_DB,
    ClassicalWatermarkError,
    ReferenceLumaWatermarker,
    psnr_to_rms_error,
)
from wmkit.types import Watermarker

PATTERN_LOW_SIGMA: float = 1.0
PATTERN_HIGH_SIGMA: float = 4.0
"""Pattern band: Gaussian(1 px) minus Gaussian(4 px), at reference resolution."""
GAUSSIAN_RADIUS_IN_SIGMAS: float = 3.0
MAX_N_BITS: int = 256
"""One full-size pattern is stored per bit (8 B x R^2 each); beyond this the band-limited
patterns are also no longer close to orthogonal."""
NULL_PATTERNS: int = 32
"""Unused patterns whose correlations estimate the noise level in the pattern band."""
HOST_CANCELLATION: float = 1.0
"""Fraction of the host/pattern correlation subtracted at embedding (0 = plain spread
spectrum, 1 = fully cancelled; 0.75 leaves a residual interference of a quarter)."""
SOFT_BIT_GAIN: float = 0.5
"""Slope of the sigmoid applied to the z-score of each correlation."""


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel as float64 [1, 1, 1, K]."""
    radius = max(1, math.ceil(GAUSSIAN_RADIUS_IN_SIGMAS * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).view(1, 1, 1, -1)


def gaussian_blur(planes: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [N, H, W] planes with reflect padding."""
    kernel = _gaussian_kernel(sigma)
    radius = kernel.shape[-1] // 2
    tensor = planes[:, None]
    tensor = torch_f.pad(tensor, (radius, radius, 0, 0), mode="reflect")
    tensor = torch_f.conv2d(tensor, kernel)
    tensor = torch_f.pad(tensor, (0, 0, radius, radius), mode="reflect")
    tensor = torch_f.conv2d(tensor, kernel.transpose(2, 3))
    blurred: torch.Tensor = tensor[:, 0]
    return blurred


def band_pass(planes: torch.Tensor) -> torch.Tensor:
    """Difference of Gaussians of [N, H, W] planes (the pattern band)."""
    return gaussian_blur(planes, PATTERN_LOW_SIGMA) - gaussian_blur(planes, PATTERN_HIGH_SIGMA)


def make_patterns(count: int, size: int, rng: np.random.Generator) -> torch.Tensor:
    """Unit-variance band-pass patterns [count, size, size] (float64)."""
    white = torch.from_numpy(rng.choice([-1.0, 1.0], size=(count, size, size)))
    patterns = band_pass(white)
    patterns = patterns - patterns.mean(dim=(1, 2), keepdim=True)
    return patterns / patterns.std(dim=(1, 2), keepdim=True)


class SpreadSpectrum(ReferenceLumaWatermarker):
    """Spread-spectrum watermarker with correlation decoding."""

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
            strength: RMS of the luminance residual; None targets `TARGET_PSNR_DB`.
            key: Secret seed of the patterns.
            reference_size: Side of the grid on which patterns live and decoding happens.

        Raises:
            ClassicalWatermarkError: Non-positive strength or too many bits.
        """
        super().__init__("spread_spectrum", n_bits, reference_size)
        if n_bits > MAX_N_BITS:
            raise ClassicalWatermarkError(
                f"spread_spectrum supports at most {MAX_N_BITS} bits, got {n_bits}"
            )
        self.strength = psnr_to_rms_error(TARGET_PSNR_DB) if strength is None else float(strength)
        if self.strength <= 0.0:
            raise ClassicalWatermarkError(f"strength must be > 0, got {strength}")
        self.amplitude = self.strength / math.sqrt(n_bits)
        rng = np.random.default_rng(key)
        self._patterns = make_patterns(n_bits, reference_size, rng)
        self._null_patterns = make_patterns(NULL_PATTERNS, reference_size, rng)

    def _embed_reference(self, plane: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        # Informed embedding: part of the host's own correlation with each pattern is
        # subtracted, so that the decoder sees about +/- amplitude whatever the host (a
        # textured image would otherwise flip bits). The residual is then rescaled to the
        # requested RMS, which keeps the PSNR constant across images.
        signs = 2.0 * bits - 1.0
        host_correlation = torch.einsum("bhw,nhw->bn", plane, self._patterns) / (
            plane.shape[-1] * plane.shape[-2]
        )
        coefficients = self.amplitude * signs - HOST_CANCELLATION * host_correlation
        residual = torch.einsum("bn,nhw->bhw", coefficients, self._patterns)
        rms = residual.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + EPSILON
        rescaled: torch.Tensor = residual * (self.strength / rms)
        return rescaled

    def _extract_reference(self, plane: torch.Tensor) -> torch.Tensor:
        # Patterns are unit-variance, so <y, p> / (R * R) is the amplitude estimate.
        pixels = plane.shape[-1] * plane.shape[-2]
        correlations = torch.einsum("bhw,nhw->bn", plane, self._patterns) / pixels
        null = torch.einsum("bhw,nhw->bn", plane, self._null_patterns) / pixels
        noise_scale = null.std(dim=1, keepdim=True) + EPSILON
        return torch.sigmoid(SOFT_BIT_GAIN * correlations / noise_scale)


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
        strength: RMS luminance residual in [0, 1] units; None targets 40 dB PSNR.
        key: Secret seed of the patterns.
        reference_size: Reference grid side.
    """
    del device
    return SpreadSpectrum(DEFAULT_N_BITS if n_bits is None else n_bits, strength, key,
                          reference_size)
