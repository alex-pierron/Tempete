"""Watermark evaluation metrics: bit accuracy, detection p-value, image quality, TPR@FPR."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from scipy.special import gammaln

DETECTION_NEG_LOG10_P: float = 6.0
"""Default detection threshold on `-log10 p` (p < 1e-6), as in the Pixel Seal paper."""

BIT_THRESHOLD: float = 0.5
"""Soft bits strictly above this value are read as 1."""

RGB_CHANNELS: int = 3
MAX_PIXEL_VALUE: float = 1.0
MIN_MSE: float = 1e-10
"""Floor on the mean squared error; caps PSNR at 100 dB instead of returning inf."""

SSIM_WINDOW_SIZE: int = 11
SSIM_GAUSSIAN_SIGMA: float = 1.5
SSIM_K1: float = 0.01
SSIM_K2: float = 0.03

FPR_FLOAT_TOLERANCE: float = 1e-12
"""Absorbs float error in `fpr * n_negatives` (e.g. 0.1 * 30 = 3.0000000000000004)."""

LPIPS_NETWORK: str = "alex"
LPIPS_INPUT_SCALE: float = 2.0
"""LPIPS expects images in [-1, 1]: x * 2 - 1."""

_LPIPS_MODELS: dict[str, Any] = {}


def _check_batch_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    """Validate two image batches [B, 3, H, W] of identical shape."""
    if x.ndim != 4 or x.shape[1] != RGB_CHANNELS:
        raise ValueError(f"expected images of shape [B, 3, H, W], got {tuple(x.shape)}")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")


def bit_accuracy(soft_or_hard: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of bits recovered correctly, after thresholding predictions at 0.5.

    Args:
        soft_or_hard: Predicted bits [B, n], soft in [0, 1] or hard in {0, 1}.
        target: Ground-truth bits [B, n] in {0, 1}.

    Returns:
        float32 tensor [B] in [0, 1].

    Raises:
        ValueError: Shapes differ or are not 2-D.
    """
    if soft_or_hard.ndim != 2 or soft_or_hard.shape != target.shape:
        raise ValueError(
            "expected two [B, n] tensors of equal shape, got "
            f"{tuple(soft_or_hard.shape)} and {tuple(target.shape)}"
        )
    predicted = soft_or_hard.detach() > BIT_THRESHOLD
    expected = target.detach().to(predicted.device) > BIT_THRESHOLD
    return (predicted == expected).to(torch.float32).mean(dim=1)


@lru_cache(maxsize=8)
def _log_binomial_tail(n_bits: int) -> np.ndarray:
    """log P(X >= k) for k = 0..n_bits, X ~ Binomial(n_bits, 1/2), computed in log space.

    scipy's `binom.logsf` is `log(sf)` for discrete laws, which underflows for
    n_bits > 1074; accumulating log-pmf terms with `logaddexp` stays finite for any n.
    """
    successes = np.arange(n_bits + 1, dtype=np.float64)
    log_pmf = (
        gammaln(n_bits + 1.0) - gammaln(successes + 1.0) - gammaln(n_bits - successes + 1.0)
        - n_bits * math.log(2.0)
    )
    tail: np.ndarray = np.logaddexp.accumulate(log_pmf[::-1])[::-1].copy()
    return tail


def neg_log10_pvalue(n_bits: int, n_correct: torch.Tensor) -> torch.Tensor:
    """Detection score `-log10 P(X >= n_correct)` with X ~ Binomial(n_bits, 0.5).

    This is the one-sided p-value of observing at least `n_correct` matching bits under
    the null hypothesis that the image carries no watermark (equation 2 of the
    Pixel Seal paper). It is finite even for n_correct == n_bits (about 308.2 at 1024).

    Args:
        n_bits: Payload size, > 0.
        n_correct: Number of matching bits per image, tensor [B] (rounded to integers).

    Returns:
        float32 tensor [B], >= 0.

    Raises:
        ValueError: `n_bits <= 0` or a count outside [0, n_bits].
    """
    if n_bits <= 0:
        raise ValueError(f"n_bits must be > 0, got {n_bits}")
    counts = torch.round(n_correct.detach().to("cpu").to(torch.float64)).to(torch.int64)
    if counts.numel() > 0 and (counts.min() < 0 or counts.max() > n_bits):
        raise ValueError(f"n_correct must lie in [0, {n_bits}], got {n_correct.tolist()}")
    log_tail = _log_binomial_tail(n_bits)
    scores = -log_tail[counts.numpy()] / math.log(10.0)
    return torch.from_numpy(np.maximum(scores, 0.0)).to(torch.float32)


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB for images in [0, 1], per batch element.

    Identical images give 100 dB (MSE floored at 1e-10) instead of infinity so that
    averages stay finite.

    Args:
        x: Images [B, 3, H, W].
        y: Images [B, 3, H, W], same shape as `x`.

    Returns:
        float32 tensor [B].
    """
    _check_batch_pair(x, y)
    mse = ((x.detach() - y.detach()) ** 2).flatten(start_dim=1).mean(dim=1)
    mse = mse.clamp_min(MIN_MSE)
    return (10.0 * torch.log10(MAX_PIXEL_VALUE**2 / mse)).to(torch.float32)


def _gaussian_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Separable-by-outer-product 2-D Gaussian window [1, 1, w, w], sums to one."""
    offsets = torch.arange(SSIM_WINDOW_SIZE, device=device, dtype=dtype) - SSIM_WINDOW_SIZE // 2
    kernel_1d = torch.exp(-(offsets**2) / (2.0 * SSIM_GAUSSIAN_SIGMA**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Structural similarity (Wang et al. 2004), Gaussian 11x11 window, mean over channels.

    Matches `skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False)` on the valid region.

    Args:
        x: Images [B, 3, H, W] in [0, 1].
        y: Images [B, 3, H, W], same shape.

    Returns:
        float32 tensor [B] in [-1, 1] (1 = identical).

    Raises:
        ValueError: Images smaller than the 11x11 window.
    """
    _check_batch_pair(x, y)
    if min(x.shape[-2:]) < SSIM_WINDOW_SIZE:
        raise ValueError(f"images must be at least {SSIM_WINDOW_SIZE}x{SSIM_WINDOW_SIZE} for SSIM")
    batch, channels, height, width = x.shape
    window = _gaussian_window(x.device, torch.float32)
    # Fold channels into the batch so one grouped-free conv filters every plane.
    xs = x.detach().to(torch.float32).reshape(batch * channels, 1, height, width)
    ys = y.detach().to(torch.float32).reshape(batch * channels, 1, height, width)
    mu_x = torch_f.conv2d(xs, window)
    mu_y = torch_f.conv2d(ys, window)
    var_x = torch_f.conv2d(xs * xs, window) - mu_x**2
    var_y = torch_f.conv2d(ys * ys, window) - mu_y**2
    covariance = torch_f.conv2d(xs * ys, window) - mu_x * mu_y
    c1 = (SSIM_K1 * MAX_PIXEL_VALUE) ** 2
    c2 = (SSIM_K2 * MAX_PIXEL_VALUE) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    per_plane = (numerator / denominator).flatten(start_dim=1).mean(dim=1)
    return per_plane.reshape(batch, channels).mean(dim=1)


def _get_lpips_model(device: torch.device) -> Any:
    """Load (once per device) the LPIPS network `alex`; the `lpips` package is lazy."""
    key = str(device)
    if key not in _LPIPS_MODELS:
        import lpips  # imported lazily: it pulls torchvision weights on first use

        model = lpips.LPIPS(net=LPIPS_NETWORK, verbose=False).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


def lpips_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Learned perceptual distance (LPIPS, AlexNet), lower is more similar.

    Args:
        x: Images [B, 3, H, W] in [0, 1].
        y: Images [B, 3, H, W], same shape.

    Returns:
        float32 tensor [B] on the device of `x`.
    """
    _check_batch_pair(x, y)
    model = _get_lpips_model(x.device)
    with torch.no_grad():
        distance = model(
            x.detach().to(torch.float32) * LPIPS_INPUT_SCALE - 1.0,
            y.detach().to(torch.float32) * LPIPS_INPUT_SCALE - 1.0,
        )
    flat: torch.Tensor = distance.reshape(-1).to(torch.float32)
    return flat


def tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> float:
    """True-positive rate at a fixed false-positive rate.

    The threshold is the smallest one whose empirical FPR (strictly greater than the
    threshold counts as a positive call) does not exceed `fpr`, so the achieved FPR
    never exceeds the requested one even with tied scores.

    Args:
        scores_pos: Detection scores of watermarked images (higher = more watermarked).
        scores_neg: Detection scores of clean images.
        fpr: Target false-positive rate in [0, 1].

    Returns:
        TPR in [0, 1].

    Raises:
        ValueError: Empty or NaN-containing inputs, or `fpr` outside [0, 1].
    """
    if not 0.0 <= fpr <= 1.0:
        raise ValueError(f"fpr must lie in [0, 1], got {fpr}")
    positives = np.asarray(scores_pos, dtype=np.float64).ravel()
    negatives = np.asarray(scores_neg, dtype=np.float64).ravel()
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("scores_pos and scores_neg must both be non-empty")
    if np.isnan(positives).any() or np.isnan(negatives).any():
        raise ValueError("scores must not contain NaN")
    allowed_false_positives = math.floor(fpr * negatives.size + FPR_FLOAT_TOLERANCE)
    if allowed_false_positives >= negatives.size:
        return 1.0
    descending = np.sort(negatives)[::-1]
    threshold = descending[allowed_false_positives]
    return float(np.mean(positives > threshold))
