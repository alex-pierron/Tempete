"""Tests for wmkit.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import binom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from wmkit.metrics import (
    DETECTION_NEG_LOG10_P,
    bit_accuracy,
    lpips_distance,
    neg_log10_pvalue,
    psnr,
    ssim,
    tpr_at_fpr,
)


def exact_neg_log10_p(n_bits: int, n_correct: int) -> float:
    """Reference computed with exact big-integer arithmetic."""
    tail = sum(math.comb(n_bits, j) for j in range(n_correct, n_bits + 1))
    return n_bits * math.log10(2.0) - math.log10(tail)


def test_detection_threshold_constant() -> None:
    assert DETECTION_NEG_LOG10_P == 6.0


def test_bit_accuracy_on_soft_and_hard_predictions() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    soft = torch.tensor([[0.9, 0.2, 0.4, 0.1], [0.5, 0.51, 0.99, 0.0]])
    # 0.5 is read as 0 (threshold is strict): row 2 = 0,1,1,0 vs 1,1,1,1 -> 2/4
    assert torch.allclose(bit_accuracy(soft, target), torch.tensor([0.75, 0.5]))
    assert torch.allclose(bit_accuracy(target, target), torch.ones(2))


def test_bit_accuracy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        bit_accuracy(torch.zeros(2, 4), torch.zeros(2, 5))
    with pytest.raises(ValueError):
        bit_accuracy(torch.zeros(4), torch.zeros(4))


@pytest.mark.parametrize("n_bits", [1, 8, 32, 48, 96, 256])
def test_neg_log10_pvalue_matches_scipy_binom_sf(n_bits: int) -> None:
    counts = torch.arange(0, n_bits + 1)
    ours = neg_log10_pvalue(n_bits, counts).numpy().astype(np.float64)
    reference = -np.log10(binom.sf(counts.numpy() - 1, n_bits, 0.5))
    assert np.allclose(ours, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_bits,n_correct", [(1024, 1024), (1024, 1000), (1024, 600), (256, 256)])
def test_neg_log10_pvalue_extremes_are_finite_and_exact(n_bits: int, n_correct: int) -> None:
    score = neg_log10_pvalue(n_bits, torch.tensor([float(n_correct)]))[0].item()
    assert math.isfinite(score)
    assert score == pytest.approx(exact_neg_log10_p(n_bits, n_correct), rel=1e-5)


def test_neg_log10_pvalue_full_payload_1024_is_about_308() -> None:
    score = neg_log10_pvalue(1024, torch.tensor([1024.0]))[0].item()
    assert score == pytest.approx(1024 * math.log10(2.0), rel=1e-6)
    # scipy's own sf agrees where it does not underflow.
    assert score == pytest.approx(-math.log10(binom.sf(1023, 1024, 0.5)), rel=1e-4)


def test_neg_log10_pvalue_stays_finite_beyond_float_underflow() -> None:
    score = neg_log10_pvalue(4096, torch.tensor([4096.0]))[0].item()
    assert score == pytest.approx(4096 * math.log10(2.0), rel=1e-6)


def test_neg_log10_pvalue_is_monotone_and_non_negative() -> None:
    scores = neg_log10_pvalue(64, torch.arange(0, 65))
    assert scores.dtype == torch.float32
    assert (scores >= 0).all()
    assert (scores[1:] >= scores[:-1]).all()
    assert scores[0].item() == 0.0


def test_neg_log10_pvalue_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        neg_log10_pvalue(0, torch.tensor([0.0]))
    with pytest.raises(ValueError):
        neg_log10_pvalue(8, torch.tensor([9.0]))
    with pytest.raises(ValueError):
        neg_log10_pvalue(8, torch.tensor([-1.0]))


def test_psnr_matches_skimage_and_caps_identical_images() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 32, 32, generator=generator)
    y = (x + 0.05 * torch.randn(2, 3, 32, 32, generator=generator)).clamp(0, 1)
    ours = psnr(x, y)
    for index in range(2):
        expected = peak_signal_noise_ratio(x[index].numpy(), y[index].numpy(), data_range=1.0)
        assert ours[index].item() == pytest.approx(expected, rel=1e-4)
    assert psnr(x, x).max().item() == pytest.approx(100.0)


def test_ssim_matches_skimage_and_is_one_for_identical_images(small_batch: torch.Tensor) -> None:
    noisy = (small_batch + 0.05 * torch.randn(small_batch.shape, generator=torch.Generator()
                                              .manual_seed(1))).clamp(0, 1)
    ours = ssim(small_batch, noisy)
    for index in range(small_batch.shape[0]):
        expected = structural_similarity(
            small_batch[index].numpy(), noisy[index].numpy(), channel_axis=0, data_range=1.0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
        assert ours[index].item() == pytest.approx(expected, abs=2e-4)
    assert torch.allclose(ssim(small_batch, small_batch), torch.ones(2), atol=1e-5)


def test_ssim_rejects_tiny_images() -> None:
    with pytest.raises(ValueError):
        ssim(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))


def test_quality_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        psnr(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 17))
    with pytest.raises(ValueError):
        psnr(torch.zeros(3, 16, 16), torch.zeros(3, 16, 16))


def test_tpr_at_fpr_separable_and_overlapping_scores() -> None:
    negatives = np.linspace(0.0, 1.0, 101)  # 101 negatives
    assert tpr_at_fpr(np.array([2.0, 3.0]), negatives, 0.0) == 1.0
    assert tpr_at_fpr(np.array([-1.0, -2.0]), negatives, 0.0) == 0.0
    # FPR 10% allows 10 negatives strictly above the threshold -> threshold = 11th largest = 0.90
    positives = np.array([0.5, 0.85, 0.905, 0.95])
    assert tpr_at_fpr(positives, negatives, 0.1) == 0.5


def test_tpr_at_fpr_never_exceeds_requested_fpr_with_ties() -> None:
    negatives = np.zeros(100)
    positives = np.array([0.0, 1.0])
    # every negative ties at 0: no positive may be called at FPR < 1 unless strictly above
    assert tpr_at_fpr(positives, negatives, 0.05) == 0.5
    assert tpr_at_fpr(positives, negatives, 1.0) == 1.0


def test_tpr_at_fpr_handles_float_products() -> None:
    negatives = np.arange(100, dtype=np.float64)
    # 0.29 * 100 == 28.999999999999996: the tolerance must still allow 29 false positives
    assert tpr_at_fpr(np.array([70.5]), negatives, 0.29) == 1.0


def test_tpr_at_fpr_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        tpr_at_fpr(np.array([1.0]), np.array([0.0]), 1.5)
    with pytest.raises(ValueError):
        tpr_at_fpr(np.array([]), np.array([0.0]), 0.1)
    with pytest.raises(ValueError):
        tpr_at_fpr(np.array([np.nan]), np.array([0.0]), 0.1)


@pytest.mark.slow
def test_lpips_distance_zero_for_identical_and_positive_for_noise(
    small_batch: torch.Tensor,
) -> None:
    noisy = (small_batch + 0.1 * torch.randn(small_batch.shape)).clamp(0, 1)
    assert lpips_distance(small_batch, small_batch).abs().max() < 1e-4
    assert (lpips_distance(small_batch, noisy) > 0).all()
