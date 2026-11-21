"""Unit tests (QA pass) of `wmkit.metrics` and `wmkit.messages`.

The existing `tests/test_metrics.py` already compares against scipy / skimage on natural
images. This module adds independent oracles (exact big-integer arithmetic, brute-force
threshold search, closed forms on constant images) and boundary cases.

Findings pinned here with `xfail(strict=True)` (see the QA report):
    F-METRICS-1  `neg_log10_pvalue` turns a NaN count into 0 instead of raising (arm64).
    F-METRICS-2  `tpr_at_fpr` loses one allowed false positive for large negative sets
                 because `FPR_FLOAT_TOLERANCE` is absolute, not relative.
"""

from __future__ import annotations

import math
import platform

import numpy as np
import pytest
import torch
from scipy.stats import binom
from skimage.metrics import structural_similarity

from wmkit.messages import (
    bits_to_hex,
    hex_length,
    hex_to_bits,
    random_messages,
)
from wmkit.metrics import (
    DETECTION_NEG_LOG10_P,
    FPR_FLOAT_TOLERANCE,
    MIN_MSE,
    SSIM_WINDOW_SIZE,
    bit_accuracy,
    neg_log10_pvalue,
    psnr,
    ssim,
    tpr_at_fpr,
)

RTOL = 1e-6
"""float32 output: 6e-8 relative resolution, so 1e-6 leaves room for the float64 log-space sum."""
ATOL = 1e-6
SCIPY_SF_FLOOR = 1e-300
"""Below this `binom.sf` is no longer a trustworthy reference (denormals)."""
PAYLOAD_SIZES = (1, 2, 3, 7, 32, 48, 64, 96, 255, 1024)
LOG10_2 = math.log10(2.0)
SSIM_K1_PAPER = 0.01
SSIM_K2_PAPER = 0.03
"""Wang et al. 2004 constants, restated here so that a drift in the module is caught."""
ZERO_SCORE_TOLERANCE = 1e-9
"""The log-space sum rounds P(X >= 0) to 1 - 1e-15 (observed: 4e-15 in -log10)."""


def exact_neg_log10_tails(n_bits: int) -> np.ndarray:
    """`-log10 P(X >= k)` for k = 0..n_bits, X ~ Bin(n_bits, 1/2), from exact integers."""
    tail = 0
    tails = [0] * (n_bits + 1)
    for successes in range(n_bits, -1, -1):
        tail += math.comb(n_bits, successes)
        tails[successes] = tail
    return np.array([n_bits * LOG10_2 - math.log10(value) for value in tails], dtype=np.float64)


# ---------------------------------------------------------------------------
# neg_log10_pvalue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_bits", PAYLOAD_SIZES)
def test_neg_log10_pvalue_matches_exact_integer_arithmetic_on_the_full_grid(n_bits: int) -> None:
    counts = torch.arange(0, n_bits + 1)
    ours = neg_log10_pvalue(n_bits, counts).numpy().astype(np.float64)
    np.testing.assert_allclose(ours, exact_neg_log10_tails(n_bits), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("n_bits", PAYLOAD_SIZES)
def test_neg_log10_pvalue_matches_scipy_binom_sf_wherever_scipy_is_reliable(n_bits: int) -> None:
    counts = np.arange(0, n_bits + 1)
    survival = binom.sf(counts - 1, n_bits, 0.5)
    reliable = survival > SCIPY_SF_FLOOR
    ours = neg_log10_pvalue(n_bits, torch.from_numpy(counts)).numpy().astype(np.float64)
    np.testing.assert_allclose(ours[reliable], -np.log10(survival[reliable]), rtol=RTOL, atol=ATOL)
    # The cut must leave a substantial part of the grid for large payloads, not an empty check.
    assert reliable.sum() >= min(n_bits + 1, 600)


def test_neg_log10_pvalue_bounds_at_zero_and_full_payload() -> None:
    for n_bits in PAYLOAD_SIZES:
        scores = neg_log10_pvalue(n_bits, torch.tensor([0.0, float(n_bits)]))
        assert 0.0 <= scores[0].item() < ZERO_SCORE_TOLERANCE, "P(X >= 0) = 1 up to log-space rounding"
        assert scores[1].item() == pytest.approx(n_bits * LOG10_2, rel=RTOL)
    assert math.isfinite(neg_log10_pvalue(1024, torch.tensor([1024.0]))[0].item())


def test_neg_log10_pvalue_single_bit_is_one_coin_flip() -> None:
    scores = neg_log10_pvalue(1, torch.tensor([0.0, 1.0]))
    assert scores[0].item() == 0.0
    assert scores[1].item() == pytest.approx(LOG10_2, rel=RTOL)


@pytest.mark.parametrize("n_bits", (8, 33, 256, 1024))
def test_neg_log10_pvalue_is_non_decreasing_and_strictly_so_above_chance(n_bits: int) -> None:
    scores = neg_log10_pvalue(n_bits, torch.arange(0, n_bits + 1)).numpy().astype(np.float64)
    assert (np.diff(scores) >= 0.0).all()
    above_chance = scores[n_bits // 2 + 1 :]
    assert (np.diff(above_chance) > 0.0).all()
    assert (scores >= 0.0).all() and np.isfinite(scores).all()


def test_neg_log10_pvalue_default_threshold_is_crossed_at_the_expected_count() -> None:
    """For 48 bits the p < 1e-6 gate needs 41 matches (computed with exact integers)."""
    n_bits = 48
    scores = exact_neg_log10_tails(n_bits)
    smallest_detecting = int(np.argmax(scores >= DETECTION_NEG_LOG10_P))
    computed = neg_log10_pvalue(n_bits, torch.arange(0, n_bits + 1))
    assert (computed >= DETECTION_NEG_LOG10_P).nonzero()[0].item() == smallest_detecting


@pytest.mark.parametrize("n_bits", (32, 48, 96, 100, 256, 1024))
def test_neg_log10_pvalue_is_exact_when_fed_float32_bit_accuracy_times_n(n_bits: int) -> None:
    """The real pipeline: bit_accuracy (float32 mean) * n_bits must round back to the count."""
    expected = exact_neg_log10_tails(n_bits)
    for correct in range(n_bits + 1):
        predicted = torch.zeros(1, n_bits)
        predicted[0, :correct] = 1.0
        accuracy = bit_accuracy(predicted, torch.ones(1, n_bits))
        score = neg_log10_pvalue(n_bits, accuracy * n_bits)[0].item()
        assert score == pytest.approx(expected[correct], rel=RTOL, abs=ATOL), correct


def test_neg_log10_pvalue_rounds_to_nearest_integer_count() -> None:
    expected = exact_neg_log10_tails(10)
    got = neg_log10_pvalue(10, torch.tensor([6.0001, 5.9999, 6.4, 5.6])).numpy()
    np.testing.assert_allclose(got[:3], expected[6], rtol=RTOL)
    np.testing.assert_allclose(got[3], expected[6], rtol=RTOL)


def test_neg_log10_pvalue_accepts_integer_and_double_tensors_and_keeps_the_shape() -> None:
    reference = neg_log10_pvalue(16, torch.tensor([3.0, 9.0, 16.0]))
    assert torch.equal(neg_log10_pvalue(16, torch.tensor([3, 9, 16])), reference)
    assert torch.equal(neg_log10_pvalue(16, torch.tensor([3.0, 9.0, 16.0], dtype=torch.float64)), reference)
    assert reference.dtype == torch.float32 and reference.shape == (3,)
    assert neg_log10_pvalue(16, torch.empty(0)).shape == (0,)


def test_neg_log10_pvalue_cache_does_not_leak_between_payload_sizes_or_after_eviction() -> None:
    """`_log_binomial_tail` is an lru_cache(maxsize=8): evicted entries must be recomputed right."""
    probe = torch.arange(0, 6)
    first = neg_log10_pvalue(5, probe).clone()
    for n_bits in range(6, 6 + 12):  # exceeds the cache size, evicting n_bits = 5
        neg_log10_pvalue(n_bits, torch.tensor([float(n_bits)]))
    assert torch.equal(neg_log10_pvalue(5, probe), first)
    assert torch.equal(neg_log10_pvalue(5, probe), first), "second hit must not be mutated"


def test_neg_log10_pvalue_stays_exact_for_a_large_payload() -> None:
    n_bits = 20_000
    score = neg_log10_pvalue(n_bits, torch.tensor([float(n_bits)]))[0].item()
    assert score == pytest.approx(n_bits * LOG10_2, rel=RTOL)


@pytest.mark.parametrize("n_bits", (0, -1, -1024))
def test_neg_log10_pvalue_rejects_non_positive_payload(n_bits: int) -> None:
    with pytest.raises(ValueError, match="n_bits"):
        neg_log10_pvalue(n_bits, torch.tensor([0.0]))


@pytest.mark.parametrize("bad", (-1.0, 9.0, 1e9, -1e9, float("inf"), float("-inf")))
def test_neg_log10_pvalue_rejects_out_of_range_counts(bad: float) -> None:
    with pytest.raises(ValueError, match="n_correct"):
        neg_log10_pvalue(8, torch.tensor([4.0, bad]))


@pytest.mark.xfail(
    strict=True,
    condition=platform.machine() in ("arm64", "aarch64"),
    reason="F-METRICS-1: NaN count -> int64 cast is 0 on arm64, so NaN scores as 'no evidence'",
)
def test_neg_log10_pvalue_rejects_nan_counts() -> None:
    with pytest.raises(ValueError):
        neg_log10_pvalue(8, torch.tensor([float("nan")]))


# ---------------------------------------------------------------------------
# bit_accuracy
# ---------------------------------------------------------------------------


def test_bit_accuracy_complement_and_extremes() -> None:
    generator = torch.Generator().manual_seed(0)
    target = random_messages(5, 40, generator)
    soft = torch.rand(5, 40, generator=generator)
    accuracy = bit_accuracy(soft, target)
    assert torch.allclose(accuracy + bit_accuracy(soft, 1.0 - target), torch.ones(5))
    assert torch.equal(bit_accuracy(target, target), torch.ones(5))
    assert torch.equal(bit_accuracy(1.0 - target, target), torch.zeros(5))
    assert accuracy.dtype == torch.float32 and not accuracy.requires_grad


def test_bit_accuracy_rows_are_independent_and_resolution_is_one_over_n() -> None:
    target = torch.ones(3, 8)
    predicted = torch.ones(3, 8)
    predicted[1, :3] = 0.0
    predicted[2] = 0.0
    assert bit_accuracy(predicted, target).tolist() == [1.0, 5 / 8, 0.0]


def test_bit_accuracy_threshold_is_strict_and_detaches_the_graph() -> None:
    predicted = torch.tensor([[0.5, 0.5000001, 0.4999999]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    assert bit_accuracy(predicted, target).item() == 1.0
    assert not bit_accuracy(predicted, target).requires_grad


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", (0.001, 0.01, 0.1, 0.5, 1.0))
def test_psnr_of_a_constant_offset_is_minus_twenty_log10_offset(offset: float) -> None:
    x = torch.full((1, 3, 16, 16), 0.25, dtype=torch.float64)
    expected = -20.0 * math.log10(offset)
    assert psnr(x, x + offset).item() == pytest.approx(expected, abs=1e-4)


def test_psnr_is_zero_db_when_every_pixel_is_maximally_wrong() -> None:
    assert psnr(torch.zeros(2, 3, 12, 12), torch.ones(2, 3, 12, 12)).tolist() == pytest.approx([0.0, 0.0])


def test_psnr_is_capped_at_one_hundred_db_for_identical_and_nearly_identical_images() -> None:
    cap = -10.0 * math.log10(MIN_MSE)
    assert cap == pytest.approx(100.0)
    x = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    assert psnr(x, x).tolist() == pytest.approx([cap, cap])
    assert psnr(x, x + 1e-7).tolist() == pytest.approx([cap, cap], abs=1e-3)  # mse 1e-14 < floor


def test_psnr_matches_a_numpy_reference_per_batch_element_and_is_symmetric() -> None:
    generator = torch.Generator().manual_seed(2)
    for _ in range(10):
        x = torch.rand(3, 3, 20, 24, generator=generator)
        y = (x + 0.1 * torch.randn(x.shape, generator=generator)).clamp(0, 1)
        reference = [
            -10.0 * math.log10(float(np.mean((x[i].numpy().astype(np.float64) - y[i].numpy()) ** 2)))
            for i in range(3)
        ]
        assert psnr(x, y).tolist() == pytest.approx(reference, abs=1e-3)
        assert torch.allclose(psnr(x, y), psnr(y, x))


def test_psnr_decreases_when_the_noise_grows_and_ignores_the_graph() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(1, 3, 32, 32, generator=generator)
    noise = torch.randn(x.shape, generator=generator)
    values = [psnr(x, (x + sigma * noise).clamp(0, 1)).item() for sigma in (0.01, 0.03, 0.1)]
    assert values == sorted(values, reverse=True)
    assert not psnr(x.clone().requires_grad_(True), x + 0.1).requires_grad


# ---------------------------------------------------------------------------
# ssim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level_x,level_y", [(0.0, 1.0), (0.2, 0.8), (0.5, 0.5), (0.9, 0.1)])
def test_ssim_of_constant_images_has_the_closed_form(level_x: float, level_y: float) -> None:
    """Zero variance and covariance: SSIM = (2ab + c1) / (a^2 + b^2 + c1)."""
    c1 = (SSIM_K1_PAPER * 1.0) ** 2
    x = torch.full((1, 3, 16, 16), level_x)
    y = torch.full((1, 3, 16, 16), level_y)
    expected = (2 * level_x * level_y + c1) / (level_x**2 + level_y**2 + c1)
    assert ssim(x, y).item() == pytest.approx(expected, abs=1e-4)


def test_ssim_is_one_for_identical_random_images_and_never_exceeds_one() -> None:
    generator = torch.Generator().manual_seed(4)
    for _ in range(8):
        x = torch.rand(2, 3, 24, 24, generator=generator)
        assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5)
        y = torch.rand(2, 3, 24, 24, generator=generator)
        assert (ssim(x, y) <= 1.0 + 1e-5).all()


def test_ssim_is_symmetric_bounded_and_low_for_anticorrelated_images() -> None:
    generator = torch.Generator().manual_seed(5)
    for _ in range(8):
        x = torch.rand(2, 3, 32, 32, generator=generator)
        y = torch.rand(2, 3, 32, 32, generator=generator)
        assert torch.allclose(ssim(x, y), ssim(y, x), atol=1e-6)
        assert ((ssim(x, y) >= -1.0 - 1e-5) & (ssim(x, y) <= 1.0 + 1e-5)).all()
    image = torch.rand(1, 3, 32, 32, generator=generator)
    assert ssim(image, 1.0 - image).item() < 0.0


def test_ssim_matches_skimage_on_unstructured_images_too() -> None:
    generator = torch.Generator().manual_seed(6)
    x = torch.rand(2, 3, 40, 28, generator=generator)
    y = (0.5 * x + 0.5 * torch.rand(x.shape, generator=generator)).clamp(0, 1)
    ours = ssim(x, y)
    for index in range(2):
        expected = structural_similarity(  # type: ignore[no-untyped-call]  # skimage ships no stubs
            x[index].numpy(), y[index].numpy(), channel_axis=0, data_range=1.0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
        assert ours[index].item() == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("contrast", (0.002, 0.01, 0.05))
def test_ssim_stabilisation_constants_matter_for_low_contrast_images(contrast: float) -> None:
    """Near-flat images make C1 / C2 dominate: a wrong K1 or K2 shows against skimage at 1e-4."""
    generator = torch.Generator().manual_seed(int(contrast * 1000))
    x = 0.5 + contrast * torch.randn(2, 3, 32, 32, generator=generator)
    y = 0.5 + contrast * torch.randn(2, 3, 32, 32, generator=generator)
    ours = ssim(x, y)
    for index in range(2):
        expected = structural_similarity(  # type: ignore[no-untyped-call]  # skimage ships no stubs
            x[index].numpy(), y[index].numpy(), channel_axis=0, data_range=1.0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
            K1=SSIM_K1_PAPER, K2=SSIM_K2_PAPER,
        )
        assert ours[index].item() == pytest.approx(expected, abs=1e-4)


def test_ssim_batch_elements_are_independent_of_each_other() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.rand(3, 3, 20, 20, generator=generator)
    y = torch.rand(3, 3, 20, 20, generator=generator)
    together = ssim(x, y)
    for index in range(3):
        assert ssim(x[index : index + 1], y[index : index + 1]).item() == pytest.approx(
            together[index].item(), abs=1e-6
        )


def test_ssim_is_the_mean_over_the_three_channels() -> None:
    """Two identical channels (SSIM 1 each) plus one perturbed channel c: (2 + s_c) / 3."""
    generator = torch.Generator().manual_seed(8)
    x = torch.rand(1, 3, 24, 24, generator=generator)
    y = x.clone()
    y[:, 1] = (y[:, 1] + 0.3 * torch.randn(y[:, 1].shape, generator=generator)).clamp(0, 1)
    single_channel = ssim(x[:, 1:2].repeat(1, 3, 1, 1), y[:, 1:2].repeat(1, 3, 1, 1)).item()
    assert ssim(x, y).item() == pytest.approx((2.0 + single_channel) / 3.0, abs=1e-5)


def test_ssim_accepts_exactly_the_window_size_and_rejects_one_less() -> None:
    side = SSIM_WINDOW_SIZE
    assert ssim(torch.zeros(1, 3, side, side), torch.zeros(1, 3, side, side)).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least"):
        ssim(torch.zeros(1, 3, side - 1, side), torch.zeros(1, 3, side - 1, side))
    with pytest.raises(ValueError, match="at least"):
        ssim(torch.zeros(1, 3, side, side - 1), torch.zeros(1, 3, side, side - 1))


def test_ssim_and_psnr_reject_mismatched_or_malformed_batches() -> None:
    good = torch.zeros(1, 3, 16, 16)
    for metric in (psnr, ssim):
        with pytest.raises(ValueError, match="shape mismatch"):
            metric(good, torch.zeros(2, 3, 16, 16))
        with pytest.raises(ValueError, match="B, 3, H, W"):
            metric(torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))


# ---------------------------------------------------------------------------
# tpr_at_fpr
# ---------------------------------------------------------------------------


def brute_force_tpr(positives: np.ndarray, negatives: np.ndarray, fpr: float) -> float:
    """Smallest threshold t among {-inf} + scores with FPR(t) = mean(neg > t) <= fpr."""
    candidates = np.concatenate([[-np.inf], np.unique(negatives)])
    for threshold in candidates:  # ascending: first feasible = smallest threshold
        if float(np.mean(negatives > threshold)) <= fpr + FPR_FLOAT_TOLERANCE:
            return float(np.mean(positives > threshold))
    return 0.0


@pytest.mark.parametrize("seed", range(12))
def test_tpr_at_fpr_matches_a_brute_force_threshold_search_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    negatives = rng.integers(0, 12, size=int(rng.integers(5, 60))).astype(np.float64)
    positives = rng.integers(2, 16, size=int(rng.integers(3, 40))).astype(np.float64)
    for fpr in (0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 1.0):
        assert tpr_at_fpr(positives, negatives, fpr) == pytest.approx(
            brute_force_tpr(positives, negatives, fpr)
        ), (seed, fpr)


@pytest.mark.parametrize("n_negatives", (7, 10, 30, 100, 101, 1000))
def test_tpr_at_fpr_allows_exactly_j_false_positives_for_fpr_j_over_n(n_negatives: int) -> None:
    """Float error in j/n * n must never cost an allowed false positive (up to 1e4 negatives)."""
    negatives = np.arange(n_negatives, dtype=np.float64)
    positives = np.arange(n_negatives, dtype=np.float64) + 0.5
    for allowed in range(n_negatives + 1):
        expected = min(1.0, (allowed + 1) / n_negatives)
        assert tpr_at_fpr(positives, negatives, allowed / n_negatives) == pytest.approx(expected)


@pytest.mark.xfail(
    strict=True,
    reason="F-METRICS-2: FPR_FLOAT_TOLERANCE (1e-12) is absolute; at 1e5 negatives j/n*n errs by 4e-12",
)
def test_tpr_at_fpr_keeps_the_allowed_false_positive_count_for_a_large_negative_set() -> None:
    n_negatives, allowed = 100_000, 25_020
    assert allowed / n_negatives * n_negatives < allowed  # the float product really is short
    negatives = np.arange(n_negatives, dtype=np.float64)
    positives = np.arange(n_negatives, dtype=np.float64) + 0.5
    assert tpr_at_fpr(positives, negatives, allowed / n_negatives) == pytest.approx(
        (allowed + 1) / n_negatives
    )


def test_tpr_at_fpr_is_monotone_in_fpr_and_in_the_positive_scores() -> None:
    rng = np.random.default_rng(0)
    negatives = rng.normal(0.0, 1.0, 500)
    positives = rng.normal(1.5, 1.0, 400)
    rates = [tpr_at_fpr(positives, negatives, fpr) for fpr in np.linspace(0.0, 1.0, 41)]
    assert rates == sorted(rates)
    assert rates[0] <= rates[-1] == 1.0
    shifted = [tpr_at_fpr(positives + 0.7, negatives, fpr) for fpr in np.linspace(0.0, 1.0, 41)]
    assert all(after >= before for before, after in zip(rates, shifted, strict=True))


def test_tpr_at_fpr_is_invariant_under_a_strictly_increasing_score_transform() -> None:
    rng = np.random.default_rng(1)
    negatives = rng.normal(0.0, 1.0, 300)
    positives = rng.normal(1.0, 1.0, 300)
    for fpr in (0.01, 0.05, 0.3):
        base = tpr_at_fpr(positives, negatives, fpr)
        assert tpr_at_fpr(np.exp(positives), np.exp(negatives), fpr) == base
        assert tpr_at_fpr(positives**3, negatives**3, fpr) == base


def test_tpr_at_fpr_realised_fpr_never_exceeds_the_request() -> None:
    rng = np.random.default_rng(2)
    negatives = rng.integers(0, 20, size=250).astype(np.float64)
    for fpr in (0.0, 0.02, 0.1, 0.33, 0.7):
        # A positive set made of every integer in range reveals the chosen threshold.
        positives = np.arange(-1, 21, dtype=np.float64)
        tpr = tpr_at_fpr(positives, negatives, fpr)
        threshold_index = round(tpr * positives.size)  # positives strictly above the threshold
        threshold = positives[positives.size - threshold_index - 1] if threshold_index else positives[-1]
        assert float(np.mean(negatives > threshold)) <= fpr + FPR_FLOAT_TOLERANCE


def test_tpr_at_fpr_handles_infinite_scores_and_accepts_nested_arrays() -> None:
    negatives = np.array([[-np.inf, 0.0], [1.0, 2.0]])
    positives = np.array([[np.inf], [3.0]])
    assert tpr_at_fpr(positives, negatives, 0.0) == 1.0
    assert tpr_at_fpr(np.array([2.0]), negatives, 0.0) == 0.0  # not strictly above the max


@pytest.mark.parametrize("fpr", (-1e-9, 1.0 + 1e-9, float("nan")))
def test_tpr_at_fpr_rejects_out_of_range_fpr(fpr: float) -> None:
    with pytest.raises(ValueError, match="fpr"):
        tpr_at_fpr(np.array([1.0]), np.array([0.0]), fpr)


# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_bits", tuple(range(1, 70)) + (128, 255, 256, 1023, 1024))
def test_hex_round_trip_for_every_payload_size_with_seeded_random_bits(n_bits: int) -> None:
    generator = torch.Generator().manual_seed(n_bits)
    for message in random_messages(4, n_bits, generator):
        text = bits_to_hex(message)
        assert len(text) == hex_length(n_bits) == math.ceil(n_bits / 4)
        assert torch.equal(hex_to_bits(text, n_bits), message)
        assert torch.equal(hex_to_bits("0x" + text.upper(), n_bits), message)
        assert torch.equal(hex_to_bits(f"  {text}\n", n_bits), message)


@pytest.mark.parametrize("n_bits", (4, 8, 32, 48, 256))
def test_hex_of_a_multiple_of_four_is_the_big_endian_integer(n_bits: int) -> None:
    generator = torch.Generator().manual_seed(100 + n_bits)
    message = random_messages(1, n_bits, generator)[0]
    as_integer = int("".join(str(int(bit)) for bit in message.tolist()), 2)
    assert bits_to_hex(message) == format(as_integer, f"0{n_bits // 4}x")


def test_hex_pads_the_last_nibble_with_zero_bits_on_the_right() -> None:
    assert bits_to_hex(torch.tensor([1.0])) == "8"
    assert bits_to_hex(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])) == "f8"
    assert bits_to_hex(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])) == "08"
    assert hex_to_bits("8", 1).tolist() == [1.0]
    assert hex_to_bits("f8", 5).tolist() == [1.0] * 5


@pytest.mark.parametrize(
    "text,n_bits",
    [
        ("1", 1),  # padding bit set
        ("f", 3),  # last padding bit set
        ("f9", 5),
        ("", 4),
        ("fff", 8),
        ("0xfff", 8),
        ("-1", 8),
        ("+f", 4),
        ("1_", 8),
        ("zz", 8),
        ("١٢", 8),  # Arabic-Indic digits are digits for int() but not hex here
        ("0x", 4),
    ],
)
def test_hex_to_bits_rejects_malformed_or_ambiguous_strings(text: str, n_bits: int) -> None:
    with pytest.raises(ValueError):
        hex_to_bits(text, n_bits)


def test_hex_to_bits_rejects_non_positive_payload() -> None:
    with pytest.raises(ValueError):
        hex_to_bits("0", 0)


def test_bits_to_hex_thresholds_soft_values_strictly_above_one_half() -> None:
    assert bits_to_hex(torch.tensor([0.5, 0.51, 0.49, 0.99])) == "5"
    assert bits_to_hex(torch.tensor([0.9, 0.1, 0.9, 0.1]).requires_grad_(True)) == "a"


def test_bits_to_hex_maps_every_nibble_to_its_digit() -> None:
    for value in range(16):
        bits = torch.tensor([float(bit) for bit in format(value, "04b")])
        assert bits_to_hex(bits) == format(value, "x")


def test_random_messages_are_binary_balanced_and_independent_across_rows() -> None:
    generator = torch.Generator().manual_seed(0)
    messages = random_messages(200, 64, generator)
    assert messages.dtype == torch.float32 and messages.shape == (200, 64)
    assert set(torch.unique(messages).tolist()) == {0.0, 1.0}
    assert abs(messages.mean().item() - 0.5) < 0.02  # 12800 draws: sigma = 0.0044
    assert len({tuple(row.tolist()) for row in messages}) == 200  # 2^-64 collision odds


def test_random_messages_stream_continues_with_a_shared_generator() -> None:
    generator = torch.Generator().manual_seed(5)
    first = random_messages(2, 16, generator)
    second = random_messages(2, 16, generator)
    assert not torch.equal(first, second)
    replay = torch.Generator().manual_seed(5)
    assert torch.equal(random_messages(4, 16, replay), torch.cat([first, second]))
