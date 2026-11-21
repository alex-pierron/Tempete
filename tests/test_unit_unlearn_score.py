"""Unit tests (QA pass) of `wmkit.unlearn.score`: the NeurIPS 2023 unlearning score.

`tests/test_unlearn.py` pins a few hand values. This module adds: an exact-rational reference
for the forgetting quality on random epsilon vectors, structural properties (permutation
invariance, monotonicity, bounds), bin boundaries at the float level, and edge cases
(negative / infinite / huge epsilon, zero reference accuracy).

Finding pinned with `xfail(strict=True)`:
    F-SCORE-1  `epsilon_bin(-inf)` and `epsilon_bin(1e308)` raise a bare OverflowError
               (`math.floor(inf)`), instead of clamping to bin 0 / returning a huge bin; the
               public docstring promises clamping of negative estimates.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from wmkit.unlearn.score import (
    EPSILON_BIN_WIDTH,
    FORGET_QUALITY_NUMERATOR,
    ScoreError,
    epsilon_bin,
    estimate_epsilon,
    forget_quality_from_epsilons,
    neurips2023_score,
)

RANDOM_TRIALS = 40
MAX_BIN_BEFORE_UNDERFLOW = 1000
"""2 / 2**1000 is still a normal float; the term only reaches 0.0 near bin 1075."""
HALF = Fraction(1, 2)


def reference_forget_quality(eps: list[float]) -> Fraction:
    """Exact mean of 2 / 2**floor(eps / 0.5) with negative eps clamped to bin 0 (no infinities)."""
    terms = [
        Fraction(2) / (2 ** max(0, math.floor(Fraction(value) / HALF))) for value in eps
    ]
    return sum(terms, start=Fraction(0)) / len(terms)


# ---------------------------------------------------------------------------
# epsilon_bin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bin_index", range(12))
def test_epsilon_bin_boundaries_are_inclusive_below_and_exclusive_above(bin_index: int) -> None:
    edge = bin_index * EPSILON_BIN_WIDTH
    assert epsilon_bin(edge) == bin_index
    assert epsilon_bin(edge + 0.25 * EPSILON_BIN_WIDTH) == bin_index
    if bin_index > 0:
        assert epsilon_bin(math.nextafter(edge, 0.0)) == bin_index - 1


def test_epsilon_bin_clamps_negative_estimates_to_zero() -> None:
    for value in (-1e-300, -0.25, -0.5, -3.0, -1e9):
        assert epsilon_bin(value) == 0


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_epsilon_bin_rejects_nan_and_plus_infinity_with_the_typed_error(value: float) -> None:
    with pytest.raises(ScoreError):
        epsilon_bin(value)


# F-SCORE-1 (-inf) fixed: epsilon_bin now clamps any non-positive epsilon to bin 0.
def test_epsilon_bin_clamps_minus_infinity_to_bin_zero() -> None:
    assert epsilon_bin(-math.inf) == 0


@pytest.mark.xfail(
    strict=True,
    raises=OverflowError,
    reason="F-SCORE-1: eps / 0.5 overflows to inf for eps > 8.9e307, then math.floor raises",
)
def test_epsilon_bin_of_a_gigantic_finite_epsilon_does_not_overflow() -> None:
    assert epsilon_bin(1e308) > 0


# ---------------------------------------------------------------------------
# forget_quality_from_epsilons
# ---------------------------------------------------------------------------


def test_forget_quality_of_a_single_epsilon_is_two_over_two_to_the_bin() -> None:
    for bin_index in range(0, MAX_BIN_BEFORE_UNDERFLOW, 37):
        value = bin_index * EPSILON_BIN_WIDTH + 0.1
        assert forget_quality_from_epsilons([value]) == math.ldexp(FORGET_QUALITY_NUMERATOR, -bin_index)
        assert forget_quality_from_epsilons([value]) == FORGET_QUALITY_NUMERATOR / 2**bin_index


def test_forget_quality_underflows_cleanly_to_zero_for_enormous_epsilon() -> None:
    assert forget_quality_from_epsilons([0.5 * 1100]) == 0.0
    assert forget_quality_from_epsilons([1e6]) == 0.0
    assert forget_quality_from_epsilons([0.5 * MAX_BIN_BEFORE_UNDERFLOW]) > 0.0


@pytest.mark.parametrize("seed", range(RANDOM_TRIALS))
def test_forget_quality_matches_exact_rational_arithmetic_on_random_vectors(seed: int) -> None:
    rng = random.Random(seed)
    size = rng.randint(1, 60)
    eps = [rng.choice([rng.uniform(-1.0, 6.0), rng.randint(0, 12) * 0.5]) for _ in range(size)]
    expected = reference_forget_quality(eps)
    got = forget_quality_from_epsilons(eps)
    assert got == pytest.approx(float(expected), rel=1e-14, abs=0.0)


def test_forget_quality_ignores_order_and_is_stable_under_repetition() -> None:
    rng = random.Random(0)
    eps = [rng.uniform(0.0, 5.0) for _ in range(50)]
    base = forget_quality_from_epsilons(eps)
    shuffled = eps[:]
    rng.shuffle(shuffled)
    assert forget_quality_from_epsilons(shuffled) == pytest.approx(base, rel=1e-15)
    assert forget_quality_from_epsilons(eps * 3) == pytest.approx(base, rel=1e-15)


def test_forget_quality_is_a_weighted_average_of_its_parts() -> None:
    left, right = [0.0, 0.7, 1.2], [2.4, 3.3]
    combined = forget_quality_from_epsilons(left + right)
    weighted = (
        len(left) * forget_quality_from_epsilons(left) + len(right) * forget_quality_from_epsilons(right)
    ) / (len(left) + len(right))
    assert combined == pytest.approx(weighted, rel=1e-15)


def test_forget_quality_never_increases_when_an_epsilon_grows() -> None:
    rng = random.Random(1)
    for _ in range(RANDOM_TRIALS):
        eps = [rng.uniform(0.0, 4.0) for _ in range(8)]
        index = rng.randrange(8)
        larger = eps[:]
        larger[index] += rng.uniform(0.0, 3.0)
        assert forget_quality_from_epsilons(larger) <= forget_quality_from_epsilons(eps) + 1e-15


def test_forget_quality_is_bounded_in_zero_two_for_finite_inputs() -> None:
    rng = random.Random(2)
    for _ in range(RANDOM_TRIALS):
        eps = [rng.uniform(-5.0, 20.0) for _ in range(rng.randint(1, 30))]
        assert 0.0 < forget_quality_from_epsilons(eps) <= 2.0
    assert forget_quality_from_epsilons([-1.0, -2.0]) == 2.0  # clamped: no credit above 2


def test_forget_quality_infinite_epsilon_contributes_exactly_zero() -> None:
    assert forget_quality_from_epsilons([math.inf, math.inf]) == 0.0
    assert forget_quality_from_epsilons([0.0, math.inf, math.inf, math.inf]) == 0.5
    assert forget_quality_from_epsilons([0.0, 1.0, math.inf]) == pytest.approx((2.0 + 0.5) / 3.0)


def test_forget_quality_agrees_with_the_per_example_bin() -> None:
    rng = random.Random(3)
    for _ in range(RANDOM_TRIALS):
        value = rng.uniform(-1.0, 10.0)
        assert forget_quality_from_epsilons([value]) == 2.0 / 2 ** epsilon_bin(value)


def test_forget_quality_sum_is_exact_where_naive_summation_would_not_be() -> None:
    """fsum: 10**5 tiny terms plus one large one lose nothing to rounding."""
    eps = [0.0] + [20.0] * 100_000  # 2.0 + 100000 * 2**-39
    expected = float((Fraction(2) + 100_000 * Fraction(2, 2**40)) / 100_001)
    assert forget_quality_from_epsilons(eps) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("bad", ([], [float("nan")], [0.5, float("nan"), 1.0]))
def test_forget_quality_rejects_empty_and_nan_inputs(bad: list[float]) -> None:
    with pytest.raises(ScoreError):
        forget_quality_from_epsilons(bad)


# ---------------------------------------------------------------------------
# neurips2023_score
# ---------------------------------------------------------------------------


def test_score_hand_values() -> None:
    # F * (RA_u / RA_r) * (TA_u / TA_r)
    assert neurips2023_score(2.0, 0.9, 0.9, 0.8, 0.8) == pytest.approx(2.0)
    assert neurips2023_score(1.0, 0.45, 0.9, 0.4, 0.8) == pytest.approx(0.25)
    assert neurips2023_score(0.75, 0.6, 0.8, 0.9, 0.6) == pytest.approx(0.75 * 0.75 * 1.5)
    assert neurips2023_score(0.9375, 0.0, 0.9, 0.8, 0.8) == 0.0


def test_score_equals_forget_quality_when_utilities_match_the_reference() -> None:
    for accuracy_pair in ((0.9, 0.8), (1.0, 1.0), (0.123, 0.456)):
        ra, ta = accuracy_pair
        assert neurips2023_score(0.6, ra, ra, ta, ta) == pytest.approx(0.6)


def test_score_is_linear_in_forget_quality_and_invariant_to_the_accuracy_scale() -> None:
    base = neurips2023_score(1.0, 0.7, 0.9, 0.6, 0.8)
    assert neurips2023_score(0.4, 0.7, 0.9, 0.6, 0.8) == pytest.approx(0.4 * base)
    # Percent instead of fraction on all four accuracies: the ratios do not change.
    assert neurips2023_score(1.0, 70.0, 90.0, 60.0, 80.0) == pytest.approx(base)


def test_score_treats_retain_and_test_utility_symmetrically() -> None:
    swapped = neurips2023_score(1.3, 0.5, 0.9, 0.7, 0.8)
    assert neurips2023_score(1.3, 0.7, 0.8, 0.5, 0.9) == pytest.approx(swapped)


def test_score_is_not_capped_when_the_unlearned_model_beats_the_reference() -> None:
    """Documented formula (eq. 2) has no clipping of the utility ratios at 1."""
    assert neurips2023_score(1.0, 0.99, 0.9, 0.88, 0.8) == pytest.approx(1.1 * 1.1)


def test_score_end_to_end_with_estimated_epsilons() -> None:
    eps = [estimate_epsilon(fpr, fnr) for fpr, fnr in ((0.5, 0.5), (0.25, 0.25), (0.0, 0.3))]
    quality = forget_quality_from_epsilons(eps)
    # eps = 0 -> 2.0 ; eps = ln 3 = 1.0986 -> bin 2 -> 0.5 ; eps = inf -> 0.
    assert quality == pytest.approx((2.0 + 0.5 + 0.0) / 3.0)
    assert neurips2023_score(quality, 0.8, 1.0, 0.5, 1.0) == pytest.approx(quality * 0.4)


@pytest.mark.parametrize(
    "arguments,name",
    [
        ((-0.01, 0.9, 0.9, 0.8, 0.8), "forget_quality"),
        ((1.0, -0.01, 0.9, 0.8, 0.8), "ra_u"),
        ((1.0, 0.9, -0.01, 0.8, 0.8), "ra_r"),
        ((1.0, 0.9, 0.9, -0.01, 0.8), "ta_u"),
        ((1.0, 0.9, 0.9, 0.8, -0.01), "ta_r"),
        ((float("nan"), 0.9, 0.9, 0.8, 0.8), "forget_quality"),
        ((1.0, 0.9, float("nan"), 0.8, 0.8), "ra_r"),
        ((1.0, 0.9, 0.9, 0.8, float("nan")), "ta_r"),
    ],
)
def test_score_names_the_offending_argument(arguments: tuple[float, ...], name: str) -> None:
    with pytest.raises(ScoreError, match=name):
        neurips2023_score(*arguments)


@pytest.mark.parametrize("arguments", [(1.0, 0.9, 0.0, 0.8, 0.8), (1.0, 0.9, 0.9, 0.8, 0.0)])
def test_score_rejects_a_zero_reference_accuracy_instead_of_dividing_by_zero(
    arguments: tuple[float, ...]
) -> None:
    with pytest.raises(ScoreError, match="strictly positive"):
        neurips2023_score(*arguments)


def test_score_accepts_a_zero_unlearned_accuracy_and_returns_zero() -> None:
    assert neurips2023_score(1.5, 0.0, 0.9, 0.0, 0.8) == 0.0


# ---------------------------------------------------------------------------
# estimate_epsilon
# ---------------------------------------------------------------------------


def reference_epsilon(fpr: float, fnr: float, delta: float) -> float:
    """eps_hat = max(0, log((1-d-FPR)/FNR), log((1-d-FNR)/FPR)) with the limit conventions."""
    candidates = [0.0]
    for numerator, denominator in ((1.0 - delta - fpr, fnr), (1.0 - delta - fnr, fpr)):
        if numerator > 0.0:
            candidates.append(math.inf if denominator == 0.0 else math.log(numerator / denominator))
    return max(candidates)


def test_epsilon_estimate_matches_the_formula_on_a_grid_including_the_corners() -> None:
    rates = [0.0, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.9, 1.0]
    for delta in (0.0, 0.05, 0.3):
        for fpr in rates:
            for fnr in rates:
                assert estimate_epsilon(fpr, fnr, delta) == pytest.approx(
                    reference_epsilon(fpr, fnr, delta), rel=1e-12, abs=1e-12
                ), (fpr, fnr, delta)


def test_epsilon_estimate_is_symmetric_in_fpr_and_fnr_and_non_negative() -> None:
    rng = random.Random(4)
    for _ in range(RANDOM_TRIALS):
        fpr, fnr, delta = rng.random(), rng.random(), rng.uniform(0.0, 0.5)
        value = estimate_epsilon(fpr, fnr, delta)
        assert value == pytest.approx(estimate_epsilon(fnr, fpr, delta), rel=1e-12, abs=1e-12)
        assert value >= 0.0


def test_epsilon_is_zero_on_the_random_guessing_diagonal() -> None:
    for fpr in (0.0, 0.1, 0.3, 0.5, 0.9, 1.0):
        assert estimate_epsilon(fpr, 1.0 - fpr) == pytest.approx(0.0, abs=1e-12)


def test_epsilon_grows_as_the_attack_gets_better() -> None:
    values = [estimate_epsilon(rate, rate) for rate in (0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001)]
    assert values == sorted(values) and values[0] == 0.0 and values[-1] > values[0]


def test_a_perfect_attack_is_infinitely_distinguishable_unless_delta_forgives_it() -> None:
    assert estimate_epsilon(0.0, 0.0) == math.inf
    assert estimate_epsilon(0.0, 0.7) == math.inf
    assert estimate_epsilon(0.8, 0.0) == math.inf
    assert estimate_epsilon(0.0, 0.0, delta=0.9) == math.inf  # numerator 0.1 > 0 over FNR = 0
    # FPR = 0 but the other numerator 1 - d - FNR = -0.05 is negative: that term is -inf, not +inf,
    # and the remaining one is log(0.9 / 0.95) < 0, clamped to 0.
    assert estimate_epsilon(0.0, 0.95, delta=0.1) == 0.0


def test_epsilon_estimate_rejects_out_of_range_rates_and_delta() -> None:
    for fpr, fnr, delta in (
        (-0.1, 0.2, 0.0),
        (0.2, 1.1, 0.0),
        (float("nan"), 0.2, 0.0),
        (0.2, 0.2, -0.1),
        (0.2, 0.2, 1.0),
        (0.2, 0.2, float("nan")),
    ):
        with pytest.raises(ScoreError):
            estimate_epsilon(fpr, fnr, delta)


def test_the_chain_estimate_then_quality_is_monotone_in_attack_strength() -> None:
    """A better membership attack can only lower the forgetting quality."""
    qualities = [
        forget_quality_from_epsilons([estimate_epsilon(rate, rate)]) for rate in (0.5, 0.4, 0.25, 0.1, 0.0)
    ]
    assert qualities == sorted(qualities, reverse=True)
    assert qualities[0] == 2.0 and qualities[-1] == 0.0
