"""Unit tests (QA pass) of the KGW and Gumbel detectors in `wmkit.text`.

Independent oracles, none of which reuses the code under test:
    * KGW p-value: exact rational arithmetic (`fractions.Fraction`) on the binomial tail.
    * Gumbel p-value: closed form of the Gamma(n, 1) tail for integer n,
      `exp(-s) * sum_{k < n} s^k / k!`, evaluated in log space (no `gammaincc`).
    * Dedup and scoring: a plain-Python re-implementation of "first occurrence of each
      (context, token) tuple", built only on the public `green_mask` / `secret_vector`.
The null-hypothesis tests draw a fresh secret key per trial, which is the statement that the
p-value is valid "over the key", and use fixed seeds so that they are deterministic.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest
import torch
from scipy.special import gammaln, logsumexp

from wmkit.text import (
    TEXT_DETECTION_P,
    GumbelConfig,
    GumbelSampler,
    KGWConfig,
    KGWLogitsProcessor,
    WatermarkConfigError,
    green_mask,
    gumbel_detect,
    kgw_detect,
    secret_vector,
)
from wmkit.text.gumbel import MAX_SCORE_TERM, gumbel_pvalue, gumbel_token_scores
from wmkit.text.kgw import (
    TORCH_SEED_MODULUS,
    context_seed,
    effective_gamma,
    green_list_size,
    kgw_green_flags,
    kgw_pvalue,
    scored_positions,
    validate_token_ids,
)

VOCAB = 999
"""Not a multiple of 4: the effective green fraction is 249/999, not 0.25."""
SMALL_ALPHABET = 6
"""At most 36 distinct (context, token) pairs in a 200-token text: heavy repetition."""
H0_TRIALS = 400
TEXT_LENGTH = 200
LOG10 = math.log(10.0)
GAMMAS = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 10), Fraction(3, 7))


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------


def exact_binomial_tail(n_green: int, n_scored: int, gamma: Fraction) -> Fraction:
    """P(X >= n_green), X ~ Bin(n_scored, gamma), in exact rational arithmetic."""
    return sum(
        (
            math.comb(n_scored, j) * gamma**j * (1 - gamma) ** (n_scored - j)
            for j in range(n_green, n_scored + 1)
        ),
        start=Fraction(0),
    )


def log10_of(fraction: Fraction) -> float:
    """log10 of a positive Fraction without overflowing a float on huge numerators."""
    return math.log10(fraction.numerator) - math.log10(fraction.denominator)


def exact_gamma_log10_sf(n_scored: int, total: float) -> float:
    """log10 P(Gamma(n, 1) >= total) for integer n >= 1, as a log-space Poisson sum."""
    if total <= 0.0:
        return 0.0
    ks = np.arange(n_scored)
    log_terms = ks * math.log(total) - gammaln(ks + 1.0)
    return float(min((-total + logsumexp(log_terms)) / LOG10, 0.0))


def reference_scored_positions(ids: Sequence[int], width: int, dedup: bool) -> list[int]:
    """Positions scored: full context and (optionally) first occurrence of (context, token)."""
    seen: set[tuple[int, ...]] = set()
    positions: list[int] = []
    for position in range(width, len(ids)):
        key = tuple(ids[position - width : position + 1])
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        positions.append(position)
    return positions


def reference_kgw(
    ids: Sequence[int], vocab: int, config: KGWConfig, dedup: bool
) -> tuple[int, int, float]:
    """(n_scored, n_green, exact log10 p) by brute force."""
    width = config.context_width
    positions = reference_scored_positions(ids, width, dedup)
    n_green = sum(
        bool(green_mask(list(ids[p - width : p]), vocab, config)[ids[p]]) for p in positions
    )
    gamma = Fraction(green_list_size(vocab, config.gamma), vocab)
    tail = exact_binomial_tail(n_green, len(positions), gamma) if positions else Fraction(1)
    return len(positions), n_green, log10_of(tail)


def reference_gumbel(
    ids: Sequence[int], vocab: int, config: GumbelConfig, dedup: bool
) -> tuple[int, float, float]:
    """(n_scored, total score, log10 p) by brute force."""
    width = config.context_width
    positions = reference_scored_positions(ids, width, dedup)
    terms = [
        -math.log1p(-float(secret_vector(list(ids[p - width : p]), vocab, config)[ids[p]]))
        for p in positions
    ]
    total = math.fsum(terms)
    return len(positions), total, exact_gamma_log10_sf(len(positions), total) if positions else 0.0


def random_ids(rng: np.random.Generator, length: int, alphabet: int = VOCAB - 1) -> list[int]:
    """Uniform i.i.d. token ids in [1, alphabet].

    Token 0 is excluded on purpose: the KGW seed is `hash_key * token`, so a context made of
    token 0 has the same green list under every key and would correlate the null trials.
    """
    return [int(token) for token in rng.integers(1, alphabet + 1, size=length)]


# ---------------------------------------------------------------------------
# KGW: p-value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", GAMMAS)
def test_kgw_pvalue_matches_exact_rational_arithmetic_on_a_grid(gamma: Fraction) -> None:
    for n_scored in (1, 2, 3, 5, 10, 17, 40, 64):
        for n_green in range(n_scored + 1):
            expected = exact_binomial_tail(n_green, n_scored, gamma)
            p_value, log10_p = kgw_pvalue(n_green, n_scored, float(gamma))
            assert p_value == pytest.approx(float(expected), rel=1e-9, abs=0.0), (n_green, n_scored)
            assert log10_p == pytest.approx(log10_of(expected), rel=1e-9, abs=1e-12)


def test_kgw_pvalue_boundaries() -> None:
    assert kgw_pvalue(0, 1, 0.25) == (1.0, 0.0)
    assert kgw_pvalue(0, 500, 0.25) == (1.0, 0.0)
    assert kgw_pvalue(0, 0, 0.25) == (1.0, 0.0)
    assert kgw_pvalue(5, -3, 0.25) == (1.0, 0.0)  # nothing scored: defensive branch
    # All green: p = gamma^n exactly; the log10 stays exact while the float p is still normal
    # (down to ~1e-308, i.e. n <= 500 at gamma = 1/4) and p itself is right in the denormals.
    for n_scored in (1, 10, 100, 300, 500):
        p_value, log10_p = kgw_pvalue(n_scored, n_scored, 0.25)
        assert log10_p == pytest.approx(n_scored * math.log10(0.25), rel=1e-9)
        assert p_value == pytest.approx(0.25**n_scored, rel=1e-9)


def exact_binomial_log10_tail(n_green: int, n_scored: int, gamma: float) -> float:
    """log10 P(X >= n_green) by log-space summation of the pmf (does not use scipy's logsf)."""
    successes = np.arange(n_green, n_scored + 1, dtype=np.float64)
    log_pmf = (
        gammaln(n_scored + 1.0) - gammaln(successes + 1.0) - gammaln(n_scored - successes + 1.0)
        + successes * math.log(gamma) + (n_scored - successes) * math.log1p(-gamma)
    )
    return float(logsumexp(log_pmf) / LOG10)


# F-TEXT-1 fixed: kgw_pvalue now computes the tail in log space, so log10 stays finite.
@pytest.mark.parametrize(
    "n_green,n_scored", [(600, 600), (1000, 1000), (2000, 2000), (1500, 2000), (3000, 5000)]
)
def test_kgw_log10_pvalue_is_finite_and_exact_below_float_underflow(
    n_green: int, n_scored: int
) -> None:
    p_value, log10_p = kgw_pvalue(n_green, n_scored, 0.25)
    assert p_value == 0.0
    assert math.isfinite(log10_p)
    assert log10_p == pytest.approx(exact_binomial_log10_tail(n_green, n_scored, 0.25), rel=1e-6)


def test_kgw_log10_pvalue_oracle_agrees_with_the_implementation_where_scipy_works() -> None:
    """Validates the log-space oracle above on the regime where the implementation is sound."""
    for n_green, n_scored in ((60, 100), (300, 300), (700, 1000), (900, 2000)):
        _, log10_p = kgw_pvalue(n_green, n_scored, 0.25)
        assert math.isfinite(log10_p)
        assert log10_p == pytest.approx(
            exact_binomial_log10_tail(n_green, n_scored, 0.25), rel=1e-9, abs=1e-9
        )


def test_kgw_pvalue_decreases_with_n_green_and_increases_with_gamma() -> None:
    log10_ps = [kgw_pvalue(k, 120, 0.25)[1] for k in range(121)]
    assert all(b < a for a, b in itertools.pairwise(log10_ps))
    by_gamma = [kgw_pvalue(40, 120, g)[1] for g in (0.1, 0.25, 0.5, 0.75)]
    assert by_gamma == sorted(by_gamma)  # a larger null green rate makes 40 greens less surprising


def test_kgw_pvalue_is_a_probability() -> None:
    for n_scored in (1, 30, 300):
        for n_green in range(0, n_scored + 1, max(1, n_scored // 10)):
            p_value, log10_p = kgw_pvalue(n_green, n_scored, 0.3)
            assert 0.0 <= p_value <= 1.0 and log10_p <= 0.0


# ---------------------------------------------------------------------------
# KGW: detection end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("dedup", (True, False))
@pytest.mark.parametrize("seed", range(5))
def test_kgw_detect_matches_a_brute_force_reference(width: int, dedup: bool, seed: int) -> None:
    rng = np.random.default_rng(1000 * width + seed)
    config = KGWConfig(gamma=0.25, hash_key=int(rng.integers(1, 10**6)), context_width=width)
    # A small alphabet forces repeated (context, token) tuples so that dedup really bites.
    ids = random_ids(rng, int(rng.integers(0, 70)), alphabet=12)
    result = kgw_detect(ids, VOCAB, config, dedup_ngrams=dedup)
    n_scored, n_green, log10_p = reference_kgw(ids, VOCAB, config, dedup)
    assert result.n_scored == n_scored
    assert result.n_green == n_green
    assert result.score == float(n_green)
    assert result.log10_p_value == pytest.approx(log10_p, rel=1e-9, abs=1e-12)
    assert result.p_value == pytest.approx(10.0**log10_p, rel=1e-9)
    gamma = green_list_size(VOCAB, 0.25) / VOCAB
    assert effective_gamma(VOCAB, 0.25) == gamma and gamma != 0.25
    if n_scored:
        expected_z = (n_green - gamma * n_scored) / math.sqrt(n_scored * gamma * (1 - gamma))
        assert result.z_score == pytest.approx(expected_z, rel=1e-12, abs=1e-12)
    else:
        assert result.z_score is None and result.p_value == 1.0 and not result.detected
    assert result.neg_log10_p == pytest.approx(-log10_p, abs=1e-12)


def test_kgw_detected_flag_is_p_strictly_below_the_threshold() -> None:
    rng = np.random.default_rng(3)
    config = KGWConfig(hash_key=11)
    for _ in range(60):
        result = kgw_detect(random_ids(rng, 60), VOCAB, config)
        assert result.detected == (result.p_value < TEXT_DETECTION_P)


def test_scored_positions_key_on_the_whole_context_and_the_token() -> None:
    # (1,2) (2,3) (3,2) are three distinct bigrams: a token-only dedup would score two.
    ids = np.array([1, 2, 3, 2])
    assert scored_positions(ids, 1, dedup_ngrams=True).tolist() == [False, True, True, True]
    # [1,2,1,2,1,2]: only the first (1,2) and the first (2,1) are scored.
    cycle = np.array([1, 2, 1, 2, 1, 2])
    assert scored_positions(cycle, 1, dedup_ngrams=True).tolist() == [False, True, True, False, False, False]
    assert scored_positions(cycle, 1, dedup_ngrams=False).tolist() == [False] + [True] * 5
    # Width 2 needs two tokens of context: positions 0 and 1 are never scored.
    assert scored_positions(np.arange(5), 2, dedup_ngrams=False).tolist() == [False, False, True, True, True]
    assert not scored_positions(np.array([7]), 1, dedup_ngrams=True).any()
    assert scored_positions(np.array([], dtype=np.int64), 1, dedup_ngrams=True).size == 0


def test_dedup_counts_the_first_occurrence_only_so_flags_are_nan_afterwards() -> None:
    config = KGWConfig(hash_key=5)
    flags = kgw_green_flags([4, 9, 4, 9, 4, 9, 4], VOCAB, config)
    assert np.isnan(flags[0])
    scored = ~np.isnan(flags)
    assert scored.tolist() == [False, True, True, False, False, False, False]
    assert set(flags[scored].tolist()) <= {0.0, 1.0}
    no_dedup = kgw_green_flags([4, 9, 4, 9, 4, 9, 4], VOCAB, config, dedup_ngrams=False)
    assert (~np.isnan(no_dedup)).sum() == 6
    assert no_dedup[1] == flags[1] and no_dedup[3] == flags[1], "same pair, same colour"


def test_a_text_with_a_single_repeated_token_scores_exactly_one_position() -> None:
    result = kgw_detect([42] * 100, VOCAB, KGWConfig())
    assert result.n_scored == 1
    assert kgw_detect([42] * 100, VOCAB, KGWConfig(), dedup_ngrams=False).n_scored == 99


# ---------------------------------------------------------------------------
# KGW: null hypothesis
# ---------------------------------------------------------------------------


def kgw_null_pvalues(alphabet: int, dedup: bool, seed: int) -> np.ndarray:
    """p-values of key-independent texts, a fresh key per trial."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            kgw_detect(
                random_ids(rng, TEXT_LENGTH, alphabet),
                VOCAB,
                KGWConfig(gamma=0.25, hash_key=int(rng.integers(1, 2**31))),
                dedup_ngrams=dedup,
            ).p_value
            for _ in range(H0_TRIALS)
        ]
    )


def test_kgw_fpr_on_uniform_random_text_is_at_most_nominal_for_several_levels() -> None:
    p_values = kgw_null_pvalues(VOCAB - 1, True, seed=11)
    # Discrete null: P(p <= u) <= u. Allow 3 binomial sd (sd = 1.1 % at 5 %, 2.0 % at 20 %).
    assert np.mean(p_values <= 0.05) <= 0.05 + 0.034
    assert np.mean(p_values <= 0.20) <= 0.20 + 0.06
    assert np.mean(p_values < TEXT_DETECTION_P) <= 0.005
    # ... and not trivially conservative: the test must be able to reject.
    assert np.mean(p_values <= 0.20) >= 0.10


def test_kgw_fpr_on_highly_repetitive_text_stays_nominal_only_with_dedup() -> None:
    """The documented reason for the dedup (Three Bricks): repeats inflate a plain z/binomial test."""
    with_dedup = kgw_null_pvalues(SMALL_ALPHABET, True, seed=12)
    without_dedup = kgw_null_pvalues(SMALL_ALPHABET, False, seed=12)
    assert np.mean(with_dedup <= 0.05) <= 0.05 + 0.034
    assert np.mean(with_dedup < TEXT_DETECTION_P) <= 0.01
    # Without dedup the effective sample is much smaller than n_scored: size explodes
    # (measured at seed 12: 26 % at 5 %, 11 % at 0.1 %).
    assert np.mean(without_dedup <= 0.05) >= 0.15
    assert np.mean(without_dedup < TEXT_DETECTION_P) >= 0.03


# ---------------------------------------------------------------------------
# KGW: seeding, green lists, validation
# ---------------------------------------------------------------------------


def test_context_seed_single_token_is_hash_key_times_token() -> None:
    for key, token in ((15485863, 0), (15485863, 1), (15485863, 49151), (1, 7), (0, 123)):
        assert context_seed([token], key) == key * token
    assert context_seed([5, 7], 3) == (3 * (7 * 1_000_003 + 5)) % TORCH_SEED_MODULUS


def test_context_seed_is_order_sensitive_and_within_torch_range() -> None:
    assert context_seed([1, 2], 11) != context_seed([2, 1], 11)
    huge = context_seed([10**9, 10**9], 2**62)
    assert 0 <= huge < TORCH_SEED_MODULUS
    seeds = {context_seed([a, b], 15485863) for a in range(40) for b in range(40)}
    assert len(seeds) == 1600, "no collision among small bigram contexts"


@pytest.mark.parametrize("vocab_size", (2, 3, 7, 100, 999, 4096))
@pytest.mark.parametrize("gamma", (0.01, 0.25, 0.5, 0.99))
def test_green_mask_has_exactly_int_vocab_gamma_greens(vocab_size: int, gamma: float) -> None:
    mask = green_mask([3], vocab_size, KGWConfig(gamma=gamma, hash_key=9))
    assert mask.dtype == torch.bool and mask.shape == (vocab_size,)
    assert int(mask.sum()) == int(vocab_size * gamma) == green_list_size(vocab_size, gamma)


def test_green_masks_of_different_contexts_overlap_about_gamma_squared() -> None:
    config = KGWConfig(gamma=0.25, hash_key=77)
    overlaps = [
        float((green_mask([a], 2000, config) & green_mask([a + 1], 2000, config)).sum()) / 2000
        for a in range(60)
    ]
    assert abs(float(np.mean(overlaps)) - 0.0625) < 0.01


def test_logits_processor_adds_delta_exactly_on_the_green_tokens() -> None:
    config = KGWConfig(gamma=0.25, delta=1.5, hash_key=31)
    processor = KGWLogitsProcessor(64, config)
    scores = torch.randn(3, 64, generator=torch.Generator().manual_seed(0))
    input_ids = torch.tensor([[1, 5], [2, 9], [3, 63]])
    out = processor(input_ids, scores)  # type: ignore[arg-type]
    for row, last in enumerate((5, 9, 63)):
        mask = green_mask([last], 64, config)
        assert torch.allclose(out[row] - scores[row], mask.float() * 1.5)
    assert not torch.equal(out, scores)


@pytest.mark.parametrize(
    "build,message",
    [
        (lambda: KGWConfig(hash_key=-1), "hash_key"),
        (lambda: KGWConfig(gamma=float("nan")), "gamma"),
        (lambda: KGWConfig(context_width=-3), "context_width"),
        (lambda: KGWLogitsProcessor(1, KGWConfig()), "vocab_size"),
        (lambda: validate_token_ids([[1, 2], [3, 4]], 10), "1-D"),  # type: ignore[list-item]  # deliberately 2-D
        (lambda: validate_token_ids([10], 10), "lie in"),
    ],
)
def test_kgw_invalid_arguments_raise_the_typed_error(build: object, message: str) -> None:
    with pytest.raises(WatermarkConfigError, match=message):
        build()  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Gumbel: p-value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_scored", (1, 2, 5, 20, 100, 500))
def test_gumbel_pvalue_matches_the_poisson_sum_closed_form(n_scored: int) -> None:
    for factor in (0.1, 0.5, 0.9, 1.0, 1.2, 1.7, 2.5, 4.0):
        total = factor * n_scored
        expected_log10 = exact_gamma_log10_sf(n_scored, total)
        p_value, log10_p = gumbel_pvalue(total, n_scored)
        # Exact branch: 1e-8. Below ~1e-323 the implementation switches to a two-term
        # asymptotic whose relative error is about 3e-7 at n = 500 (see the underflow test).
        relative = 1e-8 if expected_log10 > -320.0 else 1e-5
        assert log10_p == pytest.approx(expected_log10, rel=relative, abs=1e-10), (n_scored, factor)
        if expected_log10 > -300.0:
            assert p_value == pytest.approx(10.0**expected_log10, rel=1e-8)


@pytest.mark.parametrize(
    "n_scored,total", [(200, 1500.0), (50, 1200.0), (5, 900.0), (2000, 4500.0), (20_000, 26_000.0)]
)
def test_gumbel_log_pvalue_is_accurate_in_the_underflow_regime(n_scored: int, total: float) -> None:
    """Where `gammaincc` returns 0.0 the asymptotic branch must still track the exact tail."""
    from scipy.special import gammaincc

    assert float(gammaincc(n_scored, total)) == 0.0, "scenario must really be underflowed"
    p_value, log10_p = gumbel_pvalue(total, n_scored)
    assert p_value == 0.0
    assert log10_p == pytest.approx(exact_gamma_log10_sf(n_scored, total), rel=1e-5)


def test_gumbel_log_pvalue_has_no_jump_where_the_float_underflows() -> None:
    """Sweep the total across the exact/asymptotic seam: monotone and within 2e-4 of exact."""
    from scipy.special import gammaincc

    n_scored = 200
    totals = np.arange(1200.0, 1400.0, 4.0)
    underflowed = np.array([float(gammaincc(n_scored, t)) == 0.0 for t in totals])
    assert underflowed.any() and not underflowed.all(), "the sweep must cross the seam"
    log10_ps = np.array([gumbel_pvalue(float(t), n_scored)[1] for t in totals])
    exact = np.array([exact_gamma_log10_sf(n_scored, float(t)) for t in totals])
    assert (np.diff(log10_ps) < 0.0).all()
    # Measured: 0 error in the exact branch, a constant ~7e-5 in the asymptotic one.
    assert np.abs(log10_ps - exact).max() < 2e-4


def test_gumbel_pvalue_boundaries() -> None:
    assert gumbel_pvalue(0.0, 0) == (1.0, 0.0)
    assert gumbel_pvalue(12.0, 0) == (1.0, 0.0)
    assert gumbel_pvalue(0.0, 10) == (1.0, 0.0)
    for n_scored in (1, 10, 100):
        p_value, log10_p = gumbel_pvalue(n_scored * 0.01, n_scored)
        assert 0.99 < p_value <= 1.0 and log10_p <= 0.0
    assert gumbel_pvalue(5.0, 1)[0] == pytest.approx(math.exp(-5.0), rel=1e-12)  # Gamma(1) is Exp(1)


def test_gumbel_pvalue_is_monotone_in_the_score() -> None:
    log10_ps = [gumbel_pvalue(total, 80)[1] for total in np.linspace(0.0, 400.0, 81)]
    assert all(b <= a for a, b in itertools.pairwise(log10_ps))


# ---------------------------------------------------------------------------
# Gumbel: detection end to end and null hypothesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("dedup", (True, False))
@pytest.mark.parametrize("seed", range(5))
def test_gumbel_detect_matches_a_brute_force_reference(width: int, dedup: bool, seed: int) -> None:
    rng = np.random.default_rng(5000 + 100 * width + seed)
    config = GumbelConfig(hash_key=int(rng.integers(1, 10**6)), context_width=width)
    ids = random_ids(rng, int(rng.integers(0, 70)), alphabet=12)
    result = gumbel_detect(ids, VOCAB, config, dedup_ngrams=dedup)
    n_scored, total, log10_p = reference_gumbel(ids, VOCAB, config, dedup)
    assert result.n_scored == n_scored and result.n_green is None
    assert result.score == pytest.approx(total, rel=1e-12, abs=1e-12)
    assert result.log10_p_value == pytest.approx(log10_p, rel=1e-8, abs=1e-10)
    if n_scored:
        assert result.z_score == pytest.approx((total - n_scored) / math.sqrt(n_scored), abs=1e-9)
    else:
        assert result.z_score is None and result.p_value == 1.0 and not result.detected
    assert result.detected == (result.p_value < TEXT_DETECTION_P)


def test_gumbel_token_scores_are_non_negative_capped_and_nan_when_unscored() -> None:
    config = GumbelConfig(hash_key=3)
    scores = gumbel_token_scores([5, 6, 5, 6, 5, 6], VOCAB, config)
    assert np.isnan(scores[0]) and np.isnan(scores[3:]).all()
    observed = scores[~np.isnan(scores)]
    assert observed.size == 2 and (observed >= 0.0).all() and (observed <= MAX_SCORE_TERM).all()
    rng = np.random.default_rng(0)
    many = gumbel_token_scores(random_ids(rng, 3000), VOCAB, config)
    many = many[~np.isnan(many)]
    assert many.max() <= MAX_SCORE_TERM
    assert abs(many.mean() - 1.0) < 5.0 / math.sqrt(many.size), "terms are Exp(1) under H0"


def gumbel_null_pvalues(alphabet: int, dedup: bool, seed: int) -> np.ndarray:
    """p-values of key-independent texts, a fresh key per trial."""
    rng = np.random.default_rng(seed)
    return np.array(
        [
            gumbel_detect(
                random_ids(rng, TEXT_LENGTH, alphabet),
                VOCAB,
                GumbelConfig(hash_key=int(rng.integers(1, 2**31))),
                dedup_ngrams=dedup,
            ).p_value
            for _ in range(H0_TRIALS)
        ]
    )


def test_gumbel_pvalues_are_uniform_under_h0_over_random_keys() -> None:
    p_values = gumbel_null_pvalues(VOCAB - 1, True, seed=21)
    assert 0.02 <= np.mean(p_values <= 0.05) <= 0.085  # sd 1.1 %
    assert 0.14 <= np.mean(p_values <= 0.20) <= 0.26  # sd 2.0 %
    assert 0.43 <= np.mean(p_values <= 0.50) <= 0.57  # sd 2.5 %
    assert np.mean(p_values < TEXT_DETECTION_P) <= 0.005


def test_gumbel_fpr_on_repetitive_text_needs_the_dedup() -> None:
    with_dedup = gumbel_null_pvalues(SMALL_ALPHABET, True, seed=22)
    without_dedup = gumbel_null_pvalues(SMALL_ALPHABET, False, seed=22)
    assert 0.02 <= np.mean(with_dedup <= 0.05) <= 0.085
    assert np.mean(with_dedup < TEXT_DETECTION_P) <= 0.01
    assert np.mean(without_dedup <= 0.05) >= 0.15
    assert np.mean(without_dedup < TEXT_DETECTION_P) >= 0.03


# ---------------------------------------------------------------------------
# Gumbel: sampler and configuration
# ---------------------------------------------------------------------------


def test_gumbel_select_never_picks_a_zero_probability_token_and_honours_the_width() -> None:
    probabilities = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
    sampler = GumbelSampler(5, GumbelConfig(hash_key=8, context_width=2))
    picks = {sampler.select([c, c + 1, c + 2], probabilities) for c in range(300)}
    assert picks == {1, 3}
    # Only the last `context_width` tokens matter.
    assert sampler.select([9, 1, 2], probabilities) == sampler.select([4, 4, 1, 2], probabilities)


def test_gumbel_secret_vector_depends_on_every_context_token_and_is_in_range() -> None:
    config = GumbelConfig(hash_key=4, context_width=2)
    base = secret_vector([1, 2], 500, config)
    assert not np.array_equal(base, secret_vector([2, 1], 500, config))
    assert not np.array_equal(base, secret_vector([1, 3], 500, config))
    assert base.shape == (500,) and (base >= 0.0).all() and (base < 1.0).all()


@pytest.mark.parametrize(
    "build,message",
    [
        (lambda: GumbelSampler(1, GumbelConfig()), "vocab_size"),
        (lambda: GumbelConfig(hash_key=-5), "hash_key"),
        (lambda: GumbelConfig(context_width=0), "context_width"),
    ],
)
def test_gumbel_invalid_arguments_raise_the_typed_error(build: object, message: str) -> None:
    with pytest.raises(WatermarkConfigError, match=message):
        build()  # type: ignore[operator]
