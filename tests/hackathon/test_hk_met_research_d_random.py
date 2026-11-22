"""Team D d8: symbol counts, chi-square statistic and permutation test on synthetic replies."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.research.d_random import (
    COIN_SYMBOLS,
    DIGIT_SYMBOLS,
    RandomTestError,
    compare_endpoints,
    d8_verdict,
    homogeneity_statistic,
    run_d8,
    symbol_counts,
)
from hackathon.model_equality.store import Answer


def test_symbol_counts_for_digits_and_coin() -> None:
    assert symbol_counts("3\n3\n7 and 10", DIGIT_SYMBOLS).tolist() == [1, 1, 0, 2, 0, 0, 0, 1, 0, 0]
    assert symbol_counts("H\nT\nH Heads", COIN_SYMBOLS).tolist() == [2, 1]


def test_homogeneity_statistic_is_zero_for_equal_distributions_and_positive_otherwise() -> None:
    same = homogeneity_statistic(np.array([5.0, 5.0]), np.array([10.0, 10.0]))
    assert same == pytest.approx(0.0)
    assert homogeneity_statistic(np.array([10.0, 0.0]), np.array([0.0, 10.0])) > 15.0
    assert homogeneity_statistic(np.zeros(2), np.array([1.0, 1.0])) == 0.0


def test_permutation_test_detects_a_biased_endpoint_and_not_an_identical_one() -> None:
    rng = np.random.default_rng(1)
    fair = ["\n".join(rng.choice(list("HT"), 30)) for _ in range(30)]
    fair_again = ["\n".join(rng.choice(list("HT"), 30)) for _ in range(30)]
    biased = ["\n".join(rng.choice(list("HT"), 30, p=[0.8, 0.2])) for _ in range(30)]
    assert compare_endpoints(fair, fair_again, COIN_SYMBOLS, permutations=300).p_value > 0.05
    assert compare_endpoints(fair, biased, COIN_SYMBOLS, permutations=300).p_value < 0.01
    with pytest.raises(RandomTestError):
        compare_endpoints(fair[:2], biased, COIN_SYMBOLS)


def test_verdict_requires_fp8_separated_and_every_disguise_not() -> None:
    rng = np.random.default_rng(2)

    def replies(p_h: float) -> list[str]:
        return ["\n".join(rng.choice(list("HT"), 30, p=[p_h, 1 - p_h])) for _ in range(30)]

    def answers(biased_endpoints: set[str]) -> list[Answer]:
        rows = []
        for e in ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08"):
            for i, text in enumerate(replies(0.85 if e in biased_endpoints else 0.5)):
                rows.append(Answer(e, "d-rand-001", i, "p", text))
        return rows

    separated = run_d8(answers({"ref-02"}), {"d-rand-001": COIN_SYMBOLS})
    assert d8_verdict(separated) == "fp8 SEPARE"
    confounded = run_d8(answers({"ref-02", "ref-06"}), {"d-rand-001": COIN_SYMBOLS})
    assert d8_verdict(confounded) == "fp8 NON separe"
