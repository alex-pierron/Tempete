"""Pairwise MMD engine: equality with a brute-force reference, null calibration, power."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.features import (
    answer_key,
    build_views,
    first_word,
    ngram_matrix,
    normalize_text,
)
from hackathon.model_equality.pairwise import (
    KERNELS,
    brute_force_mmd,
    gram_matrix,
    pair_statistic,
)

IntVector = np.ndarray


def random_design(
    rng: np.random.Generator, n_prompts: int, per_side: int, shift: float
) -> tuple[list[str], IntVector, IntVector, IntVector]:
    """Texts of two endpoints over shared prompts; `shift` moves endpoint b's word distribution."""
    texts: list[str] = []
    prompts: list[int] = []
    sides: list[int] = []
    for prompt in range(n_prompts):
        base = rng.dirichlet(np.ones(6))
        other = (1 - shift) * base + shift * rng.dirichlet(np.ones(6))
        for side, distribution in ((0, base), (1, other)):
            for _ in range(per_side):
                words = rng.choice(6, size=8, p=distribution)
                texts.append(" ".join(f"w{w}" for w in words))
                prompts.append(prompt)
                sides.append(side)
    order = np.argsort(prompts, kind="stable")
    return ([texts[i] for i in order], np.array(prompts)[order], np.array(sides)[order],
            np.arange(len(texts)))


@pytest.mark.parametrize("kernel", KERNELS)
def test_fast_statistic_equals_brute_force(kernel: str) -> None:
    rng = np.random.default_rng(1)
    texts, prompts, sides, _ = random_design(rng, 4, 3, 0.5)
    views = build_views("freeform", texts)
    gram = gram_matrix(kernel, views, prompts.astype(np.int64))
    rows_a = np.flatnonzero(sides == 0).astype(np.int64)
    rows_b = np.flatnonzero(sides == 1).astype(np.int64)
    stat = pair_statistic(gram, prompts.astype(np.int64), rows_a, rows_b, 20,
                          np.random.default_rng(0))
    assert stat is not None
    assert stat.mmd == pytest.approx(brute_force_mmd(gram, prompts.astype(np.int64), rows_a, rows_b))


def test_hamming_kernel_matches_the_official_definition() -> None:
    # official: k(y, y') = sum_i 1{y_i = y'_i} / L on padded sequences (two paddings match)
    views = build_views("freeform", ["a b c", "a b d", "a b c"])
    gram = gram_matrix("hamming_word", views, np.zeros(3, dtype=np.int64))
    length = views.words.shape[1]
    assert gram[0, 2] == pytest.approx(1.0)
    assert gram[0, 1] == pytest.approx((length - 1) / length)  # same except one word
    assert gram[0, 0] == 0.0  # diagonal removed (i != j in the unbiased estimator)


def test_z_scores_are_near_standard_normal_under_the_null() -> None:
    rng = np.random.default_rng(2)
    zs: list[float] = []
    for _ in range(40):
        texts, prompts, sides, _ = random_design(rng, 6, 4, 0.0)
        views = build_views("freeform", texts)
        p = prompts.astype(np.int64)
        gram = gram_matrix("spectrum", views, p)
        stat = pair_statistic(gram, p, np.flatnonzero(sides == 0).astype(np.int64),
                              np.flatnonzero(sides == 1).astype(np.int64), 100,
                              np.random.default_rng(3))
        assert stat is not None
        zs.append(stat.z)
    assert abs(float(np.mean(zs))) < 0.6
    assert 0.6 < float(np.std(zs)) < 1.5


def test_a_real_shift_gives_a_large_z_and_a_small_p_value() -> None:
    rng = np.random.default_rng(4)
    texts, prompts, sides, _ = random_design(rng, 8, 5, 1.0)
    views = build_views("freeform", texts)
    p = prompts.astype(np.int64)
    gram = gram_matrix("spectrum", views, p)
    stat = pair_statistic(gram, p, np.flatnonzero(sides == 0).astype(np.int64),
                          np.flatnonzero(sides == 1).astype(np.int64), 199,
                          np.random.default_rng(5))
    assert stat is not None and stat.z > 4 and stat.p_value <= 0.01 and stat.mmd > 0


def test_statistic_is_undefined_without_within_endpoint_pairs() -> None:
    # one answer per prompt and endpoint: no same-prompt pair inside an endpoint
    texts = ["a", "b", "c", "d"]
    prompts = np.array([0, 0, 1, 1], dtype=np.int64)
    gram = gram_matrix("onehot_key", build_views("freeform", texts), prompts)
    stat = pair_statistic(gram, prompts, np.array([0, 2]), np.array([1, 3]), 10,
                          np.random.default_rng(0))
    assert stat is None


def test_permutations_are_reproducible_with_a_fixed_seed() -> None:
    rng = np.random.default_rng(6)
    texts, prompts, sides, _ = random_design(rng, 5, 3, 0.4)
    views = build_views("freeform", texts)
    p = prompts.astype(np.int64)
    gram = gram_matrix("hamming_char", views, p)
    a, b = np.flatnonzero(sides == 0).astype(np.int64), np.flatnonzero(sides == 1).astype(np.int64)
    first = pair_statistic(gram, p, a, b, 50, np.random.default_rng(9))
    second = pair_statistic(gram, p, a, b, 50, np.random.default_rng(9))
    assert first == second


def test_feature_extraction_rules() -> None:
    assert answer_key("arith", "The result is 1,234,567.") == "1234567"
    assert answer_key("arith", "no digits") == ""
    assert answer_key("format", "  Yes, it is.") == "yes"
    assert answer_key("freeform", "Hello   World\n") == "hello world"
    assert first_word("Arr, matey!") == "arr" and first_word("") == ""
    assert normalize_text(" a \n b\t c ") == "a b c"


def test_ngram_matrix_rows_are_unit_norm_and_shift_invariant() -> None:
    matrix = ngram_matrix(["the cat sat on the mat", "arr matey the cat sat on the mat", ""])
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)
    similarity = float((matrix[0] @ matrix[1].T).toarray()[0, 0])
    assert similarity > 0.7  # a prefix barely moves a bag of n-grams
    assert float((matrix[0] @ matrix[2].T).toarray()[0, 0]) == 0.0
