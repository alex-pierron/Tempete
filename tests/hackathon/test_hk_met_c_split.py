"""Team C: public split shapes and the fixed-mask acceptance rate (no network)."""

from __future__ import annotations

import numpy as np

from hackathon.model_equality.research.c_pairs import Partition, pair_index
from hackathon.model_equality.research.c_split import (
    SPLIT_ENDPOINTS_ANY,
    SPLIT_ENDPOINTS_BOTH,
    SPLIT_PAIRS,
    acceptance_rate,
    masked_score,
    sample_mask,
)

SIZE = 33


def test_split_shapes_have_the_expected_public_sizes() -> None:
    index = pair_index(SIZE)
    rng = np.random.default_rng(1)
    pairs = np.mean([sample_mask(SPLIT_PAIRS, index, rng).sum() for _ in range(200)])
    both = {int(sample_mask(SPLIT_ENDPOINTS_BOTH, index, rng).sum()) for _ in range(50)}
    either = {int(sample_mask(SPLIT_ENDPOINTS_ANY, index, rng).sum()) for _ in range(50)}
    assert abs(pairs - 0.3 * index.count) < 8
    assert both == {45}  # 10 public endpoints, C(10, 2) pairs
    assert either == {index.count - 253}  # pairs with at least one of 10 endpoints: 528 - C(23, 2)


def test_a_perfect_submission_scores_one_on_any_mask_with_both_classes() -> None:
    index = pair_index(SIZE)
    truth = Partition.from_groups([list(range(12)), list(range(12, 20)), [20], list(range(21, 33))],
                                  [0, 0, 0, 1])
    rng = np.random.default_rng(2)
    mask = sample_mask(SPLIT_ENDPOINTS_ANY, index, rng)
    assert masked_score(truth.states(index), truth.states(index), mask) == 1.0


def test_acceptance_rate_is_one_when_the_observation_is_always_true() -> None:
    index = pair_index(SIZE)
    alone = Partition.from_groups([[i] for i in range(SIZE)]).states(index)
    rate = acceptance_rate("balanced", SPLIT_PAIRS, [list(range(23)), list(range(23, 33))],
                           frozenset(), {"zero": (alone, 0.0)}, np.random.default_rng(3), 40)
    assert rate == 1.0
