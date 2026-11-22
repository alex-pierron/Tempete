"""Team C: organiser-like truths, ABC acceptance and the ref power helpers (no network)."""

from __future__ import annotations

import numpy as np

from hackathon.model_equality.research.c_abc import abc_posterior, balanced_truth, organiser_truth
from hackathon.model_equality.research.c_pairs import Partition, pair_index
from hackathon.model_equality.research.c_ref_power import auc, style_scores

SIZE = 33
GROUPS = [list(range(23)), list(range(23, 33))]


def test_organiser_truth_covers_every_endpoint_with_nested_labels() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        truth = organiser_truth(rng, GROUPS, frozenset({15, 19}), 0.5, [(1, 2, 3), (1, 2)])
        assert truth.size == SIZE
        # each creator group keeps its endpoints in families of its own: no family mixes groups
        for family in truth.family_groups():
            assert all(m < 23 for m in family) or all(m >= 23 for m in family)
        assert 2 <= len(truth.family_groups()) <= 5


def test_balanced_truth_has_several_models_per_family_and_no_dominant_one() -> None:
    rng = np.random.default_rng(8)
    for _ in range(40):
        truth = balanced_truth(rng, GROUPS, [(1, 2, 3), (1, 2)])
        assert truth.size == SIZE
        for family in truth.family_groups():
            assert all(m < 23 for m in family) or all(m >= 23 for m in family)
        assert len(truth.model_groups()) >= len(truth.family_groups())


def test_balanced_prior_is_selectable_in_abc() -> None:
    rng = np.random.default_rng(9)
    index = pair_index(SIZE)
    all_alone = Partition.from_groups([[i] for i in range(SIZE)]).states(index)
    result = abc_posterior(GROUPS, frozenset(), {"zero": (all_alone, 0.0)}, {"zero": all_alone},
                           rng, draws=30, prior="balanced")
    assert result.accepted == 30


def test_a_probe_that_always_scores_zero_accepts_every_truth() -> None:
    rng = np.random.default_rng(4)
    index = pair_index(SIZE)
    all_alone = Partition.from_groups([[i] for i in range(SIZE)]).states(index)
    result = abc_posterior(GROUPS, frozenset(), {"zero": (all_alone, 0.0)}, {"zero": all_alone},
                           rng, draws=50)
    assert result.accepted == 50
    assert result.acceptance_rate == 1.0
    assert float(result.mean_full[0]) == 0.0


def test_an_impossible_observation_accepts_nothing() -> None:
    rng = np.random.default_rng(5)
    index = pair_index(SIZE)
    one_model = Partition.from_groups([list(range(SIZE))], [0]).states(index)
    result = abc_posterior(GROUPS, frozenset(), {"one": (one_model, 0.95)}, {"one": one_model},
                           rng, draws=200)
    assert result.accepted == 0


def test_auc_extremes_and_ties() -> None:
    assert auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auc([1.0], [1.0]) == 0.5
    assert auc([0.0], [1.0]) == 0.0


def test_style_score_flags_the_odd_endpoint() -> None:
    rng = np.random.default_rng(6)
    features = {f"e{k}": rng.normal(0.0, 1.0, size=(6, 5)) for k in range(7)}
    features["odd"] = rng.normal(8.0, 1.0, size=(6, 5))
    scores = style_scores(features)
    assert max(scores, key=lambda e: scores[e]) == "odd"


def test_anchors_are_always_in_the_main_model_of_their_family() -> None:
    """Endpoints disguised by a persona are the main model's copies: never variants."""
    rng = np.random.default_rng(21)
    groups = [list(range(11)), list(range(11, 22)), list(range(22, 33))]
    anchors = frozenset({1, 7, 15})
    for _ in range(200):
        truth = organiser_truth(rng, groups, frozenset({1, 15, 20}), 1.0, [(1,)] * 3, anchors)
        assert len(truth.family_groups()) == 3
        for family in truth.family_groups():
            holders = [m for m in family if m in anchors]
            if not holders:
                continue
            main = next(g for g in truth.model_groups() if holders[0] in g)
            assert all(a in main for a in holders)
            assert len(main) >= len(holders)
