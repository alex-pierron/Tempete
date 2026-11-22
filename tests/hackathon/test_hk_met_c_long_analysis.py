"""Team C: long-answer agreement statistics on synthetic text (no network)."""

from __future__ import annotations

import numpy as np

from hackathon.model_equality.research.c_data import AgreementData
from hackathon.model_equality.research.c_long_analysis import (
    build_long_form,
    decoder_groups,
    extrapolate_refinement,
    log_evidence_long,
    prefix_length,
    refinement_curve,
    split_clique,
)

BASE = "the lighthouse keeper found a sealed bottle on the rocks after the storm and read it twice"


def synthetic() -> AgreementData:
    """4 endpoints: 0, 1 identical; 2 diverges late; 3 different from the first word."""
    texts = {
        "long-a": {"ep-01": BASE, "ep-02": BASE, "ep-03": BASE[:60] + " quietly", "ep-04": "a " + BASE},
        "long-b": {"ep-01": BASE + " b", "ep-02": BASE + " b", "ep-03": BASE + " c", "ep-04": "zzz"},
    }
    return AgreementData(("ep-01", "ep-02", "ep-03", "ep-04"), ("long-a", "long-b"), texts)


def test_prefix_length_counts_shared_leading_characters() -> None:
    assert prefix_length("abcdef", "abcxyz") == 3
    assert prefix_length("", "abc") == 0
    assert prefix_length("same", "same") == 4


def test_identical_count_and_groups() -> None:
    form = build_long_form(synthetic(), ["long-a", "long-b"])
    assert form.identical_count[0, 1] == 2
    assert form.identical_count[0, 2] == 0
    assert decoder_groups(form) == [[0, 1]]
    assert decoder_groups(form, required=0)  # everything linked when nothing is required


def test_prefix_ratio_orders_late_divergence_above_early() -> None:
    form = build_long_form(synthetic(), ["long-a", "long-b"])
    mean = form.mean_prefix
    assert mean[0, 1] == 1.0
    assert mean[0, 2] > mean[0, 3]
    assert np.allclose(mean, mean.T)


def test_split_clique_separates_a_late_diverging_member() -> None:
    form = build_long_form(synthetic(), ["long-a", "long-b"])
    assert split_clique(form, [0, 1, 2], threshold=0.99) == [[0, 1], [2]]
    assert split_clique(form, [0, 1, 2], threshold=0.5) == [[0, 1, 2]]


def test_long_evidence_prefers_same_model_for_identical_pairs() -> None:
    form = build_long_form(synthetic(), ["long-a", "long-b"])
    pmf = {"cross": (0.6, 0.3, 0.1), "family": (0.4, 0.4, 0.2), "model": (0.05, 0.15, 0.8)}
    first, second = np.array([0, 0]), np.array([1, 3])
    table = log_evidence_long(form, first, second, pmf, power=1.0)
    assert table[2, 0] > table[1, 0] > table[0, 0]  # identical on both prompts: model most likely
    assert table[0, 1] > table[2, 1]  # nothing identical: cross-family beats same model


def test_refinement_curve_rises_then_ends_at_the_full_signature_count() -> None:
    data = synthetic()
    curve = refinement_curve(data, ["long-a", "long-b"], [0, 1, 2, 3], np.random.default_rng(1))
    assert curve[0] == 1.0
    assert curve[-1] == 3.0  # ep-01 and ep-02 stay together, ep-03 and ep-04 split off
    assert curve == sorted(curve)


def test_extrapolation_saturates_at_the_fitted_asymptote() -> None:
    curve = [1.0 + 4.0 * (1.0 - 0.6**k) for k in range(6)]
    predicted = extrapolate_refinement(curve, [5, 10, 20])
    assert abs(predicted[5] - curve[5]) < 0.05
    assert abs(predicted[20] - 5.0) < 0.1
    assert predicted[10] <= predicted[20]
