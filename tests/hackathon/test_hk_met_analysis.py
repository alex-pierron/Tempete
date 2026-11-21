"""Offline analysis on synthetic endpoints: boilerplate, clustering, calibration, accounting.

The synthetic data (hackathon.model_equality.simulate) proves the software works end to end;
it says nothing about the real power of the method on the real server.
"""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality import analysis as analysis_module
from hackathon.model_equality.analysis import (
    Calibration,
    calibrate_on_reference,
    exact_match_matrix,
    family_names,
    model_mcc,
    run_analysis,
    select_answers,
)
from hackathon.model_equality.boilerplate import strip_boilerplate
from hackathon.model_equality.clustering import (
    ClusteringError,
    calibrate_model_threshold,
    choose_family_threshold,
    combine_channels,
    cut_two_levels,
    to_distance,
)
from hackathon.model_equality.prompts import PromptRepeat, expand_phase, get_phase
from hackathon.model_equality.scoring import official_score
from hackathon.model_equality.simulate import (
    KIND_PERSONA,
    KIND_PLAIN,
    KIND_SYSTEM,
    KIND_TEMPERATURE,
    KIND_WATERMARK,
    STRENGTH_LORA_MID,
    SimEndpoint,
    reference_endpoints,
    simulate_answers,
)
from hackathon.model_equality.store import Answer

PERMUTATIONS = 40
STRONG = 1.6


def ref_units() -> list[PromptRepeat]:
    return expand_phase(get_phase("ref-1")) + expand_phase(get_phase("ref-2"))


def small_scored_world() -> list[SimEndpoint]:
    return [
        SimEndpoint("ep-01", "A", "A1", 0.0, KIND_PLAIN),
        SimEndpoint("ep-02", "A", "A1", 0.0, KIND_PERSONA),
        SimEndpoint("ep-03", "A", "A1", 0.0, KIND_WATERMARK),
        SimEndpoint("ep-04", "A", "A1", 0.0, KIND_TEMPERATURE),
        SimEndpoint("ep-05", "A", "A2", STRONG, KIND_PLAIN),
        SimEndpoint("ep-06", "A", "A2", STRONG, KIND_SYSTEM),
        SimEndpoint("ep-07", "B", "B1", 0.0, KIND_PLAIN),
        SimEndpoint("ep-08", "B", "B1", 0.0, KIND_SYSTEM),
        SimEndpoint("ep-09", "B", "B1", 0.0, KIND_PERSONA),
        SimEndpoint("ep-10", "B", "B2", STRONG, KIND_PLAIN),
        SimEndpoint("ep-11", "B", "B2", STRONG, KIND_WATERMARK),
        SimEndpoint("ep-12", "B", "B1", 0.0, KIND_WATERMARK),
    ]


@pytest.fixture(scope="module")
def calibration() -> Calibration:
    ref = reference_endpoints()
    answers = simulate_answers(ref, ref_units())
    truth = {e.name: (e.family, e.model, e.kind) for e in ref}
    result, diagnostics, ref_analysis = calibrate_on_reference(
        answers, truth, n_permutations=PERMUTATIONS)
    assert diagnostics and ref_analysis.endpoints == sorted(truth)
    return result


def test_boilerplate_prefix_is_stripped_only_when_spread_over_categories() -> None:
    answers: list[Answer] = []
    for category in ("arith", "fact", "code", "list"):
        for index in range(4):
            answers.append(Answer("ep-01", f"{category}-{index:03d}", 0, "p",
                                  f"Arr matey the answer {category} {index}"))
    # a prefix confined to ONE category is content (e.g. "def" for code), not boilerplate
    for index in range(12):
        answers.append(Answer("ep-02", f"code-{index:03d}", 0, "p", f"def f{index}() return {index}"))
    cleaned, notes = strip_boilerplate(answers)
    assert {n.endpoint for n in notes} == {"ep-01"}
    assert any(n.phrase == "arr matey" for n in notes)
    assert all(not a.response.lower().startswith("arr matey") for a in cleaned if a.endpoint == "ep-01")
    untouched = [a for a in cleaned if a.endpoint == "ep-02"]
    assert all(a.response.startswith("def") for a in untouched)


def test_boilerplate_never_empties_an_answer() -> None:
    answers = [Answer("ep-01", f"{c}-000", r, "p", "Sure")
               for c in ("arith", "fact", "code") for r in range(5)]
    cleaned, _ = strip_boilerplate(answers)
    assert all(a.response.strip() for a in cleaned)


def test_pipeline_recovers_families_and_models_despite_disguises(calibration: Calibration) -> None:
    world = small_scored_world()
    units = expand_phase(get_phase("ep-1")) + expand_phase(get_phase("ep-2"))
    answers = simulate_answers(world, units)
    result = run_analysis(answers, [e.name for e in world], calibration=calibration,
                          n_permutations=PERMUTATIONS)
    families, models = family_names(result.clustering)
    prediction = {n: (f, m) for n, f, m in zip(result.endpoints, families, models, strict=True)}
    score = official_score({e.name: (e.family, e.model) for e in world}, prediction)
    assert score.family.mcc == 1.0
    assert score.model.mcc >= 0.9
    # persona / system prompt / watermark / temperature endpoints stay with their base model
    assert prediction["ep-01"] == prediction["ep-02"] == prediction["ep-03"] == prediction["ep-04"]
    assert prediction["ep-05"] == prediction["ep-06"] != prediction["ep-01"]
    assert any(n.endpoint == "ep-02" and n.phrase == "arr matey" for n in result.boilerplate)
    assert result.accounting.rows_in == result.accounting.rows_used == len(answers)


def test_labels_are_nested_models_inside_families(calibration: Calibration) -> None:
    world = small_scored_world()
    answers = simulate_answers(world, expand_phase(get_phase("ep-1")))
    result = run_analysis(answers, [e.name for e in world], calibration=calibration,
                          n_permutations=PERMUTATIONS)
    families, models = family_names(result.clustering)
    family_of_model: dict[str, set[str]] = {}
    for family, model in zip(families, models, strict=True):
        family_of_model.setdefault(model, set()).add(family)
    assert all(len(v) == 1 for v in family_of_model.values())


def test_calibration_json_roundtrip_and_in_sample_mcc(calibration: Calibration) -> None:
    again = Calibration.from_json(calibration.to_json())
    assert again.weights == calibration.weights and again.tau_model == calibration.tau_model
    assert calibration.mcc_model_in_sample > 0.8
    assert all(w >= 0 for w in calibration.weights.values()) and max(calibration.weights.values()) > 0


def test_reference_in_sample_labels_match_truth(calibration: Calibration) -> None:
    ref = reference_endpoints()
    answers = simulate_answers(ref, ref_units())
    result = run_analysis(answers, [e.name for e in ref], calibration=calibration,
                          n_permutations=PERMUTATIONS)
    labels = [(e.family, e.model) for e in ref]
    assert model_mcc(result, labels) > 0.8


def test_row_accounting_reports_rejected_rows_with_reasons() -> None:
    answers = [Answer("ep-01", "arith-000", 0, "p", "1"), Answer("ep-01", "arith-000", 1, "p", ""),
               Answer("ep-99", "arith-000", 0, "p", "2"), Answer("ep-01", "arith-009", 0, "p", "3")]
    kept, accounting = select_answers(answers, ["ep-01"], units={("arith-000", 0), ("arith-000", 1)})
    assert [a.endpoint for a in kept] == ["ep-01", "ep-01"]
    assert accounting.rows_in == 4 and accounting.rows_used == 2
    assert accounting.reason_counts() == {"endpoint_not_selected": 1, "outside_selected_units": 1}
    assert accounting.empty_responses == 1


def test_exact_match_matrix_measures_agreement_and_determinism() -> None:
    answers = [
        Answer("ep-01", "arith-000", 0, "p", "42"), Answer("ep-01", "arith-000", 1, "p", "42"),
        Answer("ep-02", "arith-000", 0, "p", "42"), Answer("ep-02", "arith-000", 1, "p", "43"),
        Answer("ep-03", "arith-001", 0, "p", "9"), Answer("ep-03", "arith-001", 1, "p", "9"),
    ]
    matrix = exact_match_matrix(answers, ["ep-01", "ep-02", "ep-03"])
    assert matrix[0, 0] == 1.0 and matrix[1, 1] == 0.0
    assert matrix[0, 1] == matrix[1, 0] == 0.5
    assert np.isnan(matrix[0, 2])  # no shared prompt


def test_threshold_calibration_finds_the_plateau_midpoint() -> None:
    distance = np.array([[0, 1, 1, 9], [1, 0, 1, 9], [1, 1, 0, 9], [9, 9, 9, 0]], dtype=float)
    labels = [("P", "S"), ("P", "S"), ("P", "S"), ("P", "T")]
    result = calibrate_model_threshold(distance, labels)
    assert result.mcc_model == 1.0 and 1.0 < result.tau_model < 9.0
    with pytest.raises(ClusteringError):
        calibrate_model_threshold(distance[:2, :2], labels[:2])


def test_two_level_cut_is_nested_and_family_gap_is_found() -> None:
    # two families of three endpoints; inside each, two models at distance 3
    distance = np.full((6, 6), 40.0)
    for block in ((0, 1, 2), (3, 4, 5)):
        for i in block:
            for j in block:
                distance[i, j] = 0.0 if i == j else 0.5
        distance[block[0], block[2]] = distance[block[2], block[0]] = 3.0
        distance[block[1], block[2]] = distance[block[2], block[1]] = 3.0
    clustering = cut_two_levels(distance, distance, tau_model=1.5)
    assert clustering.family_ids[:3] == [clustering.family_ids[0]] * 3
    assert clustering.family_ids[0] != clustering.family_ids[3]
    assert len(set(clustering.model_ids)) == 4
    for model in set(clustering.model_ids):
        members = [i for i, m in enumerate(clustering.model_ids) if m == model]
        assert len({clustering.family_ids[i] for i in members}) == 1


def test_family_threshold_falls_back_without_a_clear_gap() -> None:
    tree = np.array([[0, 1, 2.5, 2], [2, 3, 2.8, 3]], dtype=float)
    assert choose_family_threshold(tree, floor=0.0) == pytest.approx(4.0)  # 2 * minimum floor


def test_combine_channels_is_a_weighted_stouffer_and_flags_undefined_pairs() -> None:
    first = np.array([[np.nan, 2.0], [2.0, np.nan]])
    second = np.array([[np.nan, 4.0], [4.0, np.nan]])
    combined = combine_channels({("c", "a"): first, ("c", "b"): second},
                                {("c", "a"): 1.0, ("c", "b"): 1.0})
    assert combined[0, 1] == pytest.approx(6.0 / np.sqrt(2))
    distance, undefined = to_distance(np.array([[0.0, np.nan], [np.nan, 0.0]]))
    assert undefined == [(0, 1)] and distance[0, 1] == distance[1, 0] >= 0.0
    with pytest.raises(ClusteringError):
        combine_channels({("c", "a"): first}, {("c", "a"): 0.0})


def test_analysis_module_exposes_a_default_cut() -> None:
    assert analysis_module.DEFAULT_TAU_MODEL > 0
    assert STRONG >= STRENGTH_LORA_MID
