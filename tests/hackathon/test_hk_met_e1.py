"""Phase E1 (5 prompts x 1 draw), the small-N statistic, and the ref-limit regression. No network."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.client import BudgetPolicy
from hackathon.model_equality.planning import SCORED_ENDPOINTS, build_plan
from hackathon.model_equality.prompts import expand_phase, get_phase
from hackathon.model_equality.prompts.extra import EXTRA_PROMPTS, FAMILY_PROMPT_IDS
from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS, EP_E1_NAME, REF_E1_NAME
from hackathon.model_equality.research.candidates import candidate_by_id
from hackathon.model_equality.smalln import (
    SmallNError,
    build_small_n_data,
    cluster_at,
    pair_distance,
)
from hackathon.model_equality.store import Answer

E1_SIZE = 5
E1_RESERVE = 100 - E1_SIZE
MAX_PROMPT_CHARS = 1000
"""Characters never exceed tokens, so 1000 characters is safely under the 1024-token limit."""
MAX_NEW_TOKENS = 256
MIN_FAMILY_PROMPTS = 2
MIN_YESNO_PROMPTS = 2


def test_e1_has_five_distinct_prompts_within_the_server_limits() -> None:
    units = expand_phase(get_phase(EP_E1_NAME))
    ids = [unit.prompt.id for unit in units]
    assert len(units) == E1_SIZE and len(set(ids)) == E1_SIZE and tuple(ids) == E1_PROMPT_IDS
    assert {unit.repeat for unit in units} == {0}
    for unit in units:
        assert 0 < len(unit.prompt.text) <= MAX_PROMPT_CHARS
        assert 0 < unit.prompt.max_tokens <= MAX_NEW_TOKENS
        assert "echo" not in unit.prompt.id
    assert [u.prompt.id for u in expand_phase(get_phase(REF_E1_NAME))] == ids


def test_e1_mixes_family_and_lora_strength_prompts() -> None:
    family = [pid for pid in E1_PROMPT_IDS if pid in FAMILY_PROMPT_IDS or pid.startswith("fact-")]
    yesno = [pid for pid in E1_PROMPT_IDS if pid.startswith("cand-yesno-")]
    assert len(family) >= MIN_FAMILY_PROMPTS and len(yesno) >= MIN_YESNO_PROMPTS


def test_plan_costs_exactly_five_per_endpoint_and_the_reserve_enforces_the_cap() -> None:
    units = expand_phase(get_phase(EP_E1_NAME))
    plan = build_plan(units, SCORED_ENDPOINTS, {}, BudgetPolicy(limit=100, reserve=E1_RESERVE))
    assert plan.total_requests == E1_SIZE * len(SCORED_ENDPOINTS)
    assert all(cost.to_request == E1_SIZE and cost.truncated == 0 for cost in plan.costs)
    assert plan.caps() == {endpoint: E1_SIZE for endpoint in SCORED_ENDPOINTS}


def test_a_rerun_after_a_complete_e1_asks_nothing() -> None:
    units = expand_phase(get_phase(EP_E1_NAME))
    stored = {
        (e, u.prompt.id, u.repeat): Answer(e, u.prompt.id, u.repeat, u.prompt.text, "x")
        for e in SCORED_ENDPOINTS for u in units
    }
    plan = build_plan(units, SCORED_ENDPOINTS, stored, BudgetPolicy(limit=100, reserve=E1_RESERVE))
    assert plan.total_requests == 0


def test_extra_prompts_are_unique_and_the_candidate_copies_match_research() -> None:
    ids = [prompt.id for prompt in EXTRA_PROMPTS]
    assert len(ids) == len(set(ids))
    research = candidate_by_id()
    for prompt in EXTRA_PROMPTS:
        if prompt.id.startswith("cand-"):
            assert research[prompt.id].text == prompt.text
            assert research[prompt.id].max_tokens == prompt.max_tokens


def _answers(texts: dict[str, list[str]]) -> list[Answer]:
    """Answers of endpoints named after the dict keys, one per text (unit i = prompt p<i>)."""
    return [Answer(endpoint, f"p{i}", 0, "q", text)
            for endpoint, items in texts.items() for i, text in enumerate(items)]


def test_small_n_distance_is_symmetric_zero_on_the_diagonal_and_splits_two_groups() -> None:
    short = ["Yes.", "No.", "Paris."]
    rambling = ["Yes. Because many long reasons follow here and there and everywhere.",
                "No. There is no such thing as an answer to this question at all really.",
                "Paris is a city in France that has been the capital for a very long time."]
    data = build_small_n_data(
        _answers({"e1": short, "e2": short, "e3": rambling, "e4": rambling}),
        ["e1", "e2", "e3", "e4"])
    distance = pair_distance(data, list(range(len(data.units))))
    assert np.allclose(distance, distance.T) and np.allclose(np.diag(distance), 0.0)
    assert distance[0, 1] == pytest.approx(0.0) and distance[2, 3] == pytest.approx(0.0)
    assert distance[0, 2] > 0.3
    labels = cluster_at(distance, 0.15)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert len(set(cluster_at(distance, 10.0))) == 1


def test_small_n_rejects_an_incomplete_grid_and_empty_selections() -> None:
    with pytest.raises(SmallNError):
        build_small_n_data(_answers({"e1": ["a"]}) + _answers({"e2": []}), ["e1", "e2"])
    data = build_small_n_data(_answers({"e1": ["a"], "e2": ["b"]}), ["e1", "e2"])
    with pytest.raises(SmallNError):
        pair_distance(data, [])


def test_two_level_labels_are_nested_and_ordered_cuts_are_enforced() -> None:
    from hackathon.model_equality.smalln import label_two_levels, top_merge_height

    block = 0.1
    far = 0.9
    distance = np.full((6, 6), far)
    for group in ((0, 1, 2), (3, 4), (5,)):
        for i in group:
            for j in group:
                distance[i, j] = block if i != j else 0.0
    distance[np.ix_([0, 1, 2], [3, 4])] = 0.5
    distance[np.ix_([3, 4], [0, 1, 2])] = 0.5
    families, models = label_two_levels(distance, tau_model=0.2, tau_family=0.6)
    assert len(set(models)) == 3 and len(set(families)) == 2
    for model in set(models):  # every model lies inside one family
        assert len({families[i] for i in range(6) if models[i] == model}) == 1
    assert top_merge_height(distance) > 0.6
    with pytest.raises(SmallNError):
        label_two_levels(distance, tau_model=0.7, tau_family=0.6)


def test_restrict_keeps_the_chosen_endpoints_and_length_matrix_shapes() -> None:
    from hackathon.model_equality.smalln import restrict

    texts = {"e1": ["a", "b"], "e2": ["a b", "c"], "e3": ["a b c d", "e"]}
    data = build_small_n_data(_answers(texts), ["e1", "e2", "e3"])
    sub = restrict(data, [0, 2])
    assert sub.endpoints == ("e1", "e3") and sub.log_length.shape == (2, 2)
    assert sub.components["ham"].shape == (2, 2, 2)
    assert np.allclose(sub.components["len"][:, 0, 1], data.components["len"][:, 0, 2])


def test_endpoint_length_scores_flag_the_rambling_endpoint_and_auc_is_rank_based() -> None:
    from hackathon.model_equality.research.model_level import auc, endpoint_scores, flagged_labels

    short = ["Yes", "No", "Paris", "Four", "Blue", "Red"]
    long_text = ["Yes " + "because " * 30] * 6
    endpoints = {f"e{i}": short for i in range(5)} | {"e5": long_text}
    data = build_small_n_data(_answers(endpoints), list(endpoints))
    scores = endpoint_scores(data, list(range(len(data.units))))
    assert int(np.argmax(scores["lenz"])) == 5 and int(np.argmax(scores["ramble"])) == 5
    assert scores["lenzr"][5] > 3.0
    labels = flagged_labels(scores["lenz"], np.zeros((6, 6)), threshold=1.0)
    assert labels[5] == 1 and set(labels[:5].tolist()) == {0}
    assert auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(0.875)
