"""Prompt bank, phased budget plan, and the exact per-endpoint cost computation."""

from __future__ import annotations

from collections import Counter

import pytest

from hackathon.model_equality.client import ENDPOINT_LIMIT_DEFAULT, BudgetPolicy
from hackathon.model_equality.planning import (
    REFERENCE_ENDPOINTS,
    SCORED_ENDPOINTS,
    build_plan,
    detect_prompt_drift,
    format_plan,
)
from hackathon.model_equality.prompts import (
    CATEGORIES,
    PHASES,
    PhaseError,
    build_bank,
    category_of,
    expand_phase,
    get_phase,
)
from hackathon.model_equality.prompts.phases import MIN_REPEATS_PER_PROMPT
from hackathon.model_equality.store import Answer

BUDGET = ENDPOINT_LIMIT_DEFAULT
RESERVE = 10
PHASE_1_MAX_SHARE = 0.4


def keys_of(phase_name: str) -> set[tuple[str, int]]:
    return {(u.prompt.id, u.repeat) for u in expand_phase(get_phase(phase_name))}


def test_bank_ids_are_unique_stable_and_well_formed() -> None:
    bank = build_bank()
    assert len(bank) == len({p.id for p in bank}) == 92
    assert [p.id for p in bank] == [p.id for p in build_bank()]
    assert {p.category for p in bank} == set(CATEGORIES)
    for prompt in bank:
        assert prompt.id.startswith(prompt.category + "-")
        assert category_of(prompt.id) == prompt.category
        assert 1 <= prompt.max_tokens <= 256
        assert prompt.text.strip() == prompt.text and len(prompt.text) < 1000
    assert category_of("custom-7") == "other"


def test_ep_phases_cost_exactly_40_30_20_and_leave_the_reserve() -> None:
    costs = [len(expand_phase(get_phase(f"ep-{n}"))) for n in (1, 2, 3)]
    assert costs == [40, 30, 20]
    assert costs[0] <= PHASE_1_MAX_SHARE * BUDGET
    assert sum(costs) == BUDGET - RESERVE


def test_ep_phases_are_disjoint_and_every_prompt_is_repeated() -> None:
    k1, k2, k3 = keys_of("ep-1"), keys_of("ep-2"), keys_of("ep-3")
    assert not (k1 & k2) and not (k1 & k3) and not (k2 & k3)
    for name in ("ep-1", "ep-2"):
        repeats = Counter(u.prompt.id for u in expand_phase(get_phase(name)))
        assert min(repeats.values()) >= MIN_REPEATS_PER_PROMPT
    # phase 3 adds repeats of phase-1 prompts, so (prompt, 4 repeats) exist after phase 3
    ids_3 = {pid for pid, _ in k3}
    assert ids_3 <= {pid for pid, _ in k1}


def test_phase_expansion_is_repeat_major_and_filters_by_category() -> None:
    units = expand_phase(get_phase("ep-1"))
    repeats = [u.repeat for u in units]
    assert repeats == sorted(repeats)
    only = expand_phase(get_phase("ep-1"), categories=["arith"])
    assert {u.prompt.category for u in only} == {"arith"} and len(only) == 8


def test_ref_phases_mirror_ep_shapes_and_ref_full_covers_the_bank() -> None:
    assert keys_of("ref-1") == keys_of("ep-1")
    assert len(expand_phase(get_phase("ref-full"))) == 92 * 3


def test_unknown_phase_lists_the_valid_ones() -> None:
    with pytest.raises(PhaseError, match="ep-1"):
        get_phase("nope")
    assert set(PHASES) >= {"ep-1", "ep-2", "ep-3", "ref-1", "ref-full"}


def stored_answers(endpoint: str, units_keys: set[tuple[str, int]]) -> dict[tuple[str, str, int], Answer]:
    return {(endpoint, pid, rep): Answer(endpoint, pid, rep, "", "r") for pid, rep in units_keys}


def test_plan_cost_is_exact_for_a_fresh_phase_one() -> None:
    plan = build_plan(expand_phase(get_phase("ep-1")), SCORED_ENDPOINTS, {},
                      BudgetPolicy(BUDGET, RESERVE))
    assert plan.total_requests == 40 * 33
    assert all(c.to_request == 40 and c.truncated == 0 and c.left_after == 60 for c in plan.costs)
    assert not plan.drift
    assert plan.caps() == {e: 40 for e in SCORED_ENDPOINTS}
    text = format_plan(plan, BudgetPolicy(BUDGET, RESERVE))
    assert "TOTAL requests to send: 1320" in text and "ep-33" in text


def test_plan_skips_stored_answers_and_counts_them() -> None:
    stored = stored_answers("ep-01", keys_of("ep-1"))
    plan = build_plan(expand_phase(get_phase("ep-1")), ["ep-01", "ep-02"], stored,
                      BudgetPolicy(BUDGET, RESERVE))
    by_endpoint = {c.endpoint: c for c in plan.costs}
    assert by_endpoint["ep-01"].stored == 40 and by_endpoint["ep-01"].to_request == 0
    assert by_endpoint["ep-02"].to_request == 40
    assert {j.endpoint for j in plan.jobs} == {"ep-02"}


def test_plan_truncates_to_protect_the_reserve() -> None:
    stored = stored_answers("ep-01", {(f"x-{i}", 0) for i in range(70)})
    plan = build_plan(expand_phase(get_phase("ep-3")), ["ep-01"], stored,
                      BudgetPolicy(BUDGET, RESERVE))
    cost = plan.costs[0]
    assert cost.wanted == 20 and cost.to_request == 20 and cost.total_after == 90
    stored_more = stored_answers("ep-01", {(f"x-{i}", 0) for i in range(85)})
    plan = build_plan(expand_phase(get_phase("ep-3")), ["ep-01"], stored_more,
                      BudgetPolicy(BUDGET, RESERVE))
    assert plan.costs[0].to_request == 5 and plan.costs[0].truncated == 15
    assert len(plan.jobs) == 5


def test_plan_uses_the_server_left_when_it_is_stricter() -> None:
    plan = build_plan(expand_phase(get_phase("ep-1")), ["ep-01"], {},
                      BudgetPolicy(BUDGET, RESERVE), server_left={"ep-01": 25})
    assert plan.costs[0].to_request == 15


def test_reference_plan_is_unlimited() -> None:
    plan = build_plan(expand_phase(get_phase("ref-full")), REFERENCE_ENDPOINTS, {}, None)
    assert plan.unlimited and plan.caps() is None
    assert plan.total_requests == 92 * 3 * 9
    assert all(c.left_after is None for c in plan.costs)


def test_jobs_are_round_robin_over_endpoints() -> None:
    plan = build_plan(expand_phase(get_phase("ep-1")), ["ep-01", "ep-02", "ep-03"], {},
                      BudgetPolicy(BUDGET, RESERVE))
    first_three = [j.endpoint for j in plan.jobs[:3]]
    assert first_three == ["ep-01", "ep-02", "ep-03"]
    assert len({(j.prompt_id, j.repeat) for j in plan.jobs[:3]}) == 1


def test_prompt_drift_is_detected_between_bank_and_store() -> None:
    bank_prompt = build_bank()[0]
    stored = {
        ("ep-01", bank_prompt.id, 0): Answer("ep-01", bank_prompt.id, 0, "edited text", "r"),
        ("ep-01", "custom-1", 0): Answer("ep-01", "custom-1", 0, "whatever", "r"),
        ("ep-02", bank_prompt.id, 0): Answer("ep-02", bank_prompt.id, 0, bank_prompt.text, "r"),
    }
    messages = detect_prompt_drift(stored)
    assert len(messages) == 1 and bank_prompt.id in messages[0]
