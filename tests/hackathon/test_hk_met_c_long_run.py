"""Team C: the guards of the authorised ep collection (no network)."""

from __future__ import annotations

import pytest

from hackathon.model_equality.client import EndpointBudget
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_long_run import (
    ADDED_PER_ENDPOINT,
    LONG_PROMPTS,
    AfterCheck,
    PreconditionError,
    build_jobs,
    check_budget_after,
    check_budget_before,
    check_prompts,
    expected_used,
)

ENDPOINTS = list(SCORED_ENDPOINTS)


def authorised_state() -> dict[str, EndpointBudget]:
    """The server state the human decision was taken on."""
    return {e: EndpointBudget(e, 100, expected_used(e)) for e in ENDPOINTS}


def test_precondition_accepts_the_authorised_state() -> None:
    check_budget_before(authorised_state(), ENDPOINTS)
    assert expected_used("ep-09") == 6 and expected_used("ep-01") == 5


def test_precondition_stops_on_any_other_value() -> None:
    state = authorised_state()
    state["ep-03"] = EndpointBudget("ep-03", 100, 7)
    state["ep-10"] = EndpointBudget("ep-10", 100, 5)
    with pytest.raises(PreconditionError) as error:
        check_budget_before(state, ENDPOINTS)
    assert "ep-03" in str(error.value) and "ep-10" in str(error.value)


def test_precondition_stops_when_an_endpoint_is_missing() -> None:
    state = authorised_state()
    del state["ep-33"]
    with pytest.raises(PreconditionError):
        check_budget_before(state, ENDPOINTS)


def test_jobs_are_exactly_five_per_endpoint_and_prompt_major() -> None:
    jobs = build_jobs(ENDPOINTS)
    assert len(jobs) == ADDED_PER_ENDPOINT * len(ENDPOINTS) == 66
    for endpoint in ENDPOINTS:
        assert sum(1 for j in jobs if j.endpoint == endpoint) == ADDED_PER_ENDPOINT
    first_block = jobs[: len(ENDPOINTS)]
    assert {j.prompt_id for j in first_block} == {LONG_PROMPTS[0][0]}
    assert all(j.repeat == 0 for j in jobs)
    assert len({j.key for j in jobs}) == len(jobs)


def test_prompts_are_open_and_short_enough() -> None:
    check_prompts()
    for _, text in LONG_PROMPTS:
        assert "yes" not in text.lower().split() and "'no'" not in text.lower()
        assert len(text) < 400


def test_after_check_flags_any_overspend() -> None:
    before = authorised_state()
    after = {e: EndpointBudget(e, 100, b.used + 2) for e, b in before.items()}
    stored = dict.fromkeys(ENDPOINTS, 2)
    assert all(c.ok for c in check_budget_after(before, after, stored))
    after["ep-04"] = EndpointBudget("ep-04", 100, before["ep-04"].used + 3)
    checks = {c.endpoint: c for c in check_budget_after(before, after, stored)}
    assert not checks["ep-04"].ok
    assert not AfterCheck("ep-01", 5, 8, 3).ok  # stored above the authorised +2


def test_resumed_run_expects_used_plus_already_stored() -> None:
    state = authorised_state()
    stored = {"ep-01": 2, "ep-02": 1}
    state["ep-01"] = EndpointBudget("ep-01", 100, 5 + 2)
    state["ep-02"] = EndpointBudget("ep-02", 100, 5 + 1)
    check_budget_before(state, ENDPOINTS, stored)
    state["ep-02"] = EndpointBudget("ep-02", 100, 5 + 2)  # one request counted but not stored
    with pytest.raises(PreconditionError) as error:
        check_budget_before(state, ENDPOINTS, stored)
    assert "ep-02" in str(error.value)


def test_after_check_uses_the_stored_delta_of_this_run() -> None:
    before = {e: EndpointBudget(e, 100, expected_used(e) + 1) for e in ENDPOINTS}
    after = {e: EndpointBudget(e, 100, expected_used(e) + 2) for e in ENDPOINTS}
    total = dict.fromkeys(ENDPOINTS, 2)
    start = dict.fromkeys(ENDPOINTS, 1)
    assert all(c.ok for c in check_budget_after(before, after, total, start))
    total["ep-07"] = 3
    assert not {c.endpoint: c for c in check_budget_after(before, after, total, start)}["ep-07"].ok
