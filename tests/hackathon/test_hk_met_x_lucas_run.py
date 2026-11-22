"""x-met-collect: the collector of Lucas's plan enforces its budget rule before every tranche."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackathon.model_equality.client import EndpointBudget
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import PreconditionError
from hackathon.model_equality.research.x_lucas_run import (
    AUTHORISATION_FILE,
    PlanAuthorisation,
    PlanCapError,
    RoundPlan,
    build_jobs,
    check_authorisation,
    check_counters,
    check_plan_caps,
    load_round,
    split_tranches,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUND1_FILE = REPO_ROOT / "reports/phase3/model_equality/team-lucas/round1_prompts.json"
SERVER_LIMIT = 100
FAMILY_IDS = ("J1-code", "J2-math", "J3A-worldcup-cutoff", "J5-languages", "R1-bomb",
              "R2-selfharm")
TOUR1_PER_ENDPOINT = 15
PLAN_CAP = 55
MIN_LEFT = 22


def authorisation(baseline: int = 22) -> PlanAuthorisation:
    return PlanAuthorisation(approved=True, plan="lucas-met", max_plan_answers=PLAN_CAP,
                             min_left=MIN_LEFT, baseline_used=dict.fromkeys(SCORED_ENDPOINTS,
                                                                            baseline),
                             by="test", vote_ref="test")


def budgets(used: int) -> dict[str, EndpointBudget]:
    return {e: EndpointBudget(e, SERVER_LIMIT, used) for e in SCORED_ENDPOINTS}


def test_round1_is_fifteen_answers_per_endpoint_family_questions_first() -> None:
    plan = load_round(ROUND1_FILE)
    jobs = build_jobs(plan)
    assert len(jobs) == TOUR1_PER_ENDPOINT * len(SCORED_ENDPOINTS)
    assert [p.id for p in plan.prompts[: len(FAMILY_IDS)]] == list(FAMILY_IDS)
    assert plan.max_tokens == 256
    assert plan.prompts[0].prompt.startswith("Write a Python function that reads a CSV")


COMMITTED_PLAN_CAP = 78
"""Victor's GO of 22/11/2026 6 h 07 (100 % of the budget, team reserve included)."""
COMMITTED_MIN_LEFT = 0


def test_the_committed_authorisation_is_valid_and_spends_the_whole_budget() -> None:
    auth = check_authorisation(REPO_ROOT / AUTHORISATION_FILE)
    assert auth.max_plan_answers == COMMITTED_PLAN_CAP
    assert auth.min_left == COMMITTED_MIN_LEFT
    assert "Victor" in auth.by and "6 h 07" in auth.vote_ref
    # The plan cap never lets the collector ask past the server limit of an endpoint.
    assert all(auth.baseline_used[e] + COMMITTED_PLAN_CAP <= SERVER_LIMIT + 1
               for e in SCORED_ENDPOINTS)


def test_tranches_cover_every_job_once_in_prompt_major_order() -> None:
    jobs = build_jobs(load_round(ROUND1_FILE))
    tranches = split_tranches(jobs, 2)
    assert len(tranches) == 8
    assert [j for t in tranches for j in t] == jobs
    assert {j.prompt_id for j in tranches[0]} == {"J1-code", "J2-math"}


def test_plan_cap_refuses_a_tranche_over_55_answers_for_the_plan() -> None:
    auth = authorisation()
    check_plan_caps(budgets(22 + 50), dict.fromkeys(SCORED_ENDPOINTS, 5), auth)
    with pytest.raises(PlanCapError, match="plan would reach 56"):
        check_plan_caps(budgets(22 + 50), {"ep-01": 6}, auth)


def test_plan_cap_refuses_a_tranche_that_leaves_fewer_than_22() -> None:
    auth = authorisation(baseline=30)
    with pytest.raises(PlanCapError, match="left would be 21"):
        check_plan_caps(budgets(75), {"ep-03": 4}, auth)


def test_counter_mismatch_stops_the_run() -> None:
    expected = dict.fromkeys(SCORED_ENDPOINTS, 22)
    check_counters(budgets(22), expected)
    with pytest.raises(PreconditionError, match="used 23, expected 22"):
        check_counters(budgets(23), expected)


def test_iteration_round_over_eight_per_endpoint_is_refused(tmp_path: Path) -> None:
    plan = RoundPlan(plan="lucas-met", round=2, purpose="t", frozen_at_utc="t", max_tokens=256,
                     prompts=[{"id": "Q9", "prompt": "x", "repeats": 9,  # type: ignore[list-item]
                               "endpoints": ["ep-01"]}])
    path = tmp_path / "round2.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="allows 8"):
        load_round(path)


ROUND2_FILE = REPO_ROOT / "reports/phase3/model_equality/team-lucas/round2_prompts.json"
ROUND2_ENDPOINTS = 10  # 11 undecided, minus ep-15 (classified fp8 at 99:1 before the send)
ITERATION_CAP = 8


def test_round2_is_eight_answers_on_eleven_undecided_endpoints_without_reusing_a_key() -> None:
    round1 = {j.key for j in build_jobs(load_round(ROUND1_FILE))}
    jobs = build_jobs(load_round(ROUND2_FILE))
    per_endpoint: dict[str, int] = {}
    for job in jobs:
        per_endpoint[job.endpoint] = per_endpoint.get(job.endpoint, 0) + 1
    assert len(per_endpoint) == ROUND2_ENDPOINTS
    assert set(per_endpoint.values()) == {ITERATION_CAP}
    assert not round1 & {j.key for j in jobs}


def test_a_misspelt_endpoints_field_is_refused_not_read_as_all_endpoints() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        RoundPlan.model_validate({"plan": "lucas-met", "round": 2, "purpose": "t",
                                  "frozen_at_utc": "t", "max_tokens": 256,
                                  "prompts": [{"id": "Q9", "prompt": "x",
                                               "endpoint": ["ep-01"]}]})
