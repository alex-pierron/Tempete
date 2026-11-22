"""Team C: the +10 collector is ready and guarded; prompts are frozen and new (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.model_equality.client import EndpointBudget
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_prompts import (
    PROMPT_COUNT,
    STAGE_SIZE,
    generate_prompts,
)
from hackathon.model_equality.research.c_plus10_run import (
    AUTHORISATION_FILE,
    PROMPTS_FILE,
    AfterCheck,
    AuthorisationError,
    PreconditionError,
    build_jobs,
    check_after,
    check_authorisation,
    check_before,
    expected_used,
    load_prompts,
)

ENDPOINTS = list(SCORED_ENDPOINTS)
REPO_ROOT = Path(__file__).resolve().parents[2]


def plan_state(extra: int = 0) -> dict[str, EndpointBudget]:
    """The server state the plan was made on (7 used, 8 on ep-09..12), plus `extra` answers."""
    return {e: EndpointBudget(e, 100, expected_used(e) + extra) for e in ENDPOINTS}


def test_generated_prompts_are_new_interleaved_and_deterministic() -> None:
    taken = {"Compute 1 + 1. Answer with only the number."}
    first = generate_prompts(taken)
    assert first == generate_prompts(taken)
    assert len(first) == PROMPT_COUNT and len({t for _, t in first}) == PROMPT_COUNT
    operators = [t.split()[2] for _, t in first]
    assert operators[:6] == ["+", "-", "*", "+", "-", "*"]
    blocked = generate_prompts({t for _, t in first})
    assert {t for _, t in blocked}.isdisjoint({t for _, t in first})


def test_frozen_prompt_file_has_two_stages_of_ten() -> None:
    path = REPO_ROOT / PROMPTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["stage_1"]) == len(payload["stage_2"]) == STAGE_SIZE
    assert len(load_prompts(path, 1)) == 10 and len(load_prompts(path, 2)) == 20
    ids = [row["id"] for key in ("stage_1", "stage_2") for row in payload[key]]
    assert len(set(ids)) == 20
    with pytest.raises(ValueError):
        load_prompts(path, 3)


def test_jobs_are_ten_per_endpoint_and_prompt_major() -> None:
    prompts = load_prompts(REPO_ROOT / PROMPTS_FILE, 1)
    jobs = build_jobs(ENDPOINTS, prompts)
    assert len(jobs) == 330
    assert {j.prompt_id for j in jobs[: len(ENDPOINTS)]} == {prompts[0][0]}
    assert all(j.repeat == 0 for j in jobs)
    for endpoint in ENDPOINTS:
        assert sum(1 for j in jobs if j.endpoint == endpoint) == 10


def test_collector_refuses_without_a_human_authorisation(tmp_path: Path) -> None:
    assert not (REPO_ROOT / AUTHORISATION_FILE).exists()  # never created by the tool
    with pytest.raises(AuthorisationError):
        check_authorisation(tmp_path / "missing.json", 1)
    bad = tmp_path / "auth.json"
    bad.write_text(json.dumps({"approved": True, "stage": 2, "by": "V", "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):
        check_authorisation(bad, 1)
    bad.write_text(json.dumps({"approved": True, "stage": 1, "by": "V"}))
    with pytest.raises(AuthorisationError):
        check_authorisation(bad, 1)
    bad.write_text(json.dumps({"approved": True, "stage": 1, "by": "V", "vote_ref": "vote 21/11"}))
    assert check_authorisation(bad, 1)["by"] == "V"


def test_precondition_accepts_the_plan_state_and_stops_on_any_other() -> None:
    check_before(plan_state(), ENDPOINTS, {}, 10)
    state = plan_state()
    state["ep-04"] = EndpointBudget("ep-04", 100, 9)
    with pytest.raises(PreconditionError) as error:
        check_before(state, ENDPOINTS, {}, 10)
    assert "ep-04" in str(error.value)
    check_before(plan_state(3), ENDPOINTS, dict.fromkeys(ENDPOINTS, 3), 10)  # a resumed run
    with pytest.raises(PreconditionError):
        check_before(plan_state(3), ENDPOINTS, dict.fromkeys(ENDPOINTS, 2), 10)  # lost request


def test_after_check_flags_overspend_and_stage_overflow() -> None:
    before, after = plan_state(), plan_state(10)
    stored = dict.fromkeys(ENDPOINTS, 10)
    assert all(c.ok for c in check_after(before, after, stored, {}, 10))
    after["ep-07"] = EndpointBudget("ep-07", 100, expected_used("ep-07") + 11)
    assert not {c.endpoint: c for c in check_after(before, after, stored, {}, 10)}["ep-07"].ok
    assert not AfterCheck("ep-01", 7, 18, 11, 11, 10).ok
