"""Team C: the v3 collector is ready and locked behind its own authorisation (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.model_equality.client import EndpointBudget
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import (
    PROMPTS_FILE,
    AuthorisationError,
    PreconditionError,
    check_before,
    expected_used,
    load_prompts,
)
from hackathon.model_equality.research.c_v3_run import (
    ARITHMETIC_PROMPTS,
    AUTHORISATION_FILE,
    IDENTITY_FILE,
    IDENTITY_UNITS,
    build_jobs,
    check_authorisation_v3,
    load_identity,
)

ENDPOINTS = list(SCORED_ENDPOINTS)
REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_CAP = 16


def test_frozen_identity_file_matches_the_ref_wave() -> None:
    formulations, draws, max_tokens = load_identity(REPO_ROOT / IDENTITY_FILE)
    assert [pid for pid, _ in formulations] == ["d-ident-000", "d-ident-001", "d-ident-002"]
    assert draws * len(formulations) == IDENTITY_UNITS and max_tokens == 256
    assert formulations[0][1] == "Give your name and creator in one line."


def test_jobs_are_sixteen_per_endpoint_arithmetic_first() -> None:
    arithmetic = load_prompts(REPO_ROOT / PROMPTS_FILE, 1)[:ARITHMETIC_PROMPTS]
    formulations, draws, max_tokens = load_identity(REPO_ROOT / IDENTITY_FILE)
    jobs = build_jobs(ENDPOINTS, arithmetic, formulations, draws, max_tokens)
    assert len(jobs) == STAGE_CAP * len(ENDPOINTS) == 528
    for endpoint in ENDPOINTS:
        assert sum(1 for j in jobs if j.endpoint == endpoint) == STAGE_CAP
    assert all(j.prompt_id.startswith("p10-arith") for j in jobs[: 10 * len(ENDPOINTS)])
    identity = jobs[10 * len(ENDPOINTS):]
    assert {j.prompt_id for j in identity} == {"d-ident-000", "d-ident-001", "d-ident-002"}
    assert {j.repeat for j in identity} == {0, 1}
    assert len({j.key for j in jobs}) == len(jobs)


def test_v3_collector_has_its_own_authorisation(tmp_path: Path) -> None:
    assert not (REPO_ROOT / AUTHORISATION_FILE).exists()  # never created by the tool
    with pytest.raises(AuthorisationError):
        check_authorisation_v3(tmp_path / "missing.json")
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"approved": True, "stage": 1, "by": "V", "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):  # the +10 authorisation does not open v3
        check_authorisation_v3(path)
    path.write_text(json.dumps({"approved": True, "stage": "v3", "by": "V"}))
    with pytest.raises(AuthorisationError):
        check_authorisation_v3(path)
    path.write_text(json.dumps({"approved": True, "stage": "v3", "by": "V", "vote_ref": "vote"}))
    assert check_authorisation_v3(path)["by"] == "V"


def test_precondition_uses_the_v3_cap() -> None:
    state = {e: EndpointBudget(e, 100, expected_used(e)) for e in ENDPOINTS}
    check_before(state, ENDPOINTS, {}, STAGE_CAP)
    resumed = {e: EndpointBudget(e, 100, expected_used(e) + 4) for e in ENDPOINTS}
    check_before(resumed, ENDPOINTS, dict.fromkeys(ENDPOINTS, 4), STAGE_CAP)
    with pytest.raises(PreconditionError):
        check_before(resumed, ENDPOINTS, dict.fromkeys(ENDPOINTS, 3), STAGE_CAP)
