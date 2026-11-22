"""Team C: the v4 collector is ready and locked behind its own authorisation (no network)."""

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
from hackathon.model_equality.research.c_v3_run import IDENTITY_FILE, load_identity
from hackathon.model_equality.research.c_v4_run import (
    ALLOWED_N,
    AUTHORISATION_FILE,
    build_jobs,
    check_authorisation_v4,
)

ENDPOINTS = list(SCORED_ENDPOINTS)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("n", ALLOWED_N)
def test_jobs_are_n_plus_three_per_endpoint(n: int) -> None:
    arithmetic = load_prompts(REPO_ROOT / PROMPTS_FILE, 2)[:n]
    formulations, _, max_tokens = load_identity(REPO_ROOT / IDENTITY_FILE)
    jobs = build_jobs(ENDPOINTS, arithmetic, formulations, max_tokens)
    assert len(jobs) == (n + 3) * len(ENDPOINTS)
    assert {j.prompt_id for j in jobs[: n * len(ENDPOINTS)]} == {pid for pid, _ in arithmetic}
    identity = jobs[n * len(ENDPOINTS):]
    assert {j.prompt_id for j in identity} == {"d-ident-000", "d-ident-001", "d-ident-002"}
    assert all(j.repeat == 0 for j in jobs) and len({j.key for j in jobs}) == len(jobs)
    # the plan costs: 13 / 18 / 23 answers per endpoint, 429 / 594 / 759 requests
    assert (n + 3, len(jobs)) in {(13, 429), (18, 594), (23, 759)}


def test_authorisation_names_n_and_is_separate_from_the_other_collectors(tmp_path: Path) -> None:
    assert not (REPO_ROOT / AUTHORISATION_FILE).exists()  # never created by the tool
    with pytest.raises(AuthorisationError):
        check_authorisation_v4(tmp_path / "missing.json")
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"approved": True, "stage": "v3", "n": 15, "by": "V", "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):  # the v3 authorisation does not open v4
        check_authorisation_v4(path)
    path.write_text(json.dumps({"approved": True, "stage": "v4", "n": 12, "by": "V", "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):  # N must be one of the frozen sizes
        check_authorisation_v4(path)
    path.write_text(json.dumps({"approved": True, "stage": "v4", "n": 15, "by": "V"}))
    with pytest.raises(AuthorisationError):
        check_authorisation_v4(path)
    path.write_text(json.dumps({"approved": True, "stage": "v4", "n": 15, "by": "V", "vote_ref": "vote"}))
    assert check_authorisation_v4(path)["n"] == 15


def test_precondition_uses_the_n_plus_three_cap() -> None:
    state = {e: EndpointBudget(e, 100, expected_used(e)) for e in ENDPOINTS}
    check_before(state, ENDPOINTS, {}, 18)
    resumed = {e: EndpointBudget(e, 100, expected_used(e) + 5) for e in ENDPOINTS}
    check_before(resumed, ENDPOINTS, dict.fromkeys(ENDPOINTS, 5), 18)
    with pytest.raises(PreconditionError):
        check_before(resumed, ENDPOINTS, dict.fromkeys(ENDPOINTS, 4), 18)
