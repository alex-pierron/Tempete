"""Team C: the v6 long-answer collector is ready and locked behind its own authorisation (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import AuthorisationError
from hackathon.model_equality.research.c_v3_run import IDENTITY_FILE, load_identity
from hackathon.model_equality.research.c_v6_run import (
    ALLOWED_LONG,
    AUTHORISATION_FILE,
    LONG_PROMPTS_FILE,
    build_jobs,
    check_authorisation_v6,
    load_long_prompts,
)

ENDPOINTS = list(SCORED_ENDPOINTS)
REPO_ROOT = Path(__file__).resolve().parents[2]
LONG_PROMPT_COUNT = 22
SERVER_TOKEN_CAP = 256


def test_frozen_long_prompts_are_distinct_and_capped() -> None:
    prompts, max_tokens = load_long_prompts(REPO_ROOT / LONG_PROMPTS_FILE)
    assert len(prompts) == LONG_PROMPT_COUNT and max_tokens == SERVER_TOKEN_CAP
    assert len({pid for pid, _ in prompts}) == LONG_PROMPT_COUNT
    assert len({text for _, text in prompts}) == LONG_PROMPT_COUNT


@pytest.mark.parametrize("long", ALLOWED_LONG)
def test_jobs_are_long_plus_three_per_endpoint(long: int) -> None:
    prompts, long_tokens = load_long_prompts(REPO_ROOT / LONG_PROMPTS_FILE)
    formulations, _, identity_tokens = load_identity(REPO_ROOT / IDENTITY_FILE)
    jobs = build_jobs(ENDPOINTS, prompts[:long], long_tokens, formulations, identity_tokens)
    assert len(jobs) == (long + 3) * len(ENDPOINTS)
    assert all(j.repeat == 0 for j in jobs) and len({j.key for j in jobs}) == len(jobs)
    long_jobs = jobs[: long * len(ENDPOINTS)]
    assert {j.max_tokens for j in long_jobs} == {SERVER_TOKEN_CAP}
    assert {j.prompt_id for j in jobs[long * len(ENDPOINTS):]} == {
        "d-ident-000", "d-ident-001", "d-ident-002"}


def test_authorisation_names_long_and_is_separate_from_the_other_collectors(tmp_path: Path) -> None:
    assert not (REPO_ROOT / AUTHORISATION_FILE).exists()  # never created by the tool
    with pytest.raises(AuthorisationError):
        check_authorisation_v6(tmp_path / "missing.json")
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"approved": True, "stage": "v4", "long": 14, "by": "V",
                                "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):  # the v4 authorisation does not open v6
        check_authorisation_v6(path)
    path.write_text(json.dumps({"approved": True, "stage": "v6", "long": 13, "by": "V",
                                "vote_ref": "x"}))
    with pytest.raises(AuthorisationError):  # L must be one of the frozen sizes
        check_authorisation_v6(path)
    path.write_text(json.dumps({"approved": True, "stage": "v6", "long": 14, "by": "V"}))
    with pytest.raises(AuthorisationError):
        check_authorisation_v6(path)
    path.write_text(json.dumps({"approved": True, "stage": "v6", "long": 14, "by": "V",
                                "vote_ref": "decision"}))
    assert check_authorisation_v6(path)["long"] == 14
