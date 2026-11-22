"""Team next: the v7 tranche collector is ready and locked behind its own authorisation (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hackathon.model_equality.client import EndpointBudget
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import (
    OVERSPENT_ENDPOINTS,
    AuthorisationError,
    PreconditionError,
)
from hackathon.model_equality.research.c_v6_run import (
    LONG_PROMPTS_FILE as V6_PROMPTS_FILE,
)
from hackathon.model_equality.research.c_v6_run import (
    load_long_prompts,
)
from hackathon.model_equality.research.n_v7_run import (
    ALLOWED_TRANCHES,
    AUTHORISATION_FILE,
    BUDGET_OFFSETS_FILE,
    LONG_PROMPTS_FILE,
    TRANCHE_SIZE,
    build_jobs,
    check_authorisation_v7,
    check_before_v7,
    expected_server_used,
    load_offsets,
    load_plan,
    prompts_up_to,
    run,
)

ENDPOINTS = list(SCORED_ENDPOINTS)
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_TOKEN_CAP = 256
V6_LONG_USED = 12
SERVER_LIMIT = 100
STORED_AFTER_V6 = 22


def test_plan_is_three_distinct_tranches_that_never_repeat_a_v6_prompt() -> None:
    plan = load_plan(REPO_ROOT / LONG_PROMPTS_FILE)
    assert plan.max_tokens == SERVER_TOKEN_CAP
    v6_prompts, _ = load_long_prompts(REPO_ROOT / V6_PROMPTS_FILE)
    spent = v6_prompts[:V6_LONG_USED]
    unused = dict(v6_prompts[V6_LONG_USED:])
    everything = prompts_up_to(plan, max(ALLOWED_TRANCHES))
    assert not {p.id for p in everything} & {pid for pid, _ in spent}
    assert not {p.prompt for p in everything} & {text for _, text in spent}
    # The unused v6 prompts keep their id and text, so the two collectors can never disagree.
    reused = {p.id: p.prompt for p in everything if p.id in unused}
    assert reused == unused


@pytest.mark.parametrize("tranche", ALLOWED_TRANCHES)
def test_jobs_are_twelve_per_tranche_and_prompt_major(tranche: int) -> None:
    plan = load_plan(REPO_ROOT / LONG_PROMPTS_FILE)
    prompts = prompts_up_to(plan, tranche)
    jobs = build_jobs(ENDPOINTS, prompts, plan.max_tokens)
    assert len(jobs) == tranche * TRANCHE_SIZE * len(ENDPOINTS)
    assert len({j.key for j in jobs}) == len(jobs)
    assert {j.max_tokens for j in jobs} == {SERVER_TOKEN_CAP}
    first_round = jobs[: len(ENDPOINTS)]
    assert {j.prompt_id for j in first_round} == {prompts[0].id}
    assert [j.endpoint for j in first_round] == ENDPOINTS


def test_tranche_out_of_range_is_rejected() -> None:
    plan = load_plan(REPO_ROOT / LONG_PROMPTS_FILE)
    with pytest.raises(ValueError):
        prompts_up_to(plan, 0)
    with pytest.raises(ValueError):
        prompts_up_to(plan, max(ALLOWED_TRANCHES) + 1)


def test_authorisation_is_separate_signed_and_bounds_the_tranche(tmp_path: Path) -> None:
    assert not (REPO_ROOT / AUTHORISATION_FILE).exists()  # never created by the tool
    with pytest.raises(AuthorisationError):
        check_authorisation_v7(tmp_path / "missing.json", 1)
    path = tmp_path / "auth.json"
    signed = {"by": "V", "vote_ref": "decision"}
    path.write_text(json.dumps({"approved": True, "stage": "v6", "long": 12, **signed}))
    with pytest.raises(AuthorisationError):  # the v6 authorisation does not open v7
        check_authorisation_v7(path, 1)
    path.write_text(json.dumps({"approved": False, "stage": "v7", "tranches": 3, **signed}))
    with pytest.raises(AuthorisationError):
        check_authorisation_v7(path, 1)
    path.write_text(json.dumps({"approved": True, "stage": "v7", "tranches": 4, **signed}))
    with pytest.raises(AuthorisationError):
        check_authorisation_v7(path, 1)
    path.write_text(json.dumps({"approved": True, "stage": "v7", "tranches": 2, "by": "V"}))
    with pytest.raises(AuthorisationError):  # unsigned decision
        check_authorisation_v7(path, 1)
    path.write_text(json.dumps({"approved": True, "stage": "v7", "tranches": 2, **signed}))
    assert check_authorisation_v7(path, 2).tranches == 2
    with pytest.raises(AuthorisationError):  # a third tranche needs a new decision
        check_authorisation_v7(path, 3)


def _budgets(used: dict[str, int]) -> dict[str, EndpointBudget]:
    return {e: EndpointBudget(e, SERVER_LIMIT, n) for e, n in used.items()}


def test_precondition_follows_the_store_and_the_known_losses() -> None:
    stored_total = dict.fromkeys(ENDPOINTS, STORED_AFTER_V6)
    expected = expected_server_used(ENDPOINTS, stored_total, {"ep-01": 1})
    assert expected["ep-02"] == STORED_AFTER_V6
    assert expected["ep-01"] == STORED_AFTER_V6 + 1
    assert all(expected[e] == STORED_AFTER_V6 + 1 for e in OVERSPENT_ENDPOINTS)
    none_stored = dict.fromkeys(ENDPOINTS, 0)
    check_before_v7(_budgets(expected), expected, none_stored, TRANCHE_SIZE)
    drifted = dict(expected, **{"ep-05": expected["ep-05"] + 1})
    with pytest.raises(PreconditionError, match="ep-05: used"):
        check_before_v7(_budgets(drifted), expected, none_stored, TRANCHE_SIZE)
    missing = {e: n for e, n in expected.items() if e != "ep-07"}
    with pytest.raises(PreconditionError, match="ep-07: not listed"):
        check_before_v7(_budgets(missing), expected, none_stored, TRANCHE_SIZE)


def test_precondition_refuses_a_stage_larger_than_what_is_left() -> None:
    nearly_spent = dict.fromkeys(ENDPOINTS, SERVER_LIMIT - TRANCHE_SIZE + 1)
    with pytest.raises(PreconditionError, match="only 11 left"):
        check_before_v7(_budgets(nearly_spent), nearly_spent, dict.fromkeys(ENDPOINTS, 0),
                        TRANCHE_SIZE)


def test_offsets_file_is_optional_and_validated(tmp_path: Path) -> None:
    assert not (REPO_ROOT / BUDGET_OFFSETS_FILE).exists()
    assert load_offsets(tmp_path / "missing.json") == {}
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"offsets": {"ep-03": 1}, "by": "V", "reason": "read timeout"}))
    assert load_offsets(path) == {"ep-03": 1}
    path.write_text(json.dumps({"offsets": {"ref-01": 1}, "by": "V", "reason": "x"}))
    with pytest.raises(ValueError):
        load_offsets(path)
    path.write_text(json.dumps({"offsets": {"ep-03": -1}, "by": "V", "reason": "x"}))
    with pytest.raises(ValueError):
        load_offsets(path)


def test_offline_dry_run_contacts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    code = run(1, 4, yes=False, answers=tmp_path / "answers.jsonl",
               stop_file=tmp_path / "STOP", offline=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "396 jobs, 396 still to send" in out and "OFFLINE DRY RUN" in out
