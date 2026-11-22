"""Collector of Lucas's plan (tour 1 and iteration rounds), in checked tranches.

Generalises `n_v7_run` (same client, same store, same stop file mechanics) to a prompt file per
round: `round<k>_prompts.json` with `{plan, round, purpose, frozen_at_utc, max_tokens, prompts}`,
each prompt `{id, prompt, repeats, first_repeat, endpoints}` (`endpoints: null` = all 33).

Budget rule of the plan (Victor's authorisation of 22/11/2026, 2 h 35), enforced per endpoint and
re-checked against the SERVER COUNTER BEFORE EVERY TRANCHE:
  - answers spent by this plan (server `used` minus the `baseline_used` of the authorisation)
    never exceed `max_plan_answers` (55);
  - the server never falls under `min_left` (22) answers left.
A tranche that would break either rule on any endpoint is not sent at all, and the run stops.
A round holds at most `ROUND_CAPS[round]` answers per endpoint (15 for tour 1, 8 afterwards).

Locked by `lucas_plan_authorisation.json` (written by hand from the humans' decision, never by
this tool): nothing is sent without `--yes` and a valid file. Graceful stop: create the stop file
(in-flight requests finish, nothing new is sent); never kill the process (abandoned requests keep
the key's 4 slots busy for about 5 minutes). Resume: same command, stored keys are skipped.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hackathon.common.config import load_settings
from hackathon.model_equality.client import (
    MAX_PARALLEL,
    EndpointBudget,
    FatalQueryError,
    QueryJob,
    run_jobs,
)
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import (
    EXIT_FAILURE,
    EXIT_NOT_AUTHORISED,
    EXIT_OK,
    EXIT_PRECONDITION,
    AuthorisationError,
    PreconditionError,
)
from hackathon.model_equality.research.d_client import StoppableClient
from hackathon.model_equality.research.n_v7_run import (
    expected_server_used,
    load_offsets,
    stored_total_per_endpoint,
)
from hackathon.model_equality.store import AnswerStore, default_answers_path

PLAN_DIR = Path("reports/phase3/model_equality/team-lucas")
AUTHORISATION_FILE = PLAN_DIR / "lucas_plan_authorisation.json"
BUDGET_OFFSETS_FILE = PLAN_DIR / "lucas_budget_offsets.json"
PLAN_NAME = "lucas-met"
FIRST_ROUND = 1
LAST_ROUND = 25
"""Rounds 7-25: W-MET targeted tours (system-prompt extraction on Llama, long answers for the
likelihood test), still bounded by the per-endpoint plan caps of the authorisation."""
TOUR1_CAP_PER_ENDPOINT = 15
ITERATION_CAP_PER_ENDPOINT = 8
ROUND_CAPS: dict[int, int] = {
    r: (TOUR1_CAP_PER_ENDPOINT if r == FIRST_ROUND else ITERATION_CAP_PER_ENDPOINT)
    for r in range(FIRST_ROUND, LAST_ROUND + 1)
}
DEFAULT_PARALLEL = 4
DEFAULT_TRANCHE_PROMPTS = 2
DEFAULT_STOP_FILE = Path("/tmp/xmc/STOP_LUCAS_PLAN")
EXIT_CAP = 5


class PlanPrompt(BaseModel):
    """One prompt of a round, asked `repeats` times (repeat indices from `first_repeat`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    repeats: int = Field(default=1, ge=1)
    first_repeat: int = Field(default=0, ge=0)
    endpoints: list[str] | None = None
    # Informational fields written by the planner (x-met-relect); never used to send. Declared
    # rather than ignored so that a misspelt `endpoints` still fails instead of meaning "all 33".
    expected_gap_nats: float | None = None
    families: list[str] | None = None


class RoundPlan(BaseModel):
    """A frozen round file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: str
    round: int = Field(ge=FIRST_ROUND, le=LAST_ROUND)
    purpose: str
    frozen_at_utc: str
    max_tokens: int = Field(gt=0)
    prompts: list[PlanPrompt] = Field(min_length=1)


class PlanAuthorisation(BaseModel):
    """The human decision that opens the plan, with its budget rule."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    approved: bool
    plan: str
    max_plan_answers: int = Field(gt=0)
    min_left: int = Field(ge=0)
    baseline_used: dict[str, int]
    by: str = Field(min_length=1)
    vote_ref: str = Field(min_length=1)


class PlanCapError(RuntimeError):
    """A tranche would break the plan's budget rule on at least one endpoint."""


def load_round(path: Path) -> RoundPlan:
    """The frozen round file.

    Raises:
        ValueError: wrong plan name, unknown endpoint, duplicated (endpoint, id, repeat) key, or
            more answers per endpoint than the round cap.
    """
    plan = RoundPlan.model_validate_json(path.read_text(encoding="utf-8"))
    if plan.plan != PLAN_NAME:
        raise ValueError(f"round file of plan {plan.plan!r}, expected {PLAN_NAME!r}")
    jobs = build_jobs(plan)
    unknown = sorted({j.endpoint for j in jobs} - set(SCORED_ENDPOINTS))
    if unknown:
        raise ValueError(f"unknown endpoints in round file: {unknown}")
    if len({j.key for j in jobs}) != len(jobs):
        raise ValueError("duplicated (endpoint, id, repeat) key in round file")
    cap = ROUND_CAPS[plan.round]
    over = {e: n for e, n in jobs_per_endpoint(jobs).items() if n > cap}
    if over:
        raise ValueError(f"round {plan.round} allows {cap} answers per endpoint: {over}")
    return plan


def build_jobs(plan: RoundPlan) -> list[QueryJob]:
    """Jobs in prompt-major order (a stop leaves the endpoints level)."""
    return [
        QueryJob(endpoint, prompt.id, repeat, prompt.prompt, plan.max_tokens)
        for prompt in plan.prompts
        for repeat in range(prompt.first_repeat, prompt.first_repeat + prompt.repeats)
        for endpoint in (prompt.endpoints if prompt.endpoints is not None else SCORED_ENDPOINTS)
    ]


def jobs_per_endpoint(jobs: Sequence[QueryJob]) -> dict[str, int]:
    """Number of jobs per endpoint."""
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.endpoint] = counts.get(job.endpoint, 0) + 1
    return counts


def split_tranches(jobs: Sequence[QueryJob], prompts_per_tranche: int) -> list[list[QueryJob]]:
    """Consecutive tranches of `prompts_per_tranche` distinct (id, repeat) pairs.

    Raises:
        ValueError: `prompts_per_tranche` < 1.
    """
    if prompts_per_tranche < 1:
        raise ValueError(f"prompts_per_tranche must be >= 1, got {prompts_per_tranche}")
    tranches: list[list[QueryJob]] = []
    seen: list[tuple[str, int]] = []
    for job in jobs:
        tag = (job.prompt_id, job.repeat)
        if tag not in seen:
            seen.append(tag)
            if (len(seen) - 1) % prompts_per_tranche == 0:
                tranches.append([])
        tranches[-1].append(job)
    return tranches


def check_authorisation(path: Path) -> PlanAuthorisation:
    """The human authorisation of the plan.

    Raises:
        AuthorisationError: file missing, unreadable, not an approval of this plan, or without
            a baseline for every scored endpoint.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not decided")
    try:
        authorisation = PlanAuthorisation.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise AuthorisationError(f"invalid authorisation file: {exc}") from exc
    if not authorisation.approved or authorisation.plan != PLAN_NAME:
        raise AuthorisationError(f"authorisation is not an approval of plan {PLAN_NAME!r}")
    missing = sorted(set(SCORED_ENDPOINTS) - set(authorisation.baseline_used))
    if missing:
        raise AuthorisationError(f"authorisation has no baseline_used for {missing}")
    return authorisation


def check_counters(budgets: Mapping[str, EndpointBudget], expected: Mapping[str, int]) -> None:
    """Stop unless every server counter is exactly what the store implies.

    Raises:
        PreconditionError: an endpoint is missing or its counter differs (all listed).
    """
    problems = []
    for endpoint, want in expected.items():
        info = budgets.get(endpoint)
        if info is None:
            problems.append(f"{endpoint}: not listed by the server")
        elif info.used != want:
            problems.append(f"{endpoint}: used {info.used}, expected {want}")
    if problems:
        raise PreconditionError("; ".join(problems))


def check_plan_caps(
    budgets: Mapping[str, EndpointBudget], to_send: Mapping[str, int],
    authorisation: PlanAuthorisation,
) -> None:
    """Refuse a tranche that would break the plan's budget rule on any endpoint.

    Raises:
        PlanCapError: plan answers above `max_plan_answers` or server left under `min_left`.
    """
    problems = []
    for endpoint, count in to_send.items():
        if count == 0:
            continue
        info = budgets[endpoint]
        spent_after = info.used - authorisation.baseline_used[endpoint] + count
        if spent_after > authorisation.max_plan_answers:
            problems.append(f"{endpoint}: plan would reach {spent_after} > "
                            f"{authorisation.max_plan_answers}")
        if info.left is not None and info.left - count < authorisation.min_left:
            problems.append(f"{endpoint}: left would be {info.left - count} < "
                            f"{authorisation.min_left}")
    if problems:
        raise PlanCapError("; ".join(problems))


def pending_per_endpoint(store: AnswerStore, jobs: Sequence[QueryJob]) -> dict[str, int]:
    """Jobs of `jobs` not yet stored, per endpoint."""
    known = store.keys()
    return jobs_per_endpoint([j for j in jobs if j.key not in known])


def run(round_file: Path, parallel: int, tranche_prompts: int, yes: bool, answers: Path | None,
        stop_file: Path, offline: bool) -> int:
    """Dry-run by default; with `--yes` and a valid authorisation, collect the round."""
    endpoints = list(SCORED_ENDPOINTS)
    plan = load_round(round_file)
    jobs = build_jobs(plan)
    tranches = split_tranches(jobs, tranche_prompts)
    store = AnswerStore(answers or default_answers_path(reference=False))
    todo = sum(pending_per_endpoint(store, jobs).values())
    print(f"store={store.path}; round {plan.round}: {len(jobs)} jobs in {len(tranches)} tranches, "
          f"{todo} still to send, parallel {parallel}")
    if offline:
        print("OFFLINE DRY RUN: server not contacted, nothing sent.")
        return EXIT_OK
    try:
        authorisation = check_authorisation(AUTHORISATION_FILE)
    except AuthorisationError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_NOT_AUTHORISED
    offsets = load_offsets(BUDGET_OFFSETS_FILE)
    client = StoppableClient(load_settings(), log=print, stop_file=stop_file)
    for index, tranche in enumerate(tranches, start=1):
        budgets = client.list_budgets()
        expected = expected_server_used(
            endpoints, stored_total_per_endpoint(store, endpoints), offsets)
        to_send = pending_per_endpoint(store, tranche)
        try:
            check_counters(budgets, expected)
            check_plan_caps(budgets, to_send, authorisation)
        except PreconditionError as exc:
            print(f"STOP before tranche {index}, budget differs from the store: {exc}")
            return EXIT_PRECONDITION
        except PlanCapError as exc:
            print(f"STOP before tranche {index}, plan cap: {exc}")
            return EXIT_CAP
        spent = {e: budgets[e].used - authorisation.baseline_used[e] for e in endpoints}
        print(f"tranche {index}/{len(tranches)}: counters OK, plan spent max {max(spent.values())}"
              f" min {min(spent.values())}, left min {min(b.left or 0 for b in budgets.values())},"
              f" {sum(to_send.values())} to send")
        if not yes:
            print("DRY RUN: nothing sent (needs --yes).")
            return EXIT_OK
        if stop_file.exists():
            print(f"STOP file {stop_file} present: nothing more sent.")
            return EXIT_OK
        stored_before = stored_total_per_endpoint(store, endpoints)
        try:
            report = run_jobs(client, store, tranche, caps=to_send, max_parallel=parallel,
                              progress=print)
            print(f"tranche {index}: {report.summary()}")
            failed = report.failed
        except FatalQueryError as exc:
            print(f"tranche {index}: STOPPED early: {exc}")
            failed = [str(exc)]
        for failure in failed:
            print(f"FAILED {failure}")
        after = client.list_budgets()
        stored_after = stored_total_per_endpoint(store, endpoints)
        bad = [e for e in endpoints if after[e].used - budgets[e].used
               != stored_after[e] - stored_before[e]]
        if bad:
            print(f"BUDGET MISMATCH on {bad}: do not query again, report.")
            return EXIT_FAILURE
        if failed:
            return EXIT_FAILURE
    final = client.list_budgets()
    print("done: plan spent per endpoint "
          + " ".join(f"{e}={final[e].used - authorisation.baseline_used[e]}" for e in endpoints))
    return EXIT_OK


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("round_file", type=Path)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--tranche-prompts", type=int, default=DEFAULT_TRANCHE_PROMPTS)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_FILE)
    parser.add_argument("--offline", action="store_true",
                        help="dry run that does not even read the server counters")
    args = parser.parse_args()
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be in [1, {MAX_PARALLEL}]")
    if args.offline and args.yes:
        parser.error("--offline and --yes are exclusive")
    sys.exit(run(args.round_file, args.parallel, args.tranche_prompts, args.yes, args.answers,
                 args.stop_file, args.offline))


if __name__ == "__main__":
    main()
