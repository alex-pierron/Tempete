"""Collector of the ep answers of configuration v7: more long answers, in tranches of 12.

READY, NOT RUN. Locked with its own authorisation file, `long_v7_authorisation.json`:
`{"approved": true, "stage": "v7", "tranches": 1 | 2 | 3, "by": "...", "vote_ref": "..."}`,
written from the humans' decision, never by this tool. It sends nothing unless `--yes` is given
AND that file is valid AND `--tranche` does not exceed the authorised number of tranches.

Per endpoint and per tranche: the 12 frozen long prompts of that tranche in
`long_v7_prompts.json` (user message only, 256 tokens at most), one draw each, that is 396
requests per tranche for 33 endpoints. `--tranche k` covers tranches 1 to k: answers already
stored are skipped, so resuming or extending is safe.

Why tranches and prompt-major order: the stop rule voted with v7 may end the collection between
two submissions; with one prompt finished on every endpoint before the next one starts, a stop
leaves the 33 endpoints at the same number of tokens (within one answer).

Safety, as `c_v6_run`: exact server-counter precondition (every answer of the store, plus the
known lost answers, must equal the server's `used`), a cap per endpoint, at most 4 requests in
flight, no retry after a read timeout, counter check after the run. Graceful stop: create the
stop file (in-flight requests finish, nothing new is sent); never kill the process.
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
    BASE_USED,
    EXIT_FAILURE,
    EXIT_NOT_AUTHORISED,
    EXIT_OK,
    EXIT_PRECONDITION,
    AuthorisationError,
    PreconditionError,
    check_after,
    expected_used,
    stored_per_endpoint,
)
from hackathon.model_equality.research.d_client import StoppableClient
from hackathon.model_equality.store import AnswerStore, default_answers_path

TEAM_DIR = Path("reports/phase3/model_equality/team-next")
LONG_PROMPTS_FILE = TEAM_DIR / "long_v7_prompts.json"
AUTHORISATION_FILE = TEAM_DIR / "long_v7_authorisation.json"
BUDGET_OFFSETS_FILE = TEAM_DIR / "long_v7_budget_offsets.json"
STAGE_NAME = "v7"
TRANCHE_SIZE = 12
TRANCHE_COUNT = 3
ALLOWED_TRANCHES: tuple[int, ...] = tuple(range(1, TRANCHE_COUNT + 1))
DEFAULT_PARALLEL = 4
DEFAULT_STOP_FILE = Path("/tmp/n_scratch/STOP_V7")


class LongPrompt(BaseModel):
    """One frozen long prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class LongPromptPlan(BaseModel):
    """The frozen v7 prompt file: three tranches of twelve prompts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    purpose: str
    frozen_at_utc: str
    max_tokens: int = Field(gt=0)
    tranches: list[list[LongPrompt]]


class Authorisation(BaseModel):
    """The human decision that opens v7 up to a number of tranches."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    approved: bool
    stage: str
    tranches: int
    by: str = Field(min_length=1)
    vote_ref: str = Field(min_length=1)


class BudgetOffsets(BaseModel):
    """Answers counted by the server but absent from the store (lost to a read timeout).

    Written by hand after a `BUDGET MISMATCH`, once a human has accepted the loss: without it the
    precondition would refuse every later run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    offsets: dict[str, int]
    by: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def load_plan(path: Path) -> LongPromptPlan:
    """The frozen prompt plan.

    Raises:
        ValueError: not `TRANCHE_COUNT` tranches of `TRANCHE_SIZE` prompts, or a duplicated id
            or text (a duplicate would silently collect fewer answers than voted).
    """
    plan = LongPromptPlan.model_validate_json(path.read_text(encoding="utf-8"))
    sizes = [len(tranche) for tranche in plan.tranches]
    if sizes != [TRANCHE_SIZE] * TRANCHE_COUNT:
        raise ValueError(f"expected {TRANCHE_COUNT} tranches of {TRANCHE_SIZE} prompts, got {sizes}")
    prompts = [prompt for tranche in plan.tranches for prompt in tranche]
    if len({p.id for p in prompts}) != len(prompts) or len({p.prompt for p in prompts}) != len(prompts):
        raise ValueError("duplicated prompt id or text in the v7 plan")
    return plan


def prompts_up_to(plan: LongPromptPlan, tranche: int) -> list[LongPrompt]:
    """Prompts of tranches 1 to `tranche`, in collection order.

    Raises:
        ValueError: `tranche` out of range.
    """
    if tranche not in ALLOWED_TRANCHES:
        raise ValueError(f"tranche must be in {ALLOWED_TRANCHES}, got {tranche}")
    return [prompt for block in plan.tranches[:tranche] for prompt in block]


def check_authorisation_v7(path: Path, tranche: int) -> Authorisation:
    """The human authorisation of v7, valid for the requested tranche.

    Raises:
        AuthorisationError: file missing, unreadable, not an approval of stage "v7", unsigned,
            or authorising fewer tranches than requested.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not decided")
    try:
        authorisation = Authorisation.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise AuthorisationError(f"invalid authorisation file: {exc}") from exc
    if not authorisation.approved or authorisation.stage != STAGE_NAME:
        raise AuthorisationError(f"authorisation is not an approval of stage {STAGE_NAME!r}")
    if authorisation.tranches not in ALLOWED_TRANCHES:
        raise AuthorisationError(f"authorisation must choose tranches in {ALLOWED_TRANCHES}")
    if tranche > authorisation.tranches:
        raise AuthorisationError(
            f"tranche {tranche} requested, only {authorisation.tranches} authorised")
    return authorisation


def load_offsets(path: Path) -> dict[str, int]:
    """Accepted lost answers per endpoint (empty when the file does not exist).

    Raises:
        ValueError: a negative offset or an endpoint that is not scored.
    """
    if not path.is_file():
        return {}
    offsets = BudgetOffsets.model_validate_json(path.read_text(encoding="utf-8")).offsets
    bad = {e: n for e, n in offsets.items() if n < 0 or e not in SCORED_ENDPOINTS}
    if bad:
        raise ValueError(f"invalid budget offsets: {bad}")
    return offsets


def stored_total_per_endpoint(store: AnswerStore, endpoints: Sequence[str]) -> dict[str, int]:
    """Every stored answer per endpoint, whatever the collector that fetched it."""
    counts = dict.fromkeys(endpoints, 0)
    for answer in store.load().values():
        if answer.endpoint in counts:
            counts[answer.endpoint] += 1
    return counts


def expected_server_used(
    endpoints: Sequence[str], stored_total: Mapping[str, int], offsets: Mapping[str, int]
) -> dict[str, int]:
    """Server counter implied by the store: stored answers + known lost answers.

    The historical loss (one answer on `ep-09` .. `ep-12`) is `expected_used - BASE_USED`.
    """
    return {
        e: stored_total[e] + (expected_used(e) - BASE_USED) + offsets.get(e, 0) for e in endpoints
    }


def check_before_v7(
    budgets: Mapping[str, EndpointBudget], expected: Mapping[str, int],
    stored_stage: Mapping[str, int], stage_cap: int,
) -> None:
    """Stop unless the server counters are exactly what the store implies.

    Raises:
        PreconditionError: an endpoint is missing, its counter differs, it holds more answers
            than the stage cap, or the stage would exceed its server limit (all problems listed).
    """
    problems = []
    for endpoint, want in expected.items():
        info = budgets.get(endpoint)
        remaining = stage_cap - stored_stage.get(endpoint, 0)
        if info is None:
            problems.append(f"{endpoint}: not listed by the server")
        elif info.used != want:
            problems.append(f"{endpoint}: used {info.used}, expected {want}")
        elif remaining < 0:
            problems.append(f"{endpoint}: {stored_stage[endpoint]} stored, above the cap {stage_cap}")
        elif info.left is not None and remaining > info.left:
            problems.append(f"{endpoint}: {remaining} to send, only {info.left} left")
    if problems:
        raise PreconditionError("; ".join(problems))


def build_jobs(
    endpoints: Sequence[str], prompts: Sequence[LongPrompt], max_tokens: int
) -> list[QueryJob]:
    """One job per (prompt, endpoint), prompt-major (a stop leaves the endpoints level)."""
    return [QueryJob(e, p.id, 0, p.prompt, max_tokens) for p in prompts for e in endpoints]


def run(tranche: int, parallel: int, yes: bool, answers: Path | None, stop_file: Path,
        offline: bool) -> int:
    """Dry-run by default; with `--yes` and a valid authorisation, collect tranches 1..`tranche`."""
    endpoints = list(SCORED_ENDPOINTS)
    plan = load_plan(LONG_PROMPTS_FILE)
    prompts = prompts_up_to(plan, tranche)
    ids = [p.id for p in prompts]
    stage_cap = len(prompts)
    store = AnswerStore(answers or default_answers_path(reference=False))
    stored_before = stored_per_endpoint(store, endpoints, ids)
    jobs = build_jobs(endpoints, prompts, plan.max_tokens)
    todo = sum(max(stage_cap - stored_before[e], 0) for e in endpoints)
    print(f"store={store.path}; v7 tranches 1..{tranche}: {stage_cap} answers per endpoint x "
          f"{len(endpoints)} = {len(jobs)} jobs, {todo} still to send, parallel {parallel}")
    if offline:
        print("OFFLINE DRY RUN: server not contacted, nothing sent.")
        return EXIT_OK
    authorisation: Authorisation | None = None
    if yes:
        try:
            authorisation = check_authorisation_v7(AUTHORISATION_FILE, tranche)
        except AuthorisationError as exc:
            print(f"REFUSED: {exc}")
            return EXIT_NOT_AUTHORISED
    client = StoppableClient(load_settings(), log=print, stop_file=stop_file)
    before = client.list_budgets()
    expected = expected_server_used(
        endpoints, stored_total_per_endpoint(store, endpoints), load_offsets(BUDGET_OFFSETS_FILE))
    try:
        check_before_v7(before, expected, stored_before, stage_cap)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the store: {exc}")
        return EXIT_PRECONDITION
    print("budget state OK")
    if authorisation is None:
        print("DRY RUN: nothing sent (needs --yes and a valid authorisation file).")
        return EXIT_OK
    print(f"authorised by {authorisation.by} ({authorisation.vote_ref}), "
          f"{authorisation.tranches} tranche(s); stop file {stop_file}")
    caps = {e: stage_cap - stored_before[e] for e in endpoints}
    failed: list[str] = []
    try:
        report = run_jobs(client, store, jobs, caps=caps, max_parallel=parallel, progress=print)
        print("run: " + report.summary())
        failed = report.failed
    except FatalQueryError as exc:
        # The stop file (or an authentication failure) ended the run: the counters are still
        # checked below, because answers were stored before the stop.
        print(f"run: STOPPED early: {exc}")
    for failure in failed:
        print(f"FAILED {failure}")
    after = client.list_budgets()
    checks = check_after(before, after, stored_per_endpoint(store, endpoints, ids), stored_before,
                         stage_cap)
    bad = [c for c in checks if not c.ok]
    if bad:
        print(f"BUDGET MISMATCH on {[c.endpoint for c in bad]}: do not query again, report.")
    return EXIT_OK if not failed and not bad else EXIT_FAILURE


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--tranche", type=int, choices=ALLOWED_TRANCHES, required=True,
                        help="collect tranches 1 to this one")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
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
    sys.exit(run(args.tranche, args.parallel, args.yes, args.answers, args.stop_file, args.offline))


if __name__ == "__main__":
    main()
