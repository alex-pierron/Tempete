"""Collector of the +10 (then +20) ep answers: 10 frozen arithmetic prompts x 33 endpoints per stage.

READY, NOT RUN. It sends nothing unless ALL of these hold:

* `--yes` is given;
* the authorisation file exists, is valid JSON and says `{"approved": true, "stage": <this stage>,
  "by": "...", "vote_ref": "..."}` (written by the humans' decision, never by this tool);
* the server budget is exactly the state the plan was made on (7 used on every `ep`, 8 on `ep-09`
  .. `ep-12`, plus the answers of this collector already stored);
* the prompts are the frozen ones of `plus10_prompts.json`.

Safety: 33 x 10 jobs per stage, a cap of 10 per endpoint per stage, at most 4 requests in flight,
the client of `main` (no retry after a read timeout, a 429 "requests open" is a busy signal), the
counter check after the run, and the run stops at the first mismatch. Resuming after an
interruption is safe: stored answers are skipped and the precondition counts them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hackathon.common.config import load_settings
from hackathon.model_equality.client import (
    MAX_PARALLEL,
    ApiClient,
    EndpointBudget,
    QueryJob,
    run_jobs,
)
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.store import AnswerStore, default_answers_path

BASE_USED = 7
BASE_USED_OVERSPENT = 8
OVERSPENT_ENDPOINTS: frozenset[str] = frozenset({"ep-09", "ep-10", "ep-11", "ep-12"})
PROMPTS_FILE = Path("reports/phase2/model_equality/team-c/plus10_prompts.json")
AUTHORISATION_FILE = Path("reports/phase2/model_equality/team-c/plus10_authorisation.json")
DEFAULT_PARALLEL = 2
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_PRECONDITION = 3
EXIT_NOT_AUTHORISED = 4


class PreconditionError(RuntimeError):
    """The server budget is not the state the plan was made on."""


class AuthorisationError(RuntimeError):
    """No valid human authorisation for this stage."""


@dataclass(frozen=True)
class AfterCheck:
    """Counter state of one endpoint after a run."""

    endpoint: str
    before: int
    after: int
    stored_in_run: int
    stage_total: int
    stage_cap: int

    @property
    def ok(self) -> bool:
        """Exactly the answers stored in this run were counted, never above the stage cap."""
        return self.after == self.before + self.stored_in_run and self.stage_total <= self.stage_cap


def expected_used(endpoint: str) -> int:
    """Used count before this collector: 7, or 8 on `ep-09` .. `ep-12`."""
    return BASE_USED_OVERSPENT if endpoint in OVERSPENT_ENDPOINTS else BASE_USED


def load_prompts(path: Path, stage: int) -> list[tuple[str, str]]:
    """`(id, text)` of the frozen prompts up to and including `stage` (1 or 2).

    Raises:
        ValueError: stage not in (1, 2).
    """
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = ["stage_1"] if stage == 1 else ["stage_1", "stage_2"]
    return [(row["id"], row["prompt"]) for key in keys for row in payload[key]]


def check_authorisation(path: Path, stage: int) -> dict[str, object]:
    """The human authorisation for this stage.

    Raises:
        AuthorisationError: file missing, unreadable, not approved, or for another stage.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not voted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise AuthorisationError(f"unreadable authorisation file: {exc}") from exc
    if payload.get("approved") is not True or payload.get("stage") != stage:
        raise AuthorisationError(f"authorisation is not an approval of stage {stage}: {payload}")
    if not payload.get("by") or not payload.get("vote_ref"):
        raise AuthorisationError("authorisation must name who approved it and the vote reference")
    return dict(payload)


def stored_per_endpoint(
    store: AnswerStore, endpoints: Sequence[str], prompt_ids: Sequence[str]
) -> dict[str, int]:
    """Stored answers to the frozen prompts, per endpoint."""
    wanted = set(prompt_ids)
    counts = dict.fromkeys(endpoints, 0)
    for answer in store.load().values():
        if answer.prompt_id in wanted and answer.endpoint in counts:
            counts[answer.endpoint] += 1
    return counts


def check_before(
    budgets: Mapping[str, EndpointBudget], endpoints: Sequence[str], stored: Mapping[str, int],
    stage_cap: int,
) -> None:
    """Stop unless every endpoint is exactly at the state the plan was made on.

    Raises:
        PreconditionError: an endpoint is missing, has another `used` value, or already holds
            more answers than the stage cap (all problems listed).
    """
    problems = []
    for endpoint in endpoints:
        info = budgets.get(endpoint)
        expected = expected_used(endpoint) + stored.get(endpoint, 0)
        if info is None:
            problems.append(f"{endpoint}: not listed by the server")
        elif info.used != expected:
            problems.append(f"{endpoint}: used {info.used}, expected {expected}")
        elif stored.get(endpoint, 0) > stage_cap:
            problems.append(f"{endpoint}: {stored[endpoint]} stored, above the cap {stage_cap}")
    if problems:
        raise PreconditionError("; ".join(problems))


def build_jobs(endpoints: Sequence[str], prompts: Sequence[tuple[str, str]]) -> list[QueryJob]:
    """One job per (prompt, endpoint), prompt-major (a stop leaves complete prompts)."""
    return [QueryJob(endpoint, pid, 0, text) for pid, text in prompts for endpoint in endpoints]


def check_after(
    before: Mapping[str, EndpointBudget], after: Mapping[str, EndpointBudget],
    stored_now: Mapping[str, int], stored_before: Mapping[str, int], stage_cap: int,
) -> list[AfterCheck]:
    """Per-endpoint consistency between the server counter and the answers stored in this run."""
    return [
        AfterCheck(e, before[e].used, after[e].used, stored_now[e] - stored_before.get(e, 0),
                   stored_now[e], stage_cap)
        for e in sorted(stored_now)
    ]


def run(stage: int, parallel: int, yes: bool, answers: Path | None) -> int:
    """Dry-run by default; with `--yes` and a valid authorisation, collect the stage."""
    endpoints = list(SCORED_ENDPOINTS)
    prompts = load_prompts(PROMPTS_FILE, stage)
    stage_cap = len(prompts)
    store = AnswerStore(answers or default_answers_path(reference=False))
    ids = [pid for pid, _ in prompts]
    stored_before = stored_per_endpoint(store, endpoints, ids)
    client = ApiClient(load_settings(), log=print)
    before = client.list_budgets()
    try:
        check_before(before, endpoints, stored_before, stage_cap)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the plan: {exc}")
        return EXIT_PRECONDITION
    jobs = build_jobs(endpoints, prompts)
    todo = sum(stage_cap - stored_before[e] for e in endpoints)
    print(f"budget state OK; stage {stage}: {len(prompts)} prompts x {len(endpoints)} endpoints = "
          f"{len(jobs)} jobs, {todo} still to send, parallel {parallel}")
    if not yes:
        print("DRY RUN: nothing sent (needs --yes and a valid authorisation file).")
        return EXIT_OK
    try:
        authorisation = check_authorisation(AUTHORISATION_FILE, stage)
    except AuthorisationError as exc:
        print(f"REFUSED: {exc}")
        return EXIT_NOT_AUTHORISED
    print(f"authorised by {authorisation['by']} ({authorisation['vote_ref']})")
    caps = {e: stage_cap - stored_before[e] for e in endpoints}
    report = run_jobs(client, store, jobs, caps=caps, max_parallel=parallel, progress=print)
    print("run: " + report.summary())
    for failure in report.failed:
        print(f"FAILED {failure}")
    after = client.list_budgets()
    checks = check_after(before, after, stored_per_endpoint(store, endpoints, ids),
                         stored_before, stage_cap)
    bad = [c for c in checks if not c.ok]
    if bad:
        print(f"BUDGET MISMATCH on {[c.endpoint for c in bad]}: do not query again, report.")
    return EXIT_OK if not report.failed and not bad else EXIT_FAILURE


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be in [1, {MAX_PARALLEL}]")
    sys.exit(run(args.stage, args.parallel, args.yes, args.answers))


if __name__ == "__main__":
    main()
