"""Collector of the ep answers of rule v3: 10 arithmetic prompts + 6 identity questions per endpoint.

READY, NOT RUN. Locked like `c_plus10_run`, with its OWN authorisation file: it sends nothing
unless `--yes` is given AND `plus10_v3_authorisation.json` exists with
`{"approved": true, "stage": "v3", "by": "...", "vote_ref": "..."}`, written by the humans' vote,
never by this tool. Per endpoint: the 10 frozen arithmetic prompts of `plus10_prompts.json`
(stage 1) and the 3 frozen identity formulations of `identity_prompts.json` x 2 draws = 16 answers
(16 % of the budget, 528 requests for 33 endpoints). Same safety as `c_plus10_run`: exact
server-counter precondition (7 used, 8 on ep-09..12, plus the answers already stored), cap of 16
per endpoint, at most 4 requests in flight, no retry after a read timeout, counter check after the
run, stop at the first mismatch; resuming is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hackathon.common.config import load_settings
from hackathon.model_equality.client import MAX_PARALLEL, ApiClient, QueryJob, run_jobs
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.research.c_plus10_run import (
    DEFAULT_PARALLEL,
    EXIT_FAILURE,
    EXIT_NOT_AUTHORISED,
    EXIT_OK,
    EXIT_PRECONDITION,
    PROMPTS_FILE,
    AuthorisationError,
    PreconditionError,
    check_after,
    check_before,
    load_prompts,
    stored_per_endpoint,
)
from hackathon.model_equality.store import AnswerStore, default_answers_path

IDENTITY_FILE = Path("reports/phase2/model_equality/team-c/identity_prompts.json")
AUTHORISATION_FILE = Path("reports/phase2/model_equality/team-c/plus10_v3_authorisation.json")
STAGE_NAME = "v3"
ARITHMETIC_PROMPTS = 10
IDENTITY_UNITS = 6


def load_identity(path: Path) -> tuple[list[tuple[str, str]], int, int]:
    """`(formulations, draws per formulation, max tokens)` of the frozen identity file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    formulations = [(row["id"], row["prompt"]) for row in payload["formulations"]]
    return formulations, int(payload["draws_per_formulation_on_ep"]), int(payload["max_tokens"])


def check_authorisation_v3(path: Path) -> dict[str, object]:
    """The human authorisation of rule v3.

    Raises:
        AuthorisationError: file missing, unreadable, not an approval of stage "v3", or unsigned.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not voted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise AuthorisationError(f"unreadable authorisation file: {exc}") from exc
    if payload.get("approved") is not True or payload.get("stage") != STAGE_NAME:
        raise AuthorisationError(f"authorisation is not an approval of stage {STAGE_NAME!r}: {payload}")
    if not payload.get("by") or not payload.get("vote_ref"):
        raise AuthorisationError("authorisation must name who approved it and the vote reference")
    return dict(payload)


def build_jobs(
    endpoints: Sequence[str], arithmetic: Sequence[tuple[str, str]],
    formulations: Sequence[tuple[str, str]], draws: int, max_tokens: int,
) -> list[QueryJob]:
    """Arithmetic prompts first (prompt-major), then identity units draw by draw."""
    jobs = [QueryJob(e, pid, 0, text) for pid, text in arithmetic for e in endpoints]
    for draw in range(draws):
        jobs += [QueryJob(e, pid, draw, text, max_tokens)
                 for pid, text in formulations for e in endpoints]
    return jobs


def run(parallel: int, yes: bool, answers: Path | None) -> int:
    """Dry-run by default; with `--yes` and a valid authorisation, collect the v3 answers."""
    endpoints = list(SCORED_ENDPOINTS)
    arithmetic = load_prompts(PROMPTS_FILE, 1)[:ARITHMETIC_PROMPTS]
    formulations, draws, max_tokens = load_identity(IDENTITY_FILE)
    ids = [pid for pid, _ in arithmetic] + [pid for pid, _ in formulations]
    stage_cap = len(arithmetic) + len(formulations) * draws
    store = AnswerStore(answers or default_answers_path(reference=False))
    stored_before = stored_per_endpoint(store, endpoints, ids)
    client = ApiClient(load_settings(), log=print)
    before = client.list_budgets()
    try:
        check_before(before, endpoints, stored_before, stage_cap)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the plan: {exc}")
        return EXIT_PRECONDITION
    jobs = build_jobs(endpoints, arithmetic, formulations, draws, max_tokens)
    todo = sum(stage_cap - stored_before[e] for e in endpoints)
    print(f"budget state OK; v3: {stage_cap} answers per endpoint x {len(endpoints)} = {len(jobs)} "
          f"jobs, {todo} still to send, parallel {parallel}")
    if not yes:
        print("DRY RUN: nothing sent (needs --yes and a valid authorisation file).")
        return EXIT_OK
    try:
        authorisation = check_authorisation_v3(AUTHORISATION_FILE)
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
    checks = check_after(before, after, stored_per_endpoint(store, endpoints, ids), stored_before,
                         stage_cap)
    bad = [c for c in checks if not c.ok]
    if bad:
        print(f"BUDGET MISMATCH on {[c.endpoint for c in bad]}: do not query again, report.")
    return EXIT_OK if not report.failed and not bad else EXIT_FAILURE


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be in [1, {MAX_PARALLEL}]")
    sys.exit(run(args.parallel, args.yes, args.answers))


if __name__ == "__main__":
    main()
