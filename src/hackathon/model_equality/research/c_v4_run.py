"""Collector of the ep answers of configuration v4: N arithmetic prompts + 3 identity questions.

READY, NOT RUN. Locked with its own authorisation file, `plus10_v4_authorisation.json`:
`{"approved": true, "stage": "v4", "n": 10 | 15 | 20, "by": "...", "vote_ref": "..."}`, written by
the humans' vote (they choose N), never by this tool. It sends nothing unless `--yes` is given AND
that file is valid. Per endpoint: the first `n` of the 20 frozen arithmetic prompts of
`plus10_prompts.json`, and the 3 frozen identity formulations of `identity_prompts.json`, one draw
each: `n + 3` answers (13 / 18 / 23 for N = 10 / 15 / 20), that is 429 / 594 / 759 requests for 33
endpoints. Same safety as `c_plus10_run`: exact server-counter precondition (7 used, 8 on
ep-09..12, plus the answers already stored), cap of `n + 3` per endpoint, at most 4 requests in
flight, no retry after a read timeout, counter check after the run, resuming is safe.
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
from hackathon.model_equality.research.c_v3_run import IDENTITY_FILE, load_identity
from hackathon.model_equality.store import AnswerStore, default_answers_path

AUTHORISATION_FILE = Path("reports/phase2/model_equality/team-c/plus10_v4_authorisation.json")
STAGE_NAME = "v4"
ALLOWED_N: tuple[int, ...] = (10, 15, 20)
IDENTITY_QUESTIONS = 3


def check_authorisation_v4(path: Path) -> dict[str, object]:
    """The human authorisation of configuration v4 (it carries the chosen N).

    Raises:
        AuthorisationError: file missing, unreadable, not an approval of stage "v4", N not allowed,
            or unsigned.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not voted")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise AuthorisationError(f"unreadable authorisation file: {exc}") from exc
    if payload.get("approved") is not True or payload.get("stage") != STAGE_NAME:
        raise AuthorisationError(f"authorisation is not an approval of stage {STAGE_NAME!r}: {payload}")
    if payload.get("n") not in ALLOWED_N:
        raise AuthorisationError(f"authorisation must choose n in {ALLOWED_N}: {payload.get('n')!r}")
    if not payload.get("by") or not payload.get("vote_ref"):
        raise AuthorisationError("authorisation must name who approved it and the vote reference")
    return dict(payload)


def build_jobs(
    endpoints: Sequence[str], arithmetic: Sequence[tuple[str, str]],
    formulations: Sequence[tuple[str, str]], max_tokens: int,
) -> list[QueryJob]:
    """Arithmetic prompts first (prompt-major), then the identity formulations, one draw each."""
    jobs = [QueryJob(e, pid, 0, text) for pid, text in arithmetic for e in endpoints]
    jobs += [QueryJob(e, pid, 0, text, max_tokens) for pid, text in formulations for e in endpoints]
    return jobs


def run(n: int | None, parallel: int, yes: bool, answers: Path | None) -> int:
    """Dry-run by default (N from `--n`, default 15); with `--yes`, N comes from the authorisation."""
    endpoints = list(SCORED_ENDPOINTS)
    authorisation: dict[str, object] | None = None
    if yes:
        try:
            authorisation = check_authorisation_v4(AUTHORISATION_FILE)
        except AuthorisationError as exc:
            print(f"REFUSED: {exc}")
            return EXIT_NOT_AUTHORISED
        n = int(str(authorisation["n"]))
    chosen = n or 15
    arithmetic = load_prompts(PROMPTS_FILE, 2)[:chosen]
    formulations, _draws, max_tokens = load_identity(IDENTITY_FILE)
    ids = [pid for pid, _ in arithmetic] + [pid for pid, _ in formulations]
    stage_cap = len(arithmetic) + len(formulations)
    store = AnswerStore(answers or default_answers_path(reference=False))
    stored_before = stored_per_endpoint(store, endpoints, ids)
    client = ApiClient(load_settings(), log=print)
    before = client.list_budgets()
    try:
        check_before(before, endpoints, stored_before, stage_cap)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the plan: {exc}")
        return EXIT_PRECONDITION
    jobs = build_jobs(endpoints, arithmetic, formulations, max_tokens)
    todo = sum(stage_cap - stored_before[e] for e in endpoints)
    print(f"budget state OK; v4 N = {chosen}: {stage_cap} answers per endpoint x {len(endpoints)} = "
          f"{len(jobs)} jobs, {todo} still to send, parallel {parallel}")
    if not yes or authorisation is None:
        print("DRY RUN: nothing sent (needs --yes and a valid authorisation file).")
        return EXIT_OK
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
    parser.add_argument("--n", type=int, choices=ALLOWED_N, default=None, help="dry-run size only")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be in [1, {MAX_PARALLEL}]")
    sys.exit(run(args.n, args.parallel, args.yes, args.answers))


if __name__ == "__main__":
    main()
