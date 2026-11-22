"""Collector of the ep answers of configuration v6: L long answers + 3 identity questions.

READY, NOT RUN. Locked with its own authorisation file, `long_v6_authorisation.json`:
`{"approved": true, "stage": "v6", "long": 4 | 8 | 12 | 14 | 15 | 22, "by": "...", "vote_ref": "..."}`,
written by the humans' decision, never by this tool. It sends nothing unless `--yes` is given AND
that file is valid. Per endpoint: the first `long` of the 22 frozen long prompts of
`long_v6_prompts.json` (user message only, 256 tokens at most) and the 3 frozen identity
formulations of `identity_prompts.json`, one draw each: `long + 3` answers (18 / 17 / 15 / 25 for
L = 15 / 14 / 12 / 22), that is 594 / 561 / 495 / 825 requests for 33 endpoints. Same safety as
`c_v4_run`: exact server-counter precondition (7 used, 8 on ep-09..12, plus the answers already
stored), cap of `long + 3` per endpoint, at most 4 requests in flight, no retry after a read
timeout, counter check after the run, resuming is safe.
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
    AuthorisationError,
    PreconditionError,
    check_after,
    check_before,
    stored_per_endpoint,
)
from hackathon.model_equality.research.c_v3_run import IDENTITY_FILE, load_identity
from hackathon.model_equality.store import AnswerStore, default_answers_path

LONG_PROMPTS_FILE = Path("reports/phase2/model_equality/team-c/long_v6_prompts.json")
AUTHORISATION_FILE = Path("reports/phase2/model_equality/team-c/long_v6_authorisation.json")
STAGE_NAME = "v6"
ALLOWED_LONG: tuple[int, ...] = (4, 8, 12, 14, 15, 22)
DEFAULT_LONG = 14


def load_long_prompts(path: Path) -> tuple[list[tuple[str, str]], int]:
    """`(id, text)` of the frozen long prompts and their token cap."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(row["id"], row["prompt"]) for row in payload["prompts"]], int(payload["max_tokens"])


def check_authorisation_v6(path: Path) -> dict[str, object]:
    """The human authorisation of configuration v6 (it carries the chosen number of long answers).

    Raises:
        AuthorisationError: file missing, unreadable, not an approval of stage "v6", `long` not
            allowed, or unsigned.
    """
    if not path.is_file():
        raise AuthorisationError(f"no authorisation file at {path}: the humans have not decided")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise AuthorisationError(f"unreadable authorisation file: {exc}") from exc
    if payload.get("approved") is not True or payload.get("stage") != STAGE_NAME:
        raise AuthorisationError(f"authorisation is not an approval of stage {STAGE_NAME!r}: {payload}")
    if payload.get("long") not in ALLOWED_LONG:
        raise AuthorisationError(f"authorisation must choose long in {ALLOWED_LONG}: {payload.get('long')!r}")
    if not payload.get("by") or not payload.get("vote_ref"):
        raise AuthorisationError("authorisation must name who approved it and the decision reference")
    return dict(payload)


def build_jobs(
    endpoints: Sequence[str], long_prompts: Sequence[tuple[str, str]], long_tokens: int,
    formulations: Sequence[tuple[str, str]], identity_tokens: int,
) -> list[QueryJob]:
    """Long prompts first (prompt-major), then the identity formulations, one draw each."""
    jobs = [QueryJob(e, pid, 0, text, long_tokens) for pid, text in long_prompts for e in endpoints]
    jobs += [QueryJob(e, pid, 0, text, identity_tokens) for pid, text in formulations
             for e in endpoints]
    return jobs


def run(long: int | None, parallel: int, yes: bool, answers: Path | None) -> int:
    """Dry-run by default (L from `--long`, default 14); with `--yes`, L comes from the authorisation."""
    endpoints = list(SCORED_ENDPOINTS)
    authorisation: dict[str, object] | None = None
    if yes:
        try:
            authorisation = check_authorisation_v6(AUTHORISATION_FILE)
        except AuthorisationError as exc:
            print(f"REFUSED: {exc}")
            return EXIT_NOT_AUTHORISED
        long = int(str(authorisation["long"]))
    chosen = long or DEFAULT_LONG
    all_long, long_tokens = load_long_prompts(LONG_PROMPTS_FILE)
    long_prompts = all_long[:chosen]
    formulations, _draws, identity_tokens = load_identity(IDENTITY_FILE)
    ids = [pid for pid, _ in long_prompts] + [pid for pid, _ in formulations]
    stage_cap = len(long_prompts) + len(formulations)
    store = AnswerStore(answers or default_answers_path(reference=False))
    stored_before = stored_per_endpoint(store, endpoints, ids)
    client = ApiClient(load_settings(), log=print)
    before = client.list_budgets()
    try:
        check_before(before, endpoints, stored_before, stage_cap)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the plan: {exc}")
        return EXIT_PRECONDITION
    jobs = build_jobs(endpoints, long_prompts, long_tokens, formulations, identity_tokens)
    todo = sum(stage_cap - stored_before[e] for e in endpoints)
    print(f"budget state OK; v6 L = {chosen}: {stage_cap} answers per endpoint x {len(endpoints)} = "
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
    parser.add_argument("--long", type=int, choices=ALLOWED_LONG, default=None,
                        help="dry-run size only")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--answers", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.parallel <= MAX_PARALLEL:
        parser.error(f"--parallel must be in [1, {MAX_PARALLEL}]")
    sys.exit(run(args.long, args.parallel, args.yes, args.answers))


if __name__ == "__main__":
    main()
