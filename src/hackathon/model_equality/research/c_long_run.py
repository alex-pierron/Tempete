"""Human-authorised `ep` collection of 21/11/2026: 5 long open prompts x 1 draw on the 33 endpoints.

Decision: +5 answers per endpoint (165 requests, 5 % of the budget), no more. Rules enforced here
rather than trusted:

* BEFORE: the server budget must be exactly the state the decision was taken on (5 used on every
  `ep`, 6 on `ep-09` .. `ep-12`); anything else stops the run before a single request;
* DURING: exactly 5 jobs per endpoint, a cap of 5 per endpoint, at most 2 requests in flight, and
  the client of `main` (no retry after a read timeout: a timed-out request may already be counted);
* AFTER: every endpoint must be at exactly its previous count + the answers stored for it, and
  never above the authorised total. A failed job is reported, never retried blindly.

The prompts are open-ended (explanation, story, list), identical on every endpoint, one draw each,
generation capped at the server limit. Yes/no prompts carry one bit; open text lets two greedy
endpoints of the same model agree on hundreds of characters, and lets a quantised or LoRA variant
diverge somewhere in them.
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
    MAX_NEW_TOKENS,
    MAX_PARALLEL,
    MAX_PROMPT_TOKENS,
    ApiClient,
    EndpointBudget,
    QueryJob,
    RunReport,
    run_jobs,
)
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.store import AnswerStore, default_answers_path

AUTHORISED_PROMPT_COUNT = 2
"""Decision (b) of 21/11/2026: only the first 2 long prompts are completed, then the collection
stops (the rest of the budget is kept for ambiguous endpoints)."""
ADDED_PER_ENDPOINT = AUTHORISED_PROMPT_COUNT
LONG_RUN_MAX_PARALLEL = 2
"""2 in flight: r-met-d queries the ref again and the two teams share the cap of 4."""
EXPECTED_USED_BEFORE = 5
EXPECTED_USED_BEFORE_OVERSPENT = 6
OVERSPENT_ENDPOINTS: frozenset[str] = frozenset({"ep-09", "ep-10", "ep-11", "ep-12"})
CHARACTERS_PER_TOKEN_FLOOR = 2.0
"""Conservative bound (real text is about 4 characters per token) for the input-token check."""
EXIT_OK = 0
EXIT_PRECONDITION = 3
EXIT_FAILURE = 1

LONG_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "long-explain-000",
        ("Explain step by step how binary search works and why its running time is logarithmic. "
         "Use a numbered list and give a short worked example on a sorted list of 16 numbers."),
    ),
    (
        "long-story-000",
        ("Write a short story of about 200 words about a lighthouse keeper who finds a sealed "
         "message in a bottle on the rocks after a storm."),
    ),
    (
        "long-list-000",
        ("List the eight planets of the solar system in order from the Sun, as a numbered list, "
         "and give two distinctive facts about each one."),
    ),
    (
        "long-explain-001",
        ("Explain to a high-school student how vaccines train the immune system, in about 200 "
         "words, and finish with one common misconception."),
    ),
    (
        "long-compare-000",
        ("Describe photosynthesis in detail, then compare it with cellular respiration in a "
         "short table-like list of differences."),
    ),
)


class PreconditionError(RuntimeError):
    """The server budget is not the state the human decision was taken on."""


@dataclass(frozen=True)
class AfterCheck:
    """Budget state after the run: per endpoint, used before / after / answers stored."""

    endpoint: str
    before: int
    after: int
    stored: int
    total_stored: int = 0

    @property
    def ok(self) -> bool:
        """Exactly the stored answers were spent, and never more than the authorised total."""
        return (self.after == self.before + self.stored
                and max(self.stored, self.total_stored) <= ADDED_PER_ENDPOINT)


def expected_used(endpoint: str) -> int:
    """Used count the decision was taken on: 5, or 6 on `ep-09` .. `ep-12`."""
    return EXPECTED_USED_BEFORE_OVERSPENT if endpoint in OVERSPENT_ENDPOINTS else EXPECTED_USED_BEFORE


def check_prompts() -> None:
    """Every prompt must fit the input limit with a wide margin and be a non-trivial question.

    Raises:
        ValueError: A prompt is too long or duplicated.
    """
    identifiers = [pid for pid, _ in LONG_PROMPTS]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicated prompt ids")
    for pid, text in LONG_PROMPTS:
        if len(text) / CHARACTERS_PER_TOKEN_FLOOR >= MAX_PROMPT_TOKENS:
            raise ValueError(f"{pid}: {len(text)} characters may exceed {MAX_PROMPT_TOKENS} tokens")


def check_budget_before(
    budgets: Mapping[str, EndpointBudget], endpoints: Sequence[str],
    already_stored: Mapping[str, int] | None = None,
) -> None:
    """Stop unless every endpoint is exactly at the state the decision was taken on.

    On a resumed run, `already_stored` (answers to the long prompts already in the store) is
    added to the expected count: a lost or double-counted request shows up as a mismatch.

    Raises:
        PreconditionError: Any endpoint is missing or has another `used` value (all listed).
    """
    problems = []
    done = already_stored or {}
    for endpoint in endpoints:
        info = budgets.get(endpoint)
        expected = expected_used(endpoint) + done.get(endpoint, 0)
        if info is None:
            problems.append(f"{endpoint}: not listed by the server")
        elif info.used != expected:
            problems.append(f"{endpoint}: used {info.used}, expected {expected}")
        elif done.get(endpoint, 0) > ADDED_PER_ENDPOINT:
            problems.append(f"{endpoint}: {done[endpoint]} stored, above the authorised "
                            f"{ADDED_PER_ENDPOINT}")
    if problems:
        raise PreconditionError("; ".join(problems))


def build_jobs(endpoints: Sequence[str]) -> list[QueryJob]:
    """One job per (authorised prompt, endpoint), prompt-major.

    Every endpoint answers prompt 1 before anyone sees prompt 2, so a run that stops early is
    balanced (complete prompts, not complete endpoints). Only the first `AUTHORISED_PROMPT_COUNT`
    prompts are ever asked.
    """
    check_prompts()
    return [
        QueryJob(endpoint, pid, 0, text, MAX_NEW_TOKENS)
        for pid, text in LONG_PROMPTS[:AUTHORISED_PROMPT_COUNT] for endpoint in endpoints
    ]


def check_budget_after(
    before: Mapping[str, EndpointBudget], after: Mapping[str, EndpointBudget],
    stored: Mapping[str, int], stored_before: Mapping[str, int] | None = None,
) -> list[AfterCheck]:
    """Per-endpoint consistency between the server counter and the answers stored in this run."""
    start = stored_before or {}
    # only the endpoints this run may query: the ref counters move under other teams' feet
    return [
        AfterCheck(e, before[e].used, after[e].used, stored[e] - start.get(e, 0), stored[e])
        for e in sorted(stored)
    ]


def stored_long_per_endpoint(store: AnswerStore, endpoints: Sequence[str]) -> dict[str, int]:
    """Number of stored answers to the long prompts, per endpoint."""
    wanted = {pid for pid, _ in LONG_PROMPTS}  # any long prompt counts, authorised or not
    counts = dict.fromkeys(endpoints, 0)
    for answer in store.load().values():
        if answer.prompt_id in wanted and answer.endpoint in counts:
            counts[answer.endpoint] += 1
    return counts


def run(answers: Path | None, dry_run: bool) -> int:
    """Execute (or dry-run) the authorised collection; returns a process exit code."""
    endpoints = list(SCORED_ENDPOINTS)
    settings = load_settings()
    client = ApiClient(settings, log=print)
    before = client.list_budgets()
    store = AnswerStore(answers or default_answers_path(reference=False))
    stored_before = stored_long_per_endpoint(store, endpoints)
    try:
        check_budget_before(before, endpoints, stored_before)
    except PreconditionError as exc:
        print(f"STOP, budget differs from the authorised state: {exc}")
        return EXIT_PRECONDITION
    jobs = build_jobs(endpoints)
    print(f"budget state OK; {len(jobs)} jobs, {ADDED_PER_ENDPOINT} per endpoint, "
          f"parallel {LONG_RUN_MAX_PARALLEL}")
    if dry_run:
        print("DRY RUN: nothing sent.")
        return EXIT_OK
    if LONG_RUN_MAX_PARALLEL > MAX_PARALLEL:
        raise ValueError("parallelism above the client maximum")
    caps = {e: ADDED_PER_ENDPOINT - stored_before[e] for e in endpoints}
    report: RunReport = run_jobs(client, store, jobs, caps=caps,
                                 max_parallel=LONG_RUN_MAX_PARALLEL, progress=print)
    print("run: " + report.summary())
    for failure in report.failed:
        print(f"FAILED {failure}")
    after = client.list_budgets()
    checks = check_budget_after(before, after, stored_long_per_endpoint(store, endpoints),
                                stored_before)
    bad = [c for c in checks if not c.ok]
    print(json.dumps([c.__dict__ for c in checks]))
    if bad:
        print(f"BUDGET MISMATCH on {[c.endpoint for c in bad]}: do not query again, report.")
    return EXIT_OK if not report.failed and not bad else EXIT_FAILURE


def main() -> None:
    """CLI entry point (`--yes` is required to send anything)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--yes", action="store_true", help="really spend the authorised budget")
    args = parser.parse_args()
    sys.exit(run(args.answers, dry_run=not args.yes))


if __name__ == "__main__":
    main()
