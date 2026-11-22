"""Quota-guarded `ref-*` querier of team B.

Why not `query-ref`: on 21/11/2026 the server reports `limit=1000000, used~15000000` for every
`ref-*` endpoint, so `left` is 0 and the kit's budget policy plans zero requests. The team rule
is a hard quota per `ref-*` endpoint for team B (50 at first, raised to 300 by the coordinator
during the session), which only the local store can enforce. This runner therefore computes its
own per-endpoint allowance from the store and hands it to the kit's `run_jobs` as hard caps.

Usage (repo root, venv of the repo)::

    MET_REFERENCE_ANSWERS_FILE=artifacts/model_equality/reference_answers_b.jsonl \
    python -m hackathon.model_equality.research.ref_runner --repeat 0 --parallel 4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence

from hackathon.common.config import load_settings
from hackathon.model_equality.client import ApiClient, QueryJob, run_jobs
from hackathon.model_equality.planning import REFERENCE_ENDPOINTS
from hackathon.model_equality.prompts.bank import Prompt
from hackathon.model_equality.research.candidates import candidate_bank, candidate_by_id
from hackathon.model_equality.store import AnswerStore, default_answers_path

TEAM_B_QUOTA = 300
"""Answers per `ref-*` endpoint that team B may store (50 until the coordinator lifted it)."""
DEFAULT_PARALLEL = 2
"""Half of the server's 4 concurrent requests per team: team A queries at the same time."""
EXIT_OK = 0
EXIT_FAILURE = 1


class QuotaError(ValueError):
    """The requested batch would exceed the team B quota or the team ceiling."""


def allowance(stored_b: int) -> int:
    """New answers team B may still request on one endpoint (quota minus what it stored)."""
    return max(0, TEAM_B_QUOTA - stored_b)


def build_jobs(
    prompts: Sequence[Prompt],
    repeat: int,
    endpoints: Sequence[str],
    stored_keys: set[tuple[str, str, int]],
) -> list[QueryJob]:
    """Jobs of one batch, prompt-major so an interrupted run is balanced across endpoints."""
    jobs: list[QueryJob] = []
    for prompt in prompts:
        for endpoint in endpoints:
            if (endpoint, prompt.id, repeat) not in stored_keys:
                jobs.append(QueryJob(endpoint, prompt.id, repeat, prompt.text, prompt.max_tokens))
    return jobs


def select_prompts(prompt_ids: Sequence[str] | None) -> list[Prompt]:
    """Candidates to query: all of them, or the listed ids.

    Raises:
        QuotaError: Unknown prompt id.
    """
    known = candidate_by_id()
    if not prompt_ids:
        return candidate_bank()
    unknown = [pid for pid in prompt_ids if pid not in known]
    if unknown:
        raise QuotaError(f"unknown candidate ids: {unknown}")
    return [known[pid] for pid in prompt_ids]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repeat", type=int, required=True, help="repeat index to fetch (0, 1, ...)")
    parser.add_argument("--prompt-ids", nargs="+", default=None,
                        help="candidate ids (default: all candidates)")
    parser.add_argument("--endpoints", nargs="+", default=list(REFERENCE_ENDPOINTS))
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help="requests in flight (the server allows 4 per team, A uses them too)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one batch within the quota; return a process exit code."""
    args = parse_args(argv)
    store = AnswerStore(default_answers_path(reference=True))
    stored = store.load()
    stored_b = Counter(key[0] for key in stored)
    prompts = select_prompts(args.prompt_ids)
    jobs = build_jobs(prompts, args.repeat, args.endpoints, set(stored))
    caps = {e: allowance(stored_b[e]) for e in args.endpoints}
    requested = Counter(job.endpoint for job in jobs)
    over = {e: (requested[e], caps[e]) for e in args.endpoints if requested[e] > caps[e]}
    print(f"store={store.path} jobs={len(jobs)} per-endpoint stored_b={dict(stored_b)} caps={caps}")
    if over:
        raise QuotaError(f"batch exceeds the allowance on {over} (requested, allowed)")
    if args.dry_run:
        print("dry run: nothing sent")
        return EXIT_OK
    client = ApiClient(load_settings(), log=print)
    report = run_jobs(client, store, jobs, caps=caps, max_parallel=args.parallel, progress=print)
    print("run: " + report.summary())
    for failure in report.failed:
        print(f"FAILED {failure}")
    return EXIT_OK if not report.failed else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
