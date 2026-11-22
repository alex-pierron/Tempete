"""Reference-endpoint collector of team D (unlimited `ref-*`, never `ep-*`).

Why a dedicated runner: the `ref-*` endpoints have no quota, but the server throughput is shared
by the whole team (4 simultaneous requests, about 10 s per request whatever the length), so every
batch is explicit, capped (`--max-requests`) and resumable. Only endpoints named `ref-NN` are
accepted: a typo that would reach a scored `ep-NN` endpoint (budget of 100 answers, irreplaceable)
is rejected before any request is sent.

Usage (repo root, venv of the repo)::

    python -m hackathon.model_equality.research.d_collect --wave d1 --repeats 2 --parallel 3

Answers are appended to `artifacts/model_equality/reference_answers_d.jsonl` (official format).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from hackathon.common.config import load_settings
from hackathon.model_equality.client import FatalQueryError, QueryJob, run_jobs
from hackathon.model_equality.research.d_client import make_busy_aware_client
from hackathon.model_equality.research.d_prompts import MAX_TOKENS, DPrompt, wave_prompts
from hackathon.model_equality.store import REPO_ROOT, STORE_DIR_RELATIVE, AnswerStore

D_STORE_FILENAME = "reference_answers_d.jsonl"
D_STORE_PATH = REPO_ROOT / STORE_DIR_RELATIVE / D_STORE_FILENAME
"""Store of team D: one file per team, so that git merges never conflict."""
DEFAULT_PARALLEL = 3
"""Requests in flight: the team limit is 4 and `m-met-1` queries too."""
DEFAULT_ENDPOINTS = ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "ref-09")
"""S plain (01) and its four disguises (03 system prompt, 05 persona, 06 watermark, 08 temperature
1.0), plus the two targets: fp8 (02) and LoRA 0.3 (09)."""
REFERENCE_ENDPOINT_PATTERN = re.compile(r"^ref-\d{2}$")
EXIT_OK = 0
EXIT_FAILURE = 1


class CollectError(ValueError):
    """The requested batch is invalid (non-reference endpoint, bad repeat count, ...)."""


def validate_reference_endpoints(endpoints: Sequence[str]) -> list[str]:
    """Return the endpoints unchanged when all are `ref-NN`.

    Raises:
        CollectError: An endpoint is not a reference endpoint (it could be a scored `ep-NN`).
    """
    invalid = [e for e in endpoints if not REFERENCE_ENDPOINT_PATTERN.match(e)]
    if invalid:
        raise CollectError(f"refusing non-reference endpoints: {invalid}")
    return list(endpoints)


def build_jobs(
    prompts: Sequence[DPrompt],
    endpoints: Sequence[str],
    repeats: int,
    stored: set[tuple[str, str, int]],
    prompt_major: bool = False,
) -> list[QueryJob]:
    """Jobs of one batch.

    Repeat-major (default): every endpoint has seen the same prompts, and a repeat is only started
    once the previous one is complete. Prompt-major: a prompt is finished on every endpoint and
    repeat before the next one starts, so that the first K prompts are COMPLETE as early as possible
    (used by the hold-out collection, where a consumer waits for 25 then 50 complete prompts).

    Raises:
        CollectError: `repeats` below 1.
    """
    if repeats < 1:
        raise CollectError(f"repeats must be >= 1, got {repeats}")
    jobs: list[QueryJob] = []

    def add(prompt: DPrompt, repeat: int) -> None:
        for endpoint in endpoints:
            if (endpoint, prompt.prompt_id, repeat) not in stored:
                jobs.append(QueryJob(endpoint, prompt.prompt_id, repeat, prompt.text, MAX_TOKENS))

    if prompt_major:
        for prompt in prompts:
            for repeat in range(repeats):
                add(prompt, repeat)
    else:
        for repeat in range(repeats):
            for prompt in prompts:
                add(prompt, repeat)
    return jobs


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--wave", required=True, help="prompt wave: d1 to d7, or h1 (hold-out)")
    parser.add_argument("--repeats", type=int, default=1, help="repeat indices 0..repeats-1")
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    parser.add_argument("--max-requests", type=int, default=None, help="hard cap on this run")
    parser.add_argument("--store", type=Path, default=D_STORE_PATH)
    parser.add_argument("--order", choices=("repeat", "prompt"), default="repeat",
                        help="job order: repeat-major (default) or prompt-major (complete prompts first)")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="JSONL of raw API fields (id, finish_reason, usage, latency) beside the store")
    parser.add_argument("--dump-prompts", type=Path, default=None,
                        help="write the wave's prompts (id, text) to this JSONL file and exit")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="create this file to stop gracefully: in-flight requests finish, no new one is sent")
    parser.add_argument("--ids", nargs="+", default=None, help="restrict the wave to these prompt ids")
    parser.add_argument("--prompts", type=int, default=None,
                        help="use only the first K prompts of the wave (the hold-out stops at 50)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one batch and return a process exit code."""
    args = parse_args(argv)
    endpoints = validate_reference_endpoints(args.endpoints)
    if args.dump_prompts is not None:
        lines = [json.dumps({"id": p.prompt_id, "prompt": p.text}, ensure_ascii=False)
                 for p in wave_prompts(args.wave)]
        args.dump_prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} prompts to {args.dump_prompts}")
        return EXIT_OK
    store = AnswerStore(args.store)
    stored = set(store.load())
    wave = wave_prompts(args.wave)
    if args.ids is not None:
        wave = [p for p in wave if p.prompt_id in set(args.ids)]
    if args.prompts is not None:
        wave = wave[: args.prompts]
    jobs = build_jobs(wave, endpoints, args.repeats, stored, prompt_major=args.order == "prompt")
    if args.max_requests is not None:
        jobs = jobs[: args.max_requests]
    print(f"store={store.path} wave={args.wave} jobs={len(jobs)} stored={len(stored)}", flush=True)
    if args.dry_run or not jobs:
        print("nothing sent")
        return EXIT_OK
    def say(message: str) -> None:
        print(message, flush=True)

    client = make_busy_aware_client(load_settings(), log=say, sidecar=args.sidecar, stop_file=args.stop_file)
    try:
        report = run_jobs(client, store, jobs, caps=None, max_parallel=args.parallel, progress=say)
    except FatalQueryError as exc:
        say(f"stopped: {exc}")
        return EXIT_OK
    print("run: " + report.summary(), flush=True)
    for failure in report.failed:
        print(f"FAILED {failure}")
    return EXIT_OK if not report.failed else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
