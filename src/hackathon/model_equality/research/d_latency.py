"""Per-token latency test: does a quantised or LoRA endpoint decode at a different speed? (team D)

Hypothesis (director): fp8 decodes faster, a LoRA adds overhead, so the time per generated token separates
ref-02 (fp8) and the LoRA endpoints from ref-01 even when the text cannot.

Design, ONE request at a time (no concurrency from our side, so the queue we cause is zero): R rounds; in
each round every endpoint gets one LONG request (`LONG_PROMPT`, max_tokens 256) and one SHORT request
(`SHORT_PROMPT`, max_tokens 1) in a fresh random order (seeded). The client-side latency of each request
is the wall time of the POST. The number of generated tokens of a long reply is counted with the model
tokenizer (the server returns no usage field).

Per-token latency of a long request = (latency of the long request - latency of the SHORT request of the
same endpoint and round) / generated tokens; the short request removes the fixed cost (about 8 s) and the
queue present at that moment.

CRITERIA FROZEN BEFORE THE MEASUREMENT:
* fp8 is SEPARATED by latency when the median per-token latency of ref-02 differs from that of ref-01 by
  at least `MIN_RELATIVE_DIFFERENCE` (10 %), the Mann-Whitney p-value of the two samples is <=
  `ALPHA` (0.01), and the sign of the difference is the same in at least `MIN_CONSISTENT_ROUNDS_SHARE`
  (7/8) of the rounds.
* A LoRA endpoint (ref-09, ref-04, ref-07) is SEPARATED under the same rule against ref-01.
* The latency is only a usable flag if the disguises (ref-03, ref-05, ref-06, ref-08) are NOT separated.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy import stats

from hackathon.common.config import load_settings
from hackathon.model_equality.research.d_client import make_busy_aware_client

LONG_PROMPT = "Write a long, detailed essay about the history of the Roman Empire."
SHORT_PROMPT = "Say hi."
LONG_MAX_TOKENS = 256
SHORT_MAX_TOKENS = 1
ROUNDS = 8
SEED = 20261121
ENDPOINTS = ("ref-01", "ref-02", "ref-09", "ref-04", "ref-07", "ref-03", "ref-05", "ref-06", "ref-08")
PLAIN = "ref-01"
MIN_RELATIVE_DIFFERENCE = 0.10
ALPHA = 0.01
MIN_CONSISTENT_ROUNDS_SHARE = 7 / 8
MIN_TOKENS = 20
"""Long replies shorter than this give a noisy per-token latency and are dropped."""


def per_token_latencies(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """`(round, seconds per token)` per endpoint from raw measurements.

    Each row: {endpoint, round, kind ('long' or 'short'), latency_s, tokens}.
    """
    short: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["kind"] == "short":
            short[(str(row["endpoint"]), int(row["round"]))] = float(row["latency_s"])
    out: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row["kind"] != "long" or int(row["tokens"]) < MIN_TOKENS:
            continue
        key = (str(row["endpoint"]), int(row["round"]))
        if key not in short:
            continue
        seconds = (float(row["latency_s"]) - short[key]) / int(row["tokens"])
        out.setdefault(str(row["endpoint"]), []).append((int(row["round"]), seconds))
    return out


def separated(values: Sequence[tuple[int, float]], plain: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Apply the frozen rule to one endpoint against the plain endpoint."""
    a = [v for _r, v in values]
    b = [v for _r, v in plain]
    if len(a) < 3 or len(b) < 3:
        return {"separated": False, "reason": "not enough long replies"}
    median_a, median_b = statistics.median(a), statistics.median(b)
    relative = (median_a - median_b) / median_b if median_b else 0.0
    p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    by_round_a = {r: v for r, v in values}
    by_round_b = {r: v for r, v in plain}
    common = sorted(set(by_round_a) & set(by_round_b))
    sign = 1 if relative >= 0 else -1
    consistent = sum(1 for r in common if (by_round_a[r] - by_round_b[r]) * sign > 0)
    share = consistent / len(common) if common else 0.0
    ok = abs(relative) >= MIN_RELATIVE_DIFFERENCE and p_value <= ALPHA and share >= MIN_CONSISTENT_ROUNDS_SHARE
    return {"separated": ok, "relative_difference": relative, "p_value": p_value, "consistent_round_share": share,
            "median_s_per_token": median_a, "plain_median_s_per_token": median_b, "n": len(a)}


def analyse(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Frozen-rule verdict for every endpoint against ref-01."""
    latencies = per_token_latencies(rows)
    plain = latencies.get(PLAIN, [])
    return {endpoint: separated(values, plain) for endpoint, values in latencies.items() if endpoint != PLAIN}


def run(out: Path, rounds: int = ROUNDS) -> None:
    """Run the sequential measurement and write one JSON row per request (append-only)."""
    from transformers import AutoTokenizer

    from hackathon.model_equality.research.d_local import DEFAULT_MODEL

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True)
    sidecar = out.with_suffix(".raw.jsonl")
    client = make_busy_aware_client(load_settings(), sidecar=sidecar)
    rng = random.Random(SEED)
    with out.open("a", encoding="utf-8") as handle:
        for round_index in range(rounds):
            order = list(ENDPOINTS)
            rng.shuffle(order)
            for endpoint in order:
                for kind, prompt, cap in (("long", LONG_PROMPT, LONG_MAX_TOKENS), ("short", SHORT_PROMPT, SHORT_MAX_TOKENS)):
                    started = time.perf_counter()
                    reply = client.chat(endpoint, prompt, cap)
                    latency = time.perf_counter() - started
                    tokens = len(tokenizer(reply.content, add_special_tokens=False).input_ids)
                    row = {"endpoint": endpoint, "round": round_index, "kind": kind, "latency_s": round(latency, 3),
                           "tokens": tokens, "chars": len(reply.content)}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
            print(f"round {round_index + 1}/{rounds} done", flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else ROUNDS)
    else:
        data = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(analyse(data), indent=1))
