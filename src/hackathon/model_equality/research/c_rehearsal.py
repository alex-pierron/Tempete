"""Rehearsal on the free ref endpoints of the decision chain planned for the +10 ep answers.

The chain, exactly as it will run on the ep:

1. N arithmetic prompts, the same for every endpoint, ONE draw each; the statistic is the rate of
   answers whose stripped text ends with a full stop;
2. band rule: flag an endpoint iff its rate lies in `[band_low, band_high]` (the lower edge sits
   above temperature-1.0 copies, below LoRA 0.3);
3. grouping recipe: each flagged endpoint is its own model, all the others are one main model
   (plain, system prompt, persona, watermark, temperature 1.0 and fp8 copies);
4. official score against the known ref partition (one family, models S / fp8 / three LoRA).

`bootstrap_scores` repeats 1-4 over random draws of the N prompts and of one answer per prompt;
the summary is the median, the 10th / 90th percentiles and the share of draws at or above the
decision threshold. Development (`DEV`) tunes `band_low` on the existing ref data; the held-out
test (`HOLDOUT`) only runs the frozen chain recorded in `preregistered.json`. Nothing here touches
the network.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_oracle import full_score
from hackathon.model_equality.research.c_pairs import Partition, pair_index

FloatArray = npt.NDArray[np.float64]
Pool = dict[str, dict[str, list[str]]]
"""`endpoint -> prompt id -> answers (one per draw)`."""
BAND_HIGH = 1.00
REPLICATES = 1000
SEED = 20261121
PERCENTILES: tuple[float, float] = (10.0, 90.0)
STORE_FILES: tuple[str, ...] = (
    "reference_answers.jsonl", "reference_answers_b.jsonl", "reference_answers_m1.jsonl",
    "reference_answers_d.jsonl",
)
HOLDOUT_FILE = "reference_answers_holdout.jsonl"
REF_ENDPOINTS: tuple[str, ...] = tuple(f"ref-{k:02d}" for k in range(1, 10))
TEMPERATURE_REF = "ref-08"
TRUE_MODEL: Mapping[str, str] = {
    "ref-01": "S", "ref-02": "T1_fp8", "ref-03": "S", "ref-04": "T3_lora07", "ref-05": "S",
    "ref-06": "S", "ref-07": "T4_lora15", "ref-08": "S", "ref-09": "T2_lora03",
}
KIND: Mapping[str, str] = {
    "ref-01": "plain", "ref-02": "fp8", "ref-03": "system prompt", "ref-04": "LoRA 0.7",
    "ref-05": "persona", "ref-06": "watermark", "ref-07": "LoRA 1.5", "ref-08": "temperature 1.0",
    "ref-09": "LoRA 0.3",
}
PRIMARY_PATTERN = re.compile(r"^arith-\d{3}$")
SECONDARY_PATTERN = re.compile(r"^d-(varith|arith|barith)-\d{3}$")
CEILING_FP8_LUMPED = 0.66
CEILING_EVERYTHING = 0.90


class RehearsalError(ValueError):
    """The pool cannot serve the requested sample size."""


@dataclass(frozen=True)
class Rule:
    """The frozen decision chain: band edges and the decision threshold of the spend rule."""

    band_low: float
    band_high: float = BAND_HIGH
    threshold_x: float = 0.40


def ends_with_full_stop(answer: str) -> bool:
    """True when the stripped answer ends with a full stop."""
    return answer.strip().endswith(".")


def _add(pool: Pool, row: dict[str, object]) -> None:
    pool[str(row["endpoint"])].setdefault(str(row["id"]), []).append(str(row["response"]))


def load_pools(store_dir: Path) -> tuple[Pool, Pool]:
    """`(primary, secondary)` development pools, duplicates removed.

    Primary = bank prompts `arith-000..015` (all 9 ref); secondary = r-met-d `d4` prompts (7 ref).
    """
    seen: set[tuple[str, str, int]] = set()
    primary: Pool = defaultdict(dict)
    secondary: Pool = defaultdict(dict)
    for name in STORE_FILES:
        path = store_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["endpoint"], row["id"], int(row["repeat"]))
            if key in seen or row["endpoint"] not in TRUE_MODEL:
                continue
            seen.add(key)
            if PRIMARY_PATTERN.match(row["id"]):
                _add(primary, row)
            elif SECONDARY_PATTERN.match(row["id"]):
                _add(secondary, row)
    return dict(primary), dict(secondary)


def load_holdout(path: Path) -> Pool:
    """The held-out pool: every row of the file for the 9 ref, whatever the prompt id."""
    seen: set[tuple[str, str, int]] = set()
    pool: Pool = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["endpoint"], row["id"], int(row["repeat"]))
        if key in seen or row["endpoint"] not in TRUE_MODEL:
            continue
        seen.add(key)
        _add(pool, row)
    return dict(pool)


def common_prompts(pool: Pool, endpoints: Sequence[str]) -> list[str]:
    """Prompt ids answered by every listed endpoint, sorted."""
    sets = [set(pool.get(e, {})) for e in endpoints]
    return sorted(set.intersection(*sets)) if sets else []


def complete_prompts(pool: Pool, endpoints: Sequence[str], repeats: int) -> list[str]:
    """Prompt ids with at least `repeats` answers from EVERY listed endpoint, sorted by id."""
    return [p for p in common_prompts(pool, endpoints)
            if all(len(pool[e][p]) >= repeats for e in endpoints)]


def snapshot_pool(pool: Pool, endpoints: Sequence[str], first: int, repeats: int = 2) -> Pool:
    """The snapshot of the first `first` complete prompts (id order), all draws kept.

    Decision of the coordinator, taken before any result: the final test runs on the instant
    snapshot of the first 25 complete prompts (9 ref x 2 repeats), not on a file that keeps
    growing. The chain itself is unchanged.

    Raises:
        RehearsalError: fewer than `first` complete prompts exist yet.
    """
    complete = complete_prompts(pool, endpoints, repeats)
    if len(complete) < first:
        raise RehearsalError(f"{len(complete)} complete prompts, need {first}")
    keep = set(complete[:first])
    return {e: {p: list(a) for p, a in prompts.items() if p in keep} for e, prompts in pool.items()}


def write_snapshot(holdout_path: Path, out: Path, first: int, repeats: int = 2) -> tuple[int, str]:
    """Write the raw rows of the first `first` complete prompts to `out`; return `(rows, sha256)`.

    Uses the same selection as `snapshot_pool`, so the committed snapshot is exactly what the final
    test reads. Rows keep the file order; nothing is modified.
    """
    import hashlib

    kept = set(complete_prompts(snapshot_pool(load_holdout(holdout_path), REF_ENDPOINTS, first,
                                              repeats), REF_ENDPOINTS, repeats))
    lines = []
    seen: set[tuple[str, str, int]] = set()
    for line in holdout_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["endpoint"], row["id"], int(row["repeat"]))
        if row["id"] in kept and row["endpoint"] in TRUE_MODEL and key not in seen:
            seen.add(key)
            lines.append(line)
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return len(lines), hashlib.sha256(text.encode("utf-8")).hexdigest()


def true_partition(endpoints: Sequence[str]) -> Partition:
    """The known partition: one family, models S / fp8 / LoRA 0.3, 0.7, 1.5."""
    labels = sorted({TRUE_MODEL[e] for e in endpoints})
    model = np.array([labels.index(TRUE_MODEL[e]) for e in endpoints], dtype=np.int64)
    return Partition(np.zeros(len(endpoints), dtype=np.int64), model)


def decision_chain(rates: Sequence[float], rule: Rule) -> Partition:
    """Band rule then grouping recipe over endpoints in the order of `rates`."""
    model = np.zeros(len(rates), dtype=np.int64)
    next_id = 1
    for position, rate in enumerate(rates):
        if rule.band_low <= rate <= rule.band_high:
            model[position] = next_id
            next_id += 1
    return Partition(np.zeros(len(rates), dtype=np.int64), model)


def official_score(truth: Partition, guess: Partition) -> float:
    """`0.1 x MCC_family + 0.9 x MCC_model` over all pairs (family MCC is 0 on one family)."""
    index = pair_index(truth.size)
    return float(full_score(truth.states(index), guess.states(index)))


@dataclass(frozen=True)
class Rehearsal:
    """Bootstrap summary of one (pool, N, endpoints) cell."""

    pool: str
    n: int
    endpoints: tuple[str, ...]
    scores: FloatArray
    flag_frequency: FloatArray
    threshold_x: float

    @property
    def median(self) -> float:
        """Median official score."""
        return float(np.median(self.scores))

    @property
    def interval(self) -> tuple[float, float]:
        """10th and 90th percentiles (80 % interval)."""
        low, high = np.percentile(self.scores, PERCENTILES)
        return float(low), float(high)

    @property
    def probability_above(self) -> float:
        """Share of draws whose score reaches the decision threshold."""
        return float(np.mean(self.scores >= self.threshold_x))


def bootstrap_scores(
    pool: Pool, endpoints: Sequence[str], n: int, rule: Rule, replicates: int,
    rng: np.random.Generator, pool_name: str = "pool",
) -> Rehearsal:
    """Draw `n` common prompts and one answer per (endpoint, prompt), `replicates` times.

    Raises:
        RehearsalError: fewer than `n` prompts are answered by every endpoint.
    """
    prompts = common_prompts(pool, endpoints)
    if len(prompts) < n:
        raise RehearsalError(f"{pool_name}: {len(prompts)} common prompts, need {n}")
    stops = {
        e: {p: np.array([ends_with_full_stop(a) for a in pool[e][p]], dtype=np.float64)
            for p in prompts}
        for e in endpoints
    }
    truth = true_partition(endpoints)
    scores = np.empty(replicates)
    flags = np.zeros(len(endpoints))
    for replicate in range(replicates):
        chosen = rng.choice(len(prompts), size=n, replace=False)
        rates = []
        for e in endpoints:
            hits = [float(rng.choice(stops[e][prompts[int(c)]])) for c in chosen]
            rates.append(float(np.mean(hits)))
        scores[replicate] = official_score(truth, decision_chain(rates, rule))
        flags += np.array([rule.band_low <= r <= rule.band_high for r in rates], dtype=np.float64)
    return Rehearsal(pool_name, n, tuple(endpoints), scores, flags / replicates, rule.threshold_x)


def verdict(median_at_10: float, rule: Rule) -> bool:
    """The frozen rule: spend the +10 iff the median at N = 10 reaches the threshold."""
    return median_at_10 >= rule.threshold_x


def summarise(cell: Rehearsal) -> dict[str, object]:
    """JSON-friendly summary of a cell."""
    return {
        "pool": cell.pool, "n": cell.n, "endpoints": list(cell.endpoints), "median": cell.median,
        "p10": cell.interval[0], "p90": cell.interval[1],
        "probability_at_or_above_X": cell.probability_above,
        "flag_frequency": {e: float(f) for e, f in zip(cell.endpoints, cell.flag_frequency,
                                                       strict=True)},
    }


def run_dev(
    store_dir: Path, band_lows: Sequence[float], replicates: int = REPLICATES,
) -> list[dict[str, object]]:
    """Development grid on the existing ref data: every band edge x pool x N (all endpoints)."""
    primary, secondary = load_pools(store_dir)
    secondary_endpoints = tuple(e for e in REF_ENDPOINTS if e in secondary)
    rows: list[dict[str, object]] = []
    for low in band_lows:
        rule = Rule(low)
        rng = np.random.default_rng(SEED)
        for pool_name, pool, endpoints, sizes in (
            ("primary", primary, REF_ENDPOINTS, (10, 16)),
            ("secondary", secondary, secondary_endpoints, (10, 20)),
        ):
            for n in sizes:
                cell = bootstrap_scores(pool, endpoints, n, rule, replicates, rng, pool_name)
                rows.append({"band_low": low, **summarise(cell)})
    return rows


def run_holdout(
    holdout_path: Path, rule: Rule, sizes: Sequence[int] = (10, 20),
    replicates: int = REPLICATES, first_prompts: int | None = None,
) -> dict[str, object]:
    """The final test: the frozen rule on the held-out pool, with and without temperature copies.

    `first_prompts` restricts the pool to the snapshot of the first complete prompts.
    """
    pool = load_holdout(holdout_path)
    if first_prompts is not None:
        pool = snapshot_pool(pool, REF_ENDPOINTS, first_prompts)
    rng = np.random.default_rng(SEED)
    without = tuple(e for e in REF_ENDPOINTS if e != TEMPERATURE_REF)
    cells = []
    for label, endpoints in (("all_ref", REF_ENDPOINTS), ("without_temperature_ref", without)):
        for n in sizes:
            cell = bootstrap_scores(pool, endpoints, n, rule, replicates, rng, label)
            cells.append(summarise(cell))
    first = next(c for c in cells if c["pool"] == "all_ref" and c["n"] == sizes[0])
    return {
        "rule": {"band_low": rule.band_low, "band_high": rule.band_high,
                 "threshold_x": rule.threshold_x},
        "replicates": replicates, "seed": SEED,
        "holdout_prompts": len(common_prompts(pool, REF_ENDPOINTS)), "cells": cells,
        "verdict_spend_plus_10": verdict(float(first["median"]), rule),  # type: ignore[arg-type]
    }


def main() -> None:
    """CLI: `dev` (tuning grid) or `holdout` (frozen chain on the held-out file)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    dev = sub.add_parser("dev")
    dev.add_argument("--store-dir", type=Path, required=True)
    dev.add_argument("--out", type=Path, required=True)
    dev.add_argument("--band-low", type=float, action="append", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--holdout", type=Path, required=True)
    snap.add_argument("--first-prompts", type=int, required=True)
    snap.add_argument("--out", type=Path, required=True)
    hold = sub.add_parser("holdout")
    hold.add_argument("--holdout", type=Path, required=True)
    hold.add_argument("--band-low", type=float, required=True)
    hold.add_argument("--threshold", type=float, required=True)
    hold.add_argument("--first-prompts", type=int, default=None,
                      help="use the snapshot of the first K complete prompts (coordinator decision: 25)")
    hold.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        row_count, digest = write_snapshot(args.holdout, args.out, args.first_prompts)
        print(f"snapshot: {row_count} rows, sha256 {digest}")
        return
    if args.mode == "dev":
        rows = run_dev(args.store_dir, args.band_low)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        for row in rows:
            print(f"low {row['band_low']:.2f} {row['pool']:9s} N={row['n']:2d} median "
                  f"{row['median']:.2f} 80% [{row['p10']:.2f}, {row['p90']:.2f}]")
        return
    payload = run_holdout(args.holdout, Rule(args.band_low, threshold_x=args.threshold),
                          first_prompts=args.first_prompts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for cell in payload["cells"]:  # type: ignore[attr-defined]
        print(f"{cell['pool']:25s} N={cell['n']:2d} median {cell['median']:.2f} "
              f"80% [{cell['p10']:.2f}, {cell['p90']:.2f}] P(>=X) {cell['probability_at_or_above_X']:.2f}")
    print("verdict (spend +10):", payload["verdict_spend_plus_10"])


if __name__ == "__main__":
    main()
