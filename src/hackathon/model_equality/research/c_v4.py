"""Configuration v4: v3 (identity override) with a RELATIVE band edge and a larger N.

Why. The v3 test showed the override works (system prompt and persona declared disguised in 100 %
of the draws, no LoRA) but the median of the official score sits on a knife edge because LoRA 0.3,
whose full-stop rate is 60 to 72 %, is flagged only about half of the time at the band edge 0.55.
With the system-prompt false positive gone, a lower edge and a larger N can be used. An ABSOLUTE
edge does not transfer to the ep (Qwen and Llama endpoints end answers with a full stop at other
base rates), so the edge is RELATIVE to the family: an endpoint is flagged when its rate exceeds the
median rate of its family by at least `delta`, and is at least `floor` (a protection for families
whose copies almost never end with a full stop). On the ref (one family, copies at about 20 %) it
is the same as an absolute edge of about 0.45.

Chain: N arithmetic prompts, one answer each -> rate per endpoint -> relative band rule -> identity
override of `c_identity` (a disguised endpoint is never isolated) -> each flagged endpoint is its
own model, every other one is the main model. Scoring and sampling are those of `c_rehearsal`.
Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_identity import is_disguised, load_identity
from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import (
    REF_ENDPOINTS,
    SEED,
    TEMPERATURE_REF,
    Pool,
    RehearsalError,
    common_prompts,
    complete_prompts,
    ends_with_full_stop,
    load_holdout,
    official_score,
    true_partition,
)

FloatArray = npt.NDArray[np.float64]
TARGET = 0.60
REPLICATES = 1000
PRIMARY_N = 15
SECONDARY_SIZES: tuple[int, ...] = (10, 20)
IDENTITY_UNITS = 3
CONFIRMATION_FIRST_PROMPT = 25
"""The confirmation set is the prompts AFTER the first 25 (ids in order), never seen before the freeze."""
CONFIRMATION_PROMPTS = 25


@dataclass(frozen=True)
class RuleV4:
    """Frozen parameters: relative edge `delta`, absolute `floor`, N prompts, identity units."""

    delta: float = 0.25
    floor: float = 0.40
    baseline_percentile: float = 25.0
    n: int = PRIMARY_N
    k: int = IDENTITY_UNITS
    target: float = TARGET


def flag_endpoints(rates: Sequence[float], rule: RuleV4) -> list[bool]:
    """Flag the endpoints whose rate reaches `max(floor, baseline + delta)`.

    The baseline is the `baseline_percentile` of the family's rates (the lower quartile by
    default): robust to the large share of high-rate endpoints (variants, disguised copies) that
    makes the median a poor reference.
    """
    baseline = float(np.percentile(rates, rule.baseline_percentile))
    threshold = max(rule.floor, baseline + rule.delta)
    return [rate >= threshold for rate in rates]


def partition_v4(rates: Sequence[float], disguised: Sequence[bool], rule: RuleV4) -> Partition:
    """Relative band rule, then the identity override, then the grouping recipe."""
    flagged = flag_endpoints(rates, rule)
    model = np.zeros(len(rates), dtype=np.int64)
    next_id = 1
    for position, (flag, disguise) in enumerate(zip(flagged, disguised, strict=True)):
        if flag and not disguise:
            model[position] = next_id
            next_id += 1
    return Partition(np.zeros(len(rates), dtype=np.int64), model)


@dataclass(frozen=True)
class CellV4:
    """Bootstrap of one (endpoints, N) cell."""

    n: int
    endpoints: tuple[str, ...]
    scores: FloatArray
    isolated: FloatArray
    disguised: FloatArray
    target: float

    def summary(self) -> dict[str, object]:
        """JSON-friendly summary."""
        low, high = np.percentile(self.scores, (10.0, 90.0))
        return {
            "n": self.n, "endpoints": list(self.endpoints), "median": float(np.median(self.scores)),
            "mean": float(self.scores.mean()), "p10": float(low), "p90": float(high),
            "probability_at_or_above_target": float(np.mean(self.scores >= self.target)),
            "isolated": {e: float(f) for e, f in zip(self.endpoints, self.isolated, strict=True)},
            "declared_disguised": {e: float(f) for e, f in zip(self.endpoints, self.disguised, strict=True)},
        }


def bootstrap_v4(
    arithmetic: Pool, identity: Mapping[str, Sequence[str]], endpoints: Sequence[str],
    rule: RuleV4, replicates: int, rng: np.random.Generator,
) -> CellV4:
    """Draw `rule.n` common prompts (one answer each) and `rule.k` identity units, `replicates` times.

    Raises:
        RehearsalError: not enough common prompts or identity answers.
    """
    prompts = common_prompts(arithmetic, endpoints)
    if len(prompts) < rule.n:
        raise RehearsalError(f"{len(prompts)} common prompts, need {rule.n}")
    for endpoint in endpoints:
        if len(identity.get(endpoint, [])) < rule.k:
            raise RehearsalError(f"{endpoint}: not enough identity answers")
    stops = {e: {p: np.array([ends_with_full_stop(a) for a in arithmetic[e][p]], dtype=np.float64)
                 for p in prompts} for e in endpoints}
    truth = true_partition(endpoints)
    size = len(endpoints)
    scores = np.empty(replicates)
    isolated = np.zeros(size)
    disguised = np.zeros(size)
    for replicate in range(replicates):
        chosen = rng.choice(len(prompts), size=rule.n, replace=False)
        rates = [float(np.mean([float(rng.choice(stops[e][prompts[int(c)]])) for c in chosen]))
                 for e in endpoints]
        flags = [is_disguised([identity[e][int(i)] for i in
                               rng.choice(len(identity[e]), size=rule.k, replace=False)])
                 for e in endpoints]
        guess = partition_v4(rates, flags, rule)
        scores[replicate] = official_score(truth, guess)
        isolated += np.array([int(guess.model[i]) != 0 for i in range(size)], dtype=np.float64)
        disguised += np.array(flags, dtype=np.float64)
    return CellV4(rule.n, tuple(endpoints), scores, isolated / replicates, disguised / replicates,
                  rule.target)


def confirmation_pool(holdout_path: Path, repeats: int = 2) -> Pool:
    """Prompts 26 to 50 of the held-out file: complete prompts in id order after the first 25."""
    pool = load_holdout(holdout_path)
    complete = complete_prompts(pool, REF_ENDPOINTS, repeats)
    keep = set(complete[CONFIRMATION_FIRST_PROMPT:CONFIRMATION_FIRST_PROMPT + CONFIRMATION_PROMPTS])
    if len(keep) < CONFIRMATION_PROMPTS:
        raise RehearsalError(f"{len(keep)} confirmation prompts complete, need {CONFIRMATION_PROMPTS}")
    return {e: {p: list(a) for p, a in prompts.items() if p in keep} for e, prompts in pool.items()}


def run_confirmation(
    holdout_path: Path, identity_path: Path, rule: RuleV4, replicates: int = REPLICATES,
) -> dict[str, object]:
    """THE ONE TEST: the frozen configuration on the confirmation prompts 26-50.

    Primary cell: N = `rule.n`, 9 ref. Reported: N in `SECONDARY_SIZES`, and without ref-08.
    Verdict: the median of the primary cell reaches `rule.target`.
    """
    arithmetic = confirmation_pool(holdout_path)
    identity = load_identity(identity_path)
    rng = np.random.default_rng(SEED)
    without = tuple(e for e in REF_ENDPOINTS if e != TEMPERATURE_REF)
    cells = []
    for label, endpoints in (("all_ref", REF_ENDPOINTS), ("without_temperature_ref", without)):
        for n in (rule.n, *SECONDARY_SIZES):
            cell = bootstrap_v4(arithmetic, identity, endpoints, RuleV4(
                rule.delta, rule.floor, rule.baseline_percentile, n, rule.k, rule.target),
                replicates, rng)
            summary = cell.summary()
            summary["pool"] = label
            cells.append(summary)
    primary = next(c for c in cells if c["pool"] == "all_ref" and c["n"] == rule.n)
    return {
        "rule": {"delta": rule.delta, "floor": rule.floor,
                 "baseline_percentile": rule.baseline_percentile, "n": rule.n, "k": rule.k,
                 "target": rule.target},
        "confirmation_prompts": len(common_prompts(arithmetic, REF_ENDPOINTS)),
        "cells": cells,
        "verdict_median_reaches_target": float(primary["median"]) >= rule.target,  # type: ignore[arg-type]
    }


def main() -> None:
    """CLI: run the confirmation test on a held-out file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_confirmation(args.holdout, args.identity, RuleV4(args.delta, args.floor, args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for cell in payload["cells"]:  # type: ignore[attr-defined]
        print(f"{cell['pool']:25s} N={cell['n']:2d} median {cell['median']:.2f} "
              f"80% [{cell['p10']:.2f}, {cell['p90']:.2f}] P(>=0.6) {cell['probability_at_or_above_target']:.2f}")
    print("verdict (median reaches 0.60):", payload["verdict_median_reaches_target"])


if __name__ == "__main__":
    main()
