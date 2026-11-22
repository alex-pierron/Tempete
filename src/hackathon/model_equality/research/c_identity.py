"""Rule v3: an endpoint that the identity question shows as DISGUISED is never isolated.

Frozen before any identity answer is read (see `preregistered_v3.json`). The v2 chain flags an
endpoint as a variant (LoRA) when at least `band_low` of its answers to the arithmetic prompts end
with a full stop. The held-out test showed that a copy with a provider system prompt is flagged
by mistake about half of the time. v3 adds one step:

1. identity answers: `k` units `(formulation, draw)` of the 3 identity formulations, drawn at
   random per endpoint;
2. an answer is MARKED when it contains none of the base-model names of `BASE_TOKENS` (the
   open-weight families the task can serve, written down before looking at any answer);
3. an endpoint is DISGUISED when MORE THAN HALF of its `k` answers are marked;
4. a disguised endpoint is never isolated: it stays in the main model, whatever its full-stop rate.

Everything else (band rule, recipe, scoring) is the v2 chain of `c_rehearsal`. Nothing here
touches the network.
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

from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import (
    REF_ENDPOINTS,
    SEED,
    TEMPERATURE_REF,
    TRUE_MODEL,
    Pool,
    RehearsalError,
    Rule,
    common_prompts,
    decision_chain,
    ends_with_full_stop,
    load_holdout,
    official_score,
    true_partition,
)

FloatArray = npt.NDArray[np.float64]
IDENTITY_PROMPT_IDS: tuple[str, ...] = ("d-ident-000", "d-ident-001", "d-ident-002")
BASE_TOKENS: tuple[str, ...] = (
    "smollm", "hugging face", "huggingface",
    "qwen", "alibaba", "tongyi",
    "llama", "meta",
    "olmo", "allen institute", "ai2",
)
"""Base-model names an undisguised endpoint can state: the reference small model (SmolLM, Hugging
Face), the Qwen and Llama families seen on the ep, and OLMo (named by the task statement)."""
DISGUISED_MARKED_SHARE = 0.5
"""Disguised iff the share of marked answers is STRICTLY above this."""
BASE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in BASE_TOKENS) + r")\b",
    re.IGNORECASE,
)
IDENTITY_FILE = "reference_answers_identity.jsonl"
ADOPTION_MEDIAN = 0.50
ADOPTION_GAIN = 0.05
ADOPTION_K = 6
LORA_ENDPOINTS: tuple[str, ...] = ("ref-04", "ref-07", "ref-09")
MAX_LORA_DISGUISED_SHARE = 0.10
K_VALUES: tuple[int, ...] = (3, 6, 9)
REPLICATES = 1000


def is_marked(answer: str) -> bool:
    """True when the identity answer names none of the base models of `BASE_TOKENS`."""
    return BASE_PATTERN.search(answer) is None


def is_disguised(answers: Sequence[str]) -> bool:
    """True when strictly more than half of the answers are marked (no answers: not disguised)."""
    if not answers:
        return False
    return sum(is_marked(a) for a in answers) / len(answers) > DISGUISED_MARKED_SHARE


def decision_chain_v3(
    rates: Sequence[float], disguised: Sequence[bool], rule: Rule
) -> Partition:
    """v2 band rule, then the override: a disguised endpoint is never isolated."""
    kept = [0.0 if flag else rate for rate, flag in zip(rates, disguised, strict=True)]
    return decision_chain(kept, rule)


def load_identity(path: Path) -> dict[str, list[str]]:
    """`endpoint -> every identity answer` (units of the 3 formulations), duplicates removed."""
    seen: set[tuple[str, str, int]] = set()
    answers: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["endpoint"], row["id"], int(row["repeat"]))
        if row["id"] in IDENTITY_PROMPT_IDS and row["endpoint"] in TRUE_MODEL and key not in seen:
            seen.add(key)
            answers[row["endpoint"]].append(row["response"])
    return dict(answers)


@dataclass(frozen=True)
class PairedResult:
    """Paired v2 / v3 bootstrap of one cell (same draws of prompts, answers and identity units)."""

    n: int
    k: int
    endpoints: tuple[str, ...]
    scores_v2: FloatArray
    scores_v3: FloatArray
    flagged_v2: FloatArray
    disguised: FloatArray
    flagged_v3: FloatArray
    threshold_x: float

    @staticmethod
    def _summary(scores: FloatArray, threshold: float) -> dict[str, float]:
        low, high = np.percentile(scores, (10.0, 90.0))
        return {"median": float(np.median(scores)), "p10": float(low), "p90": float(high),
                "probability_at_or_above_X": float(np.mean(scores >= threshold))}

    def summary(self) -> dict[str, object]:
        """JSON-friendly summary: v2, v3, paired gain, per-endpoint frequencies."""
        gain = self.scores_v3 - self.scores_v2
        low, high = np.percentile(gain, (10.0, 90.0))
        names = self.endpoints
        return {
            "n": self.n, "k": self.k, "endpoints": list(names),
            "v2": self._summary(self.scores_v2, self.threshold_x),
            "v3": self._summary(self.scores_v3, self.threshold_x),
            "paired_gain_median": float(np.median(gain)), "paired_gain_p10": float(low),
            "paired_gain_p90": float(high),
            "flagged_v2": {e: float(f) for e, f in zip(names, self.flagged_v2, strict=True)},
            "declared_disguised": {e: float(f) for e, f in zip(names, self.disguised, strict=True)},
            "isolated_v3": {e: float(f) for e, f in zip(names, self.flagged_v3, strict=True)},
        }


def bootstrap_paired(
    arithmetic: Pool, identity: Mapping[str, Sequence[str]], endpoints: Sequence[str], n: int,
    k: int, rule: Rule, replicates: int, rng: np.random.Generator,
) -> PairedResult:
    """Draw `n` common arithmetic prompts (one answer each) and `k` identity units per endpoint.

    Raises:
        RehearsalError: not enough common prompts or identity answers.
    """
    prompts = common_prompts(arithmetic, endpoints)
    if len(prompts) < n:
        raise RehearsalError(f"{len(prompts)} common arithmetic prompts, need {n}")
    for endpoint in endpoints:
        if len(identity.get(endpoint, [])) < k:
            raise RehearsalError(f"{endpoint}: {len(identity.get(endpoint, []))} identity answers, need {k}")
    stops = {e: {p: np.array([ends_with_full_stop(a) for a in arithmetic[e][p]], dtype=np.float64)
                 for p in prompts} for e in endpoints}
    truth = true_partition(endpoints)
    size = len(endpoints)
    scores2 = np.empty(replicates)
    scores3 = np.empty(replicates)
    flagged2 = np.zeros(size)
    disguised = np.zeros(size)
    isolated3 = np.zeros(size)
    for replicate in range(replicates):
        chosen = rng.choice(len(prompts), size=n, replace=False)
        rates = [float(np.mean([float(rng.choice(stops[e][prompts[int(c)]])) for c in chosen]))
                 for e in endpoints]
        flags = [is_disguised([identity[e][int(i)] for i in
                               rng.choice(len(identity[e]), size=k, replace=False)])
                 for e in endpoints]
        guess2 = decision_chain(rates, rule)
        guess3 = decision_chain_v3(rates, flags, rule)
        scores2[replicate] = official_score(truth, guess2)
        scores3[replicate] = official_score(truth, guess3)
        flagged2 += np.array([rule.band_low <= r <= rule.band_high for r in rates], dtype=np.float64)
        disguised += np.array(flags, dtype=np.float64)
        isolated3 += np.array([int(guess3.model[i]) != 0 for i in range(size)], dtype=np.float64)
    return PairedResult(n, k, tuple(endpoints), scores2, scores3, flagged2 / replicates,
                        disguised / replicates, isolated3 / replicates, rule.threshold_x)


def adoption_verdict(cell: Mapping[str, object]) -> dict[str, object]:
    """The frozen adoption rule on the (N = 10, k = ADOPTION_K, 9 ref) cell.

    Spend the +10 arithmetic + `ADOPTION_K` identity questions iff: v3 median >= 0.50; v3 median
    minus v2 median >= 0.05; no LoRA endpoint is declared disguised more than 10 % of the time.
    """
    v3 = cell["v3"]
    v2 = cell["v2"]
    assert isinstance(v3, dict) and isinstance(v2, dict)
    declared = cell["declared_disguised"]
    assert isinstance(declared, dict)
    median_ok = float(v3["median"]) >= ADOPTION_MEDIAN
    gain_ok = float(v3["median"]) - float(v2["median"]) >= ADOPTION_GAIN
    lora_ok = all(float(declared[e]) <= MAX_LORA_DISGUISED_SHARE for e in LORA_ENDPOINTS if e in declared)
    return {"median_at_least_0.50": median_ok, "gain_at_least_0.05": gain_ok,
            "no_lora_read_as_disguised": lora_ok, "adopt": median_ok and gain_ok and lora_ok}


def run_identity_test(
    snapshot: Path, identity_path: Path, rule: Rule, k_values: Sequence[int] = K_VALUES,
    n: int = 10, replicates: int = REPLICATES, first_prompts: int | None = 25,
) -> dict[str, object]:
    """The frozen evaluation: snapshot arithmetic + identity wave, 9 ref and without ref-08."""
    from hackathon.model_equality.research.c_rehearsal import snapshot_pool

    arithmetic = load_holdout(snapshot)
    if first_prompts is not None:
        arithmetic = snapshot_pool(arithmetic, REF_ENDPOINTS, first_prompts)
    identity = load_identity(identity_path)
    rng = np.random.default_rng(SEED)
    without = tuple(e for e in REF_ENDPOINTS if e != TEMPERATURE_REF)
    cells = []
    for label, endpoints in (("all_ref", REF_ENDPOINTS), ("without_temperature_ref", without)):
        for k in k_values:
            summary = bootstrap_paired(arithmetic, identity, endpoints, n, k, rule, replicates, rng).summary()
            summary["pool"] = label
            cells.append(summary)
    decisive = next(c for c in cells if c["pool"] == "all_ref" and c["k"] == ADOPTION_K)
    return {"rule": {"band_low": rule.band_low, "threshold_x": rule.threshold_x},
            "identity_answers_per_ref": {e: len(identity.get(e, [])) for e in REF_ENDPOINTS},
            "cells": cells, "adoption": adoption_verdict(decisive)}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--band-low", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_identity_test(args.snapshot, args.identity,
                                Rule(args.band_low, threshold_x=args.threshold))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for cell in payload["cells"]:  # type: ignore[attr-defined]
        print(f"{cell['pool']:25s} k={cell['k']} v2 {cell['v2']['median']:.2f} v3 {cell['v3']['median']:.2f} "
              f"gain {cell['paired_gain_median']:+.2f}")
    print("adoption:", payload["adoption"])


if __name__ == "__main__":
    main()
