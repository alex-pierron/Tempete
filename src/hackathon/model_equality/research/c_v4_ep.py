"""Configuration v4 applied to the ep with the TRUE families: one relative band per family.

Why. `c_v4` was frozen on the ref, a single family of 9 endpoints, where the relative band edge
(`max(floor, p25 + delta)`) is computed over all of them. The ep hold three base models (r-met-e's
identification by likelihood: Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B, 11 endpoints each) whose
answers end with a full stop at different base rates, so the baseline MUST be computed inside each
family of 11, never over the 33 at once. On the ref the two are the same computation, which is why
the single confirmation test of v4 covers this adaptation.

Chain (frozen parameters of `preregistered_v5.json`, nothing is tuned on ep data):
1. families = `best` column of r-met-e's `identification.csv` (group `ep`);
2. rate per endpoint = share of its first N frozen arithmetic answers ending with a full stop;
3. flagged = rate >= max(floor, p25(family rates) + delta), per family;
4. identity override of `c_identity`: an endpoint whose 3 identity answers are mostly marked
   (name no base model) is disguised and is never isolated;
5. recipe per family: flagged and not disguised = its own model, every other endpoint = the main
   model of its family. The output carries the three exact families.

An endpoint with fewer than `MIN_ANSWER_SHARE` of its N arithmetic answers (failed requests) cannot
be flagged: it stays in the main model and the report says so. Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_identity import IDENTITY_PROMPT_IDS, is_disguised
from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import ends_with_full_stop
from hackathon.model_equality.research.c_v4 import RuleV4, flag_endpoints

FROZEN_DELTA = 0.20
FROZEN_FLOOR = 0.45
FROZEN_N = 15
MIN_ANSWER_SHARE = 0.8
EP_GROUP = "ep"
FAMILY_COLUMN = "best"


class EpChainError(ValueError):
    """The families file or the answers do not allow the ep chain to run."""


@dataclass(frozen=True)
class EpDecision:
    """Everything the chain decided, for the report and the tests."""

    endpoints: tuple[str, ...]
    family_of: tuple[int, ...]
    family_names: tuple[str, ...]
    rates: tuple[float, ...]
    answer_counts: tuple[int, ...]
    disguised: tuple[bool, ...]
    flagged: tuple[bool, ...]
    isolated: tuple[bool, ...]
    thresholds: tuple[float, ...]
    partition: Partition

    def summary(self) -> dict[str, object]:
        """JSON-friendly summary (no answer text)."""
        rows = [
            {"endpoint": e, "family": self.family_names[f], "rate": r, "answers": c,
             "disguised": d, "flagged": g, "isolated": i}
            for e, f, r, c, d, g, i in zip(
                self.endpoints, self.family_of, self.rates, self.answer_counts, self.disguised,
                self.flagged, self.isolated, strict=True)
        ]
        return {
            "families": {name: {"threshold": t} for name, t in
                         zip(self.family_names, self.thresholds, strict=True)},
            "isolated_count": int(sum(self.isolated)),
            "disguised_count": int(sum(self.disguised)),
            "models": len(self.partition.model_groups()),
            "endpoints": rows,
        }


def read_families(path: Path, endpoints: Sequence[str]) -> tuple[list[int], list[str]]:
    """`(family index per endpoint, family names)` from r-met-e's identification file.

    Families are numbered in order of first appearance along `endpoints`.

    Raises:
        EpChainError: an endpoint is missing from the file.
    """
    best: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["group"] == EP_GROUP:
                best[row["endpoint"]] = row[FAMILY_COLUMN]
    missing = [e for e in endpoints if e not in best]
    if missing:
        raise EpChainError(f"{path.name}: no family for {missing}")
    names: list[str] = []
    for endpoint in endpoints:
        if best[endpoint] not in names:
            names.append(best[endpoint])
    return [names.index(best[e]) for e in endpoints], names


def read_answers(
    path: Path, endpoints: Sequence[str], wanted_ids: Sequence[str]
) -> dict[tuple[str, str], list[str]]:
    """`(endpoint, prompt id) -> answers`, duplicates `(endpoint, id, repeat)` removed."""
    wanted = set(wanted_ids)
    known = set(endpoints)
    seen: set[tuple[str, str, int]] = set()
    out: dict[tuple[str, str], list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["endpoint"]), str(row["id"]), int(row["repeat"]))
        if key[0] in known and key[1] in wanted and key not in seen:
            seen.add(key)
            out.setdefault((key[0], key[1]), []).append(str(row["response"]))
    return out


def decide(
    endpoints: Sequence[str], family_of: Sequence[int], family_names: Sequence[str],
    answers: Mapping[tuple[str, str], Sequence[str]], arithmetic_ids: Sequence[str],
    rule: RuleV4,
) -> EpDecision:
    """Per-family band rule, identity override and recipe, from stored answers."""
    rates: list[float] = []
    counts: list[int] = []
    disguised: list[bool] = []
    for endpoint in endpoints:
        stops = [ends_with_full_stop(a) for pid in arithmetic_ids
                 for a in answers.get((endpoint, pid), [])]
        counts.append(len(stops))
        rates.append(float(np.mean(stops)) if stops else 0.0)
        disguised.append(is_disguised([a for pid in IDENTITY_PROMPT_IDS
                                       for a in answers.get((endpoint, pid), [])]))
    flagged = [False] * len(endpoints)
    thresholds: list[float] = []
    groups: list[list[int]] = []
    family_of_group: list[int] = []
    for family in range(len(family_names)):
        members = [i for i, f in enumerate(family_of) if f == family]
        flags = flag_endpoints([rates[i] for i in members], rule)
        baseline = float(np.percentile([rates[i] for i in members], rule.baseline_percentile))
        thresholds.append(max(rule.floor, baseline + rule.delta))
        for position, flag in zip(members, flags, strict=True):
            enough = counts[position] >= MIN_ANSWER_SHARE * rule.n
            flagged[position] = bool(flag and enough)
        isolated_members = [i for i in members if flagged[i] and not disguised[i]]
        main = [i for i in members if i not in isolated_members]
        if main:
            groups.append(main)
            family_of_group.append(family)
        for position in isolated_members:
            groups.append([position])
            family_of_group.append(family)
    partition = Partition.from_groups(groups, family_of_group)
    isolated = [flagged[i] and not disguised[i] for i in range(len(endpoints))]
    return EpDecision(
        tuple(endpoints), tuple(family_of), tuple(family_names), tuple(rates), tuple(counts),
        tuple(disguised), tuple(flagged), tuple(isolated), tuple(thresholds), partition)


def file_sha256(path: Path) -> str:
    """SHA-256 of a file (provenance of the families file)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """CLI: stored ep answers + families file -> CSV and JSON report."""
    from hackathon.model_equality.planning import SCORED_ENDPOINTS
    from hackathon.model_equality.research.c_plus10_run import PROMPTS_FILE, load_prompts
    from hackathon.model_equality.store import default_answers_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--n", type=int, default=FROZEN_N)
    parser.add_argument("--delta", type=float, default=FROZEN_DELTA)
    parser.add_argument("--floor", type=float, default=FROZEN_FLOOR)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    endpoints = list(SCORED_ENDPOINTS)
    family_of, names = read_families(args.identification, endpoints)
    arithmetic_ids = [pid for pid, _ in load_prompts(PROMPTS_FILE, 2)[:args.n]]
    store = args.answers or default_answers_path(reference=False)
    answers = read_answers(store, endpoints, [*arithmetic_ids, *IDENTITY_PROMPT_IDS])
    rule = RuleV4(delta=args.delta, floor=args.floor, n=args.n)
    decision = decide(endpoints, family_of, names, answers, arithmetic_ids, rule)
    decision.partition.write_csv(args.out_csv)
    payload = decision.summary()
    payload["identification_sha256"] = file_sha256(args.identification)
    payload["rule"] = {"delta": args.delta, "floor": args.floor, "n": args.n,
                       "baseline_percentile": rule.baseline_percentile}
    args.out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"families {names}; thresholds {[round(t, 3) for t in decision.thresholds]}")
    print(f"isolated {sum(decision.isolated)}, disguised {sum(decision.disguised)}, "
          f"models {len(decision.partition.model_groups())}")


if __name__ == "__main__":
    main()
