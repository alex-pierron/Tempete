"""Posterior and next probes on the real ep data: `python -m ...c_plan_real`.

Inputs: the ep answers file, the submissions already scored (`NAME=PATH=SCORE`), optional extra
candidate CSVs. Output (in `--out`): `plan.json` (posterior marginals, best response, probe table,
known candidates scored under the posterior) and `posterior.npz` (arrays for the figures).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_data import (
    AgreementData,
    clique_components,
    load_agreement,
)
from hackathon.model_equality.research.c_long_analysis import LongForm, build_long_form
from hackathon.model_equality.research.c_pairs import (
    Partition,
    endpoint_names,
    pair_index,
    read_partition,
)
from hackathon.model_equality.research.c_particles import expected_scores
from hackathon.model_equality.research.c_real import (
    LIBRARY_EVIDENCE_POWER,
    library_hypotheses,
    log_evidence_from_data,
)
from hackathon.model_equality.research.c_search import (
    Planner,
    PlanTable,
    SearchConfig,
    initial_planner,
)

DEFAULT_SEED = 20261121
TIGHT_CLIQUE_AGREEMENT = 5
LOOSE_CLIQUE_AGREEMENT = 4
EXPAND_ROUNDS = 0
"""No greedy expansion by default: it collapses the population onto one mode (see report)."""
TOP_TABLE_ROWS = 12
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ScoredSubmission:
    """A submission already scored on the public board."""

    name: str
    partition: Partition
    score: float


def parse_observed(items: list[str]) -> list[ScoredSubmission]:
    """`NAME=PATH=SCORE` triples into scored submissions."""
    out: list[ScoredSubmission] = []
    for item in items:
        name, path, score = item.split("=")
        out.append(ScoredSubmission(name, read_partition(Path(path)), float(score)))
    return out


def structured_candidates(data: AgreementData) -> dict[str, Partition]:
    """Named candidates built from the answers: identical-answer cliques as models.

    * `tight5`: connected groups with 5 of 5 identical answers are one model each, all other
      endpoints alone (high precision, low recall);
    * `c7_only`: the largest strict clique alone as one model;
    * `tight5_llama4`: `tight5` with the Llama-like group linked at 4 of 5 (the creator-`meta`
      component) replacing its 5-of-5 part;
    * `clique4`: groups linked at 4 of 5 (large, low precision, kept as a reference).
    Each exists as `_onefam` (every endpoint in one family: score = 0.9 x MCC_model) and
    `_creator` (family = stated creator).
    """
    agreement = data.agreement_count()
    creators = np.array(data.creators)
    creator_id = {c: k for k, c in enumerate(sorted(set(creators.tolist())))}
    strict = [g for g in clique_components(agreement, TIGHT_CLIQUE_AGREEMENT) if len(g) > 1]
    largest = max(strict, key=len)
    meta = [i for i, c in enumerate(creators.tolist()) if c == "meta"]
    llama_core = max(clique_components(agreement, LOOSE_CLIQUE_AGREEMENT, meta), key=len)
    size = len(data.endpoints)

    def with_singletons(groups: list[list[int]]) -> list[list[int]]:
        taken = {i for g in groups for i in g}
        return [*groups, *[[i] for i in range(size) if i not in taken]]

    llama_groups = [g for g in strict if not set(g) <= set(llama_core)] + [llama_core]
    recipes: dict[str, list[list[int]]] = {
        "tight5": with_singletons(strict),
        "c7_only": with_singletons([largest]),
        "tight5_llama4": with_singletons(llama_groups),
        "clique4": clique_components(agreement, LOOSE_CLIQUE_AGREEMENT),
    }
    out: dict[str, Partition] = {}
    for label, groups in recipes.items():
        out[f"{label}_onefam"] = Partition.from_groups(groups, [0] * len(groups))
        families = [creator_id[str(creators[g[0]])] for g in groups]
        out[f"{label}_creator"] = Partition.from_groups(groups, families)
    return out


def partition_rows(partition: Partition) -> list[list[str]]:
    """JSON-friendly `[endpoint, family, model]` rows."""
    return [list(row) for row in partition.to_rows()]


def build_planner(
    data: AgreementData, observed: list[ScoredSubmission], config: SearchConfig, seed: int,
    power: float = LIBRARY_EVIDENCE_POWER, long_form: LongForm | None = None,
) -> Planner:
    """Planner seeded from the answers, then updated with every observed score."""
    rng = np.random.default_rng(seed)
    evidence = log_evidence_from_data(data, power=power, long_form=long_form)
    planner = initial_planner(
        evidence, library_hypotheses(data, rng, long_form=long_form), config, rng)
    planner.extras = [item.partition for item in observed]
    for item in observed:
        planner.observe(item.partition, item.score)
    for _ in range(EXPAND_ROUNDS):
        planner.expand()
    return planner


def marginals(planner: Planner) -> tuple[FloatArray, FloatArray]:
    """`(P(same family), P(same model))` as `[E, E]` matrices under the posterior weights."""
    weights = planner.particles.weights()
    index = planner.index
    states = planner.particles.states
    size = index.size
    fam = np.eye(size)
    mod = np.eye(size)
    pair_family = weights @ (states >= 1)
    pair_model = weights @ (states == 2)
    fam[index.first, index.second] = fam[index.second, index.first] = pair_family
    mod[index.first, index.second] = mod[index.second, index.first] = pair_model
    return fam, mod


def run(
    answers: Path, observed_items: list[str], extra_csv: list[str], out: Path,
    step: float | None, seed: int, population: int, power: float = LIBRARY_EVIDENCE_POWER,
    long_prompts: Sequence[str] = (), use_long_evidence: bool = False,
) -> dict[str, Any]:
    """Build the posterior and the one-step probe table; write `plan.json` and `posterior.npz`."""
    short_ids = [p for p in load_agreement(answers, drop_incomplete=True).prompts
                 if p not in set(long_prompts)]
    data = load_agreement(answers, prompts=short_ids)
    long_form = None
    if long_prompts and use_long_evidence:
        long_data = load_agreement(answers, prompts=list(long_prompts), drop_incomplete=True)
        long_form = build_long_form(long_data, list(long_prompts)) if long_data.prompts else None
    observed = parse_observed(observed_items)
    config = SearchConfig(step=step, population=population)
    planner = build_planner(data, observed, config, seed, power, long_form)
    named = structured_candidates(data)
    for item in extra_csv:
        name, path = item.split("=")
        named[name] = read_partition(Path(path))
    planner.extras = [*planner.extras, *named.values()]
    table: PlanTable = planner.plan_table()
    index = pair_index(len(data.endpoints))
    known = {item.name: item.partition for item in observed} | named
    known_states = np.stack([p.states(index) for p in known.values()])
    known_expected, _ = expected_scores(planner.particles, known_states)
    fam, mod = marginals(planner)
    order = np.argsort(table.voi)[::-1][:TOP_TABLE_ROWS]
    payload: dict[str, Any] = {
        "endpoints": list(endpoint_names(len(data.endpoints))),
        "step": step,
        "effective_sample_size": planner.particles.effective_size(),
        "baseline_expected_score": table.baseline,
        "best_response": partition_rows(table.pool[0]),
        "best_response_expected": float(table.expected[0]),
        "known_candidates": {
            name: {"expected_final_score": float(value), "rows": partition_rows(known[name])}
            for name, value in zip(known, known_expected, strict=True)
        },
        "top_probes": [
            {"voi": float(table.voi[k]), "mutual_bits": float(table.mutual_bits[k]),
             "expected_final_score": float(table.expected[k]),
             "rows": partition_rows(table.pool[int(k)])}
            for k in order
        ],
        "observed": [{"name": o.name, "score": o.score} for o in observed],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    np.savez_compressed(
        out / "posterior.npz", family=planner.particles.family, model=planner.particles.model,
        weights=planner.particles.weights(), p_same_family=fam, p_same_model=mod,
    )
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--observed", action="append", default=[], help="NAME=PATH=SCORE")
    parser.add_argument("--extra", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population", type=int, default=1500)
    parser.add_argument("--power", type=float, default=LIBRARY_EVIDENCE_POWER)
    parser.add_argument("--long-prompt", action="append", default=[],
                        help="id of a long prompt (kept out of the short-prompt evidence)")
    parser.add_argument("--long-evidence", action="store_true",
                        help="use identical-count / prefix of the long answers as evidence. OFF by "
                        "default: the ep endpoints sample, so two same-model endpoints do not give "
                        "identical long texts and the identical-count is uninformative")
    args = parser.parse_args()
    payload = run(args.answers, args.observed, args.extra, args.out, args.step, args.seed,
                  args.population, args.power, args.long_prompt, args.long_evidence)
    print(f"ESS {payload['effective_sample_size']:.0f}; baseline expected score "
          f"{payload['baseline_expected_score']:.3f}")
    for name, info in payload["known_candidates"].items():
        print(f"  {name:18s} expected final score {info['expected_final_score']:.3f}")
    for row in payload["top_probes"][:6]:
        print(f"  probe voi {row['voi']:.3f} mi {row['mutual_bits']:.2f} bits "
              f"expected {row['expected_final_score']:.3f}")


if __name__ == "__main__":
    main()
