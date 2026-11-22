"""Choose the next carve-out probe by value of information over an ABC posterior sample.

The posterior sample is the set of truths kept by rejection under one or several priors
(`c_abc`): each kept truth reproduces every public score read so far. The candidates are edits of
the current carve set (add one endpoint, remove one, add a pair of suspects), each a CSV the queue
can take. For each candidate the planner reports the expected final score on all pairs, the
probability that its public score exceeds 0.05, the mutual information between its public score
and the truth (bits) and the one-step value of information (expected gain of the best submission
after reading it).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_abc import (
    PRIOR_BALANCED,
    PRIOR_ORGANISER,
    abc_sample,
)
from hackathon.model_equality.research.c_carve import carve
from hackathon.model_equality.research.c_oracle import (
    cross_tabulate_many,
    official_score_from_cells,
    public_score_draws,
)
from hackathon.model_equality.research.c_pairs import (
    Partition,
    StateArray,
    pair_index,
    read_partition,
)
from hackathon.model_equality.research.c_particles import ParticleSet, value_of_information

DEFAULT_SEED = 20261121
PAIR_POOL_LIMIT = 8
FLOOR_LEVEL = 0.05
STEP = 0.05


def candidate_carves(
    base: Partition, current: Sequence[int], pool: Sequence[int], pair_pool: Sequence[int] = (),
) -> dict[str, Partition]:
    """Edits of the `current` carve set (1-based endpoints): keep, add one, remove one, add a pair."""
    current_set = sorted(set(current))
    out: dict[str, Partition] = {"keep:" + ",".join(map(str, current_set)): carve(base, current_set)}
    outside = [e for e in pool if e not in current_set]
    for endpoint in outside:
        out[f"+{endpoint}"] = carve(base, sorted({*current_set, endpoint}))
    for endpoint in current_set:
        out[f"-{endpoint}"] = carve(base, [e for e in current_set if e != endpoint])
    pairs = [e for e in (pair_pool or outside) if e in outside][:PAIR_POOL_LIMIT]
    for first, second in itertools.combinations(pairs, 2):
        out[f"+{first}+{second}"] = carve(base, sorted({*current_set, first, second}))
    return out


def plan(
    observed: dict[str, tuple[StateArray, float]], base: Partition, current: Sequence[int],
    suspects: frozenset[int], draws: int, seed: int, priors: Sequence[str],
    split_options: Sequence[Sequence[int]] | None = None, anchors: frozenset[int] = frozenset(),
    candidates: Mapping[str, Partition] | None = None, floor_level: float = FLOOR_LEVEL,
) -> list[dict[str, float | str]]:
    """Rank the candidate carve edits by one-step value of information.

    `split_options` fixes how many true families each base group holds (`(1,)` per group when the
    families are known); `anchors` are endpoints known to be in the main model of their family;
    `candidates` replaces the default carve edits by an explicit set of probes; `floor_level` is
    the public score a probe must beat to count as informative (0.10 once the families are known).
    """
    rng = np.random.default_rng(seed)
    groups = base.family_groups()
    truths: list[Partition] = []
    for prior in priors:
        truths += abc_sample(groups, suspects, observed, rng, draws // len(priors), prior=prior,
                             split_options=split_options, anchors=anchors)
    if not truths:
        raise ValueError("no truth is consistent with the observations; widen the priors")
    particles = ParticleSet.from_partitions(truths)
    index = pair_index(base.size)
    probes = dict(candidates) if candidates is not None else candidate_carves(
        base, current, list(range(1, base.size + 1)), sorted(s + 1 for s in suspects))
    names = list(probes)
    states = np.stack([probes[n].states(index) for n in names])
    value = value_of_information(particles, states, states, STEP, rng)
    cells = cross_tabulate_many(particles.states, states)
    full = official_score_from_cells(cells)
    levels = public_score_draws(cells.reshape(-1, 9), 1, rng, step=STEP).reshape(cells.shape[:2])
    rows: list[dict[str, float | str]] = []
    for k, name in enumerate(names):
        rows.append({
            "candidate": name, "voi": float(value.voi[k]),
            "mutual_bits": float(value.mutual_information_bits[k]),
            "expected_full": float(full[:, k].mean()),
            "p_above_floor": float((levels[:, k] > floor_level + STEP / 2).mean()),
        })
    rows.sort(key=lambda r: -float(r["voi"]))
    rows.insert(0, {"candidate": "_posterior", "voi": 0.0, "mutual_bits": 0.0,
                    "expected_full": float(len(truths)), "p_above_floor": float(draws)})
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="E1 partition (families, 1 model each)")
    parser.add_argument("--observed", action="append", default=[], help="PATH=SCORE")
    parser.add_argument("--current", default="", help="comma-separated carved endpoints (1-based)")
    parser.add_argument("--suspects", default="", help="comma-separated suspect endpoints (1-based)")
    parser.add_argument("--draws", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = read_partition(args.base)
    index = pair_index(base.size)
    observed = {}
    for k, item in enumerate(args.observed):
        path, score = item.rsplit("=", 1)
        observed[f"o{k}"] = (read_partition(Path(path)).states(index), float(score))
    current = [int(x) for x in args.current.split(",") if x]
    suspects = frozenset(int(x) - 1 for x in args.suspects.split(",") if x)
    rows = plan(observed, base, current, suspects, args.draws, args.seed,
                (PRIOR_ORGANISER, PRIOR_BALANCED))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"posterior sample: {int(float(rows[0]['expected_full']))} truths")
    for row in rows[1:13]:
        print(f"  {row['candidate']:14s} voi {row['voi']:.3f} mi {row['mutual_bits']:.2f} bits "
              f"E[full] {row['expected_full']:.3f} P(>0.05) {row['p_above_floor']:.2f}")


if __name__ == "__main__":
    main()
