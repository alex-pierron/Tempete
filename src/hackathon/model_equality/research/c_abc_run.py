"""CLI of the ABC posterior predictive: `python -m ...c_abc_run --groups-from E1.csv ...`.

Writes `abc.json` (acceptance, predictive public-score levels and expected full score of each
candidate). The creator groups are the families of `--groups-from` (the E1 partition: Qwen-like and
Llama-like endpoints); `--suspects` are 1-based endpoint numbers read as likely strong variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_abc import PUBLIC_LEVELS, abc_posterior
from hackathon.model_equality.research.c_pairs import StateArray, pair_index, read_partition

DEFAULT_SEED = 20261121


def parse_states(items: list[str], with_score: bool) -> dict[str, tuple[StateArray, float]]:
    """`NAME=PATH[=SCORE]` items to `name -> (pair states, score)`."""
    index = pair_index(33)
    out: dict[str, tuple[StateArray, float]] = {}
    for item in items:
        parts = item.split("=")
        score = float(parts[2]) if with_score else float("nan")
        out[parts[0]] = (read_partition(Path(parts[1])).states(index), score)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-from", type=Path, required=True)
    parser.add_argument("--observed", action="append", default=[], help="NAME=PATH=SCORE")
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--suspects", default="", help="comma-separated endpoint numbers (1-based)")
    parser.add_argument("--suspect-probability", type=float, default=0.55)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--prior", choices=["organiser", "balanced"], default="organiser")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = read_partition(args.groups_from)
    groups = reference.family_groups()
    suspects = frozenset(int(x) - 1 for x in args.suspects.split(",") if x)
    observed = parse_states(args.observed, with_score=True)
    candidates = {k: v[0] for k, v in parse_states(args.candidate, with_score=False).items()}
    candidates.update({k: v[0] for k, v in observed.items()})
    result = abc_posterior(groups, suspects, observed, candidates, np.random.default_rng(args.seed),
                           args.draws, suspect_outlier_probability=args.suspect_probability,
                           prior=args.prior)
    payload = {
        "accepted": result.accepted, "drawn": result.drawn,
        "acceptance_rate": result.acceptance_rate, "prior": args.prior, "levels": [*PUBLIC_LEVELS, ">=0.30"],
        "suspects": sorted(suspects), "suspect_probability": args.suspect_probability,
        "candidates": {
            name: {"mean_full": float(result.mean_full[k]), "sd_full": float(result.sd_full[k]),
                   "p_above_0_05": float(result.p_above_floor[k]),
                   "levels": [float(x) for x in result.level_probabilities[k]]}
            for k, name in enumerate(result.names)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"accepted {result.accepted}/{result.drawn} ({100 * result.acceptance_rate:.1f} %)")
    for name, info in payload["candidates"].items():
        print(f"  {name:14s} E[full] {info['mean_full']:.3f} +- {info['sd_full']:.3f} "
              f"P(>0.05) {info['p_above_0_05']:.2f}")


if __name__ == "__main__":
    main()
