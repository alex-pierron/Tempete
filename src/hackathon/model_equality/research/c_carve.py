"""Carve-out probes: a base partition (one model per family) with chosen endpoints taken out.

Reading rule used by the plan. With the E1 families as base (every endpoint of a family in one
model) the model MCC is 0 by construction. Taking a set of endpoints out of their model
(`carved`, each alone, or `grouped`, together as one new model) makes the guess "different" for the
pairs between a carved endpoint and the rest: a true variant carved out adds true "different"
pairs (TN), a main-model copy carved out adds false ones (FN). The public score then reads
`0.05 + 0.9 x MCC_model` when the family MCC is the known ~0.5.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hackathon.model_equality.research.c_pairs import Partition, read_partition


def carve(
    base: Partition, carved: Sequence[int] = (), grouped: Sequence[Sequence[int]] = (),
) -> Partition:
    """Partition with the `carved` endpoints (1-based) alone and each `grouped` set as one model.

    Every other endpoint keeps the model of `base`. A carved or grouped endpoint stays in its
    family. Raises `ValueError` if an endpoint is listed twice.
    """
    listed = [*carved, *(m for group in grouped for m in group)]
    if len(set(listed)) != len(listed):
        raise ValueError("an endpoint is carved or grouped twice")
    group_of = {m: k for k, group in enumerate(grouped) for m in group}
    keys: dict[tuple[int, str, int], int] = {}
    models: list[int] = []
    for position in range(base.size):
        number = position + 1
        family = int(base.family[position])
        if number in carved:
            key = (family, "alone", position)
        elif number in group_of:
            key = (family, "group", group_of[number])
        else:
            key = (family, "base", int(base.model[position]))
        models.append(keys.setdefault(key, len(keys)))
    return Partition(base.family.copy(), np.array(models, dtype=np.int64))


def main() -> None:
    """CLI: write one CSV per `--set NAME=1,2,3` (carved) or `--group NAME=1,2/3,4`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="NAME=1,2,3 (each alone)")
    parser.add_argument("--group", action="append", default=[], help="NAME=1,2/3,4 (each group a model)")
    parser.add_argument("--carve-and-group", action="append", default=[],
                        help="NAME=alone:1,2;group:3,4/5,6")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    base = read_partition(args.base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.set:
        name, values = item.split("=")
        carve(base, [int(x) for x in values.split(",")]).write_csv(args.out_dir / f"{name}.csv")
    for item in args.group:
        name, values = item.split("=")
        group_sets = [[int(x) for x in part.split(",")] for part in values.split("/")]
        carve(base, grouped=group_sets).write_csv(args.out_dir / f"{name}.csv")
    for item in args.carve_and_group:
        name, spec = item.split("=")
        alone: list[int] = []
        groups: list[list[int]] = []
        for part in spec.split(";"):
            kind, values = part.split(":")
            if kind == "alone":
                alone = [int(x) for x in values.split(",")]
            else:
                groups = [[int(x) for x in g.split(",")] for g in values.split("/")]
        carve(base, alone, groups).write_csv(args.out_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
