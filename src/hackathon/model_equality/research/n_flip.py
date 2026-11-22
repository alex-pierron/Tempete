"""One-flip probes: a submission CSV that differs from an already-read base by ONE endpoint.

The public score is read once per 30 minutes, so each probe changes a single endpoint of a base
whose score is known: either `--isolate ep-NN` (the endpoint leaves its model and becomes a model
of its own) or `--to-main ep-NN` (the endpoint joins the main model of its family, the largest
one). Families are never touched. The output keeps the base's labels; an isolated endpoint gets
the next free model label of its family.

Usage: python -m hackathon.model_equality.research.n_flip <base.csv> <out.csv>
           (--isolate ep-NN | --to-main ep-NN)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from hackathon.common.errors import HackathonError

COLUMNS: tuple[str, ...] = ("endpoint", "family", "model")
MODEL_INFIX = "M"


class FlipError(HackathonError):
    """The requested flip does not change the partition or names an unknown endpoint."""


def read_submission(path: Path) -> list[dict[str, str]]:
    """Rows of a submission CSV, in file order.

    Raises:
        FlipError: the header is not `endpoint,family,model`.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise FlipError(f"{path.name}: expected columns {COLUMNS}, got {reader.fieldnames}")
        return [dict(row) for row in reader]


def main_model(rows: Sequence[dict[str, str]], family: str) -> str:
    """Label of the largest model of a family.

    Raises:
        FlipError: two models tie for the largest size (the main model is then ambiguous).
    """
    sizes = Counter(row["model"] for row in rows if row["family"] == family).most_common(2)
    if len(sizes) > 1 and sizes[0][1] == sizes[1][1]:
        raise FlipError(f"{family}: no single largest model ({sizes})")
    return sizes[0][0]


def _row_of(rows: Sequence[dict[str, str]], endpoint: str) -> dict[str, str]:
    for row in rows:
        if row["endpoint"] == endpoint:
            return row
    raise FlipError(f"unknown endpoint {endpoint}")


def isolate(rows: Sequence[dict[str, str]], endpoint: str) -> list[dict[str, str]]:
    """Copy of the rows where `endpoint` is alone in a new model of its family.

    Raises:
        FlipError: unknown endpoint, or the endpoint is already alone in its model.
    """
    out = [dict(row) for row in rows]
    target = _row_of(out, endpoint)
    if sum(row["model"] == target["model"] for row in out) == 1:
        raise FlipError(f"{endpoint} is already isolated")
    family = target["family"]
    used = {row["model"] for row in out if row["family"] == family}
    index = 1
    while f"{family}{MODEL_INFIX}{index}" in used:
        index += 1
    target["model"] = f"{family}{MODEL_INFIX}{index}"
    return out


def to_main(rows: Sequence[dict[str, str]], endpoint: str) -> list[dict[str, str]]:
    """Copy of the rows where `endpoint` is in the main model of its family.

    Raises:
        FlipError: unknown endpoint, or the endpoint is already in the main model.
    """
    out = [dict(row) for row in rows]
    target = _row_of(out, endpoint)
    main = main_model(out, target["family"])
    if target["model"] == main:
        raise FlipError(f"{endpoint} is already in the main model {main}")
    target["model"] = main
    return out


def write_submission(rows: Sequence[dict[str, str]], path: Path) -> None:
    """Write the rows as a submission CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the one-flip probe and say what changed."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--isolate")
    group.add_argument("--to-main")
    args = parser.parse_args(argv)
    rows = read_submission(args.base)
    flipped = isolate(rows, args.isolate) if args.isolate else to_main(rows, args.to_main)
    write_submission(flipped, args.out)
    changed = [(a["endpoint"], a["model"], b["model"]) for a, b in zip(rows, flipped, strict=True)
               if a != b]
    print(f"{args.out.name}: changed {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
