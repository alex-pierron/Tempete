"""Pair algebra for the team-C score analysis: partitions, pair states, exact official score.

The official score only sees PAIRS. Every pair of endpoints is in one of three truth states:

* `STATE_CROSS_FAMILY` (0): different families (never the same model, excluded from MCC_model);
* `STATE_SAME_FAMILY` (1): same family, different model;
* `STATE_SAME_MODEL` (2): same family and same model.

A submission defines the same three states for its own labelling. Everything the probing plan
needs (score of a candidate against a hypothesis, information of a public score) is a function of
the 3 x 3 cross-tabulation of truth state against submitted state, which `c_oracle` builds from
the arrays defined here. Pair order is `itertools.combinations(range(size), 2)`, the order of
`scoring.official_score`, so results can be cross-checked against it.
"""

from __future__ import annotations

import csv
import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ENDPOINT_COUNT = 33
ENDPOINT_PREFIX = "ep-"
STATE_CROSS_FAMILY = 0
STATE_SAME_FAMILY = 1
STATE_SAME_MODEL = 2
STATE_COUNT = 3
SUBMISSION_HEADER = ("endpoint", "family", "model")

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
StateArray = npt.NDArray[np.uint8]


class PartitionError(ValueError):
    """A labelling is malformed (wrong length, model label shared across families...)."""


def endpoint_names(size: int = ENDPOINT_COUNT) -> tuple[str, ...]:
    """`ep-01 ... ep-<size>`, the server's endpoint ids."""
    return tuple(f"{ENDPOINT_PREFIX}{k + 1:02d}" for k in range(size))


@dataclass(frozen=True)
class PairIndex:
    """First / second endpoint position of every unordered pair, in `combinations` order."""

    size: int
    first: IntArray
    second: IntArray

    @property
    def count(self) -> int:
        """Number of unordered pairs (528 for 33 endpoints)."""
        return int(self.first.shape[0])


@functools.lru_cache(maxsize=8)
def pair_index(size: int = ENDPOINT_COUNT) -> PairIndex:
    """The (cached) pair index of `size` endpoints."""
    first, second = np.triu_indices(size, k=1)
    return PairIndex(size=size, first=first.astype(np.int64), second=second.astype(np.int64))


def same_label_pairs(labels: IntArray, index: PairIndex) -> BoolArray:
    """Boolean "same label" per pair, for labels of shape `[..., size]` (batch friendly)."""
    result: BoolArray = labels[..., index.first] == labels[..., index.second]
    return result


def pair_states(family: IntArray, model: IntArray, index: PairIndex) -> StateArray:
    """Pair state (0 cross-family, 1 same family, 2 same model) for labellings `[..., size]`.

    `model` labels must be globally unique ids (a model label shared by two families is a
    different model per family; `Partition.from_labels` builds such ids).
    """
    same_family = same_label_pairs(family, index)
    same_model = same_label_pairs(model, index) & same_family
    states: StateArray = same_family.astype(np.uint8) + same_model.astype(np.uint8)
    return states


def canonical(labels: Sequence[int] | IntArray) -> IntArray:
    """Relabel to 0..k-1 in order of first appearance (a canonical form for comparison)."""
    mapping: dict[int, int] = {}
    out = np.empty(len(labels), dtype=np.int64)
    for position, label in enumerate(labels):
        out[position] = mapping.setdefault(int(label), len(mapping))
    return out


@dataclass(frozen=True)
class Partition:
    """A two-level labelling of the endpoints: family ids and globally unique model ids.

    Invariant (checked): two endpoints with the same model id have the same family id.
    """

    family: IntArray
    model: IntArray

    def __post_init__(self) -> None:
        if self.family.shape != self.model.shape or self.family.ndim != 1:
            raise PartitionError("family and model must be 1-D arrays of the same length")
        seen: dict[int, int] = {}
        for model_id, family_id in zip(self.model.tolist(), self.family.tolist(), strict=True):
            if seen.setdefault(model_id, family_id) != family_id:
                raise PartitionError(f"model id {model_id} spans several families")

    @property
    def size(self) -> int:
        """Number of endpoints."""
        return int(self.family.shape[0])

    @staticmethod
    def from_labels(family_labels: Sequence[str], model_labels: Sequence[str]) -> Partition:
        """Build from submission-style text labels; "same model" = same (family, model) pair."""
        if len(family_labels) != len(model_labels):
            raise PartitionError("family and model label lists differ in length")
        family_ids: dict[str, int] = {}
        model_ids: dict[tuple[str, str], int] = {}
        family = [family_ids.setdefault(f, len(family_ids)) for f in family_labels]
        model = [
            model_ids.setdefault((f, m), len(model_ids))
            for f, m in zip(family_labels, model_labels, strict=True)
        ]
        return Partition(np.array(family, dtype=np.int64), np.array(model, dtype=np.int64))

    @staticmethod
    def from_groups(
        groups: Sequence[Sequence[int]], family_of_group: Sequence[int] | None = None
    ) -> Partition:
        """Build from groups of endpoint positions (each group = one model).

        Args:
            groups: Disjoint lists of endpoint positions covering `0..size-1`.
            family_of_group: Family id of each group (default: every group its own family).
        """
        size = sum(len(g) for g in groups)
        family = np.full(size, -1, dtype=np.int64)
        model = np.full(size, -1, dtype=np.int64)
        for group_id, members in enumerate(groups):
            for position in members:
                if model[position] != -1 or not 0 <= position < size:
                    raise PartitionError(f"endpoint position {position} is repeated or out of range")
                model[position] = group_id
                family[position] = group_id if family_of_group is None else family_of_group[group_id]
        if (model < 0).any():
            raise PartitionError("groups do not cover every endpoint")
        return Partition(family, model)

    def states(self, index: PairIndex | None = None) -> StateArray:
        """Pair-state vector `[pairs]` of this partition."""
        return pair_states(self.family, self.model, index or pair_index(self.size))

    def model_groups(self) -> list[list[int]]:
        """Endpoint positions of each model, ordered by first member."""
        groups: dict[int, list[int]] = {}
        for position, model_id in enumerate(self.model.tolist()):
            groups.setdefault(model_id, []).append(position)
        return list(groups.values())

    def family_groups(self) -> list[list[int]]:
        """Endpoint positions of each family, ordered by first member."""
        groups: dict[int, list[int]] = {}
        for position, family_id in enumerate(self.family.tolist()):
            groups.setdefault(family_id, []).append(position)
        return list(groups.values())

    def to_rows(self, names: Sequence[str] | None = None) -> list[tuple[str, str, str]]:
        """`(endpoint, family, model)` rows with readable labels (F1.., F1M1..)."""
        labels = names or endpoint_names(self.size)
        family_canon = canonical(self.family)
        model_canon = canonical(self.model)
        within: dict[int, dict[int, int]] = {}
        rows: list[tuple[str, str, str]] = []
        for name, f_id, m_id in zip(labels, family_canon.tolist(), model_canon.tolist(), strict=True):
            numbering = within.setdefault(f_id, {})
            number = numbering.setdefault(m_id, len(numbering) + 1)
            rows.append((name, f"F{f_id + 1}", f"F{f_id + 1}M{number}"))
        return rows

    def write_csv(self, path: Path, names: Sequence[str] | None = None) -> None:
        """Write the submission CSV (`endpoint,family,model`, header included)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(SUBMISSION_HEADER)
            writer.writerows(self.to_rows(names))


def read_partition(path: Path, names: Sequence[str] | None = None) -> Partition:
    """Read a submission CSV into a `Partition`, endpoints in `names` order."""
    order = list(names or endpoint_names())
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle) if any(row)]
    if rows and tuple(rows[0][:3]) == SUBMISSION_HEADER:
        rows = rows[1:]
    table: Mapping[str, tuple[str, str]] = {r[0]: (r[1], r[2]) for r in rows if len(r) >= 3}
    missing = [name for name in order if name not in table]
    if missing:
        raise PartitionError(f"{path.name}: no line for {missing}")
    return Partition.from_labels([table[n][0] for n in order], [table[n][1] for n in order])
