"""Team C: candidate generation and planning of the next carve-out probe (no network)."""

from __future__ import annotations

import numpy as np

from hackathon.model_equality.research.c_abc import abc_sample
from hackathon.model_equality.research.c_next_probe import candidate_carves, plan
from hackathon.model_equality.research.c_pairs import Partition, pair_index

SIZE = 33


def base_partition() -> Partition:
    """Two families of 23 and 10 endpoints, one model each (the E1 shape)."""
    family = np.array([0] * 23 + [1] * 10, dtype=np.int64)
    return Partition(family, family.copy())


def test_candidates_cover_keep_add_remove_and_suspect_pairs() -> None:
    candidates = candidate_carves(base_partition(), [16, 20], list(range(1, SIZE + 1)), [28, 29, 31])
    assert "keep:16,20" in candidates
    assert "+28" in candidates and "-16" in candidates
    assert "+28+29" in candidates and "+28+16" not in candidates
    carved = candidates["+28"].model_groups()
    assert [27] in carved  # 0-based position of endpoint 28 stands alone


def test_abc_sample_returns_only_consistent_truths() -> None:
    rng = np.random.default_rng(2)
    index = pair_index(SIZE)
    all_alone = Partition.from_groups([[i] for i in range(SIZE)]).states(index)
    kept = abc_sample(base_partition().family_groups(), frozenset(), {"z": (all_alone, 0.0)}, rng,
                      draws=20)
    assert len(kept) == 20
    none = abc_sample(base_partition().family_groups(), frozenset(),
                      {"one": (Partition.from_groups([list(range(SIZE))], [0]).states(index), 0.95)},
                      rng, draws=50)
    assert none == []


def test_plan_ranks_candidates_and_reports_the_sample_size() -> None:
    index = pair_index(SIZE)
    base = base_partition()
    observed = {"e1": (base.states(index), 0.05)}
    rows = plan(observed, base, [16], frozenset({16, 20, 28}), draws=1500, seed=3,
                priors=("organiser", "balanced"))
    assert rows[0]["candidate"] == "_posterior"
    assert float(rows[0]["expected_full"]) > 0
    assert all(float(r["mutual_bits"]) >= 0 for r in rows[1:])
    vois = [float(r["voi"]) for r in rows[1:]]
    assert vois == sorted(vois, reverse=True)
