"""Team C: carve-out probes keep families, isolate endpoints and group sets (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.research.c_carve import carve
from hackathon.model_equality.research.c_pairs import Partition

SIZE = 12


def base_partition() -> Partition:
    """Two families (0-7 and 8-11), one model per family."""
    family = np.array([0] * 8 + [1] * 4, dtype=np.int64)
    return Partition(family, family.copy())


def test_carved_endpoints_are_alone_and_keep_their_family() -> None:
    probe = carve(base_partition(), carved=[3, 10])
    assert probe.family.tolist() == base_partition().family.tolist()
    groups = {tuple(g) for g in probe.model_groups()}
    assert (2,) in groups and (9,) in groups  # 1-based 3 and 10 are 0-based 2 and 9
    assert len(probe.model_groups()) == 4


def test_grouped_endpoints_share_one_model_inside_their_family() -> None:
    probe = carve(base_partition(), grouped=[[1, 2], [9, 10]])
    groups = {tuple(g) for g in probe.model_groups()}
    assert (0, 1) in groups and (8, 9) in groups
    assert probe.family[0] == probe.family[1] and probe.family[8] == probe.family[9]


def test_carve_and_group_do_not_overlap() -> None:
    with pytest.raises(ValueError):
        carve(base_partition(), carved=[1], grouped=[[1, 2]])


def test_nothing_carved_returns_the_base_models() -> None:
    probe = carve(base_partition())
    assert probe.model_groups() == base_partition().model_groups()
