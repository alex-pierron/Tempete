"""Tests of the refined (second-generation) plan."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from hackathon.unlearning.research.retain_proxy.concepts import (
    FULLY_FORGOTTEN_CLASSES,
    LARGE_PROBE_NOUNS,
    LARGE_PROBE_NOUNS_2,
    PROBE_VOCABULARY_2,
    SMALL_PROBE_NOUNS,
    SMALL_PROBE_NOUNS_2,
)
from hackathon.unlearning.research.retain_proxy.plan import build_plan
from hackathon.unlearning.research.retain_proxy.refine import (
    REFINED_SEED_BASE,
    RefineError,
    build_refined_plan,
    reliable_concepts,
)


def _concept(
    key: str, top_class: int, share: float = 0.9, confidence: float = 0.9
) -> dict[str, Any]:
    return {
        "concept_key": key,
        "top_class": top_class,
        "top_share": share,
        "mean_confidence": confidence,
    }


def _analysis_for(plan_keys: dict[str, int]) -> dict[str, Any]:
    return {"concepts": [_concept(key, cls) for key, cls in plan_keys.items()]}


def test_second_vocabulary_is_disjoint_from_the_first() -> None:
    first = set(SMALL_PROBE_NOUNS) | set(LARGE_PROBE_NOUNS)
    second = [noun for noun, _ in PROBE_VOCABULARY_2]
    assert len(second) == len(set(second)), "duplicate noun in vocabulary 2"
    assert not first & set(second)
    assert 0 < len(second) <= len(SMALL_PROBE_NOUNS_2) + len(LARGE_PROBE_NOUNS_2)


def test_reliable_concepts_filters_and_sorts() -> None:
    analysis = {
        "concepts": [
            _concept("a", 3, share=0.6),
            _concept("b", 3, share=0.9),
            _concept("c", 4, share=0.4),  # too few agreeing images
            _concept("d", 5, confidence=0.3),  # not confident enough
        ]
    }
    reliable = reliable_concepts(analysis)
    assert list(reliable) == [3]
    assert [c["concept_key"] for c in reliable[3]] == ["b", "a"]


def test_refined_plan_budgets_per_class_and_never_reuses_seeds() -> None:
    previous = build_plan(known_per_concept=2, probe_per_noun=0)
    forgotten = FULLY_FORGOTTEN_CLASSES[0]  # class 0: concepts class00#0..2
    kept = "class01#0", "class01#1"
    analysis = _analysis_for({"class00#0": forgotten, kept[0]: 1, kept[1]: 1})
    plan = build_refined_plan(
        analysis, previous, per_class=10, forgotten_per_class=4, probe2_per_noun=1
    )
    counts = Counter(entry.class_hint for entry in plan if entry.kind == "known")
    assert counts[1] == 10 and counts[forgotten] == 4
    assert [entry.index for entry in plan] == list(range(len(plan)))
    assert min(entry.seed for entry in plan) >= REFINED_SEED_BASE
    assert len({entry.seed for entry in plan}) == len(plan)
    probes = [entry for entry in plan if entry.kind == "probe"]
    assert len(probes) == len(PROBE_VOCABULARY_2) and all(e.class_hint is None for e in probes)
    assert all(entry.concept in entry.prompt for entry in plan)


def test_refined_plan_needs_a_reliable_concept() -> None:
    with pytest.raises(RefineError):
        build_refined_plan({"concepts": [_concept("x", 1, share=0.1)]}, build_plan())
