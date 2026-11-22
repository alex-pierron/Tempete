"""Second-generation plan: spend the images where the first run showed the original reacts.

The first run (`plan.build_plan`) tells, concept by concept, which class the original gives to the
images (`analysis.json`). The refined plan keeps only the concepts with a reliable answer, spreads a
fixed image budget over the classes they reach (more for retained classes, fewer for the nine
fully forgotten ones, which are only kept for evaluation) and appends a second discovery
vocabulary aimed at the classes still unreached.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from hackathon.common.errors import HackathonError
from hackathon.unlearning.research.retain_proxy.analyze import (
    DISCOVERY_MIN_CONFIDENCE,
    DISCOVERY_MIN_SHARE,
)
from hackathon.unlearning.research.retain_proxy.concepts import (
    FULLY_FORGOTTEN_CLASSES,
    PROBE_VOCABULARY_2,
    Scale,
)
from hackathon.unlearning.research.retain_proxy.plan import (
    KNOWN_KIND,
    PROBE_KIND,
    PromptEntry,
    compose_prompt,
    read_plan,
    write_plan,
)

DEFAULT_PER_CLASS = 150
DEFAULT_FORGOTTEN_PER_CLASS = 40
DEFAULT_PROBE2_PER_NOUN = 4
MAX_CONCEPTS_PER_CLASS = 6
"""Best concepts kept per class: enough variety, not a long tail of weak ones."""
REFINED_SEED_BASE = 1_000_000
"""Seeds of the refined run never collide with the first run's (which start at 0)."""


class RefineError(HackathonError):
    """The first-run analysis has no reliable concept."""


def reliable_concepts(analysis: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Class -> its reliable concepts (best share first), from an `analysis.json` payload."""
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for concept in analysis["concepts"]:
        if (
            concept["top_share"] >= DISCOVERY_MIN_SHARE
            and concept["mean_confidence"] >= DISCOVERY_MIN_CONFIDENCE
        ):
            by_class[int(concept["top_class"])].append(concept)
    for concepts in by_class.values():
        concepts.sort(key=lambda c: (-c["top_share"], -c["mean_confidence"], c["concept_key"]))
    return dict(sorted(by_class.items()))


def build_refined_plan(
    analysis: Mapping[str, Any],
    previous_plan: Sequence[PromptEntry],
    *,
    per_class: int = DEFAULT_PER_CLASS,
    forgotten_per_class: int = DEFAULT_FORGOTTEN_PER_CLASS,
    probe2_per_noun: int = DEFAULT_PROBE2_PER_NOUN,
    base_seed: int = REFINED_SEED_BASE,
) -> list[PromptEntry]:
    """Refined plan: reliable concepts of the first run, budgeted per class, plus vocabulary 2.

    Raises:
        RefineError: no concept of the analysis is reliable.
    """
    concept_of_key = {entry.concept_key: entry for entry in previous_plan}
    reliable = reliable_concepts(analysis)
    if not reliable:
        raise RefineError("No reliable concept in the analysis: nothing to refine.")
    entries: list[PromptEntry] = []

    def add(
        concept_key: str,
        concept: str,
        scale: Scale,
        class_hint: int | None,
        kind: Literal["known", "probe"],
        count: int,
    ) -> None:
        for _ in range(count):
            index = len(entries)
            seed = base_seed + index
            entries.append(
                PromptEntry(
                    index=index,
                    kind=kind,
                    concept_key=concept_key,
                    class_hint=class_hint,
                    concept=concept,
                    scale=scale,
                    prompt=compose_prompt(concept, scale, seed),
                    seed=seed,
                )
            )

    for class_index, concepts in reliable.items():
        chosen = concepts[:MAX_CONCEPTS_PER_CLASS]
        budget = forgotten_per_class if class_index in FULLY_FORGOTTEN_CLASSES else per_class
        each = math.ceil(budget / len(chosen))
        for reliable_concept in chosen:
            source = concept_of_key[reliable_concept["concept_key"]]
            add(
                source.concept_key,
                source.concept,
                source.scale,
                class_index,
                KNOWN_KIND,
                each,
            )
    for concept, scale in PROBE_VOCABULARY_2:
        add(f"{PROBE_KIND}2#{concept}", concept, scale, None, PROBE_KIND, probe2_per_noun)
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: refined plan from a first run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--forgotten-per-class", type=int, default=DEFAULT_FORGOTTEN_PER_CLASS)
    parser.add_argument("--probe2-per-noun", type=int, default=DEFAULT_PROBE2_PER_NOUN)
    parser.add_argument("--seed", type=int, default=REFINED_SEED_BASE)
    args = parser.parse_args(argv)
    entries = build_refined_plan(
        json.loads(args.analysis.read_text()),
        read_plan(args.previous_plan),
        per_class=args.per_class,
        forgotten_per_class=args.forgotten_per_class,
        probe2_per_noun=args.probe2_per_noun,
        base_seed=args.seed,
    )
    write_plan(entries, args.out)
    known = sum(1 for entry in entries if entry.kind == KNOWN_KIND)
    print(
        f"refined plan {args.out}: {len(entries)} images ({known} reliable-concept, {len(entries) - known} probe)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
