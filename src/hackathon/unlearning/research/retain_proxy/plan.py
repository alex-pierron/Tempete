"""Prompt plan of the generated retain proxy: deterministic, shardable, resumable.

A plan is a JSON-lines file of `PromptEntry`. The GPU generator reads it, keeps the entries whose
`index % num_shards == shard` and writes one `.npz` chunk per `CHUNK_SIZE` of its entries, so a
crashed or time-boxed job resumes by skipping the chunks already on disk.

The prompt template copies what the forget images show: cinematic photographs, shallow depth of
field, golden hour or window light, objects in context, a few aerial or phone-screen views.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import DEFAULT_SEED
from hackathon.unlearning.research.retain_proxy.concepts import (
    KNOWN_CLASS_CONCEPTS,
    PROBE_VOCABULARY,
    ConceptList,
    Scale,
)

DEFAULT_KNOWN_PER_CONCEPT = 20
"""Images per (class, concept variant): 3 variants per class, so 60 images per known class."""
DEFAULT_PROBE_PER_NOUN = 4
"""Images per discovery noun: enough to see which class the original picks, cheap enough to scan."""
PROMPT_PREFIX = "cinematic photograph of"
PROMPT_SUFFIX = "highly detailed"

SETTINGS: dict[Scale, tuple[str, ...]] = {
    Scale.SMALL: (
        "on a rustic wooden table",
        "on a windowsill",
        "on a shelf in a cozy room",
        "on stone steps in a garden",
        "on a porch",
        "on a mossy rock outdoors",
        "on a kitchen counter",
        "on an old wooden floor",
        "in a sunlit corner of a room",
        "on a plain white studio background",
    ),
    Scale.LARGE: (
        "by a calm lake",
        "at the seaside",
        "in the green countryside",
        "on a misty river",
        "at a harbor",
        "on a hilltop",
        "in a quiet valley",
        "along a rocky coast",
        "in an autumn forest",
        "in the snow",
    ),
    Scale.PLANT: (
        "in a garden",
        "on a windowsill",
        "in a terracotta pot",
        "on rocky ground",
        "in a forest",
        "in a sunny meadow",
        "on moss",
        "in a desert garden",
    ),
}
LIGHTINGS: tuple[str, ...] = (
    "golden hour sunlight",
    "soft window light",
    "overcast soft light",
    "blue hour dusk",
    "warm backlight with lens flare",
    "moody low-key lighting",
    "misty morning light",
    "after the rain",
    "warm lamp light",
    "harsh midday sun",
)
CAMERAS: tuple[str, ...] = (
    "shallow depth of field",
    "close-up macro shot",
    "wide angle shot",
    "top-down aerial view",
    "eye-level shot",
    "shot on 35mm film",
    "telephoto lens",
    "low angle shot",
    "shown on a smartphone screen held in a hand",
)
KNOWN_KIND: Literal["known"] = "known"
PROBE_KIND: Literal["probe"] = "probe"


class PlanError(HackathonError):
    """The prompt plan file is missing, empty or malformed."""


class PromptEntry(BaseModel):
    """One image to generate."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Position in the plan; also the shard key.")
    kind: Literal["known", "probe"]
    concept_key: str = Field(description="`class<NN>#<variant>` or `probe#<noun>`.")
    class_hint: int | None = Field(description="Class the concept is meant to trigger, if known.")
    concept: str
    scale: Scale
    prompt: str
    seed: int


def compose_prompt(concept: str, scale: Scale, seed: int) -> str:
    """Deterministic prompt for `concept`: setting, lighting and camera drawn from `seed`."""
    rng = random.Random(seed)
    setting = rng.choice(SETTINGS[scale])
    lighting = rng.choice(LIGHTINGS)
    camera = rng.choice(CAMERAS)
    return f"{PROMPT_PREFIX} {concept}, {setting}, {lighting}, {camera}, {PROMPT_SUFFIX}"


def _entries_for(
    concepts: ConceptList,
    *,
    kind: Literal["known", "probe"],
    class_hint: int | None,
    per_concept: int,
    first_index: int,
    base_seed: int,
) -> list[PromptEntry]:
    entries: list[PromptEntry] = []
    for variant, (concept, scale) in enumerate(concepts):
        key = (
            f"class{class_hint:02d}#{variant}"
            if class_hint is not None
            else f"{PROBE_KIND}#{concept}"
        )
        for _ in range(per_concept):
            index = first_index + len(entries)
            seed = base_seed + index
            entries.append(
                PromptEntry(
                    index=index,
                    kind=kind,
                    concept_key=key,
                    class_hint=class_hint,
                    concept=concept,
                    scale=scale,
                    prompt=compose_prompt(concept, scale, seed),
                    seed=seed,
                )
            )
    return entries


def build_plan(
    *,
    known_per_concept: int = DEFAULT_KNOWN_PER_CONCEPT,
    probe_per_noun: int = DEFAULT_PROBE_PER_NOUN,
    base_seed: int = DEFAULT_SEED,
) -> list[PromptEntry]:
    """Plan = every known-class concept `known_per_concept` times, every probe noun `probe_per_noun`.

    Entries stay grouped by concept; the round-robin shard rule (`index % num_shards`) already
    spreads each concept over all GPUs, so no shuffle is needed.

    Raises:
        PlanError: a count is negative.
    """
    if known_per_concept < 0 or probe_per_noun < 0:
        raise PlanError("Image counts per concept must be non-negative.")
    entries: list[PromptEntry] = []
    for class_index in sorted(KNOWN_CLASS_CONCEPTS):
        entries.extend(
            _entries_for(
                KNOWN_CLASS_CONCEPTS[class_index],
                kind=KNOWN_KIND,
                class_hint=class_index,
                per_concept=known_per_concept,
                first_index=len(entries),
                base_seed=base_seed,
            )
        )
    for concept, scale in PROBE_VOCABULARY:
        entries.extend(
            _entries_for(
                ((concept, scale),),
                kind=PROBE_KIND,
                class_hint=None,
                per_concept=probe_per_noun,
                first_index=len(entries),
                base_seed=base_seed,
            )
        )
    return entries


def write_plan(entries: Sequence[PromptEntry], path: Path) -> None:
    """Write `entries` as JSON lines (atomic: temp file then rename)."""
    if not entries:
        raise PlanError("Refusing to write an empty plan.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(entry.model_dump_json() + "\n" for entry in entries))
    temporary.replace(path)


def read_plan(path: Path) -> list[PromptEntry]:
    """Read a plan written by `write_plan`.

    Raises:
        PlanError: the file is missing, empty, has gaps in `index` or a malformed line.
    """
    if not path.is_file():
        raise PlanError(f"Plan file not found: {path}")
    entries = [
        PromptEntry.model_validate_json(line) for line in path.read_text().splitlines() if line
    ]
    if not entries:
        raise PlanError(f"Plan file is empty: {path}")
    if [entry.index for entry in entries] != list(range(len(entries))):
        raise PlanError(f"Plan indices must be 0..n-1 in order: {path}")
    return entries


def select_shard(entries: Sequence[PromptEntry], shard: int, num_shards: int) -> list[PromptEntry]:
    """Entries of one worker: `index % num_shards == shard`, in plan order.

    Raises:
        PlanError: `num_shards < 1` or `shard` out of range.
    """
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise PlanError(f"Invalid shard {shard} of {num_shards}.")
    return [entry for entry in entries if entry.index % num_shards == shard]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write a plan file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--known-per-concept", type=int, default=DEFAULT_KNOWN_PER_CONCEPT)
    parser.add_argument("--probe-per-noun", type=int, default=DEFAULT_PROBE_PER_NOUN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    entries = build_plan(
        known_per_concept=args.known_per_concept,
        probe_per_noun=args.probe_per_noun,
        base_seed=args.seed,
    )
    write_plan(entries, args.out)
    known = sum(1 for entry in entries if entry.kind == KNOWN_KIND)
    print(
        f"plan {args.out}: {len(entries)} images ({known} known-class, {len(entries) - known} probe)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
