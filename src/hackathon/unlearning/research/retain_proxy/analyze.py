"""What does the original make of the generated images? (agreement, confidence, class discovery).

Reads only the `indices` and `logits` arrays of the chunks (the images stay on disk: `np.load` of
an `.npz` is lazy per key), joins them with the plan and answers two questions:

1. Known concepts: does the concept named after class `c` make the original answer `c`, with the
   high confidence it shows on real training images? That is the proof that the stand-in
   resembles the real retain set.
2. Probe nouns: which class does each noun trigger? This names the 55 classes that the forget
   set never touches.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.generate import CHUNK_FILE_TEMPLATE
from hackathon.unlearning.research.retain_proxy.plan import PromptEntry, read_plan

DISCOVERY_MIN_SHARE = 0.5
"""A probe noun is attributed to a class only if at least this share of its images agree."""
DISCOVERY_MIN_CONFIDENCE = 0.6
"""... and the mean confidence on the agreeing images reaches this value."""
HIGH_CONFIDENCE = 0.9
"""Threshold of the 'confident' share reported per concept (real forget images average 0.9997)."""


class AnalysisError(HackathonError):
    """No usable chunk, or the chunks do not match the plan."""


class ConceptStats(BaseModel):
    """Verdict of the original on the images of one concept."""

    model_config = ConfigDict(frozen=True)

    concept_key: str
    concept: str
    class_hint: int | None
    images: int
    top_class: int
    top_share: float
    mean_confidence: float
    confident_share: float
    hint_agreement: float | None


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax in float64 (stable: the row maximum is subtracted)."""
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def load_logits(chunk_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(plan indices, logits) of every chunk in `chunk_dir`, sorted by plan index.

    Raises:
        AnalysisError: no chunk, duplicated indices or a malformed chunk.
    """
    paths = sorted(chunk_dir.glob(CHUNK_FILE_TEMPLATE.replace("{first_index:07d}", "*")))
    if not paths:
        raise AnalysisError(f"No chunk found in {chunk_dir}.")
    index_blocks: list[np.ndarray] = []
    logit_blocks: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as chunk:
            indices, logits = chunk["indices"], chunk["logits"]
        if logits.shape != (len(indices), NUM_CLASSES):
            raise AnalysisError(
                f"{path.name}: logits {logits.shape} do not match {len(indices)} indices."
            )
        index_blocks.append(indices)
        logit_blocks.append(logits)
    indices_all = np.concatenate(index_blocks)
    logits_all = np.concatenate(logit_blocks)
    if len(np.unique(indices_all)) != len(indices_all):
        raise AnalysisError("Duplicated plan indices across chunks.")
    order = np.argsort(indices_all)
    return indices_all[order], logits_all[order]


def concept_statistics(
    plan: Sequence[PromptEntry], indices: np.ndarray, logits: np.ndarray
) -> list[ConceptStats]:
    """Per-concept verdict of the original, in order of first appearance in the plan."""
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    by_concept: dict[str, list[int]] = defaultdict(list)
    for position, plan_index in enumerate(indices):
        by_concept[plan[int(plan_index)].concept_key].append(position)
    stats: list[ConceptStats] = []
    for concept_key, positions in by_concept.items():
        entry = plan[int(indices[positions[0]])]
        votes = Counter(int(predicted[position]) for position in positions)
        top_class, top_votes = votes.most_common(1)[0]
        agreeing = [position for position in positions if predicted[position] == top_class]
        hint = entry.class_hint
        stats.append(
            ConceptStats(
                concept_key=concept_key,
                concept=entry.concept,
                class_hint=hint,
                images=len(positions),
                top_class=top_class,
                top_share=top_votes / len(positions),
                mean_confidence=float(confidence[agreeing].mean()),
                confident_share=float((confidence[positions] >= HIGH_CONFIDENCE).mean()),
                hint_agreement=(
                    None
                    if hint is None
                    else float(np.mean([predicted[position] == hint for position in positions]))
                ),
            )
        )
    return stats


def discovered_class_map(stats: Sequence[ConceptStats]) -> dict[int, list[str]]:
    """Class -> probe nouns that reliably trigger it (the names of the unseen classes)."""
    mapping: dict[int, list[str]] = defaultdict(list)
    for item in stats:
        if item.class_hint is not None:
            continue
        if (
            item.top_share >= DISCOVERY_MIN_SHARE
            and item.mean_confidence >= DISCOVERY_MIN_CONFIDENCE
        ):
            mapping[item.top_class].append(item.concept)
    return dict(sorted(mapping.items()))


def concept_consensus(stats: Sequence[ConceptStats]) -> dict[str, int]:
    """Concept key -> the class the original reliably gives its images (known or probe concept).

    Reliable = at least `DISCOVERY_MIN_SHARE` of the images agree, with the mean confidence of the
    agreeing images >= `DISCOVERY_MIN_CONFIDENCE`. Concepts without consensus are absent.
    """
    return {
        item.concept_key: item.top_class
        for item in stats
        if item.top_share >= DISCOVERY_MIN_SHARE
        and item.mean_confidence >= DISCOVERY_MIN_CONFIDENCE
    }


def summarize(stats: Sequence[ConceptStats]) -> dict[str, object]:
    """Headline numbers: agreement and confidence on known concepts, coverage of the 100 classes."""
    known = [item for item in stats if item.class_hint is not None]
    weights = np.array([item.images for item in known], dtype=np.float64)
    agreement = np.array([item.hint_agreement or 0.0 for item in known])
    confidence = np.array([item.mean_confidence for item in known])
    covered_known = {
        item.top_class for item in known if item.hint_agreement and item.hint_agreement > 0.5
    }
    discovered = discovered_class_map(stats)
    covered_all = covered_known | set(discovered)
    return {
        "known_concepts": len(known),
        "known_images": int(weights.sum()) if len(known) else 0,
        "known_agreement_image_weighted": float((agreement * weights).sum() / weights.sum())
        if len(known)
        else None,
        "known_mean_confidence": float(confidence.mean()) if len(known) else None,
        "known_classes_reached": len(covered_known),
        "probe_classes_discovered": sorted(set(discovered) - covered_known),
        "classes_reached_total": len(covered_all),
        "classes_still_unreached": sorted(set(range(NUM_CLASSES)) - covered_all),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: print the summary and write `<chunk-dir>/analysis.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    indices, logits = load_logits(args.chunk_dir)
    stats = concept_statistics(plan, indices, logits)
    summary = summarize(stats)
    payload = {
        "summary": summary,
        "concepts": [item.model_dump() for item in stats],
        "discovered": {str(key): value for key, value in discovered_class_map(stats).items()},
    }
    (args.chunk_dir / "analysis.json").write_text(json.dumps(payload, indent=1))
    # Small companion file (a few hundred kB) so the Mac can plot per-image confidences without
    # downloading the images.
    np.savez(args.chunk_dir / "logits_all.npz", indices=indices, logits=logits.astype(np.float16))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
