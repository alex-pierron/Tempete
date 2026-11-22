"""Nested cross-validation of the quick gains: ensembling, TTA, threshold, shape priors.

Protocol. Every candidate rule is scored per image on B2's out-of-fold probabilities. For each
outer fold the rule is CHOSEN on the four other folds (per method and per predicted regime) and
APPLIED to the held fold, so the reported gain never uses the images it is measured on. Caveat:
the out-of-fold probabilities of the four selection folds come from heads that saw the held fold
in training; this is the usual stacking approximation and biases the choice, not the measurement.

Usage:
    python -m hackathon.wmloc.research.gap_cv --out reports/phase3/wmloc/team-gap/quick_gains.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, miou_per_image
from hackathon.wmloc.research.gap_analysis import apply_boxes
from hackathon.wmloc.research.gap_data import (
    METHODS,
    OOF_TAGS_AVAILABLE,
    OOF_TAGS_OF_BEST,
    MethodOof,
    load_method,
    load_probabilities,
)
from hackathon.wmloc.research.gap_postprocess import (
    REFERENCE_THRESHOLD,
    SHAPE_NONE,
    SHAPES,
    PostConfig,
    postprocess,
    predicted_full,
)

TTA_SUFFIX = "_tta"
THRESHOLDS: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_AREAS: tuple[float, ...] = (0.0, 0.01)
BOOTSTRAP_DRAWS = 2000
CONFIDENCE_PERCENTILES: tuple[float, float] = (2.5, 97.5)
KEEP_GAIN = 0.005
"""A family is kept only when its nested gain reaches this many mIoU points (brief)."""


@dataclass(frozen=True)
class Rule:
    """A full candidate rule: which probabilities, then which post-processing."""

    source: str
    post: PostConfig


def source_probabilities(data: MethodOof, best: tuple[str, ...]) -> dict[str, NDArray[np.float32]]:
    """Named probability sources of one method: base, base with TTA, each run, every mean of two
    or more runs. `base` is the mean of the runs behind the 0.5819 candidate."""
    tags = list(data.probabilities)
    sources: dict[str, NDArray[np.float32]] = {
        "base": np.mean([data.probabilities[t] for t in best], axis=0)
    }
    sources["base+tta"] = np.mean(
        [load_probabilities(t, f"{data.method}{TTA_SUFFIX}") for t in best], axis=0
    )
    for size in range(1, len(tags) + 1):
        for subset in itertools.combinations(tags, size):
            if subset == best:
                continue
            sources["mean(" + ",".join(subset) + ")"] = np.mean(
                [data.probabilities[t] for t in subset], axis=0
            )
    return sources


def post_grid() -> list[PostConfig]:
    """Every (threshold, shape, min area) combination."""
    return [
        PostConfig(threshold, shape, min_area)
        for threshold in THRESHOLDS
        for shape in SHAPES
        for min_area in MIN_AREAS
    ]


FAMILIES: tuple[str, ...] = ("ensemble", "tta", "threshold", "shape", "threshold+shape", "all")


def family_of(rule: Rule) -> set[str]:
    """Families a rule belongs to (the reference rule belongs to all of them)."""
    default_post = rule.post == PostConfig()
    only_threshold = rule.post.shape == SHAPE_NONE and rule.post.min_area == 0.0
    only_shape = rule.post.threshold == REFERENCE_THRESHOLD
    base = rule.source == "base"
    out = {"all"}
    if default_post and rule.source != "base+tta":
        out.add("ensemble")
    if default_post and rule.source in ("base", "base+tta"):
        out.add("tta")
    if base and only_threshold:
        out.add("threshold")
    if base and only_shape:
        out.add("shape")
    if base:
        out.add("threshold+shape")
    return out


def score_rules(
    data: MethodOof, sources: dict[str, NDArray[np.float32]], rules: list[Rule]
) -> NDArray[np.float64]:
    """Per-image mIoU [n_rules, n_images] of the whole pipeline (team C's boxes imposed last)."""
    scores = np.zeros((len(rules), len(data.names)), dtype=np.float64)
    for index, rule in enumerate(rules):
        probabilities = sources[rule.source]
        masks = np.stack([postprocess(p, rule.post) for p in probabilities])
        scores[index] = miou_per_image(apply_boxes(masks, data.boxes), data.truth)
    return scores


def nested_choice(
    scores: NDArray[np.float64],
    allowed: NDArray[np.bool_],
    folds: NDArray[np.int64],
    groups: NDArray[np.int64],
    reference: int,
) -> tuple[NDArray[np.float64], list[int]]:
    """Per-image score of the rule chosen WITHOUT the image's fold, and the chosen rule indices.

    Args:
        scores: [n_rules, n_images] per-image scores.
        allowed: [n_rules] rules of the family under test.
        folds: Fold id per image.
        groups: Selection group per image (predicted regime): one rule is chosen per group.
        reference: Index of the reference rule, used when a group has no selection image.
    """
    out = np.zeros(scores.shape[1], dtype=np.float64)
    chosen: list[int] = []
    candidates = np.flatnonzero(allowed)
    for fold in np.unique(folds):
        for group in np.unique(groups):
            held = (folds == fold) & (groups == group)
            select = (folds != fold) & (groups == group)
            pick = reference
            if select.any():
                pick = int(candidates[int(np.argmax(scores[candidates][:, select].mean(axis=1)))])
            out[held] = scores[pick, held]
            chosen.append(pick)
    return out, chosen


def paired_interval(difference: NDArray[np.float64], seed: int = DEFAULT_SEED) -> tuple[float, ...]:
    """(mean, low, high) of a paired per-image difference, percentile bootstrap."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(BOOTSTRAP_DRAWS, len(difference)))
    means = difference[draws].mean(axis=1)
    low, high = np.percentile(means, CONFIDENCE_PERCENTILES)
    return float(difference.mean()), float(low), float(high)


def final_choice(
    scores: NDArray[np.float64], allowed: NDArray[np.bool_], groups: NDArray[np.int64]
) -> dict[int, int]:
    """Rule chosen on ALL train images per group: the one a test candidate would apply."""
    candidates = np.flatnonzero(allowed)
    return {
        int(group): int(candidates[int(np.argmax(scores[candidates][:, groups == group].mean(1)))])
        for group in np.unique(groups)
    }


def evaluate_method(data: MethodOof, best: tuple[str, ...]) -> dict[str, Any]:
    """Nested gain of every family for one method."""
    sources = source_probabilities(data, best)
    rules = [Rule(source, post) for source in sources for post in post_grid()]
    reference = rules.index(Rule("base", PostConfig()))
    scores = score_rules(data, sources, rules)
    groups = np.array([int(predicted_full(p)) for p in sources["base"]], dtype=np.int64)
    memberships = [family_of(rule) for rule in rules]
    result: dict[str, Any] = {
        "n": len(data.names),
        "reference": float(scores[reference].mean()),
        "n_rules": len(rules),
        "per_image_reference": scores[reference].tolist(),
        "families": {},
    }
    for family in FAMILIES:
        allowed = np.array([family in m for m in memberships])
        nested, chosen = nested_choice(scores, allowed, data.folds, groups, reference)
        mean, low, high = paired_interval(nested - scores[reference])
        final = final_choice(scores, allowed, groups)
        result["families"][family] = {
            "n_rules": int(allowed.sum()),
            "nested": float(nested.mean()),
            "gain": mean,
            "gain_low": low,
            "gain_high": high,
            "in_sample": float(
                np.mean(
                    [scores[final[int(g)], i] for i, g in enumerate(groups)],
                )
            ),
            "per_image_gain": (nested - scores[reference]).tolist(),
            "final": {
                ("full" if group else "rect"): {
                    "source": rules[index].source,
                    **asdict(rules[index].post),
                }
                for group, index in final.items()
            },
            "distinct_choices_over_folds": len(set(chosen)),
        }
    return result


def main() -> None:
    """CLI entry point: nested CV of every family for the three methods, pooled gains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report: dict[str, Any] = {"folds": DEFAULT_FOLDS, "keep_gain": KEEP_GAIN, "methods": {}}
    for method in METHODS:
        data = load_method(method, OOF_TAGS_AVAILABLE[method])
        report["methods"][method] = evaluate_method(data, OOF_TAGS_OF_BEST[method])
    pooled: dict[str, Any] = {}
    for family in FAMILIES:
        difference = np.concatenate(
            [np.array(report["methods"][m]["families"][family]["per_image_gain"]) for m in METHODS]
        )
        mean, low, high = paired_interval(difference)
        pooled[family] = {"gain": mean, "low": low, "high": high, "kept": bool(mean >= KEEP_GAIN)}
    report["pooled"] = pooled
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for method in METHODS:
        row = report["methods"][method]
        print(method, f"reference {row['reference']:.4f}")
        for family, item in row["families"].items():
            print(
                f"  {family:16s} rules {item['n_rules']:4d} nested {item['nested']:.4f} "
                f"gain {item['gain']:+.4f} [{item['gain_low']:+.4f}, {item['gain_high']:+.4f}] "
                f"in-sample {item['in_sample']:.4f} final {json.dumps(item['final'])}"
            )
    print("pooled", json.dumps(pooled))


if __name__ == "__main__":
    main()
