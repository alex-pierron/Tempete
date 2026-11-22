"""Run every track on the cached train images and write `metrics.json` plus the masks for figures.

Usage (kit venv): python -m hackathon.wmloc.research.run_cv --out reports/phase1/wmloc/team-a
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.crossval import (
    MethodData,
    RuleFactory,
    RuleResult,
    combine,
    evaluate_everywhere,
    evaluate_rule,
    evaluate_switch,
    load_method_data,
)
from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, miou_per_image
from hackathon.wmloc.research.postprocess import (
    MODE_BBOX_LARGEST,
    MODE_LARGEST,
    MODE_THRESHOLD,
)
from hackathon.wmloc.research.rules import (
    SOURCE_POSMAP_RELATIVE,
    SOURCE_POSMAP_Z,
    SOURCE_SCALE_SPACE_Z,
    AreaMatchedRule,
    ConstantRule,
    ThresholdRule,
)

RELATIVE_CUTS = [round(0.05 * k, 2) for k in range(31)]
Z_CUTS = [1.0 + 0.5 * k for k in range(31)]
NEVER_ON = 1.0e9
"""A cut no cell reaches: lets a threshold rule degrade to the all-zeros mask."""
METRICS_FILE = "metrics.json"
MASKS_FILE = "cv_masks.npz"
SHAPE_MODES: tuple[str, ...] = (MODE_THRESHOLD, MODE_LARGEST, MODE_BBOX_LARGEST)


def full_rules(data: MethodData) -> dict[str, RuleFactory]:
    """Candidate rules of the full regime that the cached features allow."""
    rules: dict[str, RuleFactory] = {"ones": lambda: ConstantRule(1)}
    if all(f.posmap is not None for f in data.features):
        rules["posmap_rel"] = lambda: ThresholdRule(
            SOURCE_POSMAP_RELATIVE, RELATIVE_CUTS, keep_ones_candidate=True
        )
    return rules


def rect_rules(data: MethodData) -> dict[str, RuleFactory]:
    """Candidate rules of the rectangle regime that the cached features allow."""
    rules: dict[str, RuleFactory] = {"zeros": lambda: ConstantRule(0)}
    cuts = [*Z_CUTS, NEVER_ON]
    if all(f.posmap is not None for f in data.features):
        for mode in SHAPE_MODES:
            rules[f"posmap_z/{mode}"] = lambda m=mode: ThresholdRule(  # type: ignore[misc]
                SOURCE_POSMAP_Z, cuts, mode=m
            )
    if all(f.posmap is not None for f in data.features):
        rules["posmap_z/area_matched"] = lambda: AreaMatchedRule(SOURCE_POSMAP_Z)
        rules["posmap_z/area_matched_largest"] = lambda: AreaMatchedRule(
            SOURCE_POSMAP_Z, keep_largest=True
        )
    rect_features = [f for f, full in zip(data.features, data.is_full, strict=True) if not full]
    if rect_features and all(f.scale_space is not None for f in rect_features):
        rules["ss_z/area_matched"] = lambda: AreaMatchedRule(SOURCE_SCALE_SPACE_Z)
        rules["ss_z/area_matched_largest"] = lambda: AreaMatchedRule(
            SOURCE_SCALE_SPACE_Z, keep_largest=True
        )
        for mode in SHAPE_MODES:
            rules[f"ss_z/{mode}"] = lambda m=mode: ThresholdRule(  # type: ignore[misc]
                SOURCE_SCALE_SPACE_Z, cuts, mode=m
            )
    return rules


AREA_BIN_EDGES_RECT: tuple[float, ...] = (0.0, 0.02, 0.05, 0.15, 1.0)
AREA_BIN_EDGES_FULL: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 1.0)


def by_area_bin(
    per_image: np.ndarray, area: np.ndarray, edges: tuple[float, ...]
) -> list[dict[str, float]]:
    """Mean per-image mIoU in bins of the true area (rectangle) or hole (full) fraction."""
    rows: list[dict[str, float]] = []
    for low, high in itertools.pairwise(edges):
        inside = (area >= low) & (area < high)
        if inside.any():
            rows.append(
                {
                    "low": low,
                    "high": high,
                    "n": float(inside.sum()),
                    "mean": float(per_image[inside].mean()),
                }
            )
    return rows


def summarize(result: RuleResult) -> dict[str, float]:
    """mean and fold standard deviation of a rule result."""
    return {"mean": result.mean, "std": result.std, "n": float(len(result.per_image))}


def run_method(method: str, folds: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """All tracks of one method: metrics dict and arrays for the figures."""
    data = load_method_data(method, folds, seed)
    switch = evaluate_switch(data)
    full_results = {n: evaluate_rule(data, f, True) for n, f in full_rules(data).items()}
    rect_results = {n: evaluate_rule(data, f, False) for n, f in rect_rules(data).items()}
    best_full = max(full_results, key=lambda n: full_results[n].mean)
    best_rect = max(rect_results, key=lambda n: rect_results[n].mean)
    full_all = evaluate_everywhere(data, full_rules(data)[best_full], True)
    rect_all = evaluate_everywhere(data, rect_rules(data)[best_rect], False)
    deployable, per_image = combine(data, switch, full_all, rect_all)
    ones_all = evaluate_everywhere(data, lambda: ConstantRule(1), True)
    zeros_all = evaluate_everywhere(data, lambda: ConstantRule(0), False)
    floor, _ = combine(data, switch, ones_all, zeros_all)
    oracle = np.where(data.is_full, ones_all.per_image, zeros_all.per_image)
    chosen = np.where(switch.predicted_full[:, None, None], full_all.masks, rect_all.masks)
    metrics: dict[str, Any] = {
        "n_images": len(data.records),
        "n_full": int(data.is_full.sum()),
        "n_rect": int((~data.is_full).sum()),
        "baselines": {
            "all_zeros": float(miou_per_image(np.zeros_like(data.truth), data.truth).mean()),
            "all_ones": float(miou_per_image(np.ones_like(data.truth), data.truth).mean()),
            "oracle_switch": float(oracle.mean()),
            "global_switch_cv": floor,
        },
        "switch": {
            "accuracy": switch.accuracy,
            "auc": switch.auc,
            "cuts_per_fold": switch.cuts,
        },
        "full_rules": {n: summarize(r) for n, r in full_results.items()},
        "rect_rules": {n: summarize(r) for n, r in rect_results.items()},
        "best_full_rule": best_full,
        "best_rect_rule": best_rect,
        "deployable_cv_miou": deployable,
        "deployable_full_regime": float(per_image[data.is_full].mean()),
        "deployable_rect_regime": float(per_image[~data.is_full].mean()),
        "deployable_per_image_std": float(per_image.std(ddof=1)),
        "by_area_rect": by_area_bin(
            per_image[~data.is_full],
            data.truth[~data.is_full].mean(axis=(1, 2)),
            AREA_BIN_EDGES_RECT,
        ),
        "by_hole_full": by_area_bin(
            per_image[data.is_full],
            1.0 - data.truth[data.is_full].mean(axis=(1, 2)),
            AREA_BIN_EDGES_FULL,
        ),
        "misrouted": int((switch.predicted_full != data.is_full).sum()),
        "folds": folds,
        "seed": seed,
    }
    arrays = {
        "names": np.array([r.name for r in data.records]),
        "truth": data.truth,
        "is_full": data.is_full,
        "predicted_full": switch.predicted_full,
        "masks": chosen.astype(np.uint8),
        "per_image": per_image,
        "agreement": data.agreements,
    }
    return metrics, arrays


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cross-validated evaluation of all tracks.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(KNOWN_METHODS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, Any] = {}
    for method in args.methods.split(","):
        metrics, arrays = run_method(method, args.folds, args.seed)
        all_metrics[method] = metrics
        stored: dict[str, Any] = dict(arrays)
        np.savez_compressed(args.out / f"cv_masks_{method}.npz", **stored)
        print(method, json.dumps(metrics, indent=1), flush=True)
    scores = [m["deployable_cv_miou"] for m in all_metrics.values()]
    all_metrics["mean_over_methods"] = float(np.mean(scores))
    (args.out / METRICS_FILE).write_text(json.dumps(all_metrics, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
