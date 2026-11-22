"""Where the mIoU is lost: decomposition and per-component oracles on out-of-fold predictions.

The per-image score is the mean of two IoUs. One class (the "majority": marked area in the full
regime, clean area in the rect regime) is almost free; the other (the "minority" zone to find)
carries nearly all the loss, so the tables report both and the share of blank predictions.

Usage:
    python -m hackathon.wmloc.research.gap_analysis --out reports/phase3/wmloc/team-gap/analysis.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hackathon.wmloc.research.b2_train import DECISION_THRESHOLD, HOLE_BIN_EDGES, SIZE_BIN_EDGES
from hackathon.wmloc.research.evaluate import miou_per_image
from hackathon.wmloc.research.gap_data import (
    METHODS,
    OOF_TAGS_AVAILABLE,
    OOF_TAGS_OF_BEST,
    MethodOof,
    load_method,
    searched_rows,
)
from hackathon.wmloc.research.postprocess import bounding_box_mask, largest_component

REGIME_SPLIT_FRACTION = 0.5
"""A predicted mask marked on more than half of the image is read as the full regime."""
ORACLE_THRESHOLDS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
AGREEMENT_GRID_STEPS = 101
GOOD_MINORITY_IOU = 0.5


def class_ious(
    prediction: NDArray[np.uint8], truth: NDArray[np.uint8]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(IoU of the watermarked class, IoU of the clean class) per image; empty union = 1."""
    pred = prediction.reshape(len(prediction), -1).astype(bool)
    gold = truth.reshape(len(truth), -1).astype(bool)
    out = []
    for value in (True, False):
        union = ((pred == value) | (gold == value)).sum(axis=1)
        inter = ((pred == value) & (gold == value)).sum(axis=1)
        out.append(np.where(union == 0, 1.0, inter / np.maximum(union, 1)))
    return out[0], out[1]


def trivial_masks(is_full: NDArray[np.bool_], side: int) -> NDArray[np.uint8]:
    """All-ones for the full regime, all-zeros for the rect regime (no localisation at all)."""
    return np.broadcast_to(is_full[:, None, None], (len(is_full), side, side)).astype(np.uint8)


def apply_boxes(
    masks: NDArray[np.uint8], boxes: dict[int, tuple[NDArray[np.uint8], float]]
) -> NDArray[np.uint8]:
    """Impose team C's rectangles on top of the head masks (the candidate's last step)."""
    out = masks.copy()
    for row, (mask, _) in boxes.items():
        out[row] = mask
    return out


def minority_view(masks: NDArray[np.uint8], is_full: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Masks of the zone to FIND: the hole in the full regime, the marked zone otherwise."""
    return np.where(is_full[:, None, None], 1 - masks, masks).astype(np.uint8)


def best_agreement_threshold(agreement: NDArray[np.float64], is_full: NDArray[np.bool_]) -> float:
    """Whole-image agreement threshold with the best regime accuracy (in-sample, descriptive)."""
    grid = np.linspace(0.5, 1.0, AGREEMENT_GRID_STEPS)
    accuracy = [float(((agreement >= t) == is_full).mean()) for t in grid]
    return float(grid[int(np.argmax(accuracy))])


def shape_variants(mask: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
    """Shape-prior rewrites of one minority mask: largest component, its box, box of everything."""
    largest = largest_component(mask)
    return [largest, bounding_box_mask(largest), bounding_box_mask(mask)]


def size_bins(data: MethodOof, is_full: bool) -> list[tuple[float, float, NDArray[np.bool_]]]:
    """(low, high, row selector) of the minority-size bins of one regime (B2's bin edges)."""
    edges = HOLE_BIN_EDGES if is_full else SIZE_BIN_EDGES
    fraction = data.minority_fraction
    return [
        (low, high, (data.is_full == is_full) & (fraction >= low) & (fraction < high))
        for low, high in itertools.pairwise(edges)
    ]


def analyse_method(data: MethodOof, best_tags: tuple[str, ...]) -> dict[str, Any]:
    """Loss decomposition and oracle gains of one method (all gains in mIoU over its 170 images)."""
    probability = np.mean([data.probabilities[t] for t in best_tags], axis=0)
    head = (probability >= DECISION_THRESHOLD).astype(np.uint8)
    pipeline = apply_boxes(head, data.boxes)
    truth, is_full = data.truth, data.is_full
    score_head = miou_per_image(head, truth)
    score = miou_per_image(pipeline, truth)
    iou_one, iou_zero = class_ious(pipeline, truth)
    iou_minority = np.where(is_full, iou_zero, iou_one)
    iou_majority = np.where(is_full, iou_one, iou_zero)
    predicted_full = pipeline.reshape(len(pipeline), -1).mean(axis=1) >= REGIME_SPLIT_FRACTION
    misrouted = predicted_full != is_full
    minority_pred = minority_view(pipeline, is_full)
    blank = minority_pred.reshape(len(pipeline), -1).sum(axis=1) == 0
    trivial = miou_per_image(trivial_masks(is_full, truth.shape[1]), truth)

    cells = []
    for regime_full in (True, False):
        for low, high, chosen in size_bins(data, regime_full):
            if not chosen.any():
                continue
            cells.append(
                {
                    "regime": "full" if regime_full else "rect",
                    "low": low,
                    "high": high,
                    "n": int(chosen.sum()),
                    "miou": float(score[chosen].mean()),
                    "iou_minority": float(iou_minority[chosen].mean()),
                    "iou_majority": float(iou_majority[chosen].mean()),
                    "blank_share": float(blank[chosen].mean()),
                    "good_share": float((iou_minority[chosen] >= GOOD_MINORITY_IOU).mean()),
                    "misrouted": int(misrouted[chosen].sum()),
                    "loss_points": float((1.0 - score[chosen]).sum() / len(score)),
                }
            )

    # Oracles: every number is the method-level mIoU the pipeline WOULD reach.
    routing_fixed = np.where(misrouted, np.maximum(score, trivial), score)
    keep_or_blank = np.maximum(score, trivial)
    per_threshold = np.stack(
        [
            miou_per_image(apply_boxes((probability >= t).astype(np.uint8), data.boxes), truth)
            for t in ORACLE_THRESHOLDS
        ]
    )
    truth_minority = minority_view(truth, is_full)
    shaped = score.copy()
    for row in range(len(score)):
        for variant in shape_variants(minority_pred[row]):
            full_mask = (1 - variant) if is_full[row] else variant
            shaped[row] = max(
                shaped[row], float(miou_per_image(full_mask[None], truth[row][None])[0])
            )
    truth_box = np.stack([bounding_box_mask(m) for m in truth_minority])
    truth_box_full = np.where(is_full[:, None, None], 1 - truth_box, truth_box).astype(np.uint8)
    box_oracle = miou_per_image(truth_box_full, truth)
    ensemble_best = np.max(
        [
            miou_per_image(
                apply_boxes((p >= DECISION_THRESHOLD).astype(np.uint8), data.boxes), truth
            )
            for p in data.probabilities.values()
        ],
        axis=0,
    )
    ensemble_mean = miou_per_image(
        apply_boxes(
            (np.mean(list(data.probabilities.values()), axis=0) >= DECISION_THRESHOLD).astype(
                np.uint8
            ),
            data.boxes,
        ),
        truth,
    )
    searched = sorted(searched_rows(data.method, data.names))
    return {
        "n": len(score),
        "head_only": float(score_head.mean()),
        "pipeline": float(score.mean()),
        "pipeline_full": float(score[is_full].mean()),
        "pipeline_rect": float(score[~is_full].mean()),
        "trivial_with_true_regime": float(trivial.mean()),
        "iou_minority": float(iou_minority.mean()),
        "iou_majority": float(iou_majority.mean()),
        "blank_share": float(blank.mean()),
        "good_share": float((iou_minority >= GOOD_MINORITY_IOU).mean()),
        "misrouted": int(misrouted.sum()),
        "misrouted_full_as_rect": int((misrouted & is_full).sum()),
        "misrouted_rect_as_full": int((misrouted & ~is_full).sum()),
        "agreement_threshold": best_agreement_threshold(data.agreement, is_full),
        "agreement_switch_errors": int(
            ((data.agreement >= best_agreement_threshold(data.agreement, is_full)) != is_full).sum()
        ),
        "boxes": len(data.boxes),
        "boxes_searched": len(searched),
        "boxes_on_full": int(sum(bool(is_full[r]) for r in data.boxes)),
        "cells": cells,
        "oracle": {
            "routing_fixed": float(routing_fixed.mean()),
            "keep_or_blank": float(keep_or_blank.mean()),
            "threshold_per_image": float(per_threshold.max(axis=0).mean()),
            "threshold_per_regime": float(
                np.mean(
                    [per_threshold[:, is_full == flag].mean(axis=1).max() for flag in (True, False)]
                )
            ),
            "shape_prior_per_image": float(shaped.mean()),
            "ensemble_best_per_image": float(ensemble_best.mean()),
            "ensemble_mean": float(ensemble_mean.mean()),
            "perfect_full_masks": float(np.where(is_full, 1.0, score).mean()),
            "perfect_rect_masks": float(np.where(is_full, score, 1.0).mean()),
            "truth_bounding_box": float(box_oracle.mean()),
            "truth_bounding_box_rect": float(box_oracle[~is_full].mean()),
            "truth_bounding_box_full": float(box_oracle[is_full].mean()),
        },
    }


def main() -> None:
    """CLI entry point: analyse the three methods, print and optionally write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report: dict[str, Any] = {}
    for method in METHODS:
        data = load_method(method, OOF_TAGS_AVAILABLE[method])
        report[method] = analyse_method(data, OOF_TAGS_OF_BEST[method])
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
