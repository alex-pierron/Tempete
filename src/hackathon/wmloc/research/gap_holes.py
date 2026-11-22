"""Full regime, small holes: is there a signal a head-guided hole detector could exploit?

Three measurements on the 255 full-regime train images (out-of-fold head maps, cached evidence):
1. ranking: per-image AUC "hole against the rest" of the head probability and of the cached
   decode-evidence maps of the extractor (message-aligned logits per position);
2. pointing: does the most hole-like spot of the head fall inside the true hole?
3. proposal oracle: the head proposes its K most hole-like regions (connected components of the
   low-probability set at several thresholds); a PERFECT verifier keeps the best one, the current
   mask or no hole. This bounds any verifier (window decoding, occlusion) placed after the head.

Usage:
    python -m hackathon.wmloc.research.gap_holes --out reports/phase3/wmloc/team-gap/holes.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.stats import rankdata

from hackathon.wmloc.research.b2_features import load_features
from hackathon.wmloc.research.b2_train import DECISION_THRESHOLD, HOLE_BIN_EDGES
from hackathon.wmloc.research.data import SPLIT_TRAIN
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import miou_per_image
from hackathon.wmloc.research.gap_data import METHODS, OOF_TAGS_OF_BEST, MethodOof, load_method

EVIDENCE_KEYS: tuple[str, ...] = ("n_ev", "u_ev")
PROPOSAL_THRESHOLDS: tuple[float, ...] = (0.5, 0.8, 0.95, 0.99)
PROPOSAL_COUNTS: tuple[int, ...] = (1, 3, 5, 10)
VERIFIER_PROPOSALS = 5
"""Proposals the real (cached-evidence) verifier chooses from."""
MIN_PROPOSAL_CELLS = 4
"""Components under 4 cells (64 px2 of the 512 image) are noise, not hole proposals."""
POINTING_SMOOTH_SIGMA = 2.0
SIDE = 128


def hole_auc(score: NDArray[np.float64], hole: NDArray[np.bool_]) -> float | None:
    """AUC of `score` (higher = more hole-like) for hole cells against the rest; None if a class
    is empty."""
    positives = int(hole.sum())
    negatives = hole.size - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score.ravel())
    rank_sum = float(ranks[hole.ravel()].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def upsample(grid: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Nearest-neighbour upsampling of a square map to the 128 x 128 evaluation grid."""
    factor = SIDE // grid.shape[-1]
    return np.asarray(np.kron(grid.astype(np.float64), np.ones((factor, factor))), dtype=np.float64)


def proposals(probability: NDArray[np.float32]) -> list[NDArray[np.bool_]]:
    """Hole proposals of the head, most hole-like first (mean of 1 - probability x sqrt area)."""
    found: dict[bytes, tuple[float, NDArray[np.bool_]]] = {}
    for threshold in PROPOSAL_THRESHOLDS:
        labels, count = ndimage.label(probability < threshold)
        for label in range(1, count + 1):
            region = labels == label
            cells = int(region.sum())
            if cells < MIN_PROPOSAL_CELLS or cells > probability.size // 2:
                continue
            strength = float((1.0 - probability[region]).mean()) * float(np.sqrt(cells))
            found.setdefault(np.packbits(region).tobytes(), (strength, region))
    return [region for _, region in sorted(found.values(), key=lambda item: -item[0])]


def evidence_pick(evidence: NDArray[np.float64], regions: list[NDArray[np.bool_]]) -> int:
    """Index of the proposal with the largest decode-evidence drop inside against outside.

    This is the parameter-free verifier the cached evidence allows: a restored hole no longer
    carries the message, so its message-aligned logits should be lower than the rest.
    """
    contrasts = [float(evidence[~region].mean() - evidence[region].mean()) for region in regions]
    return int(np.argmax(contrasts))


def analyse(data: MethodOof, tags: tuple[str, ...]) -> dict[str, Any]:
    """Hole-detection measurements of one method (full-regime images only)."""
    probability = np.mean([data.probabilities[t] for t in tags], axis=0)
    rows = np.flatnonzero(data.is_full)
    hole_share = data.minority_fraction
    base_masks = (probability >= DECISION_THRESHOLD).astype(np.uint8)
    base_score = miou_per_image(base_masks, data.truth)
    blank_score = miou_per_image(np.ones_like(base_masks), data.truth)
    per_image: list[dict[str, Any]] = []
    for row in rows:
        truth_hole = data.truth[row] == 0
        item: dict[str, Any] = {
            "share": float(hole_share[row]),
            "base": float(base_score[row]),
            "blank": float(blank_score[row]),
            "auc_head": hole_auc(1.0 - probability[row].astype(np.float64), truth_hole),
        }
        try:
            features = load_features(SPLIT_TRAIN, data.method, data.names[row])
        except ResearchError:  # a method without an extractor cache has no evidence map
            features = {}
        for key in EVIDENCE_KEYS:
            if key in features:
                evidence = upsample(np.asarray(features[key], dtype=np.float64).mean(axis=0))
                item[f"auc_{key}"] = hole_auc(-evidence, truth_hole)
        smooth = ndimage.gaussian_filter(probability[row].astype(np.float64), POINTING_SMOOTH_SIGMA)
        peak = np.unravel_index(int(np.argmin(smooth)), smooth.shape)
        item["pointing_hit"] = bool(truth_hole[peak]) if truth_hole.any() else None
        regions = proposals(probability[row])
        scores = [
            float(miou_per_image((~region).astype(np.uint8)[None], data.truth[row][None])[0])
            for region in regions[: max(PROPOSAL_COUNTS)]
        ]
        floor = max(item["base"], item["blank"])
        for count in PROPOSAL_COUNTS:
            item[f"oracle_top{count}"] = max([floor, *scores[:count]])
        item["first_proposal"] = scores[0] if scores else item["blank"]
        for key in EVIDENCE_KEYS:
            if key in features and regions:
                evidence = upsample(np.asarray(features[key], dtype=np.float64).mean(axis=0))
                pick = evidence_pick(evidence, regions[:VERIFIER_PROPOSALS])
                item[f"verifier_{key}"] = scores[pick]
            elif key in features:
                item[f"verifier_{key}"] = item["base"]
        per_image.append(item)

    def mean_of(items: list[dict[str, Any]], key: str) -> float | None:
        values = [i[key] for i in items if i.get(key) is not None]
        return float(np.mean(values)) if values else None

    keys = [
        "base",
        "blank",
        "auc_head",
        *(f"auc_{k}" for k in EVIDENCE_KEYS),
        "pointing_hit",
        "first_proposal",
        *(f"verifier_{k}" for k in EVIDENCE_KEYS),
        *(f"oracle_top{c}" for c in PROPOSAL_COUNTS),
    ]
    bins = []
    for low, high in itertools.pairwise(HOLE_BIN_EDGES):
        chosen = [i for i in per_image if low <= i["share"] < high]
        entry: dict[str, Any] = {"low": low, "high": high, "n": len(chosen)}
        entry.update({k: mean_of(chosen, k) for k in keys})
        bins.append(entry)
    n_method = len(data.names)
    return {
        "n_full": len(per_image),
        "all": {k: mean_of(per_image, k) for k in keys},
        "bins": bins,
        "method_gain_if_perfect_verifier": {
            f"top{c}": float(sum(i[f"oracle_top{c}"] - i["base"] for i in per_image) / n_method)
            for c in PROPOSAL_COUNTS
        },
        "method_gain_small_holes_only": {
            f"top{c}": float(
                sum(
                    i[f"oracle_top{c}"] - i["base"]
                    for i in per_image
                    if i["share"] < HOLE_BIN_EDGES[1]
                )
                / n_method
            )
            for c in PROPOSAL_COUNTS
        },
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {
        method: analyse(load_method(method, OOF_TAGS_OF_BEST[method]), OOF_TAGS_OF_BEST[method])
        for method in METHODS
    }
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for method, row in report.items():
        print(method, "full images", row["n_full"])
        print(
            "  all ", {k: (round(v, 3) if v is not None else None) for k, v in row["all"].items()}
        )
        for item in row["bins"]:
            print(
                f"  hole [{item['low']}, {item['high']}) n={item['n']}",
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in item.items()
                    if k not in ("low", "high", "n")
                },
            )
        print("  gain if perfect verifier", row["method_gain_if_perfect_verifier"])
        print("  of which holes < 5 %", row["method_gain_small_holes_only"])


if __name__ == "__main__":
    main()
