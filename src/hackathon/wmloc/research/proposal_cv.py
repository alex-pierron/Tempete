"""Cross-validated selection of segment proposals (phase 2, team A).

Reads the cached proposals (`proposals.py`) of the TRAIN images and the phase-1 out-of-fold masks
(`reports/phase1/wmloc/team-a/cv_masks_<method>.npz`, the fallback). For every method and regime:

* rect regime: a proposal is positive when its mask has IoU >= `POSITIVE_IOU` with the marked zone;
  the picked proposal is the most probable one and is used only when its probability clears a cut
  chosen by an inner cross-validation; otherwise the phase-1 mask is kept;
* full regime: same with the proposal read as the clean HOLE (mask = 1 - proposal).

Everything reported is out-of-fold (same stratified folds as phase 1) and the images are routed by
the phase-1 out-of-fold regime switch, so a routing error costs what it costs.

Usage: python -m hackathon.wmloc.research.proposal_cv --cache <proposals dir> --phase1 <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.crossval import load_features
from hackathon.wmloc.research.data import Record, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import (
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    miou_per_image,
    stratified_folds,
)
from hackathon.wmloc.research.maps import DEFAULT_SIDES
from hackathon.wmloc.research.proposals import STAT_COLUMNS, proposal_cache_path

POSITIVE_IOU = 0.6
"""A proposal is a positive example when its mIoU with the target zone reaches this value."""
INNER_FOLDS = 4
CUT_GRID = tuple(np.linspace(0.2, 0.95, 16))
CHANCE = 0.5
EVIDENCE_COLUMNS = 3
FEATURE_COUNT = 10 + EVIDENCE_COLUMNS
"""Model features per proposal: 10 decode / size columns + 3 scale-space evidence columns."""
MODEL_SEED = DEFAULT_SEED
MIN_TRAIN_ROWS = 20


class ProposalCvError(ResearchError):
    """A cache file is missing or the shapes are inconsistent."""


@dataclass
class ImageProposals:
    """Cached proposals of one image."""

    masks: np.ndarray
    features: np.ndarray


def evidence_columns(evidence: np.ndarray | None, masks: np.ndarray) -> np.ndarray:
    """[P, 3]: mean scale-space evidence inside the mask, outside, and the contrast (zeros if absent)."""
    rows = np.zeros((len(masks), EVIDENCE_COLUMNS), dtype=np.float32)
    if evidence is None:
        return rows
    for k, mask in enumerate(masks):
        inside = mask.astype(bool)
        mean_in = float(evidence[inside].mean()) if inside.any() else 0.0
        mean_out = float(evidence[~inside].mean()) if (~inside).any() else 0.0
        rows[k] = (mean_in, mean_out, mean_in - mean_out)
    return rows


def load_evidence(split: str, record: Record) -> np.ndarray | None:
    """Phase-1 best-scale z-score evidence [128, 128] of an image, or None when not cached."""
    return load_features(split, record, DEFAULT_SIDES, use_scale_space=True).scale_space


def load_proposals(
    cache: Path, split: str, method: str, name: str, evidence: np.ndarray | None = None
) -> ImageProposals:
    """Proposals of one image with the model features (`proposal_features` + evidence columns).

    Raises:
        ProposalCvError: cache file missing.
    """
    path = proposal_cache_path(cache, split, method, name)
    if not path.is_file():
        raise ProposalCvError(f"missing {path}")
    with np.load(path) as handle:
        masks, stats = handle["masks"], handle["stats"]
    return ImageProposals(
        masks, np.hstack([proposal_features(stats), evidence_columns(evidence, masks)])
    )


def proposal_features(stats: np.ndarray) -> np.ndarray:
    """Model features [P, 10] from the stored statistics: decode scores (NaN -> chance), size, rank."""
    col = {name: i for i, name in enumerate(STAT_COLUMNS)}
    if len(stats) == 0:
        return np.zeros((0, 10), dtype=np.float32)
    inscribed = stats[:, col["inscr_agree"]]
    has_inscribed = ~np.isnan(inscribed)
    best_box = stats[:, col["bbox_agree"]].max()
    return np.column_stack(
        [
            stats[:, col["bbox_agree"]],
            stats[:, col["bbox_margin"]],
            np.where(has_inscribed, inscribed, CHANCE),
            np.where(has_inscribed, stats[:, col["inscr_margin"]], 0.0),
            has_inscribed.astype(np.float32),
            stats[:, col["det_score"]],
            stats[:, col["area"]],
            stats[:, col["bbox_agree"]] - best_box,
            np.log1p(np.nan_to_num(stats[:, col["inscr_h_px"]] * stats[:, col["inscr_w_px"]])),
            np.arange(len(stats), dtype=np.float32),
        ]
    ).astype(np.float32)


def target_masks(proposal_masks: np.ndarray, full_regime: bool) -> np.ndarray:
    """The mask a proposal would predict: itself (rect) or its complement (full)."""
    return (1 - proposal_masks) if full_regime else proposal_masks


def proposal_ious(
    images: list[ImageProposals], truth: np.ndarray, full_regime: bool
) -> list[np.ndarray]:
    """Official per-image mIoU of every proposal's predicted mask against the truth, per image."""
    out: list[np.ndarray] = []
    for item, gold in zip(images, truth, strict=True):
        if len(item.masks) == 0:
            out.append(np.zeros(0))
            continue
        predicted = target_masks(item.masks, full_regime)
        out.append(miou_per_image(predicted, np.broadcast_to(gold, predicted.shape)))
    return out


def new_model() -> HistGradientBoostingClassifier:
    """Small, regularized gradient boosting (a few hundred rows per fit)."""
    return HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.08,
        max_leaf_nodes=6,
        min_samples_leaf=8,
        l2_regularization=1.0,
        random_state=MODEL_SEED,
    )


def stack_rows(
    images: list[ImageProposals], ious: list[np.ndarray], indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and 0/1 labels of the proposals of the images at `indices`."""
    xs = [images[i].features for i in indices if len(images[i].features)]
    ys = [(ious[i] >= POSITIVE_IOU).astype(int) for i in indices if len(images[i].features)]
    if not xs:
        return np.zeros((0, FEATURE_COUNT), dtype=np.float32), np.zeros(0, dtype=int)
    return np.concatenate(xs), np.concatenate(ys)


def fit_or_none(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier | None:
    """Fitted model, or None when there are too few rows or a single class."""
    if len(y) < MIN_TRAIN_ROWS or len(set(y.tolist())) < 2:
        return None
    return new_model().fit(x, y)


def best_pick(
    model: HistGradientBoostingClassifier | None, item: ImageProposals
) -> tuple[int, float]:
    """(index of the most probable proposal, its probability); (-1, 0) when no proposal or no model."""
    if model is None or len(item.features) == 0:
        return -1, 0.0
    prob = model.predict_proba(item.features)[:, 1]
    index = int(np.argmax(prob))
    return index, float(prob[index])


def choose_cut(
    images: list[ImageProposals],
    ious: list[np.ndarray],
    fallback: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> float:
    """Probability cut maximizing the mean mIoU on `indices`, from inner out-of-fold probabilities.

    Args:
        images: Proposals of every image of the regime.
        ious: Proposal IoUs, aligned with `images`.
        fallback: Per-image mIoU of the phase-1 fallback mask, aligned with `images`.
        indices: Positions of the training images.
        seed: Fold seed.
    """
    inner = stratified_folds(["x"] * len(indices), INNER_FOLDS, seed)
    picks: dict[int, tuple[int, float]] = {}
    for fold in range(INNER_FOLDS):
        model = fit_or_none(*stack_rows(images, ious, indices[inner != fold]))
        for i in indices[inner == fold]:
            picks[int(i)] = best_pick(model, images[int(i)])
    best_cut, best_score = 1.0, -1.0
    for cut in CUT_GRID:
        score = float(
            np.mean(
                [
                    ious[i][picks[i][0]] if picks[i][1] >= cut and picks[i][0] >= 0 else fallback[i]
                    for i in picks
                ]
            )
        )
        if score > best_score:
            best_cut, best_score = float(cut), score
    return best_cut


def evaluate_regime(
    images: list[ImageProposals], ious: list[np.ndarray], fallback: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    """Out-of-fold mIoU of "pick a proposal if confident, else fallback" over one regime's images."""
    final = np.zeros(len(images))
    used = np.zeros(len(images), dtype=bool)
    for fold in sorted(set(folds.tolist())):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        cut = choose_cut(images, ious, fallback, train, MODEL_SEED + fold)
        model = fit_or_none(*stack_rows(images, ious, train))
        for i in held:
            index, prob = best_pick(model, images[int(i)])
            if index >= 0 and prob >= cut:
                final[i], used[i] = ious[int(i)][index], True
            else:
                final[i] = fallback[i]
    upper = np.array([ious[i].max() if len(ious[i]) else fallback[i] for i in range(len(images))])
    return {
        "n": float(len(images)),
        "fallback": float(fallback.mean()),
        "with_proposals": float(final.mean()),
        "gain": float(final.mean() - fallback.mean()),
        "used_fraction": float(used.mean()),
        "upper_bound_best_proposal": float(np.maximum(upper, 0).mean()),
    }


def run(cache: Path, phase1: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Per method and regime results (regime = truth regime, images routed like phase 1 for the global number)."""
    results: dict[str, dict[str, dict[str, float]]] = {}
    for method in KNOWN_METHODS:
        z = np.load(phase1 / f"cv_masks_{method}.npz", allow_pickle=True)
        names = [str(n) for n in z["names"]]
        truth = load_truth(names)
        is_full = np.asarray(z["is_full"], dtype=bool)
        fallback_masks = np.asarray(z["masks"])
        folds = stratified_folds([f"{method}/{f}" for f in is_full], DEFAULT_FOLDS, DEFAULT_SEED)
        by_name = {r.name: r for r in load_records("train")}
        if any(n not in by_name for n in names):
            raise ProposalCvError("phase-1 names not in the train metadata")
        items = [
            load_proposals(cache, "train", method, n, load_evidence("train", by_name[n]))
            for n in names
        ]
        fallback_iou = miou_per_image(fallback_masks, truth)
        results[method] = {}
        for regime, flag in (("rect", False), ("full", True)):
            idx = np.flatnonzero(is_full == flag)
            sub_items = [items[i] for i in idx]
            sub_ious = proposal_ious(sub_items, truth[idx], flag)
            results[method][regime] = evaluate_regime(
                sub_items, sub_ious, fallback_iou[idx], folds[idx]
            )
            print(method, regime, results[method][regime], flush=True)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Out-of-fold evaluation of proposal selection.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.cache, args.phase1), indent=1))


if __name__ == "__main__":
    main()
