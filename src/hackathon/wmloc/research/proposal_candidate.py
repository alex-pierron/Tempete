"""Build the phase-2 candidate: phase-1 masks, with confident segment proposals substituted.

For each method and regime the selection model and the probability cut are fitted on ALL train
images (`proposal_cv` functions), then applied to the test images routed to that regime by the
phase-1 switch. A test image whose best proposal is not confident keeps its fallback mask, taken
from an existing validated candidate file (default: the phase-1 scale-space candidate). Only the
substitution counts are new; the file is validated by the official validator before being written.

Usage (needs the proposals cache of train AND test):
    python -m hackathon.wmloc.research.proposal_candidate --cache <proposals dir> \
        --phase1 reports/phase1/wmloc/team-a --fallback <candidate.pt> --out <new candidate.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.candidate import fit_switch, test_agreement
from hackathon.wmloc.research.data import SPLIT_TEST, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import miou_per_image
from hackathon.wmloc.research.proposal_cv import (
    MODEL_SEED,
    best_pick,
    choose_cut,
    fit_or_none,
    load_evidence,
    load_proposals,
    proposal_ious,
    stack_rows,
    target_masks,
)
from hackathon.wmloc.submission import save_submission


class ProposalCandidateError(ResearchError):
    """Fallback file unreadable or inconsistent with the test set."""


def load_fallback(path: Path) -> tuple[list[str], np.ndarray]:
    """Names and masks [N, 128, 128] of an existing submission file."""
    data = torch.load(path, weights_only=True)
    names = [str(n) for n in data["image_names"]]
    masks = data["masks"].numpy().astype(np.uint8)
    if masks.shape != (len(names), 128, 128):
        raise ProposalCandidateError(f"{path}: unexpected mask shape {masks.shape}")
    return names, masks


def build(
    cache: Path,
    phase1: Path,
    fallback_path: Path,
    out_path: Path,
    allowed: frozenset[str] | None = None,
    cut_override: float | None = None,
) -> dict[str, object]:
    """Write the candidate and return substitution statistics per method and regime.

    Args:
        cache: Proposals cache (train and test).
        phase1: Folder with the phase-1 `cv_masks_<method>.npz` (fallback scores for the cut).
        fallback_path: Submission whose masks are kept unless a confident proposal replaces them.
        out_path: Output submission file.
        allowed: `method:regime` pairs where substitution is permitted (default: all); use it to
            keep only the pairs whose cross-validated gain is clear.
        cut_override: Fixed probability cut instead of the inner-CV argmax (the out-of-fold curve
            is flat between 0.4 and 0.65, so a fixed mid-plateau value is more stable).
    """
    fallback_names, masks = load_fallback(fallback_path)
    position = {n: i for i, n in enumerate(fallback_names)}
    test_records = load_records(SPLIT_TEST)
    if {r.name for r in test_records} != set(fallback_names):
        raise ProposalCandidateError("fallback names differ from the test manifest")
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for method in KNOWN_METHODS:
        z = np.load(phase1 / f"cv_masks_{method}.npz", allow_pickle=True)
        names = [str(n) for n in z["names"]]
        truth = load_truth(names)
        is_full = np.asarray(z["is_full"], dtype=bool)
        fallback_iou = miou_per_image(np.asarray(z["masks"]), truth)
        switch = fit_switch(method)
        by_name = {r.name: r for r in load_records("train")}
        items = [
            load_proposals(cache, "train", method, n, load_evidence("train", by_name[n]))
            for n in names
        ]
        stats[method] = {}
        for regime, flag in (("rect", False), ("full", True)):
            if allowed is not None and f"{method}:{regime}" not in allowed:
                continue
            idx = np.flatnonzero(is_full == flag)
            sub = [items[i] for i in idx]
            ious = proposal_ious(sub, truth[idx], flag)
            cut = (
                cut_override
                if cut_override is not None
                else choose_cut(sub, ious, fallback_iou[idx], np.arange(len(sub)), MODEL_SEED)
            )
            model = fit_or_none(*stack_rows(sub, ious, np.arange(len(sub))))
            used = seen = 0
            agreement_with_base: list[float] = []
            for record in test_records:
                if record.method != method:
                    continue
                routed_full = switch.is_full(test_agreement(record))
                if routed_full != flag:
                    continue
                seen += 1
                item = load_proposals(
                    cache, "test", method, record.name, load_evidence(SPLIT_TEST, record)
                )
                index, prob = best_pick(model, item)
                if index >= 0 and prob >= cut:
                    picked = target_masks(item.masks[index : index + 1], flag)[0]
                    base = masks[position[record.name]]
                    agreement_with_base.append(float(miou_per_image(picked[None], base[None])[0]))
                    masks[position[record.name]] = picked
                    used += 1
            stats[method][regime] = {
                "cut": cut,
                "test_images": float(seen),
                "substituted": float(used),
                "mean_miou_between_pick_and_base": float(np.mean(agreement_with_base))
                if agreement_with_base
                else float("nan"),
            }
    report = save_submission(
        out_path, fallback_names, torch.from_numpy(masks), expected_names=set(fallback_names)
    )
    return {
        "path": str(out_path),
        "bytes": report.n_bytes,
        "watermarked_fraction": report.watermarked_fraction,
        "substitutions": stats,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase-2 candidate (never submits).")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only", default=None, help="comma-separated method:regime pairs, e.g. pixelseal:rect"
    )
    parser.add_argument("--cut", type=float, default=None, help="fixed probability cut")
    args = parser.parse_args()
    allowed = frozenset(args.only.split(",")) if args.only else None
    print(
        json.dumps(
            build(args.cache, args.phase1, args.fallback, args.out, allowed, args.cut), indent=1
        )
    )


if __name__ == "__main__":
    main()
