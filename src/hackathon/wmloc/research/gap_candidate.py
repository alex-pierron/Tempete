"""Apply the kept quick gain (Pixel Seal thresholds) to an existing candidate (never submits).

The base candidate mixes three things per image: the learned head, team C's rectangle overlay and
nothing else. Only the Pixel Seal images whose base mask IS the head mask at 0.5 are rewritten;
overlaid rectangles, TrustMark and StegaStamp stay byte-identical. The head probabilities are
re-derived from the B2 test fits and checked against the base before anything is changed.

Usage:
    python -m hackathon.wmloc.research.gap_candidate --base <base.pt> --out <candidate.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.research.b2_candidate import average_fits
from hackathon.wmloc.research.c_candidate import load_base
from hackathon.wmloc.research.data import SPLIT_TEST, load_records
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.gap_postprocess import PostConfig, postprocess, predicted_full
from hackathon.wmloc.submission import save_submission

TEST_FITS: dict[str, tuple[str, ...]] = {
    "pixelseal": ("loc2_ps_s0", "loc2_ps_s1", "loc2_ps_s2"),
}
"""B2 test fits behind the Pixel Seal masks of b2_v7c / c_v3 (3 seeds, averaged)."""

PIXELSEAL_RULE: dict[str, PostConfig] = {
    "full": PostConfig(threshold=0.8),
    "rect": PostConfig(threshold=0.2),
}
"""Rule chosen by nested CV (gap_cv, family `threshold`): the head under-sizes the zone to find
in both regimes, so the hole grows in the full regime and the marked zone in the rect regime."""

MAX_UNEXPLAINED_SHARE = 0.15
"""Largest share of a method's images allowed to differ from the head mask (the overlay)."""


class GapCandidateError(ResearchError):
    """The base candidate is not explained by the head fits, or lacks a test image."""


def build(
    base_path: Path, out_path: Path, rules: dict[str, dict[str, PostConfig]]
) -> dict[str, Any]:
    """Write the candidate and return its change report.

    Args:
        base_path: Candidate to start from (e.g. the 0.5819 file).
        out_path: Where the validated candidate is written.
        rules: `{method: {"full" | "rect": PostConfig}}` applied to the head images of `method`.

    Raises:
        GapCandidateError: a test image is missing, or too many base masks differ from the head.
    """
    names, base_masks = load_base(base_path)
    masks = base_masks.numpy().astype(np.uint8).copy()
    index = {name: row for row, name in enumerate(names)}
    test_names = [r.name for r in load_records(SPLIT_TEST)]
    missing = [n for n in test_names if n not in index]
    if missing:
        raise GapCandidateError(f"base lacks {len(missing)} test images, e.g. {missing[0]}")
    report: dict[str, Any] = {"base": str(base_path), "out": str(out_path), "methods": {}}
    for method, rule in rules.items():
        fit_names, probabilities = average_fits(list(TEST_FITS[method]), method)
        from_head = kept_overlay = changed = 0
        cells_before = cells_after = 0
        by_regime = {"full": 0, "rect": 0}
        for name, probability in zip(fit_names, probabilities, strict=True):
            row = index[name]
            head_mask = postprocess(probability, PostConfig())
            if not np.array_equal(head_mask, masks[row]):
                kept_overlay += 1
                continue
            from_head += 1
            regime = "full" if predicted_full(probability) else "rect"
            new_mask = postprocess(probability, rule[regime])
            by_regime[regime] += 1
            changed += int(not np.array_equal(new_mask, head_mask))
            minority_before = (1 - head_mask) if regime == "full" else head_mask
            minority_after = (1 - new_mask) if regime == "full" else new_mask
            cells_before += int(minority_before.sum())
            cells_after += int(minority_after.sum())
            masks[row] = new_mask
        if kept_overlay > MAX_UNEXPLAINED_SHARE * len(fit_names):
            raise GapCandidateError(
                f"{method}: {kept_overlay} of {len(fit_names)} base masks differ from the head "
                f"fits {TEST_FITS[method]}: wrong fits for this base"
            )
        report["methods"][method] = {
            "images": len(fit_names),
            "head_images": from_head,
            "untouched_non_head_images": kept_overlay,
            "changed": changed,
            "by_predicted_regime": by_regime,
            "minority_cells_before": cells_before,
            "minority_cells_after": cells_after,
            "rule": {k: vars(v) for k, v in rule.items()},
        }
    untouched = [
        row
        for row, name in enumerate(names)
        if not np.array_equal(masks[row], base_masks[row].numpy())
    ]
    report["images_changed_total"] = len(untouched)
    saved = save_submission(
        out_path, names, torch.from_numpy(masks), expected_names=set(test_names)
    )
    report.update(
        {"images_out": saved.n_images, "bytes": saved.n_bytes, "shape": list(saved.shape)}
    )
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.out, {"pixelseal": PIXELSEAL_RULE}), indent=1))


if __name__ == "__main__":
    main()
