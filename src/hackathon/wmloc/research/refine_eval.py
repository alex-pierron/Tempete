"""Does fitting the box edges on the exact crop improve the scale-space mask? (rectangle regime)

For each train rectangle image: seed box = bounding box of the largest connected component of
the scale-space evidence above `SEED_Z_CUT` (dilated by `SEED_PAD_PX`), then `fit_rectangle`.
Reports, per image, the mIoU of the threshold mask, of the seed box and of the fitted box, plus
the soft margin / agreement of the fitted crop (used to gate the refinement).
The refinement has no trained parameter; only the gate would be fitted, on these numbers.

Usage (kit venv): python -m hackathon.wmloc.research.refine_eval --method pixelseal --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hackathon.wmloc.constants import MAX_ASPECT_RATIO_BY_METHOD
from hackathon.wmloc.research.crossval import load_features
from hackathon.wmloc.research.data import SPLIT_TRAIN, load_image, load_records, load_truth
from hackathon.wmloc.research.decoding import Decoder
from hackathon.wmloc.research.evaluate import miou_per_image
from hackathon.wmloc.research.maps import DEFAULT_SIDES
from hackathon.wmloc.research.postprocess import largest_component
from hackathon.wmloc.research.rectangles import (
    Box,
    box_from_grid,
    box_to_mask,
    clamp_box,
    fit_rectangle,
    is_valid,
)

SEED_Z_CUT = 4.0
"""Evidence z-score of the seed region (the typical CV-fitted cut of the scale-space rule)."""
SEED_PAD_PX = 16


def run(method: str, limit: int | None) -> list[dict[str, float]]:
    """Fit every rectangle-regime train image of `method` that has scale-space caches."""
    records = [
        r for r in load_records(SPLIT_TRAIN) if r.method == method and r.full_watermark is False
    ]
    decoder = Decoder(method)
    max_aspect = MAX_ASPECT_RATIO_BY_METHOD[method]
    rows: list[dict[str, float]] = []
    for record in records[:limit]:
        feature = load_features(SPLIT_TRAIN, record, DEFAULT_SIDES, use_scale_space=True)
        if feature.scale_space is None:
            continue
        truth = load_truth([record.name])
        threshold_mask = (feature.scale_space >= SEED_Z_CUT).astype(np.uint8)
        seed_mask = largest_component(threshold_mask)
        seed = box_from_grid(seed_mask.astype(float), 0.5)
        row: dict[str, float] = {
            "threshold_miou": float(miou_per_image(threshold_mask[None], truth)[0])
        }
        if seed is None:
            row.update(
                seed_miou=0.0, fit_miou=0.0, margin=0.0, agreement=0.0, forwards=0.0, seconds=0.0
            )
            rows.append(row)
            continue
        padded: Box = clamp_box(
            (
                seed[0] - SEED_PAD_PX,
                seed[1] - SEED_PAD_PX,
                seed[2] + SEED_PAD_PX,
                seed[3] + SEED_PAD_PX,
            )
        )
        padded = padded if is_valid(padded, max_aspect) else seed
        start = time.time()
        if not is_valid(padded, max_aspect):
            rows.append(
                row
                | {
                    "seed_miou": 0.0,
                    "fit_miou": 0.0,
                    "margin": 0.0,
                    "agreement": 0.0,
                    "forwards": 0.0,
                    "seconds": 0.0,
                }
            )
            continue
        fit = fit_rectangle(
            decoder, load_image(SPLIT_TRAIN, record.name), record.message, padded, max_aspect
        )
        row.update(
            seed_miou=float(miou_per_image(box_to_mask(padded)[None], truth)[0]),
            fit_miou=float(miou_per_image(box_to_mask(fit.box)[None], truth)[0]),
            margin=fit.margin,
            agreement=fit.agreement,
            forwards=float(fit.forwards),
            seconds=time.time() - start,
        )
        rows.append(row)
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Edge-fit evaluation on train rectangles.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = run(args.method, args.limit)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keys = ("threshold_miou", "seed_miou", "fit_miou", "forwards", "seconds")
    print({k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}, len(rows))


if __name__ == "__main__":
    main()
