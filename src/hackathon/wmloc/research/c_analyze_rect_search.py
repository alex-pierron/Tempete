"""Score cached rectangle searches against the train ground truth (team C).

For every searched train image of the rect regime, compares three masks with the official
per-image mIoU: all zeros (the constant that wins without localization), the searched box when
`detected` (else zeros), and the box without the detection gate. Images are split into true
rectangles (mask fills >= `TRUE_RECT_FILL` of its bounding box) and object-shaped masks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.research.c_rect_search import Box
from hackathon.wmloc.research.c_results import load_all
from hackathon.wmloc.research.data import load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import miou_per_image

TRUE_RECT_FILL = 0.97
"""A mask whose pixels fill at least this share of its bounding box is a true rectangle."""
AREA_BIN_EDGES: tuple[float, ...] = (0.0, 0.02, 0.05, 0.15, 1.0)
"""Share of the image covered by the mask, the same bins as the phase 1 reports."""
CELL_PX = 4


class AnalysisError(ResearchError):
    """No results found or a result file is malformed."""


@dataclass(frozen=True)
class ImageOutcome:
    """Scores of one searched image."""

    name: str
    true_rect: bool
    area: float
    detected: bool
    box_iou: float
    miou_zeros: float
    miou_gated: float
    miou_ungated: float
    seconds: float
    n_decodes: int


def ground_truth_geometry(mask: np.ndarray) -> tuple[bool, float, Box | None]:
    """(is a true rectangle, image share, pixel box) of a 128 x 128 ground-truth mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return False, 0.0, None
    box = Box(
        int(ys.min()) * CELL_PX,
        int(xs.min()) * CELL_PX,
        (int(ys.max()) + 1) * CELL_PX,
        (int(xs.max()) + 1) * CELL_PX,
    )
    fill = float(mask.sum()) / ((box.height // CELL_PX) * (box.width // CELL_PX))
    return fill >= TRUE_RECT_FILL, float(mask.mean()), box


def load_results(directory: Path) -> list[dict[str, Any]]:
    """Every JSON payload of a results directory (searched images only).

    Raises:
        AnalysisError: the directory has no result file.
    """
    payloads = load_all(directory)
    searched = [p for p in payloads if p.get("searched")]
    if not searched:
        raise AnalysisError(f"no searched image in {directory}")
    return searched


def score_outcomes(payloads: list[dict[str, Any]]) -> list[ImageOutcome]:
    """Score `payloads` (train images with labels) against the official masks."""
    names = [str(p["name"]) for p in payloads]
    truth = load_truth(names)
    zeros = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    outcomes: list[ImageOutcome] = []
    for payload, gold in zip(payloads, truth, strict=True):
        true_rect, area, gold_box = ground_truth_geometry(gold)
        raw = payload.get("box")
        box = None if raw is None else Box(**raw)
        mask = zeros if box is None else box.to_mask()
        detected = bool(payload["detected"])
        gated = mask if detected else zeros
        scores = miou_per_image(np.stack([zeros, gated, mask]), np.stack([gold, gold, gold]))
        iou = 0.0 if box is None or gold_box is None else box.iou(gold_box)
        outcomes.append(
            ImageOutcome(
                name=str(payload["name"]),
                true_rect=true_rect,
                area=area,
                detected=detected,
                box_iou=iou,
                miou_zeros=float(scores[0]),
                miou_gated=float(scores[1]),
                miou_ungated=float(scores[2]),
                seconds=float(payload["seconds"]),
                n_decodes=int(payload["n_decodes"]),
            )
        )
    return outcomes


def summarize(outcomes: list[ImageOutcome]) -> dict[str, Any]:
    """Aggregate scores: overall, per mask kind and per area bin."""

    def block(rows: list[ImageOutcome]) -> dict[str, float]:
        if not rows:
            return {"n": 0.0}
        return {
            "n": float(len(rows)),
            "detected_rate": float(np.mean([r.detected for r in rows])),
            "mean_box_iou": float(np.mean([r.box_iou for r in rows])),
            "miou_zeros": float(np.mean([r.miou_zeros for r in rows])),
            "miou_gated": float(np.mean([r.miou_gated for r in rows])),
            "miou_ungated": float(np.mean([r.miou_ungated for r in rows])),
            "seconds_per_image": float(np.mean([r.seconds for r in rows])),
            "decodes_per_image": float(np.mean([r.n_decodes for r in rows])),
        }

    by_area = {
        f"{low:g}-{high:g}": block([r for r in outcomes if low <= r.area < high])
        for low, high in itertools.pairwise(AREA_BIN_EDGES)
    }
    return {
        "all": block(outcomes),
        "true_rect": block([r for r in outcomes if r.true_rect]),
        "object_shaped": block([r for r in outcomes if not r.true_rect]),
        "true_rect_by_area": {
            f"{low:g}-{high:g}": block(
                [r for r in outcomes if r.true_rect and low <= r.area < high]
            )
            for low, high in itertools.pairwise(AREA_BIN_EDGES)
        },
        "by_area": by_area,
    }


def main() -> None:
    """CLI: print the summary of a results directory as JSON."""
    parser = argparse.ArgumentParser(description="Score cached rectangle searches (train).")
    parser.add_argument("results", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    outcomes = score_outcomes(load_results(args.results))
    summary = summarize(outcomes)
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text, encoding="utf-8")
    for row in outcomes:
        print(
            f"{row.name} true_rect={row.true_rect} area={row.area:.3f} det={row.detected} "
            f"iou={row.box_iou:.2f} miou zeros/gated/ungated="
            f"{row.miou_zeros:.2f}/{row.miou_gated:.2f}/{row.miou_ungated:.2f}"
        )


if __name__ == "__main__":
    main()
