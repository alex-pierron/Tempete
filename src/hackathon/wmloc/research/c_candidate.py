"""Merge detected rectangles of the window search into a base submission (team C).

The base submission (the best current pipeline: Pixel Seal and the regime routing) stays as it is
for every image the search did not detect a rectangle in. For the images of `method` whose search
was run, `detected` is true and the box is valid, the base mask is replaced by the box mask.
Nothing else changes, so the candidate cannot lose on images the search abstained on.

Usage (main venv):
    python -m hackathon.wmloc.research.c_candidate --base <base.pt> \
        --merge trustmark=<results dir> --merge pixelseal=<results dir> --out <candidate.pt>
Then validate with `python -m hackathon.wmloc validate <candidate.pt> --manifest <test manifest>`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.constants import (
    GRID_SIZE,
    KNOWN_METHODS,
    MAX_ASPECT_RATIO_BY_METHOD,
    SUBMISSION_KEY_MASKS,
    SUBMISSION_KEY_NAMES,
)
from hackathon.wmloc.research.c_rect_search import Box, is_detection
from hackathon.wmloc.research.c_results import ResultStoreError, load_all
from hackathon.wmloc.research.data import load_records
from hackathon.wmloc.research.errors import ResearchError


class CandidateError(ResearchError):
    """The base submission or a result file is inconsistent with the release."""


@dataclass(frozen=True)
class MergeReport:
    """What the merge changed.

    Attributes:
        n_images: Images of the base submission.
        n_method: Test images of the merged method.
        n_searched: Of those, images with a result file that was searched.
        n_replaced: Base masks replaced by a detected box.
        n_missing: Images of the method without any result file (base kept).
    """

    n_images: int
    n_method: int
    n_searched: int
    n_replaced: int
    n_missing: int


def load_base(path: Path) -> tuple[list[str], torch.Tensor]:
    """Names and uint8 masks [N, 128, 128] of a base submission.

    Raises:
        CandidateError: wrong keys or shape.
    """
    payload = torch.load(path, weights_only=True)
    if not isinstance(payload, dict) or SUBMISSION_KEY_NAMES not in payload:
        raise CandidateError(f"{path} is not a submission file")
    names = [str(n) for n in payload[SUBMISSION_KEY_NAMES]]
    masks = torch.as_tensor(payload[SUBMISSION_KEY_MASKS]).to(torch.uint8)
    if masks.shape != (len(names), GRID_SIZE, GRID_SIZE):
        raise CandidateError(f"unexpected mask shape {tuple(masks.shape)}")
    return names, masks


def accepted_box(payload: dict[str, Any], method: str) -> Box | None:
    """The detected box of a result payload, or None (not searched, abstained, invalid)."""
    if not payload.get("searched") or not payload.get("detected") or payload.get("box") is None:
        return None
    box = Box(**payload["box"])
    if not is_detection(box, float(payload.get("agreement", 0.0))):
        return None
    return box if box.is_valid(MAX_ASPECT_RATIO_BY_METHOD[method]) else None


def merge_rectangles(
    names: list[str],
    masks: torch.Tensor,
    method_names: set[str],
    results_dir: Path,
    method: str,
) -> tuple[torch.Tensor, MergeReport]:
    """Replace the masks of detected rectangles; returns the new masks and a report."""
    merged = masks.clone()
    index = {name: i for i, name in enumerate(names)}
    searched = replaced = missing = 0
    unknown = sorted(method_names - index.keys())
    if unknown:
        raise CandidateError(f"{unknown[0]} is not in the base submission")
    try:
        stored = {str(p["name"]): p for p in load_all(results_dir)}
    except ResultStoreError as error:
        raise CandidateError(str(error)) from error
    for name in sorted(method_names):
        payload = stored.get(name)
        if payload is None:
            missing += 1
            continue
        searched += int(bool(payload.get("searched")))
        box = accepted_box(payload, method)
        if box is not None:
            merged[index[name]] = torch.from_numpy(box.to_mask())
            replaced += 1
    report = MergeReport(len(names), len(method_names), searched, replaced, missing)
    return merged, report


def parse_merge(spec: str) -> tuple[str, Path]:
    """`method=directory` -> (method, directory).

    Raises:
        CandidateError: malformed spec or unknown method.
    """
    method, separator, directory = spec.partition("=")
    if not separator or method not in KNOWN_METHODS:
        raise CandidateError(
            f"--merge expects <{'|'.join(KNOWN_METHODS)}>=<results dir>, got {spec!r}"
        )
    return method, Path(directory)


def collect_overlay(
    method_names: set[str], results_dir: Path, method: str
) -> dict[str, tuple[np.ndarray, float]]:
    """`{image name: (128 x 128 uint8 box mask, confidence)}` of the accepted detections.

    The confidence is the bit agreement of the box with the known message (0.8 to 1.0: the
    detection gate is 0.8, chance level 0.5).

    Raises:
        CandidateError: no result in the directory.
    """
    try:
        stored = {str(p["name"]): p for p in load_all(results_dir)}
    except ResultStoreError as error:
        raise CandidateError(str(error)) from error
    entries: dict[str, tuple[np.ndarray, float]] = {}
    for name in sorted(method_names & stored.keys()):
        box = accepted_box(stored[name], method)
        if box is not None:
            entries[name] = (box.to_mask(), float(stored[name]["agreement"]))
    return entries


def write_overlay(path: Path, entries: dict[str, tuple[np.ndarray, float]]) -> None:
    """Cumulative overlay `{"image_names", "masks" uint8 [n,128,128], "confidence" float [n]}`.

    Written to `<path>.part` then renamed, so a reader never sees a half-written file.
    """
    names = sorted(entries)
    payload = {
        SUBMISSION_KEY_NAMES: names,
        SUBMISSION_KEY_MASKS: torch.from_numpy(np.stack([entries[n][0] for n in names])).to(
            torch.uint8
        )
        if names
        else torch.zeros((0, GRID_SIZE, GRID_SIZE), dtype=torch.uint8),
        "confidence": torch.tensor([entries[n][1] for n in names], dtype=torch.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    """CLI entry point: `--merge method=dir` (repeatable) into a base and/or a cumulative overlay."""
    parser = argparse.ArgumentParser(description="Merge detected rectangles into a submission.")
    parser.add_argument("--base", type=Path, help="submission to overlay (needed with --out)")
    parser.add_argument(
        "--merge", action="append", required=True, help="method=results directory (repeatable)"
    )
    parser.add_argument("--out", type=Path, help="complete candidate: base + detected rectangles")
    parser.add_argument("--overlay-out", type=Path, help="cumulative overlay file (.pt)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="only detections at least this confident go into --out (the overlay keeps all)",
    )
    args = parser.parse_args()
    if args.out is None and args.overlay_out is None:
        parser.error("give --out and/or --overlay-out")
    if args.out is not None and args.base is None:
        parser.error("--out needs --base")
    records = load_records("test")
    entries: dict[str, tuple[np.ndarray, float]] = {}
    for spec in args.merge:
        method, directory = parse_merge(spec)
        method_names = {r.name for r in records if r.method == method}
        for name, entry in collect_overlay(method_names, directory, method).items():
            if name not in entries or entry[1] > entries[name][1]:
                entries[name] = entry
        print(method, directory, "detections so far:", len(entries))
    if args.overlay_out is not None:
        write_overlay(args.overlay_out, entries)
        print("overlay:", args.overlay_out, len(entries), "images")
    if args.out is not None:
        names, masks = load_base(args.base)
        index = {name: i for i, name in enumerate(names)}
        unknown = sorted(entries.keys() - index.keys())
        if unknown:
            raise CandidateError(f"{unknown[0]} is not in the base submission")
        for name, (mask, confidence) in entries.items():
            if confidence >= args.min_confidence:
                masks[index[name]] = torch.from_numpy(mask)
        torch.save({SUBMISSION_KEY_NAMES: names, SUBMISSION_KEY_MASKS: masks}, args.out)


if __name__ == "__main__":
    main()
