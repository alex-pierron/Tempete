"""Candidate assembly and train-side evaluation for the full-resolution networks (w-loc-small).

`evaluate` scores the probability maps of `small_train` on the real train split (overall, by
regime, by size of the zone to find), alone and averaged with an out-of-fold map of team B2, and
sweeps the decision threshold. `build` replaces, in an existing complete candidate, the masks of
the chosen methods by the thresholded network maps, then re-applies the rectangle overlay of
team C (confident exact rectangles win, as in every candidate since v7c).

Usage:
    python -m hackathon.wmloc.research.small_candidate evaluate --maps out/u1_ss.npz \
        --method stegastamp --reference team_b2/cv/e_spatial3_loc.npz
    python -m hackathon.wmloc.research.small_candidate build --base b2n_v9_syn3.pt \
        --replace stegastamp=out/u1_ss.npz:test_finetuned --overlay c_rect_overlay.pt --out new.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.constants import GRID_SIZE, KNOWN_METHODS
from hackathon.wmloc.research.data import SPLIT_TEST, SPLIT_TRAIN, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError

DECISION_THRESHOLD = 0.5
OVERLAY_CONFIDENCE = 0.85
"""Rule of teams B and C: a rectangle replaces the mask when its confidence reaches this."""
SMALL_ZONE_FRACTION = 0.05
THRESHOLD_GRID: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
TRAIN_KEYS: tuple[str, ...] = ("train_synthetic", "train_finetuned_oof", "train_finetuned_oof_b")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2026


class SmallCandidateError(ResearchError):
    """A map container, a candidate or an overlay is inconsistent with the release."""


def miou_per_image(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Official per-image score: mean IoU of the two classes (1 when a union is empty)."""
    scores = np.zeros(len(truth), dtype=np.float64)
    for value in (False, True):
        p, g = prediction == value, truth == value
        union = (p | g).sum(axis=(1, 2))
        inter = (p & g).sum(axis=(1, 2))
        scores += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return scores / 2


def bootstrap_interval(values: NDArray[np.float64]) -> tuple[float, float]:
    """95 % bootstrap interval of the mean of `values`."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def breakdown(
    scores: NDArray[np.float64], truth: NDArray[np.bool_], is_full: NDArray[np.bool_]
) -> dict[str, float]:
    """Mean score overall, by regime and by size of the zone to find."""
    marked = truth.mean(axis=(1, 2))
    zone = np.where(is_full, 1 - marked, marked)
    small = zone < SMALL_ZONE_FRACTION
    low, high = bootstrap_interval(scores)
    report = {"all": float(scores.mean()), "ci_low": low, "ci_high": high}
    for label, group in (("full", is_full), ("rect", ~is_full)):
        report[label] = float(scores[group].mean())
        report[f"{label}_small"] = float(scores[group & small].mean())
        report[f"{label}_large"] = float(scores[group & ~small].mean())
    return report


def train_frame(method: str) -> tuple[list[str], NDArray[np.bool_], NDArray[np.bool_]]:
    """(names, truth bool [n, 128, 128], regime flags) of the train records of `method`."""
    records = [r for r in load_records(SPLIT_TRAIN) if r.method == method]
    names = [r.name for r in records]
    truth = load_truth(names).astype(bool)
    return names, truth, np.array([bool(r.full_watermark) for r in records])


def mean_maps(paths: list[Path], key: str, names_key: str) -> tuple[list[str], NDArray[np.float32]]:
    """Names and the mean probability map `key` of several containers (seeds of one method).

    Raises:
        SmallCandidateError: the containers do not list the same images in the same order.
    """
    names: list[str] = []
    total: NDArray[np.float32] | None = None
    for path in paths:
        with np.load(path) as archive:
            current = [str(n) for n in archive[names_key]]
            probability = archive[key].astype(np.float32)
        if names and current != names:
            raise SmallCandidateError(f"{path} does not list the images of {paths[0]}")
        names = current
        total = probability if total is None else total + probability
    if total is None:
        raise SmallCandidateError("no map container given")
    return names, (total / np.float32(len(paths))).astype(np.float32)


def evaluate(maps: list[Path], method: str, reference: Path | None) -> dict[str, dict[str, float]]:
    """Train-side report of the train maps (mean over `maps`), alone and averaged with `reference`.

    Raises:
        SmallCandidateError: a container does not follow the train records of the method.
    """
    names, truth, is_full = train_frame(method)
    report: dict[str, dict[str, float]] = {}
    with np.load(maps[0]) as archive:
        keys = [k for k in TRAIN_KEYS if k in archive.files]
    mine: dict[str, NDArray[np.float32]] = {}
    for key in keys:
        map_names, mine[key] = mean_maps(maps, key, "names_train")
        if map_names != names:
            raise SmallCandidateError(f"{maps[0]} does not follow the train records of {method}")
    other: NDArray[np.float32] | None = None
    if reference is not None:
        with np.load(reference) as archive:
            other = archive[method].astype(np.float32)
        report["reference"] = breakdown(
            miou_per_image(other >= DECISION_THRESHOLD, truth), truth, is_full
        )
    for key, probability in mine.items():
        for threshold in THRESHOLD_GRID:
            scores = miou_per_image(probability >= threshold, truth)
            report[f"{key}@{threshold}"] = breakdown(scores, truth, is_full)
        if other is not None:
            blend = (probability + other) / 2
            report[f"{key}+reference@0.5"] = breakdown(
                miou_per_image(blend >= DECISION_THRESHOLD, truth), truth, is_full
            )
    return report


def parse_replacement(text: str) -> tuple[str, list[Path], str, float]:
    """`method=a.npz[+b.npz]:key[:threshold]` -> (method, paths to average, key, threshold).

    Raises:
        SmallCandidateError: malformed specification or unknown method.
    """
    method, _, rest = text.partition("=")
    parts = rest.split(":")
    if method not in KNOWN_METHODS or len(parts) not in (2, 3):
        raise SmallCandidateError(f"expected method=path.npz:key[:threshold], got {text!r}")
    threshold = float(parts[2]) if len(parts) == 3 else DECISION_THRESHOLD
    return method, [Path(p) for p in parts[0].split("+")], parts[1], threshold


def build(
    base: Path,
    replacements: list[tuple[str, list[Path], str, float]],
    overlay: Path | None,
    out: Path,
) -> dict[str, int]:
    """Write the candidate; returns the number of changed masks per source.

    Raises:
        SmallCandidateError: a name of a map or of the overlay is not a test image of its method.
    """
    loaded = torch.load(base, weights_only=True)
    names = [str(n) for n in loaded["image_names"]]
    masks = loaded["masks"].to(torch.uint8).clone()
    if masks.shape != (len(names), GRID_SIZE, GRID_SIZE):
        raise SmallCandidateError(f"unexpected base shape {tuple(masks.shape)}")
    row = {name: index for index, name in enumerate(names)}
    method_of = {r.name: r.method for r in load_records(SPLIT_TEST)}
    changed: dict[str, int] = {}
    for method, paths, key, threshold in replacements:
        map_names, probability = mean_maps(paths, key, "names_test")
        count = 0
        for index, name in enumerate(map_names):
            if method_of.get(name) != method or name not in row:
                raise SmallCandidateError(f"{name} of {paths[0]} is not a test image of {method}")
            new = torch.from_numpy((probability[index] >= threshold).astype(np.uint8))
            count += int(not torch.equal(new, masks[row[name]]))
            masks[row[name]] = new
        changed[method] = count
    if overlay is not None:
        boxes = torch.load(overlay, weights_only=True)
        count = 0
        for name, mask, confidence in zip(
            boxes["image_names"], boxes["masks"], boxes["confidence"], strict=True
        ):
            if float(confidence) < OVERLAY_CONFIDENCE:
                continue
            if str(name) not in row:
                raise SmallCandidateError(f"overlay image {name} is not in the base candidate")
            new = mask.to(torch.uint8)
            count += int(not torch.equal(new, masks[row[str(name)]]))
            masks[row[str(name)]] = new
        changed["overlay"] = count
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": masks}, out)
    return changed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and assemble w-loc-small candidates.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--maps", type=Path, nargs="+", required=True)
    evaluation.add_argument("--method", choices=KNOWN_METHODS, required=True)
    evaluation.add_argument("--reference", type=Path)
    assembly = commands.add_parser("build")
    assembly.add_argument("--base", type=Path, required=True)
    assembly.add_argument("--replace", nargs="+", required=True)
    assembly.add_argument("--overlay", type=Path)
    assembly.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        for key, row in evaluate(args.maps, args.method, args.reference).items():
            print(key, json.dumps({k: round(v, 4) for k, v in row.items()}))
        return
    replacements = [parse_replacement(text) for text in args.replace]
    print(json.dumps(build(args.base, replacements, args.overlay, args.out)))


if __name__ == "__main__":
    main()
