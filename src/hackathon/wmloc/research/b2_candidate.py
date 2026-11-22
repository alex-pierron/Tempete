"""Turn B2 test probabilities into a validated 1 020-image submission file (never submits).

Methods the learned head covers take its (seed-averaged, thresholded) masks; every other image
keeps the mask of a fallback submission (team A's candidate for StegaStamp, which has no spatial
head). The output is checked by the official validator of `hackathon.wmloc.submission`.

Usage (kit venv or main venv):
    python -m hackathon.wmloc.research.b2_candidate --fits fit_j_s0 fit_j_s1 \
        --methods pixelseal trustmark --fallback <team A .pt> --out <candidate .pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.research.b2_train import DECISION_THRESHOLD, results_dir
from hackathon.wmloc.research.data import SPLIT_TEST, load_records
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.submission import save_submission

NAMES_KEY_PREFIX = "names_"


class CandidateBuildError(ResearchError):
    """The fits disagree on the test images, or the fallback does not cover them."""


def load_fit(
    tag: str, directory: Path | None = None
) -> dict[str, tuple[list[str], NDArray[np.float32]]]:
    """`{method: (test names, probabilities [n, 128, 128])}` of one `fit` output.

    Raises:
        CandidateBuildError: the file is missing.
    """
    path = (directory or results_dir()) / f"{tag}_test.npz"
    if not path.is_file():
        raise CandidateBuildError(f"missing fit output {path}")
    out: dict[str, tuple[list[str], NDArray[np.float32]]] = {}
    with np.load(path) as archive:
        for key in archive.files:
            if key.startswith(NAMES_KEY_PREFIX):
                method = key[len(NAMES_KEY_PREFIX) :]
                out[method] = (
                    [str(name) for name in archive[key]],
                    archive[method].astype(np.float32),
                )
    return out


def average_fits(
    tags: list[str], method: str, directory: Path | None = None
) -> tuple[list[str], NDArray[np.float32]]:
    """Seed-averaged probabilities of `method` over the fits `tags`.

    Raises:
        CandidateBuildError: no fit, method absent, or fits ordered over different images.
    """
    if not tags:
        raise CandidateBuildError("need at least one fit")
    names: list[str] | None = None
    total: NDArray[np.float32] | None = None
    for tag in tags:
        fit = load_fit(tag, directory)
        if method not in fit:
            raise CandidateBuildError(f"{tag} has no {method}")
        fit_names, probabilities = fit[method]
        if names is None:
            names, total = fit_names, probabilities.copy()
        elif fit_names != names or total is None:
            raise CandidateBuildError(f"{tag} is ordered over different images")
        else:
            total += probabilities
    if names is None or total is None:
        raise CandidateBuildError("no probabilities")
    return names, (total / len(tags)).astype(np.float32)


OVERLAY_MIN_CONFIDENCE = 0.85
"""Below this bit agreement a detected rectangle is not trusted over the learned head (team C)."""


def load_overlay(
    path: Path, base: Path | None = None, min_confidence: float = OVERLAY_MIN_CONFIDENCE
) -> dict[str, NDArray[np.uint8]]:
    """Masks that an overlay file imposes, keyed by image name.

    The overlay is `{"image_names": [...], "masks": uint8 [n, 128, 128]}` with an optional
    `"confidence"` float [n] (bit agreement of the detected box with the known message): images
    below `min_confidence` are dropped. When `base` is given the overlay is a FULL submission built
    on top of `base` (team C's rectangle candidates are v5 plus some replaced masks): only the
    images whose mask differs from `base` are kept.

    Raises:
        CandidateBuildError: names and masks disagree, or the base lacks an overlay image.
    """
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    names = [str(n) for n in loaded["image_names"]]
    masks = loaded["masks"].numpy().astype(np.uint8)
    if len(names) != len(masks):
        raise CandidateBuildError(f"{path}: {len(names)} names for {len(masks)} masks")
    if "confidence" in loaded:
        confidence = loaded["confidence"].numpy()
        keep = [i for i in range(len(names)) if float(confidence[i]) >= min_confidence]
        names, masks = [names[i] for i in keep], masks[keep]
    if base is None:
        return dict(zip(names, masks, strict=True))
    reference = torch.load(base, map_location="cpu", weights_only=True)
    index = {str(n): i for i, n in enumerate(reference["image_names"])}
    missing = [n for n in names if n not in index]
    if missing:
        raise CandidateBuildError(f"{base} lacks {len(missing)} overlay images, e.g. {missing[0]}")
    base_masks = reference["masks"].numpy().astype(np.uint8)
    return {
        name: mask
        for name, mask in zip(names, masks, strict=True)
        if (base_masks[index[name]] != mask).any()
    }


def build_candidate(
    tags: list[str],
    methods: list[str],
    fallback_path: Path,
    out_path: Path,
    threshold: float = DECISION_THRESHOLD,
    directory: Path | None = None,
    method_fits: dict[str, list[str]] | None = None,
    overlays: list[tuple[Path, Path | None]] | None = None,
) -> dict[str, Any]:
    """Write and validate the candidate; returns image counts in / out and the replaced share.

    Args:
        tags: Fit outputs averaged for every method in `methods`.
        methods: Methods taken from the learned head (the others keep the fallback masks).
        fallback_path: Submission whose masks fill every image the head does not cover.
        out_path: Where the validated file is written.
        threshold: Probability threshold of the binary mask.
        directory: Folder of the fit outputs (default: `results_dir()`).
        method_fits: Per-method fit lists that override `tags` (e.g. a Pixel Seal head with
            extractor features next to content-only heads for the other two methods).

    Raises:
        CandidateBuildError: the fallback lacks a test image.
    """
    test_names = [r.name for r in load_records(SPLIT_TEST)]
    fallback = torch.load(fallback_path, map_location="cpu", weights_only=True)
    index_of = {str(name): i for i, name in enumerate(fallback["image_names"])}
    missing = [n for n in test_names if n not in index_of]
    if missing:
        raise CandidateBuildError(f"fallback lacks {len(missing)} test images, e.g. {missing[0]}")
    masks = np.stack([fallback["masks"][index_of[n]].numpy() for n in test_names]).astype(np.uint8)
    position = {name: i for i, name in enumerate(test_names)}
    replaced: dict[str, int] = {}
    overrides = method_fits or {}
    for method in dict.fromkeys([*methods, *overrides]):
        names, probabilities = average_fits(overrides.get(method, tags), method, directory)
        for name, probability in zip(names, probabilities, strict=True):
            masks[position[name]] = (probability >= threshold).astype(np.uint8)
        replaced[method] = len(names)
    overlaid = 0
    for overlay_path, overlay_base in overlays or []:
        for name, mask in load_overlay(overlay_path, overlay_base).items():
            if name not in position:
                raise CandidateBuildError(f"overlay image {name} is not a test image")
            masks[position[name]] = mask
            overlaid += 1
    report = save_submission(
        out_path, test_names, torch.from_numpy(masks), expected_names=set(test_names)
    )
    return {
        "path": str(out_path),
        "images_in": len(test_names),
        "images_out": report.n_images,
        "shape": list(report.shape),
        "bytes": report.n_bytes,
        "watermarked_fraction": report.watermarked_fraction,
        "replaced_by_learned_head": replaced,
        "overlay_masks_applied": overlaid,
        "grid": GRID_SIZE,
        "threshold": threshold,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the B2 candidate (never submits).")
    parser.add_argument("--fits", nargs="*", default=[], help="tags of `fit` outputs")
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument(
        "--method-fits",
        nargs="*",
        default=[],
        help="method=tag[,tag...] pairs overriding --fits for that method",
    )
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument(
        "--overlay",
        nargs="*",
        default=[],
        help="overlay[:base] pairs: masks imposed last; with a base, only the images that differ "
        "from that submission are kept (team C's rectangle candidates are built on v5)",
    )
    args = parser.parse_args()
    overrides = {
        pair.split("=", 1)[0]: pair.split("=", 1)[1].split(",") for pair in args.method_fits
    }
    overlays: list[tuple[Path, Path | None]] = []
    for item in args.overlay:
        overlay, _, base = item.partition(":")
        overlays.append((Path(overlay), Path(base) if base else None))
    summary = build_candidate(
        args.fits,
        args.methods,
        args.fallback,
        args.out,
        args.threshold,
        method_fits=overrides,
        overlays=overlays,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
