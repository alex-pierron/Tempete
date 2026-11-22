"""Evaluation helpers for the full-resolution network runs of x-loc-unet.

Three commands, all on the Mac (seconds, no GPU):

* `split`: a merged container written by a two-method GPU task (keys `<method>__<key>`) becomes
  one standard `small_train` container per method (`<out-dir>/<prefix>_<method>.npz`).
* `evaluate`: out-of-fold mIoU of the MEAN of any list of `path:key` train maps (several seeds,
  several fine-tuning recipes), at several thresholds, alone and averaged with a reference map.
* `combine`: writes that mean (train and test, matching keys) as a standard container, so that
  `small_candidate build --replace <method>=<combined.npz>:test_finetuned` assembles it.

A spec `path:key` names a TRAIN key (`train_finetuned_oof`, `train_finetuned_oof_b`, ...); the
matching test key replaces the `train_` prefix by `test_` and drops `_oof`.

Usage:
    python -m hackathon.wmloc.research.x_unet_tools split --container x3.npz --out-dir d --prefix x3
    python -m hackathon.wmloc.research.x_unet_tools evaluate --method stegastamp \
        --maps u1.npz:train_finetuned_oof x2.npz:train_finetuned_oof_b [--reference ref.npz]
    python -m hackathon.wmloc.research.x_unet_tools combine --method stegastamp --maps ... \
        --out ens.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.small_candidate import (
    DECISION_THRESHOLD,
    THRESHOLD_GRID,
    breakdown,
    miou_per_image,
    train_frame,
)

METHOD_SEPARATOR = "__"
TRAIN_PREFIX = "train_"
TEST_PREFIX = "test_"
OOF_MARK = "_oof"


class UnetToolsError(ResearchError):
    """A spec, a container or a key is inconsistent."""


def test_key_of(train_key: str) -> str:
    """`train_finetuned_oof_b` -> `test_finetuned_b`; `train_synthetic` -> `test_synthetic`.

    Raises:
        UnetToolsError: the key is not a train key.
    """
    if not train_key.startswith(TRAIN_PREFIX):
        raise UnetToolsError(f"{train_key!r} is not a train key")
    return TEST_PREFIX + train_key[len(TRAIN_PREFIX) :].replace(OOF_MARK, "", 1)


def parse_spec(spec: str) -> tuple[Path, str]:
    """`path:key` -> (path, train key).

    Raises:
        UnetToolsError: no key given.
    """
    path, separator, key = spec.rpartition(":")
    if not separator or not path or not key:
        raise UnetToolsError(f"expected path:train_key, got {spec!r}")
    return Path(path), key


def split_container(container: Path, out_dir: Path, prefix: str) -> list[Path]:
    """Write one standard container per method found in a merged container; returns the paths.

    Raises:
        UnetToolsError: no `<method>__<key>` array in the container.
    """
    per_method: dict[str, dict[str, NDArray[np.generic]]] = {}
    with np.load(container, allow_pickle=False) as archive:
        for name in archive.files:
            method, separator, key = name.partition(METHOD_SEPARATOR)
            if separator and method in KNOWN_METHODS:
                per_method.setdefault(method, {})[key] = archive[name]
    if not per_method:
        raise UnetToolsError(f"no <method>__<key> array in {container}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for method, arrays in sorted(per_method.items()):
        target = out_dir / f"{prefix}_{method}.npz"
        np.savez(target, **arrays)  # type: ignore[arg-type]  # numpy stubs: kwargs of arrays
        written.append(target)
    return written


def mean_of_specs(
    specs: list[tuple[Path, str]], test: bool
) -> tuple[list[str], NDArray[np.float32]]:
    """Names and the mean map of the specs (train keys, or their test keys when `test`).

    Raises:
        UnetToolsError: the containers do not list the same images in the same order.
    """
    names: list[str] = []
    total: NDArray[np.float32] | None = None
    names_key = "names_test" if test else "names_train"
    for path, key in specs:
        with np.load(path, allow_pickle=False) as archive:
            current = [str(n) for n in archive[names_key]]
            wanted = test_key_of(key) if test else key
            if wanted not in archive.files:
                raise UnetToolsError(f"{path} has no {wanted}")
            probability = archive[wanted].astype(np.float32)
        if names and current != names:
            raise UnetToolsError(f"{path} does not list the images of {specs[0][0]}")
        names = current
        total = probability if total is None else total + probability
    if total is None:
        raise UnetToolsError("no map given")
    return names, (total / np.float32(len(specs))).astype(np.float32)


def evaluate_specs(
    specs: list[tuple[Path, str]], method: str, reference: Path | None
) -> dict[str, dict[str, float]]:
    """Out-of-fold report of the mean train map, alone and averaged with `reference`.

    Raises:
        UnetToolsError: the maps do not follow the train records of the method.
    """
    names, truth, is_full = train_frame(method)
    map_names, probability = mean_of_specs(specs, test=False)
    if map_names != names:
        raise UnetToolsError(f"the maps do not follow the train records of {method}")
    report: dict[str, dict[str, float]] = {}
    for threshold in THRESHOLD_GRID:
        report[f"mean@{threshold}"] = breakdown(
            miou_per_image(probability >= threshold, truth), truth, is_full
        )
    if reference is not None:
        with np.load(reference, allow_pickle=False) as archive:
            other = archive[method].astype(np.float32)
        report["reference"] = breakdown(
            miou_per_image(other >= DECISION_THRESHOLD, truth), truth, is_full
        )
        report["mean+reference@0.5"] = breakdown(
            miou_per_image((probability + other) / 2 >= DECISION_THRESHOLD, truth), truth, is_full
        )
    return report


def regime_binarise(
    probability: NDArray[np.float32], thresholds: tuple[float, float]
) -> NDArray[np.float32]:
    """0/1 maps with one threshold per GUESSED regime (full when half the map is >= 0.5).

    The output thresholded at 0.5 (as `small_candidate build` does) gives back these masks.
    """
    full_threshold, rect_threshold = thresholds
    guess_full = (probability >= DECISION_THRESHOLD).mean(axis=(1, 2)) >= DECISION_THRESHOLD
    threshold = np.where(guess_full, full_threshold, rect_threshold)[:, None, None]
    return (probability >= threshold).astype(np.float32)


def combine_specs(
    specs: list[tuple[Path, str]], out: Path, thresholds: tuple[float, float] | None = None
) -> Path:
    """Write the mean train and test maps as a standard container (keys of round a).

    With `thresholds` (full, rect), the maps are written already binarised by `regime_binarise`.
    """
    train_names, train_mean = mean_of_specs(specs, test=False)
    test_names, test_mean = mean_of_specs(specs, test=True)
    if thresholds is not None:
        train_mean = regime_binarise(train_mean, thresholds)
        test_mean = regime_binarise(test_mean, thresholds)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        names_train=np.array(train_names),
        names_test=np.array(test_names),
        train_finetuned_oof=train_mean.astype(np.float16),
        test_finetuned=test_mean.astype(np.float16),
        metrics_json=np.array(json.dumps({"combined": [f"{p}:{k}" for p, k in specs]})),
    )
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="x-loc-unet evaluation helpers.")
    commands = parser.add_subparsers(dest="command", required=True)
    splitting = commands.add_parser("split")
    splitting.add_argument("--container", type=Path, required=True)
    splitting.add_argument("--out-dir", type=Path, required=True)
    splitting.add_argument("--prefix", required=True)
    for name in ("evaluate", "combine"):
        sub = commands.add_parser(name)
        sub.add_argument("--method", choices=KNOWN_METHODS, required=True)
        sub.add_argument("--maps", nargs="+", required=True)
        if name == "evaluate":
            sub.add_argument("--reference", type=Path)
        else:
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--regime-thresholds", type=float, nargs=2, metavar=("FULL", "RECT"))
    args = parser.parse_args()
    if args.command == "split":
        for path in split_container(args.container, args.out_dir, args.prefix):
            print(path)
        return
    specs = [parse_spec(spec) for spec in args.maps]
    if args.command == "evaluate":
        for key, row in evaluate_specs(specs, args.method, args.reference).items():
            print(key, json.dumps({k: round(v, 4) for k, v in row.items()}))
        return
    thresholds = (
        None
        if args.regime_thresholds is None
        else (float(args.regime_thresholds[0]), float(args.regime_thresholds[1]))
    )
    print(combine_specs(specs, args.out, thresholds))


if __name__ == "__main__":
    main()
