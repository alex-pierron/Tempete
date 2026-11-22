"""Assemble a "best source per method" localisation candidate from a complete base candidate.

Steps, in this order (the order is the precedence, last writer wins):
  1. start from a complete base candidate (`--base`, 1020 masks);
  2. for each `--replace method=a.npz[+b.npz]:key[:threshold]`, overwrite the masks of that
     method's test images by the (mean) probability map thresholded (default 0.5);
  3. for each `--overlay path[:min_confidence]`, overwrite the masks it lists whose confidence is
     at least `min_confidence` (default 0.85, the rule used by every previous candidate).
Prints the number of masks changed per step and per method, as JSON.

Usage:
    PYTHONPATH=src HACKATHON_DATA_DIR=... python solutions/34-watermark-localisation/assemble.py \
        --base base.pt --replace trustmark=u1_trustmark.npz:test_finetuned \
        --overlay c_rect_overlay.pt --overlay rect_ps_occlusion_overlay.pt --out new.pt
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.research.data import load_records

GRID_SIZE = 128
DECISION_THRESHOLD = 0.5
OVERLAY_CONFIDENCE = 0.85
NAMES_TEST_KEY = "names_test"
METHOD_NAMES_PREFIX = "names_"


class AssemblyError(ValueError):
    """A map, an overlay or the base is inconsistent with the test release."""


def load_test_map(path: Path, key: str) -> tuple[list[str], NDArray[np.float32]]:
    """Test probabilities of one container: `names_test` or `names_<key>` gives the order."""
    archive = np.load(path)
    if key not in archive.files:
        raise AssemblyError(f"{path} has no key {key!r} (keys: {archive.files})")
    for names_key in (NAMES_TEST_KEY, METHOD_NAMES_PREFIX + key):
        if names_key in archive.files:
            return [str(n) for n in archive[names_key]], archive[key].astype(np.float32)
    raise AssemblyError(f"{path} has no name list for {key!r}")


def mean_map(paths: list[Path], key: str) -> tuple[list[str], NDArray[np.float32]]:
    """Mean of several test maps, aligned on the names of the first one."""
    names, total = load_test_map(paths[0], key)
    for path in paths[1:]:
        other_names, other = load_test_map(path, key)
        order = {n: i for i, n in enumerate(other_names)}
        if set(order) != set(names):
            raise AssemblyError(f"{path} does not cover the same images as {paths[0]}")
        total = total + other[[order[n] for n in names]]
    return names, (total / np.float32(len(paths))).astype(np.float32)


def parse_replacement(text: str) -> tuple[str, list[Path], str, float]:
    """`method=a.npz[+b.npz]:key[:threshold]` -> (method, paths, key, threshold)."""
    method, _, rest = text.partition("=")
    parts = rest.split(":")
    if not method or len(parts) not in (2, 3):
        raise AssemblyError(f"expected method=path.npz:key[:threshold], got {text!r}")
    threshold = float(parts[2]) if len(parts) == 3 else DECISION_THRESHOLD
    return method, [Path(p) for p in parts[0].split("+")], parts[1], threshold


def parse_overlay(text: str) -> tuple[Path, float]:
    """`path[:min_confidence]` -> (path, min_confidence)."""
    path, _, confidence = text.partition(":")
    return Path(path), float(confidence) if confidence else OVERLAY_CONFIDENCE


def assemble(
    base: Path,
    replacements: list[tuple[str, list[Path], str, float]],
    overlays: list[tuple[Path, float]],
    out: Path,
) -> dict[str, dict[str, int]]:
    """Write the candidate and return the changed-mask counts per step and method.

    Raises:
        AssemblyError: a map or overlay name is not a test image (of the expected method).
    """
    loaded = torch.load(base, weights_only=True)
    names = [str(n) for n in loaded["image_names"]]
    masks = loaded["masks"].to(torch.uint8).clone()
    if masks.shape != (len(names), GRID_SIZE, GRID_SIZE):
        raise AssemblyError(f"unexpected base shape {tuple(masks.shape)}")
    row = {name: index for index, name in enumerate(names)}
    method_of = {r.name: r.method for r in load_records("test")}
    report: dict[str, dict[str, int]] = {}
    for method, paths, key, threshold in replacements:
        map_names, probability = mean_map(paths, key)
        changed: collections.Counter[str] = collections.Counter()
        for index, name in enumerate(map_names):
            if method_of.get(name) != method or name not in row:
                raise AssemblyError(f"{name} of {paths[0]} is not a test image of {method}")
            new = torch.from_numpy((probability[index] >= threshold).astype(np.uint8))
            changed[method] += int(not torch.equal(new, masks[row[name]]))
            masks[row[name]] = new
        report[f"replace:{method}:{'+'.join(p.stem for p in paths)}"] = dict(changed)
    for path, min_confidence in overlays:
        boxes = torch.load(path, weights_only=True)
        changed = collections.Counter()
        for name, mask, confidence in zip(
            boxes["image_names"], boxes["masks"], boxes["confidence"], strict=True
        ):
            if float(confidence) < min_confidence:
                continue
            if str(name) not in row:
                raise AssemblyError(f"overlay image {name} of {path} is not a test image")
            new = mask.to(torch.uint8)
            changed[method_of[str(name)]] += int(not torch.equal(new, masks[row[str(name)]]))
            masks[row[str(name)]] = new
        report[f"overlay:{path.stem}"] = dict(changed)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": masks}, out)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replace", action="append", default=[])
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(
        args.base,
        [parse_replacement(t) for t in args.replace],
        [parse_overlay(t) for t in args.overlay],
        args.out,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
