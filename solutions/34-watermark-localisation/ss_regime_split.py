"""StegaStamp map with one seed mean per PREDICTED regime, scored against the in-place ensemble.

Why: the zoom + strength chain (x6) is a model of the FULL-regime pattern (magnified, weaker);
adding it to the seed mean gains in full (+0.024) and loses in rect (+0.0085 against +0.015
without it). The regime is predicted from the rect members' mean (share of cells >= 0.5 below
0.5 -> rect, 99.4 % right on u1), then: predicted rect -> mean of the rect members cut at 0.2
(the public rule), predicted full -> mean of the full members cut at 0.5.

Writes a map ALREADY shifted (rect images moved by 0.5 - 0.2) so that assemble.py at 0.5 applies
the rule, with the train out-of-fold map under the same keys (seeds_eval.py format).

Usage: ss_regime_split.py --rect <spec...> --full <spec...> --reference <spec...> --out map.npz
    spec = file.npz[:suffix] -> keys train_finetuned_oof<_suffix> / test_finetuned<_suffix>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.metric import per_image_miou
from hackathon.wmloc.research.data import load_records, load_truth

METHOD = "stegastamp"
DECISION_THRESHOLD = 0.5
# LG_REGIME_CUT moves the share of cells >= 0.5 under which an image is predicted rect (probes).
REGIME_CUT = float(os.environ.get("LG_REGIME_CUT", "0.5"))
RECT_THRESHOLD = 0.2
SMALL_ZONE_FRACTION = 0.05
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 2026


def parse_spec(text: str) -> tuple[Path, str]:
    """`file.npz[:suffix]` -> (path, key suffix)."""
    path, _, suffix = text.partition(":")
    return Path(path), f"_{suffix}" if suffix else ""


def mean_map(specs: list[str], split: str, names: list[str]) -> NDArray[np.float32]:
    """Mean map of the specs on `split` ("train" out of fold, or "test"), ordered on `names`."""
    maps = []
    for text in specs:
        path, suffix = parse_spec(text)
        archive = np.load(path)
        key = ("train_finetuned_oof" if split == "train" else "test_finetuned") + suffix
        order = {str(n): i for i, n in enumerate(archive[f"names_{split}"])}
        maps.append(archive[key][[order[n] for n in names]].astype(np.float32))
    return np.mean(maps, axis=0).astype(np.float32)


def predicted_rect(maps: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Regime rule of the public candidate: under half of the cells at >= 0.5 -> rect."""
    return np.asarray((maps >= DECISION_THRESHOLD).mean(axis=(1, 2)) < REGIME_CUT)


def split_map(rect_maps: NDArray[np.float32], full_maps: NDArray[np.float32]) -> NDArray[np.float32]:
    """Shifted map: predicted rect -> rect members + (0.5 - 0.2), predicted full -> full members."""
    rect = predicted_rect(rect_maps)
    shifted = rect_maps + (DECISION_THRESHOLD - RECT_THRESHOLD)
    return np.where(rect[:, None, None], shifted, full_maps).astype(np.float32)


def official(binary: NDArray[np.bool_], truth: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Official per-image mIoU."""
    prediction = torch.from_numpy(binary.astype(np.uint8))
    return per_image_miou(prediction, torch.from_numpy(truth)).numpy().astype(np.float64)


def interval(delta: NDArray[np.float64]) -> str:
    """Mean and 95 % bootstrap interval of a paired delta."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = delta[rng.integers(0, len(delta), (BOOTSTRAP_ROUNDS, len(delta)))].mean(axis=1)
    return f"{delta.mean():+.4f} [{np.quantile(means, 0.025):+.4f};{np.quantile(means, 0.975):+.4f}]"


def main() -> None:
    """Score the split map against the reference + rule, write the shifted map."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rect", nargs="+", required=True)
    parser.add_argument("--full", nargs="+", required=True)
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in load_records("train") if r.method == METHOD]
    names = [r.name for r in records]
    full = np.array([bool(r.full_watermark) for r in records])
    truth = load_truth(names)
    marked = truth.astype(bool).mean(axis=(1, 2))
    small = np.where(full, 1 - marked, marked) < SMALL_ZONE_FRACTION
    train_rect = mean_map(args.rect, "train", names)
    train_split = split_map(train_rect, mean_map(args.full, "train", names))
    reference = mean_map(args.reference, "train", names)
    reference_masks = split_map(reference, reference) >= DECISION_THRESHOLD
    base = official(reference_masks, truth)
    scores = official(train_split >= DECISION_THRESHOLD, truth)
    delta = scores - base
    regime_right = float((predicted_rect(train_rect) == ~full).mean())
    print(f"regime predicted right on {regime_right:.3f} of the train images")
    print(f"reference + rule all {base.mean():.4f} | split all {scores.mean():.4f} full "
          f"{scores[full].mean():.4f} rect {scores[~full].mean():.4f}")
    print(f"delta {interval(delta)} | full {interval(delta[full])} | rect {interval(delta[~full])}")
    groups = {"full_small": full & small, "full_large": full & ~small,
              "rect_small": ~full & small, "rect_large": ~full & ~small}
    print(" | ".join(f"{k} {interval(delta[v])}" for k, v in groups.items()))
    first = np.load(parse_spec(args.rect[0])[0])
    names_test = [str(n) for n in first["names_test"]]
    test_rect = mean_map(args.rect, "test", names_test)
    test_split = split_map(test_rect, mean_map(args.full, "test", names_test))
    folds = first["folds"][[{str(n): i for i, n in enumerate(first["names_train"])}[n] for n in names]]
    np.savez_compressed(args.out, names_test=np.array(names_test), names_train=np.array(names),
                        folds=folds, test_finetuned=test_split, train_finetuned_oof=train_split)
    print({"out": str(args.out), "test_predicted_rect": int(predicted_rect(test_rect).sum())})


if __name__ == "__main__":
    main()
