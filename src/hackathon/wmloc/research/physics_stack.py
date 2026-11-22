"""Stack physically motivated pixel statistics on top of the base head's out-of-fold probabilities.

Evidence used (each map on the 128 grid, z-scored within the image so that it ranks the cells of
ONE image, which is what the per-image mIoU rewards):

* ``lc_ratio``: log ratio of luma to chroma high-pass energy. Pixel Seal predicts its residual on
  the Y channel only (videoseal card ``pixelseal``: ``unet_base_yuv_quant`` 1 -> 1, ``jnd_1_1``,
  additive blending), so the residual is achromatic and raises luma energy only.
* ``hp_energy``: log local high-pass energy (the JND-scaled residual adds energy; inpainted or
  outpainted content is smoother).
* ``per2_rel``: period-2 phase energy relative to the high-pass energy (upsampling traces of the
  generative editor or of the x2 residual upscaling).

Model: per method, a logistic regression on [logit(p_base), features, logit x features], fitted on
4 folds of images and applied to the 5th (fold = image index mod 5). Score: official per-image mIoU
of ``p > 0.5`` for the base and for the stack, on exactly the same images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression

GRID_SIZE = 128
CELL = 4
EPS = 1e-3
N_FOLDS = 5
CELLS_PER_IMAGE = 2048
SEED = 0
PROB_CLIP = 1e-4
THRESHOLD = 0.5
FEATURES = ("lc_ratio", "hp_energy", "per2_rel", "chroma_energy")


def to_grid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average 4 x 4 cells to the 128 grid."""
    return x.reshape(GRID_SIZE, CELL, GRID_SIZE, CELL).mean(axis=(1, 3))


def zscore(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardise within the image."""
    return (x - x.mean()) / (x.std() + 1e-9)


def physics_maps(img: NDArray[np.uint8], window: int) -> dict[str, NDArray[np.float64]]:
    """The four evidence maps of one image, z-scored, on the 128 grid."""
    r, g, b = (img[..., c].astype(np.float64) for c in range(3))
    lum = (r + g + b) / 3
    hp_l = lum - uniform_filter(lum, 3, mode="reflect")
    hp_c1 = (r - g) - uniform_filter(r - g, 3, mode="reflect")
    hp_c2 = (b - g) - uniform_filter(b - g, 3, mode="reflect")
    e_l = uniform_filter(hp_l**2, window) + EPS
    e_c = uniform_filter(hp_c1**2 + hp_c2**2, window) + EPS
    side = lum.shape[0]
    rows = np.arange(side)[:, None] % 2
    cols = np.arange(side)[None, :] % 2
    phase_means = []
    for a in range(2):
        for c in range(2):
            ind = ((rows == a) & (cols == c)).astype(np.float64)
            phase_means.append(uniform_filter(hp_l * ind, window) / (uniform_filter(ind, window) + 1e-9))
    per2 = np.stack(phase_means).var(axis=0) + 1e-6
    return {
        "lc_ratio": zscore(to_grid(np.log(e_l) - np.log(e_c))),
        "hp_energy": zscore(to_grid(np.log(e_l))),
        "per2_rel": zscore(to_grid(np.log(per2) - np.log(e_l))),
        "chroma_energy": zscore(to_grid(np.log(e_c))),
    }


def miou(pred: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    """Official per-image mIoU (classes 0 and 1, empty union counts 1)."""
    total = 0.0
    for cls in (False, True):
        p, t = pred == cls, truth == cls
        union = int((p | t).sum())
        total += 1.0 if union == 0 else int((p & t).sum()) / union
    return total / 2


def design(logit: NDArray[np.float64], feats: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stacking design matrix: base logit, features, and their products with the logit."""
    return np.concatenate([logit[:, None], feats, feats * logit[:, None]], axis=1)


def main() -> None:
    """Compute train features, stack out of fold, print base vs stack mIoU per method and regime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True, help="npz with <method> (N, 128, 128) probabilities")
    parser.add_argument("--methods", nargs="+", default=["pixelseal", "trustmark"])
    parser.add_argument("--window", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m.astype(bool) for n, m in zip(packed["names"], packed["masks"], strict=True)}
    oof = np.load(args.oof)
    rng = np.random.default_rng(SEED)
    report: dict[str, dict[str, float]] = {}
    for method in args.methods:
        items = [m for m in meta if m["method"] == method]
        probs = oof[method].astype(np.float64)
        names = [str(m["image_name"]) for m in items]
        is_full = np.array([bool(m["full_watermark"]) for m in items])
        truth = np.stack([masks[n] for n in names])
        feats = np.stack([
            np.stack([v for k, v in physics_maps(
                np.asarray(Image.open(args.data / "images" / f"{n}.png").convert("RGB")), args.window
            ).items() if k in FEATURES], axis=-1)
            for n in names
        ])  # N x 128 x 128 x F
        logits = np.log(np.clip(probs, PROB_CLIP, 1 - PROB_CLIP) / (1 - np.clip(probs, PROB_CLIP, 1 - PROB_CLIP)))
        folds = np.arange(len(names)) % N_FOLDS
        stacked = np.zeros_like(probs)
        for regime in (True, False):
            for fold in range(N_FOLDS):
                train_idx = np.where((folds != fold) & (is_full == regime))[0]
                test_idx = np.where((folds == fold) & (is_full == regime))[0]
                xs, ys = [], []
                for i in train_idx:
                    cells = rng.choice(GRID_SIZE * GRID_SIZE, CELLS_PER_IMAGE, replace=False)
                    xs.append(design(logits[i].reshape(-1)[cells], feats[i].reshape(-1, feats.shape[-1])[cells]))
                    ys.append(truth[i].reshape(-1)[cells])
                clf = LogisticRegression(C=1.0, max_iter=500).fit(np.concatenate(xs), np.concatenate(ys))
                for i in test_idx:
                    x = design(logits[i].reshape(-1), feats[i].reshape(-1, feats.shape[-1]))
                    stacked[i] = clf.predict_proba(x)[:, 1].reshape(GRID_SIZE, GRID_SIZE)
        base_scores = np.array([miou(probs[i] > THRESHOLD, truth[i]) for i in range(len(names))])
        stack_scores = np.array([miou(stacked[i] > THRESHOLD, truth[i]) for i in range(len(names))])
        diff = stack_scores - base_scores
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        report[method] = {
            "base_all": float(base_scores.mean()), "stack_all": float(stack_scores.mean()),
            "base_full": float(base_scores[is_full].mean()), "stack_full": float(stack_scores[is_full].mean()),
            "base_rect": float(base_scores[~is_full].mean()), "stack_rect": float(stack_scores[~is_full].mean()),
            "delta_ci95_low": float(np.percentile(boot, 2.5)), "delta_ci95_high": float(np.percentile(boot, 97.5)),
        }
        print(method, json.dumps({k: round(v, 4) for k, v in report[method].items()}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
