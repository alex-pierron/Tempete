"""Decision theory for the official metric: per image, pick the binary mask that maximises the EXPECTED
mIoU under the head's probability map, instead of thresholding at 0.5.

Uncertainty model (spatially coherent): the true zone is a level set of the calibrated map,
G_u = {p_T >= u}, u ~ U(0, 1), with p_T = sigmoid(logit(p) / T). Candidates M_t = {p >= t}.
Choose t* = argmax_t mean_u mIoU(M_t, G_u). T chosen in nested CV (5 folds). Train OOF only.

Usage: python emiou_eval.py <method> [npz key file: d_syn8 | d_syn3]
"""

from __future__ import annotations

import sys

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import HEAD_CV, gt128, miou, train_meta  # noqa: E402

CANDIDATE_T = np.round(np.arange(0.05, 0.96, 0.05), 2)
LEVELS_U = np.round(np.arange(0.025, 1.0, 0.05), 3)
TEMPERATURES = (0.5, 1.0, 2.0, 4.0)
FOLDS = 5
SEED = 2026
SMALL = 0.05


def miou_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Two-class mIoU of boolean arrays (empty union counts 1)."""
    out = 0.0
    for pa, pb in ((a, b), (~a, ~b)):
        union = np.count_nonzero(pa | pb)
        out += 1.0 if union == 0 else np.count_nonzero(pa & pb) / union
    return out / 2


def choose(p: np.ndarray, temp: float) -> float:
    """Threshold maximising expected mIoU under the level-set model."""
    lg = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    pt = 1 / (1 + np.exp(-lg / temp))
    cands = [p >= t for t in CANDIDATE_T]
    truths = [pt >= u for u in LEVELS_U]
    scores = [np.mean([miou_fast(c, g) for g in truths]) for c in cands]
    return float(CANDIDATE_T[int(np.argmax(scores))])


def main() -> None:
    """Print base vs nested expected-mIoU decision, by regime and zone size."""
    method = sys.argv[1]
    key_file = sys.argv[2] if len(sys.argv) > 2 else "d_syn8"
    meta = train_meta(method)
    maps = np.load(HEAD_CV / f"{key_file}.npz")[method].astype(np.float32)
    gts = gt128()
    g = [gts[r["image_name"]] > 0 for r in meta]
    base = np.array([miou_fast(p >= 0.5, gi) for p, gi in zip(maps, g, strict=True)])
    print(f"{method} {key_file} base {base.mean():.4f} (official check {np.mean([miou((p >= 0.5).astype(np.uint8), gi.astype(np.uint8)) for p, gi in zip(maps, g, strict=True)]):.4f})")
    table = np.zeros((len(meta), len(TEMPERATURES)))
    for i, (p, gi) in enumerate(zip(maps, g, strict=True)):
        for j, temp in enumerate(TEMPERATURES):
            table[i, j] = miou_fast(p >= choose(p, temp), gi)
    rng = np.random.default_rng(SEED)
    folds = rng.integers(0, FOLDS, len(meta))
    nested = np.zeros(len(meta))
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        j = int(np.argmax(table[tr].mean(0)))
        nested[te] = table[te, j]
        print(f"  fold {f}: T={TEMPERATURES[j]}")
    print("  per T:", {t: round(float(table[:, j].mean()), 4) for j, t in enumerate(TEMPERATURES)})
    d = nested - base
    full = np.array([bool(r["full_watermark"]) for r in meta])
    small = np.array([min(gi.mean(), 1 - gi.mean()) < SMALL for gi in g])
    for name, m in (("all", np.ones(len(meta), bool)), ("full", full), ("rect", ~full), ("small", small), ("full small", full & small), ("rect small", ~full & small)):
        b = [d[m][rng.integers(0, m.sum(), m.sum())].mean() for _ in range(2000)]
        print(f"{name:11s} n={m.sum():3d} base {base[m].mean():.4f} emiou {nested[m].mean():.4f} delta {d[m].mean():+.4f} "
              f"[{np.percentile(b, 2.5):+.4f};{np.percentile(b, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
