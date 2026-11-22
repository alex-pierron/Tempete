"""Size prior: threshold chosen per (predicted regime, predicted zone size) bin, out of fold.

Predicted size = minority share of the cells >= 0.5 in the image's own map (full: holes, rect: marked share).
For each bin, the threshold is chosen leave-one-out on a grid; delta vs the deployed rule (TM: ll_v12 regime
thresholds 0.6 / 0.4 on d_syn8 + i_syn8; PS: 0.5 on the 3-head mean). Also prints the trivial mask per bin.

Usage: python size_prior.py <method>
"""

from __future__ import annotations

import sys

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import HEAD_CV, ROOT, gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402

HEADS = {"pixelseal": ("d_syn3", "d_syn8", "i_syn3"), "trustmark": ("d_syn8", "i_syn8")}
DEPLOYED = {"pixelseal": (0.5, 0.5), "trustmark": (0.6, 0.4), "stegastamp": (0.5, 0.5)}
SS_SPLIT7 = str(ROOT / "team_final" / "ss_split7_u1x5_zoom2.npz")
GRID = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
SIZE_EDGES = tuple(float(v) for v in __import__("os").environ.get("SIZE_EDGES", "0.03,0.08").split(","))
DECISION = 0.5
SEED = 2026
BOOTSTRAP = 4000


def train_map(method: str, names: list[str]) -> np.ndarray:
    """OOF map of the deployed rule's input: head mean (PS / TM) or the split7 SS map (already regime-shifted)."""
    if method == "stegastamp":
        z = np.load(SS_SPLIT7)
        pos = {str(n): i for i, n in enumerate(z["names_train"])}
        return z["train_finetuned_oof"].astype(np.float32)[[pos[n] for n in names]]
    return np.mean([np.load(HEAD_CV / f"{h}.npz")[method].astype(np.float32) for h in HEADS[method]], axis=0)


def main() -> None:
    """Print the LOO size-prior delta and per-bin details."""
    method = sys.argv[1]
    meta = train_meta(method)
    gts = gt128()
    g = [gts[r["image_name"]] > 0 for r in meta]
    p = train_map(method, [str(r["image_name"]) for r in meta])
    share = (p >= DECISION).mean(axis=(1, 2))
    full = share >= DECISION
    minority = np.where(full, 1 - share, share)
    size_bin = np.digitize(minority, SIZE_EDGES)
    groups = full.astype(int) * len(SIZE_EDGES + (0,)) + size_bin
    t_dep = np.where(full, DEPLOYED[method][0], DEPLOYED[method][1])
    base = np.array([miou_fast(m >= t, x) for m, t, x in zip(p, t_dep, g, strict=True)])
    table = {t: np.array([miou_fast(m >= t, x) for m, x in zip(p, g, strict=True)]) for t in GRID}
    trivial = np.array([miou_fast(np.full_like(x, f), x) for x, f in zip(g, full, strict=True)])
    loo = np.empty(len(g))
    for i in range(len(g)):
        same = (groups == groups[i]) & (np.arange(len(g)) != i)
        best = max(GRID, key=lambda t: table[t][same].sum()) if same.any() else t_dep[i]
        loo[i] = table[best][i] if same.any() else base[i]
    d = loo - base
    rng = np.random.default_rng(SEED)
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
    print(f"{method}: deployed {base.mean():.4f} | size-prior LOO {loo.mean():.4f} delta {d.mean():+.4f} "
          f"[{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}]")
    for grp in np.unique(groups):
        m = groups == grp
        best = max(GRID, key=lambda t: table[t][m].sum())
        print(f"  {'full' if grp >= len(SIZE_EDGES) + 1 else 'rect'} size bin {grp % (len(SIZE_EDGES) + 1)} n={int(m.sum()):3d}"
              f" deployed {base[m].mean():.3f} best t={best} {table[best][m].mean():.3f} trivial {trivial[m].mean():.3f}")


if __name__ == "__main__":
    main()
