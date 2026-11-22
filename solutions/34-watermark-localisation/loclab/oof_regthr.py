"""Paired OOF of two TrustMark head ensembles, both with the regime thresholds of ll_v12 (full 0.6 / rect 0.4).

The regime of each image is predicted from its own ensemble map (share of cells >= 0.5 above one half), as in
the test assembly. Prints the candidate, the reference (the ll_v12 ensemble by default: d_syn8 + i_syn8) and
the paired bootstrap CI by regime. Decision for ll_v11: propose only if the paired delta is > 0.

Usage: python oof_regthr.py "<candidate cv stems>" ["<reference cv stems>"] [t_full,t_rect]
   e.g. python oof_regthr.py "d_syn8 i_syn8 i_syn8c i_syn8d"
"""

from __future__ import annotations

import sys

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import HEAD_CV, gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402

METHOD = "trustmark"
REFERENCE = "d_syn8 i_syn8"
THRESHOLDS = "0.6,0.4"
DECISION = 0.5
SEED = 2026
BOOTSTRAP = 4000


def scores(stems: list[str], t_full: float, t_rect: float, g: list[np.ndarray]) -> np.ndarray:
    """Per-image mIoU of the mean map of `stems` under the regime thresholds."""
    p = np.mean([np.load(HEAD_CV / f"{s}.npz")[METHOD].astype(np.float32) for s in stems], axis=0)
    full = (p >= DECISION).mean(axis=(1, 2)) >= DECISION
    return np.array([miou_fast(m >= (t_full if f else t_rect), x) for m, f, x in zip(p, full, g, strict=True)])


def main() -> None:
    """Print the paired comparison."""
    candidate = sys.argv[1].split()
    reference = (sys.argv[2] if len(sys.argv) > 2 else REFERENCE).split()
    t_full, t_rect = (float(v) for v in (sys.argv[3] if len(sys.argv) > 3 else THRESHOLDS).split(","))
    meta = train_meta(METHOD)
    gts = gt128()
    g = [gts[r["image_name"]] > 0 for r in meta]
    full = np.array([bool(r["full_watermark"]) for r in meta])
    s_c, s_r = scores(candidate, t_full, t_rect, g), scores(reference, t_full, t_rect, g)
    d = s_c - s_r
    rng = np.random.default_rng(SEED)
    print(f"{METHOD} {candidate} {s_c.mean():.4f} vs {reference} {s_r.mean():.4f} (thresholds {t_full}/{t_rect})")
    for key, m in {"all": np.ones(len(d), bool), "full": full, "rect": ~full}.items():
        bt = [d[m][rng.integers(0, m.sum(), m.sum())].mean() for _ in range(BOOTSTRAP)]
        print(f"  {key:5s} delta {d[m].mean():+.4f} [{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
