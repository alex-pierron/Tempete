"""Uniform ensembles of b2 heads (OOF from team_b2/cv/<cv>.npz, test from team_b2/cv/<fit>.npz) vs the deployed
head of one method; paired bootstrap CI by regime / zone size; writes the ensemble test map for assemble.py.

Usage: python ens_heads.py <method> <out.npz> <cv1>:<fit1> [<cv2>:<fit2> ...]
   e.g. python ens_heads.py pixelseal out.npz d_syn3:fit_d_syn3_s0_test d_syn8:fit_n_syn8_s1_test i_syn3:fit_i_syn3_s0_test
"""

from __future__ import annotations

import sys

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import HEAD_CV, gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402
from heads_eval import HEAD_OOF, HEAD_TEST, SMALL  # noqa: E402

SEED = 2026


def main() -> None:
    """Print the paired comparison and write the test map."""
    method, out = sys.argv[1], sys.argv[2]
    pairs = [s.split(":") for s in sys.argv[3:]]
    meta = train_meta(method)
    gts = gt128()
    g = [gts[r["image_name"]] > 0 for r in meta]
    full = np.array([bool(r["full_watermark"]) for r in meta])
    small = np.array([min(gi.mean(), 1 - gi.mean()) < SMALL for gi in g])
    oof = np.mean([np.load(HEAD_CV / f"{cv}.npz")[method].astype(np.float32) for cv, _ in pairs], axis=0)
    dep = np.load(HEAD_CV / f"{HEAD_OOF[method]}.npz")[method].astype(np.float32)
    s_new = np.array([miou_fast(m >= 0.5, gi) for m, gi in zip(oof, g, strict=True)])
    s_dep = np.array([miou_fast(m >= 0.5, gi) for m, gi in zip(dep, g, strict=True)])
    d = s_new - s_dep
    rng = np.random.default_rng(SEED)
    print(f"{method} ensemble {[c for c, _ in pairs]} vs deployed {HEAD_OOF[method]} ({s_dep.mean():.4f})")
    for gk, gm in {"all": np.ones(len(g), bool), "full": full, "rect": ~full, "full_small": full & small, "rect_small": ~full & small}.items():
        bt = [d[gm][rng.integers(0, gm.sum(), gm.sum())].mean() for _ in range(4000)]
        print(f"  {gk:10s} {s_new[gm].mean():.4f} delta {d[gm].mean():+.4f} [{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}]")
    ref = np.load(HEAD_CV / f"{HEAD_TEST[method]}.npz")
    names = [str(n) for n in ref[f"names_{method}"]]
    maps = []
    for _, fit in pairs:
        z = np.load(HEAD_CV / f"{fit}.npz")
        pos = {str(n): i for i, n in enumerate(z[f"names_{method}"])}
        maps.append(z[method].astype(np.float32)[[pos[n] for n in names]])
    test = np.mean(maps, axis=0)
    changed = int(sum(not np.array_equal(a >= 0.5, b >= 0.5) for a, b in zip(test, ref[method].astype(np.float32), strict=True)))
    np.savez(out, **{method: test.astype(np.float16), f"names_{method}": np.array(names)})
    print(f"  wrote {out}; test masks differing from the deployed head at 0.5: {changed}")


if __name__ == "__main__":
    main()
