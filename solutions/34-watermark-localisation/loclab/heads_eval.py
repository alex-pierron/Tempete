"""Evaluate re-trained b2 heads (e.g. x23 iso synthetic) against the deployed heads, and write test maps.

Container npz keys are "<stem>/<array>" (LOC-GPU packing): cv stem with arrays pixelseal / trustmark
(170 x 128 x 128, train metadata order, OOF) and a fit stem "<fit>_test" with arrays <method> and
names_<method>. Deployed heads: d_syn3 (PS) and d_syn8 (TM) OOF; test fit_d_syn3_s0_test / fit_n_syn8_s1_test.
For each method: OOF at 0.5 by regime / zone size for new, deployed and their mean, paired delta vs deployed
with a bootstrap CI; writes <out_dir>/<tag>_<method>_{new,mean}.npz for assemble.py --replace.

Usage: python heads_eval.py <container.npz> <cv_stem> <fit_stem> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import HEAD_CV, gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402

HEAD_OOF = {"pixelseal": "d_syn3", "trustmark": "d_syn8"}
HEAD_TEST = {"pixelseal": "fit_d_syn3_s0_test", "trustmark": "fit_n_syn8_s1_test"}
SMALL = 0.05
SEED = 2026


def main() -> None:
    """Print the comparison and write the maps."""
    box = np.load(sys.argv[1], allow_pickle=True)
    cv_stem, fit_stem, out_dir = sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    gts = gt128()
    rng = np.random.default_rng(SEED)
    for method in ("pixelseal", "trustmark"):
        key = f"{cv_stem}/{method}"
        if key not in box.files:
            print(f"{method}: no {key} in container")
            continue
        meta = train_meta(method)
        g = [gts[r["image_name"]] > 0 for r in meta]
        full = np.array([bool(r["full_watermark"]) for r in meta])
        small = np.array([min(gi.mean(), 1 - gi.mean()) < SMALL for gi in g])
        new = box[key].astype(np.float32)
        old = np.load(HEAD_CV / f"{HEAD_OOF[method]}.npz")[method].astype(np.float32)
        variants = {"deployed": old, "new": new, "mean": (old + new) / 2}
        scores = {k: np.array([miou_fast(m >= 0.5, gi) for m, gi in zip(v, g, strict=True)]) for k, v in variants.items()}
        groups = {"all": np.ones(len(g), bool), "full": full, "rect": ~full, "full_small": full & small, "rect_small": ~full & small}
        print(f"== {method}")
        for k in ("new", "mean"):
            d = scores[k] - scores["deployed"]
            parts = []
            for gk, gm in groups.items():
                bt = [d[gm][rng.integers(0, gm.sum(), gm.sum())].mean() for _ in range(2000)]
                parts.append(f"{gk} {scores[k][gm].mean():.4f} ({d[gm].mean():+.4f} [{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}])")
            print(f"  {k:5s} vs deployed {scores['deployed'].mean():.4f}: " + " | ".join(parts))
        tkey, nkey = f"{fit_stem}/{method}", f"{fit_stem}/names_{method}"
        if tkey in box.files:
            names = [str(n) for n in box[nkey]]
            zt = np.load(HEAD_CV / f"{HEAD_TEST[method]}.npz")
            pos = {str(n): i for i, n in enumerate(zt[f"names_{method}"])}
            old_t = zt[method].astype(np.float32)[[pos[n] for n in names]]
            new_t = box[tkey].astype(np.float32)
            for k, v in (("new", new_t), ("mean", (old_t + new_t) / 2)):
                path = out_dir / f"{cv_stem}_{method}_{k}.npz"
                np.savez(path, **{method: v.astype(np.float16), f"names_{method}": np.array(names)})
                print(f"  wrote {path}")


if __name__ == "__main__":
    main()
