"""X-LOC transductive bet: pseudo-labels of the TEST images (PS / TM) from the ll_v20x composition, high confidence only.

Composition (as ll_v20x): head map on head-predicted full images, mean(head, xb1) on predicted rect images, SAM samGx
masks on its apply images (0.99 / 0.01), c_rect overlay masks last. Label = 1 if p > HIGH, 0 if p < LOW, -1 otherwise,
and -1 within ERODE cells of the p >= 0.5 boundary. The same rule on the train OOF composition gives the label precision.

Usage: python pseudo_labels.py <out_dir>
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # loclab helper modules
from common import gt128, train_meta  # noqa: E402
from integrate_map import HEAD_TEST, load_map  # noqa: E402
from size_prior import train_map  # noqa: E402

HIGH, LOW, ERODE = 0.9, 0.1, 2
XB1 = __import__("os").environ.get(
    "PSEUDO_XB", (__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loc_gpu_in/xlocb_xb1/xb1_{s}_{k}.npz"))
"""Pattern of the e2e map used on rect images ({s} = ps / tm, {k} = oof / test); round 2 points it at xpl1."""
SAM = (__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x_loc_c/samGx_{m}_{k}.npz")
CRECT = (__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/overlays/c_rect_overlay_tmshrink2.pt")
SHORT = {"pixelseal": "ps", "trustmark": "tm"}


def compose(head: np.ndarray, xb1: np.ndarray, sam: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """ll_v20x probability composition (before the size-prior shift)."""
    full = (head >= 0.5).mean(axis=(1, 2)) >= 0.5
    p = np.where(full[:, None, None], head, 0.5 * (head + xb1))
    return np.where(apply[:, None, None], np.where(sam > 0.5, 0.99, 0.01), p).astype(np.float32)


def labels_of(p: np.ndarray) -> np.ndarray:
    """1 / 0 / -1 labels with the boundary band ignored."""
    lab = np.full(p.shape, -1, dtype=np.int8)
    lab[p > HIGH] = 1
    lab[p < LOW] = 0
    for i in range(len(p)):
        m = p[i] >= 0.5
        band = ndimage.binary_dilation(m, iterations=ERODE) & ~ndimage.binary_erosion(m, iterations=ERODE)
        lab[i][band] = -1
    return lab


def main() -> None:
    """Write pseudo_<method>_test.npz and print the train-equivalent precision and coverage."""
    out = sys.argv[1]
    crect = torch.load(CRECT, weights_only=True)
    crect_masks = {n: m.numpy() > 0 for n, m, c in zip(crect["image_names"], crect["masks"], crect["confidence"]) if float(c) >= 0.85}
    gts = gt128()
    for method, s in SHORT.items():
        names = [str(r["image_name"]) for r in train_meta(method)]
        head = train_map(method, names)
        xb1 = load_map(XB1.format(s=s, k="oof"), method, names)[0]
        sam, apply = load_map(SAM.format(m=method, k="train"), method, names)
        lab = labels_of(compose(head, xb1, sam, apply))
        y = np.stack([gts[v] > 0 for v in names])
        pos, neg = lab == 1, lab == 0
        print(f"{method} train-equivalent: coverage {(lab >= 0).mean():.3f}, precision of 1 {y[pos].mean():.4f}, "
              f"of 0 {1 - y[neg].mean():.4f}, share of 1 among labelled {pos.sum() / max(1, (lab >= 0).sum()):.3f}")
        path, key, names_key = HEAD_TEST[method]
        zt = np.load(path)
        tn = [str(v) for v in zt[names_key]]
        ht = zt[key].astype(np.float32)
        xt = load_map(XB1.format(s=s, k="test"), method, tn)[0]
        st, at = load_map(SAM.format(m=method, k="test"), method, tn)
        p = compose(ht, xt, st, at)
        for i, v in enumerate(tn):
            if v in crect_masks:
                p[i] = np.where(crect_masks[v], 0.99, 0.01)
        lt = labels_of(p)
        tag = __import__("os").environ.get("PSEUDO_TAG", "")
        np.savez_compressed(f"{out}/pseudo{tag}_{method}_test.npz", labels=lt, names=np.array(tn))
        print(f"{method} test: coverage {(lt >= 0).mean():.3f}, share of 1 {(lt == 1).sum() / max(1, (lt >= 0).sum()):.3f}, "
              f"c_rect images {sum(v in crect_masks for v in tn)} -> {out}/pseudo_{method}_test.npz")


if __name__ == "__main__":
    main()
