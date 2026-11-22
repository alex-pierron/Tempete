"""X-LOC transductive round 2: pseudo-labels of the TEST images from a configurable composition (the deposited one).

Per method: rect map (mean with the head or alone) on head-predicted rect images, full map (mean or alone) on predicted
full images, SAM masks (0.99 / 0.01) on its apply images, c_rect masks last. Label = 1 if p > 0.9, 0 if p < 0.1, -1
otherwise and in a 2-cell band around the p >= 0.5 boundary. Also prints the train-equivalent label precision.

Usage: python pseudo_labels2.py <out_dir> <tag> <method>:<rect_prefix>:<rect_mode>:<full_prefix>:<full_mode>:<sam_prefix> ...
       a prefix P means P_oof.npz / P_test.npz (e2e maps) or P_train.npz / P_test.npz (SAM); '-' = none (head).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # loclab helper modules
from common import gt128, train_meta  # noqa: E402
from integrate_map import HEAD_TEST, load_map  # noqa: E402
from pseudo_labels import CRECT, labels_of  # noqa: E402
from size_prior import train_map  # noqa: E402


def compose(head: np.ndarray, rect: np.ndarray | None, rect_mode: str, full_map: np.ndarray | None, full_mode: str,
            sam: np.ndarray | None, apply: np.ndarray | None) -> np.ndarray:
    """Probability composition before the size-prior shift."""
    full = (head >= 0.5).mean(axis=(1, 2)) >= 0.5

    def mix(other: np.ndarray | None, mode: str) -> np.ndarray:
        if other is None:
            return head
        return other if mode == "alone" else 0.5 * (head + other)

    p = np.where(full[:, None, None], mix(full_map, full_mode), mix(rect, rect_mode))
    if sam is not None and apply is not None:
        p = np.where(apply[:, None, None], np.where(sam > 0.5, 0.99, 0.01), p)
    return p.astype(np.float32)


def main() -> None:
    """Write pseudo<tag>_<method>_test.npz for every spec."""
    out, tag = sys.argv[1], sys.argv[2]
    crect = torch.load(CRECT, weights_only=True)
    crect_masks = {n: m.numpy() > 0 for n, m, c in zip(crect["image_names"], crect["masks"], crect["confidence"]) if float(c) >= 0.85}
    gts = gt128()
    for spec in sys.argv[3:]:
        method, rp, rmode, fp, fmode, sp = spec.split(":")
        names = [str(r["image_name"]) for r in train_meta(method)]
        head = train_map(method, names)
        get = lambda p, k, n: None if p == "-" else load_map(f"{p}_{k}.npz", method, n)  # noqa: E731
        r, f, s = get(rp, "oof", names), get(fp, "oof", names), get(sp, "train", names)
        lab = labels_of(compose(head, None if r is None else r[0], rmode, None if f is None else f[0], fmode,
                                None if s is None else s[0], None if s is None else s[1]))
        y = np.stack([gts[v] > 0 for v in names])
        print(f"{method} train-equivalent: coverage {(lab >= 0).mean():.3f}, precision 1 {y[lab == 1].mean():.4f}, "
              f"0 {1 - y[lab == 0].mean():.4f}")
        path, key, names_key = HEAD_TEST[method]
        zt = np.load(path)
        tn = [str(v) for v in zt[names_key]]
        ht = zt[key].astype(np.float32)
        r, f, s = get(rp, "test", tn), get(fp, "test", tn), get(sp, "test", tn)
        p = compose(ht, None if r is None else r[0], rmode, None if f is None else f[0], fmode,
                    None if s is None else s[0], None if s is None else s[1])
        for i, v in enumerate(tn):
            if v in crect_masks:
                p[i] = np.where(crect_masks[v], 0.99, 0.01)
        lt = labels_of(p)
        target = os.path.join(out, f"pseudo{tag}_{method}_test.npz")
        np.savez_compressed(target, labels=lt, names=np.array(tn))
        print(f"{method} test: coverage {(lt >= 0).mean():.3f}, share of 1 {(lt == 1).sum() / max(1, (lt >= 0).sum()):.3f} -> {target}")


if __name__ == "__main__":
    main()
