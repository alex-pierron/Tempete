"""X-LOC integrator: combination of a mean-with-head map (X-LOC-B xb1) and a SAM replacement (X-LOC-C), out of fold.

Map = mean(head, new) (or head when no new map), then SAM masks on the SAM apply images, then the size prior
(LOO thresholds), against the ll_v14 recipe of the method. With --build, writes the test map for assemble.py
(size-prior thresholds folded in, SAM images as 0/1 so any threshold keeps them) and a SAM overlay .pt.

Usage: SIZE_EDGES=... python combo_eval.py <method> <new_train|-> <new_test|-> <sam_train|-> <sam_test|-> [--build <out_prefix>]
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # loclab helper modules
from common import gt128, train_meta  # noqa: E402
from fuse_dense import ci, size_prior_loo  # noqa: E402
from integrate_map import HEAD_TEST, load_map, thresholds_shift  # noqa: E402
from size_prior import DECISION, DEPLOYED, train_map  # noqa: E402

RECT_ONLY = __import__("os").environ.get("COMBO_RECT_ONLY", "") == "1"  # mean only on head-predicted rect images
# Optional map for the head-predicted FULL images (e.g. X-LOC-B denoiser): env COMBO_FULL_TRAIN / COMBO_FULL_TEST
# (same npz format), COMBO_FULL_MODE = mean (mean with the head) or alone.
FULL_MODE = __import__("os").environ.get("COMBO_FULL_MODE", "mean")
RECT_MODE = __import__("os").environ.get("COMBO_RECT_MODE", "mean")  # mean with the head, or alone (new map only)
SAM_HIGH = 0.99
SAM_LOW = 0.01


def combine(head: np.ndarray, new: np.ndarray | None, sam: np.ndarray | None, apply: np.ndarray | None,
            full_map: np.ndarray | None = None) -> np.ndarray:
    """mean(head, new), then SAM 0/1 masks (as 0.01 / 0.99, beyond every grid threshold) on the apply images."""
    p = head if new is None else (new if RECT_MODE == "alone" else 0.5 * (head + new))
    if full_map is not None:
        predicted_full = (head >= DECISION).mean(axis=(1, 2)) >= DECISION
        alt = full_map if FULL_MODE == "alone" else 0.5 * (head + full_map)
        p = np.where(predicted_full[:, None, None], alt, p)
    if new is not None and RECT_ONLY and full_map is None:
        predicted_full = (head >= DECISION).mean(axis=(1, 2)) >= DECISION
        p = np.where(predicted_full[:, None, None], head, p)
    if sam is not None and apply is not None:
        hard = np.where(sam > 0.5, SAM_HIGH, SAM_LOW).astype(np.float32)
        p = np.where(apply[:, None, None], hard, p)
    return p.astype(np.float32)


def main() -> None:
    """Evaluate head / +new / +sam / +new+sam out of fold; optionally build the test map."""
    method, new_tr, new_te, sam_tr, sam_te = sys.argv[1:6]
    meta = train_meta(method)
    names = [str(r["image_name"]) for r in meta]
    gts = gt128()
    g = [gts[v] > 0 for v in names]
    head = train_map(method, names)
    new = None if new_tr == "-" else load_map(new_tr, method, names)[0]
    sam, apply = (None, None) if sam_tr == "-" else load_map(sam_tr, method, names)
    env = __import__("os").environ
    full_tr = load_map(env["COMBO_FULL_TRAIN"], method, names)[0] if env.get("COMBO_FULL_TRAIN") else None
    t_head = np.where((head >= DECISION).mean(axis=(1, 2)) >= DECISION, DEPLOYED[method][0], DEPLOYED[method][1])
    half = np.full(len(g), DECISION)
    ref = size_prior_loo(head, g, t_head)[1]
    variants = {"+new": (new, None, None), "+sam": (None, sam, apply), "+new+sam": (new, sam, apply)}
    chosen_all = {}
    for label, (n, s, a) in variants.items():
        if (label != "+sam" and n is None) or (label != "+new" and s is None):
            continue
        _, loo, chosen = size_prior_loo(combine(head, n, s, a, full_tr), g, half)
        chosen_all[label] = chosen
        print(f"{method} {label:9s} LOO {loo.mean():.4f} vs ll_v14 recipe {ref.mean():.4f}: {ci(loo - ref)}")
    if "--build" not in sys.argv:
        return
    prefix = sys.argv[sys.argv.index("--build") + 1]
    path, key, names_key = HEAD_TEST[method]
    zt = np.load(path)
    test_names = [str(v) for v in zt[names_key]]
    head_test = zt[key].astype(np.float32)
    new_test = None if new_te == "-" else load_map(new_te, method, test_names)[0]
    sam_test, apply_test = (None, None) if sam_te == "-" else load_map(sam_te, method, test_names)
    # SAM given on train but not on test: thresholds of "+new+sam" (SAM images are threshold-free), SAM overlaid later.
    with_sam = sam_tr != "-"
    label = "+new+sam" if new_test is not None and with_sam else ("+new" if new_test is not None else "+sam")
    full_te = load_map(env["COMBO_FULL_TEST"], method, test_names)[0] if env.get("COMBO_FULL_TEST") else None
    q = combine(head_test, new_test, sam_test, apply_test, full_te)
    shifted = thresholds_shift(q, chosen_all[label])
    np.savez(f"{prefix}_{method}.npz", **{method: shifted, f"names_{method}": np.array(test_names)})
    n_apply = 0 if apply_test is None else int(apply_test.sum())
    print(f"built {label} -> {prefix}_{method}.npz (SAM images {n_apply}) thresholds {chosen_all[label]}")


if __name__ == "__main__":
    main()
