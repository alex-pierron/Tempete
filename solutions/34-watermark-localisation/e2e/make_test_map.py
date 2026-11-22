"""X-LOC-B (Mac, numpy only): turn e2e OOF / test maps into a thresholded test map for assemble.py.

Blend: p = w * mean(e2e runs) + (1 - w) * heads (w = 1 means e2e alone). The decision threshold is chosen per
predicted regime (share of cells >= 0.5) on the WHOLE OOF set (deployment), and its leave-one-out score is
printed next to the deployed heads, so the number used for the decision is the honest one. The written map
holds 0 / 1 values (key <method>, names_<method>), to be replaced with threshold 0.5.

Usage: python make_test_map.py <method> <weight> <out.npz> <run prefix> [<run prefix> ...]
       (prefix = path without the _oof.npz / _test.npz suffix)
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "loclab"))  # loclab helper modules
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # e2e helper modules
from common import gt128, train_meta  # noqa: E402
from compare_oof import THRESHOLDS, miou_batch  # noqa: E402
from size_prior import DEPLOYED, train_map  # noqa: E402

HEADS_TEST = {
    "pixelseal": (__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_ps_3.npz"),
    "trustmark": (__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_tm_2.npz"),
}
DECISION = 0.5


class MapError(ValueError):
    """Name orders differ or an OOF map is incomplete."""


def regime(prob: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Predicted full regime: at least half of the cells >= DECISION."""
    return (prob >= DECISION).mean((1, 2)) >= DECISION


def choose(prob: NDArray[np.float32], gt: NDArray[np.bool_], mask: NDArray[np.bool_]) -> float:
    """Best grid threshold on the images selected by mask."""
    return max(THRESHOLDS, key=lambda t: miou_batch(prob[mask] >= t, gt[mask]).sum())


def main() -> None:
    """Print the OOF numbers and write the thresholded test map."""
    method, weight, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    prefixes = sys.argv[4:]
    meta = train_meta(method)
    names = [str(r["image_name"]) for r in meta]
    g = gt128()
    gt = np.stack([g[n] > 0 for n in names])
    oofs, tests, test_names = [], [], None
    for prefix in prefixes:
        z = np.load(f"{prefix}_oof.npz")
        if list(z[f"names_{method}"]) != names or not bool(z["seen"].all()):
            raise MapError(f"{prefix}: OOF names differ or incomplete")
        oofs.append(z[method].astype(np.float32))
        t = np.load(f"{prefix}_test.npz")
        current = [str(n) for n in t[f"names_{method}"]]
        if test_names is not None and current != test_names:
            raise MapError(f"{prefix}: test names differ")
        test_names = current
        tests.append(t[method].astype(np.float32))
    assert test_names is not None
    heads_oof = train_map(method, names)
    h = np.load(HEADS_TEST[method])
    heads_names = [str(n) for n in h[f"names_{method}"]]
    position = {n: i for i, n in enumerate(heads_names)}
    heads_test = h[method].astype(np.float32)[[position[n] for n in test_names]]
    oof = weight * np.mean(oofs, 0) + (1 - weight) * heads_oof
    test = weight * np.mean(tests, 0) + (1 - weight) * heads_test
    full_oof, full_test = regime(oof), regime(test)
    thresholds = {f: choose(oof, gt, full_oof == f) for f in (True, False)}
    loo = np.empty(len(gt))
    table = {t: miou_batch(oof >= t, gt) for t in THRESHOLDS}
    for i in range(len(gt)):
        same = (full_oof == full_oof[i]) & (np.arange(len(gt)) != i)
        loo[i] = table[max(THRESHOLDS, key=lambda t: table[t][same].sum())][i]
    hp = regime(heads_oof)
    base = miou_batch(heads_oof >= np.where(hp, DEPLOYED[method][0], DEPLOYED[method][1])[:, None, None], gt)
    print(f"{method} w={weight} runs={len(prefixes)}: heads {base.mean():.4f} | blend@0.5 "
          f"{miou_batch(oof >= DECISION, gt).mean():.4f} | regime LOO {loo.mean():.4f} (delta {loo.mean() - base.mean():+.4f})"
          f" | thresholds full {thresholds[True]} rect {thresholds[False]} | test full share {full_test.mean():.3f}")
    binary = test >= np.where(full_test, thresholds[True], thresholds[False])[:, None, None]
    changed = (binary != (heads_test >= DECISION)).mean()
    np.savez(out, **{method: binary.astype(np.float16), f"names_{method}": np.array(test_names)})
    print(f"wrote {out}: {len(test_names)} maps, cells differing from heads@0.5 {changed:.3f}")


if __name__ == "__main__":
    main()
