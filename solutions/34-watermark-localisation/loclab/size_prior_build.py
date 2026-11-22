"""Test maps with the size-prior thresholds (ll_v13): threshold per (predicted regime, predicted zone size) bin.

Thresholds are chosen on the whole train OOF (size_prior.py evaluates the same rule leave-one-out), then each
test map is shifted by 0.5 - t(image) so that assemble.py at 0.5 applies them. Bins and predictions exactly as
size_prior.py (regime: share of cells >= 0.5 at least one half; size: minority share, edges 0.03 / 0.08).
With FORCE="0.5,0.5" (PS) or "0.6,0.4" (TM) for every bin it must reproduce ll_v12 (sanity mode).

SS: pass the split7 container (team_final/ss_split7_u1x5_zoom2.npz) as the test map. Env SIZE_EDGES as size_prior.

Usage: python size_prior_build.py <method> <test_map.npz> <out.npz> [sanity]
"""

from __future__ import annotations

import sys

import numpy as np

# Sibling modules (common, emiou_eval, heads_eval, size_prior) are imported from this directory.
from common import gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402
from size_prior import DECISION, DEPLOYED, GRID, SIZE_EDGES, train_map  # noqa: E402

N_SIZE_BINS = len(SIZE_EDGES) + 1


def groups_of(p: np.ndarray) -> np.ndarray:
    """(regime, size bin) group of each map: full * N_SIZE_BINS + size bin."""
    share = (p >= DECISION).mean(axis=(1, 2))
    full = share >= DECISION
    minority = np.where(full, 1 - share, share)
    return full.astype(int) * N_SIZE_BINS + np.digitize(minority, SIZE_EDGES)


def main() -> None:
    """Choose thresholds on train, shift the test map, write it."""
    method, test_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sanity = len(sys.argv) > 4 and sys.argv[4] == "sanity"
    meta = train_meta(method)
    gts = gt128()
    g = [gts[r["image_name"]] > 0 for r in meta]
    p = train_map(method, [str(r["image_name"]) for r in meta])
    grp = groups_of(p)
    table = {t: np.array([miou_fast(m >= t, x) for m, x in zip(p, g, strict=True)]) for t in GRID}
    chosen: dict[int, float] = {}
    for k in range(2 * N_SIZE_BINS):
        if sanity:
            chosen[k] = DEPLOYED[method][0] if k >= N_SIZE_BINS else DEPLOYED[method][1]
        elif (grp == k).any():
            chosen[k] = max(GRID, key=lambda t: table[t][grp == k].sum())
        else:
            chosen[k] = DEPLOYED[method][0] if k >= N_SIZE_BINS else DEPLOYED[method][1]
    z = np.load(test_path)
    # SS: the split7 container (test_finetuned / names_test, already regime-shifted); PS / TM: ensemble maps.
    names_key, map_key = (("names_test", "test_finetuned") if method == "stegastamp" else (f"names_{method}", method))
    q = z[map_key].astype(np.float32)
    tgrp = groups_of(q)
    t_img = np.array([chosen[int(k)] for k in tgrp], dtype=np.float32)
    shifted = q + (DECISION - t_img)[:, None, None]
    np.savez(out, **{method: shifted, f"names_{method}": z[names_key]})
    counts = {int(k): int((tgrp == k).sum()) for k in np.unique(tgrp)}
    print(f"{method} thresholds {chosen} test groups {counts} -> {out}")


if __name__ == "__main__":
    main()
