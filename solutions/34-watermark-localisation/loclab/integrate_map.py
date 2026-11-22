"""X-LOC integrator: evaluate a new per-image map (X-LOC-B fine-tuned extractors, X-LOC-C SAM, ...) against the
ll_v14 recipe of its method, out of fold, and build the test map for assemble.py --replace.

Input maps (.npz): key <method> = [n, 128, 128] probability (1 = watermarked), names_<method> = [n]; optional
apply_<method> = [n] bool (replace only those images). Train file: the 170 train images, out of fold. Test: 340.

Modes, each followed by the size-prior decision (regime x predicted size, LOO thresholds, same edges as ll_v14):
  head   the deployed head map (ll_v14 recipe for this method; reference)
  alone  the new map
  mean   mean of head and new probabilities
  fuse   per-cell logistic fusion (head logit, new logit, product, new smoothed), 5 folds by image
  apply  new map on the apply_<method> images only, head map elsewhere (when the key exists)
Prints mean LOO score per mode, delta vs head with a bootstrap 95 % CI, and per true zone size class.

Usage: SIZE_EDGES=... python integrate_map.py <method> <train.npz> <test.npz> [--build <mode> <out.npz>]
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # loclab helper modules
from common import gt128, train_meta  # noqa: E402
from fuse_dense import CELLS_PER_IMAGE, FOLDS, ci, fit_logistic, predict, size_prior_loo  # noqa: E402
from size_prior import DECISION, DEPLOYED, SEED, SIZE_EDGES, train_map  # noqa: E402

HEAD_TEST = {
    "pixelseal": ((__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_ps_3.npz"), "pixelseal", "names_pixelseal"),
    "trustmark": ((__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_tm_2.npz"), "trustmark", "names_trustmark"),
    "stegastamp": ((__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/team_final/ss_split7_u1x5_zoom2.npz"), "test_finetuned", "names_test"),
}
EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit."""
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fuse_features(h: np.ndarray, n: np.ndarray) -> np.ndarray:
    """[n, 128, 128, 5]: 1, head logit, new logit, product, new logit smoothed (sigma 2 cells)."""
    lh, ln = logit(h), logit(n)
    smooth = np.stack([ndimage.gaussian_filter(x, 2) for x in ln])
    return np.stack([np.ones_like(lh), lh, ln, lh * ln, smooth], axis=-1).astype(np.float32)


def load_map(path: str, method: str, names: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Map (and optional apply flags) of `method` in the order of `names`."""
    z = np.load(path)
    pos = {str(v): i for i, v in enumerate(z[f"names_{method}"])}
    missing = [v for v in names if v not in pos]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} names missing, e.g. {missing[:3]}")
    order = [pos[v] for v in names]
    apply = z[f"apply_{method}"].astype(bool)[order] if f"apply_{method}" in z.files else None
    return z[method].astype(np.float32)[order], apply


def thresholds_shift(p: np.ndarray, chosen: dict[int, float]) -> np.ndarray:
    """Shift each map by 0.5 - t(group) so that assemble.py at 0.5 applies the size-prior thresholds."""
    share = (p >= DECISION).mean(axis=(1, 2))
    full = share >= DECISION
    groups = full.astype(int) * (len(SIZE_EDGES) + 1) + np.digitize(np.where(full, 1 - share, share), SIZE_EDGES)
    t = np.array([chosen.get(int(k), DECISION) for k in groups], dtype=np.float32)
    return p + (DECISION - t)[:, None, None]


def main() -> None:
    """Evaluate every mode out of fold; optionally build the test map of one mode."""
    method, train_path, test_path = sys.argv[1], sys.argv[2], sys.argv[3]
    build = sys.argv[sys.argv.index("--build") + 1 : sys.argv.index("--build") + 3] if "--build" in sys.argv else None
    meta = train_meta(method)
    names = [str(r["image_name"]) for r in meta]
    regime = np.array([bool(r["full_watermark"]) for r in meta])
    gts = gt128()
    g = [gts[v] > 0 for v in names]
    y = np.stack(g).astype(np.float64)
    head = train_map(method, names)
    new, apply = load_map(train_path, method, names)
    rng = np.random.default_rng(SEED)
    folds = np.empty(len(names), dtype=np.int64)
    for flag in (True, False):
        members = np.flatnonzero(regime == flag)
        folds[members] = rng.permutation(len(members)) % FOLDS
    x = fuse_features(head, new)
    fused = np.empty(head.shape, dtype=np.float32)
    for k in range(FOLDS):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        xs, ys = [], []
        for i in tr:
            cells = rng.choice(128 * 128, CELLS_PER_IMAGE, replace=False)
            xs.append(x[i].reshape(-1, x.shape[-1])[cells])
            ys.append(y[i].reshape(-1)[cells])
        fused[te] = predict(fit_logistic(np.concatenate(xs), np.concatenate(ys)), x[te])
    modes = {"head": head, "alone": new, "mean": 0.5 * (head + new), "fuse": fused}
    if apply is not None:
        modes["apply"] = np.where(apply[:, None, None], new, head)
    t_head = np.where((head >= DECISION).mean(axis=(1, 2)) >= DECISION, DEPLOYED[method][0], DEPLOYED[method][1])
    results = {}
    for mode, p in modes.items():
        t = t_head if mode == "head" else np.full(len(g), DECISION)
        results[mode] = size_prior_loo(p, g, t)
    loo_head = results["head"][1]
    share = y.mean(axis=(1, 2))
    minority = np.where(regime, 1 - share, share)
    classes = ((0, 0.03), (0.03, 0.1), (0.1, 1.01))
    print(f"{method} (edges {SIZE_EDGES}); apply images: {None if apply is None else int(apply.sum())}")
    for mode, (base, loo, _) in results.items():
        by = " ".join(f"[{lo:.2f},{hi:.2f}) {loo[(minority >= lo) & (minority < hi)].mean():.3f}" for lo, hi in classes)
        delta = "" if mode == "head" else f" delta {ci(loo - loo_head)}"
        print(f"  {mode:6s} @thr {base.mean():.4f} size-prior LOO {loo.mean():.4f}{delta} | {by}")
    if build is None:
        return
    mode, out = build
    path, key, names_key = HEAD_TEST[method]
    zt = np.load(path)
    tpos = {str(v): i for i, v in enumerate(zt[names_key])}
    new_test, apply_test = load_map(test_path, method, [str(v) for v in np.load(test_path)[f"names_{method}"]])
    test_names = [str(v) for v in np.load(test_path)[f"names_{method}"]]
    head_test = zt[key].astype(np.float32)[[tpos[v] for v in test_names]]
    if mode == "alone":
        q = new_test
    elif mode == "mean":
        q = 0.5 * (head_test + new_test)
    elif mode == "apply":
        if apply_test is None:
            raise SystemExit("no apply flags in the test file")
        q = np.where(apply_test[:, None, None], new_test, head_test)
    elif mode == "fuse":
        xs, ys = [], []
        for i in range(len(names)):
            cells = rng.choice(128 * 128, CELLS_PER_IMAGE, replace=False)
            xs.append(x[i].reshape(-1, x.shape[-1])[cells])
            ys.append(y[i].reshape(-1)[cells])
        q = predict(fit_logistic(np.concatenate(xs), np.concatenate(ys)), fuse_features(head_test, new_test))
    else:
        raise SystemExit(f"unknown mode {mode}")
    shifted = thresholds_shift(q.astype(np.float32), results[mode][2])
    np.savez(out, **{method: shifted.astype(np.float32), f"names_{method}": np.array(test_names)})
    print(f"  built {mode} -> {out} thresholds {results[mode][2]}")


if __name__ == "__main__":
    main()
