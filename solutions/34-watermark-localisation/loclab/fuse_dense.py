"""X-LOC x30 fusion: head ensemble map + dense paired-residual logits -> per-cell logistic fusion, out of fold.

Features per cell: head logit, dense logit smoothed (Gaussian 1 / 2 / 4 / 8 cells), dense logit minus the image
median (per-image calibration), head x dense interaction. Logistic regression (IRLS, L2), 5 folds by image
stratified by regime, fitted on a cell subsample. Then the size-prior decision (regime x predicted size, LOO
thresholds) on the fused map, against the same decision on the head map (the ll_v14 recipe for this method).

Usage: python fuse_dense.py <method> <x30_dense_<method>.npz> [out_test.npz]
Env SIZE_EDGES as size_prior.py (PS 0.03,0.08 ; TM 0.02,0.05,0.10).
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import gt128, train_meta  # noqa: E402
from emiou_eval import miou_fast  # noqa: E402
from size_prior import BOOTSTRAP, DECISION, DEPLOYED, GRID, SEED, SIZE_EDGES, train_map  # noqa: E402

TEST_HEAD = {
    "pixelseal": __import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_ps_3.npz",
    "trustmark": __import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/x23b/ens_tm_2.npz",
}
FOLDS = 5
CELLS_PER_IMAGE = 2000
RIDGE = 1.0
IRLS_STEPS = 25
SIGMAS = (1, 2, 4, 8)
EPS = 1e-4


def features(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """[n, 128, 128, F] float32 feature stack."""
    lh = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
    feats = [np.ones_like(p), lh]
    smooth = {s: np.stack([ndimage.gaussian_filter(x, s) for x in d]) for s in SIGMAS}
    feats += [smooth[s] for s in SIGMAS]
    med = np.median(d.reshape(len(d), -1), axis=1)[:, None, None]
    feats.append(smooth[4] - med)
    feats.append(lh * (smooth[4] - med))
    return np.stack(feats, axis=-1).astype(np.float32)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """IRLS logistic regression with an L2 penalty (intercept included in x, not penalised)."""
    mu_x, sd_x = x.mean(0), x.std(0) + 1e-6
    mu_x[0], sd_x[0] = 0.0, 1.0
    z = (x - mu_x) / sd_x
    w = np.zeros(x.shape[1])
    pen = RIDGE * np.eye(x.shape[1])
    pen[0, 0] = 0.0
    for _ in range(IRLS_STEPS):
        prob = 1 / (1 + np.exp(-(z @ w)))
        grad = z.T @ (prob - y) + pen @ w
        hess = (z * (prob * (1 - prob))[:, None]).T @ z + pen
        w -= np.linalg.solve(hess, grad)
    return np.concatenate([w, mu_x, sd_x])


def predict(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probability of the fused model."""
    k = x.shape[-1]
    w, mu_x, sd_x = params[:k], params[k : 2 * k], params[2 * k :]
    return 1 / (1 + np.exp(-(((x - mu_x) / sd_x) @ w)))


def sample_cells(x: np.ndarray, y: np.ndarray, images: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of cells of the given images."""
    xs, ys = [], []
    for i in images:
        flat = rng.choice(x.shape[1] * x.shape[2], CELLS_PER_IMAGE, replace=False)
        xs.append(x[i].reshape(-1, x.shape[-1])[flat])
        ys.append(y[i].reshape(-1)[flat])
    return np.concatenate(xs), np.concatenate(ys).astype(np.float64)


def size_prior_loo(p: np.ndarray, g: list[np.ndarray], t_dep: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """(deployed score, LOO size-prior score, thresholds chosen on all) of maps p (same rule as size_prior.py)."""
    share = (p >= DECISION).mean(axis=(1, 2))
    full = share >= DECISION
    minority = np.where(full, 1 - share, share)
    n_bins = len(SIZE_EDGES) + 1
    groups = full.astype(int) * n_bins + np.digitize(minority, SIZE_EDGES)
    base = np.array([miou_fast(m >= t, x) for m, t, x in zip(p, t_dep, g, strict=True)])
    table = {t: np.array([miou_fast(m >= t, x) for m, x in zip(p, g, strict=True)]) for t in GRID}
    loo = np.empty(len(g))
    for i in range(len(g)):
        same = (groups == groups[i]) & (np.arange(len(g)) != i)
        loo[i] = table[max(GRID, key=lambda t: table[t][same].sum())][i] if same.any() else base[i]
    chosen = {int(k): max(GRID, key=lambda t: table[t][groups == k].sum()) for k in np.unique(groups)}
    return base, loo, chosen


def ci(d: np.ndarray) -> str:
    """Mean and bootstrap 95 % interval."""
    rng = np.random.default_rng(SEED)
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
    return f"{d.mean():+.4f} [{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}]"


def main() -> None:
    """Out-of-fold fusion, comparison, optional test map."""
    method, dense_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    meta = train_meta(method)
    names = [str(r["image_name"]) for r in meta]
    regime = np.array([bool(r["full_watermark"]) for r in meta])
    gts = gt128()
    g = [gts[n] > 0 for n in names]
    y = np.stack(g).astype(np.float64)
    p = train_map(method, names)
    z = np.load(dense_path)
    pos = {str(n): i for i, n in enumerate(z["names_train"])}
    d = z["train_logit"].astype(np.float32)[[pos[n] for n in names]]
    x = features(p, d)
    rng = np.random.default_rng(SEED)
    folds = np.empty(len(names), dtype=np.int64)
    for flag in (True, False):
        members = np.flatnonzero(regime == flag)
        folds[members] = rng.permutation(len(members)) % FOLDS
    fused = np.empty(p.shape, dtype=np.float32)
    dense_only = np.empty(p.shape, dtype=np.float32)
    keep_dense = [0] + list(range(2, x.shape[-1] - 1))
    for k in range(FOLDS):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        xs, ys = sample_cells(x, y, tr, rng)
        fused[te] = predict(fit_logistic(xs, ys), x[te])
        dense_only[te] = predict(fit_logistic(xs[:, keep_dense], ys), x[te][..., keep_dense])
    t_dep = np.where((p >= DECISION).mean(axis=(1, 2)) >= DECISION, DEPLOYED[method][0], DEPLOYED[method][1])
    base_h, loo_h, _ = size_prior_loo(p, g, t_dep)
    half = np.full(len(g), DECISION)
    base_d, loo_d, _ = size_prior_loo(dense_only, g, half)
    base_f, loo_f, chosen = size_prior_loo(fused, g, half)
    true_share = y.mean(axis=(1, 2))
    minority_true = np.where(regime, 1 - true_share, true_share)
    print(f"{method}: head deployed {base_h.mean():.4f} size-prior LOO {loo_h.mean():.4f} (ll_v14 recipe)")
    print(f"  dense only      @0.5 {base_d.mean():.4f} size-prior LOO {loo_d.mean():.4f}")
    print(f"  fused           @0.5 {base_f.mean():.4f} size-prior LOO {loo_f.mean():.4f}")
    print(f"  fused LOO - head LOO {ci(loo_f - loo_h)} ; fused@0.5 - head deployed {ci(base_f - base_h)}")
    for lo, hi in ((0, 0.03), (0.03, 0.1), (0.1, 1.01)):
        m = (minority_true >= lo) & (minority_true < hi)
        print(f"   true zone [{lo:.2f},{hi:.2f}) n={m.sum():3d} head {loo_h[m].mean():.3f} fused {loo_f[m].mean():.3f} "
              f"dense {loo_d[m].mean():.3f}")
    if out is None:
        return
    xs, ys = sample_cells(x, y, np.arange(len(names)), rng)
    params = fit_logistic(xs, ys)
    th = np.load(TEST_HEAD[method])
    tpos = {str(n): i for i, n in enumerate(th[f"names_{method}"])}
    test_names = [str(n) for n in z["names_test"]]
    pt = th[method].astype(np.float32)[[tpos[n] for n in test_names]]
    q = predict(params, features(pt, z["test_logit"].astype(np.float32)))
    share = (q >= DECISION).mean(axis=(1, 2))
    full = share >= DECISION
    n_bins = len(SIZE_EDGES) + 1
    groups = full.astype(int) * n_bins + np.digitize(np.where(full, 1 - share, share), SIZE_EDGES)
    t_img = np.array([chosen.get(int(k), DECISION) for k in groups], dtype=np.float32)
    np.savez(out, **{method: (q + (DECISION - t_img)[:, None, None]).astype(np.float32), f"names_{method}": np.array(test_names)})
    print(f"  test map -> {out} thresholds {chosen} groups {dict(zip(*np.unique(groups, return_counts=True)))}")


if __name__ == "__main__":
    main()
