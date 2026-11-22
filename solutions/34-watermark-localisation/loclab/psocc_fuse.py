"""M-LOC: PS full regime, logistic fusion of the decoder occlusion drops with the e2e maps (OOF).

The record's PS full masks come from an occlusion overlay fused with the OLD heads. Here the same occlusion
drops (blur12 / gray fills, windows 32-96 px, stride 16) are fused per cell with the heads AND the mean of the
e2e maps (xb2, xpl1, xpl2, xbig, ...), 5 folds by image, on head-predicted full images. Reference = the same
fusion without the e2e maps (proxy of the current overlay). Size prior LOO on top, as combo_eval.

Usage: SIZE_EDGES=0.03,0.08 python psocc_fuse.py <e2e tags comma> [--build <out_prefix>]
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import gt128, train_meta  # noqa: E402
from fuse_dense import CELLS_PER_IMAGE, FOLDS, ci, fit_logistic, predict, size_prior_loo  # noqa: E402
from integrate_map import HEAD_TEST, load_map, logit, thresholds_shift  # noqa: E402
from size_prior import DECISION, SEED, train_map  # noqa: E402

METHOD = "pixelseal"
W = __import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc")
OCC_DIR = f"{W}/research_cache_rect/out/occ_pixelseal_"
OCC_KEYS = ("drop_blur12_b32", "drop_blur12_b48", "drop_blur12_b64", "drop_blur12_b96", "drop_gray_b48", "drop_gray_b96")
STRIDE_PX, CELL_PX, GRID = 16, 4, 128
E2E = {"xb1": "loc_gpu_in/xlocb_xb1/xb1_ps", "xb2": "loc_gpu_in/xlocb_xb1/xb2_ps", "xb3": "loc_gpu_in/xlocb_xb1/xb3_ps",
       "xpl1": "loclab/xloc/xpl1_ps", "xpl2": "loclab/xloc/xpl2_ps", "xbig": "loc_gpu_in/xlocb_xb1/xbig_ps",
       "xbig2": "loc_gpu_in/xlocb_xb1/xbig2_ps", "dn3": "loclab/xloc/dn3sam", "dn2": "loclab/xloc/dn2avg"}
SAM_TRAIN = f"{W}/loclab/x_loc_c/samGx_pixelseal_train.npz"
SMOOTH_CELLS = 2.0
Z_CLIP = 10.0


def window_to_cells(d: np.ndarray, block_px: int) -> np.ndarray:
    """Mean drop of every window covering each 4 px cell (windows of block_px at stride 16 px)."""
    acc = np.zeros((GRID, GRID), np.float32)
    cnt = np.zeros((GRID, GRID), np.float32)
    span = block_px // CELL_PX
    step = STRIDE_PX // CELL_PX
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            acc[i * step:i * step + span, j * step:j * step + span] += d[i, j]
            cnt[i * step:i * step + span, j * step:j * step + span] += 1
    return acc / np.maximum(cnt, 1)


def load_occ(split: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """([n, K, 128, 128] occlusion drops on the cell grid, [n] availability), in the order of names."""
    raw: dict[str, dict[str, np.ndarray]] = {}
    for f in glob.glob(f"{OCC_DIR}{split}/*.npz"):
        z = np.load(f)
        for k in z.files:
            n, key = k.split("/", 1)
            if key in OCC_KEYS:
                raw.setdefault(n, {})[key] = z[k]
    out = np.zeros((len(names), len(OCC_KEYS), GRID, GRID), np.float32)
    have = np.array([n in raw and len(raw[n]) == len(OCC_KEYS) for n in names])
    for a, n in enumerate(names):
        if have[a]:
            for b, key in enumerate(OCC_KEYS):
                out[a, b] = window_to_cells(raw[n][key], int(key.rsplit("_b", 1)[1]))
    return out, have


TEXTURE_SIGMAS = (2.0, 6.0, 10.0)
TEXTURE_POOL_CELLS = 4.0
USE_TEXTURE = __import__("os").environ.get("MLOC_TEXTURE", "0") == "1"
_TEX: dict[str, np.ndarray] = {}


def texture(split: str, names: list[str]) -> np.ndarray:
    """[n, 3, 128, 128] log local energy of (Y - blur_sigma(Y)), pooled (the content part of an occlusion drop)."""
    from PIL import Image
    folder = f"{W}/release/{'train' if split == 'train' else 'test'}/images"
    out = np.zeros((len(names), len(TEXTURE_SIGMAS), GRID, GRID), np.float32)
    for a, n in enumerate(names):
        if n in _TEX:
            out[a] = _TEX[n]
            continue
        y = np.asarray(Image.open(f"{folder}/{n}.png").convert("L"), dtype=np.float32)
        for b, sg in enumerate(TEXTURE_SIGMAS):
            e = (y - ndimage.gaussian_filter(y, sg)) ** 2
            cell = e.reshape(GRID, CELL_PX, GRID, CELL_PX).mean(axis=(1, 3))
            out[a, b] = np.log(ndimage.gaussian_filter(cell, TEXTURE_POOL_CELLS) + 1e-3)
        _TEX[n] = out[a]
    return out


def features(head: np.ndarray, e2e: np.ndarray | None, occ: np.ndarray, tex: np.ndarray | None = None) -> np.ndarray:
    """[n, 128, 128, F]: 1, head logit, (e2e logit), robust z of each drop map and its smoothed copy, product."""
    feats = [np.ones_like(head), logit(head)]
    if e2e is not None:
        feats.append(logit(e2e))
    for k in range(occ.shape[1]):
        o = occ[:, k]
        med = np.median(o.reshape(len(o), -1), axis=1)[:, None, None]
        mad = np.median(np.abs(o - med).reshape(len(o), -1), axis=1)[:, None, None] + 1e-6
        z = np.clip((o - med) / mad, -Z_CLIP, Z_CLIP)
        feats += [z, np.stack([ndimage.gaussian_filter(x, SMOOTH_CELLS) for x in z])]
    feats.append(feats[1] * feats[-1])
    if tex is not None:
        z48 = feats[2 + (e2e is not None) + 2]  # robust z of the b48 blur drop
        for b in range(tex.shape[1]):
            feats += [tex[:, b], tex[:, b] * z48]
    return np.stack(feats, axis=-1).astype(np.float32)


def fit_cells(x: np.ndarray, y: np.ndarray, images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Logistic fit on a cell subsample of the given images."""
    xs, ys = [], []
    for i in images:
        cells = rng.choice(GRID * GRID, CELLS_PER_IMAGE, replace=False)
        xs.append(x[i].reshape(-1, x.shape[-1])[cells])
        ys.append(y[i].reshape(-1)[cells])
    return fit_logistic(np.concatenate(xs), np.concatenate(ys))


def main() -> None:
    tags = sys.argv[1].split(",")
    meta = train_meta(METHOD)
    names = [str(r["image_name"]) for r in meta]
    regime = np.array([bool(r["full_watermark"]) for r in meta])
    gts = gt128()
    g = [gts[v] > 0 for v in names]
    y = np.stack(g).astype(np.float64)
    head = train_map(METHOD, names)
    e2e = np.mean([load_map(f"{W}/{E2E[t]}_oof.npz", METHOD, names)[0] for t in tags], axis=0)
    xb1 = load_map(f"{W}/{E2E['xb1']}_oof.npz", METHOD, names)[0]
    sam, apply = load_map(SAM_TRAIN, METHOD, names)
    occ, have = load_occ("train", names)
    full = ((head >= DECISION).mean(axis=(1, 2)) >= DECISION) & have
    print(f"train: drops {int(have.sum())}, head-predicted full with drops {int(full.sum())}, "
          f"pred-full without drops {int((((head >= DECISION).mean(axis=(1, 2)) >= DECISION) & ~have).sum())}")
    rect_comp = 0.5 * (head + xb1)  # record: mean(heads, xb1) on predicted-rect images
    tex = None
    if USE_TEXTURE:
        tex = np.zeros((len(names), len(TEXTURE_SIGMAS), GRID, GRID), np.float32)
        idx = np.flatnonzero(full)
        tex[idx] = texture("train", [names[i] for i in idx])
    variants = {"head+occ (proxy record)": (head, None), "head+e2e+occ": (head, e2e),
                "e2e+occ": (e2e, None), "head+e2e no occ": (0.5 * (head + e2e), "noocc")}
    rng0 = np.random.default_rng(SEED)
    folds = np.empty(len(names), dtype=np.int64)
    for flag in (True, False):
        members = np.flatnonzero(regime == flag)
        folds[members] = rng0.permutation(len(members)) % FOLDS
    half = np.full(len(g), DECISION)
    results = {}
    for label, (base, extra) in variants.items():
        rng = np.random.default_rng(SEED)
        if isinstance(extra, str):
            fused_full = base
        else:
            x = features(base, extra, occ, tex)
            fused_full = base.copy()
            for k in range(FOLDS):
                tr = np.flatnonzero((folds != k) & full)
                te = np.flatnonzero((folds == k) & full)
                fused_full[te] = predict(fit_cells(x, y, tr, rng), x[te])
        p = np.where(full[:, None, None], fused_full, rect_comp)
        p = np.where(apply[:, None, None], np.where(sam > 0.5, 0.99, 0.01), p).astype(np.float32)
        _, loo, chosen = size_prior_loo(p, g, half)
        results[label] = (loo, chosen)
        np.savez(f"oof_{label.split()[0].replace('+', '_')}.npz", p=p, full=full, g=np.stack(g), loo=loo)
    ref = results["head+occ (proxy record)"][0]
    for label, (loo, chosen) in results.items():
        print(f"{label:24s} all {loo.mean():.4f} pred-full {loo[full].mean():.4f}  dfull {ci((loo - ref)[full])}  dall {ci(loo - ref)}")
    if "--build" in sys.argv:
        build(sys.argv[sys.argv.index("--build") + 1], tags, names, head, e2e, occ, y, full, results, tex)


def build(prefix: str, tags: list[str], names: list[str], head: np.ndarray, e2e: np.ndarray, occ: np.ndarray,
          y: np.ndarray, full: np.ndarray, results: dict, tex: np.ndarray | None = None) -> None:
    """Fit on all predicted-full train images, predict the test, fold the size-prior thresholds in."""
    rng = np.random.default_rng(SEED)
    params = fit_cells(features(head, e2e, occ, tex), y, np.flatnonzero(full), rng)
    path, key, names_key = HEAD_TEST[METHOD]
    zt = np.load(path)
    tn = [str(v) for v in zt[names_key]]
    ht = zt[key].astype(np.float32)
    et = np.mean([load_map(f"{W}/{E2E[t]}_test.npz", METHOD, tn)[0] for t in tags], axis=0)
    occ_t, have_t = load_occ("test", tn)
    full_t = ((ht >= DECISION).mean(axis=(1, 2)) >= DECISION) & have_t
    print(f"test: drops {int(have_t.sum())}, predicted full with drops {int(full_t.sum())}")
    tex_t = None
    if tex is not None:
        tex_t = np.zeros((len(tn), len(TEXTURE_SIGMAS), GRID, GRID), np.float32)
        idx = np.flatnonzero(full_t)
        tex_t[idx] = texture("test", [tn[i] for i in idx])
    fused_t = np.where(full_t[:, None, None], predict(params, features(ht, et, occ_t, tex_t)), ht).astype(np.float32)
    chosen = results["head+e2e+occ"][1]
    np.savez(f"{prefix}_pixelseal.npz", **{METHOD: thresholds_shift(fused_t, chosen),
                                          f"names_{METHOD}": np.array(tn), "full_pixelseal": full_t, "raw_pixelseal": fused_t})
    # reference (head + occlusion) on the same images, to check the proxy against the deployed overlay
    rng = np.random.default_rng(SEED)
    params_ref = fit_cells(features(head, None, occ, tex), y, np.flatnonzero(full), rng)
    ref_t = np.where(full_t[:, None, None], predict(params_ref, features(ht, None, occ_t, tex_t)), ht).astype(np.float32)
    np.savez(f"{prefix}_ref_pixelseal.npz", **{METHOD: thresholds_shift(ref_t, results["head+occ (proxy record)"][1]),
                                              f"names_{METHOD}": np.array(tn), "full_pixelseal": full_t})
    print(f"built -> {prefix}_pixelseal.npz (full test {int(full_t.sum())}) thresholds {chosen}")


if __name__ == "__main__":
    main()
