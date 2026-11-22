"""X-LOC-C sam_v2: learned object selector over SAM candidates (+ the head mask as candidate 0).

Each candidate of a rect-regime image gets features (head vote, overlaps with the head map at several
thresholds, size, shape, seam, HF contrast, SAM scores, overlap with the head box) and the target
mIoU(candidate, ground truth). A gradient-boosted regressor is trained out of fold BY IMAGE (GroupKFold);
per image the candidate with the highest predicted mIoU is kept, the head mask (candidate 0) included,
so the gate is learned too. Reported against the head mask and the SAM oracle, per method.

Usage: python sam_learn.py <sam_train.npz> <out_prefix> [<sam_test.npz>]
Writes <out_prefix>_<method>_train.npz (OOF choice) and, with a test file, <out_prefix>_<method>_test.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from sam_select import (  # noqa: E402
    DECISION, GRID, N_TRAIN_TOTAL, RECT_THRESHOLD, TRAIN, fill_of_bbox, head_maps, load_candidates, miou,
)

METHODS = ("pixelseal", "trustmark", "stegastamp")
HARD_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
N_FOLDS = 5
MARGINS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.1)
SEED = 2026
BOOTSTRAP = 2000
GUIDE_DIR = Path((__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc") + "/loclab/xloc"))
USE_GUIDE = True
# Candidate 0 = the mask of the base the objects are applied to. ll_v16_xb1rect thresholds the guide (heads + xb1)
# in the rect regime, so for PS/TM candidate 0 is guide >= 0.5; SS keeps its head mask.
BASE_FROM_GUIDE = True
GUIDE_BASE_T = 0.5


EXTRA_GUIDES: tuple[str, ...] = tuple(g for g in __import__("os").environ.get("EXTRA_GUIDES", "").split(",") if g)


GUIDE_MAIN = __import__("os").environ.get("GUIDE_MAIN", "guide")


def guide_maps(split: str, prefix: str = "") -> dict[str, np.ndarray]:
    """X-LOC guide maps (deployed heads + xb1 mean, out of fold on train), PS and TM only."""
    out: dict[str, np.ndarray] = {}
    prefix = prefix or GUIDE_MAIN
    for m in ("pixelseal", "trustmark"):
        z = np.load((Path(prefix) if "/" in prefix else GUIDE_DIR / prefix).parent / f"{Path(prefix).name}_{m}_{split}.npz")
        out.update({str(n): g.astype(np.float32) for n, g in zip(z[f"names_{m}"], z[m], strict=True)})
    return out


UNION_TOP = 6
UNION_TRIPLE_TOP = 4
USE_UNIONS = False


def augment(cand: dict[str, np.ndarray], guide: np.ndarray) -> dict[str, np.ndarray]:
    """Add unions of 2 (top 6) and 3 (top 4) candidates ranked by soft IoU with the guide map:
    a transplant is often split by SAM (a hand and a cup, a sign and its post)."""
    masks = cand["masks"]
    if not USE_UNIONS or len(masks) < 2:
        return cand
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    gf = guide.reshape(-1)
    inter = flat @ gf
    soft = inter / np.maximum(flat.sum(1) + gf.sum() - inter, 1e-6)
    top = list(np.argsort(-soft)[:UNION_TOP])
    combos = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    t3 = top[:UNION_TRIPLE_TOP]
    combos += [(a, b, c) for i, a in enumerate(t3) for j, b in enumerate(t3[i + 1:], i + 1) for c in t3[j + 1:]]
    new_masks, seam, hf, piou = [], [], [], []
    for cb in combos:
        u = np.logical_or.reduce([masks[k] for k in cb])
        if any(np.array_equal(u, masks[k]) for k in cb):
            continue
        new_masks.append(u)
        seam.append(float(np.mean([cand["seam"][k] for k in cb])))
        hf.append(float(np.mean([cand["hf"][k] for k in cb])))
        piou.append(float(np.min([cand["piou"][k] for k in cb])))
    if not new_masks:
        return cand
    return {"masks": np.concatenate([masks, np.stack(new_masks)]),
            "seam": np.concatenate([cand["seam"], np.array(seam, np.float32)]),
            "hf": np.concatenate([cand["hf"], np.array(hf, np.float32)]),
            "piou": np.concatenate([cand["piou"], np.array(piou, np.float32)])}


GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)


def box_of(m: np.ndarray) -> np.ndarray:
    """Bounding box of a mask, as a mask (empty stays empty)."""
    out = np.zeros_like(m)
    if m.any():
        ys, xs = np.nonzero(m)
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def features(p: np.ndarray, method: str, cand: dict[str, np.ndarray],
             guide: np.ndarray | None = None, base: np.ndarray | None = None,
             extra: list[np.ndarray | None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix (row 0 = head mask) and the candidate masks (row 0 = head mask)."""
    head = p >= RECT_THRESHOLD[method]
    if BASE_FROM_GUIDE and guide is not None:
        head = guide >= GUIDE_BASE_T
    if base is not None:
        head = base
    masks = np.concatenate([head[None], cand["masks"]], axis=0)
    k = len(masks)
    flat = masks.reshape(k, -1).astype(np.float32)
    area = flat.sum(1)
    pf = p.reshape(-1)
    inter = flat @ pf
    cols: list[np.ndarray] = [
        area / (GRID * GRID),
        inter / np.maximum(area, 1),                                   # vote inside
        (pf.sum() - inter) / np.maximum(GRID * GRID - area, 1),        # vote outside
        inter / np.maximum(area + pf.sum() - inter, 1e-6),             # soft IoU
    ]
    p3 = pf ** 3
    i3 = flat @ p3
    cols.append(i3 / np.maximum(area + p3.sum() - i3, 1e-6))
    for t in HARD_THRESHOLDS:
        h = (pf >= t).astype(np.float32)
        ih = flat @ h
        cols += [ih / np.maximum(area + h.sum() - ih, 1e-6), ih / np.maximum(h.sum(), 1), ih / np.maximum(area, 1)]
    hb = box_of(head).reshape(-1).astype(np.float32)
    ib = flat @ hb
    cols += [ib / np.maximum(area + hb.sum() - ib, 1e-6), ib / np.maximum(area, 1)]
    cols.append(np.array([fill_of_bbox(m) for m in masks], np.float32))
    seam = np.concatenate([[np.nan], cand["seam"]])
    hf = np.concatenate([[np.nan], cand["hf"]])
    piou = np.concatenate([[np.nan], cand["piou"]])
    cols += [seam, hf, piou]
    is_head = np.zeros(k, np.float32)
    is_head[0] = 1
    share = float((p >= DECISION).mean())
    img = [np.full(k, v, np.float32) for v in (
        share, fill_of_bbox(head), float(head.mean()), float(len(cand["masks"])),
        *(float(method == m) for m in METHODS))]
    if USE_GUIDE:
        gf = (p if guide is None else guide).reshape(-1)
        ig = flat @ gf
        cols += [ig / np.maximum(area, 1), ig / np.maximum(area + gf.sum() - ig, 1e-6)]
        for t in (0.3, 0.5):
            h = (gf >= t).astype(np.float32)
            ih = flat @ h
            cols += [ih / np.maximum(area + h.sum() - ih, 1e-6), ih / np.maximum(h.sum(), 1)]
    for em in extra or []:
        ef = (p if em is None else em).reshape(-1)
        ie = flat @ ef
        cols += [ie / np.maximum(area, 1), ie / np.maximum(area + ef.sum() - ie, 1e-6)]
        for t in (0.3, 0.5):
            h = (ef >= t).astype(np.float32)
            ih = flat @ h
            cols += [ih / np.maximum(area + h.sum() - ih, 1e-6), ih / np.maximum(h.sum(), 1)]
    x = np.stack(cols + [is_head] + img, axis=1).astype(np.float32)
    return x, masks


def main() -> None:
    """Out-of-fold selection on the train rect images; optional test export."""
    cands = load_candidates(Path(sys.argv[1]))
    meta = {s["name"]: s for s in json.loads((TRAIN / "manifest.json").read_text())["samples"]}
    z = np.load(TRAIN / "masks_128.npz")
    gts = {str(n): m > 0 for n, m in zip(z["names"], z["masks"], strict=True)}
    maps = head_maps()
    guides = guide_maps("train")
    extras = [guide_maps("train", g) for g in EXTRA_GUIDES]
    names = [n for n in cands if n in maps]
    xs, ys, groups, masks_all = [], [], [], []
    for gi, n in enumerate(names):
        gd = guides.get(n)
        x, ms = features(maps[n], meta[n]["method"], augment(cands[n], maps[n] if gd is None else gd), gd, extra=[e.get(n) for e in extras])
        y = np.array([miou(m, gts[n]) for m in ms], np.float32)
        xs.append(x)
        ys.append(y)
        groups.append(np.full(len(y), gi))
        masks_all.append(ms)
    X, Y, G = np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)
    pred = np.zeros(len(Y), np.float32)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, Y, G):
        model = HistGradientBoostingRegressor(random_state=SEED, **GBM).fit(X[tr], Y[tr])
        pred[te] = model.predict(X[te])
    base, chosen, oracle = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
    methods = np.array([meta[n]["method"] for n in names])
    transplant = np.zeros(len(names), bool)
    changed = np.zeros(len(names), bool)
    by_margin = np.zeros((len(MARGINS), len(names)))
    oof_mask: dict[str, np.ndarray] = {}
    for gi, n in enumerate(names):
        sel = G == gi
        y, pr = Y[sel], pred[sel]
        base[gi], oracle[gi] = y[0], y.max()
        c = int(np.argmax(pr))
        chosen[gi], changed[gi] = y[c], c != 0
        oof_mask[n] = masks_all[gi][c]
        for mi, mg in enumerate(MARGINS):
            by_margin[mi, gi] = y[c] if pr[c] - pr[0] > mg else y[0]
        g = gts[n]
        transplant[gi] = fill_of_bbox(g if g.mean() < DECISION else ~g) < 0.98
    d = chosen - base
    rng = np.random.default_rng(SEED)
    print(f"OOF selector: delta on the 510 score {d.sum() / N_TRAIN_TOTAL:+.4f} "
          f"[{np.percentile([d[rng.integers(0, len(d), len(d))].sum() / N_TRAIN_TOTAL for _ in range(BOOTSTRAP)], [2.5, 97.5]).round(4)}]"
          f", changed {int(changed.sum())}/{len(names)}")
    for m in METHODS:
        k = methods == m
        dm = d[k]
        bt = [dm[rng.integers(0, len(dm), len(dm))].sum() / N_TRAIN_TOTAL for _ in range(BOOTSTRAP)]
        print(f"  {m}: {dm.sum() / N_TRAIN_TOTAL:+.4f} [{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}] "
              f"changed {int(changed[k].sum())}")
        for lab, s in (("transplant", transplant), ("rectangle", ~transplant)):
            kk = k & s
            print(f"    {lab:10s} n={kk.sum():3d} base {base[kk].mean():.3f} chosen {chosen[kk].mean():.3f} "
                  f"oracle {oracle[kk].mean():.3f}")
    for m in METHODS:
        k = methods == m
        print(f"  margins {m}: " + " ".join(f"{mg}:{(by_margin[mi, k] - base[k]).sum() / N_TRAIN_TOTAL:+.4f}"
                                            for mi, mg in enumerate(MARGINS)))
    prefix = Path(sys.argv[2])
    meta_rows = json.loads((TRAIN / "metadata.json").read_text())
    for m in METHODS:
        all_names = [r["image_name"] for r in meta_rows if r["method"] == m]
        out = np.zeros((len(all_names), GRID, GRID), np.uint8)
        apply = np.zeros(len(all_names), bool)
        for j, n in enumerate(all_names):
            if n in oof_mask:
                gi = names.index(n)
                out[j], apply[j] = oof_mask[n], bool(changed[gi])
        tmp = prefix.parent / f"{prefix.name}_{m}_train.tmp.npz"
        np.savez_compressed(tmp, **{m: out, f"names_{m}": np.array(all_names), f"apply_{m}": apply})
        tmp.replace(prefix.parent / f"{prefix.name}_{m}_train.npz")
    if len(sys.argv) > 3:
        export_test(X, Y, Path(sys.argv[3]), prefix)


def export_test(X: np.ndarray, Y: np.ndarray, test_npz: Path, prefix: Path) -> None:
    """Fit on all train candidates, choose per test image; integrator npz per method (apply = not the head)."""
    model = HistGradientBoostingRegressor(random_state=SEED, **GBM).fit(X, Y)
    cands = load_candidates(test_npz)
    guides = guide_maps("test")
    extras = [guide_maps("test", g) for g in EXTRA_GUIDES]
    from sam_build import TEST_MAPS  # local import: torch only needed for the test export
    for method, (path, key, nkey) in TEST_MAPS.items():
        z = np.load(path)
        all_names = [str(n) for n in z[nkey]]
        out = np.zeros((len(all_names), GRID, GRID), np.uint8)
        apply = np.zeros(len(all_names), bool)
        score = np.zeros(len(all_names), np.float32)
        for j, (n, p) in enumerate(zip(all_names, z[key], strict=True)):
            if n not in cands:
                continue
            gd = guides.get(n)
            pp = p.astype(np.float32)
            x, ms = features(pp, method, augment(cands[n], pp if gd is None else gd), gd, extra=[e.get(n) for e in extras])
            pr = model.predict(x)
            c = int(np.argmax(pr))
            out[j], apply[j], score[j] = ms[c], c != 0, pr[c] - pr[0]
        tmp = prefix.parent / f"{prefix.name}_{method}_test.tmp.npz"
        np.savez_compressed(tmp, **{method: out, f"names_{method}": np.array(all_names), f"apply_{method}": apply,
                                    f"gain_{method}": score})
        tmp.replace(prefix.parent / f"{prefix.name}_{method}_test.npz")
        print(f"test export {method}: apply {int(apply.sum())}/{len(all_names)}")


if __name__ == "__main__":
    main()
