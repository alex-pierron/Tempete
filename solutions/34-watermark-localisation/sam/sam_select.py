"""X-LOC-C: choose the transplanted object among SAM candidates, out of fold on the train rect images.

Per image of the rect regime, SAM candidates (128 x 128) are scored against the head map p:
soft IoU(candidate, p) * seam ** W_SEAM (seam = luma step across the candidate contour / 4 px away).
Greedy union of up to MAX_PARTS candidates while the score improves (a pasted object split by SAM).
The chosen mask replaces the deployed head mask when the image is gated (predicted rect regime, deployed
mask fill-of-bbox < F_T) and the score >= S_T. Parameters chosen leave-one-out on the train rect images.

Usage: python sam_select.py <sam_train.npz> [report]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc"))
TRAIN = ROOT / "release" / "train"
HEAD_CV = ROOT / "team_b2" / "cv"
SS_SPLIT7 = ROOT / "team_final" / "ss_split7_u1x5_zoom2.npz"
HEADS = {"pixelseal": ("d_syn3", "d_syn8", "i_syn3"), "trustmark": ("d_syn8", "i_syn8")}
RECT_THRESHOLD = {"pixelseal": 0.5, "trustmark": 0.4, "stegastamp": 0.5}
GRID = 128
DECISION = 0.5
MAX_PARTS = 3
FILL_GRID = (0.6, 0.7, 0.8, 0.9, 1.01)
SCORE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
SEAM_W_GRID = (0.0,)
HF_W_GRID = (0.0,)
TARGET_GRID = ("soft", "p3", "h0.3", "h0.4", "h0.5", "h0.6", "h0.7")
HF_CLIP = (0.5, 3.0)
UNION_GRID = (1, MAX_PARTS)
SEED = 2026
BOOTSTRAP = 2000
N_TRAIN_TOTAL = 510


def miou(a: np.ndarray, b: np.ndarray) -> float:
    """Official two-class mIoU of boolean arrays (empty union counts 1)."""
    out = 0.0
    for pa, pb in ((a, b), (~a, ~b)):
        union = np.count_nonzero(pa | pb)
        out += 1.0 if union == 0 else np.count_nonzero(pa & pb) / union
    return out / 2


def head_maps() -> dict[str, np.ndarray]:
    """Train head maps (honest: heads trained on synthetic; SS = out-of-fold split7), by name."""
    meta = json.loads((TRAIN / "metadata.json").read_text())
    out: dict[str, np.ndarray] = {}
    for method in ("pixelseal", "trustmark"):
        names = [r["image_name"] for r in meta if r["method"] == method]
        p = np.mean([np.load(HEAD_CV / f"{h}.npz")[method].astype(np.float32) for h in HEADS[method]], axis=0)
        out.update(dict(zip(names, p, strict=True)))
    z = np.load(SS_SPLIT7)
    out.update({str(n): m.astype(np.float32) for n, m in zip(z["names_train"], z["train_finetuned_oof"], strict=True)})
    return out


def fill_of_bbox(m: np.ndarray) -> float:
    """Share of the bounding box covered by the mask (1 for an empty mask)."""
    if not m.any():
        return 1.0
    ys, xs = np.nonzero(m)
    return float(m.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)))


def load_candidates(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Unpack the SAM npz into per-image candidate arrays."""
    z = np.load(path)
    masks = np.unpackbits(z["packed"], axis=1).reshape(-1, GRID, GRID).astype(bool)
    out: dict[str, dict[str, np.ndarray]] = {}
    start = 0
    for name, k in zip(z["names"], z["counts"], strict=True):
        sl = slice(start, start + int(k))
        out[str(name)] = {"masks": masks[sl], "seam": z["seam"][sl], "hf": z["hf"][sl], "piou": z["pred_iou"][sl]}
        start += int(k)
    return out


def target(p: np.ndarray, kind: str) -> np.ndarray:
    """Map the candidates are matched to: soft head map, its cube, or the head mask at a threshold."""
    if kind == "soft":
        return p
    if kind == "p3":
        return p ** 3
    return (p >= float(kind[1:])).astype(np.float32)


def choose(p: np.ndarray, cand: dict[str, np.ndarray], seam_w: float, parts: int,
           hf_w: float = 0.0, kind: str = "soft") -> tuple[np.ndarray, float]:
    """Greedy union of candidates maximising soft IoU(mask, p) * seam ** seam_w * hf ** hf_w."""
    masks = cand["masks"]
    p = target(p, kind)
    if len(masks) == 0:
        return np.zeros((GRID, GRID), bool), 0.0
    seam = np.clip(cand["seam"], 1.0, None) ** seam_w * np.clip(cand["hf"], *HF_CLIP) ** hf_w
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    pf = p.reshape(-1)
    inter = flat @ pf
    union = flat.sum(1) + pf.sum() - inter
    score = inter / np.maximum(union, 1e-6) * seam
    best = int(np.argmax(score))
    cur, cur_s = masks[best].copy(), float(score[best])
    for _ in range(parts - 1):
        improved = False
        for j in np.argsort(-score)[:20]:
            trial = cur | masks[j]
            t = trial.reshape(-1)
            s = float((pf[t].sum()) / max(t.sum() + pf.sum() - pf[t].sum(), 1e-6) * seam[j])
            if s > cur_s + 1e-4:
                cur, cur_s, improved = trial, s, True
        if not improved:
            break
    return cur, cur_s


def main() -> None:
    """LOO evaluation of the SAM object selection against the deployed head mask."""
    cands = load_candidates(Path(sys.argv[1]))
    meta = {s["name"]: s for s in json.loads((TRAIN / "manifest.json").read_text())["samples"]}
    z = np.load(TRAIN / "masks_128.npz")
    gts = {str(n): m > 0 for n, m in zip(z["names"], z["masks"], strict=True)}
    maps = head_maps()
    names = [n for n in cands if n in maps]
    configs = list(itertools.product(FILL_GRID, SCORE_GRID, TARGET_GRID, UNION_GRID))
    base = np.zeros(len(names))
    oracle = np.zeros(len(names))
    table = np.zeros((len(configs), len(names)))
    methods = np.array([meta[n]["method"] for n in names])
    transplant = np.zeros(len(names), bool)
    chosen_cache: dict[tuple[str, int], list[tuple[np.ndarray, float]]] = {}
    for kind, parts in itertools.product(TARGET_GRID, UNION_GRID):
        chosen_cache[(kind, parts)] = [choose(maps[n], cands[n], 0.0, parts, 0.0, kind) for n in names]
    for i, n in enumerate(names):
        p, g = maps[n], gts[n]
        dep = p >= RECT_THRESHOLD[meta[n]["method"]]
        base[i] = miou(dep, g)
        gfill = fill_of_bbox(g if g.mean() < DECISION else ~g)
        transplant[i] = meta[n]["group"] == "rect_watermark" and gfill < 0.98
        ms = cands[n]["masks"]
        oracle[i] = max([miou(m, g) for m in ms], default=base[i])
        pred_rect = (p >= DECISION).mean() < DECISION
        fill = fill_of_bbox(dep)
        for c, (ft, st, kind, parts) in enumerate(configs):
            m, s = chosen_cache[(kind, parts)][i]
            use = pred_rect and fill < ft and s >= st and m.any()
            table[c, i] = miou(m, g) if use else base[i]
    loo = np.empty(len(names))
    loo_cfg = np.empty(len(names), dtype=int)
    for i in range(len(names)):
        keep = (np.arange(len(names)) != i) & (methods == methods[i])
        c = int(np.argmax(table[:, keep].sum(1)))
        loo[i] = table[c, i]
        loo_cfg[i] = c
    d = loo - base
    rng = np.random.default_rng(SEED)
    bt = [d[rng.integers(0, len(d), len(d))].sum() / N_TRAIN_TOTAL for _ in range(BOOTSTRAP)]
    rng = np.random.default_rng(SEED)
    best = int(np.argmax(table.sum(1)))
    best_m = {m: int(np.argmax(table[:, methods == m].sum(1))) for m in ("pixelseal", "trustmark", "stegastamp")}
    print(f"images {len(names)} (transplant {transplant.sum()}), per-method in-sample configs "
          f"{ {m: configs[c] for m, c in best_m.items()} }")
    for m in ("pixelseal", "trustmark", "stegastamp"):
        k = methods == m
        dm = d[k]
        btm = [dm[rng.integers(0, len(dm), len(dm))].sum() / N_TRAIN_TOTAL for _ in range(BOOTSTRAP)]
        print(f"  {m}: LOO delta on the 510 score {dm.sum() / N_TRAIN_TOTAL:+.4f} "
              f"[{np.percentile(btm, 2.5):+.4f};{np.percentile(btm, 97.5):+.4f}], changed (LOO) "
              f"{int((np.abs(loo[k] - base[k]) > 0).sum())}")
    print(f"delta on the 510-image score: LOO {d.sum() / N_TRAIN_TOTAL:+.4f} "
          f"[{np.percentile(bt, 2.5):+.4f};{np.percentile(bt, 97.5):+.4f}] | in-sample "
          f"{(table[best] - base).sum() / N_TRAIN_TOTAL:+.4f}")
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        for lab, sel in (("transplant", transplant), ("rectangle", ~transplant)):
            k = (methods == meth) & sel
            if k.any():
                print(f"  {meth:10s} {lab:10s} n={k.sum():3d} base {base[k].mean():.3f} LOO {loo[k].mean():.3f} "
                      f"in-sample {table[best][k].mean():.3f} SAM oracle {oracle[k].mean():.3f}")
    if len(sys.argv) > 2:
        export_train(Path(sys.argv[2]), names, methods, maps, configs, loo_cfg, chosen_cache)


def export_train(prefix: Path, names: list[str], methods: np.ndarray, maps: dict[str, np.ndarray],
                 configs: list[tuple[float, float, str, int]], loo_cfg: np.ndarray,
                 chosen_cache: dict[tuple[str, int], list[tuple[np.ndarray, float]]]) -> None:
    """Integrator format, one npz per method over its 170 train images: <method> = chosen mask,
    apply_<method> = LOO gate decision (False outside the SAM set)."""
    meta = json.loads((TRAIN / "metadata.json").read_text())
    pos = {n: i for i, n in enumerate(names)}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        all_names = [r["image_name"] for r in meta if r["method"] == method]
        out = np.zeros((len(all_names), GRID, GRID), np.uint8)
        apply = np.zeros(len(all_names), bool)
        for j, n in enumerate(all_names):
            if n not in pos:
                continue
            i = pos[n]
            ft, st, kind, parts = configs[int(loo_cfg[i])]
            m, s = chosen_cache[(kind, parts)][i]
            p = maps[n]
            gate = (p >= DECISION).mean() < DECISION and fill_of_bbox(p >= RECT_THRESHOLD[method]) < ft
            apply[j] = bool(gate and s >= st and m.any())
            out[j] = m
        tmp = prefix.parent / f"{prefix.name}_{method}_train.tmp.npz"
        np.savez_compressed(tmp, **{method: out, f"names_{method}": np.array(all_names), f"apply_{method}": apply})
        tmp.replace(prefix.parent / f"{prefix.name}_{method}_train.npz")
        print(f"export {method}: apply {int(apply.sum())}/{len(all_names)}")


if __name__ == "__main__":
    main()
