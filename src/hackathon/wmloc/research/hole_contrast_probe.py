"""Probe: are the real full-regime HOLES original content or smoother generated content?

Statistic: per-image ROC AUC of the local log high-pass energy against the edited class (1 - mask).
Real full-regime images: measured directly. Synthetic composites: the SAME real masks applied to
clean covers (rect-regime train images of the other methods), watermarked by ``embed`` at 512 inside
the mask, with the hole either (a) the clean original cover (the current synthetic recipe) or
(b) filled by OpenCV Telea inpainting (a smooth generated fill). If the real AUC matches (b) and not
(a), the synthetic recipe must inpaint the hole instead of restoring the original.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata

GRID_SIZE = 128
CELL = 4
WINDOW = 17
EPS = 1e-3
SEED = 0
INPAINT_RADIUS = 5


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def hp_energy_grid(img: np.ndarray) -> np.ndarray:
    """Local log high-pass energy of the luma on the 128 grid."""
    y = img.astype(np.float64).mean(axis=2)
    hp = y - uniform_filter(y, 3, mode="reflect")
    e = np.log(uniform_filter(hp**2, WINDOW) + EPS)
    return e.reshape(GRID_SIZE, CELL, GRID_SIZE, CELL).mean(axis=(1, 3))


def main() -> None:
    """Compare the hole contrast of real images with two synthetic hole recipes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", default="pixelseal")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    wm = load_watermark(args.method, models_dir=args.models, device=args.device)
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m.astype(np.uint8) for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    load = lambda n: np.asarray(Image.open(args.data / "images" / f"{n}.png").convert("RGB"))
    real = [m for m in meta if m["method"] == args.method and m["full_watermark"]]
    real = [real[i] for i in rng.permutation(len(real))[: args.n]]
    covers = [m for m in meta if m["method"] != args.method and not m["full_watermark"]]
    covers = [covers[i] for i in rng.permutation(len(covers))[: args.n]]
    out: dict[str, list[float]] = {"real": [], "syn_original_hole": [], "syn_inpainted_hole": []}
    for item, cover_item in zip(real, covers, strict=True):
        name = str(item["image_name"])
        m128 = masks[name]
        edited = 1 - m128.reshape(-1)
        out["real"].append(roc_auc(hp_energy_grid(load(name)).reshape(-1), edited))
        cover = load(str(cover_item["image_name"]))
        bits = [int(c) for c in rng.integers(0, 2, wm.nbits)]
        marked = np.asarray(wm.embed(cover, bits), dtype=np.uint8)
        m512 = np.kron(m128, np.ones((CELL, CELL), np.uint8)).astype(bool)[..., None]
        comp_a = np.where(m512, marked, cover)
        hole = (~m512[..., 0]).astype(np.uint8) * 255
        filled = cv2.inpaint(np.ascontiguousarray(marked[..., ::-1]), hole, INPAINT_RADIUS, cv2.INPAINT_TELEA)[..., ::-1]
        comp_b = np.where(m512, marked, filled)
        out["syn_original_hole"].append(roc_auc(hp_energy_grid(comp_a).reshape(-1), edited))
        out["syn_inpainted_hole"].append(roc_auc(hp_energy_grid(comp_b).reshape(-1), edited))
    summary = {k: round(float(np.mean(v)), 3) for k, v in out.items()}
    print(args.method, json.dumps(summary), flush=True)
    args.out.write_text(json.dumps({"summary": summary, "values": out}))


if __name__ == "__main__":
    main()
