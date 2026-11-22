"""Probe: Pixel Seal writes an ACHROMATIC residual; does the luma/chroma high-pass energy ratio localise it?

Mechanism (videoseal code, card ``pixelseal``: embedder ``unet_base_yuv_quant`` with in/out channels 1,
JND ``jnd_1_1``, additive blending): the residual is predicted on the Y channel only and broadcast to
R, G and B, so after rounding on an integer cover the per-pixel change is identical in the three
channels. Chroma differences (R - G, B - G) are untouched, the luma gains the watermark energy. In a
watermarked area the local ratio E[hp(luma)^2] / E[hp(chroma)^2] therefore rises above the natural one.

The probe prints the mean per-image ROC AUC of several variants of that ratio against the official
128 x 128 mask, per method and regime. Pure numpy/scipy, no model, train split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata

GRID_SIZE = 128
CELL = 4
EPS = 1e-3
WINDOWS = (9, 17, 33)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def highpass(x: np.ndarray, size: int) -> np.ndarray:
    """Image minus its box blur."""
    return x - uniform_filter(x, size=size, mode="reflect")


def to_grid(x: np.ndarray) -> np.ndarray:
    """Average 4 x 4 cells to the 128 grid."""
    return x.reshape(GRID_SIZE, CELL, GRID_SIZE, CELL).mean(axis=(1, 3))


def ratio_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    """Log luma/chroma high-pass energy ratios at several window sizes (128 x 128 each)."""
    r, g, b = (img[..., c].astype(np.float64) for c in range(3))
    out: dict[str, np.ndarray] = {}
    for hp_size in (3, 5):
        lum = highpass((r + g + b) / 3, hp_size)
        c1 = highpass(r - g, hp_size)
        c2 = highpass(b - g, hp_size)
        e_l, e_c = lum**2, c1**2 + c2**2
        for win in WINDOWS:
            out[f"hp{hp_size}_w{win}"] = to_grid(
                np.log(uniform_filter(e_l, win) + EPS) - np.log(uniform_filter(e_c, win) + EPS)
            )
        out[f"hp{hp_size}_lum"] = to_grid(np.log(uniform_filter(e_l, 17) + EPS))
    return out


def main() -> None:
    """Compute per-image AUCs and print per (method, regime) means."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rows: list[dict[str, object]] = []
    for item in meta:
        name = str(item["image_name"])
        img = np.asarray(Image.open(args.data / "images" / f"{name}.png").convert("RGB"))
        gt = masks[name].astype(np.uint8).reshape(-1)
        aucs = {k: roc_auc(v.reshape(-1), gt) for k, v in ratio_maps(img).items()}
        rows.append({"name": name, "method": item["method"], "full": bool(item["full_watermark"]), **aucs})
    keys = [k for k in rows[0] if k not in ("name", "method", "full")]
    summary: dict[str, dict[str, float]] = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        for full in (True, False):
            sel = [r for r in rows if r["method"] == method and r["full"] == full]
            summary[f"{method}_{'full' if full else 'rect'}"] = {
                k: round(float(np.mean([float(r[k]) for r in sel])), 3) for k in keys  # type: ignore[arg-type]
            }
    for group, vals in summary.items():
        print(group, json.dumps(vals))
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}))


if __name__ == "__main__":
    main()
