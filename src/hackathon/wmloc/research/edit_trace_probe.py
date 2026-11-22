"""Probe: localise the EDIT instead of the watermark, through generative-model traces.

Hypothesis (forensics literature: Corvi et al. 2023 and Ricker et al. 2022 on the spectral peaks left
by latent-diffusion VAE decoders; Li and Huang 2019 on deep-inpainting localisation with high-pass
residuals): the clean areas of this dataset were produced by an inpainting or outpainting model, whose
decoder upsamples by powers of two and leaves periodic traces (period 8 for a latent-diffusion VAE)
in the high-pass residual. A watermarked area is the original photo plus a weak residual and carries
no such periodicity. If true, the periodic energy is a strong, method-agnostic evidence for
"clean = 0", independent of how weak the watermark is.

Per image and per period P, the local periodic energy is the variance across the P x P pixel phases
of the windowed mean of the high-pass residual. The probe prints the mean per-image ROC AUC of each
map against the EDITED class (1 - mask), per method and regime. Train split only.
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
PERIODS = (2, 4, 8)
WINDOW = 32
EPS = 1e-6


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def to_grid(x: np.ndarray) -> np.ndarray:
    """Average 4 x 4 cells to the 128 grid."""
    return x.reshape(GRID_SIZE, CELL, GRID_SIZE, CELL).mean(axis=(1, 3))


def periodic_energy(hp: np.ndarray, period: int, window: int) -> np.ndarray:
    """Variance across the period x period phases of the windowed phase-mean of ``hp``."""
    side = hp.shape[0]
    rows = np.arange(side)[:, None] % period
    cols = np.arange(side)[None, :] % period
    means = []
    for a in range(period):
        for b in range(period):
            ind = ((rows == a) & (cols == b)).astype(np.float64)
            means.append(uniform_filter(hp * ind, window) / (uniform_filter(ind, window) + EPS))
    stack = np.stack(means)
    return stack.var(axis=0)


def trace_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    """Periodic-energy maps (log, normalised by local high-pass energy) on the 128 grid."""
    y = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    hp = y - uniform_filter(y, 3, mode="reflect")
    local = uniform_filter(hp**2, WINDOW) + EPS
    out: dict[str, np.ndarray] = {"hp_energy": to_grid(np.log(local))}
    for period in PERIODS:
        energy = periodic_energy(hp, period, WINDOW)
        out[f"per{period}"] = to_grid(np.log(energy + EPS))
        out[f"per{period}_rel"] = to_grid(np.log(energy + EPS) - np.log(local))
    return out


def main() -> None:
    """Compute per-image AUCs (edited class) and print per (method, regime) means."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    meta = json.loads((args.data / "metadata.json").read_text())
    if args.limit:
        rng = np.random.default_rng(0)
        meta = [meta[i] for i in rng.permutation(len(meta))[: args.limit]]
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rows: list[dict[str, object]] = []
    for item in meta:
        name = str(item["image_name"])
        img = np.asarray(Image.open(args.data / "images" / f"{name}.png").convert("RGB"))
        edited = 1 - masks[name].astype(np.uint8).reshape(-1)
        aucs = {k: roc_auc(v.reshape(-1), edited) for k, v in trace_maps(img).items()}
        rows.append({"name": name, "method": item["method"], "full": bool(item["full_watermark"]), **aucs})
    keys = [k for k in rows[0] if k not in ("name", "method", "full")]
    summary: dict[str, dict[str, float]] = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        for full in (True, False):
            sel = [r for r in rows if r["method"] == method and r["full"] == full]
            if sel:
                summary[f"{method}_{'full' if full else 'rect'}"] = {
                    k: round(float(np.mean([float(r[k]) for r in sel])), 3) for k in keys  # type: ignore[arg-type]
                }
    for group, vals in summary.items():
        print(group, json.dumps(vals))
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}))


if __name__ == "__main__":
    main()
