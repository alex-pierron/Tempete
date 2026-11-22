"""Probe: does decoding at the REAL pattern scale sharpen the per-position margin map?

The real pattern is magnified (~x1.08 for TrustMark and Pixel Seal, zoom probe). Shrinking the image by
1/k and padding with noise lets the decoder see the pattern at its native scale. The per-position
known-message margin map (kit ``features()['per_position_logits']``) is computed at z = 1 and at
z = 1/k, mapped back to the image frame, and scored by per-image ROC AUC against the official mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

SEED = 0
SIDE = 512
GRID = 128
PAD_NOISE_STD = 0.15


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def margin_map(wm: object, x: torch.Tensor, sign: torch.Tensor, z: float, gen: torch.Generator) -> np.ndarray:
    """Per-position margin at zoom z (z <= 1: shrink + noise pad), mapped to the 128 grid of the image."""
    side = round(SIDE * z)
    small = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False, antialias=True)
    canvas = (0.5 + torch.randn(x.shape, generator=gen) * PAD_NOISE_STD).clamp(0, 1)
    o = (SIDE - side) // 2
    canvas[..., o : o + side, o : o + side] = small
    per_pos = wm.features(canvas)["per_position_logits"].float().cpu()  # type: ignore[attr-defined]
    m = (per_pos * sign[None, :, None, None]).mean(dim=1, keepdim=True)
    up = F.interpolate(m, size=(SIDE, SIDE), mode="bilinear", align_corners=False)[..., o : o + side, o : o + side]
    return F.interpolate(up, size=(GRID, GRID), mode="bilinear", align_corners=False)[0, 0].numpy()


def main() -> None:
    """Per-image AUC of the margin map at z = 1 and z = 1 / k, by regime."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", default="trustmark")
    parser.add_argument("--k", type=float, default=1.08)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    wm = load_watermark(args.method, models_dir=args.models, device=args.device)
    meta = [m for m in json.loads((args.data / "metadata.json").read_text()) if m["method"] == args.method]
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    zooms = (1.0, 1.0 / args.k)
    for full in (True, False):
        group = [m for m in meta if bool(m["full_watermark"]) == full]
        aucs: list[list[float]] = [[], []]
        for i in rng.permutation(len(group))[: args.n]:
            item = group[int(i)]
            name = str(item["image_name"])
            a = np.asarray(Image.open(args.data / "images" / f"{name}.png").convert("RGB"), np.float32) / 255
            x = torch.from_numpy(a).permute(2, 0, 1)[None].contiguous()
            sign = 2 * torch.tensor([int(c) for c in str(item["message"])], dtype=torch.float32) - 1
            gt = masks[name].reshape(-1)
            for j, z in enumerate(zooms):
                aucs[j].append(roc_auc(margin_map(wm, x.to(args.device), sign, z, gen).reshape(-1), gt))
        print(args.method, "full" if full else "rect", {f"z={z:.3f}": round(float(np.mean(v)), 3)
                                                         for z, v in zip(zooms, aucs, strict=True)},
              "paired delta", round(float(np.mean(np.array(aucs[1]) - np.array(aucs[0]))), 3), flush=True)


if __name__ == "__main__":
    main()
