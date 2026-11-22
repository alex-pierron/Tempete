"""Probe: blur-baseline Integrated Gradients of the KNOWN-message margin as a per-pixel watermark map.

Mechanism. The decoders of Pixel Seal and TrustMark pool globally, so their margin
m(x) = mean_k (2 b_k - 1) logit_k(x) for the known message b is a sum of local contributions.
Integrated Gradients (Sundararajan et al. 2017) with a blurred baseline (Sturmfels et al. 2020;
Kapishnikov et al. 2019) distributes m(x) - m(blur(x)) over the pixels (completeness axiom): the
blur removes the high-frequency watermark, so a watermarked pixel receives positive credit, a clean
pixel only the noise of natural high frequencies. This is the per-pixel, differentiable counterpart
of the occlusion map that already helps on Pixel Seal (+0.034 on mapped images).

The probe prints the mean per-image ROC AUC (watermarked class) of the IG map, next to the AUC of
the base head's out-of-fold probabilities on the same images. Train split only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

GRID_SIZE = 128
BLUR_KERNEL = 7
SMOOTH_KERNEL = 5
SEED = 0


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def blur(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Box blur with reflect padding (the baseline without the high-frequency watermark)."""
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x, (pad,) * 4, mode="reflect"), kernel, stride=1)


def margin_fn(wm: object, method: str) -> object:
    """Differentiable logits of the official wrapper (its public methods run under no_grad)."""
    if method == "pixelseal":
        return lambda x: wm.det(wm._resize(x))[:, 1:]  # type: ignore[attr-defined]
    return lambda x: wm.net(wm._prep(x))  # type: ignore[attr-defined]


def blur_ig(logits_of: object, x: torch.Tensor, bits: torch.Tensor, steps: int) -> torch.Tensor:
    """IG attribution of the known-message margin, summed over channels, on the 128 grid."""
    base = blur(x, BLUR_KERNEL)
    sign = (2 * bits - 1).to(x.device)
    total = torch.zeros_like(x)
    for alpha in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:
        xa = (base + float(alpha) * (x - base)).detach().requires_grad_(True)
        m = (logits_of(xa)[0].float() * sign).mean()  # type: ignore[operator]
        (grad,) = torch.autograd.grad(m, xa)
        total += grad
    attribution = ((x - base) * total / steps).sum(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(attribution, GRID_SIZE)
    return F.avg_pool2d(pooled, SMOOTH_KERNEL, stride=1, padding=SMOOTH_KERNEL // 2, count_include_pad=False)


def main() -> None:
    """Score IG maps against the masks and against the base head's OOF probabilities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--method", default="pixelseal")
    parser.add_argument("--per-regime", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    torch.set_num_threads(4)
    wm = load_watermark(args.method, models_dir=args.models, device=args.device)
    logits_of = margin_fn(wm, args.method)
    meta = [m for m in json.loads((args.data / "metadata.json").read_text()) if m["method"] == args.method]
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    oof = np.load(args.oof)[args.method]
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    t0 = time.time()
    for full in (True, False):
        idx = [i for i, m in enumerate(meta) if bool(m["full_watermark"]) == full]
        for i in rng.permutation(idx)[: args.per_regime]:
            item = meta[int(i)]
            name = str(item["image_name"])
            arr = np.asarray(Image.open(args.data / "images" / f"{name}.png").convert("RGB"), np.float32) / 255
            x = torch.from_numpy(arr).permute(2, 0, 1)[None].contiguous().to(args.device)
            bits = torch.tensor([int(c) for c in str(item["message"])], dtype=torch.float32)
            ig = blur_ig(logits_of, x, bits, args.steps)[0, 0].detach().cpu().numpy()
            gt = masks[name].reshape(-1)
            rows.append({
                "name": name, "full": full, "area": float(gt.mean()),
                "auc_ig": roc_auc(ig.reshape(-1), gt), "auc_base": roc_auc(oof[int(i)].astype(np.float64).reshape(-1), gt),
            })
            print(json.dumps(rows[-1]), flush=True)
    summary = {
        regime: {k: float(np.mean([r[k] for r in rows if r["full"] == (regime == "full")])) for k in ("auc_ig", "auc_base")}
        for regime in ("full", "rect")
    }
    print(json.dumps({"method": args.method, "summary": summary, "seconds": round(time.time() - t0, 1)}))
    args.out.write_text(json.dumps({"rows": rows, "summary": summary}))


if __name__ == "__main__":
    main()
