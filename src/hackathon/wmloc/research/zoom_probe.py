"""Probe: decoding margin versus zoom, real full-regime images vs ``embed()`` at 512.

T_z(x): resize the 512 image by z (bilinear, antialias), then centre-crop (z > 1) or pad with
mid-grey (z < 1) back to 512, then decode with the official wrapper. If the real watermark pattern
is magnified by k relative to ``embed()`` at 512 (the 512 image being a crop of a larger watermarked
original), the real margin peaks at z_real ~ z_syn / k. Prints the mean curves and the per-image
argmax for real and synthetic (covers: rect-regime train images of the other methods, composited
with the real masks).
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

SEED = 0
SIDE = 512
FULL_AREA_MIN = 0.9
ZOOMS = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4)
PAD_VALUE = 0.5
PAD_NOISE_STD = 0.15
PAD_GENERATOR = torch.Generator().manual_seed(SEED)
PAD_MODE = {"mode": "grey"}  # x-loc-unet: a uniform grey band alone drops the real PS margin


def zoom(x: torch.Tensor, z: float) -> torch.Tensor:
    """Resize by z then centre-crop or pad back to SIDE."""
    side = round(SIDE * z)
    y = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False, antialias=True)
    if side >= SIDE:
        o = (side - SIDE) // 2
        return y[..., o : o + SIDE, o : o + SIDE]
    canvas = torch.full_like(x, PAD_VALUE)
    if PAD_MODE["mode"] == "noise":
        noise = torch.randn(x.shape, generator=PAD_GENERATOR) * PAD_NOISE_STD
        canvas = (canvas + noise.to(x.device)).clamp(0, 1)
    o = (SIDE - side) // 2
    canvas[..., o : o + side, o : o + side] = y
    return canvas


def main() -> None:
    """Margin-vs-zoom curves for real and synthetic full-regime images of one method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", default="pixelseal")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pad", choices=("grey", "noise"), default="grey")
    args = parser.parse_args()
    PAD_MODE["mode"] = args.pad
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    wm = load_watermark(args.method, models_dir=args.models, device=args.device)
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m.astype(np.uint8) for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    load = lambda n: np.asarray(Image.open(args.data / "images" / f"{n}.png").convert("RGB"))
    to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].contiguous()
    real = [m for m in meta if m["method"] == args.method and m["full_watermark"]
            and masks[str(m["image_name"])].mean() >= FULL_AREA_MIN]
    real = [real[i] for i in rng.permutation(len(real))[: args.n]]
    covers = [m for m in meta if m["method"] != args.method and not m["full_watermark"]]
    covers = [covers[i] for i in rng.permutation(len(covers))[: args.n]]

    def curve(img: np.ndarray, bits: list[int]) -> list[float]:
        sign = 2 * torch.tensor(bits, dtype=torch.float32) - 1
        x = to_t(img)
        batch = torch.cat([zoom(x, z) for z in ZOOMS])
        logits = wm.logits(batch.to(args.device) if args.method != "stegastamp" else batch).cpu()
        return [float(v) for v in (logits * sign).mean(dim=1)]

    real_curves, syn_curves = [], []
    for item, cover_item in zip(real, covers, strict=True):
        bits = [int(c) for c in str(item["message"])]
        real_curves.append(curve(load(str(item["image_name"])), bits))
        cover = load(str(cover_item["image_name"]))
        sbits = [int(c) for c in rng.integers(0, 2, wm.nbits)]
        marked = np.asarray(wm.embed(cover, sbits), dtype=np.uint8)
        m512 = np.kron(masks[str(item["image_name"])], np.ones((4, 4), np.uint8)).astype(bool)[..., None]
        syn_curves.append(curve(np.where(m512, marked, cover), sbits))
    r, s = np.array(real_curves), np.array(syn_curves)
    rn, sn = r / r.max(axis=1, keepdims=True), s / s.max(axis=1, keepdims=True)
    report = {
        "method": args.method, "zooms": ZOOMS,
        "real_mean_norm": [round(float(v), 3) for v in rn.mean(0)],
        "syn_mean_norm": [round(float(v), 3) for v in sn.mean(0)],
        "real_argmax": [ZOOMS[i] for i in r.argmax(1)],
        "syn_argmax": [ZOOMS[i] for i in s.argmax(1)],
        "real_mean_margin": [round(float(v), 2) for v in r.mean(0)],
        "syn_mean_margin": [round(float(v), 2) for v in s.mean(0)],
    }
    for k, v in report.items():
        print(k, v, flush=True)
    args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
