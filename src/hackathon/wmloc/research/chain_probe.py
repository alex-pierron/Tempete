"""Probe: does ``embed()`` reproduce the REAL Pixel Seal residual of the dataset (subtraction test)?

Pixel Seal is deterministic given (cover, message): delta = 0.2 * JND(x) * up(E(Y(down256(x)), m)).
Since the residual is small, delta(x_w) ~ delta(x0). If the dataset chain is ``embed`` on the
512 x 512 image, subtracting the re-embedding residual from a watermarked area REMOVES the watermark:
the decoder margin of ``x - delta'`` falls to ~0 while ``x + delta'`` doubles it. A chain mismatch
(other resolution, other crop, other strength) leaves the margin of ``x - delta'`` high. Hypotheses
are scored on real full-regime train images; a synthetic control (COCO cover embedded at 512) gives
the expected values when the chain matches.

Margin = mean over bits of logit * (2 * bit - 1), computed on the watermarked cells only via the
masked composite (hole pixels kept from x, so only the watermarked area changes).
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


def margin(wm: object, img: torch.Tensor, bits: torch.Tensor) -> float:
    """Mean signed logit of the known message (> 0: watermark present)."""
    logits = wm.logits(img)  # type: ignore[attr-defined]
    return float((logits.cpu() * (2 * bits - 1)).mean())


def reembed_residual(wm: object, img: torch.Tensor, bits: torch.Tensor, scale: int) -> torch.Tensor:
    """Float residual of re-embedding, with the cover optionally rescaled by ``scale`` before embedding."""
    model = wm.model  # type: ignore[attr-defined]
    x = img
    if scale != 1:
        x = F.interpolate(img, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)
    with torch.no_grad():
        out = model.embed(x, msgs=bits[None].to(x.device), is_video=False)
    res = out["imgs_w"] - x
    if scale != 1:
        res = F.interpolate(res, size=img.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
    return res


def main() -> None:
    """Run the subtraction test for several chain hypotheses and print a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    from wmloc.watermarks import load_watermark

    wm = load_watermark("pixelseal", models_dir=args.models, device=args.device)
    meta = [m for m in json.loads((args.data / "metadata.json").read_text()) if m["method"] == "pixelseal"]
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    full = [m for m in meta if m["full_watermark"]]
    picks = [full[i] for i in rng.permutation(len(full))[: args.n]]

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None].contiguous().to(args.device)

    results: list[dict[str, object]] = []
    for item in picks:
        name = str(item["image_name"])
        x = to_t(np.asarray(Image.open(args.data / "images" / f"{name}.png").convert("RGB")))
        bits = torch.tensor([int(c) for c in str(item["message"])], dtype=torch.float32)
        m128 = torch.from_numpy(masks[name].astype(np.float32))[None, None].to(args.device)
        mask = F.interpolate(m128, size=(SIDE, SIDE), mode="nearest")
        row: dict[str, object] = {"name": name, "area": float(m128.mean()), "m_x": margin(wm, x, bits)}
        for scale in (1, 2):
            res = reembed_residual(wm, x, bits, scale) * mask
            row[f"m_minus_s{scale}"] = margin(wm, (x - res).clamp(0, 1), bits)
            row[f"m_plus_s{scale}"] = margin(wm, (x + res).clamp(0, 1), bits)
        results.append(row)
        print(json.dumps(row), flush=True)

    controls: list[dict[str, object]] = []
    for path in sorted(args.coco.iterdir())[:3]:
        cover = np.asarray(Image.open(path).convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC))
        bits = torch.from_numpy(rng.integers(0, 2, 256).astype(np.float32))
        xw = to_t(wm.embed(cover, bits.int().tolist()))
        row = {"name": path.name, "m_x": margin(wm, xw, bits)}
        for scale in (1, 2):
            res = reembed_residual(wm, xw, bits, scale)
            row[f"m_minus_s{scale}"] = margin(wm, (xw - res).clamp(0, 1), bits)
            row[f"m_plus_s{scale}"] = margin(wm, (xw + res).clamp(0, 1), bits)
        controls.append(row)
        print("control", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
