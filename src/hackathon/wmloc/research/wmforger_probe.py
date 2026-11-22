"""Probe: does the WMForger preference model (Soucek et al., NeurIPS 2025) localise post-hoc watermarks?

Mechanism under test. WMForger trains a ConvNeXt "preference model" to rank clean images above images
carrying procedurally generated watermark-like artefacts; gradient ascent on its score removes the
watermark of unseen post-hoc schemes (black-box, transferable). Two per-pixel evidences follow:

* ``score_map``: the score head applied per position before the mean pooling (low = watermarked);
* ``delta_map``: local energy of the perturbation found by the optimisation (high = watermarked).

The probe reports, per method and regime, the mean per-image ROC AUC of each map against the official
128 x 128 ground truth, plus an out-of-fold mIoU of a fraction-of-area threshold rule. Nothing here
touches the test split.
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

GRID_SIZE = 128
WORK_RESOLUTION = 768  # resolution used by the authors' optimize_image.py
DEFAULT_STEPS = 32
DEFAULT_LR = 0.2
SMOOTH_KERNEL = 5
SEED = 0


def load_preference_model(repo_dir: Path, ckpt: Path, device: str) -> torch.nn.Module:
    """Build the WMForger ConvNeXt preference model and load its public checkpoint."""
    sys.path.insert(0, str(repo_dir))
    import omegaconf
    from wmforger.models import build_extractor

    params = omegaconf.OmegaConf.load(repo_dir / "configs" / "extractor.yaml")["convnext_tiny"]
    model = build_extractor("convnext_tiny", params, img_size=256, nbits=0)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")["model"]
    model.load_state_dict(state)
    return model.eval().to(device)


def score_map(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """Per-position preference score (pre-pooling), resized to the 128 grid. Low = watermarked."""
    with torch.no_grad():
        lat = model.convnext(model.preprocess(img))
        emb = model.pixel_decoder.output_upscaling(lat)
        lin = model.pixel_decoder.linear
        per_pos = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
    return F.interpolate(per_pos[:, :1], size=(GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=False)


def delta_map(model: torch.nn.Module, img: torch.Tensor, steps: int, lr: float) -> torch.Tensor:
    """Energy of the WMForger removal perturbation, pooled to the 128 grid. High = watermarked."""
    img = img.contiguous()
    param = torch.zeros(img.shape, device=img.device, requires_grad=True)
    optim = torch.optim.SGD([param], lr=lr)
    for _ in range(steps):
        optim.zero_grad()
        loss = -model((img + param).clamp(0, 1)).mean()
        loss.backward()
        optim.step()
    energy = param.detach().pow(2).sum(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(energy, GRID_SIZE)
    return F.avg_pool2d(pooled, SMOOTH_KERNEL, stride=1, padding=SMOOTH_KERNEL // 2, count_include_pad=False)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC; 0.5 when one class is absent."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    """Run the probe on a train subset and write one JSON with per-image results and maps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--resolution", type=int, default=WORK_RESOLUTION)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--all", action="store_true", help="every train image instead of a subset")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    meta = json.loads((args.data / "metadata.json").read_text())
    packed = np.load(args.data / "masks_128.npz")
    masks = {str(n): m for n, m in zip(packed["names"], packed["masks"], strict=True)}
    rng = np.random.default_rng(SEED)
    chosen: list[dict[str, object]] = []
    for method in ("pixelseal", "trustmark", "stegastamp"):
        for full in (True, False):
            group = [m for m in meta if m["method"] == method and bool(m["full_watermark"]) == full]
            idx = np.arange(len(group)) if args.all else rng.permutation(len(group))[: args.per_group]
            chosen += [group[i] for i in idx]

    model = load_preference_model(args.repo, args.ckpt, args.device)
    rows: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {}
    t0 = time.time()
    for item in chosen:
        name = str(item["image_name"])
        pil = Image.open(args.data / "images" / f"{name}.png").convert("RGB")
        x = torch.from_numpy(np.asarray(pil, np.float32) / 255).permute(2, 0, 1)[None].to(args.device)
        x = F.interpolate(x.contiguous(), size=(args.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True).contiguous()
        gt = masks[name].astype(np.uint8).reshape(-1)
        s_map = score_map(model, x)[0, 0].float().cpu().numpy()
        if args.steps > 0:
            d_map = delta_map(model, x, args.steps, args.lr)[0, 0].float().cpu().numpy()
        else:
            d_map = np.zeros_like(s_map)
        maps[f"s_{name}"] = s_map.astype(np.float16)
        maps[f"d_{name}"] = d_map.astype(np.float16)
        rows.append({
            "name": name, "method": item["method"], "full": bool(item["full_watermark"]),
            "area": float(gt.mean()),
            "auc_score": roc_auc(-s_map.reshape(-1), gt),
            "auc_delta": roc_auc(d_map.reshape(-1), gt),
        })
        print(json.dumps(rows[-1]), flush=True)
    elapsed = time.time() - t0
    summary: dict[str, dict[str, float]] = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        for full in (True, False):
            sel = [r for r in rows if r["method"] == method and r["full"] == full]
            if sel:
                summary[f"{method}_{'full' if full else 'rect'}"] = {
                    "n": len(sel),
                    "auc_score": float(np.mean([r["auc_score"] for r in sel])),
                    "auc_delta": float(np.mean([r["auc_delta"] for r in sel])),
                }
    print(json.dumps({"summary": summary, "seconds": round(elapsed, 1)}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **maps)
    args.out.write_text(json.dumps({"rows": rows, "summary": summary, "seconds": elapsed}))


if __name__ == "__main__":
    main()
