"""X-LOC-B (JURECA GPU): learned MAP DENOISER on the 128 grid (shape prior the thresholds do not capture).

Input per image (denoise_pack.py): the out-of-fold source map of its method (heads / split7), the e2e map (PS /
TM), the RGB thumbnail, the method one-hot and the regime proxy (share of cells >= 0.5). A small U-Net predicts
a correction of the input logit (last layer initialised at zero, so training starts from the input map).
Trained on the 510 OOF train maps, 5 folds stratified by (method, regime), dihedral augmentations, several
seeds; the OOF denoised maps are honest (the denoiser never saw the image, and its inputs are OOF too).

Outputs: <prefix>_oof.npz and <prefix>_test.npz, keys <method> [n, 128, 128] float16 + names_<method>
(train / test metadata order of each method), and <prefix>.json (per-method OOF mIoU @0.5, input vs denoised).

Usage: python denoise.py <pack.npz> <prefix>
Env: XLB_DN_EPOCHS (60), XLB_DN_SEEDS (3), XLB_DN_LR (1e-3), XLB_DN_WIDTH (32), XLB_DN_RGB (1), XLB_LIMIT.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
FOLDS = 5
EPOCHS = int(os.environ.get("XLB_DN_EPOCHS", "40"))
SEEDS = int(os.environ.get("XLB_DN_SEEDS", "1"))
FIRST_SEED = int(os.environ.get("XLB_DN_FIRST_SEED", "0"))
LR = float(os.environ.get("XLB_DN_LR", "1e-3"))
WIDTH = int(os.environ.get("XLB_DN_WIDTH", "32"))
USE_RGB = os.environ.get("XLB_DN_RGB", "1") == "1"
USE_EXTRA = os.environ.get("XLB_DN_EXTRA", "1") == "1"
ONLY = os.environ.get("XLB_DN_ONLY", "")
"""Comma-separated methods to train and predict on (empty = all three)."""
LIMIT = int(os.environ.get("XLB_LIMIT", "0"))
BATCH = 16
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 20
PROB_CLIP = 1e-3
SMOOTH = 1.0
THRESHOLD = 0.5
METHODS = ("pixelseal", "trustmark", "stegastamp")
E2E_METHODS = (0, 1)
GROUPS = 8


def set_seed(seed: int) -> None:
    """Seed numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logit(p: Tensor) -> Tensor:
    """Clipped logit."""
    p = p.clamp(PROB_CLIP, 1 - PROB_CLIP)
    return torch.log(p) - torch.log1p(-p)


def build_inputs(head: NDArray[Any], e2e: NDArray[Any], rgb: NDArray[np.uint8], method: NDArray[np.int64],
                 extra: NDArray[Any] | None = None) -> tuple[Tensor, Tensor]:
    """Input tensor [N, C, 128, 128] and base logit [N, 1, 128, 128] (mean of head and e2e where e2e exists)."""
    h = torch.from_numpy(head.astype(np.float32))[:, None]
    e = torch.from_numpy(e2e.astype(np.float32))
    e = e[:, None] if e.ndim == 3 else e
    has = torch.from_numpy(np.isin(method, E2E_METHODS).astype(np.float32))[:, None, None, None]
    onehot = functional.one_hot(torch.from_numpy(method), len(METHODS)).float()[:, :, None, None].expand(-1, -1, *h.shape[-2:])
    share = (h >= THRESHOLD).float().mean((2, 3), keepdim=True).expand_as(h)
    parts = [h, e * has, has.expand_as(h), onehot, share]
    if USE_RGB:
        parts.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
    if extra is not None:
        parts.append(torch.from_numpy(extra.astype(np.float32)))
    base_prob = torch.where(has > 0, (h + e.mean(1, keepdim=True)) / 2, h)
    return torch.cat(parts, 1), logit(base_prob)


def block(cin: int, cout: int) -> nn.Sequential:
    """Two 3 x 3 convs with GroupNorm and GELU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(GROUPS, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(GROUPS, cout), nn.GELU(),
    )


class Denoiser(nn.Module):
    """U-Net 128 -> 8 with a global branch; outputs a correction of the base logit."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        widths = [width, width * 2, width * 4, width * 8, width * 8]
        self.down = nn.ModuleList()
        cin = channels
        for w in widths:
            self.down.append(block(cin, w))
            cin = w
        self.global_branch = nn.Sequential(nn.Linear(widths[-1], widths[-1]), nn.GELU(), nn.Linear(widths[-1], widths[-1]))
        self.up = nn.ModuleList(block(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 1))
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, base: Tensor) -> Tensor:
        """Logits [N, 1, 128, 128]."""
        skips = []
        for index, layer in enumerate(self.down):
            x = layer(x if index == 0 else functional.max_pool2d(x, 2))
            skips.append(x)
        x = x + self.global_branch(x.mean((2, 3)))[:, :, None, None]
        for index in range(len(self.up) - 1, -1, -1):
            x = functional.interpolate(x, size=skips[index].shape[-2:], mode="bilinear", align_corners=False)
            x = self.up[index](torch.cat([x, skips[index]], 1))
        return base + self.head(x)


def loss_fn(logits: Tensor, target: Tensor) -> Tensor:
    """BCE + (1 - soft two-class per-image mIoU)."""
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    q, u = 1 - p, 1 - target
    miou = (((p * target).sum(dims) + SMOOTH) / ((p + target - p * target).sum(dims) + SMOOTH)
            + ((q * u).sum(dims) + SMOOTH) / ((q + u - q * u).sum(dims) + SMOOTH)) / 2
    return functional.binary_cross_entropy_with_logits(logits, target) + (1 - miou).mean()


def dihedral(x: Tensor, k: int) -> Tensor:
    """Element k of the dihedral group of the square on the last two axes."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def undo_dihedral(x: Tensor, k: int) -> Tensor:
    """Inverse of dihedral(., k)."""
    x = x.flip(-1) if k >= 4 else x
    return torch.rot90(x, -(k % 4), dims=(-2, -1))


def miou_batch(pred: NDArray[np.bool_], gt: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Official per-image two-class mIoU (empty union = 1)."""
    scores = []
    for p, g in ((pred, gt), (~pred, ~gt)):
        inter, union = (p & g).sum((1, 2)), (p | g).sum((1, 2))
        scores.append(np.where(union == 0, 1.0, inter / np.maximum(union, 1)))
    return (scores[0] + scores[1]) / 2


@torch.no_grad()
def predict(model: Denoiser, x: Tensor, base: Tensor) -> NDArray[np.float32]:
    """Probabilities with the 8-element dihedral TTA."""
    model.eval()
    out = []
    for start in range(0, x.shape[0], 64):
        xb, bb = x[start:start + 64].to(DEVICE), base[start:start + 64].to(DEVICE)
        acc = torch.zeros_like(bb)
        for k in range(8):
            acc += undo_dihedral(torch.sigmoid(model(dihedral(xb, k), dihedral(bb, k))), k)
        out.append((acc / 8)[:, 0].cpu().numpy())
    return np.concatenate(out)


def train_one(x: Tensor, base: Tensor, gt: Tensor, index: NDArray[np.int64], seed: int) -> Denoiser:
    """One denoiser on the images of index."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = Denoiser(x.shape[1], WIDTH).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = math.ceil(len(index) / BATCH) * EPOCHS
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    xd, bd, gd = x.to(DEVICE), base.to(DEVICE), gt.to(DEVICE)
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(index)
        for start in range(0, len(order), BATCH):
            chunk = torch.from_numpy(order[start:start + BATCH]).to(DEVICE)
            k = int(rng.integers(8))
            logits = model(dihedral(xd[chunk], k), dihedral(bd[chunk], k))
            loss = loss_fn(logits, dihedral(gd[chunk], k))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
    return model


class _Archive(dict):  # type: ignore[type-arg]
    """dict with the `files` attribute of an NpzFile (used after the method filter)."""

    @property
    def files(self) -> list[str]:
        return list(self.keys())


def make_folds(method: NDArray[np.int64], full: NDArray[np.bool_], seed: int) -> NDArray[np.int64]:
    """Folds stratified by (method, regime)."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(method), dtype=np.int64)
    for m in np.unique(method):
        for f in (True, False):
            idx = np.flatnonzero((method == m) & (full == f))
            fold[rng.permutation(idx)] = np.arange(len(idx)) % FOLDS
    return fold


def main() -> None:
    """Train SEEDS x FOLDS denoisers, write OOF and test maps."""
    pack_path, prefix = Path(sys.argv[1]), sys.argv[2]
    start = time.time()
    z = np.load(pack_path)
    keep: Any = slice(0, LIMIT) if LIMIT else slice(None)
    if ONLY:
        wanted = [METHODS.index(m) for m in ONLY.split(",")]
        train_keep = np.flatnonzero(np.isin(z["train_method"], wanted))
        test_keep = np.flatnonzero(np.isin(z["test_method"], wanted))
        z = {**{k: z[k][train_keep] for k in z.files if k.startswith("train_")},
             **{k: z[k][test_keep] for k in z.files if k.startswith("test_")}}
        z = _Archive(z)
    method, full = z["train_method"][keep], z["train_full"][keep].astype(bool)
    gt_np = z["train_gt"][keep] > 0
    has_extra = "train_extra" in z.files and USE_EXTRA
    x, base = build_inputs(z["train_head"][keep], z["train_e2e"][keep], z["train_rgb"][keep], method,
                           z["train_extra"][keep] if has_extra else None)
    xt, baset = build_inputs(z["test_head"][keep], z["test_e2e"][keep], z["test_rgb"][keep], z["test_method"][keep],
                             z["test_extra"][keep] if has_extra else None)
    print(f"inputs {tuple(x.shape)} extra={has_extra}", flush=True)
    gt = torch.from_numpy(gt_np.astype(np.float32))[:, None]
    oof_sum = np.zeros(gt_np.shape, dtype=np.float32)
    test_sum = np.zeros((len(xt), 128, 128), dtype=np.float32)
    for done, s in enumerate(range(FIRST_SEED, FIRST_SEED + SEEDS), start=1):
        fold = make_folds(method, full, SEED + s)
        for k in range(FOLDS):
            model = train_one(x, base, gt, np.flatnonzero(fold != k), SEED + 100 * s + k)
            held = np.flatnonzero(fold == k)
            oof_sum[held] += predict(model, x[held], base[held])
            test_sum += predict(model, xt, baset) / FOLDS
        print(f"seed {s} done [{time.time() - start:.0f} s]", flush=True)
        write_outputs(z, keep, prefix, oof_sum / done, test_sum / done, base, gt_np, method, full, done, start)


def write_outputs(z: Any, keep: slice, prefix: str, oof: NDArray[np.float32], test: NDArray[np.float32], base: Tensor,
                  gt_np: NDArray[np.bool_], method: NDArray[np.int64], full: NDArray[np.bool_], seeds_done: int,
                  start: float) -> None:
    """Write OOF / test maps and the summary (called after every seed, atomically)."""
    summary: dict[str, Any] = {"epochs": EPOCHS, "seeds_done": seeds_done, "first_seed": FIRST_SEED, "lr": LR,
                               "width": WIDTH, "rgb": USE_RGB, "extra": USE_EXTRA, "only": ONLY}
    base_prob = torch.sigmoid(base)[:, 0].numpy()
    names, test_names, test_method = z["train_names"][keep], z["test_names"][keep], z["test_method"][keep]
    oof_out: dict[str, NDArray[Any]] = {}
    test_out: dict[str, NDArray[Any]] = {}
    for m_id, m in enumerate(METHODS):
        sel = method == m_id
        if not sel.any():
            continue
        s_in = miou_batch(base_prob[sel] >= THRESHOLD, gt_np[sel])
        s_dn = miou_batch(oof[sel] >= THRESHOLD, gt_np[sel])
        summary[m] = {"input": float(s_in.mean()), "denoised": float(s_dn.mean()),
                      "denoised_full": float(s_dn[full[sel]].mean()), "denoised_rect": float(s_dn[~full[sel]].mean()),
                      "input_full": float(s_in[full[sel]].mean()), "input_rect": float(s_in[~full[sel]].mean())}
        print(f"{m}: input {s_in.mean():.4f} -> denoised {s_dn.mean():.4f} | full {s_in[full[sel]].mean():.3f} -> "
              f"{s_dn[full[sel]].mean():.3f} rect {s_in[~full[sel]].mean():.3f} -> {s_dn[~full[sel]].mean():.3f}", flush=True)
        oof_out[m], oof_out[f"names_{m}"] = oof[sel].astype(np.float16), names[sel]
        tsel = test_method == m_id
        test_out[m], test_out[f"names_{m}"] = test[tsel].astype(np.float16), test_names[tsel]

    for path, arrays in ((f"{prefix}_oof.npz", oof_out), (f"{prefix}_test.npz", test_out)):
        temporary = path.replace(".npz", ".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    summary["seconds"] = time.time() - start
    Path(f"{prefix}.json").write_text(json.dumps(summary, indent=1))
    print(f"written seeds={seeds_done} [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
