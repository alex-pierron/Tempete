"""X-LOC-B (JURECA GPU): fine-tune the OFFICIAL extractor end to end for localization on the REAL train images.

Why: our PS / TM heads (0.586 / 0.583 out of fold) sit on FROZEN extractor features pooled to 8-32 cells,
and synthetic training data does not transfer to the real chain. Here the whole extractor (Pixel Seal
ConvNeXt-tiny + pixel decoder, or TrustMark ResNet-50) is trained on the 170 real images of its method with
their true masks, 5 folds, native 512 px input (stride-4 stage = the 128 evaluation grid), plus a FPN decoder.
The extractor's per-position bit logits are compared with the KNOWN message (agreement maps at the 512 pass
and at the native 256 / 245 pass), so fine-tuning can sharpen exactly the signal the network already has.

Outputs (one call = one method):
    <out>_oof.npz   key <method> [N, 128, 128] float16 out-of-fold probabilities (train metadata order),
                    names_<method>, fold
    <out>_test.npz  key <method> [M, 128, 128] float16 mean of the fold models, names_<method>
    <out>.json      settings, flip check, per-fold and overall scores, timings
Written after every fold, so a timeout keeps the finished folds.

Usage: python e2e_loc.py <method> <pack.npz> <out prefix>
Env: WMLOC_ROOT (models/), WMLOC_KIT (kit package `wmloc`), XLB_EPOCHS, XLB_LR_BACKBONE, XLB_LR_HEAD,
     XLB_FOLDS, XLB_BUDGET_SECONDS, XLB_FLIP (auto | on | off), XLB_SEED.
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

ROOT = Path(os.environ["WMLOC_ROOT"])
sys.path.insert(0, os.environ["WMLOC_KIT"])
from wmloc.watermarks import load_watermark  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("XLB_SEED", "2026"))
FOLDS = int(os.environ.get("XLB_FOLDS", "5"))
EPOCHS = int(os.environ.get("XLB_EPOCHS", "30"))
LR_BACKBONE = float(os.environ.get("XLB_LR_BACKBONE", "1e-4"))
LR_HEAD = float(os.environ.get("XLB_LR_HEAD", "1e-3"))
BUDGET_SECONDS = float(os.environ.get("XLB_BUDGET_SECONDS", "2100"))
FLIP_MODE = os.environ.get("XLB_FLIP", "auto")
FREEZE_STAGES = int(os.environ.get("XLB_FREEZE_STAGES", "0"))
"""Number of early backbone stages kept frozen (fewer free parameters on 136 images)."""
SYNTH_DIR = os.environ.get("XLB_SYNTH_DIR", "")
"""Folder with <method>/s_*.npz synthetic composites (gen_synth.py); empty = real images only."""
SYNTH_PER_BATCH = int(os.environ.get("XLB_SYNTH_PER_BATCH", "4"))
PSEUDO = os.environ.get("XLB_PSEUDO", "")
"""npz with test pseudo-labels (labels int8 [n, 128, 128], 1 / 0 / -1 = ignore, names): transductive self-training."""
PSEUDO_PER_BATCH = int(os.environ.get("XLB_PSEUDO_PER_BATCH", "4"))
B2_FOLDS = os.environ.get("XLB_B2_FOLDS", "0") == "1"
"""Use the b2 fold assignment (the synthetic samples carry the b2 fold of their shape's source image)."""
LIMIT = int(os.environ.get("XLB_LIMIT", "0"))
"""Smoke test only: keep the first LIMIT train and test images (0 = all)."""
BATCH = 8
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 30
IMAGE_SIDE = 512
GRID_SIDE = 128
CELL = IMAGE_SIDE // GRID_SIDE
CROP_SIDE = 384
CROP_PROBABILITY = 0.5
FLIP_PROBABILITY = 0.5
FLIP_MIN_BIT_ACCURACY = 0.8
"""Flips are used only if the pretrained extractor still decodes flipped full-regime images this well."""
DECODER_CHANNELS = 64
NORM_GROUPS = 8
SMOOTH = 1.0
DETECTION_SCALE = 5.0
MAGNITUDE_EPSILON = 1e-3
MARGIN_CLIP = 8.0
EVAL_EVERY = 10
PREDICT_BATCH = 16
THRESHOLD = 0.5
NATIVE_SIDE = {"pixelseal": 256, "trustmark": 245}


class E2EError(RuntimeError):
    """Unsupported method or inconsistent pack."""


def set_seed(seed: int) -> None:
    """Seed numpy and torch (CPU and CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Backbone(nn.Module):
    """Official extractor exposed as 4 stages (strides 4, 8, 16, 32) and per-position logits."""

    stage_channels: tuple[int, int, int, int]
    has_detection: bool

    def stages(self, images: Tensor) -> list[Tensor]:
        """Feature maps of the 4 stages for images in [0, 1]."""
        raise NotImplementedError

    def position_logits(self, deepest: Tensor) -> tuple[Tensor, Tensor | None]:
        """Per-position bit logits [B, nbits, h, w] and detection logit [B, 1, h, w] (or None)."""
        raise NotImplementedError


class PixelSealBackbone(Backbone):
    """ConvNeXt-tiny + pixel decoder of Pixel Seal; logits are linear, so their spatial mean is the decode."""

    stage_channels = (96, 192, 384, 768)
    has_detection = True

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.convnext = detector.convnext
        self.upscaling = detector.pixel_decoder.output_upscaling
        self.linear = detector.pixel_decoder.linear

    def stages(self, images: Tensor) -> list[Tensor]:
        x = images * 2 - 1
        out: list[Tensor] = []
        for index in range(4):
            x = self.convnext.downsample_layers[index](x)
            x = self.convnext.stages[index](x)
            out.append(x)
        return out

    def position_logits(self, deepest: Tensor) -> tuple[Tensor, Tensor | None]:
        embedding = self.upscaling(deepest)
        logits = torch.einsum("bchw,oc->bohw", embedding, self.linear.weight)
        logits = logits + self.linear.bias[None, :, None, None]
        return logits[:, 1:], logits[:, :1]


class TrustMarkBackbone(Backbone):
    """ResNet-50 decoder of TrustMark Q; BatchNorm statistics stay frozen (batches of 8 are too small)."""

    stage_channels = (256, 512, 1024, 2048)
    has_detection = False

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def train(self, mode: bool = True) -> TrustMarkBackbone:
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def stages(self, images: Tensor) -> list[Tensor]:
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(images * 2 - 1))))
        out: list[Tensor] = []
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, name)(x)
            out.append(x)
        return out

    def position_logits(self, deepest: Tensor) -> tuple[Tensor, Tensor | None]:
        logits = torch.einsum("bchw,oc->bohw", deepest, self.net.fc.weight)
        return logits + self.net.fc.bias[None, :, None, None], None


def build_backbone(method: str) -> Backbone:
    """Load the official extractor of the method (kit wrapper) as a trainable Backbone."""
    wm = load_watermark(method, str(ROOT / "models"), device="cpu")
    if method == "pixelseal":
        backbone: Backbone = PixelSealBackbone(wm.det)
    elif method == "trustmark":
        backbone = TrustMarkBackbone(wm.net)
    else:
        raise E2EError(f"unsupported method {method}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    for module in frozen_modules(backbone):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return backbone.float()


def frozen_modules(backbone: Backbone) -> list[nn.Module]:
    """The FREEZE_STAGES earliest stages (with their stem / downsampling layers)."""
    if FREEZE_STAGES <= 0:
        return []
    if isinstance(backbone, PixelSealBackbone):
        c = backbone.convnext
        return [m for i in range(FREEZE_STAGES) for m in (c.downsample_layers[i], c.stages[i])]
    n = backbone.net  # type: ignore[attr-defined]
    stem: list[nn.Module] = [n.conv1, n.bn1]
    return stem + [getattr(n, f"layer{i + 1}") for i in range(FREEZE_STAGES)]


def evidence_maps(backbone: Backbone, deepest: Tensor, signs: Tensor) -> tuple[Tensor, Tensor]:
    """Message-aware maps [B, 3 (+1), h, w] and global decode scalars [B, 2] of one pass."""
    bits, detection = backbone.position_logits(deepest)
    weighted = bits * signs[:, :, None, None]
    magnitude = bits.abs().mean(1, keepdim=True)
    correlation = weighted.mean(1, keepdim=True) / (magnitude + MAGNITUDE_EPSILON)
    hard = (weighted > 0).float().mean(1, keepdim=True)
    maps = [correlation, hard, torch.log1p(magnitude)]
    if detection is not None:
        maps.append(detection.clamp(-DETECTION_SCALE * 2, DETECTION_SCALE * 2) / DETECTION_SCALE)
    global_logits = weighted.mean((2, 3))
    scalars = torch.stack(
        [(global_logits > 0).float().mean(1), global_logits.mean(1).clamp(-MARGIN_CLIP, MARGIN_CLIP) / MARGIN_CLIP],
        dim=1,
    )
    return torch.cat(maps, 1), scalars


def conv_block(channels: int, dilation: int = 1) -> nn.Sequential:
    """3 x 3 conv, GroupNorm, GELU."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
        nn.GroupNorm(NORM_GROUPS, channels),
        nn.GELU(),
    )


class Localizer(nn.Module):
    """Fine-tuned extractor + FPN decoder fed with the agreement maps of the 512 and the native pass."""

    def __init__(self, backbone: Backbone, native_side: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.native_side = native_side
        d = DECODER_CHANNELS
        per_pass = 3 + (1 if backbone.has_detection else 0)
        evidence_channels = 2 * per_pass + 2
        self.lateral = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, d, 1), nn.GroupNorm(NORM_GROUPS, d)) for c in backbone.stage_channels
        )
        self.evidence = nn.Sequential(nn.Conv2d(evidence_channels, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.context = nn.Sequential(conv_block(d, 1), conv_block(d, 2), conv_block(d, 4))
        self.global_branch = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.fuse = nn.ModuleList(conv_block(d) for _ in range(3))
        self.head = nn.Conv2d(d, 1, 1)

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the new layers (everything but the backbone)."""
        own = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in own]

    def forward(self, images: Tensor, signs: Tensor) -> Tensor:
        """Logits on the stride-4 grid ([B, 1, H / 4, W / 4]) for images in [0, 1]."""
        stages = self.backbone.stages(images)
        deepest = stages[3]
        maps_full, _ = evidence_maps(self.backbone, deepest, signs)
        side = [max(32, round(s * self.native_side / IMAGE_SIDE)) for s in images.shape[-2:]]
        native = functional.interpolate(images, size=side, mode="bilinear", align_corners=False, antialias=True)
        maps_native, scalars = evidence_maps(self.backbone, self.backbone.stages(native)[3], signs)
        maps_native = functional.interpolate(maps_native, size=deepest.shape[-2:], mode="bilinear", align_corners=False)
        broadcast = scalars[:, :, None, None].expand(-1, -1, *deepest.shape[-2:])
        p = self.lateral[3](deepest) + self.evidence(torch.cat([maps_full, maps_native, broadcast], 1))
        p = p + self.context(p)
        p = p + self.global_branch(p.mean((2, 3)))[:, :, None, None]
        for level in (2, 1, 0):
            p = functional.interpolate(p, size=stages[level].shape[-2:], mode="bilinear", align_corners=False)
            p = self.fuse[level](p + self.lateral[level](stages[level]))
        return self.head(p)


def iou_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """BCE + (1 - soft two-class per-image mIoU) over the valid cells (label -1 = ignored, pseudo-labels)."""
    weight = (labels >= 0).float()
    target = labels.clamp(min=0)
    bce = (functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum()
    bce = bce / weight.sum().clamp(min=1)
    p = torch.sigmoid(logits) * weight
    target = target * weight
    dims = (1, 2, 3)
    inter1 = (p * target).sum(dims)
    union1 = (p + target - p * target).sum(dims)
    q, u = weight - p, weight - target
    inter0 = (q * u).sum(dims)
    union0 = (q + u - q * u).sum(dims)
    miou = ((inter1 + SMOOTH) / (union1 + SMOOTH) + (inter0 + SMOOTH) / (union0 + SMOOTH)) / 2
    return bce + (1 - miou).mean()


def miou_batch(pred: NDArray[np.bool_], gt: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Official per-image two-class mIoU, vectorised over the first axis (empty union = 1)."""
    axes = (1, 2)
    scores = []
    for p, g in ((pred, gt), (~pred, ~gt)):
        inter = (p & g).sum(axes)
        union = (p | g).sum(axes)
        scores.append(np.where(union == 0, 1.0, inter / np.maximum(union, 1)))
    return (scores[0] + scores[1]) / 2


def make_folds(full: NDArray[np.bool_], method: str = "", names: NDArray[Any] | None = None) -> NDArray[np.int64]:
    """Fold of every image, stratified by regime, fixed seed (or the b2 assignment when B2_FOLDS)."""
    if B2_FOLDS:
        from hackathon.wmloc.research.data import load_records
        from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, stratified_folds

        records = [r for r in load_records("train") if r.method == method]
        if names is None or [r.name for r in records] != [str(n) for n in names]:
            raise E2EError("b2 record order differs from the pack order")
        strata = [f"{method}/{bool(r.full_watermark)}" for r in records]
        return np.asarray(stratified_folds(strata, DEFAULT_FOLDS, DEFAULT_SEED), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(full), dtype=np.int64)
    for regime in (True, False):
        index = np.flatnonzero(full == regime)
        fold[rng.permutation(index)] = np.arange(len(index)) % FOLDS
    return fold


def to_float(images_uint8: Tensor) -> Tensor:
    """[B, H, W, 3] uint8 on device -> [B, 3, H, W] float in [0, 1]."""
    return images_uint8.permute(0, 3, 1, 2).float().div(255)


def augment_batch(
    images: Tensor, masks: Tensor, rng: np.random.Generator, flips: bool
) -> tuple[Tensor, Tensor]:
    """Random crop (same pixel scale, cell-aligned) and optional horizontal flips."""
    if rng.random() < CROP_PROBABILITY:
        crops_i, crops_m = [], []
        for k in range(images.shape[0]):
            top, left = (int(v) * CELL for v in rng.integers(0, (IMAGE_SIDE - CROP_SIDE) // CELL + 1, 2))
            crops_i.append(images[k, :, top:top + CROP_SIDE, left:left + CROP_SIDE])
            g0, g1 = top // CELL, left // CELL
            crops_m.append(masks[k, :, g0:g0 + CROP_SIDE // CELL, g1:g1 + CROP_SIDE // CELL])
        images, masks = torch.stack(crops_i), torch.stack(crops_m)
    if flips:
        flip = torch.from_numpy(rng.random(images.shape[0]) < FLIP_PROBABILITY).to(images.device)
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
        masks = torch.where(flip[:, None, None, None], masks.flip(-1), masks)
    return images, masks


@torch.no_grad()
def predict(model: Localizer, images: Tensor, signs: Tensor, flips: bool) -> NDArray[np.float32]:
    """Probabilities [N, 128, 128] (horizontal-flip TTA when flips are used)."""
    model.eval()
    out = []
    for start in range(0, images.shape[0], PREDICT_BATCH):
        x = to_float(images[start:start + PREDICT_BATCH])
        s = signs[start:start + PREDICT_BATCH]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            p = torch.sigmoid(model(x, s).float())
            if flips:
                p = (p + torch.sigmoid(model(x.flip(-1), s).float()).flip(-1)) / 2
        out.append(p[:, 0].float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def flip_check(method: str, images: Tensor, signs: Tensor, full: NDArray[np.bool_]) -> tuple[float, float]:
    """Bit accuracy of the pretrained extractor on full-regime images, straight and flipped (native pass)."""
    backbone = build_backbone(method).to(DEVICE).eval()
    index = torch.from_numpy(np.flatnonzero(full)).to(images.device)
    side = NATIVE_SIDE[method]
    accuracies = []
    for flipped in (False, True):
        hits = []
        for start in range(0, len(index), PREDICT_BATCH):
            chunk = index[start:start + PREDICT_BATCH]
            x = to_float(images[chunk])
            x = x.flip(-1) if flipped else x
            x = functional.interpolate(x, size=(side, side), mode="bilinear", align_corners=False, antialias=True)
            bits, _ = backbone.position_logits(backbone.stages(x)[3])
            hits.append(((bits.mean((2, 3)) * signs[chunk]) > 0).float().mean(1).cpu())
        accuracies.append(float(torch.cat(hits).mean()))
    del backbone
    torch.cuda.empty_cache()
    return accuracies[0], accuracies[1]


def train_fold(
    method: str, images: Tensor, masks: Tensor, signs: Tensor, train_index: NDArray[np.int64],
    eval_index: NDArray[np.int64], eval_gt: NDArray[np.bool_], flips: bool, fold: int, log: list[str],
    synth: tuple[Tensor, Tensor, Tensor] | None = None,
    pseudo: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Localizer:
    """Train one fold model; logs the held-out score every EVAL_EVERY epochs (information only)."""
    set_seed(SEED + fold)
    rng = np.random.default_rng(SEED + 100 + fold)
    model = Localizer(build_backbone(method), NATIVE_SIDE[method]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
            {"params": model.head_parameters(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    real_per_batch = BATCH - (SYNTH_PER_BATCH if synth is not None else 0) - (PSEUDO_PER_BATCH if pseudo is not None else 0)
    steps_per_epoch = math.ceil(len(train_index) / real_per_batch)
    total = steps_per_epoch * EPOCHS
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)),
    )
    eval_images = images[torch.from_numpy(eval_index).to(images.device)]
    eval_signs = signs[torch.from_numpy(eval_index).to(signs.device)]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(train_index)
        running = 0.0
        for start in range(0, len(order), real_per_batch):
            chunk = torch.from_numpy(order[start:start + real_per_batch]).to(images.device)
            x, m, s = to_float(images[chunk]), masks[chunk][:, None].float(), signs[chunk]
            if synth is not None:
                pick = torch.from_numpy(rng.integers(0, synth[0].shape[0], SYNTH_PER_BATCH)).to(images.device)
                x = torch.cat([x, to_float(synth[0][pick])])
                m = torch.cat([m, synth[1][pick][:, None].float()])
                s = torch.cat([s, synth[2][pick]])
            if pseudo is not None:
                pick = torch.from_numpy(rng.integers(0, pseudo[0].shape[0], PSEUDO_PER_BATCH)).to(images.device)
                x = torch.cat([x, to_float(pseudo[0][pick])])
                m = torch.cat([m, pseudo[1][pick][:, None].float()])
                s = torch.cat([s, pseudo[2][pick]])
            x, m = augment_batch(x, m, rng, flips)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(x, s)
            loss = iou_loss(logits.float(), m)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running += float(loss) * len(chunk)
        line = f"fold {fold} epoch {epoch} loss {running / len(order):.4f}"
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            p = predict(model, eval_images, eval_signs, flips)
            line += f" | held-out mIoU@0.5 {miou_batch(p >= THRESHOLD, eval_gt).mean():.4f}"
        print(line, flush=True)
        log.append(line)
    return model


def load_synth(method: str) -> tuple[Tensor, Tensor, Tensor, NDArray[np.int64]] | None:
    """Synthetic images, masks, signs (on DEVICE) and shape folds, or None without XLB_SYNTH_DIR."""
    if not SYNTH_DIR:
        return None
    files = sorted(Path(SYNTH_DIR, method).glob("s_*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    if not files:
        raise E2EError(f"no synthetic file in {SYNTH_DIR}/{method}")
    parts = [np.load(f) for f in files]
    images = torch.from_numpy(np.concatenate([z["images"] for z in parts])).to(DEVICE)
    masks = torch.from_numpy(np.concatenate([z["masks"] for z in parts]).astype(np.uint8)).to(DEVICE)
    signs = torch.from_numpy(np.concatenate([z["messages"] for z in parts]).astype(np.float32) * 2 - 1).to(DEVICE)
    folds = np.concatenate([z["fold"] for z in parts])
    full = np.concatenate([z["full"] for z in parts])
    print(f"synthetic {method}: {len(folds)} samples from {len(files)} files, full share {full.mean():.2f}", flush=True)
    return images, masks, signs, folds


def main() -> None:
    """Train FOLDS models of one method, write OOF and test maps after each fold."""
    method, pack_path, prefix = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if method not in NATIVE_SIDE:
        raise E2EError(f"unsupported method {method}")
    start = time.time()
    pack = np.load(pack_path)
    keep = slice(0, LIMIT) if LIMIT > 0 else slice(None)
    train_np, test_np = pack["train_images"][keep], pack["test_images"][keep]
    gt = pack["train_masks"][keep] > 0
    full = pack["train_full"][keep].astype(bool)
    names, test_names = pack["train_names"][keep], pack["test_names"][keep]
    images = torch.from_numpy(train_np).to(DEVICE)
    test_images = torch.from_numpy(test_np).to(DEVICE)
    masks = torch.from_numpy(gt.astype(np.uint8)).to(DEVICE)
    signs = torch.from_numpy(pack["train_messages"][keep].astype(np.float32) * 2 - 1).to(DEVICE)
    test_signs = torch.from_numpy(pack["test_messages"][keep].astype(np.float32) * 2 - 1).to(DEVICE)
    print(f"{method}: train {tuple(images.shape)} test {tuple(test_images.shape)} [{time.time() - start:.0f} s]", flush=True)
    straight, flipped = flip_check(method, images, signs, full)
    flips = FLIP_MODE == "on" or (FLIP_MODE == "auto" and flipped >= FLIP_MIN_BIT_ACCURACY)
    print(f"flip check: bit acc straight {straight:.3f} flipped {flipped:.3f} -> flips {flips}", flush=True)
    fold = make_folds(full, method, names)
    synth_all = load_synth(method)
    pseudo = None
    if PSEUDO:
        pz = np.load(PSEUDO)
        position = {str(n): i for i, n in enumerate(pz["names"])}
        missing = [str(n) for n in test_names if str(n) not in position]
        if missing:
            raise E2EError(f"pseudo-labels miss {len(missing)} test images")
        order = [position[str(n)] for n in test_names]
        labels = torch.from_numpy(pz["labels"][order].astype(np.int8)).to(DEVICE)
        pseudo = (test_images, labels, test_signs)
        print(f"pseudo-labels: {len(order)} test images, labelled cells {(labels >= 0).float().mean():.3f}, "
              f"positive {(labels == 1).float().mean():.3f}", flush=True)
    oof = np.full(gt.shape, np.nan, dtype=np.float32)
    test_sum = np.zeros((len(test_names), GRID_SIDE, GRID_SIDE), dtype=np.float32)
    log: list[str] = []
    summary: dict[str, Any] = {
        "method": method, "seed": SEED, "folds": FOLDS, "epochs": EPOCHS, "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD, "freeze_stages": FREEZE_STAGES, "synth_dir": SYNTH_DIR,
        "synth_per_batch": SYNTH_PER_BATCH, "b2_folds": B2_FOLDS, "pseudo": PSEUDO,
        "pseudo_per_batch": PSEUDO_PER_BATCH, "batch": BATCH, "crop": CROP_SIDE, "flip_check": [straight, flipped], "flips": flips,
        "fold_scores": [], "log": log,
    }
    done = 0
    for k in range(FOLDS):
        if time.time() - start > BUDGET_SECONDS:
            print(f"budget reached before fold {k}", flush=True)
            break
        eval_index = np.flatnonzero(fold == k)
        synth = None
        if synth_all is not None:
            keep_synth = torch.from_numpy(np.flatnonzero(synth_all[3] != k)).to(DEVICE)
            synth = (synth_all[0][keep_synth], synth_all[1][keep_synth], synth_all[2][keep_synth])
            print(f"fold {k}: {len(keep_synth)} synthetic samples", flush=True)
        model = train_fold(
            method, images, masks, signs, np.flatnonzero(fold != k), eval_index, gt[eval_index], flips, k, log, synth,
            pseudo,
        )
        del synth
        oof[eval_index] = predict(model, images[torch.from_numpy(eval_index).to(DEVICE)],
                                  signs[torch.from_numpy(eval_index).to(DEVICE)], flips)
        test_sum += predict(model, test_images, test_signs, flips)
        done += 1
        fold_score = float(miou_batch(oof[eval_index] >= THRESHOLD, gt[eval_index]).mean())
        summary["fold_scores"].append(fold_score)
        seen = ~np.isnan(oof[:, 0, 0])
        s = miou_batch(oof[seen] >= THRESHOLD, gt[seen])
        summary.update({
            "folds_done": done, "oof_images": int(seen.sum()), "oof_miou": float(s.mean()),
            "oof_full": float(s[full[seen]].mean()), "oof_rect": float(s[~full[seen]].mean()),
            "seconds": time.time() - start,
        })
        print(f"fold {k}: {fold_score:.4f} | OOF so far {s.mean():.4f} (full {summary['oof_full']:.4f} rect "
              f"{summary['oof_rect']:.4f}, n={int(seen.sum())}) [{time.time() - start:.0f} s]", flush=True)
        np.savez(f"{prefix}_oof.npz", **{method: np.nan_to_num(oof, nan=0.0).astype(np.float16),
                                          f"names_{method}": names, "fold": fold, "seen": seen})
        np.savez(f"{prefix}_test.npz", **{method: (test_sum / done).astype(np.float16), f"names_{method}": test_names})
        Path(f"{prefix}.json").write_text(json.dumps(summary, indent=1))
        del model
        torch.cuda.empty_cache()
    print(f"done {method} folds={done} [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
