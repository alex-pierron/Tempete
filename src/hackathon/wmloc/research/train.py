"""Training, zero-shot evaluation and k-fold fine-tuning of the learned localizer.

Sub-commands (run in the kit venv through `reports/phase1/wmloc/team-b/scripts/run_kit.sh`):
- `pretrain`: train on a synthetic set (`synth_data.py`), monitor a held-out synthetic part.
- `zeroshot`: evaluate a synthetic-only checkpoint on the whole real train split (no real label
  used for training: this is a clean estimate of the synthetic -> real transfer).
- `crossval`: k-fold on the real train split; each fold fine-tunes the pretrained checkpoint
  (or trains from scratch with `--scratch`) on k-1 blocks mixed with synthetic data and scores the
  held-out block. Reports mean and standard deviation over folds.

The score is the official per-image mIoU (classes 0 and 1, empty union counts 1), recomputed here
in vectorised form and unit-tested against the kit's `utils.miou_image`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.research.kit import METHODS
from hackathon.wmloc.research.model import Localizer, count_parameters

DEFAULT_SEED = 2026
BATCH_SIZE = 8
CPU_THREADS = 2
"""Shared machine: cap intra-op CPU threads (coordinator rule)."""
torch.set_num_threads(CPU_THREADS)
PRETRAIN_LR = 2e-3
FINETUNE_LR = 4e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0
EVAL_BATCH = 16
DECISION_THRESHOLD = 0.5
SYNTH_VALIDATION_FRACTION = 0.1
SMOOTH = 1e-6
DEFAULT_FOLDS = 5
SMALL_AREA_FRACTION = 0.03
LARGE_AREA_FRACTION = 0.15

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


@dataclass
class ArrayData:
    """Arrays of one (split, method): images and extras always, masks when labelled."""

    images: NDArray[np.uint8]
    maps: NDArray[np.float16]
    scalars: NDArray[np.float32]
    masks: NDArray[np.uint8] | None
    full: NDArray[np.bool_] | None

    def __len__(self) -> int:
        return int(self.images.shape[0])


def load_arrays(folder: Path, mmap: bool = True) -> ArrayData:
    """Load a folder written by `synth_data.py` or `real_data.py`."""
    mode: Literal["r"] | None = "r" if mmap else None
    masks_path = folder / "masks128.npy"
    full_path = folder / "full.npy"
    full: NDArray[np.bool_] | None = None
    if full_path.exists():
        full = np.load(full_path)
    elif (folder / "meta.json").exists():
        info = json.loads((folder / "meta.json").read_text())
        full = np.array([s["regime"] == "full" for s in info["samples"]])
    return ArrayData(
        images=np.load(folder / "images.npy", mmap_mode=mode),
        maps=np.load(folder / "extras_maps.npy"),
        scalars=np.load(folder / "extras_scalars.npy"),
        masks=np.load(masks_path) if masks_path.exists() else None,
        full=full,
    )


def official_miou(pred: BoolArray, truth: BoolArray) -> NDArray[np.float64]:
    """Per-image official mIoU for [N, H, W] boolean arrays (classes 0 and 1, empty union = 1)."""
    total = np.zeros(pred.shape[0])
    for cls in (False, True):
        p, t = pred == cls, truth == cls
        inter = np.logical_and(p, t).sum(axis=(1, 2))
        union = np.logical_or(p, t).sum(axis=(1, 2))
        total += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return total / 2.0


def localization_loss(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """BCE + (1 - mean soft IoU over the two classes), the latter matching the official metric."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
    prob = torch.sigmoid(logits)
    ious = []
    for p, t in ((prob, truth), (1.0 - prob, 1.0 - truth)):
        inter = (p * t).sum(dim=(1, 2))
        union = (p + t - p * t).sum(dim=(1, 2))
        ious.append((inter + SMOOTH) / (union + SMOOTH))
    soft_miou = (ious[0] + ious[1]) / 2.0
    return bce + (1.0 - soft_miou).mean()


def pick_device() -> torch.device:
    """MPS when available (Apple GPU), else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch(
    data: ArrayData, idx: NDArray[np.int64], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    order = np.sort(idx)
    images = torch.from_numpy(np.ascontiguousarray(data.images[order])).to(device)
    maps = torch.from_numpy(data.maps[order].astype(np.float32)).to(device)
    scalars = torch.from_numpy(data.scalars[order]).to(device)
    masks = None
    if data.masks is not None:
        masks = torch.from_numpy(data.masks[order].astype(np.float32)).to(device)
    return images, maps, scalars, masks


def _augment(
    images: torch.Tensor, masks: torch.Tensor, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random flip / transpose applied to the whole batch (valid for the residual-only variant)."""
    if rng.random() < 0.5:
        images, masks = images.flip(2), masks.flip(2)
    if rng.random() < 0.5:
        images, masks = images.flip(1), masks.flip(1)
    if rng.random() < 0.5:
        images, masks = images.transpose(1, 2), masks.transpose(1, 2)
    return images, masks


@torch.no_grad()
def predict(
    model: Localizer, data: ArrayData, idx: NDArray[np.int64], device: torch.device
) -> FloatArray:
    """Sigmoid probabilities [len(idx), 128, 128] in the order of `idx`."""
    model.eval()
    out = np.zeros((len(idx), 128, 128), dtype=np.float32)
    for start in range(0, len(idx), EVAL_BATCH):
        chunk = idx[start : start + EVAL_BATCH]
        order = np.argsort(chunk)
        images, maps, scalars, _ = _batch(data, chunk[order], device)
        probs = torch.sigmoid(model(images, maps, scalars)).float().cpu().numpy()
        restored = np.empty_like(probs)
        restored[order] = probs
        out[start : start + len(chunk)] = restored
    return out


def evaluate(
    model: Localizer, data: ArrayData, idx: NDArray[np.int64], device: torch.device
) -> NDArray[np.float64]:
    """Per-image official mIoU on `idx` (data must be labelled)."""
    if data.masks is None:
        raise ValueError("evaluation needs masks")
    probs = predict(model, data, idx, device)
    return official_miou(probs >= DECISION_THRESHOLD, data.masks[idx].astype(bool))


def fit(
    model: Localizer,
    train_sets: list[tuple[ArrayData, NDArray[np.int64]]],
    epochs: int,
    steps_per_epoch: int,
    lr: float,
    seed: int,
    device: torch.device,
    augment: bool,
    monitor: tuple[ArrayData, NDArray[np.int64]] | None = None,
    log: Any = print,
) -> list[dict[str, float]]:
    """AdamW + cosine schedule. Each step draws half of its batch from each set when two are given.

    Args:
        train_sets: [(data, indices), ...]; batches are split evenly among the sets.
        steps_per_epoch: Optimizer steps per epoch.
        monitor: Optional labelled held-out part scored at the end of each epoch.

    Returns:
        One record per epoch: train loss and, when monitored, held-out mean mIoU.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.15
    )
    per_set = max(1, BATCH_SIZE // len(train_sets))
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            parts = []
            for data, idx in train_sets:
                chosen = rng.choice(idx, size=per_set, replace=len(idx) < per_set)
                parts.append(_batch(data, chosen.astype(np.int64), device))
            images = torch.cat([p[0] for p in parts])
            maps = torch.cat([p[1] for p in parts])
            scalars = torch.cat([p[2] for p in parts])
            masks = torch.cat([p[3] for p in parts if p[3] is not None])
            if augment and not model.use_extras:
                images, masks = _augment(images, masks, rng)
            loss = localization_loss(model(images, maps, scalars), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "seconds": time.time() - started,
        }
        if monitor is not None:
            record["val_miou"] = float(evaluate(model, monitor[0], monitor[1], device).mean())
        history.append(record)
        log(json.dumps(record))
    return history


def stratified_folds(full: NDArray[np.bool_], n_folds: int, seed: int) -> list[NDArray[np.int64]]:
    """Fold index arrays, stratified by regime (full vs region), deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for flag in (True, False):
        members = rng.permutation(np.flatnonzero(full == flag))
        for position, index in enumerate(members):
            folds[position % n_folds].append(int(index))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def summarise(scores: NDArray[np.float64], data: ArrayData) -> dict[str, float]:
    """Mean mIoU overall, by regime and by region-area bucket (full: the hole is the minority)."""
    result = {"all": float(scores.mean())}
    if data.full is not None:
        result["full_regime"] = float(scores[data.full].mean()) if data.full.any() else float("nan")
        result["region_regime"] = (
            float(scores[~data.full].mean()) if (~data.full).any() else float("nan")
        )
    if data.masks is not None:
        area = data.masks.mean(axis=(1, 2))
        minority = np.where(data.full, 1.0 - area, area) if data.full is not None else area
        result["minority_small"] = float(scores[minority < SMALL_AREA_FRACTION].mean())
        mid = (minority >= SMALL_AREA_FRACTION) & (minority < LARGE_AREA_FRACTION)
        result["minority_mid"] = float(scores[mid].mean()) if mid.any() else float("nan")
        large = minority >= LARGE_AREA_FRACTION
        result["minority_large"] = float(scores[large].mean()) if large.any() else float("nan")
    return result


def trivial_baselines(data: ArrayData) -> dict[str, float]:
    """Mean mIoU of all-zeros, all-ones and the global-agreement switch (no localization)."""
    if data.masks is None:
        raise ValueError("baselines need masks")
    truth = data.masks.astype(bool)
    zeros = official_miou(np.zeros_like(truth), truth).mean()
    ones = official_miou(np.ones_like(truth), truth).mean()
    switch_on = data.scalars[:, 1] > 0.75
    switched = np.where(switch_on[:, None, None], np.ones_like(truth), np.zeros_like(truth))
    return {
        "all_zeros": float(zeros),
        "all_ones": float(ones),
        "global_switch": float(official_miou(switched, truth).mean()),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def cmd_pretrain(args: argparse.Namespace) -> None:
    """Train on synthetic data; save checkpoint + history."""
    device = pick_device()
    synth = load_arrays(Path(args.synth) / args.method)
    n = len(synth)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * SYNTH_VALIDATION_FRACTION))
    val_idx, train_idx = np.sort(order[:n_val]), np.sort(order[n_val:])
    model = Localizer(use_extras=args.variant == "extras")
    print(
        f"{args.method}/{args.variant}: {count_parameters(model)} parameters, device={device}, n_train={len(train_idx)}",
        flush=True,
    )
    steps = max(1, len(train_idx) // BATCH_SIZE)
    started = time.time()
    history = fit(
        model,
        [(synth, train_idx)],
        args.epochs,
        steps,
        PRETRAIN_LR,
        args.seed,
        device,
        True,
        (synth, val_idx),
        lambda s: print(s, flush=True),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.method}_{args.variant}.pt")
    _write(
        out / f"{args.method}_{args.variant}_pretrain.json",
        {
            "method": args.method,
            "variant": args.variant,
            "seed": args.seed,
            "epochs": args.epochs,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "parameters": count_parameters(model),
            "seconds": time.time() - started,
            "history": history,
            "device": str(device),
        },
    )


def _load_model(args: argparse.Namespace, device: torch.device) -> Localizer:
    model = Localizer(use_extras=args.variant == "extras")
    checkpoint = Path(args.ckpt) / f"{args.method}_{args.variant}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device)


def cmd_zeroshot(args: argparse.Namespace) -> None:
    """Score a synthetic-only checkpoint on the whole real train split."""
    device = pick_device()
    real = load_arrays(Path(args.real) / "train" / args.method)
    model = _load_model(args, device)
    idx = np.arange(len(real), dtype=np.int64)
    probs = predict(model, real, idx, device)
    assert real.masks is not None
    scores = official_miou(probs >= DECISION_THRESHOLD, real.masks.astype(bool))
    out = Path(args.out)
    np.savez_compressed(
        out / f"{args.method}_{args.variant}_zeroshot.npz",
        probs=probs.astype(np.float16),
        scores=scores,
    )
    _write(
        out / f"{args.method}_{args.variant}_zeroshot.json",
        {
            "method": args.method,
            "variant": args.variant,
            "summary": summarise(scores, real),
            "baselines": trivial_baselines(real),
        },
    )
    print(json.dumps(summarise(scores, real)), flush=True)


def cmd_crossval(args: argparse.Namespace) -> None:
    """k-fold fine-tuning on the real train split (mixed with synthetic data)."""
    device = pick_device()
    real = load_arrays(Path(args.real) / "train" / args.method, mmap=False)
    assert real.full is not None and real.masks is not None
    synth = load_arrays(Path(args.synth) / args.method)
    synth_idx = np.arange(len(synth), dtype=np.int64)
    folds = stratified_folds(real.full, args.folds, args.seed)
    all_scores = np.zeros(len(real))
    all_probs = np.zeros((len(real), 128, 128), dtype=np.float16)
    fold_means: list[float] = []
    histories: list[list[dict[str, float]]] = []
    for k, held_out in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(real)), held_out).astype(np.int64)
        torch.manual_seed(args.seed + k)
        if args.scratch:
            model = Localizer(use_extras=args.variant == "extras")
        else:
            model = _load_model(args, device)
        sets = [(real, train_idx)] if args.no_synth_mix else [(real, train_idx), (synth, synth_idx)]
        steps = max(1, len(train_idx) // BATCH_SIZE)
        lr = PRETRAIN_LR if args.scratch else FINETUNE_LR
        histories.append(
            fit(
                model,
                sets,
                args.epochs,
                steps,
                lr,
                args.seed + k,
                device,
                True,
                None,
                lambda s: None,
            )
        )
        scores = evaluate(model, real, held_out, device)
        all_probs[held_out] = predict(model, real, held_out, device).astype(np.float16)
        all_scores[held_out] = scores
        fold_means.append(float(scores.mean()))
        print(
            f"{args.method}/{args.variant} fold {k}: mIoU={scores.mean():.4f} (n={len(held_out)})",
            flush=True,
        )
    out = Path(args.out)
    tag = f"{args.method}_{args.variant}_{'scratch' if args.scratch else 'ft'}{'_nomix' if args.no_synth_mix else ''}_seed{args.seed}"
    np.savez_compressed(out / f"{tag}_cv.npz", probs=all_probs, scores=all_scores)
    _write(
        out / f"{tag}_cv.json",
        {
            "method": args.method,
            "variant": args.variant,
            "scratch": args.scratch,
            "seed": args.seed,
            "folds": args.folds,
            "epochs": args.epochs,
            "fold_means": fold_means,
            "fold_std": float(np.std(fold_means)),
            "summary": summarise(all_scores, real),
            "baselines": trivial_baselines(real),
            "histories": histories,
        },
    )
    print(
        f"{tag}: mean {np.mean(all_scores):.4f} +- {np.std(fold_means):.4f} (fold std)", flush=True
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pretrain", "zeroshot", "crossval"):
        p = sub.add_parser(name)
        p.add_argument("--method", required=True, choices=METHODS)
        p.add_argument("--variant", required=True, choices=("resid", "extras"))
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--synth", default="")
        p.add_argument("--real", default="")
        p.add_argument("--ckpt", default="")
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
        p.add_argument("--scratch", action="store_true")
        p.add_argument("--no-synth-mix", action="store_true")
    args = parser.parse_args(argv)
    {"pretrain": cmd_pretrain, "zeroshot": cmd_zeroshot, "crossval": cmd_crossval}[args.command](
        args
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
