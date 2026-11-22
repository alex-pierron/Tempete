"""Data assembly, training and 5-fold cross-validation of the B2 segmentation head.

Everything reported is out-of-fold on the 510 real train images: folds are those of team A
(`stratified_folds` by method x regime, seed 2026), so numbers are directly comparable. The metric
is the official per-image mIoU, broken down by method, by regime (full / rectangle) and by size of
the minority zone (the hole in the full regime, the rectangle in the rectangle regime).

Usage (kit venv, see /tmp/b2/run.sh):
    python -m hackathon.wmloc.research.b2_train cv --methods pixelseal --epochs 60 --tag ps_base
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from hackathon.wmloc.research.b2_dino import load_dino
from hackathon.wmloc.research.b2_features import MARGIN_CLIP, cache_root, load_features
from hackathon.wmloc.research.b2_model import (
    DIHEDRAL_ELEMENTS,
    DINO_KEY,
    OUT_SIDE,
    PIXEL_KEY,
    SOURCES_BY_METHOD,
    WINDOW_KEY,
    B2Net,
    HeadConfig,
    combined_loss,
    count_parameters,
    dihedral,
    inverse_dihedral,
)
from hackathon.wmloc.research.b2_pack import load_global_table
from hackathon.wmloc.research.b2_synth_features import SynthFeatureError, append_synthetic
from hackathon.wmloc.research.b2_windows import load_window_maps
from hackathon.wmloc.research.data import Record, load_image, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import (
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    miou_per_image,
    stratified_folds,
)
from hackathon.wmloc.research.scoring import global_logits

MARGIN_SCALE = 8.0
SPREAD_SCALE = 2.0
AGREEMENT_CENTRE = 0.5
RGB_CENTRE = 0.5
RGB_POOL = 4
"""512 -> 128 area pooling of the RGB thumbnail."""
THREADS = 2
"""Torch intra-op threads by default: the Mac is shared by about nine agents."""
AUTO_DEVICE = "auto"
MONITOR_EVERY = 5
"""Epochs between two held-out scorings of the learning curve."""
DECISION_THRESHOLD = 0.5
SIZE_BIN_EDGES: tuple[float, ...] = (0.0, 0.02, 0.05, 0.15, 1.0)
HOLE_BIN_EDGES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 1.0)
RESULTS_SUBDIR = "team_b2/cv"
RGB_SUBDIR = "team_b2/rgb128"


class TrainingError(ResearchError):
    """Inconsistent training inputs (empty split, unknown method)."""


@dataclass
class MethodArrays:
    """All cached inputs of one (split, method), stacked over images.

    Attributes:
        method: Method name.
        names: Image names.
        sources: Feature maps by source name, float16 [N, C, h, w] (DINO included).
        scalars: Normalised regime scalars [N, 3].
        rgb: uint8 [N, 3, 128, 128] thumbnails.
        truth: Ground-truth masks [N, 128, 128] (train only).
        is_full: Regime flag (train only).
        folds: Fold id per image (train only). For synthetic rows it is the fold of the REAL image
            the mask shape came from.
        n_real: Number of real rows; the arrays may carry synthetic rows AFTER them, which
            training can use but evaluation and `names` never include.
    """

    method: str
    names: list[str]
    sources: dict[str, torch.Tensor]
    scalars: torch.Tensor
    rgb: torch.Tensor
    truth: NDArray[np.uint8] | None
    is_full: NDArray[np.bool_] | None
    folds: NDArray[np.int64] | None
    n_real: int


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings (all named: no magic numbers in the loop)."""

    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    bce_weight: float = 0.5
    soft_weight: float = 1.0
    lovasz_weight: float = 0.0
    seed: int = DEFAULT_SEED
    device: str = "auto"
    augment: bool = True
    use_synthetic: bool = False
    synthetic_per_real: float = 2.0
    """Synthetic samples drawn per real training sample in each epoch of the mixed phase."""
    finetune_fraction: float = 0.25
    """Last share of the epochs run on real images only (synthetic pretraining, real fine-tuning)."""


WRITE_ATTEMPTS = 20
WRITE_RETRY_SECONDS = 15.0
"""The shared scratch runs out of inodes for minutes at a time (quota common to every team): a
result that took a GPU task to compute is kept in memory and its write is retried."""


MAX_RSS_GB_ENV = "WMLOC_MAX_RSS_GB"
MAX_RSS_GB_DEFAULT = 90.0
"""Resident memory above which a run aborts instead of taking the node down with it: on 22/11 two
fits and a neighbour's task filled the 503 GB of the node and the kernel killed processes."""
KB_PER_GB = 1024.0 * 1024.0
PROC_STATUS = Path("/proc/self/status")
GUARD_EVERY_SHARDS = 8
GUARD_EVERY_STEPS = 500


class MemoryGuardError(ResearchError):
    """The process holds more resident memory than `$WMLOC_MAX_RSS_GB` allows."""


def resident_gb() -> float | None:
    """Anonymous resident memory of this process in GB (Linux `/proc`), None where not exposed.

    `RssAnon` (heap, arrays) rather than `VmRSS`: on the A100 nodes VmRSS also counts file and
    driver mappings (96 GB before any data on 22/11, 298 GB right after the first `.to("cuda")`
    while the node still had 430 GB free), which made the guard abort healthy runs. Falls back to
    VmRSS on kernels without RssAnon.
    """
    try:
        fields = {
            line.split(":", 1)[0]: float(line.split()[1])
            for line in PROC_STATUS.read_text().splitlines()
            if line.startswith(("RssAnon:", "VmRSS:"))
        }
    except (OSError, ValueError, IndexError):
        return None
    kilobytes = fields.get("RssAnon", fields.get("VmRSS"))
    return None if kilobytes is None else kilobytes / KB_PER_GB


def memory_guard(stage: str) -> None:
    """Print the resident memory at `stage` and abort above the cap.

    Raises:
        MemoryGuardError: resident memory exceeds `$WMLOC_MAX_RSS_GB` (default 90 GB).
    """
    resident = resident_gb()
    if resident is None:
        return
    limit = float(os.environ.get(MAX_RSS_GB_ENV, MAX_RSS_GB_DEFAULT))
    print(f"  memory {resident:.1f} GB resident at {stage}", flush=True)
    if resident > limit:
        raise MemoryGuardError(
            f"{resident:.1f} GB resident at {stage}, above the {limit:.0f} GB cap"
        )


def retry_write(action: Callable[[], None], what: str) -> None:
    """Run a file write, retrying on `OSError` (quota, I/O) before giving up.

    Raises:
        OSError: the last of `WRITE_ATTEMPTS` attempts failed.
    """
    for attempt in range(1, WRITE_ATTEMPTS + 1):
        try:
            action()
            return
        except OSError as exc:
            if attempt == WRITE_ATTEMPTS:
                raise
            print(f"write of {what} failed ({exc}); attempt {attempt}/{WRITE_ATTEMPTS}", flush=True)
            time.sleep(WRITE_RETRY_SECONDS)


def save_atomically(path: Path, arrays: dict[str, Any]) -> None:
    """Write a compressed npz to a temporary name, then rename: readers never see half a file."""
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")

    def write() -> None:
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    retry_write(write, path.name)


def write_text(path: Path, text: str) -> None:
    """Write a small text file (report, metadata), retrying like `save_atomically`."""

    def write() -> None:
        path.write_text(text, encoding="utf-8")

    retry_write(write, path.name)


def results_dir() -> Path:
    """Folder holding the `cv` and `fit` outputs (`<data>/wmloc/team_b2/cv`)."""
    return cache_root().parent.parent / RESULTS_SUBDIR


def _rgb_thumbnails(split: str, method: str, names: list[str]) -> torch.Tensor:
    """uint8 [N, 3, 128, 128] thumbnails, cached on disk."""
    path = cache_root().parent / "rgb128" / f"{split}_{method}.npy"
    if path.is_file():
        return torch.from_numpy(np.load(path))
    thumbs = []
    for name in names:
        image = load_image(split, name)[None]
        pooled = torch.nn.functional.avg_pool2d(image, RGB_POOL)[0]
        thumbs.append((pooled * 255.0).round().clamp(0, 255).to(torch.uint8))
    stacked = torch.stack(thumbs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stacked.numpy())
    return stacked


def global_decode_scalars(split: str, record: Record) -> NDArray[np.float32]:
    """Regime scalars [margin, agreement, spread] of the whole-image decode (team A's cache).

    This is what a method WITHOUT spatial features (StegaStamp) or a content-only ablation uses:
    only the global decode of the known message, the same evidence as the regime switch.
    """
    table = load_global_table(split, record.method)
    logits = (
        table[record.name]
        if table is not None and record.name in table
        else global_logits(split, record.method, record.name)
    )
    signed = logits * (record.message * 2.0 - 1.0)
    return np.array(
        [
            float(np.clip(signed.mean(), -MARGIN_CLIP, MARGIN_CLIP)),
            float((signed > 0).mean()),
            float(signed.std()),
        ],
        dtype=np.float32,
    )


def _normalise_scalars(raw: NDArray[np.float32]) -> torch.Tensor:
    """[N, 3] raw (margin, agreement, spread) -> centred and scaled network input."""
    return torch.from_numpy(
        np.stack(
            [
                raw[:, 0] / MARGIN_SCALE,
                raw[:, 1] - AGREEMENT_CENTRE,
                raw[:, 2] / SPREAD_SCALE,
            ],
            axis=1,
        ).astype(np.float32)
    )


def load_arrays(
    split: str,
    method: str,
    with_dino: bool = True,
    with_windows: bool = False,
    with_extractor: bool = True,
    synthetic: bool = False,
    with_pixels: bool = False,
) -> MethodArrays:
    """Stack every cached input of `method` in `split` (metadata order).

    Args:
        split: `train` or `test`.
        method: Method name.
        with_dino: Also stack the shared DINOv2 content features.
        with_windows: Also stack the multi-scale window-decode maps (`b2_windows` cache).
        with_extractor: Stack the extractor feature maps. False (or a method without any, such as
            StegaStamp) falls back to the global-decode scalars of team A's cache.
        synthetic: Append the cached synthetic samples (`b2_synth_features`) after the real rows
            (train split only, extractor features only).
        with_pixels: Also stack the full-resolution images (uint8) under the `pixels` source, for
            the trainable pixel stem.

    Raises:
        TrainingError: no image of that method, or a cache file is missing.
    """
    records = [r for r in load_records(split) if r.method == method]
    if not records:
        raise TrainingError(f"no {method} image in {split}")
    names = [r.name for r in records]
    with_extractor = with_extractor and method in SOURCES_BY_METHOD
    per_image = [load_features(split, method, name) for name in names] if with_extractor else []
    wanted = SOURCES_BY_METHOD[method] if with_extractor else {}
    sources = {key: torch.from_numpy(np.stack([row[key] for row in per_image])) for key in wanted}
    if with_dino:
        sources[DINO_KEY] = torch.from_numpy(np.stack([load_dino(split, n) for n in names]))
    if with_windows:
        sources[WINDOW_KEY] = torch.from_numpy(
            np.stack([load_window_maps(split, method, n) for n in names])
        )
    raw = (
        np.stack([row["scalars"] for row in per_image]).astype(np.float32)
        if with_extractor
        else np.stack([global_decode_scalars(split, r) for r in records])
    )
    scalars = _normalise_scalars(raw)
    truth = is_full = folds = None
    if split == "train":
        truth = load_truth(names)
        is_full = np.array([bool(r.full_watermark) for r in records])
        folds = stratified_folds(
            [f"{method}/{flag}" for flag in is_full], DEFAULT_FOLDS, DEFAULT_SEED
        )
    rgb = _rgb_thumbnails(split, method, names)
    n_real = len(names)
    if with_pixels:
        sources[PIXEL_KEY] = torch.stack(
            [(load_image(split, n) * 255.0).round().to(torch.uint8) for n in names]
        )
    wants_synthetic = synthetic and split == "train"
    if wants_synthetic and truth is not None and is_full is not None and folds is not None:
        # One allocation per array at its final size (see `append_synthetic`): the former
        # load-then-concatenate held three copies and got eight parallel fits killed on the node.
        real_rows: dict[str, NDArray[Any]] = {key: value.numpy() for key, value in sources.items()}
        real_rows.update(
            {
                "scalars": raw,
                "rgb": rgb.numpy(),
                "truth": truth,
                "is_full": is_full,
                "fold": folds,
            }
        )
        memory_guard(f"{method}: real rows loaded")

        def after_shard(done: int) -> None:
            if done % GUARD_EVERY_SHARDS == 0:
                memory_guard(f"{method}: {done} synthetic shards read")

        try:
            joined = append_synthetic(
                method, real_rows, rename={PIXEL_KEY: "image"}, after_shard=after_shard
            )
        except SynthFeatureError as exc:
            raise TrainingError(f"synthetic shards of {method} are unusable: {exc}") from exc
        sources = {key: torch.from_numpy(joined[key]) for key in sources}
        scalars = _normalise_scalars(joined["scalars"])
        rgb = torch.from_numpy(joined["rgb"])
        truth = joined["truth"]
        is_full = joined["is_full"].astype(np.bool_)
        folds = joined["fold"].astype(np.int64)
        memory_guard(f"{method}: synthetic rows joined")
    return MethodArrays(method, names, sources, scalars, rgb, truth, is_full, folds, n_real)


def _batch(
    arrays: MethodArrays, indices: NDArray[np.int64], device: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Slice a mini-batch and move it to `device` (float conversion happens in the adapters)."""
    index = torch.from_numpy(indices)
    sources = {
        key: value[index.to(value.device)].to(device) for key, value in arrays.sources.items()
    }
    scalars = arrays.scalars[index].to(device)
    rgb = arrays.rgb[index].to(device).float() / 255.0 - RGB_CENTRE
    return sources, scalars, rgb


def resolve_device(requested: str) -> str:
    """`auto` -> cuda, then mps, then cpu; any other value is returned unchanged."""
    if requested != AUTO_DEVICE:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrainResult:
    """A trained model and its learning curves.

    Attributes:
        model: The trained head (eval mode not forced).
        epoch_losses: Mean training loss of each epoch.
        monitor: `(epoch, held-out mIoU)` pairs, every `MONITOR_EVERY` epochs, when monitoring.
    """

    model: B2Net
    epoch_losses: list[float]
    monitor: list[tuple[int, float]]


def monitor_score(
    model: B2Net,
    arrays: dict[str, MethodArrays],
    held: dict[str, NDArray[np.int64]],
    device: str,
) -> float:
    """Mean official mIoU (threshold 0.5) over the held-out images of every method."""
    scores: list[NDArray[np.float64]] = []
    for method, indices in held.items():
        truth = arrays[method].truth
        if truth is None or len(indices) == 0:
            continue
        probabilities = predict_probabilities(model, arrays[method], indices, device)
        prediction = (probabilities >= DECISION_THRESHOLD).astype(np.uint8)
        scores.append(miou_per_image(prediction, truth[indices]))
    model.train()
    return float(np.concatenate(scores).mean()) if scores else 0.0


GPU_DATA_GB_ENV = "WMLOC_GPU_DATA_GB"
GPU_DATA_GB_DEFAULT = 30.0
"""Per process: an A100 has 40 GB; the head and its activations need a few, and 5 GB stay free."""
BYTES_PER_GB = 1e9


def place_on_device(arrays: dict[str, MethodArrays], device: str) -> float:
    """Keep the feature maps in the accelerator's memory when they fit; returns the GB moved.

    A head of a few hundred thousand parameters is cheap on an A100: a step is dominated by copying
    mini-batches from host memory. Holding the (fp16) feature maps on the GPU removes that copy and
    uses the memory the node offers. Maps are placed smallest first until the per-process budget
    (`$WMLOC_GPU_DATA_GB`, default 30) is spent; whatever does not fit, or hits an out-of-memory
    error, stays on the host and is read from there as before. Idempotent.
    """
    if not device.startswith("cuda"):
        return 0.0
    budget = float(os.environ.get(GPU_DATA_GB_ENV, GPU_DATA_GB_DEFAULT)) * BYTES_PER_GB
    holders = [
        (int(owner[key].element_size() * owner[key].numel()), owner, key)
        for owner in (data.sources for data in arrays.values())
        for key in owner
    ]
    used = sum(size for size, owner, key in holders if owner[key].device.type == "cuda")
    moved = 0
    for size, owner, key in sorted(holders, key=lambda item: item[0]):
        if owner[key].device.type == "cuda" or used + size > budget:
            continue
        try:
            owner[key] = owner[key].to(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        used += size
        moved += size
    memory_guard(f"maps placed on {device}")
    return moved / BYTES_PER_GB


def release_device_data(arrays: dict[str, MethodArrays]) -> None:
    """Move the feature maps back to the host and free the accelerator memory they held.

    A sweep keeps several array sets alive; only the one being trained should sit on the GPU.
    """
    for data in arrays.values():
        for key, tensor in data.sources.items():
            if tensor.device.type == "cuda":
                data.sources[key] = tensor.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mixed_epoch_size(indices: NDArray[np.int64], n_real: int, config: TrainConfig) -> int:
    """Samples of one epoch of the mixed phase: all real rows plus the synthetic quota."""
    n_real_rows = int((indices < n_real).sum())
    n_synthetic = len(indices) - n_real_rows
    if not config.use_synthetic or n_synthetic == 0:
        return n_real_rows
    return n_real_rows + min(n_synthetic, int(config.synthetic_per_real * n_real_rows))


def _epoch_order(
    indices: NDArray[np.int64],
    n_real: int,
    config: TrainConfig,
    use_synthetic: bool,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Shuffled rows of one epoch: real only, or real plus a random synthetic subset."""
    real = indices[indices < n_real]
    synthetic = indices[indices >= n_real]
    if not use_synthetic or len(synthetic) == 0:
        return np.asarray(rng.permutation(real), dtype=np.int64)
    quota = min(len(synthetic), int(config.synthetic_per_real * len(real)))
    drawn = rng.choice(synthetic, size=quota, replace=False)
    return np.asarray(rng.permutation(np.concatenate([real, drawn])), dtype=np.int64)


def train_model(
    arrays: dict[str, MethodArrays],
    train_indices: dict[str, NDArray[np.int64]],
    head: HeadConfig,
    config: TrainConfig,
    monitor: dict[str, NDArray[np.int64]] | None = None,
) -> TrainResult:
    """Train one joint model on the given images of every method.

    Args:
        arrays: Inputs by method.
        train_indices: Training images by method.
        head: Architecture settings.
        config: Optimisation settings.
        monitor: Optional held-out images by method, scored every `MONITOR_EVERY` epochs to draw a
            learning curve (never used for any decision inside the loop).
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    model = B2Net(head).to(device)
    epoch_losses: list[float] = []
    monitor_curve: list[tuple[int, float]] = []
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    mixed_sizes = {
        m: _mixed_epoch_size(idx, arrays[m].n_real, config) for m, idx in train_indices.items()
    }
    steps_per_epoch = max(int(np.ceil(size / config.batch_size)) for size in mixed_sizes.values())
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.learning_rate, total_steps=config.epochs * steps_per_epoch
    )
    finetune_start = round(config.epochs * (1.0 - config.finetune_fraction))
    started_at = time.time()
    model.train()
    for epoch in range(config.epochs):
        use_synthetic = config.use_synthetic and epoch < finetune_start
        orders = {
            m: _epoch_order(idx, arrays[m].n_real, config, use_synthetic, rng)
            for m, idx in train_indices.items()
        }
        running = 0.0
        for step in range(steps_per_epoch):
            element = int(rng.integers(DIHEDRAL_ELEMENTS)) if config.augment else 0
            optimiser.zero_grad()
            for method, order in orders.items():
                start = (step * config.batch_size) % len(order)
                chosen = order[start : start + config.batch_size]
                sources, scalars, rgb = _batch(arrays[method], chosen, device)
                truth_array = arrays[method].truth
                if truth_array is None:
                    raise TrainingError("training needs ground-truth masks")
                target = torch.from_numpy(truth_array[chosen]).to(device)
                sources = {k: dihedral(v, element) for k, v in sources.items()}
                logits = model(method, sources, scalars, dihedral(rgb, element))
                loss = combined_loss(
                    logits,
                    dihedral(target, element),
                    config.bce_weight,
                    config.soft_weight,
                    config.lovasz_weight,
                )
                loss.backward()  # type: ignore[no-untyped-call]  # torch stubs leave backward untyped
                running += float(loss.detach())
            optimiser.step()
            schedule.step()
            if (epoch * steps_per_epoch + step) % GUARD_EVERY_STEPS == 0:
                memory_guard(f"training step {epoch * steps_per_epoch + step}")
        epoch_losses.append(running / (steps_per_epoch * len(orders)))
        if monitor is not None and (epoch + 1) % MONITOR_EVERY == 0:
            monitor_curve.append((epoch + 1, monitor_score(model, arrays, monitor, device)))
    elapsed = time.time() - started_at
    total_steps = config.epochs * steps_per_epoch
    print(
        f"  trained {total_steps} steps in {elapsed:.0f}s ({total_steps / max(elapsed, 1e-9):.1f} "
        f"steps/s, {len(arrays)} method(s), device {device})",
        flush=True,
    )
    return TrainResult(model, epoch_losses, monitor_curve)


@torch.no_grad()
def predict_probabilities(
    model: B2Net,
    arrays: MethodArrays,
    indices: NDArray[np.int64],
    device: str,
    tta: bool = False,
) -> NDArray[np.float32]:
    """Sigmoid probabilities [n, 128, 128] of the images `indices` (eval mode).

    With `tta`, the 8 symmetries of the square are applied to every input map, the predictions are
    mapped back and averaged: the head is trained under the same symmetries, so this only reduces
    variance (no information is added).
    """
    model.eval()
    elements = range(DIHEDRAL_ELEMENTS) if tta else range(1)
    outputs = []
    for start in range(0, len(indices), 16):
        chunk = indices[start : start + 16]
        sources, scalars, rgb = _batch(arrays, chunk, device)
        total: torch.Tensor | None = None
        for element in elements:
            transformed = {key: dihedral(value, element) for key, value in sources.items()}
            logits = model(arrays.method, transformed, scalars, dihedral(rgb, element))
            probability = inverse_dihedral(torch.sigmoid(logits), element)
            total = probability if total is None else total + probability
        assert total is not None
        outputs.append((total / len(elements)).float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, OUT_SIDE, OUT_SIDE), np.float32)


def breakdown(
    per_image: NDArray[np.float64], truth: NDArray[np.uint8], is_full: NDArray[np.bool_]
) -> dict[str, object]:
    """Mean mIoU overall, by regime and by minority-zone size bin."""
    result: dict[str, object] = {
        "all": float(per_image.mean()),
        "full": float(per_image[is_full].mean()),
        "rect": float(per_image[~is_full].mean()),
    }
    fraction = truth.reshape(len(truth), -1).mean(axis=1)
    for regime, flag, edges, measure in (
        ("rect", False, SIZE_BIN_EDGES, fraction),
        ("full", True, HOLE_BIN_EDGES, 1.0 - fraction),
    ):
        bins = []
        for low, high in itertools.pairwise(edges):
            chosen = (is_full == flag) & (measure >= low) & (measure < high)
            bins.append(
                {
                    "low": low,
                    "high": high,
                    "n": int(chosen.sum()),
                    "mean": float(per_image[chosen].mean()) if chosen.any() else None,
                }
            )
        result[f"by_size_{regime}"] = bins
    return result


def run_cv(
    methods: list[str], head: HeadConfig, config: TrainConfig, folds: int = DEFAULT_FOLDS
) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, object]], list[dict[str, object]]]:
    """Joint 5-fold CV over `methods`.

    Returns:
        OOF probabilities by method, per-method breakdowns, and one learning-curve record per
        fold (`fold`, `losses`, `monitor`, `seconds`).
    """
    arrays = {
        m: load_arrays(
            "train",
            m,
            with_windows=WINDOW_KEY in head.sources[m],
            with_extractor=bool(head.sources[m]),
            synthetic=config.use_synthetic,
            with_pixels=head.use_pixels,
        )
        for m in methods
    }
    return cross_validate(arrays, head, config, folds)


def cross_validate(
    arrays: dict[str, MethodArrays],
    head: HeadConfig,
    config: TrainConfig,
    folds: int = DEFAULT_FOLDS,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, object]], list[dict[str, object]]]:
    """5-fold CV on already loaded arrays (lets a sweep load the data once per method)."""
    device = resolve_device(config.device)
    moved = place_on_device(arrays, device)
    if moved:
        print(f"  {moved:.1f} GB of feature maps placed on {device}", flush=True)
    probabilities = {
        m: np.zeros((len(a.names), OUT_SIDE, OUT_SIDE), np.float32) for m, a in arrays.items()
    }
    probabilities_tta = {m: p.copy() for m, p in probabilities.items()}
    curves: list[dict[str, object]] = []
    for fold in range(folds):
        started = time.time()
        held, train = {}, {}
        for method, data in arrays.items():
            assert data.folds is not None
            real_rows = np.arange(len(data.folds)) < data.n_real
            held[method] = np.flatnonzero((data.folds == fold) & real_rows)
            train[method] = np.flatnonzero(data.folds != fold)
        fold_config = TrainConfig(**{**asdict(config), "seed": config.seed + fold})
        result = train_model(arrays, train, head, fold_config, monitor=held)
        for method, data in arrays.items():
            probabilities[method][held[method]] = predict_probabilities(
                result.model, data, held[method], device
            )
            probabilities_tta[method][held[method]] = predict_probabilities(
                result.model, data, held[method], device, tta=True
            )
        seconds = time.time() - started
        curves.append(
            {
                "fold": fold,
                "losses": result.epoch_losses,
                "monitor": result.monitor,
                "seconds": seconds,
            }
        )
        print(
            f"fold {fold}: {seconds:.0f}s, {count_parameters(result.model)} params, "
            f"last monitor {result.monitor[-1] if result.monitor else None}",
            flush=True,
        )
    reports: dict[str, dict[str, object]] = {}
    for method, data in arrays.items():
        assert data.truth is not None and data.is_full is not None
        truth = data.truth[: data.n_real]
        regimes = data.is_full[: data.n_real]
        prediction = (probabilities[method] >= DECISION_THRESHOLD).astype(np.uint8)
        per_image = miou_per_image(prediction, truth)
        reports[method] = breakdown(per_image, truth, regimes)
        tta_prediction = (probabilities_tta[method] >= DECISION_THRESHOLD).astype(np.uint8)
        tta_report = breakdown(miou_per_image(tta_prediction, truth), truth, regimes)
        reports[method].update({f"tta_{key}": tta_report[key] for key in ("all", "full", "rect")})
        print(
            method,
            json.dumps({k: v for k, v in reports[method].items() if not k.startswith("by_")}),
            flush=True,
        )
    everything: dict[str, NDArray[np.float32]] = dict(probabilities)
    everything.update({f"{m}_tta": p for m, p in probabilities_tta.items()})
    return everything, reports, curves


def fit_and_predict_test(
    methods: list[str], head: HeadConfig, config: TrainConfig, tta: bool = False
) -> tuple[dict[str, NDArray[np.float32]], dict[str, list[str]]]:
    """Train on ALL train images, predict the test images (probabilities, names).

    `tta` averages the 8 symmetric predictions (see `predict_probabilities`).
    """
    kwargs = {
        m: {
            "with_windows": WINDOW_KEY in head.sources[m],
            "with_extractor": bool(head.sources[m]),
            "with_pixels": head.use_pixels,
        }
        for m in methods
    }
    train_arrays = {
        m: load_arrays("train", m, synthetic=config.use_synthetic, **kwargs[m]) for m in methods
    }
    test_arrays = {m: load_arrays("test", m, **kwargs[m]) for m in methods}
    everything = {m: np.arange(a.scalars.shape[0], dtype=np.int64) for m, a in train_arrays.items()}
    device = resolve_device(config.device)
    moved = place_on_device(train_arrays, device)
    if moved:
        print(f"  {moved:.1f} GB of feature maps placed on {device}", flush=True)
    result = train_model(train_arrays, everything, head, config)
    probabilities = {
        m: predict_probabilities(
            result.model, a, np.arange(len(a.names), dtype=np.int64), device, tta=tta
        )
        for m, a in test_arrays.items()
    }
    return probabilities, {m: a.names for m, a in test_arrays.items()}


def _selected_sources(method: str, args: argparse.Namespace) -> tuple[str, ...]:
    """Sources of `method`: `--sources` if given, else every extractor map (+ windows on request)."""
    if args.content_only:
        return ()
    if args.sources:
        return tuple(args.sources)
    names = tuple(SOURCES_BY_METHOD.get(method, {}))
    return (*names, WINDOW_KEY) if args.windows and names else names


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--methods", nargs="+", default=["pixelseal"])
    parser.add_argument("--sources", nargs="+", help="explicit source names (window key: win)")
    parser.add_argument("--windows", action="store_true", help="add the window-decode maps")
    parser.add_argument(
        "--content-only",
        action="store_true",
        help="ablation: no extractor evidence at all (DINO content prior + RGB + regime scalars)",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lovasz", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=HeadConfig.width)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--bce-weight", type=float, default=TrainConfig.bce_weight)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-dino", action="store_true")
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--tta", action="store_true", help="fit: average 8 symmetric predictions")
    parser.add_argument("--synthetic", action="store_true", help="also train on synthetic shards")
    parser.add_argument("--pixels", action="store_true", help="add the trainable pixel stem")
    parser.add_argument("--synthetic-per-real", type=float, default=TrainConfig.synthetic_per_real)
    parser.add_argument("--finetune-fraction", type=float, default=TrainConfig.finetune_fraction)
    parser.add_argument("--tag", required=True)


def main() -> None:
    """CLI entry point: `cv` (5-fold out-of-fold evaluation) or `fit` (train all, predict test)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_common_arguments(sub.add_parser("cv"))
    _add_common_arguments(sub.add_parser("fit"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    head = HeadConfig(
        sources={m: _selected_sources(m, args) for m in args.methods},
        use_dino=not args.no_dino,
        use_rgb=not args.no_rgb,
        dropout=args.dropout,
        width=args.width,
        use_pixels=args.pixels,
    )
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        batch_size=args.batch_size,
        lovasz_weight=args.lovasz,
        device=args.device,
        seed=args.seed,
        use_synthetic=args.synthetic,
        synthetic_per_real=args.synthetic_per_real,
        finetune_fraction=args.finetune_fraction,
    )
    out = results_dir()
    out.mkdir(parents=True, exist_ok=True)
    head_record = {"dropout": args.dropout, "dino": not args.no_dino, "rgb": not args.no_rgb}
    if args.command == "fit":
        test_probabilities, test_names = fit_and_predict_test(
            args.methods, head, config, tta=args.tta
        )
        fitted: dict[str, Any] = {m: p.astype(np.float16) for m, p in test_probabilities.items()}
        fitted.update({f"names_{m}": np.array(n) for m, n in test_names.items()})
        meta = {"config": asdict(config), "head": head_record, "methods": args.methods}
        write_text(out / f"{args.tag}_test.json", json.dumps(meta, indent=1))
        # The npz goes last and atomically: its existence means "this fit is complete" (a filler
        # task can be killed at any moment).
        save_atomically(out / f"{args.tag}_test.npz", fitted)
        return
    probabilities, reports, curves = run_cv(args.methods, head, config)
    archive: dict[str, Any] = {m: p.astype(np.float16) for m, p in probabilities.items()}
    save_atomically(out / f"{args.tag}.npz", archive)
    payload = {
        "config": asdict(config),
        "head": head_record,
        "reports": reports,
        "curves": curves,
    }
    write_text(out / f"{args.tag}.json", json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
