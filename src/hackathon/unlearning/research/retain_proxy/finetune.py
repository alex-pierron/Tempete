"""Fine-tune the original on the generated retain proxy, and read what it does to the forget set.

Idea: the gold never saw the forget images and never saw nine classes. Training the original only
on retain-like images (no forget images, no image of the nine classes) is the closest thing to
"retraining without the forget set" that a few minutes of A100 time allow. The run records, at the
end of each epoch, how the model now answers the forget images: do the nine removed classes lose
their images to sister classes, do the partial classes keep a moderate confidence (unseen-image
behaviour) instead of the memorised 0.9997?

Memory: gradient accumulation over micro-batches whose size follows the same adaptive rule as the
generator (target >= 30 GB per A100), with a full-step retry after an out-of-memory error.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torchvision.ops import roi_align

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import DEFAULT_SEED as GLOBAL_SEED
from hackathon.unlearning.constants import (
    FORGET_IMAGES_KEY,
    IMAGE_SIZE,
    NUM_CLASSES,
    PIXEL_MAX,
    SCORING_TEMPERATURE,
)
from hackathon.unlearning.model import load_model, predict, server_autocast
from hackathon.unlearning.research.retain_proxy.blending import blend_state_dicts
from hackathon.unlearning.research.retain_proxy.concepts import (
    FULLY_FORGOTTEN_CLASSES,
    sister_classes,
)
from hackathon.unlearning.research.retain_proxy.export_views import (
    EVAL_SUBSET,
    build_views,
    export_model,
    subset_indices,
)
from hackathon.unlearning.research.retain_proxy.gpu_memory import (
    DEFAULT_MARGIN_GB,
    DEFAULT_TARGET_GB,
    BatchSizer,
    MemoryProbe,
    device_budget_gb,
    make_probe,
)
from hackathon.unlearning.training import kd_from_teacher, learning_rate_at

TRAIN_SEED = GLOBAL_SEED
"""Named seed of the shuffles and augmentations of every run."""
ASPECT_RATIO_RANGE = (3 / 4, 4 / 3)
JITTER_RANGE = (0.85, 1.15)
ROI_SAMPLING_RATIO = 2
PREDICT_BATCH = 256


class FinetuneError(HackathonError):
    """Bad dataset or configuration for the fine-tuning run."""


class FinetuneConfig(BaseModel):
    """One fine-tuning variant; immutable and dumped next to its results."""

    model_config = ConfigDict(frozen=True)

    name: str
    dataset_path: Path
    original_path: Path
    forget_path: Path
    out_dir: Path
    lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    epochs: int = Field(default=3, ge=1)
    effective_batch: int = Field(default=256, ge=2)
    warmup_fraction: float = Field(default=0.1, ge=0, lt=1)
    ce_weight: float = Field(default=1.0, ge=0)
    kd_weight: float = Field(default=0.0, ge=0)
    kd_temperature: float = Field(default=SCORING_TEMPERATURE, gt=0)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    crop_min_area: float = Field(default=0.6, gt=0, le=1)
    seed: int = TRAIN_SEED
    start_micro_batch: int = Field(default=128, ge=1)
    max_micro_batch: int = Field(default=512, ge=1)
    target_memory_gb: float = DEFAULT_TARGET_GB
    memory_margin_gb: float = DEFAULT_MARGIN_GB
    holdout_fraction: float = Field(default=0.1, ge=0, lt=0.5)
    save_model: bool = False
    export_logits_dir: Path | None = Field(
        default=None,
        description="Write team C's per-view logits of the final model and its blends here.",
    )
    export_alphas: tuple[float, ...] = (1.0,)
    export_subset_size: int = Field(default=EVAL_SUBSET, ge=1)
    device: str = "cuda"


class ForgetStatistics(BaseModel):
    """How a model answers the 3500 forget images, against the original's answers."""

    model_config = ConfigDict(frozen=True)

    agreement_with_original: float
    agreement_forgotten_classes: float
    agreement_partial_classes: float
    mass_on_forgotten_classes: float
    mean_confidence: float
    mean_confidence_forgotten_classes: float
    mean_confidence_partial_classes: float
    mean_entropy: float
    kl_original_to_model_t2: float
    kl_partial_classes_t2: float
    kl_forgotten_classes_t2: float
    loss_mean: float
    loss_std: float
    loss_mean_forgotten_classes: float
    loss_mean_partial_classes: float
    loss_target_distance: float
    mean_std_score_estimate: float
    sister_mass_forgotten_images: float
    sister_argmax_share_forgotten_images: float
    lantern_to_5_mass: float
    lantern_to_5_argmax_share: float
    boats_to_32_mass: float
    boats_to_32_argmax_share: float
    mean_entropy_t2: float
    tail_mass_beyond_top5_t1: float
    tail_mass_beyond_top5_t2: float
    true_prob_below_0_1_partial: float
    true_prob_above_0_9_partial: float
    true_prob_middle_partial: float
    true_prob_below_0_1_forgotten: float
    mean_entropy_partial_classes: float


LANTERN_CLASSES = (6, 7, 8)
LANTERN_SISTER = 5
BOAT_CLASSES = (29, 30, 31, 33)
BOAT_SISTER = 32
TAIL_TOP_K = 5
LOW_TRUE_PROB = 0.1
HIGH_TRUE_PROB = 0.9
"""Tail mass = probability left outside the `TAIL_TOP_K` most likely classes (flatness check)."""

GOLD_LOSS_MEAN_TARGET = 2.5
"""Mid-point of the 2.25-2.7 range that team C estimated for the gold's mean CE on exact images."""
GOLD_LOSS_STD_TARGET = 3.0
"""Team C's estimate of the gold's standard deviation of the per-image CE (hence the heterogeneity)."""


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row softmax in float64 with the max subtracted."""
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def forget_statistics(
    model_logits: np.ndarray,
    original_logits: np.ndarray,
    forgotten_classes: Sequence[int] = FULLY_FORGOTTEN_CLASSES,
) -> ForgetStatistics:
    """Statistics of `model_logits` on the forget images (rows aligned with `original_logits`).

    The original's argmax stands for the (unknown) true label: it is almost surely right on its
    own training images, and it is exactly the class the gold is not obliged to reproduce.
    """
    probabilities = softmax_np(model_logits)
    original_class = original_logits.argmax(axis=1)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    forgotten_mask = np.isin(original_class, np.asarray(forgotten_classes))
    partial_mask = ~forgotten_mask

    def mean_over(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    agree = predicted == original_class
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)
    p_original = softmax_np(original_logits, SCORING_TEMPERATURE)
    p_model = softmax_np(model_logits, SCORING_TEMPERATURE)
    kl = (
        p_original
        * (np.log(np.clip(p_original, 1e-12, None)) - np.log(np.clip(p_model, 1e-12, None)))
    ).sum(axis=1)
    # Per-image cross-entropy against the original's class (the score's loss L, up to the labels).
    loss = -np.log(
        np.clip(probabilities[np.arange(len(original_class)), original_class], 1e-12, None)
    )
    loss_mean, loss_std = float(loss.mean()), float(loss.std())
    distance = abs(loss_mean - GOLD_LOSS_MEAN_TARGET) + abs(loss_std - GOLD_LOSS_STD_TARGET)
    # The original's own distance to the target (its loss is ~0 everywhere): the official
    # mean/std score of the exact images is 1 - clip(d / d_original, 0, 1).
    original_loss = -np.log(
        np.clip(
            softmax_np(original_logits)[np.arange(len(original_class)), original_class], 1e-12, None
        )
    )
    original_distance = abs(float(original_loss.mean()) - GOLD_LOSS_MEAN_TARGET) + abs(
        float(original_loss.std()) - GOLD_LOSS_STD_TARGET
    )
    # Where do the images of the removed classes go? Sister classes = the other members of the
    # head neighbourhood that the gold did see.
    sister_mass, sister_argmax = [], []
    for row in np.nonzero(forgotten_mask)[0]:
        sisters = sister_classes(int(original_class[row]), forgotten_classes)
        if sisters:
            sister_mass.append(float(probabilities[row, sisters].sum()))
            sister_argmax.append(float(predicted[row] in sisters))
    p2 = softmax_np(model_logits, SCORING_TEMPERATURE)
    entropy_t2 = -(p2 * np.log(np.clip(p2, 1e-12, None))).sum(axis=1)

    def tail_mass(distribution: np.ndarray) -> float:
        top = np.sort(distribution, axis=1)[:, -TAIL_TOP_K:].sum(axis=1)
        return float((1.0 - top).mean())

    def routing(classes: Sequence[int], sister: int) -> tuple[float, float]:
        rows = np.nonzero(np.isin(original_class, np.asarray(classes)))[0]
        if len(rows) == 0:
            return float("nan"), float("nan")
        return float(probabilities[rows, sister].mean()), float((predicted[rows] == sister).mean())

    # Shape of p(original's class): a model that never saw the image is bimodal (sure or sure-not).
    p_true = probabilities[np.arange(len(original_class)), original_class]
    lantern_mass, lantern_argmax = routing(LANTERN_CLASSES, LANTERN_SISTER)
    boats_mass, boats_argmax = routing(BOAT_CLASSES, BOAT_SISTER)
    return ForgetStatistics(
        agreement_with_original=float(agree.mean()),
        agreement_forgotten_classes=mean_over(agree.astype(np.float64), forgotten_mask),
        agreement_partial_classes=mean_over(agree.astype(np.float64), partial_mask),
        mass_on_forgotten_classes=float(
            probabilities[:, list(forgotten_classes)].sum(axis=1).mean()
        ),
        mean_confidence=float(confidence.mean()),
        mean_confidence_forgotten_classes=mean_over(confidence, forgotten_mask),
        mean_confidence_partial_classes=mean_over(confidence, partial_mask),
        mean_entropy=float(entropy.mean()),
        kl_original_to_model_t2=float(kl.mean()),
        kl_partial_classes_t2=mean_over(kl, partial_mask),
        kl_forgotten_classes_t2=mean_over(kl, forgotten_mask),
        loss_mean=loss_mean,
        loss_std=loss_std,
        loss_mean_forgotten_classes=mean_over(loss, forgotten_mask),
        loss_mean_partial_classes=mean_over(loss, partial_mask),
        loss_target_distance=distance,
        mean_std_score_estimate=float(1.0 - np.clip(distance / original_distance, 0.0, 1.0)),
        sister_mass_forgotten_images=float(np.mean(sister_mass)) if sister_mass else float("nan"),
        sister_argmax_share_forgotten_images=(
            float(np.mean(sister_argmax)) if sister_argmax else float("nan")
        ),
        lantern_to_5_mass=lantern_mass,
        lantern_to_5_argmax_share=lantern_argmax,
        boats_to_32_mass=boats_mass,
        boats_to_32_argmax_share=boats_argmax,
        mean_entropy_t2=float(entropy_t2.mean()),
        tail_mass_beyond_top5_t1=tail_mass(probabilities),
        tail_mass_beyond_top5_t2=tail_mass(p2),
        true_prob_below_0_1_partial=mean_over(
            (p_true < LOW_TRUE_PROB).astype(np.float64), partial_mask
        ),
        true_prob_above_0_9_partial=mean_over(
            (p_true > HIGH_TRUE_PROB).astype(np.float64), partial_mask
        ),
        true_prob_middle_partial=mean_over(
            ((p_true >= LOW_TRUE_PROB) & (p_true <= HIGH_TRUE_PROB)).astype(np.float64),
            partial_mask,
        ),
        true_prob_below_0_1_forgotten=mean_over(
            (p_true < LOW_TRUE_PROB).astype(np.float64), forgotten_mask
        ),
        mean_entropy_partial_classes=mean_over(entropy, partial_mask),
    )


def augment_batch(images_uint8: Tensor, generator: torch.Generator, min_area: float) -> Tensor:
    """uint8 [B, 224, 224, 3] on device -> float [B, 3, 224, 224] in [-1, 1], augmented.

    Random resized crop (area in [min_area, 1], aspect 3/4..4/3, one `roi_align` call for the
    whole batch), horizontal flip, brightness and contrast jitter. Deterministic given `generator`.
    """
    batch = images_uint8.shape[0]
    device = images_uint8.device
    pixels = images_uint8.permute(0, 3, 1, 2).float() / PIXEL_MAX

    def uniform(low: float, high: float) -> Tensor:
        return low + (high - low) * torch.rand(batch, generator=generator, device="cpu").to(device)

    area = uniform(min_area, 1.0)
    log_ratio = uniform(math.log(ASPECT_RATIO_RANGE[0]), math.log(ASPECT_RATIO_RANGE[1]))
    ratio = log_ratio.exp()
    width = (area * ratio).sqrt().clamp(max=1.0) * IMAGE_SIZE
    height = (area / ratio).sqrt().clamp(max=1.0) * IMAGE_SIZE
    left = uniform(0.0, 1.0) * (IMAGE_SIZE - width)
    top = uniform(0.0, 1.0) * (IMAGE_SIZE - height)
    boxes = torch.stack([left, top, left + width, top + height], dim=1)
    crops = roi_align(
        pixels,
        list(boxes.unsqueeze(1).unbind(0)),  # one [1, 4] box per image
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
        spatial_scale=1.0,
        sampling_ratio=ROI_SAMPLING_RATIO,
        aligned=True,
    )
    flip = (torch.rand(batch, generator=generator, device="cpu") < 0.5).to(device)
    crops = torch.where(flip.view(-1, 1, 1, 1), crops.flip(3), crops)
    brightness = uniform(*JITTER_RANGE).view(-1, 1, 1, 1)
    contrast = uniform(*JITTER_RANGE).view(-1, 1, 1, 1)
    mean = crops.mean(dim=(1, 2, 3), keepdim=True)
    jittered = ((crops - mean) * contrast + mean) * brightness
    return (jittered.clamp(0.0, 1.0) - 0.5) / 0.5


class ProxyData:
    """The assembled dataset, resident on the device as uint8."""

    def __init__(self, path: Path, device: torch.device) -> None:
        if not path.is_file():
            raise FinetuneError(f"Dataset not found: {path}")
        with np.load(path) as archive:
            images = archive["images"]
            labels = archive["labels"]
        if len(images) != len(labels) or len(images) == 0:
            raise FinetuneError(f"Inconsistent or empty dataset: {path}")
        if int(labels.min()) < 0 or int(labels.max()) >= NUM_CLASSES:
            raise FinetuneError("Labels outside [0, 99].")
        self.images = torch.from_numpy(images).to(device)
        self.labels = torch.from_numpy(labels).to(device)

    def __len__(self) -> int:
        return len(self.labels)

    def hold_out(self, fraction: float, seed: int) -> tuple[Tensor, Tensor]:
        """Remove a seeded `fraction` of the images from training; return (images, labels) of it."""
        count = int(len(self) * fraction)
        if count == 0:
            return self.images[:0], self.labels[:0]
        order = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        held, kept = order[:count].to(self.images.device), order[count:].to(self.images.device)
        held_images, held_labels = self.images[held], self.labels[held]
        self.images, self.labels = self.images[kept], self.labels[kept]
        return held_images, held_labels


class HoldoutStatistics(BaseModel):
    """Drift of a model on held-out proxy images (the stand-in of the hidden control set).

    The server's retain score is `clip(acc / acc_orig, 0, 1) / (1 + KL(orig || sub))` at T = 2 on
    in-distribution images; here the labels are the original's own answers, so `acc_orig` is 1.
    """

    model_config = ConfigDict(frozen=True)

    kl_original_to_model_t2: float
    accuracy: float
    mean_confidence: float
    retain_estimate: float


def holdout_statistics(
    model_logits: np.ndarray, original_logits: np.ndarray, labels: np.ndarray
) -> HoldoutStatistics:
    """KL(orig || model) at T = 2, accuracy against `labels`, confidence and the retain estimate."""
    p_original = softmax_np(original_logits, SCORING_TEMPERATURE)
    p_model = softmax_np(model_logits, SCORING_TEMPERATURE)
    kl = float(
        (
            p_original
            * (np.log(np.clip(p_original, 1e-12, None)) - np.log(np.clip(p_model, 1e-12, None)))
        )
        .sum(axis=1)
        .mean()
    )
    accuracy = float((model_logits.argmax(axis=1) == labels).mean())
    return HoldoutStatistics(
        kl_original_to_model_t2=kl,
        accuracy=accuracy,
        mean_confidence=float(softmax_np(model_logits).max(axis=1).mean()),
        retain_estimate=float(min(accuracy, 1.0) / (1.0 + kl)),
    )


def _load_forget(path: Path) -> Tensor:
    with np.load(path) as archive:
        return torch.from_numpy(archive[FORGET_IMAGES_KEY])


def _loss(
    student: nn.Module,
    teacher: nn.Module | None,
    images: Tensor,
    labels: Tensor,
    config: FinetuneConfig,
) -> Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
        logits = student(images).float()
    loss = config.ce_weight * torch_f.cross_entropy(
        logits, labels, label_smoothing=config.label_smoothing
    )
    if config.kd_weight > 0 and teacher is not None:
        with torch.no_grad(), server_autocast(images.device):
            teacher_logits = teacher(images).float()
        loss = loss + config.kd_weight * kd_from_teacher(
            teacher_logits, logits, config.kd_temperature
        )
    return loss


def run_finetune(
    config: FinetuneConfig,
    *,
    original: nn.Module | None = None,
    forget_images: Tensor | None = None,
) -> dict[str, object]:
    """Train one variant; write `<out_dir>/<name>/{config,history}.json`, logits and (optionally) the model.

    `original` and `forget_images` are injectable for tests; the CLI loads them from disk.
    """
    device = torch.device(config.device)
    started = time.time()
    run_dir = config.out_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=1))

    base = original if original is not None else load_model(config.original_path, device)
    base = base.to(device).eval()
    forget = forget_images if forget_images is not None else _load_forget(config.forget_path)
    original_forget_logits = predict(base, forget, PREDICT_BATCH).numpy()
    data = ProxyData(config.dataset_path, device)
    holdout_images, holdout_labels = data.hold_out(config.holdout_fraction, config.seed)
    holdout_labels_np = holdout_labels.cpu().numpy()
    has_holdout = len(holdout_labels_np) > 0
    original_holdout_logits = (
        predict(base, holdout_images, PREDICT_BATCH).numpy() if has_holdout else None
    )

    teacher: nn.Module | None = None
    if config.kd_weight > 0:
        teacher = copy.deepcopy(base).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    student = copy.deepcopy(base).train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    probe = make_probe(device)
    cuda = device.type == "cuda"
    sizer = BatchSizer(
        size=min(config.start_micro_batch, config.max_micro_batch, config.effective_batch),
        maximum=min(config.max_micro_batch, config.effective_batch),
        target_gb=config.target_memory_gb,
        cap_gb=device_budget_gb(device, config.memory_margin_gb) if cuda else 1e9,
        weights_gb=torch.cuda.memory_allocated(device) / 1e9 if cuda else 0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    steps_per_epoch = max(1, math.ceil(len(data) / config.effective_batch))
    total_steps = steps_per_epoch * config.epochs
    history: list[dict[str, object]] = []
    step = 0
    for epoch in range(config.epochs):
        order = torch.randperm(len(data), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(data), config.effective_batch):
            batch_index = order[start : start + config.effective_batch].to(device)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, total_steps, config.lr, config.warmup_fraction)
            step_loss = _accumulated_step(
                student, teacher, optimizer, data, batch_index, generator, config, sizer, probe
            )
            epoch_loss += step_loss
            step += 1
        student.eval()
        logits = predict(student, forget, PREDICT_BATCH).numpy()
        student.train()
        stats = forget_statistics(logits, original_forget_logits)
        entry: dict[str, object] = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": epoch_loss / steps_per_epoch,
            "micro_batch": sizer.size,
            "peak_memory_gb": max((peak for _, peak in sizer.history), default=None),
            "oom_events": sizer.oom_events,
            "seconds": time.time() - started,
            "forget": json.loads(stats.model_dump_json()),
        }
        if original_holdout_logits is not None:
            student.eval()
            held = predict(student, holdout_images, PREDICT_BATCH).numpy()
            student.train()
            entry["holdout"] = json.loads(
                holdout_statistics(
                    held, original_holdout_logits, holdout_labels_np
                ).model_dump_json()
            )
        history.append(entry)
        print(json.dumps(entry), flush=True)
    np.savez(run_dir / "forget_logits.npz", model=logits, original=original_forget_logits)
    if config.save_model:
        student.eval()
        torch.save(
            {k: v.detach().cpu().float() for k, v in student.state_dict().items()},
            run_dir / "model.pt",
        )
    exported: list[str] = []
    blends: list[dict[str, object]] = []
    if config.export_logits_dir is not None:
        holdout = (
            (holdout_images, holdout_labels_np, original_holdout_logits)
            if original_holdout_logits is not None
            else None
        )
        paths, blends = _export_team_c_logits(
            config, base, student, forget, original_forget_logits, holdout
        )
        exported = [str(path) for path in paths]
        (run_dir / "blends.json").write_text(json.dumps(blends, indent=1))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {
        "name": config.name,
        "history": history,
        "seconds": time.time() - started,
        "exported_logits": exported,
        "blends": blends,
    }


def _export_team_c_logits(
    config: FinetuneConfig,
    base: nn.Module,
    student: nn.Module,
    forget: Tensor,
    original_forget_logits: np.ndarray,
    holdout: tuple[Tensor, np.ndarray, np.ndarray] | None,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Team C's per-view logits of `original + alpha * (student - original)` for each alpha.

    Also measures every blend (forget statistics and held-out proxy drift), so the alpha that keeps
    KL(orig || sub) under the team's ceiling can be read without another run.
    """
    assert config.export_logits_dir is not None
    indices = subset_indices(len(forget), config.export_subset_size)
    views = build_views(forget[indices])
    base_state = {k: v.detach().cpu().float() for k, v in base.state_dict().items()}
    student_state = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
    blended_model = copy.deepcopy(base).eval()
    paths: list[Path] = []
    blends: list[dict[str, object]] = []
    for alpha in config.export_alphas:
        blended_model.load_state_dict(blend_state_dicts(base_state, student_state, alpha))
        name = f"d_{config.name}_a{alpha:.2f}"
        paths.append(export_model(blended_model, name, config.export_logits_dir, indices, views))
        forget_logits = predict(blended_model, forget, PREDICT_BATCH).numpy()
        record: dict[str, object] = {
            "name": name,
            "alpha": alpha,
            "forget": json.loads(
                forget_statistics(forget_logits, original_forget_logits).model_dump_json()
            ),
        }
        if holdout is not None:
            images, labels, original_logits = holdout
            held = predict(blended_model, images, PREDICT_BATCH).numpy()
            record["holdout"] = json.loads(
                holdout_statistics(held, original_logits, labels).model_dump_json()
            )
        blends.append(record)
        print(json.dumps(record), flush=True)
    return paths, blends


def _accumulated_step(
    student: nn.Module,
    teacher: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    data: ProxyData,
    batch_index: Tensor,
    generator: torch.Generator,
    config: FinetuneConfig,
    sizer: BatchSizer,
    probe: MemoryProbe,
) -> float:
    """One optimiser step over `batch_index`, in micro-batches; retries whole on OOM."""
    oom_errors: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
    while True:
        optimizer.zero_grad(set_to_none=True)
        probe.reset_peak()
        total = 0.0
        try:
            for start in range(0, len(batch_index), sizer.size):
                chunk = batch_index[start : start + sizer.size]
                images = augment_batch(data.images[chunk], generator, config.crop_min_area)
                loss = _loss(student, teacher, images, data.labels[chunk], config)
                weight = len(chunk) / len(batch_index)
                (loss * weight).backward()  # type: ignore[no-untyped-call]  # torch stub gap
                total += float(loss.detach()) * weight
            break
        except oom_errors:
            probe.release_cache()
            optimizer.zero_grad(set_to_none=True)
            sizer.after_oom()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    sizer.after_success(min(sizer.size, len(batch_index)), probe.peak_gb())
    if not math.isfinite(total):
        raise FinetuneError("Non-finite loss.")
    return total


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run one variant from a JSON file of `FinetuneConfig` fields (plus CLI overrides)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    fields = json.loads(args.config_json.read_text())
    for override in args.set:
        key, _, value = override.partition("=")
        fields[key] = json.loads(value)
    fields["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config = FinetuneConfig(**fields)
    run_finetune(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
