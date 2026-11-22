"""Few-shot unlearning as meta-learning (team G): every deletion request is a task.

A *task* is a handful of forget images (1, 4, 16 or 64) with their per-image targets (`g_targets`).
**Inner loop**: k optimiser steps on the task (exact images and one fresh altered copy of each,
because half of the forget score lives on hidden altered copies) plus a retention anchor, with a
per-image BUDGET: an image stops contributing as soon as its distance to its own target has fallen
to `budget_fraction` of its initial distance. **Outer loop** (Reptile, Nichol et al. 2018):
`theta <- theta + epsilon * (theta_k - theta)`. Streaming the tasks over the whole forget set gives
the candidate; the same code on a held-out part gives the few-shot measurements; the same code from
the original on N random images gives the capacity of the operator (steps, seconds, drift).

What this module does NOT claim: a meta-learner does not create information about the gold. The
targets carry every belief about it; the loop only decides how faithfully, how cheaply and how
locally the student reaches them. Fidelity, drift and cost are therefore measured, not assumed.

Memory: images go through the student by micro-batches of `micro_batch` (gradients accumulate), so
a 64-image task fits next to other workers on one 40 GB GPU.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as torch_f
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.constants import DEFAULT_SEED, SCORING_TEMPERATURE
from hackathon.unlearning.model import preprocess
from hackathon.unlearning.training import (
    kd_from_teacher,
    teacher_logits,
    temper_probabilities,
    training_autocast,
)

LOGGER = logging.getLogger("hackathon.unlearning.research.g")

NATIVE_TEMPERATURE = 1.0
PROBABILITY_FLOOR = 1e-12
BLOCKS_ATTRIBUTE = "blocks"
LAST_BLOCKS_TRAINED = 4
ADAM_BETAS = (0.9, 0.999)
MICRO_BATCH_CAP_UNBOUNDED = 1 << 30
SELF_ANCHOR_BOX_AREA = (0.3, 0.6)
"""Area range of the pasted box of a CutMix anchor image (fraction of the image)."""
SELF_ANCHOR_MASK_RATIO = (0.2, 0.4)
"""Fraction of the 16x16 patches replaced by gray in a masked anchor image."""
PATCH_SIZE = 16
MASK_GRAY = 128
InnerOptimizer = Literal["sgd", "adam"]
AnchorMode = Literal["self", "pool"]
TrainableMode = Literal["all", "last4", "head"]


_micro_batch_cap = MICRO_BATCH_CAP_UNBOUNDED


def halve_micro_batch_cap(current: int) -> int:
    """Lower the process-wide micro-batch ceiling to half of `current` (>= 1); return the new cap.

    Called after a CUDA out-of-memory: the same task then runs with half the images per backward
    (gradients accumulate, so the update is unchanged, only slower).
    """
    global _micro_batch_cap
    _micro_batch_cap = max(1, min(current, _micro_batch_cap) // 2)
    return _micro_batch_cap


def reset_micro_batch_cap() -> None:
    """Remove the ceiling (tests, and the start of a fresh job)."""
    global _micro_batch_cap
    _micro_batch_cap = MICRO_BATCH_CAP_UNBOUNDED


class MetaLearningError(HackathonError):
    """A task, a spec or a plan is inconsistent (empty task, diverged loss, bad sizes)."""


class MetaSpec(BaseModel):
    """Every knob of one streamed run (JSON-friendly, frozen)."""

    model_config = ConfigDict(frozen=True)

    name: str
    seed: int = DEFAULT_SEED
    task_sizes: tuple[int, ...] = (16,)
    inner_optimizer: InnerOptimizer = "adam"
    inner_lr: float = Field(default=2e-5, gt=0.0)
    inner_steps: int = Field(default=3, ge=1)
    budget_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    outer_lr: float = Field(default=1.0, gt=0.0, le=1.0)
    passes: int = Field(default=1, ge=1)
    retain_batch: int = Field(default=8, ge=0)
    retain_weight: float = Field(default=1.0, ge=0.0)
    altered_weight: float = Field(default=1.0, ge=0.0)
    strong_weight: float = Field(default=1.0, ge=0.0)
    strong_min_area: float = Field(default=0.35, gt=0.0, lt=1.0)
    holdout_count: int = Field(default=0, ge=0)
    trainable: TrainableMode = "all"
    anchor: AnchorMode = Field(
        default="self",
        description="'self': anchors derived from the forget set only (CutMix, masking); 'pool': external images.",
    )
    persistent_optimizer: bool = Field(
        default=False,
        description="Keep one optimiser (and its moments) across tasks: ordinary training.",
    )
    micro_batch: int = Field(default=16, ge=1)
    grad_clip: float | None = Field(default=5.0, gt=0.0)
    monitor_every: int = Field(default=16, ge=0)

    @model_validator(mode="after")
    def _sizes_are_positive(self) -> MetaSpec:
        if not self.task_sizes or min(self.task_sizes) < 1:
            raise ValueError("task_sizes must be a non-empty tuple of positive integers")
        return self


@dataclass
class AdaptRecord:
    """What one inner loop did (all KL values are the matching KL of the task images, exact copies)."""

    task_size: int
    steps: int = 0
    kl_per_step: list[float] = field(default_factory=list)
    active_per_step: list[int] = field(default_factory=list)
    grad_norm_per_step: list[float] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class MetaContext:
    """The live objects of a run; `targets` and `forget_images` share the same row order."""

    device: torch.device
    student: nn.Module
    teacher: nn.Module
    forget_images: Tensor
    targets: Tensor
    retain_pool: Tensor
    params: list[nn.Parameter] = field(default_factory=list)
    targets_altered: Tensor | None = None
    optimizer: torch.optim.Optimizer | None = None


def per_sample_matching_kl(target: Tensor, logits: Tensor) -> Tensor:
    """Per-sample distance to a target: the mean of KL at T = 1 and T^2 * KL at the scoring T = 2.

    Same objective as the research trainer (`_matching_loss`, batch-mean there), kept per sample
    here so that the budget can switch single images off.
    """
    log_target = target.clamp_min(PROBABILITY_FLOOR).log()
    native = (target * (log_target - torch_f.log_softmax(logits, dim=1))).sum(dim=1)
    tempered = temper_probabilities(target, SCORING_TEMPERATURE)
    tempered_kl = (
        tempered
        * (
            tempered.clamp_min(PROBABILITY_FLOOR).log()
            - torch_f.log_softmax(logits / SCORING_TEMPERATURE, dim=1)
        )
    ).sum(dim=1) * SCORING_TEMPERATURE**2
    return 0.5 * (native + tempered_kl)


def set_trainable(model: nn.Module, mode: TrainableMode) -> list[nn.Parameter]:
    """Freeze everything but the parameters of `mode`; return the trainable ones.

    `all`: every weight. `last4`: the last four blocks, the final norm and the head.
    `head`: the classifier only (the cheapest, most local operator).
    """
    blocks = getattr(model, BLOCKS_ATTRIBUTE)
    first_block = len(blocks) - LAST_BLOCKS_TRAINED
    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        top = name.split(".")[0]
        if mode == "all":
            keep = True
        elif mode == "head":
            keep = top == "head"
        else:
            keep = int(name.split(".")[1]) >= first_block if top == BLOCKS_ATTRIBUTE else top in (
                "norm", "fc_norm", "head"
            )  # fmt: skip
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(parameter)
    if not trainable:
        raise MetaLearningError(f"no trainable parameter for mode '{mode}'")
    return trainable


def student_logits(student: nn.Module, images: Tensor, device: torch.device) -> Tensor:
    """Float32 logits with gradient (bfloat16 autocast on CUDA, fp32 master weights)."""
    batch = preprocess(images, device)
    with training_autocast(device, True):
        logits: Tensor = student(batch)
    return logits.float()


def make_optimizer(params: Sequence[nn.Parameter], spec: MetaSpec) -> torch.optim.Optimizer:
    """A fresh inner optimiser (no state carried between tasks: a task is self-contained)."""
    if spec.inner_optimizer == "sgd":
        return torch.optim.SGD(params, lr=spec.inner_lr, momentum=0.0)
    return torch.optim.Adam(params, lr=spec.inner_lr, betas=ADAM_BETAS, weight_decay=0.0)


def _chunks(count: int, size: int) -> list[slice]:
    return [slice(start, min(start + size, count)) for start in range(0, count, size)]


def build_self_anchor(images: Tensor, generator: torch.Generator, count: int) -> Tensor:
    """Anchor images derived ONLY from the forget set: CutMix of two images, or patch masking.

    The original's behaviour on such inputs is what the student must keep: they look unlike any
    single forget image, so anchoring them does not pull a processed forget image back toward the
    original (unlike a strong crop of that very image, whose target is deliberately different).
    Half of the batch is CutMix (a random box of another forget image pasted in, 30-60 % of the
    area), half is masking (20-40 % of the 16x16 patches replaced by gray).

    Args:
        images: uint8 [N, 224, 224, 3] forget images (the only source).
        generator: Seeded CPU generator (the draw is reproducible).
        count: Number of anchor images.

    Raises:
        MetaLearningError: no source image or a non-positive count.
    """
    if len(images) == 0 or count < 1:
        raise MetaLearningError("a self anchor needs source images and a positive count")
    height, width = int(images.shape[1]), int(images.shape[2])
    first = torch.randint(len(images), (count,), generator=generator)
    second = torch.randint(len(images), (count,), generator=generator)
    out = images[first].clone()
    cutmix = count - count // 2
    for row in range(cutmix):
        area = SELF_ANCHOR_BOX_AREA[0] + (
            SELF_ANCHOR_BOX_AREA[1] - SELF_ANCHOR_BOX_AREA[0]
        ) * float(torch.rand(1, generator=generator))
        box_h = max(1, round(height * area**0.5))
        box_w = max(1, round(width * area**0.5))
        top = int(torch.randint(height - box_h + 1, (1,), generator=generator))
        left = int(torch.randint(width - box_w + 1, (1,), generator=generator))
        out[row, top : top + box_h, left : left + box_w] = images[
            second[row], top : top + box_h, left : left + box_w
        ]
    patches_h, patches_w = height // PATCH_SIZE, width // PATCH_SIZE
    for row in range(cutmix, count):
        ratio = SELF_ANCHOR_MASK_RATIO[0] + (
            SELF_ANCHOR_MASK_RATIO[1] - SELF_ANCHOR_MASK_RATIO[0]
        ) * float(torch.rand(1, generator=generator))
        hidden = torch.rand(patches_h, patches_w, generator=generator) < ratio
        mask = hidden.repeat_interleave(PATCH_SIZE, dim=0).repeat_interleave(PATCH_SIZE, dim=1)
        out[row][mask] = MASK_GRAY
    return out


def _retain_step(ctx: MetaContext, spec: MetaSpec, generator: torch.Generator) -> float:
    """Backward of the retention anchor (KD to the original at T = 2); returns its value."""
    if spec.retain_batch == 0 or spec.retain_weight == 0.0:
        return 0.0
    if spec.anchor == "self":
        batch = build_self_anchor(ctx.forget_images, generator, spec.retain_batch)
    else:
        if len(ctx.retain_pool) == 0:
            return 0.0
        picked = torch.randint(len(ctx.retain_pool), (spec.retain_batch,), generator=generator)
        batch = random_augment(ctx.retain_pool[picked], generator)
    reference = teacher_logits(ctx.teacher, batch, ctx.device)
    loss = spec.retain_weight * kd_from_teacher(
        reference, student_logits(ctx.student, batch, ctx.device), SCORING_TEMPERATURE
    )
    torch.autograd.backward(loss)
    return float(loss.item())


def adapt_task(
    ctx: MetaContext, indices: Tensor, spec: MetaSpec, generator: torch.Generator
) -> AdaptRecord:
    """Inner loop on the forget images `indices`, then the Reptile interpolation of the weights.

    The student is modified in place. With `outer_lr == 1` the result is plain sequential
    fine-tuning; with `outer_lr < 1` only that fraction of the inner displacement is kept.

    Raises:
        MetaLearningError: empty task or a non-finite loss (lower `inner_lr`).
    """
    count = len(indices)
    if count == 0:
        raise MetaLearningError("a task needs at least one image")
    started = time.time()
    ctx.student.train()
    snapshot = [parameter.detach().clone() for parameter in ctx.params]
    if spec.persistent_optimizer:
        if ctx.optimizer is None:
            ctx.optimizer = make_optimizer(ctx.params, spec)
        optimizer = ctx.optimizer
    else:
        optimizer = make_optimizer(ctx.params, spec)
    images = ctx.forget_images[indices]
    targets = ctx.targets[indices].to(ctx.device)
    strong_targets = (
        ctx.targets_altered[indices].to(ctx.device) if ctx.targets_altered is not None else targets
    )
    tolerance = torch.zeros(count, device=ctx.device)
    record = AdaptRecord(task_size=count)
    weight_total = 1.0 + spec.altered_weight + spec.strong_weight
    for step in range(spec.inner_steps):
        optimizer.zero_grad(set_to_none=True)
        altered = random_augment(images, generator)
        strong = strong_view(images, generator, spec.strong_min_area)
        active_total = 0
        kl_total = 0.0
        for window in _chunks(count, min(spec.micro_batch, _micro_batch_cap)):
            exact_kl = per_sample_matching_kl(
                targets[window], student_logits(ctx.student, images[window], ctx.device)
            )
            if step == 0:
                tolerance[window] = spec.budget_fraction * exact_kl.detach()
                active = torch.ones_like(exact_kl)
            else:
                active = (exact_kl.detach() > tolerance[window]).float()
            altered_kl = per_sample_matching_kl(
                targets[window], student_logits(ctx.student, altered[window], ctx.device)
            )
            strong_kl = per_sample_matching_kl(
                strong_targets[window], student_logits(ctx.student, strong[window], ctx.device)
            )
            combined = exact_kl + spec.altered_weight * altered_kl + spec.strong_weight * strong_kl
            chunk_loss = (active * combined).sum() / (weight_total * count)
            if not torch.isfinite(chunk_loss):
                raise MetaLearningError("non-finite forget loss: lower inner_lr")
            torch.autograd.backward(chunk_loss)
            active_total += int(active.sum().item())
            kl_total += float(exact_kl.detach().sum().item())
        record.kl_per_step.append(kl_total / count)
        record.active_per_step.append(active_total)
        if step > 0 and active_total == 0:
            break
        _retain_step(ctx, spec, generator)
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                ctx.params, spec.grad_clip if spec.grad_clip is not None else float("inf")
            )
        )
        record.grad_norm_per_step.append(norm)
        optimizer.step()
        record.steps += 1
    optimizer.zero_grad(set_to_none=True)
    if spec.outer_lr < 1.0:
        with torch.no_grad():
            for parameter, origin in zip(ctx.params, snapshot, strict=True):
                parameter.copy_(origin + spec.outer_lr * (parameter - origin))
    record.seconds = time.time() - started
    return record


def build_task_plan(
    indices: Tensor, task_sizes: Sequence[int], generator: torch.Generator
) -> list[Tensor]:
    """Consecutive tasks over a seeded permutation of `indices`, sizes drawn from `task_sizes`.

    Every index appears in exactly one task; the last task is truncated to what is left.
    """
    if len(indices) == 0:
        raise MetaLearningError("no image to plan")
    order = indices[torch.randperm(len(indices), generator=generator)]
    sizes = torch.tensor(list(task_sizes))
    plan: list[Tensor] = []
    cursor = 0
    while cursor < len(order):
        size = int(sizes[int(torch.randint(len(sizes), (1,), generator=generator))])
        plan.append(order[cursor : cursor + size])
        cursor += size
    return plan


@dataclass
class StreamResult:
    """Outcome of a streamed run: per-task records and the monitor history."""

    records: list[AdaptRecord] = field(default_factory=list)
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


Monitor = Callable[[int, list[Tensor]], dict[str, float]]
"""`monitor(tasks_done, processed_task_indices) -> metrics` (called every `monitor_every` tasks)."""


def run_stream(
    ctx: MetaContext, indices: Tensor, spec: MetaSpec, monitor: Monitor | None = None
) -> StreamResult:
    """Process every image of `indices` as a stream of tasks, `spec.passes` times.

    The first monitor call (before any task) records the starting point; the last one follows the
    final task of each pass.
    """
    generator = torch.Generator().manual_seed(spec.seed)
    result = StreamResult()
    started = time.time()
    done: list[Tensor] = []
    task_counter = 0
    if monitor is not None:
        result.history.append({"tasks": 0.0, "pass": 0.0, **monitor(0, done)})
    for pass_index in range(spec.passes):
        plan = build_task_plan(indices, spec.task_sizes, generator)
        for task in plan:
            result.records.append(adapt_task(ctx, task, spec, generator))
            done.append(task)
            task_counter += 1
            due = spec.monitor_every > 0 and task_counter % spec.monitor_every == 0
            if monitor is not None and (due or task is plan[-1]):
                metrics = monitor(task_counter, done)
                result.history.append(
                    {"tasks": float(task_counter), "pass": float(pass_index + 1), **metrics}
                )
                LOGGER.info("%s tasks %d: %s", spec.name, task_counter, result.history[-1])
    result.seconds = time.time() - started
    return result


def holdout_split(count: int, holdout: int, seed: int) -> tuple[Tensor, Tensor]:
    """Seeded split of range(count) into (meta-train, held-out) index tensors, both sorted."""
    if holdout < 0 or holdout >= count:
        raise MetaLearningError(f"holdout must lie in [0, {count}), got {holdout}")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    held = order[:holdout].sort().values
    train = order[holdout:].sort().values
    return train, held
