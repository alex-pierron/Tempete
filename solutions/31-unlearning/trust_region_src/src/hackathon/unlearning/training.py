"""Building blocks shared by the training-based unlearning methods (pure helpers, easy to test)."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.data import class_grouped_order
from hackathon.unlearning.model import preprocess, server_autocast

LOG_PROBABILITY_FLOOR = 1e-12
"""Floor applied before log() of a probability vector (a target can contain exact zeros)."""
ALWAYS_TRAINED_PREFIXES = ("norm", "fc_norm", "head")
"""Parameters outside the transformer blocks that stay trainable when blocks are frozen."""
BLOCKS_ATTRIBUTE = "blocks"


def learning_rate_at(step: int, total_steps: int, peak_lr: float, warmup_fraction: float) -> float:
    """Linear warmup to `peak_lr`, then cosine decay to zero; `step` is 0-indexed.

    Args:
        step: Index of the optimiser step about to run.
        total_steps: Total number of optimiser steps of the run (>= 1).
        peak_lr: Peak learning rate.
        warmup_fraction: Fraction of `total_steps` spent warming up (0 disables warmup).
    """
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * peak_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def steps_per_epoch(sample_count: int, config: UnlearnConfig) -> int:
    """Optimiser steps in one pass over `sample_count` images (last partial batch kept)."""
    full_pass = math.ceil(sample_count / config.batch_size)
    if config.max_steps_per_epoch is None:
        return full_pass
    return max(1, min(full_pass, config.max_steps_per_epoch))


def epoch_batches(
    sample_count: int, batch_size: int, steps: int, generator: torch.Generator
) -> Iterator[Tensor]:
    """Yield `steps` index batches from a fresh permutation (wraps around if steps need more)."""
    emitted = 0
    while emitted < steps:
        order = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            if emitted >= steps:
                return
            yield order[start : start + batch_size]
            emitted += 1


def stratified_batches(
    labels: Tensor, batch_size: int, steps: int, generator: torch.Generator
) -> Iterator[Tensor]:
    """Like `epoch_batches`, but batch i takes every n-th index of the class-grouped order.

    Each batch therefore spans the labels in the proportions of the whole set, so no step pushes
    the classifier head toward a few classes.
    """
    batch_count = math.ceil(len(labels) / batch_size)
    emitted = 0
    while emitted < steps:
        order = class_grouped_order(labels, generator)
        for start in torch.randperm(batch_count, generator=generator).tolist():
            if emitted >= steps:
                return
            yield order[start::batch_count]
            emitted += 1


def prepare_teacher(model: nn.Module, device: torch.device) -> nn.Module:
    """Frozen eval-mode copy of the original model (the reference of every target)."""
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def prepare_student(model: nn.Module, device: torch.device, config: UnlearnConfig) -> nn.Module:
    """Trainable copy of `model`, honouring `trainable_blocks` and `grad_checkpointing`."""
    student = copy.deepcopy(model).to(device).train()
    if config.trainable_blocks >= 0:
        blocks = getattr(student, BLOCKS_ATTRIBUTE)
        frozen_until = len(blocks) - config.trainable_blocks
        for name, parameter in student.named_parameters():
            top = name.split(".")[0]
            if top == BLOCKS_ATTRIBUTE:
                parameter.requires_grad_(int(name.split(".")[1]) >= frozen_until)
            else:
                parameter.requires_grad_(top in ALWAYS_TRAINED_PREFIXES)
    enable_checkpointing = getattr(student, "set_grad_checkpointing", None)
    if config.grad_checkpointing and callable(enable_checkpointing):
        enable_checkpointing(True)
    return student


def training_autocast(device: torch.device, enabled: bool) -> AbstractContextManager[None]:
    """bfloat16 autocast on CUDA for training (fp32 master weights); no-op elsewhere."""
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def student_logits(
    student: nn.Module, images_uint8: Tensor, device: torch.device, config: UnlearnConfig
) -> Tensor:
    """Float32 logits of the student with gradient."""
    batch = preprocess(images_uint8, device)
    with training_autocast(device, config.amp):
        logits: Tensor = student(batch)
    return logits.float()


@torch.no_grad()
def teacher_logits(teacher: nn.Module, images_uint8: Tensor, device: torch.device) -> Tensor:
    """Float32 logits of the teacher, computed like the server (fp16 autocast on CUDA)."""
    batch = preprocess(images_uint8, device)
    with server_autocast(device):
        logits: Tensor = teacher(batch)
    return logits.float()


def temper_probabilities(probabilities: Tensor, temperature: float) -> Tensor:
    """softmax(log(p) / T): the distribution p would have if its logits were divided by T."""
    log_p = probabilities.clamp_min(LOG_PROBABILITY_FLOOR).log()
    return torch_f.softmax(log_p / temperature, dim=1)


def kl_to_student(target_probs: Tensor, logits: Tensor, temperature: float) -> Tensor:
    """T^2 * KL(target || softmax(logits / T)), batch mean (target is already at temperature T)."""
    return (
        torch_f.kl_div(
            torch_f.log_softmax(logits / temperature, dim=1), target_probs, reduction="batchmean"
        )
        * temperature
        * temperature
    )


def kd_from_teacher(teacher_logits_value: Tensor, logits: Tensor, temperature: float) -> Tensor:
    """T^2 * KL(softmax(z_teacher / T) || softmax(z_student / T)) (the SCRUB distance)."""
    return kl_to_student(
        torch_f.softmax(teacher_logits_value / temperature, dim=1), logits, temperature
    )
