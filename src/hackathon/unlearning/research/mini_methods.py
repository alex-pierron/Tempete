"""The kit's unlearning methods, transcribed for the miniature CNN, plus a few variants.

`identity`, `scrub` and `gold_soft` follow `hackathon.unlearning.methods` step for step (same loss
terms, same cap, same warmup + cosine schedule, same retain anchor on out-of-distribution proxy
images); only the scale changes (a CNN instead of a ViT-B, `steps` instead of epochs, a larger
learning rate because the net is trained from scratch). The pure loss helpers are imported from the
kit (`hackathon.unlearning.training`), not re-implemented, so a bug there would show here too.

Extra methods, there to give the correlation study a spread of behaviours:

* `grad_ascent`   naive gradient ascent on the pseudo-label (what the subject says NOT to do);
* `uniform`       distil towards the uniform distribution on forget images (pure confusion);
* `finetune_retain`  only the retain-anchor term (moves nothing on forget by design);
* `weight_noise`  Gaussian noise on the weights (a random perturbation baseline).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.research.mini_core import (
    MiniNet,
    random_alteration,
    strong_view,
)
from hackathon.unlearning.training import (
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    temper_probabilities,
)

METHOD_IDENTITY = "identity"
METHOD_SCRUB = "scrub"
METHOD_GOLD_SOFT = "gold_soft"
METHOD_GRAD_ASCENT = "grad_ascent"
METHOD_UNIFORM = "uniform"
METHOD_FINETUNE_RETAIN = "finetune_retain"
METHOD_WEIGHT_NOISE = "weight_noise"
METHOD_NAMES = (
    METHOD_IDENTITY,
    METHOD_SCRUB,
    METHOD_GOLD_SOFT,
    METHOD_GRAD_ASCENT,
    METHOD_UNIFORM,
    METHOD_FINETUNE_RETAIN,
    METHOD_WEIGHT_NOISE,
)
NATIVE_TEMPERATURE = 1.0
"""Temperature of the per-sample cross-entropy (mean / std score), as in the kit."""
GRADIENT_CLIP_NORM = 1.0
FIXED_TARGET_SEED_OFFSET = 7000
FIXED_TARGET_CHUNK = 500
WARMUP_FRACTION = 0.1
DEFAULT_LEARNING_RATE = 3e-4
"""AdamW rate of the miniature: the kit's 1e-5 is a ViT fine-tuning rate, a CNN needs more."""


class MiniUnlearnError(HackathonError):
    """Invalid configuration or a diverging run of a miniature unlearning method."""


@dataclass(frozen=True)
class MiniUnlearnConfig:
    """Knobs of one miniature unlearning run (a flat superset, like the kit's `UnlearnConfig`).

    Attributes:
        method: one of `METHOD_NAMES`.
        label: unique name of the candidate in the benchmark tables.
        steps: optimiser steps (scrub: max-steps; its min-steps are `steps * scrub_min_ratio`).
        lr: peak AdamW learning rate (warmup then cosine).
        batch_size: images per forget / altered / retain batch.
        seed: seed of batches and augmentations.
        target_temperature: gold_soft: logits are divided by it to build the targets.
        target_views: gold_soft: strong views averaged per target (0 = tempered exact image).
        view_min_area: gold_soft: smallest crop area fraction of a strong view.
        altered_weight: weight of the altered-copy loss relative to the exact one.
        retain_weight: weight of the KL-to-teacher anchor on retain-proxy images.
        kd_temperature: temperature of the KL terms (the scoring one, 2).
        scrub_kl_cap: scrub: cap of the maximised (T^2-scaled) KL.
        scrub_min_ratio: scrub: min-steps per max-step.
        noise_scale: weight_noise: std of the noise relative to each tensor's RMS.
        fixed_targets: gold_soft: draw the strong views ONCE per image (a fixed pseudo-gold model)
            instead of fresh views every step (whose running average is smoother than any single model).
    """

    method: str
    label: str
    steps: int = 100
    lr: float = DEFAULT_LEARNING_RATE
    batch_size: int = 128
    seed: int = 0
    target_temperature: float = 2.0
    target_views: int = 3
    view_min_area: float = 0.35
    altered_weight: float = 1.0
    retain_weight: float = 1.0
    kd_temperature: float = 2.0
    scrub_kl_cap: float = 2.0
    scrub_min_ratio: float = 2.0
    noise_scale: float = 0.0
    fixed_targets: bool = False

    def __post_init__(self) -> None:
        """Reject configurations that would silently do nothing or crash late."""
        if self.method not in METHOD_NAMES:
            raise MiniUnlearnError(f"unknown method '{self.method}'; choose from {METHOD_NAMES}")
        if self.steps < 0 or self.batch_size < 1 or self.lr <= 0:
            raise MiniUnlearnError(f"invalid steps/batch_size/lr in {self.label}")
        if self.target_temperature <= 0 or self.kd_temperature <= 0:
            raise MiniUnlearnError(f"temperatures must be > 0 in {self.label}")
        if not 0.0 < self.view_min_area <= 1.0:
            raise MiniUnlearnError(f"view_min_area must be in (0, 1] in {self.label}")


@dataclass(frozen=True)
class UnlearnInputs:
    """What a method is allowed to see: the original, forget images, OOD proxy (and a pseudo-label)."""

    orig: MiniNet
    forget_x: Tensor
    forget_pseudo_y: Tensor
    """Argmax of the original on the forget images (stands for 'no labels are given')."""
    proxy_x: Tensor
    device: torch.device


def _check_finite(value: Tensor, where: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise MiniUnlearnError(f"non-finite loss in {where}")


def _matching_loss(target_probs: Tensor, logits: Tensor, config: MiniUnlearnConfig) -> Tensor:
    """Mean of KL(target || student) at T = 1 and at the scoring temperature (kit `_matching_loss`)."""
    native = kl_to_student(target_probs, logits, NATIVE_TEMPERATURE)
    tempered = kl_to_student(
        temper_probabilities(target_probs, config.kd_temperature), logits, config.kd_temperature
    )
    return 0.5 * (native + tempered)


def _draw(source: Tensor, size: int, generator: torch.Generator) -> Tensor:
    """Random batch (with replacement) of `size` images from `source`."""
    return source[torch.randint(len(source), (size,), generator=generator)]


def _prepare(inputs: UnlearnInputs) -> tuple[MiniNet, MiniNet]:
    """(frozen eval teacher, trainable student) copies of the original on the device."""
    teacher = copy.deepcopy(inputs.orig).to(inputs.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = copy.deepcopy(inputs.orig).to(inputs.device).train()
    return teacher, student


def _step(optimizer: torch.optim.Optimizer, student: nn.Module, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _backward(loss: Tensor, where: str) -> None:
    _check_finite(loss, where)
    torch.autograd.backward(loss)  # one backward per term: only one student graph alive at a time


def _gold_soft_target(
    teacher: MiniNet, exact: Tensor, config: MiniUnlearnConfig, generator: torch.Generator,
    device: torch.device,
) -> Tensor:  # fmt: skip
    """Average of tempered teacher softmax over strong views (kit target); no views = exact image."""
    with torch.no_grad():
        if config.target_views == 0:
            return torch_f.softmax(teacher(exact.to(device)) / config.target_temperature, dim=1)
        probabilities = torch.zeros(len(exact), teacher.head.out_features, device=device)
        for _ in range(config.target_views):
            view = strong_view(exact, generator, config.view_min_area).to(device)
            probabilities += torch_f.softmax(teacher(view) / config.target_temperature, dim=1)
        return probabilities / config.target_views


def _retain_loss(
    teacher: MiniNet, student: MiniNet, inputs: UnlearnInputs, config: MiniUnlearnConfig,
    generator: torch.Generator,
) -> Tensor:  # fmt: skip
    """KL(teacher || student) at T = 2 on an augmented OOD proxy batch (the retain anchor)."""
    batch = random_alteration(_draw(inputs.proxy_x, config.batch_size, generator), generator)
    batch = batch.to(inputs.device)
    with torch.no_grad():
        teacher_out = teacher(batch)
    return kd_from_teacher(teacher_out, student(batch), config.kd_temperature)


def _fixed_targets(teacher: MiniNet, inputs: UnlearnInputs, config: MiniUnlearnConfig) -> Tensor:
    """One target per forget image, from views drawn once (seeded): a single pseudo-gold model."""
    generator = torch.Generator().manual_seed(config.seed + FIXED_TARGET_SEED_OFFSET)
    parts = [
        _gold_soft_target(
            teacher, inputs.forget_x[start : start + FIXED_TARGET_CHUNK], config, generator,
            inputs.device,
        )
        for start in range(0, len(inputs.forget_x), FIXED_TARGET_CHUNK)
    ]  # fmt: skip
    return torch.cat(parts)


def _run_gold_soft(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    teacher, student = _prepare(inputs)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    weight_total = 1.0 + config.altered_weight
    fixed = _fixed_targets(teacher, inputs, config) if config.fixed_targets else None
    for step in range(config.steps):
        index = torch.randint(len(inputs.forget_x), (config.batch_size,), generator=generator)
        exact = inputs.forget_x[index]
        altered = random_alteration(exact, generator)
        if fixed is not None:
            target = fixed[index.to(inputs.device)]
        else:
            target = _gold_soft_target(teacher, exact, config, generator, inputs.device)
        exact_loss = _matching_loss(target, student(exact.to(inputs.device)), config)
        _backward(exact_loss / weight_total, "gold_soft exact")
        altered_loss = _matching_loss(target, student(altered.to(inputs.device)), config)
        _backward(config.altered_weight * altered_loss / weight_total, "gold_soft altered")
        _backward(
            config.retain_weight * _retain_loss(teacher, student, inputs, config, generator),
            "gold_soft retain",
        )
        _step(optimizer, student, learning_rate_at(step, config.steps, config.lr, WARMUP_FRACTION))
    return student.eval()


def _run_scrub(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    teacher, student = _prepare(inputs)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    min_steps = round(config.steps * config.scrub_min_ratio)
    total = config.steps + min_steps
    weight_total = 1.0 + config.altered_weight
    for step in range(config.steps):
        exact = _draw(inputs.forget_x, config.batch_size, generator).to(inputs.device)
        altered = random_alteration(exact.cpu(), generator).to(inputs.device)
        for images, weight in ((exact, 1.0), (altered, config.altered_weight)):
            with torch.no_grad():
                teacher_out = teacher(images)
            divergence = kd_from_teacher(teacher_out, student(images), config.kd_temperature)
            # clamp(max=cap): beyond the cap the gradient is exactly zero, the ascent stops.
            _backward(-weight * torch.clamp(divergence, max=config.scrub_kl_cap) / weight_total,
                      "scrub max-step")  # fmt: skip
        _step(optimizer, student, learning_rate_at(step, total, config.lr, WARMUP_FRACTION))
    for offset in range(min_steps):
        _backward(
            config.retain_weight * _retain_loss(teacher, student, inputs, config, generator),
            "scrub min-step",
        )
        _step(
            optimizer, student,
            learning_rate_at(config.steps + offset, total, config.lr, WARMUP_FRACTION),
        )  # fmt: skip
    return student.eval()


def _run_finetune_retain(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    teacher, student = _prepare(inputs)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    for step in range(config.steps):
        _backward(
            config.retain_weight * _retain_loss(teacher, student, inputs, config, generator),
            "finetune_retain",
        )
        _step(optimizer, student, learning_rate_at(step, config.steps, config.lr, WARMUP_FRACTION))
    return student.eval()


def _run_grad_ascent(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    _, student = _prepare(inputs)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    for step in range(config.steps):
        index = torch.randint(len(inputs.forget_x), (config.batch_size,), generator=generator)
        images = inputs.forget_x[index].to(inputs.device)
        labels = inputs.forget_pseudo_y[index].to(inputs.device)
        _backward(-torch_f.cross_entropy(student(images), labels), "grad_ascent")
        _step(optimizer, student, learning_rate_at(step, config.steps, config.lr, WARMUP_FRACTION))
    return student.eval()


def _run_uniform(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    teacher, student = _prepare(inputs)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    classes = student.head.out_features
    weight_total = 1.0 + config.altered_weight
    for step in range(config.steps):
        exact = _draw(inputs.forget_x, config.batch_size, generator)
        altered = random_alteration(exact, generator)
        uniform = torch.full((len(exact), classes), 1.0 / classes, device=inputs.device)
        for images, weight in ((exact, 1.0), (altered, config.altered_weight)):
            loss = _matching_loss(uniform, student(images.to(inputs.device)), config)
            _backward(weight * loss / weight_total, "uniform")
        _backward(
            config.retain_weight * _retain_loss(teacher, student, inputs, config, generator),
            "uniform retain",
        )
        _step(optimizer, student, learning_rate_at(step, config.steps, config.lr, WARMUP_FRACTION))
    return student.eval()


def _run_weight_noise(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    student = copy.deepcopy(inputs.orig).to(inputs.device).eval()
    generator = torch.Generator().manual_seed(config.seed)
    with torch.no_grad():
        for parameter in student.parameters():
            rms = parameter.pow(2).mean().sqrt()
            noise = torch.randn(parameter.shape, generator=generator) * config.noise_scale
            parameter.add_(noise.to(inputs.device) * rms)
    return student


def unlearn_mini(inputs: UnlearnInputs, config: MiniUnlearnConfig) -> MiniNet:
    """Run `config.method` on a copy of the original; the original is never modified.

    Raises:
        MiniUnlearnError: unknown method (config validation) or a non-finite loss.
    """
    if config.method == METHOD_IDENTITY:
        return copy.deepcopy(inputs.orig).to(inputs.device).eval()
    runners = {
        METHOD_GOLD_SOFT: _run_gold_soft,
        METHOD_SCRUB: _run_scrub,
        METHOD_FINETUNE_RETAIN: _run_finetune_retain,
        METHOD_GRAD_ASCENT: _run_grad_ascent,
        METHOD_UNIFORM: _run_uniform,
        METHOD_WEIGHT_NOISE: _run_weight_noise,
    }
    return runners[config.method](inputs, config)
