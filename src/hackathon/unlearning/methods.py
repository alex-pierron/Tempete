"""Unlearning methods behind one interface: `unlearn(model, forget_images, config)`.

Three methods, from safest to most ambitious:

identity
    Returns an untouched copy. Score about 0 by construction (sub = orig): the reference point
    that every other method must beat, and a way to test the whole submit pipeline.

scrub
    SCRUB (Kurmanji et al., NeurIPS 2023) adapted to a setting with NO labels and NO retain set:
    alternate "max steps" (maximise KL(teacher || student) on the forget images, but capped, so it
    is not the naive gradient ascent the subject penalises) and "min steps" (minimise the KL to the
    teacher on retain-proxy images, i.e. out-of-distribution natural images). The original CE
    term of the retain step is dropped: we have no labels.

gold_soft  (recommended starting point)
    Targets the behaviour of the GOLD models instead of "being wrong". A gold model never saw the
    forget image, so on it it behaves like on any unseen image of the distribution: lower
    confidence, a smooth distribution whose runner-up classes are the semantically close ones,
    and the same behaviour on slightly altered copies. We approximate that with the only model we
    have: the original teacher, queried on ALTERED views of each forget image (off the memorised
    pixel pattern), averaged (an ensemble, like the average of several gold models), then
    tempered (logits / target_temperature) to remove the over-confidence of memorisation while
    keeping the class RANKING (which wrong class is preferred). The student is trained to output
    this target on the exact image AND on fresh altered copies (half of the forget score), with a
    KL anchor to the teacher on retain-proxy images to protect the retain score. No gradient
    ascent: every loss is a bounded KL toward a sensible target.

Training reads the forget images as uint8 [N, 224, 224, 3]; nothing is written to disk here.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

from hackathon.common.errors import ConfigError
from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.configs import (
    METHOD_GOLD_SOFT,
    METHOD_IDENTITY,
    METHOD_SCRUB,
    UnlearnConfig,
)
from hackathon.unlearning.data import load_proxy_images, select_subset
from hackathon.unlearning.model import model_device, select_device
from hackathon.unlearning.training import (
    epoch_batches,
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    prepare_student,
    prepare_teacher,
    steps_per_epoch,
    student_logits,
    teacher_logits,
    temper_probabilities,
)

LOGGER = logging.getLogger("hackathon.unlearning")
LOG_EVERY_STEPS = 10
GRADIENT_CLIP_NORM = 1.0
"""Global gradient-norm clip: a single bad batch must not undo the careful low learning rate."""
NATIVE_TEMPERATURE = 1.0
"""The server's cross-entropy (mean/std score) is at temperature 1."""

History = list[dict[str, float]]


class UnlearnError(RuntimeError):
    """A training run diverged (non-finite loss) or lacks an input."""


def _check_finite(value: Tensor, where: str) -> float:
    number = float(value.item())
    if not math.isfinite(number):
        raise UnlearnError(f"Non-finite loss in {where}: lower lr or increase the KL cap.")
    return number


def _build_optimizer(student: nn.Module, config: UnlearnConfig) -> torch.optim.Optimizer:
    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise ConfigError("No trainable parameter: check trainable_blocks.")
    return torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)


def _backward_term(loss: Tensor, where: str) -> float:
    """Backpropagate one loss term right away (frees its graph: peak memory = one forward)."""
    value = _check_finite(loss, where)
    torch.autograd.backward(loss)
    return value


def _apply_update(
    optimizer: torch.optim.Optimizer, student: nn.Module, learning_rate: float
) -> None:
    """Clip the accumulated gradient, step at `learning_rate`, clear the gradients."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _resolve_retain(config: UnlearnConfig, retain_images: Tensor | None) -> Tensor:
    if retain_images is not None:
        if len(retain_images) == 0:
            raise UnlearnError("retain_images is empty.")
        return retain_images
    if config.proxy_root is None:
        raise ConfigError(
            "This method needs retain-proxy images: pass retain_images or set proxy_root "
            "(folder with coco_val_subset / div2k_valid / kodak)."
        )
    return load_proxy_images(Path(config.proxy_root), config.retain_count, config.seed)


def _retain_batch(retain: Tensor, config: UnlearnConfig, generator: torch.Generator) -> Tensor:
    """Random augmented retain-proxy batch (augmentation widens the few local images)."""
    indices = torch.randint(len(retain), (config.batch_size,), generator=generator)
    return random_augment(retain[indices], generator)


def _matching_loss(target_probs: Tensor, logits: Tensor, config: UnlearnConfig) -> Tensor:
    """KL(target || student) at temperature 1 plus the same at the scoring temperature.

    Temperature 1 drives the per-sample loss (mean/std score); temperature 2 is the one the
    prediction score uses. Matching both costs nothing and avoids tuning for one at the expense
    of the other.
    """
    native = kl_to_student(target_probs, logits, NATIVE_TEMPERATURE)
    tempered = kl_to_student(
        temper_probabilities(target_probs, config.kd_temperature), logits, config.kd_temperature
    )
    return 0.5 * (native + tempered)


def identity_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """Baseline: an untouched copy of the model."""
    del forget_images, config, retain_images
    return copy.deepcopy(model), []


def gold_soft_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """Distil tempered, view-averaged teacher targets on forget images, anchored on retain proxy."""
    device = select_device(config.device) if config.device else model_device(model)
    retain = _resolve_retain(config, retain_images)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    teacher = prepare_teacher(model, device)
    student = prepare_student(model, device, config)
    optimizer = _build_optimizer(student, config)

    epoch_steps = steps_per_epoch(len(forget_images), config)
    total_steps = epoch_steps * config.epochs
    history: History = []
    step = 0
    for epoch in range(1, config.epochs + 1):
        forget_total = retain_total = 0.0
        batches = 0
        for indices in epoch_batches(len(forget_images), config.batch_size, epoch_steps, generator):
            exact = forget_images[indices]
            altered = random_augment(exact, generator)
            # Target: ensemble of the teacher over altered views, tempered. Computed without
            # gradient and once per step, then reused for the exact image and its altered copy
            # (an altered copy follows its source image in the hidden scoring set).
            probabilities = torch.zeros(len(exact), model_classes(teacher), device=device)
            for _ in range(config.target_views):
                view = strong_view(exact, generator, config.target_view_min_area)
                probabilities += torch.softmax(
                    teacher_logits(teacher, view, device) / config.target_temperature, dim=1
                )
            target = probabilities / config.target_views
            # One backward per term, so only one student graph is alive at a time.
            weight_total = 1.0 + config.altered_weight
            exact_loss = _matching_loss(
                target, student_logits(student, exact, device, config), config
            )
            forget_loss = _backward_term(exact_loss / weight_total, f"gold_soft exact e{epoch}")
            altered_loss = _matching_loss(
                target, student_logits(student, altered, device, config), config
            )
            forget_loss += _backward_term(
                config.altered_weight * altered_loss / weight_total, f"gold_soft altered e{epoch}"
            )
            retain_batch = _retain_batch(retain, config, generator)
            retain_loss = kd_from_teacher(
                teacher_logits(teacher, retain_batch, device),
                student_logits(student, retain_batch, device, config),
                config.kd_temperature,
            )
            _backward_term(config.retain_weight * retain_loss, f"gold_soft retain e{epoch}")
            _apply_update(
                optimizer, student,
                learning_rate_at(step, total_steps, config.lr, config.warmup_fraction),
            )  # fmt: skip
            forget_total += forget_loss
            retain_total += float(retain_loss.item())
            batches += 1
            step += 1
            if step % LOG_EVERY_STEPS == 0:
                LOGGER.info(
                    "gold_soft step %d/%d forget=%.4f retain=%.4f",
                    step,
                    total_steps,
                    forget_loss,
                    retain_loss.item(),
                )
        history.append(
            {
                "epoch": float(epoch),
                "forget_loss": forget_total / batches,
                "retain_loss": retain_total / batches,
            }
        )
        LOGGER.info("gold_soft epoch %d/%d %s", epoch, config.epochs, history[-1])
    return student.eval(), history


def scrub_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """SCRUB without labels: capped max-steps on forget, min-steps on retain proxy."""
    device = select_device(config.device) if config.device else model_device(model)
    retain = _resolve_retain(config, retain_images)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    teacher = prepare_teacher(model, device)
    student = prepare_student(model, device, config)
    optimizer = _build_optimizer(student, config)

    epoch_steps = steps_per_epoch(len(forget_images), config)
    total_steps = epoch_steps * (config.epochs + config.scrub_msteps)
    history: History = []
    step = 0
    weight_total = 1.0 + config.altered_weight
    for epoch in range(1, config.epochs + 1):
        max_total = 0.0
        if epoch <= config.scrub_msteps:
            batches = 0
            for indices in epoch_batches(
                len(forget_images), config.batch_size, epoch_steps, generator
            ):
                exact = forget_images[indices]
                altered = random_augment(exact, generator)
                divergence_exact = kd_from_teacher(
                    teacher_logits(teacher, exact, device),
                    student_logits(student, exact, device, config),
                    config.kd_temperature,
                )
                divergence = divergence_exact
                # clamp(max=cap): once a term exceeds the cap its gradient is exactly zero, so
                # the ascent stops instead of running away (naive ascent is penalised).
                _backward_term(
                    -torch.clamp(divergence_exact, max=config.scrub_kl_cap) / weight_total,
                    f"scrub max-step epoch {epoch}",
                )
                if config.altered_weight > 0:
                    divergence_altered = kd_from_teacher(
                        teacher_logits(teacher, altered, device),
                        student_logits(student, altered, device, config),
                        config.kd_temperature,
                    )
                    _backward_term(
                        -config.altered_weight
                        * torch.clamp(divergence_altered, max=config.scrub_kl_cap)
                        / weight_total,
                        f"scrub max-step (altered) epoch {epoch}",
                    )
                _apply_update(
                    optimizer, student,
                    learning_rate_at(step, total_steps, config.lr, config.warmup_fraction),
                )  # fmt: skip
                max_total += float(divergence.item())
                batches += 1
                step += 1
            max_total /= max(1, batches)
        min_total = 0.0
        for _ in range(epoch_steps):
            retain_batch = _retain_batch(retain, config, generator)
            loss = config.retain_weight * kd_from_teacher(
                teacher_logits(teacher, retain_batch, device),
                student_logits(student, retain_batch, device, config),
                config.kd_temperature,
            )
            _backward_term(loss, f"scrub min-step epoch {epoch}")
            _apply_update(
                optimizer, student,
                learning_rate_at(step, total_steps, config.lr, config.warmup_fraction),
            )  # fmt: skip
            min_total += float(loss.item())
            step += 1
        history.append(
            {"epoch": float(epoch), "max_kl": max_total, "min_loss": min_total / epoch_steps}
        )
        LOGGER.info("scrub epoch %d/%d %s", epoch, config.epochs, history[-1])
    return student.eval(), history


def model_classes(model: nn.Module) -> int:
    """Number of output classes of the classifier head."""
    head = getattr(model, "head", None)
    out_features = getattr(head, "out_features", None)
    if not isinstance(out_features, int):
        raise UnlearnError("The model has no `head.out_features` classifier head.")
    return out_features


MethodFunction = Callable[
    [nn.Module, Tensor, UnlearnConfig, Tensor | None], tuple[nn.Module, History]
]
METHODS: dict[str, MethodFunction] = {
    METHOD_IDENTITY: identity_unlearn,
    METHOD_SCRUB: scrub_unlearn,
    METHOD_GOLD_SOFT: gold_soft_unlearn,
}


def unlearn_with_history(
    model: nn.Module,
    forget_images: Tensor,
    config: UnlearnConfig,
    retain_images: Tensor | None = None,
) -> tuple[nn.Module, History]:
    """Run the configured method; returns the unlearned model and the per-epoch history.

    The input model is never modified. `config.forget_subset` restricts the forget images
    (smoke runs); a copy of the result is returned in eval mode on the training device.

    Raises:
        ConfigError: unknown method or missing retain-proxy source.
        UnlearnError: empty forget set or a diverging run.
    """
    if len(forget_images) == 0:
        raise UnlearnError("forget_images is empty.")
    if config.forget_subset is not None:
        forget_images = select_subset(forget_images, config.forget_subset, config.seed)
    return METHODS[config.method](model, forget_images, config, retain_images)


def unlearn(
    model: nn.Module,
    forget_images: Tensor,
    config: UnlearnConfig,
    retain_images: Tensor | None = None,
) -> nn.Module:
    """Official-style entry point: `unlearn(model, forget_images, config) -> model`."""
    return unlearn_with_history(model, forget_images, config, retain_images)[0]
