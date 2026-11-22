"""SCRUB and SCRUB+R (Kurmanji, Triantafillou, Hayes, Triantafillou, NeurIPS 2023, arXiv 2302.09880).

SCRUB distils the original model (the frozen *teacher*) into a copy of itself (the *student*) with
an asymmetric rule: the student must **disobey** the teacher on the forget set and **obey** it on
the retain set. With `d(x) = KL(softmax(z_teacher / T) || softmax(z_student / T)) * T^2`:

* max-step, one epoch over the forget set:   minimise `-d(x_f)`;
* min-step, one epoch over the retain set:   minimise `alpha * d(x_r) + gamma * CE(f(x_r), y_r)`.

Training alternates a max-step then a min-step for the first `msteps` epochs, then continues with
min-steps only to restore the retain performance (the optimiser state and the step-decayed learning
rate carry over, as in the official notebook). SCRUB+R additionally keeps one checkpoint per epoch
and rewinds to the one whose forget error is closest to the error on a validation set drawn from
the forget distribution, so the forgotten examples are not made *more* wrong than unseen ones
(which would make them identifiable by a membership attack).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as torch_f
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from wmkit.unlearn.mia import BatchIterable

OptimizerName = Literal["sgd", "adam"]

# Defaults of the contract (docs/CONTRACT-wmkit.md 2.3). They are the values of the *large-scale*
# column (CIFAR-10 / Lacuna-10) of the official notebooks, see docs/research/scrub-unlearning.md 4.
DEFAULT_LEARNING_RATE: float = 5e-4
DEFAULT_MSTEPS: int = 2
DEFAULT_EPOCHS: int = 3
DEFAULT_ALPHA: float = 0.001
DEFAULT_GAMMA: float = 0.99
DEFAULT_KD_TEMPERATURE: float = 4.0
DEFAULT_WEIGHT_DECAY: float = 5e-4
DEFAULT_LR_DECAY_EPOCHS: tuple[int, ...] = (3, 5, 9)

# `ScrubConfig.large()` of the contract = the *small-scale* column (CIFAR-5 / Lacuna-5) of the notebooks.
PRESET_LARGE_MSTEPS: int = 3
PRESET_LARGE_EPOCHS: int = 10
PRESET_LARGE_ALPHA: float = 0.5
PRESET_LARGE_GAMMA: float = 1.0
PRESET_LARGE_KD_TEMPERATURE: float = 2.0
PRESET_LARGE_WEIGHT_DECAY: float = 0.1
PRESET_LARGE_LR_DECAY_EPOCHS: tuple[int, ...] = (5, 8, 9)

DEFAULT_LR_DECAY_RATE: float = 0.1
SGD_MOMENTUM: float = 0.9
NOT_APPLICABLE: float = float("nan")


class ScrubError(RuntimeError):
    """Raised when SCRUB cannot run or diverges (empty loader, non-finite loss)."""


class ScrubConfig(BaseModel):
    """Hyper-parameters of SCRUB.

    The defaults and `ScrubConfig.large()` follow `docs/CONTRACT-wmkit.md` 2.3 numerically. Note
    that the contract labels them "small" and "large" the other way round compared with the
    official notebooks: the default values below are the notebooks' large-scale (CIFAR-10) ones and
    `large()` is the notebooks' small-scale (CIFAR-5) one. The numbers are what matter and are kept.

    Attributes:
        lr: Initial learning rate.
        msteps: Number of leading epochs that include a max-step (forget set).
        epochs: Total number of epochs; each one contains a min-step (retain set).
        alpha: Weight of the retain-set distillation term.
        gamma: Weight of the retain-set cross-entropy term.
        kd_T: Temperature of the distillation KL.
        wd: Weight decay of the optimiser.
        optimizer: `"sgd"` (momentum 0.9, official large-scale) or `"adam"` (official small-scale).
        lr_decay_epochs: Epochs after which the learning rate is multiplied by `lr_decay_rate`.
        lr_decay_rate: Multiplicative decay applied at each milestone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: float = Field(default=DEFAULT_LEARNING_RATE, gt=0.0)
    msteps: int = Field(default=DEFAULT_MSTEPS, ge=0)
    epochs: int = Field(default=DEFAULT_EPOCHS, ge=1)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0)
    gamma: float = Field(default=DEFAULT_GAMMA, ge=0.0)
    kd_T: float = Field(default=DEFAULT_KD_TEMPERATURE, gt=0.0)
    wd: float = Field(default=DEFAULT_WEIGHT_DECAY, ge=0.0)
    optimizer: OptimizerName = "sgd"
    lr_decay_epochs: tuple[int, ...] = DEFAULT_LR_DECAY_EPOCHS
    lr_decay_rate: float = Field(default=DEFAULT_LR_DECAY_RATE, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _msteps_within_epochs(self) -> ScrubConfig:
        """`msteps` cannot exceed the number of epochs."""
        if self.msteps > self.epochs:
            raise ValueError(f"msteps ({self.msteps}) must not exceed epochs ({self.epochs})")
        return self

    @classmethod
    def large(cls) -> ScrubConfig:
        """Contract preset `large()`: msteps 3, epochs 10, alpha 0.5, gamma 1, T 2, wd 0.1, Adam."""
        return cls(
            msteps=PRESET_LARGE_MSTEPS,
            epochs=PRESET_LARGE_EPOCHS,
            alpha=PRESET_LARGE_ALPHA,
            gamma=PRESET_LARGE_GAMMA,
            kd_T=PRESET_LARGE_KD_TEMPERATURE,
            wd=PRESET_LARGE_WEIGHT_DECAY,
            optimizer="adam",
            lr_decay_epochs=PRESET_LARGE_LR_DECAY_EPOCHS,
        )


def kl_teacher_student(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float
) -> Tensor:
    """Tempered distillation distance `T^2 * KL(softmax(z_t / T) || softmax(z_s / T))`, batch mean.

    This is the official `DistillKL`: `F.kl_div(log_softmax(z_s/T), softmax(z_t/T), 'batchmean')`.

    Args:
        student_logits: Logits of the student, `[B, C]`.
        teacher_logits: Logits of the (frozen) teacher, `[B, C]`.
        temperature: Softmax temperature `T`.

    Returns:
        A scalar tensor, differentiable with respect to `student_logits`.
    """
    return (
        torch_f.kl_div(
            torch_f.log_softmax(student_logits / temperature, dim=1),
            torch_f.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature
        * temperature
    )


def _build_optimizer(model: nn.Module, config: ScrubConfig) -> torch.optim.Optimizer:
    """Instantiate the optimiser named in `config` over all student parameters."""
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    return torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=SGD_MOMENTUM, weight_decay=config.wd
    )


def _learning_rate_for_epoch(config: ScrubConfig, epoch: int) -> float:
    """Step-decayed learning rate of the 1-indexed `epoch`: `lr * rate^(#milestones passed)`."""
    passed = sum(1 for milestone in config.lr_decay_epochs if epoch > milestone)
    return config.lr * config.lr_decay_rate**passed


def _check_finite(loss: Tensor, phase: str) -> float:
    """Return `loss` as a float, or raise `ScrubError` if it is NaN/inf (the max-step is unbounded)."""
    value = float(loss.item())
    if not math.isfinite(value):
        raise ScrubError(
            f"non-finite loss during the {phase}: the student diverged; lower `lr` or `msteps`"
        )
    return value


def _distillation_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: BatchIterable,
    optimizer: torch.optim.Optimizer,
    config: ScrubConfig,
    device: torch.device | str,
    *,
    maximize: bool,
) -> float:
    """Run one max-step (forget set) or min-step (retain set) epoch; return the mean loss shown.

    For a max-step the returned value is the mean KL divergence being *maximised* (positive); for
    a min-step it is the mean of `alpha * KL + gamma * CE` being minimised.
    """
    student.train()
    teacher.eval()
    phase = "max-step" if maximize else "min-step"
    total = 0.0
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        divergence = kl_teacher_student(student_logits, teacher_logits, config.kd_T)
        if maximize:
            loss = -divergence
            shown = _check_finite(divergence, phase)
        else:
            loss = config.alpha * divergence + config.gamma * torch_f.cross_entropy(
                student_logits, labels
            )
            shown = _check_finite(loss, phase)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        total += shown
        batches += 1
    if batches == 0:
        raise ScrubError(f"the {phase} loader yielded no batch")
    return total / batches


@torch.no_grad()
def classification_error(
    model: nn.Module, loader: BatchIterable, device: torch.device | str
) -> float:
    """Top-1 error of `model` (eval mode) on `loader`; the model's train/eval mode is restored.

    Raises:
        ScrubError: If the loader yields no example.
    """
    was_training = model.training
    model.eval()
    wrong = 0
    seen = 0
    try:
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            wrong += int((predictions != labels.cpu()).sum().item())
            seen += len(labels)
    finally:
        model.train(was_training)
    if seen == 0:
        raise ScrubError("cannot measure an error on an empty loader")
    return wrong / seen


def scrub(
    student: nn.Module,
    teacher: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    config: ScrubConfig,
    device: torch.device | str,
    *,
    rewind_val_loader: BatchIterable | None = None,
    epoch_callback: Callable[[int, nn.Module], None] | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Unlearn the forget set with SCRUB, or SCRUB+R when `rewind_val_loader` is given.

    Neither `student` nor `teacher` is modified: both are deep-copied, the teacher is frozen
    (eval mode, no gradient) and the returned module is the unlearned copy of the student.

    Args:
        student: Model to unlearn, usually a copy of the original weights.
        teacher: The original model (reference for the distillation).
        retain_loader: Batches `(images, labels)` of the data to keep.
        forget_loader: Batches `(images, labels)` of the data to forget.
        config: SCRUB hyper-parameters.
        device: Device used for the optimisation.
        rewind_val_loader: Optional validation batches drawn from the *same distribution as the
            forget set* but never trained on. When given, runs SCRUB+R: the returned model is the
            per-epoch checkpoint whose forget error is closest to the final validation error.
        epoch_callback: Optional hook called as `epoch_callback(epoch, student)` at the end of each
            epoch, before any rewinding; the module is the live student (copy it to keep it). It
            lets one run yield both bare SCRUB (the last epoch) and SCRUB+R from the same trajectory.

    Returns:
        `(unlearned_model, history)`. `history` has one dict per epoch with `epoch`, `lr`,
        `max_kl` (mean KL maximised, NaN if the epoch had no max-step), `min_loss` and
        `forget_error`, plus `val_error` when rewinding. With rewinding, a last dict carries
        `rewind_epoch`, `rewind_target_error` and `rewind_forget_error`.

    Raises:
        ScrubError: If a loader is empty or the loss becomes non-finite.
    """
    student_model = copy.deepcopy(student).to(device)
    teacher_model = copy.deepcopy(teacher).to(device).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    optimizer = _build_optimizer(student_model, config)

    history: list[dict[str, float]] = []
    checkpoints: list[tuple[float, dict[str, Tensor]]] = []
    for epoch in range(1, config.epochs + 1):
        learning_rate = _learning_rate_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        max_kl = NOT_APPLICABLE
        if epoch <= config.msteps:
            max_kl = _distillation_epoch(
                student_model, teacher_model, forget_loader, optimizer, config, device,
                maximize=True,
            )  # fmt: skip
        min_loss = _distillation_epoch(
            student_model, teacher_model, retain_loader, optimizer, config, device, maximize=False
        )
        forget_error = classification_error(student_model, forget_loader, device)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "lr": learning_rate,
            "max_kl": max_kl,
            "min_loss": min_loss,
            "forget_error": forget_error,
        }
        if rewind_val_loader is not None:
            row["val_error"] = classification_error(student_model, rewind_val_loader, device)
            checkpoints.append(
                (
                    forget_error,
                    {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()},
                )
            )
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(epoch, student_model)

    if rewind_val_loader is not None:
        target_error = history[-1]["val_error"]
        best_index = min(
            range(len(checkpoints)), key=lambda index: abs(checkpoints[index][0] - target_error)
        )
        student_model.load_state_dict(checkpoints[best_index][1])
        history.append(
            {
                "rewind_epoch": float(best_index + 1),
                "rewind_target_error": target_error,
                "rewind_forget_error": checkpoints[best_index][0],
            }
        )
    return student_model, history
