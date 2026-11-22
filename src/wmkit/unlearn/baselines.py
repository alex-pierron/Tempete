"""Unlearning baselines compared with SCRUB in the paper (Kurmanji et al., sec. 5.1 and 9).

All four share the signature `(model, retain_loader, forget_loader, config, device)` and return
`(unlearned_model, history)` like `wmkit.unlearn.scrub.scrub`; the input model is never modified.

* `finetune_retain`: plain fine-tuning on the retain set (the forget set is only "forgotten" by
  catastrophic interference; paper: 10 epochs, learning rate 0.01).
* `negative_gradient` (NegGrad): gradient *ascent* on the cross-entropy of the forget set alone.
* `neggrad_plus` (NegGrad+): `beta * CE(retain) - (1 - beta) * CE(forget)`, the stronger version
  proposed by the SCRUB authors (paper: beta 0.95 small scale, 0.9999 large scale).
* `random_labels`: fine-tune on forget examples relabelled with a random *wrong* class, together
  with retain examples to protect the utility.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn.functional as torch_f
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from wmkit.unlearn.mia import BatchIterable

BaselineName = Literal["finetune_retain", "negative_gradient", "neggrad_plus", "random_labels"]

FINETUNE_LEARNING_RATE: float = 0.01
FINETUNE_EPOCHS: int = 10
NEGGRAD_LEARNING_RATE: float = 1e-3
NEGGRAD_EPOCHS: int = 5
NEGGRAD_PLUS_LEARNING_RATE: float = 0.01
NEGGRAD_PLUS_EPOCHS: int = 5
NEGGRAD_PLUS_BETA: float = 0.95
RANDOM_LABELS_LEARNING_RATE: float = 0.01
RANDOM_LABELS_EPOCHS: int = 5
BASELINE_MOMENTUM: float = 0.9
BASELINE_WEIGHT_DECAY: float = 5e-4
BASELINE_MAX_GRAD_NORM: float = 5.0
"""Gradient-norm clip applied to every baseline: ascent losses are unbounded and explode otherwise."""
BASELINE_SEED: int = 0
MIN_CLASSES: int = 2


class BaselineError(RuntimeError):
    """Raised when a baseline cannot run or diverges (empty loader, non-finite loss)."""


class BaselineConfig(BaseModel):
    """Hyper-parameters shared by the baselines (SGD with momentum).

    Attributes:
        lr: Learning rate.
        epochs: Passes over the loader that drives the iteration (retain for finetune, NegGrad+ and
            random labels; forget for plain NegGrad).
        wd: Weight decay.
        beta: NegGrad+ trade-off between retain and forget terms, in `(0, 1]`.
        max_grad_norm: Gradient clipping norm; `None` disables clipping.
        seed: Seed of the random label draw (random_labels).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: float = Field(default=FINETUNE_LEARNING_RATE, gt=0.0)
    epochs: int = Field(default=FINETUNE_EPOCHS, ge=1)
    wd: float = Field(default=BASELINE_WEIGHT_DECAY, ge=0.0)
    beta: float = Field(default=NEGGRAD_PLUS_BETA, gt=0.0, le=1.0)
    max_grad_norm: float | None = Field(default=BASELINE_MAX_GRAD_NORM, gt=0.0)
    seed: int = BASELINE_SEED

    @classmethod
    def for_method(cls, name: BaselineName) -> BaselineConfig:
        """Default configuration of a baseline (values of the SCRUB paper where it gives them)."""
        if name == "negative_gradient":
            return cls(lr=NEGGRAD_LEARNING_RATE, epochs=NEGGRAD_EPOCHS)
        if name == "neggrad_plus":
            return cls(lr=NEGGRAD_PLUS_LEARNING_RATE, epochs=NEGGRAD_PLUS_EPOCHS)
        if name == "random_labels":
            return cls(lr=RANDOM_LABELS_LEARNING_RATE, epochs=RANDOM_LABELS_EPOCHS)
        return cls(lr=FINETUNE_LEARNING_RATE, epochs=FINETUNE_EPOCHS)


Batch = tuple[Tensor, Tensor]


def _cycle(loader: BatchIterable) -> Iterator[Batch]:
    """Iterate over `loader` forever, re-iterating it (no caching of batches in memory)."""
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise BaselineError("the loader yielded no batch")


def _prepare(
    model: nn.Module, config: BaselineConfig, device: torch.device | str
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Deep-copy `model` onto `device` in train mode and build its SGD optimiser."""
    working = copy.deepcopy(model).to(device)
    working.train()
    optimizer = torch.optim.SGD(
        working.parameters(), lr=config.lr, momentum=BASELINE_MOMENTUM, weight_decay=config.wd
    )
    return working, optimizer


def _apply(
    model: nn.Module, optimizer: torch.optim.Optimizer, loss: Tensor, config: BaselineConfig
) -> float:
    """Backpropagate `loss`, clip the gradient, step; return the loss value or raise if non-finite."""
    value = float(loss.item())
    if not math.isfinite(value):
        raise BaselineError("non-finite loss: the baseline diverged; lower the learning rate")
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(loss)
    if config.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return value


def finetune_retain(
    model: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    config: BaselineConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fine-tune on the retain set only (the forget loader is ignored, kept for a uniform API).

    Args:
        model: Original model; not modified.
        retain_loader: Batches of the data to keep.
        forget_loader: Unused.
        config: Hyper-parameters; defaults to `BaselineConfig.for_method("finetune_retain")`.
        device: Compute device.

    Returns:
        `(unlearned_model, history)` with one `{"epoch", "loss"}` dict per epoch.

    Raises:
        BaselineError: If the loader is empty or the loss diverges.
    """
    del forget_loader
    cfg = config or BaselineConfig.for_method("finetune_retain")
    working, optimizer = _prepare(model, cfg, device)
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for images, labels in retain_loader:
            loss = torch_f.cross_entropy(working(images.to(device)), labels.to(device))
            total += _apply(working, optimizer, loss, cfg)
            batches += 1
        if batches == 0:
            raise BaselineError("the retain loader yielded no batch")
        history.append({"epoch": float(epoch), "loss": total / batches})
    return working, history


def negative_gradient(
    model: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    config: BaselineConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """NegGrad: gradient ascent on the forget-set cross-entropy (the retain loader is ignored).

    Args:
        model: Original model; not modified.
        retain_loader: Unused.
        forget_loader: Batches of the data to forget; drives the iteration.
        config: Hyper-parameters; defaults to `BaselineConfig.for_method("negative_gradient")`.
        device: Compute device.

    Returns:
        `(unlearned_model, history)` with one `{"epoch", "forget_ce"}` dict per epoch.

    Raises:
        BaselineError: If the loader is empty or the loss diverges.
    """
    del retain_loader
    cfg = config or BaselineConfig.for_method("negative_gradient")
    working, optimizer = _prepare(model, cfg, device)
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for images, labels in forget_loader:
            ce = torch_f.cross_entropy(working(images.to(device)), labels.to(device))
            # `_apply` checks the value of the loss it is given: pass -CE, report +CE.
            total += -_apply(working, optimizer, -ce, cfg)
            batches += 1
        if batches == 0:
            raise BaselineError("the forget loader yielded no batch")
        history.append({"epoch": float(epoch), "forget_ce": total / batches})
    return working, history


def neggrad_plus(
    model: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    config: BaselineConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, list[dict[str, float]]]:
    """NegGrad+: minimise `beta * CE(retain) - (1 - beta) * CE(forget)`.

    Each step pairs one retain batch with one forget batch (the forget loader is cycled); an epoch
    is one pass over the retain loader.

    Args:
        model: Original model; not modified.
        retain_loader: Batches of the data to keep; drives the iteration.
        forget_loader: Batches of the data to forget (cycled).
        config: Hyper-parameters (`beta` included); defaults to the method preset.
        device: Compute device.

    Returns:
        `(unlearned_model, history)` with one `{"epoch", "loss"}` dict per epoch.

    Raises:
        BaselineError: If a loader is empty or the loss diverges.
    """
    cfg = config or BaselineConfig.for_method("neggrad_plus")
    working, optimizer = _prepare(model, cfg, device)
    forget_batches = _cycle(forget_loader)
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for images, labels in retain_loader:
            forget_images, forget_labels = next(forget_batches)
            retain_ce = torch_f.cross_entropy(working(images.to(device)), labels.to(device))
            forget_ce = torch_f.cross_entropy(
                working(forget_images.to(device)), forget_labels.to(device)
            )
            loss = cfg.beta * retain_ce - (1.0 - cfg.beta) * forget_ce
            total += _apply(working, optimizer, loss, cfg)
            batches += 1
        if batches == 0:
            raise BaselineError("the retain loader yielded no batch")
        history.append({"epoch": float(epoch), "loss": total / batches})
    return working, history


def random_wrong_labels(labels: Tensor, num_classes: int, generator: torch.Generator) -> Tensor:
    """Draw, for each label, a uniformly random class different from it.

    Args:
        labels: True labels `[B]`.
        num_classes: Number of classes (>= 2).
        generator: Random generator on CPU (reproducible draws).

    Returns:
        Labels `[B]` on the device of `labels`, never equal to the input labels.

    Raises:
        BaselineError: If `num_classes < 2`.
    """
    if num_classes < MIN_CLASSES:
        raise BaselineError("random wrong labels need at least 2 classes")
    offsets = torch.randint(1, num_classes, labels.shape, generator=generator)
    return (labels + offsets.to(labels.device)) % num_classes


def random_labels(
    model: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    config: BaselineConfig | None = None,
    device: torch.device | str = "cpu",
    *,
    num_classes: int | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fine-tune on forget examples with random wrong labels, alongside retain examples.

    Each step pairs one retain batch (true labels) with one forget batch (labels replaced by a
    random wrong class); loss = sum of the two cross-entropies. An epoch is one pass over the
    retain loader.

    Args:
        model: Original model; not modified.
        retain_loader: Batches of the data to keep; drives the iteration.
        forget_loader: Batches of the data to forget (cycled, labels randomised).
        config: Hyper-parameters; defaults to the method preset.
        device: Compute device.
        num_classes: Number of classes; inferred from the model's output size when omitted.

    Returns:
        `(unlearned_model, history)` with one `{"epoch", "loss"}` dict per epoch.

    Raises:
        BaselineError: If a loader is empty, the loss diverges or the class count is unknown.
    """
    cfg = config or BaselineConfig.for_method("random_labels")
    working, optimizer = _prepare(model, cfg, device)
    forget_batches = _cycle(forget_loader)
    generator = torch.Generator().manual_seed(cfg.seed)
    classes = num_classes
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        total, batches = 0.0, 0
        for images, labels in retain_loader:
            forget_images, forget_labels = next(forget_batches)
            forget_logits = working(forget_images.to(device))
            if classes is None:
                classes = int(forget_logits.shape[1])
            wrong = random_wrong_labels(forget_labels.to(device), classes, generator)
            retain_ce = torch_f.cross_entropy(working(images.to(device)), labels.to(device))
            loss = retain_ce + torch_f.cross_entropy(forget_logits, wrong)
            total += _apply(working, optimizer, loss, cfg)
            batches += 1
        if batches == 0:
            raise BaselineError("the retain loader yielded no batch")
        history.append({"epoch": float(epoch), "loss": total / batches})
    return working, history
