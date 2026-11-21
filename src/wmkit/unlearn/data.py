"""CIFAR-10 data, unlearning splits, a CIFAR ResNet-18 and its (cached) training.

Everything the unlearning demo needs besides the algorithms themselves lives here: the dataset in
memory as `uint8`, the retain / forget / validation / test splits of the two SCRUB scenarios
(whole-class forgetting and selective forgetting), a light batch loader, a ResNet-18 adapted to
32x32 inputs, and the training of the *original* model, cached on disk so it is paid once.

The original model is trained without data augmentation, exactly like the SCRUB paper, so that it
over-fits its training set (and the membership attack has something to find).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from wmkit.config import data_dir, weights_dir

NUM_CLASSES: int = 10
CIFAR_IMAGE_SIZE: int = 32
CIFAR_CHANNELS: int = 3
CIFAR_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
PIXEL_MAX: float = 255.0

DEFAULT_FORGET_CLASS: int = 5
"""Class forgotten in the SCRUB large-scale experiments (CIFAR-10 'dog')."""

SELECTIVE_FORGET_SIZE: int = 100
"""Number of examples of the forget class forgotten in the selective scenario (paper sec. 5.1)."""

VALIDATION_FRACTION_OF_FORGET_CLASS_TEST: float = 0.5
"""Share of the forget class's test images reserved for SCRUB+R rewinding (never used to evaluate)."""

DEFAULT_SPLIT_SEED: int = 0

# Batch sizes of the SCRUB paper (sec. 9) for ResNet / CIFAR-10: (forget, retain).
CLASS_FORGET_BATCH_SIZE: int = 512
CLASS_RETAIN_BATCH_SIZE: int = 128
SELECTIVE_FORGET_BATCH_SIZE: int = 16
SELECTIVE_RETAIN_BATCH_SIZE: int = 64
EVAL_BATCH_SIZE: int = 500

ORIGINAL_EPOCHS: int = 12
ORIGINAL_BATCH_SIZE: int = 128
ORIGINAL_PEAK_LR: float = 0.1
ORIGINAL_MOMENTUM: float = 0.9
ORIGINAL_WEIGHT_DECAY: float = 5e-4
ORIGINAL_WARMUP_FRACTION: float = 0.15
ORIGINAL_SEED: int = 0
ORIGINAL_CACHE_FORMAT_VERSION: int = 1


class UnlearnDataError(ValueError):
    """Raised for an impossible split request (unknown class, class too small, empty subset)."""


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and accelerator) for a reproducible run.

    Args:
        seed: The global seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class CifarArrays:
    """CIFAR-10 held in memory as `uint8`.

    Attributes:
        train_x: Training images `[N, 3, 32, 32]`, `uint8`.
        train_y: Training labels `[N]`, `int64`.
        test_x: Test images `[M, 3, 32, 32]`, `uint8`.
        test_y: Test labels `[M]`, `int64`.
    """

    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor


def load_cifar10(
    root: Path | None = None, *, subset_per_class: int | None = None, seed: int = DEFAULT_SPLIT_SEED
) -> CifarArrays:
    """Load CIFAR-10 (downloading it once into `data_dir()`), optionally class-balanced subsampled.

    The download is idempotent: torchvision verifies the files already on disk and only fetches
    the archive (~170 MB) when it is missing or corrupt.

    Args:
        root: Dataset folder; defaults to `wmkit.config.data_dir()`.
        subset_per_class: If set, keep this many random *training* images per class (the test set
            is always kept whole). Used to shorten the demo on slow hardware.
        seed: Seed of the subsampling.

    Returns:
        The dataset as `uint8` tensors.

    Raises:
        UnlearnDataError: If `subset_per_class` is not positive or exceeds the class size.
    """
    from torchvision.datasets import CIFAR10  # lazy: keeps `import wmkit.unlearn` light

    folder = root if root is not None else data_dir()
    train = CIFAR10(root=str(folder), train=True, download=True)
    test = CIFAR10(root=str(folder), train=False, download=True)
    train_x = torch.from_numpy(np.ascontiguousarray(train.data)).permute(0, 3, 1, 2).contiguous()
    test_x = torch.from_numpy(np.ascontiguousarray(test.data)).permute(0, 3, 1, 2).contiguous()
    train_y = torch.tensor(train.targets, dtype=torch.long)
    test_y = torch.tensor(test.targets, dtype=torch.long)
    if subset_per_class is not None:
        keep = _stratified_subset(train_y, subset_per_class, seed)
        train_x, train_y = train_x[keep], train_y[keep]
    return CifarArrays(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def _stratified_subset(labels: Tensor, per_class: int, seed: int) -> Tensor:
    """Return sorted indices of `per_class` random examples of every class."""
    if per_class <= 0:
        raise UnlearnDataError(f"subset_per_class must be positive, got {per_class}")
    generator = torch.Generator().manual_seed(seed)
    chosen: list[Tensor] = []
    for label in range(NUM_CLASSES):
        members = torch.nonzero(labels == label).flatten()
        if len(members) < per_class:
            raise UnlearnDataError(f"class {label} has only {len(members)} examples")
        chosen.append(members[torch.randperm(len(members), generator=generator)[:per_class]])
    return torch.cat(chosen).sort().values


class TensorBatchLoader:
    """Re-iterable batch loader over in-memory `uint8` images (no worker processes).

    Yields `(images, labels)` with `images` normalised `float32 [B, 3, 32, 32]`. With
    `shuffle=True` every pass uses a fresh permutation drawn from a seeded generator, so a run is
    reproducible while epochs differ.

    Args:
        images: `uint8` images `[N, 3, H, W]`.
        labels: `int64` labels `[N]`.
        batch_size: Examples per batch (the last batch may be smaller).
        shuffle: Whether to reshuffle at every pass.
        seed: Seed of the shuffling generator.
        device: Device where the uint8 images are stored (batches come out on that device).
    """

    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int = DEFAULT_SPLIT_SEED,
        device: torch.device | str = "cpu",
    ) -> None:
        if len(images) != len(labels):
            raise UnlearnDataError("images and labels must have the same length")
        if batch_size <= 0:
            raise UnlearnDataError(f"batch_size must be positive, got {batch_size}")
        self._images = images.to(device)
        self._labels = labels.to(device)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._generator = torch.Generator().manual_seed(seed)
        self._mean = torch.tensor(CIFAR_MEAN, device=device).view(1, -1, 1, 1)
        self._std = torch.tensor(CIFAR_STD, device=device).view(1, -1, 1, 1)

    def __len__(self) -> int:
        """Number of batches per pass."""
        return math.ceil(len(self._labels) / self._batch_size)

    @property
    def num_examples(self) -> int:
        """Number of examples per pass."""
        return len(self._labels)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over one pass of normalised `(images, labels)` batches."""
        count = len(self._labels)
        order = (
            torch.randperm(count, generator=self._generator)
            if self._shuffle
            else torch.arange(count)
        ).to(self._labels.device)
        for start in range(0, count, self._batch_size):
            index = order[start : start + self._batch_size]
            pixels = self._images[index].float() / PIXEL_MAX
            yield (pixels - self._mean) / self._std, self._labels[index]


@dataclass(frozen=True)
class UnlearnSplit:
    """Index sets of one unlearning scenario.

    Attributes:
        name: `"class"` (whole class forgotten) or `"selective"` (a few examples forgotten).
        forget_class: Class the forget set is drawn from.
        retain_idx: Indices (into the training set) of the retained examples.
        forget_idx: Indices (into the training set) of the forgotten examples.
        val_idx: Indices (into the test set) of same-distribution validation examples, used only
            by SCRUB+R to pick the checkpoint. Never used for evaluation.
        test_idx: Indices (into the test set) used for test accuracy and the MIA.
    """

    name: str
    forget_class: int
    retain_idx: Tensor
    forget_idx: Tensor
    val_idx: Tensor
    test_idx: Tensor


def _split_forget_class_test_set(
    test_y: Tensor, forget_class: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Split the test set into (validation, evaluation) indices.

    Half of the forget class's test images become the SCRUB+R validation set (same distribution as
    the forget set); the other half and every other class form the evaluation test set. The split
    is disjoint so rewinding never selects a checkpoint on data it is later scored on.
    """
    class_members = torch.nonzero(test_y == forget_class).flatten()
    generator = torch.Generator().manual_seed(seed + 1)
    shuffled = class_members[torch.randperm(len(class_members), generator=generator)]
    n_val = int(len(shuffled) * VALIDATION_FRACTION_OF_FORGET_CLASS_TEST)
    if n_val == 0 or n_val == len(shuffled):
        raise UnlearnDataError(f"test class {forget_class} is too small to be split")
    val_idx = shuffled[:n_val].sort().values
    evaluation = torch.ones(len(test_y), dtype=torch.bool)
    evaluation[val_idx] = False
    return val_idx, torch.nonzero(evaluation).flatten()


def make_split(
    arrays: CifarArrays,
    *,
    scenario: str,
    forget_class: int = DEFAULT_FORGET_CLASS,
    n_forget: int = SELECTIVE_FORGET_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> UnlearnSplit:
    """Build the retain / forget / validation / test index sets of a scenario.

    Args:
        arrays: The dataset.
        scenario: `"class"` forgets every training image of `forget_class`; `"selective"` forgets
            `n_forget` random training images of `forget_class`.
        forget_class: The class concerned by the forget request.
        n_forget: Forget set size for the selective scenario.
        seed: Seed of the random choices.

    Returns:
        The `UnlearnSplit`.

    Raises:
        UnlearnDataError: If the scenario or class is unknown or the forget set cannot be formed.
    """
    if scenario not in ("class", "selective"):
        raise UnlearnDataError(f"unknown scenario {scenario!r}; expected 'class' or 'selective'")
    if not 0 <= forget_class < NUM_CLASSES:
        raise UnlearnDataError(f"forget_class must be in [0, {NUM_CLASSES}), got {forget_class}")
    members = torch.nonzero(arrays.train_y == forget_class).flatten()
    if scenario == "class":
        forget_idx = members
    else:
        if not 0 < n_forget < len(members):
            raise UnlearnDataError(
                f"n_forget must be in (0, {len(members)}) for class {forget_class}, got {n_forget}"
            )
        generator = torch.Generator().manual_seed(seed)
        forget_idx = members[torch.randperm(len(members), generator=generator)[:n_forget]]
        forget_idx = forget_idx.sort().values
    if len(forget_idx) == 0:
        raise UnlearnDataError(f"no training example of class {forget_class}")
    keep = torch.ones(len(arrays.train_y), dtype=torch.bool)
    keep[forget_idx] = False
    val_idx, test_idx = _split_forget_class_test_set(arrays.test_y, forget_class, seed)
    return UnlearnSplit(
        name=scenario,
        forget_class=forget_class,
        retain_idx=torch.nonzero(keep).flatten(),
        forget_idx=forget_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


@dataclass(frozen=True)
class UnlearnLoaders:
    """The four loaders of one scenario (all share the same normalisation)."""

    retain: TensorBatchLoader
    forget: TensorBatchLoader
    val: TensorBatchLoader
    test: TensorBatchLoader


def build_loaders(
    arrays: CifarArrays,
    split: UnlearnSplit,
    *,
    device: torch.device | str = "cpu",
    seed: int = DEFAULT_SPLIT_SEED,
    forget_batch_size: int | None = None,
    retain_batch_size: int | None = None,
) -> UnlearnLoaders:
    """Create the retain / forget / validation / test loaders of a split.

    Retain and forget loaders shuffle (they are used for training); validation and test do not.
    Default batch sizes are the SCRUB paper's for ResNet / CIFAR-10 (sec. 9).

    Args:
        arrays: The dataset.
        split: Index sets from `make_split`.
        device: Device where the images are stored.
        seed: Seed of the shuffling generators.
        forget_batch_size: Overrides the paper's forget batch size for this scenario.
        retain_batch_size: Overrides the paper's retain batch size for this scenario.

    Returns:
        The `UnlearnLoaders`.
    """
    is_class = split.name == "class"
    forget_bs = forget_batch_size or (
        CLASS_FORGET_BATCH_SIZE if is_class else SELECTIVE_FORGET_BATCH_SIZE
    )
    retain_bs = retain_batch_size or (
        CLASS_RETAIN_BATCH_SIZE if is_class else SELECTIVE_RETAIN_BATCH_SIZE
    )

    def make(
        x: Tensor, y: Tensor, index: Tensor, batch_size: int, shuffle: bool, offset: int
    ) -> TensorBatchLoader:
        return TensorBatchLoader(
            x[index], y[index], batch_size, shuffle=shuffle, seed=seed + offset, device=device
        )

    return UnlearnLoaders(
        retain=make(arrays.train_x, arrays.train_y, split.retain_idx, retain_bs, True, 11),
        forget=make(arrays.train_x, arrays.train_y, split.forget_idx, forget_bs, True, 12),
        val=make(arrays.test_x, arrays.test_y, split.val_idx, EVAL_BATCH_SIZE, False, 13),
        test=make(arrays.test_x, arrays.test_y, split.test_idx, EVAL_BATCH_SIZE, False, 14),
    )


def resnet18_cifar(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 for 32x32 inputs: 3x3 stride-1 stem convolution and no initial max-pool.

    Args:
        num_classes: Size of the classification head.

    Returns:
        A randomly initialised torchvision ResNet-18 with the CIFAR stem.
    """
    from torchvision.models import resnet18

    model: nn.Module = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(CIFAR_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@dataclass(frozen=True)
class TrainingSummary:
    """Measurements of one training run of the original model."""

    epochs: int
    seconds: float
    final_train_loss: float
    train_acc: float
    test_acc: float
    cached: bool


def _learning_rate_at(step: int, total_steps: int, peak: float, warmup_fraction: float) -> float:
    """Linear warm-up to `peak`, then linear decay to 0 (one-cycle shape, no extra dependency)."""
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    return peak * max(0.0, (total_steps - step) / (total_steps - warmup_steps))


def train_classifier(
    model: nn.Module,
    loader: TensorBatchLoader,
    device: torch.device | str,
    *,
    epochs: int = ORIGINAL_EPOCHS,
    peak_lr: float = ORIGINAL_PEAK_LR,
    log: Callable[[str], None] | None = None,
) -> float:
    """Train `model` in place with SGD + warm-up/decay schedule, no data augmentation.

    Args:
        model: Classifier to train (already on `device`).
        loader: Shuffling training loader.
        device: Device of the model and of the loader's images.
        epochs: Number of passes over the data.
        peak_lr: Peak learning rate of the schedule.
        log: Optional callback receiving one progress line per epoch.

    Returns:
        The mean training loss of the last epoch.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=peak_lr,
        momentum=ORIGINAL_MOMENTUM,
        weight_decay=ORIGINAL_WEIGHT_DECAY,
        nesterov=True,
    )
    total_steps = epochs * len(loader)
    step = 0
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for images, labels in loader:
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate_at(
                    step, total_steps, peak_lr, ORIGINAL_WARMUP_FRACTION
                )
            loss = torch_f.cross_entropy(model(images.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            running += float(loss.item())
            batches += 1
            step += 1
        last_loss = running / max(1, batches)
        if log is not None:
            log(f"epoch {epoch}/{epochs} train_loss={last_loss:.4f}")
    return last_loss


def _cache_path(arrays: CifarArrays, epochs: int, seed: int) -> Path:
    """Cache file name keyed by every input that changes the trained weights."""
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "v": ORIGINAL_CACHE_FORMAT_VERSION,
                "n_train": len(arrays.train_y),
                "epochs": epochs,
                "seed": seed,
                "bs": ORIGINAL_BATCH_SIZE,
                "lr": ORIGINAL_PEAK_LR,
                "wd": ORIGINAL_WEIGHT_DECAY,
                "label_sum": int(arrays.train_y.sum().item()),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    folder = weights_dir() / "unlearn"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"cifar10_resnet18_n{len(arrays.train_y)}_e{epochs}_s{seed}_{fingerprint}.pt"


def load_or_train_original(
    arrays: CifarArrays,
    device: torch.device | str,
    *,
    epochs: int = ORIGINAL_EPOCHS,
    seed: int = ORIGINAL_SEED,
    log: Callable[[str], None] | None = None,
) -> tuple[nn.Module, TrainingSummary]:
    """Return the original ResNet-18 (trained on the whole training set), training it only once.

    The weights are cached under `weights_dir()/unlearn/`; a second call with the same inputs
    loads them in a second. The cache write is atomic (temp file then `os.replace`), so a crash
    during training never leaves a truncated checkpoint behind: re-running resumes from scratch.

    Args:
        arrays: The dataset; the model is trained on all of `train_x`.
        device: Device used for training and for the returned model.
        epochs: Training epochs.
        seed: Global seed of the weight initialisation and shuffling.
        log: Optional callback receiving one progress line per epoch.

    Returns:
        `(model, summary)`; `summary.cached` is True when the weights came from disk.
    """
    from wmkit.unlearn.mia import accuracy

    path = _cache_path(arrays, epochs, seed)
    seed_everything(seed)
    model = resnet18_cifar().to(device)
    train_loader = TensorBatchLoader(
        arrays.train_x, arrays.train_y, ORIGINAL_BATCH_SIZE, shuffle=True, seed=seed, device=device
    )
    eval_train = TensorBatchLoader(
        arrays.train_x, arrays.train_y, EVAL_BATCH_SIZE, shuffle=False, device=device
    )
    eval_test = TensorBatchLoader(
        arrays.test_x, arrays.test_y, EVAL_BATCH_SIZE, shuffle=False, device=device
    )
    if path.exists():
        payload = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(payload["state_dict"])
        seconds, final_loss = float(payload["seconds"]), float(payload["final_train_loss"])
        cached = True
    else:
        started = time.perf_counter()
        final_loss = train_classifier(model, train_loader, device, epochs=epochs, log=log)
        seconds = time.perf_counter() - started
        temporary = path.with_suffix(f".tmp{os.getpid()}")
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "seconds": seconds,
                "final_train_loss": final_loss,
            },
            temporary,
        )
        os.replace(temporary, path)
        cached = False
    summary = TrainingSummary(
        epochs=epochs,
        seconds=seconds,
        final_train_loss=final_loss,
        train_acc=accuracy(model, eval_train, device),
        test_acc=accuracy(model, eval_test, device),
        cached=cached,
    )
    return model, summary


def summary_as_dict(summary: TrainingSummary) -> dict[str, float | int | bool]:
    """Return a `TrainingSummary` as a plain dictionary (for printing or JSON)."""
    return asdict(summary)
