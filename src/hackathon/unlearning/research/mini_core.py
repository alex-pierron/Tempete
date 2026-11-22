"""Building blocks of the miniature benchmark: CIFAR-10 worlds, alterations, a small CNN, training.

The real task cannot be re-run (ViT-B, 3500 forget images, hidden gold models). The miniature
reproduces its PROTOCOL at a scale a laptop trains in minutes:

* an "original" model trained on `retain + forget` until it memorises (train confidence ~ 1);
* several "gold" models trained on `retain` only (the forget images never seen);
* a held-out control set and validation set, and altered copies of the forget images.

Images are float32 [N, 3, 32, 32] in [0, 1] on CPU (alterations run there); the networks run on
MPS/CUDA when available.

Differences with the real task, all deliberate and listed in the report: 10 classes instead of 100,
a 0.65 M-parameter CNN trained from scratch instead of a pretrained ViT-B, 32 px images.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from numpy.typing import NDArray
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.data import list_proxy_image_files

NUM_CLASSES = 10
IMAGE_SIZE = 32
CIFAR_TRAIN_FILES = tuple(f"data_batch_{index}" for index in range(1, 6))
CIFAR_TEST_FILE = "test_batch"
PIXEL_MAX = 255.0
NORMALISATION_CENTER = 0.5
GROUP_COUNT = 8
"""GroupNorm groups: like the ViT's LayerNorm, no batch statistics, so a forget-only batch cannot
silently shift running statistics (a BatchNorm artefact that has no counterpart in the real task)."""
DEFAULT_WIDTH = 32
PAD_PIXELS = 4
"""Random-translation padding of the training augmentation (flip + translate)."""
EVAL_BATCH = 500

CROP_RANGE = (0.6, 0.9)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.6, 1.4)
JPEG_QUALITY = 60
"""Quality 50 of the kit is brutal on a 32 px image; 60 keeps the class visible like q50 does at 224."""
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
ROTATION_DEGREES = 15.0
NOISE_STD = 0.06
CUTOUT_SIDE = 10
STRONG_VIEW_SPREAD = 0.1
ROI_SAMPLING_RATIO = 2
GRAY_FILL = 0.5

KNOWN_FAMILY = ("flip", "crop", "jitter", "jpeg")
"""The 4 alterations of the kit: what methods train on and what the main score is computed with."""
UNKNOWN_FAMILY = ("rotate", "noise", "blur", "cutout")
"""Alterations NO method sees: the sensitivity test of the kit's 'a fifth alteration' risk."""


class MiniBenchError(HackathonError):
    """A miniature-benchmark input is invalid (missing dataset, impossible split)."""


# ------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cifar10:
    """CIFAR-10 as float32 [N, 3, 32, 32] in [0, 1] plus int64 labels."""

    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor


def _read_cifar_batch(path: Path) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    if not path.is_file():
        raise MiniBenchError(f"CIFAR-10 batch not found: {path}")
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    images = np.asarray(raw["data"], dtype=np.uint8).reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    return images, np.asarray(raw["labels"], dtype=np.int64)


def load_cifar10(root: Path) -> Cifar10:
    """Read the python-version CIFAR-10 folder (`cifar-10-batches-py`)."""
    train = [_read_cifar_batch(root / name) for name in CIFAR_TRAIN_FILES]
    test_images, test_labels = _read_cifar_batch(root / CIFAR_TEST_FILE)
    train_images = np.concatenate([images for images, _ in train])
    train_labels = np.concatenate([labels for _, labels in train])
    return Cifar10(
        train_x=torch.from_numpy(train_images).float() / PIXEL_MAX,
        train_y=torch.from_numpy(train_labels),
        test_x=torch.from_numpy(test_images).float() / PIXEL_MAX,
        test_y=torch.from_numpy(test_labels),
    )


def load_ood_proxy(data_root: Path, count: int, seed: int) -> Tensor:
    """Out-of-distribution natural images (COCO / DIV2K / Kodak of the kit) at 32 px.

    The miniature analogue of the kit's retain proxy: real photographs that are NOT from the task
    distribution, centre-cropped to a square and resized. Falls back to an empty-folder error.
    """
    from PIL import Image  # lazy: Pillow ships with torchvision

    files = list_proxy_image_files(data_root)
    if not files:
        raise MiniBenchError(f"No proxy images under {data_root}")
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(files), generator=generator)[:count].sort().values.tolist()
    tensors: list[Tensor] = []
    for index in chosen:
        with Image.open(files[index]) as handle:
            image = handle.convert("RGB")
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        square = image.crop((left, top, left + side, top + side)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC
        )
        array = np.asarray(square, dtype=np.uint8).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).float() / PIXEL_MAX)
    return torch.stack(tensors)


@dataclass(frozen=True)
class WorldSpec:
    """Sizes and seed of one benchmark world (a draw of splits and of every model's seed)."""

    seed: int
    retain_size: int = 4000
    forget_size: int = 400
    control_size: int = 2000
    validation_size: int = 2000
    member_control_size: int = 800


@dataclass(frozen=True)
class World:
    """The data of one world; `retain` is what the golds train on, `retain + forget` the orig."""

    spec: WorldSpec
    retain_x: Tensor
    retain_y: Tensor
    forget_x: Tensor
    forget_y: Tensor
    control_x: Tensor
    control_y: Tensor
    member_control_x: Tensor
    member_control_y: Tensor
    validation_x: Tensor
    validation_y: Tensor


def make_world(data: Cifar10, spec: WorldSpec) -> World:
    """Disjoint retain / forget (from the train split) and control / validation (from test).

    Control and validation are held out from EVERY model, as the subject says ("held-out control
    set"); `member_control` (a slice of retain, seen by every model) is kept for the sensitivity
    check "what if the hidden control set is actually part of the training data".

    Raises:
        MiniBenchError: the requested sizes do not fit in the dataset.
    """
    train_total = spec.retain_size + spec.forget_size
    test_total = spec.control_size + spec.validation_size
    if train_total > len(data.train_x) or test_total > len(data.test_x):
        raise MiniBenchError(f"world {spec} does not fit in CIFAR-10")
    if spec.member_control_size > spec.retain_size:
        raise MiniBenchError("member_control_size exceeds retain_size")
    generator = torch.Generator().manual_seed(spec.seed)
    train_order = torch.randperm(len(data.train_x), generator=generator)
    test_order = torch.randperm(len(data.test_x), generator=generator)
    retain_idx = train_order[: spec.retain_size]
    forget_idx = train_order[spec.retain_size : train_total]
    control_idx = test_order[: spec.control_size]
    validation_idx = test_order[spec.control_size : test_total]
    member_idx = retain_idx[: spec.member_control_size]
    return World(
        spec=spec,
        retain_x=data.train_x[retain_idx],
        retain_y=data.train_y[retain_idx],
        forget_x=data.train_x[forget_idx],
        forget_y=data.train_y[forget_idx],
        control_x=data.test_x[control_idx],
        control_y=data.test_y[control_idx],
        member_control_x=data.train_x[member_idx],
        member_control_y=data.train_y[member_idx],
        validation_x=data.test_x[validation_idx],
        validation_y=data.test_y[validation_idx],
    )


# ------------------------------------------------------------------------------------------
# Alterations (float [N, 3, 32, 32] in [0, 1], CPU, deterministic given the generator)
# ------------------------------------------------------------------------------------------

Alteration = Callable[[Tensor, torch.Generator], Tensor]


def _uniform(count: int, bounds: tuple[float, float], generator: torch.Generator) -> Tensor:
    low, high = bounds
    return (low + (high - low) * torch.rand(count, generator=generator)).view(count, 1, 1, 1)


def flip(images: Tensor, generator: torch.Generator) -> Tensor:
    """Mirror left-right."""
    del generator
    return images.flip(dims=(3,)).contiguous()


def crop_resize(
    images: Tensor,
    generator: torch.Generator,
    low: float = CROP_RANGE[0],
    high: float = CROP_RANGE[1],
) -> Tensor:
    """Random square crop covering `low`..`high` of the area, resized back to 32 px."""
    from torchvision.ops import roi_align

    count = images.shape[0]
    area = low + (high - low) * torch.rand(count, generator=generator)
    side = area.sqrt() * IMAGE_SIZE
    top = torch.rand(count, generator=generator) * (IMAGE_SIZE - side)
    left = torch.rand(count, generator=generator) * (IMAGE_SIZE - side)
    boxes = torch.stack([torch.arange(count).float(), left, top, left + side, top + side], dim=1)
    out: Tensor = roi_align(
        images,
        boxes,
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
        spatial_scale=1.0,
        sampling_ratio=ROI_SAMPLING_RATIO,
        aligned=True,
    )
    return out.clamp(0.0, 1.0)


def jitter(images: Tensor, generator: torch.Generator) -> Tensor:
    """Random brightness, contrast, saturation (each factor drawn per image)."""
    count = images.shape[0]
    luma = torch.tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    pixels = images * _uniform(count, BRIGHTNESS_RANGE, generator)
    gray_mean = (pixels * luma).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    pixels = (pixels - gray_mean) * _uniform(count, CONTRAST_RANGE, generator) + gray_mean
    gray = (pixels * luma).sum(dim=1, keepdim=True)
    pixels = (pixels - gray) * _uniform(count, SATURATION_RANGE, generator) + gray
    return pixels.clamp(0.0, 1.0)


def jpeg(images: Tensor, generator: torch.Generator) -> Tensor:
    """JPEG round trip at `JPEG_QUALITY` (torchvision codec, one call per image)."""
    del generator
    from torchvision.io import decode_jpeg, encode_jpeg

    if len(images) == 0:
        return images.clone()
    decoded: list[Tensor] = []
    for image in images:
        as_uint8 = (image.clamp(0.0, 1.0) * PIXEL_MAX).round().to(torch.uint8)
        decoded.append(decode_jpeg(encode_jpeg(as_uint8, quality=JPEG_QUALITY)).float() / PIXEL_MAX)
    return torch.stack(decoded)


def rotate(images: Tensor, generator: torch.Generator) -> Tensor:
    """Rotation by U(-15, 15) degrees, reflection padding (UNKNOWN family)."""
    count = images.shape[0]
    angle = (torch.rand(count, generator=generator) * 2 - 1) * ROTATION_DEGREES * np.pi / 180.0
    cos, sin = angle.cos(), angle.sin()
    zeros = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, zeros], 1), torch.stack([sin, cos, zeros], 1)], 1)
    grid = torch_f.affine_grid(theta, list(images.shape), align_corners=False)
    return torch_f.grid_sample(images, grid, padding_mode="reflection", align_corners=False)


def noise(images: Tensor, generator: torch.Generator) -> Tensor:
    """Additive Gaussian noise of std `NOISE_STD` (UNKNOWN family)."""
    return (images + NOISE_STD * torch.randn(images.shape, generator=generator)).clamp(0.0, 1.0)


def blur(images: Tensor, generator: torch.Generator) -> Tensor:
    """3x3 binomial blur (UNKNOWN family)."""
    del generator
    kernel_1d = torch.tensor([1.0, 2.0, 1.0])
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]) / 16.0
    weight = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    padded = torch_f.pad(images, (1, 1, 1, 1), mode="reflect")
    return torch_f.conv2d(padded, weight, groups=3)


def cutout(images: Tensor, generator: torch.Generator) -> Tensor:
    """Gray square of `CUTOUT_SIDE` px at a random position (UNKNOWN family)."""
    out = images.clone()
    count = images.shape[0]
    tops = torch.randint(0, IMAGE_SIZE - CUTOUT_SIDE + 1, (count,), generator=generator)
    lefts = torch.randint(0, IMAGE_SIZE - CUTOUT_SIDE + 1, (count,), generator=generator)
    for index in range(count):
        top, left = int(tops[index]), int(lefts[index])
        out[index, :, top : top + CUTOUT_SIDE, left : left + CUTOUT_SIDE] = GRAY_FILL
    return out


ALTERATIONS: dict[str, Alteration] = {
    "flip": flip,
    "crop": crop_resize,
    "jitter": jitter,
    "jpeg": jpeg,
    "rotate": rotate,
    "noise": noise,
    "blur": blur,
    "cutout": cutout,
}


def apply_family(images: Tensor, family: tuple[str, ...], seed: int) -> Tensor:
    """One altered copy per image: image i gets alteration `family[i % len(family)]`.

    Mixing the family inside one set mirrors the hidden altered copies ("modified versions"): each
    forget image has one copy, not all of them. Deterministic given `seed`.
    """
    out = images.clone()
    for offset, name in enumerate(family):
        indices = torch.arange(offset, len(images), len(family))
        if len(indices) == 0:
            continue
        generator = torch.Generator().manual_seed(seed + offset)
        out[indices] = ALTERATIONS[name](images[indices], generator)
    return out


def strong_view(images: Tensor, generator: torch.Generator, min_area: float) -> Tensor:
    """Small crop (`min_area` .. `min_area + 0.1`) plus colour jitter: a view off the memorised pixels."""
    high = min(1.0, min_area + STRONG_VIEW_SPREAD)
    return jitter(crop_resize(images, generator, min(min_area, high), high), generator)


def random_alteration(images: Tensor, generator: torch.Generator) -> Tensor:
    """One randomly chosen KNOWN-family alteration applied to the whole batch (training use)."""
    name = KNOWN_FAMILY[int(torch.randint(len(KNOWN_FAMILY), (1,), generator=generator))]
    return ALTERATIONS[name](images, generator)


# ------------------------------------------------------------------------------------------
# Model and training
# ------------------------------------------------------------------------------------------


class MiniNet(nn.Module):
    """Small VGG-style CNN with GroupNorm (0.65 M parameters at width 48)."""

    def __init__(self, width: int = DEFAULT_WIDTH, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        def block(channels_in: int, channels_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.GroupNorm(GROUP_COUNT, channels_out),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            block(3, width),
            block(width, width),
            nn.MaxPool2d(2),
            block(width, 2 * width),
            block(2 * width, 2 * width),
            nn.MaxPool2d(2),
            block(2 * width, 4 * width),
            block(4 * width, 4 * width),
            nn.MaxPool2d(2),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Logits [N, classes] from images in [0, 1]."""
        centred = (images - NORMALISATION_CENTER) / NORMALISATION_CENTER
        pooled = self.features(centred).mean(dim=(2, 3))
        logits: Tensor = self.head(pooled)
        return logits


def select_device() -> torch.device:
    """MPS when available (Apple GPU), else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class TrainSpec:
    """Hyper-parameters of the from-scratch training of orig / gold models.

    Few images and many epochs on purpose: the original must OVERFIT (memorise) the forget set,
    like the real ViT whose confidence on it is 0.9997.
    """

    epochs: int = 60
    batch_size: int = 256
    peak_lr: float = 0.08
    weight_decay: float = 0.0
    warmup_fraction: float = 0.1
    momentum: float = 0.9
    width: int = DEFAULT_WIDTH
    augment: bool = False

    def fingerprint(self) -> str:
        """Short stable id of the spec (cache key)."""
        return hashlib.sha256(repr(self).encode()).hexdigest()[:10]


def cosine_with_warmup(step: int, total_steps: int, peak: float, warmup_fraction: float) -> float:
    """Linear warmup then cosine decay to zero; `step` is 0-indexed."""
    warmup = int(total_steps * warmup_fraction)
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return float(0.5 * peak * (1.0 + np.cos(np.pi * progress)))


def augment_batch(images: Tensor, generator: torch.Generator) -> Tensor:
    """Random flip and random translation (reflection padding), vectorised, on the images' device."""
    count = images.shape[0]
    flip_mask = (torch.rand(count, generator=generator) < 0.5).to(images.device)
    images = torch.where(flip_mask.view(-1, 1, 1, 1), images.flip(dims=(3,)), images)
    padded = torch_f.pad(images, (PAD_PIXELS,) * 4, mode="reflect")
    span = 2 * PAD_PIXELS + 1
    tops = torch.randint(0, span, (count,), generator=generator).to(images.device)
    lefts = torch.randint(0, span, (count,), generator=generator).to(images.device)
    ramp = torch.arange(IMAGE_SIZE, device=images.device)
    rows = (tops[:, None] + ramp[None, :])[:, :, None]
    cols = (lefts[:, None] + ramp[None, :])[:, None, :]
    batch_index = torch.arange(count, device=images.device)[:, None, None]
    gathered = padded.permute(0, 2, 3, 1)[batch_index, rows, cols]
    return gathered.permute(0, 3, 1, 2).contiguous()


def train_classifier(
    images: Tensor, labels: Tensor, spec: TrainSpec, seed: int, device: torch.device
) -> MiniNet:
    """Train a MiniNet from scratch with SGD + nesterov, warmup + cosine, optional augmentation.

    Reproducible up to the (non-deterministic) accelerator kernels: the seed fixes the
    initialisation, the shuffling and the augmentation draws.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = MiniNet(spec.width).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=spec.peak_lr,
        momentum=spec.momentum,
        nesterov=True,
        weight_decay=spec.weight_decay,
    )
    images_device, labels_device = images.to(device), labels.to(device)
    count = len(images)
    steps_per_epoch = max(1, count // spec.batch_size)
    total_steps = steps_per_epoch * spec.epochs
    step = 0
    for _ in range(spec.epochs):
        order = torch.randperm(count, generator=generator).to(device)
        for batch in range(steps_per_epoch):
            index = order[batch * spec.batch_size : (batch + 1) * spec.batch_size]
            batch_x = images_device[index]
            if spec.augment:
                batch_x = augment_batch(batch_x, generator)
            for group in optimizer.param_groups:
                group["lr"] = cosine_with_warmup(
                    step, total_steps, spec.peak_lr, spec.warmup_fraction
                )
            loss = torch_f.cross_entropy(model(batch_x), labels_device[index])
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            step += 1
    return model.eval()


@torch.no_grad()
def logits_of(model: nn.Module, images: Tensor, device: torch.device) -> NDArray[np.float64]:
    """float64 logits [N, C] of `model` on CPU images (eval mode, batches of `EVAL_BATCH`)."""
    model.eval()
    outputs = [
        model(images[start : start + EVAL_BATCH].to(device)).cpu().double()
        for start in range(0, len(images), EVAL_BATCH)
    ]
    return torch.cat(outputs).numpy().astype(np.float64)


def logits_on_sets(
    model: nn.Module, sets: Mapping[str, Tensor], device: torch.device
) -> dict[str, NDArray[np.float64]]:
    """Logits of `model` on every named set."""
    return {name: logits_of(model, images, device) for name, images in sets.items()}


def load_or_train(
    cache_path: Path | None,
    images: Tensor,
    labels: Tensor,
    spec: TrainSpec,
    seed: int,
    device: torch.device,
) -> MiniNet:
    """Train, or reload the identical model from `cache_path` (rerunning the bench is idempotent)."""
    if cache_path is not None and cache_path.is_file():
        model = MiniNet(spec.width)
        model.load_state_dict(torch.load(cache_path, map_location="cpu", weights_only=True))
        return model.to(device).eval()
    model = train_classifier(images, labels, spec, seed, device)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, temporary)
        temporary.replace(cache_path)  # atomic: a crash never leaves a half-written model
    return model
