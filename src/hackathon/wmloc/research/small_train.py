"""Full-resolution segmentation network trained on synthetic composites (w-loc-small).

The heads of team B2 read frozen extractor features on 16 to 64 px cells: a region under 5 % of
the image is a handful of cells. This network reads the 512 px image itself (plus a fixed
high-pass copy) and predicts the official 128 grid, so a 4 px cell is its native unit.

Protocol (one method per process, every stage bounded by wall-clock time):

1. train on synthetic composites only (`small_gen`): real hole and object shapes cut out of
   (clean, marked) pairs on the fly, plus pre-built rectangle composites;
2. score the REAL train images of the method: an honest validation, since neither their pixels
   nor their mask shapes were used (shapes and bases come from the other two methods);
3. 5-fold fine-tuning on the real train images (folds of team A, seed 2026) for out-of-fold
   probabilities, then one fine-tuning on all of them for the test split.

Outputs ONE `.npz` and nothing else (shared inode quota): probabilities float16 on the 128 grid
for train and test, synthetic-only and fine-tuned, plus the metrics as a JSON string.

Usage:
    python -m hackathon.wmloc.research.small_train --method stegastamp --in-dir /dev/shm/wls \
        --out-dir out --tag ss_a --train-seconds 1200 --finetune-seconds 40
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn
from torch.nn import functional

from hackathon.wmloc.constants import (
    CELL_WATERMARK_FRACTION,
    GRID_SIZE,
    IMAGE_SIZE,
    KNOWN_METHODS,
    METHOD_PIXELSEAL,
)
from hackathon.wmloc.research.data import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    Record,
    load_records,
    load_truth,
    split_dir,
)
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import stratified_folds
from hackathon.wmloc.research.small_gen import load_mask512, other_method_records

CELL = IMAGE_SIZE // GRID_SIZE
BYTE_MAX = 255.0
FOLDS = 5
FOLD_SEED = 2026
"""Folds of team A, reused by every localization team: out-of-fold maps stay comparable."""
HIGH_PASS_GAIN = 10.0
"""A watermark is a few grey levels at most: the high-pass copy is amplified to unit scale."""
TEMPLATE_GAIN = 50.0
INPUT_MEAN = 0.5
INPUT_SCALE = 0.25
PLAIN_CHANNELS = 6
TEMPLATE_CHANNELS = 12
WIDTHS: tuple[int, int, int, int, int] = (32, 64, 128, 256, 384)
PEAK_LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE_FRACTION = 0.05
WARMUP_FRACTION = 0.03
WEIGHT_DECAY = 1e-4
FINETUNE_LEARNING_RATE = 2e-4
DICE_WEIGHT = 0.5
DICE_SMOOTHING = 1.0
RECTANGLE_FILL = 0.95
"""A real region whose bounding box is filled above this is an outpainting rectangle."""
SCALE_RANGE: tuple[float, float] = (0.6, 1.5)
SHIFT_FRACTION = 0.3
MIN_AREA_FRACTION = 0.004
MAX_AREA_FRACTION = 0.45
SMALL_ZONE_FRACTION = 0.05
EVAL_BATCH = 16
EVAL_EVERY_SECONDS = 300.0
LOG_EVERY_STEPS = 200
DECISION_THRESHOLD = 0.5
# Share of each training batch, by kind of composite.
HOLE_SHARE = 0.45
TRANSPLANT_SHARE = 0.2
FINETUNE_REAL_SHARE = 0.25
SAVE_FINAL_ATTEMPTS = 20
GPU_RESERVE_BYTES = 12 * 2**30
REAL_CROP = 384
REAL_CROP_LATTICE = 16
"""Crop offsets stay on the coarsest stride of the network and on the 4 px label grid."""
REAL_CROP_PROBABILITY = 0.5
EXTRACTOR_SIDES: dict[str, tuple[int, int]] = {"pixelseal": (256, 512), "trustmark": (245, 490)}
"""Native side of each extractor and twice that side (finer cells on the 512 px image)."""
EXTRACTOR_CHANNELS: dict[str, int] = {"pixelseal": 6, "trustmark": 4}
EVIDENCE_MARGIN_SCALE = 4.0
EVIDENCE_CLIP = 2.0
SAVE_RETRY_SECONDS = 15.0


class SmallTrainError(ResearchError):
    """No synthetic material, or an inconsistent real split."""


def conv_block(inputs: int, outputs: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """3x3 convolution, batch norm, SiLU."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(outputs),
        nn.SiLU(inplace=True),
    )


class FullResolutionNet(nn.Module):
    """U-shaped network: encoder from 512 px down to 32 px, decoder back up to the 128 grid."""

    def __init__(self, in_channels: int, width_scale: float = 1.0) -> None:
        super().__init__()
        # Widths stay multiples of 8 whatever the scale.
        w0, w1, w2, w3, w4 = (max(8, 8 * round(width * width_scale / 8)) for width in WIDTHS)
        self.enc0 = nn.Sequential(conv_block(in_channels, w0), conv_block(w0, w0))
        self.enc1 = nn.Sequential(conv_block(w0, w1, stride=2), conv_block(w1, w1))
        self.enc2 = nn.Sequential(conv_block(w1, w2, stride=2), conv_block(w2, w2))
        self.enc3 = nn.Sequential(conv_block(w2, w3, stride=2), conv_block(w3, w3))
        self.enc4 = nn.Sequential(
            conv_block(w3, w4, stride=2),
            conv_block(w4, w4, dilation=2),
            conv_block(w4, w4, dilation=4),
            conv_block(w4, w4),
        )
        self.dec3 = nn.Sequential(conv_block(w4 + w3, w3), conv_block(w3, w3))
        self.dec2 = nn.Sequential(conv_block(w3 + w2, w2), conv_block(w2, w2))
        self.head = nn.Conv2d(w2, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Logits [B, 1, 128, 128] of inputs [B, C, 512, 512]."""
        skip2 = self.enc2(self.enc1(self.enc0(inputs)))
        skip3 = self.enc3(skip2)
        deep = self.enc4(skip3)
        up3 = functional.interpolate(deep, scale_factor=2.0, mode="bilinear", align_corners=False)
        mid = self.dec3(torch.cat([up3, skip3], dim=1))
        up2 = functional.interpolate(mid, scale_factor=2.0, mode="bilinear", align_corners=False)
        logits: torch.Tensor = self.head(self.dec2(torch.cat([up2, skip2], dim=1)))
        return logits


class Featurizer:
    """uint8 images -> network inputs (image, high-pass copy, optional re-embedding template)."""

    def __init__(self, method: str, device: str, template: bool, extra_channels: int = 0) -> None:
        self.device = device
        self.extra_channels = extra_channels
        self.embedder: Any = None
        if template:
            if method != METHOD_PIXELSEAL:
                raise SmallTrainError("the re-embedding template is only wired for pixelseal")
            from hackathon.wmloc.research.kit import load_wrapper  # kit environment only

            self.embedder = load_wrapper(method, device).model

    @property
    def channels(self) -> int:
        """Number of input channels of the network."""
        base = TEMPLATE_CHANNELS if self.embedder is not None else PLAIN_CHANNELS
        return base + self.extra_channels

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, bits: torch.Tensor | None, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inputs [B, C, h, w] of uint8 images [B, h, w, 3] (and bits [B, n] if template).

        `extra` holds cached extractor maps [B, extra_channels, h / 4, w / 4], stacked after a
        bilinear upsampling to the image size.
        """
        pixels = images.permute(0, 3, 1, 2).float() / BYTE_MAX
        high = pixels - functional.avg_pool2d(
            functional.pad(pixels, (1, 1, 1, 1), mode="reflect"), 3, stride=1
        )
        parts = [(pixels - INPUT_MEAN) / INPUT_SCALE, high * HIGH_PASS_GAIN]
        if self.embedder is not None:
            if bits is None:
                raise SmallTrainError("the template needs the message of every image")
            marked = self.embedder.embed(pixels, msgs=bits.float(), is_video=False)["imgs_w"]
            residual = (marked.float() - pixels) * TEMPLATE_GAIN
            parts += [residual, residual * high * HIGH_PASS_GAIN]
        if self.extra_channels:
            if extra is None or extra.shape[1] != self.extra_channels:
                raise SmallTrainError("the extractor maps of the batch are missing")
            parts.append(
                functional.interpolate(
                    extra.float(), size=pixels.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        return torch.cat(parts, dim=1).contiguous(memory_format=torch.channels_last)


@dataclass
class Material:
    """Synthetic material of one method, on the training device (uint8)."""

    clean: torch.Tensor
    marked: torch.Tensor
    pair_bits: torch.Tensor
    rect: torch.Tensor
    rect_label: torch.Tensor
    rect_bits: torch.Tensor


def load_material(in_dir: Path, method: str, device: str) -> Material:
    """Concatenate the containers of every generation worker of `method`.

    Raises:
        SmallTrainError: no pair or no rectangle composite was generated.
    """
    merged: dict[str, torch.Tensor] = {}
    keys = ("clean", "marked", "pair_bits", "rect", "rect_label", "rect_bits")
    listing = {
        key: [p for p in sorted(in_dir.glob(f"{method}_w*_{key}.npy")) if ".tmp." not in p.name]
        for key in keys
    }
    # The GPU may be shared with another task: keep only the share of every worker's samples
    # that leaves `GPU_RESERVE_BYTES` free for the network (the same share for every key, so the
    # rows of a pair or of a rectangle stay aligned).
    keep = 1.0
    if device.startswith("cuda"):
        needed = sum(
            int(np.prod(np.load(path, mmap_mode="r").shape))
            for key in keys
            for path in listing[key]
        )
        free, _ = torch.cuda.mem_get_info()
        keep = min(1.0, max(0.0, (free - GPU_RESERVE_BYTES) / max(1, needed)))
        print(f"  material {needed / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB, keep {keep:.2f}")
    for key in keys:
        paths = listing[key]
        if not paths:
            raise SmallTrainError(f"no synthetic container {key} for {method} in {in_dir}")
        # Sized from the headers, then filled file by file: never two copies on the device.
        shapes = [np.load(path, mmap_mode="r").shape for path in paths]
        counts = [int(shape[0] * keep) for shape in shapes]
        store = torch.empty((sum(counts), *shapes[0][1:]), dtype=torch.uint8, device=device)
        cursor = 0
        for path, count in zip(paths, counts, strict=True):
            store[cursor : cursor + count] = torch.from_numpy(np.load(path)[:count]).to(device)
            cursor += count
        merged[key] = store
    if len(merged["clean"]) == 0 or len(merged["rect"]) == 0:
        raise SmallTrainError(f"empty synthetic material for {method}")
    return Material(**merged)


def shape_tensors(method: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Real 512 px shapes of the OTHER methods as float [K, 1, 512, 512]: (holes, objects).

    Holes are the restored regions of the full regime; objects are the non-rectangular regions of
    the other regime (transplantation), the rectangles being covered by `small_gen`.
    """
    holes: list[NDArray[np.bool_]] = []
    objects: list[NDArray[np.bool_]] = []
    for record in other_method_records(method, SPLIT_TRAIN):
        mask = load_mask512(record.name)
        if record.full_watermark:
            holes.append(~mask)
            continue
        rows, columns = np.nonzero(mask)
        if len(rows) == 0:
            continue
        box = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
        if mask.sum() / box < RECTANGLE_FILL:
            objects.append(mask)
    if not holes or not objects:
        raise SmallTrainError("no real shape for the other methods")

    def pack(shapes: list[NDArray[np.bool_]]) -> torch.Tensor:
        return torch.from_numpy(np.stack(shapes)).to(device)[:, None].float()

    return pack(holes), pack(objects)


def augment_shapes(shapes: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random symmetry, scale and shift of float shapes [B, 1, 512, 512]; bool, area kept in range.

    A draw whose area leaves the accepted range falls back to the symmetry alone.
    """
    batch = shapes.shape[0]
    device = shapes.device

    def uniform(low: float, high: float, *size: int) -> torch.Tensor:
        return torch.rand(*size, device=device, generator=generator) * (high - low) + low

    quarter = torch.randint(0, 4, (batch,), device=device, generator=generator).float()
    cosine = torch.cos(quarter * math.pi / 2).round()
    sine = torch.sin(quarter * math.pi / 2).round()
    flip = torch.randint(0, 2, (batch,), device=device, generator=generator).float() * 2 - 1
    # Transpose of the symmetry Q = rotation x diag(flip, 1): rows of Q^T.
    q_t = torch.stack(
        [torch.stack([cosine * flip, sine * flip], 1), torch.stack([-sine, cosine], 1)], 1
    )
    scale = uniform(*SCALE_RANGE, batch)
    shift = uniform(-SHIFT_FRACTION, SHIFT_FRACTION, batch, 2) * 2.0  # normalised units

    def warp(linear: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        theta = torch.cat([linear, offset[:, :, None]], dim=2)
        grid = functional.affine_grid(theta, list(shapes.shape), align_corners=False)
        return functional.grid_sample(shapes, grid, mode="bilinear", align_corners=False) > 0.5

    linear = q_t / scale[:, None, None]
    moved = warp(linear, -(linear @ shift[:, :, None])[:, :, 0])
    plain = warp(q_t, torch.zeros(batch, 2, device=device))
    area = moved.float().mean(dim=(1, 2, 3))
    base = plain.float().mean(dim=(1, 2, 3))
    keep = (area >= MIN_AREA_FRACTION) & (area <= MAX_AREA_FRACTION) & (area > 0.25 * base)
    return torch.where(keep[:, None, None, None], moved, plain)


def official_grid(mask: torch.Tensor) -> torch.Tensor:
    """Official 512 -> 128 rule on bool masks [B, 1, 512, 512]; float labels [B, 1, 128, 128]."""
    return (functional.avg_pool2d(mask.float(), CELL) >= CELL_WATERMARK_FRACTION).float()


class SyntheticSampler:
    """Draws training batches of composites from the material and the real shape banks."""

    def __init__(self, material: Material, method: str, device: str, seed: int) -> None:
        self.material = material
        self.holes, self.objects = shape_tensors(method, device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.device = device

    def _index(self, count: int, size: int) -> torch.Tensor:
        return torch.randint(0, count, (size,), device=self.device, generator=self.generator)

    def draw(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(images uint8 [B, 512, 512, 3], bits [B, n], labels float [B, 1, 128, 128])."""
        material = self.material
        n_hole = max(1, round(batch * HOLE_SHARE))
        n_object = max(1, round(batch * TRANSPLANT_SHARE))
        n_rect = max(1, batch - n_hole - n_object)
        pair = self._index(len(material.clean), n_hole + n_object)
        clean = material.clean[pair]
        marked = material.marked[pair]
        hole = augment_shapes(self.holes[self._index(len(self.holes), n_hole)], self.generator)
        thing = augment_shapes(
            self.objects[self._index(len(self.objects), n_object)], self.generator
        )
        # Marked everywhere but the hole; marked only on the transplanted object.
        is_marked = torch.cat([~hole, thing])
        composite = torch.where(is_marked.permute(0, 2, 3, 1), marked, clean)
        rect = self._index(len(material.rect), n_rect)
        images = torch.cat([composite, material.rect[rect]])
        bits = torch.cat([material.pair_bits[pair], material.rect_bits[rect]])
        labels = torch.cat([official_grid(is_marked), material.rect_label[rect][:, None].float()])
        return images, bits, labels


@dataclass
class RealSplit:
    """Real images of one method (uint8, CPU) with their messages; truth for the train split."""

    names: list[str]
    images: torch.Tensor
    bits: torch.Tensor
    truth: torch.Tensor | None
    is_full: NDArray[np.bool_] | None
    extra: torch.Tensor | None = None
    """Cached extractor maps [n, C, 128, 128] (float16, CPU) when the network stacks them."""


def load_real(method: str, split: str, limit: int | None) -> RealSplit:
    """Images, messages and (train) truth of the records of `method` in `split`."""
    records: list[Record] = [r for r in load_records(split) if r.method == method]
    if limit is not None:
        records = records[:limit]
    names = [r.name for r in records]
    arrays = []
    for name in names:
        with Image.open(split_dir(split) / "images" / f"{name}.png") as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
    images = torch.from_numpy(np.stack(arrays))
    bits = torch.from_numpy(np.stack([r.message for r in records]).astype(np.uint8))
    if split != SPLIT_TRAIN:
        return RealSplit(names, images, bits, None, None)
    truth = torch.from_numpy(load_truth(names))
    is_full = np.array([bool(r.full_watermark) for r in records])
    return RealSplit(names, images, bits, truth, is_full)


def segmentation_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus the Dice loss of BOTH classes (the metric averages both IoU)."""
    logits = logits.float()
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, labels)
    probability = torch.sigmoid(logits)
    dice = logits.new_zeros(())
    for positive, target in ((probability, labels), (1 - probability, 1 - labels)):
        overlap = (positive * target).sum(dim=(1, 2, 3))
        total = positive.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = dice + (1 - (2 * overlap + DICE_SMOOTHING) / (total + DICE_SMOOTHING)).mean()
    return cross_entropy + DICE_WEIGHT * dice


@torch.no_grad()
def predict(
    model: nn.Module, featurizer: Featurizer, split: RealSplit, device: str
) -> NDArray[np.float16]:
    """Probabilities [n, 128, 128] (float16) of the watermarked class."""
    model.eval()
    outputs = []
    for start in range(0, len(split.names), EVAL_BATCH):
        images = split.images[start : start + EVAL_BATCH].to(device)
        bits = split.bits[start : start + EVAL_BATCH].to(device)
        extra = None if split.extra is None else split.extra[start : start + EVAL_BATCH].to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = model(featurizer(images, bits, extra))
        outputs.append(torch.sigmoid(logits.float())[:, 0].cpu())
    model.train()
    return torch.cat(outputs).numpy().astype(np.float16)


def miou_per_image(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Official per-image score: mean IoU of the two classes (1 when a union is empty)."""
    scores = np.zeros(len(truth), dtype=np.float64)
    for value in (False, True):
        p, g = prediction == value, truth == value
        union = (p | g).sum(axis=(1, 2))
        inter = (p & g).sum(axis=(1, 2))
        scores += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return scores / 2


def summarise(probability: NDArray[np.float16], split: RealSplit) -> dict[str, float]:
    """mIoU of thresholded probabilities: overall, by regime, and by size of the zone to find."""
    if split.truth is None or split.is_full is None:
        raise SmallTrainError("no truth for this split")
    truth = split.truth.numpy().astype(bool)
    scores = miou_per_image(probability.astype(np.float32) >= DECISION_THRESHOLD, truth)
    marked_area = truth.mean(axis=(1, 2))
    zone = np.where(split.is_full, 1 - marked_area, marked_area)
    small = zone < SMALL_ZONE_FRACTION
    report = {"all": float(scores.mean())}
    for label, group in (("full", split.is_full), ("rect", ~split.is_full)):
        report[label] = float(scores[group].mean()) if group.any() else float("nan")
        for size, chosen in (("small", group & small), ("large", group & ~small)):
            report[f"{label}_{size}"] = (
                float(scores[chosen].mean()) if chosen.any() else float("nan")
            )
    return report


def learning_rate(fraction: float) -> float:
    """Linear warm-up then cosine decay, as a function of the elapsed share of the time budget."""
    if fraction < WARMUP_FRACTION:
        return PEAK_LEARNING_RATE * fraction / WARMUP_FRACTION
    progress = min(1.0, (fraction - WARMUP_FRACTION) / (1 - WARMUP_FRACTION))
    floor = PEAK_LEARNING_RATE * FINAL_LEARNING_RATE_FRACTION
    return floor + (PEAK_LEARNING_RATE - floor) * 0.5 * (1 + math.cos(math.pi * progress))


class Trainer:
    """Holds the network, the optimiser and the mixed-precision state of one run."""

    def __init__(self, featurizer: Featurizer, device: str, width_scale: float = 1.0) -> None:
        self.device = device
        self.featurizer = featurizer
        network = FullResolutionNet(featurizer.channels, width_scale).to(device)
        self.model: nn.Module = network.to(memory_format=torch.channels_last)
        self.amp = device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=PEAK_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    def step(
        self,
        images: torch.Tensor,
        bits: torch.Tensor,
        labels: torch.Tensor,
        lr: float,
        extra: torch.Tensor | None = None,
    ) -> float:
        """One optimisation step; returns the loss."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        inputs = self.featurizer(images, bits, extra)
        with torch.autocast(device_type=self.device.split(":")[0], enabled=self.amp):
            logits = self.model(inputs)
        loss = segmentation_loss(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        scaled = self.scaler.scale(loss)
        scaled.backward()  # type: ignore[no-untyped-call]  # torch stubs leave backward untyped
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach())

    def snapshot(self) -> dict[str, Any]:
        """Deep copy of the network and optimiser states (restored before each fine-tuning)."""
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Back to a snapshot."""
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


def finetune(
    trainer: Trainer,
    sampler: SyntheticSampler,
    real: RealSplit,
    members: NDArray[np.int64],
    seconds: float,
    batch: int,
    rng: np.random.Generator,
    real_share: float = FINETUNE_REAL_SHARE,
    mapper: ExtractorMapper | None = None,
) -> int:
    """Fine-tune on the real images `members` mixed with synthetic composites; returns the steps.

    With a `mapper`, the synthetic part gets fresh extractor maps and the real part its cached
    ones (`real.extra`).
    """
    if real.truth is None:
        raise SmallTrainError("fine-tuning needs the train truth")
    if mapper is not None and real.extra is None:
        raise SmallTrainError("the extractor maps of the real images are not cached")
    n_real = min(batch - 1, max(1, round(batch * real_share)))
    deadline = time.time() + seconds
    steps = 0
    while time.time() < deadline:
        chosen = torch.from_numpy(rng.choice(members, size=n_real, replace=len(members) < n_real))
        synthetic_images, synthetic_bits, labels = sampler.draw(batch - n_real)
        images = torch.cat([synthetic_images, real.images[chosen].to(trainer.device)])
        bits = torch.cat([synthetic_bits, real.bits[chosen].to(trainer.device)])
        labels = torch.cat([labels, real.truth[chosen][:, None].float().to(trainer.device)])
        extra = None
        if mapper is not None and real.extra is not None:
            extra = torch.cat(
                [mapper(synthetic_images, synthetic_bits), real.extra[chosen].to(trainer.device)]
            )
        trainer.step(images, bits, labels, FINETUNE_LEARNING_RATE, extra)
        steps += 1
    return steps


def _evidence(per_position: torch.Tensor, signs: torch.Tensor) -> list[torch.Tensor]:
    """Message-aware maps of per-position bit logits [B, n, h, w]: scaled margin and agreement."""
    signed = per_position * signs[:, :, None, None]
    margin = (signed.mean(dim=1) / EVIDENCE_MARGIN_SCALE).clamp(-EVIDENCE_CLIP, EVIDENCE_CLIP)
    agreement = (signed > 0).float().mean(dim=1) - 0.5
    return [margin, agreement]


class ExtractorMapper:
    """The method's OWN frozen extractor, loaded once: per-position evidence on the 128 grid.

    The extractor reads the image at its native side and at twice that side (finer cells); each
    pass gives the margin and the agreement of every position with the KNOWN message, plus the
    detection channel for Pixel Seal. C = 6 for Pixel Seal, 4 for TrustMark. Called on cached
    real images (`cache`) and on every synthetic batch (`__call__`), so the network sees the same
    kind of evidence in pre-training and on the real split.
    """

    def __init__(self, method: str, device: str) -> None:
        """Load the frozen extractor of `method` on `device`.

        Raises:
            SmallTrainError: the method has no per-position extractor output (StegaStamp).
        """
        if method not in EXTRACTOR_SIDES:
            raise SmallTrainError(f"no per-position extractor map for {method}")
        from hackathon.wmloc.research.kit import load_wrapper  # kit environment only

        self.method = method
        self.device = device
        self.wrapper = load_wrapper(method, device)

    @property
    def channels(self) -> int:
        """Number of evidence channels."""
        return EXTRACTOR_CHANNELS[self.method]

    @torch.no_grad()
    def _chunk(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Evidence float16 [b, C, 128, 128] on the device of uint8 images [b, h, w, 3]."""
        pixels = images.to(self.device).permute(0, 3, 1, 2).float() / BYTE_MAX
        signs = bits.to(self.device).float() * 2 - 1
        maps: list[torch.Tensor] = []
        for side in EXTRACTOR_SIDES[self.method]:
            resized = functional.interpolate(
                pixels,
                size=(side, side),
                mode="bilinear",
                align_corners=False,
                antialias=side < pixels.shape[-1],
            )
            if self.method == METHOD_PIXELSEAL:
                detector = self.wrapper.det
                latent = detector.convnext(detector.preprocess(resized))
                embedding = detector.pixel_decoder.output_upscaling(latent)
                head = detector.pixel_decoder.linear
                per_position = torch.einsum("bchw,oc->bohw", embedding, head.weight)
                per_position = (per_position + head.bias[None, :, None, None]).float()
                detection = per_position[:, 0] / EVIDENCE_MARGIN_SCALE
                maps += _evidence(per_position[:, 1:], signs)
                maps.append(detection.clamp(-EVIDENCE_CLIP, EVIDENCE_CLIP))
            else:
                net = self.wrapper.net
                hidden = net.maxpool(net.relu(net.bn1(net.conv1(resized * 2 - 1))))
                for layer in ("layer1", "layer2", "layer3", "layer4"):
                    hidden = getattr(net, layer)(hidden)
                per_position = torch.einsum("bchw,oc->bohw", hidden, net.fc.weight)
                per_position = (per_position + net.fc.bias[None, :, None, None]).float()
                maps += _evidence(per_position, signs)
        stacked = torch.stack(
            [
                functional.interpolate(
                    m[:, None], size=(GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=False
                )[:, 0]
                for m in maps
            ],
            dim=1,
        )
        return stacked.half()

    def __call__(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Evidence float16 [B, C, 128, 128] on the device, by chunks of `EVAL_BATCH` images."""
        return torch.cat(
            [
                self._chunk(images[start : start + EVAL_BATCH], bits[start : start + EVAL_BATCH])
                for start in range(0, len(images), EVAL_BATCH)
            ]
        )

    def cache(self, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Evidence float16 [n, C, 128, 128] on the CPU (cached once per real image)."""
        return torch.cat(
            [
                self._chunk(
                    images[start : start + EVAL_BATCH], bits[start : start + EVAL_BATCH]
                ).cpu()
                for start in range(0, len(images), EVAL_BATCH)
            ]
        )


def extractor_maps(
    method: str, device: str, images: torch.Tensor, bits: torch.Tensor
) -> torch.Tensor:
    """Maps of the method's OWN frozen extractor, cached once per image: float16 [n, C, 128, 128].

    Raises:
        SmallTrainError: the method has no per-position extractor output (StegaStamp).
    """
    return ExtractorMapper(method, device).cache(images, bits)


def train_on_real(
    trainer: Trainer,
    images: torch.Tensor,
    bits: torch.Tensor,
    truth: torch.Tensor,
    members: NDArray[np.int64],
    seconds: float,
    batch: int,
    rng: np.random.Generator,
    extra: torch.Tensor | None = None,
) -> int:
    """Train from scratch on the real images `members` only, for `seconds`; returns the steps.

    The only augmentation is a random crop on the 16 px lattice of the network (half of the
    batches): a crop keeps the watermark texture as it is, a flip or a resize would not.

    Args:
        images: uint8 [n, 512, 512, 3] on the training device.
        bits: uint8 [n, n_bits] on the training device.
        truth: uint8 [n, 128, 128] on the training device.
    """
    start = time.time()
    steps = 0
    positions = (IMAGE_SIZE - REAL_CROP) // REAL_CROP_LATTICE + 1
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            return steps
        chosen = torch.from_numpy(rng.choice(members, size=batch, replace=len(members) < batch))
        chosen = chosen.to(images.device)
        batch_images = images[chosen]
        labels = truth[chosen][:, None].float()
        batch_extra = None if extra is None else extra[chosen]
        if rng.random() < REAL_CROP_PROBABILITY:
            top, left = (int(v) * REAL_CROP_LATTICE for v in rng.integers(0, positions, size=2))
            batch_images = batch_images[:, top : top + REAL_CROP, left : left + REAL_CROP]
            if batch_extra is not None:
                batch_extra = batch_extra[
                    :,
                    :,
                    top // CELL : (top + REAL_CROP) // CELL,
                    left // CELL : (left + REAL_CROP) // CELL,
                ]
            labels = labels[
                :,
                :,
                top // CELL : (top + REAL_CROP) // CELL,
                left // CELL : (left + REAL_CROP) // CELL,
            ]
        trainer.step(
            batch_images, bits[chosen], labels, learning_rate(elapsed / seconds), batch_extra
        )
        steps += 1


def run_real_only(args: argparse.Namespace) -> None:
    """Real images only, no synthetic material: 5 folds + all, for several seeds, then the mean.

    Writes the same keys as the fine-tuned recipe (`train_finetuned_oof`, `test_finetuned`), which
    hold the MEAN probability over the seeds, so the assembly tool reads both kinds alike.
    """
    started = time.time()
    device = args.device
    torch.backends.cudnn.benchmark = True
    extra_channels = EXTRACTOR_CHANNELS.get(args.method, 0) if args.extractor_maps else 0
    if args.extractor_maps and not extra_channels:
        raise SmallTrainError(f"no extractor map for {args.method}")
    featurizer = Featurizer(args.method, device, args.template, extra_channels)
    real_train = load_real(args.method, SPLIT_TRAIN, args.limit_real)
    real_test = load_real(args.method, SPLIT_TEST, args.limit_real)
    if real_train.truth is None or real_train.is_full is None:
        raise SmallTrainError("the train split has no truth")
    extra: torch.Tensor | None = None
    if extra_channels:
        real_train.extra = extractor_maps(args.method, device, real_train.images, real_train.bits)
        real_test.extra = extractor_maps(args.method, device, real_test.images, real_test.bits)
        extra = real_train.extra.to(device)
        print(f"{args.tag}: extractor maps cached ({time.time() - started:.0f}s)", flush=True)
    images = real_train.images.to(device)
    bits = real_train.bits.to(device)
    truth = real_train.truth.to(device)
    strata = [f"{args.method}/{flag}" for flag in real_train.is_full]
    folds = stratified_folds(strata, FOLDS, FOLD_SEED)
    everyone = np.arange(len(real_train.names), dtype=np.int64)
    oof_sum = np.zeros((len(real_train.names), GRID_SIZE, GRID_SIZE), dtype=np.float32)
    test_sum = np.zeros((len(real_test.names), GRID_SIZE, GRID_SIZE), dtype=np.float32)
    outputs: dict[str, Any] = {
        "names_train": np.array(real_train.names),
        "names_test": np.array(real_test.names),
        "folds": folds,
    }
    metrics: dict[str, Any] = {"tag": args.tag, "method": args.method, "real_only": True}
    per_seed: list[dict[str, float]] = []
    for seed_index in range(args.real_seeds):
        seed = args.seed + seed_index
        rng = np.random.default_rng(seed)
        out_of_fold = np.zeros_like(oof_sum)
        fold_test = np.zeros_like(test_sum)
        for fold in range(FOLDS):
            torch.manual_seed(seed * FOLDS + fold)
            trainer = Trainer(featurizer, device, args.width_scale)
            members = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            steps = train_on_real(
                trainer,
                images,
                bits,
                truth,
                members,
                args.real_seconds,
                args.real_batch,
                rng,
                extra,
            )
            subset = RealSplit(
                [real_train.names[i] for i in held],
                real_train.images[held],
                real_train.bits[held],
                None,
                None,
                None if real_train.extra is None else real_train.extra[held],
            )
            out_of_fold[held] = predict(trainer.model, featurizer, subset, device)
            if args.fold_ensemble:
                fold_test += predict(trainer.model, featurizer, real_test, device).astype(
                    np.float32
                )
            print(f"  seed {seed} fold {fold}: {steps} steps", flush=True)
        report = summarise(out_of_fold.astype(np.float16), real_train)
        per_seed.append(report)
        print(f"{args.tag} REAL-ONLY seed {seed} out-of-fold: {json.dumps(report)}", flush=True)
        if args.fold_ensemble:
            # The five fold models vote on the test split: no sixth fit, longer folds instead.
            test_sum += fold_test / FOLDS
        else:
            torch.manual_seed(seed * FOLDS + FOLDS)
            trainer = Trainer(featurizer, device, args.width_scale)
            train_on_real(
                trainer,
                images,
                bits,
                truth,
                everyone,
                args.real_seconds,
                args.real_batch,
                rng,
                extra,
            )
            test_sum += predict(trainer.model, featurizer, real_test, device).astype(np.float32)
        oof_sum += out_of_fold
        done = seed_index + 1
        outputs["train_finetuned_oof"] = (oof_sum / done).astype(np.float16)
        outputs["test_finetuned"] = (test_sum / done).astype(np.float16)
        mean_report = summarise(outputs["train_finetuned_oof"], real_train)
        metrics.update({"seeds": done, "per_seed": per_seed, "finetuned_out_of_fold": mean_report})
        print(
            f"{args.tag} REAL-ONLY mean of {done} seed(s) out-of-fold: {json.dumps(mean_report)} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        save(args, outputs, metrics, SAVE_FINAL_ATTEMPTS if done == args.real_seeds else 1)
    print(f"{args.tag} done in {time.time() - started:.0f}s", flush=True)


def run(args: argparse.Namespace) -> None:
    """Train, validate on the real train split, fine-tune, predict the test split, write outputs."""
    if args.real_only:
        run_real_only(args)
        return
    if args.in_dir is None:
        raise SmallTrainError("--in-dir is required unless --real-only is given")
    started = time.time()
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)
    material = load_material(args.in_dir, args.method, device)
    print(
        f"{args.tag}: {len(material.clean)} pairs, {len(material.rect)} rects on {device} "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )
    sampler = SyntheticSampler(material, args.method, device, args.seed)
    # Optional extractor evidence: computed on every synthetic batch, cached for the real images.
    mapper = ExtractorMapper(args.method, device) if args.extractor_maps else None
    featurizer = Featurizer(
        args.method, device, args.template, 0 if mapper is None else mapper.channels
    )
    trainer = Trainer(featurizer, device, args.width_scale)
    real_train = load_real(args.method, SPLIT_TRAIN, args.limit_real)
    real_test = load_real(args.method, SPLIT_TEST, args.limit_real)
    if mapper is not None:
        real_train.extra = mapper.cache(real_train.images, real_train.bits)
        real_test.extra = mapper.cache(real_test.images, real_test.bits)
        print(f"{args.tag}: extractor maps cached ({time.time() - started:.0f}s)", flush=True)
    print(
        f"{args.tag}: {len(real_train.names)} real train, {len(real_test.names)} real test, "
        f"{len(sampler.holes)} holes, {len(sampler.objects)} objects ({time.time() - started:.0f}s)",
        flush=True,
    )
    history: list[dict[str, float]] = []
    train_start = time.time()
    last_eval = train_start
    step = 0
    running = 0.0
    while True:
        elapsed = time.time() - train_start
        if elapsed >= args.train_seconds:
            break
        images, bits, labels = sampler.draw(args.batch)
        extra = None if mapper is None else mapper(images, bits)
        running += trainer.step(
            images, bits, labels, learning_rate(elapsed / args.train_seconds), extra
        )
        step += 1
        if step % LOG_EVERY_STEPS == 0:
            rate = step * args.batch / (time.time() - train_start)
            print(
                f"  step {step} loss {running / LOG_EVERY_STEPS:.4f} {rate:.0f} img/s "
                f"t={time.time() - train_start:.0f}s",
                flush=True,
            )
            running = 0.0
        if time.time() - last_eval > EVAL_EVERY_SECONDS:
            last_eval = time.time()
            report = summarise(predict(trainer.model, featurizer, real_train, device), real_train)
            history.append({"seconds": time.time() - train_start, **report})
            print(
                f"  real train at {time.time() - train_start:.0f}s: {json.dumps(report)}",
                flush=True,
            )
    train_synthetic = predict(trainer.model, featurizer, real_train, device)
    test_synthetic = predict(trainer.model, featurizer, real_test, device)
    synthetic_report = summarise(train_synthetic, real_train)
    print(f"{args.tag} SYNTHETIC-ONLY real train: {json.dumps(synthetic_report)}", flush=True)
    outputs: dict[str, Any] = {
        "names_train": np.array(real_train.names),
        "names_test": np.array(real_test.names),
        "train_synthetic": train_synthetic,
        "test_synthetic": test_synthetic,
    }
    metrics: dict[str, Any] = {
        "tag": args.tag,
        "method": args.method,
        "template": bool(args.template),
        "extractor_maps": mapper is not None,
        "width_scale": args.width_scale,
        "pairs": len(material.clean),
        "rects": len(material.rect),
        "steps": step,
        "samples_per_second": step * args.batch / max(1e-9, time.time() - train_start),
        "synthetic_only": synthetic_report,
        "history": history,
    }
    wants_finetune = args.finetune_seconds > 0 and real_train.is_full is not None
    save(args, outputs, metrics, 1 if wants_finetune else SAVE_FINAL_ATTEMPTS)
    if wants_finetune and real_train.is_full is not None:
        base_state = trainer.snapshot()
        strata = [f"{args.method}/{flag}" for flag in real_train.is_full]
        folds = stratified_folds(strata, FOLDS, FOLD_SEED)
        outputs["folds"] = folds
        # Round "" is the reference recipe; round "_b" is a longer one with more real images.
        rounds = [("", args.finetune_seconds, FINETUNE_REAL_SHARE)]
        if args.finetune_b_seconds > 0:
            rounds.append(("_b", args.finetune_b_seconds, args.finetune_b_real_share))
        for index, (suffix, seconds, share) in enumerate(rounds):
            out_of_fold = np.zeros_like(train_synthetic)
            for fold in range(FOLDS):
                trainer.restore(base_state)
                members = np.flatnonzero(folds != fold)
                held = np.flatnonzero(folds == fold)
                steps = finetune(
                    trainer, sampler, real_train, members, seconds, args.batch, rng, share, mapper
                )
                subset = RealSplit(
                    [real_train.names[i] for i in held],
                    real_train.images[held],
                    real_train.bits[held],
                    None,
                    None,
                    None if real_train.extra is None else real_train.extra[held],
                )
                out_of_fold[held] = predict(trainer.model, featurizer, subset, device)
                print(f"  round {suffix or 'a'} fold {fold}: {steps} fine-tuning steps", flush=True)
            finetuned_report = summarise(out_of_fold, real_train)
            print(
                f"{args.tag} FINE-TUNED{suffix} ({seconds:.0f}s, real share {share}) out-of-fold "
                f"real train: {json.dumps(finetuned_report)}",
                flush=True,
            )
            outputs[f"train_finetuned_oof{suffix}"] = out_of_fold
            metrics[f"finetuned_out_of_fold{suffix}"] = finetuned_report
            save(args, outputs, metrics, 1)
            trainer.restore(base_state)
            everyone = np.arange(len(real_train.names), dtype=np.int64)
            finetune(
                trainer, sampler, real_train, everyone, seconds, args.batch, rng, share, mapper
            )
            outputs[f"test_finetuned{suffix}"] = predict(
                trainer.model, featurizer, real_test, device
            )
            last = index == len(rounds) - 1
            save(args, outputs, metrics, SAVE_FINAL_ATTEMPTS if last else 1)
    print(f"{args.tag} done in {time.time() - started:.0f}s", flush=True)


def save(
    args: argparse.Namespace, outputs: dict[str, Any], metrics: dict[str, Any], attempts: int
) -> bool:
    """Write ONE container (probabilities and the metrics as a JSON string); True when written.

    The inode quota of the shared scratch is full at times: a single file per run, no temporary
    file once the container exists (rewriting in place needs no new inode), and `attempts` tries
    `SAVE_RETRY_SECONDS` apart. Not atomic: a kill during the write leaves a torn container.
    """
    container = args.out_dir / f"{args.tag}.npz"
    payload = {**outputs, "metrics_json": np.array(json.dumps(metrics))}
    for attempt in range(attempts):
        try:
            with open(container, "wb") as handle:
                np.savez(handle, **payload)
            return True
        except OSError as error:
            print(
                f"  write of {container} failed ({error}), try {attempt + 1}/{attempts}", flush=True
            )
            if attempt + 1 < attempts:
                time.sleep(SAVE_RETRY_SECONDS)
    return False


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Full-resolution segmentation on synthetic data.")
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument("--in-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-seconds", type=float, default=0.0)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--real-seconds", type=float, default=78.0)
    parser.add_argument("--real-seeds", type=int, default=3)
    parser.add_argument("--real-batch", type=int, default=16)
    parser.add_argument("--extractor-maps", action="store_true")
    parser.add_argument("--fold-ensemble", action="store_true")
    parser.add_argument("--width-scale", type=float, default=1.0)
    parser.add_argument("--finetune-seconds", type=float, default=0.0)
    parser.add_argument("--finetune-b-seconds", type=float, default=0.0)
    parser.add_argument("--finetune-b-real-share", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--limit-real", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
