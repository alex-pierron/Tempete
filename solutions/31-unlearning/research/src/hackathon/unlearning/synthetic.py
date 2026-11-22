"""Synthetic images drawn from the original model alone (V_Fusion), and the gates they must pass.

Generation, after Plug-In Inversion (Ghiasi et al., ICML 2022): each image starts from noise at
low resolution and is optimised so that the original predicts its class through an ensemble of
random augmentations (pixel jitter, per-channel colour shift), at a resolution that grows by
stages (zoom). No external image, no pretrained generator, no forget image: only the original's
weights, as the subject requires. In the last stage an image stops once the original gives its
class `confidence_cap` (0.9, the level of the strong views): at T = 2 its output must still shape
the wrong classes. Flips, crops and blur are never used here: gate a needs transformations the
generation did not see.

Pools, by class: A = classes of the forget set's pseudo-labels minus the set-aside ones, B =
classes absent from the forget set, C = the set-aside classes (team hypothesis, not a measure:
classes the gold may never have seen). Per class, the last `thermometer_per_class` images are
kept for measuring drift, never for a loss.

Gates, thresholds fixed before any look at the results:

a. robustness: under flip, crop (60-90 % of the area) and blur, the original keeps the class on
   at least 80 % of the images, for each transformation;
b. in the class: the head-input class token is nearer its class centroid on real forget images
   than any other centroid, for at least 70 % of the images. Pool B has no real image of its
   classes: not measurable (a head-weight prototype would be circular, the generation maximises
   exactly that score);
c. variety: at most 5 % of same-class pairs are as similar as an image and its own flip;
c'. not the forget set: an image whose middle-block class token is as near a forget image as a
   typical forget image is to its own strong view is dropped (`kept` = False); the gate fails if
   more than 10 % of pool A goes;
d. repair: noise on the weights makes the original change its mind on about 20 % of strong views
   of real forget images; distillation at T = 2 on a pool's kept anchor images alone must bring
   the agreement back to at least 0.95. A control repairs on strong views of the other half of
   the forget set with the same budget: if it fails too, the budget is short and gate d says
   nothing about the pool.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn
from torchvision.transforms.functional import gaussian_blur

from hackathon.unlearning.alterations import (
    horizontal_flip,
    random_augment,
    random_crop_resize,
    strong_view,
)
from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.constants import (
    DEFAULT_BATCH_SIZE,
    IMAGE_SIZE,
    NORMALISATION_CENTER,
    NORMALISATION_SCALE,
    NUM_CLASSES,
    PIXEL_MAX,
    SCORING_TEMPERATURE,
)
from hackathon.unlearning.data import DataError, stratified_split
from hackathon.unlearning.diagnose import PRE_LOGITS, capture_cls
from hackathon.unlearning.model import model_device, predict, server_autocast
from hackathon.unlearning.training import (
    kd_from_teacher,
    learning_rate_at,
    prepare_student,
    prepare_teacher,
    student_logits,
    teacher_logits,
    training_autocast,
)

SET_ASIDE_CLASSES = (0, 6, 7, 8, 29, 30, 31, 33, 38)
"""Classes the gold may never have seen (team hypothesis from Lucas's note, not a measure)."""
POOLS = ("A", "B", "C")
"""Pool codes, stored as indices into this tuple in the synthetic file."""
POOL_A, POOL_B, POOL_C = POOLS
ZOOM_STAGES = (56, 112, IMAGE_SIZE)
JITTER_PIXELS = 16
"""Largest circular shift of an augmented copy: one ViT patch."""
COLOR_SHIFT = 0.5
"""Per image and channel, pixels -> pixels * exp(U(-c, c)) + U(-c, c), in the model's [-1, 1]."""
INIT_SCALE = 0.2
CAP_CHECK_EVERY = 10
MAX_PER_CLASS = 1000
"""Image seeds are (seed * classes + class) * MAX_PER_CLASS + index: unique below this count."""

ROBUSTNESS_MIN = 0.8
IN_CLASS_MIN = 0.7
NEAR_COPY_MAX = 0.05
NEAR_COPY_QUANTILE = 0.05
"""A pair is a near-copy above this quantile of the image-vs-its-own-flip similarities."""
NOT_FORGET_MAX_DROPPED = 0.10
NOT_FORGET_VIEW_AREA = 0.5
"""Strong views of the forget images that calibrate c' (the bench's middle crop area)."""
BLUR_KERNEL, BLUR_SIGMA = 5, 1.0
REPAIR_MIN = 0.95
REPAIR_EVAL_AREA = 0.5
REPAIR_CONTROL_AREA = 0.25
"""Control repair views: the anchor area of SCRUB conforme (ANCHOR_VIEW_MIN_AREA)."""
REPAIR_WARMUP = 0.1
DAMAGE_SEARCH_STEPS = 12
DAMAGE_SIGMA_BOUNDS = (1e-4, 1.0)
"""Noise scale, relative to each weight tensor's standard deviation."""
CONTROL = "views"
GRID_SIZE = 64


@dataclass(frozen=True)
class SynthConfig:
    """Generation settings (defaults proposed, not measured)."""

    per_class: int = 20
    thermometer_per_class: int = 5
    steps_per_stage: int = 100
    lr: float = 0.05
    ensemble: int = 4
    batch_size: int = 32
    confidence_cap: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.per_class <= MAX_PER_CLASS:
            raise ValueError(f"per_class must be in 1..{MAX_PER_CLASS}, got {self.per_class}")
        if not 0 <= self.thermometer_per_class < self.per_class:
            raise ValueError("thermometer_per_class must leave at least one anchor image")
        if self.ensemble < 1 or self.batch_size < 1 or self.steps_per_stage < 0:
            raise ValueError("ensemble and batch_size must be >= 1, steps_per_stage >= 0")
        if not 0.0 <= self.confidence_cap <= 1.0:
            raise ValueError(f"confidence_cap must be in [0, 1], got {self.confidence_cap}")


@dataclass(frozen=True)
class RepairConfig:
    """Gate d settings."""

    steps: int = 300
    lr: float = 3e-5
    batch_size: int = 32
    disagreement: float = 0.2
    seed: int = 0


def class_pools(
    forget_labels: Tensor, set_aside: Sequence[int], classes: int = NUM_CLASSES
) -> dict[int, str]:
    """Pool of every class: C if set aside, else A if the forget set has it, else B."""
    in_forget, aside = set(forget_labels.tolist()), set(set_aside)
    return {
        c: POOL_C if c in aside else POOL_A if c in in_forget else POOL_B for c in range(classes)
    }


def shard_plan(
    config: SynthConfig, shard: int, shards: int, classes: int = NUM_CLASSES
) -> tuple[Tensor, Tensor, Tensor]:
    """(labels, seeds, thermometer) of the images of shard `shard` (classes c with c % shards ==
    shard), `per_class` per class; seeds depend on (seed, class, index) only, not on the shard."""
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in 0..{shards - 1}, got {shard}")
    mine = [c for c in range(classes) if c % shards == shard]
    index = torch.arange(config.per_class).repeat(len(mine))
    labels = torch.tensor(mine, dtype=torch.long).repeat_interleave(config.per_class)
    seeds = (config.seed * classes + labels) * MAX_PER_CLASS + index
    thermometer = index >= config.per_class - config.thermometer_per_class
    return labels, seeds, thermometer


def _full_size(pixels: Tensor) -> Tensor:
    if pixels.shape[-1] == IMAGE_SIZE:
        return pixels
    return torch_f.interpolate(pixels, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")


def _augment(pixels: Tensor, generator: torch.Generator) -> Tensor:
    """One augmented copy: a circular shift of the batch, a colour shift per image and channel."""
    count, device = pixels.shape[0], pixels.device
    dy, dx = torch.randint(-JITTER_PIXELS, JITTER_PIXELS + 1, (2,), generator=generator).tolist()
    shift = (2 * torch.rand(count, 3, 1, 1, generator=generator) - 1) * COLOR_SHIFT
    scale = ((2 * torch.rand(count, 3, 1, 1, generator=generator) - 1) * COLOR_SHIFT).exp()
    return pixels.roll(shifts=(dy, dx), dims=(2, 3)) * scale.to(device) + shift.to(device)


@torch.no_grad()
def _target_confidence(teacher: nn.Module, pixels: Tensor, labels: Tensor) -> Tensor:
    with server_autocast(pixels.device):
        logits = teacher(_full_size(pixels)).float()
    return torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)


def _invert(
    teacher: nn.Module, labels: Tensor, seeds: Tensor, config: SynthConfig
) -> tuple[Tensor, Tensor]:
    """(pixels in [-1, 1] at full size, reached the cap) for one batch of target classes."""
    device = model_device(teacher)
    first = ZOOM_STAGES[0]
    noise = [
        torch.randn(3, first, first, generator=torch.Generator().manual_seed(int(s))) for s in seeds
    ]
    pixels = (torch.stack(noise) * INIT_SCALE).to(device)
    target = labels.to(device)
    generator = torch.Generator().manual_seed(int(seeds[0]))
    done = torch.zeros(len(labels), dtype=torch.bool, device=device)
    for stage, size in enumerate(ZOOM_STAGES):
        pixels = torch_f.interpolate(pixels, size=(size, size), mode="bilinear").detach()
        pixels.requires_grad_(True)
        optimizer = torch.optim.Adam([pixels], lr=config.lr)
        last = stage == len(ZOOM_STAGES) - 1
        frozen = pixels.detach().clone()
        for step in range(config.steps_per_stage + (1 if last else 0)):
            if last and (step % CAP_CHECK_EVERY == 0 or step == config.steps_per_stage):
                newly = (
                    _target_confidence(teacher, pixels, target) >= config.confidence_cap
                ) & ~done
                frozen[newly] = pixels.detach()[newly]
                done |= newly
                if bool(done.all()) or step == config.steps_per_stage:
                    break
            full = _full_size(pixels)
            copies = torch.cat([_augment(full, generator) for _ in range(config.ensemble)])
            with training_autocast(device, True):
                logits = teacher(copies).float()
            losses = torch_f.cross_entropy(logits, target.repeat(config.ensemble), reduction="none")
            loss = (losses.view(config.ensemble, -1).mean(dim=0) * (~done)).sum()
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(-1.0, 1.0)
                pixels[done] = frozen[done]  # Adam's momentum must not move a stopped image
    return pixels.detach(), done.cpu()


def to_uint8(pixels: Tensor) -> Tensor:
    """Model-space float NCHW in [-1, 1] -> uint8 NHWC on CPU (the forget images' format)."""
    unit = (pixels.float().cpu() * NORMALISATION_SCALE + NORMALISATION_CENTER).clamp(0.0, 1.0)
    return (unit * PIXEL_MAX).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def generate(
    model: nn.Module, labels: Tensor, seeds: Tensor, config: SynthConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """(uint8 images [N, 224, 224, 3], the original's confidence in their class on those uint8
    images, computed like the server, and whether the optimisation reached the cap)."""
    teacher = prepare_teacher(model, model_device(model))
    images: list[Tensor] = []
    reached: list[Tensor] = []
    for start in range(0, len(labels), config.batch_size):
        part = slice(start, start + config.batch_size)
        pixels, done = _invert(teacher, labels[part], seeds[part], config)
        images.append(to_uint8(pixels))
        reached.append(done)
    out = torch.cat(images)
    probs = torch.softmax(predict(teacher, out, config.batch_size), dim=1)
    return out, probs.gather(1, labels[:, None]).squeeze(1), torch.cat(reached)


# ----- gates ----------------------------------------------------------------------------------


def _tokens(
    model: nn.Module, images: Tensor, blocks: Sequence[int], batch: int
) -> tuple[Tensor, dict[int, Tensor]]:
    """(logits, L2-normalised class token after each block of `blocks`, PRE_LOGITS = the head
    input), one forward pass, float32 CPU."""
    with capture_cls(model, blocks) as store:
        logits = predict(model, images, batch)
    return logits, {block: torch_f.normalize(torch.cat(store[block]), dim=1) for block in blocks}


def _classes(labels: Tensor) -> Tensor:
    """The distinct labels, in increasing order."""
    return torch.tensor(sorted(set(labels.tolist())), dtype=torch.long)


def _share(mask: Tensor) -> float:
    return float(mask.double().mean()) if len(mask) else math.nan


def _blur(images: Tensor) -> Tensor:
    nchw = images.permute(0, 3, 1, 2)
    blurred: Tensor = gaussian_blur(nchw, [BLUR_KERNEL, BLUR_KERNEL], [BLUR_SIGMA, BLUR_SIGMA])
    return blurred.permute(0, 2, 3, 1).contiguous()


def gate_robustness(
    model: nn.Module, images: Tensor, labels: Tensor, seed: int, batch: int
) -> dict[str, Tensor]:
    """Per transformation never used by the generation: does the original keep the class?"""
    transformed = {
        "flip": horizontal_flip(images, torch.Generator().manual_seed(seed)),
        "crop": random_crop_resize(images, torch.Generator().manual_seed(seed)),
        "blur": _blur(images),
    }
    return {
        name: predict(model, view, batch).argmax(dim=1) == labels
        for name, view in transformed.items()
    }


def gate_in_class(
    synth_tokens: Tensor, labels: Tensor, real_tokens: Tensor, real_labels: Tensor
) -> Tensor:
    """Per image: nearest real class centroid is its own class (False where it has no centroid)."""
    classes = _classes(real_labels)
    centroids = torch_f.normalize(
        torch.stack([real_tokens[real_labels == c].mean(dim=0) for c in classes]), dim=1
    )
    nearest = classes[(synth_tokens @ centroids.T).argmax(dim=1)]
    return nearest == labels


def gate_variety(tokens: Tensor, flipped: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """(near-copy count, pair count) per class of `labels`, in increasing order."""
    threshold = torch.quantile((tokens * flipped).sum(dim=1), NEAR_COPY_QUANTILE)
    copies, pairs = [], []
    for c in _classes(labels):
        group = tokens[labels == c]
        upper = torch.triu(torch.ones(len(group), len(group), dtype=torch.bool), diagonal=1)
        similarity = (group @ group.T)[upper]
        copies.append(int((similarity >= threshold).sum()))
        pairs.append(len(similarity))
    return torch.tensor(copies), torch.tensor(pairs)


def gate_not_forget(
    synth_tokens: Tensor, forget_tokens: Tensor, forget_view_tokens: Tensor
) -> tuple[Tensor, float]:
    """(kept, threshold): drop an image as near a forget image as a forget image's own view."""
    threshold = float((forget_tokens * forget_view_tokens).sum(dim=1).median())
    nearest = (synth_tokens @ forget_tokens.T).max(dim=1).values
    return nearest < threshold, threshold


def _weight_noise(model: nn.Module, seed: int) -> list[Tensor]:
    """Fixed N(0, 1) draws times each floating weight tensor's std (zero for constant tensors)."""
    generator = torch.Generator().manual_seed(seed)
    noise = []
    for parameter in model.parameters():
        scale = float(parameter.detach().float().std()) if parameter.numel() > 1 else 0.0
        draw = torch.randn(parameter.shape, generator=generator) * scale
        noise.append(draw.to(parameter.device, parameter.dtype))
    return noise


def _damaged(model: nn.Module, noise: Sequence[Tensor], sigma: float) -> nn.Module:
    copy_ = copy.deepcopy(model)
    with torch.no_grad():
        for parameter, draw in zip(copy_.parameters(), noise, strict=True):
            parameter.add_(draw, alpha=sigma)
    return copy_.eval()


def _agreement(model: nn.Module, images: Tensor, reference: Tensor, batch: int) -> float:
    return float((predict(model, images, batch).argmax(dim=1) == reference).double().mean())


def calibrate_damage(
    model: nn.Module, images: Tensor, reference: Tensor, config: RepairConfig, batch: int
) -> tuple[float, nn.Module]:
    """(sigma, damaged model): bisection in log-sigma for 1 - agreement = config.disagreement."""
    noise = _weight_noise(model, config.seed)
    low, high = (math.log(bound) for bound in DAMAGE_SIGMA_BOUNDS)
    for _ in range(DAMAGE_SEARCH_STEPS):
        middle = (low + high) / 2
        disagreement = 1.0 - _agreement(
            _damaged(model, noise, math.exp(middle)), images, reference, batch
        )
        low, high = (middle, high) if disagreement < config.disagreement else (low, middle)
    sigma = math.exp((low + high) / 2)
    return sigma, _damaged(model, noise, sigma)


def repair(
    original: nn.Module, damaged: nn.Module, anchor: Tensor, config: RepairConfig, *, views: bool
) -> nn.Module:
    """Distil the original into a copy of `damaged` at T = 2 on augmented `anchor` images (strong
    views of them when `views`, the control), with the repository's warmup-cosine schedule."""
    device = model_device(original)
    settings = UnlearnConfig(lr=config.lr, batch_size=config.batch_size)
    teacher = prepare_teacher(original, device)
    student = prepare_student(damaged, device, settings)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(config.seed)
    for step in range(config.steps):
        picked = anchor[torch.randint(len(anchor), (config.batch_size,), generator=generator)]
        batch = (
            strong_view(picked, generator, REPAIR_CONTROL_AREA)
            if views
            else random_augment(picked, generator)
        )
        loss = kd_from_teacher(
            teacher_logits(teacher, batch, device),
            student_logits(student, batch, device, settings),
            SCORING_TEMPERATURE,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, config.steps, config.lr, REPAIR_WARMUP)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return student.eval()


def gate_repair(
    model: nn.Module,
    forget: Tensor,
    forget_labels: Tensor,
    anchors: Mapping[str, Tensor],
    config: RepairConfig,
    batch: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    """Gate d: damage, then repair on each pool's anchors and on the control's forget views."""
    control_index, eval_index = stratified_split(forget_labels, 0.5, config.seed)
    evaluation = strong_view(
        forget[eval_index], torch.Generator().manual_seed(config.seed), REPAIR_EVAL_AREA
    )
    reference = predict(model, evaluation, batch).argmax(dim=1)
    sigma, damaged = calibrate_damage(model, evaluation, reference, config, batch)
    after: dict[str, float | None] = {}
    for name, images in {**anchors, CONTROL: forget[control_index]}.items():
        fixed = (
            repair(model, damaged, images, config, views=name == CONTROL) if len(images) else None
        )
        after[name] = None if fixed is None else _agreement(fixed, evaluation, reference, batch)
    control = after[CONTROL]
    return {
        "sigma": sigma,
        "damaged_agreement": _agreement(damaged, evaluation, reference, batch),
        "agreement": after,  # None: no kept anchor image left in the pool
        "passed": {name: None if v is None else v >= REPAIR_MIN for name, v in after.items()},
        "interpretable": control is not None and control >= REPAIR_MIN,
        "eval_images": len(evaluation),
    }


def run_gates(
    model: nn.Module,
    forget: Tensor,
    synth: Mapping[str, Tensor],
    repair_config: RepairConfig | None,
    batch: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, object], Tensor]:
    """(report, kept) for a synthetic set {images, labels, pool, thermometer}; gate d is skipped
    when `repair_config` is None or has no step."""
    images, labels, pool = synth["images"], synth["labels"], synth["pool"]
    seed = repair_config.seed if repair_config else 0
    middle = len(cast(nn.ModuleList, model.blocks)) // 2
    blocks = (PRE_LOGITS, middle)
    _, synth_tokens = _tokens(model, images, blocks, batch)
    forget_logits, forget_tokens = _tokens(model, forget, blocks, batch)
    forget_labels = forget_logits.argmax(dim=1)
    flip = horizontal_flip(images, torch.Generator())
    _, flipped = _tokens(model, flip, (PRE_LOGITS,), batch)
    views = strong_view(forget, torch.Generator().manual_seed(seed), NOT_FORGET_VIEW_AREA)
    _, view_tokens = _tokens(model, views, (middle,), batch)
    kept, threshold = gate_not_forget(
        synth_tokens[middle], forget_tokens[middle], view_tokens[middle]
    )
    head = synth_tokens[PRE_LOGITS]
    robust = gate_robustness(model, images, labels, seed, batch)
    in_class = gate_in_class(head, labels, forget_tokens[PRE_LOGITS], forget_labels)
    has_centroid = torch.isin(labels, _classes(forget_labels))
    report: dict[str, dict[str, object]] = {}
    for code, name in enumerate(POOLS):
        member = pool == code
        if not bool(member.any()):
            continue
        robustness = {k: _share(v[member]) for k, v in robust.items()}
        copies, pairs = gate_variety(head[member], flipped[PRE_LOGITS][member], labels[member])
        near_copy = float(copies.sum()) / max(1, int(pairs.sum()))
        measurable = member & has_centroid
        in_class_share = _share(in_class[measurable]) if bool(measurable.any()) else None
        dropped = 1.0 - _share(kept[member])
        report[name] = {
            "images": int(member.sum()),
            "robustness": robustness,
            "a_passed": min(robustness.values()) >= ROBUSTNESS_MIN,
            "in_class": in_class_share,
            "b_passed": None if in_class_share is None else in_class_share >= IN_CLASS_MIN,
            "near_copy_share": near_copy,
            "c_passed": near_copy <= NEAR_COPY_MAX,
            "dropped_near_forget": dropped,
            "c_prime_passed": dropped <= NOT_FORGET_MAX_DROPPED if name == POOL_A else None,
        }
    result: dict[str, object] = {"pools": report, "not_forget_threshold": threshold}
    if repair_config is not None and repair_config.steps > 0:
        anchor = kept & ~synth["thermometer"]
        anchors = {POOLS[c]: images[anchor & (pool == c)] for c in (0, 1)}
        result["repair"] = gate_repair(model, forget, forget_labels, anchors, repair_config, batch)
    return result, kept


SYNTHETIC_KEYS = ("images", "labels", "pool", "thermometer")


def save_synthetic(path: Path, tensors: Mapping[str, Tensor], meta: Mapping[str, object]) -> None:
    """Tensors in `path` (weights-only loadable), metadata in the JSON sidecar next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(tensors), path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_synthetic(paths: Sequence[Path]) -> dict[str, Tensor]:
    """Concatenation of synthetic files (the shards of one generation), keys checked."""
    parts = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    for path, part in zip(paths, parts, strict=True):
        missing = [key for key in SYNTHETIC_KEYS if key not in part]
        if missing or set(part) != set(parts[0]):
            raise DataError(f"{path}: not a synthetic file of the same kind (missing {missing})")
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def anchor_images(synth: Mapping[str, Tensor], pools: str) -> Tensor:
    """The V_Fusion anchor: kept, non-thermometer images of `pools` (codes of POOLS, e.g. "AB")."""
    if "kept" not in synth:
        raise DataError("not a gated synthetic file (no `kept`): run synth-gates on it first")
    unknown = set(pools) - set(POOLS)
    if unknown:
        raise DataError(f"unknown pools {sorted(unknown)}; choose from {list(POOLS)}")
    codes = torch.tensor([POOLS.index(name) for name in pools])
    chosen = synth["kept"].bool() & ~synth["thermometer"].bool()
    chosen &= torch.isin(synth["pool"].long(), codes)
    return synth["images"][chosen]


def save_grid(images: Tensor, labels: Tensor, path: Path, per_class: int = 4) -> None:
    """Thumbnails, `per_class` per class in class order, four classes per row: to look at."""
    from torchvision.utils import save_image

    picked = torch.cat([torch.nonzero(labels == c).flatten()[:per_class] for c in _classes(labels)])
    unit = images[picked].permute(0, 3, 1, 2).float() / PIXEL_MAX
    small = torch_f.interpolate(unit, size=(GRID_SIZE, GRID_SIZE), mode="area")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(small, path, nrow=4 * per_class, padding=2)


__all__ = [
    "POOLS",
    "SET_ASIDE_CLASSES",
    "SYNTHETIC_KEYS",
    "RepairConfig",
    "SynthConfig",
    "calibrate_damage",
    "class_pools",
    "gate_in_class",
    "gate_not_forget",
    "gate_repair",
    "gate_robustness",
    "gate_variety",
    "generate",
    "load_synthetic",
    "repair",
    "run_gates",
    "save_grid",
    "save_synthetic",
    "shard_plan",
    "to_uint8",
]
