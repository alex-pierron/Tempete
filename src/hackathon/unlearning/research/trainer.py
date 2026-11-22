"""Research trainer: one loop, several objectives, several parameter-update restrictions.

Objectives (`RunSpec.method`):

gold_soft
    The kit's idea (tempered, view-averaged teacher targets), re-implemented here so that every
    knob can be swept and the loop can report a trajectory.
fallback
    Per-image target `rho_i * onehot(original class) + (1 - rho_i) * q_i`, where `rho_i` falls with
    the fraction of strong views that disagree with the exact prediction (fragile image = low
    confidence) and `q_i` is the "wrong class a model that never saw the image would pick"
    (second class of the original, or the most likely wrong class over the strong views).
npo
    Negative Preference Optimisation (Zhang et al., 2024) on the pseudo-label: a bounded loss
    `(2 / beta) * softplus(beta * log(p_student(y) / p_original(y)))` (no raw gradient ascent),
    optionally switched off per sample once `p_student(y)` has fallen below a floor.
calibrated
    Phase 2. Per-image target `q_i * onehot(original class) + (1 - q_i) * r_i` where `q_i` is read
    from a quantile law fixed by the spec (a fraction of images is made "wrong" on purpose, the
    others keep a moderate confidence) and ranked by the fragility of the image under strong views.
    It aims at the loss mean/accuracy a model that never saw the images would show on them, instead
    of whatever the averaged views happen to give.
scrub
    The kit's label-free SCRUB, as a reference.

Restrictions (`RunSpec.trainable`): all | head | last2 | last4 | layerscale | layerscale_head |
salient (SalUn-style: only the weights with the largest forget-loss gradient magnitude move).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.configs import METHOD_SCRUB, UnlearnConfig
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.methods import scrub_unlearn
from hackathon.unlearning.model import preprocess
from hackathon.unlearning.research.context import EvalResult, ResearchContext, evaluate
from hackathon.unlearning.training import (
    epoch_batches,
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    temper_probabilities,
)

LOGGER = logging.getLogger("hackathon.unlearning.research")

GRADIENT_CLIP_NORM = 1.0
NATIVE_TEMPERATURE = 1.0
MAX_PROBABILITY_ON_ORIGINAL_CLASS = 0.98
"""Upper bound of rho_i in the fallback target (a one-hot target would re-create memorisation)."""
CACHE_SEED_OFFSET = 777
SALIENCY_BATCHES = 4
SALIENCY_QUANTILE_SAMPLE = 2_000_000
"""Number of gradient magnitudes sampled to estimate the saliency threshold (avoids a global sort)."""
NUM_CLASSES = 100
TRAIN_SUBSET_A = "a"
TRAIN_SUBSET_ALL = "all"
METHODS = ("gold_soft", "fallback", "calibrated", "npo", "scrub", "identity")
TRAINABLE_MODES = (
    "all", "head", "last2", "last4", "layerscale", "layerscale_head", "salient"
)  # fmt: skip
FALLBACK_KINDS = ("rank2", "view_mode", "view_soft")
LAYERSCALE_MARKERS = ("ls1.gamma", "ls2.gamma")


class ResearchError(RuntimeError):
    """A research run was misconfigured or diverged."""


@dataclass(frozen=True)
class RunSpec:
    """Every knob of one research run (JSON-friendly, hashable by name)."""

    name: str
    method: str = "gold_soft"
    steps: int = 32
    batch_size: int = 16
    retain_batch_size: int = 16
    lr: float = 1e-5
    warmup_fraction: float = 0.1
    target_temperature: float = 2.0
    target_area: float = 0.35
    target_views: int = 3
    altered_weight: float = 1.0
    retain_weight: float = 1.0
    trainable: str = "all"
    saliency_keep: float = 0.1
    npo_beta: float = 0.1
    npo_p_floor: float = 0.0
    fallback_kind: str = "rank2"
    fallback_floor: float = 0.5
    seed: int = 0
    eval_every: int = 0
    alphas: tuple[float, ...] = ()
    train_on: str = TRAIN_SUBSET_A
    cache_views: bool = False
    init_from: str = ""
    cal_wrong_fraction: float = 0.22
    cal_wrong_low: float = 0.005
    cal_wrong_high: float = 0.12
    cal_right_low: float = 0.55
    cal_right_high: float = 0.97
    cal_wrong_sharpen: float = 2.0
    cal_wrong_top_share: float = 0.5
    anchor_view_fraction: float = 0.0
    mask_classes: tuple[int, ...] = ()
    mask_floor: float = 1e-4
    note: str = ""

    def __post_init__(self) -> None:
        """Reject unknown names early (a typo must not silently run the default)."""
        if self.method not in METHODS:
            raise ResearchError(f"Unknown method '{self.method}', choose from {METHODS}")
        if self.trainable not in TRAINABLE_MODES:
            raise ResearchError(
                f"Unknown trainable '{self.trainable}', choose from {TRAINABLE_MODES}"
            )
        if self.fallback_kind not in FALLBACK_KINDS:
            raise ResearchError(f"Unknown fallback_kind '{self.fallback_kind}'")
        if self.train_on not in (TRAIN_SUBSET_A, TRAIN_SUBSET_ALL):
            raise ResearchError(f"train_on must be '{TRAIN_SUBSET_A}' or '{TRAIN_SUBSET_ALL}'")
        if self.steps < 1 or self.batch_size < 1:
            raise ResearchError("steps and batch_size must be >= 1")


@dataclass
class TrainOutcome:
    """Result of one training run."""

    student: nn.Module
    history: list[dict[str, float]] = field(default_factory=list)
    trajectory: list[dict[str, object]] = field(default_factory=list)
    seconds: float = 0.0


def _set_trainable(student: nn.Module, mode: str) -> int:
    """Freeze every parameter except those of `mode`; return the trainable parameter count."""
    blocks = getattr(student, "blocks")  # noqa: B009 (nn.Module attribute, typed as Tensor | Module)
    first_trained_block = {"last2": len(blocks) - 2, "last4": len(blocks) - 4}.get(mode)
    for name, parameter in student.named_parameters():
        top = name.split(".")[0]
        if mode in ("all", "salient"):
            trainable = True
        elif mode == "head":
            trainable = top == "head"
        elif mode in ("layerscale", "layerscale_head"):
            trainable = any(marker in name for marker in LAYERSCALE_MARKERS) or (
                mode == "layerscale_head" and top == "head"
            )
        else:
            assert first_trained_block is not None
            if top == "blocks":
                trainable = int(name.split(".")[1]) >= first_trained_block
            else:
                trainable = top in ("norm", "fc_norm", "head")
        parameter.requires_grad_(trainable)
    return sum(p.numel() for p in student.parameters() if p.requires_grad)


def _student_logits(student: nn.Module, images: Tensor, device: torch.device) -> Tensor:
    logits: Tensor = student(preprocess(images, device))
    return logits.float()


@torch.no_grad()
def _teacher_logits(teacher: nn.Module, images: Tensor, device: torch.device) -> Tensor:
    logits: Tensor = teacher(preprocess(images, device))
    return logits.float()


def _matching_loss(target_probs: Tensor, logits: Tensor) -> Tensor:
    """KL(target || student) at temperature 1 plus the same at the scoring temperature (T = 2)."""
    native = kl_to_student(target_probs, logits, NATIVE_TEMPERATURE)
    tempered = kl_to_student(
        temper_probabilities(target_probs, SCORING_TEMPERATURE), logits, SCORING_TEMPERATURE
    )
    return 0.5 * (native + tempered)


def _view_probabilities(
    teacher: nn.Module,
    exact: Tensor,
    spec: RunSpec,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Softmax (T = 1) of the teacher on `target_views` strong views, stacked [K, B, C]."""
    return torch.stack(
        [
            torch.softmax(
                _teacher_logits(teacher, strong_view(exact, generator, spec.target_area), device),
                dim=1,
            )
            for _ in range(spec.target_views)
        ]
    )


def remove_classes_from_target(target: Tensor, classes: tuple[int, ...], floor: float) -> Tensor:
    """Give `classes` the probability `floor` each and renormalise the rest (rows still sum to 1).

    A gold that never saw a class never predicts it, on ANY image: this applies to the images of
    that class and to every other image alike. `floor` is the residual probability (1e-4 gives a
    loss of about 9.2 on the pseudo-label of an image of a removed class).
    """
    if not classes:
        return target
    index = list(classes)
    kept = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
    kept[index] = False
    rest = target * kept
    rest = rest / rest.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out = rest * (1.0 - floor * len(index))
    out[:, index] = floor
    return out


def build_target(
    teacher: nn.Module,
    exact: Tensor,
    spec: RunSpec,
    generator: torch.Generator,
    device: torch.device,
    views: Tensor | None = None,
) -> Tensor:
    """Target distribution [B, C] the student must output on `exact` (gold_soft / fallback).

    `views` ([K, B, C] teacher softmax on K strong views) can be passed from a precomputed cache,
    which removes the teacher forward passes from the training step.
    """
    if views is None:
        views = _view_probabilities(teacher, exact, spec, generator, device)
    if spec.method == "gold_soft":
        tempered = torch.stack(
            [temper_probabilities(view, spec.target_temperature) for view in views]
        )
        return remove_classes_from_target(tempered.mean(dim=0), spec.mask_classes, spec.mask_floor)
    exact_logits = _teacher_logits(teacher, exact, device)
    original_class = exact_logits.argmax(dim=1)
    agreement = (views.argmax(dim=2) == original_class.unsqueeze(0)).float().mean(dim=0)
    rho = (spec.fallback_floor + (1.0 - spec.fallback_floor) * agreement).clamp(
        max=MAX_PROBABILITY_ON_ORIGINAL_CLASS
    )
    onehot_original = torch_f.one_hot(original_class, NUM_CLASSES).float()
    if spec.fallback_kind == "rank2":
        masked = exact_logits.masked_fill(onehot_original.bool(), float("-inf"))
        wrong = torch_f.one_hot(masked.argmax(dim=1), NUM_CLASSES).float()
    else:
        mass = views.sum(dim=0).masked_fill(onehot_original.bool(), 0.0)
        if spec.fallback_kind == "view_mode":
            wrong = torch_f.one_hot(mass.argmax(dim=1), NUM_CLASSES).float()
        else:
            wrong = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return rho.unsqueeze(1) * onehot_original + (1.0 - rho).unsqueeze(1) * wrong


@torch.no_grad()
def precompute_view_cache(
    context: ResearchContext, spec: RunSpec, forget: Tensor, batch_size: int
) -> Tensor:
    """Teacher softmax on `target_views` strong views of every image: [N, K, C] (float16, CPU).

    One draw per image, fixed for the whole run (the per-step resampling of the exploration runs
    is traded for speed: the teacher forwards leave the training step).
    """
    generator = torch.Generator().manual_seed(spec.seed + CACHE_SEED_OFFSET)
    chunks: list[Tensor] = []
    for start in range(0, len(forget), batch_size):
        exact = forget[start : start + batch_size]
        views = _view_probabilities(context.original, exact, spec, generator, context.device)
        chunks.append(views.transpose(0, 1).half().cpu())
    return torch.cat(chunks)


def calibrated_targets(views: Tensor, exact_logits: Tensor, spec: RunSpec) -> Tensor:
    """Per-image targets [N, C] of the `calibrated` method (see the module docstring).

    Args:
        views: [N, K, C] teacher softmax on K strong views of every image (the view cache).
        exact_logits: [N, C] teacher logits on the exact images (their argmax is the pseudo-label).
        spec: Provides the quantile law (`cal_*`) and the temperature of the non-label mass.

    Images are ranked by the mean strong-view probability of their pseudo-label (fragile first).
    The `cal_wrong_fraction` most fragile get a label probability spread over
    [`cal_wrong_low`, `cal_wrong_high`] and a sharpened non-label mass (a clear wrong top-1); the
    others get a label probability spread over [`cal_right_low`, `cal_right_high`], by rank.
    """
    count = len(views)
    labels = exact_logits.argmax(dim=1)
    onehot = torch_f.one_hot(labels, NUM_CLASSES).float()
    view_probs = views.float()
    support = view_probs.mean(dim=1).gather(1, labels[:, None]).squeeze(1)
    order = torch.argsort(support)
    rank = torch.empty(count)
    rank[order] = torch.arange(count, dtype=torch.float32) / max(1, count - 1)
    wrong = rank < spec.cal_wrong_fraction
    within_wrong = (rank / max(spec.cal_wrong_fraction, 1e-9)).clamp(0.0, 1.0)
    within_right = (
        (rank - spec.cal_wrong_fraction) / max(1.0 - spec.cal_wrong_fraction, 1e-9)
    ).clamp(0.0, 1.0)
    q_wrong = spec.cal_wrong_low + (spec.cal_wrong_high - spec.cal_wrong_low) * within_wrong
    q_right = spec.cal_right_low + (spec.cal_right_high - spec.cal_right_low) * within_right
    label_mass = torch.where(wrong, q_wrong, q_right)
    tempered_views = torch.stack(
        [temper_probabilities(view_probs[:, k], spec.target_temperature) for k in range(views.shape[1])]
    ).mean(dim=0)
    tempered_exact = temper_probabilities(
        torch.softmax(exact_logits.float(), dim=1), spec.target_temperature
    )
    other = (0.5 * tempered_views + 0.5 * tempered_exact).masked_fill(onehot.bool(), 0.0)
    sharpened = other.clamp_min(1e-12).pow(spec.cal_wrong_sharpen).masked_fill(onehot.bool(), 0.0)
    sharpened = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-12)
    top_wrong = torch_f.one_hot(sharpened.argmax(dim=1), NUM_CLASSES).float()
    # A "wrong" image must really be wrong: a guaranteed share of the non-label mass goes to its
    # most supported wrong class, so its top-1 is that class whatever the views looked like.
    sharpened = (
        (1.0 - spec.cal_wrong_top_share) * sharpened + spec.cal_wrong_top_share * top_wrong
    )
    other = other / other.sum(dim=1, keepdim=True).clamp_min(1e-12)
    other = torch.where(wrong[:, None], sharpened, other)
    return label_mass[:, None] * onehot + (1.0 - label_mass)[:, None] * other


def _with_self_anchor(
    retain_batch: Tensor, forget: Tensor, spec: RunSpec, generator: torch.Generator
) -> Tensor:
    """Replace a fraction of the retain batch by strong views of random forget images.

    The teacher (original) behaves like on test images on such views, and they are in-distribution
    (the retain proxy is not), so keeping the student equal to the teacher there protects the
    behaviour of the hidden control set while the exact images are pushed to the soft targets.
    """
    count = round(len(retain_batch) * spec.anchor_view_fraction)
    if count == 0:
        return retain_batch
    source = torch.randint(len(forget), (count,), generator=generator)
    views = strong_view(forget[source], generator, spec.target_area)
    return torch.cat([retain_batch[: len(retain_batch) - count], views])


def npo_loss(
    student_logits: Tensor, teacher_logits_value: Tensor, labels: Tensor, beta: float, floor: float
) -> Tensor:
    """NPO on pseudo-labels: (2 / beta) * mean softplus(beta * (log p_s(y) - log p_o(y)))."""
    log_p_student = torch_f.log_softmax(student_logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    log_p_teacher = (
        torch_f.log_softmax(teacher_logits_value, dim=1).gather(1, labels[:, None]).squeeze(1)
    )
    per_sample = torch_f.softplus(beta * (log_p_student - log_p_teacher)) * (2.0 / beta)
    if floor > 0:
        per_sample = per_sample * (log_p_student.detach().exp() > floor).float()
    return per_sample.mean()


def _forget_loss(
    spec: RunSpec,
    student: nn.Module,
    teacher: nn.Module,
    batch: Tensor,
    target: Tensor | None,
    labels: Tensor | None,
    device: torch.device,
) -> Tensor:
    logits = _student_logits(student, batch, device)
    if spec.method == "npo":
        assert labels is not None
        return npo_loss(
            logits, _teacher_logits(teacher, batch, device), labels, spec.npo_beta, spec.npo_p_floor
        )
    assert target is not None
    return _matching_loss(target, logits)


def _make_batches(
    count: int, batch_size: int, steps: int, generator: torch.Generator
) -> Iterator[Tensor]:
    return epoch_batches(count, batch_size, steps, generator)


def _saliency_masks(
    student: nn.Module,
    teacher: nn.Module,
    forget: Tensor,
    spec: RunSpec,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """SalUn masks: keep the `saliency_keep` fraction of weights with the largest |gradient|."""
    accumulated = {n: torch.zeros_like(p) for n, p in student.named_parameters()}
    batches = _make_batches(len(forget), spec.batch_size, SALIENCY_BATCHES, generator)
    for indices in batches:
        exact = forget[indices]
        target = build_target(teacher, exact, spec, generator, device)
        loss = _matching_loss(target, _student_logits(student, exact, device))
        student.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        for name, parameter in student.named_parameters():
            if parameter.grad is not None:
                accumulated[name] += parameter.grad.abs()
    student.zero_grad(set_to_none=True)
    flat = torch.cat([a.flatten() for a in accumulated.values()]).cpu()
    sample = flat[torch.randint(len(flat), (SALIENCY_QUANTILE_SAMPLE,), generator=generator)]
    threshold = torch.quantile(sample, 1.0 - spec.saliency_keep).item()
    return {name: (a >= threshold) for name, a in accumulated.items()}


def _apply_update(
    optimizer: torch.optim.Optimizer,
    student: nn.Module,
    learning_rate: float,
    masks: dict[str, Tensor] | None,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    if masks is not None:
        for name, parameter in student.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(masks[name])
    torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], GRADIENT_CLIP_NORM
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _check_finite(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise ResearchError(f"Non-finite loss in {where}: lower the learning rate.")
    return value


def train(
    context: ResearchContext,
    spec: RunSpec,
    retain_pool: Tensor,
    forget: Tensor,
    view_cache: Tensor | None = None,
    target_cache: Tensor | None = None,
) -> TrainOutcome:
    """Run `spec` on `forget` (uint8 NHWC) with `retain_pool` as the KL anchor; return the student.

    `target_cache` ([N, C], same order as `forget`) holds fixed per-image targets (`calibrated`).
    `spec.init_from` warm-starts the student from a saved state dict (the teacher stays the original).
    """
    started = time.time()
    device = context.device
    teacher = context.original
    if spec.method == "identity":
        return TrainOutcome(copy.deepcopy(teacher), seconds=0.0)
    if spec.method == "scrub":
        return _train_scrub(context, spec, retain_pool, forget, started)
    torch.manual_seed(spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    student = copy.deepcopy(teacher)
    if spec.init_from:
        student.load_state_dict(
            torch.load(spec.init_from, map_location=device, weights_only=True)
        )
    student.train()
    trainable_count = _set_trainable(student, spec.trainable)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=spec.lr, weight_decay=0.0
    )
    masks: dict[str, Tensor] | None = None
    if spec.trainable == "salient":
        masks = _saliency_masks(student, teacher, forget, spec, generator, device)
    outcome = TrainOutcome(student)
    forget_total = retain_total = 0.0
    window = 0
    batches = _make_batches(len(forget), spec.batch_size, spec.steps, generator)
    for step, indices in enumerate(batches):
        exact = forget[indices]
        altered = random_augment(exact, generator)
        target = labels = None
        if spec.method == "npo":
            labels = _teacher_logits(teacher, exact, device).argmax(dim=1)
        else:
            cached = None
            if target_cache is not None:
                target = target_cache[indices].float().to(device)
            elif view_cache is not None:
                cached = view_cache[indices].transpose(0, 1).float().to(device)
            if target is None:
                target = build_target(teacher, exact, spec, generator, device, cached)
        weight_total = 1.0 + spec.altered_weight
        exact_loss = _forget_loss(spec, student, teacher, exact, target, labels, device)
        forget_value = _check_finite(float(exact_loss.item()), "forget exact")
        torch.autograd.backward(exact_loss / weight_total)
        altered_loss = _forget_loss(spec, student, teacher, altered, target, labels, device)
        torch.autograd.backward(spec.altered_weight * altered_loss / weight_total)
        retain_index = torch.randint(
            len(retain_pool), (spec.retain_batch_size,), generator=generator
        )
        retain_batch = random_augment(retain_pool[retain_index], generator)
        retain_batch = _with_self_anchor(retain_batch, forget, spec, generator)
        retain_loss = kd_from_teacher(
            _teacher_logits(teacher, retain_batch, device),
            _student_logits(student, retain_batch, device),
            SCORING_TEMPERATURE,
        )
        torch.autograd.backward(spec.retain_weight * retain_loss)
        _apply_update(
            optimizer,
            student,
            learning_rate_at(step, spec.steps, spec.lr, spec.warmup_fraction),
            masks,
        )
        forget_total += forget_value
        retain_total += float(retain_loss.item())
        window += 1
        if (step + 1) % 8 == 0 or step + 1 == spec.steps:
            outcome.history.append(
                {
                    "step": float(step + 1),
                    "forget_loss": forget_total / window,
                    "retain_loss": retain_total / window,
                    "trainable_parameters": float(trainable_count),
                }
            )
            LOGGER.info("%s %s", spec.name, outcome.history[-1])
            forget_total = retain_total = 0.0
            window = 0
        if spec.eval_every and (step + 1) % spec.eval_every == 0 and step + 1 < spec.steps:
            slim = evaluate(context, student, full=False)
            outcome.trajectory.append({"step": step + 1, **_slim_view(slim)})
    outcome.student = student.eval()
    outcome.seconds = time.time() - started
    return outcome


def _slim_view(result: EvalResult) -> dict[str, object]:
    groups = result.metrics["groups"]
    assert isinstance(groups, dict)
    exact = groups["exact"]
    return {
        "forget_kl_t2": result.metrics["forget_kl_t2"],
        "retain_kl_t2": result.metrics["retain_kl_t2"],
        "loss_mean_sub": exact["loss_mean_sub"],
        "loss_std_sub": exact["loss_std_sub"],
        "confidence_sub": exact["confidence_sub"],
        "argmax_agreement": exact["argmax_agreement"],
    }


def _train_scrub(
    context: ResearchContext, spec: RunSpec, retain_pool: Tensor, forget: Tensor, started: float
) -> TrainOutcome:
    """The kit's SCRUB at the same optimiser-step budget (half max-steps, half min-steps)."""
    half = max(1, spec.steps // 2)
    config = UnlearnConfig(
        method=METHOD_SCRUB,
        seed=spec.seed,
        batch_size=spec.batch_size,
        epochs=1,
        lr=spec.lr,
        max_steps_per_epoch=half,
        retain_weight=spec.retain_weight,
        altered_weight=spec.altered_weight,
        scrub_msteps=1,
        amp=False,
        warmup_fraction=spec.warmup_fraction,
    )
    student, history = scrub_unlearn(context.original, forget, config, retain_pool)
    return TrainOutcome(student, history, [], time.time() - started)


def spec_to_dict(spec: RunSpec) -> dict[str, object]:
    """Plain dict of a spec for the results file."""
    return asdict(spec)
