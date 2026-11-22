"""Unlearning methods behind one interface: `unlearn(model, forget_images, config)`.

Four methods, from safest to most ambitious:

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

trust_region
    SCRUB+R without data: the "rewind" point (forget error = error of a validation set we do not
    have) becomes a radius. The student is pulled toward a softened target of the original on
    the EXACT forget images (convex KL, no augmentation, no image outside the forget set) in
    class-stratified batches, and training stops as soon as the mean KL(original || student) at
    T = 1 on those images reaches `kl_radius`, the forget budget. Collateral drift is measured on
    a held-out stratified fold of the forget set (`holdout_fraction`, see cli.run).
    `trust_target` picks the target (`trust_target_logits`). "tempered" (V1, default):
    softmax(z_orig / tau), the same softening for every image, class ranking kept. "views" (V2):
    per image, the mean over `target_views` strong views of softmax(z_orig(view) / tau), computed
    once before training without gradient, so each image keeps its own confidence and its own
    preferred wrong class (the gold is heterogeneous: some images are misclassified, toward a
    precise class); the target gives the direction, the radius the amount. Known bias: the
    bench's gold stand-in is also the original on strong views, so part of a V2 gain on the bench
    is mechanical (views drawn from `seed`, never from the bench's seeds).
    "calibrated" (V3, `calibrated_target_logits`): the score compares two different gold
    objects, per-model loss statistics (mean/std score) and the MIXTURE of the gold
    distributions (prediction score). From the same strong views: the direction at T = 2 is the
    views' mixture hedged with softmax(z_orig / tau), and only the true-class logit is then moved
    (closed form) so that the loss at T = 1 is l* = l_mix + trust_lambda * (mean per-view loss -
    l_mix); the ratios between wrong classes stay exactly those of the direction.
    `radius_fraction` (any target) sets the radius to that share of KL(original || target)
    instead of `kl_radius` (on all the images; with a V4 option, on the probe the stop is tested
    on, see `_trust_phase`); `altered_fraction` replaces that
    share of each batch by a flip or a 70-100 % crop of the same image, trained toward its source's
    target (jitter and JPEG are left unseen, so the bench measures the transfer to families never
    trained on). V4 adds two independent options. `anchor_folds` (K >= 2): at full scale ~88 % of
    the V1 movement is common to every memorised image, trained or not (the audit fold moves
    0.80-0.96x as much as the exact trained images in the 31 V1 runs checked at every step); so
    the images are forgotten fold after fold while the other folds, memorised images of the same
    distribution, are held (never pushed: at the original's outputs; already forgotten: at their
    state when the phase began; `anchor_views` of them as strong views): a retain proxy drawn
    from the forget set alone, whose drift each phase measures (`pending_kl`).
    `radius_dual_lr` > 0: the radius becomes an active constraint (dual ascent) instead of a
    stop, so training does not freeze mid-warmup at the first crossing.
    scrub and gold_soft take `retain_source="forget_views"`: strong views of the forget images
    replace the external proxy images, which a submitted method may not use.

band_ascent  (V5, rapport section "V5 : oubli en bande")
    A bounded gradient ASCENT instead of a descent toward a target. With L_i the cross-entropy of
    the student on the pseudo-label and l_i the original's mean loss on target_views strong
    views, the share band_release of the images with the highest l_i is pushed up to
    t_i = rho * l_i and kept below h_i = band_tolerance * t_i + band_floor; the others stay below
    band_floor. Loss: mean of [pushed * softplus(beta (t - L)) + softplus(beta (L - h))] / beta,
    so the force on an image vanishes once it sits in its band and collapse costs loss. The dose
    rho climbs band_levels levels (fresh AdamW and schedule per level). With cv_folds = K >= 2, a
    sub-run on K - 1 folds keeps the last level whose drift on the untouched fold (scaled by
    (K / (K - 1)) ** cv_scale_exponent) stays within cv_max_drift; the final run climbs those
    levels on every image. Hard stop (back to the last level done) when the drift monitor exceeds
    twice the budget or more than half a fixed probe sits above its ceiling. Nothing but the exact
    forget images enters the loss.

Training reads the forget images as uint8 [N, 224, 224, 3]; nothing is written to disk here.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.common.errors import ConfigError
from hackathon.unlearning.alterations import (
    horizontal_flip,
    random_augment,
    random_crop_resize,
    strong_view,
)
from hackathon.unlearning.configs import (
    METHOD_BAND_ASCENT,
    METHOD_GOLD_SOFT,
    METHOD_IDENTITY,
    METHOD_SCRUB,
    METHOD_TRUST_REGION,
    RETAIN_SOURCE_FORGET_VIEWS,
    TRUST_TARGET_CALIBRATED,
    TRUST_TARGET_TEMPERED,
    UnlearnConfig,
)
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.data import class_grouped_order, load_proxy_images, select_subset
from hackathon.unlearning.model import model_device, predict, select_device
from hackathon.unlearning.proxy_eval import tempered_kl
from hackathon.unlearning.training import (
    LOG_PROBABILITY_FLOOR,
    epoch_batches,
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    prepare_student,
    prepare_teacher,
    steps_per_epoch,
    stratified_batches,
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
RADIUS_CHECK_IMAGES = 256
"""trust_region: the stop is tested every `radius_check_every` steps on a fixed random sample of
RADIUS_CHECK_IMAGES training images (~25 % extra compute at batch 32 every 10 steps; ~7 s per run
every step on the RTX 5080), then measured exactly on all of them at the end. The KL of the
current training batch cannot serve as the estimate: in a reduced run (M2, 32 images) the images
an update did not train on stayed at KL ~1e-6 while the trained ones jumped. At full scale the
forgetting is no longer local (movement ratio 1.2-1.8, experiments/unlearning/
trust-region-campaign.md), but the batch is still not a sample of the trained set.
"""
VIEW_TARGET_SEED_OFFSET = 1_000_003
"""trust_region "views" and "calibrated": the views are drawn from seed + this offset, a stream
apart from the batch order (seeded by `seed`) and from the bench's gold views (seeds 31000-31503)."""
TARGET_LOSS_FLOOR = 1e-6
"""calibrated: floor of l* (the closed form needs l* > 0; below 1e-6 nat nothing is measurable)."""
ALTERED_SEED_OFFSET = 2_000_003
"""trust_region `altered_fraction`: its draws come from seed + this offset, so the batch order is
the same with and without soft copies."""
SOFT_CROP_AREA = (0.70, 1.00)
"""Area of the training crops: milder than the bench's 60-90 % crop copies (not tuned on them)."""
ANCHOR_SEED_OFFSET = 3_000_003
"""trust_region `anchor_folds` (V4): the folds come from seed + this offset, the anchor batches of
phase k from seed + this offset + k: the batch order and the soft copies keep their own streams."""
PHASE_SEED_STRIDE = 1000
"""trust_region `anchor_folds`: the batch order of phase k comes from seed + ANCHOR_SEED_OFFSET +
k * this stride (apart from the folds, seed + offset, and the anchor draws, + k)."""
ANCHOR_VIEW_MIN_AREA = 0.25
"""Crop area [0.25, 0.35] of the strong views a forget-only retain proxy holds (trust_region
`anchor_views`, retain_source "forget_views"): apart from every area the bench draws (gold crops
from 0.40, audit view 0.50); the bench's colour jitter is close, so for these variants the patch
masking family (`score_min_family.mask`) and `audit_kl` decide."""
DUAL_KL_FLOOR = 1e-12
"""radius_dual_lr: floor of the probe KL inside log(KL / radius) (KL = 0 before any update)."""
CV_SEED_OFFSET = 4_000_003
"""band_ascent (V5): the cross-validation folds come from seed + this offset and the sub-run's
batch order from seed + this offset + 1, apart from every other stream (views, copies, V4)."""
BAND_HARD_STOP_DRIFT = 2.0
"""band_ascent: hard stop once a drift monitor exceeds this multiple of cv_max_drift."""
BAND_HARD_STOP_ABOVE = 0.5
"""band_ascent: hard stop once more than this share of the probe sits above its ceiling h_i."""

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
    return torch.optim.AdamW(
        trainable, lr=config.lr, weight_decay=config.weight_decay, fused=trainable[0].is_cuda
    )


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


def _resolve_retain(
    config: UnlearnConfig, retain_images: Tensor | None, forget_images: Tensor
) -> Tensor:
    """Retain-proxy images: the forget images themselves when retain_source is "forget_views"
    (`_retain_batch` then shows them as strong views), else `retain_images` or the proxy folders."""
    if config.retain_source == RETAIN_SOURCE_FORGET_VIEWS:
        return forget_images
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
    """Random augmented retain-proxy batch (augmentation widens the few local images); with
    retain_source "forget_views", strong views of forget images, where the original is not
    memorised and the gold behaves like it: holding them keeps the function, not the spike."""
    indices = torch.randint(len(retain), (config.batch_size,), generator=generator)
    if config.retain_source == RETAIN_SOURCE_FORGET_VIEWS:
        return strong_view(retain[indices], generator, ANCHOR_VIEW_MIN_AREA)
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
    retain = _resolve_retain(config, retain_images, forget_images).to(device)
    forget_images = forget_images.to(device)  # augmentations then run on the GPU
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


def _forget_kl(student: nn.Module, images: Tensor, original_probs: Tensor, batch: int) -> float:
    """Mean KL(original || student) at T = 1 over `images`; leaves the student in train mode."""
    logits = predict(student, images, batch).to(original_probs.device)
    student.train()
    return float(kl_to_student(original_probs, logits, NATIVE_TEMPERATURE))


def view_logits(
    model: nn.Module, images: Tensor, config: UnlearnConfig, device: torch.device
) -> Iterator[Tensor]:
    """The original's logits on target_views strong views of `images`, from seed + offset."""
    generator = torch.Generator().manual_seed(config.seed + VIEW_TARGET_SEED_OFFSET)
    # shortcut: each view holds a few float32 copies of the set at once (1.7 GB each for 2800
    # images, as in bench.run_bench); draw per-image seeds and chunk if a small GPU runs out.
    for _ in range(config.target_views):
        view = strong_view(images, generator, config.target_view_min_area)
        yield predict(model, view, config.batch_size).to(device)


def calibrated_target_logits(
    original_logits: Tensor, view_logits: Iterable[Tensor], config: UnlearnConfig
) -> Tensor:
    """Logits of the "calibrated" target (V3) from the original's logits on the exact images
    and on their views; float64 on the CPU inside (MPS has no float64, the tensors are N x 100),
    returned on the device and in the dtype of `original_logits`.

    With y = argmax of the original, T = 2 the scoring temperature, b = trust_hedge and
    tau = target_temperature: p_hat = mean_v softmax(z_v / T); l_bar = mean_v CE(z_v, y);
    p_dir = (1 - b) p_hat + b softmax(z_orig / tau); z = T log p_dir; l_mix = CE(z, y);
    l* = l_mix + trust_lambda (l_bar - l_mix). Since CE(z, y) = softplus(A - z_y) with
    A = logsumexp of the other logits, z_y <- A - log(expm1(l*)) gives CE = l* exactly and keeps
    every other logit, hence the ratios between wrong classes of p_dir at T = 2.
    """
    original = original_logits.cpu().double()
    labels = original.argmax(dim=1)
    views = torch.stack([view.cpu().double() for view in view_logits])  # [V, N, classes]
    mixture = torch.softmax(views / SCORING_TEMPERATURE, dim=2).mean(dim=0)
    view_loss = torch.stack(
        [torch_f.cross_entropy(view, labels, reduction="none") for view in views]
    ).mean(dim=0)
    hedge = torch.softmax(original / config.target_temperature, dim=1)
    direction = (1.0 - config.trust_hedge) * mixture + config.trust_hedge * hedge
    logits = SCORING_TEMPERATURE * direction.clamp_min(LOG_PROBABILITY_FLOOR).log()
    mixture_loss = torch_f.cross_entropy(logits, labels, reduction="none")
    target_loss = mixture_loss + config.trust_lambda * (view_loss - mixture_loss)
    target_loss = target_loss.clamp_min(TARGET_LOSS_FLOOR)
    others = logits.scatter(1, labels[:, None], -math.inf).logsumexp(dim=1)
    true_logit = others - torch.expm1(target_loss).log()
    logits = logits.scatter(1, labels[:, None], true_logit[:, None])
    return logits.to(original_logits.device, original_logits.dtype)


def trust_target_logits(
    model: nn.Module, images: Tensor, original_logits: Tensor, config: UnlearnConfig
) -> Tensor:
    """Logits of the trust_region target [N, classes], on the device of original_logits.

    softmax(result) is the target; log_softmax(result) its log-probabilities, so the result feeds
    `kl_to_student` directly. "tempered" (V1): z_orig / tau. "views" (V2): the log of the mean,
    over target_views strong views, of softmax(z_orig(view) / tau) (already normalised).
    "calibrated" (V3): `calibrated_target_logits` on the same views. The views are drawn on the
    whole set at once from seed + VIEW_TARGET_SEED_OFFSET, so the target of an image does not
    depend on the batch size, and the global RNG is left untouched.
    """
    if config.trust_target == TRUST_TARGET_TEMPERED:
        return original_logits / config.target_temperature
    views = view_logits(model, images, config, original_logits.device)
    if config.trust_target == TRUST_TARGET_CALIBRATED:
        return calibrated_target_logits(original_logits, views, config)
    total = torch.zeros_like(original_logits)
    for logits in views:
        total += torch.softmax(logits / config.target_temperature, dim=1)
    return (total / config.target_views).clamp_min(LOG_PROBABILITY_FLOOR).log()


def _soft_copies(images: Tensor, fraction: float, generator: torch.Generator) -> Tensor:
    """Copy of the batch where round(fraction * n) images are replaced, half by their flip,
    half by a SOFT_CROP_AREA crop of themselves (positions drawn from `generator`)."""
    chosen = torch.randperm(len(images), generator=generator)[: round(fraction * len(images))]
    flips, crops = chosen[: len(chosen) // 2], chosen[len(chosen) // 2 :]
    out = images.clone()
    out[flips] = horizontal_flip(images[flips], generator)
    out[crops] = random_crop_resize(images[crops], generator, *SOFT_CROP_AREA)
    return out


def anchor_fold_indices(
    labels: Tensor, count: int, seed: int, offset: int = ANCHOR_SEED_OFFSET
) -> list[Tensor]:
    """`count` disjoint folds covering all indices, each spanning the labels (every count-th
    index of the class-grouped order drawn from seed + offset)."""
    order = class_grouped_order(labels, torch.Generator().manual_seed(seed + offset))
    return [order[fold::count] for fold in range(count)]


def _held_outputs(
    images: Tensor,
    pending: Tensor,
    holders: tuple[nn.Module, nn.Module | None],
    config: UnlearnConfig,
) -> Tensor:
    """No-grad logits an anchor batch is held at: the original's for images never pushed
    (`pending`), the phase-start copy's for images already forgotten. Same autocast as the
    student's own forward, so the anchor is exactly zero before the first update."""
    device = images.device
    pending = pending.to(device)
    logits: Tensor | None = None
    with torch.no_grad():
        for mask, holder in ((pending, holders[0]), (~pending, holders[1])):
            if holder is None or not bool(mask.any()):
                continue
            part = student_logits(holder, images[mask], device, config)
            if logits is None:
                logits = part.new_empty((len(images), part.shape[1]))
            logits[mask] = part
    if logits is None:
        raise UnlearnError("Empty anchor batch.")
    return logits


def _mean_divergence(reference: nn.Module, student: nn.Module, images: Tensor, batch: int) -> float:
    """Mean KL(reference || student) at the scoring temperature (CPU float64, like the bench)."""
    divergence = tempered_kl(predict(reference, images, batch), predict(student, images, batch))
    student.train()
    return float(divergence.mean())


def _trust_phase(
    student: nn.Module,
    forget_images: Tensor,
    push: Tensor,
    anchor: tuple[Tensor, Tensor, nn.Module] | None,
    targets: tuple[Tensor, Tensor, Tensor, Tensor],
    config: UnlearnConfig,
    generators: tuple[torch.Generator, torch.Generator],
    phase: int,
) -> tuple[History, float]:
    """One trust_region phase: pull images[push] toward their target until the KL of a fixed
    probe of them reaches the radius; returns the rows and that radius.

    radius_dual_lr > 0: the phase does not stop there; it runs its whole schedule with the radius
    as an active constraint, a multiplier mu on KL(original || student) at T = 1 of the trained
    batch, updated at every check by mu <- max(0, mu + radius_dual_lr * log(probe KL / radius))
    (symmetric: an overshoot is undone as fast as it was paid for).
    `anchor` = (pending, forgotten, original): each step also holds an anchor batch drawn from
    both sets (a share `anchor_views` as strong views), the images never pushed at the original's
    outputs, those already forgotten at a frozen copy of the student taken when the phase began.

    `targets` is (target probabilities, target logits, original probabilities, pseudo-labels)
    for all images; `generators` is (batch order, soft copies). The radius is kl_radius, or
    radius_fraction * KL(original || target) on `push`; with a V4 option (anchor or dual), on the
    probe itself, the images the stop is tested on (on all of `push`, a probe whose target sits
    under radius_fraction x the mean of the set can never reach the radius). V1-V3 keep the whole
    set, so every measured run (ref4-v3...) replays exactly. Without anchor nor dual, the RNG
    draws are those of the single-phase V1-V3 loop.
    """
    target, target_logits, original_probs, labels = targets
    generator, altered = generators
    device = model_device(student)
    optimizer = _build_optimizer(student, config)
    epoch_steps = steps_per_epoch(len(push), config)
    total_steps = epoch_steps * config.epochs
    probe_order = torch.Generator().manual_seed(config.seed)
    probe = push[torch.randperm(len(push), generator=probe_order)[:RADIUS_CHECK_IMAGES]]
    radius = config.kl_radius
    if config.radius_fraction is not None:
        scope = probe if anchor is not None or config.radius_dual_lr > 0 else push
        scope_kl = kl_to_student(original_probs[scope], target_logits[scope], 1.0)
        radius = config.radius_fraction * float(scope_kl)
    if not radius > 0:
        raise ConfigError(f"trust_region phase {phase}: the target is the original on the probe.")
    LOGGER.info("trust_region phase %d: %d images, radius %.4f", phase, len(push), radius)
    held = forgotten = pending = None
    holders: tuple[nn.Module, nn.Module | None] | None = None
    if anchor is not None:
        pending, forgotten, original = anchor
        held = torch.cat([pending, forgotten])
        is_pending = torch.arange(len(held)) < len(pending)
        holders = (original, prepare_teacher(student, device) if len(forgotten) else None)
    anchor_draws = torch.Generator().manual_seed(config.seed + ANCHOR_SEED_OFFSET + phase)
    dual = config.radius_dual_lr > 0
    multiplier = 0.0
    history: History = []
    step = 0
    estimate = 0.0
    for epoch in range(1, config.epochs + 1):
        loss_total = anchor_total = penalty_total = 0.0
        batches = 0
        for local in stratified_batches(labels[push], config.batch_size, epoch_steps, generator):
            indices = push[local]
            images = forget_images[indices]
            if config.altered_fraction > 0:  # no extra draw at 0: V1/V2 stay bit-identical
                images = _soft_copies(images, config.altered_fraction, altered)
            logits = student_logits(student, images, device, config)
            loss = _matching_loss(target[indices], logits, config)
            if multiplier > 0:  # the constraint is active: pull back toward the original
                penalty = kl_to_student(original_probs[indices], logits, NATIVE_TEMPERATURE)
                loss_total += float(loss.detach())
                penalty_total += _backward_term(
                    loss + multiplier * penalty, f"trust_region dual e{epoch}"
                ) - float(loss.detach())
            else:
                loss_total += _backward_term(loss, f"trust_region e{epoch}")
            if held is not None and holders is not None:
                pick = torch.randint(len(held), (len(indices),), generator=anchor_draws)
                batch = _anchor_views(forget_images[held[pick]], config.anchor_views, anchor_draws)
                anchor_loss = kd_from_teacher(
                    _held_outputs(batch, is_pending[pick], holders, config),
                    student_logits(student, batch, device, config),
                    config.kd_temperature,
                )
                anchor_total += _backward_term(
                    config.anchor_weight * anchor_loss, f"trust_region anchor p{phase} e{epoch}"
                )
            _apply_update(
                optimizer, student,
                learning_rate_at(step, total_steps, config.lr, config.warmup_fraction),
            )  # fmt: skip
            batches += 1
            step += 1
            if step % config.radius_check_every == 0 or step == total_steps:
                estimate = _forget_kl(
                    student, forget_images[probe], original_probs[probe], config.batch_size
                )
                LOGGER.info("trust_region step %d/%d forget KL ~%.4f", step, total_steps, estimate)
                if dual:
                    ratio = max(estimate, DUAL_KL_FLOOR) / radius
                    multiplier = max(0.0, multiplier + config.radius_dual_lr * math.log(ratio))
                elif estimate >= radius:
                    break
        row = {"epoch": float(epoch), "forget_loss": loss_total / batches,
               "forget_kl_estimate": estimate, "steps": float(step)}  # fmt: skip
        if anchor is not None:
            row |= {"phase": float(phase), "anchor_loss": anchor_total / batches, "radius": radius}
        if dual:
            row |= {"dual_multiplier": multiplier, "dual_penalty": penalty_total / batches}
        history.append(row)
        LOGGER.info("trust_region phase %d epoch %d/%d %s", phase, epoch, config.epochs, row)
        if not dual and estimate >= radius:
            break
    if holders is not None and pending is not None and forgotten is not None:
        if len(pending):  # never pushed, against the original: the run's own audit fold
            history[-1]["pending_kl"] = _mean_divergence(
                holders[0], student, forget_images[pending], config.batch_size
            )
        if holders[1] is not None:  # already forgotten, against their state at phase start
            history[-1]["forgotten_kl"] = _mean_divergence(
                holders[1], student, forget_images[forgotten], config.batch_size
            )
    return history, radius


def _anchor_views(images: Tensor, share: float, generator: torch.Generator) -> Tensor:
    """Copy of the anchor batch where round(share * n) images become a strong view of
    themselves (crop area ANCHOR_VIEW_MIN_AREA + 0-10 %, then colour), drawn from `generator`."""
    if share <= 0:
        return images
    chosen = torch.randperm(len(images), generator=generator)[: round(share * len(images))]
    out = images.clone()
    out[chosen] = strong_view(images[chosen], generator, ANCHOR_VIEW_MIN_AREA)
    return out


def trust_region_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """Pull forget outputs toward the trust target; stop once their KL reaches the radius.

    The radius is kl_radius, or radius_fraction * KL(original || target) (V1-V3: on all the
    images, as measured; V4 options: on the probe the stop is tested on). With anchor_folds =
    K >= 2 (V4), the images are forgotten fold after fold, each phase with its own probe and
    radius, while the other folds are held (`_trust_phase`); the
    last history row then carries, per phase k, radius_pk, reached_pk, pending_kl_pk (folds never
    pushed, against the original: the run's own audit) and forgotten_kl_pk.
    """
    del retain_images  # nothing beyond the forget set: the radius is the guard
    device = select_device(config.device) if config.device else model_device(model)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    forget_images = forget_images.to(device)
    original_logits = predict(model, forget_images, config.batch_size).to(device)
    original_probs = torch.softmax(original_logits, dim=1)
    target_logits = trust_target_logits(model, forget_images, original_logits, config)
    target = torch.softmax(target_logits, dim=1)
    reachable = float(kl_to_student(original_probs, target_logits, 1.0))
    radius = config.kl_radius
    if config.radius_fraction is not None:
        radius = config.radius_fraction * reachable
    if config.anchor_folds > len(forget_images):
        raise ConfigError(
            f"anchor_folds ({config.anchor_folds}) exceeds the {len(forget_images)} forget images."
        )
    if reachable <= radius:
        hint = f"raise target_temperature (now {config.target_temperature})"
        if config.trust_target != TRUST_TARGET_TEMPERED:
            hint += f" or lower target_view_min_area (now {config.target_view_min_area})"
        raise ConfigError(
            f"trust_region: the {config.trust_target} target sits at KL {reachable:.3f} <= "
            f"radius {radius:.4g}; {hint}."
        )
    labels = original_logits.argmax(dim=1).cpu()
    agreement = float((target.argmax(dim=1).cpu() == labels).double().mean())
    target_loss = torch_f.cross_entropy(target_logits.cpu().double(), labels, reduction="none")
    LOGGER.info(
        "trust_region %s target KL %.4f (argmax agreement %.3f, loss %.4f +- %.4f), "
        "radius on the whole set %.4f",
        config.trust_target, reachable, agreement, target_loss.mean(),
        target_loss.std(unbiased=False), radius,
    )  # fmt: skip
    student = prepare_student(model, device, config)
    everything = torch.arange(len(forget_images))
    folds = [everything]
    original: nn.Module | None = None
    if config.anchor_folds:
        folds = anchor_fold_indices(labels, config.anchor_folds, config.seed)
        original = prepare_teacher(model, device)
    targets = (target, target_logits, original_probs, labels)
    altered = torch.Generator().manual_seed(config.seed + ALTERED_SEED_OFFSET)
    history: History = []
    phase_rows: dict[str, float] = {}
    radii: list[float] = []
    for phase, push in enumerate(folds, start=1):
        anchor = None
        generators = (generator, altered)
        if original is not None:  # one stream per phase: a phase's length never shifts the next
            none = torch.empty(0, dtype=torch.long)
            anchor = (torch.cat([none, *folds[phase:]]), torch.cat([none, *folds[: phase - 1]]),
                      original)  # fmt: skip
            order_seed = config.seed + ANCHOR_SEED_OFFSET + PHASE_SEED_STRIDE * phase
            generators = (
                torch.Generator().manual_seed(order_seed),
                torch.Generator().manual_seed(config.seed + ALTERED_SEED_OFFSET + phase),
            )
        rows, phase_radius = _trust_phase(
            student, forget_images, push, anchor, targets, config, generators, phase
        )
        radii.append(phase_radius)
        done = history[-1]["steps"] if history else 0.0
        for row in rows:  # steps count over the whole run, phases included
            row |= {"steps": row["steps"] + done, "target_kl": reachable,
                    "target_agreement": agreement}  # fmt: skip
        if original is not None:
            last = rows[-1]
            phase_rows[f"radius_p{phase}"] = phase_radius
            phase_rows[f"reached_p{phase}"] = float(last["forget_kl_estimate"] >= phase_radius)
            for key in ("pending_kl", "forgotten_kl"):
                if key in last:
                    phase_rows[f"{key}_p{phase}"] = last[key]
        history += rows
    radius = sum(radii) / len(radii)  # the folds are equal to within one image
    forget_kl = _forget_kl(student, forget_images, original_probs, config.batch_size)
    history[-1]["forget_kl"] = forget_kl  # exact, after the last update
    history[-1]["radius_reached"] = float(forget_kl >= radius)
    history[-1]["kl_radius_effective"] = radius
    if original is not None or config.radius_dual_lr > 0:
        history[-1]["radius_ratio"] = forget_kl / radius
    if original is not None:
        history[-1] |= phase_rows
        history[-1]["radius_reached"] = min(
            phase_rows[f"reached_p{k}"] for k in range(1, len(folds) + 1)
        )
        # the self-audit of the last phase that still had never-pushed folds
        history[-1]["pending_kl"] = phase_rows[f"pending_kl_p{len(folds) - 1}"]
    # CE of the target on the pseudo-label at T = 1 (calibrated: exactly l*).
    history[-1]["target_loss_mean"] = float(target_loss.mean())
    history[-1]["target_loss_std"] = float(target_loss.std(unbiased=False))
    if forget_kl < radius:
        LOGGER.warning("trust_region: radius %.4f not reached (KL %.4f)", radius, forget_kl)
    return student.eval(), history


def scrub_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """SCRUB without labels: capped max-steps on forget, min-steps on retain proxy."""
    device = select_device(config.device) if config.device else model_device(model)
    retain = _resolve_retain(config, retain_images, forget_images).to(device)
    forget_images = forget_images.to(device)  # augmentations then run on the GPU
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


def band_targets(
    view_loss: Tensor, config: UnlearnConfig, rho: float
) -> tuple[Tensor, Tensor, Tensor]:
    """(pushed, target t, ceiling h) per image (rapport eq. v5targets): the round(band_release *
    n) images of highest view loss l are pushed to t = rho * l, below h = band_tolerance * t +
    band_floor; the others have t = 0 and h = band_floor (+inf when band_floor is None)."""
    count = max(1, round(config.band_release * len(view_loss)))
    pushed = torch.zeros(len(view_loss), dtype=torch.bool)
    pushed[torch.topk(view_loss, count).indices] = True
    target = torch.where(pushed, rho * view_loss, torch.zeros_like(view_loss))
    floor = 0.0 if config.band_floor is None else config.band_floor
    ceiling = config.band_tolerance * target + floor
    if config.band_floor is None:
        ceiling = torch.where(pushed, ceiling, torch.full_like(ceiling, math.inf))
    return pushed, target, ceiling


def band_loss(
    losses: Tensor, pushed: Tensor, target: Tensor, ceiling: Tensor, sharpness: float
) -> Tensor:
    """Batch mean of psi_i = [pushed_i softplus(beta (t_i - L_i)) + softplus(beta (L_i - h_i))]
    / beta (rapport eq. v5band): ascent up to t_i, nothing inside the band, pull back above h_i."""
    lift = torch_f.softplus(sharpness * (target - losses)) * pushed
    pull = torch_f.softplus(sharpness * (losses - ceiling))
    return (lift + pull).mean() / sharpness


def _band_levels(
    student: nn.Module,
    forget_images: Tensor,
    run: Tensor,
    witness: tuple[Tensor, float] | None,
    reference: tuple[Tensor, Tensor, Tensor],
    config: UnlearnConfig,
    generator: torch.Generator,
    levels: int,
) -> tuple[History, int, bool]:
    """Climb the dose levels rho_j = j / band_levels, j = 1..levels, on images[run]; returns
    (rows, levels completed, hard stop).

    `reference` = (original logits, pseudo-labels, view losses) of every image (CPU). With
    `witness` = (untouched fold, scale), a cross-validation sub-run: a level counts only if
    scale * drift of the fold <= cv_max_drift and its agreement >= cv_min_agreement, and the climb
    ends at the first that does not. Without it, the final run: a hard stop restores the state of
    the last level completed. The prequential drift is KL(original || student) at T = 2 of each
    batch BEFORE its update; on the first level no image of the batch has been pushed yet.
    """
    original_logits, labels, view_loss = reference
    device = model_device(student)
    steps = steps_per_epoch(len(run), config) * config.epochs
    local_probe = torch.randperm(len(run), generator=torch.Generator().manual_seed(config.seed))
    probe = local_probe[:RADIUS_CHECK_IMAGES]
    phase = 0.0 if witness is not None else 1.0
    budget = BAND_HARD_STOP_DRIFT * config.cv_max_drift

    def snapshot() -> dict[str, Tensor]:  # on the CPU: a ViT-B copy is 344 MB
        return {k: v.detach().to("cpu", copy=True) for k, v in student.state_dict().items()}

    kept = snapshot() if witness is None else {}
    history: History = []
    done = elapsed = 0
    for level in range(1, levels + 1):
        rho = level / config.band_levels
        pushed, target, ceiling = band_targets(view_loss[run], config, rho)
        optimizer = _build_optimizer(student, config)
        loss_total = preq_total = window = above = 0.0
        window_count = step = 0
        stopped = False
        for step, local in enumerate(stratified_batches(labels[run], config.batch_size, steps,
                                                        generator)):  # fmt: skip
            indices = run[local]
            logits = student_logits(student, forget_images[indices], device, config)
            preq = float(tempered_kl(original_logits[indices], logits.detach().cpu()).mean())
            losses = torch_f.cross_entropy(logits, labels[indices].to(device), reduction="none")
            loss = band_loss(losses, pushed[local].to(device), target[local].to(device),
                             ceiling[local].to(device), config.band_sharpness)  # fmt: skip
            loss_total += _backward_term(loss, f"band_ascent level {level}")
            _apply_update(
                optimizer, student,
                learning_rate_at(step, steps, config.lr, config.warmup_fraction),
            )  # fmt: skip
            preq_total += preq
            window += preq
            window_count += 1
            if (step + 1) % config.cv_check_every and step + 1 < steps:
                continue
            probe_losses = torch_f.cross_entropy(
                predict(student, forget_images[run[probe]], config.batch_size).cpu(),
                labels[run[probe]], reduction="none",
            )  # fmt: skip
            above = float((probe_losses > ceiling[probe]).double().mean())
            drift = window / window_count if level == 1 and witness is None else 0.0
            if witness is not None:
                drift = witness[1] * _band_drift(student, forget_images, witness[0], reference,
                                                 config)[0]  # fmt: skip
            student.train()
            window, window_count = 0.0, 0
            LOGGER.info("band_ascent phase %d level %d step %d/%d drift %.4f above %.3f",
                        phase, level, step + 1, steps, drift, above)  # fmt: skip
            if drift > budget or above > BAND_HARD_STOP_ABOVE:
                stopped = True
                break
        elapsed += step + 1
        row = {"phase": phase, "level": float(level), "rho": rho, "steps": float(elapsed),
               "band_loss": loss_total / (step + 1), "preq_kl": preq_total / (step + 1),
               "above_ceiling": above, "hard_stop": float(stopped)}  # fmt: skip
        accepted = not stopped
        if witness is not None:
            drift, agreement = _band_drift(student, forget_images, witness[0], reference, config)
            student.train()
            row |= {"witness_kl": drift, "witness_agreement": agreement,
                    "witness_kl_scaled": witness[1] * drift}  # fmt: skip
            accepted = accepted and witness[1] * drift <= config.cv_max_drift
            accepted = accepted and agreement >= config.cv_min_agreement
        history.append(row)
        LOGGER.info("band_ascent phase %d level %d %s", phase, level, row)
        if not accepted:
            if witness is None:  # final run: back to the last level completed
                student.load_state_dict(kept)
            return history, done, stopped
        done = level
        if witness is None and level < levels:
            kept = snapshot()
    return history, done, False


def _band_drift(
    student: nn.Module,
    forget_images: Tensor,
    fold: Tensor,
    reference: tuple[Tensor, Tensor, Tensor],
    config: UnlearnConfig,
) -> tuple[float, float]:
    """(mean KL(original || student) at T = 2, argmax agreement) on images[fold]; eval mode."""
    logits = predict(student, forget_images[fold], config.batch_size).cpu()
    drift = float(tempered_kl(reference[0][fold], logits).mean())
    return drift, float((logits.argmax(dim=1) == reference[1][fold]).double().mean())


def band_ascent_unlearn(
    model: nn.Module, forget_images: Tensor, config: UnlearnConfig, retain_images: Tensor | None
) -> tuple[nn.Module, History]:
    """V5: bounded per-image ascent in a band, dose chosen on an untouched fold (module doc).

    The last history row carries cv_stop_level (levels kept by the cross-validation, the
    band_levels without it), cv_monitor_kl and cv_agreement (the untouched fold at that level),
    cv_scale, band_levels_done and hard_stop of the final run, forget_kl (exact KL(original ||
    student) at T = 1), pushed_count and the view and target losses of the pushed images.
    """
    del retain_images  # nothing beyond the forget set: the bands and the untouched fold guard
    device = select_device(config.device) if config.device else model_device(model)
    torch.manual_seed(config.seed)
    forget_images = forget_images.to(device)
    original_logits = predict(model, forget_images, config.batch_size).cpu()
    labels = original_logits.argmax(dim=1)
    view_loss = torch.stack([
        torch_f.cross_entropy(view.cpu().float(), labels, reduction="none")
        for view in view_logits(model, forget_images, config, device)
    ]).mean(dim=0)  # fmt: skip
    reference = (original_logits, labels, view_loss)
    history: History = []
    summary: dict[str, float] = {}
    levels = config.band_levels
    if config.cv_folds:
        if config.cv_folds > len(forget_images):
            raise ConfigError(
                f"cv_folds ({config.cv_folds}) exceeds the {len(forget_images)} forget images."
            )
        folds = anchor_fold_indices(labels, config.cv_folds, config.seed, CV_SEED_OFFSET)
        scale = (config.cv_folds / (config.cv_folds - 1)) ** config.cv_scale_exponent
        rows, levels, stopped = _band_levels(
            prepare_student(model, device, config), forget_images, torch.cat(folds[1:]),
            (folds[0], scale), reference, config,
            torch.Generator().manual_seed(config.seed + CV_SEED_OFFSET + 1), config.band_levels,
        )  # fmt: skip
        history += rows
        chosen = rows[levels - 1] if levels else {}
        summary |= {"cv_stop_level": float(levels), "cv_stop_rho": levels / config.band_levels,
                    "cv_scale": scale, "cv_hard_stop": float(stopped),
                    "cv_monitor_kl": chosen.get("witness_kl", 0.0),
                    "cv_agreement": chosen.get("witness_agreement", 1.0)}  # fmt: skip
        if not levels:
            LOGGER.warning("band_ascent: no level passed the cross-validation; model unchanged.")
    student = prepare_student(model, device, config)
    done, stopped = 0, False
    if levels:
        rows, done, stopped = _band_levels(
            student, forget_images, torch.arange(len(forget_images)), None, reference, config,
            torch.Generator().manual_seed(config.seed), levels,
        )  # fmt: skip
        history += rows
    if not history or history[-1]["phase"] != 1.0:
        history.append({"phase": 1.0, "level": 0.0, "steps": 0.0})
    pushed, target, _ = band_targets(view_loss, config, done / config.band_levels)
    original_probs = torch.softmax(original_logits, dim=1).to(device)
    summary |= {"band_levels_done": float(done), "hard_stop": float(stopped),
                "forget_kl": _forget_kl(student, forget_images, original_probs, config.batch_size),
                "pushed_count": float(pushed.sum()),
                "pushed_view_loss": float(view_loss[pushed].mean()),
                "pushed_target_loss": float(target[pushed].mean())}  # fmt: skip
    history[-1] |= summary
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
    METHOD_TRUST_REGION: trust_region_unlearn,
    METHOD_BAND_ASCENT: band_ascent_unlearn,
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
