"""Jobs of team G: pilot, zoo, singletons, held-out study and the streamed candidates.

Usage (see `jureca/unlearning_g_4gpu.sbatch` for the node script)::

    python -m hackathon.unlearning.research.g_run build-bank --out bank_a.pt --name a
    python -m hackathon.unlearning.research.g_run run --job calibrate --agent-dir DIR ...
    python -m hackathon.unlearning.research.g_run run --job stream_main --agent-dir DIR ...
    python -m hackathon.unlearning.research.g_run list-jobs

Layout under `--agent-dir` (nothing is written anywhere else, one container per output):

    data/unlearning/{model.pt, forget_images.npz}     inputs (symlinks to the shared copies)
    data/unlearning/research_g/bank_<id>.pt           per-image targets built by `build-bank`
    data/unlearning/research_g/calibration.json       the pilot's choice of inner optimiser/lr
    data/unlearning/research_g/results/<job>.json     every measurement of a job
    data/unlearning/research_g/exports/<name>.npz     team C's per-view logits/features
    data/unlearning/candidates/g/<variant>.pt         validated base state dicts (alpha = 1)

A job never submits anything to the evaluation server.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.constants import (
    DEFAULT_SEED,
    FORGET_IMAGES_FILENAME,
    MODEL_FILENAME,
)
from hackathon.unlearning.data import load_forget_images, load_proxy_images
from hackathon.unlearning.model import load_model, predict, select_device
from hackathon.unlearning.proxy_eval import tempered_kl
from hackathon.unlearning.research.g_classes import (
    class_counts,
    cosine_groups,
    fully_forgotten_classes,
    present_siblings,
)
from hackathon.unlearning.research.g_meta import (
    AdaptRecord,
    MetaContext,
    MetaSpec,
    TrainableMode,
    adapt_task,
    halve_micro_batch_cap,
    holdout_split,
    per_sample_matching_kl,
    run_stream,
    set_trainable,
)
from hackathon.unlearning.research.g_probe import (
    ProbeSet,
    build_probe_set,
    distribution_report,
    export_views,
    load_generated_images,
    measure_probe,
)
from hackathon.unlearning.research.g_targets import TargetSpec, build_target_bank
from hackathon.unlearning.submission import save_submission

LOGGER = logging.getLogger("hackathon.unlearning.research.g")
RESEARCH_SUBDIR = "research_g"
RESULTS_SUBDIR = "results"
EXPORTS_SUBDIR = "exports"
CANDIDATES_SUBDIR = Path("candidates") / "g"
CALIBRATION_FILE = "calibration.json"
THREAD_LIMIT = 4
DEFAULT_MICRO_BATCH = 16
SELF_ANCHOR_BATCH = 16
MAIN_BANK = "a"
POOL_ANCHOR_BATCH = 8
RETAIN_OOD_PROBE = 128
GENERATED_PROBE = 128
GENERATED_TRAIN = 1024
GENERATED_PROBE_SKIP_CHUNKS = 6
"""Team D's chunks hold 256 images: the first 6 (1536 images) feed the anchor, the next feeds the probe."""
FORGET_PROBE = 192
EXPORT_ALPHAS = (0.9, 1.0, 1.1)
ADAPT_STEPS_PILOT = 5
PILOT_TASKS = 6
PILOT_TASK_SIZE = 4
PILOT_RATIO_TARGET = 0.5
ZOO_PLAN = ((1, 24), (4, 16), (16, 8), (64, 4), (256, 2))
"""(task size, repetitions) of the capacity study."""
HOLDOUT_FEW_SHOT_PLAN = ((1, 12), (4, 8), (16, 4))
HOLDOUT_COUNT = 500
SINGLETON_REQUESTS = 128
PILOT_GRID: tuple[tuple[str, float, float | None], ...] = (
    ("sgd", 1e-3, None), ("sgd", 3e-3, None), ("sgd", 1e-2, None), ("sgd", 3e-2, None),
    ("adam", 1e-5, 5.0), ("adam", 2e-5, 5.0), ("adam", 5e-5, 5.0), ("adam", 1e-4, 5.0),
)  # fmt: skip
"""(inner optimiser, learning rate, gradient clip) explored by the pilot."""


class JobError(HackathonError):
    """A job was asked for something it cannot do (unknown name, missing bank, bad layout)."""


@dataclass(frozen=True)
class Variant:
    """One streamed candidate: a bank, a loop and the multiplier of the calibrated learning rate."""

    bank: str
    lr_scale: float = 1.0
    task_sizes: tuple[int, ...] = (16,)
    inner_steps: int = 3
    budget_fraction: float = 0.1
    outer_lr: float = 1.0
    passes: int = 1
    trainable: TrainableMode = "all"
    strong_weight: float = 1.0
    persistent_optimizer: bool = False
    anchor: str = "self"
    seed: int = DEFAULT_SEED

    @property
    def submittable(self) -> bool:
        """Only anchors derived from the forget set and the model alone respect the subject."""
        return self.anchor == "self"


def _gold_soft(anchor: str = "self", seed: int = DEFAULT_SEED) -> Variant:
    """The gold_soft protocol: plain view targets, one step per task, one persistent AdamW."""
    return Variant(
        "v2",
        inner_steps=1,
        budget_fraction=0.0,
        strong_weight=0.0,
        persistent_optimizer=True,
        anchor=anchor,
        seed=seed,
    )


VARIANTS: dict[str, Variant] = {
    "main": Variant("a"),
    "main_s1": Variant("a", seed=1),
    "main_s2": Variant("a", seed=2),
    "reptile": Variant("a", outer_lr=0.5, passes=2),
    "mixed": Variant("a", task_sizes=(1, 4, 16, 64), outer_lr=0.5, passes=2),
    "norouting": Variant("b"),
    "sharp": Variant("d"),
    "blur_m": Variant("m"),
    "blur_f": Variant("f"),
    "head": Variant("a", lr_scale=20.0, trainable="head"),
    "last4": Variant("a", trainable="last4"),
    "nobudget": Variant("a", budget_fraction=0.0),
    # The gold_soft protocol on the forget-only anchor, seeds 0 and 1 (A/B partner of main, main_s1).
    "baseline_s1": _gold_soft(),
    "baseline_s2": _gold_soft(seed=1),
    # Controls (NEVER submittable: their anchor uses images from outside the forget set):
    "baseline": _gold_soft(anchor="ood"),
    "main_ood": Variant("a", anchor="ood"),
}
CHAINS: dict[str, tuple[str, ...]] = {"stream_sharp": ("stream_blur_m", "stream_blur_f")}
"""Jobs run right after another one in the same process (the queued node script lists fixed names)."""
ANALYSIS_JOBS = ("calibrate", "zoo", "singletons", "holdout")
MID_BLUR_POWER = 6.0
FLAT_BLUR_POWER = 0.4
BANK_SPECS: dict[str, TargetSpec] = {
    # Bimodality ladder at equal entropy (blur = share of images with p(true) in (0.1, 0.9)), on the
    # exact images: a ~12 %, m ~27 %, f ~61 %. Team C measured that sharper files score higher
    # (c_sib2 head x1.25: 0.138, x1.75: 0.157), so a (the sharpest) is the main bank; m and f are
    # kept for comparison.
    "a": TargetSpec(),
    "m": TargetSpec(other_tail_fraction=0.0, other_shape_power=MID_BLUR_POWER),
    "f": TargetSpec(other_tail_fraction=0.0, other_shape_power=FLAT_BLUR_POWER),
    # Ablations of the main bank: no sibling routing; a lower entropy (1.2 nat at T = 2).
    "b": TargetSpec(routing_share=0.0),
    "d": TargetSpec(entropy_target=1.2),
    "v2": TargetSpec(kind="v2"),
}


@dataclass(frozen=True)
class PilotRow:
    """One pilot configuration: how close it got to the targets and what it disturbed on the way."""

    optimizer: str
    lr: float
    grad_clip: float | None
    diverged: bool
    ratio: float = float("nan")
    steps: float = float("nan")
    drift_generated: float = float("nan")
    drift_forget_other: float = float("nan")

    @property
    def collateral(self) -> float:
        """Drift on retention-like images plus drift on forget images of other classes."""
        return self.drift_generated + self.drift_forget_other


@dataclass(frozen=True)
class Bank:
    """Targets of the exact images and of strong crops (rows aligned with the forget set)."""

    exact: Tensor
    altered: Tensor


@dataclass
class World:
    """Everything a job reads: model, images, targets, retention pools, class structure."""

    device: torch.device
    original: nn.Module
    original_state: dict[str, Tensor]
    forget_all: Tensor
    retain_ood: Tensor
    retain_generated: Tensor
    ood_probe: Tensor
    generated_probe: Tensor
    saturated: list[int]
    siblings: dict[int, list[int]]
    labels_all: Tensor
    orig_logits_all: Tensor
    agent_dir: Path

    def anchor_pool(self, anchor: str) -> Tensor:
        """Images the retention anchor draws from during TRAINING.

        `self`: none (the anchor is built from the forget set, `g_meta.build_self_anchor`): the only
        kind a submitted file may use (the subject: "from the original model and the forget set
        alone"). `ood`: natural images (COCO, DIV2K, Kodak). `all`: natural + team D's SDXL images.
        The last two are CONTROLS, never submittable; the same images stay available for MEASURING.
        """
        if anchor == "self":
            return self.retain_ood[:0]
        if anchor == "ood":
            return self.retain_ood
        if anchor == "all":
            return torch.cat([self.retain_ood, self.retain_generated])
        raise JobError(f"unknown anchor '{anchor}', choose 'all' or 'ood'")

    @property
    def research_dir(self) -> Path:
        """Folder of every output of this team."""
        return self.agent_dir / "data" / "unlearning" / RESEARCH_SUBDIR


def _load_bank(world: World, bank_id: str) -> Bank:
    path = world.research_dir / f"bank_{bank_id}.pt"
    if not path.is_file():
        raise JobError(f"missing target bank {path}: run `build-bank --name {bank_id}`")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return Bank(payload["targets"], payload["targets_altered"])


def build_world(
    agent_dir: Path, proxy_root: Path, generated_dir: Path, device: torch.device
) -> World:
    """Load the model, the forget images, the retention pools and the class structure."""
    task_dir = agent_dir / "data" / "unlearning"
    original = load_model(task_dir / MODEL_FILENAME, device)
    forget_all = load_forget_images(task_dir / FORGET_IMAGES_FILENAME)
    ood = load_proxy_images(proxy_root, 10_000, DEFAULT_SEED)
    order = torch.randperm(len(ood), generator=torch.Generator().manual_seed(DEFAULT_SEED))
    ood_probe, ood_train = ood[order[:RETAIN_OOD_PROBE]], ood[order[RETAIN_OOD_PROBE:]]
    generated_train = load_generated_images(generated_dir, GENERATED_TRAIN)
    generated_probe = load_generated_images(
        generated_dir, GENERATED_PROBE, skip_chunks=GENERATED_PROBE_SKIP_CHUNKS
    )
    orig_logits_all = predict(original, forget_all)
    labels = orig_logits_all.argmax(dim=1)
    counts = class_counts(labels.numpy())
    saturated = fully_forgotten_classes(counts)
    weight = original.state_dict()["head.weight"].double().cpu().numpy()
    return World(
        device, original, {k: v.detach().cpu().clone() for k, v in original.state_dict().items()},
        forget_all, ood_train, generated_train, ood_probe, generated_probe,
        saturated, present_siblings(cosine_groups(weight), saturated), labels, orig_logits_all,
        agent_dir,
    )  # fmt: skip


def make_context(world: World, bank: Bank, spec: MetaSpec, anchor: str = "all") -> MetaContext:
    """A fresh student (copy of the original) with the trainable subset of `spec`."""
    student = copy.deepcopy(world.original).to(world.device)
    teacher = world.original
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    params = set_trainable(student, spec.trainable)
    return MetaContext(
        world.device,
        student,
        teacher,
        world.forget_all,
        bank.exact,
        world.anchor_pool(anchor),
        params,
        bank.altered,
    )


def make_probe(world: World, targets: Tensor, pool: Tensor, seed: int) -> ProbeSet:
    """Probe set whose forget part is a seeded draw of `FORGET_PROBE` images from `pool`."""
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))
    chosen = pool[order[:FORGET_PROBE]].sort().values
    return build_probe_set(
        world.original, world.forget_all, targets, chosen, world.generated_probe, world.ood_probe,
        world.saturated, world.siblings,
    )  # fmt: skip


def restore_original(ctx: MetaContext, world: World) -> None:
    """Put the student back on the original weights (in place: parameter objects are kept)."""
    ctx.student.load_state_dict(world.original_state)


@torch.no_grad()
def task_distance(ctx: MetaContext, indices: Tensor, seed: int) -> tuple[float, float]:
    """Mean matching KL of the task images to their targets: (exact copies, one altered copy)."""
    ctx.student.eval()
    images = ctx.forget_images[indices]
    targets = ctx.targets[indices]
    exact = predict(ctx.student, images)
    altered = predict(ctx.student, random_augment(images, torch.Generator().manual_seed(seed)))
    return (
        float(per_sample_matching_kl(targets, exact).mean()),
        float(per_sample_matching_kl(targets, altered).mean()),
    )


def _spec_from(
    variant: Variant, name: str, optimizer: str, lr: float, clip: float | None, **extra: object
) -> MetaSpec:
    return MetaSpec(
        name=name, task_sizes=variant.task_sizes, inner_optimizer=optimizer,  # type: ignore[arg-type]
        inner_lr=lr * variant.lr_scale, inner_steps=variant.inner_steps,
        budget_fraction=variant.budget_fraction, outer_lr=variant.outer_lr,
        passes=variant.passes, trainable=variant.trainable, grad_clip=clip,
        anchor="self" if variant.anchor == "self" else "pool",
        retain_batch=SELF_ANCHOR_BATCH if variant.anchor == "self" else POOL_ANCHOR_BATCH,
        strong_weight=variant.strong_weight, persistent_optimizer=variant.persistent_optimizer,
        seed=variant.seed,
        **extra,  # type: ignore[arg-type]
    )  # fmt: skip


def read_calibration(research_dir: Path) -> tuple[str, float, float | None]:
    """The pilot's choice (optimiser, lr, clip); Adam 2e-5 (the base candidate's setting) if absent."""
    path = research_dir / CALIBRATION_FILE
    if not path.is_file():
        LOGGER.warning("no %s: falling back to adam 2e-5", CALIBRATION_FILE)
        return "adam", 2e-5, 5.0
    chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    return str(chosen["optimizer"]), float(chosen["lr"]), chosen["grad_clip"]


def _record_summary(record: AdaptRecord, distance: tuple[float, float]) -> dict[str, float]:
    return {
        "size": float(record.task_size),
        "steps": float(record.steps),
        "seconds": record.seconds,
        "kl_initial": record.kl_per_step[0],
        "kl_final_exact": distance[0],
        "kl_final_altered": distance[1],
        "grad_norm_first": record.grad_norm_per_step[0] if record.grad_norm_per_step else 0.0,
    }


def _collateral(
    ctx: MetaContext, probe: ProbeSet, task_labels: Tensor, skip_forget: Tensor | None = None
) -> dict[str, float]:
    """Drift caused by one task on retention images and on the other forget images, by class overlap."""
    forget = predict(ctx.student, probe.forget)
    generated = predict(ctx.student, probe.generated)
    ood = predict(ctx.student, probe.ood)
    per_image = tempered_kl(probe.orig_forget, forget)
    keep = torch.ones(len(per_image), dtype=torch.bool) if skip_forget is None else ~skip_forget
    same = torch.isin(probe.labels, task_labels) & keep
    other = ~torch.isin(probe.labels, task_labels) & keep
    return {
        "drift_generated_kl_t2": float(tempered_kl(probe.orig_generated, generated).mean()),
        "drift_ood_kl_t2": float(tempered_kl(probe.orig_ood, ood).mean()),
        "drift_forget_same_class": float(per_image[same].mean()) if same.any() else float("nan"),
        "drift_forget_other_class": float(per_image[other].mean()) if other.any() else float("nan"),
    }


def _task_pool(world: World, probe: ProbeSet) -> Tensor:
    """Forget images that are not probe images (tasks must not be measured on themselves)."""
    keep = torch.ones(len(world.forget_all), dtype=torch.bool)
    keep[probe.forget_indices] = False
    return keep.nonzero(as_tuple=True)[0]


def job_calibrate(world: World, bank: Bank) -> dict[str, object]:
    """Pilot: which inner optimiser and step reach the targets of small tasks with the least collateral.

    This is the one quantity we META-LEARN by search: the step size of the inner loop. Every task
    of the pilot starts from the original; a configuration is eligible when the mean ratio of final
    to initial distance is below `PILOT_RATIO_TARGET` within `ADAPT_STEPS_PILOT` steps.
    """
    probe = make_probe(world, bank.exact, torch.arange(len(world.forget_all)), DEFAULT_SEED)
    pool = _task_pool(world, probe)
    saturated_pool = pool[torch.isin(world.labels_all[pool], torch.tensor(world.saturated))]
    other_pool = pool[~torch.isin(world.labels_all[pool], torch.tensor(world.saturated))]
    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    half = PILOT_TASK_SIZE // 2
    tasks = [
        torch.cat([
            saturated_pool[torch.randperm(len(saturated_pool), generator=generator)[:half]],
            other_pool[torch.randperm(len(other_pool), generator=generator)[:half]],
        ])
        for _ in range(PILOT_TASKS)
    ]  # fmt: skip
    rows: list[PilotRow] = []
    ctx = make_context(world, bank, MetaSpec(name="pilot"))
    for optimizer, lr, clip in PILOT_GRID:
        spec = MetaSpec(
            name=f"pilot_{optimizer}_{lr}",
            inner_optimizer=optimizer,  # type: ignore[arg-type]
            inner_lr=lr,
            inner_steps=ADAPT_STEPS_PILOT,
            grad_clip=clip,
        )
        ratios: list[float] = []
        steps: list[float] = []
        collateral: list[dict[str, float]] = []
        for index, task in enumerate(tasks):
            restore_original(ctx, world)
            try:
                record = adapt_task(ctx, task, spec, torch.Generator().manual_seed(index))
            except HackathonError as error:
                LOGGER.warning("%s diverged: %s", spec.name, error)
                break
            exact, _ = task_distance(ctx, task, index)
            ratios.append(exact / max(record.kl_per_step[0], 1e-9))
            steps.append(float(record.steps))
            collateral.append(_collateral(ctx, probe, world.labels_all[task]))
        if len(ratios) < len(tasks):
            rows.append(PilotRow(optimizer, lr, clip, diverged=True))
            continue
        row = PilotRow(
            optimizer,
            lr,
            clip,
            diverged=False,
            ratio=float(np.mean(ratios)),
            steps=float(np.mean(steps)),
            drift_generated=float(np.mean([c["drift_generated_kl_t2"] for c in collateral])),
            drift_forget_other=float(
                np.nanmean([c["drift_forget_other_class"] for c in collateral])
            ),
        )
        rows.append(row)
        LOGGER.info("pilot %s", row)
    live = [row for row in rows if not row.diverged]
    if not live:
        raise JobError("every pilot configuration diverged")
    eligible = [row for row in live if row.ratio <= PILOT_RATIO_TARGET]
    if eligible:
        chosen = min(eligible, key=lambda row: row.collateral)
    else:
        chosen = min(live, key=lambda row: row.ratio)
    payload: dict[str, object] = {
        "grid": [asdict(row) for row in rows],
        "chosen": asdict(chosen),
        "eligible_count": len(eligible),
    }
    (world.research_dir / CALIBRATION_FILE).write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )
    return payload


def job_zoo(world: World, bank: Bank) -> dict[str, object]:
    """Capacity of the operator from the original: 1..256 images per request, cost, fidelity, drift."""
    optimizer, lr, clip = read_calibration(world.research_dir)
    probe = make_probe(world, bank.exact, torch.arange(len(world.forget_all)), DEFAULT_SEED)
    pool = _task_pool(world, probe)
    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    spec = MetaSpec(
        name="zoo", inner_optimizer=optimizer, inner_lr=lr, grad_clip=clip,  # type: ignore[arg-type]
        inner_steps=ADAPT_STEPS_PILOT, budget_fraction=0.1,
    )  # fmt: skip
    ctx = make_context(world, bank, spec)
    rows: list[dict[str, float]] = []
    for size, repetitions in ZOO_PLAN:
        for repetition in range(repetitions):
            task = pool[torch.randperm(len(pool), generator=generator)[:size]]
            restore_original(ctx, world)
            record = adapt_task(ctx, task, spec, torch.Generator().manual_seed(repetition))
            row = _record_summary(record, task_distance(ctx, task, repetition))
            row.update(_collateral(ctx, probe, world.labels_all[task]))
            rows.append(row)
        LOGGER.info("zoo size %d done", size)
    return {"spec": spec.model_dump(), "rows": rows}


def job_singletons(world: World, bank: Bank) -> dict[str, object]:
    """128 single-image requests in a row from the original: cumulative drift and fidelity curve."""
    optimizer, lr, clip = read_calibration(world.research_dir)
    spec = MetaSpec(
        name="singletons", task_sizes=(1,), inner_optimizer=optimizer, inner_lr=lr,  # type: ignore[arg-type]
        grad_clip=clip, inner_steps=ADAPT_STEPS_PILOT, budget_fraction=0.1, monitor_every=8,
    )  # fmt: skip
    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    order = torch.randperm(len(world.forget_all), generator=generator)
    stream_indices = order[:SINGLETON_REQUESTS].sort().values
    probe = make_probe(world, bank.exact, stream_indices, DEFAULT_SEED)
    ctx = make_context(world, bank, spec)
    result = run_stream(ctx, stream_indices, spec, _monitor_factory(ctx, probe))
    return {
        "spec": spec.model_dump(), "history": result.history,
        "steps_per_request": [r.steps for r in result.records],
        "seconds_per_request": [r.seconds for r in result.records],
        "stream_seconds": result.seconds,
    }  # fmt: skip


def _monitor_factory(
    ctx: MetaContext, probe: ProbeSet
) -> Callable[[int, list[Tensor]], dict[str, float]]:
    def monitor(_: int, done: list[Tensor]) -> dict[str, float]:
        processed = torch.zeros(len(probe.forget_indices), dtype=torch.bool)
        if done:
            processed = torch.isin(probe.forget_indices, torch.cat(done))
        metrics = measure_probe(ctx.student, probe, processed)
        ctx.student.train()
        return metrics

    return monitor


def _arm_spec(
    name: str, variant: Variant, calibration: tuple[str, float, float | None], **extra: object
) -> MetaSpec:
    optimizer, lr, clip = calibration
    return _spec_from(variant, name, optimizer, lr, clip, **extra)


def _state_of(student: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}


def _leak_metrics(ctx: MetaContext, world: World, held_idx: Tensor, bank: Bank) -> dict[str, float]:
    """What a stream trained WITHOUT `held_idx` did to those memorised, never-requested images.

    This is the quantity the baseline `gold_soft` is suspected to inflate (a leak onto every
    memorised image, which is what the hidden control set looks like): KL to the original at the
    scoring temperature, argmax flips, confidence, loss moments and fidelity to the targets, plus
    the same on one altered (strong crop) copy.
    """
    ctx.student.eval()
    images = world.forget_all[held_idx]
    labels = world.labels_all[held_idx]
    reference = world.orig_logits_all[held_idx]
    logits = predict(ctx.student, images)
    strong = strong_view(images, torch.Generator().manual_seed(DEFAULT_SEED + 5), 0.35)
    strong_logits = predict(ctx.student, strong)
    strong_reference = predict(world.original, strong)
    in_saturated = torch.isin(labels, torch.tensor(world.saturated))
    per_image = tempered_kl(reference, logits)
    loss = torch.nn.functional.cross_entropy(logits.double(), labels, reduction="none")
    return {
        "kl_t2": float(per_image.mean()),
        "kl_t2_nonsat": float(per_image[~in_saturated].mean()),
        "kl_t2_sat": float(per_image[in_saturated].mean()) if in_saturated.any() else 0.0,
        "flip_rate": float((logits.argmax(1) != reference.argmax(1)).float().mean()),
        "confidence": float(torch.softmax(logits.double(), 1).amax(1).mean()),
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std(unbiased=False)),
        "fidelity_mean": float(per_sample_matching_kl(bank.exact[held_idx], logits).mean()),
        "strong_kl_t2": float(tempered_kl(strong_reference, strong_logits).mean()),
        "strong_loss_mean": float(
            torch.nn.functional.cross_entropy(
                strong_logits.double(), labels, reduction="none"
            ).mean()
        ),
    }


def job_holdout(world: World, bank: Bank) -> dict[str, object]:
    """A/B on 3000 meta-train images against 500 never-requested ones, then few-shot from two inits.

    Arm `meta`: calibrated targets, Reptile (3 steps per task, epsilon 0.5, two passes, per-image
    budget, strong-crop copies). Arm `baseline`: the gold_soft protocol re-run in this framework
    (plain tempered view targets, one step per task of 16, one persistent AdamW). Both are scored on
    the same held-out images: KL to the original (the leak), flips, confidence, loss, altered copy.
    """
    calibration = read_calibration(world.research_dir)
    train_idx, held_idx = holdout_split(len(world.forget_all), HOLDOUT_COUNT, DEFAULT_SEED)
    plain = _load_bank(world, "v2")
    probe = make_probe(world, bank.exact, train_idx, DEFAULT_SEED)
    arms: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, Tensor]] = {}
    for arm, arm_bank, variant in (
        ("meta", bank, Variant(MAIN_BANK, task_sizes=(4, 16, 64), outer_lr=0.5, passes=2)),
        ("baseline_self", plain, VARIANTS["baseline_s1"]),
        ("baseline_ood", plain, VARIANTS["baseline"]),
    ):
        spec = _arm_spec(f"holdout_{arm}", variant, calibration, holdout_count=HOLDOUT_COUNT)
        ctx = make_context(world, arm_bank, spec, variant.anchor)
        stream = run_stream(ctx, train_idx, spec, _monitor_factory(ctx, probe))
        arms[arm] = {
            "spec": spec.model_dump(), "history": stream.history, "seconds": stream.seconds,
            "leak_on_held_out": _leak_metrics(ctx, world, held_idx, bank),
            "mean_steps_per_task": float(np.mean([r.steps for r in stream.records])),
        }  # fmt: skip
        states[arm] = _state_of(ctx.student)
        LOGGER.info("holdout arm %s: %s", arm, arms[arm]["leak_on_held_out"])
    optimizer, lr, clip = calibration
    shot = MetaSpec(
        name="few_shot", inner_optimizer=optimizer, inner_lr=lr, grad_clip=clip,  # type: ignore[arg-type]
        inner_steps=ADAPT_STEPS_PILOT, budget_fraction=0.1,
    )  # fmt: skip
    ctx = make_context(world, bank, shot)
    generator = torch.Generator().manual_seed(DEFAULT_SEED + 1)
    rows: list[dict[str, float | str]] = []
    for size, repetitions in HOLDOUT_FEW_SHOT_PLAN:
        for repetition in range(repetitions):
            task = held_idx[torch.randperm(len(held_idx), generator=generator)[:size]]
            for init_name, state in (("original", world.original_state), ("meta", states["meta"])):
                ctx.student.load_state_dict(state)
                record = adapt_task(ctx, task, shot, torch.Generator().manual_seed(repetition))
                row: dict[str, float | str] = {"init": init_name, **_record_summary(
                    record, task_distance(ctx, task, repetition)
                )}  # fmt: skip
                row.update(_collateral(ctx, probe, world.labels_all[task]))
                rows.append(row)
        LOGGER.info("holdout few-shot size %d done", size)
    return {"arms": arms, "few_shot_rows": rows, "held_out_count": len(held_idx)}


def job_stream(world: World, variant_name: str) -> dict[str, object]:
    """Stream the whole forget set, export the base at several alphas, report fidelity, shape, drift."""
    variant = VARIANTS[variant_name]
    bank = _load_bank(world, variant.bank)
    spec = _arm_spec(f"g_{variant_name}", variant, read_calibration(world.research_dir))
    everything = torch.arange(len(world.forget_all))
    probe = make_probe(world, bank.exact, everything, DEFAULT_SEED)
    ctx = make_context(world, bank, spec, variant.anchor)
    result = run_stream(ctx, everything, spec, _monitor_factory(ctx, probe))
    ctx.student.eval()
    logits = predict(ctx.student, world.forget_all)
    fidelity = per_sample_matching_kl(bank.exact, logits)
    saturated_rows = torch.isin(world.labels_all, torch.tensor(world.saturated))
    drift = tempered_kl(world.orig_logits_all, logits)
    summary: dict[str, object] = {
        "spec": spec.model_dump(), "variant": {**variant.__dict__, "submittable": variant.submittable}, "history": result.history,
        "stream_seconds": result.seconds, "tasks": len(result.records),
        "mean_steps_per_task": float(np.mean([r.steps for r in result.records])),
        "fidelity_all": {
            "mean": float(fidelity.mean()), "median": float(fidelity.median()),
            "p90": float(fidelity.quantile(0.9)),
            "mean_saturated": float(fidelity[saturated_rows].mean()),
            "mean_other": float(fidelity[~saturated_rows].mean()),
        },
        "kl_t2_to_original": {
            "nonsat_forget": float(drift[~saturated_rows].mean()),
            "sat_forget": float(drift[saturated_rows].mean()),
            "all_forget": float(drift.mean()),
        },
        "shape_all": distribution_report(logits, world.labels_all, world.saturated, world.siblings),
    }  # fmt: skip
    stem = f"g_{variant_name}" if variant.submittable else f"ctl_{variant_name}"
    summary["submittable"] = variant.submittable
    summary["file_stem"] = stem
    path = world.agent_dir / "data" / "unlearning" / CANDIDATES_SUBDIR / f"{stem}.pt"
    save_submission(ctx.student, path)
    trained = _state_of(ctx.student)
    scratch = copy.deepcopy(world.original).eval()
    exports = world.research_dir / EXPORTS_SUBDIR
    shapes: dict[str, dict[str, dict[str, float]]] = {}
    for alpha in EXPORT_ALPHAS:
        blended = {
            key: world.original_state[key] + alpha * (value - world.original_state[key])
            if value.is_floating_point() else value
            for key, value in trained.items()
        }  # fmt: skip
        scratch.load_state_dict(blended)
        name = f"{stem}_a{alpha:.2f}"
        views = export_views(scratch, world.forget_all, exports / f"{name}.npz")
        subset = torch.from_numpy(np.load(exports / f"{name}.npz")["subset_indices"])
        labels = world.labels_all[subset]
        shapes[name] = {
            view.removeprefix("logits_"): distribution_report(
                torch.from_numpy(array), labels, world.saturated, world.siblings
            )
            for view, array in views.items()
        }
        shapes[name]["kl_t2_nonsat_exact_subset"] = {
            "value": float(
                tempered_kl(
                    world.orig_logits_all[subset], torch.from_numpy(views["logits_exact"])
                )[~torch.isin(labels, torch.tensor(world.saturated))].mean()
            )
        }  # fmt: skip
    summary["shape_by_alpha"] = shapes
    return summary


def run_job(job: str, world: World) -> dict[str, object]:
    """Dispatch a job by name (`list-jobs` shows them)."""
    if job.startswith("stream_"):
        name = job.removeprefix("stream_")
        if name not in VARIANTS:
            raise JobError(f"unknown variant '{name}', choose from {sorted(VARIANTS)}")
        return job_stream(world, name)
    analysis: dict[str, Callable[[World, Bank], dict[str, object]]] = {
        "calibrate": job_calibrate, "zoo": job_zoo, "singletons": job_singletons,
        "holdout": job_holdout,
    }  # fmt: skip
    if job not in analysis:
        raise JobError(f"unknown job '{job}'")
    return analysis[job](world, _load_bank(world, MAIN_BANK))


def run_job_with_retry(job: str, world: World, retries: int = 1) -> dict[str, object]:
    """Run a job; on CUDA out-of-memory, free the cache and retry with a smaller micro-batch."""
    for attempt in range(retries + 1):
        try:
            return run_job(job, world)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cap = halve_micro_batch_cap(DEFAULT_MICRO_BATCH)
            LOGGER.warning(
                "out of memory on %s (attempt %d): micro-batch cap %d", job, attempt + 1, cap
            )
            if attempt == retries:
                raise
    raise JobError("unreachable")


def build_bank_command(args: argparse.Namespace) -> None:
    """Build a target bank from the cached view targets and original logits (CPU, seconds)."""
    task_dir = Path(args.task_dir)
    views = torch.load(args.views, map_location="cpu", weights_only=True).float()
    logits = torch.load(args.exact_logits, map_location="cpu", weights_only=True)
    state = torch.load(task_dir / MODEL_FILENAME, map_location="cpu", weights_only=True)
    groups = cosine_groups(state["head.weight"].double().numpy())
    counts = class_counts(logits.argmax(dim=1).numpy())
    spec = BANK_SPECS[args.name]
    bank = build_target_bank(views, logits, counts, groups, spec)
    saturated = fully_forgotten_classes(counts)
    torch.save(
        {
            "targets": bank.targets, "targets_altered": bank.targets_altered,
            "loss_targets": bank.loss_targets, "spec": spec.model_dump(),
            "statistics": bank.statistics, "gamma": bank.gamma, "saturated": saturated,
            "siblings": {str(k): v for k, v in present_siblings(groups, saturated).items()},
        },
        args.out,
    )  # fmt: skip
    print(json.dumps({"bank": args.name, "gamma": bank.gamma, **bank.statistics}, indent=1))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bank = sub.add_parser("build-bank")
    bank.add_argument("--name", choices=sorted(BANK_SPECS), required=True)
    bank.add_argument("--task-dir", required=True)
    bank.add_argument("--views", required=True)
    bank.add_argument("--exact-logits", required=True)
    bank.add_argument("--out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--job", required=True)
    run.add_argument("--agent-dir", type=Path, required=True)
    run.add_argument("--proxy-root", type=Path, required=True)
    run.add_argument("--generated-dir", type=Path, required=True)
    sub.add_parser("list-jobs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(THREAD_LIMIT)
    if args.command == "list-jobs":
        print("\n".join([*ANALYSIS_JOBS, *(f"stream_{name}" for name in VARIANTS)]))
        return
    if args.command == "build-bank":
        build_bank_command(args)
        return
    started = time.time()
    world = build_world(args.agent_dir, args.proxy_root, args.generated_dir, select_device(None))
    results = world.research_dir / RESULTS_SUBDIR
    results.mkdir(parents=True, exist_ok=True)
    for job in (args.job, *CHAINS.get(args.job, ())):
        payload = run_job_with_retry(job, world)
        payload["job_seconds"] = time.time() - started
        partial = results / f"{job}.json.partial"
        partial.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        partial.replace(results / f"{job}.json")
        LOGGER.info("job %s done in %.0f s", job, payload["job_seconds"])
        started = time.time()


if __name__ == "__main__":
    main()
