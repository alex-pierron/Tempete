"""Stage 2: soft targets on the forget images, anchored on the in-distribution generated proxy.

`gold_soft` (team A) trains the student on the forget images towards the teacher's view-averaged
softmax and anchors it with KL on out-of-distribution COCO/Kodak images; the anchor is what makes
the server's retain term drop (KL on in-distribution control images 1.4 to 2.1). Here the anchor
is the generated proxy (in distribution, unseen by the original), and the teacher can be the
retain-fine-tuned student of `finetune` (it already routes the nine removed classes to their
sisters) instead of the original.

Per step, one batch of forget images (exact or altered copy) is pushed towards its target
(KL at T = 1 and T = 2) and one batch of proxy images towards the original's own answer (CE on the
proxy label and KD at T = 2). Same statistics and exports as `finetune`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as torch_f
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.constants import DEFAULT_SEED, NUM_CLASSES, SCORING_TEMPERATURE
from hackathon.unlearning.model import load_model, predict, preprocess, server_autocast
from hackathon.unlearning.research.retain_proxy.concepts import FULLY_FORGOTTEN_CLASSES
from hackathon.unlearning.research.retain_proxy.finetune import (
    PREDICT_BATCH,
    FinetuneConfig,
    ProxyData,
    _export_team_c_logits,
    _load_forget,
    augment_batch,
    forget_statistics,
    holdout_statistics,
)
from hackathon.unlearning.research.retain_proxy.gpu_memory import (
    DEFAULT_MARGIN_GB,
    DEFAULT_TARGET_GB,
    BatchSizer,
    MemoryProbe,
    device_budget_gb,
    make_probe,
)
from hackathon.unlearning.training import kd_from_teacher, kl_to_student, learning_rate_at

NATIVE_TEMPERATURE = 1.0
VIEW_SEED_OFFSET = 101
ALTERED_SEED_OFFSET = 202
LOG_FLOOR = 1e-12


class DistillError(HackathonError):
    """Bad configuration or non-finite loss in the stage-2 run."""


class DistillConfig(BaseModel):
    """One stage-2 variant; immutable and dumped next to its results."""

    model_config = ConfigDict(frozen=True)

    name: str
    dataset_path: Path
    original_path: Path
    forget_path: Path
    out_dir: Path
    teacher_path: Path | None = Field(
        default=None, description="Teacher state dict (a fine-tuned student); None = the original."
    )
    init_from: Literal["original", "teacher"] = "original"
    target_views: int = Field(default=4, ge=1)
    target_area: float = Field(default=0.35, gt=0, le=1)
    target_temperature: float = Field(default=2.0, gt=0)
    removal_floor: float | None = Field(
        default=None,
        gt=0,
        lt=0.01,
        description="Give the nine removed classes this probability in every target (teacher = original).",
    )
    altered_copies: int = Field(default=2, ge=0)
    lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    epochs: int = Field(default=6, ge=1)
    steps_per_epoch: int = Field(default=40, ge=1)
    forget_batch: int = Field(default=128, ge=1)
    proxy_batch: int = Field(default=256, ge=1)
    warmup_fraction: float = Field(default=0.1, ge=0, lt=1)
    forget_weight: float = Field(default=1.0, ge=0)
    proxy_ce_weight: float = Field(default=0.5, ge=0)
    proxy_kd_weight: float = Field(default=1.0, ge=0)
    crop_min_area: float = Field(default=0.6, gt=0, le=1)
    holdout_fraction: float = Field(default=0.1, ge=0, lt=0.5)
    seed: int = DEFAULT_SEED
    start_micro_batch: int = Field(default=128, ge=1)
    max_micro_batch: int = Field(default=512, ge=1)
    target_memory_gb: float = DEFAULT_TARGET_GB
    memory_margin_gb: float = DEFAULT_MARGIN_GB
    save_model: bool = False
    export_logits_dir: Path | None = None
    export_alphas: tuple[float, ...] = (1.0,)
    export_subset_size: int = Field(default=1000, ge=1)
    device: str = "cuda"


def remove_classes(target: Tensor, classes: Sequence[int], floor: float) -> Tensor:
    """Give `classes` the probability `floor` each, renormalise the rest (rows still sum to 1)."""
    index = list(classes)
    kept = torch.ones(target.shape[1], dtype=torch.bool, device=target.device)
    kept[index] = False
    rest = target * kept
    rest = rest / rest.sum(dim=1, keepdim=True).clamp_min(LOG_FLOOR)
    out = rest * (1.0 - floor * len(index))
    out[:, index] = floor
    return out


def temper(probabilities: Tensor, temperature: float) -> Tensor:
    """softmax(log p / T): the distribution p would have with its logits divided by T."""
    return torch_f.softmax(probabilities.clamp_min(LOG_FLOOR).log() / temperature, dim=1)


@torch.no_grad()
def build_targets(
    teacher: nn.Module, forget: Tensor, config: DistillConfig, device: torch.device
) -> Tensor:
    """View-averaged tempered softmax of the teacher on strong views of every image: [N, 100]."""
    generator = torch.Generator().manual_seed(config.seed + VIEW_SEED_OFFSET)
    rows: list[Tensor] = []
    for start in range(0, len(forget), PREDICT_BATCH):
        batch = forget[start : start + PREDICT_BATCH]
        views: list[Tensor] = []
        for _ in range(config.target_views):
            altered = strong_view(batch, generator, config.target_area)
            with server_autocast(device):
                logits = teacher(preprocess(altered, device)).float()
            views.append(temper(torch_f.softmax(logits, dim=1), config.target_temperature))
        target = torch.stack(views).mean(dim=0)
        if config.removal_floor is not None:
            target = remove_classes(target, FULLY_FORGOTTEN_CLASSES, config.removal_floor)
        rows.append(target.cpu())
    return torch.cat(rows)


def altered_copies(forget: Tensor, copies: int, seed: int) -> list[Tensor]:
    """`copies` fixed random alterations (flip, soft crop, jitter, JPEG) of every forget image."""
    generator = torch.Generator().manual_seed(seed + ALTERED_SEED_OFFSET)
    result: list[Tensor] = []
    for _ in range(copies):
        parts = [
            random_augment(forget[start : start + PREDICT_BATCH], generator)
            for start in range(0, len(forget), PREDICT_BATCH)
        ]
        result.append(torch.cat(parts))
    return result


def matching_loss(target: Tensor, logits: Tensor) -> Tensor:
    """Mean of KL(target || student) at T = 1 and of the same at the scoring temperature T = 2."""
    native = kl_to_student(target, logits, NATIVE_TEMPERATURE)
    tempered = kl_to_student(temper(target, SCORING_TEMPERATURE), logits, SCORING_TEMPERATURE)
    return 0.5 * (native + tempered)


def _micro_loss(
    student: nn.Module,
    original: nn.Module,
    images: Tensor,
    is_forget: Tensor,
    targets: Tensor,
    labels: Tensor,
    config: DistillConfig,
) -> Tensor:
    """Per-micro-batch loss SUM (the caller divides by the step's batch size)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
        logits = student(images).float()
    total = logits.new_zeros(())
    if is_forget.any() and config.forget_weight > 0:
        selected = logits[is_forget]
        total = total + config.forget_weight * matching_loss(targets[is_forget], selected) * len(
            selected
        )
    proxy = ~is_forget
    if proxy.any():
        proxy_logits = logits[proxy]
        if config.proxy_ce_weight > 0:
            total = total + config.proxy_ce_weight * torch_f.cross_entropy(
                proxy_logits, labels[proxy], reduction="sum"
            )
        if config.proxy_kd_weight > 0:
            with torch.no_grad(), server_autocast(images.device):
                teacher_logits: Tensor = original(images[proxy]).float()
            total = total + config.proxy_kd_weight * kd_from_teacher(
                teacher_logits, proxy_logits, SCORING_TEMPERATURE
            ) * len(proxy_logits)
    result: Tensor = total
    return result


def run_distill(
    config: DistillConfig,
    *,
    original: nn.Module | None = None,
    teacher: nn.Module | None = None,
    forget_images: Tensor | None = None,
) -> dict[str, object]:
    """Train one stage-2 variant; same outputs as `finetune.run_finetune`."""
    device = torch.device(config.device)
    started = time.time()
    run_dir = config.out_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=1))

    base = (original if original is not None else load_model(config.original_path, device)).to(
        device
    )
    base = base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    if teacher is None:
        teacher = base
        if config.teacher_path is not None:
            teacher = load_model(config.teacher_path, device)
    teacher = teacher.to(device).eval()
    forget = forget_images if forget_images is not None else _load_forget(config.forget_path)
    original_forget_logits = predict(base, forget, PREDICT_BATCH).numpy()
    data = ProxyData(config.dataset_path, device)
    holdout_images, holdout_labels = data.hold_out(config.holdout_fraction, config.seed)
    holdout_np = holdout_labels.cpu().numpy()
    original_holdout = (
        predict(base, holdout_images, PREDICT_BATCH).numpy() if len(holdout_np) else None
    )

    targets = build_targets(teacher, forget, config, device)
    sources = [forget, *altered_copies(forget, config.altered_copies, config.seed)]
    student = copy.deepcopy(teacher if config.init_from == "teacher" else base)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    probe: MemoryProbe = make_probe(device)
    cuda = device.type == "cuda"
    step_images = config.forget_batch + config.proxy_batch
    sizer = BatchSizer(
        size=min(config.start_micro_batch, config.max_micro_batch, step_images),
        maximum=min(config.max_micro_batch, step_images),
        target_gb=config.target_memory_gb,
        cap_gb=device_budget_gb(device, config.memory_margin_gb) if cuda else 1e9,
        weights_gb=torch.cuda.memory_allocated(device) / 1e9 if cuda else 0.0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    total_steps = config.epochs * config.steps_per_epoch
    history: list[dict[str, object]] = []
    logits = original_forget_logits
    step = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for _ in range(config.steps_per_epoch):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, total_steps, config.lr, config.warmup_fraction)
            epoch_loss += _distill_step(
                student, base, optimizer, data, sources, targets, config, sizer, probe, generator
            )
            step += 1
        student.eval()
        logits = predict(student, forget, PREDICT_BATCH).numpy()
        entry: dict[str, object] = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": epoch_loss / config.steps_per_epoch,
            "micro_batch": sizer.size,
            "peak_memory_gb": max((peak for _, peak in sizer.history), default=None),
            "oom_events": sizer.oom_events,
            "seconds": time.time() - started,
            "forget": json.loads(
                forget_statistics(logits, original_forget_logits).model_dump_json()
            ),
        }
        if original_holdout is not None:
            held = predict(student, holdout_images, PREDICT_BATCH).numpy()
            entry["holdout"] = json.loads(
                holdout_statistics(held, original_holdout, holdout_np).model_dump_json()
            )
        student.train()
        history.append(entry)
        print(json.dumps(entry), flush=True)
    student.eval()
    np.savez(run_dir / "forget_logits.npz", model=logits, original=original_forget_logits)
    if config.save_model:
        torch.save(
            {k: v.detach().cpu().float() for k, v in student.state_dict().items()},
            run_dir / "model.pt",
        )
    blends: list[dict[str, object]] = []
    if config.export_logits_dir is not None:
        holdout = (
            (holdout_images, holdout_np, original_holdout) if original_holdout is not None else None
        )
        export_config = FinetuneConfig(
            name=config.name,
            dataset_path=config.dataset_path,
            original_path=config.original_path,
            forget_path=config.forget_path,
            out_dir=config.out_dir,
            export_logits_dir=config.export_logits_dir,
            export_alphas=config.export_alphas,
            export_subset_size=config.export_subset_size,
            device=config.device,
        )
        _, blends = _export_team_c_logits(
            export_config, base, student, forget, original_forget_logits, holdout
        )
        (run_dir / "blends.json").write_text(json.dumps(blends, indent=1))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {
        "name": config.name,
        "history": history,
        "seconds": time.time() - started,
        "blends": blends,
    }


def _prepare_images(
    pixels: Tensor,
    is_forget: Tensor,
    generator: torch.Generator,
    config: DistillConfig,
    device: torch.device,
) -> Tensor:
    """Model input: forget rows as they are (their alterations are already in the pool), proxy rows augmented."""
    images = torch.empty((len(pixels), 3, pixels.shape[1], pixels.shape[2]), device=device)
    if is_forget.any():
        images[is_forget] = preprocess(pixels[is_forget], device)
    if (~is_forget).any():
        images[~is_forget] = augment_batch(pixels[~is_forget], generator, config.crop_min_area)
    return images


def _distill_step(
    student: nn.Module,
    original: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: ProxyData,
    sources: Sequence[Tensor],
    targets: Tensor,
    config: DistillConfig,
    sizer: BatchSizer,
    probe: MemoryProbe,
    generator: torch.Generator,
) -> float:
    """One optimiser step on a forget batch plus a proxy batch; retried whole on OOM."""
    device = data.images.device
    forget_index = torch.randint(0, len(targets), (config.forget_batch,), generator=generator)
    source_pick = torch.randint(0, len(sources), (config.forget_batch,), generator=generator)
    forget_pixels = torch.stack(
        [sources[int(pick)][int(row)] for pick, row in zip(source_pick, forget_index, strict=True)]
    ).to(device)
    proxy_index = torch.randint(0, len(data), (config.proxy_batch,), generator=generator)
    proxy_pixels = data.images[proxy_index.to(device)]
    pixels = torch.cat([forget_pixels, proxy_pixels])
    step_targets = torch.cat(
        [targets[forget_index], torch.zeros(config.proxy_batch, NUM_CLASSES)]
    ).to(device)
    labels = torch.cat(
        [
            torch.zeros(config.forget_batch, dtype=torch.long, device=device),
            data.labels[proxy_index.to(device)],
        ]
    )
    is_forget = torch.cat(
        [
            torch.ones(config.forget_batch, dtype=torch.bool),
            torch.zeros(config.proxy_batch, dtype=torch.bool),
        ]
    ).to(device)
    count = len(pixels)
    oom_errors: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
    while True:
        optimizer.zero_grad(set_to_none=True)
        probe.reset_peak()
        total = 0.0
        try:
            for start in range(0, count, sizer.size):
                rows = slice(start, start + sizer.size)
                images = _prepare_images(pixels[rows], is_forget[rows], generator, config, device)
                loss = _micro_loss(
                    student,
                    original,
                    images,
                    is_forget[rows],
                    step_targets[rows],
                    labels[rows],
                    config,
                )
                (loss / count).backward()  # type: ignore[no-untyped-call]  # torch stub gap
                total += float(loss.detach()) / count
            break
        except oom_errors:
            probe.release_cache()
            optimizer.zero_grad(set_to_none=True)
            sizer.after_oom()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    sizer.after_success(min(sizer.size, count), probe.peak_gb())
    if not math.isfinite(total):
        raise DistillError("Non-finite loss.")
    return total


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: one variant from a JSON file of `DistillConfig` fields (plus `--set key=json` overrides)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    fields = json.loads(args.config_json.read_text())
    for override in args.set:
        key, _, value = override.partition("=")
        fields[key] = json.loads(value)
    fields["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    run_distill(DistillConfig(**fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
