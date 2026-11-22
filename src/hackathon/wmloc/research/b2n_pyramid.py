"""Full-resolution segmenter: the extractor's own backbone, fine-tuned on synthetic data (w-loc-b2n).

Gap analysis (reports/phase3/wmloc/team-gap): the score is lost on the zone to find (IoU 0.20),
and zones under 5 % of the image are half of the loss. The B2 head reads FROZEN features on a
16 to 32 cell grid (one cell = 16 to 32 px), so a zone of 60 px is two or three cells wide. The
synthetic shards (`b2_synth_features`) keep the full 512 px image of every sample, which makes a
trainable pixel pathway affordable: 12 288 labelled images per method instead of 136.

`PyramidNet` is the B2 head plus a trainable copy of the method's extractor backbone (ConvNeXt-tiny
for Pixel Seal, ResNet-50 for TrustMark) run on the 512 px image. Its four stages (strides 4, 8,
16, 32, i.e. grids 128, 64, 32, 16) feed a top-down decoder: the stride-16 level joins the trunk
of the head, the stride-8 and stride-4 levels are added on the way up, so the last decoder layer
works on the 128 x 128 evaluation grid with features computed at that resolution. The cached
frozen features (message-aware evidence included) are still read, so the model can only add to
what the head already knows.

Training is budgeted in SECONDS (a GPU task has a hard limit and the throughput is not known in
advance): a short probe measures the step rate, the number of steps is fixed from it before the
schedule is built, and it is logged so that a run can be repeated with `--max-steps`.

Usage (kit venv, GPU):
    python -m hackathon.wmloc.research.b2n_pyramid --method pixelseal --tag n_pyr_ps \
        --fold-budget 230 --fit-budget 290 --fit-seeds 2
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional

from hackathon.wmloc.constants import METHOD_PIXELSEAL, METHOD_TRUSTMARK
from hackathon.wmloc.research.b2_model import (
    DECODER_CHANNELS,
    DIHEDRAL_ELEMENTS,
    DINO_KEY,
    GRID,
    OUT_SIDE,
    PIXEL_KEY,
    RGB_CHANNELS,
    SOURCES_BY_METHOD,
    B2Net,
    HeadConfig,
    combined_loss,
    conv_block,
    count_parameters,
    dihedral,
)
from hackathon.wmloc.research.b2_train import (
    DECISION_THRESHOLD,
    GUARD_EVERY_STEPS,
    MethodArrays,
    TrainingError,
    _batch,
    breakdown,
    load_arrays,
    memory_guard,
    place_on_device,
    predict_probabilities,
    resolve_device,
    results_dir,
    save_atomically,
    write_text,
)
from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, miou_per_image

PYRAMID_CHANNELS = 48
MID_GRID = 64
"""Grid of the stride-8 level, between the trunk grid (32) and the evaluation grid (128)."""
STAGE_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    METHOD_PIXELSEAL: (96, 192, 384, 768),
    METHOD_TRUSTMARK: (256, 512, 1024, 2048),
}
"""Channels of the four backbone stages (strides 4, 8, 16, 32)."""
PIXEL_MAX = 255.0
PROBE_STEPS = 12
PROBE_WARMUP_STEPS = 3
"""First probe steps are not timed (cuDNN autotuning, memory pools)."""
MIN_STEPS = 40
BUDGET_SAFETY = 0.92
"""Share of the time budget turned into steps: the probe is a little optimistic."""


class ExtractorBackbone(nn.Module):
    """Trainable copy of an official extractor's backbone, returning its four stage maps.

    Both extractors normalise their input as `x * 2 - 1`. Batch-norm layers (ResNet-50) keep their
    pretrained statistics: batches of 16 images are too small to re-estimate them.
    """

    def __init__(self, method: str, network: nn.Module) -> None:
        super().__init__()
        if method not in STAGE_CHANNELS:
            raise TrainingError(f"no trainable backbone for {method!r}")
        self.method = method
        self.network = network

    def train(self, mode: bool = True) -> ExtractorBackbone:
        """Training mode for everything except the batch-norm layers, which stay frozen."""
        super().train(mode)
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        """uint8 [B, 3, 512, 512] -> stage maps at strides 4, 8, 16 and 32."""
        hidden = pixels.float() / PIXEL_MAX * 2.0 - 1.0
        network: Any = self.network
        stages: list[torch.Tensor] = []
        if self.method == METHOD_PIXELSEAL:
            for index in range(len(network.stages)):
                hidden = network.stages[index](network.downsample_layers[index](hidden))
                stages.append(hidden)
            return stages
        hidden = network.maxpool(network.relu(network.bn1(network.conv1(hidden))))
        for layer in ("layer1", "layer2", "layer3", "layer4"):
            hidden = getattr(network, layer)(hidden)
            stages.append(hidden)
        return stages


def load_backbone_template(method: str) -> ExtractorBackbone:
    """The pretrained backbone of `method` on the CPU, every weight trainable (kit venv only).

    Each fold and each seed starts from a deep copy of this template.

    Raises:
        TrainingError: the method has no torch backbone (StegaStamp is an ONNX graph).
    """
    from hackathon.wmloc.research.kit import load_wrapper

    if method not in STAGE_CHANNELS:
        raise TrainingError(f"no trainable backbone for {method!r}")
    wrapper = load_wrapper(method, "cpu")
    source = wrapper.det.convnext if method == METHOD_PIXELSEAL else wrapper.net
    network = copy.deepcopy(source).float()
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    return ExtractorBackbone(method, network)


class PyramidNet(B2Net):
    """The B2 head with a trainable full-resolution pyramid; see the module docstring."""

    def __init__(
        self, config: HeadConfig, backbone: nn.Module, stage_channels: tuple[int, int, int, int]
    ) -> None:
        super().__init__(config)
        self.backbone = backbone
        self.laterals = nn.ModuleList(
            [conv_block(channels, PYRAMID_CHANNELS, kernel=1) for channels in stage_channels]
        )
        self.to_trunk = nn.Conv2d(PYRAMID_CHANNELS, config.width, 1)
        self.decode_mid = conv_block(config.width + PYRAMID_CHANNELS, DECODER_CHANNELS, 3)
        rgb_in = RGB_CHANNELS if config.use_rgb else 0
        self.decode_fine = conv_block(
            DECODER_CHANNELS + PYRAMID_CHANNELS + rgb_in, DECODER_CHANNELS, 3
        )

    def head_parameters(self) -> list[nn.Parameter]:
        """Every parameter outside the backbone (trained at the full learning rate)."""
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def forward(
        self,
        method: str,
        sources: dict[str, torch.Tensor],
        scalars: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Logits [B, 128, 128]; `sources` must hold the uint8 image under `pixels`."""
        levels = [
            lateral(stage)
            for lateral, stage in zip(self.laterals, self.backbone(sources[PIXEL_KEY]), strict=True)
        ]
        fine, mid, coarse, coarsest = levels
        top = coarse + _resize(coarsest, coarse.shape[-1])
        parts = [
            adapter(sources[name])
            for name, adapter in self.adapters[method].items()  # type: ignore[operator]
        ]
        if self.dino_adapter is not None:
            parts.append(self.dino_adapter(sources[DINO_KEY]))
        features = self.fuse[method](torch.cat(parts, dim=1))
        features = features + self.scalar_proj[method](scalars.float())[:, :, None, None]
        features = features + self.to_trunk(_resize(top, GRID))
        for block in self.trunk:
            features = features + block(features)
        features = features + self.global_proj(features.mean(dim=(2, 3)))[:, :, None, None]
        middle = self.decode_mid(
            torch.cat([_resize(features, MID_GRID), _resize(mid, MID_GRID)], dim=1)
        )
        last = [_resize(middle, OUT_SIDE), _resize(fine, OUT_SIDE)]
        if self.rgb_stem is not None:
            last.append(self.rgb_stem(rgb))
        logits: torch.Tensor = self.head(self.refine(self.decode_fine(torch.cat(last, dim=1))))[
            :, 0
        ]
        return logits


def _resize(maps: torch.Tensor, side: int) -> torch.Tensor:
    """Bilinear resize of [B, C, h, w] to a square grid (identity when already there)."""
    if maps.shape[-1] == side and maps.shape[-2] == side:
        return maps
    return functional.interpolate(maps, size=(side, side), mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class PyramidTrainConfig:
    """Optimisation settings of one budgeted training run."""

    budget_seconds: float = 230.0
    max_steps: int = 20000
    batch_size: int = 16
    learning_rate: float = 2e-3
    backbone_lr_scale: float = 0.05
    """The pretrained backbone moves 20 times slower than the freshly initialised decoder."""
    weight_decay: float = 1e-2
    bce_weight: float = 0.5
    soft_weight: float = 1.0
    synthetic_per_real: float = 8.0
    finetune_fraction: float = 0.15
    """Last share of the steps run on real images only."""
    seed: int = DEFAULT_SEED


def planned_steps(rate: float, config: PyramidTrainConfig) -> int:
    """Steps that fit in the time budget at `rate` steps per second; `max_steps` always caps."""
    affordable = int(rate * config.budget_seconds * BUDGET_SAFETY)
    return min(config.max_steps, max(MIN_STEPS, affordable))


class BatchSampler:
    """Mini-batches of a fixed real / synthetic mix, each pool visited in shuffled passes."""

    def __init__(
        self, rows: NDArray[np.int64], n_real: int, config: PyramidTrainConfig, seed: int
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.real = rows[rows < n_real]
        self.synthetic = rows[rows >= n_real]
        if len(self.real) == 0:
            raise TrainingError("training needs at least one real image")
        self.batch_size = config.batch_size
        share = 1.0 / (1.0 + config.synthetic_per_real)
        self.real_per_batch = (
            self.batch_size
            if len(self.synthetic) == 0
            else min(self.batch_size, max(1, round(self.batch_size * share)))
        )
        self._queues: dict[str, NDArray[np.int64]] = {
            "real": np.zeros(0, np.int64),
            "synthetic": np.zeros(0, np.int64),
        }

    def _take(self, pool_name: str, count: int) -> NDArray[np.int64]:
        pool = self.real if pool_name == "real" else self.synthetic
        queue = self._queues[pool_name]
        while len(queue) < count:
            queue = np.concatenate([queue, self.rng.permutation(pool)])
        self._queues[pool_name] = queue[count:]
        return np.asarray(queue[:count], dtype=np.int64)

    def draw(self, real_only: bool) -> NDArray[np.int64]:
        """Row indices of the next batch (real rows only in the fine-tuning phase)."""
        n_real = self.batch_size if real_only else self.real_per_batch
        chosen = self._take("real", n_real)
        if n_real < self.batch_size:
            chosen = np.concatenate([chosen, self._take("synthetic", self.batch_size - n_real)])
        return chosen


@dataclass
class PyramidResult:
    """Outcome of one budgeted run: the model, the steps done and the measured rate."""

    model: PyramidNet
    steps: int
    steps_per_second: float
    losses: list[float]


def _training_step(
    model: PyramidNet,
    arrays: MethodArrays,
    chosen: NDArray[np.int64],
    element: int,
    config: PyramidTrainConfig,
    device: str,
    use_amp: bool,
) -> torch.Tensor:
    """Loss of one mini-batch under one symmetry of the square."""
    if arrays.truth is None:
        raise TrainingError("training needs ground-truth masks")
    sources, scalars, rgb = _batch(arrays, chosen, device)
    target = dihedral(torch.from_numpy(arrays.truth[chosen]).to(device), element)
    sources = {key: dihedral(value, element) for key, value in sources.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(arrays.method, sources, scalars, dihedral(rgb, element))
    # The IoU surrogates sum over 16 384 cells: computed in float32, outside autocast.
    return combined_loss(logits.float(), target, config.bce_weight, config.soft_weight, 0.0)


def train_pyramid(
    model: PyramidNet,
    arrays: MethodArrays,
    rows: NDArray[np.int64],
    config: PyramidTrainConfig,
    device: str,
) -> PyramidResult:
    """Train `model` on `rows` of `arrays` within the time budget.

    Args:
        model: A fresh model (pretrained backbone, random decoder) already on `device`.
        arrays: Inputs of one method (real rows first, synthetic rows after).
        rows: Training rows.
        config: Optimisation settings.
        device: Torch device.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    sampler = BatchSampler(rows, arrays.n_real, config, config.seed)
    use_amp = device.startswith("cuda")
    model.train()
    probe_started = time.time()
    for step in range(PROBE_STEPS):
        if step == PROBE_WARMUP_STEPS:
            _synchronise(device)
            probe_started = time.time()
        model.zero_grad()
        _training_step(model, arrays, sampler.draw(False), 0, config, device, use_amp).backward()  # type: ignore[no-untyped-call]
    _synchronise(device)
    rate = (PROBE_STEPS - PROBE_WARMUP_STEPS) / max(time.time() - probe_started, 1e-9)
    total_steps = planned_steps(rate, config)
    optimiser = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters())},
            {"params": model.head_parameters()},
        ],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=[config.learning_rate * config.backbone_lr_scale, config.learning_rate],
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    finetune_start = round(total_steps * (1.0 - config.finetune_fraction))
    losses: list[float] = []
    running, window = 0.0, max(1, total_steps // 20)
    started = time.time()
    for step in range(total_steps):
        element = int(rng.integers(DIHEDRAL_ELEMENTS))
        optimiser.zero_grad()
        chosen = sampler.draw(real_only=step >= finetune_start)
        loss = _training_step(model, arrays, chosen, element, config, device, use_amp)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimiser)
        scaler.update()
        schedule.step()
        running += float(loss.detach())
        if step % GUARD_EVERY_STEPS == 0:
            memory_guard(f"pyramid step {step}")
        if (step + 1) % window == 0:
            losses.append(running / window)
            running = 0.0
    _synchronise(device)
    elapsed = time.time() - started
    measured = total_steps / max(elapsed, 1e-9)
    print(
        f"  trained {total_steps} steps in {elapsed:.0f}s ({measured:.1f} steps/s, probe "
        f"{rate:.1f}, batch {config.batch_size}, device {device})",
        flush=True,
    )
    return PyramidResult(model, total_steps, measured, losses)


def _synchronise(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build_model(
    method: str, template: ExtractorBackbone, head: HeadConfig, device: str
) -> PyramidNet:
    """A fresh model: deep copy of the pretrained backbone, random decoder."""
    return PyramidNet(head, copy.deepcopy(template), STAGE_CHANNELS[method]).to(device)


class OutputSink:
    """Destination of the results: separate files, or ONE npz container (scratch inode quota).

    With `single` set, every npz / text output becomes keys `<file stem>/<array>` (text as a 0-d
    string array under `<file name>`) of that one container, rewritten atomically at each save, so
    a task leaves exactly one file whatever the number of folds and fits.
    """

    def __init__(self, single: Path | None) -> None:
        self.single = single
        self.items: dict[str, Any] = {}

    def save_npz(self, path: Path, arrays: dict[str, Any]) -> None:
        """Save `arrays` as `path`, or merge them into the single container."""
        if self.single is None:
            save_atomically(path, arrays)
            return
        prefix = path.name.removesuffix(".npz")
        self.items.update({f"{prefix}/{key}": value for key, value in arrays.items()})
        save_atomically(self.single, self.items)

    def save_text(self, path: Path, text: str) -> None:
        """Write `text` as `path`, or store it in the single container."""
        if self.single is None:
            write_text(path, text)
            return
        self.items[path.name] = np.array(text)
        save_atomically(self.single, self.items)


def _write_cv(
    out: Path,
    tag: str,
    method: str,
    arrays: MethodArrays,
    probabilities: dict[str, NDArray[np.float32]],
    done: NDArray[np.bool_],
    payload: dict[str, Any],
    sink: OutputSink,
) -> None:
    """Out-of-fold probabilities and report of the folds finished so far (atomic npz)."""
    if arrays.truth is None or arrays.is_full is None:
        raise TrainingError("cross-validation needs ground-truth masks")
    truth, regimes = arrays.truth[: arrays.n_real][done], arrays.is_full[: arrays.n_real][done]
    reports: dict[str, Any] = {}
    for key, values in probabilities.items():
        prediction = (values[done] >= DECISION_THRESHOLD).astype(np.uint8)
        reports[key] = breakdown(miou_per_image(prediction, truth), truth, regimes)
    main_report = dict(reports[method])
    main_report.update({f"tta_{k}": reports[f"{method}_tta"][k] for k in ("all", "full", "rect")})
    sink.save_text(
        out / f"{tag}.json",
        json.dumps(
            {**payload, "reports": {method: main_report}, "images_scored": int(done.sum())},
            indent=1,
        ),
    )
    sink.save_npz(
        out / f"{tag}.npz",
        {**{k: v.astype(np.float16) for k, v in probabilities.items()}, "done": done},
    )
    print(
        method,
        json.dumps({k: v for k, v in main_report.items() if not k.startswith("by_")}),
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    """Budgeted 5-fold CV, then fits on all train images with test-time symmetries."""
    started = time.time()
    method: str = args.method
    device = resolve_device(args.device)
    head = HeadConfig(
        sources={method: tuple(SOURCES_BY_METHOD[method])}, dropout=args.dropout, width=args.width
    )
    base = PyramidTrainConfig(
        budget_seconds=args.fold_budget,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        synthetic_per_real=args.synthetic_per_real,
        finetune_fraction=args.finetune_fraction,
        seed=args.seed,
    )
    arrays = load_arrays("train", method, synthetic=True, with_pixels=True)
    test = load_arrays("test", method, with_pixels=True)
    if arrays.folds is None:
        raise TrainingError("the train split has no folds")
    print(
        f"loaded {len(arrays.folds)} rows ({arrays.n_real} real) in {time.time() - started:.0f}s",
        flush=True,
    )
    moved = place_on_device({method: arrays}, device)
    print(f"  {moved:.1f} GB of feature maps placed on {device}", flush=True)
    template = load_backbone_template(method)
    sink = OutputSink(args.single_output)
    out = results_dir()
    if sink.single is None:
        out.mkdir(parents=True, exist_ok=True)
    real = np.arange(len(arrays.folds)) < arrays.n_real
    probabilities = {
        key: np.zeros((arrays.n_real, OUT_SIDE, OUT_SIDE), np.float32)
        for key in (method, f"{method}_tta")
    }
    done = np.zeros(arrays.n_real, dtype=np.bool_)
    folds_log: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "config": {**asdict(base), "use_synthetic": True, "epochs": None, "lovasz_weight": 0.0},
        "head": {"dropout": args.dropout, "dino": True, "rgb": True, "pyramid": True},
        "curves": folds_log,
    }
    for fold in range(min(args.folds, DEFAULT_FOLDS)):
        fold_started = time.time()
        held = np.flatnonzero((arrays.folds == fold) & real)
        config = PyramidTrainConfig(**{**asdict(base), "seed": args.seed + fold})
        model = build_model(method, template, head, device)
        result = train_pyramid(model, arrays, np.flatnonzero(arrays.folds != fold), config, device)
        for key, tta in ((method, False), (f"{method}_tta", True)):
            probabilities[key][held] = predict_probabilities(model, arrays, held, device, tta=tta)
        done[held] = True
        folds_log.append(
            {
                "fold": fold,
                "steps": result.steps,
                "steps_per_second": result.steps_per_second,
                "losses": result.losses,
                "seconds": time.time() - fold_started,
                "parameters": count_parameters(model),
            }
        )
        print(f"fold {fold}: {time.time() - fold_started:.0f}s, {result.steps} steps", flush=True)
        _write_cv(out, args.tag, method, arrays, probabilities, done, payload, sink)
        del model, result
    everything = np.arange(len(arrays.folds), dtype=np.int64)
    # `--test-limit` only serves smoke runs on a laptop: a real fit predicts every test image.
    n_test = len(test.names) if args.test_limit is None else min(args.test_limit, len(test.names))
    test_rows = np.arange(n_test, dtype=np.int64)
    for k in range(args.fit_seeds):
        if time.time() - started + args.fit_budget > args.deadline:
            print(f"fit seed {k} skipped: deadline", flush=True)
            break
        config = PyramidTrainConfig(
            **{**asdict(base), "seed": args.seed + 100 + k, "budget_seconds": args.fit_budget}
        )
        model = build_model(method, template, head, device)
        result = train_pyramid(model, arrays, everything, config, device)
        fitted: dict[str, Any] = {
            method: predict_probabilities(model, test, test_rows, device, tta=True).astype(
                np.float16
            ),
            f"names_{method}": np.array(test.names[:n_test]),
        }
        tag = f"fit_{args.tag}_s{k}"
        sink.save_text(
            out / f"{tag}_test.json",
            json.dumps(
                {"config": asdict(config), "steps": result.steps, "methods": [method]}, indent=1
            ),
        )
        # The npz goes last and atomically: its existence means "this fit is complete".
        sink.save_npz(out / f"{tag}_test.npz", fitted)
        print(
            f"fit seed {k}: {result.steps} steps, {time.time() - started:.0f}s since start",
            flush=True,
        )
        del model, result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(STAGE_CHANNELS), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--fold-budget", type=float, default=PyramidTrainConfig.budget_seconds)
    parser.add_argument("--fit-budget", type=float, default=290.0)
    parser.add_argument("--fit-seeds", type=int, default=2)
    parser.add_argument(
        "--deadline",
        type=float,
        default=2400.0,
        help="seconds since start after which no fit begins",
    )
    parser.add_argument("--max-steps", type=int, default=PyramidTrainConfig.max_steps)
    parser.add_argument("--batch-size", type=int, default=PyramidTrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=PyramidTrainConfig.learning_rate)
    parser.add_argument(
        "--backbone-lr-scale", type=float, default=PyramidTrainConfig.backbone_lr_scale
    )
    parser.add_argument(
        "--synthetic-per-real", type=float, default=PyramidTrainConfig.synthetic_per_real
    )
    parser.add_argument(
        "--finetune-fraction", type=float, default=PyramidTrainConfig.finetune_fraction
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=HeadConfig.width)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--single-output",
        type=Path,
        help="write every result into this ONE npz container instead of the results folder",
    )
    parser.add_argument(
        "--test-limit", type=int, help="smoke runs: predict only the first test images"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
