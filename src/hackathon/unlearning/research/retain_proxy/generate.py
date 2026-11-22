"""GPU worker: generate the retain-proxy images and classify them with the original model.

One process per GPU (`CUDA_VISIBLE_DEVICES`), each on its own shard of the plan. Output: one
uncompressed `.npz` per chunk (images, the original's logits, plan indices), so the shared
filesystem sees a few dozen files, never one per image.

Generator: SDXL-Turbo (Stability AI Non-Commercial Research Community License), 512x512, a few
steps, no classifier-free guidance, resized to the 224x224 of the task with a bicubic filter.
Weights come from a local directory (compute nodes have no Internet).
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.model import load_model, predict
from hackathon.unlearning.research.retain_proxy.gpu_memory import (
    DEFAULT_MARGIN_GB,
    DEFAULT_TARGET_GB,
    BatchSizer,
    MemoryProbe,
    device_budget_gb,
    make_probe,
    process_adaptively,
)
from hackathon.unlearning.research.retain_proxy.plan import (
    PromptEntry,
    read_plan,
    select_shard,
)

DEFAULT_CHUNK_SIZE = 256
DEFAULT_BATCH_SIZE = 48
"""First batch size; `BatchSizer` then walks it towards the memory target (>= 30 GB per A100)."""
DEFAULT_MAX_BATCH_SIZE = 192
DEFAULT_STEPS = 4
"""SDXL-Turbo is distilled for 1 to 4 steps; 4 gives the sharpest objects."""
DEFAULT_RENDER_SIZE = 512
"""SDXL-Turbo native resolution (it degrades above it)."""
GUIDANCE_SCALE = 0.0
"""Turbo models are trained without classifier-free guidance."""
CHUNK_FILE_TEMPLATE = "chunk_{first_index:07d}.npz"
PREVIEW_COLUMNS = 8
PREVIEW_ROWS = 4
PREVIEW_TILE = 112
DEADLINE_SAFETY_FACTOR = 1.25
"""A new chunk starts only if `factor x` the last chunk duration still fits before the deadline."""
GENERATION_SEED_OFFSET = 0


class GenerationError(HackathonError):
    """Generation could not run (missing weights, GPU, plan) or produced unusable output."""


class WorkerConfig(BaseModel):
    """Everything one worker needs; immutable and printable for provenance."""

    model_config = ConfigDict(frozen=True)

    plan_path: Path
    model_dir: Path = Field(description="Local diffusers folder of SDXL-Turbo (fp16 variant).")
    original_path: Path = Field(
        description="model.pt of the task (the classifier to read through)."
    )
    out_dir: Path
    shard: int = Field(ge=0)
    num_shards: int = Field(ge=1)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)
    max_batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, ge=1)
    target_memory_gb: float = Field(default=DEFAULT_TARGET_GB, gt=0)
    memory_margin_gb: float = Field(default=DEFAULT_MARGIN_GB, ge=0)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, ge=1)
    steps: int = Field(default=DEFAULT_STEPS, ge=1)
    render_size: int = Field(default=DEFAULT_RENDER_SIZE, ge=64)
    deadline_epoch: float | None = Field(default=None)
    device: str = "cuda"


class WorkerReport(BaseModel):
    """What a worker did, for the job log and the report."""

    model_config = ConfigDict(frozen=True)

    shard: int
    chunks_written: int
    chunks_skipped: int
    images_written: int
    seconds: float
    stopped_by_deadline: bool
    final_batch_size: int
    oom_events: int
    peak_memory_gb: float | None


def chunk_path(out_dir: Path, first_index: int) -> Path:
    """Where the chunk whose first plan index is `first_index` lives."""
    return out_dir / CHUNK_FILE_TEMPLATE.format(first_index=first_index)


def split_chunks(entries: Sequence[PromptEntry], chunk_size: int) -> list[list[PromptEntry]]:
    """Consecutive groups of `chunk_size` entries (the last one may be shorter)."""
    if chunk_size < 1:
        raise GenerationError("chunk_size must be at least 1.")
    return [
        list(entries[start : start + chunk_size]) for start in range(0, len(entries), chunk_size)
    ]


def to_dataset_resolution(image: Image.Image) -> np.ndarray:
    """PIL image -> uint8 [224, 224, 3] (RGB, bicubic, antialiased on downscale)."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def save_chunk(path: Path, indices: np.ndarray, images: np.ndarray, logits: np.ndarray) -> None:
    """Write one chunk atomically (temp file then rename), validating shapes first.

    Raises:
        GenerationError: inconsistent lengths, wrong image shape or non-finite logits.
    """
    count = len(indices)
    if images.shape != (count, IMAGE_SIZE, IMAGE_SIZE, 3) or images.dtype != np.uint8:
        raise GenerationError(f"Bad image block {images.shape} {images.dtype} for {count} indices.")
    if logits.shape != (count, NUM_CLASSES) or not np.isfinite(logits).all():
        raise GenerationError(f"Bad logits block {logits.shape} (non-finite or wrong shape).")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, indices=indices.astype(np.int64), images=images, logits=logits)
    temporary.replace(path)


def write_preview(path: Path, images: np.ndarray) -> None:
    """Contact sheet of the first images of a chunk (a quick visual check of the generator)."""
    count = PREVIEW_COLUMNS * PREVIEW_ROWS
    tiles = [
        np.asarray(Image.fromarray(image).resize((PREVIEW_TILE, PREVIEW_TILE)))
        for image in images[:count]
    ]
    blank = np.zeros((PREVIEW_TILE, PREVIEW_TILE, 3), dtype=np.uint8)
    tiles += [blank] * (count - len(tiles))
    rows = [
        np.concatenate(tiles[row * PREVIEW_COLUMNS : (row + 1) * PREVIEW_COLUMNS], axis=1)
        for row in range(PREVIEW_ROWS)
    ]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def load_generator_pipeline(model_dir: Path, device: str) -> Any:
    """Load SDXL-Turbo from a local folder in fp16 (offline; imports diffusers lazily)."""
    if not (model_dir / "model_index.json").is_file():
        raise GenerationError(f"No diffusers model at {model_dir} (model_index.json missing).")
    from diffusers import StableDiffusionXLPipeline  # heavy import, only on the GPU job

    # diffusers ships `from_pretrained` without annotations: nothing to type-check on our side.
    pipeline = StableDiffusionXLPipeline.from_pretrained(  # type: ignore[no-untyped-call]
        str(model_dir), torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.set_progress_bar_config(disable=True)
    # Decode the VAE one image at a time: a 128-image batch at 512x512 would exceed the 2^31
    # elements that cuDNN convolutions index with 32 bits, and slicing barely costs anything.
    pipeline.vae.enable_slicing()
    return pipeline.to(device)


def render_batch(pipeline: Any, batch: Sequence[PromptEntry], config: WorkerConfig) -> np.ndarray:
    """One pipeline call: uint8 [len(batch), 224, 224, 3], every image seeded by its plan entry."""
    generators = [
        torch.Generator(device=config.device).manual_seed(entry.seed + GENERATION_SEED_OFFSET)
        for entry in batch
    ]
    output = pipeline(
        prompt=[entry.prompt for entry in batch],
        num_inference_steps=config.steps,
        guidance_scale=GUIDANCE_SCALE,
        height=config.render_size,
        width=config.render_size,
        generator=generators,
    )
    return np.stack([to_dataset_resolution(image) for image in output.images])


def generate_images(
    pipeline: Any,
    entries: Sequence[PromptEntry],
    config: WorkerConfig,
    sizer: BatchSizer,
    probe: MemoryProbe,
) -> np.ndarray:
    """Images of `entries` as uint8 [n, 224, 224, 3]; the batch size adapts to the GPU memory."""
    blocks = process_adaptively(
        entries, sizer, lambda batch: render_batch(pipeline, batch, config), probe
    )
    return np.concatenate(blocks) if blocks else np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)


def run_worker(config: WorkerConfig) -> WorkerReport:
    """Generate and classify every chunk of this worker's shard that is not yet on disk."""
    entries = select_shard(read_plan(config.plan_path), config.shard, config.num_shards)
    chunks = split_chunks(entries, config.chunk_size)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pipeline = load_generator_pipeline(config.model_dir, config.device)
    classifier = load_model(config.original_path, config.device)
    probe = make_probe(config.device)
    cuda = torch.device(config.device).type == "cuda"
    weights_gb = torch.cuda.memory_allocated() / 1e9 if cuda else 0.0
    cap_gb = device_budget_gb(config.device, config.memory_margin_gb) if cuda else 1e9
    sizer = BatchSizer(
        size=min(config.batch_size, config.max_batch_size),
        maximum=config.max_batch_size,
        target_gb=config.target_memory_gb,
        cap_gb=cap_gb,
        weights_gb=weights_gb,
    )
    print(
        f"[shard {config.shard}] models loaded in {time.time() - started:.0f} s "
        f"(weights {weights_gb:.1f} GB, cap {cap_gb:.1f} GB, target {sizer.target_gb:.1f} GB), "
        f"{len(entries)} images in {len(chunks)} chunks",
        flush=True,
    )

    written = skipped = images_done = 0
    peak_gb: float | None = None
    last_duration = 0.0
    stopped = False
    for chunk in chunks:
        target = chunk_path(config.out_dir, chunk[0].index)
        if target.is_file():
            skipped += 1
            continue
        if (
            config.deadline_epoch is not None
            and time.time() + DEADLINE_SAFETY_FACTOR * last_duration > config.deadline_epoch
        ):
            stopped = True
            break
        chunk_started = time.time()
        images = generate_images(pipeline, chunk, config, sizer, probe)
        logits = predict(classifier, torch.from_numpy(images)).numpy()
        indices = np.array([entry.index for entry in chunk], dtype=np.int64)
        save_chunk(target, indices, images, logits)
        if written == 0:
            write_preview(config.out_dir / f"preview_shard{config.shard}.png", images)
        last_duration = time.time() - chunk_started
        written += 1
        images_done += len(chunk)
        peak_gb = max((peak for _, peak in sizer.history), default=None)
        print(
            f"[shard {config.shard}] chunk {target.name} {len(chunk)} images in "
            f"{last_duration:.1f} s ({len(chunk) / last_duration:.1f} img/s), batch {sizer.size}, "
            f"peak {'n/a' if peak_gb is None else f'{peak_gb:.1f} GB'}, oom {sizer.oom_events}",
            flush=True,
        )
    return WorkerReport(
        shard=config.shard,
        chunks_written=written,
        chunks_skipped=skipped,
        images_written=images_done,
        seconds=time.time() - started,
        stopped_by_deadline=stopped,
        final_batch_size=sizer.size,
        oom_events=sizer.oom_events,
        peak_memory_gb=peak_gb,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI of one worker (one GPU)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument("--target-memory-gb", type=float, default=DEFAULT_TARGET_GB)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--deadline-epoch", type=float, default=None)
    args = parser.parse_args(argv)
    config = WorkerConfig(
        plan_path=args.plan,
        model_dir=args.model_dir,
        original_path=args.original,
        out_dir=args.out_dir,
        shard=args.shard,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        target_memory_gb=args.target_memory_gb,
        chunk_size=args.chunk_size,
        steps=args.steps,
        deadline_epoch=args.deadline_epoch,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if config.device != "cuda" and os.environ.get("RETAIN_PROXY_ALLOW_CPU") != "1":
        raise GenerationError("CUDA is not available (set RETAIN_PROXY_ALLOW_CPU=1 to override).")
    report = run_worker(config)
    print(report.model_dump_json(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
