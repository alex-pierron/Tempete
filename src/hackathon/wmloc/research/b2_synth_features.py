"""Synthetic training samples with their frozen-extractor features cached (team B2).

The learned head peaks at 20 epochs and then overfits: only 136 real images per fold. More data
cannot come from the 510 real images, but it can be made: clean images are watermarked by the
official embedder and edited with REAL mask shapes (`b2_synth`), then the very same frozen
extractor, DINOv2 backbone and thumbnail pipeline used on real images produce the head's inputs.

Each shard `<cache>/synth/<method>/shard_<i>.npz` holds N samples: every feature array of
`b2_features` stacked on axis 0, plus `dino` [N, 384, 32, 32], `rgb` uint8 [N, 3, 128, 128],
`truth` uint8 [N, 128, 128], `is_full`, and `fold`, the cross-validation fold of the REAL image
the mask shape came from. A training fold drops the synthetic samples of its own fold, so no shape
of a held-out image ever reaches training.

Usage (kit venv, GPU advised):
    python -m hackathon.wmloc.research.b2_synth_features --method pixelseal --shards 4 --shard-size 128
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.nn import functional

from hackathon.wmloc.constants import METHOD_PIXELSEAL, METHOD_STEGASTAMP, METHOD_TRUSTMARK
from hackathon.wmloc.research.b2_dino import DinoEncoder
from hackathon.wmloc.research.b2_features import (
    accelerator,
    build_probe,
    cache_root,
    extraction_batch_size,
)
from hackathon.wmloc.research.b2_synth import (
    IMAGE,
    EmbedFn,
    ShapeLibrary,
    build_library,
    sample_pair,
)
from hackathon.wmloc.research.data import data_root, load_records, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, stratified_folds
from hackathon.wmloc.research.kit import N_BITS_BY_METHOD

CLEAN_DIR_ENV = "WMLOC_CLEAN_IMAGES"
CLEAN_DIR_DEFAULT = "coco_val_subset"
"""Folder of clean photographs, next to the `hackathon` data directory by default."""
THUMB_POOL = 4
FEATURE_BATCH = 8
CUDA_FEATURE_BATCH = 32
THREAD_LIMIT = 2
TRUSTMARK_MAX_ASPECT = 2.0
"""TrustMark only watermarks the whole patch when its aspect ratio is at most 2 (kit docstring)."""
SYNTH_SUBDIR = "synth"
WRITE_ATTEMPTS = 20
WRITE_RETRY_SECONDS = 15.0
MAX_SYNTHETIC_ENV = "WMLOC_MAX_SYNTHETIC"
MAX_SYNTHETIC_DEFAULT = 6144
"""Samples per method kept in memory (about 18 GB with images): 4 processes share 512 GB."""


class SynthFeatureError(ResearchError):
    """No clean image is available, or a shard is missing."""


def synth_dir(method: str) -> Path:
    """Folder of the synthetic shards of `method`."""
    return cache_root().parent / SYNTH_SUBDIR / method


def clean_pool() -> list[NDArray[np.uint8]]:
    """The clean photographs as uint8 arrays (short side 512), from ONE container when it exists.

    Falls back to the folder of photographs (`$WMLOC_CLEAN_IMAGES` or `<data>/../coco_val_subset`),
    resized on the fly, when `clean_images.npz` has not been packed (see `b2_pack`).

    Raises:
        SynthFeatureError: no photograph anywhere.
    """
    from hackathon.wmloc.research.b2_pack import clean_path

    container = clean_path()
    if container.is_file():
        with np.load(container) as archive:
            return [np.asarray(archive[key], dtype=np.uint8) for key in archive.files]
    configured = os.environ.get(CLEAN_DIR_ENV)
    folder = Path(configured) if configured else data_root().parent.parent / CLEAN_DIR_DEFAULT
    paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SynthFeatureError(f"no clean image in {folder} and no container {container}")
    return [_resize_short_side(path) for path in paths]


def _resize_short_side(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as handle:
        picture = handle.convert("RGB")
    width, height = picture.size
    scale = IMAGE / min(width, height)
    size = (max(IMAGE, round(width * scale)), max(IMAGE, round(height * scale)))
    return np.asarray(picture.resize(size, Image.Resampling.BICUBIC), dtype=np.uint8)


def clean_crop(picture: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """A random 512 x 512 crop (random flip) of a photograph whose short side is 512."""
    height, width = picture.shape[:2]
    left = int(rng.integers(0, width - IMAGE + 1))
    top = int(rng.integers(0, height - IMAGE + 1))
    array = picture[top : top + IMAGE, left : left + IMAGE]
    return (
        np.ascontiguousarray(array[:, ::-1]) if rng.random() < 0.5 else np.ascontiguousarray(array)
    )


def real_library() -> ShapeLibrary:
    """Shape library of all 510 real train masks, tagged with each image's own CV fold."""
    records = load_records("train")
    masks = load_truth([r.name for r in records])
    is_full = np.array([bool(r.full_watermark) for r in records])
    folds = np.zeros(len(records), dtype=np.int64)
    for method in sorted({r.method for r in records}):
        indices = [i for i, r in enumerate(records) if r.method == method]
        strata = [f"{method}/{is_full[i]}" for i in indices]
        folds[indices] = stratified_folds(strata, DEFAULT_FOLDS, DEFAULT_SEED)
    return build_library(masks, is_full, folds)


def _thumbnail(image: torch.Tensor) -> NDArray[np.uint8]:
    pooled = functional.avg_pool2d(image[None], THUMB_POOL)[0]
    return np.asarray((pooled * 255.0).round().clamp(0, 255).to(torch.uint8).numpy())


class ShardGenerator:
    """Everything a worker needs to generate shards, built ONCE per process.

    Loading the embedder, the extractor, DINOv2 and the photograph pool costs tens of seconds: a
    worker that writes a dozen shards must not pay it a dozen times.
    """

    def __init__(self, method: str) -> None:
        from hackathon.wmloc.research.kit import load_wrapper, wmloc_dir  # kit venv only

        torch.set_num_threads(THREAD_LIMIT)
        self.method = method
        device = accelerator()
        wrapper = load_wrapper(method, device)
        self.probe: Any
        if method == METHOD_STEGASTAMP:
            # No torch module to tap: the decoder is an ONNX graph run on the CPU (b2_stegastamp).
            from hackathon.wmloc.research.b2_stegastamp import StegaStampProbe

            self.probe = StegaStampProbe(
                wmloc_dir() / "models" / "stegastamp" / "stega_stamp_logits.onnx"
            )
        else:
            self.probe = build_probe(method, wrapper, device)
        self.encoder = DinoEncoder()
        self.library = real_library()
        self.sources = clean_pool()
        self.n_bits = N_BITS_BY_METHOD[method]
        self.embed: EmbedFn = wrapper.embed
        self.max_aspect = TRUSTMARK_MAX_ASPECT if method == "trustmark" else None
        self.batch_size = extraction_batch_size(CUDA_FEATURE_BATCH, FEATURE_BATCH)

    def generate(self, index: int, size: int, seed: int) -> Path:
        """Generate, featurise and save shard `index` (an existing shard is kept)."""
        path = synth_dir(self.method) / f"shard_{index:04d}.npz"
        if path.exists():
            return path
        rng = np.random.default_rng([seed, index])
        rows: list[dict[str, Any]] = []
        started = time.time()
        for start in range(0, size, self.batch_size):
            rows.extend(self._batch(rng, min(self.batch_size, size - start)))
            print(
                f"  shard {index}: {len(rows)}/{size} ({len(rows) / (time.time() - started):.2f}/s)",
                flush=True,
            )
        stacked: dict[str, Any] = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
        # The shard stays in memory until the write succeeds (inode quota of the shared scratch).
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                np.savez(temporary, **stacked)
                os.replace(temporary, path)
                break
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                if attempt == WRITE_ATTEMPTS:
                    raise
                print(f"  shard {index}: write failed ({exc}); attempt {attempt}", flush=True)
                time.sleep(WRITE_RETRY_SECONDS)
        return path

    def _batch(self, rng: np.random.Generator, count: int) -> list[dict[str, Any]]:
        images, messages, truths, fulls, folds = [], [], [], [], []
        for _ in range(count):
            bits = rng.integers(0, 2, self.n_bits).astype(np.uint8)
            clean = clean_crop(self.sources[int(rng.integers(len(self.sources)))], rng)
            image, label, is_full, fold = sample_pair(
                clean, bits, self.library, self.embed, rng, max_aspect=self.max_aspect
            )
            images.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)
            messages.append(bits.astype(np.float32))
            truths.append(label)
            fulls.append(is_full)
            folds.append(fold)
        batch = torch.stack(images)
        features = self.probe.extract(batch, np.stack(messages))
        dino = self.encoder.encode(batch).numpy().astype(np.float16)
        rows: list[dict[str, Any]] = []
        for k, row in enumerate(features):
            row["dino"] = dino[k]
            row["rgb"] = _thumbnail(images[k])
            # The full-resolution pixels let a later head learn from the image itself.
            row["image"] = (images[k] * 255.0).round().to(torch.uint8).numpy()
            row["truth"] = truths[k]
            row["is_full"] = np.array(fulls[k])
            row["fold"] = np.array(folds[k], dtype=np.int64)
            rows.append(row)
        return rows


def generate_shard(method: str, index: int, size: int, seed: int) -> Path:
    """One shard with a throw-away generator (use `ShardGenerator` for several shards)."""
    path = synth_dir(method) / f"shard_{index:04d}.npz"
    if path.exists():
        return path
    return ShardGenerator(method).generate(index, size, seed)


def select_shards(shards: list[Path], per_shard: int, max_samples: int) -> list[Path]:
    """At most `max_samples` worth of shards, spread evenly over the sorted list.

    Spreading (instead of taking the first ones) lets shards generated later by a filler task
    enter the training pool while the memory footprint stays bounded.
    """
    keep = max(1, max_samples // max(per_shard, 1))
    if len(shards) <= keep:
        return shards
    positions = np.unique(np.linspace(0, len(shards) - 1, keep).round().astype(int))
    return [shards[i] for i in positions]


def load_synthetic(
    method: str, keys: set[str] | None = None, max_samples: int | None = None
) -> dict[str, NDArray[Any]]:
    """The shards of `method` concatenated along the sample axis.

    Args:
        method: Method name.
        keys: Arrays to read (default: all). The full-resolution `image` array is five times the
            size of everything else, so callers that do not need it should not load it.
        max_samples: Cap on the number of samples loaded (default `$WMLOC_MAX_SYNTHETIC` or
            `MAX_SYNTHETIC_DEFAULT`): four training processes share one node's RAM.

    Raises:
        SynthFeatureError: no shard exists.
    """
    # A killed writer leaves `shard_i.npz.<pid>.tmp.npz` behind: never read those.
    shards = sorted(p for p in synth_dir(method).glob("shard_*.npz") if ".tmp." not in p.name)
    if not shards:
        raise SynthFeatureError(f"no synthetic shard for {method} in {synth_dir(method)}")
    cap = max_samples or int(os.environ.get(MAX_SYNTHETIC_ENV, MAX_SYNTHETIC_DEFAULT))
    with np.load(shards[0]) as first:
        per_shard = int(first["truth"].shape[0])
    shards = select_shards(shards, per_shard, cap)
    parts: dict[str, list[NDArray[Any]]] = {}
    for shard in shards:
        with np.load(shard) as archive:
            for key in archive.files:
                if keys is None or key in keys:
                    parts.setdefault(key, []).append(archive[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def selected_shards(method: str, max_samples: int | None = None) -> list[Path]:
    """The shards a training run reads: finished ones only, capped and spread (`select_shards`).

    Raises:
        SynthFeatureError: no shard exists.
    """
    # A killed writer leaves `shard_i.npz.<pid>.tmp.npz` behind: never read those.
    shards = sorted(p for p in synth_dir(method).glob("shard_*.npz") if ".tmp." not in p.name)
    if not shards:
        raise SynthFeatureError(f"no synthetic shard for {method} in {synth_dir(method)}")
    cap = max_samples or int(os.environ.get(MAX_SYNTHETIC_ENV, MAX_SYNTHETIC_DEFAULT))
    with np.load(shards[0]) as first:
        per_shard = int(first["truth"].shape[0])
    return select_shards(shards, per_shard, cap)


def append_synthetic(
    method: str,
    real: dict[str, NDArray[Any]],
    rename: dict[str, str] | None = None,
    max_samples: int | None = None,
    after_shard: Callable[[int], None] | None = None,
) -> dict[str, NDArray[Any]]:
    """`real` arrays followed by the synthetic rows of the same keys, allocated ONCE.

    `load_synthetic` then a concatenation with the real rows holds three copies of the synthetic
    set at its peak (shard list, concatenation, joined array); eight such loaders were killed by
    the kernel on the GPU node. Here every output array is allocated at its final size and filled
    shard by shard, so the peak is one copy plus one shard.

    Args:
        method: Method name.
        real: Real rows by key (each [n_real, ...]); the result keeps their dtype and row shape.
        rename: Key of `real` -> name of the array inside a shard, when they differ.
        max_samples: Cap on the synthetic samples (default `$WMLOC_MAX_SYNTHETIC`).
        after_shard: Called with the number of shards read so far (memory guard of the caller).

    Raises:
        SynthFeatureError: no shard exists, or a shard lacks one of the keys.
    """
    shards = selected_shards(method, max_samples)
    names = {key: (rename or {}).get(key, key) for key in real}
    counts = []
    for shard in shards:
        with np.load(shard) as archive:
            missing = [name for name in names.values() if name not in archive.files]
            if missing:
                raise SynthFeatureError(f"{shard} lacks the arrays {missing}")
            counts.append(int(archive["fold"].shape[0]))
    total = int(sum(counts))
    joined: dict[str, NDArray[Any]] = {}
    for key, rows in real.items():
        joined[key] = np.empty((len(rows) + total, *rows.shape[1:]), dtype=rows.dtype)
        joined[key][: len(rows)] = rows
    offsets = np.cumsum([0, *counts])
    for done, (shard, offset, count) in enumerate(zip(shards, offsets, counts, strict=False), 1):
        with np.load(shard) as archive:
            for key, name in names.items():
                start = len(real[key]) + int(offset)
                joined[key][start : start + count] = archive[name]
        if after_shard is not None:
            after_shard(done)
    return joined


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic samples with features.")
    parser.add_argument(
        "--method",
        choices=(METHOD_PIXELSEAL, METHOD_TRUSTMARK, METHOD_STEGASTAMP),
        required=True,
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    wanted = range(args.first_shard, args.first_shard + args.shards)
    if all((synth_dir(args.method) / f"shard_{i:04d}.npz").exists() for i in wanted):
        return
    generator = ShardGenerator(args.method)
    for index in wanted:
        print(generator.generate(index, args.shard_size, args.seed), flush=True)


if __name__ == "__main__":
    main()
