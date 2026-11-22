"""Multi-scale sliding-window decoding with the KNOWN message: the strongest local evidence (team B2).

The extractors read a whole image; asked about a CROP they answer "does this crop still carry the
message?" with a bit agreement of 1.0 when it lies inside the watermarked zone and 0.5 (chance) when
it lies in the restored hole or outside the marked rectangle. Decoding square windows of several
sizes on a regular grid therefore yields, for each size, a map of how well each window position
reads the message. A window straddling a border is partly clean, so its agreement degrades smoothly:
the stack of maps over sizes localises both the hole and the rectangle, and a learned head can
combine them with the image content to find the boundaries.

One `.npz` per (split, method, image) under `<cache>/win/<split>/<method>/<name>.npz` with, for each
window size `S`, `w{S}_m` (mean signed margin) and `w{S}_a` (bit agreement), float16 [n, n] where
n = (512 - S) / STRIDE + 1. `window_maps` resamples them onto the 32 x 32 trunk grid.

Usage (kit venv, GPU strongly advised: about 1000 decodes per image):
    python -m hackathon.wmloc.research.b2_windows --split train --method pixelseal
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from hackathon.wmloc.constants import IMAGE_SIZE, METHOD_PIXELSEAL, METHOD_TRUSTMARK
from hackathon.wmloc.research.b2_features import accelerator, cache_root, message_signs
from hackathon.wmloc.research.data import SPLITS, Record, load_image, load_records
from hackathon.wmloc.research.errors import ResearchError

WINDOW_SIZES: tuple[int, ...] = (64, 96, 128, 160, 192, 256, 320)
WINDOW_STRIDE = 32
"""Pixels between two window origins; every size in `WINDOW_SIZES` fits the 512 image exactly."""
DECODE_CHUNK = 256
TRUNK_GRID = 32
CELL_PX = IMAGE_SIZE // TRUNK_GRID
CHANNELS_PER_SIZE = 2
THREAD_LIMIT = 2
PROGRESS_EVERY = 10


class WindowError(ResearchError):
    """Window cache missing or a window size does not tile the image."""


def window_path(split: str, method: str, name: str) -> Path:
    """Cache file of one (split, method, image)."""
    return cache_root().parent / "win" / split / method / f"{name}.npz"


def grid_side(size: int) -> int:
    """Windows per axis for a window of `size` px.

    Raises:
        WindowError: `size` does not tile the image at `WINDOW_STRIDE`.
    """
    if size > IMAGE_SIZE or (IMAGE_SIZE - size) % WINDOW_STRIDE:
        raise WindowError(f"window {size} does not tile {IMAGE_SIZE} at stride {WINDOW_STRIDE}")
    return (IMAGE_SIZE - size) // WINDOW_STRIDE + 1


@torch.no_grad()
def decode_windows(wrapper: Any, image: torch.Tensor, signs: torch.Tensor) -> dict[str, Any]:
    """Margin and agreement maps of every window size for one image.

    Args:
        wrapper: Official kit wrapper (`logits` resizes crops to the decoder resolution).
        image: [3, 512, 512] float in [0, 1], already on the wrapper's device.
        signs: [n_bits] message signs on the same device.
    """
    arrays: dict[str, Any] = {}
    for size in WINDOW_SIZES:
        side = grid_side(size)
        crops = image.unfold(1, size, WINDOW_STRIDE).unfold(2, size, WINDOW_STRIDE)
        crops = crops.permute(1, 2, 0, 3, 4).reshape(side * side, 3, size, size)
        logits = torch.cat(
            [
                wrapper.logits(crops[start : start + DECODE_CHUNK]).float()
                for start in range(0, len(crops), DECODE_CHUNK)
            ]
        )
        signed = logits * signs[None, :]
        arrays[f"w{size}_m"] = (
            signed.mean(dim=1).reshape(side, side).cpu().numpy().astype(np.float16)
        )
        arrays[f"w{size}_a"] = (
            (signed > 0).float().mean(dim=1).reshape(side, side).cpu().numpy().astype(np.float16)
        )
    return arrays


def _resample_axis(side: int, size: int) -> NDArray[np.float64]:
    """Fractional window index seen by each of the 32 trunk cells along one axis."""
    centres = np.arange(TRUNK_GRID) * CELL_PX + CELL_PX / 2.0
    return np.clip((centres - size / 2.0) / WINDOW_STRIDE, 0.0, side - 1.0)


def window_maps(cached: dict[str, NDArray[np.float16]]) -> NDArray[np.float32]:
    """Stack the cached maps on the trunk grid: [len(WINDOW_SIZES) * 2, 32, 32] float32.

    Each window value is attributed to the cell under its centre and bilinearly interpolated
    between window centres; cells nearer the border than half a window take the nearest value.
    """
    channels = []
    for size in WINDOW_SIZES:
        side = grid_side(size)
        axis = _resample_axis(side, size)
        rows, cols = np.meshgrid(axis, axis, indexing="ij")
        for suffix in ("m", "a"):
            grid = cached[f"w{size}_{suffix}"].astype(np.float32)
            resampled = ndimage.map_coordinates(grid, [rows, cols], order=1, mode="nearest")
            channels.append(resampled.astype(np.float32))
    return np.stack(channels)


def load_window_maps(split: str, method: str, name: str) -> NDArray[np.float32]:
    """Cached window maps of one image on the trunk grid.

    Raises:
        WindowError: the cache file does not exist.
    """
    path = window_path(split, method, name)
    if not path.is_file():
        raise WindowError(f"missing window cache {path}")
    with np.load(path) as archive:
        return window_maps({key: archive[key] for key in archive.files})


def _atomic_save(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def extract_windows(
    split: str,
    method: str,
    limit: int | None = None,
    shard: int = 0,
    n_shards: int = 1,
) -> int:
    """Fill the window cache of `method` in `split`; returns files written.

    Idempotent and shardable: image `i` belongs to shard `i % n_shards`, so several GPUs can fill
    the same cache without overlapping.
    """
    from hackathon.wmloc.research.kit import load_wrapper  # kit venv only

    torch.set_num_threads(THREAD_LIMIT)
    records = [r for r in load_records(split) if r.method == method]
    records = [r for i, r in enumerate(records) if i % n_shards == shard]
    if limit is not None:
        records = records[:limit]
    pending: list[Record] = [r for r in records if not window_path(split, method, r.name).exists()]
    print(f"windows {split}/{method} shard {shard}/{n_shards}: {len(pending)} to do", flush=True)
    if not pending:
        return 0
    device = accelerator()
    wrapper = load_wrapper(method, device)
    started = time.time()
    for done, record in enumerate(pending, start=1):
        image = load_image(split, record.name).to(device)
        signs = message_signs(record.message).to(device)
        _atomic_save(window_path(split, method, record.name), decode_windows(wrapper, image, signs))
        if done % PROGRESS_EVERY == 0:
            print(
                f"  {done}/{len(pending)} ({done / (time.time() - started):.2f} img/s)", flush=True
            )
    return len(pending)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache multi-scale window decodes (team B2).")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--method", choices=(METHOD_PIXELSEAL, METHOD_TRUSTMARK), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    count = extract_windows(args.split, args.method, args.limit, args.shard, args.n_shards)
    print(f"done, {count} cache files written", flush=True)


if __name__ == "__main__":
    main()
