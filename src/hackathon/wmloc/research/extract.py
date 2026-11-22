"""Expensive stage: decode crops of every image once and cache the raw logits on disk.

One cache file per (split, method, image, pass): `<cache>/<split>/<method>/<name>.<pass>.npz`,
written atomically (temp file then rename) so a crash never leaves a half-written file and a
restart simply skips what exists (idempotent, resumable). Passes:

* `global`  : logits of the whole image, [1, n_bits].
* `posmap`  : per-position logits of the 2x2 shifted whole image, [4, n_bits, 8, 8] (PixelSeal,
              TrustMark only), see `shift_geometry`.
* `ss<L>`   : per-position logits [n_bits, h, w] of the whole image enlarged to L x L px
              (scale-space probe, PixelSeal and TrustMark only), cell = 32 * 512 / L px.
* `w<S>s<T>`: logits of the square windows of side S px at stride T px, [W, n_bits] and the
              boxes [W, 4] (top, left, bottom, right).

Usage (kit venv, see reports/phase1/wmloc/team-a/scripts/run_kit.sh):
    python -m hackathon.wmloc.research.extract --split train --method pixelseal \
        --passes global,posmap,w128s64
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from hackathon.wmloc.constants import IMAGE_SIZE, KNOWN_METHODS, METHOD_STEGASTAMP
from hackathon.wmloc.research.data import SPLITS, Record, data_root, load_image, load_records
from hackathon.wmloc.research.decoding import Decoder
from hackathon.wmloc.research.errors import ResearchError

CACHE_ENV = "WMLOC_RESEARCH_CACHE"
CACHE_SUBDIR = "research_cache"
PASS_GLOBAL = "global"
PASS_POSMAP = "posmap"
WINDOW_PASS_PATTERN = re.compile(r"^w(\d+)s(\d+)$")
SCALE_PASS_PATTERN = re.compile(r"^ss(\d+)$")
DECODE_CHUNK = 64
CELL_PX = 64
"""Pixels (at 512) covered by one cell of the per-position map (8x8 map on a 512 image)."""
SHIFT_STEP_PX = CELL_PX // 2
"""Half-cell shifts densify the 8x8 map to 16x16 (spacing 32 px)."""
SHIFT_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
"""(row, column) shift multiples of SHIFT_STEP_PX applied to the image, in `posmap` order."""
PROGRESS_EVERY = 10


class ExtractionError(ResearchError):
    """A pass name is unknown or a method does not support it."""


@dataclass(frozen=True)
class WindowSpec:
    """Square windows of `size` px on a regular grid of step `stride` px."""

    size: int
    stride: int


def parse_window_pass(name: str) -> WindowSpec | None:
    """`w128s64` -> WindowSpec(128, 64); None if `name` is not a window pass."""
    match = WINDOW_PASS_PATTERN.match(name)
    if match is None:
        return None
    return WindowSpec(size=int(match.group(1)), stride=int(match.group(2)))


def window_boxes(spec: WindowSpec) -> np.ndarray:
    """Boxes [W, 4] (top, left, bottom, right) covering the 512 image, last row/column flush.

    Raises:
        ExtractionError: size outside (0, 512] or non-positive stride.
    """
    if not 0 < spec.size <= IMAGE_SIZE or spec.stride <= 0:
        raise ExtractionError(f"bad window spec {spec}")
    origins = list(range(0, IMAGE_SIZE - spec.size + 1, spec.stride))
    if origins[-1] != IMAGE_SIZE - spec.size:
        origins.append(IMAGE_SIZE - spec.size)
    return np.array(
        [(y, x, y + spec.size, x + spec.size) for y in origins for x in origins], dtype=np.int32
    )


def cache_dir() -> Path:
    """Root of the logit cache (`$WMLOC_RESEARCH_CACHE` or `<data>/wmloc/research_cache`)."""
    configured = os.environ.get(CACHE_ENV)
    return Path(configured).expanduser() if configured else data_root() / CACHE_SUBDIR


def cache_path(split: str, method: str, name: str, pass_name: str) -> Path:
    """Cache file of one (split, method, image, pass)."""
    return cache_dir() / split / method / f"{name}.{pass_name}.npz"


def _atomic_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def shifted_image(image: torch.Tensor, row_steps: int, col_steps: int) -> torch.Tensor:
    """The image moved down/right by `steps * SHIFT_STEP_PX` px (reflect-padded), [3, 512, 512]."""
    dy, dx = row_steps * SHIFT_STEP_PX, col_steps * SHIFT_STEP_PX
    if dy == 0 and dx == 0:
        return image
    padded = functional.pad(image[None], (dx, 0, dy, 0), mode="reflect")
    return padded[0, :, :IMAGE_SIZE, :IMAGE_SIZE]


def crops_of(image: torch.Tensor, boxes: np.ndarray) -> torch.Tensor:
    """Stack the crops of `boxes` (all of the same size) into [W, 3, s, s]."""
    return torch.stack([image[:, t:b, l:r] for t, l, b, r in boxes.tolist()])


def decode_chunked(decoder: Decoder, crops: torch.Tensor) -> torch.Tensor:
    """Logits [W, n_bits] of many crops, in chunks of `DECODE_CHUNK`."""
    parts = [
        decoder.logits(crops[i : i + DECODE_CHUNK]) for i in range(0, len(crops), DECODE_CHUNK)
    ]
    return torch.cat(parts)


def run_pass(decoder: Decoder, image: torch.Tensor, pass_name: str) -> dict[str, np.ndarray]:
    """Compute one pass for one image.

    Raises:
        ExtractionError: unknown pass, or `posmap` on StegaStamp.
    """
    if pass_name == PASS_GLOBAL:
        logits = decoder.logits(image[None])
        return {"logits": logits.numpy().astype(np.float32)}
    if pass_name == PASS_POSMAP:
        if decoder.method == METHOD_STEGASTAMP:
            raise ExtractionError("stegastamp has no per-position map")
        shifts = torch.stack([shifted_image(image, r, c) for r, c in SHIFT_OFFSETS])
        maps = decoder.position_logits(shifts)
        return {"maps": maps.numpy().astype(np.float16)}
    scale = SCALE_PASS_PATTERN.match(pass_name)
    if scale is not None:
        if decoder.method == METHOD_STEGASTAMP:
            raise ExtractionError("stegastamp has no per-position map")
        maps = decoder.position_logits_at_side(image[None], int(scale.group(1)))[0]
        return {"maps": maps.numpy().astype(np.float16)}
    spec = parse_window_pass(pass_name)
    if spec is None:
        raise ExtractionError(f"unknown pass {pass_name!r}")
    boxes = window_boxes(spec)
    logits = decode_chunked(decoder, crops_of(image, boxes))
    return {"logits": logits.numpy().astype(np.float16), "boxes": boxes}


def extract(
    split: str,
    method: str,
    passes: list[str],
    names: set[str] | None = None,
    limit: int | None = None,
    device: str | None = None,
) -> int:
    """Fill the cache for the images of `method` in `split`; returns the number of files written.

    Args:
        split: `train` or `test`.
        method: One of the three methods (the decoder is loaded once).
        passes: Pass names, e.g. `["global", "posmap", "w128s64"]`.
        names: If given, only these image names.
        limit: If given, stop after this many images (smoke tests).
        device: Torch device; default cuda / mps / cpu.
    """
    records = [r for r in load_records(split) if r.method == method]
    if names is not None:
        records = [r for r in records if r.name in names]
    if limit is not None:
        records = records[:limit]
    pending = [
        r for r in records if any(not cache_path(split, method, r.name, p).exists() for p in passes)
    ]
    print(f"{split}/{method}: {len(pending)} images to do out of {len(records)}", flush=True)
    if not pending:
        return 0
    decoder = Decoder(method, device)
    written = 0
    started = time.time()
    for index, record in enumerate(pending):
        written += _extract_one(decoder, split, record, passes)
        if (index + 1) % PROGRESS_EVERY == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(pending)} images ({rate:.2f} img/s)", flush=True)
    return written


def _extract_one(decoder: Decoder, split: str, record: Record, passes: list[str]) -> int:
    todo = [p for p in passes if not cache_path(split, record.method, record.name, p).exists()]
    if not todo:
        return 0
    image = load_image(split, record.name)
    for pass_name in todo:
        arrays = run_pass(decoder, image, pass_name)
        _atomic_save(cache_path(split, record.method, record.name, pass_name), **arrays)
    return len(todo)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache official-decoder logits of crops.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument(
        "--passes", required=True, help="comma-separated, e.g. global,posmap,w128s64"
    )
    parser.add_argument("--names-file", type=Path, help="text file, one image name per line")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    names = None
    if args.names_file is not None:
        names = {line.strip() for line in args.names_file.read_text().splitlines() if line.strip()}
    count = extract(args.split, args.method, args.passes.split(","), names, args.limit, args.device)
    print(f"done, {count} cache files written", flush=True)


if __name__ == "__main__":
    main()
