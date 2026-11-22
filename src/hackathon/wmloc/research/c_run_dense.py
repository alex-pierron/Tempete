"""Cache dense window-decode maps for a split and a method, chunked NPZ (team C).

Idempotent, resumable and shardable like `c_run_rect_search`; results go through
`ArrayChunkWriter` (ten images per file, few inodes). Train: every image of the method (both
regimes: the cached map is the evidence, the regime only matters offline). Test: every image.

Usage (kit venv):
    python -m hackathon.wmloc.research.c_run_dense --split train --method pixelseal \
        --out <dir> [--shard 0 --shards 4]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.c_batching import configure_batch
from hackathon.wmloc.research.c_decoder import LightDecoder
from hackathon.wmloc.research.c_dense_windows import (
    DEFAULT_SIZES,
    DEFAULT_STRIDE_FRACTION,
    dense_map,
    to_arrays,
)
from hackathon.wmloc.research.c_results import ArrayChunkWriter, done_array_names
from hackathon.wmloc.research.c_run_rect_search import global_agreement
from hackathon.wmloc.research.data import SPLITS, Record, load_image, load_records
from hackathon.wmloc.research.errors import ResearchError

PROGRESS_EVERY = 5


class DenseRunError(ResearchError):
    """Invalid command-line combination."""


def select_records(split: str, method: str, limit: int | None) -> list[Record]:
    """Every image of the method in the split (optionally the first `limit`)."""
    records = [r for r in load_records(split) if r.method == method]
    return records[:limit] if limit is not None else records


def run(args: argparse.Namespace) -> int:
    """Process this shard; returns the number of images written."""
    sizes = tuple(int(s) for s in args.sizes.split(","))
    records = select_records(args.split, args.method, args.limit)
    records = [r for i, r in enumerate(records) if i % args.shards == args.shard]
    out_dir: Path = args.out
    already_done = done_array_names(out_dir)
    pending = [r for r in records if r.name not in already_done]
    print(
        f"{args.split}/{args.method}: shard {args.shard}/{args.shards}, "
        f"{len(pending)} to do of {len(records)}",
        flush=True,
    )
    if not pending:
        return 0
    writer = ArrayChunkWriter(out_dir, f"shard{args.shard}of{args.shards}")
    decoder = LightDecoder(args.method, args.device)
    if args.decode_batch != "env":
        print(configure_batch(decoder, args.decode_batch), flush=True)
    started = time.time()
    for index, record in enumerate(pending):
        image = load_image(args.split, record.name)
        arrays = to_arrays(dense_map(decoder, image, record.message, sizes, args.stride))
        whole = global_agreement(decoder, image, record.message)
        arrays["global_agreement"] = np.asarray(whole, dtype=np.float32)
        writer.add(record.name, arrays)
        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(pending):
            seconds = (time.time() - started) / (index + 1)
            print(f"  {index + 1}/{len(pending)} ({seconds:.1f} s/image)", flush=True)
    writer.flush()
    return len(pending)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Dense window-decode maps.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE_FRACTION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--decode-batch",
        default="env",
        help="windows per decode pass: an integer, `auto` (autotune on the GPU) or `env` "
        "(WMC_DECODE_BATCH or 32)",
    )
    parser.add_argument("--device")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if not 0 <= args.shard < args.shards:
        raise DenseRunError("--shard must be in [0, --shards)")
    count = run(args)
    print(f"done, {count} images written", flush=True)


if __name__ == "__main__":
    main()
