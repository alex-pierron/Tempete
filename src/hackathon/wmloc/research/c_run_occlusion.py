"""Cache occlusion maps (full regime) for a split and a method, one NPZ per image (team C).

Idempotent, resumable and shardable like `c_run_rect_search`. Train: the images of the full
regime (`full_watermark` true). Test: the images whose whole-image agreement reaches
`--route-agreement` (the routing of the rect search, mirrored).

Usage (kit venv):
    python -m hackathon.wmloc.research.c_run_occlusion --split train --method trustmark \
        --sigmas 6,12 --out <dir> [--shard 0 --shards 4]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.c_batching import configure_batch
from hackathon.wmloc.research.c_decoder import LightDecoder
from hackathon.wmloc.research.c_occlusion import (
    DEFAULT_BLOCK_PX,
    DEFAULT_STRIDE_PX,
    occlusion_map,
)
from hackathon.wmloc.research.c_results import ArrayChunkWriter, done_array_names
from hackathon.wmloc.research.c_run_rect_search import (
    DEFAULT_ROUTE_AGREEMENT,
    global_agreement,
)
from hackathon.wmloc.research.data import SPLITS, Record, load_image, load_records
from hackathon.wmloc.research.errors import ResearchError

DEFAULT_SIGMAS = "6,12"
PROGRESS_EVERY = 5


class OcclusionRunError(ResearchError):
    """Invalid command-line combination."""


def select_records(split: str, method: str, limit: int | None) -> list[Record]:
    """Train: full-regime images; test: every image (routing happens after the global decode)."""
    records = [r for r in load_records(split) if r.method == method]
    if split == "train":
        records = [r for r in records if r.full_watermark is True]
    return records[:limit] if limit is not None else records


def run(args: argparse.Namespace) -> int:
    """Process this shard; returns the number of images written."""
    sigmas = [float(s) for s in args.sigmas.split(",")]
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
        whole = global_agreement(decoder, image, record.message)
        arrays: dict[str, Any] = {"global_agreement": np.float32(whole)}
        if args.split == "train" or whole >= args.route_agreement:
            for sigma in sigmas:
                found = occlusion_map(
                    decoder, image, record.message, sigma, args.block, args.stride
                )
                arrays[f"drop_s{sigma:g}"] = found.drop
                arrays["base_margin"] = np.float32(found.base_margin)
                arrays["origins"] = found.origins
            arrays["block"] = np.int32(args.block)
        writer.add(record.name, arrays)
        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(pending):
            print(
                f"  {index + 1}/{len(pending)} ({(time.time() - started) / (index + 1):.1f} s/image)",
                flush=True,
            )
    writer.flush()
    return len(pending)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Occlusion maps (full regime).")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS)
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK_PX)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE_PX)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--route-agreement", type=float, default=DEFAULT_ROUTE_AGREEMENT)
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
        raise OcclusionRunError("--shard must be in [0, --shards)")
    count = run(args)
    print(f"done, {count} images written", flush=True)


if __name__ == "__main__":
    main()
