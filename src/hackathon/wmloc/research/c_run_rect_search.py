"""Run the rectangle search on a split and cache one JSON per image (team C).

Idempotent and resumable: an image whose JSON exists is skipped; files are written atomically.
Sharding (`--shard i --shards n`) splits the work by index so that several processes, one per
GPU (`CUDA_VISIBLE_DEVICES`), never touch the same image.

Usage (kit venv, see reports/phase2/wmloc/team-c/scripts/):
    python -m hackathon.wmloc.research.c_run_rect_search --split train --method trustmark \
        --regime rect --out <dir> [--shard 0 --shards 4]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.constants import (
    IMAGE_SIZE,
    KNOWN_METHODS,
    MAX_ASPECT_RATIO_BY_METHOD,
)
from hackathon.wmloc.research.c_batching import configure_batch
from hackathon.wmloc.research.c_decoder import LightDecoder
from hackathon.wmloc.research.c_rect_search import (
    DEFAULT_REFINE_STEPS_PX,
    DEFAULT_SIZE_RATIO,
    DEFAULT_STRIDE_FRACTION,
    MIN_SIDE_PX,
    DecoderWindowScorer,
    SearchConfig,
    resize_crop,
    search_rectangle,
)
from hackathon.wmloc.research.c_results import ResultStoreError, ShardWriter, done_names, load_all
from hackathon.wmloc.research.data import SPLITS, Record, load_image, load_records
from hackathon.wmloc.research.errors import ResearchError

REGIME_RECT = "rect"
REGIME_FULL = "full"
REGIME_ALL = "all"
DEFAULT_ROUTE_AGREEMENT = 0.85
"""Test images whose whole-image agreement reaches this are in the full regime: no search.
The full regime decodes at 0.98 on average, the rect regime at 0.52-0.55 (train)."""
PROGRESS_EVERY = 5


class RunError(ResearchError):
    """Invalid command-line combination."""


def select_records(
    split: str, method: str, regime: str, limit: int | None, names: set[str] | None
) -> list[Record]:
    """Records to process: one method, optionally one train regime (rect or full), optional subset."""
    if regime in (REGIME_RECT, REGIME_FULL) and split != "train":
        raise RunError(f"--regime {regime} needs the train labels")
    records = [r for r in load_records(split) if r.method == method]
    if regime == REGIME_RECT:
        records = [r for r in records if r.full_watermark is False]
    elif regime == REGIME_FULL:
        records = [r for r in records if r.full_watermark is True]
    if names is not None:
        records = [r for r in records if r.name in names]
    if limit is not None:
        records = records[:limit]
    return records


def undetected_names(directory: Path) -> set[str]:
    """Names searched but not detected in a previous pass (the candidates of a finer second pass).

    Raises:
        RunError: the directory has no result.
    """
    try:
        payloads = load_all(directory)
    except ResultStoreError as error:
        raise RunError(str(error)) from error
    return {str(p["name"]) for p in payloads if p.get("searched") and not p.get("detected")}


def global_agreement(decoder: LightDecoder, image: torch.Tensor, message: np.ndarray) -> float:
    """Bit agreement of the whole image (routing signal between the two regimes)."""
    crop = resize_crop(image, decoder.method, decoder.resolution)[None]
    logits = decoder.logits(crop)[0].numpy()
    return float(((logits > 0).astype(np.float32) == message).mean())


def process_image(
    decoder: LightDecoder,
    split: str,
    record: Record,
    config: SearchConfig,
    route_agreement: float,
) -> dict[str, Any]:
    """Search one image and return the JSON payload."""
    started = time.time()
    image = load_image(split, record.name)
    whole = global_agreement(decoder, image, record.message)
    payload: dict[str, Any] = {
        "name": record.name,
        "method": record.method,
        "global_agreement": whole,
        "full_watermark": record.full_watermark,
    }
    if split == "test" and whole >= route_agreement:
        payload.update(searched=False, detected=False, box=None, n_decodes=0)
        payload["seconds"] = time.time() - started
        return payload
    scorer = DecoderWindowScorer(decoder, image, record.message)
    result = search_rectangle(scorer, config)
    payload.update(
        searched=True,
        detected=result.detected,
        box=None if result.box is None else asdict(result.box),
        margin=result.margin,
        evidence=result.evidence,
        agreement=result.agreement,
        n_decodes=result.n_decodes,
    )
    payload["seconds"] = time.time() - started
    return payload


def run(args: argparse.Namespace) -> int:
    """Process this shard; returns the number of images written."""
    names = None
    if args.names_file is not None:
        names = {line.strip() for line in args.names_file.read_text().splitlines() if line.strip()}
    if args.retry_undetected_in is not None:
        retry = undetected_names(args.retry_undetected_in)
        names = retry if names is None else names & retry
    records = select_records(args.split, args.method, args.regime, args.limit, names)
    records = [r for i, r in enumerate(records) if i % args.shards == args.shard]
    out_dir: Path = args.out
    already_done = done_names(out_dir)
    pending = [r for r in records if r.name not in already_done]
    writer = ShardWriter(out_dir, f"shard{args.shard}of{args.shards}")
    print(
        f"{args.split}/{args.method}: shard {args.shard}/{args.shards}, "
        f"{len(pending)} to do of {len(records)}",
        flush=True,
    )
    if not pending:
        return 0
    decoder = LightDecoder(args.method, args.device)
    if args.decode_batch != "env":
        print(configure_batch(decoder, args.decode_batch), flush=True)
    config = SearchConfig(
        max_aspect=MAX_ASPECT_RATIO_BY_METHOD[args.method],
        size_ratio=args.size_ratio,
        stride_fraction=args.stride,
        min_side=args.min_side,
        max_side=args.max_side,
        refine_steps=tuple(int(s) for s in args.refine_steps.split(",")),
    )
    started = time.time()
    for index, record in enumerate(pending):
        payload = process_image(decoder, args.split, record, config, args.route_agreement)
        writer.append(payload)
        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(pending):
            rate = (time.time() - started) / (index + 1)
            print(f"  {index + 1}/{len(pending)} images ({rate:.1f} s/image)", flush=True)
    return len(pending)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Rectangle search by window decoding.")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument(
        "--regime", choices=(REGIME_RECT, REGIME_FULL, REGIME_ALL), default=REGIME_ALL
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--size-ratio", type=float, default=DEFAULT_SIZE_RATIO)
    parser.add_argument("--stride", type=float, default=DEFAULT_STRIDE_FRACTION)
    parser.add_argument("--min-side", type=int, default=MIN_SIDE_PX)
    parser.add_argument("--max-side", type=int, default=IMAGE_SIZE)
    parser.add_argument("--refine-steps", default=",".join(str(s) for s in DEFAULT_REFINE_STEPS_PX))
    parser.add_argument(
        "--retry-undetected-in",
        type=Path,
        help="second pass: only the images this earlier results directory searched without detection",
    )
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
        raise RunError("--shard must be in [0, --shards)")
    count = run(args)
    print(f"done, {count} images written", flush=True)


if __name__ == "__main__":
    main()
