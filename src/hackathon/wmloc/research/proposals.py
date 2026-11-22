"""Segment proposals scored by decoding the known message (phase 2, team A).

About half of the marked zones of the rect regime are object silhouettes and the clean holes of
the full regime are blobs. An off-the-shelf instance segmenter (torchvision Mask R-CNN, COCO)
proposes candidate masks; for each one the official decoder is run on (a) the bounding-box crop and
(b) the largest axis-aligned rectangle fully inside the mask, and bit agreement / aligned margin
with the KNOWN message are stored. Selection rules are studied offline on these cached numbers.

One cache file per image: `<out>/<split>/<method>/<name>.props.npz`, written atomically; existing
files are skipped, so a killed job restarts where it stopped (idempotent).

Usage (kit venv, e.g. on JURECA; one process per GPU):
    CUDA_VISIBLE_DEVICES=0 python -m hackathon.wmloc.research.proposals \
        --splits train,test --shard 0 --n-shards 4 --out <cache dir>
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, KNOWN_METHODS
from hackathon.wmloc.research.data import SPLITS, Record, load_image, load_records
from hackathon.wmloc.research.decoding import Decoder
from hackathon.wmloc.research.errors import ResearchError

SCORE_MIN = 0.25
"""Lowest Mask R-CNN confidence kept as a proposal."""
MAX_PROPOSALS = 12
"""Proposals kept per image (highest confidence first): bounds the decode cost."""
MASK_THRESHOLD = 0.5
MIN_BOX_PX = 24
MIN_INSCRIBED_PX = 48
"""Smallest side (px at 512) of an inscribed rectangle worth decoding."""
BLOCK = IMAGE_SIZE // GRID_SIZE
PROPOSAL_FILE_SUFFIX = "props"
NO_VALUE = float("nan")
PROGRESS_EVERY = 25


class ProposalError(ResearchError):
    """Bad shard arguments or an unusable cache directory."""


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned box in px: rows [top, bottom), columns [left, right)."""

    top: int
    left: int
    bottom: int
    right: int

    def scaled(self, factor: int) -> Rectangle:
        """The same box in a grid `factor` times finer."""
        return Rectangle(
            self.top * factor, self.left * factor, self.bottom * factor, self.right * factor
        )

    @property
    def height(self) -> int:
        """Rows."""
        return self.bottom - self.top

    @property
    def width(self) -> int:
        """Columns."""
        return self.right - self.left


def downsample_mask(mask512: np.ndarray) -> np.ndarray:
    """Majority vote of each BLOCK x BLOCK block: [512, 512] -> [128, 128] uint8."""
    blocks = mask512.reshape(GRID_SIZE, BLOCK, GRID_SIZE, BLOCK).mean((1, 3))
    return (blocks >= 0.5).astype(np.uint8)


def largest_rectangle(mask: np.ndarray) -> Rectangle | None:
    """Largest axis-aligned all-ones rectangle of a binary mask (histogram method, O(H*W)).

    Returns:
        The rectangle, or None when the mask has no foreground.
    """
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int64)
    best_area = 0
    best: Rectangle | None = None
    for row in range(height):
        heights = np.where(mask[row] > 0, heights + 1, 0)
        stack: list[int] = []
        for col in range(width + 1):
            current = int(heights[col]) if col < width else 0
            while stack and heights[stack[-1]] >= current:
                top_col = stack.pop()
                bar = int(heights[top_col])
                left = stack[-1] + 1 if stack else 0
                area = bar * (col - left)
                if area > best_area:
                    best_area = area
                    best = Rectangle(row - bar + 1, left, row + 1, col)
            stack.append(col)
    return best


def decode_stats(decoder: Decoder, crop: torch.Tensor, message: np.ndarray) -> tuple[float, float]:
    """(bit agreement, mean aligned margin) of one [3, h, w] crop against the known message."""
    logit = decoder.logits(crop[None])[0]
    sign = torch.from_numpy(2.0 * message - 1.0).float()
    agreement = float(((logit > 0).float() == torch.from_numpy(message)).float().mean())
    return agreement, float((logit * sign).mean())


def proposal_cache_path(out_dir: Path, split: str, method: str, name: str) -> Path:
    """Cache file of one image."""
    return out_dir / split / method / f"{name}.{PROPOSAL_FILE_SUFFIX}.npz"


def build_segmenter(device: str) -> torch.nn.Module:
    """Mask R-CNN (COCO) in eval mode; weights come from `$TORCH_HOME` (no internet on compute nodes)."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model: torch.nn.Module = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
    return model


@torch.no_grad()
def proposals_for_image(
    segmenter: torch.nn.Module,
    decoder: Decoder,
    image: torch.Tensor,
    message: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """All proposal arrays of one image (first axis = proposal; empty arrays when none)."""
    detection = segmenter([image.to(device)])[0]
    masks: list[np.ndarray] = []
    rows: list[list[float]] = []
    for index in range(len(detection["scores"])):
        if len(masks) >= MAX_PROPOSALS:
            break
        score = float(detection["scores"][index])
        if score < SCORE_MIN:
            continue
        mask512 = (detection["masks"][index, 0] > MASK_THRESHOLD).cpu().numpy().astype(np.uint8)
        ys, xs = np.nonzero(mask512)
        if len(ys) == 0 or ys.max() - ys.min() < MIN_BOX_PX or xs.max() - xs.min() < MIN_BOX_PX:
            continue
        mask128 = downsample_mask(mask512)
        box_agree, box_margin = decode_stats(
            decoder, image[:, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], message
        )
        inner = largest_rectangle(mask128)
        in_agree, in_margin, in_h, in_w = NO_VALUE, NO_VALUE, 0.0, 0.0
        if inner is not None:
            big = inner.scaled(BLOCK)
            if big.height >= MIN_INSCRIBED_PX and big.width >= MIN_INSCRIBED_PX:
                in_agree, in_margin = decode_stats(
                    decoder, image[:, big.top : big.bottom, big.left : big.right], message
                )
                in_h, in_w = float(big.height), float(big.width)
        masks.append(mask128)
        rows.append(
            [
                score,
                float(detection["labels"][index]),
                float(mask128.mean()),
                box_agree,
                box_margin,
                in_agree,
                in_margin,
                in_h,
                in_w,
            ]
        )
    stats = np.array(rows, dtype=np.float32).reshape(-1, 9)
    return {
        "masks": np.stack(masks).astype(np.uint8)
        if masks
        else np.zeros((0, GRID_SIZE, GRID_SIZE), np.uint8),
        "stats": stats,
    }


STAT_COLUMNS = (
    "det_score",
    "label",
    "area",
    "bbox_agree",
    "bbox_margin",
    "inscr_agree",
    "inscr_margin",
    "inscr_h_px",
    "inscr_w_px",
)
"""Column order of the `stats` array."""


def run(splits: list[str], shard: int, n_shards: int, out_dir: Path, limit: int | None) -> int:
    """Fill the cache for this shard; returns the number of files written.

    Raises:
        ProposalError: invalid shard numbers.
    """
    if not 0 <= shard < n_shards:
        raise ProposalError(f"shard {shard} outside [0, {n_shards})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = build_segmenter(device)
    decoders = {method: Decoder(method, device) for method in KNOWN_METHODS}
    todo: list[tuple[str, Record]] = []
    for split in splits:
        records = sorted(load_records(split), key=lambda r: r.name)
        todo += [(split, r) for position, r in enumerate(records) if position % n_shards == shard]
    if limit is not None:
        todo = todo[:limit]
    written, started = 0, time.time()
    for index, (split, record) in enumerate(todo):
        path = proposal_cache_path(out_dir, split, record.method, record.name)
        if path.exists():
            continue
        arrays = proposals_for_image(
            segmenter,
            decoders[record.method],
            load_image(split, record.name),
            record.message,
            device,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, masks=arrays["masks"], stats=arrays["stats"])
        os.replace(temporary, path)
        written += 1
        if written % PROGRESS_EVERY == 0:
            rate = written / (time.time() - started)
            print(
                f"shard {shard}: {written} written, {index + 1}/{len(todo)} ({rate:.2f} img/s)",
                flush=True,
            )
    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache segment proposals scored by decoding.")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = run(args.splits.split(","), args.shard, args.n_shards, args.out, args.limit)
    print(f"done, {count} files written", flush=True)


if __name__ == "__main__":
    main()
