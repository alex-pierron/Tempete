"""Logits of candidate files on the 1000-image evaluation subset of team C (small files, no .pt).

Team C scores candidates from per-view logits (`research_c/logits/<name>.npz`, produced by its
`extract_models.py`). Candidate files are 344 MB and live on JURECA; the logits are ~9 MB. This
module reproduces team C's subset and views EXACTLY (same seeds and offsets) so that its scoring
model can rank an A3 candidate without the file ever leaving the cluster.

Format of one `<name>.npz`: `subset_indices` [1000], and for each view in `VIEWS_EXPORTED`
`logits_<view>` [1000, 100] (float32) and `feats_<view>` [1000, 768] (float16).

Usage (shard i of n over the files, one process per GPU):
    python -m hackathon.unlearning.research.a3_export --out-dir <dir> --shard 0 --shards 4 \
        NAME=PATH [NAME=PATH ...]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hackathon.common.config import load_settings
from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import ALTERATIONS, random_crop_resize, strong_view
from hackathon.unlearning.constants import DEFAULT_SEED, FORGET_IMAGES_FILENAME, default_task_dir
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.model import load_model, preprocess, select_device

LOGGER = logging.getLogger("hackathon.unlearning.research")
EVAL_SUBSET = 1000
PREDICT_BATCH = 32
CROP_35_AREAS = (0.33, 0.37)
STRONG_MIN_AREA = 0.35
KIT_OFFSETS = {"flip": 1, "crop": 2, "jitter": 3, "jpeg": 4}
CROP_35_OFFSET = 11
STRONG_OFFSET = 20
VIEWS_EXPORTED = ("exact", "crop", "jitter", "crop_35", "strong_view")
PARTIAL_SUFFIX = ".partial.npz"


class ExportError(HackathonError):
    """A candidate could not be exported (bad argument, missing file)."""


def subset_indices(total: int) -> Tensor:
    """Team C's draw: first 1000 of a seed-0 permutation, sorted."""
    order = torch.randperm(total, generator=torch.Generator().manual_seed(DEFAULT_SEED))
    return order[:EVAL_SUBSET].sort().values


def build_views(forget_subset: Tensor) -> dict[str, Tensor]:
    """The five views of team C (kit alterations, 35 % crop, strong view), seeds identical."""
    views: dict[str, Tensor] = {"exact": forget_subset}
    for name in ("crop", "jitter"):
        generator = torch.Generator().manual_seed(DEFAULT_SEED + KIT_OFFSETS[name])
        views[name] = ALTERATIONS[name](forget_subset, generator)
    views["crop_35"] = random_crop_resize(
        forget_subset,
        torch.Generator().manual_seed(DEFAULT_SEED + CROP_35_OFFSET),
        *CROP_35_AREAS,
    )
    views["strong_view"] = strong_view(
        forget_subset, torch.Generator().manual_seed(DEFAULT_SEED + STRONG_OFFSET), STRONG_MIN_AREA
    )
    return views


@torch.no_grad()
def logits_and_features(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    """Logits [N, 100] and pre-head features [N, 768] on CPU, eval mode, fp32."""
    model.eval()
    device = next(model.parameters()).device
    logits: list[Tensor] = []
    features: list[Tensor] = []
    for start in range(0, len(images), PREDICT_BATCH):
        batch = preprocess(images[start : start + PREDICT_BATCH], device)
        feats = model.forward_head(model.forward_features(batch), pre_logits=True)  # type: ignore[operator]
        logits.append(model.head(feats).float().cpu())  # type: ignore[operator]
        features.append(feats.float().cpu())
    return torch.cat(logits), torch.cat(features)


def export_model(
    model: nn.Module, name: str, out_dir: Path, indices: Tensor, views: dict[str, Tensor]
) -> Path:
    """Write `<out_dir>/<name>.npz` atomically; an existing file is left untouched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.npz"
    if target.exists():
        return target
    payload: dict[str, np.ndarray] = {"subset_indices": indices.numpy()}
    for view_name in VIEWS_EXPORTED:
        logits, features = logits_and_features(model, views[view_name])
        payload[f"logits_{view_name}"] = logits.numpy()
        payload[f"feats_{view_name}"] = features.numpy().astype(np.float16)
    partial = target.with_suffix(PARTIAL_SUFFIX)
    try:
        np.savez_compressed(partial, **payload)  # type: ignore[arg-type]
        partial.rename(target)
    except BaseException:
        # SIGTERM (SystemExit) or a full disk: never leave a half file that looks like an export.
        partial.unlink(missing_ok=True)
        raise
    return target


def parse_pairs(arguments: list[str]) -> list[tuple[str, Path]]:
    """`NAME=PATH` arguments to (name, path); raises on a malformed one."""
    pairs: list[tuple[str, Path]] = []
    for argument in arguments:
        name, separator, raw_path = argument.partition("=")
        if not separator or not name or not raw_path:
            raise ExportError(f"expected NAME=PATH, got '{argument}'")
        pairs.append((name, Path(raw_path)))
    return pairs


def shard(pairs: list[tuple[str, Path]], index: int, count: int) -> list[tuple[str, Path]]:
    """Every `count`-th pair starting at `index` (round-robin split over the GPUs)."""
    if count < 1 or not 0 <= index < count:
        raise ExportError(f"bad shard {index}/{count}")
    return pairs[index::count]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("pairs", nargs="+", help="NAME=PATH")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mine = shard(parse_pairs(args.pairs), args.shard, args.shards)
    device = select_device(None)
    task_dir = default_task_dir(load_settings(env_file=None).data_dir)
    forget_all = load_forget_images(task_dir / FORGET_IMAGES_FILENAME)
    indices = subset_indices(len(forget_all))
    views = build_views(forget_all[indices])
    for name, path in mine:
        started = time.time()
        model = load_model(path, device)
        export_model(model, name, args.out_dir, indices, views)
        LOGGER.info("%s exported in %.0f s", name, time.time() - started)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
