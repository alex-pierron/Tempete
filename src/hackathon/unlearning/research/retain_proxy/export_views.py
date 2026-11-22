"""Logits of a model on team C's 1000-image subset and five views (the format of `research_c/logits`).

Team C scores candidates from per-view logits. This module reproduces its subset and views EXACTLY
(same seeds and offsets as `a3_export`) so that a fine-tuned model of team D, or an alpha blend of it
with the original, can be ranked by its scoring model without the 344 MB file leaving the cluster.

Format of one `<name>.npz`: `subset_indices` [1000], and for each view of `VIEWS_EXPORTED`
`logits_<view>` [1000, 100] float32 and `feats_<view>` [1000, 768] float16.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.alterations import ALTERATIONS, random_crop_resize, strong_view
from hackathon.unlearning.constants import DEFAULT_SEED
from hackathon.unlearning.model import preprocess

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
    """The subset cannot be drawn (fewer forget images than the subset size)."""


def subset_indices(total: int, size: int = EVAL_SUBSET) -> Tensor:
    """Team C's draw: the first `size` of a seed-0 permutation, sorted."""
    if total < size:
        raise ExportError(f"Need at least {size} forget images, got {total}.")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(DEFAULT_SEED))
    return order[:size].sort().values


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
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    logits: list[Tensor] = []
    features: list[Tensor] = []
    for start in range(0, len(images), PREDICT_BATCH):
        batch = preprocess(images[start : start + PREDICT_BATCH], device)
        feats = model.forward_head(model.forward_features(batch), pre_logits=True)  # type: ignore[operator]
        logits.append(model.head(feats).float().cpu())  # type: ignore[operator]
        features.append(feats.float().cpu())
    model.train(was_training)
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
        # A full disk or SIGTERM must never leave a half file that looks like an export.
        partial.unlink(missing_ok=True)
        raise
    return target
