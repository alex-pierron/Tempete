"""Shared helpers for the LOC-LAB probes (train split only; the test split is never read here)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.stats import rankdata

ROOT = Path(os.environ.get("WMLOC_DATA_DIR", "data/hackathon/wmloc"))
"""Local task folder: release/{train,test} of the official dataset, starter_kit/, models/, and the cached maps."""
TRAIN = ROOT / "release" / "train"
HEAD_CV = ROOT / "team_b2" / "cv"
GRID = 128
SIZE = 512


def kit_watermark(method: str, device: str = "cpu") -> Any:
    """Load the starter-kit wrapper (the exact embedders used to build the dataset)."""
    sys.path.insert(0, str(ROOT / "starter_kit"))
    from wmloc.watermarks import load_watermark

    return load_watermark(method, str(ROOT / "models"), device=device)


def train_meta(method: str) -> list[dict[str, Any]]:
    """Train metadata rows of one method, in file order (the order of the head OOF arrays)."""
    rows: list[dict[str, Any]] = json.loads((TRAIN / "metadata.json").read_text())
    return [r for r in rows if r["method"] == method]


def load_image(name: str) -> NDArray[np.uint8]:
    """512 x 512 x 3 uint8 train image."""
    return np.asarray(Image.open(TRAIN / "images" / f"{name}.png").convert("RGB"), dtype=np.uint8)


def gt128() -> dict[str, NDArray[np.uint8]]:
    """Official 128 x 128 ground truth, by name."""
    z = np.load(TRAIN / "masks_128.npz")
    return {str(n): (m > 0).astype(np.uint8) for n, m in zip(z["names"], z["masks"], strict=True)}


def auc(lab: NDArray[np.int_], score: NDArray[np.floating[Any]]) -> float:
    """Mann-Whitney AUC; 0.5 when one class is missing."""
    pos = lab == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def miou(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> float:
    """Official per-image two-class mIoU (empty union counts 1)."""
    out = []
    for c in (0, 1):
        p, g = pred == c, gt == c
        union = int((p | g).sum())
        out.append(1.0 if union == 0 else int((p & g).sum()) / union)
    return float(np.mean(out))


def pool128(a: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Mean-pool a H x W map (H, W multiples of 128) to the 128 grid."""
    h, w = a.shape
    return a.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3)).astype(np.float64)
