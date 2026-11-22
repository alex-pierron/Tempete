"""Loading of the official release splits (train and test) for the research code.

Paths come from `HACKATHON_DATA_DIR` only: this module deliberately avoids
`hackathon.common.config`, which parses the `.env` file (research code never needs a secret).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, WMLOC_DATA_SUBDIR
from hackathon.wmloc.research.errors import ResearchError

DATA_DIR_ENV = "HACKATHON_DATA_DIR"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLITS: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_TEST)
RELEASE_SUBDIR = "release"
METADATA_FILE = "metadata.json"
MASKS_128_FILE = "masks_128.npz"
BYTE_MAX = 255.0


class ResearchDataError(ResearchError):
    """The release data is missing or inconsistent."""


@dataclass(frozen=True)
class Record:
    """One image of a release split.

    Attributes:
        name: Image name (file name without `.png`), the key of the submission.
        method: `pixelseal`, `trustmark` or `stegastamp`.
        message: Known embedded message as a float32 array of 0/1, shape [n_bits].
        full_watermark: Regime flag (train only): True = everything marked but a restored hole.
    """

    name: str
    method: str
    message: np.ndarray
    full_watermark: bool | None


def data_root() -> Path:
    """`<HACKATHON_DATA_DIR>/wmloc`.

    Raises:
        ResearchDataError: the environment variable is not set.
    """
    configured = os.environ.get(DATA_DIR_ENV)
    if not configured:
        raise ResearchDataError(f"set {DATA_DIR_ENV} to the hackathon data directory")
    return Path(configured).expanduser() / WMLOC_DATA_SUBDIR


def split_dir(split: str) -> Path:
    """Folder of a split (`release/train` or `release/test`)."""
    if split not in SPLITS:
        raise ResearchDataError(f"split must be one of {SPLITS}, got {split!r}")
    return data_root() / RELEASE_SUBDIR / split


def load_records(split: str) -> list[Record]:
    """All records of a split, in metadata order.

    Raises:
        ResearchDataError: metadata file missing.
    """
    path = split_dir(split) / METADATA_FILE
    if not path.is_file():
        raise ResearchDataError(f"missing {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Record(
            name=str(item["image_name"]),
            method=str(item["method"]),
            message=np.array([float(char) for char in item["message"]], dtype=np.float32),
            full_watermark=item.get("full_watermark"),
        )
        for item in raw
    ]


def load_image(split: str, name: str) -> torch.Tensor:
    """The image as float32 [3, 512, 512] in [0, 1].

    Raises:
        ResearchDataError: file missing or not 512x512.
    """
    path = split_dir(split) / "images" / f"{name}.png"
    if not path.is_file():
        raise ResearchDataError(f"missing image {path}")
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ResearchDataError(f"{path} is {array.shape}, expected 512x512x3")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / BYTE_MAX


def load_truth(names: list[str]) -> np.ndarray:
    """Ground-truth 128x128 masks (uint8, 1 = watermarked) of train images, in `names` order.

    Raises:
        ResearchDataError: a name is not in the official `masks_128.npz`.
    """
    archive = np.load(split_dir(SPLIT_TRAIN) / MASKS_128_FILE, allow_pickle=True)
    index = {str(name): i for i, name in enumerate(archive["names"])}
    missing = [name for name in names if name not in index]
    if missing:
        raise ResearchDataError(f"{len(missing)} names without a truth mask, e.g. {missing[0]}")
    masks = np.asarray(archive["masks"], dtype=np.uint8)
    out = np.stack([masks[index[name]] for name in names])
    if out.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise ResearchDataError(f"unexpected truth shape {out.shape}")
    return out
