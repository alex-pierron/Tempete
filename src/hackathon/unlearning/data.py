"""Forget set loading and local retain-proxy images.

IMPORTANT limitation, repeated in the docs: the retain proxy images (COCO / DIV2K / Kodak,
resized to 224) are NOT from the task distribution (a custom 100-class dataset). They only tell
us how much the model's outputs drift on natural images in general; they say nothing about the
hidden control set accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import (
    EXPECTED_FORGET_IMAGES,
    FORGET_IMAGES_KEY,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    IMAGE_SUFFIXES,
    PROXY_IMAGE_DIRS,
)


class DataError(HackathonError):
    """An image file is missing or malformed."""


def load_forget_images(path: Path, limit: int | None = None) -> Tensor:
    """Forget images as uint8 [N, 224, 224, 3] (raw pixels, channels last), optionally the first N.

    Mirrors the checks of the official loader (key, dtype, rank, spatial shape) and tolerates a
    channels-first array, like it does. A count different from 3500 only warns through the
    returned shape: callers decide (a smoke subset is legitimate).

    Raises:
        DataError: missing file, missing key, wrong dtype or shape.
    """
    if not path.is_file():
        raise DataError(f"{path} not found (run `fetch` first).")
    with np.load(path, allow_pickle=False) as archive:
        if FORGET_IMAGES_KEY not in archive.files:
            raise DataError(f"{path} has no '{FORGET_IMAGES_KEY}' array; found {archive.files}.")
        # mmap is not possible inside an npz; slice before the copy when a limit is given.
        array = archive[FORGET_IMAGES_KEY]
        if limit is not None:
            array = array[:limit]
        images = torch.from_numpy(np.ascontiguousarray(array))
    if images.dtype != torch.uint8:
        raise DataError(f"Expected uint8 pixels, got {images.dtype}")
    if images.ndim != 4:
        raise DataError(f"Expected 4 dimensions [N, H, W, 3], got {tuple(images.shape)}")
    if images.shape[-1] != IMAGE_CHANNELS and images.shape[1] == IMAGE_CHANNELS:
        images = images.permute(0, 2, 3, 1).contiguous()
    if tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
        raise DataError(
            f"Expected [N, {IMAGE_SIZE}, {IMAGE_SIZE}, {IMAGE_CHANNELS}], got {tuple(images.shape)}"
        )
    return images


def is_full_forget_set(images: Tensor) -> bool:
    """True when `images` has the official 3500 images (False for a smoke subset)."""
    return len(images) == EXPECTED_FORGET_IMAGES


def select_subset(images: Tensor, count: int, seed: int) -> Tensor:
    """Reproducible random subset of `count` images (all of them when count >= len)."""
    if count >= len(images):
        return images
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(images), generator=generator)[:count].sort().values
    return images[indices]


def list_proxy_image_files(data_root: Path) -> list[Path]:
    """Sorted image files of the proxy folders present under `data_root` (missing ones skipped)."""
    files: list[Path] = []
    for folder in PROXY_IMAGE_DIRS:
        directory = data_root / folder
        if directory.is_dir():
            files.extend(
                sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
    return files


def load_resized_image(path: Path) -> Tensor:
    """One image as uint8 [224, 224, 3]: shorter side resized to 224, then center crop.

    Raises:
        DataError: the file cannot be decoded (typed, not a raw PIL traceback).
    """
    from PIL import Image  # lazy: Pillow ships with torchvision, which timm already requires

    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise DataError(f"Cannot read image {path}: {exc}") from exc
    width, height = image.size
    scale = IMAGE_SIZE / min(width, height)
    resized = image.resize(
        (max(IMAGE_SIZE, round(width * scale)), max(IMAGE_SIZE, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - IMAGE_SIZE) // 2
    top = (resized.height - IMAGE_SIZE) // 2
    cropped = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    return torch.from_numpy(np.asarray(cropped, dtype=np.uint8).copy())


def load_proxy_images(data_root: Path, count: int, seed: int) -> Tensor:
    """Up to `count` out-of-distribution natural images, uint8 [N, 224, 224, 3], reproducible.

    Args:
        data_root: Folder holding coco_val_subset / div2k_valid / kodak (the repo `data/`).
        count: Maximum number of images; files are drawn without replacement with `seed`.
        seed: Named seed of the draw.

    Raises:
        DataError: no proxy folder found, or `count` is not positive.
    """
    if count <= 0:
        raise DataError("The retain proxy needs at least one image.")
    files = list_proxy_image_files(data_root)
    if not files:
        raise DataError(f"No proxy images under {data_root} ({', '.join(PROXY_IMAGE_DIRS)}).")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(files), generator=generator)[:count].sort().values.tolist()
    return torch.stack([load_resized_image(files[index]) for index in order])
