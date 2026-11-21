"""Shared, small image fixtures (skimage sample images, no download)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from skimage import data as skimage_data

from wmkit.io import save_image

SMALL_IMAGE_SIDE = 96
"""Side of the small fixture images: big enough for SSIM (window 11), fast to attack."""

SAMPLE_IMAGE_LOADERS: dict[str, Callable[[], np.ndarray]] = {
    "astronaut": skimage_data.astronaut,
    "coffee": skimage_data.coffee,
    "chelsea": skimage_data.chelsea,
    "rocket": skimage_data.rocket,
}


def load_sample(name: str, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Return a skimage sample as a [3, H, W] float32 tensor, optionally resized to `size`."""
    array = SAMPLE_IMAGE_LOADERS[name]()
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
    if size is not None:
        tensor = torch_f.interpolate(
            tensor[None], size=size, mode="bilinear", antialias=True, align_corners=False
        )[0].clamp(0.0, 1.0)
    return tensor


@pytest.fixture(scope="session")
def small_image() -> torch.Tensor:
    """One natural image [3, 96, 96]."""
    return load_sample("astronaut", (SMALL_IMAGE_SIDE, SMALL_IMAGE_SIDE))


@pytest.fixture(scope="session")
def small_batch() -> torch.Tensor:
    """Two natural images [2, 3, 96, 96]."""
    first = load_sample("astronaut", (SMALL_IMAGE_SIDE, SMALL_IMAGE_SIDE))
    second = load_sample("coffee", (SMALL_IMAGE_SIDE, SMALL_IMAGE_SIDE))
    return torch.stack([first, second])


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Folder with three small images of different sizes (png, png, jpg)."""
    folder = tmp_path / "images"
    save_image(load_sample("astronaut", (112, 112)), folder / "a_astronaut.png")
    save_image(load_sample("coffee", (96, 128)), folder / "b_coffee.png")
    save_image(load_sample("chelsea", (100, 140)), folder / "c_chelsea.jpg")
    return folder
