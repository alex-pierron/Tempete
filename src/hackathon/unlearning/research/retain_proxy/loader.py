"""Drop-in loader of the generated retain proxy for the existing research code.

`hackathon.unlearning.data.load_proxy_images(data_root, count, seed)` returns uint8 [N, 224, 224, 3]
natural images from COCO / DIV2K / Kodak folders (out of distribution). The generated proxy is in
distribution, stored as a few `.npz` files; this module returns the same kind of tensor from them,
so `context.build_context` (retain / retain_train) and `trainer.train` (the KL anchor) can use it
by swapping one call.

Two sources: the assembled dataset (`v1_proxy.npz`: images, labels, orig_logits) or the raw chunk
folders of a generation run (`chunk_*.npz`, every generated image, confident or not).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.research.retain_proxy.generate import CHUNK_FILE_TEMPLATE

IMAGES_KEY = "images"


class LoaderError(HackathonError):
    """The proxy source is missing or empty, or `count` is not positive."""


def _image_arrays(source: Path) -> list[np.ndarray]:
    if source.is_file():
        with np.load(source) as archive:
            return [archive[IMAGES_KEY]]
    chunks = sorted(source.glob(CHUNK_FILE_TEMPLATE.replace("{first_index:07d}", "*")))
    if not chunks:
        raise LoaderError(f"No proxy images at {source} (expected a .npz file or chunk_*.npz).")
    arrays = []
    for path in chunks:
        with np.load(path) as archive:
            arrays.append(archive[IMAGES_KEY])
    return arrays


def load_generated_proxy_images(source: Path, count: int, seed: int) -> Tensor:
    """`count` images (all of them when fewer exist) as uint8 [N, 224, 224, 3], seeded draw.

    Args:
        source: an assembled `.npz` dataset or a folder of generation chunks.
        count: number of images wanted.
        seed: seed of the draw (same seed, same images).

    Raises:
        LoaderError: `count < 1`, or no image at `source`.
    """
    if count < 1:
        raise LoaderError("The retain proxy needs at least one image.")
    images = torch.from_numpy(np.concatenate(_image_arrays(source)))
    if len(images) <= count:
        return images
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(seed))
    return images[order[:count].sort().values]
