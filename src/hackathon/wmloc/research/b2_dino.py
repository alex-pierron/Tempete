"""DINOv2 patch features of every release image: the semantic / object-boundary prior (team B2).

The edited regions of the dataset follow object and stuff silhouettes (see the montages in the
phase-2 report), so a learned segmenter needs a content prior that 510 labelled masks cannot teach
from scratch. A frozen self-supervised ViT provides it: its patch tokens separate objects from
background, and a small head trained on the real masks only has to learn WHICH regions get edited.

One `.npy` of float16 [384, 32, 32] per image under `<cache>/dino/<split>/<name>.npy`; the model
runs at 448 px (a 32 x 32 grid of 14 px patches, one cell per 16 px of the 512 original), which is
also the grid of the extractor features, so both can be concatenated without resampling.

Usage (kit venv): python -m hackathon.wmloc.research.b2_dino --split train
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional

from hackathon.wmloc.research.b2_features import (
    accelerator,
    cache_root,
    extraction_batch_size,
)
from hackathon.wmloc.research.data import SPLITS, load_image, load_records

DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
DINO_INPUT_SIDE = 448
DINO_PATCH = 14
DINO_GRID = DINO_INPUT_SIDE // DINO_PATCH
DINO_DIM = 384
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
BATCH_SIZE = 8
CUDA_BATCH_SIZE = 64
THREAD_LIMIT = 2


DINO_CHUNK_IMAGES = 256
_DINO_TABLES: dict[str, dict[str, NDArray[np.float16]]] = {}


def dino_dir(split: str) -> Path:
    """Folder of the DINO containers of `split`."""
    return cache_root().parent / "dino" / split


def load_dino_table(split: str, refresh: bool = False) -> dict[str, NDArray[np.float16]]:
    """Every cached image of `split`: `{name: float16 [384, 32, 32]}` (containers and legacy files)."""
    if split in _DINO_TABLES and not refresh:
        return _DINO_TABLES[split]
    table: dict[str, NDArray[np.float16]] = {}
    directory = dino_dir(split)
    if directory.is_dir():
        for path in sorted(directory.glob("*.np[yz]")):
            if ".tmp." in path.name:
                continue
            if path.suffix == ".npy":  # legacy: one file per image
                table[path.name[: -len(".npy")]] = np.load(path)
                continue
            with np.load(path) as archive:
                for index, name in enumerate(str(n) for n in archive["names"]):
                    table[name] = archive["features"][index]
    _DINO_TABLES[split] = table
    return table


def write_dino_chunk(split: str, names: list[str], features: NDArray[np.float16]) -> Path:
    """Write `features` [n, 384, 32, 32] of `names` as ONE container file, atomically."""
    path = dino_dir(split) / f"chunk_{int(time.time() * 1000)}_{os.getpid()}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, names=np.array(names), features=features)
    os.replace(temporary, path)
    _DINO_TABLES.pop(split, None)
    return path


def load_dino(split: str, name: str) -> NDArray[np.float16]:
    """Cached patch features [384, 32, 32] of one image.

    Raises:
        FileNotFoundError: the image is in no container of `split`.
    """
    table = load_dino_table(split)
    if name not in table:
        raise FileNotFoundError(f"missing DINO cache for {split}/{name} in {dino_dir(split)}")
    return table[name]


class DinoEncoder:
    """The frozen DINOv2 backbone, loaded once; used for real and synthetic images alike."""

    def __init__(self) -> None:
        import timm  # kit venv only: keeps this module importable where timm is absent

        self.device = accelerator()
        model = timm.create_model(DINO_MODEL_NAME, pretrained=True, img_size=DINO_INPUT_SIDE)
        self.model = model.eval().to(self.device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device)[None, :, None, None]
        self.std = torch.tensor(IMAGENET_STD, device=self.device)[None, :, None, None]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Patch features float [B, 384, 32, 32] (CPU) of images [B, 3, 512, 512] in [0, 1]."""
        resized = functional.interpolate(
            images.to(self.device),
            size=(DINO_INPUT_SIDE,) * 2,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        tokens = self.model.forward_features((resized - self.mean) / self.std)
        patches = tokens[:, -DINO_GRID * DINO_GRID :]
        grid = patches.reshape(len(images), DINO_GRID, DINO_GRID, DINO_DIM).permute(0, 3, 1, 2)
        features: torch.Tensor = grid.float().cpu()
        return features


def extract_dino(split: str, limit: int | None = None) -> int:
    """Fill the DINO cache of `split` (all methods share it); returns images cached."""
    torch.set_num_threads(THREAD_LIMIT)
    records = load_records(split)
    if limit is not None:
        records = records[:limit]
    done = load_dino_table(split, refresh=True)
    pending = [r.name for r in records if r.name not in done]
    print(f"dino {split}: {len(pending)} to do out of {len(records)}", flush=True)
    if not pending:
        return 0
    encoder = DinoEncoder()
    started = time.time()
    written = 0
    buffer_names: list[str] = []
    buffer: list[NDArray[np.float16]] = []
    batch_size = extraction_batch_size(CUDA_BATCH_SIZE, BATCH_SIZE)
    for start in range(0, len(pending), batch_size):
        names = pending[start : start + batch_size]
        images = torch.stack([load_image(split, name) for name in names])
        grid = encoder.encode(images).numpy().astype(np.float16)
        buffer_names.extend(names)
        buffer.extend(grid)
        written += len(names)
        if len(buffer_names) >= DINO_CHUNK_IMAGES:
            write_dino_chunk(split, buffer_names, np.stack(buffer))
            buffer_names, buffer = [], []
        if (start // batch_size + 1) % max(1, 10 * BATCH_SIZE // batch_size) == 0:
            print(
                f"  {written}/{len(pending)} ({written / (time.time() - started):.2f} img/s)",
                flush=True,
            )
    if buffer_names:
        write_dino_chunk(split, buffer_names, np.stack(buffer))
    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache DINOv2 patch features (team B2).")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(f"done, {extract_dino(args.split, args.limit)} images cached", flush=True)


if __name__ == "__main__":
    main()
