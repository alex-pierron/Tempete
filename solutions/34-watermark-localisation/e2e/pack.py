"""X-LOC-B (JURECA, CPU): pack the real train and test images of one method into one uncompressed npz.

Preprocessing runs on CPU BEFORE the GPU task (team rule): the GPU task then reads one file instead of
decoding hundreds of PNG files. Output keys:
    train_images uint8 [N, 512, 512, 3], train_masks uint8 [N, 128, 128], train_full bool [N],
    train_messages uint8 [N, nbits], train_names <U12 [N]  (train metadata order of the method),
    test_images, test_messages, test_names                 (test metadata order of the method).

Usage: python pack.py <method> <out.npz>
Env: WMLOC_ROOT (dir with release/{train,test}).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ROOT = Path(os.environ["WMLOC_ROOT"])
IMAGE_SIDE = 512
DECODE_THREADS = 32


class PackError(RuntimeError):
    """The release folder does not hold what the packer expects."""


def load_rgb(path: Path) -> NDArray[np.uint8]:
    """One 512 x 512 x 3 uint8 image; raises PackError on another size."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if image.shape != (IMAGE_SIDE, IMAGE_SIDE, 3):
        raise PackError(f"{path.name}: shape {image.shape}, expected {IMAGE_SIDE} square")
    return image


def split_rows(split: str, method: str) -> list[dict[str, Any]]:
    """Metadata rows of one (split, method), in file order."""
    rows: list[dict[str, Any]] = json.loads((ROOT / "release" / split / "metadata.json").read_text())
    return [r for r in rows if r["method"] == method]


def pack_split(split: str, method: str) -> dict[str, NDArray[Any]]:
    """Arrays of one split, keys prefixed by the split name."""
    rows = split_rows(split, method)
    if not rows:
        raise PackError(f"{split}/{method}: no rows")
    folder = ROOT / "release" / split / "images"
    with ThreadPoolExecutor(DECODE_THREADS) as pool:
        images = np.stack(list(pool.map(lambda r: load_rgb(folder / f"{r['image_name']}.png"), rows)))
    out: dict[str, NDArray[Any]] = {
        f"{split}_images": images,
        f"{split}_names": np.array([r["image_name"] for r in rows]),
        f"{split}_messages": np.array([[int(c) for c in r["message"]] for r in rows], dtype=np.uint8),
    }
    if split == "train":
        archive = np.load(ROOT / "release" / "train" / "masks_128.npz")
        position = {str(n): i for i, n in enumerate(archive["names"])}
        out["train_masks"] = (archive["masks"][[position[r["image_name"]] for r in rows]] > 0).astype(np.uint8)
        out["train_full"] = np.array([bool(r["full_watermark"]) for r in rows])
    return out


def main() -> None:
    """Pack train and test of the method given on the command line."""
    method, target = sys.argv[1], Path(sys.argv[2])
    start = time.time()
    arrays = {**pack_split("train", method), **pack_split("test", method)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(target)
    shapes = {k: v.shape for k, v in arrays.items()}
    print(f"{method}: {shapes} -> {target} [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
