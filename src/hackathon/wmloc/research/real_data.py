"""Cache the real splits (train / test) as arrays, with the same extras as the synthetic sets.

Per method and split, in `<out>/<split>/<method>/`: `names.json`, `images.npy`, `messages.npy`,
`extras_maps.npy`, `extras_scalars.npy` and, for train, `masks128.npy` (official grid, taken from
the kit's `masks_128.npz`) and `full.npy` (the `full_watermark` flag). Run in the kit venv:

    python -m hackathon.wmloc.research.real_data --split train --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image

from hackathon.wmloc.research.extras import N_MAP_CHANNELS, N_SCALARS, compute_extras
from hackathon.wmloc.research.kit import (
    GRID_SIZE,
    IMAGE_SIZE,
    METHODS,
    N_BITS_BY_METHOD,
    load_wrapper,
    test_dir,
    train_dir,
)
from hackathon.wmloc.research.synth_data import DEVICE_BY_METHOD

SPLITS: tuple[str, ...] = ("train", "test")
EXTRAS_CHUNK = 16


def build_split(split: str, method: str, out_root: Path) -> None:
    """Write the cache of one (split, method) pair; skipped when `names.json` exists."""
    out_dir = out_root / split / method
    if (out_dir / "names.json").exists():
        print(f"{split}/{method} cached", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    root = train_dir() if split == "train" else test_dir()
    records = [m for m in json.loads((root / "metadata.json").read_text()) if m["method"] == method]
    n = len(records)
    n_bits = N_BITS_BY_METHOD[method]
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    messages = np.zeros((n, n_bits), dtype=np.uint8)
    for i, record in enumerate(records):
        images[i] = np.asarray(
            Image.open(root / "images" / f"{record['image_name']}.png").convert("RGB")
        )
        messages[i] = np.array([int(c) for c in record["message"]], dtype=np.uint8)
    wrapper = load_wrapper(method, DEVICE_BY_METHOD[method])
    maps = np.zeros((n, N_MAP_CHANNELS, 8, 8), dtype=np.float16)
    scalars = np.zeros((n, N_SCALARS), dtype=np.float32)
    started = time.time()
    for start in range(0, n, EXTRAS_CHUNK):
        end = min(n, start + EXTRAS_CHUNK)
        maps[start:end], scalars[start:end] = compute_extras(
            wrapper, method, images[start:end], messages[start:end]
        )
        print(f"{split}/{method} extras {end}/{n} ({time.time() - started:.0f}s)", flush=True)
    names = [r["image_name"] for r in records]
    if split == "train":
        archive = np.load(root / "masks_128.npz")
        by_name = dict(zip(archive["names"].tolist(), archive["masks"]))
        masks = np.stack([by_name[name] for name in names]).astype(np.uint8)
        assert masks.shape[1:] == (GRID_SIZE, GRID_SIZE)
        np.save(out_dir / "masks128.npy", masks)
        np.save(out_dir / "full.npy", np.array([bool(r["full_watermark"]) for r in records]))
    np.save(out_dir / "images.npy", images)
    np.save(out_dir / "messages.npy", messages)
    np.save(out_dir / "extras_maps.npy", maps)
    np.save(out_dir / "extras_scalars.npy", scalars)
    (out_dir / "names.json").write_text(json.dumps(names))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    args = parser.parse_args(argv)
    for method in args.methods:
        build_split(args.split, method, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
