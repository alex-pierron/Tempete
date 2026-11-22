"""Synthetic training data that mimics the two real regimes, with the OFFICIAL embedders.

Regimes (50/50 in the real train, every method):
- `full`: the whole image is watermarked, then a region (rectangle, blob, silhouette) is restored
  to the original clean pixels. Truth = everything except the hole.
- `region`: only a region is watermarked. A rectangle is watermarked as an image of its own (crop
  mode, as the organizers do: decoding the exact crop works, the whole image does not); a blob or a
  silhouette is cut from a whole-image embedding (hypothesis, see report).

Outputs, in `--out` (one folder per method): `images.npy` uint8 [N, 512, 512, 3], `masks128.npy`
uint8 [N, 128, 128], `meta.json`, `extras_maps.npy` float16 [N, 2, 8, 8], `extras_scalars.npy`
float32 [N, 2]. Deterministic given the seed; `--max-seconds` stops cleanly after a time budget.

Run in the kit venv (needs TrustMark and onnxruntime):
    python -m hackathon.wmloc.research.synth_data --method pixelseal --n 1500 --out DIR
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
from numpy.typing import NDArray
from PIL import Image

from hackathon.wmloc.research.extras import N_MAP_CHANNELS, N_SCALARS, compute_extras
from hackathon.wmloc.research.kit import (
    GRID_SIZE,
    IMAGE_SIZE,
    METHODS,
    N_BITS_BY_METHOD,
    data_dir,
    load_wrapper,
)
from hackathon.wmloc.research.shapes import (
    KIND_BLOB,
    KIND_POLYGON,
    KIND_RECT,
    RegionSpec,
    downsample_mask,
    sample_region,
)

DEFAULT_SEED = 2026
"""Seed of the synthetic set; sample i uses the generator seeded with [seed, method_index, i]."""
FULL_REGIME_PROBABILITY = 0.5
FULL_KIND_PROBABILITIES = {KIND_RECT: 0.25, KIND_BLOB: 0.45, KIND_POLYGON: 0.30}
REGION_KIND_PROBABILITIES = {KIND_RECT: 0.60, KIND_BLOB: 0.20, KIND_POLYGON: 0.20}
MAX_RECT_ASPECT_BY_METHOD = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}
SOURCE_DIRS: tuple[str, ...] = ("coco_val_subset", "div2k_valid", "kodak")
SOURCE_SUFFIXES = (".png", ".jpg", ".jpeg")
CROP_SCALE_RANGE = (0.35, 1.0)
"""Random square crop side as a fraction of the shorter source side (content scale variety)."""
DEFAULT_GROUP_SIZE = 3
PROGRESS_EVERY = 10
MIN_EMBED_SIDE = 8
REGIME_FULL = "full"
REGIME_REGION = "region"
DEVICE_BY_METHOD = {"pixelseal": "mps", "trustmark": "cpu", "stegastamp": "cpu"}

UInt8Array = NDArray[np.uint8]


def list_sources() -> list[Path]:
    """Source photo paths under `<data_dir>/..` (the repo's `data/` sibling folders).

    `data_dir()` is `<repo>/data/hackathon`; the photo folders live in `<repo>/data/`.
    """
    root = data_dir().parent
    found: list[Path] = []
    for name in SOURCE_DIRS:
        folder = root / name
        if folder.is_dir():
            found.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in SOURCE_SUFFIXES))
    if len(found) < 2:
        raise FileNotFoundError(f"no source photos under {root} ({SOURCE_DIRS})")
    return found


def load_source(path: Path, rng: np.random.Generator) -> UInt8Array:
    """Random square crop of a photo, resized to 512x512 (Lanczos), uint8 [512, 512, 3]."""
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        short = min(rgb.size)
        side = max(64, int(short * rng.uniform(*CROP_SCALE_RANGE)))
        left = int(rng.integers(0, rgb.width - side + 1))
        top = int(rng.integers(0, rgb.height - side + 1))
        crop = rgb.crop((left, top, left + side, top + side)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        return np.asarray(crop, dtype=np.uint8).copy()


def _embed_rect_crop(
    wrapper: object, clean: UInt8Array, region: RegionSpec, bits: NDArray[np.int64]
) -> UInt8Array:
    """Watermark the bounding box of a rectangle region as an image of its own (crop mode)."""
    y0, y1, x0, x1 = region.bbox
    out = clean.copy()
    out[y0:y1, x0:x1] = wrapper.embed(  # type: ignore[attr-defined]
        np.ascontiguousarray(clean[y0:y1, x0:x1]), bits
    )
    return out


def make_group(
    wrapper: object, method: str, sources: list[Path], seed: int, group: int, group_size: int
) -> list[tuple[UInt8Array, UInt8Array, NDArray[np.uint8], dict[str, object]]]:
    """`group_size` samples sharing one clean photo, one message and one whole-image embedding.

    Sharing the (expensive) whole-image embedding divides the generation cost by up to
    `group_size`; samples differ by regime, region shape and position. Deterministic given
    (seed, method, group).

    Returns:
        List of (image uint8 [512, 512, 3], truth128 uint8 [128, 128], bits uint8, info).
    """
    method_index = METHODS.index(method)
    rng = np.random.default_rng([seed, method_index, group])
    bits = rng.integers(0, 2, N_BITS_BY_METHOD[method]).astype(np.int64)
    clean = load_source(sources[int(rng.integers(0, len(sources)))], rng)
    max_aspect = MAX_RECT_ASPECT_BY_METHOD[method]
    marked: UInt8Array | None = None
    samples: list[tuple[UInt8Array, UInt8Array, NDArray[np.uint8], dict[str, object]]] = []
    for _ in range(group_size):
        is_full = rng.random() < FULL_REGIME_PROBABILITY
        probabilities = FULL_KIND_PROBABILITIES if is_full else REGION_KIND_PROBABILITIES
        region = sample_region(rng, probabilities, max_aspect)
        needs_whole = is_full or region.kind != KIND_RECT
        if needs_whole and marked is None:
            marked = wrapper.embed(clean, bits)  # type: ignore[attr-defined]
        if is_full:
            assert marked is not None
            image = marked.copy()
            image[region.mask] = clean[region.mask]
            truth = downsample_mask(~region.mask)
            info: dict[str, object] = {
                "regime": REGIME_FULL,
                "kind": region.kind,
                "area": float(1.0 - region.mask.mean()),
            }
        else:
            if region.kind == KIND_RECT:
                image = _embed_rect_crop(wrapper, clean, region, bits)
            else:
                assert marked is not None
                image = clean.copy()
                image[region.mask] = marked[region.mask]
            truth = downsample_mask(region.mask)
            info = {"regime": REGIME_REGION, "kind": region.kind, "area": float(region.mask.mean())}
        samples.append((image, truth, bits.astype(np.uint8), info))
    return samples


def generate(
    method: str, n_samples: int, out_dir: Path, seed: int, group_size: int, max_seconds: float
) -> None:
    """Generate `n_samples` samples (or as many as fit in `max_seconds`) into `out_dir`.

    Idempotent: a folder that already holds `meta.json` is left untouched; the data files are
    renamed into place only at the end, so a crash never leaves a half-written set that looks
    complete.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final_meta = out_dir / "meta.json"
    if final_meta.exists():
        print(f"{out_dir} already complete, nothing to do", flush=True)
        return
    wrapper = load_wrapper(method, DEVICE_BY_METHOD[method])
    sources = list_sources()
    n_bits = N_BITS_BY_METHOD[method]
    images = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    masks = np.zeros((n_samples, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    maps = np.zeros((n_samples, N_MAP_CHANNELS, 8, 8), dtype=np.float16)
    scalars = np.zeros((n_samples, N_SCALARS), dtype=np.float32)
    messages = np.zeros((n_samples, n_bits), dtype=np.uint8)
    infos: list[dict[str, object]] = []
    started = time.time()
    done = 0
    group = 0
    while done < n_samples and time.time() - started < max_seconds:
        for image, truth, bits, info in make_group(
            wrapper, method, sources, seed, group, group_size
        ):
            if done >= n_samples:
                break
            images[done], masks[done], messages[done] = image, truth, bits
            infos.append(info)
            m, s = compute_extras(wrapper, method, image[None], bits[None])
            maps[done], scalars[done] = m[0], s[0]
            done += 1
        group += 1
        if group % PROGRESS_EVERY == 0:
            print(
                f"{method} {done}/{n_samples} ({done / (time.time() - started):.2f} img/s)",
                flush=True,
            )
    np.save(out_dir / "images.npy", images[:done])
    np.save(out_dir / "masks128.npy", masks[:done])
    np.save(out_dir / "extras_maps.npy", maps[:done])
    np.save(out_dir / "extras_scalars.npy", scalars[:done])
    np.save(out_dir / "messages.npy", messages[:done])
    final_meta.write_text(json.dumps({"method": method, "seed": seed, "samples": infos}))
    print(f"{method}: done {done} samples in {time.time() - started:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args(argv)
    generate(args.method, args.n, args.out, args.seed, args.group_size, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
