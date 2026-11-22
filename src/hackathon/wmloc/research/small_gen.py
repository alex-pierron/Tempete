"""Synthetic composites at full resolution, built with the official embedders (w-loc-small).

One worker process writes, for one watermarking method, two kinds of material into a RAM folder:

* PAIRS `(clean, marked)`: a clean 512 px base and the same base watermarked as a whole. The
  trainer cuts real hole shapes (object removal / addition) or keeps real object shapes
  (transplantation) out of a pair on the fly, so one pair yields unlimited composites.
* RECTS: finished composites of the outpainting kind, where the bounding box of a real region
  shape is watermarked ON ITS OWN CROP (the embedder rescales its pattern to the crop) and pasted
  back inside the shape, with the official 128 grid label.

Bases are real TRAIN images of the OTHER two methods (same image domain as the test split; their
watermark is a different signal) and packed clean photographs. No test image is read here. Shapes come from the real train
masks of the OTHER two methods, so the real train images of this method stay an untouched
validation set: neither their pixels nor their mask shapes are ever used here.

A base is flipped or rotated BEFORE embedding only: a watermark never appears mirrored in the data.

Usage (kit environment):
    python -m hackathon.wmloc.research.small_gen --method stegastamp --worker 0 \
        --out-dir /dev/shm/wls --seconds 600
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from hackathon.wmloc.constants import (
    CELL_WATERMARK_FRACTION,
    GRID_SIZE,
    IMAGE_SIZE,
    KNOWN_METHODS,
    MAX_ASPECT_RATIO_BY_METHOD,
)
from hackathon.wmloc.research.data import (
    SPLIT_TRAIN,
    Record,
    data_root,
    load_records,
    split_dir,
)
from hackathon.wmloc.research.errors import ResearchError

CELL = IMAGE_SIZE // GRID_SIZE
CLEAN_CONTAINER = "team_b2/clean_images.npz"
REAL_BASE_PROBABILITY = 0.6
"""Share of bases drawn from the train images of the other methods (test-like image domain)."""
RECTS_PER_PAIR = 3
"""A pair is reused with many shapes by the trainer, a rectangle composite is not."""
SCALE_RANGE: tuple[float, float] = (0.6, 1.5)
SHIFT_FRACTION = 0.3
MIN_AREA_FRACTION = 0.004
MAX_AREA_FRACTION = 0.45
MIN_CROP_SIDE = 16
MAX_SHAPE_TRIES = 12
SYMMETRIES = 8
MASK_THRESHOLD = 127
BITS_BY_METHOD: dict[str, int] = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
PROGRESS_SECONDS = 60.0
BYTE_MAX = 255


class SmallGenError(ResearchError):
    """No base image, no shape, or an embedder failure that cannot be skipped."""


def symmetry(array: NDArray[Any], element: int) -> NDArray[Any]:
    """One of the 8 symmetries of the square applied to the two leading axes."""
    turned = np.rot90(array, element % 4, axes=(0, 1))
    return np.ascontiguousarray(turned[:, ::-1] if element >= 4 else turned)


def load_mask512(name: str) -> NDArray[np.bool_]:
    """Ground-truth 512 px mask of a train image (True = watermarked)."""
    with Image.open(split_dir(SPLIT_TRAIN) / "masks" / f"{name}.png") as handle:
        return np.asarray(handle.convert("L")) > MASK_THRESHOLD


def other_method_records(method: str, split: str) -> list[Record]:
    """Records of `split` whose method is NOT `method`."""
    return [r for r in load_records(split) if r.method != method]


def shape_bank(method: str) -> tuple[list[NDArray[np.bool_]], list[NDArray[np.bool_]]]:
    """Real 512 px shapes of the other methods: (holes of the full regime, regions of the others)."""
    holes: list[NDArray[np.bool_]] = []
    regions: list[NDArray[np.bool_]] = []
    for record in other_method_records(method, SPLIT_TRAIN):
        mask = load_mask512(record.name)
        if record.full_watermark:
            holes.append(~mask)
        else:
            regions.append(mask)
    if not holes or not regions:
        raise SmallGenError("no real shape for the other methods")
    return holes, regions


def augment_shape(shape: NDArray[np.bool_], rng: np.random.Generator) -> NDArray[np.bool_]:
    """Random symmetry, isotropic scale and shift of a 512 px shape, area kept in range."""
    moved = symmetry(shape, int(rng.integers(SYMMETRIES)))
    base_area = float(moved.mean())
    centre = (IMAGE_SIZE - 1) / 2.0
    for _ in range(MAX_SHAPE_TRIES):
        scale = float(rng.uniform(*SCALE_RANGE))
        shift = rng.uniform(-SHIFT_FRACTION, SHIFT_FRACTION, size=2) * IMAGE_SIZE
        matrix = np.eye(2) / scale
        offset = np.array([centre, centre]) - matrix @ (np.array([centre, centre]) + shift)
        warped = ndimage.affine_transform(moved.astype(np.float32), matrix, offset=offset, order=1)
        candidate = warped > 0.5
        area = float(candidate.mean())
        if MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION and area > 0.25 * base_area:
            return np.asarray(candidate, dtype=np.bool_)
    return np.asarray(moved, dtype=np.bool_)


def official_label(mask512: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """512 -> 128 with the official rule (a cell is marked when at least half of it is)."""
    blocks = mask512.reshape(GRID_SIZE, CELL, GRID_SIZE, CELL).mean(axis=(1, 3))
    return np.asarray(blocks >= CELL_WATERMARK_FRACTION, dtype=np.uint8)


class BasePool:
    """Clean-enough 512 px bases: train images of the other methods and packed photographs."""

    def __init__(self, method: str, max_real: int | None, max_photos: int | None) -> None:
        self.real_paths: list[Path] = []
        # TRAIN split only: no test image is ever read by the generator (team rule).
        for record in other_method_records(method, SPLIT_TRAIN):
            self.real_paths.append(split_dir(SPLIT_TRAIN) / "images" / f"{record.name}.png")
        if max_real is not None:
            self.real_paths = self.real_paths[:max_real]
        self.photos: list[NDArray[np.uint8]] = []
        container = data_root() / CLEAN_CONTAINER
        if container.is_file() and max_photos != 0:
            with np.load(container) as archive:
                keys = archive.files if max_photos is None else archive.files[:max_photos]
                self.photos = [np.asarray(archive[key], dtype=np.uint8) for key in keys]
        if not self.real_paths and not self.photos:
            raise SmallGenError("no base image available")

    def draw(self, rng: np.random.Generator) -> NDArray[np.uint8]:
        """A random 512 x 512 x 3 uint8 base, with a random symmetry."""
        use_real = bool(self.real_paths) and (
            not self.photos or rng.random() < REAL_BASE_PROBABILITY
        )
        if use_real:
            path = self.real_paths[int(rng.integers(len(self.real_paths)))]
            with Image.open(path) as handle:
                array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        else:
            photo = self.photos[int(rng.integers(len(self.photos)))]
            height, width = photo.shape[:2]
            top = int(rng.integers(0, height - IMAGE_SIZE + 1))
            left = int(rng.integers(0, width - IMAGE_SIZE + 1))
            array = photo[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
        return symmetry(array, int(rng.integers(SYMMETRIES)))


def compose_rect(
    clean: NDArray[np.uint8],
    bits: NDArray[np.uint8],
    region: NDArray[np.bool_],
    wrapper: Any,
    max_aspect: float,
) -> NDArray[np.uint8] | None:
    """Watermark the bounding-box crop of `region` on its own and paste it inside the region.

    Returns None when the box is too thin or too elongated for the method.
    """
    rows, columns = np.nonzero(region)
    if len(rows) == 0:
        return None
    top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
    short, long = sorted((int(bottom - top), int(right - left)))
    if short < MIN_CROP_SIDE or long > max_aspect * short:
        return None
    crop = np.ascontiguousarray(clean[top:bottom, left:right])
    try:
        marked = np.asarray(wrapper.embed(crop, bits.tolist()), dtype=np.uint8)
    except (RuntimeError, ValueError) as error:
        # An embedder may refuse a very small or odd crop: the caller redraws.
        print(f"  embed refused a {crop.shape} crop: {type(error).__name__}", flush=True)
        return None
    if marked.shape != crop.shape:
        return None
    image = clean.copy()
    inside = region[top:bottom, left:right, None]
    image[top:bottom, left:right] = np.where(inside, marked, crop)
    return image


def zoomed_pair(
    clean: NDArray[np.uint8],
    bits: NDArray[np.uint8],
    wrapper: Any,
    rng: np.random.Generator,
    zoom_range: tuple[float, float],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """(clean, marked) 512 px crops of a base watermarked at a LARGER, possibly non-square size.

    The decode margin of the REAL full-regime images peaks when the image is shrunk by 0.8 to 1.0
    before decoding, where embed() at 512 needs 1.05 to 1.1 (x-loc-unet probe on TrustMark): the
    real pattern is magnified, as if the 512 image were a crop of a larger watermarked original.
    Each axis is magnified independently by a draw in `zoom_range`, the base upsampled (bicubic),
    watermarked, then the SAME random 512 window is cut from the clean and the marked images.
    """
    low, high = zoom_range
    width = max(IMAGE_SIZE, round(IMAGE_SIZE * rng.uniform(low, high)))
    height = max(IMAGE_SIZE, round(IMAGE_SIZE * rng.uniform(low, high)))
    large = np.asarray(
        Image.fromarray(clean).resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8
    )
    marked = np.asarray(wrapper.embed(large, bits.tolist()), dtype=np.uint8)
    top = int(rng.integers(0, height - IMAGE_SIZE + 1))
    left = int(rng.integers(0, width - IMAGE_SIZE + 1))
    window = (slice(top, top + IMAGE_SIZE), slice(left, left + IMAGE_SIZE))
    return large[window], marked[window]


def weaken(
    clean: NDArray[np.uint8], marked: NDArray[np.uint8], strength: float
) -> NDArray[np.uint8]:
    """clean + strength x (marked - clean), rounded and clipped to uint8.

    The real full-regime watermark is weaker than embed() at strength 1 (median ~0.4 for
    StegaStamp, x-review-loc, content-matched): pairs drawn at a random strength cover it.
    """
    residual = marked.astype(np.float32) - clean.astype(np.float32)
    weakened = clean.astype(np.float32) + np.float32(strength) * residual
    return np.clip(np.rint(weakened), 0, BYTE_MAX).astype(np.uint8)


def generate(
    method: str,
    worker: int,
    out_dir: Path,
    seconds: float,
    max_pairs: int,
    max_rects: int,
    device: str,
    max_real: int | None = None,
    max_photos: int | None = None,
    pair_zoom: tuple[float, float] | None = None,
    pair_strength: tuple[float, float] | None = None,
) -> dict[str, int]:
    """Fill the worker's containers until the deadline or the caps; returns the counts written.

    With `pair_zoom`, every (clean, marked) pair comes from `zoomed_pair` (magnified pattern);
    with `pair_strength`, its residual is then scaled by a uniform draw in that range (`weaken`).
    """
    from hackathon.wmloc.research.kit import load_wrapper  # kit environment only

    started = time.time()
    rng = np.random.default_rng(1_000_003 * (KNOWN_METHODS.index(method) + 1) + worker)
    wrapper = load_wrapper(method, device)
    session = getattr(wrapper, "sess", None)
    if session is not None:
        print(f"onnx providers: {session.get_providers()}", flush=True)
    pool = BasePool(method, max_real, max_photos)
    _, regions = shape_bank(method)
    n_bits = BITS_BY_METHOD[method]
    max_aspect = MAX_ASPECT_RATIO_BY_METHOD[method]
    print(
        f"{method} worker {worker}: {len(pool.real_paths)} real bases, {len(pool.photos)} photos, "
        f"{len(regions)} region shapes, ready in {time.time() - started:.0f}s",
        flush=True,
    )
    # Preallocated at the caps (pages are committed lazily): no second copy when writing.
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    clean_store = np.empty((max_pairs, *image_shape), dtype=np.uint8)
    marked_store = np.empty((max_pairs, *image_shape), dtype=np.uint8)
    rect_store = np.empty((max_rects, *image_shape), dtype=np.uint8)
    label_store = np.empty((max_rects, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    pair_bits = np.empty((max_pairs, n_bits), dtype=np.uint8)
    rect_bits = np.empty((max_rects, n_bits), dtype=np.uint8)
    n_pairs = 0
    n_rects = 0
    last_report = time.time()
    deadline = started + seconds
    while time.time() < deadline:
        pairs_full = n_pairs >= max_pairs
        rects_full = n_rects >= max_rects
        if pairs_full and rects_full:
            break
        want_pair = not pairs_full and (rects_full or n_rects >= RECTS_PER_PAIR * n_pairs)
        clean = pool.draw(rng)
        bits = rng.integers(0, 2, n_bits).astype(np.uint8)
        if want_pair and pair_zoom is not None:
            clean_store[n_pairs], marked_store[n_pairs] = zoomed_pair(
                clean, bits, wrapper, rng, pair_zoom
            )
            pair_bits[n_pairs] = bits
            n_pairs += 1
        elif want_pair:
            clean_store[n_pairs] = clean
            marked_store[n_pairs] = np.asarray(wrapper.embed(clean, bits.tolist()), dtype=np.uint8)
            pair_bits[n_pairs] = bits
            n_pairs += 1
        if want_pair and pair_strength is not None:
            marked_store[n_pairs - 1] = weaken(
                clean_store[n_pairs - 1],
                marked_store[n_pairs - 1],
                float(rng.uniform(*pair_strength)),
            )
        else:
            region = augment_shape(regions[int(rng.integers(len(regions)))], rng)
            image = compose_rect(clean, bits, region, wrapper, max_aspect)
            if image is None:
                continue
            rect_store[n_rects] = image
            label_store[n_rects] = official_label(region)
            rect_bits[n_rects] = bits
            n_rects += 1
        if time.time() - last_report > PROGRESS_SECONDS:
            last_report = time.time()
            print(
                f"  {n_pairs} pairs, {n_rects} rects, "
                f"{(n_pairs + n_rects) / (time.time() - started):.1f} embeds/s",
                flush=True,
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.uint8]] = {
        "clean": clean_store[:n_pairs],
        "marked": marked_store[:n_pairs],
        "pair_bits": pair_bits[:n_pairs],
        "rect": rect_store[:n_rects],
        "rect_label": label_store[:n_rects],
        "rect_bits": rect_bits[:n_rects],
    }
    for key, array in arrays.items():
        temporary = out_dir / f"{method}_w{worker}_{key}.{os.getpid()}.tmp.npy"
        np.save(temporary, array)
        os.replace(temporary, out_dir / f"{method}_w{worker}_{key}.npy")
    counts = {"pairs": n_pairs, "rects": n_rects}
    print(f"{method} worker {worker}: wrote {counts} in {time.time() - started:.0f}s", flush=True)
    return counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Synthetic full-resolution composites.")
    parser.add_argument("--method", choices=KNOWN_METHODS, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--max-pairs", type=int, default=1500)
    parser.add_argument("--max-rects", type=int, default=4500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-real", type=int)
    parser.add_argument("--max-photos", type=int)
    parser.add_argument(
        "--pair-zoom",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="magnify the watermark pattern of the pairs by a per-axis draw in [LOW, HIGH]",
    )
    parser.add_argument(
        "--pair-strength",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="scale the watermark residual of the pairs by a uniform draw in [LOW, HIGH]",
    )
    args = parser.parse_args()
    generate(
        args.method,
        args.worker,
        args.out_dir,
        args.seconds,
        args.max_pairs,
        args.max_rects,
        args.device,
        args.max_real,
        args.max_photos,
        None if args.pair_zoom is None else (float(args.pair_zoom[0]), float(args.pair_zoom[1])),
        None
        if args.pair_strength is None
        else (float(args.pair_strength[0]), float(args.pair_strength[1])),
    )


if __name__ == "__main__":
    main()
