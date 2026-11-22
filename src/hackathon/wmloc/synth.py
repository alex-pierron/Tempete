"""LOCAL ONLY synthetic bench: stand-in for the task data while the gated dataset is unreachable.

We watermark a rectangle of a local photo with an injected `embed_fn` (Pixel Seal through
`local_backend.PixelSealBackend.embed`), then simulate the four edit families of the task with
exact ground truth. Requires Pillow and OpenCV (not part of the runtime of the other modules).

WHAT THIS IS NOT: the organizers' generator. Real edits are done by generative models
(inpainting / outpainting networks), real rectangles and embedding procedure are unknown to us.
Our edits are crude (cv2 inpainting, foreign-pixel patches, hard seams), so absolute scores here
say little about the leaderboard; they are good for comparing localizers with each other and for
catching bugs. Assumptions to revisit when the train split arrives:

* `embed_mode="crop"`: the rectangle is cropped, watermarked as an image of its own (Pixel Seal
  resizes it to 256 internally) and pasted back. `"full"`: the whole image is watermarked and
  only the rectangle is kept. The two give different pattern scales; the bench supports both.
* Rectangle: area 15-100 % of the image, aspect <= 4:1, sides >= 96 px.
* Everything is quantized to 8 bits (PNG), as the real files are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hackathon.wmloc.constants import (
    DEFAULT_SEED,
    IMAGE_SIZE,
    METHOD_PIXELSEAL,
    PIXELSEAL_N_BITS,
)
from hackathon.wmloc.errors import ManifestError
from hackathon.wmloc.manifest import ManifestSample, bits_to_string, write_manifest

EDIT_OUTPAINTING = "outpainting"
EDIT_REMOVAL = "removal"
EDIT_ADDITION = "addition"
EDIT_TRANSPLANT = "transplant"
EDIT_TYPES: tuple[str, ...] = (EDIT_OUTPAINTING, EDIT_REMOVAL, EDIT_ADDITION, EDIT_TRANSPLANT)
EMBED_CROP = "crop"
EMBED_FULL = "full"
EMBED_MODES: tuple[str, ...] = (EMBED_CROP, EMBED_FULL)

RECT_MIN_AREA_FRACTION = 0.15
RECT_MAX_ASPECT = 4.0
RECT_MIN_SIDE = 96
OUTPAINT_MAX_AREA_FRACTION = 0.9
"""An outpainted image always leaves part of the canvas unwatermarked."""
HOLE_AREA_FRACTION_RANGE = (0.03, 0.40)
"""Edited object size as a fraction of the watermarked rectangle."""
ELONGATED_PROBABILITY = 0.4
ELONGATED_ASPECT_RANGE = (2.5, 7.0)
POLYGON_VERTICES = 14
POLYGON_RADIUS_JITTER = (0.65, 1.35)
INPAINT_RADIUS_PX = 5
BYTE_MAX = 255
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
RECT_TRIES = 50

EmbedFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""(images [B, 3, H, W] in [0, 1], message [n_bits]) -> watermarked images, same shape."""


@dataclass(frozen=True)
class SynthSample:
    """One synthetic image with its exact ground truth.

    Attributes:
        name: Unique name.
        edit: One of EDIT_TYPES.
        image: uint8 [3, 512, 512].
        mask512: uint8 [512, 512], 1 = watermarked pixel.
        message: float32 [n_bits] of 0/1.
        rect: (y0, x0, height, width) of the watermarked rectangle before editing.
    """

    name: str
    edit: str
    image: torch.Tensor
    mask512: torch.Tensor
    message: torch.Tensor
    rect: tuple[int, int, int, int]


def list_source_images(roots: Sequence[Path]) -> list[Path]:
    """Sorted image files found in `roots` (non-recursive).

    Raises:
        ManifestError: a root is missing or no image is found.
    """
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise ManifestError(f"source image folder not found: {root}")
        found.extend(p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES)
    if not found:
        raise ManifestError(f"no image in {[str(r) for r in roots]}")
    return found


def load_square_image(path: Path) -> torch.Tensor:
    """Center-crop to a square, resize to 512 (Lanczos), return float32 [3, 512, 512] in [0, 1]."""
    from PIL import Image

    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        side = min(rgb.size)
        left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
        square = rgb.crop((left, top, left + side, top + side)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        array = np.asarray(square, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / BYTE_MAX


def sample_rect(
    rng: np.random.Generator, max_area_fraction: float = 1.0
) -> tuple[int, int, int, int]:
    """Random (y0, x0, h, w) with the area, side and aspect limits of this module."""
    for _ in range(RECT_TRIES):
        height = int(rng.integers(RECT_MIN_SIDE, IMAGE_SIZE + 1))
        width = int(rng.integers(RECT_MIN_SIDE, IMAGE_SIZE + 1))
        area = height * width / IMAGE_SIZE**2
        aspect = max(height, width) / min(height, width)
        if RECT_MIN_AREA_FRACTION <= area <= max_area_fraction and aspect <= RECT_MAX_ASPECT:
            y0 = int(rng.integers(0, IMAGE_SIZE - height + 1))
            x0 = int(rng.integers(0, IMAGE_SIZE - width + 1))
            return y0, x0, height, width
    side = int(IMAGE_SIZE * max_area_fraction**0.5)
    return 0, 0, side, side


def rect_mask(rect: tuple[int, int, int, int]) -> np.ndarray:
    """Boolean [512, 512] mask of a rectangle."""
    y0, x0, height, width = rect
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[y0 : y0 + height, x0 : x0 + width] = True
    return mask


def sample_blob(rng: np.random.Generator, centre_yx: tuple[int, int], area_px: float) -> np.ndarray:
    """Random irregular or elongated object of roughly `area_px` pixels, boolean [512, 512]."""
    import cv2

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    centre = (int(centre_yx[1]), int(centre_yx[0]))  # cv2 wants (x, y)
    angle = float(rng.uniform(0.0, 180.0))
    if rng.random() < ELONGATED_PROBABILITY:
        aspect = float(rng.uniform(*ELONGATED_ASPECT_RANGE))
        major = (area_px * aspect / np.pi) ** 0.5
        axes = (max(2, int(major)), max(2, int(major / aspect)))
        cv2.ellipse(canvas, centre, axes, angle, 0.0, 360.0, 1, thickness=-1)
    else:
        radius = (area_px / np.pi) ** 0.5
        angles = np.linspace(0.0, 2.0 * np.pi, POLYGON_VERTICES, endpoint=False)
        radii = radius * rng.uniform(*POLYGON_RADIUS_JITTER, size=POLYGON_VERTICES)
        points = np.stack(
            [centre[0] + radii * np.cos(angles), centre[1] + radii * np.sin(angles)], axis=1
        ).astype(np.int32)
        cv2.fillPoly(canvas, [points], 1)
    return canvas.astype(bool)


def _watermark_rect(
    base: torch.Tensor,
    rect: tuple[int, int, int, int],
    message: torch.Tensor,
    embed_fn: EmbedFn,
    embed_mode: str,
) -> torch.Tensor:
    """Return `base` [3, 512, 512] with only `rect` watermarked, per `embed_mode`."""
    y0, x0, height, width = rect
    window = (slice(None), slice(y0, y0 + height), slice(x0, x0 + width))
    marked = base.clone()
    if embed_mode == EMBED_CROP:
        marked[window] = embed_fn(base[window][None], message)[0]
    elif embed_mode == EMBED_FULL:
        marked[window] = embed_fn(base[None], message)[0][window]
    else:
        raise ValueError(f"embed_mode must be one of {EMBED_MODES}, got {embed_mode!r}")
    return marked.clamp(0.0, 1.0)


def _quantize(image: torch.Tensor) -> torch.Tensor:
    """Float [0, 1] to uint8, rounding like an 8-bit PNG save."""
    return (image.clamp(0.0, 1.0) * BYTE_MAX).round().to(torch.uint8)


def _hole_blob(rng: np.random.Generator, rect: tuple[int, int, int, int]) -> np.ndarray:
    """An object centred inside `rect`, sized as a fraction of the rectangle."""
    y0, x0, height, width = rect
    fraction = float(rng.uniform(*HOLE_AREA_FRACTION_RANGE))
    centre = (int(rng.integers(y0, y0 + height)), int(rng.integers(x0, x0 + width)))
    return sample_blob(rng, centre, fraction * height * width)


def make_sample(
    name: str,
    edit: str,
    base: torch.Tensor,
    other: torch.Tensor,
    message: torch.Tensor,
    embed_fn: EmbedFn,
    rng: np.random.Generator,
    embed_mode: str = EMBED_CROP,
) -> SynthSample:
    """Build one watermarked-then-edited image and its exact 512x512 truth mask.

    Args:
        name: Sample name.
        edit: One of EDIT_TYPES.
        base: Source photo, float [3, 512, 512].
        other: Unrelated photo providing foreign content.
        message: float [n_bits] of 0/1 hidden in the rectangle.
        embed_fn: Watermark embedder.
        rng: Randomness of this sample (rectangle, object shape).
        embed_mode: EMBED_CROP or EMBED_FULL.

    Raises:
        ValueError: unknown edit or embed mode.
    """
    import cv2

    if edit not in EDIT_TYPES:
        raise ValueError(f"edit must be one of {EDIT_TYPES}, got {edit!r}")
    rect = sample_rect(rng, OUTPAINT_MAX_AREA_FRACTION if edit == EDIT_OUTPAINTING else 1.0)
    inside = rect_mask(rect)
    marked = _watermark_rect(base, rect, message, embed_fn, embed_mode)
    marked_u8, base_u8, other_u8 = _quantize(marked), _quantize(base), _quantize(other)
    inside_t = torch.from_numpy(inside)

    if edit == EDIT_OUTPAINTING:
        image = torch.where(inside_t, marked_u8, other_u8)
        truth = inside
    elif edit == EDIT_TRANSPLANT:
        blob = _hole_blob(rng, rect) & inside
        image = torch.where(torch.from_numpy(blob), marked_u8, other_u8)
        truth = blob
    else:
        blob = _hole_blob(rng, rect)
        scene = torch.where(inside_t, marked_u8, base_u8)
        if edit == EDIT_REMOVAL:
            bgr = np.ascontiguousarray(scene.permute(1, 2, 0).numpy()[:, :, ::-1])
            filled = cv2.inpaint(bgr, blob.astype(np.uint8), INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
            patch = torch.from_numpy(np.ascontiguousarray(filled[:, :, ::-1])).permute(2, 0, 1)
        else:
            patch = other_u8
        image = torch.where(torch.from_numpy(blob), patch, scene)
        truth = inside & ~blob
    return SynthSample(
        name=name,
        edit=edit,
        image=image,
        mask512=torch.from_numpy(truth.astype(np.uint8)),
        message=message,
        rect=rect,
    )


def generate_samples(
    sources: Sequence[Path],
    embed_fn: EmbedFn,
    n_per_edit: int,
    seed: int = DEFAULT_SEED,
    embed_mode: str = EMBED_CROP,
    n_bits: int = PIXELSEAL_N_BITS,
) -> Iterator[SynthSample]:
    """Yield `n_per_edit` samples of each edit type, deterministic given `seed`.

    Each sample draws its own generator from (seed, global index), so a subset regenerates
    identically. Base and foreign photos are chosen by a seeded permutation of `sources`.
    """
    if len(sources) < 2:
        raise ManifestError("need at least two source images (a base and a foreign one)")
    order = np.random.default_rng(seed).permutation(len(sources))
    index = 0
    for edit in EDIT_TYPES:
        for k in range(n_per_edit):
            rng = np.random.default_rng([seed, index])
            base_path = sources[int(order[index % len(sources)])]
            other_path = sources[int(order[(index + 1 + k) % len(sources)])]
            if other_path == base_path:
                other_path = sources[int(order[(index + 2) % len(sources)])]
            message = torch.from_numpy(rng.integers(0, 2, n_bits).astype(np.float32))
            yield make_sample(
                f"{edit}_{k:04d}",
                edit,
                load_square_image(base_path),
                load_square_image(other_path),
                message,
                embed_fn,
                rng,
                embed_mode,
            )
            index += 1


def write_synth_dataset(
    out_dir: Path, samples: Iterator[SynthSample] | Sequence[SynthSample]
) -> int:
    """Write PNG images, PNG masks (0/255) and `manifest.json`; return the number of samples."""
    from PIL import Image

    images_dir, masks_dir = out_dir / "images", out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records: list[ManifestSample] = []
    for sample in samples:
        array = sample.image.permute(1, 2, 0).numpy()
        Image.fromarray(array).save(images_dir / f"{sample.name}.png")
        Image.fromarray(sample.mask512.numpy() * BYTE_MAX).save(masks_dir / f"{sample.name}.png")
        records.append(
            ManifestSample(
                name=sample.name,
                image=f"images/{sample.name}.png",
                method=METHOD_PIXELSEAL,
                message=bits_to_string(sample.message),
                mask512=f"masks/{sample.name}.png",
                edit=sample.edit,
            )
        )
    write_manifest(out_dir / "manifest.json", records)
    return len(records)
