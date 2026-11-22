"""Image input/output: lossless-by-default load and save of [3, H, W] float tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
"""Extensions picked up by `list_images` (compared case-insensitively)."""

JPEG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg"})
JPEG_SAVE_QUALITY: int = 95
"""Quality used when `save_image` targets a JPEG file."""

UINT8_MAX: int = 255
RGB_CHANNELS: int = 3


class ImageIOError(ValueError):
    """Raised when an image file cannot be decoded or a tensor cannot be encoded."""


def load_image(path: str | Path) -> torch.Tensor:
    """Read an image file as an RGB float32 tensor in [0, 1].

    EXIF orientation is applied, alpha channels are dropped and grayscale or palette
    images are expanded to three channels.

    Args:
        path: Image file path.

    Returns:
        Tensor of shape [3, H, W], dtype float32, values in [0, 1].

    Raises:
        FileNotFoundError: `path` does not exist.
        ImageIOError: The file exists but is not a decodable image.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    try:
        with Image.open(file_path) as handle:
            rgb = ImageOps.exif_transpose(handle).convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
    except (OSError, Image.DecompressionBombError) as error:
        # DecompressionBombError is not an OSError, so it would otherwise escape as a traceback.
        raise ImageIOError(f"cannot decode image {file_path}: {error}") from error
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / UINT8_MAX
    return tensor.contiguous()


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Quantise a [3, H, W] float tensor in [0, 1] to a [H, W, 3] uint8 array.

    Values are clamped then rounded to nearest (never truncated).

    Raises:
        ImageIOError: `image` is not a [3, H, W] tensor.
    """
    if image.ndim != 3 or image.shape[0] != RGB_CHANNELS:
        raise ImageIOError(f"expected an image of shape [3, H, W], got {tuple(image.shape)}")
    scaled = image.detach().to("cpu", torch.float32).clamp(0.0, 1.0) * UINT8_MAX
    array: np.ndarray = scaled.round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    return array


def save_image(img: torch.Tensor, path: str | Path) -> None:
    """Write a [3, H, W] float tensor in [0, 1] to disk.

    The format follows the extension: JPEG (quality 95) for .jpg/.jpeg, otherwise the
    format Pillow infers (PNG, lossless, when the extension is .png). Parent
    directories are created.

    Args:
        img: Tensor of shape [3, H, W], values clamped to [0, 1] before quantisation.
        path: Destination file.

    Raises:
        ImageIOError: `img` has the wrong shape or the extension is unsupported.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ImageIOError(f"unsupported output extension {suffix!r}; use one of {supported}")
    array = tensor_to_uint8(img)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(array, mode="RGB")
    if suffix in JPEG_EXTENSIONS:
        pil_image.save(file_path, format="JPEG", quality=JPEG_SAVE_QUALITY)
    else:
        pil_image.save(file_path)


def list_images(directory: str | Path, max_images: int | None = None) -> list[Path]:
    """List the image files of a directory (non recursive), sorted lexically.

    Args:
        directory: Folder to scan.
        max_images: Keep only the first `max_images` paths after sorting; None keeps all.

    Returns:
        Sorted paths whose suffix is png, jpg, jpeg or webp (case-insensitive).

    Raises:
        NotADirectoryError: `directory` is not a folder.
        ValueError: `max_images` is negative.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    if max_images is not None and max_images < 0:
        raise ValueError(f"max_images must be >= 0, got {max_images}")
    paths = sorted(
        entry for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths if max_images is None else paths[:max_images]
