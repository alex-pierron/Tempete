"""Sample manifest: our own neutral description of a split (synthetic, train or test).

The real dataset layout is only known once the starter kit is available, so every command works
from this small JSON instead: a list of records {name, image, method, message, mask512?, edit?}.
The synthetic bench writes it, and a ~20 line adapter will write it from the kit's dataset
objects (see docs/hackathon/wmloc.md). Paths are relative to the manifest's directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.constants import IMAGE_SIZE, KNOWN_METHODS
from hackathon.wmloc.errors import ManifestError
from hackathon.wmloc.metric import downsample_mask_to_grid

MANIFEST_VERSION = 1
RGB_CHANNELS = 3
BYTE_MAX = 255.0
MASK_FILE_ON_VALUE = 255


@dataclass(frozen=True)
class ManifestSample:
    """One image of a split.

    Attributes:
        name: Unique image name (the key of the submission).
        image: Image path relative to the manifest directory.
        method: One of KNOWN_METHODS.
        message: Embedded message as a string of '0'/'1' characters.
        mask512: Ground-truth mask PNG (0/255) relative path, None on the test split.
        edit: Edit type label (synthetic data only), else None.
        group: Free label to break scores down by (real data: whether the whole image was
            watermarked then partly restored), else None.
    """

    name: str
    image: str
    method: str
    message: str
    mask512: str | None = None
    edit: str | None = None
    group: str | None = None


def parse_message_bits(message: str) -> torch.Tensor:
    """Turn a '0101...' string into a float32 tensor [n_bits] of 0.0 / 1.0.

    Raises:
        ManifestError: empty string or a character other than 0 and 1.
    """
    text = message.strip()
    if not text or set(text) - {"0", "1"}:
        raise ManifestError(f"message must be a non-empty string of 0/1, got {message[:20]!r}")
    return torch.tensor([float(char) for char in text], dtype=torch.float32)


def bits_to_string(bits: torch.Tensor) -> str:
    """Inverse of `parse_message_bits` (any tensor of 0/1 values)."""
    return "".join("1" if value > 0.5 else "0" for value in bits.flatten().tolist())


def write_manifest(path: Path, samples: list[ManifestSample]) -> None:
    """Write the manifest as JSON (creates the directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "samples": [s.__dict__ for s in samples]}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def read_manifest(path: Path) -> list[ManifestSample]:
    """Load and validate a manifest.

    Raises:
        ManifestError: missing file, bad JSON, unknown method, duplicate names, missing image.
    """
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        records = payload["samples"]
        samples = [ManifestSample(**record) for record in records]
    except (ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"malformed manifest {path}: {exc}") from exc
    names = [sample.name for sample in samples]
    if len(set(names)) != len(names):
        raise ManifestError(f"{path}: duplicate image names")
    for sample in samples:
        if sample.method not in KNOWN_METHODS:
            raise ManifestError(f"{sample.name}: unknown method {sample.method!r}")
        if not (path.parent / sample.image).is_file():
            raise ManifestError(f"{sample.name}: image not found: {path.parent / sample.image}")
    return samples


def read_image_tensor(path: Path) -> torch.Tensor:
    """Read an RGB image as float32 [3, H, W] in [0, 1] (Pillow imported lazily).

    Raises:
        ManifestError: unreadable file or wrong size.
    """
    from PIL import Image

    try:
        with Image.open(path) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    except OSError as exc:
        raise ManifestError(f"cannot read image {path}: {exc}") from exc
    if array.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ManifestError(f"{path}: expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {array.shape[:2]}")
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / BYTE_MAX


def read_truth_mask128(path: Path) -> torch.Tensor:
    """Read a 512x512 mask PNG (0/255) and reduce it to the 128 grid with the task rule."""
    from PIL import Image

    try:
        with Image.open(path) as handle:
            array = np.asarray(handle.convert("L"), dtype=np.uint8).copy()
    except OSError as exc:
        raise ManifestError(f"cannot read mask {path}: {exc}") from exc
    binary = torch.from_numpy(array == MASK_FILE_ON_VALUE).to(torch.uint8)
    if binary.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ManifestError(f"{path}: expected a {IMAGE_SIZE}x{IMAGE_SIZE} mask")
    return downsample_mask_to_grid(binary)


GROUP_FULL_WATERMARK = "full_watermark"
GROUP_RECT_WATERMARK = "rect_watermark"
METADATA_FILE = "metadata.json"


def manifest_from_release(split_dir: Path) -> list[ManifestSample]:
    """Build manifest samples from an official release split (`<split>/metadata.json`).

    The official layout (starter kit `wmloc/data.py`): `images/<name>.png`, `metadata.json`
    with `image_name`, `method`, `message` (and `full_watermark` on train), and on train
    `masks/<name>.png` (255 = watermarked). Names are the file names without `.png`.

    Raises:
        ManifestError: missing or malformed metadata, or an image/mask that is not on disk.
    """
    meta_path = split_dir / METADATA_FILE
    if not meta_path.is_file():
        raise ManifestError(f"metadata not found: {meta_path}")
    has_masks = (split_dir / "masks").is_dir()
    samples: list[ManifestSample] = []
    try:
        records: list[dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
        for record in records:
            name = str(record["image_name"])
            group = None
            if "full_watermark" in record:
                group = GROUP_FULL_WATERMARK if record["full_watermark"] else GROUP_RECT_WATERMARK
            samples.append(
                ManifestSample(
                    name=name,
                    image=f"images/{name}.png",
                    method=str(record["method"]),
                    message=str(record["message"]),
                    mask512=f"masks/{name}.png" if has_masks else None,
                    group=group,
                )
            )
    except (ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"malformed {meta_path}: {exc}") from exc
    for sample in samples:
        for relative in (sample.image, sample.mask512):
            if relative is not None and not (split_dir / relative).is_file():
                raise ManifestError(f"{sample.name}: file not found: {split_dir / relative}")
    return samples
