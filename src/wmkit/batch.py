"""Batched watermarking of image folders.

`Watermarker.embed` / `extract` already accept `[B, 3, H, W]` tensors; this module feeds
them whole batches instead of one image at a time, which is what lets a GPU (or the Apple
unified memory) amortise kernel launches and weight reads.

Design rules:
    * Images of different sizes are never resized: they are grouped by `(H, W)` and batched
      inside each group, because a tensor batch needs one shape.
    * Results do not depend on the batch size. Messages are drawn once, in sorted file
      order, before any batching, and every result is put back at its file's position.
    * Memory stays bounded by one batch: folder functions read only the image headers up
      front (to learn the shapes) and decode the pixels batch by batch.
    * The batch size is either explicit or derived from the free memory (`auto_batch_size`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from wmkit.io import ImageIOError, list_images, load_image, save_image
from wmkit.messages import bits_to_hex, hex_to_bits, random_messages
from wmkit.metrics import DETECTION_NEG_LOG10_P, bit_accuracy, neg_log10_pvalue
from wmkit.types import Watermarker

MIN_BATCH_SIZE: int = 1
"""Floor of the automatic batch size (an image that barely fits still runs)."""

MAX_BATCH_SIZE: int = 256
"""Ceiling of the automatic batch size: past this, throughput is flat while latency and
peak memory keep growing."""

FALLBACK_BATCH_SIZE: int = 8
"""Conservative automatic batch size when free memory cannot be measured (no psutil)."""

MEMORY_BUDGET_FRACTION: float = 0.25
"""Share of the free memory one batch may plan to use. Deliberately low: the model weights,
the allocator caches and the other processes of a shared machine are not counted."""

BYTES_PER_FLOAT32: int = 4
RGB_CHANNELS: int = 3
ACTIVATION_MEMORY_FACTOR: int = 96
"""Peak memory of a neural watermarker per pixel, in multiples of one float32 RGB pixel
(input, output, residual, and the full-resolution feature maps of a 64-channel U-Net style
embedder alive at the same time). An analytic, deliberately pessimistic estimate, not a
measurement: MPS exposes no peak-memory counter. Sanity check on a 16 GB Mac with 3.4 GiB
free: it picked 16 images at 256x256 (fast), where 64 images ran 5-10x slower (memory
pressure)."""

EXIF_ORIENTATION_TAG: int = 0x0112
TRANSPOSING_ORIENTATIONS: frozenset[int] = frozenset({5, 6, 7, 8})
"""EXIF orientations that swap width and height when applied (`load_image` applies them)."""

OUTPUT_SUFFIX: str = ".png"
"""Watermarked images are always written losslessly: a JPEG round trip would damage the mark."""

ImageShape = tuple[int, int]
"""`(height, width)` of an image."""


class BatchError(ValueError):
    """Invalid batching request: bad batch size, empty folder, inconsistent inputs."""


@dataclass(frozen=True)
class BatchPlan:
    """One batch to run: the positions (in the input order) of images sharing one shape."""

    shape: ImageShape
    indices: tuple[int, ...]


@dataclass(frozen=True)
class EmbedRecord:
    """Outcome of embedding one image of a folder."""

    image_name: str
    output_path: Path
    message_hex: str


@dataclass(frozen=True)
class EmbedFolderResult:
    """Outcome of `embed_folder`: one record per image (sorted file order) and the batch
    size actually used for each image shape."""

    records: tuple[EmbedRecord, ...]
    batch_sizes: dict[ImageShape, int]


@dataclass(frozen=True)
class ExtractFolderResult:
    """Outcome of `extract_folder`: names and soft bits [N, n_bits] in sorted file order."""

    image_names: tuple[str, ...]
    soft_bits: torch.Tensor
    batch_sizes: dict[ImageShape, int]


@dataclass(frozen=True)
class ExtractionScore:
    """Detection statistics of one image against its expected message."""

    bit_accuracy: float
    neg_log10_p: float
    detected: bool


def _available_memory_bytes(device: torch.device) -> int | None:
    """Free memory the device can draw on, or None when it cannot be measured.

    CUDA reports its own free memory. CPU and MPS share the system RAM (unified memory on
    Apple Silicon), read through psutil when it is installed.
    """
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    try:
        psutil: Any = importlib.import_module("psutil")
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def auto_batch_size(device: str | torch.device, image_shape: ImageShape) -> int:
    """Pick a batch size from the free memory.

    The budget is `MEMORY_BUDGET_FRACTION` of the free memory; one image is assumed to
    peak at `ACTIVATION_MEMORY_FACTOR` float32 values per pixel. The result is clamped to
    `[MIN_BATCH_SIZE, MAX_BATCH_SIZE]`. Without a way to measure free memory (psutil not
    installed on CPU/MPS) it returns the conservative `FALLBACK_BATCH_SIZE`.

    Args:
        device: Torch device on which the batch will run.
        image_shape: `(height, width)` of the images of the batch.

    Returns:
        A batch size in `[MIN_BATCH_SIZE, MAX_BATCH_SIZE]`.

    Raises:
        BatchError: A non-positive image side.
    """
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise BatchError(f"image shape must be positive, got {image_shape}")
    available = _available_memory_bytes(torch.device(device))
    if available is None:
        return FALLBACK_BATCH_SIZE
    bytes_per_image = height * width * RGB_CHANNELS * BYTES_PER_FLOAT32 * ACTIVATION_MEMORY_FACTOR
    fitting = int(available * MEMORY_BUDGET_FRACTION) // bytes_per_image
    return max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, fitting))


def resolve_batch_size(
    batch_size: int | None, device: str | torch.device, image_shape: ImageShape
) -> int:
    """Turn the user's request into a concrete size: None or 0 means automatic.

    Raises:
        BatchError: `batch_size` is negative.
    """
    if batch_size is None or batch_size == 0:
        return auto_batch_size(device, image_shape)
    if batch_size < 0:
        raise BatchError(f"batch size must be >= 0 (0 = automatic), got {batch_size}")
    return batch_size


def plan_batches(
    shapes: Sequence[ImageShape], batch_size: int | None, device: str | torch.device
) -> list[BatchPlan]:
    """Group image positions by shape and cut every group into batches.

    Groups keep the order of their first image; positions stay increasing inside a group.

    Args:
        shapes: `(H, W)` of every image, in input order.
        batch_size: Explicit size, or None / 0 for `auto_batch_size` (computed per shape).
        device: Device used by the automatic sizing.

    Returns:
        The batches to run, each holding only images of one shape.

    Raises:
        BatchError: No image, or a negative batch size.
    """
    if not shapes:
        raise BatchError("no image to process")
    positions_by_shape: dict[ImageShape, list[int]] = {}
    for position, shape in enumerate(shapes):
        positions_by_shape.setdefault(shape, []).append(position)
    plans: list[BatchPlan] = []
    for shape, positions in positions_by_shape.items():
        size = resolve_batch_size(batch_size, device, shape)
        for start in range(0, len(positions), size):
            plans.append(BatchPlan(shape=shape, indices=tuple(positions[start:start + size])))
    return plans


def max_batch_size(plans: Sequence[BatchPlan]) -> int:
    """Number of images of the biggest batch of `plans`."""
    return max(len(plan.indices) for plan in plans)


def sizes_by_shape(plans: Sequence[BatchPlan]) -> dict[ImageShape, int]:
    """Largest batch of each shape in `plans` (the size that was resolved for the shape)."""
    sizes: dict[ImageShape, int] = {}
    for plan in plans:
        sizes[plan.shape] = max(sizes.get(plan.shape, 0), len(plan.indices))
    return sizes


INTERNAL_BATCH_ATTRIBUTE: str = "batch_size"
"""Attribute under which a watermarker (e.g. `SealWatermarker`, default 8) caps the number
of images per forward pass and silently splits bigger batches itself."""


@contextmanager
def _internal_batch_at_least(watermarker: Watermarker, size: int) -> Iterator[None]:
    """Let the watermarker run `size` images per forward pass, then restore its setting.

    Without this, a batch of 64 handed to a watermarker capped at 8 is split back into 8
    forward passes and the larger batch buys nothing. A watermarker with no such attribute
    (the classical methods) is left alone; a larger cap set by the caller is kept.
    """
    current = getattr(watermarker, INTERNAL_BATCH_ATTRIBUTE, None)
    if not isinstance(current, int) or isinstance(current, bool) or current >= size:
        yield
        return
    setattr(watermarker, INTERNAL_BATCH_ATTRIBUTE, size)
    try:
        yield
    finally:
        setattr(watermarker, INTERNAL_BATCH_ATTRIBUTE, current)


def _stack(images: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Stack same-shape [3, H, W] images into one [B, 3, H, W] float32 tensor on `device`."""
    return torch.stack([image.to(torch.float32) for image in images]).to(device)


def _release_cache(device: torch.device) -> None:
    """Hand the allocator's cached blocks back after a shape group (shapes change between
    groups, so cached blocks of the previous shape are dead weight on unified memory)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _embed_batch(
    watermarker: Watermarker, images: torch.Tensor, messages: torch.Tensor
) -> torch.Tensor:
    """One `embed` call with the contract checks (output has the input's shape)."""
    with torch.no_grad():
        marked = watermarker.embed(images, messages.to(images.device))
    if marked.shape != images.shape:
        raise BatchError(
            f"{watermarker.name}.embed returned {tuple(marked.shape)} for an input of "
            f"{tuple(images.shape)}"
        )
    return marked.detach()


def _extract_batch(watermarker: Watermarker, images: torch.Tensor) -> torch.Tensor:
    """One `extract` call with the contract checks; returns soft bits on the CPU."""
    with torch.no_grad():
        soft_bits = watermarker.extract(images)
    expected = (images.shape[0], watermarker.n_bits)
    if tuple(soft_bits.shape) != expected:
        raise BatchError(
            f"{watermarker.name}.extract returned {tuple(soft_bits.shape)}, expected {expected}"
        )
    return soft_bits.detach().cpu()


def _check_messages(watermarker: Watermarker, messages: torch.Tensor, n_images: int) -> None:
    expected = (n_images, watermarker.n_bits)
    if tuple(messages.shape) != expected:
        raise BatchError(f"messages must have shape {expected}, got {tuple(messages.shape)}")


def _run_plans(
    plans: Sequence[BatchPlan],
    load: Callable[[int], torch.Tensor],
    device: torch.device,
) -> Iterator[tuple[BatchPlan, torch.Tensor]]:
    """Yield `(plan, stacked images on device)`, loading pixels one batch at a time."""
    previous_shape: ImageShape | None = None
    for plan in plans:
        if previous_shape is not None and plan.shape != previous_shape:
            _release_cache(device)
        previous_shape = plan.shape
        images = [load(position) for position in plan.indices]
        for position, image in zip(plan.indices, images, strict=True):
            if tuple(image.shape[-2:]) != plan.shape:
                raise BatchError(
                    f"image {position} has shape {tuple(image.shape[-2:])}, planned {plan.shape}"
                )
        yield plan, _stack(images, device)
    if previous_shape is not None:
        _release_cache(device)


def embed_images(
    watermarker: Watermarker,
    images: Sequence[torch.Tensor],
    messages: torch.Tensor,
    *,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Watermark in-memory images by batches, whatever their sizes.

    Args:
        watermarker: Method to apply.
        images: `[3, H, W]` tensors in [0, 1]; sizes may differ.
        messages: `[N, n_bits]` payloads in {0, 1}, row i for image i.
        batch_size: Images per batch; None or 0 = `auto_batch_size`.
        device: Device the batches run on.

    Returns:
        Watermarked `[3, H, W]` CPU tensors, same order and shapes as `images`.

    Raises:
        BatchError: Empty input, bad batch size, wrong `messages` shape, or a watermarker
            output that breaks the contract.
    """
    torch_device = torch.device(device)
    _check_messages(watermarker, messages, len(images))
    plans = plan_batches([(int(i.shape[-2]), int(i.shape[-1])) for i in images],
                         batch_size, torch_device)
    outputs: list[torch.Tensor | None] = [None] * len(images)
    with _internal_batch_at_least(watermarker, max_batch_size(plans)):
        for plan, batch in _run_plans(plans, lambda position: images[position], torch_device):
            marked = _embed_batch(watermarker, batch, messages[list(plan.indices)]).cpu()
            for row, position in enumerate(plan.indices):
                outputs[position] = marked[row]
    if any(output is None for output in outputs):
        missing = [index for index, output in enumerate(outputs) if output is None]
        raise BatchError(f"images {missing} were not covered by any batch")
    return [output for output in outputs if output is not None]


def extract_images(
    watermarker: Watermarker,
    images: Sequence[torch.Tensor],
    *,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Read the soft bits of in-memory images by batches, whatever their sizes.

    Args:
        watermarker: Method to read with.
        images: `[3, H, W]` tensors in [0, 1]; sizes may differ.
        batch_size: Images per batch; None or 0 = `auto_batch_size`.
        device: Device the batches run on.

    Returns:
        Soft bits `[N, n_bits]` on the CPU, row i for image i.

    Raises:
        BatchError: Empty input, bad batch size, or a watermarker output that breaks the
            contract.
    """
    torch_device = torch.device(device)
    plans = plan_batches([(int(i.shape[-2]), int(i.shape[-1])) for i in images],
                         batch_size, torch_device)
    soft_bits = torch.empty(len(images), watermarker.n_bits)
    with _internal_batch_at_least(watermarker, max_batch_size(plans)):
        for plan, batch in _run_plans(plans, lambda position: images[position], torch_device):
            soft_bits[list(plan.indices)] = _extract_batch(watermarker, batch)
    return soft_bits


def probe_image_shape(path: Path) -> ImageShape:
    """`(H, W)` of an image as `load_image` will return it, reading only the header.

    Raises:
        ImageIOError: The file cannot be opened as an image.
    """
    try:
        with Image.open(path) as handle:
            width, height = handle.size
            orientation = handle.getexif().get(EXIF_ORIENTATION_TAG)
    except (OSError, Image.DecompressionBombError) as error:
        # DecompressionBombError is not an OSError, so it would otherwise escape as a traceback.
        raise ImageIOError(f"cannot read image {path}: {error}") from error
    if orientation in TRANSPOSING_ORIENTATIONS:
        return width, height
    return height, width


def _list_folder(images_dir: Path) -> list[Path]:
    """Sorted image paths of `images_dir`; an empty folder is an error."""
    paths = list_images(images_dir)
    if not paths:
        raise BatchError(f"no image (png, jpg, jpeg, webp) found in {images_dir}")
    return paths


def output_path_for(image_path: Path, out_dir: Path) -> Path:
    """Where the watermarked version of `image_path` is written (same stem, PNG)."""
    return out_dir / (image_path.stem + OUTPUT_SUFFIX)


def embed_folder(
    watermarker: Watermarker,
    images_dir: Path,
    out_dir: Path,
    *,
    batch_size: int | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> EmbedFolderResult:
    """Watermark every image of a folder, by batches per image size.

    Each image gets its own random message, drawn from one generator seeded with `seed`, in
    sorted file order: the payloads (and so the output) do not depend on `batch_size`.
    Outputs keep the input stem and are written as PNG (lossless) in `out_dir`.

    Args:
        watermarker: Method to apply.
        images_dir: Folder of png / jpg / jpeg / webp images (non recursive).
        out_dir: Destination folder, created if missing.
        batch_size: Images per batch; None or 0 = `auto_batch_size` per image shape.
        seed: Seed of the per-image messages.
        device: Device the batches run on.

    Returns:
        One record per image in sorted order, and the batch size used per shape.

    Raises:
        BatchError: Empty folder, bad batch size, or two inputs that would write the same
            output file (e.g. `a.jpg` and `a.png`).
        NotADirectoryError: `images_dir` is not a folder.
        ImageIOError: An undecodable image.
    """
    torch_device = torch.device(device)
    if out_dir.resolve() == images_dir.resolve():
        raise BatchError(
            f"out_dir must differ from images_dir ({images_dir}): writing the marked images "
            f"back over the inputs would destroy them"
        )
    paths = _list_folder(images_dir)
    output_paths = [output_path_for(path, out_dir) for path in paths]
    # Compare case-insensitively: APFS (macOS) and NTFS treat 'Photo.png' and 'photo.png' as the
    # same file, so a case-only difference would still silently overwrite one output.
    casefolded = [str(path).casefold() for path in output_paths]
    if len(set(casefolded)) != len(casefolded):
        raise BatchError(
            f"several inputs of {images_dir} would write the same output file in {out_dir} "
            f"(names differing only by stem or case, e.g. 'a.jpg'/'a.png' or 'A.png'/'a.png')"
        )
    plans = plan_batches([probe_image_shape(path) for path in paths], batch_size, torch_device)
    generator = torch.Generator().manual_seed(seed)
    messages = random_messages(len(paths), watermarker.n_bits, generator)
    hexes: list[str] = [bits_to_hex(row) for row in messages]
    with _internal_batch_at_least(watermarker, max_batch_size(plans)):
        for plan, batch in _run_plans(plans, lambda position: load_image(paths[position]),
                                      torch_device):
            marked = _embed_batch(watermarker, batch, messages[list(plan.indices)]).cpu()
            for row, position in enumerate(plan.indices):
                save_image(marked[row], output_paths[position])
    records = tuple(
        EmbedRecord(image_name=path.name, output_path=output, message_hex=message_hex)
        for path, output, message_hex in zip(paths, output_paths, hexes, strict=True)
    )
    return EmbedFolderResult(records=records, batch_sizes=sizes_by_shape(plans))


def extract_folder(
    watermarker: Watermarker,
    images_dir: Path,
    *,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> ExtractFolderResult:
    """Read the soft bits of every image of a folder, by batches per image size.

    Args:
        watermarker: Method to read with.
        images_dir: Folder of png / jpg / jpeg / webp images (non recursive).
        batch_size: Images per batch; None or 0 = `auto_batch_size` per image shape.
        device: Device the batches run on.

    Returns:
        Image names and soft bits `[N, n_bits]`, in sorted file order.

    Raises:
        BatchError: Empty folder or bad batch size.
        NotADirectoryError: `images_dir` is not a folder.
        ImageIOError: An undecodable image.
    """
    torch_device = torch.device(device)
    paths = _list_folder(images_dir)
    plans = plan_batches([probe_image_shape(path) for path in paths], batch_size, torch_device)
    soft_bits = torch.empty(len(paths), watermarker.n_bits)
    with _internal_batch_at_least(watermarker, max_batch_size(plans)):
        for plan, batch in _run_plans(plans, lambda position: load_image(paths[position]),
                                      torch_device):
            soft_bits[list(plan.indices)] = _extract_batch(watermarker, batch)
    return ExtractFolderResult(
        image_names=tuple(path.name for path in paths),
        soft_bits=soft_bits,
        batch_sizes=sizes_by_shape(plans),
    )


def read_expected_messages(
    path: Path, n_bits: int, image_names: Sequence[str]
) -> torch.Tensor:
    """Read a file of expected messages and align it to `image_names`.

    Two line formats are accepted (blank lines ignored):

    - ``output_name,hex`` (what ``batch-embed --messages-out`` writes): each message is matched
      to its image by file name, so the pairing survives any reordering between embed and
      extract (renaming ``.webp`` to ``.png`` changes the sort order).
    - ``hex`` alone (a hand-written file): messages are matched by position, in the order of
      ``image_names``; the line count must then equal the number of images.

    Args:
        path: Text file.
        n_bits: Payload size of the method.
        image_names: The images to cover, in extraction order.

    Returns:
        Bits ``[len(image_names), n_bits]`` in {0, 1}, row ``i`` for ``image_names[i]``.

    Raises:
        BatchError: A line is malformed, a name is unknown, duplicated or missing, or (in
            positional mode) the line count differs from the number of images.
        OSError: The file cannot be read.
    """
    entries = [
        (number, line.strip())
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    if not entries:
        raise BatchError(f"{path} holds no expected message for {len(image_names)} image(s)")
    if any("," in line for _, line in entries):
        return _read_named_messages(path, n_bits, image_names, entries)
    return _read_positional_messages(path, n_bits, image_names, entries)


def _parse_payload(path: Path, number: int, payload: str, n_bits: int) -> torch.Tensor:
    """Decode one hex payload, blaming the actual source line on failure."""
    try:
        return hex_to_bits(payload, n_bits)
    except ValueError as error:
        raise BatchError(f"{path}, line {number}: {error}") from error


def _read_positional_messages(
    path: Path, n_bits: int, image_names: Sequence[str], entries: Sequence[tuple[int, str]]
) -> torch.Tensor:
    """Legacy format: one hex per line, matched to `image_names` by position."""
    if len(entries) != len(image_names):
        raise BatchError(
            f"{path} holds {len(entries)} expected message(s) for {len(image_names)} image(s)"
        )
    return torch.stack([_parse_payload(path, number, line, n_bits) for number, line in entries])


def _read_named_messages(
    path: Path, n_bits: int, image_names: Sequence[str], entries: Sequence[tuple[int, str]]
) -> torch.Tensor:
    """`name,hex` format: each message matched to its image by file name."""
    by_name: dict[str, torch.Tensor] = {}
    for number, line in entries:
        name, separator, payload = line.partition(",")
        name = name.strip()
        if not separator or not name:
            raise BatchError(f"{path}, line {number}: expected 'name,hex', got {line!r}")
        if name in by_name:
            raise BatchError(f"{path}, line {number}: duplicate entry for {name!r}")
        by_name[name] = _parse_payload(path, number, payload.strip(), n_bits)
    wanted = set(image_names)
    missing = [name for name in image_names if name not in by_name]
    unexpected = [name for name in by_name if name not in wanted]
    if missing or unexpected:
        raise BatchError(
            f"{path}: expected messages do not match the images "
            f"(missing {missing[:5]}, unexpected {unexpected[:5]})"
        )
    return torch.stack([by_name[name] for name in image_names])


def score_extractions(soft_bits: torch.Tensor, expected: torch.Tensor) -> list[ExtractionScore]:
    """Bit accuracy, `-log10 p` and detection verdict of every row.

    Args:
        soft_bits: `[N, n_bits]` extracted soft bits.
        expected: `[N, n_bits]` embedded bits.

    Returns:
        One `ExtractionScore` per row; `detected` uses `DETECTION_NEG_LOG10_P`.

    Raises:
        ValueError: Shapes differ (raised by `bit_accuracy`).
    """
    accuracy = bit_accuracy(soft_bits, expected)
    n_bits = soft_bits.shape[1]
    scores = neg_log10_pvalue(n_bits, torch.round(accuracy * n_bits))
    return [
        ExtractionScore(
            bit_accuracy=float(row_accuracy),
            neg_log10_p=float(row_score),
            detected=float(row_score) > DETECTION_NEG_LOG10_P,
        )
        for row_accuracy, row_score in zip(accuracy, scores, strict=True)
    ]
