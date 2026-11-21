"""Build and validate the localization submission file.

Format (task PDF): a `.pt` holding {"image_names": list[str], "masks": Tensor[N, 128, 128]} with
values in {0, 1}, every image name exactly once, file size <= 20 MB. The server rejects any
violation automatically, and a rejected attempt can still start a cooldown, so everything is
checked locally first. Masks are stored as uint8: 1 020 x 128 x 128 in float32 is 67 MB (rejected),
in uint8 it is 16.7 MB.
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hackathon.common.errors import SubmissionInvalidError
from hackathon.wmloc.constants import (
    GRID_SIZE,
    MASK_CLEAN,
    MASK_WATERMARKED,
    MAX_SUBMISSION_BYTES,
    SUBMISSION_KEY_MASKS,
    SUBMISSION_KEY_NAMES,
)

COMPACT_DTYPES: tuple[torch.dtype, ...] = (torch.uint8, torch.bool)
ACCEPTED_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
)
MASKS_NDIM = 3
MAX_NAMES_IN_MESSAGE = 5


@dataclass(frozen=True)
class SubmissionReport:
    """What a validated submission contains.

    Attributes:
        n_images: Number of images.
        shape: Shape of the mask tensor.
        dtype: Mask dtype as a string.
        n_bytes: Serialized size in bytes (file size when validated from a file).
        watermarked_fraction: Fraction of cells set to 1, over all images.
        warnings: Non-blocking remarks (e.g. non-compact dtype).
    """

    n_images: int
    shape: tuple[int, ...]
    dtype: str
    n_bytes: int
    watermarked_fraction: float
    warnings: tuple[str, ...]


def build_submission(names: Sequence[str], masks: torch.Tensor) -> dict[str, Any]:
    """Return the submission dict with masks cast to uint8 (validated, nothing written).

    Raises:
        SubmissionInvalidError: names/masks break the format rules.
    """
    submission: dict[str, Any] = {
        SUBMISSION_KEY_NAMES: list(names),
        SUBMISSION_KEY_MASKS: _cast_binary(masks),
    }
    validate_submission(submission)
    return submission


def _cast_binary(masks: torch.Tensor) -> torch.Tensor:
    """Cast to uint8 only if every value is already 0 or 1 (never silently round)."""
    if masks.is_floating_point() and bool(torch.isnan(masks).any()):
        raise SubmissionInvalidError("masks contain NaN")
    is_binary = (masks == MASK_CLEAN) | (masks == MASK_WATERMARKED)
    if not bool(is_binary.all()):
        raise SubmissionInvalidError("masks must only contain 0 and 1 (threshold them first)")
    return masks.to(torch.uint8)


def _serialized_size(submission: dict[str, Any]) -> int:
    buffer = io.BytesIO()
    torch.save(submission, buffer)
    return buffer.getbuffer().nbytes


def _check_names(names: object, expected: Collection[str] | None) -> list[str]:
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise SubmissionInvalidError(f"{SUBMISSION_KEY_NAMES} must be a list[str]")
    if any(not name for name in names):
        raise SubmissionInvalidError(f"{SUBMISSION_KEY_NAMES} contains an empty name")
    seen = set(names)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise SubmissionInvalidError(
            f"duplicate image names ({len(duplicates)}): {duplicates[:MAX_NAMES_IN_MESSAGE]}"
        )
    if expected is not None:
        wanted = set(expected)
        missing = sorted(wanted - seen)
        unknown = sorted(seen - wanted)
        if missing or unknown:
            raise SubmissionInvalidError(
                f"names differ from the expected set: {len(missing)} missing "
                f"{missing[:MAX_NAMES_IN_MESSAGE]}, {len(unknown)} unknown "
                f"{unknown[:MAX_NAMES_IN_MESSAGE]}"
            )
    return names


def _check_masks(masks: object, n_names: int) -> torch.Tensor:
    if not isinstance(masks, torch.Tensor):
        raise SubmissionInvalidError(f"{SUBMISSION_KEY_MASKS} must be a torch.Tensor")
    if masks.dtype not in ACCEPTED_DTYPES:
        raise SubmissionInvalidError(f"unsupported mask dtype {masks.dtype}")
    expected_shape = (n_names, GRID_SIZE, GRID_SIZE)
    if masks.ndim != MASKS_NDIM or tuple(masks.shape) != expected_shape:
        raise SubmissionInvalidError(
            f"masks must have shape {expected_shape} (one {GRID_SIZE}x{GRID_SIZE} mask per name), "
            f"got {tuple(masks.shape)}"
        )
    if masks.is_floating_point() and bool(torch.isnan(masks).any()):
        raise SubmissionInvalidError("masks contain NaN")
    is_binary = (masks == MASK_CLEAN) | (masks == MASK_WATERMARKED)
    if not bool(is_binary.all()):
        raise SubmissionInvalidError("masks must only contain 0 and 1")
    return masks


def validate_submission(
    submission: object,
    expected_names: Collection[str] | None = None,
    *,
    n_bytes: int | None = None,
) -> SubmissionReport:
    """Check every format rule of the task and return a report; raise on the first violation.

    Args:
        submission: The loaded object (should be a dict).
        expected_names: Names the file must contain exactly (the test split), if known.
        n_bytes: Size already measured on disk; when None the size is measured by serializing.

    Raises:
        SubmissionInvalidError: any rule is broken (keys, names, shape, values, size).
    """
    if not isinstance(submission, dict):
        raise SubmissionInvalidError("submission must be a dict")
    keys = set(submission)
    required = {SUBMISSION_KEY_NAMES, SUBMISSION_KEY_MASKS}
    if keys != required:
        raise SubmissionInvalidError(
            f"keys must be exactly {sorted(required)}, got {sorted(map(str, keys))}"
        )
    names = _check_names(submission[SUBMISSION_KEY_NAMES], expected_names)
    masks = _check_masks(submission[SUBMISSION_KEY_MASKS], len(names))
    size = n_bytes if n_bytes is not None else _serialized_size(submission)
    if size > MAX_SUBMISSION_BYTES:
        raise SubmissionInvalidError(
            f"file is {size} bytes, over the {MAX_SUBMISSION_BYTES} byte limit "
            f"(masks dtype {masks.dtype}; use uint8)"
        )
    warnings: list[str] = []
    if masks.dtype not in COMPACT_DTYPES:
        warnings.append(f"dtype {masks.dtype} is not compact; uint8 is recommended")
    return SubmissionReport(
        n_images=len(names),
        shape=tuple(masks.shape),
        dtype=str(masks.dtype),
        n_bytes=size,
        watermarked_fraction=float(masks.to(torch.float32).mean().item()) if len(names) else 0.0,
        warnings=tuple(warnings),
    )


def save_submission(
    path: Path,
    names: Sequence[str],
    masks: torch.Tensor,
    expected_names: Collection[str] | None = None,
) -> SubmissionReport:
    """Validate, then write the submission atomically (temp file renamed once it re-validates).

    Raises:
        SubmissionInvalidError: the content breaks a rule; nothing is written.
    """
    submission = build_submission(names, masks)
    validate_submission(submission, expected_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    torch.save(submission, partial)
    try:
        report = validate_submission_file(partial, expected_names)
    except SubmissionInvalidError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return report


def validate_submission_file(
    path: Path, expected_names: Collection[str] | None = None
) -> SubmissionReport:
    """Load `path` (tensors only, `weights_only=True`) and validate it, size included.

    Raises:
        SubmissionInvalidError: the file is missing, unreadable, or breaks a rule.
    """
    if not path.is_file():
        raise SubmissionInvalidError(f"submission file not found: {path}")
    n_bytes = path.stat().st_size
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SubmissionInvalidError(f"cannot load {path} with torch.load: {exc}") from exc
    return validate_submission(loaded, expected_names, n_bytes=n_bytes)
