"""Validation of a submission file against the server rules, before anything is sent.

Server rules (Unlearning.pdf, "Submission Format"): a `.pt`/`.pth` weights-only state dict that
loads with `strict=True` into the task architecture, float32 weights only, no NaN/Inf, at most
450 MB. Every violation raises `SubmissionInvalidError` with a message that names the tensor.

CLI: `python -m hackathon.unlearning.validate FILE` (also `python -m hackathon.unlearning validate`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError, SubmissionInvalidError
from hackathon.unlearning.constants import (
    ALLOWED_SUFFIXES,
    BYTES_PER_MEGABYTE,
    MAX_SUBMISSION_BYTES,
)
from hackathon.unlearning.model import build_model

MAX_LISTED_PROBLEMS = 8
"""Cap on the offending tensors named in one error message (the rest is counted)."""
REQUIRED_FLOAT_DTYPE = torch.float32

ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ValidationReport:
    """Facts about a file that passed validation."""

    path: Path
    size_bytes: int
    tensor_count: int
    parameter_count: int


def _summarise(problems: list[str]) -> str:
    shown = problems[:MAX_LISTED_PROBLEMS]
    hidden = len(problems) - len(shown)
    return "; ".join(shown) + (f"; ... and {hidden} more" if hidden > 0 else "")


def reference_shapes(reference: nn.Module) -> dict[str, tuple[int, ...]]:
    """Key -> shape of the target architecture (timm cannot build on the meta device)."""
    return {key: tuple(value.shape) for key, value in reference.state_dict().items()}


def check_file_metadata(path: Path) -> int:
    """Check existence, suffix and size; return the size in bytes."""
    if not path.is_file():
        raise SubmissionInvalidError(f"Submission file not found: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SubmissionInvalidError(
            f"{path.name}: expected a {' or '.join(ALLOWED_SUFFIXES)} file, got '{path.suffix}'."
        )
    size = path.stat().st_size
    if size > MAX_SUBMISSION_BYTES:
        raise SubmissionInvalidError(
            f"{path.name}: {size / BYTES_PER_MEGABYTE:.1f} MB exceeds the "
            f"{MAX_SUBMISSION_BYTES // BYTES_PER_MEGABYTE} MB limit."
        )
    return size


def check_state_dict(
    state_dict: Mapping[str, Tensor], expected: Mapping[str, tuple[int, ...]]
) -> None:
    """Check keys, shapes, dtype float32 and finiteness of an in-memory state dict.

    Raises:
        SubmissionInvalidError: with the offending tensor names.
    """
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    problems: list[str] = []
    if missing:
        problems.append(f"missing keys: {_summarise(missing)}")
    if unexpected:
        problems.append(f"unexpected keys: {_summarise(unexpected)}")
    if problems:
        raise SubmissionInvalidError("; ".join(problems))

    shape_problems: list[str] = []
    dtype_problems: list[str] = []
    value_problems: list[str] = []
    for key, tensor in state_dict.items():
        if not isinstance(tensor, Tensor):
            shape_problems.append(f"{key} is a {type(tensor).__name__}, not a tensor")
            continue
        if tuple(tensor.shape) != expected[key]:
            shape_problems.append(
                f"{key} has shape {tuple(tensor.shape)}, expected {expected[key]}"
            )
        if tensor.is_floating_point() and tensor.dtype != REQUIRED_FLOAT_DTYPE:
            dtype_problems.append(f"{key} is {tensor.dtype}")
        elif tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            value_problems.append(f"{key} contains NaN or Inf")
    if shape_problems:
        raise SubmissionInvalidError(f"shape mismatch: {_summarise(shape_problems)}")
    if dtype_problems:
        raise SubmissionInvalidError(
            f"only float32 weights are accepted: {_summarise(dtype_problems)}"
        )
    if value_problems:
        raise SubmissionInvalidError(f"non-finite weights: {_summarise(value_problems)}")


def validate_submission(path: Path, model_factory: ModelFactory = build_model) -> ValidationReport:
    """Validate `path` exactly like the server will, and return a report.

    Args:
        path: Candidate `.pt` / `.pth` file.
        model_factory: Builder of the target architecture (the official one by default; tests
            inject a tiny ViT).

    Raises:
        SubmissionInvalidError: any rule violated (never a raw torch traceback).
    """
    size = check_file_metadata(path)
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises pickle/zip/runtime errors
        raise SubmissionInvalidError(
            f"{path.name} is not a weights-only torch file: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise SubmissionInvalidError(
            f"{path.name} must contain a plain state dict, got {type(loaded).__name__} "
            "(save model.state_dict(), not the module)."
        )
    target = model_factory()
    check_state_dict(loaded, reference_shapes(target))

    # Final proof, the server's own operation: strict load into a real instance.
    try:
        target.load_state_dict(loaded, strict=True)
    except RuntimeError as exc:
        raise SubmissionInvalidError(f"strict load failed: {exc}") from exc
    return ValidationReport(
        path=path,
        size_bytes=size,
        tensor_count=len(loaded),
        parameter_count=sum(int(tensor.numel()) for tensor in loaded.values()),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 0 when valid, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Validate an unlearning submission file.")
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_submission(args.file)
    except HackathonError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"VALID: {report.path} ({report.size_bytes / BYTES_PER_MEGABYTE:.1f} MB, "
        f"{report.tensor_count} tensors, {report.parameter_count:,} parameters)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
