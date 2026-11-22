"""Saving, validating and (explicitly) sending a submission file."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from hackathon.common.config import Settings
from hackathon.common.submit import SubmitResult, submit_file
from hackathon.unlearning.constants import TASK_ID
from hackathon.unlearning.model import build_model, cpu_float32_state_dict
from hackathon.unlearning.validate import ModelFactory, ValidationReport, validate_submission

PARTIAL_MARKER = ".partial"
"""Inserted before the suffix of the temporary file (keeps the .pt suffix the validator needs)."""


def save_submission(
    model: nn.Module, path: Path, model_factory: ModelFactory = build_model
) -> ValidationReport:
    """Write the CPU float32 state dict of `model` to `path` and validate it.

    The file is written under a temporary name, validated, then moved into place, so a crash or an
    invalid model never leaves a half-written or wrong file at `path`.

    Args:
        model: The unlearned model (any device, any floating dtype: it is cast to CPU float32).
        path: Destination, `.pt` or `.pth`.
        model_factory: Target architecture builder (tests inject a tiny ViT).

    Returns:
        The validation report of the saved file.

    Raises:
        SubmissionInvalidError: the state dict breaks a server rule (the file is not created).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + PARTIAL_MARKER + path.suffix)
    torch.save(cpu_float32_state_dict(model), partial)
    try:
        report = validate_submission(partial, model_factory)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    return ValidationReport(
        path=path,
        size_bytes=report.size_bytes,
        tensor_count=report.tensor_count,
        parameter_count=report.parameter_count,
    )


def submit_validated(path: Path, settings: Settings, *, confirm: bool) -> SubmitResult | None:
    """Validate `path`, then send it only when `confirm` is True.

    Returns:
        The SubmitResult when sent, or None for a dry run (`confirm` False): nothing leaves the
        machine and the cooldown log is untouched.

    Raises:
        SubmissionInvalidError: the file is invalid (nothing is sent).
        HackathonError: any failure of the shared submission client.
    """
    validate_submission(path)
    if not confirm:
        return None
    return submit_file(TASK_ID, path, settings, confirm=True)
