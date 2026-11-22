"""Turn a streamed base and a recipe (alpha, head scale, class bias) into a submittable state dict.

The recipe is what the scoring grid ranks (`g_score`): the exports are logits of
`theta_orig + alpha * (theta_base - theta_orig)`, and the post-hoc axes act on those logits as

    logits' = scale * logits - delta * 1[class in saturated]

which is EXACTLY a head edit: multiply `head.weight` and `head.bias` by `scale`, then subtract
`delta` from the bias of the saturated classes. Nothing here needs a GPU or a forward pass.
`write_candidate` validates the file with the official rules before moving it into place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.model import build_model
from hackathon.unlearning.validate import ValidationReport, validate_submission

HEAD_WEIGHT_KEY = "head.weight"
HEAD_BIAS_KEY = "head.bias"
PARTIAL_SUFFIX = ".partial"
StateDict = dict[str, Tensor]


class CandidateError(HackathonError):
    """The recipe or the state dicts are inconsistent (key mismatch, bad scale, class out of range)."""


def blend(origin: Mapping[str, Tensor], base: Mapping[str, Tensor], alpha: float) -> StateDict:
    """`origin + alpha * (base - origin)` on every floating tensor (alpha > 1 extrapolates).

    Raises:
        CandidateError: the two state dicts do not have the same keys and shapes.
    """
    if set(origin) != set(base):
        raise CandidateError("origin and base do not have the same keys")
    blended: StateDict = {}
    for key, value in base.items():
        if tuple(value.shape) != tuple(origin[key].shape):
            raise CandidateError(f"shape mismatch on {key}")
        blended[key] = (
            origin[key] + alpha * (value - origin[key])
            if value.is_floating_point()
            else value.detach().clone()
        )
    return blended


def edit_head(
    state: Mapping[str, Tensor], scale: float, delta: float, saturated: Sequence[int]
) -> StateDict:
    """Head edit of the recipe: scale weight and bias by `scale`, then `delta` off the saturated biases.

    Raises:
        CandidateError: non-positive scale, negative delta, a class outside the head, or no head.
    """
    if scale <= 0.0:
        raise CandidateError(f"head scale must be positive, got {scale}")
    if delta < 0.0:
        raise CandidateError(f"delta is subtracted: it must be >= 0, got {delta}")
    if HEAD_WEIGHT_KEY not in state or HEAD_BIAS_KEY not in state:
        raise CandidateError("the state dict has no classifier head")
    edited = {key: value.detach().clone() for key, value in state.items()}
    edited[HEAD_WEIGHT_KEY] = edited[HEAD_WEIGHT_KEY] * scale
    bias = edited[HEAD_BIAS_KEY] * scale
    for index in saturated:
        if not 0 <= index < len(bias):
            raise CandidateError(f"class {index} outside [0, {len(bias)})")
        bias[index] = bias[index] - delta
    edited[HEAD_BIAS_KEY] = bias
    return edited


def build_candidate(
    origin: Mapping[str, Tensor],
    base: Mapping[str, Tensor],
    alpha: float,
    scale: float,
    delta: float,
    saturated: Sequence[int],
) -> StateDict:
    """The full recipe: blend at `alpha`, then the head edit (`scale`, `delta`)."""
    return edit_head(blend(origin, base, alpha), scale, delta, saturated)


def write_candidate(
    state: Mapping[str, Tensor],
    path: Path,
    model_factory: Callable[[], nn.Module] = build_model,
) -> ValidationReport:
    """Write a float32 CPU state dict, validate it, and only then move it to `path`.

    Raises:
        SubmissionInvalidError: the file breaks a server rule (nothing is left at `path`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {
        key: (value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu())
        for key, value in state.items()
    }
    partial = path.with_name(path.stem + PARTIAL_SUFFIX + path.suffix)
    torch.save(cleaned, partial)
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
