"""Weight-space blend `original + alpha * (candidate - original)` (pure torch, no I/O)."""

from __future__ import annotations

from torch import Tensor

from hackathon.common.errors import HackathonError


class BlendError(HackathonError):
    """The two state dicts cannot be blended (different keys or shapes)."""


def blend_state_dicts(
    original: dict[str, Tensor], candidate: dict[str, Tensor], alpha: float
) -> dict[str, Tensor]:
    """`original + alpha * (candidate - original)` on every floating tensor (others copied).

    Raises:
        BlendError: the key sets or a tensor shape differ.
    """
    if original.keys() != candidate.keys():
        raise BlendError("The two state dicts do not have the same keys.")
    blended: dict[str, Tensor] = {}
    for key, base in original.items():
        other = candidate[key]
        if base.shape != other.shape:
            raise BlendError(
                f"Shape mismatch on {key}: {tuple(base.shape)} vs {tuple(other.shape)}."
            )
        if base.is_floating_point():
            blended[key] = base.double().add(other.double().sub(base.double()), alpha=alpha).float()
        else:
            blended[key] = base.clone()
    return blended
