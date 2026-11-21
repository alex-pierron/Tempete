"""Shared, frozen types of the wmkit toolkit.

Every module (classical baselines, Meta Seal wrappers, neural trainer, attacks,
evaluation harness) talks through these types. Changing them is a contract change:
update docs/CONTRACT-wmkit.md in the same commit.

Tensor conventions (all modules):
    images   float32, shape [B, 3, H, W], RGB, values in [0, 1], any device.
    messages float32, shape [B, n_bits], values in {0.0, 1.0}.
    soft bits float32, shape [B, n_bits], values in [0, 1] = P(bit == 1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch

AttackGroup = Literal[
    "identity",
    "valuemetric",
    "compression",
    "geometric",
    "combined",
    "removal",
]
"""Attack families. The first five mirror the Pixel Seal paper (Table 1)."""

ATTACK_GROUPS: tuple[AttackGroup, ...] = (
    "identity",
    "valuemetric",
    "compression",
    "geometric",
    "combined",
    "removal",
)


@runtime_checkable
class Watermarker(Protocol):
    """A multi-bit post-hoc image watermarking method.

    Implementations must be stateless between calls (no hidden per-image state),
    accept images of arbitrary H and W (resizing internally if needed) and return
    watermarked images of exactly the input size.
    """

    name: str
    n_bits: int

    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Hide `messages` [B, n_bits] in `images` [B, 3, H, W]; return [B, 3, H, W] in [0, 1]."""
        ...

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Read soft bits [B, n_bits] in [0, 1] from possibly attacked images [B, 3, H', W']."""
        ...


AttackFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps images [B, 3, H, W] in [0, 1] to attacked images [B, 3, H', W'] in [0, 1]."""


@dataclass(frozen=True)
class AttackSpec:
    """A named, parameterised image transformation used to stress a watermark.

    Attributes:
        name: Unique registry key, lowercase, format `<op>_<param>` (see contract).
        group: Attack family, one of ATTACK_GROUPS.
        fn: The transformation itself.
        differentiable: True if gradients flow through `fn` (usable in training as is).
            Non-differentiable attacks are wrapped with a straight-through estimator
            (`wmkit.attacks.straight_through`) when used in training.
        params: Human-readable parameters, for reports only.
    """

    name: str
    group: AttackGroup
    fn: AttackFn
    differentiable: bool
    params: dict[str, float | int | str] = field(default_factory=dict)
