"""Shared helpers of the unlearning tests: a tiny ViT with the real head size, random images."""

from __future__ import annotations

import torch
from torch import nn

from hackathon.unlearning.constants import IMAGE_CHANNELS, IMAGE_SIZE
from hackathon.unlearning.model import build_model

TINY_EMBED_DIM = 32
TINY_DEPTH = 2
TINY_HEADS = 2
IMAGE_SEED = 1234


def tiny_model(seed: int = 0) -> nn.Module:
    """ViT with the official patch size, image size and 100-class head, but 2 tiny blocks."""
    torch.manual_seed(seed)
    return build_model(embed_dim=TINY_EMBED_DIM, depth=TINY_DEPTH, num_heads=TINY_HEADS).eval()


def tiny_factory() -> nn.Module:
    """Argument-free builder, the shape `validate_submission(model_factory=...)` expects."""
    return build_model(embed_dim=TINY_EMBED_DIM, depth=TINY_DEPTH, num_heads=TINY_HEADS)


def random_images(count: int, seed: int = IMAGE_SEED) -> torch.Tensor:
    """Random uint8 [count, 224, 224, 3] images."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (count, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), generator=generator,
        dtype=torch.uint8,
    )  # fmt: skip
