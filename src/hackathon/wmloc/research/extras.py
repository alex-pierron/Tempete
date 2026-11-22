"""Message-aware side inputs computed from the frozen extractors (tracks 2 and 4).

For every image we cache, with the SAME official wrapper that built the dataset:
- `maps` [2, 8, 8]: per-position agreement with the known message, from the wrapper's
  `features()["per_position_logits"]` (PixelSeal and TrustMark; zeros for StegaStamp whose
  wrapper only exposes global logits). Channel 0 is the mean signed margin (logit x message sign),
  channel 1 the fraction of bits whose per-position logit agrees with the message.
- `scalars` [2]: global margin and global bit accuracy of the whole-image decode (the regime
  switch: ~0.98 when the whole image is marked, ~0.5 when only a small region is).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

EXTRAS_MAP_SIDE = 8
N_MAP_CHANNELS = 2
N_SCALARS = 2
FEATURE_KEY_PER_POSITION = "per_position_logits"
MARGIN_CLIP = 8.0
"""Margins are clipped so that a few huge logits cannot dominate the input scale."""
METHODS_WITH_SPATIAL_MAP: tuple[str, ...] = ("pixelseal", "trustmark")
EXTRAS_BATCH = 8


def message_signs(bits: NDArray[np.uint8]) -> torch.Tensor:
    """+1 for bit 1 and -1 for bit 0, float32 with the shape of `bits`."""
    return torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)


def compute_extras(
    wrapper: Any,
    method: str,
    images: NDArray[np.uint8],
    bits: NDArray[np.uint8],
) -> tuple[NDArray[np.float16], NDArray[np.float32]]:
    """Agreement maps and global scalars for a batch of images.

    Args:
        wrapper: Official kit wrapper of `method`.
        method: Method name.
        images: uint8 [B, H, W, 3].
        bits: uint8 [B, n_bits], the known message of each image.

    Returns:
        (maps float16 [B, 2, 8, 8], scalars float32 [B, 2]).
    """
    signs = message_signs(bits)
    batch = images.shape[0]
    maps = np.zeros((batch, N_MAP_CHANNELS, EXTRAS_MAP_SIDE, EXTRAS_MAP_SIDE), dtype=np.float16)
    scalars = np.zeros((batch, N_SCALARS), dtype=np.float32)
    for start in range(0, batch, EXTRAS_BATCH):
        chunk = torch.from_numpy(np.ascontiguousarray(images[start : start + EXTRAS_BATCH]))
        tensor = chunk.permute(0, 3, 1, 2).float() / 255.0
        end = start + chunk.shape[0]
        has_map = method in METHODS_WITH_SPATIAL_MAP
        if has_map:
            # The decoder head is linear after a spatial mean, so the global logits are EXACTLY
            # the mean of the per-position logits: one forward pass gives both.
            per_position = wrapper.features(tensor)[FEATURE_KEY_PER_POSITION].detach().float().cpu()
            logits = per_position.mean(dim=(2, 3))
        else:
            logits = wrapper.logits(tensor).detach().float().cpu()
        signed = logits * signs[start:end]
        scalars[start:end, 0] = signed.mean(dim=1).clamp(-MARGIN_CLIP, MARGIN_CLIP).numpy()
        scalars[start:end, 1] = (signed > 0).float().mean(dim=1).numpy()
        if has_map:
            signed_map = per_position * signs[start:end, :, None, None]
            margin = signed_map.mean(dim=1).clamp(-MARGIN_CLIP, MARGIN_CLIP)
            agree = (signed_map > 0).float().mean(dim=1)
            side = EXTRAS_MAP_SIDE
            if margin.shape[-2:] != (side, side):
                margin = torch.nn.functional.adaptive_avg_pool2d(margin[:, None], side)[:, 0]
                agree = torch.nn.functional.adaptive_avg_pool2d(agree[:, None], side)[:, 0]
            maps[start:end, 0] = margin.numpy().astype(np.float16)
            maps[start:end, 1] = agree.numpy().astype(np.float16)
    return maps, scalars
