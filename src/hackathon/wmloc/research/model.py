"""Small fully-convolutional localizer: high-frequency residual (+ optional message extras) -> 128x128.

Design (~150 k parameters, trains on a shared 16 GB M5):
- Input at 512x512: RGB centred and its high-pass residual (the watermark is a high-frequency
  pattern), 6 channels.
- Stem: a 4x4 stride-4 convolution brings the signal to the 128x128 evaluation grid (one 4x4
  pixel cell = one grid cell), then a 3x3 convolution.
- Context: a strided step to 64x64 then four dilated 3x3 convolutions (dilation 1, 2, 4, 8, an
  effective field of ~250 px) and a global-average branch, so that the model can tell "the whole
  image is marked, find the hole" from "a small region is marked".
- Decoder: upsample, concatenate with the 128-grid features, one convolution, 1x1 logits.
- Optional extras (track 4): per-position and global agreement of the frozen extractor with the
  known message, resized to the 128 grid and concatenated at the 128 stage.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

HIGHPASS_KERNEL = 5
HIGHPASS_GAIN = 12.0
"""Residuals are ~1-2 grey levels (out of 255); the gain brings them to O(0.05-0.1)."""
STEM_CHANNELS = 48
CELL_PIXELS = 4
GRID_CHANNELS = 48
CONTEXT_CHANNELS = 64
DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
GROUPS = 8
N_RESIDUAL_INPUT_CHANNELS = 6
N_EXTRAS_CHANNELS = 4
MARGIN_SCALE = 8.0
AGREEMENT_CENTRE = 0.5


def conv_block(
    in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1
) -> nn.Sequential:
    """Conv + GroupNorm + SiLU with 'same'-style padding."""
    padding = dilation * (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation, bias=False),
        nn.GroupNorm(GROUPS, out_ch),
        nn.SiLU(inplace=True),
    )


def residual_input(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 512, 512, 3] -> float [B, 6, 512, 512]: centred RGB and gained high-pass residual."""
    x = images_uint8.permute(0, 3, 1, 2).float() / 255.0
    pad = HIGHPASS_KERNEL // 2
    blurred = F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="reflect"), HIGHPASS_KERNEL, stride=1
    )
    return torch.cat([x - 0.5, (x - blurred) * HIGHPASS_GAIN], dim=1)


def extras_input(maps: torch.Tensor, scalars: torch.Tensor, grid: int) -> torch.Tensor:
    """Cached extras -> float [B, 4, grid, grid]: margin map, agreement map, two global planes."""
    margin = F.interpolate(
        maps[:, 0:1].float() / MARGIN_SCALE, size=(grid, grid), mode="bilinear", align_corners=False
    )
    agree = F.interpolate(
        maps[:, 1:2].float() - AGREEMENT_CENTRE,
        size=(grid, grid),
        mode="bilinear",
        align_corners=False,
    )
    planes = torch.stack([scalars[:, 0] / MARGIN_SCALE, scalars[:, 1] - AGREEMENT_CENTRE], dim=1)
    broadcast = planes[:, :, None, None].expand(-1, -1, grid, grid)
    return torch.cat([margin, agree, broadcast], dim=1)


class Localizer(nn.Module):
    """Residual CNN localizer; `use_extras` adds the message-agreement planes at the 128 stage."""

    def __init__(self, use_extras: bool) -> None:
        super().__init__()
        self.use_extras = use_extras
        self.stem = nn.Sequential(
            # Patchify: one 4x4 / stride-4 convolution maps each grid cell (4x4 px) straight to
            # the 128 grid. Much cheaper than two strided convs at 256x256 (memory-bound on MPS).
            nn.Conv2d(
                N_RESIDUAL_INPUT_CHANNELS,
                STEM_CHANNELS,
                CELL_PIXELS,
                stride=CELL_PIXELS,
                bias=False,
            ),
            nn.GroupNorm(GROUPS, STEM_CHANNELS),
            nn.SiLU(inplace=True),
            conv_block(STEM_CHANNELS, GRID_CHANNELS, 3),
        )
        grid_in = GRID_CHANNELS + (N_EXTRAS_CHANNELS if use_extras else 0)
        self.grid_fuse = conv_block(grid_in, GRID_CHANNELS, 3)
        self.down = conv_block(GRID_CHANNELS, CONTEXT_CHANNELS, 3, stride=2)
        self.context = nn.ModuleList(
            [conv_block(CONTEXT_CHANNELS, CONTEXT_CHANNELS, 3, dilation=d) for d in DILATIONS]
        )
        self.global_proj = nn.Linear(CONTEXT_CHANNELS, CONTEXT_CHANNELS)
        self.up_fuse = conv_block(CONTEXT_CHANNELS + GRID_CHANNELS, GRID_CHANNELS, 3)
        self.head = nn.Conv2d(GRID_CHANNELS, 1, 1)

    def forward(
        self,
        images_uint8: torch.Tensor,
        maps: torch.Tensor | None,
        scalars: torch.Tensor | None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits [B, 128, 128].

        Args:
            images_uint8: [B, 512, 512, 3] uint8.
            maps: [B, 2, 8, 8] extras maps (required when `use_extras`).
            scalars: [B, 2] extras scalars (required when `use_extras`).
        """
        # `residual` lets saliency code differentiate w.r.t. the network input.
        features = self.stem(residual if residual is not None else residual_input(images_uint8))
        if self.use_extras:
            if maps is None or scalars is None:
                raise ValueError("this model was built with use_extras=True")
            features = torch.cat([features, extras_input(maps, scalars, features.shape[-1])], dim=1)
        grid = self.grid_fuse(features)
        context = self.down(grid)
        for block in self.context:
            context = context + block(context)
        pooled = self.global_proj(context.mean(dim=(2, 3)))
        context = context + pooled[:, :, None, None]
        up = F.interpolate(context, size=grid.shape[-2:], mode="bilinear", align_corners=False)
        logits: torch.Tensor = self.head(self.up_fuse(torch.cat([up, grid], dim=1)))[:, 0]
        return logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
