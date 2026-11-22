"""Segmentation head over frozen-extractor features, with IoU-targeting losses (team B2).

The head never sees raw pixels except a 128 px RGB thumbnail used for sharp boundaries. Its inputs
are the cached feature maps of `b2_features` / `b2_dino` (per-method adapters, one 1x1 projection
per source) fused on a 32 x 32 grid (one cell = 16 px of the 512 image), refined by a small
dilated trunk with a global branch (so it can tell the "whole image is marked, find the hole"
regime from the "small marked region" regime) and decoded to the 128 x 128 evaluation grid.

The trunk and decoder are SHARED by all methods: the adapters are the only method-specific
parameters, so the shape / object prior is learned from all 510 real masks instead of 170.

Losses target the metric directly: the per-image mIoU averages the IoU of the clean and of the
watermarked class, so the surrogate is the (soft or Lovasz) IoU of BOTH classes, per image, plus a
BCE term for stable gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional

GRID = 32
"""Working grid of the trunk: one cell per 16 px of the 512 image."""
OUT_SIDE = 128
"""Side of the evaluation grid."""
ADAPTER_CHANNELS = 16
TRUNK_CHANNELS = 48
RGB_CHANNELS = 16
DECODER_CHANNELS = 32
GROUPS = 8
DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
N_SCALARS = 3
EPSILON = 1e-6
SMOOTH = 1.0
"""Additive smoothing of the soft IoU (also makes an empty union score 1)."""

SOURCES_PIXELSEAL: dict[str, tuple[int, int]] = {
    "n_st1": (192, 32),
    "n_st2": (384, 16),
    "n_st3": (768, 8),
    "u_st2": (384, 16),
    "u_st3": (768, 16),
    "n_ev": (4, 8),
    "u_ev": (4, 16),
}
SOURCES_TRUSTMARK: dict[str, tuple[int, int]] = {
    "n_l2": (512, 16),
    "n_l3": (1024, 16),
    "n_l4": (2048, 8),
    "n_ev": (3, 8),
    "u_ev": (3, 16),
}
SOURCES_STEGASTAMP: dict[str, tuple[int, int]] = {
    "ss_c3": (64, 25),
    "ss_c5": (64, 25),
    "ss_c7": (128, 13),
}
SOURCES_BY_METHOD: dict[str, dict[str, tuple[int, int]]] = {
    "pixelseal": SOURCES_PIXELSEAL,
    "trustmark": SOURCES_TRUSTMARK,
    "stegastamp": SOURCES_STEGASTAMP,
}
PIXEL_KEY = "pixels"
"""Key of the uint8 [B, 3, 512, 512] full-resolution image in the `sources` dict."""
PIXEL_CHANNELS = 32
PIXEL_HIGHPASS_KERNEL = 5
PIXEL_HIGHPASS_GAIN = 12.0
"""Residuals are 1-2 grey levels out of 255: the gain brings them to O(0.05-0.1)."""
PIXEL_PATCH = 4
"""One 4 x 4 pixel patch per cell of the 128 evaluation grid."""
WINDOW_KEY = "win"
WINDOW_SPEC: tuple[int, int] = (14, 32)
"""Window-decode maps on the trunk grid: 7 window sizes x (margin, agreement), see `b2_windows`."""
DINO_KEY = "dino"
DINO_SPEC: tuple[int, int] = (384, 32)


@dataclass(frozen=True)
class HeadConfig:
    """What the head reads and how it regularises.

    Attributes:
        sources: Feature sources used per method (names from `SOURCES_BY_METHOD`).
        use_dino: Whether the shared DINOv2 content prior is read.
        use_rgb: Whether the 128 px RGB thumbnail feeds the decoder.
        dropout: Channel dropout on every adapter input (the main regulariser: 136 images).
        width: Trunk width.
        use_pixels: Whether a small TRAINABLE stem reads the full-resolution pixels (image and its
            high-pass residual) next to the frozen features: it can learn the embedder's own
            texture statistics, which synthetic data makes learnable.
    """

    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    use_dino: bool = True
    use_rgb: bool = True
    dropout: float = 0.3
    width: int = TRUNK_CHANNELS
    use_pixels: bool = False


def conv_block(in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1) -> nn.Sequential:
    """Conv + GroupNorm + SiLU with 'same' padding."""
    padding = dilation * (kernel - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
        nn.GroupNorm(min(GROUPS, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class Adapter(nn.Module):
    """Standardise, drop and project one source to `ADAPTER_CHANNELS` on the trunk grid."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)
        self.project = nn.Conv2d(channels, ADAPTER_CHANNELS, 1)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """[B, C, h, w] -> [B, ADAPTER_CHANNELS, GRID, GRID]."""
        projected: torch.Tensor = self.project(self.drop(self.norm(maps.float())))
        if projected.shape[-1] != GRID:
            projected = functional.interpolate(
                projected, size=(GRID, GRID), mode="bilinear", align_corners=False
            )
        return projected


class PixelStem(nn.Module):
    """Trainable stem on [image - 0.5, gained high-pass residual], 512 px -> 128 grid features."""

    def __init__(self) -> None:
        super().__init__()
        self.patchify = nn.Sequential(
            nn.Conv2d(6, PIXEL_CHANNELS, PIXEL_PATCH, stride=PIXEL_PATCH, bias=False),
            nn.GroupNorm(GROUPS, PIXEL_CHANNELS),
            nn.SiLU(inplace=True),
        )
        self.refine = conv_block(PIXEL_CHANNELS, PIXEL_CHANNELS, 3)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """uint8 [B, 3, 512, 512] -> float [B, PIXEL_CHANNELS, 128, 128]."""
        image = pixels.float() / 255.0
        pad = PIXEL_HIGHPASS_KERNEL // 2
        blurred = functional.avg_pool2d(
            functional.pad(image, (pad, pad, pad, pad), mode="reflect"),
            PIXEL_HIGHPASS_KERNEL,
            stride=1,
        )
        stacked = torch.cat([image - 0.5, (image - blurred) * PIXEL_HIGHPASS_GAIN], dim=1)
        features: torch.Tensor = self.refine(self.patchify(stacked))
        return features


class B2Net(nn.Module):
    """Multi-method segmentation head; see the module docstring."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.adapters = nn.ModuleDict()
        self.fuse = nn.ModuleDict()
        self.scalar_proj = nn.ModuleDict()
        for method, names in config.sources.items():
            specs = {**SOURCES_BY_METHOD.get(method, {}), WINDOW_KEY: WINDOW_SPEC}
            self.adapters[method] = nn.ModuleDict(
                {name: Adapter(specs[name][0], config.dropout) for name in names}
            )
            n_inputs = len(names) + (1 if config.use_dino else 0)
            self.fuse[method] = conv_block(n_inputs * ADAPTER_CHANNELS, width, kernel=1)
            self.scalar_proj[method] = nn.Linear(N_SCALARS, width)
        self.dino_adapter = Adapter(DINO_SPEC[0], config.dropout) if config.use_dino else None
        self.trunk = nn.ModuleList([conv_block(width, width, 3, d) for d in DILATIONS])
        self.global_proj = nn.Linear(width, width)
        rgb_in = RGB_CHANNELS if config.use_rgb else 0
        self.rgb_stem = conv_block(3, RGB_CHANNELS, 3) if config.use_rgb else None
        self.pixel_stem = PixelStem() if config.use_pixels else None
        pixel_in = PIXEL_CHANNELS if config.use_pixels else 0
        self.up_fuse = conv_block(width + rgb_in + pixel_in, DECODER_CHANNELS, 3)
        self.refine = conv_block(DECODER_CHANNELS, DECODER_CHANNELS, 3)
        self.head = nn.Conv2d(DECODER_CHANNELS, 1, 1)

    def forward(
        self,
        method: str,
        sources: dict[str, torch.Tensor],
        scalars: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Logits [B, 128, 128].

        Args:
            method: Key of the per-method adapters.
            sources: Feature maps by source name (each [B, C, h, w]), plus `dino` if enabled.
            scalars: Whole-image regime scalars [B, N_SCALARS].
            rgb: [B, 3, 128, 128] float in [-0.5, 0.5].
        """
        parts = [
            adapter(sources[name])
            for name, adapter in self.adapters[method].items()  # type: ignore[operator]
        ]
        if self.dino_adapter is not None:
            parts.append(self.dino_adapter(sources[DINO_KEY]))
        features = self.fuse[method](torch.cat(parts, dim=1))
        features = features + self.scalar_proj[method](scalars.float())[:, :, None, None]
        for block in self.trunk:
            features = features + block(features)
        pooled = self.global_proj(features.mean(dim=(2, 3)))
        features = features + pooled[:, :, None, None]
        upsampled = functional.interpolate(
            features, size=(OUT_SIDE, OUT_SIDE), mode="bilinear", align_corners=False
        )
        if self.rgb_stem is not None:
            upsampled = torch.cat([upsampled, self.rgb_stem(rgb)], dim=1)
        if self.pixel_stem is not None:
            upsampled = torch.cat([upsampled, self.pixel_stem(sources[PIXEL_KEY])], dim=1)
        logits: torch.Tensor = self.head(self.refine(self.up_fuse(upsampled)))[:, 0]
        return logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def soft_miou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean over images of the soft mIoU of both classes. logits, target: [B, H, W]."""
    probability = torch.sigmoid(logits)
    target = target.float()
    ious = []
    for pred, gold in ((probability, target), (1 - probability, 1 - target)):
        intersection = (pred * gold).flatten(1).sum(1)
        union = (pred + gold - pred * gold).flatten(1).sum(1)
        ious.append((intersection + SMOOTH) / (union + SMOOTH))
    return 1.0 - torch.stack(ious).mean()


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-image Lovasz hinge (Berman et al. 2018) of the class `labels == 1`, [B].

    Args:
        logits: [B, N] scores of the positive class.
        labels: [B, N] 0/1.
    """
    labels = labels.float()
    errors = 1.0 - logits * (2.0 * labels - 1.0)
    errors_sorted, permutation = torch.sort(errors, dim=1, descending=True)
    gold = torch.gather(labels, 1, permutation)
    total = gold.sum(dim=1, keepdim=True)
    intersection = total - gold.cumsum(dim=1)
    union = total + (1.0 - gold).cumsum(dim=1)
    jaccard = 1.0 - intersection / union.clamp(min=EPSILON)
    jaccard = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], dim=1)
    return (functional.relu(errors_sorted) * jaccard).sum(dim=1)


def lovasz_miou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over images of the Lovasz hinge of BOTH classes. logits, target: [B, H, W]."""
    flat_logits = logits.flatten(1)
    flat_target = target.flatten(1).float()
    foreground = lovasz_hinge(flat_logits, flat_target)
    background = lovasz_hinge(-flat_logits, 1.0 - flat_target)
    return 0.5 * (foreground + background).mean()


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    soft_weight: float,
    lovasz_weight: float,
) -> torch.Tensor:
    """Weighted BCE + soft mIoU + Lovasz mIoU. Zero-weight terms are skipped."""
    loss = logits.new_zeros(())
    if bce_weight > 0:
        loss = loss + bce_weight * functional.binary_cross_entropy_with_logits(
            logits, target.float()
        )
    if soft_weight > 0:
        loss = loss + soft_weight * soft_miou_loss(logits, target)
    if lovasz_weight > 0:
        loss = loss + lovasz_weight * lovasz_miou_loss(logits, target)
    return loss


DIHEDRAL_ELEMENTS = 8


def dihedral(tensor: torch.Tensor, element: int) -> torch.Tensor:
    """Apply one of the 8 symmetries of the square to the last two dims (rot90 x flip)."""
    rotated = torch.rot90(tensor, element % 4, dims=(-2, -1))
    return torch.flip(rotated, dims=(-1,)) if element >= 4 else rotated


def inverse_dihedral(tensor: torch.Tensor, element: int) -> torch.Tensor:
    """Undo `dihedral(., element)`: `inverse_dihedral(dihedral(x, e), e) == x`."""
    unflipped = torch.flip(tensor, dims=(-1,)) if element >= 4 else tensor
    return torch.rot90(unflipped, -(element % 4), dims=(-2, -1))
