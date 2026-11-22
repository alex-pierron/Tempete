"""Fast, numerically faithful access to the three official decoders.

The kit wrappers resize every input with `antialias=True`, an operator torch 2.5 lacks on MPS: with
`PYTORCH_ENABLE_MPS_FALLBACK=1` it silently runs on the CPU and dominates the cost of small crops
(5.8 crops/s measured for 64 px crops against 31 crops/s for 256 px ones). Antialiasing only matters
when DOWNSCALING, so for upscaling (every crop smaller than the decoder resolution) the plain
bilinear op on the device gives the same numbers. `Decoder` does that, keeps the CPU path for
downscaling, and calls the same network the kit calls. `check_parity` proves the equivalence.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional

from hackathon.wmloc.constants import METHOD_PIXELSEAL, METHOD_STEGASTAMP, METHOD_TRUSTMARK
from hackathon.wmloc.kit_backend import load_kit_module, models_dir
from hackathon.wmloc.research.errors import ResearchError

DEVICE_CPU = "cpu"
DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
TRUSTMARK_INPUT_OFFSET = 1.0
"""TrustMark feeds the network `x * 2 - 1`."""
THREAD_LIMIT = 4
"""Torch intra-op threads: the machine is shared by several agents."""
PARITY_TOLERANCE = 1e-3
"""Largest accepted |difference| between our logits and the kit's, in logit units."""


class DecoderParityError(ResearchError):
    """Our fast path disagrees with the official wrapper."""


def pick_device() -> str:
    """cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return DEVICE_CUDA
    if torch.backends.mps.is_available():
        return DEVICE_MPS
    return DEVICE_CPU


class Decoder:
    """One loaded official decoder (load each model ONCE per process).

    Attributes:
        method: `pixelseal`, `trustmark` or `stegastamp`.
        n_bits: Message length.
        resolution: Square input size of the network.
        device: Device the network runs on (StegaStamp always runs in ONNX Runtime on CPU).
    """

    def __init__(self, method: str, device: str | None = None) -> None:
        torch.set_num_threads(THREAD_LIMIT)
        self.method = method
        self.device = device if device is not None else pick_device()
        module = load_kit_module()
        self._wrapper: Any = module.load_watermark(
            method, models_dir=models_dir(), device=self.device
        )
        self.n_bits: int = int(self._wrapper.nbits)
        self.resolution: int = int(self._wrapper.resolution)

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to the network resolution: antialiased on CPU if shrinking, else on device."""
        size = self.resolution
        if images.shape[-2:] == (size, size):
            return images.to(self.device)
        shrinking = images.shape[-1] > size or images.shape[-2] > size
        if shrinking:
            resized = functional.interpolate(
                images.cpu(),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return resized.to(self.device)
        return functional.interpolate(
            images.to(self.device), size=(size, size), mode="bilinear", align_corners=False
        )

    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Bit logits [B, n_bits] on the CPU (> 0 means bit 1) for [B, 3, H, W] images in [0, 1]."""
        if self.method == METHOD_STEGASTAMP:
            return torch.as_tensor(self._wrapper.logits(images)).float().cpu()
        x = self._resize(images)
        if self.method == METHOD_PIXELSEAL:
            out = self._wrapper.det(x)[:, 1:]
        elif self.method == METHOD_TRUSTMARK:
            out = self._wrapper.net(x * 2 - TRUSTMARK_INPUT_OFFSET)
        else:
            raise ResearchError(f"unknown method {self.method!r}")
        return torch.as_tensor(out).float().cpu()

    @torch.no_grad()
    def position_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Per-position bit logits [B, n_bits, h, w] on the CPU (PixelSeal and TrustMark).

        Both heads are `linear(mean over positions)`, so the mean over (h, w) of this map equals
        `logits` exactly: each position's share of the global decision.

        Raises:
            ResearchError: StegaStamp has no spatial head (dense layers after the convolutions).
        """
        if self.method == METHOD_STEGASTAMP:
            raise ResearchError("stegastamp exposes no per-position logits")
        x = self._resize(images)
        features = self._wrapper.features(x)["per_position_logits"]
        return torch.as_tensor(features).float().cpu()

    @torch.no_grad()
    def position_logits_at_side(self, images: torch.Tensor, side: int) -> torch.Tensor:
        """Per-position bit logits of the images resized to `side` x `side`, [B, n_bits, h, w].

        The extractor is fully convolutional up to its head, so feeding a larger image yields a
        finer map (one cell per 32 network pixels). This is a scale-space probe: a rectangle that
        was watermarked at its own scale (crop resized to the decoder resolution) is read at the
        right scale when the whole image is enlarged so that the rectangle spans the resolution.

        Raises:
            ResearchError: StegaStamp (no spatial head).
        """
        if self.method == METHOD_STEGASTAMP:
            raise ResearchError("stegastamp exposes no per-position logits")
        shrinking = side < images.shape[-1]
        resized = functional.interpolate(
            images.cpu() if shrinking else images.to(self.device),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
            antialias=shrinking,
        ).to(self.device)
        wrapper = self._wrapper
        if self.method == METHOD_PIXELSEAL:
            det = wrapper.det
            latent = det.convnext(det.preprocess(resized))
            embedding = det.pixel_decoder.output_upscaling(latent)
            head = det.pixel_decoder.linear
            out = torch.einsum("bchw,oc->bohw", embedding, head.weight)
            out = (out + head.bias[None, :, None, None])[:, 1:]
        else:
            net = wrapper.net
            x = net.maxpool(net.relu(net.bn1(net.conv1(resized * 2 - TRUSTMARK_INPUT_OFFSET))))
            for layer in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(net, layer)(x)
            out = torch.einsum("bchw,oc->bohw", x, net.fc.weight) + net.fc.bias[None, :, None, None]
        return torch.as_tensor(out).float().cpu()

    def kit_logits(self, images: torch.Tensor) -> torch.Tensor:
        """The unmodified kit output, for the parity check."""
        return torch.as_tensor(self._wrapper.logits(images)).float().cpu()


def check_parity(decoder: Decoder, images: torch.Tensor) -> float:
    """Largest |ours - kit| over `images` (a few crops of mixed sizes are enough).

    Raises:
        DecoderParityError: the difference exceeds `PARITY_TOLERANCE`.
    """
    worst = float((decoder.logits(images) - decoder.kit_logits(images)).abs().max().item())
    if worst > PARITY_TOLERANCE:
        raise DecoderParityError(f"{decoder.method}: fast path differs by {worst:.2e}")
    return worst
