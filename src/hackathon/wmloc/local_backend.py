"""LOCAL ONLY: Pixel Seal backend built on wmkit + videoseal (not available in the JURECA env).

This is the one module of the package allowed to import wmkit, and it does so lazily, inside
functions. On JURECA the official starter-kit wrappers replace it (see `DecodeBackend`, the
adapter for a decode-only wrapper, and docs/hackathon/wmloc.md for what to wire).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from hackathon.wmloc.constants import METHOD_PIXELSEAL, PIXELSEAL_N_BITS, PIXELSEAL_NATIVE_SIZE
from hackathon.wmloc.errors import LocalBackendError, UnsupportedMethodError

WEIGHTS_ENV = "WMKIT_WEIGHTS_DIR"
PIXELSEAL_SLIM_FILE = "pixelseal.slim.pth"
PIXELSEAL_MAP_STRIDE = 32
"""ConvNeXt-tiny extractor: 4 (stem) x 2 x 2 x 2 (downsamplings) = 32 native px per map cell."""
DEVICE_AUTO = "auto"


def pick_device(requested: str = DEVICE_AUTO) -> str:
    """Resolve a device string: explicit value as given, else cuda, then mps, then cpu."""
    if requested != DEVICE_AUTO:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PixelSealBackend:
    """Pixel Seal decoder and embedder through wmkit's `SealWatermarker`.

    Attributes:
        n_bits: 256.
        native_size: 256 (inputs are resized to it by Pixel Seal).
        map_stride: 32 native pixels per cell of the spatial logit map.
    """

    n_bits: int = PIXELSEAL_N_BITS
    native_size: int = PIXELSEAL_NATIVE_SIZE
    map_stride: int | None = PIXELSEAL_MAP_STRIDE

    def __init__(self, watermarker: Any) -> None:
        """Wrap a loaded `wmkit.seal.SealWatermarker` (typed Any: wmkit is a lazy dependency)."""
        self._watermarker = watermarker
        self.device: torch.device = watermarker.device

    @torch.no_grad()
    def bit_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Global bit logits [B, 256] (the same numbers the whole-image decoder gives)."""
        logits: torch.Tensor = self._watermarker.extract_logits(images)
        return logits.detach().cpu()

    @torch.no_grad()
    def logit_map(self, images: torch.Tensor) -> torch.Tensor:
        """Per-location bit logits [B, 256, 8, 8] for [B, 3, 256, 256] images.

        The extractor's head is Linear(768, 1 + 256) applied after a spatial mean; a linear map
        commutes with the mean, so applying it per location gives a map whose spatial average
        equals `bit_logits` exactly (checked in tests). Channel 0 (the unused detection logit) is
        dropped.
        """
        if images.shape[-2:] != (self.native_size, self.native_size):
            raise UnsupportedMethodError(
                f"logit_map needs {self.native_size}x{self.native_size} inputs, "
                f"got {tuple(images.shape[-2:])}"
            )
        detector = self._watermarker.model.detector
        batch = images.to(self.device, torch.float32)
        features = detector.convnext(detector.preprocess(batch))
        upscaled = detector.pixel_decoder.output_upscaling(features)
        head = detector.pixel_decoder.linear
        per_location = functional.linear(upscaled.permute(0, 2, 3, 1), head.weight, head.bias)
        return per_location.permute(0, 3, 1, 2)[:, 1:].detach().cpu()

    @torch.no_grad()
    def embed(self, images: torch.Tensor, message_bits: torch.Tensor) -> torch.Tensor:
        """Watermark `images` [B, 3, H, W] in [0, 1] with one message [256]; returns [B, 3, H, W]."""
        marked: torch.Tensor = self._watermarker.embed(images, message_bits.reshape(1, -1))
        return marked.detach().cpu()


class DecodeBackend:
    """Adapter for a decode-only wrapper (TrustMark, StegaStamp, or the kit's PixelSeal).

    Args:
        decode: Callable mapping [B, 3, H, W] images in [0, 1] to [B, n_bits] bit scores.
        n_bits: Message length of the method.
        native_size: Square input size of the decoder (used by callers that resize).
        bit_threshold: Value separating bit 0 from bit 1 in the scores (0.0 for logits, 0.5 for
            probabilities or hard bits); scores are shifted by it so that their sign is the bit.
    """

    map_stride: int | None = None

    def __init__(
        self,
        decode: Callable[[torch.Tensor], torch.Tensor],
        n_bits: int,
        native_size: int,
        bit_threshold: float = 0.0,
    ) -> None:
        self._decode = decode
        self.n_bits = n_bits
        self.native_size = native_size
        self._bit_threshold = bit_threshold

    @torch.no_grad()
    def bit_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Decoded scores shifted so that sign = bit, [B, n_bits]."""
        return self._decode(images).detach().to(torch.float32).cpu() - self._bit_threshold

    def logit_map(self, images: torch.Tensor) -> torch.Tensor:
        """Not available: a decode-only wrapper has no spatial map."""
        raise UnsupportedMethodError("decode-only backend has no per-location logit map")


def load_local_pixelseal(device: str = DEVICE_AUTO) -> PixelSealBackend:
    """Load Pixel Seal through wmkit (weights from $WMKIT_WEIGHTS_DIR or <wmkit root>/weights).

    Raises:
        LocalBackendError: wmkit/videoseal missing, or the checkpoint is not on disk (we never
            trigger a download from here).
    """
    try:
        from wmkit.config import weights_dir
        from wmkit.seal import load_seal
    except ImportError as exc:
        raise LocalBackendError(f"wmkit / videoseal not importable: {exc}") from exc
    checkpoint = Path(weights_dir()) / "seal" / PIXELSEAL_SLIM_FILE
    if not checkpoint.is_file():
        raise LocalBackendError(
            f"Pixel Seal checkpoint not found at {checkpoint}: set {WEIGHTS_ENV} to the folder "
            "holding seal/pixelseal.slim.pth (git-ignored, not in worktrees)."
        )
    return PixelSealBackend(load_seal(device=pick_device(device)))


def make_backend(method: str, device: str = DEVICE_AUTO) -> PixelSealBackend | None:
    """Default backend factory of the CLI: Pixel Seal through wmkit, nothing for other methods.

    Factory contract (for `--backend-factory module:function`): `(method, device)` returns a
    `WatermarkBackend` for the method, or None if it is not wired (the CLI then falls back to
    the all-ones mask for those images).
    """
    if method == METHOD_PIXELSEAL:
        return load_local_pixelseal(device)
    return None
