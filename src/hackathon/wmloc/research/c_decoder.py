"""Official decoders without any dependency on `hackathon.common` (team C).

`research.decoding.Decoder` imports `kit_backend`, which pulls the Pydantic settings (and so the
`.env` parser); that is unwanted on JURECA, whose official venv carries only the kit's packages.
`LightDecoder` builds the same wrappers through `research.kit.load_wrapper` and exposes the two
things the window search needs: `logits` for crops already at the decoder resolution, and the
`method` / `resolution` attributes.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from hackathon.wmloc.constants import METHOD_PIXELSEAL, METHOD_STEGASTAMP, METHOD_TRUSTMARK
from hackathon.wmloc.research.c_batching import DEFAULT_BATCH
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.kit import load_wrapper

TRUSTMARK_INPUT_OFFSET = 1.0
"""TrustMark feeds the network `x * 2 - 1`."""
BATCH_ENV = "WMC_DECODE_BATCH"
"""Environment override of the decode batch size (default `DEFAULT_BATCH`)."""
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"


def pick_device() -> str:
    """cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return DEVICE_CUDA
    if torch.backends.mps.is_available():
        return DEVICE_MPS
    return DEVICE_CPU


class LightDecoder:
    """One loaded official decoder (load each model ONCE per process).

    Attributes:
        method: `pixelseal`, `trustmark` or `stegastamp`.
        n_bits: Message length.
        resolution: Square input size of the network.
        device: Device the network runs on (StegaStamp runs in ONNX Runtime).
    """

    def __init__(self, method: str, device: str | None = None) -> None:
        self.method = method
        self.batch_size = int(os.environ.get(BATCH_ENV, DEFAULT_BATCH))
        self.device = device if device is not None else pick_device()
        self._wrapper: Any = load_wrapper(method, self.device)
        self.n_bits: int = int(self._wrapper.nbits)
        self.resolution: int = int(self._wrapper.resolution)

    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Bit logits [B, n_bits] on the CPU for [B, 3, resolution, resolution] images in [0, 1].

        The caller resizes (see `c_rect_search.resize_crop`); a batch already at the network
        resolution passes through the kit's own resize unchanged.

        Raises:
            ResearchError: unknown method.
        """
        if self.method == METHOD_STEGASTAMP:
            return torch.as_tensor(self._wrapper.logits(images)).float().cpu()
        x = images.to(self.device)
        if self.method == METHOD_PIXELSEAL:
            out = self._wrapper.det(x)[:, 1:]
        elif self.method == METHOD_TRUSTMARK:
            out = self._wrapper.net(x * 2 - TRUSTMARK_INPUT_OFFSET)
        else:
            raise ResearchError(f"unknown method {self.method!r}")
        return torch.as_tensor(out).float().cpu()
