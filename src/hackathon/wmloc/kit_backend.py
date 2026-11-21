"""Backends built on the official starter-kit wrappers (PixelSeal, TrustMark, StegaStamp).

The kit's `wmloc/watermarks.py` is loaded BY PATH under a private module name, never through
`sys.path`: its top-level package is also called `wmloc`, and a path load keeps it from clashing
with our `hackathon.wmloc`. The wrappers are exactly the embedders and extractors that built the
dataset, so this is the backend to use on real data (and on JURECA).

Wrapper interface (from the kit): `nbits`, `resolution` (square input size of the decoder),
`logits(images [B,3,H,W] in [0,1]) -> [B, nbits]` with "> 0 means bit 1", and `features(images)`,
which for PixelSeal includes `per_position_logits` [B, nbits, 8, 8].

Directories: $WMLOC_STARTER_KIT_DIR (default `<data_dir>/wmloc/starter_kit`) and $WMLOC_MODELS
(default `<data_dir>/wmloc/models`, where `download_models.py --models_dir` put the weights).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from hackathon.common.config import load_settings
from hackathon.wmloc.constants import (
    KNOWN_METHODS,
    METHOD_PIXELSEAL,
    STARTER_KIT_SUBDIR,
    WMLOC_DATA_SUBDIR,
)
from hackathon.wmloc.errors import LocalBackendError, UnsupportedMethodError

KIT_DIR_ENV = "WMLOC_STARTER_KIT_DIR"
MODELS_DIR_ENV = "WMLOC_MODELS"
MODELS_SUBDIR = "models"
WATERMARKS_RELATIVE = Path("wmloc") / "watermarks.py"
KIT_MODULE_NAME = "_wmloc_kit_watermarks"
PIXELSEAL_MAP_STRIDE = 32
"""PixelSeal's ConvNeXt maps 256 px to 8x8 cells, i.e. 32 native pixels per cell."""
FEATURE_KEY_PER_POSITION = "per_position_logits"
DEVICE_CPU = "cpu"
DECODE_CHUNK = 64

_MODULE_CACHE: dict[Path, ModuleType] = {}


def kit_dir() -> Path:
    """Folder of the unpacked starter kit."""
    configured = os.environ.get(KIT_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return load_settings().data_dir / WMLOC_DATA_SUBDIR / STARTER_KIT_SUBDIR


def models_dir() -> Path:
    """Folder where `download_models.py` stored the weights."""
    configured = os.environ.get(MODELS_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return load_settings().data_dir / WMLOC_DATA_SUBDIR / MODELS_SUBDIR


def load_kit_module(directory: Path | None = None) -> ModuleType:
    """Import the kit's `wmloc/watermarks.py` by path (cached).

    Raises:
        LocalBackendError: the file is missing or does not import (missing dependency).
    """
    base = directory if directory is not None else kit_dir()
    path = base / WATERMARKS_RELATIVE
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    if not path.is_file():
        raise LocalBackendError(
            f"starter kit not found: {path} (run `fetch`, or set {KIT_DIR_ENV})"
        )
    spec = importlib.util.spec_from_file_location(KIT_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise LocalBackendError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise LocalBackendError(f"kit wrappers need a missing dependency: {exc}") from exc
    _MODULE_CACHE[path] = module
    return module


class KitBackend:
    """A kit wrapper seen as a `WatermarkBackend`.

    Attributes:
        n_bits: Message length of the method.
        native_size: Square input resolution of the decoder.
        map_stride: Native pixels per cell of `logit_map` (PixelSeal only), else None.
    """

    def __init__(self, wrapper: Any, has_spatial_map: bool) -> None:
        self._wrapper = wrapper
        self.n_bits: int = int(wrapper.nbits)
        self.native_size: int = int(wrapper.resolution)
        self.map_stride: int | None = PIXELSEAL_MAP_STRIDE if has_spatial_map else None

    @torch.no_grad()
    def bit_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Soft outputs [B, n_bits] (sign = decoded bit), computed in chunks."""
        chunks = [
            self._wrapper.logits(images[start : start + DECODE_CHUNK]).detach().float().cpu()
            for start in range(0, images.shape[0], DECODE_CHUNK)
        ]
        return torch.cat(chunks) if chunks else torch.empty(0, self.n_bits)

    @torch.no_grad()
    def logit_map(self, images: torch.Tensor) -> torch.Tensor:
        """Per-position bit logits [B, n_bits, 8, 8] (PixelSeal's `features`).

        Raises:
            UnsupportedMethodError: the wrapper has no spatial map, or the size is not native.
        """
        if self.map_stride is None:
            raise UnsupportedMethodError("this kit wrapper exposes no per-position logit map")
        if images.shape[-2:] != (self.native_size, self.native_size):
            raise UnsupportedMethodError(
                f"logit_map needs {self.native_size}x{self.native_size} inputs"
            )
        maps = self._wrapper.features(images)[FEATURE_KEY_PER_POSITION]
        return torch.as_tensor(maps).detach().float().cpu()


def make_backend(method: str, device: str = DEVICE_CPU) -> KitBackend | None:
    """Backend factory over the kit wrappers (the CLI default for real data).

    Args:
        method: `pixelseal`, `trustmark` or `stegastamp`; anything else returns None.
        device: Torch device for PixelSeal and TrustMark ("auto" picks cuda, then mps, then cpu);
            StegaStamp runs in ONNX Runtime and only uses CUDA when `device` is cuda.

    Raises:
        LocalBackendError: kit, weights or a dependency (videoseal clone, trustmark, onnxruntime)
            is missing.
    """
    if method not in KNOWN_METHODS:
        return None
    from hackathon.wmloc.local_backend import pick_device

    resolved = pick_device(device)
    module = load_kit_module()
    try:
        wrapper = module.load_watermark(method, models_dir=models_dir(), device=resolved)
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise LocalBackendError(
            f"cannot build the {method} wrapper ({type(exc).__name__}: {exc}); "
            f"did you run `download_models.py --models_dir {models_dir()}`?"
        ) from exc
    return KitBackend(wrapper, has_spatial_map=method == METHOD_PIXELSEAL)
