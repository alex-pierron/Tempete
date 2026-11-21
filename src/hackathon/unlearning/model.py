"""Model builder, loading, preprocessing and prediction, identical to the official task_template.py.

The only deliberate difference: the device is a parameter instead of a module global, so the same
code runs on the A100 (CUDA, fp16 autocast like the server), on the Apple M5 (MPS, fp32) and on
CPU (fp32, used by the tests). fp16 autocast is enabled on CUDA only, because the server does it
there and because MPS/CPU autocast would change numerics without matching the server anyway.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import timm
import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import (
    DEFAULT_BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    LAYER_SCALE_INIT,
    MODEL_NAME,
    NORMALISATION_CENTER,
    NORMALISATION_SCALE,
    NUM_CLASSES,
    PIXEL_MAX,
)


class ModelLoadError(HackathonError):
    """The checkpoint is missing or does not fit the official architecture."""


def select_device(requested: str | None = None) -> torch.device:
    """Pick the compute device: an explicit request, else CUDA, else MPS, else CPU.

    Args:
        requested: "cuda", "mps" or "cpu"; None means automatic.

    Raises:
        ModelLoadError: the requested accelerator is not available (fail loudly: a silent CPU
            fallback on JURECA would waste the reservation).
    """
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("CUDA requested but not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ModelLoadError("MPS requested but not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    num_classes: int = NUM_CLASSES,
    model_name: str = MODEL_NAME,
    **overrides: object,
) -> nn.Module:
    """ViT-B/16 at 224x224 with LayerScale (init 1e-4), exactly as the original and gold models.

    `overrides` is for tests only (a tiny ViT through `embed_dim`, `depth`, ...); the real task
    never passes any.
    """
    model: nn.Module = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        init_values=LAYER_SCALE_INIT,
        **overrides,
    )
    return model


def load_state_dict_file(path: Path) -> dict[str, Tensor]:
    """Read a weights-only state dict from disk on CPU (what the server does with a submission)."""
    if not path.is_file():
        raise ModelLoadError(f"Checkpoint not found: {path} (run `fetch` first).")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ModelLoadError(f"{path} does not contain a state dict (got {type(loaded).__name__}).")
    return loaded


def load_model(path: Path, device: torch.device | str = "cpu") -> nn.Module:
    """Build the architecture, load `path` with strict=True, move to `device`, set eval mode."""
    model = build_model()
    try:
        model.load_state_dict(load_state_dict_file(path), strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"{path} does not load strictly into the architecture: {exc}") from exc
    return model.to(device).eval()


def preprocess(images_uint8: Tensor, device: torch.device | str) -> Tensor:
    """uint8 [B, 224, 224, 3] -> float32 [B, 3, 224, 224] in [-1, 1] (mean = std = 0.5).

    Exactly the server preprocessing: do not resize or normalise a second time.
    """
    expected = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    if images_uint8.dtype != torch.uint8 or tuple(images_uint8.shape[1:]) != expected:
        raise ValueError(
            f"Expected uint8 [B, {IMAGE_SIZE}, {IMAGE_SIZE}, {IMAGE_CHANNELS}], "
            f"got {images_uint8.dtype} {tuple(images_uint8.shape)}"
        )
    pixels = images_uint8.to(device).permute(0, 3, 1, 2).float() / PIXEL_MAX
    return (pixels - NORMALISATION_CENTER) / NORMALISATION_SCALE


def model_device(model: nn.Module) -> torch.device:
    """Device of the first parameter of `model`."""
    return next(model.parameters()).device


@contextmanager
def server_autocast(device: torch.device | str) -> Iterator[None]:
    """fp16 autocast on CUDA only, like the server's forward pass (no-op elsewhere)."""
    device_type = torch.device(device).type
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device_type == "cuda"):
        yield


@torch.no_grad()
def predict(model: nn.Module, images_uint8: Tensor, batch_size: int = DEFAULT_BATCH_SIZE) -> Tensor:
    """Float32 logits [N, 100] on CPU for uint8 images [N, 224, 224, 3], computed like the server.

    The model stays on its own device; eval mode is forced (and not restored: the evaluation of a
    submission is always in eval mode).
    """
    model.eval()
    device = model_device(model)
    outputs: list[Tensor] = []
    for start in range(0, len(images_uint8), batch_size):
        batch = preprocess(images_uint8[start : start + batch_size], device)
        with server_autocast(device):
            outputs.append(model(batch).float().cpu())
    if not outputs:
        return torch.empty((0, NUM_CLASSES))
    return torch.cat(outputs)


def cpu_float32_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Detached CPU float32 copy of the state dict (floating tensors only are cast).

    Integer buffers, if any, keep their dtype: the architecture has none today, but casting them
    would break `strict=True` loading on the day timm adds one.
    """
    # clone(): the result must not alias the live parameters, which a later step may keep training.
    return {
        key: (
            value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()
        ).clone()
        for key, value in model.state_dict().items()
    }
