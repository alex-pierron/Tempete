"""Light access to the official starter kit for the learned-localization research (team B).

This module deliberately avoids `hackathon.common` (Pydantic settings) so that it also imports in
the dedicated kit venv (`wmloc/.venv-kit`, Python 3.11, torch 2.5.1, trustmark, onnxruntime),
which is the only environment where all three embedders (PixelSeal, TrustMark, StegaStamp) run.
The data directory comes from `HACKATHON_DATA_DIR`.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any

DATA_DIR_ENV = "HACKATHON_DATA_DIR"
KIT_MODULE_NAME = "_wmloc_research_kit_watermarks"
METHODS: tuple[str, ...] = ("pixelseal", "trustmark", "stegastamp")
N_BITS_BY_METHOD: dict[str, int] = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
IMAGE_SIZE = 512
GRID_SIZE = 128

_MODULE_CACHE: dict[str, ModuleType] = {}


class ResearchKitError(RuntimeError):
    """Raised when the data directory, the starter kit or a model is missing."""


def data_dir() -> Path:
    """Root `data/hackathon` directory taken from `HACKATHON_DATA_DIR`.

    Raises:
        ResearchKitError: the environment variable is unset or points to nothing.
    """
    raw = os.environ.get(DATA_DIR_ENV)
    if not raw:
        raise ResearchKitError(f"set {DATA_DIR_ENV} to the data/hackathon directory")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ResearchKitError(f"{DATA_DIR_ENV}={path} is not a directory")
    return path


def wmloc_dir() -> Path:
    """Folder holding the task data (`release/`, `starter_kit/`, `models/`)."""
    return data_dir() / "wmloc"


def train_dir() -> Path:
    """Real labelled split (510 images, masks, metadata)."""
    return wmloc_dir() / "release" / "train"


def test_dir() -> Path:
    """Real unlabelled split (1020 images)."""
    return wmloc_dir() / "release" / "test"


def load_kit_module() -> ModuleType:
    """Import the kit's `wmloc/watermarks.py` by path under a private module name (cached).

    Raises:
        ResearchKitError: the kit is missing or one of its dependencies cannot be imported.
    """
    path = wmloc_dir() / "starter_kit" / "wmloc" / "watermarks.py"
    key = str(path)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    if not path.is_file():
        raise ResearchKitError(f"starter kit not found: {path}")
    spec = importlib.util.spec_from_file_location(KIT_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ResearchKitError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def load_wrapper(method: str, device: str) -> Any:
    """Official wrapper (`embed`, `logits`, `features`, `nbits`, `resolution`) of one method.

    Raises:
        ResearchKitError: unknown method, or weights / dependency missing.
    """
    if method not in METHODS:
        raise ResearchKitError(f"unknown method {method!r}, expected one of {METHODS}")
    module = load_kit_module()
    try:
        return module.load_watermark(method, models_dir=wmloc_dir() / "models", device=device)
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise ResearchKitError(f"cannot build the {method} wrapper: {exc}") from exc
