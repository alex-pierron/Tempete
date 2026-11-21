"""Runtime configuration: device selection and directories, overridable by environment."""

from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

ENV_DEVICE = "WMKIT_DEVICE"
ENV_DATA_DIR = "WMKIT_DATA_DIR"
ENV_WEIGHTS_DIR = "WMKIT_WEIGHTS_DIR"
ENV_RESULTS_DIR = "WMKIT_RESULTS_DIR"


def get_device() -> torch.device:
    """Return the compute device: $WMKIT_DEVICE if set, else cuda, then mps, then cpu."""
    forced = os.environ.get(ENV_DEVICE, "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dir_from_env(env_name: str, default_name: str) -> Path:
    """Resolve a directory from an env var, defaulting to PROJECT_ROOT/<default_name>."""
    raw = os.environ.get(env_name, "").strip()
    path = Path(raw).expanduser() if raw else PROJECT_ROOT / default_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Directory holding downloaded datasets (git-ignored)."""
    return _dir_from_env(ENV_DATA_DIR, "data")


def weights_dir() -> Path:
    """Directory holding checkpoints (git-ignored)."""
    return _dir_from_env(ENV_WEIGHTS_DIR, "weights")


def results_dir() -> Path:
    """Directory holding evaluation outputs (git-ignored)."""
    return _dir_from_env(ENV_RESULTS_DIR, "results")
