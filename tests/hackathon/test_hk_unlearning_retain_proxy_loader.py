"""Tests of the drop-in loader of the generated retain proxy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.generate import chunk_path, save_chunk
from hackathon.unlearning.research.retain_proxy.loader import (
    LoaderError,
    load_generated_proxy_images,
)


def _images(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def _chunks(root: Path) -> Path:
    for first, count in ((0, 4), (4, 3)):
        save_chunk(
            chunk_path(root, first),
            np.arange(first, first + count),
            _images(count, first),
            np.zeros((count, NUM_CLASSES), dtype=np.float32),
        )
    return root


def test_loads_chunks_and_assembled_files_as_uint8_nhwc(tmp_path: Path) -> None:
    root = _chunks(tmp_path / "chunks")
    everything = load_generated_proxy_images(root, count=100, seed=0)
    assert everything.shape == (7, IMAGE_SIZE, IMAGE_SIZE, 3) and everything.dtype == torch.uint8
    dataset = tmp_path / "proxy.npz"
    np.savez(dataset, images=_images(5, 9), labels=np.zeros(5, dtype=np.int64))
    assert load_generated_proxy_images(dataset, count=100, seed=0).shape[0] == 5


def test_draw_is_seeded_sorted_and_bounded(tmp_path: Path) -> None:
    root = _chunks(tmp_path / "chunks")
    first = load_generated_proxy_images(root, count=3, seed=1)
    second = load_generated_proxy_images(root, count=3, seed=1)
    other = load_generated_proxy_images(root, count=3, seed=2)
    assert first.shape[0] == 3 and torch.equal(first, second)
    assert not torch.equal(first, other)


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(LoaderError):
        load_generated_proxy_images(tmp_path / "empty", count=3, seed=0)
    with pytest.raises(LoaderError):
        load_generated_proxy_images(_chunks(tmp_path / "chunks"), count=0, seed=0)
