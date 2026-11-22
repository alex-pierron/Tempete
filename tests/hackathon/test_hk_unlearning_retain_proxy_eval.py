"""Tests of the candidate evaluation on held-out proxy images (tiny ViT, CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hackathon.unlearning.constants import IMAGE_SIZE
from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.retain_proxy.finetune import ProxyData
from hackathon.unlearning.research.retain_proxy.proxy_eval import (
    ProxyEvalError,
    evaluate_candidates,
    holdout_indices,
    load_holdout,
)

IMAGE_COUNT = 20


def _tiny() -> nn.Module:
    return build_model(embed_dim=32, depth=2, num_heads=2)


def _checkpoint(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    torch.save(_tiny().state_dict(), path)
    return path


def _dataset(path: Path) -> Path:
    rng = np.random.default_rng(0)
    np.savez(
        path,
        images=rng.integers(0, 255, (IMAGE_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        labels=np.arange(IMAGE_COUNT, dtype=np.int64) % 7,
    )
    return path


def test_holdout_matches_the_split_used_by_fine_tuning(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "proxy.npz")
    images, labels = load_holdout(dataset, 0.25, seed=3)
    data = ProxyData(dataset, torch.device("cpu"))
    held_images, held_labels = data.hold_out(0.25, seed=3)
    assert len(labels) == 5 and np.array_equal(labels, held_labels.numpy())
    assert torch.equal(images, held_images)
    assert (
        holdout_indices(IMAGE_COUNT, 0.25, 3).tolist()
        == holdout_indices(IMAGE_COUNT, 0.25, 3).tolist()
    )


def test_identical_candidate_has_zero_drift_and_other_has_some(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "proxy.npz")
    original = _checkpoint(tmp_path / "orig.pt", seed=0)
    other = _checkpoint(tmp_path / "other.pt", seed=1)
    stats = evaluate_candidates(
        original,
        {"same": original, "other": other},
        dataset,
        fraction=0.5,
        model_factory=_tiny,
    )
    assert stats["same"].kl_original_to_model_t2 == pytest.approx(0.0, abs=1e-9)
    assert stats["same"].mean_confidence > 0
    assert stats["other"].kl_original_to_model_t2 > 0.0


def test_empty_holdout_and_missing_dataset_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ProxyEvalError):
        holdout_indices(5, 0.1, 0)
    with pytest.raises(ProxyEvalError):
        load_holdout(tmp_path / "missing.npz", 0.1, 0)


def test_confidence_filter_keeps_only_sure_images_and_can_empty_the_set(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "proxy.npz")
    original = _checkpoint(tmp_path / "orig.pt", seed=0)
    everything = evaluate_candidates(
        original, {"same": original}, dataset, fraction=1.0, model_factory=_tiny
    )
    assert everything["same"].mean_confidence > 0
    with pytest.raises(ProxyEvalError):
        evaluate_candidates(
            original,
            {"same": original},
            dataset,
            fraction=1.0,
            model_factory=_tiny,
            min_original_confidence=1.1,
        )
