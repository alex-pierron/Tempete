"""Tests of the team C format exporter (subset, views, npz layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.retain_proxy.export_views import (
    VIEWS_EXPORTED,
    ExportError,
    build_views,
    export_model,
    logits_and_features,
    subset_indices,
)

SUBSET = 5


def _images(count: int) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(
        rng.integers(0, 255, (count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    )


def test_subset_is_sorted_seeded_and_bounded() -> None:
    first, second = subset_indices(20, SUBSET), subset_indices(20, SUBSET)
    assert torch.equal(first, second) and first.tolist() == sorted(first.tolist())
    assert len(set(first.tolist())) == SUBSET and int(first.max()) < 20
    with pytest.raises(ExportError):
        subset_indices(3, SUBSET)


def test_views_have_team_c_names_and_shapes() -> None:
    subset = _images(4)
    views = build_views(subset)
    assert tuple(views) == VIEWS_EXPORTED
    assert all(view.shape == subset.shape and view.dtype == torch.uint8 for view in views.values())
    assert torch.equal(views["exact"], subset)
    assert not torch.equal(views["strong_view"], subset)
    again = build_views(subset)
    assert all(torch.equal(views[name], again[name]) for name in views), "seeds must be fixed"


def test_export_model_layout_and_idempotence(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(embed_dim=32, depth=2, num_heads=2)
    subset = _images(4)
    indices = torch.arange(4)
    logits, features = logits_and_features(model, subset)
    assert logits.shape == (4, NUM_CLASSES) and features.shape == (4, 32)
    path = export_model(model, "d_demo", tmp_path, indices, build_views(subset))
    with np.load(path) as archive:
        assert set(archive.files) == {"subset_indices"} | {
            f"{kind}_{view}" for view in VIEWS_EXPORTED for kind in ("logits", "feats")
        }
        np.testing.assert_allclose(archive["logits_exact"], logits.numpy(), atol=1e-5)
    stamp = path.stat().st_mtime_ns
    assert export_model(model, "d_demo", tmp_path, indices, build_views(subset)) == path
    assert path.stat().st_mtime_ns == stamp, "an existing export is left untouched"
    assert not list(tmp_path.glob("*.partial*"))
