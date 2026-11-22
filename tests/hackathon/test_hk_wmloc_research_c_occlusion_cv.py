"""Team C occlusion detector: ridge fit, hole evidence and the quantile mask rule."""

from __future__ import annotations

import numpy as np
import torch

from hackathon.wmloc.research.c_occlusion_cv import (
    TEXTURE_SIGMAS,
    design_matrix,
    fit_ridge,
    hole_evidence,
    mask_from_evidence,
    texture_features,
)

GRID = 128


def test_fit_ridge_recovers_a_linear_relation_in_the_original_feature_units() -> None:
    rng = np.random.default_rng(0)
    features = rng.uniform(0.5, 2.0, size=(4000, 3))
    design = np.concatenate([features, np.ones((4000, 1))], axis=1)
    target = design @ np.array([2.0, -1.0, 0.5, 3.0]) + rng.normal(0, 0.01, size=4000)
    coefficients = fit_ridge(design, target, penalty=1e-9)
    assert np.allclose(coefficients, [2.0, -1.0, 0.5, 3.0], atol=0.05)


def test_mask_from_evidence_sets_the_requested_share_to_zero() -> None:
    evidence = np.arange(GRID * GRID, dtype=np.float32).reshape(GRID, GRID)
    mask = mask_from_evidence(evidence, 0.1)
    assert abs(float((mask == 0).mean()) - 0.1) < 0.01
    assert mask[-1, -1] == 0 and mask[0, 0] == 1  # highest evidence -> hole
    assert mask_from_evidence(evidence, 0.0).min() == 1


def test_hole_evidence_without_a_model_is_high_where_the_drop_is_low() -> None:
    drop = np.full((GRID, GRID), 0.5, dtype=np.float32)
    drop[:, :32] = 0.0  # a region that loses nothing when blurred
    texture = np.zeros((len(TEXTURE_SIGMAS) + 1, GRID, GRID), dtype=np.float32)
    evidence = hole_evidence(drop, texture, None)
    assert evidence[:, 4:28].mean() > evidence[:, 60:].mean() + 1.0


def test_texture_correction_removes_the_drop_explained_by_texture() -> None:
    rng = np.random.default_rng(1)
    texture = rng.uniform(0.0, 1.0, size=(len(TEXTURE_SIGMAS) + 1, GRID, GRID)).astype(np.float32)
    drop = (0.2 + texture[0]).astype(np.float32)  # drop explained by texture alone
    design = design_matrix(texture)
    coefficients = fit_ridge(design, drop.reshape(-1), penalty=1e-9)
    corrected = hole_evidence(drop, texture, coefficients)
    raw = hole_evidence(drop, texture, None)
    assert float(np.abs(corrected).mean()) < float(np.abs(raw).mean())


def test_texture_features_shape_and_flat_image() -> None:
    flat = torch.full((3, 512, 512), 0.4)
    features = texture_features(flat)
    assert features.shape == (len(TEXTURE_SIGMAS) + 1, GRID, GRID)
    assert float(np.abs(features).max()) < 1e-5
