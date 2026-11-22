"""Cross-validated hole detector from occlusion maps (team C, full regime).

The margin drop of an occluded window mixes two things: how much watermark the window carried
(what we want) and how much blurring changes the image content (texture, which the watermark
does not explain). The raw drop therefore tracks texture (measured: AUC 0.63 against 0.54 for
texture alone on 8 images). This module removes the texture part with a ridge regression fitted
on the NON-hole cells of the training images, and keeps the residual as hole evidence: a cell
that loses LESS margin than its texture predicts has little watermark to lose.

Prediction rule (one parameter, chosen on the training folds): the share `q` of cells with the
highest hole evidence is set to 0 (hole), the rest to 1. `q = 0` is the "all ones" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional

from hackathon.wmloc.research.c_occlusion import OcclusionMap, gaussian_blur, to_grid
from hackathon.wmloc.research.c_results import ResultStoreError, load_arrays
from hackathon.wmloc.research.data import load_image, load_truth
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import miou_per_image, roc_auc, stratified_folds

TEXTURE_SIGMAS: tuple[float, ...] = (2.0, 6.0, 10.0)
"""Blur radii (px) whose residual energy describes the texture a blur destroys."""
TEXTURE_WINDOW_CELLS = 8
"""Texture energy is averaged over 8 cells = 32 px, the occlusion window."""
SMOOTH_CELLS = 5
"""Smoothing of the evidence map before ranking cells, in cells."""
RIDGE_PENALTY = 1e-3
HOLE_SHARE_CANDIDATES: tuple[float, ...] = tuple(float(q) for q in np.arange(0.0, 0.31, 0.02))
CV_FOLDS = 5
CV_SEED = 2026
"""Same fold seed as the other team experiments (`evaluate.CV_SEED` convention)."""
MIN_BASE_MARGIN = 1e-3
GRID_SIZE_FACTOR = 4
"""Pixels per grid cell (512 / 128)."""


class OcclusionCvError(ResearchError):
    """Missing or inconsistent occlusion files."""


@dataclass(frozen=True)
class CellData:
    """Everything the detector needs for N full-regime images on the 128 grid.

    Attributes:
        names: Image names.
        drops: Relative margin drop per sigma, [N, n_sigma, 128, 128].
        texture: Texture features, [N, F, 128, 128].
        truth: Ground-truth masks (1 = watermarked, 0 = hole), [N, 128, 128].
        sigmas: Blur radii of the occlusion maps.
    """

    names: list[str]
    drops: np.ndarray
    texture: np.ndarray
    truth: np.ndarray
    sigmas: tuple[float, ...]


def texture_features(image: torch.Tensor) -> np.ndarray:
    """Texture descriptors [F, 128, 128] of a [3, 512, 512] image (no watermark knowledge).

    One channel per `TEXTURE_SIGMAS` (mean absolute blur residual) plus the gradient magnitude,
    each pooled to the 128 grid and averaged over the occlusion window.
    """
    channels = []
    for sigma in TEXTURE_SIGMAS:
        residual = (image - gaussian_blur(image, sigma)).abs().mean(dim=0)
        channels.append(residual)
    gray = image.mean(dim=0, keepdim=True)[None]
    grad_x = functional.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    grad_y = functional.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    channels.append((grad_x**2 + grad_y**2).sqrt()[0, 0])
    stacked = torch.stack(channels)[None]
    pooled = functional.avg_pool2d(stacked, GRID_SIZE_FACTOR)[0].numpy()
    return np.stack(
        [ndimage.uniform_filter(c, size=TEXTURE_WINDOW_CELLS, mode="nearest") for c in pooled]
    ).astype(np.float32)


def design_matrix(texture: np.ndarray) -> np.ndarray:
    """Regression design [cells, 2F + 1] of texture features: raw, log and an intercept."""
    flat = texture.reshape(texture.shape[0], -1).T
    return np.concatenate([flat, np.log(flat + 1e-4), np.ones((len(flat), 1))], axis=1)


def fit_ridge(design: np.ndarray, target: np.ndarray, penalty: float = RIDGE_PENALTY) -> np.ndarray:
    """Ridge coefficients of `target ~ design` (features standardized inside the fit).

    Returns:
        Coefficients [n_features] for the ORIGINAL design (standardization folded back in).
    """
    mean = design[:, :-1].mean(axis=0)
    scale = design[:, :-1].std(axis=0) + 1e-9
    z = np.concatenate([(design[:, :-1] - mean) / scale, design[:, -1:]], axis=1)
    gram = z.T @ z + penalty * len(z) * np.eye(z.shape[1])
    beta = np.linalg.solve(gram, z.T @ target)
    folded = np.empty_like(beta)
    folded[:-1] = beta[:-1] / scale
    folded[-1] = beta[-1] - float((beta[:-1] / scale) @ mean)
    return folded


def hole_evidence(
    drop: np.ndarray, texture: np.ndarray, coefficients: np.ndarray | None
) -> np.ndarray:
    """Hole evidence map [128, 128]: how much LESS margin a cell loses than its texture predicts.

    With `coefficients=None` the raw drop is used (higher evidence = smaller drop).
    """
    if coefficients is None:
        residual = -drop
    else:
        prediction = (design_matrix(texture) @ coefficients).reshape(drop.shape)
        residual = prediction - drop
    spread = float(residual.std()) + 1e-9
    smoothed = ndimage.uniform_filter(residual / spread, size=SMOOTH_CELLS, mode="nearest")
    return np.asarray(smoothed, dtype=np.float32)


def mask_from_evidence(evidence: np.ndarray, hole_share: float) -> np.ndarray:
    """1 everywhere except the `hole_share` of cells with the highest hole evidence."""
    mask = np.ones(evidence.shape, dtype=np.uint8)
    if hole_share <= 0.0:
        return mask
    threshold = np.quantile(evidence, 1.0 - hole_share)
    mask[evidence >= threshold] = 0
    return mask


def load_cell_data(occlusion_dir: Path, sigmas: tuple[float, ...]) -> CellData:
    """Read every NPZ of `occlusion_dir` (train, full regime) with its image and truth.

    Raises:
        OcclusionCvError: no file, or a sigma missing from a file.
    """
    try:
        stored = load_arrays(occlusion_dir)
    except ResultStoreError as error:
        raise OcclusionCvError(str(error)) from error
    names = sorted(stored)
    drops, textures = [], []
    for name in names:
        archive = stored[name]
        base = max(float(archive["base_margin"]), MIN_BASE_MARGIN)
        per_sigma = []
        for sigma in sigmas:
            key = f"drop_s{sigma:g}"
            if key not in archive:
                raise OcclusionCvError(f"{name} has no {key}")
            window = OcclusionMap(base, archive[key], archive["origins"], int(archive["block"]))
            per_sigma.append(to_grid(window) / base)
        drops.append(np.stack(per_sigma))
        textures.append(texture_features(load_image("train", name)))
    return CellData(
        names=names,
        drops=np.stack(drops).astype(np.float32),
        texture=np.stack(textures),
        truth=load_truth(names),
        sigmas=sigmas,
    )


def cell_auc(data: CellData, sigma_index: int, coefficients: np.ndarray | None) -> float:
    """Mean per-image AUC (hole cells vs the rest) of the hole evidence."""
    scores = []
    for i in range(len(data.names)):
        labels = (data.truth[i] == 0).reshape(-1)
        if not 0 < labels.sum() < labels.size:
            continue
        evidence = hole_evidence(data.drops[i, sigma_index], data.texture[i], coefficients)
        scores.append(roc_auc(evidence.reshape(-1), labels))
    return float(np.mean(scores))


@dataclass(frozen=True)
class DetectorScores:
    """Out-of-fold mIoU of the detectors on the same folds.

    Attributes:
        ones: The all-ones constant (the baseline of the full regime).
        raw_drop: Rule on the raw margin drop.
        texture_corrected: Rule on the drop minus its texture prediction.
        fold_std_raw: Standard deviation of the held-out fold means, raw rule.
        fold_std_corrected: Same for the corrected rule.
        hole_share_corrected: Hole share chosen on each training fold, corrected rule.
    """

    ones: float
    raw_drop: float
    texture_corrected: float
    fold_std_raw: float
    fold_std_corrected: float
    hole_share_corrected: list[float]


def predict_masks(
    data: CellData,
    sigma_index: int,
    indices: np.ndarray,
    hole_share: float,
    coefficients: np.ndarray | None,
) -> np.ndarray:
    """Predicted masks [len(indices), 128, 128] of the selected images."""
    return np.stack(
        [
            mask_from_evidence(
                hole_evidence(data.drops[i, sigma_index], data.texture[i], coefficients),
                hole_share,
            )
            for i in indices
        ]
    )


def _cross_validate(
    data: CellData, sigma_index: int, use_texture: bool
) -> tuple[float, float, list[float]]:
    """Mean out-of-fold mIoU, fold std and the hole share chosen on each training fold."""
    fold_ids = stratified_folds(["full"] * len(data.names), CV_FOLDS, CV_SEED)
    per_image = np.zeros(len(data.names))
    fold_means: list[float] = []
    shares: list[float] = []
    for fold in sorted(set(fold_ids.tolist())):
        held = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        coefficients = fit_texture_model(data, sigma_index, train) if use_texture else None
        scores = {
            q: float(
                miou_per_image(
                    predict_masks(data, sigma_index, train, q, coefficients), data.truth[train]
                ).mean()
            )
            for q in HOLE_SHARE_CANDIDATES
        }
        best = max(scores, key=lambda key: scores[key])
        per_image[held] = miou_per_image(
            predict_masks(data, sigma_index, held, best, coefficients), data.truth[held]
        )
        fold_means.append(float(per_image[held].mean()))
        shares.append(best)
    return float(per_image.mean()), float(np.std(fold_means, ddof=1)), shares


def cross_validate_detectors(data: CellData, sigma_index: int) -> DetectorScores:
    """Out-of-fold mIoU of `all ones`, the raw-drop rule and the texture-corrected rule."""
    baseline = float(miou_per_image(np.ones_like(data.truth), data.truth).mean())
    raw_mean, raw_std, _ = _cross_validate(data, sigma_index, use_texture=False)
    corrected_mean, corrected_std, shares = _cross_validate(data, sigma_index, use_texture=True)
    return DetectorScores(baseline, raw_mean, corrected_mean, raw_std, corrected_std, shares)


def fit_texture_model(data: CellData, sigma_index: int, train_ids: np.ndarray) -> np.ndarray:
    """Ridge model of the relative drop from texture, fitted on non-hole cells of `train_ids`."""
    designs, targets = [], []
    for i in train_ids:
        keep = (data.truth[i] == 1).reshape(-1)
        designs.append(design_matrix(data.texture[i])[keep])
        targets.append(data.drops[i, sigma_index].reshape(-1)[keep])
    return fit_ridge(np.concatenate(designs), np.concatenate(targets))
