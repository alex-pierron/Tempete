"""Calibrated, learning-free mask rules and the regime switch.

A rule maps the cached features of one image to a [128, 128] mask. Rules have one scalar
parameter fitted on the training folds (`fit`), so every number we report is out-of-fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.evaluate import miou_per_image
from hackathon.wmloc.research.maps import robust_z
from hackathon.wmloc.research.postprocess import (
    MODE_THRESHOLD,
    evidence_to_mask,
    largest_component,
    top_fraction_mask,
)

SOURCE_POSMAP_RELATIVE = "posmap_rel"
SOURCE_POSMAP_Z = "posmap_z"
SOURCE_SCALE_SPACE_Z = "ss_z"
SOURCES: tuple[str, ...] = (SOURCE_POSMAP_RELATIVE, SOURCE_POSMAP_Z, SOURCE_SCALE_SPACE_Z)
RELATIVE_FLOOR = 1e-6
"""Floor of the median used to normalize a margin map (a flat map must not explode)."""
SWITCH_GRID_POINTS = 201
MIN_AREA_FRACTION = 1e-3
MIN_LOG_SPREAD = 1e-6
"""Floor of the area fraction and of the margin before taking logarithms."""


class RuleError(ResearchError):
    """A rule was used before `fit`, or a needed feature is missing."""


@dataclass(frozen=True)
class ImageFeatures:
    """Everything the rules read about one image.

    Attributes:
        name: Image name.
        agreement: Hard bit agreement of the whole-image decode against the known message.
        margin: Soft margin of the whole-image decode.
        posmap: [128, 128] soft-margin map of the (shifted) per-position head, or None.
        scale_space: [128, 128] best-scale z-score evidence, or None.
    """

    name: str
    agreement: float
    margin: float
    posmap: np.ndarray | None
    scale_space: np.ndarray | None


class MaskRule(Protocol):
    """A fitted-then-applied mask rule."""

    name: str

    def fit(self, features: Sequence[ImageFeatures], truth: np.ndarray) -> None:
        """Calibrate the parameter on `features` and their ground-truth masks [N, 128, 128]."""
        ...

    def predict(self, feature: ImageFeatures) -> np.ndarray:
        """uint8 mask [128, 128]."""
        ...


class ConstantRule:
    """The same mask for every image (all ones / all zeros baselines)."""

    def __init__(self, value: int) -> None:
        self.name = f"constant_{value}"
        self._value = value

    def fit(self, features: Sequence[ImageFeatures], truth: np.ndarray) -> None:
        """Nothing to fit."""

    def predict(self, feature: ImageFeatures) -> np.ndarray:
        """The constant mask."""
        return np.full((GRID_SIZE, GRID_SIZE), self._value, dtype=np.uint8)


class ThresholdRule:
    """Threshold of an evidence map, followed by a shape prior; the cut is fitted.

    Args:
        source: Which evidence map (`SOURCES`).
        mode: Post-processing (`postprocess.MODES`).
        candidates: Thresholds tried.
        keep_ones_candidate: Also try "everything on" (cut below the minimum), so the rule can
            fall back to the constant baseline when the map carries no information.
    """

    def __init__(
        self,
        source: str,
        candidates: Sequence[float],
        mode: str = MODE_THRESHOLD,
        keep_ones_candidate: bool = False,
    ) -> None:
        if source not in SOURCES:
            raise RuleError(f"source must be one of {SOURCES}")
        self.name = f"{source}/{mode}"
        self._source = source
        self._mode = mode
        self._candidates = list(candidates)
        self._keep_ones = keep_ones_candidate
        self.cut: float | None = None

    def evidence(self, feature: ImageFeatures) -> np.ndarray:
        """The evidence map of this rule for one image.

        Raises:
            RuleError: the needed feature was not extracted.
        """
        if self._source == SOURCE_SCALE_SPACE_Z:
            if feature.scale_space is None:
                raise RuleError(f"{feature.name}: no scale-space evidence")
            return feature.scale_space
        if feature.posmap is None:
            raise RuleError(f"{feature.name}: no per-position map")
        if self._source == SOURCE_POSMAP_Z:
            return robust_z(feature.posmap)
        median = max(float(np.median(feature.posmap)), RELATIVE_FLOOR)
        return np.asarray(feature.posmap / median, dtype=np.float32)

    def fit(self, features: Sequence[ImageFeatures], truth: np.ndarray) -> None:
        """Pick the cut with the best mean mIoU on the training images."""
        maps = [self.evidence(f) for f in features]
        best_cut, best_score = self._candidates[0], -1.0
        options = list(self._candidates)
        if self._keep_ones:
            options.append(float(min(m.min() for m in maps)) - 1.0)
        for cut in options:
            masks = np.stack([evidence_to_mask(m, cut, self._mode) for m in maps])
            score = float(miou_per_image(masks, truth).mean())
            if score > best_score:
                best_cut, best_score = cut, score
        self.cut = best_cut

    def predict(self, feature: ImageFeatures) -> np.ndarray:
        """Mask at the fitted cut.

        Raises:
            RuleError: `fit` was not called.
        """
        if self.cut is None:
            raise RuleError(f"{self.name}: fit() first")
        return evidence_to_mask(self.evidence(feature), self.cut, self._mode)


class AreaMatchedRule:
    """Keep the `area` fraction of cells with the highest evidence, `area` predicted from the margin.

    On the rectangle regime the whole-image soft margin grows with the watermarked area
    (correlation measured on train), so a log-log line fitted on the training images predicts
    the area; the evidence only has to RANK the cells, which is easier than calibrating a cut.

    Args:
        source: Evidence map used for the ranking (`SOURCES`).
        keep_largest: Keep only the largest connected component of the selected cells.
    """

    def __init__(self, source: str, keep_largest: bool = False) -> None:
        self.name = f"{source}/area_matched" + ("_largest" if keep_largest else "")
        self._ranker = ThresholdRule(source, [0.0])
        self._keep_largest = keep_largest
        self._line: tuple[float, float] | None = None

    def fit(self, features: Sequence[ImageFeatures], truth: np.ndarray) -> None:
        """Fit log(area fraction) = slope * log(margin) + intercept on the training images."""
        areas = np.maximum(truth.reshape(len(truth), -1).mean(axis=1), MIN_AREA_FRACTION)
        margins = np.maximum([f.margin for f in features], MIN_AREA_FRACTION)
        log_margins = np.log(margins)
        if float(np.ptp(log_margins)) < MIN_LOG_SPREAD:
            self._line = (0.0, float(np.log(areas).mean()))  # no margin spread: constant area
            return
        slope, intercept = np.polyfit(log_margins, np.log(areas), 1)
        self._line = (float(slope), float(intercept))

    def predict(self, feature: ImageFeatures) -> np.ndarray:
        """Mask of the top-evidence cells covering the predicted area.

        Raises:
            RuleError: `fit` was not called.
        """
        if self._line is None:
            raise RuleError(f"{self.name}: fit() first")
        slope, intercept = self._line
        margin = max(feature.margin, MIN_AREA_FRACTION)
        fraction = float(np.clip(np.exp(slope * np.log(margin) + intercept), 0.0, 1.0))
        mask = top_fraction_mask(self._ranker.evidence(feature), fraction)
        return largest_component(mask) if self._keep_largest else mask


class SwitchRule:
    """Regime classifier on the whole-image bit agreement: full if agreement >= the cut.

    The cut maximizes the training accuracy (ties resolved at the middle of the plateau).
    """

    def __init__(self) -> None:
        self.cut: float | None = None

    def fit(self, agreements: np.ndarray, is_full: np.ndarray) -> None:
        """Fit the cut on agreements [N] and boolean regime labels [N]."""
        cuts = np.linspace(0.45, 1.0, SWITCH_GRID_POINTS)
        accuracy = np.array([float(((agreements >= c) == is_full).mean()) for c in cuts])
        plateau = np.flatnonzero(accuracy >= accuracy.max() - 1e-12)
        self.cut = float(cuts[plateau[len(plateau) // 2]])

    def is_full(self, agreement: float) -> bool:
        """Predicted regime of one image.

        Raises:
            RuleError: `fit` was not called.
        """
        if self.cut is None:
            raise RuleError("switch: fit() first")
        return agreement >= self.cut
