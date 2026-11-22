"""k-fold evaluation of the learning-free rules, per method and per regime.

Folds are stratified by regime, so each held-out fold has both kinds of images. A rule is fitted
on the training folds (one scalar), applied to the held-out fold; the regime switch is fitted the
same way. Three scores are reported for a (full rule, rectangle rule) pair:

* `oracle`: each image gets the rule of its TRUE regime (what the rules could reach with a
  perfect switch);
* `switched`: the regime is predicted by the whole-image agreement (the deployable score).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hackathon.wmloc.research.data import Record, load_records, load_truth
from hackathon.wmloc.research.evaluate import (
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    miou_per_image,
    roc_auc,
    stratified_folds,
)
from hackathon.wmloc.research.extract import PASS_GLOBAL, PASS_POSMAP, cache_path
from hackathon.wmloc.research.maps import DEFAULT_SIDES, evidence_grid, pass_name_for_side
from hackathon.wmloc.research.rules import ImageFeatures, MaskRule, RuleError, SwitchRule
from hackathon.wmloc.research.scoring import agreement, global_logits, margin, posmap_score_grid

RuleFactory = Callable[[], MaskRule]


@dataclass
class MethodData:
    """Cached train data of one method.

    Attributes:
        method: Method name.
        records: Images that have every needed cache file.
        truth: Ground-truth masks [N, 128, 128].
        is_full: Regime flag of each image.
        features: Cached features of each image.
        folds: Fold id of each image.
    """

    method: str
    records: list[Record]
    truth: np.ndarray
    is_full: np.ndarray
    features: list[ImageFeatures]
    folds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def agreements(self) -> np.ndarray:
        """Whole-image bit agreement of every image."""
        return np.array([f.agreement for f in self.features])


def load_features(
    split: str, record: Record, sides: Sequence[int], use_scale_space: bool
) -> ImageFeatures:
    """Cached features of one image (None for the maps whose cache is absent)."""
    logits = global_logits(split, record.method, record.name)
    has_posmap = cache_path(split, record.method, record.name, PASS_POSMAP).exists()
    has_scales = use_scale_space and all(
        cache_path(split, record.method, record.name, pass_name_for_side(s)).exists() for s in sides
    )
    return ImageFeatures(
        name=record.name,
        agreement=float(agreement(logits, record.message)),
        margin=float(margin(logits, record.message)),
        posmap=posmap_score_grid(split, record.method, record.name, record.message)
        if has_posmap
        else None,
        scale_space=evidence_grid(split, record.method, record.name, record.message, tuple(sides))
        if has_scales
        else None,
    )


def load_method_data(
    method: str,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    sides: Sequence[int] = DEFAULT_SIDES,
) -> MethodData:
    """Train images of `method` that have a global decode cached, with their features."""
    records = [
        r
        for r in load_records("train")
        if r.method == method and cache_path("train", method, r.name, PASS_GLOBAL).exists()
    ]
    truth = load_truth([r.name for r in records])
    is_full = np.array([bool(r.full_watermark) for r in records])
    features = [load_features("train", r, sides, use_scale_space=True) for r in records]
    fold_ids = stratified_folds([f"{method}/{flag}" for flag in is_full], folds, seed)
    return MethodData(method, records, truth, is_full, features, fold_ids)


@dataclass(frozen=True)
class RuleResult:
    """Out-of-fold result of one rule on one regime.

    Attributes:
        name: Rule name.
        per_image: Out-of-fold mIoU of each image of the regime.
        fold_means: Mean per held-out fold.
        masks: Out-of-fold predicted masks [n, 128, 128].
        indices: Positions (in `MethodData`) of the images of the regime.
    """

    name: str
    per_image: np.ndarray
    fold_means: np.ndarray
    masks: np.ndarray
    indices: np.ndarray

    @property
    def mean(self) -> float:
        """Mean out-of-fold mIoU."""
        return float(self.per_image.mean())

    @property
    def std(self) -> float:
        """Standard deviation of the held-out fold means."""
        return float(self.fold_means.std(ddof=1)) if len(self.fold_means) > 1 else 0.0


def evaluate_rule(data: MethodData, factory: RuleFactory, regime_full: bool) -> RuleResult:
    """Fit on the training folds, predict the held-out fold, for the images of one regime."""
    regime = np.flatnonzero(data.is_full == regime_full)
    masks = np.zeros((len(regime), *data.truth.shape[1:]), dtype=np.uint8)
    fold_means: list[float] = []
    name = factory().name
    for fold in sorted(set(data.folds.tolist())):
        train = [i for i in regime if data.folds[i] != fold]
        held = [k for k, i in enumerate(regime) if data.folds[i] == fold]
        rule = factory()
        rule.fit([data.features[i] for i in train], data.truth[train])
        for k in held:
            masks[k] = rule.predict(data.features[regime[k]])
        fold_means.append(float(miou_per_image(masks[held], data.truth[regime[held]]).mean()))
    per_image = miou_per_image(masks, data.truth[regime])
    return RuleResult(name, per_image, np.array(fold_means), masks, regime)


@dataclass(frozen=True)
class SwitchResult:
    """Out-of-fold behaviour of the regime switch.

    Attributes:
        accuracy: Out-of-fold accuracy.
        auc: ROC AUC of the agreement for the "full" class.
        predicted_full: Out-of-fold predicted regime of every image.
        cuts: Cut fitted on each training fold.
    """

    accuracy: float
    auc: float
    predicted_full: np.ndarray
    cuts: list[float]


def evaluate_switch(data: MethodData) -> SwitchResult:
    """Fit the agreement cut per training fold, predict the held-out fold."""
    predicted = np.zeros(len(data.records), dtype=bool)
    cuts: list[float] = []
    agreements = data.agreements
    for fold in sorted(set(data.folds.tolist())):
        train = data.folds != fold
        switch = SwitchRule()
        switch.fit(agreements[train], data.is_full[train])
        assert switch.cut is not None
        cuts.append(switch.cut)
        for i in np.flatnonzero(~train):
            predicted[i] = switch.is_full(float(agreements[i]))
    return SwitchResult(
        float((predicted == data.is_full).mean()),
        roc_auc(agreements, data.is_full),
        predicted,
        cuts,
    )


def combine(
    data: MethodData, switch: SwitchResult, full_all: RuleResult, rect_all: RuleResult
) -> tuple[float, np.ndarray]:
    """Deployable mIoU of a (full rule, rectangle rule) pair behind the switch.

    Args:
        data: The method data.
        switch: Out-of-fold predicted regimes.
        full_all: Full-regime rule from `evaluate_everywhere` (a mask for EVERY image).
        rect_all: Rectangle-regime rule from `evaluate_everywhere`.

    Returns:
        (mean mIoU, per-image mIoU): each image scored with the mask of its predicted regime,
        so a routing error costs what it costs.
    """
    masks = np.where(switch.predicted_full[:, None, None], full_all.masks, rect_all.masks).astype(
        np.uint8
    )
    per_image = miou_per_image(masks, data.truth)
    return float(per_image.mean()), per_image


def evaluate_everywhere(data: MethodData, factory: RuleFactory, regime_full: bool) -> RuleResult:
    """Like `evaluate_rule` but trained on one regime and predicting EVERY held-out image.

    A mis-routed image (wrong regime predicted) still receives a mask from the rule of the
    predicted regime, so the deployable score includes the cost of routing errors.
    """
    everyone = np.arange(len(data.records))
    masks = np.zeros_like(data.truth)
    fold_means: list[float] = []
    name = factory().name
    for fold in sorted(set(data.folds.tolist())):
        train = [i for i in everyone if data.folds[i] != fold and data.is_full[i] == regime_full]
        held = [i for i in everyone if data.folds[i] == fold]
        rule = factory()
        rule.fit([data.features[i] for i in train], data.truth[train])
        for i in held:
            try:
                masks[i] = rule.predict(data.features[i])
            except RuleError:
                # A feature extracted only for the images the switch can route here is missing:
                # the image keeps an all-zeros mask and never reaches `combine`.
                continue
        fold_means.append(float(miou_per_image(masks[held], data.truth[held]).mean()))
    return RuleResult(
        name, miou_per_image(masks, data.truth), np.array(fold_means), masks, everyone
    )
