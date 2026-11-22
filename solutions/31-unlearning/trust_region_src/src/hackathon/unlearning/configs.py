"""Hyper-parameters of the unlearning methods, named presets and JSON/override loading.

One flat frozen dataclass serves every method so a config file is a plain JSON object and
`--set key=value` works uniformly. A preset is a PARTIAL dict applied over the defaults:

* `full`  : the real run on the A100 (all 3500 forget images, 600 retain-proxy images).
* `smoke` : a < 2 minute check on the Apple M5 (tiny subsets, 1 epoch, a few steps).

Defaults are deliberately conservative (small learning rate, few epochs): the retain score is
multiplicative, so a method that is too aggressive loses more than it wins.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

from hackathon.common.errors import ConfigError
from hackathon.unlearning.constants import DEFAULT_SEED, SCORING_TEMPERATURE

METHOD_IDENTITY = "identity"
METHOD_SCRUB = "scrub"
METHOD_GOLD_SOFT = "gold_soft"
METHOD_TRUST_REGION = "trust_region"
METHOD_BAND_ASCENT = "band_ascent"
METHOD_NAMES = (
    METHOD_IDENTITY, METHOD_SCRUB, METHOD_GOLD_SOFT, METHOD_TRUST_REGION, METHOD_BAND_ASCENT
)  # fmt: skip
TRUST_TARGET_TEMPERED = "tempered"
TRUST_TARGET_VIEWS = "views"
TRUST_TARGET_CALIBRATED = "calibrated"
TRUST_TARGETS = (TRUST_TARGET_TEMPERED, TRUST_TARGET_VIEWS, TRUST_TARGET_CALIBRATED)
RETAIN_SOURCE_PROXY = "proxy"
RETAIN_SOURCE_FORGET_VIEWS = "forget_views"
RETAIN_SOURCES = (RETAIN_SOURCE_PROXY, RETAIN_SOURCE_FORGET_VIEWS)

DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-5
"""Typical ViT-B fine-tuning rate; larger values drift on retain within a few steps."""
DEFAULT_WARMUP_FRACTION = 0.1
DEFAULT_RETAIN_COUNT = 600
"""All the local proxy images (500 COCO + 100 DIV2K + 24 Kodak = 624)."""
DEFAULT_RETAIN_WEIGHT = 1.0
DEFAULT_TARGET_TEMPERATURE = 2.0
DEFAULT_TARGET_VIEWS = 3
DEFAULT_TARGET_VIEW_MIN_AREA = 0.35
"""Crop area of the target views: measured on the original, ~0.35 gives 87 % argmax agreement."""
DEFAULT_ALTERED_WEIGHT = 1.0
DEFAULT_SCRUB_MSTEPS = 1
DEFAULT_SCRUB_KL_CAP = 2.0
"""Upper bound of the maximised KL (scaled by T^2): stops the max-step before it destroys the model."""
DEFAULT_KL_RADIUS = 0.5
"""Forget budget at T = 1 (about the mean pseudo-label loss of a gold model): a prior, not measured."""
DEFAULT_TRUST_LAMBDA = 0.25
DEFAULT_TRUST_HEDGE = 0.25
"""calibrated target: starting values of the research note (robust over two stand-in families)."""
DEFAULT_RADIUS_CHECK_EVERY = 10
"""trust_region: steps between two radius checks. Measured at full scale (RTX 5080, 21/11): the
KL jumps from 0.07 to 0.65 within 10 steps at the end of the warmup, so radii 0.2 to 0.6 stopped
at the same step; set 1 for a stop that tracks the radius (one probe forward per step)."""
DEFAULT_BAND_RELEASE = 0.25
"""band_ascent: share of images pushed, ~ the 8-23 % the six bench stand-ins get wrong (rapport V5)."""
DEFAULT_BAND_FLOOR = 0.05
"""band_ascent: loss ceiling of the held images, -log(0.95): a 95 %-sure unseen typical image."""


@dataclass(frozen=True)
class UnlearnConfig:
    """All knobs of `unlearn`. Unused fields are ignored by methods that do not need them.

    Attributes:
        method: identity | scrub | gold_soft | trust_region | band_ascent.
        seed: Seed of subsets, augmentations, shuffling.
        device: cuda | mps | cpu, or None for automatic.
        batch_size: Images per forward batch (each of forget / altered / retain per step).
        epochs: Passes over the forget set.
        lr: Peak AdamW learning rate (linear warmup then cosine to zero).
        weight_decay: AdamW weight decay (0: decay would pull weights away from the original).
        warmup_fraction: Fraction of the steps used for the linear warmup.
        max_steps_per_epoch: Cap on steps per epoch (smoke tests); None for a full pass.
        forget_subset: Use only this many forget images (smoke tests); None for all.
        retain_count: Number of retain-proxy images loaded when none are passed in.
        proxy_root: Folder holding coco_val_subset / div2k_valid / kodak (default: data root).
        trainable_blocks: Train only the last N transformer blocks (+ norm + head); -1 trains all.
        grad_checkpointing: Recompute activations to save memory (slower).
        amp: bfloat16 autocast on CUDA during training (weights stay float32).
        kd_temperature: Softmax temperature of the retain/scrub KL (the scoring one, 2).
        retain_weight: Weight of the KL-to-teacher term on retain-proxy images.
        target_temperature: gold_soft, trust_region: temperature applied to the teacher logits to
            build targets (trust_region: the target must sit beyond kl_radius; tempered: ~5-6,
            views: ~4; calibrated: temperature of the hedge, 6).
        target_views: gold_soft, trust_region (views, calibrated): number of strong views
            averaged into one target.
        target_view_min_area: gold_soft, trust_region (views, calibrated): smallest crop area
            fraction of a strong view.
        altered_weight: gold_soft: weight of the loss on altered copies relative to exact images.
        scrub_msteps: scrub: number of leading epochs that include a max-step on forget.
        scrub_kl_cap: scrub: cap of the maximised (T^2-scaled) KL.
        kl_radius: trust_region: training stops once the mean KL(original || student) at T = 1 on
            the training forget images reaches this radius (the forget budget).
        trust_target: trust_region: "tempered" (V1, softmax(z_orig / target_temperature) on the
            exact image), "views" (V2, per image: mean over target_views strong views of
            softmax(z_orig(view) / target_temperature), drawn once with a seed from `seed`) or
            "calibrated" (V3, see `trust_target_logits`: direction at T = 2 from the views mixed
            with softmax(z_orig / target_temperature), true-class logit shifted so that the loss
            at T = 1 is l* = l_mix + trust_lambda * (mean per-view loss - l_mix)).
        trust_lambda: trust_region (calibrated): in [0, 1]; 0 aims the loss of the mixture, 1 the
            mean per-view loss (the per-model statistics the mean/std score compares).
        trust_hedge: trust_region (calibrated): in [0, 1); weight b of the hedge
            softmax(z_orig / target_temperature) in the direction (1 - b) * views + b * hedge.
        radius_fraction: trust_region: None uses kl_radius; else in (0, 1), the radius becomes
            radius_fraction * KL(original || target) (any target), measured on the run's images.
        radius_check_every: trust_region: steps between two checks of the radius (1 = every step).
        radius_dual_lr: trust_region: 0 stops at the first check that reaches the radius (V1-V3);
            > 0 runs every epoch with the radius as an active constraint (dual ascent step of the
            multiplier of KL(original || student), per check, relative to the radius).
        altered_fraction: trust_region: in [0, 1); share of each batch replaced by a flip or a
            70-100 % crop of the same image, trained toward the target of its source image.
        anchor_folds: trust_region (V4): 0 (off, V1-V3 unchanged) or K >= 2. The forget images are
            split into K folds stratified by pseudo-label and forgotten one fold after the other;
            while a fold is pushed to its target (and stops at its own radius), the other folds
            are held at their outputs from the start of that phase: memorised images of the same
            distribution, the only retain proxy the forget set itself offers.
        anchor_weight: trust_region (V4): weight of the anchor term, T^2 KL at kd_temperature
            between those held outputs and the student, one anchor batch per training batch.
        anchor_views: trust_region (V4): in [0, 1]; share of each anchor batch shown as a strong
            view (crop of 35-45 % of the area, then colour) instead of the exact image, so the
            anchor also holds the behaviour on inputs the original has not memorised.
        retain_source: scrub, gold_soft: "proxy" (images outside the forget set: retain_images or
            proxy_root) or "forget_views" (strong views of the forget images themselves, the only
            source the subject allows in a submitted method).
        holdout_fraction: `run` only, any method: hold this share of the forget set out of training
            (stratified by the original's argmax) and evaluate the drift on it instead of the
            retain proxy. A tuning run: its file has NOT forgotten the held-out images.
        band_release: band_ascent (V5): in (0, 1]; share phi of the images pushed, those whose
            mean loss of the original on target_views strong views is highest.
        band_levels: band_ascent (V5): number J of dose levels rho_j = j / J; each level is
            `epochs` passes with a fresh AdamW and schedule.
        band_tolerance: band_ascent (V5): >= 1; ceiling h_i = band_tolerance * t_i + floor of a
            pushed image whose target loss is t_i = rho * (its views' loss).
        band_floor: band_ascent (V5): loss ceiling delta_0 of the images not pushed (and added to
            the pushed ones'); None drops the ceiling of the images not pushed (ablation).
        band_sharpness: band_ascent (V5): stiffness beta of the two softplus hinges.
        cv_folds: band_ascent (V5): 0 runs every level; K >= 2 first runs the levels on K - 1
            folds and keeps the last level whose drift on the untouched fold, times
            (K / (K - 1)) ** cv_scale_exponent, stays within cv_max_drift with an argmax
            agreement >= cv_min_agreement; the final run then climbs those levels on every image.
        cv_max_drift: band_ascent (V5): budget of mean KL(original || student) at T = 2 on the
            untouched fold (the retain divergence); twice it is a hard stop.
        cv_min_agreement: band_ascent (V5): argmax agreement required on the untouched fold.
        cv_scale_exponent: band_ascent (V5): >= 0; exponent of the K / (K - 1) correction from
            the cross-validation sub-run (K - 1 folds) to the final run (all images).
        cv_check_every: band_ascent (V5): steps between two hard-stop checks.
    """

    method: str = METHOD_GOLD_SOFT
    seed: int = DEFAULT_SEED
    device: str | None = None
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LEARNING_RATE
    weight_decay: float = 0.0
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION
    max_steps_per_epoch: int | None = None
    forget_subset: int | None = None
    retain_count: int = DEFAULT_RETAIN_COUNT
    proxy_root: str | None = None
    trainable_blocks: int = -1
    grad_checkpointing: bool = False
    amp: bool = True
    kd_temperature: float = SCORING_TEMPERATURE
    retain_weight: float = DEFAULT_RETAIN_WEIGHT
    target_temperature: float = DEFAULT_TARGET_TEMPERATURE
    target_views: int = DEFAULT_TARGET_VIEWS
    target_view_min_area: float = DEFAULT_TARGET_VIEW_MIN_AREA
    altered_weight: float = DEFAULT_ALTERED_WEIGHT
    scrub_msteps: int = DEFAULT_SCRUB_MSTEPS
    scrub_kl_cap: float = DEFAULT_SCRUB_KL_CAP
    kl_radius: float = DEFAULT_KL_RADIUS
    trust_target: str = TRUST_TARGET_TEMPERED
    trust_lambda: float = DEFAULT_TRUST_LAMBDA
    trust_hedge: float = DEFAULT_TRUST_HEDGE
    radius_fraction: float | None = None
    radius_check_every: int = DEFAULT_RADIUS_CHECK_EVERY
    radius_dual_lr: float = 0.0
    altered_fraction: float = 0.0
    anchor_folds: int = 0
    anchor_weight: float = 1.0
    anchor_views: float = 0.0
    retain_source: str = RETAIN_SOURCE_PROXY
    holdout_fraction: float = 0.0
    band_release: float = DEFAULT_BAND_RELEASE
    band_levels: int = 4
    band_tolerance: float = 2.0
    band_floor: float | None = DEFAULT_BAND_FLOOR
    band_sharpness: float = 10.0
    cv_folds: int = 5
    cv_max_drift: float = 0.1
    cv_min_agreement: float = 0.97
    cv_scale_exponent: float = 2.0
    cv_check_every: int = 5

    def __post_init__(self) -> None:
        """Reject values that would make a run meaningless or crash late."""
        if self.method not in METHOD_NAMES:
            raise ConfigError(f"Unknown method '{self.method}'; choose from {list(METHOD_NAMES)}")
        if self.retain_source not in RETAIN_SOURCES:
            raise ConfigError(
                f"Unknown retain_source '{self.retain_source}'; choose from {list(RETAIN_SOURCES)}"
            )
        if self.radius_dual_lr < 0:
            raise ConfigError(f"radius_dual_lr must be >= 0, got {self.radius_dual_lr}")
        if self.trust_target not in TRUST_TARGETS:
            raise ConfigError(
                f"Unknown trust_target '{self.trust_target}'; choose from {list(TRUST_TARGETS)}"
            )
        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "retain_count": self.retain_count,
            "target_views": self.target_views,
            "radius_check_every": self.radius_check_every,
            "band_levels": self.band_levels,
            "cv_check_every": self.cv_check_every,
        }
        for name, value in positive.items():
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}")
        for name, real in {
            "lr": self.lr,
            "kd_temperature": self.kd_temperature,
            "target_temperature": self.target_temperature,
            "scrub_kl_cap": self.scrub_kl_cap,
            "kl_radius": self.kl_radius,
            "anchor_weight": self.anchor_weight,
            "band_sharpness": self.band_sharpness,
            "cv_max_drift": self.cv_max_drift,
        }.items():
            if real <= 0:
                raise ConfigError(f"{name} must be > 0, got {real}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ConfigError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ConfigError(f"holdout_fraction must be in [0, 1), got {self.holdout_fraction}")
        if not 0.0 <= self.trust_lambda <= 1.0:
            raise ConfigError(f"trust_lambda must be in [0, 1], got {self.trust_lambda}")
        if not 0.0 <= self.trust_hedge < 1.0:
            raise ConfigError(f"trust_hedge must be in [0, 1), got {self.trust_hedge}")
        if self.radius_fraction is not None and not 0.0 < self.radius_fraction < 1.0:
            raise ConfigError(f"radius_fraction must be in (0, 1), got {self.radius_fraction}")
        if not 0.0 <= self.altered_fraction < 1.0:
            raise ConfigError(f"altered_fraction must be in [0, 1), got {self.altered_fraction}")
        if not 0.0 <= self.anchor_views <= 1.0:
            raise ConfigError(f"anchor_views must be in [0, 1], got {self.anchor_views}")
        if self.anchor_folds == 1 or self.anchor_folds < 0:
            raise ConfigError(f"anchor_folds must be 0 (off) or >= 2, got {self.anchor_folds}")
        if not 0.0 < self.band_release <= 1.0:
            raise ConfigError(f"band_release must be in (0, 1], got {self.band_release}")
        if self.band_tolerance < 1.0:
            raise ConfigError(f"band_tolerance must be >= 1, got {self.band_tolerance}")
        if self.band_floor is not None and self.band_floor < 0:
            raise ConfigError(f"band_floor must be >= 0 or None, got {self.band_floor}")
        if self.cv_folds == 1 or self.cv_folds < 0:
            raise ConfigError(f"cv_folds must be 0 (off) or >= 2, got {self.cv_folds}")
        if not 0.0 <= self.cv_min_agreement <= 1.0:
            raise ConfigError(f"cv_min_agreement must be in [0, 1], got {self.cv_min_agreement}")
        if self.cv_scale_exponent < 0:
            raise ConfigError(f"cv_scale_exponent must be >= 0, got {self.cv_scale_exponent}")
        if self.scrub_msteps > self.epochs:
            raise ConfigError(f"scrub_msteps ({self.scrub_msteps}) exceeds epochs ({self.epochs})")


PRESETS: dict[str, dict[str, object]] = {
    "full": {},
    "smoke": {
        "forget_subset": 64,
        "retain_count": 32,
        "batch_size": 8,
        "epochs": 1,
        "max_steps_per_epoch": 4,
        "target_views": 2,
        "scrub_msteps": 1,
        "warmup_fraction": 0.0,
    },
}
"""Partial configs by name; `full` is the dataclass defaults."""


def _field_types() -> dict[str, object]:
    return typing.get_type_hints(UnlearnConfig)


def coerce_value(name: str, raw: object) -> object:
    """Convert a JSON value or a `--set` string to the declared type of field `name`.

    Raises:
        ConfigError: unknown field or a value that does not parse.
    """
    hints = _field_types()
    if name not in hints:
        raise ConfigError(f"Unknown config field '{name}'; known: {sorted(hints)}")
    annotation = str(hints[name])
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.lower() in ("none", "null"):
        return None
    try:
        if "bool" in annotation:
            if text.lower() not in ("true", "false", "1", "0"):
                raise ValueError(text)
            return text.lower() in ("true", "1")
        if "int" in annotation:
            return int(text)
        if "float" in annotation:
            return float(text)
    except ValueError as exc:
        raise ConfigError(f"Cannot parse '{raw}' for field '{name}' ({annotation})") from exc
    return text


def build_config(values: Mapping[str, object]) -> UnlearnConfig:
    """UnlearnConfig from a partial mapping (unknown keys raise ConfigError)."""
    coerced = {name: coerce_value(name, raw) for name, raw in values.items()}
    return UnlearnConfig(**coerced)  # type: ignore[arg-type]


def load_config(spec: str | None, overrides: Mapping[str, object] | None = None) -> UnlearnConfig:
    """Resolve `spec` (preset name, JSON file path, or None for defaults) plus overrides.

    Args:
        spec: "full", "smoke", a path to a JSON object, or None.
        overrides: Field values applied last (typically `--method` and `--set`).

    Raises:
        ConfigError: unknown preset/field, unreadable or non-object JSON.
    """
    base: dict[str, object] = {}
    if spec is not None:
        if spec in PRESETS:
            base = dict(PRESETS[spec])
        else:
            path = Path(spec)
            if not path.is_file():
                raise ConfigError(f"'{spec}' is neither a preset {sorted(PRESETS)} nor a file.")
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(f"Cannot read config {path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path} must contain a JSON object.")
            base = loaded
    base.update(overrides or {})
    return build_config(base)


def config_to_dict(config: UnlearnConfig) -> dict[str, object]:
    """Plain dict of a config, for logs and the run report."""
    return {field.name: getattr(config, field.name) for field in fields(config)}


def with_overrides(config: UnlearnConfig, **changes: object) -> UnlearnConfig:
    """Copy of `config` with some fields replaced (validated again)."""
    return replace(config, **changes)  # type: ignore[arg-type]
