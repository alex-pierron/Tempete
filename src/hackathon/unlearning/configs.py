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
METHOD_NAMES = (METHOD_IDENTITY, METHOD_SCRUB, METHOD_GOLD_SOFT)

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


@dataclass(frozen=True)
class UnlearnConfig:
    """All knobs of `unlearn`. Unused fields are ignored by methods that do not need them.

    Attributes:
        method: identity | scrub | gold_soft.
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
        target_temperature: gold_soft: temperature applied to the teacher logits to build targets.
        target_views: gold_soft: number of strong views averaged into one target.
        target_view_min_area: gold_soft: smallest crop area fraction of a strong view.
        altered_weight: gold_soft: weight of the loss on altered copies relative to exact images.
        scrub_msteps: scrub: number of leading epochs that include a max-step on forget.
        scrub_kl_cap: scrub: cap of the maximised (T^2-scaled) KL.
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

    def __post_init__(self) -> None:
        """Reject values that would make a run meaningless or crash late."""
        if self.method not in METHOD_NAMES:
            raise ConfigError(f"Unknown method '{self.method}'; choose from {list(METHOD_NAMES)}")
        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "retain_count": self.retain_count,
            "target_views": self.target_views,
        }
        for name, value in positive.items():
            if value < 1:
                raise ConfigError(f"{name} must be >= 1, got {value}")
        for name, real in {
            "lr": self.lr,
            "kd_temperature": self.kd_temperature,
            "target_temperature": self.target_temperature,
            "scrub_kl_cap": self.scrub_kl_cap,
        }.items():
            if real <= 0:
                raise ConfigError(f"{name} must be > 0, got {real}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ConfigError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
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
