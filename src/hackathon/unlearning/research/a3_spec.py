"""Variant specification of team A3's class-removed generation (read at JOB START, not at submit).

The Slurm queue of the reservation is ordered by age: a job submitted early starts early, but its
parameters should be decided as LATE as possible (the public scores of the post-hoc probes of
team C arrive while the job waits). The sbatch script therefore reads `jureca/a3_spec.json` when it
starts; this module validates that file so a typo fails in seconds, not after 20 minutes of GPU.

Plain frozen dataclasses with explicit checks, not Pydantic: the official `unlearning/.venv` of the
cluster does not ship it and the rules forbid installing packages in the shared environment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hackathon.common.errors import HackathonError

EPOCH_STEPS = 219
"""Optimiser steps of one pass over the 3500 forget images at batch 16."""
MAX_STEPS = 12 * EPOCH_STEPS
CLASS_COUNT = 100
DEFAULT_REMOVED_CLASSES = (0, 6, 7, 8, 29, 30, 31, 33, 38)
"""The 9 classes holding 129-130 forget images (team C), see `a3_targets`."""
MAX_ALPHA = 1.5
"""Above this the extrapolation destroys retention (measured by team A2: alpha 1.5 flips 55 %)."""
MAX_DELTA = 20.0
MAX_HEAD_SCALE = 3.0
MAX_BATCH = 64
MAX_LEARNING_RATE = 1e-3
MAX_VARIANTS = 200
"""Size of the pool the four workers of a job draw from."""
MAX_SAVE_TOP = 8
MAX_DUAL_WEIGHT = 10.0
MAX_FLATTEN = 10.0
MAX_CAL_WRONG = 0.6
MAX_VIEW_COUNT = 12
MAX_FALLBACK_TEMPERATURE = 10.0
DEFAULT_VIEW_COUNT = 3
DEFAULT_VIEW_AREA = 0.35
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,60}$")
GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


class SpecError(HackathonError):
    """The variant file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class VariantSpec:
    """One training run and the candidate grid derived from it.

    Attributes:
        name: File stem prefix of every candidate of the run.
        note: Free text kept in the result JSON.
        steps: Optimiser steps (`EPOCH_STEPS` = one epoch at batch 16).
        lr: Peak learning rate (warmup then cosine, the research trainer's schedule).
        batch_size: Forget batch (exact images; the altered copies double it).
        retain_batch_size: Anchor batch.
        retain_weight: Weight of the anchor KL towards the original.
        anchor_view_fraction: Share of the anchor batch replaced by strong views of forget images
            (instance-level suppression: the original keeps the class on views, only the exact
            images lose it).
        target_temperature: gold_soft target temperature (student logits shrink by this factor).
        delta_full: Log-probability offset (nats, = logit units) on the fully removed classes.
        delta_partial: Same, graded by the forgotten share of the other forgotten classes.
        removed_flatten: Extra temperature (>= 1) on the targets of the images of the removed
            classes: the gold has no class for them and hesitates between several classes.
        cal_wrong_fraction: When > 0, the targets of ALL images come from the research trainer's
            calibrated law (that share of the most fragile images is made wrong on purpose, the
            others keep a moderate confidence) instead of the plain tempered view average. The
            public scores favour candidates that are LESS accurate on the non-removed images than
            the plain targets produce (0.74 for gold_soft x1.1 against 0.84-0.89).
        cal_right_low: Lowest probability of the true class among the images kept RIGHT by the
            calibrated law (0.55 = the research trainer's default, a diffuse middle; 0.85 makes them
            confident). The gold is bimodal: gold_soft x1.1, the best public family, has 11 % of the
            non-removed images with p(true) < 0.1 and 25 % above 0.9; the A3 students have 4 % and 13-17 %.
        cal_wrong_high: Highest probability of the true class among the images made WRONG.
        cal_wrong_top_share: Share of the non-label mass put on the most supported wrong class (a
            clear wrong top-1 instead of a diffuse one).
        cal_class_aware: In [0, 1]. With cal_wrong_fraction > 0, the images of the PARTIALLY
            forgotten classes are ranked as more fragile in proportion to the share of their class
            that is in the forget set (a gold saw fewer images of it, so it is less accurate there).
        fallback_temperature: When > 0, the targets of the images of the removed classes are the
            ORIGINAL's exact-image logits with the removed classes masked, softmax at this
            temperature, with no view average and no crop (team C: after masking, the trained
            students recover the original's first remaining class in only 31-41 % of the cases).
        sibling_mass: Share of the probability of a removed-class image put on its PRESENT sibling
            classes (team C, the organisers' class groups), spread by the original's logits; the rest
            is the original fallback. Needs fallback_temperature > 0.
        external_anchor: Allow the retention anchor on EXTERNAL images (COCO, DIV2K, Kodak). The official
            rule says the method must work from the original model and the forget set alone, so the
            default is False: the anchor is then made ONLY of strong crops of forget images with the
            original as teacher (`anchor_view_fraction` is forced to 1 and the anchor pool is the
            forget set). True reproduces the phase 1 recipe (a grey zone, kept for comparison only).
        anchor_skip_removed: Draw the forget-derived anchor only from forget images whose ORIGINAL
            prediction is not a removed class. The anchor asks the student to equal the original on
            strong crops; on a removed-class image that is the opposite of the forgetting target (measured:
            anchor weight 1 on all forget images leaves a raw state at 92 % accuracy on the other classes,
            weight 4 forgets nothing). The removed classes are read off the original model and the forget
            set, so the rule 'original model and forget set alone' still holds.
        save_raw: Also save the non-extrapolated state (alpha 1, no bias) as a file.
        holdout: Train WITHOUT the images of subset B (256 of the 3500) and measure the drift on them
            (`b_*` card fields): in-distribution images the original memorised and the student never
            saw, the closest local stand-in for the hidden control set (diagnostic variants).
        no_batch_scale: Ignore the task's batch scale for this variant (diagnostics compared with
            other recipes at their own batch size).
        view_count: Number of strong views averaged into the targets (3 = the cached default; a
            gold is an average of models, more views give a smoother, more ensemble-like target).
        view_area: Minimum crop area of those views (0.35 = the cached default).
        removed_classes: Fully removed classes.
        alphas: Weight extrapolation factors of the candidate grid.
        post_deltas: Extra head-bias offsets applied after the blend (team C's post-hoc lever).
        head_scales: Post-hoc sharpening: head weight and bias multiplied by this (logits x scale), after
            the bias shift. Team C's best public file (0.138) is a head x 1.25 of a gold_soft base.
        save_top: Number of grid points (best proxy first) written as .pt files and exported as
            logits; every grid point gets a proxy card regardless (a file is 344 MB).
        dual_weight: Weight of the keep-the-class-on-strong-crops term (`a3_train`, team C's dual
            targets); 0 trains the single-target loop.
        dual_area: Minimum crop area of those strong crops.
        seed: Training seed.
    """

    name: str
    note: str = ""
    steps: int = EPOCH_STEPS
    lr: float = 3e-5
    batch_size: int = 16
    retain_batch_size: int = 16
    retain_weight: float = 4.0
    anchor_view_fraction: float = 0.0
    target_temperature: float = 2.0
    delta_full: float = 5.5
    delta_partial: float = 0.0
    removed_flatten: float = 1.0
    cal_wrong_fraction: float = 0.0
    cal_class_aware: float = 0.0
    cal_right_low: float = 0.55
    cal_wrong_high: float = 0.12
    cal_wrong_top_share: float = 0.5
    fallback_temperature: float = 0.0
    sibling_mass: float = 0.0
    holdout: bool = False
    external_anchor: bool = False
    anchor_skip_removed: bool = False
    save_raw: bool = False
    no_batch_scale: bool = False
    view_count: int = 3
    view_area: float = 0.35
    removed_classes: tuple[int, ...] = DEFAULT_REMOVED_CLASSES
    alphas: tuple[float, ...] = (1.0, 1.1)
    post_deltas: tuple[float, ...] = (0.0,)
    head_scales: tuple[float, ...] = (1.0,)
    save_top: int = 1
    dual_weight: float = 0.0
    dual_area: float = 0.35
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject unusable values early (every message names the field)."""
        problems: list[str] = []
        if not NAME_PATTERN.match(self.name):
            problems.append("name must match [A-Za-z0-9_.-]{1,60}")
        if not 1 <= self.steps <= MAX_STEPS:
            problems.append(f"steps must lie in [1, {MAX_STEPS}]")
        if not 0.0 < self.lr <= MAX_LEARNING_RATE:
            problems.append(f"lr must lie in (0, {MAX_LEARNING_RATE}]")
        if not 1 <= self.batch_size <= MAX_BATCH:
            problems.append(f"batch_size must lie in [1, {MAX_BATCH}]")
        if not 1 <= self.retain_batch_size <= MAX_BATCH:
            problems.append(f"retain_batch_size must lie in [1, {MAX_BATCH}]")
        if self.retain_weight < 0.0:
            problems.append("retain_weight must be >= 0")
        if not 0.0 <= self.anchor_view_fraction <= 1.0:
            problems.append("anchor_view_fraction must lie in [0, 1]")
        if self.target_temperature <= 0.0:
            problems.append("target_temperature must be > 0")
        if not 0.0 <= self.delta_full <= MAX_DELTA:
            problems.append(f"delta_full must lie in [0, {MAX_DELTA}]")
        if not 0.0 <= self.delta_partial <= MAX_DELTA:
            problems.append(f"delta_partial must lie in [0, {MAX_DELTA}]")
        if any(not 0 <= index < CLASS_COUNT for index in self.removed_classes):
            problems.append(f"removed_classes must lie in [0, {CLASS_COUNT})")
        if not self.alphas or any(not 0.0 < alpha <= MAX_ALPHA for alpha in self.alphas):
            problems.append(f"alphas must be non-empty and lie in (0, {MAX_ALPHA}]")
        if not self.head_scales or any(
            not 1.0 <= scale <= MAX_HEAD_SCALE for scale in self.head_scales
        ):
            problems.append(f"head_scales must be non-empty and lie in [1, {MAX_HEAD_SCALE}]")
        if not self.post_deltas or any(not 0.0 <= delta <= MAX_DELTA for delta in self.post_deltas):
            problems.append(f"post_deltas must be non-empty and lie in [0, {MAX_DELTA}]")
        if not 0.05 <= self.cal_right_low <= 0.99:
            problems.append("cal_right_low must lie in [0.05, 0.99]")
        if not 0.005 <= self.cal_wrong_high <= 0.5:
            problems.append("cal_wrong_high must lie in [0.005, 0.5]")
        if not 0.0 <= self.cal_wrong_top_share <= 1.0:
            problems.append("cal_wrong_top_share must lie in [0, 1]")
        if not 0.0 <= self.fallback_temperature <= MAX_FALLBACK_TEMPERATURE:
            problems.append(f"fallback_temperature must lie in [0, {MAX_FALLBACK_TEMPERATURE}]")
        if not 0.0 <= self.sibling_mass <= 1.0:
            problems.append("sibling_mass must lie in [0, 1]")
        if self.sibling_mass > 0.0 and self.fallback_temperature <= 0.0:
            problems.append("sibling_mass needs fallback_temperature > 0")
        if not 0.0 <= self.cal_class_aware <= 1.0:
            problems.append("cal_class_aware must lie in [0, 1]")
        if self.cal_class_aware > 0.0 and self.cal_wrong_fraction <= 0.0:
            problems.append("cal_class_aware needs cal_wrong_fraction > 0")
        if not 1 <= self.view_count <= MAX_VIEW_COUNT:
            problems.append(f"view_count must lie in [1, {MAX_VIEW_COUNT}]")
        if not 0.1 <= self.view_area <= 0.9:
            problems.append("view_area must lie in [0.1, 0.9]")
        if not 0.0 <= self.cal_wrong_fraction <= MAX_CAL_WRONG:
            problems.append(f"cal_wrong_fraction must lie in [0, {MAX_CAL_WRONG}]")
        if not 1.0 <= self.removed_flatten <= MAX_FLATTEN:
            problems.append(f"removed_flatten must lie in [1, {MAX_FLATTEN}]")
        if not 0.0 <= self.dual_weight <= MAX_DUAL_WEIGHT:
            problems.append(f"dual_weight must lie in [0, {MAX_DUAL_WEIGHT}]")
        if not 0.05 <= self.dual_area <= 1.0:
            problems.append("dual_area must lie in [0.05, 1]")
        if not 0 <= self.save_top <= MAX_SAVE_TOP:
            problems.append(f"save_top must lie in [0, {MAX_SAVE_TOP}]")
        if problems:
            raise SpecError(f"variant '{self.name}': " + "; ".join(problems))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> VariantSpec:
        """Build a variant from parsed JSON (lists become tuples, unknown keys are refused).

        Raises:
            SpecError: unknown key, missing name, or any value rejected by `__post_init__`.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise SpecError(f"unknown variant field(s): {unknown}")
        if "name" not in raw:
            raise SpecError("a variant needs a name")
        values: dict[str, Any] = {
            key: tuple(value) if isinstance(value, list) else value for key, value in raw.items()
        }
        try:
            return cls(**values)
        except TypeError as error:
            raise SpecError(f"variant '{raw['name']}': {error}") from error


@dataclass(frozen=True)
class SpecFile:
    """The JSON file: a generation tag and the pool of variants (taken in order by the workers)."""

    generation: str
    variants: tuple[VariantSpec, ...]

    def __post_init__(self) -> None:
        """Check the tag, the variant count and the uniqueness of the names."""
        if not GENERATION_PATTERN.match(self.generation):
            raise SpecError("generation must match [A-Za-z0-9_.-]{1,40}")
        if not 1 <= len(self.variants) <= MAX_VARIANTS:
            raise SpecError(f"between 1 and {MAX_VARIANTS} variants are needed")
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise SpecError("variant names must be unique")


def load_spec_file(path: Path) -> SpecFile:
    """Read and validate a spec file.

    Raises:
        SpecError: file missing, not JSON, or failing validation (the message says which field).
    """
    if not path.is_file():
        raise SpecError(f"spec file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpecError(f"unreadable spec file {path}: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {"generation", "variants"}:
        raise SpecError("spec file must be an object with exactly 'generation' and 'variants'")
    variants_raw = raw["variants"]
    if not isinstance(variants_raw, list) or not all(isinstance(v, dict) for v in variants_raw):
        raise SpecError("'variants' must be a list of objects")
    return SpecFile(
        generation=str(raw["generation"]),
        variants=tuple(VariantSpec.from_mapping(entry) for entry in variants_raw),
    )
