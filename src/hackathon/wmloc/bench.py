"""Run localizers over a manifest: predictions for a submission, or scores for the bench.

Score maps are computed once per localizer and image (the expensive part: model forward
passes); thresholds, Otsu and cross-validation are then cheap tensor operations on them.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from hackathon.wmloc.calibrate import (
    CV_FOLDS,
    DEFAULT_THRESHOLD_GRID,
    bootstrap_mean_ci,
    cross_validated_per_image,
    per_image_at_threshold,
    sweep_thresholds,
)
from hackathon.wmloc.constants import GRID_SIZE
from hackathon.wmloc.errors import LocalizerError
from hackathon.wmloc.localizers import (
    ConstantLocalizer,
    GlobalSwitchLocalizer,
    Localizer,
    PixelMapLocalizer,
    RectRefineLocalizer,
    ScoreLocalizer,
    SlidingWindowLocalizer,
    WatermarkBackend,
    apply_threshold,
)
from hackathon.wmloc.manifest import (
    ManifestSample,
    parse_message_bits,
    read_image_tensor,
    read_truth_mask128,
)
from hackathon.wmloc.metric import per_image_miou

ALL_GROUP = "all"
PROGRESS_EVERY = 10
MIN_IMAGES_FOR_CV = 2 * CV_FOLDS
LocalizerFactory = Callable[[WatermarkBackend], Localizer]

LOCALIZER_FACTORIES: dict[str, LocalizerFactory] = {
    "all_ones": lambda backend: ConstantLocalizer(1),
    "global_switch": lambda backend: GlobalSwitchLocalizer(backend),
    "all_zeros": lambda backend: ConstantLocalizer(0),
    "pixel_map_s1": lambda backend: PixelMapLocalizer(backend, 1, name="pixel_map_s1"),
    "pixel_map_s2": lambda backend: PixelMapLocalizer(backend, 2, name="pixel_map_s2"),
    "pixel_map_s4": lambda backend: PixelMapLocalizer(backend, 4, name="pixel_map_s4"),
    "sw_mean_128_192_256": lambda backend: SlidingWindowLocalizer(
        backend, (128, 192, 256), aggregation="mean", name="sw_mean_128_192_256"
    ),
    "sw_max_96_128": lambda backend: SlidingWindowLocalizer(
        backend, (96, 128), name="sw_max_96_128"
    ),
    "sw_max_128_192": lambda backend: SlidingWindowLocalizer(
        backend, (128, 192), name="sw_max_128_192"
    ),
    "rect_refine_96_128": lambda backend: RectRefineLocalizer(
        SlidingWindowLocalizer(backend, (96, 128), name="sw_inner"), name="rect_refine_96_128"
    ),
    "rect_refine_128_192": lambda backend: RectRefineLocalizer(
        SlidingWindowLocalizer(backend, (128, 192), name="sw_inner"), name="rect_refine_128_192"
    ),
    "sw_max_64_96_128": lambda backend: SlidingWindowLocalizer(
        backend, (64, 96, 128), name="sw_max_64_96_128"
    ),
}
"""Named localizers selectable from the CLI (`--localizers`)."""


@dataclass(frozen=True)
class GroupResult:
    """Scores of one localizer on one group of images (an edit type, or all)."""

    n_images: int
    miou_default: float
    miou_otsu: float | None
    miou_calibrated: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass
class LocalizerReport:
    """Bench result of one localizer."""

    name: str
    seconds_per_image: float
    best_threshold: float | None
    groups: dict[str, GroupResult] = field(default_factory=dict)


def load_inputs(sample: ManifestSample, root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(image [3, 512, 512], message bits [n_bits]) of a manifest sample."""
    return read_image_tensor(root / sample.image), parse_message_bits(sample.message)


def compute_outputs(
    localizer: Localizer,
    samples: list[ManifestSample],
    root: Path,
    *,
    scores: bool,
) -> torch.Tensor:
    """Stack per-image outputs: score maps (float) if `scores`, else binary masks (uint8).

    Raises:
        LocalizerError: `scores=True` on a localizer that has no score map.
    """
    if scores and not isinstance(localizer, ScoreLocalizer):
        raise LocalizerError(f"{localizer.name} has no score map")
    outputs: list[torch.Tensor] = []
    started = time.monotonic()
    for index, sample in enumerate(samples):
        image, bits = load_inputs(sample, root)
        if scores and isinstance(localizer, ScoreLocalizer):
            outputs.append(localizer.score_map(image, sample.method, bits))
        else:
            outputs.append(localizer.predict(image, sample.method, bits))
        if (index + 1) % PROGRESS_EVERY == 0:
            rate = (time.monotonic() - started) / (index + 1)
            print(
                f"  [{localizer.name}] {index + 1}/{len(samples)} ({rate:.2f} s/img)",
                file=sys.stderr,
            )
    if not outputs:
        return torch.empty(0, GRID_SIZE, GRID_SIZE)
    return torch.stack(outputs)


def load_truth(samples: list[ManifestSample], root: Path) -> torch.Tensor:
    """Ground-truth masks [N, 128, 128] from the manifest's 512 masks.

    Raises:
        LocalizerError: a sample has no ground-truth mask (test split).
    """
    masks: list[torch.Tensor] = []
    for sample in samples:
        if sample.mask512 is None:
            raise LocalizerError(f"{sample.name}: no ground-truth mask in the manifest")
        masks.append(read_truth_mask128(root / sample.mask512))
    return torch.stack(masks)


def group_indices(samples: list[ManifestSample]) -> dict[str, list[int]]:
    """Image indices per group: `all`, then one group per edit type (when labelled)."""
    groups: dict[str, list[int]] = {ALL_GROUP: list(range(len(samples)))}
    by_edit: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        label = sample.group or sample.edit
        if label:
            by_edit[label].append(index)
    groups.update(dict(sorted(by_edit.items())))
    return groups


def evaluate_score_localizer(
    name: str,
    default_threshold: float | None,
    score_maps: torch.Tensor,
    truth: torch.Tensor,
    samples: list[ManifestSample],
    seconds_per_image: float,
) -> LocalizerReport:
    """Report a score-map localizer: fixed cut, per-image Otsu, and cross-validated cut."""
    report = LocalizerReport(name, seconds_per_image, None)
    can_calibrate = score_maps.shape[0] >= MIN_IMAGES_FOR_CV
    calibrated = (
        cross_validated_per_image(score_maps, truth, DEFAULT_THRESHOLD_GRID)
        if can_calibrate
        else None
    )
    if can_calibrate:
        report.best_threshold = sweep_thresholds(score_maps, truth).best_threshold
    fixed = (
        per_image_at_threshold(score_maps, truth, default_threshold)
        if default_threshold is not None
        else per_image_miou(torch.stack([apply_threshold(m, None) for m in score_maps]), truth)
    )
    otsu = per_image_miou(torch.stack([apply_threshold(m, None) for m in score_maps]), truth)
    for group, indices in group_indices(samples).items():
        pick = torch.tensor(indices, dtype=torch.long)
        if calibrated is not None:
            mean, low, high = bootstrap_mean_ci(calibrated[pick])
            cal: float | None = mean
            ci_low: float | None = low
            ci_high: float | None = high
        else:
            cal = ci_low = ci_high = None
        report.groups[group] = GroupResult(
            n_images=len(indices),
            miou_default=float(fixed[pick].mean().item()),
            miou_otsu=float(otsu[pick].mean().item()),
            miou_calibrated=cal,
            ci_low=ci_low,
            ci_high=ci_high,
        )
    return report


def evaluate_mask_localizer(
    name: str,
    masks: torch.Tensor,
    truth: torch.Tensor,
    samples: list[ManifestSample],
    seconds_per_image: float,
) -> LocalizerReport:
    """Report a localizer that only outputs masks (constants, routing)."""
    per_image = per_image_miou(masks, truth)
    report = LocalizerReport(name, seconds_per_image, None)
    for group, indices in group_indices(samples).items():
        pick = torch.tensor(indices, dtype=torch.long)
        mean, low, high = bootstrap_mean_ci(per_image[pick])
        report.groups[group] = GroupResult(len(indices), mean, None, None, low, high)
    return report


def run_bench(
    names: list[str],
    backend: WatermarkBackend,
    samples: list[ManifestSample],
    root: Path,
    on_report: Callable[[LocalizerReport], None] | None = None,
) -> list[LocalizerReport]:
    """Evaluate the named localizers on `samples` against their manifest ground truth.

    Raises:
        LocalizerError: unknown localizer name or missing ground truth.
    """
    unknown = [n for n in names if n not in LOCALIZER_FACTORIES]
    if unknown:
        raise LocalizerError(
            f"unknown localizer(s) {unknown}; known: {sorted(LOCALIZER_FACTORIES)}"
        )
    truth = load_truth(samples, root)
    reports: list[LocalizerReport] = []
    for name in names:
        localizer = LOCALIZER_FACTORIES[name](backend)
        started = time.monotonic()
        if isinstance(localizer, ScoreLocalizer):
            maps = compute_outputs(localizer, samples, root, scores=True)
            seconds = (time.monotonic() - started) / max(1, len(samples))
            reports.append(
                evaluate_score_localizer(name, localizer.threshold, maps, truth, samples, seconds)
            )
        else:
            masks = compute_outputs(localizer, samples, root, scores=False)
            seconds = (time.monotonic() - started) / max(1, len(samples))
            reports.append(evaluate_mask_localizer(name, masks, truth, samples, seconds))
        if on_report is not None:
            on_report(reports[-1])
    return reports


def format_reports(reports: list[LocalizerReport]) -> str:
    """Plain-text table: one block per localizer, one line per group."""
    lines: list[str] = []
    for report in reports:
        best = f", best cut {report.best_threshold:.2f}" if report.best_threshold else ""
        lines.append(f"== {report.name} ({report.seconds_per_image:.2f} s/img{best})")
        lines.append(
            f"  {'group':<13}{'n':>5}{'default':>9}{'otsu':>8}{'cv-calib':>10}{'95% CI':>18}"
        )
        for group, res in report.groups.items():
            otsu = f"{res.miou_otsu:.3f}" if res.miou_otsu is not None else "-"
            cal = f"{res.miou_calibrated:.3f}" if res.miou_calibrated is not None else "-"
            interval = (
                f"[{res.ci_low:.3f}, {res.ci_high:.3f}]"
                if res.ci_low is not None and res.ci_high is not None
                else "-"
            )
            lines.append(
                f"  {group:<13}{res.n_images:>5}{res.miou_default:>9.3f}{otsu:>8}{cal:>10}"
                f"{interval:>18}"
            )
    return "\n".join(lines)
