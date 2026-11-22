"""Figures of the team A report (matplotlib, Agg backend). Every figure has a one-line takeaway
that the report puts in its caption.

Usage (kit venv): python -m hackathon.wmloc.research.figures --report-dir reports/phase1/wmloc/team-a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hackathon.wmloc.constants import KNOWN_METHODS
from hackathon.wmloc.research.data import SPLIT_TRAIN, load_image
from hackathon.wmloc.research.maps import DEFAULT_SIDES, margin_map, robust_z
from hackathon.wmloc.research.rectangles import box_from_grid

FIGURE_DPI = 110
COLOR_TRUTH = "#00c853"
COLOR_PRED = "#ff1744"
COLOR_FULL = "#1f77b4"
COLOR_RECT = "#d62728"
EXAMPLES_PER_CELL = 2
CELL_PX = 4


def _load(report_dir: Path, method: str) -> dict[str, np.ndarray] | None:
    path = report_dir / f"cv_masks_{method}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def figure_switch(report_dir: Path, figures: Path) -> None:
    """Histogram of whole-image agreement per regime, per method."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for axis, method in zip(axes, KNOWN_METHODS, strict=True):
        data = _load(report_dir, method)
        axis.set_title(method)
        if data is None:
            continue
        bins = np.linspace(0.3, 1.0, 36)
        axis.hist(
            data["agreement"][data["is_full"]], bins=bins, color=COLOR_FULL, alpha=0.7, label="full"
        )
        axis.hist(
            data["agreement"][~data["is_full"]],
            bins=bins,
            color=COLOR_RECT,
            alpha=0.7,
            label="rectangle",
        )
        axis.set_xlabel("whole-image bit agreement")
    axes[0].set_ylabel("images")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_switch_agreement.png", dpi=FIGURE_DPI)
    plt.close(fig)


def figure_tracks(metrics: dict[str, Any], figures: Path) -> None:
    """Out-of-fold mIoU of each rule, per method and regime."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
    for column, method in enumerate(KNOWN_METHODS):
        if method not in metrics:
            continue
        for row, (key, title) in enumerate(
            (("full_rules", "full regime"), ("rect_rules", "rectangle regime"))
        ):
            rules = metrics[method][key]
            names = list(rules)
            means = [rules[n]["mean"] for n in names]
            errors = [rules[n]["std"] for n in names]
            axis = axes[row, column]
            axis.barh(names, means, xerr=errors, color=COLOR_FULL if row == 0 else COLOR_RECT)
            axis.set_xlim(0.3, max(0.7, max(means) + 0.05))
            axis.set_title(f"{method}: {title}")
            axis.set_xlabel("out-of-fold mIoU (error bar = std over folds)")
    fig.tight_layout()
    fig.savefig(figures / "fig_miou_by_rule.png", dpi=FIGURE_DPI)
    plt.close(fig)


def figure_iou_distribution(report_dir: Path, figures: Path) -> None:
    """Distribution of the per-image mIoU of the deployable pipeline."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for axis, method in zip(axes, KNOWN_METHODS, strict=True):
        data = _load(report_dir, method)
        axis.set_title(method)
        if data is None:
            continue
        bins = np.linspace(0.0, 1.0, 26)
        axis.hist(
            data["per_image"][data["is_full"]], bins=bins, color=COLOR_FULL, alpha=0.7, label="full"
        )
        axis.hist(
            data["per_image"][~data["is_full"]],
            bins=bins,
            color=COLOR_RECT,
            alpha=0.7,
            label="rectangle",
        )
        axis.set_xlabel("per-image mIoU (out-of-fold)")
    axes[0].set_ylabel("images")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "fig_iou_distribution.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _contour(axis: Any, mask: np.ndarray, color: str) -> None:
    upsampled = np.kron(mask.astype(float), np.ones((CELL_PX, CELL_PX)))
    if upsampled.max() > 0 and upsampled.min() < 1:
        axis.contour(upsampled, levels=[0.5], colors=color, linewidths=1.6)


def figure_overlays(report_dir: Path, figures: Path, best: bool) -> None:
    """Image + truth (green) + prediction (red) for successes (best) or failures (worst)."""
    rows = len(KNOWN_METHODS) * 2
    fig, axes = plt.subplots(rows, EXAMPLES_PER_CELL, figsize=(2.6 * EXAMPLES_PER_CELL, 2.6 * rows))
    row = 0
    for method in KNOWN_METHODS:
        data = _load(report_dir, method)
        for regime_full in (True, False):
            for column in range(EXAMPLES_PER_CELL):
                axis = axes[row, column]
                axis.axis("off")
                if data is None:
                    continue
                indices = np.flatnonzero(data["is_full"] == regime_full)
                order = indices[np.argsort(data["per_image"][indices])]
                order = order[::-1] if best else order
                if column >= len(order):
                    continue
                index = int(order[column])
                name = str(data["names"][index])
                axis.imshow(load_image(SPLIT_TRAIN, name).permute(1, 2, 0).numpy())
                _contour(axis, data["truth"][index], COLOR_TRUTH)
                _contour(axis, data["masks"][index], COLOR_PRED)
                axis.set_title(
                    f"{method[:5]} {'full' if regime_full else 'rect'} mIoU {data['per_image'][index]:.2f}",
                    fontsize=8,
                )
            row += 1
    fig.tight_layout()
    fig.savefig(
        figures / ("fig_overlays_success.png" if best else "fig_overlays_failure.png"),
        dpi=FIGURE_DPI,
    )
    plt.close(fig)


def figure_scale_space(report_dir: Path, figures: Path, method: str) -> bool:
    """Best-contrast probe side versus the side predicted by the scale-matching hypothesis."""
    data = _load(report_dir, method)
    if data is None:
        return False
    from hackathon.wmloc.research.data import load_records, load_truth
    from hackathon.wmloc.research.extract import cache_path
    from hackathon.wmloc.research.maps import pass_name_for_side, to_grid

    records = [
        r
        for r in load_records(SPLIT_TRAIN)
        if r.method == method
        and r.full_watermark is False
        and all(
            cache_path(SPLIT_TRAIN, method, r.name, pass_name_for_side(s)).exists()
            for s in DEFAULT_SIDES
        )
    ]
    if len(records) < 5:
        return False
    truth = load_truth([r.name for r in records])
    predicted: list[float] = []
    observed: list[float] = []
    for record, mask in zip(records, truth, strict=True):
        box = box_from_grid(mask.astype(float), 0.5)
        if box is None:
            continue
        size = float(np.sqrt((box[2] - box[0]) * (box[3] - box[1])))
        contrasts = []
        for side in DEFAULT_SIDES:
            z = to_grid(
                robust_z(margin_map(SPLIT_TRAIN, method, record.name, record.message, side))
            )
            inside, outside = z[mask == 1], z[mask == 0]
            contrasts.append(
                float(inside.mean() - outside.mean()) if len(inside) and len(outside) else 0.0
            )
        predicted.append(512.0 * 256.0 / size if method == "pixelseal" else 512.0 * 245.0 / size)
        observed.append(float(DEFAULT_SIDES[int(np.argmax(contrasts))]))
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.scatter(predicted, observed, s=18, color=COLOR_RECT, alpha=0.7)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("side predicted by scale matching: 512 x resolution / sqrt(rect area)")
    axis.set_ylabel("probe side with the best inside/outside contrast")
    axis.set_title(f"{method}: scale-space probe")
    fig.tight_layout()
    fig.savefig(figures / f"fig_scale_space_{method}.png", dpi=FIGURE_DPI)
    plt.close(fig)
    return True


def figure_evidence_maps(report_dir: Path, figures: Path, method: str) -> bool:
    """Evidence maps (per-position margin z-score, scale-space z-score) beside the truth."""
    data = _load(report_dir, method)
    if data is None:
        return False
    from hackathon.wmloc.research.data import load_records
    from hackathon.wmloc.research.extract import cache_path
    from hackathon.wmloc.research.maps import evidence_grid, pass_name_for_side
    from hackathon.wmloc.research.scoring import posmap_score_grid

    lookup = {r.name: r for r in load_records(SPLIT_TRAIN)}
    rect_indices = np.flatnonzero(~data["is_full"])
    order = rect_indices[
        np.argsort(-data["truth"][rect_indices].reshape(len(rect_indices), -1).mean(axis=1))
    ]
    shown = [int(i) for i in order[: 6 * 2]][::2][:6]
    fig, axes = plt.subplots(3, len(shown), figsize=(2.5 * len(shown), 7.2), squeeze=False)
    for column, index in enumerate(shown):
        record = lookup[str(data["names"][index])]
        axes[0, column].imshow(load_image(SPLIT_TRAIN, record.name).permute(1, 2, 0).numpy())
        _contour(axes[0, column], data["truth"][index], COLOR_TRUTH)
        axes[1, column].imshow(
            robust_z(posmap_score_grid(SPLIT_TRAIN, method, record.name, record.message)),
            cmap="magma",
        )
        _contour(axes[1, column], np.kron(data["truth"][index], np.ones((1, 1))), COLOR_TRUTH)
        has_scales = all(
            cache_path(SPLIT_TRAIN, method, record.name, pass_name_for_side(s)).exists()
            for s in DEFAULT_SIDES
        )
        if has_scales:
            evidence = evidence_grid(
                SPLIT_TRAIN, method, record.name, record.message, DEFAULT_SIDES
            )
            axes[2, column].imshow(evidence, cmap="magma")
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_title("image + truth", fontsize=8)
    axes[1, 0].set_title("per-position z (256)", fontsize=8)
    axes[2, 0].set_title("scale-space z", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / f"fig_evidence_maps_{method}.png", dpi=FIGURE_DPI)
    plt.close(fig)
    return True


def main() -> None:
    """CLI entry point: draw every figure the available data allows."""
    parser = argparse.ArgumentParser(description="Draw the team A figures.")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    figures = args.report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.report_dir / "metrics.json").read_text(encoding="utf-8"))
    figure_switch(args.report_dir, figures)
    figure_tracks(metrics, figures)
    figure_iou_distribution(args.report_dir, figures)
    figure_overlays(args.report_dir, figures, best=True)
    figure_overlays(args.report_dir, figures, best=False)
    for method in ("pixelseal", "trustmark"):
        figure_scale_space(args.report_dir, figures, method)
        figure_evidence_maps(args.report_dir, figures, method)


if __name__ == "__main__":
    main()
