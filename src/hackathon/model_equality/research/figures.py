"""Matplotlib figures of the phase-1 report (light surface, thin marks, one axis per chart).

Colours follow the team's data-visualisation palette: categorical slots are used in order and
bound to an entity (base model S = blue, other models = orange), magnitudes use one blue ramp,
and no chart has two y-axes. Every function writes one PNG and returns its path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from hackathon.model_equality.clustering import build_linkage, to_distance
from hackathon.model_equality.research.dataset import ReferenceTruth
from hackathon.model_equality.research.evaluate import pairs_of
from hackathon.model_equality.research.pipeline import ResearchArtifacts

FloatMatrix = npt.NDArray[np.float64]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
MAGENTA = "#e87ba4"
VIOLET = "#6250d6"
YELLOW = "#eda100"
RED = "#e34948"
SERIES = (BLUE, ORANGE, AQUA, MAGENTA, VIOLET, YELLOW, RED)
RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])
DPI = 130
BASE_MODEL_LABEL = "S"
"""Model label of the undisguised base model and of its four disguises in `practice_answers.csv`."""
ROC_TOP_CHANNELS = 4
AUC_CHANCE = 0.5


def _style() -> None:
    """Global rcParams: surface, ink, hairline grid."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    })


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _labels(truth: ReferenceTruth) -> list[str]:
    return [truth.short_name(e) for e in truth.endpoints]


def _is_base(truth: ReferenceTruth, index: int) -> bool:
    return truth.model[truth.endpoints[index]] == BASE_MODEL_LABEL


def heatmap(
    matrix: FloatMatrix, truth: ReferenceTruth, title: str, path: Path, vmax: float | None = None,
    label: str = "pair score (z)",
) -> Path:
    """Annotated 9 x 9 heatmap of a pair matrix, endpoints ordered S first then T1..T4."""
    _style()
    order = sorted(range(len(truth.endpoints)),
                   key=lambda i: (0 if _is_base(truth, i) else 1, truth.model[truth.endpoints[i]],
                                  truth.endpoints[i]))
    ordered = matrix[np.ix_(order, order)]
    names = [_labels(truth)[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    shown = np.where(np.isfinite(ordered), ordered, 0.0)
    image = ax.imshow(shown, cmap=RAMP, vmin=0.0, vmax=vmax if vmax else float(np.nanmax(shown)))
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                dark = shown[i, j] > 0.55 * image.get_clim()[1]
                ax.text(j, i, f"{shown[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
    n_base = sum(1 for i in order if _is_base(truth, i))
    ax.axhline(n_base - 0.5, color=INK_SECONDARY, linewidth=0.8)
    ax.axvline(n_base - 0.5, color=INK_SECONDARY, linewidth=0.8)
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, shrink=0.8, label=label)
    return _save(fig, path)


def dendrogram_figure(
    matrix: FloatMatrix, truth: ReferenceTruth, title: str, path: Path,
) -> Path:
    """Average-linkage dendrogram of a pair-score matrix; leaves of the base model S in blue."""
    _style()
    distance, _ = to_distance(matrix)
    tree = build_linkage(distance)
    names = _labels(truth)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.grid(False)
    result = dendrogram(tree, labels=names, ax=ax, orientation="left", color_threshold=0,
                        above_threshold_color=INK_SECONDARY, leaf_font_size=8)
    for tick in ax.get_yticklabels():
        index = names.index(tick.get_text())
        tick.set_color(BLUE if _is_base(truth, index) else ORANGE)
    ax.set_xlabel("average-linkage height (combined z)")
    ax.set_title(title, loc="left")
    del result
    return _save(fig, path)


def channel_auc_bars(
    pooled_rows: Sequence[Mapping[str, Any]], path: Path, title: str,
) -> Path:
    """Horizontal bars: AUC of each pooled channel, same-model versus different-model pairs."""
    _style()
    rows = sorted((r for r in pooled_rows if r["auc"] is not None), key=lambda r: r["auc"])
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.2))
    values = [r["auc"] for r in rows]
    ax.barh([r["kernel"] for r in rows], values, color=BLUE, height=0.62)
    ax.axvline(AUC_CHANCE, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    for position, value in enumerate(values):
        ax.text(value + 0.01, position, f"{value:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("AUC (different-model pair scored higher than same-model pair)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def category_kernel_heatmap(
    rows: Sequence[Mapping[str, Any]], path: Path, title: str,
) -> Path:
    """Category x kernel table of AUC values."""
    _style()
    categories = sorted({r["category"] for r in rows})
    kernels = sorted({r["kernel"] for r in rows})
    grid = np.full((len(categories), len(kernels)), np.nan)
    for row in rows:
        if row["auc"] is not None:
            grid[categories.index(row["category"]), kernels.index(row["kernel"])] = row["auc"]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(categories) + 1.6))
    ax.grid(False)
    image = ax.imshow(np.nan_to_num(grid, nan=AUC_CHANCE), cmap=RAMP, vmin=AUC_CHANCE, vmax=1.0,
                      aspect="auto")
    ax.set_xticks(range(len(kernels)), kernels, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)), categories)
    for i in range(len(categories)):
        for j in range(len(kernels)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if grid[i, j] > 0.78 else INK)
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, shrink=0.8, label="AUC (0.5 = chance)")
    return _save(fig, path)


def roc_points(different: Sequence[float], same: Sequence[float]) -> tuple[list[float], list[float]]:
    """ROC of "pair is different-model" with the pair score as the detector."""
    d = np.asarray(different, dtype=np.float64)
    s = np.asarray(same, dtype=np.float64)
    thresholds = np.unique(np.concatenate([d, s, [d.max() + 1.0, s.min() - 1.0]]))[::-1]
    fpr = [float((s >= t).mean()) for t in thresholds]
    tpr = [float((d >= t).mean()) for t in thresholds]
    return fpr, tpr


def roc_figure(
    pooled: Mapping[str, FloatMatrix], extra: Mapping[str, FloatMatrix], truth: ReferenceTruth,
    pooled_rows: Sequence[Mapping[str, Any]], path: Path, title: str,
) -> Path:
    """ROC curves of the best pooled channels and of the combined scores."""
    _style()
    ranked = [r["kernel"] for r in sorted((r for r in pooled_rows if r["auc"] is not None),
                                          key=lambda r: -r["auc"])][:ROC_TOP_CHANNELS]
    curves: dict[str, FloatMatrix] = {k: pooled[k] for k in ranked}
    curves.update(extra)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for position, (name, matrix) in enumerate(curves.items()):
        same: list[float] = []
        different: list[float] = []
        for i, j, is_same in pairs_of(truth):
            if np.isfinite(matrix[i, j]):
                (same if is_same else different).append(float(matrix[i, j]))
        fpr, tpr = roc_points(different, same)
        ax.plot(fpr, tpr, color=SERIES[position % len(SERIES)], linewidth=1.6, label=name)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("false alarm: same-model pair called different")
    ax.set_ylabel("detection: different-model pair called different")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    return _save(fig, path)


def trap_heatmap(
    traps: Mapping[str, Mapping[str, Mapping[str, float | None]]], truth: ReferenceTruth,
    path: Path, title: str,
) -> Path:
    """Pooled z of the plain base model against each other endpoint, one column per kernel."""
    _style()
    kernels = list(traps)
    others = list(next(iter(traps.values())))
    grid = np.array([[traps[k][o]["z"] if traps[k][o]["z"] is not None else np.nan
                      for k in kernels] for o in others], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(others) + 1.8))
    ax.grid(False)
    image = ax.imshow(np.nan_to_num(np.clip(grid, 0, None)), cmap=RAMP, aspect="auto",
                      vmin=0, vmax=max(6.0, float(np.nanmax(grid))))
    ax.set_xticks(range(len(kernels)), kernels, rotation=45, ha="right")
    ax.set_yticks(range(len(others)), [truth.short_name(o) for o in others])
    top = image.get_clim()[1]
    for i in range(len(others)):
        for j in range(len(kernels)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if grid[i, j] > 0.55 * top else INK)
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, shrink=0.8, label="pooled z vs ref-01 (plain S)")
    return _save(fig, path)


def mcc_vs_n_figure(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> Path:
    """MCC (leave-one-endpoint-out) versus answers per endpoint, mean and 10-90 percent band."""
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for position, method in enumerate(methods):
        mine = sorted((r for r in rows if r["method"] == method and r["mcc_mean"] is not None),
                      key=lambda r: r["answers_per_endpoint"])
        x = [r["answers_per_endpoint"] for r in mine]
        color = SERIES[position % len(SERIES)]
        ax.plot(x, [r["mcc_mean"] for r in mine], color=color, linewidth=1.8, marker="o",
                markersize=4, label=method)
        ax.fill_between(x, [r["mcc_p10"] for r in mine], [r["mcc_p90"] for r in mine],
                        color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("answers per endpoint (prompts x 2 repeats)")
    ax.set_ylabel("model MCC, leave-one-endpoint-out")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def z_growth_figure(
    rows: Sequence[Mapping[str, Any]], truth: ReferenceTruth, path: Path, title: str,
) -> Path:
    """Pooled z of plain S against each other endpoint versus answers per endpoint, per kernel.

    Blue = disguises of S (should stay flat near 0), orange = other models (should grow).
    """
    _style()
    kernels = list(dict.fromkeys(r["kernel"] for r in rows))
    fig, axes = plt.subplots(1, len(kernels), figsize=(3.6 * len(kernels), 3.8), sharex=True)
    for ax, kernel in zip(np.atleast_1d(axes), kernels, strict=True):
        for endpoint in truth.endpoints:
            mine = sorted((r for r in rows if r["kernel"] == kernel and r["endpoint"] == endpoint
                           and r["z_mean"] is not None), key=lambda r: r["answers_per_endpoint"])
            if not mine:
                continue
            is_base = truth.model[endpoint] == BASE_MODEL_LABEL
            x = [r["answers_per_endpoint"] for r in mine]
            y = [r["z_mean"] for r in mine]
            ax.plot(x, y, color=BLUE if is_base else ORANGE, linewidth=1.2,
                    linestyle="-" if is_base else "--", marker="o", markersize=2.5)
            ax.text(x[-1] + 1, y[-1], endpoint[-2:], fontsize=7, va="center", color=INK_SECONDARY)
        ax.set_title(kernel, loc="left")
        ax.set_xlabel("answers per endpoint")
    np.atleast_1d(axes)[0].set_ylabel("pooled z vs plain S (ref-01)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    return _save(fig, path)


def build_all(
    metrics: Mapping[str, Any], artifacts: ResearchArtifacts, truth: ReferenceTruth, out_dir: Path,
) -> dict[str, Path]:
    """Write every figure of the report into `out_dir` and return their paths by name."""
    figures: dict[str, Path] = {}
    pooled_rows = metrics["pooled"]
    figures["matrix_prior"] = heatmap(
        artifacts.combined_prior, truth, "Combined z between the 9 reference endpoints (kit prior)",
        out_dir / "matrix_prior.png", vmax=12.0)
    figures["matrix_calibrated"] = heatmap(
        artifacts.combined_calibrated, truth,
        "Combined z, calibrated weights (in-sample, optimistic)", out_dir / "matrix_calibrated.png",
        vmax=12.0)
    figures["dendrogram"] = dendrogram_figure(
        artifacts.combined_prior, truth, "Dendrogram of the 9 reference endpoints (prior weights)",
        out_dir / "dendrogram.png")
    figures["channel_auc"] = channel_auc_bars(
        pooled_rows, out_dir / "channel_auc.png", "Which channel separates models: pooled AUC")
    figures["category_kernel"] = category_kernel_heatmap(
        metrics["channels"], out_dir / "category_kernel_auc.png",
        "AUC per prompt category and kernel")
    figures["roc"] = roc_figure(
        artifacts.pooled, {"kit prior (combined)": artifacts.combined_prior,
                           "kit calibrated (in-sample)": artifacts.combined_calibrated},
        truth, pooled_rows, out_dir / "roc.png", "ROC, same model versus different model")
    figures["traps"] = trap_heatmap(
        metrics["traps"], truth, out_dir / "traps.png",
        "Plain S against each other endpoint: disguises should stay low, T1..T4 high")
    if "mcc_vs_n" in metrics:
        figures["mcc_vs_n"] = mcc_vs_n_figure(
            metrics["mcc_vs_n"], out_dir / "mcc_vs_n.png", "MCC as a function of answers per endpoint")
    if "z_vs_n" in metrics:
        figures["z_vs_n"] = z_growth_figure(
            metrics["z_vs_n"], truth, out_dir / "z_vs_n.png",
            "Separation from plain S versus n (blue: S disguises, orange dashed: other models)")
    return figures
