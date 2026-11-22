"""Figures of the m-met-1 report: ep dendrogram, ordered distance matrix, MCC versus N on ref.

Same conventions as `figures.py`: light surface, thin marks, categorical colours bound to an
entity (families in palette order), one blue ramp for magnitudes, no dual axes.
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

from hackathon.model_equality.smalln import build_tree

FloatMatrix = npt.NDArray[np.float64]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#e8f0fb", "#2a78d6", "#0b2f5e"])
DPI = 150
MODEL_COLOR = "#2a78d6"
FAMILY_COLOR = "#eb6834"
METHOD_COLORS = {"ham+len": "#2a78d6", "ham+spec+len": "#1baf7a", "ham": "#eda100", "len": "#6250d6"}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def dendrogram_figure(
    distance: FloatMatrix, endpoints: Sequence[str], families: Sequence[str],
    tau_family: float, path: Path,
) -> Path:
    """Average-linkage dendrogram of the ep family distance with its cut; labels coloured by family."""
    tree = build_tree(distance)
    figure, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    _style(ax)
    dendrogram(tree, labels=list(endpoints), ax=ax, color_threshold=0,
                        above_threshold_color=INK_SECONDARY, leaf_font_size=8)
    ax.axhline(tau_family, color=FAMILY_COLOR, linestyle="--", linewidth=1.0)
    ax.text(0.01, tau_family, f" family cut {tau_family:.2f}", color=FAMILY_COLOR, fontsize=8,
            va="bottom", transform=ax.get_yaxis_transform())
    names = sorted(set(families))
    color_of = {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(names)}
    family_of = dict(zip(endpoints, families, strict=True))
    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)
        label.set_color(color_of[family_of[label.get_text()]])
    ax.set_ylabel("average-linkage family distance (identity + yes/no)", color=INK_SECONDARY, fontsize=9)
    ax.set_title("ep endpoints: dendrogram after E1 (5 answers each); label colour = family",
                 color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return path


def heatmap_figure(
    distance: FloatMatrix, endpoints: Sequence[str], path: Path, title: str,
) -> Path:
    """Distance matrix with rows and columns in dendrogram leaf order."""
    tree = build_tree(distance)
    order = dendrogram(tree, no_plot=True)["leaves"]
    ordered = distance[np.ix_(order, order)]
    figure, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=SURFACE)
    image = ax.imshow(ordered, cmap=BLUE_RAMP, vmin=0.0, vmax=float(np.percentile(distance, 98)))
    labels = [endpoints[i] for i in order]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5, color=INK_SECONDARY)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return path


def mcc_vs_n_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Held-out model MCC versus N prompts (one draw each), per method, band = p10 to p90."""
    figure, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style(ax)
    for method, color in METHOD_COLORS.items():
        subset = sorted((r for r in rows if r["method"] == method), key=lambda r: r["n_prompts"])
        if not subset:
            continue
        sizes = [r["n_prompts"] for r in subset]
        ax.plot(sizes, [r["mcc_mean"] for r in subset], color=color, linewidth=1.6, marker="o",
                markersize=4, label=method)
        ax.fill_between(sizes, [r["mcc_p10"] for r in subset], [r["mcc_p90"] for r in subset],
                        color=color, alpha=0.10, linewidth=0)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.axhline(0.52, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.text(0.2, 0.53, "coarse split ceiling 0.52", fontsize=8, color=INK_SECONDARY,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("prompts per endpoint (one answer each)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("model MCC on ref, held-out prompts", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Small-N statistic on the 9 ref: MCC against N (mean, band p10 to p90)",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylim(-0.1, 0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return path
