"""Reusable figure toolbox of the unlearning research teams: pure functions, arrays in, Figure out.

Every function takes plain numpy arrays / mappings and returns a `matplotlib.figure.Figure` built
WITHOUT pyplot (no global state, no backend, safe in tests and in parallel). Save with
`save_figure(fig, path)`. Colours follow the dataviz reference palette (fixed categorical order,
one sequential hue, a gray-centred diverging pair); text is always ink-coloured, never series-coloured.

Typical use:

    fig = plot_ecdf({"orig": loss_orig, "unlearned": loss_sub}, xlabel="loss", log_x=True)
    save_figure(fig, Path("figures/loss_ecdf.png"))
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
FAMILY_COLORS = {
    "gold_soft": SERIES_COLORS[0],
    "scrub": SERIES_COLORS[1],
    "grad_ascent": SERIES_COLORS[7],
    "uniform": SERIES_COLORS[6],
    "finetune_retain": SERIES_COLORS[2],
    "weight_noise": SERIES_COLORS[3],
    "identity": INK_SECONDARY,
    "gold_heldout": SERIES_COLORS[5],
}
"""One colour per method family, identical in every figure of every report (colour-blind-safe set)."""
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#eaf2fb", "#2a78d6", "#0d2f5c"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f1f0ec", "#2a78d6"])
DEFAULT_DPI = 130
AXIS_LABEL_SIZE = 12
BASE_FONT_SIZE = 11
TITLE_WRAP_CHARS = 62
TITLE_LINE_INCHES = 0.26
TITLE_PAD_INCHES = 0.15
DEFAULT_SIZE = (7.2, 4.2)
LINE_WIDTH = 2.0
MARKER_AREA = 22.0
HIGHLIGHT_ALPHA = 0.9
MIN_POINTS_FOR_FIT = 3


rcParams.update({"axes.labelsize": AXIS_LABEL_SIZE, "font.size": BASE_FONT_SIZE})


class VizError(ValueError):
    """Figure inputs are inconsistent (empty data, mismatched lengths, too many series)."""


def _color(index: int) -> str:
    """Categorical colour in fixed order; a 9th series is refused instead of cycled."""
    if index >= len(SERIES_COLORS):
        raise VizError(
            f"at most {len(SERIES_COLORS)} series per figure; fold the rest into 'other'"
        )
    return SERIES_COLORS[index]


def _new_figure(size: tuple[float, float]) -> tuple[Figure, Axes]:
    figure = Figure(figsize=size, dpi=DEFAULT_DPI, facecolor=SURFACE)
    axes = figure.subplots()
    _style_axes(axes)
    return figure, axes


def _style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(INK_SECONDARY)
    axes.tick_params(colors=INK_SECONDARY, labelsize=11)
    axes.xaxis.label.set_color(INK)
    axes.yaxis.label.set_color(INK)
    axes.title.set_color(INK)


def _finish(figure: Figure, title: str) -> Figure:
    """Add the (wrapped) conclusion title above the axes and lay the figure out."""
    wrapped = textwrap.fill(title, TITLE_WRAP_CHARS) if title else ""
    lines = wrapped.count("\n") + 1 if wrapped else 0
    if wrapped:
        figure.suptitle(wrapped, color=INK, fontsize=13, x=0.01, ha="left", fontweight="bold")
    height = figure.get_figheight()
    top = 1.0 - (TITLE_LINE_INCHES * lines + TITLE_PAD_INCHES) / height if lines else 1.0
    figure.tight_layout(rect=(0, 0, 1, top))
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """Write the figure as PNG (creates the parent folder) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DEFAULT_DPI, facecolor=SURFACE)
    return path


def _require(samples: Mapping[str, FloatArray]) -> None:
    if not samples:
        raise VizError("no series to plot")
    for name, values in samples.items():
        if len(values) == 0:
            raise VizError(f"series '{name}' is empty")


def plot_ecdf(
    samples: Mapping[str, FloatArray],
    xlabel: str,
    *,
    log_x: bool = False,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
    floor: float = 1e-6,
) -> Figure:
    """Empirical CDF of each series (the honest way to compare distributions with long tails).

    Args:
        samples: name -> 1-D values.
        xlabel: axis label.
        log_x: log-scaled x (values are floored at `floor` so zeros stay visible).
        title: figure title.
        size: figure size in inches.
        floor: lower clip for the log axis.
    """
    _require(samples)
    figure, axes = _new_figure(size)
    for index, (name, values) in enumerate(samples.items()):
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        if log_x:
            ordered = np.maximum(ordered, floor)
        axes.step(
            ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
            color=_color(index), linewidth=LINE_WIDTH, label=f"{name} (mean {ordered.mean():.3g})",
        )  # fmt: skip
    if log_x:
        axes.set_xscale("log")
    axes.set_xlabel(xlabel)
    axes.set_ylabel("fraction of images <= x")
    axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    return _finish(figure, title)


def plot_histograms(
    samples: Mapping[str, FloatArray],
    xlabel: str,
    *,
    bins: int = 40,
    value_range: tuple[float, float] | None = None,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """Overlaid step histograms (density) on shared bin edges."""
    _require(samples)
    everything = np.concatenate([np.asarray(v, dtype=np.float64) for v in samples.values()])
    low, high = value_range if value_range else (float(everything.min()), float(everything.max()))
    edges = np.linspace(low, high if high > low else low + 1.0, bins + 1)
    figure, axes = _new_figure(size)
    for index, (name, values) in enumerate(samples.items()):
        axes.hist(
            np.clip(values, low, high), bins=edges.tolist(), density=True, histtype="step",
            color=_color(index), linewidth=LINE_WIDTH, label=name,
        )  # fmt: skip
    axes.set_xlabel(xlabel)
    axes.set_ylabel("density")
    axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    return _finish(figure, title)


def plot_boxes(
    samples: Mapping[str, FloatArray],
    ylabel: str,
    *,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """One box per group (median, quartiles, 5-95 % whiskers); the mean is the dot."""
    _require(samples)
    figure, axes = _new_figure(size)
    names = list(samples)
    data = [np.asarray(samples[name], dtype=np.float64) for name in names]
    boxes = axes.boxplot(
        data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
        medianprops={"color": INK, "linewidth": 1.6},
    )  # fmt: skip
    for patch in boxes["boxes"]:
        patch.set_facecolor("#cfe0f6")
        patch.set_edgecolor("#2a78d6")
    axes.scatter(
        range(1, len(names) + 1), [d.mean() for d in data], color=SERIES_COLORS[1],
        s=MARKER_AREA * 1.6, zorder=3, label="mean",
    )  # fmt: skip
    axes.set_xticks(range(1, len(names) + 1), names, rotation=20, ha="right")
    axes.set_ylabel(ylabel)
    axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    return _finish(figure, title)


def plot_ranked_bars(
    names: Sequence[str],
    values: FloatArray,
    xlabel: str,
    *,
    errors: FloatArray | None = None,
    highlight: Sequence[str] = (),
    color_by: Mapping[str, str] | None = None,
    title: str = "",
    size: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal bars sorted by value (best on top), optional error bars, highlighted names."""
    if len(names) != len(values) or len(names) == 0:
        raise VizError("names and values must be non-empty and of equal length")
    order = np.argsort(np.asarray(values))
    height = size[1] if size else max(2.6, 0.27 * len(names) + 1.0)
    figure, axes = _new_figure(size or (7.4, height))
    colors = [
        (color_by or {}).get(
            names[i], SERIES_COLORS[1] if names[i] in highlight else SERIES_COLORS[0]
        )
        for i in order
    ]
    axes.barh(
        [names[i] for i in order], np.asarray(values)[order], color=colors, height=0.62,
        xerr=None if errors is None else np.asarray(errors)[order],
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.0, "capsize": 2},
    )  # fmt: skip
    axes.set_xlabel(xlabel)
    axes.grid(False, axis="y")
    return _finish(figure, title)


def plot_scatter(
    x: FloatArray,
    y: FloatArray,
    xlabel: str,
    ylabel: str,
    *,
    labels: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    group_colors: Mapping[str, str] | None = None,
    annotate_top: int = 0,
    diagonal: bool = False,
    log_x: bool = False,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """Scatter with optional group colours, diagonal, and labels on the `annotate_top` highest y."""
    if len(x) != len(y) or len(x) == 0:
        raise VizError("x and y must be non-empty and of equal length")
    figure, axes = _new_figure(size)
    if groups is None:
        axes.scatter(x, y, s=MARKER_AREA, color=SERIES_COLORS[0], alpha=HIGHLIGHT_ALPHA,
                     edgecolors=SURFACE, linewidths=0.6)  # fmt: skip
    else:
        unique = list(dict.fromkeys(groups))
        for index, group in enumerate(unique):
            mask = np.asarray([g == group for g in groups])
            axes.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=MARKER_AREA,
                         color=(group_colors or {}).get(group, _color(index)),
                alpha=HIGHLIGHT_ALPHA,
                label=group,
                         edgecolors=SURFACE, linewidths=0.6)  # fmt: skip
        axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    if diagonal:
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        axes.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    if labels is not None and annotate_top > 0:
        for index in np.argsort(-np.asarray(y))[:annotate_top]:
            axes.annotate(labels[int(index)], (x[index], y[index]), fontsize=10, color=INK,
                          xytext=(4, 4), textcoords="offset points")  # fmt: skip
    if log_x:
        axes.set_xscale("log")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return _finish(figure, title)


def plot_heatmap(
    matrix: FloatArray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    diverging: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    colorbar_label: str = "",
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """Annotated heatmap (e.g. a correlation matrix); NaN cells stay blank."""
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise VizError(
            f"matrix {matrix.shape} does not match {len(row_labels)}x{len(col_labels)} labels"
        )
    figure, axes = _new_figure(size)
    axes.grid(False)
    image = axes.imshow(matrix, cmap=DIVERGING if diverging else SEQUENTIAL, vmin=vmin, vmax=vmax,
                        aspect="auto")  # fmt: skip
    axes.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    axes.set_yticks(range(len(row_labels)), row_labels)
    if annotate:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if np.isfinite(matrix[row, col]):
                    axes.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",                              fontsize=10, color=INK)  # fmt: skip
    colorbar = figure.colorbar(image, ax=axes, fraction=0.04)
    colorbar.set_label(colorbar_label, color=INK)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=11)
    return _finish(figure, title)


def plot_transition_matrix(
    counts: FloatArray,
    *,
    title: str = "",
    size: tuple[float, float] = (6.4, 5.6),
    xlabel: str = "fallback class (argmax on the strongly cropped view)",
    ylabel: str = "class on the exact image",
) -> Figure:
    """Class-to-class transition counts on a log colour scale (zeros blank)."""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise VizError("counts must be a square matrix")
    figure, axes = _new_figure(size)
    axes.grid(False)
    masked = np.ma.masked_less_equal(counts, 0)
    image = axes.imshow(masked, cmap=SEQUENTIAL, norm=LogNorm(vmin=1, vmax=max(2.0, float(counts.max()))),
                        aspect="equal", interpolation="nearest")  # fmt: skip
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.046)
    colorbar.set_label("images", color=INK)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=11)
    return _finish(figure, title)


def plot_lines(
    x: FloatArray,
    series: Mapping[str, FloatArray],
    xlabel: str,
    ylabel: str,
    *,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """Several y-series against one shared x (legend always present for 2+ series)."""
    _require(series)
    figure, axes = _new_figure(size)
    for index, (name, values) in enumerate(series.items()):
        axes.plot(x, values, color=_color(index), linewidth=LINE_WIDTH, marker="o",
                  markersize=4, label=name)  # fmt: skip
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if len(series) > 1:
        axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    return _finish(figure, title)


def plot_grouped_bars(
    categories: Sequence[str],
    series: Mapping[str, FloatArray],
    ylabel: str,
    *,
    title: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
) -> Figure:
    """Bars of several series side by side per category."""
    _require(series)
    figure, axes = _new_figure(size)
    width = 0.8 / len(series)
    positions = np.arange(len(categories))
    for index, (name, values) in enumerate(series.items()):
        axes.bar(positions + (index - (len(series) - 1) / 2) * width, values, width=width * 0.92,
                 color=_color(index), label=name)  # fmt: skip
    axes.set_xticks(positions, categories, rotation=20, ha="right")
    axes.set_ylabel(ylabel)
    axes.legend(frameon=False, fontsize=11, labelcolor=INK)
    axes.grid(False, axis="x")
    return _finish(figure, title)


def plot_image_grid(
    images: Sequence[NDArray[np.uint8]],
    titles: Sequence[str],
    columns: int,
    *,
    title: str = "",
    cell_size: float = 1.7,
) -> Figure:
    """Grid of uint8 [H, W, 3] images with a short title each (examples, successes and failures)."""
    if len(images) != len(titles) or len(images) == 0 or columns < 1:
        raise VizError("images and titles must be non-empty, equal in length; columns >= 1")
    rows = -(-len(images) // columns)
    size = (cell_size * columns, (cell_size + 0.35) * rows + 0.4)
    figure = Figure(figsize=size, dpi=DEFAULT_DPI, facecolor=SURFACE)
    axes_grid = figure.subplots(rows, columns, squeeze=False)
    for index, axes in enumerate(axes_grid.ravel()):
        axes.set_axis_off()
        if index < len(images):
            axes.imshow(images[index])
            axes.set_title(titles[index], fontsize=10, color=INK)
    return _finish(figure, title)
