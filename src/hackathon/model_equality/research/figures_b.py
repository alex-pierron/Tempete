"""Figures of the team-B phase-1 report (matplotlib, light surface, validated palette).

Colour roles follow the endpoint's *model*, not its rank: S (base model, all five disguises) is
blue, T1 (fp8) orange, T2 (LoRA 0.3) aqua, T3 (LoRA 0.7) violet, T4 (LoRA 1.5) red. The set was
checked with the dataviz validator (adjacent CVD delta E 9.2, normal-vision 27.6); aqua is
below 3:1 on the surface, so every aqua mark also carries a text label or a distinct marker.
Magnitude heat maps use a one-hue blue ramp.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
MODEL_COLOURS: Mapping[str, str] = {
    "S": "#2a78d6", "T1": "#eb6834", "T2": "#1baf7a", "T3": "#6250d6", "T4": "#e34948",
}
BLUE_RAMP: tuple[str, ...] = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", list(BLUE_RAMP))
SEQUENTIAL.set_bad("#ecebe7")
DPI = 130
COLUMN_ORDER: tuple[str, ...] = (
    "ref-01", "ref-03", "ref-05", "ref-06", "ref-08", "ref-02", "ref-09", "ref-04", "ref-07")
SHORT_KIND: Mapping[str, str] = {
    "plain": "plain", "system prompt": "system", "persona": "persona", "watermark": "watermark",
    "sampling at temperature 1.0": "temp 1.0", "quantized (fp8)": "fp8",
    "LoRA at strength 0.3": "LoRA 0.3", "LoRA at strength 0.7": "LoRA 0.7",
    "LoRA at strength 1.5": "LoRA 1.5",
}
KEY_GROUPS: frozenset[str] = frozenset(
    {"yesno", "choice", "numeric", "arith", "bank:arith", "bank:format"})
GROUP_KERNEL_KEY = "onehot_key"
GROUP_KERNEL_TEXT = "hamming_char"
KERNEL_LABELS: Mapping[str, str] = {
    "onehot_key": "exact answer", "hamming_char": "characters (Hamming)",
    "spectrum": "n-gram bag", "length": "length", "onehot_first": "first word",
}
KERNEL_MARKERS: Mapping[str, str] = {
    "onehot_key": "o", "hamming_char": "s", "spectrum": "D", "length": "^", "onehot_first": "v",
}


def _style(ax: Any) -> None:
    """Recessive grid and axes, ink text."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def endpoint_label(truth: Mapping[str, Any], endpoint: str) -> str:
    """`S plain`, `T1 fp8`, ... label of an endpoint."""
    info = truth[endpoint]
    return f"{info['model']} {SHORT_KIND.get(info['kind'], info['kind'])}"


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_heatmap(res: Mapping[str, Any], path: Path) -> None:
    """Prompts x endpoints: mean distance of each endpoint to the S pool (itself excluded)."""
    rows = sorted(res["per_prompt"], key=lambda r: (r["group"], r["prompt"]))
    columns = [e for e in COLUMN_ORDER if e in res["endpoints"]]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        kernel = GROUP_KERNEL_KEY if row["group"] in KEY_GROUPS else GROUP_KERNEL_TEXT
        pool = row["kernels"][kernel]["pool_distance"]
        for j, endpoint in enumerate(columns):
            value = pool.get(endpoint)
            matrix[i, j] = np.nan if value is None else value
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.2 * len(rows) + 1.2)))
    image = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([endpoint_label(res["truth"], e) for e in columns], rotation=45,
                       ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['prompt']}" for r in rows], fontsize=6, color=INK)
    previous = None
    for i, row in enumerate(rows):
        if previous is not None and row["group"] != previous:
            ax.axhline(i - 0.5, color=SURFACE, linewidth=2.0)
        previous = row["group"]
    n_base = sum(res["truth"][e]["model"] == "S" for e in columns)
    ax.axvline(n_base - 0.5, color=SURFACE, linewidth=3.0)
    for i in range(len(rows)):
        for j in range(len(columns)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=5,
                        color="#ffffff" if matrix[i, j] > 0.55 else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("mean distance to the S pool (0 = identical, 1 = never equal)", fontsize=7,
                  color=INK_SECONDARY)
    bar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    ax.set_title("Heat map: how far each endpoint sits from the base model, per prompt",
                 fontsize=10, color=INK, loc="left")
    _save(fig, path)


def figure_group_auc(res: Mapping[str, Any], path: Path, kernels: Sequence[str]) -> None:
    """AUC of (T1|T2 vs S) and (T3|T4 vs S) against S-S pairs, per prompt group and kernel."""
    groups = sorted({g["group"] for g in res["groups"]})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 0.42 * len(groups) + 1.6), sharey=True)
    for ax, role, title in zip(axes, ("hard", "easy"),
                               ("hard: fp8 and LoRA 0.3 against S", "easy: LoRA 0.7 and 1.5 against S"),
                               strict=True):
        _style(ax)
        for k_index, kernel in enumerate(kernels):
            for g_index, group in enumerate(groups):
                match = [g for g in res["groups"] if g["group"] == group and g["kernel"] == kernel]
                if not match or not np.isfinite(match[0][f"auc_{role}"]):
                    continue
                entry = match[0]
                y = g_index + (k_index - (len(kernels) - 1) / 2) * 0.16
                ax.plot([entry[f"auc_{role}_low"], entry[f"auc_{role}_high"]], [y, y],
                        color=MODEL_COLOURS["S"], linewidth=1.0, alpha=0.5)
                ax.plot(entry[f"auc_{role}"], y, KERNEL_MARKERS[kernel], color=MODEL_COLOURS["S"],
                        markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.6,
                        label=KERNEL_LABELS[kernel] if g_index == 0 else None)
        ax.axvline(0.5, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
        ax.set_xlim(0.0, 1.02)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("AUC (0.5 = chance, 1 = perfect); bars = 5-95 % over prompts", fontsize=8,
                      color=INK_SECONDARY)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(groups, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    _save(fig, path)


REPEAT_STYLES: Mapping[int, tuple[str, str]] = {
    1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 3: ("#1baf7a", "^"), 4: ("#6250d6", "D"),
}


def figure_budget_curve(res: Mapping[str, Any], path: Path, kernel: str) -> None:
    """Separation AUC against answers per endpoint, one line per number of draws per prompt.

    Rows: pair statistic (T-S pairs against S-S pairs) and pooled-reference statistic (each
    endpoint against the pooled S reference); columns: hard (fp8, LoRA 0.3) and easy (LoRA 0.7, 1.5).
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), sharey=True, sharex=True)
    for row, statistic in enumerate(("pair", "pool")):
        for col, role in enumerate(("hard", "easy")):
            ax = axes[row][col]
            _style(ax)
            for repeats, (colour, marker) in REPEAT_STYLES.items():
                points = res["repeat_vs_diversity"].get(f"{kernel}/{role}/r{repeats}/{statistic}", [])
                if not points:
                    continue
                x = [p["budget"] for p in points]
                ax.fill_between(x, [p["auc_low"] for p in points],
                                [p["auc_high"] for p in points], color=colour, alpha=0.10,
                                linewidth=0)
                ax.plot(x, [p["auc_mean"] for p in points], color=colour, linewidth=1.8,
                        label=f"{repeats} draw(s) per prompt")
                solid = [p for p in points if p["prompts"] <= 0.75 * p["pool"]]
                ax.plot([p["budget"] for p in solid], [p["auc_mean"] for p in solid], marker,
                        color=colour, markersize=4.5, markeredgecolor=SURFACE, linestyle="none")
                hollow = [p for p in points if p["prompts"] > 0.75 * p["pool"]]
                ax.plot([p["budget"] for p in hollow], [p["auc_mean"] for p in hollow], marker,
                        color=SURFACE, markersize=4.5, markeredgecolor=colour, linestyle="none")
            ax.axhline(0.5, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
            ax.set_ylim(0.3, 1.03)
            label = "pairs" if statistic == "pair" else "endpoints vs S pool"
            ax.set_title(f"{role}: {label}", fontsize=9, color=INK, loc="left")
            if row == 1:
                ax.set_xlabel("answers per endpoint (N)", fontsize=8, color=INK_SECONDARY)
            if col == 0:
                ax.set_ylabel("separation AUC", fontsize=8, color=INK_SECONDARY)
    axes[0][0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(f"Budget to quality at equal N, kernel: {KERNEL_LABELS[kernel]} "
                 "(hollow marker: resamples overlap, interval understated)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    _save(fig, path)


def figure_prompt_ranking(res: Mapping[str, Any], path: Path, kernel: str, top: int = 18) -> None:
    """Per-prompt separation score (T1|T2 pool distance minus S pool distance), best and worst."""
    rows = [(r["prompt"], r["group"], r["kernels"][kernel]["excess_hard"],
             r["kernels"][kernel]["disguise_leak"]) for r in res["per_prompt"]
            if np.isfinite(r["kernels"][kernel]["excess_hard"])]
    rows.sort(key=lambda t: t[2], reverse=True)
    shown = rows[:top] + rows[-top // 2:]
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(shown) + 1.2))
    _style(ax)
    y = np.arange(len(shown))
    ax.barh(y, [t[2] for t in shown], color=MODEL_COLOURS["T1"], height=0.6, label="T1/T2 excess")
    ax.plot([t[3] for t in shown], y, "o", color=MODEL_COLOURS["S"], markersize=4,
            markeredgecolor=SURFACE, label="largest disguise excess (S endpoint)")
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in shown], fontsize=6.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xlabel("distance to the S pool: (T1,T2) mean minus S mean  [prompt, kernel "
                  f"{KERNEL_LABELS[kernel]}]", fontsize=8, color=INK_SECONDARY)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title("Prompts ranked by raw separation (selected and shown on the same draws: "
                 "optimistic, see split-half)", fontsize=8.5, color=INK, loc="left")
    _save(fig, path)


def figure_length_signature(
    lengths: Mapping[str, Mapping[str, Sequence[int]]], truth: Mapping[str, Any], path: Path,
) -> None:
    """Box plot of reply length (words) per endpoint, one panel per family of prompts."""
    families = list(lengths)
    fig, axes = plt.subplots(1, len(families), figsize=(3.1 * len(families), 3.4), sharey=False)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families, strict=True):
        _style(ax)
        data = [list(lengths[family].get(e, [])) for e in COLUMN_ORDER]
        parts = ax.boxplot([d if d else [np.nan] for d in data], vert=False, patch_artist=True,
                           widths=0.6, showfliers=False)
        for patch, endpoint in zip(parts["boxes"], COLUMN_ORDER, strict=True):
            colour = MODEL_COLOURS[truth[endpoint]["model"]]
            patch.set(facecolor=colour, alpha=0.55, edgecolor=colour)
        for element in ("whiskers", "caps", "medians"):
            for line in parts[element]:
                line.set(color=INK_SECONDARY, linewidth=0.9)
        ax.set_yticks(range(1, len(COLUMN_ORDER) + 1))
        ax.set_yticklabels([endpoint_label(truth, e) for e in COLUMN_ORDER] if ax is axes[0]
                           else [], fontsize=7.5, color=INK)
        ax.invert_yaxis()
        ax.set_title(family, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("reply length (words)", fontsize=8, color=INK_SECONDARY)
    _save(fig, path)


def figure_adaptive(steps: Sequence[Mapping[str, float]], summary: Mapping[str, float],
                    path: Path) -> None:
    """Uniform-spend error and resolved fraction against prompts per endpoint, adaptive point."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    x = [s["prompts"] for s in steps]
    _style(axes[0])
    axes[0].plot(x, [s["pair_error_uniform"] for s in steps], "-o", color=MODEL_COLOURS["S"],
                 markersize=4, label="all pairs, uniform spend")
    axes[0].plot(x, [s["hard_pair_error_uniform"] for s in steps], "-s", color=MODEL_COLOURS["T1"],
                 markersize=4, label="hard + same-model pairs, uniform spend")
    axes[0].plot([summary["adaptive_mean_spend"]], [summary["adaptive_hard_pair_error"]], "*",
                 color=INK, markersize=12, label="adaptive rule (hard + same pairs)")
    axes[0].set_xlabel("prompts per endpoint (1 draw each)", fontsize=8, color=INK_SECONDARY)
    axes[0].set_ylabel("pair classification error", fontsize=8, color=INK_SECONDARY)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("error against spend", fontsize=9, color=INK, loc="left")
    _style(axes[1])
    axes[1].plot(x, [s["resolved_fraction"] for s in steps], "-o", color=MODEL_COLOURS["T2"],
                 markersize=4)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("prompts per endpoint", fontsize=8, color=INK_SECONDARY)
    axes[1].set_ylabel("fraction of pairs resolved (|z| >= 2)", fontsize=8, color=INK_SECONDARY)
    axes[1].set_title("what an adaptive rule can stop early", fontsize=9, color=INK, loc="left")
    _save(fig, path)


DIVERGING = LinearSegmentedColormap.from_list("blue_orange", ["#2a78d6", "#ecebe7", "#eb6834"])
T_CLIP = 6.0
"""t statistics are clipped to +-6 in the colour scale (the number stays printed)."""


def figure_style_t(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], truth: Mapping[str, Any], path: Path,
) -> None:
    """Style-feature t statistics of each endpoint against the S pool, one panel per subset."""
    names = list(tables)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names, strict=True):
        rows = tables[name]
        matrix = np.array([[row["t"][e] for e in COLUMN_ORDER] for row in rows], dtype=float)
        ax.imshow(np.clip(matrix, -T_CLIP, T_CLIP), cmap=DIVERGING, vmin=-T_CLIP, vmax=T_CLIP,
                  aspect="auto")
        ax.set_xticks(range(len(COLUMN_ORDER)))
        ax.set_xticklabels([endpoint_label(truth, e) for e in COLUMN_ORDER], rotation=60,
                           ha="right", fontsize=7, color=INK)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["feature"] for row in rows], fontsize=7.5, color=INK)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6,
                            color=INK)
        ax.set_title(f"{name} ({rows[0]['prompts']} prompts)", fontsize=9, color=INK, loc="left")
    _save(fig, path)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion (well behaved at 0 and 1, unlike the normal one)."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(centre - half), float(centre + half)


def figure_adherence(
    rates: Mapping[str, Mapping[str, Sequence[int]]], truth: Mapping[str, Any], path: Path,
    xlabel: str = "share of replies of at most 2 words",
) -> None:
    """Share of short replies (<= 2 words) to constrained prompts, per endpoint, with 95 % Wilson."""
    families = list(rates)
    fig, axes = plt.subplots(1, len(families), figsize=(3.4 * len(families), 3.4), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families, strict=True):
        _style(ax)
        for index, endpoint in enumerate(COLUMN_ORDER):
            successes, total = rates[family].get(endpoint, (0, 0))
            if total == 0:
                continue
            low, high = wilson_interval(successes, total)
            colour = MODEL_COLOURS[truth[endpoint]["model"]]
            ax.plot([low, high], [index, index], color=colour, linewidth=1.6)
            ax.plot(successes / total, index, "o", color=colour, markersize=6,
                    markeredgecolor=SURFACE)
            ax.text(1.03, index, f"{successes}/{total}", fontsize=6.5, va="center",
                    color=INK_SECONDARY)
        ax.set_xlim(0, 1.0)
        if ax is axes[0]:
            ax.set_yticks(range(len(COLUMN_ORDER)))
            ax.set_yticklabels([endpoint_label(truth, e) for e in COLUMN_ORDER], fontsize=7.5,
                               color=INK)
            ax.invert_yaxis()
        ax.set_title(family, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=INK_SECONDARY)
    _save(fig, path)
