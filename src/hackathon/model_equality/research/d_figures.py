"""Figures of the team-D reports (matplotlib, same palette as the team-B figures).

Model colours are shared with team B (S blue, T1 orange, T2 aqua, T3 violet, T4 red; aqua is
below 3:1 on the surface, so every aqua mark carries a label). Methods use neutral hues that never
collide with a model colour, and keep their colour from one figure to the next.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
DPI = 130
LABEL_SIZE = 11
TICK_SIZE = 10
MODEL_COLOURS: Mapping[str, str] = {
    "S": "#2a78d6", "T1": "#eb6834", "T2": "#1baf7a", "T3": "#6250d6", "T4": "#e34948",
}
METHOD_COLOURS: Mapping[str, str] = {
    "length": "#7f7f7f", "ends_period": "#a3a000", "sentences": "#8c564b", "style22": "#c0558f",
    "transfer": "#222222", "oracle_fp8": "#222222", "oracle_lora03": "#222222",
    "selected": "#b8860b", "held_out": "#222222",
}
METHOD_LABELS: Mapping[str, str] = {
    "length": "length (log characters)", "ends_period": "reply ends with a period",
    "sentences": "sentence count", "style22": "22 style features, L2",
    "transfer": "direction learned on LoRA 0.7 and 1.5 (usable at ep)",
    "oracle_fp8": "oracle direction trained on fp8 labels (not usable)",
    "oracle_lora03": "oracle direction trained on LoRA 0.3 labels (not usable)",
}
KIND_SHORT: Mapping[str, str] = {
    "plain": "plain", "system prompt": "system prompt", "persona": "persona",
    "watermark": "watermark", "sampling at temperature 1.0": "temp 1.0",
    "quantized (fp8)": "fp8", "LoRA at strength 0.3": "LoRA 0.3",
    "LoRA at strength 0.7": "LoRA 0.7", "LoRA at strength 1.5": "LoRA 1.5",
}
ENDPOINT_ORDER: tuple[str, ...] = (
    "ref-01", "ref-03", "ref-05", "ref-06", "ref-08", "ref-02", "ref-09", "ref-04", "ref-07")
CHANCE_AUC = 0.5
LATENCY_SECONDS: Mapping[int, tuple[float, ...]] = {
    1: (9.3, 8.1), 8: (9.2, 10.6, 20.5), 64: (11.9, 11.6), 256: (21.7, 22.6, 21.5)}
"""Sequential latencies measured on 21/11/2026 at 19h20 with a free team slot, by max_tokens (the
first request of each series, queued behind other teams for 148 s or 31 s, is left out)."""


def _style(ax: Any) -> None:
    """Recessive grid, ink text, no top/right spine."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=TICK_SIZE)


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def endpoint_label(truth: Mapping[str, Any], endpoint: str) -> str:
    """`ref-02 T1 fp8` style label."""
    info = truth[endpoint]
    return f"{endpoint[-2:]} {info['model']} {KIND_SHORT.get(info['kind'], info['kind'])}"


def _curve_point(curve: Mapping[str, Any], n: int, role: str) -> tuple[float, float, float]:
    """(mean, low, high) of an AUC curve entry."""
    entry = curve[str(n)][role]
    return float(entry["mean"]), float(entry["low"]), float(entry["high"])


def figure_dashboard_bars(metrics: Mapping[str, Any], path: Path, n: int = 40) -> None:
    """Page-1 bars: AUC at N = `n` for fp8 and for LoRA 0.3, one pair of bars per statistic."""
    curves = metrics["discovery"]["auc_curves"]
    keys = ("length", "ends_period", "sentences", "style22", "transfer", "oracle_lora03")
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    _style(ax)
    y = np.arange(len(keys))[::-1].astype(float)
    height = 0.34
    targets = (("fp8", "T1", "fp8 vs S", height / 2 + 0.01), ("lora_weak", "T2", "LoRA 0.3 vs S", -height / 2 - 0.01))
    for role, model, label, shift in targets:
        for yi, key in zip(y, keys, strict=True):
            source = "oracle_fp8" if (role == "fp8" and key == "oracle_lora03") else key
            mean, low, high = _curve_point(curves[source], n, role)
            hatch = "//" if source.startswith("oracle") else None
            ax.barh(yi + shift, mean - CHANCE_AUC, left=CHANCE_AUC, height=height, color=MODEL_COLOURS[model],
                    hatch=hatch, edgecolor=SURFACE, label=label if yi == y[0] else None)
            ax.plot([low, high], [yi + shift, yi + shift], color=INK, linewidth=1.1)
            ax.text(max(high, mean) + 0.012, yi + shift, f"{mean:.2f}", va="center", fontsize=LABEL_SIZE - 1,
                    color=INK)
    ax.axvline(CHANCE_AUC, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    labels = [METHOD_LABELS[k].replace("oracle direction trained on LoRA 0.3 labels", "oracle (labels of the target)")
              for k in keys]
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlim(0.2, 1.08)
    ax.set_xlabel(f"AUC against the 10 S-S pairs, N = {n} answers per endpoint (0.5 = chance; bar = 2.5 - 97.5 %)",
                  fontsize=LABEL_SIZE - 1)
    ax.legend(fontsize=LABEL_SIZE - 1, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ends = {r: _curve_point(curves["ends_period"], n, r)[0] for r in ("fp8", "lora_weak")}
    ax.set_title(f"N = {n}: fp8 stays at chance ({ends['fp8']:.2f}); LoRA 0.3 reaches {ends['lora_weak']:.2f} "
                 "on one feature (in-sample)", fontsize=12, loc="left")
    _save(fig, path)


def figure_auc_vs_n(metrics: Mapping[str, Any], path: Path) -> None:
    """AUC vs N (log axis) with replicate intervals, one panel per target."""
    curves = metrics["discovery"]["auc_curves"]
    sizes = metrics["sample_sizes"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    panels = (("fp8 (T1) vs S", "fp8", ("length", "ends_period", "sentences", "style22", "transfer", "oracle_fp8")),
              ("LoRA 0.3 (T2) vs S", "lora_weak",
               ("length", "ends_period", "sentences", "style22", "transfer", "oracle_lora03")))
    for ax, (title, role, keys) in zip(axes, panels, strict=True):
        _style(ax)
        for key in keys:
            points = [_curve_point(curves[key], n, role) for n in sizes]
            mean = np.asarray([p[0] for p in points])
            low = np.asarray([p[1] for p in points])
            high = np.asarray([p[2] for p in points])
            style = "--" if key.startswith("oracle") else "-"
            ax.plot(sizes, mean, style, color=METHOD_COLOURS[key], linewidth=2.0, marker="o",
                    markersize=4, label=METHOD_LABELS[key])
            ax.fill_between(sizes, low, high, color=METHOD_COLOURS[key], alpha=0.10)
        ax.axhline(CHANCE_AUC, color=INK_SECONDARY, linewidth=1.0, linestyle=":")
        ax.axvline(100, color=INK_SECONDARY, linewidth=1.0, linestyle="-.")
        ax.text(101, 0.12, "ep budget", rotation=90, fontsize=9, color=INK_SECONDARY)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_xlabel("answers per endpoint N (log scale)", fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=12, loc="left")
    axes[0].set_ylabel("AUC (shaded: 2.5 - 97.5 % over draws)", fontsize=LABEL_SIZE)
    axes[0].set_ylim(0.1, 1.0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    best_n = {r: [_curve_point(curves["ends_period"], n, r)[0] for n in sizes] for r in ("fp8", "lora_weak")}
    fig.suptitle(
        f"fp8 stays at chance for every N (max {max(best_n['fp8']):.2f}); LoRA 0.3 rises with N for the period "
        f"feature only ({best_n['lora_weak'][sizes.index(40)]:.2f} at N = 40)", fontsize=12, x=0.01, ha="left")
    _save(fig, path)


def figure_dose_response(metrics: Mapping[str, Any], path: Path) -> None:
    """Mean transfer score per endpoint: the LoRA direction grows with strength, T2 and fp8 sit with S."""
    truth = metrics["discovery"]["truth"]
    scores = metrics["discovery"]["transfer_mean_score"]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    _style(ax)
    x = np.arange(len(ENDPOINT_ORDER))
    for xi, endpoint in zip(x, ENDPOINT_ORDER, strict=True):
        entry = scores[endpoint]
        colour = MODEL_COLOURS[truth[endpoint]["model"]]
        ax.bar(xi, entry["mean"], color=colour, width=0.66)
        ax.plot([xi, xi], [entry["low"], entry["high"]], color=INK, linewidth=1.3)
        ax.text(float(xi), entry["high"] + 0.04, f"{entry['mean']:.2f}", ha="center", fontsize=LABEL_SIZE - 1)
    ax.set_xticks(x)
    ax.set_xticklabels([endpoint_label(truth, e) for e in ENDPOINT_ORDER], rotation=30, ha="right",
                       fontsize=TICK_SIZE)
    ax.set_ylabel("mean margin of the LoRA direction", fontsize=LABEL_SIZE)
    ax.set_title("The LoRA direction grows with strength (0.7 to 1.5) but LoRA 0.3 and fp8 sit inside the S spread",
                 fontsize=12, loc="left")
    _save(fig, path)


def figure_effect_heatmap(metrics: Mapping[str, Any], path: Path) -> None:
    """t of every endpoint against the other S endpoints for every scalar feature."""
    truth = metrics["discovery"]["truth"]
    effect = metrics["discovery"]["effect_t"]
    features = list(next(iter(effect.values())))
    matrix = np.asarray([[effect[e][f] for f in features] for e in ENDPOINT_ORDER])
    limit = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=60, ha="right", fontsize=TICK_SIZE)
    ax.set_yticks(range(len(ENDPOINT_ORDER)))
    ax.set_yticklabels([endpoint_label(truth, e) for e in ENDPOINT_ORDER], fontsize=TICK_SIZE)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if abs(matrix[i, j]) >= 3.0:
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(image, ax=ax, label="paired t vs the other S endpoints")
    strong = np.abs(matrix) >= 3.0
    rows = {e: int(strong[i].sum()) for i, e in enumerate(ENDPOINT_ORDER)}
    disguise_cells = sum(v for e, v in rows.items() if truth[e]["model"] == "S")
    ax.set_title(f"Cells with |t| >= 3: the 5 S endpoints {disguise_cells} (of 110), fp8 {rows['ref-02']}, "
                 f"LoRA 0.3 {rows['ref-09']}, LoRA 0.7 {rows['ref-04']}, LoRA 1.5 {rows['ref-07']}",
                 fontsize=12, loc="left")
    _save(fig, path)


def figure_selection(metrics: Mapping[str, Any], path: Path) -> None:
    """Optimistic vs honest AUC of the screened winner, per random split of the prompts."""
    selection = metrics["discovery"].get("selection")
    if not selection or not selection["held_out_auc"]:
        return
    sel = np.asarray(selection["selected_auc"])
    held = np.asarray(selection["held_out_auc"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _style(ax)
    rng = np.random.default_rng(0)
    ax.scatter(np.zeros_like(sel) + rng.normal(0, 0.04, len(sel)), sel, color=METHOD_COLOURS["selected"],
               s=36, label="chosen on half 1 (optimistic)")
    ax.scatter(np.ones_like(held) + rng.normal(0, 0.04, len(held)), held, color=METHOD_COLOURS["held_out"],
               s=36, label="same winner on half 2 (honest)")
    ax.axhline(selection["mean_candidate_auc"], color=INK_SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(1.35, selection["mean_candidate_auc"] + 0.005, "mean of all candidates", fontsize=9,
            color=INK_SECONDARY, ha="right")
    ax.axhline(CHANCE_AUC, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["screening half", "unseen half"], fontsize=LABEL_SIZE)
    ax.set_ylabel("AUC (N = 20, fp8 and LoRA 0.3 vs S)", fontsize=LABEL_SIZE)
    ax.set_xlim(-0.5, 1.5)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_title(
        f"A screened best AUC of {sel.mean():.2f} falls to {held.mean():.2f} on unseen prompts",
        fontsize=12, loc="left")
    _save(fig, path)


def figure_t_growth(metrics: Mapping[str, Any], path: Path) -> None:
    """t vs N per endpoint with the square-root fit extrapolated (an estimate beyond N = 120)."""
    truth = metrics["discovery"]["truth"]
    growth = metrics["discovery"]["t_growth"]
    names = ("transfer", "ends_period")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    grid = np.linspace(5, 400, 80)
    for ax, name in zip(axes, names, strict=True):
        _style(ax)
        for endpoint in ENDPOINT_ORDER:
            model = truth[endpoint]["model"]
            if model in ("T3", "T4"):
                continue
            row = growth[name][endpoint]
            style = "-" if model in ("T1", "T2") else ":"
            ax.plot(grid, row["slope"] * np.sqrt(grid), style, color=MODEL_COLOURS[model],
                    linewidth=2.2 if model != "S" else 1.3, alpha=0.95 if model != "S" else 0.7)
            sizes = [int(k.split("_")[-1]) for k in row if k.startswith("t_at_")]
            ax.plot(sizes, [row[f"t_at_{n}"] for n in sizes], "o", color=MODEL_COLOURS[model], markersize=4)
            if model in ("T1", "T2"):
                ax.text(402, row["slope"] * math.sqrt(400), endpoint_label(truth, endpoint), fontsize=9,
                        color=INK, va="center")
        ax.axhline(3.0, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
        ax.axhline(-3.0, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
        ax.axvline(100, color=INK_SECONDARY, linewidth=1.0, linestyle="-.")
        ax.set_xlim(0, 520)
        ax.set_xlabel("answers per endpoint N (dots: measured; lines: sqrt(N) fit)", fontsize=LABEL_SIZE - 1)
        ax.set_title({"transfer": "LoRA direction score", "ends_period": "reply ends with a period"}[name],
                     fontsize=12, loc="left")
    axes[0].set_ylabel("t vs the other S endpoints (blue dotted = S disguises)", fontsize=LABEL_SIZE - 1)
    period = growth["ends_period"]
    lora_t = period["ref-09"]["t_at_120"]
    peak = max(period[e]["t_at_120"] for e in ENDPOINT_ORDER if truth[e]["model"] == "S")
    fig.suptitle(f"At N = 120 the period feature gives LoRA 0.3 t = {lora_t:.1f} against {peak:.1f} for the best "
                 f"disguise and {period['ref-02']['t_at_120']:.1f} for fp8", fontsize=12, x=0.01, ha="left")
    _save(fig, path)


def figure_effect_sizes(metrics: Mapping[str, Any], path: Path) -> None:
    """Paired effect size d per endpoint, with its bootstrap interval, for the usable direction."""
    truth = metrics["discovery"]["truth"]
    cohen = metrics["discovery"]["cohen_d"]["transfer"]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    _style(ax)
    x = np.arange(len(ENDPOINT_ORDER))
    for xi, endpoint in zip(x, ENDPOINT_ORDER, strict=True):
        entry = cohen[endpoint]
        ax.bar(xi, entry["d"], color=MODEL_COLOURS[truth[endpoint]["model"]], width=0.66)
        ax.plot([xi, xi], [entry["low"], entry["high"]], color=INK, linewidth=1.3)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([endpoint_label(truth, e) for e in ENDPOINT_ORDER], rotation=30, ha="right",
                       fontsize=TICK_SIZE)
    ax.set_ylabel("paired effect size d (per answer)", fontsize=LABEL_SIZE)
    s_values = [cohen[e]["d"] for e in ENDPOINT_ORDER if truth[e]["model"] == "S"]
    ax.set_title(
        f"LoRA direction, per-answer effect d: disguises {min(s_values):.2f} to {max(s_values):.2f}, fp8 "
        f"{cohen['ref-02']['d']:.2f}, LoRA 0.3 {cohen['ref-09']['d']:.2f}, LoRA 0.7 {cohen['ref-04']['d']:.2f}, "
        f"LoRA 1.5 {cohen['ref-07']['d']:.2f}", fontsize=11, loc="left")
    _save(fig, path)


def figure_exact(metrics: Mapping[str, Any], path: Path) -> None:
    """Mismatch matrix on low-entropy cells and agreement of every endpoint with the consensus."""
    exact = metrics["discovery"]["exact"]
    truth = metrics["discovery"]["truth"]
    endpoints = exact["endpoints"]
    order = [e for e in ENDPOINT_ORDER if e in endpoints]
    idx = [endpoints.index(e) for e in order]
    matrix = np.asarray(exact["mismatch"])[np.ix_(idx, idx)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.95})
    image = left.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [endpoint_label(truth, e) for e in order]
    left.set_xticks(range(len(order)))
    left.set_xticklabels(labels, rotation=60, ha="right", fontsize=TICK_SIZE - 1)
    left.set_yticks(range(len(order)))
    left.set_yticklabels(labels, fontsize=TICK_SIZE - 1)
    for i in range(len(order)):
        for j in range(len(order)):
            left.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if matrix[i, j] > 0.55 else INK)
    fig.colorbar(image, ax=left, fraction=0.046, pad=0.03, label="exact mismatch rate")
    left.set_title(f"Mismatch between endpoints, {exact['matrix_cells']} low-entropy cells", fontsize=12, loc="left")
    _style(right)
    agree = [exact["mode_agreement"][e] for e in order]
    right.barh(range(len(order))[::-1], agree, color=[MODEL_COLOURS[truth[e]["model"]] for e in order], height=0.66)
    right.set_yticks(range(len(order))[::-1])
    right.set_yticklabels(labels, fontsize=TICK_SIZE - 1)
    for yi, value in zip(range(len(order))[::-1], agree, strict=True):
        right.text(value + 0.01, yi, f"{value:.2f}", va="center", fontsize=LABEL_SIZE - 1)
    right.set_xlim(0, 1.1)
    right.set_xlabel("share of cells where the endpoint gives the consensus answer", fontsize=LABEL_SIZE - 1)
    right.set_title("fp8 and LoRA 0.3 look like S plain; the disguises deviate more", fontsize=12, loc="left")
    _save(fig, path)


FAMILY_COLOURS: Mapping[str, str] = {"Qwen": "#8c564b", "Llama": "#bcbd22", "?": "#ffffff"}


def figure_ep_structure(structure: Mapping[str, Any], path: Path) -> None:
    """Exact-mismatch heat map of the 33 ep endpoints (5 prompts) and prefix distance on the completion prompt."""
    endpoints = structure["endpoints"]
    hints = structure["family_hints"]
    groups = structure["clusters_at_cut"]
    mismatch = np.asarray(structure["mismatch"])
    lcp = np.asarray(structure["lcp_distance"]) if structure["lcp_distance"] else None
    panels = [("exact mismatch over 5 prompts", mismatch)]
    if lcp is not None:
        panels.append((f"prefix distance on {structure['lcp_prompt']}", lcp))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 6.2), squeeze=False)
    labels = [e[-2:] for e in endpoints]
    cluster_of = {e: index for index, group in enumerate(groups) for e in group}
    boundaries = [
        i + 0.5 for i in range(len(endpoints) - 1) if cluster_of[endpoints[i]] != cluster_of[endpoints[i + 1]]]
    for ax, (title, matrix) in zip(axes[0], panels, strict=True):
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for tick, endpoint in zip(ax.get_yticklabels(), endpoints, strict=True):
            tick.set_bbox({"facecolor": FAMILY_COLOURS[hints.get(endpoint, "?")], "edgecolor": GRID, "pad": 1.5})
        for b in boundaries:
            ax.axhline(b, color=INK, linewidth=0.8)
            ax.axvline(b, color=INK, linewidth=0.8)
        ax.set_title(title, fontsize=12, loc="left")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    sizes = [len(g) for g in groups]
    fig.suptitle(f"ep, first phase (5 answers per endpoint): clusters of sizes {sizes}; "
                 "label colour = family the endpoint names (brown Qwen, olive Llama, white unclear)",
                 fontsize=11, x=0.01, ha="left")
    _save(fig, path)


def figure_latency(path: Path) -> None:
    """Measured server latency by generation cap."""
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    _style(ax)
    caps = sorted(LATENCY_SECONDS)
    means = [float(np.mean(LATENCY_SECONDS[c])) for c in caps]
    ax.bar(range(len(caps)), means, color="#7f7f7f", width=0.6)
    for xi, c in enumerate(caps):
        ax.scatter([xi] * len(LATENCY_SECONDS[c]), LATENCY_SECONDS[c], color=INK, s=14, zorder=3)
        ax.text(xi, means[xi] + 1.0, f"{means[xi]:.0f} s", ha="center", fontsize=LABEL_SIZE)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([str(c) for c in caps], fontsize=LABEL_SIZE)
    ax.set_xlabel("max_tokens", fontsize=LABEL_SIZE)
    ax.set_ylabel("latency of one request (s)", fontsize=LABEL_SIZE)
    ax.set_title("About 8 s fixed + 0.05 s per token", fontsize=12, loc="left")
    _save(fig, path)


def figure_fresh(metrics: Mapping[str, Any], path: Path) -> bool:
    """Fresh-prompt validation: mean transfer score per endpoint. Returns False when there is no data."""
    groups = metrics["fresh"]["groups"]
    usable = {g: v for g, v in groups.items() if v.get("cells", 0) >= 2}
    if not usable:
        return False
    truth = metrics["discovery"]["truth"]
    endpoints = [e for e in ENDPOINT_ORDER if e in metrics["fresh"]["endpoints"]]
    fig, axes = plt.subplots(1, len(usable), figsize=(5.6 * len(usable), 4.0), squeeze=False)
    for ax, (group, info) in zip(axes[0], usable.items(), strict=True):
        _style(ax)
        values = [info["transfer_mean"][e] for e in endpoints]
        ax.bar(range(len(endpoints)), values, color=[MODEL_COLOURS[truth[e]["model"]] for e in endpoints], width=0.66)
        ax.set_xticks(range(len(endpoints)))
        ax.set_xticklabels([endpoint_label(truth, e) for e in endpoints], rotation=35, ha="right", fontsize=TICK_SIZE)
        ax.set_title(f"{group}: {info['cells']} fresh cells", fontsize=12, loc="left")
    axes[0][0].set_ylabel("mean LoRA-direction margin", fontsize=LABEL_SIZE)
    _save(fig, path)
    return True


def make_all_figures(metrics: Mapping[str, Any], out_dir: Path) -> list[str]:
    """Write every figure; returns the file names produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    jobs: Sequence[tuple[str, Any]] = (
        ("dashboard_bars.png", figure_dashboard_bars), ("auc_vs_n.png", figure_auc_vs_n),
        ("dose_response.png", figure_dose_response), ("effect_heatmap.png", figure_effect_heatmap),
        ("selection.png", figure_selection), ("t_growth.png", figure_t_growth),
        ("effect_sizes.png", figure_effect_sizes), ("exact.png", figure_exact),
    )
    for name, function in jobs:
        function(metrics, out_dir / name)
        if (out_dir / name).exists():
            produced.append(name)
    figure_latency(out_dir / "latency.png")
    produced.append("latency.png")
    if "ep_structure" in metrics:
        figure_ep_structure(metrics["ep_structure"], out_dir / "ep_structure.png")
        produced.append("ep_structure.png")
    if figure_fresh(metrics, out_dir / "fresh.png"):
        produced.append("fresh.png")
    return produced
