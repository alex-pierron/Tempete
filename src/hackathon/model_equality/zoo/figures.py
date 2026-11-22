"""Figures of the zoo report (matplotlib, static PNG for the HTML and the PDF).

Palette: the colour-blind-safe set used by the other team reports; the four family colours were run
through `validate_palette.js` (light surface: lightness and chroma pass, normal-vision floor 24,
deutan floor 7.6 which is legal with the direct labels and the tables that accompany every figure;
orange and pink are under 3:1 against the surface, hence labels in ink, never in the series colour).
One axis per chart, thin marks, no dual axis, values in text colour.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from hackathon.model_equality.zoo.config import FAMILY_BY_KEY, VARIANTS

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FAMILY_COLOR: Mapping[str, str] = {
    "qwen15": "#0072B2", "qwen3": "#E69F00", "llama1": "#009E73", "smollm": "#CC79A7",
}
THRESHOLD_COLOR = "#D55E00"
FAMILY_BY_KEY_DISPLAY: Mapping[str, str] = {k: f.display for k, f in FAMILY_BY_KEY.items()}
SEQUENTIAL = LinearSegmentedColormap.from_list("zoo_blue", ["#f4f8fb", "#9cc3de", "#0072B2",
                                                            "#003a5c"])
FIGURE_DPI = 170
TEXT_ON_DARK_FROM = 0.55
SCORE_THRESHOLD = 0.40
DECISION_SCORE = 0.60
RULE_LABEL: Mapping[str, str] = {
    "v2": "v2 point final", "v3": "v3 identite (lexique fige)", "v3ext": "v3, lexique + Mistral",
    "v4": "v4 bande relative", "v4ext": "v4, lexique + Mistral",
    "lik_plain": "vraisemblance, ref = plain", "lik_median": "vraisemblance, ref = mediane",
    "lik_median40": "vraisemblance 40 reponses", "v3ext+lik_median": "identite + vraisemblance",
    "v3ext+lik_median40": "identite + vraisemblance 40", "oracle": "plafond : deguisement connu",
}
VARIANT_ORDER: tuple[str, ...] = tuple(v.key for v in VARIANTS)
VARIANT_LABEL: Mapping[str, str] = {
    "plain": "plain", "fp8_w": "fp8 (poids)", "sysprompt": "system prompt", "lora07": "LoRA 0,7",
    "persona": "persona", "watermark": "watermark", "lora15": "LoRA 1,5",
    "temp1": "temperature 1,0", "lora03": "LoRA 0,3", "fp8_w8a8": "fp8 W8A8",
}
FAMILY_SHORT: Mapping[str, str] = {
    "qwen15": "Qwen 1,5B", "qwen3": "Qwen 3B", "llama1": "Llama 1B", "smollm": "SmolLM 360M (ctrl)",
}
SCOPE_LABEL = {"oracle_families": "4 familles, familles connues",
               "likelihood_families": "4 familles, familles par vraisemblance",
               "oracle_families_w8a8": "4 familles, fp8 = W8A8"}
IMAGE_WIDTH_PERCENT: Mapping[str, int] = {"heatmap": 62, "dots": 100, "shift": 80}


def to_base64(figure: Figure) -> str:
    """PNG of a figure as base64 (for a `data:` URI), then close it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _style(axis: Any) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)


def heatmap(
    values: np.ndarray, rows: Sequence[str], columns: Sequence[str], title: str,
    percent: bool = True, vmax: float = 1.0, axis: Any = None,
) -> Figure | None:
    """Annotated heatmap, one hue, values printed in the cells (text colour switches on dark).

    Draws on `axis` when given (returns None), else on a new figure (returned).
    """
    own = axis is None
    figure = None
    if own:
        figure, axis = plt.subplots(figsize=(1.0 + 1.15 * len(columns), 0.9 + 0.36 * len(rows)))
    axis.imshow(values, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax, aspect="auto")
    axis.set_xticks(range(len(columns)), columns, fontsize=8, color=INK)
    axis.set_yticks(range(len(rows)), rows, fontsize=8, color=INK)
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    for r in range(len(rows)):
        for c in range(len(columns)):
            value = values[r, c]
            if np.isnan(value):
                continue
            label = f"{100 * value:.0f}" if percent else f"{value:.2f}"
            colour = "#fcfcfb" if value / vmax > TEXT_ON_DARK_FROM else INK
            axis.text(c, r, label, ha="center", va="center", fontsize=8, color=colour)
    for edge in range(len(columns) + 1):
        axis.axvline(edge - 0.5, color=SURFACE, linewidth=2)
    for edge in range(len(rows) + 1):
        axis.axhline(edge - 0.5, color=SURFACE, linewidth=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, fontsize=9, color=INK, loc="left", pad=22)
    return figure


def score_dots(
    cells: Sequence[Mapping[str, Any]], scopes: Sequence[str], rules: Sequence[str], n: int,
    ceilings: Mapping[str, Mapping[str, float]], titles: Mapping[str, str],
) -> Figure:
    """Median and 10-90 % interval of the official score, one panel per scope, one row per rule."""
    columns = 2 if len(scopes) > 3 else len(scopes)
    rows_count = int(np.ceil(len(scopes) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(4.8 * columns, 0.27 * len(rules) *
                                                              rows_count + 0.9 * rows_count),
                                sharex=True, squeeze=False)
    for index, scope in enumerate(scopes):
        axis = axes[index // columns][index % columns]
        _style(axis)
        for position, rule in enumerate(rules):
            cell = next(c for c in cells if c["scope"] == scope and c["rule"] == rule
                        and c["n"] == n)
            y = len(rules) - 1 - position
            colour = INK if rule == "oracle" else FAMILY_COLOR.get(scope.split(":")[-1], "#0072B2")
            axis.plot([cell["p10"], cell["p90"]], [y, y], color=colour, linewidth=1.4,
                      solid_capstyle="round")
            axis.plot(cell["median"], y, "o", color=colour, markersize=5.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2)
            axis.text(min(cell["p90"] + 0.02, 0.98), y, f"{cell['median']:.2f}".replace(".", ","),
                      fontsize=7.5, color=INK, va="center")
        axis.axvline(SCORE_THRESHOLD, color=THRESHOLD_COLOR, linewidth=0.9, linestyle=(0, (4, 3)))
        axis.axvline(DECISION_SCORE, color=THRESHOLD_COLOR, linewidth=1.3)
        lumped = ceilings.get(scope, {}).get("fp8_lumped_with_S")
        if lumped is not None:
            axis.axvline(lumped, color=MUTED, linewidth=0.9, linestyle=":")
        axis.set_yticks(range(len(rules)), [RULE_LABEL[r] for r in reversed(rules)], fontsize=7.5,
                        color=INK)
        axis.set_xlim(0, 1.05)
        axis.set_title(titles.get(scope, scope), fontsize=8.5, color=INK, loc="left")
        axis.grid(axis="x", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    for index in range(len(scopes), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    figure.supxlabel("score officiel (mediane, intervalle 10-90 %) ; trait plein = 0,60 (decision), "
                     "tirets = 0,40, pointille = plafond fp8 fondu dans S", fontsize=7.5,
                     color=MUTED)
    figure.tight_layout()
    return figure


def likelihood_shift(
    variants: Mapping[str, Mapping[str, Any]], families: Sequence[str]
) -> Figure:
    """Mean per-token log-likelihood of each variant minus the plain one, per family (nats)."""
    figure, axis = plt.subplots(figsize=(6.4, 0.42 * len(VARIANT_ORDER) + 1.1))
    _style(axis)
    for f_index, family in enumerate(families):
        plain = variants[f"zoo-{family}-01"]["mean_ll_per_token"]
        for position, key in enumerate(VARIANT_ORDER):
            slot = next(v.slot for v in VARIANTS if v.key == key)
            entry = variants.get(f"zoo-{family}-{slot:02d}")
            if entry is None or entry["mean_ll_per_token"] is None or plain is None:
                continue
            shift = float(entry["mean_ll_per_token"]) - float(plain)
            y = len(VARIANT_ORDER) - 1 - position + (f_index - 1.5) * 0.14
            axis.plot(shift, y, "o", color=FAMILY_COLOR[family], markersize=5.5,
                      markeredgecolor=SURFACE, markeredgewidth=1.2,
                      label=FAMILY_BY_KEY[family].display.split(" (")[0] if position == 0 else None)
    axis.axvline(0.0, color=MUTED, linewidth=0.8)
    axis.set_yticks(range(len(VARIANT_ORDER)), [VARIANT_LABEL[k] for k in reversed(VARIANT_ORDER)],
                    fontsize=8, color=INK)
    axis.set_xlabel("vraisemblance moyenne par token sous le modele de base, moins celle de plain "
                    "(nats)", fontsize=8, color=MUTED)
    axis.grid(axis="x", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(fontsize=7.5, frameon=False, loc="lower left", labelcolor=INK)
    figure.tight_layout()
    return figure
