"""Figures of the m-met-1 v2 report, to the team's visual standard.

Every figure has a title that states its conclusion, named axes, fonts of at least 11 pt, a
colour-blind-safe palette and the same colour for the same entity from one figure to the next:

* methods: `pair` (ham+len clustering, v1) violet, `lenz+cluster` (v2) green, `ramble` amber,
  `ham` pink;
* families: A blue, B orange, unknown grey;
* us blue on the leaderboard, other teams grey.

Every function writes one PNG and returns its path.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram

from hackathon.model_equality.smalln import build_tree

FloatMatrix = npt.NDArray[np.float64]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
GREY = "#898781"
LIGHT_GREY = "#c3c2b7"
US_BLUE = "#2a78d6"
FAMILY_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#6250d6"}
UNKNOWN_COLOR = GREY
METHOD_COLORS = {"pair": "#6250d6", "lenz+cluster": "#1baf7a", "lenz": "#1baf7a", "lenzr": "#1baf7a",
                 "ramble": "#eda100", "ham": "#e87ba4", "ham+len": "#6250d6"}
METHOD_LABELS = {"pair": "v1 : regroupement ham+len", "lenz+cluster": "v2 : longueur relative + regroupement",
                 "lenz": "longueur relative", "lenzr": "v2 : longueur relative (z robuste)", "ramble": "taux de bavardage", "ham": "Hamming seul"}
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#e8f0fb", "#2a78d6", "#0b2f5e"])
DPI = 150
FONT = 11
TITLE_FONT = 12.5

plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": TITLE_FONT, "axes.labelsize": FONT,
    "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_SECONDARY, "text.color": INK, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
})


def _clean(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def leaderboard_figure(
    top: Sequence[tuple[str, float]], ours: float, rank: str, path: Path,
) -> Path:
    """Bars: all-zero baseline, the best teams, and us."""
    entries = [("tout a zero", 0.0, LIGHT_GREY), *[(n, s, GREY) for n, s in top],
               (f"nous ({rank})", ours, US_BLUE)]
    entries.sort(key=lambda e: e[1])
    figure, ax = plt.subplots(figsize=(8.5, 4.0))
    _clean(ax)
    ax.barh([e[0] for e in entries], [e[1] for e in entries], color=[e[2] for e in entries])
    for position, entry in enumerate(entries):
        ax.text(entry[1] + 0.005, position, f"{entry[1]:.2f}".replace(".", ","), va="center",
                fontsize=FONT, color=INK)
    ax.set_xlabel("score public Model Equality (0 a 1)")
    ax.set_xlim(0, max(e[1] for e in entries) * 1.18)
    ax.set_title("Nous sommes a 0,05 ; la tete est a 0,40 : tout reste a gagner sur les modeles",
                 loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    return _save(figure, path)


def budget_figure(used: Mapping[str, int], cap: int, path: Path) -> Path:
    """Server-counted answers per ep against the human cap."""
    endpoints = sorted(used)
    figure, ax = plt.subplots(figsize=(11, 3.6))
    _clean(ax)
    colors = [FAMILY_COLORS["B"] if used[e] > cap else US_BLUE for e in endpoints]
    ax.bar([e[3:] for e in endpoints], [used[e] for e in endpoints], color=colors)
    ax.axhline(cap, color=INK, linewidth=1.0, linestyle="--")
    ax.text(len(endpoints) - 0.5, cap + 0.12, f"plafond humain : {cap}", ha="right", fontsize=FONT)
    ax.set_ylim(0, cap + 1.6)
    over = [e for e in endpoints if used[e] > cap]
    ax.set_xlabel("endpoint ep (numero)")
    ax.set_ylabel("reponses comptees par le serveur")
    ax.set_title(f"165 reponses payees pour 169 comptees : {', '.join(over)} depassent d'une "
                 "reponse (ReadTimeout)", loc="left")
    return _save(figure, path)


def dendrogram_figure(
    distance: FloatMatrix, endpoints: Sequence[str], families: Sequence[str], tau: float,
    path: Path,
) -> Path:
    """Dendrogram of the family distance, labels coloured by family, with the cut."""
    tree = build_tree(distance)
    figure, ax = plt.subplots(figsize=(11, 5.2))
    _clean(ax)
    dendrogram(tree, labels=list(endpoints), ax=ax, color_threshold=0,
               above_threshold_color=INK_SECONDARY, leaf_font_size=FONT)
    ax.axhline(tau, color=INK, linewidth=1.0, linestyle="--")
    ax.text(0.99, tau + 0.01, f"coupe famille {tau:.2f} (max observe sur les ref)", ha="right",
            fontsize=FONT, transform=ax.get_yaxis_transform())
    family_of = dict(zip(endpoints, families, strict=True))
    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)
        label.set_color(FAMILY_COLORS.get(family_of[label.get_text()], UNKNOWN_COLOR))
    ax.set_ylabel("distance famille (liaison moyenne)")
    ax.set_xlabel("endpoint ep (couleur = famille : bleu A, orange B)")
    ax.set_title("Deux blocs nets : les 33 ep se coupent en 2 familles, 10 ep (B) contre 23 (A)",
                 loc="left")
    return _save(figure, path)


def heatmap_figure(
    distance: FloatMatrix, endpoints: Sequence[str], families: Sequence[str], path: Path,
    title: str, colorbar_label: str,
) -> Path:
    """Distance matrix in dendrogram order with family colour strips."""
    tree = build_tree(distance)
    order = dendrogram(tree, no_plot=True)["leaves"]
    ordered = distance[np.ix_(order, order)]
    figure, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(ordered, cmap=BLUE_RAMP, vmin=0.0, vmax=float(np.percentile(distance, 98)))
    labels = [endpoints[i] for i in order]
    ax.set_xticks(range(len(labels)), [lab[3:] for lab in labels], rotation=90, fontsize=9)
    ax.set_yticks(range(len(labels)), [lab[3:] for lab in labels], fontsize=9)
    for tick, label in zip(ax.get_xticklabels(), labels, strict=True):
        tick.set_color(FAMILY_COLORS.get(families[endpoints.index(label)], UNKNOWN_COLOR))
    for tick, label in zip(ax.get_yticklabels(), labels, strict=True):
        tick.set_color(FAMILY_COLORS.get(families[endpoints.index(label)], UNKNOWN_COLOR))
    bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(colorbar_label, fontsize=FONT)
    ax.set_xlabel("endpoint ep")
    ax.set_ylabel("endpoint ep")
    ax.set_title(title, loc="left", fontsize=TITLE_FONT)
    return _save(figure, path)


def examples_figure(
    rows: Sequence[tuple[str, str, str, str]], path: Path,
) -> Path:
    """Answers that tell the families apart: (endpoint, family, identity, yes/no answer)."""
    figure, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1)
    ax.text(0.0, len(rows) + 0.45, "endpoint", fontweight="bold")
    ax.text(0.15, len(rows) + 0.45, "\"Qui t'a cree ?\" (identite)", fontweight="bold")
    ax.text(0.72, len(rows) + 0.45, "\"L'eau bout-elle plus bas en altitude ?\"", fontweight="bold",
            fontsize=FONT)
    for position, (endpoint, family, identity, yesno) in enumerate(rows):
        y = len(rows) - position - 0.3
        color = FAMILY_COLORS.get(family, UNKNOWN_COLOR)
        ax.add_patch(Rectangle((0.0, y - 0.38), 0.008, 0.76, color=color, transform=ax.transData))
        ax.text(0.015, y, f"{endpoint}  ({family})", va="center", color=color, fontweight="bold")
        ax.text(0.15, y, textwrap.fill(identity, 58), va="center", fontsize=FONT)
        ax.text(0.72, y, textwrap.fill(yesno, 24), va="center", fontsize=FONT,
                family="monospace")
    ax.set_title("L'identite et l'ecriture du oui/non separent Qwen (A) et Llama (B) ; "
                 "3 ep sont ambigus", loc="left")
    return _save(figure, path)


def mcc_vs_n_figure(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str], path: Path, ceiling: float,
) -> Path:
    """Model MCC against N with error bars (sd over random prompt sets), one line per method."""
    figure, ax = plt.subplots(figsize=(9, 4.8))
    _clean(ax)
    for offset, method in enumerate(methods):
        subset = sorted((r for r in rows if r["statistic"] == method), key=lambda r: r["n_prompts"])
        if not subset:
            continue
        sizes = np.array([r["n_prompts"] for r in subset]) + (offset - 0.5) * 0.8
        ax.errorbar(sizes, [r["mcc_one_class"] for r in subset],
                    yerr=[r["mcc_one_class_sd"] for r in subset], color=METHOD_COLORS[method],
                    marker="o", linewidth=2.0, capsize=4, label=METHOD_LABELS[method])
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.9)
    ax.axhline(ceiling, color=INK_SECONDARY, linewidth=0.9, linestyle=":")
    ax.text(41, ceiling + 0.015, f"decoupage grossier parfait : {ceiling:.2f}".replace(".", ","),
            fontsize=FONT, ha="right")
    ax.set_xlabel("N : reponses par endpoint (un prompt chacune)")
    ax.set_ylabel("MCC modele sur les 9 ref (hors echantillon)")
    ax.set_ylim(-0.1, 0.85)
    ax.set_title("La longueur relative (v2) gagne +0,1 a +0,2 de MCC modele sur le regroupement (v1)",
                 loc="left")
    ax.legend(frameon=False, loc="lower right")
    return _save(figure, path)


def detection_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """AUC of the strong (LoRA 0.7, 1.5) and weak (fp8, LoRA 0.3) variants against the copies."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key, title in ((axes[0], "auc_strong", "LoRA fort (0,7 et 1,5)"),
                           (axes[1], "auc_weak", "fp8 et LoRA 0,3")):
        _clean(ax)
        for method in ("lenzr", "ramble", "pair", "ham"):
            subset = sorted((r for r in rows if r["statistic"] == method),
                            key=lambda r: r["n_prompts"])
            ax.plot([r["n_prompts"] for r in subset], [r[key] for r in subset], marker="o",
                    linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.axhline(0.5, color=INK_SECONDARY, linewidth=0.9, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("N : reponses par endpoint")
        ax.set_ylim(0.1, 1.05)
    axes[0].set_ylabel("AUC contre les copies de la base (0,5 = hasard)")
    axes[0].legend(frameon=False, loc="lower right")
    figure.suptitle("Les LoRA forts se voient des N = 5 (AUC 0,95) ; fp8 et LoRA 0,3 restent "
                    "incertains (2 endpoints seulement)", x=0.01, ha="left", fontsize=TITLE_FONT)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def tiers_figure(
    tiers: Sequence[Mapping[str, Any]], path: Path,
) -> Path:
    """Expected MCC gain of each extra tier, with its estimated duration under each bar."""
    figure, ax = plt.subplots(figsize=(9, 4.6))
    _clean(ax)
    labels = [f"{t['label']}\n{t['requests']} requetes\n~{t['minutes']:.0f} min" for t in tiers]
    gains = [t["gain"] for t in tiers]
    errors = [t["gain_sd"] for t in tiers]
    colors = [METHOD_COLORS["lenz+cluster"] if g > 0 else LIGHT_GREY for g in gains]
    ax.bar(range(len(tiers)), gains, yerr=errors, color=colors, capsize=5)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.9)
    ax.set_xticks(range(len(tiers)), labels)
    ax.set_ylabel("gain de MCC modele vs E1 seul (estimation)")
    ax.set_title("+10 par ep est le palier le plus rentable, mais le gain (+0,13) reste sous l'incertitude",
                 loc="left", fontsize=TITLE_FONT)
    return _save(figure, path)
