"""Figures of the team-C reports (matplotlib, light surface, colour-blind-safe Okabe-Ito palette).

One role per colour, stable across figures: ours = blue, leaders = grey, creator Qwen/Alibaba =
blue, Llama/Meta = orange, other creators = green / vermillion / purple. Each function writes one
PNG and returns its path; titles state the conclusion, axes carry names and units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
OURS = "#0072B2"
LEADER = "#9a9a95"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
CREATOR_COLOURS: Mapping[str, str] = {
    "alibaba": OURS, "meta": ORANGE, "anthropic": GREEN, "openai": VERMILION, "none": PURPLE,
}
FAMILY_PALETTE: tuple[str, ...] = (OURS, ORANGE, GREEN, VERMILION, PURPLE, "#56B4E9", "#F0E442")
RAMP = LinearSegmentedColormap.from_list("ramp", ["#eef5fd", "#86b6ef", "#1c5cab", "#0d366b"])
DPI = 130
FloatArray = npt.NDArray[np.float64]


def _axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig: Any, path: Path) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_dashboard_bars(
    path: Path, leaderboard: Mapping[str, float], ours: Mapping[str, float],
    expected: tuple[str, float, float] | None,
) -> Path:
    """Bars: leaderboard, our submitted scores, and the expected score of the pending probe."""
    labels = [*leaderboard, *ours]
    values = [*leaderboard.values(), *ours.values()]
    colours = [LEADER] * len(leaderboard) + [OURS] * len(ours)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    _axes(ax)
    positions = np.arange(len(labels) + (1 if expected else 0))
    ax.bar(positions[: len(labels)], values, color=colours)
    for pos, value in zip(range(len(values)), values, strict=True):
        ax.text(float(pos), value + 0.012, f"{value:.2f}", ha="center", fontsize=11, color=INK)
    if expected:
        name, mean, sd = expected
        ax.bar(positions[-1], mean, color="white", edgecolor=OURS, hatch="//", linewidth=1.3)
        ax.errorbar(positions[-1], mean, yerr=sd, color=INK, capsize=4)
        labels.append(name)
        ax.text(positions[-1], mean + sd + 0.012, f"{mean:.2f} +- {sd:.2f}", ha="center",
                fontsize=10, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("score public (0 a 1)", fontsize=11)
    ax.set_title("Nous a 0,05 ; le 1er a 0,75 : l'ecart est entierement au niveau modele", fontsize=12)
    return _save(fig, path)


def fig_score_decomposition(path: Path, rows: Sequence[tuple[str, float, float]]) -> Path:
    """Stacked bars: contribution of 0.1 x MCC_family and 0.9 x MCC_model to a score."""
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    _axes(ax)
    labels = [r[0] for r in rows]
    family = np.array([0.1 * r[1] for r in rows])
    model = np.array([0.9 * r[2] for r in rows])
    ax.barh(labels, family, color=ORANGE, label="0,1 x MCC famille")
    ax.barh(labels, model, left=family, color=OURS, label="0,9 x MCC modele")
    for pos, (f_part, m_part) in enumerate(zip(family, model, strict=True)):
        ax.text(f_part + m_part + 0.01, pos, f"{f_part + m_part:.2f}", va="center", fontsize=11)
    ax.set_xlabel("score", fontsize=11)
    ax.set_xlim(0, max(0.85, float((family + model).max()) + 0.1))
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("La famille pese au plus 0,10 : tout le reste vient du MCC modele", fontsize=12)
    return _save(fig, path)


def fig_family_constraint(
    path: Path, sizes: Sequence[int], mcc_values: Sequence[float], band: tuple[float, float]
) -> Path:
    """MCC_family of the E1 partition if the Qwen-like group were two true families of given sizes."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    _axes(ax)
    ax.axhspan(*band, color=ORANGE, alpha=0.25, label="MCC famille public lu (E1 = 0,05)")
    ax.plot(sizes, mcc_values, color=OURS, marker="o", label="MCC famille de E1 (23/10) si A = deux familles")
    ax.axhline(1.0, color=LEADER, linestyle="--", label="si A/B etait la vraie structure")
    ax.set_xlabel("taille de la plus petite des deux familles dans le groupe Qwen-like (sur 23)", fontsize=11)
    ax.set_ylabel("MCC famille (528 paires)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("0,05 a E1 exclut 'A/B = vraie structure' : le groupe Qwen-like se coupe", fontsize=12)
    return _save(fig, path)


def fig_agreement_matrix(
    path: Path, agreement: npt.NDArray[np.int64], names: Sequence[str], creators: Sequence[str],
    prompts: int,
) -> tuple[Path, list[int]]:
    """Agreement count matrix (0..prompts) ordered by an average-linkage tree; returns the order."""
    distance = (prompts - agreement) / prompts
    np.fill_diagonal(distance, 0.0)
    order = list(leaves_list(linkage(squareform(distance, checks=False), "average")))
    ordered = agreement[np.ix_(order, order)].astype(float)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    image = ax.imshow(ordered, cmap=RAMP, vmin=0, vmax=prompts)
    labels = [names[i][3:] for i in order]
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for tick, i in zip(ax.get_xticklabels(), order, strict=True):
        tick.set_color(CREATOR_COLOURS.get(creators[i], INK))
    for tick, i in zip(ax.get_yticklabels(), order, strict=True):
        tick.set_color(CREATOR_COLOURS.get(creators[i], INK))
    fig.colorbar(image, ax=ax, fraction=0.04, label="prompts (sur 5) ou les deux reponses sont identiques")
    ax.set_title("Cliques d'accord : 7 endpoints Qwen-like identiques sur 5 prompts", fontsize=12)
    return _save(fig, path), order


def fig_dendrogram(
    path: Path, agreement: npt.NDArray[np.int64], names: Sequence[str], creators: Sequence[str],
    prompts: int,
) -> Path:
    """Average-linkage dendrogram of the answer distance, labels coloured by stated creator."""
    distance = (prompts - agreement) / prompts
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), "average")
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    _axes(ax)
    result = dendrogram(tree, labels=[n[3:] for n in names], ax=ax, color_threshold=0,
                        above_threshold_color=INK2, leaf_font_size=10)
    for tick in ax.get_xticklabels():
        index = [n[3:] for n in names].index(tick.get_text())
        tick.set_color(CREATOR_COLOURS.get(creators[index], INK))
    del result
    ax.set_ylabel("distance = part des 5 prompts en desaccord", fontsize=11)
    ax.set_title("Deux blocs (Qwen bleu, Llama orange) ; les identites system-prompt sortent des blocs",
                 fontsize=12)
    return _save(fig, path)


def fig_prompt_information(path: Path, prompts: Sequence[str], distinct: Sequence[int]) -> Path:
    """Distinct normalised answers per prompt over the 33 endpoints: the information per prompt."""
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    _axes(ax)
    colours = [OURS if d > 5 else LEADER for d in distinct]
    ax.barh(prompts, distinct, color=colours)
    for pos, value in enumerate(distinct):
        ax.text(value + 0.3, pos, str(value), va="center", fontsize=11)
    ax.set_xlabel("reponses distinctes parmi 33 endpoints", fontsize=11)
    ax.set_title("3 prompts sur 5 ne portent qu'1 bit : les prompts ouverts manquent", fontsize=12)
    return _save(fig, path)


def fig_ref_agreement(
    path: Path, pair_rates: Mapping[str, float], self_rates: Mapping[str, float]
) -> Path:
    """Exact-agreement rates on the ref (sampling endpoints): by pair class and with oneself."""
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax in axes:
        _axes(ax)
    axes[0].bar(list(pair_rates), list(pair_rates.values()), color=[OURS, ORANGE, GREEN])
    for pos, value in enumerate(pair_rates.values()):
        axes[0].text(pos, value + 0.005, f"{value:.2f}", ha="center", fontsize=11)
    axes[0].set_ylabel("accord exact (102 prompts)", fontsize=11)
    axes[0].set_title("Entre endpoints : SS 0,17 vs ST 0,13", fontsize=11)
    axes[1].bar(list(self_rates), list(self_rates.values()), color=OURS)
    axes[1].set_xticks(range(len(self_rates)))
    axes[1].set_xticklabels(list(self_rates), rotation=30, ha="right", fontsize=9)
    axes[1].set_title("Un endpoint avec lui-meme (2 tirages) : 0,01 a 0,43", fontsize=11)
    fig.suptitle("Les ref echantillonnent : l'accord exact y est faible, y compris avec soi-meme",
                 fontsize=12)
    return _save(fig, path)


def fig_library_structure(
    path: Path, families: FloatArray, models: FloatArray, weights: FloatArray
) -> Path:
    """Number of families / models in the hypothesis library, prior and after E1 and E1b."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, values, label in ((axes[0], families, "familles"), (axes[1], models, "modeles")):
        _axes(ax)
        top = int(values.max()) + 1
        bins = np.arange(1, top + 1) - 0.5
        ax.hist(values, bins=bins, color=LEADER, density=True, label="bibliotheque (a priori)")
        ax.hist(values, bins=bins, weights=weights, color=OURS, density=True, alpha=0.7,
                label="apres E1 = 0,05 et E1b = 0,05")
        ax.set_xlabel(f"nombre de {label}", fontsize=11)
        ax.set_ylabel("densite", fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle("Les deux scores a 0,05 deplacent la masse vers moins de modeles et plus de familles",
                 fontsize=12)
    return _save(fig, path)


def fig_marginals(
    path: Path, same_family: FloatArray, same_model: FloatArray, order: Sequence[int],
    names: Sequence[str],
) -> Path:
    """Posterior P(same family) and P(same model) per pair, endpoints in dendrogram order."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    labels = [names[i][3:] for i in order]
    for ax, matrix, title in (
        (axes[0], same_family, "P(meme famille)"), (axes[1], same_model, "P(meme modele)"),
    ):
        image = ax.imshow(matrix[np.ix_(order, order)], cmap=RAMP, vmin=0, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=11)
        fig.colorbar(image, ax=ax, fraction=0.04)
    fig.suptitle("Posterieur : le coeur Qwen est probablement un modele, le reste est incertain",
                 fontsize=12)
    return _save(fig, path)


def fig_partition_graph(
    path: Path, rows: Sequence[Sequence[str]], title: str,
    edge_strength: FloatArray | None = None,
) -> Path:
    """Endpoints as nodes: colour = family, a ring joins the endpoints of one model."""
    model_members: dict[tuple[str, str], list[str]] = {}
    family_ids: dict[str, int] = {}
    for endpoint, family, model in rows:
        family_ids.setdefault(family, len(family_ids))
        model_members.setdefault((family, model), []).append(endpoint)
    groups = sorted(model_members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    graph = nx.Graph()
    positions: dict[str, tuple[float, float]] = {}
    big = [g for g in groups if len(g[1]) > 1]
    small = [g for g in groups if len(g[1]) == 1]
    for k, ((family, _), members) in enumerate(big):
        angle = 2 * np.pi * k / max(len(big), 1)
        centre = np.array([np.cos(angle), np.sin(angle)]) * 2.4
        for j, endpoint in enumerate(members):
            sub = 2 * np.pi * j / len(members)
            radius = 0.18 + 0.075 * len(members)
            positions[endpoint] = (centre[0] + radius * np.cos(sub), centre[1] + radius * np.sin(sub))
            graph.add_node(endpoint, family=family)
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                graph.add_edge(members[left], members[right])
    for k, ((family, _), members) in enumerate(small):
        angle = 2 * np.pi * k / max(len(small), 1)
        positions[members[0]] = (0.95 * np.cos(angle), 0.95 * np.sin(angle))
        graph.add_node(members[0], family=family)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    colours = [FAMILY_PALETTE[family_ids[graph.nodes[n]["family"]] % len(FAMILY_PALETTE)]
               for n in graph.nodes]
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color=INK2, width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colours, node_size=330,
                           edgecolors=INK, linewidths=0.6)
    nx.draw_networkx_labels(graph, positions, labels={n: n[3:] for n in graph.nodes}, ax=ax,
                            font_size=8, font_color="white")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=FAMILY_PALETTE[i % 7],
                      label=f"famille {name}") for name, i in family_ids.items()]
    ax.legend(handles=handles, fontsize=9, loc="lower left")
    del edge_strength
    return _save(fig, path)


def fig_candidates(
    path: Path, names: Sequence[str], means: FloatArray, sds: FloatArray,
    observed: Mapping[str, float],
    title: str = "Bleu = deja soumis (losange = score public lu) ; les attentes sont celles de la bibliotheque",
) -> Path:
    """Expected final score (all pairs) of each candidate, with posterior sd, and observed ones."""
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(names) + 1.4))
    _axes(ax)
    colours = [OURS if names[i] in observed else LEADER for i in order]
    ax.barh([names[i] for i in order], means[order], xerr=sds[order], color=colours, capsize=3)
    for pos, i in enumerate(order):
        if names[i] in observed:
            ax.plot(observed[names[i]], pos, marker="D", color=VERMILION, markersize=8)
    ax.set_xlabel("score attendu sur les 528 paires (barre = ecart-type du posterieur)", fontsize=11)
    ax.set_title(title, fontsize=11)
    return _save(fig, path)


def fig_predictive(path: Path, levels: Sequence[float], probabilities: Mapping[str, Sequence[float]]) -> Path:
    """Posterior predictive distribution of the public score (0.05 steps) of several probes."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    _axes(ax)
    width = 0.8 / max(len(probabilities), 1)
    for k, (name, values) in enumerate(probabilities.items()):
        ax.bar(np.arange(len(levels)) + k * width, values, width=width,
               color=FAMILY_PALETTE[k % 7], label=name)
    ax.set_xticks(np.arange(len(levels)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{x:.2f}" if x < 0.3 else ">= 0,30" for x in levels], fontsize=10)
    ax.set_xlabel("score public observe (pas de 0,05)", fontsize=11)
    ax.set_ylabel("probabilite", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Un 0,05 exact est quasi certain pour E1 et probable pour E1b ; la sonde 3 s'etale",
                 fontsize=11)
    return _save(fig, path)


def fig_voi(path: Path, names: Sequence[str], voi: FloatArray, mutual: FloatArray) -> Path:
    """Value of information (score units) and mutual information (bits) of the best probes."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.3 * len(names) + 1.6), sharey=True)
    for ax in axes:
        _axes(ax)
    axes[0].barh(names, voi, color=OURS)
    axes[0].set_xlabel("gain attendu du meilleur envoi apres lecture (score)", fontsize=10)
    axes[1].barh(names, mutual, color=ORANGE)
    axes[1].set_xlabel("information du score public sur la structure (bits)", fontsize=10)
    fig.suptitle("Valeur d'information a un pas : les sondes proches du consensus dominent", fontsize=12)
    return _save(fig, path)


def fig_closed_loop(path: Path, curves: Sequence[Sequence[float]], labels: Sequence[str]) -> Path:
    """Score of the posterior-best submission as probes are spent, per synthetic truth."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    _axes(ax)
    for k, (curve, label) in enumerate(zip(curves, labels, strict=True)):
        ax.plot(range(len(curve)), curve, marker="o", color=FAMILY_PALETTE[k % 7], label=label)
    mean = np.mean(np.array(curves), axis=0)
    ax.plot(range(len(mean)), mean, color=INK, linewidth=2.6, label="moyenne")
    ax.set_xlabel("sondes consommees (0 = texte seul)", fontsize=11)
    ax.set_ylabel("score complet de la meilleure reponse", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Boucle fermee (3 verites synthetiques) : +0,15 en moyenne apres 6 sondes, de +0,04 a +0,31",
                 fontsize=11)
    return _save(fig, path)


def fig_history(path: Path, points: Sequence[tuple[str, float]], pending: Sequence[str]) -> Path:
    """Public scores of the team's Model Equality submissions, in sending order."""
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    _axes(ax)
    ax.plot(range(len(points)), [p[1] for p in points], marker="o", color=OURS)
    ax.set_xticks(range(len(points) + len(pending)))
    ax.set_xticklabels([p[0] for p in points] + [f"{n}\n(en attente)" for n in pending], fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.axhline(0.75, color=LEADER, linestyle="--", label="1er du classement (0,75)")
    ax.set_ylabel("score public", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Historique : deux soumissions a 0,05 ; la sonde 3 doit lever l'hypothese modele", fontsize=11)
    return _save(fig, path)


def fig_ref_power(
    path: Path, curve: Mapping[int, Mapping[str, float]], paid: Sequence[int] = (5, 10, 20),
) -> Path:
    """AUC of the long-answer style score vs number of long units per endpoint, measured on the ref."""
    sizes = sorted(curve)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    _axes(ax)
    for key, label, colour in (("strong", "LoRA 0,7 et 1,5 (variantes fortes)", OURS),
                               ("weak", "fp8 et LoRA 0,3 (variantes faibles)", VERMILION)):
        values = np.array([curve[s][key] for s in sizes])
        spread = np.array([curve[s][key + "_sd"] for s in sizes])
        ax.plot(sizes, values, marker="o", color=colour, label=label)
        ax.fill_between(sizes, values - spread, values + spread, color=colour, alpha=0.15)
    ax.axhline(0.5, color=LEADER, linestyle="--", label="hasard (AUC 0,5)")
    for step in paid:
        ax.axvline(step, color=LEADER, linewidth=0.8, linestyle=":")
        ax.text(step + 0.3, 0.12, f"+{step}", fontsize=10, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("unites longues par endpoint (prompt x tirage)", fontsize=11)
    ax.set_ylabel("AUC variante contre copies du modele de base", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("MESURE sur ref : les variantes fortes montent avec N, les faibles restent au hasard",
                 fontsize=11)
    return _save(fig, path)


def fig_long_style(
    path: Path, prompts: Sequence[str], counts: Mapping[str, Mapping[str, Mapping[str, int]]]
) -> Path:
    """Opening word of the long answers by stated creator group: the style separates families."""
    fig, axes = plt.subplots(1, len(prompts), figsize=(4.6 * len(prompts), 3.4), squeeze=False)
    palette = {"sure": OURS, "certainly": GREEN, "here": ORANGE, "other": LEADER}
    for ax, prompt in zip(axes[0], prompts, strict=True):
        _axes(ax)
        groups = list(counts[prompt])
        bottoms = np.zeros(len(groups))
        for word, colour in palette.items():
            values = np.array([counts[prompt][g].get(word, 0) for g in groups], dtype=float)
            ax.bar(groups, values, bottom=bottoms, color=colour, label=word)
            bottoms += values
        ax.set_title(prompt, fontsize=10)
        ax.set_ylabel("endpoints", fontsize=10)
    axes[0][0].legend(fontsize=9, title="premier mot")
    fig.suptitle("Qwen-like : 'Sure' / 'Certainly' ; Llama-like : 'Here's' : le style separe les familles",
                 fontsize=12)
    return _save(fig, path)
