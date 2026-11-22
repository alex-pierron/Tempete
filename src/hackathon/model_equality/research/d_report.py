"""Salvo report of team D: analysis, figures and one self-contained HTML (figures embedded).

Usage (repo root, venv of the repo, 2 BLAS threads, heavy)::

    python -m hackathon.model_equality.research.d_report --salvo 1 \
        --truth data/hackathon/hf/model_equality/practice_answers.csv \
        --discovery artifacts/model_equality/reference_answers.jsonl \
                    artifacts/model_equality/reference_answers_b.jsonl \
                    artifacts/model_equality/reference_answers_m1.jsonl

Writes `reports/phase2/model_equality/team-d/report-<N>.html`, `figures-<N>/` and `metrics-<N>.json`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hackathon.model_equality.research.d_analysis import AnalysisConfig, refresh_fresh, run_analysis
from hackathon.model_equality.research.d_clique import structure_metrics
from hackathon.model_equality.research.d_collect import D_STORE_PATH
from hackathon.model_equality.research.d_figures import (
    ENDPOINT_ORDER,
    endpoint_label,
    make_all_figures,
)
from hackathon.model_equality.research.report_b import STYLE, embed_png, fmt, table_html
from hackathon.model_equality.store import REPO_ROOT, AnswerStore

REPORT_DIR = REPO_ROOT / "reports" / "phase2" / "model_equality" / "team-d"
DEFAULT_ENDPOINTS = tuple(f"ref-{i:02d}" for i in range(1, 10))
DEFAULT_REPLICATES = 200
DEFAULT_SELECTION_SPLITS = 30
LEADERBOARD_SNAPSHOT = {"time": "21/11/2026 19 h 41", "rank": "13e sur 22", "score": 0.05, "best": 0.75,
                        "second": 0.70}
GIT_TIMEOUT_S = 10.0
D4_SUCCESS = 0.65
"""Pre-registered success threshold of the d4 validation: AUC at N = 20 (set before reading any d4 answer)."""
SCORE_MODEL_WEIGHT = 0.9
"""Weight of the model-level MCC in the official score (0.1 family + 0.9 model)."""
ASSUMED_LORA_PAIR_SHARE = 8 / 55
"""Share of the within-family pairs that are (LoRA 0.3, S), for an ASSUMED family of 11 endpoints
split 4 / 2 / 2 / 2 / 1 over S, T1, T2, T3, T4: 2 x 4 = 8 pairs out of 55."""
EXTRA_STYLE = """
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .big{font-size:34px;font-weight:700;line-height:1.1}
.tile .lbl{font-size:12.5px;color:var(--ink2);margin-top:4px}
.tile.warn .big{color:var(--warn)}.tile.good .big{color:var(--good)}.tile.acc .big{color:var(--accent)}
mark{background:#ffe08a;color:#0b0b0b;padding:0 1px}
.est{font-style:italic;color:var(--ink2)}
section.page1{break-after:page}
"""


def _auc(metrics: Mapping[str, Any], stat: str, n: int, role: str) -> tuple[float, float, float]:
    """(mean, low, high) of the discovery AUC of one statistic."""
    entry = metrics["discovery"]["auc_curves"][stat][str(n)][role]
    return float(entry["mean"]), float(entry["low"]), float(entry["high"])


def _fig(figures: Path, name: str, caption: str) -> str:
    """Embedded figure with its 'what to conclude' caption (empty when the file is missing)."""
    path = figures / name
    if not path.exists():
        return ""
    return (f'<figure><img alt="{html.escape(caption[:80])}" src="{embed_png(path)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def auc_table(metrics: Mapping[str, Any]) -> str:
    """AUC of every statistic at N = 5, 10, 20, 40, 80 for both targets."""
    sizes = (5, 10, 20, 40, 80)
    stats = (("length", "longueur"), ("ends_period", "fin par un point"), ("sentences", "nb de phrases"),
             ("style22", "22 traits de style"), ("transfer", "direction LoRA apprise (transfert)"))
    rows: list[list[str]] = []
    for role, label in (("fp8", "fp8"), ("lora_weak", "LoRA 0,3")):
        for key, name in stats:
            cells = [f"{_auc(metrics, key, n, role)[0]:.2f}" for n in sizes]
            low, high = _auc(metrics, key, 40, role)[1:]
            rows.append([label, name, *cells, f"[{low:.2f} ; {high:.2f}]"])
    for key, name, role, label in (("oracle_fp8", "oracle (étiquettes de fp8)", "fp8", "fp8"),
                                   ("oracle_lora03", "oracle (étiquettes de LoRA 0,3)", "lora_weak", "LoRA 0,3")):
        cells = [f"{_auc(metrics, key, n, role)[0]:.2f}" for n in sizes]
        low, high = _auc(metrics, key, 40, role)[1:]
        rows.append([label, name, *cells, f"[{low:.2f} ; {high:.2f}]"])
    header = ["cible", "statistique", *[f"N = {n}" for n in sizes], "intervalle N = 40"]
    return table_html(header, rows, numeric_from=2)


def effect_table(metrics: Mapping[str, Any]) -> str:
    """Per-answer effect size and extrapolated N of the LoRA direction and of the period feature."""
    truth = metrics["discovery"]["truth"]
    cohen = metrics["discovery"]["cohen_d"]["transfer"]
    growth_period = metrics["discovery"]["t_growth"]["ends_period"]
    growth_transfer = metrics["discovery"]["t_growth"]["transfer"]
    rows: list[list[str]] = []
    for endpoint in ENDPOINT_ORDER:
        d = cohen[endpoint]
        period = growth_period[endpoint]
        transfer = growth_transfer[endpoint]
        rows.append([
            endpoint_label(truth, endpoint),
            f"{d['d']:.2f} [{d['low']:.2f} ; {d['high']:.2f}]",
            fmt(transfer.get("t_at_120"), 1),
            fmt(period.get("t_at_120"), 1),
            _needed(period),
        ])
    return table_html(["endpoint", "d de la direction LoRA [IC 95 %]", "t à N = 120 (direction)",
                       "t à N = 120 (fin par un point)", "N pour t = 3 (estimation)"], rows)


def _needed(row: Mapping[str, float]) -> str:
    """Extrapolated N for t = 3, or a dash when the slope is negative or null."""
    needed = row["n_for_target"]
    if needed == float("inf") or row["slope"] <= 0:
        return "jamais (pente nulle ou négative)"
    return f"{needed:.0f}"


def exact_table(metrics: Mapping[str, Any]) -> str:
    """AUC of the exact-mismatch distance by consensus level."""
    rows: list[list[str]] = []
    for threshold, entry in metrics["discovery"]["exact"]["table"].items():
        curve = entry["curve"]
        for n in ("5", "10", "20", "40"):
            if n in curve:
                rows.append([threshold, str(entry["cells"]), n, f"{curve[n]['fp8']['mean']:.2f}",
                             f"{curve[n]['lora_weak']['mean']:.2f}"])
    return table_html(["consensus minimal", "cellules retenues", "N", "AUC fp8", "AUC LoRA 0,3"], rows,
                      numeric_from=0)


def exact_auc_range(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Min and max AUC of the exact-mismatch distance at N = 20 and 40 for consensus >= 0.34."""
    values: list[float] = []
    for threshold, entry in metrics["discovery"]["exact"]["table"].items():
        if float(threshold) < 0.34:
            continue
        for n in ("20", "40"):
            if n in entry["curve"]:
                values.extend(entry["curve"][n][role]["mean"] for role in ("fp8", "lora_weak"))
    return (min(values), max(values)) if values else (float("nan"), float("nan"))


def repeat_table(metrics: Mapping[str, Any]) -> str:
    """AUC of the collision and diversity statistics for both targets."""
    rows: list[list[str]] = []
    for name, label in (("collisions", "taux de collision"), ("diversity", "diversité")):
        curve = metrics["discovery"]["repeat_stats"][name]
        for role, target in (("fp8", "fp8"), ("lora_weak", "LoRA 0,3")):
            sizes = [n for n in ("5", "10", "20", "40", "80") if n in curve]
            rows.append([label, target, *[f"{curve[n][role]['mean']:.2f}" for n in sizes]])
    header = ["statistique", "cible", *[f"N = {n}" for n in ("5", "10", "20", "40", "80")]]
    width = len(header)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return table_html(header, rows, numeric_from=2)


def mcc_table(metrics: Mapping[str, Any]) -> str:
    """Held-out MCC proxy of the period-ending feature for fp8 and LoRA 0.3."""
    rows: list[list[str]] = []
    for role, label in (("fp8", "fp8"), ("lora_weak", "LoRA 0,3")):
        for n, entry in metrics["discovery"]["mcc_proxy"][role].items():
            rows.append([label, n, f"{entry['mean']:.2f} [{entry['low']:.2f} ; {entry['high']:.2f}]",
                         f"{entry['share_above_half']:.0%}"])
    return table_html(["cible", "N", "MCC proxy [2,5 ; 97,5 %]", "tirages avec MCC >= 0,5"], rows, numeric_from=1)


TARGET_LABELS = {"ref-02": "fp8", "ref-09": "LoRA 0,3", "ref-04": "LoRA 0,7", "ref-07": "LoRA 1,5",
                 "ref-03": "system prompt (déguisement)", "ref-05": "persona (déguisement)",
                 "ref-06": "watermark (déguisement)", "ref-08": "température 1,0 (déguisement)"}
STATISTIC_LABELS = {"ends_period": "fin par un point", "length": "longueur", "sentences": "nb de phrases",
                    "transfer": "direction LoRA apprise"}


def separability_html(metrics: Mapping[str, Any]) -> str:
    """Tables of the AUC of each target against S, at N = 10 / 20 / 30 / 50, discovery and fresh d4."""
    sep = metrics.get("separability")
    if not sep:
        return ""
    blocks = [(
        "<h3>Séparabilité de chaque cible contre S, N = 10 / 20 / 30 / 50</h3>"
        "<p>AUC des paires (cible, S) au-dessus des paires (S, S), la cible étant retirée du groupe S. "
        "Pour fp8 et LoRA, haut = bien ; pour un déguisement, haut = <b>mal</b> (le déguisement est pris "
        "pour un changement de modèle).</p>")]
    for key, title in (("d4", f"Prompts neufs d4 ({sep.get('d4_cells', 0)} cellules arithmétiques)"),
                       ("discovery", "Découverte (228 cellules, tous prompts)")):
        table = sep.get(key)
        if not table:
            continue
        sizes = sorted({n for per in table.values() for row in per.values() for n in row}, key=int)
        rows: list[list[str]] = []
        for stat, per in table.items():
            for target, label in TARGET_LABELS.items():
                if target in per:
                    rows.append([STATISTIC_LABELS.get(stat, stat), label, *[f"{per[target].get(n, float('nan')):.2f}"
                                                                           for n in sizes]])
        blocks.append(f"<h3>{html.escape(title)}</h3>"
                      + table_html(["statistique", "cible", *[f"N = {n}" for n in sizes]], rows, numeric_from=2))
    families = sep.get("families")
    if families:
        rows = []
        for family, entry in families.items():
            cells = int(entry["cells"]["n"]["cells"])
            for feature in ("ends_period", "log_chars", "sentence_count"):
                rows.append([html.escape(family), str(cells), feature,
                             *[f"{entry[feature][t]['10']:.2f}" for t in ("ref-02", "ref-09", "ref-03")]])
        blocks.append("<h3>Catégories de prompts (N = 10, choisies après coup : pistes, pas mesures)</h3>"
                      + table_html(["famille", "cellules", "statistique", "AUC fp8", "AUC LoRA 0,3", "AUC system prompt"],
                                   rows, numeric_from=1))
    return "".join(blocks)


def examples_html(metrics: Mapping[str, Any]) -> str:
    """Side-by-side replies on cells where LoRA 0.3 (or the system prompt) ends with a period."""
    truth = metrics["discovery"]["truth"]
    order = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08", "ref-02", "ref-09")
    blocks: list[str] = []
    for row in metrics["discovery"]["examples"]:
        cells = []
        for endpoint in order:
            text = html.escape(row["responses"][endpoint])
            stripped = text.rstrip()
            marked = (f"{stripped[:-1]}<mark>.</mark>" if stripped.endswith(".") else text)
            cells.append(f"<tr><td>{html.escape(endpoint_label(truth, endpoint))}</td>"
                         f'<td class="resp">{marked}</td></tr>')
        blocks.append(f"<h3>{html.escape(row['kind'])}</h3><p class=\"resp\">{html.escape(row['prompt'])}</p>"
                      f"<table><tbody>{''.join(cells)}</tbody></table>")
    return "".join(blocks) or "<p>Aucun exemple retenu.</p>"


PILOT_FEATURES = (("b_line_words", "mots par ligne"), ("b_period", "part de lignes finies par un point"),
                  ("b_facts_acc", "items justes"))


CRITERION_FEATURES = ("b_line_words", "b_period")
CRITERION_TARGET = "ref-09"
CRITERION_FP8 = "ref-02"


def outside(value: float, others: Sequence[float]) -> bool:
    """True when `value` is above the largest or below the smallest of `others`."""
    return value > max(others) or value < min(others)


def d5_criterion(metrics: Mapping[str, Any]) -> tuple[int, int]:
    """(cells passed, cells) of the pre-registered d5 criterion.

    A cell passes when LoRA 0.3 lies outside the range of the five S endpoints AND of fp8 on both
    `CRITERION_FEATURES`. Only cells answered by all 7 endpoints count.
    """
    truth = metrics["discovery"]["truth"]
    family = metrics["fresh"]["groups"].get("d5_facts", {}).get("family_means", {}).get("bfact2")
    if not family:
        return 0, 0
    passed = 0
    for values in family["cell_values"].values():
        others = [e for e in values if e != CRITERION_TARGET and (truth[e]["model"] == "S" or e == CRITERION_FP8)]
        ok = all(outside(values[CRITERION_TARGET][f], [values[e][f] for e in others]) for f in CRITERION_FEATURES)
        passed += int(ok)
    return passed, len(family["cell_values"])


def pilot_lines(battery: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    """One data-driven sentence per fact battery: LoRA 0.3 against the S endpoints and fp8."""
    truth = metrics["discovery"]["truth"]
    base = [e for e in ENDPOINT_ORDER if truth[e]["model"] == "S"]
    items: list[str] = []
    for family in ("bfacts", "bfact2"):
        info = battery.get(family)
        if not info or "ref-09" not in info["means"]:
            continue
        parts: list[str] = []
        for feature, label in PILOT_FEATURES:
            if feature not in info["features"]:
                continue
            s_values = [info["means"][e][feature] for e in base if e in info["means"]]
            parts.append(f"{label} : LoRA 0,3 {info['means']['ref-09'][feature]:.2f}, fp8 "
                         f"{info['means']['ref-02'][feature]:.2f}, S de {min(s_values):.2f} à {max(s_values):.2f}")
        items.append(f"<li>Batterie de faits {html.escape(family)} ({info['cells']} cellules) : " + " ; ".join(parts) + ".</li>")
    passed, total = d5_criterion(metrics)
    verdict = (f"<p><b>Critère d5 : {passed} cellule(s) sur {total}</b> sortent de la plage (seuil de succès : 2 sur 3 "
               "des cellules reçues).</p>") if total else ""
    if not items:
        return ""
    return (verdict + "<h3>Pilote : batteries de faits (lecture, pas mesure)</h3><ul>" + "".join(items) + "</ul>"
            "<p>Protocole d5 écrit avant les réponses : 4 batteries de 12 questions (graine 20261121 + 4), 3 répétitions, "
            "7 endpoints. Critère : LoRA 0,3 sort de la plage des 5 S et de fp8 sur les mots par ligne ET la part de lignes "
            "à point dans au moins 2 cellules sur 3. Dans une réponse, les lignes partagent un même mode de format : "
            "l'unité statistique est la réponse, pas la ligne.</p>")


def fresh_html(metrics: Mapping[str, Any], figures: Path) -> str:
    """Out-of-sample section: what the fresh prompts show so far."""
    groups = metrics["fresh"]["groups"]
    lines = [f"<li>{html.escape(name)} : {info.get('cells', 0)} cellules communes aux 7 endpoints "
             f"({info.get('answers', 0)} réponses stockées)</li>" for name, info in groups.items()]
    body = "<ul>" + "".join(lines) + "</ul>"
    truth_map = metrics["discovery"]["truth"]
    for name, info in groups.items():
        means = info.get("feature_means")
        if not means or info.get("cells", 0) < 2:
            continue
        order = [e for e in ENDPOINT_ORDER if e in means]
        rows = [[endpoint_label(truth_map, e), f"{means[e]['ends_period']:.2f}", f"{means[e]['log_chars']:.2f}"]
                for e in order]
        body += (f"<h3>{html.escape(name)} : part de réponses finissant par un point et log-longueur "
                 f"({info['cells']} cellules fraîches)</h3>" + table_html(["endpoint", "fin par un point", "log longueur"], rows))
        if "auc" in info:
            auc_rows = []
            for stat, curve in info["auc"].items():
                for n, entry in curve.items():
                    auc_rows.append([stat, n, f"{entry['fp8']['mean']:.2f}", f"{entry['lora_weak']['mean']:.2f}"])
            body += table_html(["statistique", "N", "AUC fp8", "AUC LoRA 0,3"], auc_rows, numeric_from=1)
    battery = {**groups.get("d3_batteries", {}).get("family_means", {}),
               **groups.get("d5_facts", {}).get("family_means", {})}
    body += pilot_lines(battery, metrics)
    for family, info in battery.items():
        features = info["features"]
        if not features:
            continue
        truth = metrics["discovery"]["truth"]
        order = [e for e in ENDPOINT_ORDER if e in info["means"]]
        rows = [[endpoint_label(truth, e), *[f"{info['means'][e][f]:.2f}" for f in features]] for e in order]
        body += (f"<h3>Famille {html.escape(family)} ({info['cells']} cellules, pilote)</h3>"
                 + table_html(["endpoint", *features], rows))
    return body + _fig(figures, "fresh.png",
                       "Marge de la direction LoRA sur les prompts neufs : à lire comme un contrôle de cohérence, "
                       "pas comme une mesure (quelques cellules seulement).")


def ep_structure_html(metrics: Mapping[str, Any], figures: Path) -> str:
    """Section on the exact-agreement structure of the first ep phase (read only)."""
    structure = metrics.get("ep_structure")
    if not structure:
        return ""
    clusters = structure["clusters_at_cut"]
    cliques = [c for c in structure["strict_cliques"] if len(c) > 1]
    patterns: dict[str, list[str]] = {}
    for endpoint, pattern in structure["binary_patterns"].items():
        patterns.setdefault(pattern, []).append(endpoint[-2:])
    rows = [[html.escape(p), str(len(e)), html.escape(", ".join(sorted(e)))]
            for p, e in sorted(patterns.items(), key=lambda kv: -len(kv[1]))]
    binary = ", ".join(structure["binary_prompts"])
    sizes = ", ".join(str(len(g)) for g in clusters)
    largest = cliques[0] if cliques else []
    clique_text = f"{len(largest)} endpoints ({', '.join(e[-2:] for e in largest)})"
    other_cliques = len(cliques) - 1
    return f"""
<h2>ep. Cliques d'égalité exacte de la phase e1 (lecture seule, aucune requête) : ce qu'elles montrent et ne montrent pas</h2>
<p>Source : <code>artifacts/model_equality/answers.jsonl</code> du checkout principal, 5 réponses par endpoint (5 prompts, répétition 0).
<b>{clique_text}</b> rendent la même chaîne sur les 5 prompts, y compris une complétion de 126 caractères (le début de la Déclaration d'indépendance, un texte mémorisé) ; {other_cliques} autre(s) clique(s) plus petites existent.
Coupe de l'arbre à 0,4 de désaccord : groupes de tailles {sizes}.</p>
<p><b>Correction du directeur (mesure de r-met-c)</b> : sur le premier prompt long, les 33 endpoints donnent 33 réponses distinctes, clique comprise. Les ep échantillonnent donc comme les ref : ces égalités sont le <i>mode</i> de la distribution sur des réponses courtes ou mémorisées, pas un décodage glouton. Une lecture précédente de cette section (décodage déterministe) est retirée.</p>
{_fig(figures, "ep_structure.png", "Les 33 endpoints se rangent en blocs de 10 à 11 (deux blocs qui se disent Qwen, un bloc qui se dit Llama). C'est une lecture des familles, pas des modèles : l'égalité exacte sur réponses courtes ne tranche ni fp8 ni LoRA.")}
<h3>Empreinte binaire (prompts à au plus 3 réponses distinctes : {html.escape(binary)})</h3>
{table_html(["motif (réponses normalisées)", "endpoints", "numéros"], rows, numeric_from=1)}
<p>Lecture prudente : l'identité auto-déclarée (Qwen / Llama) trace les familles, et un motif binaire différent (ex. cand-yesno-002 : « yes » chez 20 endpoints Qwen, « no » chez 5 Llama et chez ep-33) signale des endpoints à surveiller, mais un seul tirage d'un prompt à probabilité intermédiaire suffit à produire ce motif sans différence de modèle. Sur ref, la même statistique donne une AUC inférieure à 0,5 (voir 3) : elle n'est pas la clé pour fp8 ou LoRA 0,3.</p>"""


def build_html(metrics: Mapping[str, Any], figures: Path, salvo: int, answers_d: int) -> str:
    """The whole report page."""
    disc = metrics["discovery"]
    selection = disc.get("selection") or {}
    held = sum(selection.get("held_out_auc", [0.0])) / max(len(selection.get("held_out_auc", [1])), 1)
    optimistic = sum(selection.get("selected_auc", [0.0])) / max(len(selection.get("selected_auc", [1])), 1)
    period_fp8 = _auc(metrics, "ends_period", 40, "fp8")[0]
    period_lora = _auc(metrics, "ends_period", 40, "lora_weak")[0]
    period_lora_80 = _auc(metrics, "ends_period", 80, "lora_weak")[0]
    period_lora_20 = _auc(metrics, "ends_period", 20, "lora_weak")[0]
    agree = disc["exact"]["mode_agreement"]
    lb = LEADERBOARD_SNAPSHOT
    d4_bullet = ""
    d4 = metrics["fresh"]["groups"].get("d4_validation", {})
    if d4.get("cells", 0) >= 2 and "feature_means" in d4:
        fmn = d4["feature_means"]
        s_rates = [fmn[e]["ends_period"] for e in fmn if disc["truth"][e]["model"] == "S"]
        d4_bullet = (
            f"<li><b>Validation sur prompts neufs (d4 : {d4['cells']} cellules sur 30 prévues, protocole écrit avant les "
            f"réponses)</b> : LoRA 0,3 finit {fmn['ref-09']['ends_period']:.0%} de ses réponses par un point, contre "
            f"{min(s_rates):.0%} à {max(s_rates):.0%} pour les S et {fmn['ref-02']['ends_period']:.0%} pour fp8.")
        if "auc" in d4:
            curve = d4["auc"]["ends_period"]
            lora = ", ".join(f"{curve[n]['lora_weak']['mean']:.2f} à N = {n}" for n in curve)
            fp8c = ", ".join(f"{curve[n]['fp8']['mean']:.2f} à N = {n}" for n in curve)
            d4_bullet += f" AUC LoRA 0,3 : {lora} ; fp8 : {fp8c}."
            if "20" in curve:
                verdict = ("critère de succès atteint" if curve["20"]["lora_weak"]["mean"] >= D4_SUCCESS
                           else "critère de succès non atteint")
                d4_bullet += f" Seuil figé (AUC &gt;= {D4_SUCCESS:.2f} à N = 20) : {verdict}."
            else:
                d4_bullet += f" Le seuil figé (AUC &gt;= {D4_SUCCESS:.2f} à N = 20) sera tranché dès 20 cellules."
        mcc_fresh = d4.get("mcc_proxy", {}).get("lora_weak", {})
        if mcc_fresh:
            last = max(mcc_fresh, key=int)
            d4_bullet += f" MCC proxy hors échantillon sur les paires (LoRA 0,3, S) : {mcc_fresh[last]['mean']:.2f} à N = {last}."
        d4_bullet += "</li>"
    pilot_bullet = d4_bullet
    period_verdict = '<span class="verdict-dig">creuser : valider hors échantillon</span>'
    d4_curve = d4.get("auc", {}).get("ends_period")
    if d4_curve:
        largest = max(d4_curve, key=int)
        if d4_curve[largest]["lora_weak"]["mean"] >= D4_SUCCESS:
            period_verdict = (f'<span class="verdict-keep">garder : confirmé sur prompts neufs '
                              f"(AUC {d4_curve[largest]['lora_weak']['mean']:.2f} à N = {largest})</span>")
    d4_ep_text = "la validation sur prompts neufs est en cours (d4)"
    if d4_curve:
        d4_sizes = sorted(d4_curve, key=int)
        d4_ep_text = ("AUC LoRA 0,3 sur prompts neufs : " + ", ".join(
            f"{d4_curve[n]['lora_weak']['mean']:.2f} à N = {n}" for n in d4_sizes)
            + " (<i class=\"est\">estimation</i> pour les ep : 10 à 20 réponses par endpoint suffiraient si le gain se transfère)")
    passed, total = d5_criterion(metrics)
    if total:
        pilot_bullet += (
            f"<li><b>Piste abandonnée : batteries de faits</b>. Le pilote d3 (1 cellule : LoRA 0,3 répondait par des "
            f"phrases et plus juste) ne se réplique pas : {passed} cellule(s) sur {total} de d5 sortent de la plage "
            f"(seuil 2 sur 3). C'était un tirage de mode isolé.</li>")
    fp8_values = [_auc(metrics, stat, 40, "fp8")[0]
                  for stat in ("length", "ends_period", "sentences", "style22", "transfer", "oracle_fp8")]
    fp8_low, fp8_high = min(fp8_values), max(fp8_values)
    rep_values = [entry[role]["mean"] for curve in disc["repeat_stats"].values()
                  for n, entry in curve.items() if n != "5" for role in ("fp8", "lora_weak")]
    rep_low, rep_high = min(rep_values), max(rep_values)
    exact_low, exact_high = exact_auc_range(metrics)
    mcc_lora = disc["mcc_proxy"]["lora_weak"]
    mcc_fp8 = disc["mcc_proxy"]["fp8"]
    score_gain_40 = SCORE_MODEL_WEIGHT * mcc_lora["40"]["mean"] * ASSUMED_LORA_PAIR_SHARE
    need_lora = disc["t_growth"]["ends_period"]["ref-09"]["n_for_target"]
    need_fp8 = disc["t_growth"]["ends_period"]["ref-02"]["n_for_target"]
    fresh_cells = {name: info.get("cells", 0) for name, info in metrics["fresh"]["groups"].items()}
    fresh_text = ", ".join(f"{name} : {count}" for name, count in fresh_cells.items())
    d4_tile = ""
    if d4_curve and "20" in d4_curve:
        verdict_word = "atteint" if d4_curve["20"]["lora_weak"]["mean"] >= D4_SUCCESS else "non atteint"
        d4_tile = (f'<div class="tile good"><div class="big">{d4_curve["20"]["lora_weak"]["mean"]:.2f}</div>'
                   f'<div class="lbl">AUC LoRA 0,3 contre S sur prompts NEUFS (d4, {d4["cells"]} cellules) à N = 20 ; '
                   f'fp8 : {d4_curve["20"]["fp8"]["mean"]:.2f}. Seuil figé {D4_SUCCESS:.2f} : {verdict_word}.</div></div>')
    tiles = f"""
<div class="tiles">
 <div class="tile"><div class="big">{period_fp8:.2f}</div><div class="lbl">AUC fp8 contre S à N = 40, fin par un point ; les autres statistiques vont de {fp8_low:.2f} à {fp8_high:.2f}. 0,5 = hasard.</div></div>
 <div class="tile acc"><div class="big">{period_lora:.2f}</div><div class="lbl">AUC LoRA 0,3 contre S à N = 40, fin par un point. En échantillon (critère repéré sur ces données) : à valider.</div></div>
 {d4_tile}
 <div class="tile warn"><div class="big">{held:.2f}</div><div class="lbl">AUC hors échantillon du meilleur critère criblé parmi environ 300 (il valait {optimistic:.2f} au criblage).</div></div>
 <div class="tile good"><div class="big">0</div><div class="lbl">requête ep envoyée. Gain de MCC ep attendu par le design actuel : non démontré, voir 1 et 6.</div></div>
</div>"""
    page1 = f"""
<section class="page1">
<h1>Model Equality, équipe D : ce qui trahit fp8 et LoRA 0,3</h1>
<p class="meta">Salve {salvo}, 21/11/2026. Mesures sur {disc['cells']} cellules (prompt, répétition) communes aux 9 endpoints ref
(équipes A, B et m-met-1), plus {answers_d} réponses fraîches de l'équipe D. Les extrapolations sont marquées <i>estimation</i>.</p>
{tiles}
<p><b>Leaderboard public Model Equality</b> ({lb['time']}) : nous {lb['score']:.2f} ({lb['rank']}), premiers {lb['best']:.2f} et {lb['second']:.2f}.
Score public sur 30 % des données : repère, jamais réglage.</p>
{_fig(figures, "dashboard_bars.png", f"À N = 40 réponses par endpoint, fp8 reste au hasard pour toutes les statistiques ; seule la fin de réponse par un point sépare LoRA 0,3 ({period_lora:.2f}), et ce critère a été repéré sur ces mêmes données. Barres noires : intervalle 2,5 - 97,5 % sur les tirages de cellules, large parce qu'il n'y a qu'un endpoint fp8, un LoRA 0,3 et 10 paires S-S.")}
</section>"""
    summary = f"""
<h2>1. Résumé décisionnel</h2>
<div class="summary"><ul>
<li><b>fp8 : rien à N = 5, 10, 20, 40, 80</b>, avec 22 statistiques de style, une direction apprise et même un classifieur « oracle » entraîné sur les étiquettes de fp8 (AUC {_auc(metrics, 'oracle_fp8', 40, 'fp8')[0]:.2f} à N = 40) : le texte de ref-02 ne se distingue pas de celui de S.</li>
<li><b>LoRA 0,3 : un seul signal, la fin de réponse par un point</b> (AUC {period_lora_20:.2f} / {period_lora:.2f} / {period_lora_80:.2f} à N = 20 / 40 / 80). Il ressort de la lecture des données d'équipes A, B et m-met-1 : sa validation sur prompts neufs est dans le point suivant (cellules fraîches communes aux 7 endpoints : {fresh_text}). La direction apprise sur LoRA 0,7 et 1,5 ne le capte pas (AUC {_auc(metrics, 'transfer', 40, 'lora_weak')[0]:.2f}). Recoupement indépendant (r-met-c, statistique de style sur ref) : fp8 et LoRA 0,3 restent entre 0,45 et 0,49 jusqu'à N = 20, seules les variantes fortes montent (0,60 à N = 1, 0,88 à N = 20) ; la fin par un point donne {period_lora_20:.2f} à N = 20, un peu au-dessus de leur plage, avec un intervalle qui contient 0,5 : le gain n'apparaît qu'à N = 40 et plus.</li>
{pilot_bullet}<li><b>Les ep échantillonnent comme les ref</b> (mesure de r-met-c : 33 réponses distinctes sur 33 endpoints au premier prompt long) : nos AUC ref s'appliquent aux ep. Les sept endpoints identiques sur les 5 prompts courts de e1 ne sont que le mode de la distribution. L'accord exact n'est donc pas la clé : sur ref, il place fp8 ({agree['ref-02']:.2f}) et LoRA 0,3 ({agree['ref-09']:.2f}) avec S plain ({agree['ref-01']:.2f}), et les déguisements s'en écartent plus ({agree['ref-08']:.2f} à {agree['ref-05']:.2f}). Ce qui sépare nettement les familles, c'est le style des réponses longues (observation r-met-c), pas les poids fins.</li>
<li><b>Recommandation</b> : (a) ne pas ajouter de réponses ep (+5, +10, +20) pour fp8 (AUC {period_fp8:.2f} à N = 40, 10 réponses de plus ne changent rien) ; (b) réponses ep nécessaires pour t = 3 sur la fin par un point (<i class="est">estimation</i> : t proportionnel à racine de N) : LoRA 0,3 environ {need_lora:.0f}, fp8 plus de {need_fp8:.0f}, contre 90 dépensables : hors de portée avec le design actuel ; MCC proxy hors échantillon sur les paires (LoRA 0,3, S) : {mcc_lora['20']['mean']:.2f} / {mcc_lora['40']['mean']:.2f} / {mcc_lora['80']['mean']:.2f} à N = 20 / 40 / 80 (fp8 : {mcc_fp8['40']['mean']:.2f} à N = 40) ; gain de score attendu, <i class="est">estimation</i> grossière et optimiste (famille de 11 endpoints répartis 4 / 2 / 2 / 2 / 1 : 8 paires (LoRA 0,3, S) sur 55) : environ {score_gain_40:.2f} à N = 40 ; passer de 90 à 100 réponses ne change rien à cette échelle ; (c) les batteries d'items (d3, d5) n'ont donné aucun signal pour LoRA 0,3 ni fp8 à ce stade ; l'arithmétique d'une ligne est le prompt à privilégier.</li>
<li><b>Décision demandée</b> : (1) autoriser, ou non, un palier ep de +10 puis +20 prompts arithmétiques d'une ligne (« Compute A op B. Answer with only the number. », prompts DISTINCTS) par endpoint, pour mesurer la part de réponses finissant par un point : c'est le seul critère retenu pour LoRA 0,3. Sur ref, {d4_ep_text}. Risque : la signature n'est pas vérifiée sur les familles Qwen et Llama des ep, et une partie des 90 réponses dépensables est déjà engagée. (2) Accepter que fp8 reste hors de portée. Ne jamais régler le critère sur le score public.</li>
</ul></div>"""
    leads = table_html(
        ["piste", "hypothèse et source", "ce qui a été fait", "résultat chiffré", "verdict"],
        [
            ["Direction LoRA apprise", "Un LoRA faible va dans le sens du LoRA fort (même adaptateur, force 0,3 / 0,7 / 1,5).",
             "Régression logistique mots + caractères, plis par prompt, 9 endpoints ref.",
             f"Marge moyenne : S -0,70 à -0,41, LoRA 0,3 {disc['transfer_mean_score']['ref-09']['mean']:.2f}, LoRA 0,7 {disc['transfer_mean_score']['ref-04']['mean']:.2f}, LoRA 1,5 {disc['transfer_mean_score']['ref-07']['mean']:.2f}. AUC LoRA 0,3 : {_auc(metrics, 'transfer', 40, 'lora_weak')[0]:.2f} à N = 40.",
             '<span class="verdict-drop">abandonner pour LoRA 0,3</span>'],
            ["Oracle (étiquettes connues)", "Borne haute : existe-t-il UNE direction de texte qui généralise ?",
             "Même classifieur entraîné sur les étiquettes de la cible, plis par prompt.",
             f"fp8 : {_auc(metrics, 'oracle_fp8', 80, 'fp8')[0]:.2f} à N = 80. LoRA 0,3 : {_auc(metrics, 'oracle_lora03', 40, 'lora_weak')[0]:.2f} (N = 40), {_auc(metrics, 'oracle_lora03', 80, 'lora_weak')[0]:.2f} (N = 80).",
             '<span class="verdict-drop">fp8 : abandonner</span> ; LoRA 0,3 : borne'],
            ["Criblage de 22 statistiques", "Un trait scalaire (point final, longueur, ponctuation) isole fp8 ou LoRA 0,3.",
             "AUC à N = 5 à 80 contre les 10 paires S-S ; expérience de sélection : choix sur la moitié des prompts, mesure sur l'autre.",
             f"Meilleure AUC criblée {optimistic:.2f}, hors échantillon {held:.2f} ({len(selection.get('held_out_auc', []))} découpages).",
             '<span class="verdict-drop">abandonner le criblage libre</span>'],
            ["Fin de réponse par un point", "Un LoRA fait écrire des phrases complètes (intuition, lecture des réponses).",
             "AUC et t en fonction de N ; contrôle sur prompts neufs (vagues d4 et d5, protocole écrit avant les réponses).",
             f"AUC LoRA 0,3 {period_lora_20:.2f} (N = 20), {period_lora:.2f} (40), {period_lora_80:.2f} (80) ; fp8 {period_fp8:.2f} (40).",
             period_verdict],
            ["Accord exact sur prompts à consensus fort", "Un token qui bascule (fp8, LoRA) se voit comme un désaccord là où les endpoints s'accordent.",
             "Consensus entre endpoints sans étiquette ; distance = taux de désaccord exact.",
             f"AUC {exact_low:.2f} à {exact_high:.2f} sur ref à N = 20 et 40 (inférieure à 0,5) : les déguisements désaccordent plus que fp8 et LoRA 0,3, qui s'alignent sur S plain. Les ep échantillonnent comme les ref (r-met-c) : le résultat s'applique.",
             '<span class="verdict-drop">abandonner pour fp8 et LoRA 0,3</span>'],
            ["Chaînes longues, tirages au hasard, batteries (d2, d3, d5)", "Beaucoup de décisions par réponse : la première divergence arrive tard, la distribution est mesurable (consigne du brief).",
             "Prompts neufs, graine fixée ; une requête porte 10 à 20 items, car le coût serveur est un plancher fixe par requête.",
             f"Batteries : pas de signal (d5 : {passed} cellule(s) sur {total} sortent de la plage ; d3 : voir les tableaux) ; chaînes longues (d2) non collectées faute de débit (cellules fraîches : {fresh_text}). Débit mesuré : 1 à 10 réponses par minute selon la charge.",
             '<span class="verdict-drop">abandonner les batteries ; chaînes longues : non testées</span>'],
            ["Logprobs, température, graine", "Accès direct aux logits ou décodage glouton.",
             "Lecture de l'énoncé.", "Énoncé : « temperature et seed sont ignorés, ne renvoie que du texte ».",
             '<span class="verdict-drop">impossible</span>'],
        ])
    metrics_section = f"""
<h2>3. Toutes les métriques</h2>
<p>Métrique de diagnostic (proxy local, pas le score officiel) : AUC qui classe les paires « fp8 contre S » et « LoRA 0,3 contre S » (positifs) au-dessus des 10 paires S contre S (négatifs, déguisements compris), à N réponses par endpoint, tirées sur les cellules communes, {metrics['replicates']} tirages. 0,5 = hasard.</p>
{auc_table(metrics)}
<h3>Collisions et diversité des répétitions</h3>
<p>Statistiques de distribution demandées par le brief : taux de collision (deux répétitions identiques après normalisation) et diversité (réponses distinctes sur répétitions), prompt par prompt, sur les {disc['repeat_prompts']} prompts à au moins 2 répétitions. AUC comme ci-dessus.</p>
{repeat_table(metrics)}
<p>Résultat : AUC entre {rep_low:.2f} et {rep_high:.2f}, donc sous le hasard à partir de N = 10 : la température 1,0 et les autres déguisements changent la diversité des répétitions plus que fp8 et LoRA 0,3, qui restent comme S plain. Aucun signal de distribution de ce type pour les deux cibles.</p>
<h3>Effet par réponse et croissance (estimations)</h3>
<p>d = moyenne / écart-type de la différence appariée entre un endpoint et la moyenne des autres S, sur la direction LoRA apprise. La colonne « N pour t = 3 » extrapole t = c racine de N (<i class="est">estimation</i> : l'effet est supposé garder sa taille).</p>
{effect_table(metrics)}
{separability_html(metrics)}
<h3>MCC proxy de la fin par un point (estimation hors échantillon sur les tirages)</h3>
<p>Règle de seuil sur la distance, seuil choisi sur une moitié des tirages et mesuré sur l'autre. Les 10 paires S-S sont « même modèle », les 5 paires (cible, S) « modèles différents ». C'est un proxy : il ignore les autres modèles et le niveau famille du vrai score.</p>
{mcc_table(metrics)}
<h3>Accord exact par niveau de consensus</h3>
{exact_table(metrics)}
<h3>Coût serveur mesuré</h3>
<p>Latence d'une requête : 8 à 9 s pour 1 token, 21 à 23 s pour 256 tokens (environ 8 s fixes + 0,05 s par token), mesurée à vide. Sous charge, le débit de l'équipe est descendu à 1 à 4 réponses par minute avec 2 à 3 places. Le coût est donc par requête : un prompt qui porte 10 à 20 items rapporte 10 à 20 fois plus d'information par seconde.</p>
"""
    visuals = f"""
<h2>4. Visuels</h2>
{_fig(figures, "auc_vs_n.png", "fp8 reste au hasard à tout N. LoRA 0,3 monte avec N seulement pour la fin par un point et l'oracle. Au-delà de N = 80 les tirages se recouvrent (228 cellules) : les bandes sous-estiment l'incertitude.")}
{_fig(figures, "dose_response.png", "La direction LoRA croît avec la force (0,7 puis 1,5), mais LoRA 0,3 et fp8 restent dans la plage des déguisements de S : cette direction ne peut pas servir à les isoler.")}
{_fig(figures, "effect_sizes.png", "Effet par réponse : LoRA 1,5 est hors de portée des déguisements, LoRA 0,3 est de la même taille que le système prompt ou le persona.")}
{_fig(figures, "effect_heatmap.png", "Chaque case est un t apparié contre les autres S. Les déguisements allument peu de cases, LoRA 0,3 en allume sur la ponctuation finale, fp8 sur aucune.")}
{_fig(figures, "t_growth.png", "Croissance de t avec N. Pour la fin par un point, LoRA 0,3 s'éloigne des déguisements ; pour la direction apprise, un déguisement (système prompt) la rejoint. Les lignes au-delà de N = 120 sont des estimations.")}
{_fig(figures, "selection.png", "Pourquoi le criblage libre est trompeur : le gagnant choisi sur une moitié des prompts perd tout son avantage sur l'autre moitié.")}
{_fig(figures, "exact.png", "Sur les prompts à consensus fort de ref, fp8 et LoRA 0,3 s'accordent avec le consensus autant que S plain, les déguisements moins : l'égalité exacte seule ne les sépare pas sur ce modèle bruité.")}
{_fig(figures, "latency.png", "Latence mesurée : un plancher fixe d'environ 8 s par requête. Mettre beaucoup d'items dans une requête est le levier de débit.")}
<h3>Prompts neufs (validation, protocole écrit avant les réponses)</h3>
<p>Vague d4 : 30 opérations arithmétiques tirées avec la graine 20261121 + 3 (10 additions, 10 soustractions, 10 multiplications), 1 réponse par endpoint sur 7 endpoints (S plain, système prompt, persona, watermark, température 1,0, fp8, LoRA 0,3). Statistique fixée à l'avance : part de réponses finissant par un point, AUC des paires LoRA 0,3 contre S au-dessus des paires S contre S. Succès : AUC &gt;= 0,65 à N = 20 (la courbe de découverte donne 0,61 sur tous les prompts, la famille arithmétique est plus favorable) ; échec : &lt;= 0,55. Seuils figés avant la première réponse d4 lue. Les cellules déjà reçues sont ci-dessous ; le critère n'est pas modifié en cours de route.</p>
{fresh_html(metrics, figures)}
<h3>Exemples de réponses (surligné : le point final)</h3>
{examples_html(metrics)}
"""
    orc120 = _auc(metrics, "oracle_fp8", 120, "fp8")[0]
    orc160 = _auc(metrics, "oracle_fp8", 160, "fp8")[0]
    rc = disc["row_counts"]
    pr = disc["period_rate"]
    pa = disc["period_rate_arith"]
    interpret = f"""
<h2>5. Interprétabilité</h2>
<ul>
<li><b>fp8</b> : la quantification ajoute une erreur fixe et aléatoire aux poids, sans direction commune entre prompts. Par token, le décalage de probabilité est de l'ordre de quelques pour cent, très inférieur au bruit d'échantillonnage d'un modèle à température élevée : aucun classifieur, même oracle, ne la voit à N &lt;= 80 (l'oracle monte à {orc120:.2f} puis {orc160:.2f} à N = 120 et 160, tirages très recouverts et au-delà du budget ep). Un basculement isolé de token (ex. « 3 pm » devenu « 3 PM » chez ref-02 seul) existe, mais un seul tirage ne le distingue pas d'un hasard d'échantillonnage.</li>
<li><b>LoRA</b> : l'adaptateur pousse vers des phrases complètes terminées par un point (S plain : {pr['ref-01']:.0%} des réponses, LoRA 0,3 : {pr['ref-09']:.0%}, LoRA 0,7 : {pr['ref-04']:.0%}). Le système prompt et le watermark imitent en partie ce trait sur les prompts arithmétiques ({pa['ref-03']:.0%} et {pa['ref-06']:.0%} contre {pa['ref-01']:.0%} pour S plain, {disc['period_rate_arith_cells']} cellules) : c'est le piège.</li>
<li><b>Déguisements</b> : ils changent le style plus que les poids ; c'est pourquoi une distance globale les classe « différents ». Un détecteur utile doit s'appuyer sur un trait que les déguisements ne touchent pas.</li>
<li><b>Limite d'identifiabilité</b> : un seul endpoint fp8 et un seul LoRA 0,3 dans ref. Un écart propre à cet endpoint ne se distingue pas d'un écart du modèle.</li>
</ul>"""
    ranking = table_html(
        ["piste", "gain attendu", "coût", "risque (sur-ajustement au public 30 %)"],
        [
            ["Valider la fin par un point sur prompts neufs (d4, d5)", "décide si LoRA 0,3 est atteignable", "0 (ref gratuits, environ 1 à 4 réponses par minute)", "moyen : critère repéré sur les données de découverte"],
            ["Prompts courts à point final (LoRA 0,3)", f"AUC {period_lora:.2f} à N = 40 en échantillon, MCC proxy {mcc_lora['40']['mean']:.2f} (à valider)", "0 si on réutilise le budget prévu", "moyen : critère repéré sur les données de découverte"],
            ["Batteries d'items (d3, d5)", "aucun signal à ce stade (d5 : réplication échouée)", "0 (ref gratuits)", "faible"],
            ["Réponses ep supplémentaires (+5, +10, +20) pour fp8", "nul tant que AUC = 0,5", "5 à 20 par endpoint", "élevé"],
        ], numeric_from=9)
    repro = f"""
<h2>7. Reproductibilité</h2>
<pre>cd &lt;worktree&gt;   # branche worktree-agent-a6ad10967eab0762a
.venv/bin/python -m hackathon.model_equality.research.d_report --salvo {salvo} \\
  --truth data/hackathon/hf/model_equality/practice_answers.csv \\
  --discovery artifacts/model_equality/reference_answers.jsonl artifacts/model_equality/reference_answers_b.jsonl \\
              artifacts/model_equality/reference_answers_m1.jsonl
.venv/bin/python -m hackathon.model_equality.research.d_collect --wave d3 --repeats 3 --parallel 3
node scripts/hackathon/report_to_pdf.mjs reports/phase2/model_equality/team-d/report-{salvo}.html</pre>
<p>Graines : tirages de cellules et plis 20261121, sélection 7, prompts 20261121. OMP_NUM_THREADS=2. Aucune requête ep.
SHA du code : <code>{git_head()}</code> ; durée de l'analyse complète : {metrics.get('duration_s', 'non mesurée')} s. Sorties : <code>reports/phase2/model_equality/team-d/</code> (report-{salvo}.html, .pdf, figures-{salvo}/, metrics-{salvo}.json).
Données : réponses des équipes A et B et de m-met-1 (lecture seule), {answers_d} réponses de l'équipe D (artifacts/model_equality/reference_answers_d.jsonl).<br>Comptes de lignes (découverte) : {rc['rows_in']} lignes lues, {rc['duplicates_dropped']} doublons de clé écartés (le store de m-met-1 contient déjà ceux de A et B), {rc['after_dedupe']} réponses distinctes, {rc['rows_used']} utilisées ({disc['cells']} cellules x 9 endpoints), {rc['rows_rejected']} rejetées car la cellule n'est pas répondue par les 9 endpoints, 0 ligne tronquée.</p>"""
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Equality, équipe D, salve {salvo}</title><style>{STYLE}{EXTRA_STYLE}</style></head><body><main>
{page1}{summary}
<h2>2. Pistes explorées</h2>{leads}
{metrics_section}{ep_structure_html(metrics, figures)}{visuals}{interpret}
<h2>6. Classement des pistes</h2>{ranking}
{repro}
</main></body></html>"""


def git_head() -> str:
    """Short SHA of the code that produced the report (`inconnu` when git is unavailable)."""
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return "inconnu"
    return done.stdout.strip() or "inconnu"


def localise_decimals(page: str) -> str:
    """French decimal comma in the text nodes (tags, CSS and image data are left alone)."""
    return re.sub(
        r">([^<>]+)<", lambda m: ">" + re.sub(r"(?<=\d)\.(?=\d)", ",", m.group(1)) + "<", page)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--salvo", type=int, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, nargs="+", required=True)
    parser.add_argument("--fresh", type=Path, default=D_STORE_PATH)
    parser.add_argument("--ep-answers", type=Path, default=None, help="ep answers file, read only")
    parser.add_argument("--refresh-fresh", action="store_true",
                        help="with --reuse-metrics: recompute the fresh-prompt part from the current d store")
    parser.add_argument("--reuse-metrics", type=Path, default=None,
                        help="metrics JSON of an earlier run: redraw figures and text without recomputing")
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--selection-splits", type=int, default=DEFAULT_SELECTION_SPLITS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis, draw the figures, write the HTML."""
    args = parse_args(argv)
    config = AnalysisConfig(tuple(args.discovery), args.fresh, args.truth, DEFAULT_ENDPOINTS,
                            args.replicates, args.selection_splits)
    if args.reuse_metrics is not None:
        metrics = json.loads(args.reuse_metrics.read_text(encoding="utf-8"))
        if args.refresh_fresh:
            metrics = refresh_fresh(config, metrics)
    else:
        started = time.perf_counter()
        metrics = run_analysis(config)
        metrics["duration_s"] = round(time.perf_counter() - started, 1)
    if args.ep_answers is not None:
        metrics["ep_structure"] = structure_metrics(list(AnswerStore(args.ep_answers).load().values()))
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / f"figures-{args.salvo}"
    make_all_figures(metrics, figures)
    (args.out / f"metrics-{args.salvo}.json").write_text(
        json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    answers_d = sum(1 for _ in args.fresh.open(encoding="utf-8")) if args.fresh.exists() else 0
    page = localise_decimals(build_html(metrics, figures, args.salvo, answers_d))
    path = args.out / f"report-{args.salvo}.html"
    path.write_text(page, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
