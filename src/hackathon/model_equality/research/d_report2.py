"""Short report of salvo 2 (team D): API audit, latency, band rule, strong LoRA, d6 echo, d7 replication.

Usage (repo root, venv of the repo)::

    python -m hackathon.model_equality.research.d_report2 --truth <practice_answers.csv> \
        --store artifacts/model_equality/reference_answers_d.jsonl \
        [--latency-raw /tmp/latency_raw.jsonl]

Writes `reports/phase2/model_equality/team-d/report-2.html` (+ `figures-2/`, `metrics-2.json`).
The thresholds applied to d6 and d7 were committed before any answer (see `d_salvo2.py`).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hackathon.model_equality.research.d_figures import (
    GRID,
    INK,
    INK_SECONDARY,
    LABEL_SIZE,
    MODEL_COLOURS,
    SURFACE,
    TICK_SIZE,
    _save,
    _style,
)
from hackathon.model_equality.research.d_report import localise_decimals
from hackathon.model_equality.research.d_salvo2 import (
    D6_FAILURE,
    D6_SUCCESS,
    D7_BAND_FPR_MAX,
    D7_BAND_THRESHOLD,
    D7_BAND_TPR_MIN,
    D7_SUCCESS,
    salvo2_metrics,
)
from hackathon.model_equality.research.dataset import load_truth
from hackathon.model_equality.research.report_b import STYLE, embed_png, table_html
from hackathon.model_equality.store import REPO_ROOT

REPORT_DIR = REPO_ROOT / "reports" / "phase2" / "model_equality" / "team-d"
DEFAULT_REPLICATES = 500
PRE_REGISTRATION_SHA = "8ffdb75"
HOLDOUT_SHA = "f53af85"
S_ENDPOINTS = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
ENDPOINT_LABELS = {
    "ref-01": "S plain", "ref-02": "fp8", "ref-03": "S system prompt", "ref-04": "LoRA 0,7", "ref-05": "S persona",
    "ref-06": "S watermark", "ref-07": "LoRA 1,5", "ref-08": "S température 1,0", "ref-09": "LoRA 0,3"}
API_AUDIT: tuple[tuple[str, str, str], ...] = (
    ("Champs de la réponse", "id (24 hex aléatoires), object, created = 0, model, choices[0] {index, message, finish_reason}, budget",
     "aucun champ usage, aucun logprobs"),
    ("finish_reason", "toujours « stop », même quand la réponse est coupée à max_tokens", "aucun signal de troncature"),
    ("En-têtes", "nginx standard + Content-Length (fonction du texte seul)", "aucun en-tête informatif"),
    ("Paramètres n, logprobs, top_logprobs, temperature, seed, stop, stream",
     "tous ignorés (temperature = 0 et seed = 1 deux fois : deux textes différents ; stream : JSON normal ; n = 2 : une réponse)",
     "décodage non contrôlable"),
    ("Schéma OpenAPI public", "ChatRequest = model, messages, max_tokens seulement", "pas de champ caché déclaré"),
    ("Erreurs", "400 « Invalid request (not counted) » générique ; 422 champ manquant ; modèle inconnu : liste des ids",
     "aucune fuite sur le modèle"),
    ("Entrée limite", "contenu vide ou espace accepté et répondu ; 900 tokens acceptés, 1 100 rejetés (limite 1 024)",
     "comportement normal"),
    ("Tokens spéciaux dans le prompt", "ChatML, Llama 3, [INST] : réponse banale", "pas de fuite de gabarit"),
)


def band_rows(table: Mapping[str, Any], s_endpoints: Sequence[str]) -> list[list[str]]:
    """Rows (threshold, N, TPR LoRA 0.3, mean FPR on the S copies, FPR fp8, FPR température 1,0)."""
    rows: list[list[str]] = []
    for threshold, per_n in table.items():
        for n, per_endpoint in per_n.items():
            fpr = float(np.mean([per_endpoint[e] for e in s_endpoints if e in per_endpoint]))
            rows.append([threshold, n, f"{per_endpoint['ref-09']:.2f}", f"{fpr:.2f}",
                         f"{per_endpoint['ref-02']:.2f}", f"{per_endpoint.get('ref-08', float('nan')):.2f}"])
    return rows


def figure_band(table: Mapping[str, Any], s_endpoints: Sequence[str], path: Path) -> None:
    """TPR of LoRA 0.3 and mean FPR on the S copies for each threshold and N."""
    thresholds = list(table)
    sizes = sorted({n for per_n in table.values() for n in per_n}, key=int)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.0 * len(thresholds), 3.8), sharey=True)
    for ax, threshold in zip(np.atleast_1d(axes), thresholds, strict=True):
        _style(ax)
        x = np.arange(len(sizes))
        tpr = [table[threshold][n]["ref-09"] for n in sizes]
        fpr = [float(np.mean([table[threshold][n][e] for e in s_endpoints if e in table[threshold][n]])) for n in sizes]
        ax.bar(x - 0.18, tpr, 0.34, color=MODEL_COLOURS["T2"], label="LoRA 0,3 (vrais positifs)")
        ax.bar(x + 0.18, fpr, 0.34, color=MODEL_COLOURS["S"], label="5 copies S (faux positifs)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"N = {n}" for n in sizes], fontsize=TICK_SIZE)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"seuil {threshold}", fontsize=12, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("part des tirages signalés", fontsize=LABEL_SIZE)
    np.atleast_1d(axes)[0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.suptitle("Règle en bande (taux de points >= seuil), prompts neufs d4", fontsize=12, x=0.01, y=1.02, ha="left")
    _save(fig, path)


def figure_auc(table: Mapping[str, Mapping[str, Mapping[str, float]]], statistic: str, targets: Sequence[str],
               title: str, path: Path, success: float | None = None) -> None:
    """AUC vs N of one statistic for several targets."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    for target in targets:
        per_n = table[statistic].get(target)
        if not per_n:
            continue
        sizes = sorted(per_n, key=int)
        colour = (MODEL_COLOURS["T1"] if target == "ref-02" else MODEL_COLOURS["T2"] if target == "ref-09"
                  else MODEL_COLOURS["S"])
        ax.plot([int(n) for n in sizes], [per_n[n] for n in sizes], "-o", color=colour, linewidth=2.0 if target in
                ("ref-02", "ref-09") else 1.2, alpha=1.0 if target in ("ref-02", "ref-09") else 0.6,
                label=ENDPOINT_LABELS[target])
    ax.axhline(0.5, color=INK_SECONDARY, linestyle=":", linewidth=1.0)
    if success is not None:
        ax.axhline(success, color=INK, linestyle="--", linewidth=1.0)
        ax.text(0.5, success + 0.01, "seuil figé", fontsize=9, color=INK)
    ax.set_xlabel("réponses par endpoint N", fontsize=LABEL_SIZE)
    ax.set_ylabel("AUC contre le groupe S", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    ax.set_title(title, fontsize=12, loc="left")
    _save(fig, path)


def figure_latency(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, dict[str, float]]:
    """Box plot of the latency of the interleaved probe; returns the median and count per endpoint."""
    by: dict[str, list[float]] = {}
    for row in rows:
        by.setdefault(str(row["endpoint"]), []).append(float(row["latency_s"]))
    order = [e for e in ENDPOINT_LABELS if e in by]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    _style(ax)
    ax.boxplot([by[e] for e in order], widths=0.55, patch_artist=True,
               boxprops={"facecolor": SURFACE, "edgecolor": INK}, medianprops={"color": MODEL_COLOURS["T4"]})
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([ENDPOINT_LABELS[e] for e in order], rotation=20, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("latence d'une requête (s)", fontsize=LABEL_SIZE)
    ax.grid(True, color=GRID, linewidth=0.6)
    _save(fig, path)
    return {e: {"median": float(np.median(by[e])), "n": float(len(by[e]))} for e in order}


def _fig(path: Path, caption: str) -> str:
    """Embedded figure with its caption (empty when the file is missing)."""
    if not path.exists():
        return ""
    return f'<figure><img alt="{html.escape(caption[:80])}" src="{embed_png(path)}"><figcaption>{caption}</figcaption></figure>'

HOLDOUT_FACTS = (
    "50 prompts h-arith-000 à 049 x 2 répétitions x 9 ref = 900 lignes, SHA256 2960ca5d614959b0956310860e222c223c491d4cc32ca5e2da314e5892891dc0 "
    "(commit fbcac47), non analysé par cette équipe : r-met-c y applique sa configuration figée."
)
IDENTITY2_FACTS = (
    "270 lignes, SHA256 3f40d80ee99bd538e936d8c9ee12178b40bb828e55b30142137957d7ae8312cb (commit f6f18a8), non lues par cette équipe : "
    "vague neuve pour que r-met-c fige son critère d'identité avant de la lire."
)
WRONG_MODEL_NOTICE = (
    "Les ref sont SmolLM2-360M-Instruct (mesure de r-met-e), pas le 135M : les tests de vraisemblance ci-dessous "
    "ont été faits avec la mauvaise copie locale et ne disent rien de la faisabilité de fp8."
)


def _read_json(path: Path) -> Any:
    """JSON file content, or None when the file does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def extra_sections(out: Path, figures: Path) -> list[str]:
    """Sections for the results of the salvo that do not come from the d-store: identity, MAUVE, likelihood, Lucas."""
    sections: list[str] = ["<h2>7. Autres résultats de la salve</h2>"]
    identity = _read_json(out / "identity-metrics.json")
    if identity:
        rows = [[ENDPOINT_LABELS.get(e, e), f"{r:.2f}", "oui" if identity["flagged"][e] else "non"]
                for e, r in identity["rates"].items()]
        sections += ["<h3>7a. Drapeau d'identité (SmolLM ou Hugging Face cités)</h3>",
                     _fig(figures / "identity.png", "Taux de réponses citant SmolLM ou Hugging Face, 3 formulations x 10 tirages par ref."),
                     table_html(["endpoint", "taux", "drapeau"], rows, numeric_from=1),
                     (f"<p>Verdict figé : {html.escape(str(identity['verdict']))} ; {identity['minimal_draws']} tirages suffisent. "
                     f"Vague neuve (identity2) : {IDENTITY2_FACTS}</p>")]
    mauve = _read_json(out / "mauve_results.json")
    if mauve:
        rows = [[ENDPOINT_LABELS.get(e, e), f"{v['median_mauve']:.3f}", "oui" if v["flagged"] else "non"]
                for e, v in mauve["endpoints"].items()]
        sections += ["<h3>7b. MAUVE (ré-implémentation, règle figée)</h3>",
                     (f"<p>Médiane de la référence (ref-01 contre ref-01) : {mauve['null_median']:.3f}, seuil {mauve['threshold']:.3f}. "
                     "Seule la LoRA 1,5 est signalée ; fp8 et LoRA 0,3 sont invisibles.</p>"),
                     table_html(["endpoint", "MAUVE médian", "signalé"], rows, numeric_from=1)]
    local = _read_json(out / "local_likelihood_results.json")
    if local:
        fp8 = local["fp8"]["w_tensor"]
        sections += ["<h3>7c. Vraisemblance locale et recherche de recette fp8 (RETIRÉ)</h3>",
                     (f"<p><strong>{html.escape(WRONG_MODEL_NOTICE)}</strong> Mesures avec le 135M : t fp8 contre plain = {fp8['t_fp8_vs_plain']:.2f} "
                     "(seuil figé 3) ; recherche de 9 variantes (protocole badf802) : meilleure sélection t = 1,35, confirmation t = 0,92. "
                     "À refaire sur le 360M (code prêt, mêmes critères).</p>")]
    lucas = _read_json(out / "lucas_results.json")
    if lucas:
        status = lucas.get("block_C_models", {}).get("status", {})
        sections += ["<h3>7d. Constitution de Lucas (blocs B et C)</h3>",
                     ("<p>Bloc B (4 questions x 5 tirages x 9 ref, 180 lignes) : seule la question de code a une réponse modale stable (part du mode 0,67) ; "
                     "les trois autres sont trop dispersées (0,22 à 0,40) pour marquer une famille. "
                     f"Bloc C étape 1 : {status.get('stage1_rows', 0)} lignes sur {status.get('stage1_rows_expected', 360)} ; "
                     "règles de rétention appliquées seulement quand tous les tirages d'une étape sont là.</p>")]
    sections.append(f"<h3>7e. Jeu tenu à l'écart</h3><p>{HOLDOUT_FACTS}</p>")
    return sections


def build_html(metrics: Mapping[str, Any], figures: Path, latency: Mapping[str, Any] | None,
               extras: Sequence[str] = ()) -> str:
    """The report page; `extras` are optional sections inserted before the reproducibility one."""
    band = metrics.get("d4_band")
    d6, d7, strong = metrics.get("d6_echo"), metrics.get("d7_replication"), metrics.get("d4_strong")
    tile_band = "n/a"
    if band:
        per = band[f"{D7_BAND_THRESHOLD:.1f}"]["10"]
        tile_band = f"{per['ref-09']:.2f} / {float(np.mean([per[e] for e in S_ENDPOINTS])):.2f}"
    tile_d6 = d6["verdict"] if d6 else "en attente"
    tile_d7 = d7["verdict"] if d7 else "en attente"
    tiles = f"""<div class="tiles">
<div class="tile good"><div class="big">0</div><div class="lbl">champ de l'API exploitable au-delà du texte (usage, logprobs, en-têtes, finish_reason, tokens spéciaux, limites)</div></div>
<div class="tile acc"><div class="big">{tile_band}</div><div class="lbl">règle en bande à N = 10, seuil {D7_BAND_THRESHOLD:.1f} : vrais positifs LoRA 0,3 / faux positifs moyens sur les 5 copies S (d4)</div></div>
<div class="tile"><div class="big">{tile_d6}</div><div class="lbl">d6, écho pour fp8 (seuil figé : AUC de la longueur &gt;= {D6_SUCCESS:.2f} à N = 20)</div></div>
<div class="tile"><div class="big">{tile_d7}</div><div class="lbl">d7, réplication du point final (2 répétitions, 30 prompts neufs)</div></div></div>"""
    audit = table_html(["élément testé", "ce que renvoie le serveur", "conclusion"],
                       [[html.escape(a), html.escape(b), html.escape(c)] for a, b, c in API_AUDIT], numeric_from=9)
    sections = [f"<h1>Model Equality, équipe D : salve 2 (API et validations gratuites)</h1>{tiles}",
                _fig(figures / "band.png", "Règle en bande sur prompts neufs : seuil 0,6 sépare LoRA 0,3 des copies S dès N = 20 ; la température 1,0 est le seul déguisement qui franchit le seuil à N = 10.")
                if (figures / "band.png").exists() else "",
                "<h2>1. Audit de l'API (0 requête ep)</h2>", audit,
                "<p>Le schéma OpenAPI public ne déclare que model, messages et max_tokens. Il ne reste que deux canaux latéraux candidats : la latence par requête et l'identifiant (aléatoire).</p>"]
    if latency:
        rows = [[ENDPOINT_LABELS[e], f"{v['median']:.1f}", f"{int(v['n'])}"] for e, v in latency.items()]
        sections += ["<h2>2. Latence (sonde entrelacée, 64 tokens)</h2>",
                     _fig(figures / "latency_probe.png", "Latence par endpoint : la dispersion due à la file du serveur recouvre toute différence entre endpoints."),
                     table_html(["endpoint", "médiane (s)", "n"], rows)]
    if band:
        sections += ["<h2>3. Règle en bande (seuil figé 0,6 par r-met-c)</h2>",
                     table_html(["seuil", "N", "vrais positifs LoRA 0,3", "faux positifs moyens (5 copies S)", "fp8", "température 1,0"],
                                band_rows(band, S_ENDPOINTS), numeric_from=1)]
    if strong:
        sections += ["<h2>4. LoRA fortes sur les prompts d4</h2>",
                     f"<p>{strong['cells']} cellules communes aux 9 ref.</p>",
                     table_html(["statistique", "cible", *[f"N = {n}" for n in ("5", "10", "20")]],
                                [[s, ENDPOINT_LABELS[t], *[f"{v.get(n, float('nan')):.2f}" for n in ("5", "10", "20")]]
                                 for s, per in strong["auc"].items() for t, v in per.items()], numeric_from=2)]
    if d6:
        sections += [f"<h2>5. Écho pour fp8 (d6, {d6['cells']} cellules) : {html.escape(d6['verdict'])}</h2>",
                     f"<p>Seuils figés avant les réponses (commit {PRE_REGISTRATION_SHA}) : AUC de la longueur contre le groupe S à N = 20 &gt;= {D6_SUCCESS:.2f} succès, &lt;= {D6_FAILURE:.2f} échec. Valeur mesurée : {d6['primary_auc_n20']}.</p>",
                     _fig(figures / "d6.png", "AUC de la longueur pour fp8 et pour chaque déguisement de S.")]
    if d7:
        sections += [f"<h2>6. Réplication du point final (d7, {d7['cells']} cellules) : {html.escape(d7['verdict'])}</h2>",
                     f"<p>Seuils figés (commit {PRE_REGISTRATION_SHA}) : AUC de LoRA 0,3 &gt;= {D7_SUCCESS:.2f} à N = 20, règle en bande {D7_BAND_THRESHOLD:.1f} avec vrais positifs &gt;= {D7_BAND_TPR_MIN:.1f} et faux positifs &lt;= {D7_BAND_FPR_MAX:.2f}. Mesuré : AUC {d7['auc_lora_n20']}, vrais positifs {d7['band_tpr_n20']}, faux positifs {d7['band_fpr_n20']}.</p>",
                     _fig(figures / "d7.png", "AUC du point final pour LoRA 0,3, fp8 et les déguisements, 60 cellules.")]
    if not d6:
        sections += ["<h2>5. Écho pour fp8 (d6) : non collecté</h2>",
                     (f"<p>Vague préparée et seuils figés (commit {PRE_REGISTRATION_SHA}) ; collecte repoussée derrière le jeu tenu à l'écart, "
                      "l'identité neuve et la constitution de Lucas, puis suspendue pour laisser les quatre places de la clé à la collecte payante de r-met-c.</p>")]
    if not d7:
        sections += ["<h2>6. Réplication du point final (d7) : non collectée</h2>",
                     (f"<p>Mêmes raisons ; seuils figés au commit {PRE_REGISTRATION_SHA}. Le point final reste validé sur d4 seulement.</p>")]
    sections += list(extras)
    sections += ["<h2>8. Seuils figés et reproductibilité</h2>",
                 f"<p>Jeu tenu à l'écart : liste commitée {HOLDOUT_SHA} avant toute réponse, analysé par r-met-c uniquement. d6 et d7 : seuils commités {PRE_REGISTRATION_SHA} avant toute réponse. Commandes : <code>python -m hackathon.model_equality.research.d_collect --wave d6|d7 ...</code> puis <code>python -m hackathon.model_equality.research.d_report2</code>.</p>"]
    body = "\n".join(s for s in sections if s)
    page = (f"<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Model Equality, équipe D, salve 2</title><style>{STYLE}.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:14px 0}}"
            f".tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}.tile .big{{font-size:30px;font-weight:700}}"
            f".tile .lbl{{font-size:12.5px;color:var(--ink2);margin-top:4px}}</style></head><body><main>{body}</main></body></html>")
    return localise_decimals(page)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--latency-raw", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the metrics, draw the figures, write report-2."""
    args = parse_args(argv)
    truth = load_truth(args.truth)
    metrics = salvo2_metrics(args.store, dict(truth.model), args.replicates)
    figures = args.out / "figures-2"
    figures.mkdir(parents=True, exist_ok=True)
    if metrics["d4_band"]:
        figure_band(metrics["d4_band"], S_ENDPOINTS, figures / "band.png")
    if metrics["d6_echo"]:
        figure_auc(metrics["d6_echo"]["auc"], "length", ["ref-02", "ref-03", "ref-05", "ref-06", "ref-08"],
                   "d6 : longueur, fp8 et déguisements", figures / "d6.png", D6_SUCCESS)
    if metrics["d7_replication"]:
        figure_auc(metrics["d7_replication"]["auc"], "ends_period", ["ref-09", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08"],
                   "d7 : fin par un point", figures / "d7.png", D7_SUCCESS)
    latency: dict[str, dict[str, float]] | None = None
    if args.latency_raw is not None and args.latency_raw.exists():
        rows = [json.loads(line) for line in args.latency_raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        latency = figure_latency(rows, figures / "latency_probe.png")
        metrics["latency"] = latency
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics-2.json").write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    page = build_html(metrics, figures, latency, extra_sections(args.out, figures))
    path = args.out / "report-2.html"
    path.write_text(re.sub(r"\n{3,}", "\n\n", page), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
