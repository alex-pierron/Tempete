"""Visual report v2 of m-met-1 (dashboard page, one figure per result, big tables in annex).

Run after `small_n_eval`, `e1_run`, `model_level` and `e1_v2` wrote their files into the analysis
directory:

    python -m hackathon.model_equality.research.report_v2 \
        --analysis artifacts/model_equality/analysis_e1 --out reports/phase2/model_equality/m-met-1

Offline: every number comes from committed stores and analysis files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS
from hackathon.model_equality.research import figures_v2
from hackathon.model_equality.research.e1_run import IDENTITY_PROMPT_ID
from hackathon.model_equality.research.report_b import STYLE
from hackathon.model_equality.research.report_m1 import (
    request_rate,
    selection_table,
    sensitivity,
)
from hackathon.model_equality.research.small_n_eval import load_ref_answers
from hackathon.model_equality.store import AnswerStore

V2_STATISTIC = "lenzr"
V1_STATISTIC = "pair"
BASE_REF_ANSWERS = 1654
"""Answers of teams A and B already in the store when this sweep started."""
TIER_EXTRA = (5, 10, 20)
TARGETED_ENDPOINTS = 10
TARGETED_ANSWERS = 10
CEILING = 0.52
EXAMPLE_ENDPOINTS = ("ep-01", "ep-24", "ep-05", "ep-13", "ep-02", "ep-16", "ep-23", "ep-20")
YESNO_EXAMPLE_ID = "cand-yesno-002"
EXTRA_CSS = """
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .big{font-size:34px;font-weight:700;line-height:1.1}
.card .lab{font-size:12.5px;color:var(--ink2);margin-top:4px}
.card.us .big{color:#2a78d6}.card.warn .big{color:#eb6834}.card.good .big{color:#0c7a50}
figure{background:#fcfcfb}
.take{font-size:13.5px;margin:2px 0 18px}
details{margin:10px 0}summary{cursor:pointer;font-weight:600}
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _table(headers: list[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def parse_leaderboard(text: str) -> tuple[list[tuple[str, float]], float, str]:
    """Top teams, our score and our rank for Model Equality, from `leaderboard.txt`."""
    line = next(row for row in text.splitlines() if row.startswith("| model_equality"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    top = [(m.group(1).strip(), float(m.group(2)))
           for m in re.finditer(r"([^,]+?) (\d\.\d+)(?:,|$)", cells[5])]
    return top, float(cells[2]), cells[1]


def _fig(path: Path, caption: str, take: str) -> str:
    return (f'<figure><img alt="{_esc(caption)}" src="{_png(path)}"></figure>'
            f'<p class="take"><b>A retenir :</b> {_esc(take)}</p>')


def _tier_rows(rows: list[dict[str, Any]], rate: float) -> list[dict[str, Any]]:
    by_n = {r["n_prompts"]: r for r in rows if r["statistic"] == V2_STATISTIC}
    base = by_n[5]
    tiers = []
    for extra in TIER_EXTRA:
        point = by_n[5 + extra]
        tiers.append({
            "label": f"+{extra} par ep (N={5 + extra})", "requests": extra * len(SCORED_ENDPOINTS),
            "minutes": extra * len(SCORED_ENDPOINTS) / rate / 60,
            "gain": point["mcc_one_class"] - base["mcc_one_class"],
            "gain_sd": float(np.hypot(point["mcc_one_class_sd"], base["mcc_one_class_sd"])),
        })
    return tiers


def build_report(analysis: Path, out_dir: Path, store_dir: Path) -> Path:
    """Write figures and `report.html` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    summary = json.loads((analysis / "e1_summary.json").read_text(encoding="utf-8"))
    level = json.loads((analysis / "model_level.json").read_text(encoding="utf-8"))
    calibration = json.loads((analysis / "calibration.json").read_text(encoding="utf-8"))
    scores_v2 = json.loads((analysis / "lenzr_scores_ep.json").read_text(encoding="utf-8"))
    labels_v1 = list(csv.DictReader((analysis / "labels_ep.csv").open(encoding="utf-8")))
    labels_v2 = list(csv.DictReader((analysis / "labels_ep_v2.csv").open(encoding="utf-8")))
    endpoints = [r["endpoint"] for r in labels_v1]
    families = [r["family"] for r in labels_v1]
    models_v2 = [r["model"] for r in labels_v2]
    family_dist = np.array([[float(v) for v in r[1:]] for r in list(csv.reader(
        (analysis / "distance_family_ep.csv").open(encoding="utf-8")))[1:]])
    model_dist = np.array([[float(v) for v in r[1:]] for r in list(csv.reader(
        (analysis / "distance_ep.csv").open(encoding="utf-8")))[1:]])
    ep_answers = list(AnswerStore(store_dir / "answers.jsonl").load().values())
    ref_store = AnswerStore(store_dir / "reference_answers_m1.jsonl").load()
    ref_answers = load_ref_answers(store_dir)
    top, our_score, our_rank = parse_leaderboard(
        (analysis / "leaderboard.txt").read_text(encoding="utf-8"))
    rate = request_rate(ep_answers)
    rows = level["rows"]
    v2_5 = next(r for r in rows if r["statistic"] == V2_STATISTIC and r["n_prompts"] == 5)
    v1_5 = next(r for r in rows if r["statistic"] == V1_STATISTIC and r["n_prompts"] == 5)
    tiers = _tier_rows(rows, rate)
    tau_family = summary["tau_family"]
    budget = {line.split()[0]: int(line.split()[2]) for line in
              (analysis / "budget_after_e1.txt").read_text(encoding="utf-8").splitlines()}
    over = sorted(e for e, used in budget.items() if used > 5)
    sens = sensitivity(ep_answers, model_dist, tau_family, summary["tau_model"])
    by_key = {(a.endpoint, a.prompt_id): a.response for a in ep_answers if a.repeat == 0}
    family_of = dict(zip(endpoints, families, strict=True))

    f_leader = figures_v2.leaderboard_figure(top, our_score, our_rank, fig_dir / "v2_leaderboard.png")
    f_budget = figures_v2.budget_figure(budget, 5, fig_dir / "v2_budget.png")
    f_dendro = figures_v2.dendrogram_figure(family_dist, endpoints, families, tau_family,
                                            fig_dir / "v2_dendrogram.png")
    f_heat_family = figures_v2.heatmap_figure(
        family_dist, endpoints, families, fig_dir / "v2_heat_family.png",
        "Deux blocs : famille A (Qwen) et B (Llama)", "distance famille")
    f_heat_model = figures_v2.heatmap_figure(
        model_dist, endpoints, families, fig_dir / "v2_heat_model.png",
        "Distance modele (ham+len) : peu de contraste a N=5", "distance modele ham+len")
    example_rows = [(e, family_of[e], by_key[(e, IDENTITY_PROMPT_ID)].replace("\n", " ")[:150],
                     by_key[(e, YESNO_EXAMPLE_ID)].replace("\n", " ")[:60])
                    for e in EXAMPLE_ENDPOINTS]
    f_examples = figures_v2.examples_figure(example_rows, fig_dir / "v2_examples.png")
    f_mcc = figures_v2.mcc_vs_n_figure(rows, (V1_STATISTIC, V2_STATISTIC, "ramble"),
                                       fig_dir / "v2_mcc_vs_n.png", CEILING)
    f_detect = figures_v2.detection_figure(rows, fig_dir / "v2_detection.png")
    f_tiers = figures_v2.tiers_figure(tiers, fig_dir / "v2_tiers.png")

    creators = Counter(str(v) for v in summary["ep_creators"].values())
    ref_new = len(ref_store) - BASE_REF_ANSWERS
    flagged = sorted(e for e, v in scores_v2["scores"].items() if v > scores_v2["threshold"])
    targeted = TARGETED_ENDPOINTS * TARGETED_ANSWERS
    low_margin = sorted(sens["margins"].items(), key=lambda kv: kv[1])[:TARGETED_ENDPOINTS - 2]
    submission_v1 = (store_dir / "submissions" / "submission_e1.csv").read_text(encoding="utf-8")
    submission_v2 = (store_dir / "submissions" / "submission_e1b.csv").read_text(encoding="utf-8")
    answers_rows = []
    for e in SCORED_ENDPOINTS:
        answers_rows.append([e] + [by_key.get((e, pid), "").replace("\n", " ")[:60]
                                   for pid in E1_PROMPT_IDS])
    cal_rows = [[r["statistic"], r["n_prompts"], _fr(r["mcc_one_class"]), _fr(r["mcc_one_class_sd"]),
                 _fr(r["auc_strong"]), _fr(r["auc_weak"]), _fr(r["threshold"])]
                for r in rows]
    gain_text = ", ".join(f"{t['label']} : {t['gain']:+.2f}".replace(".", ",") for t in tiers)

    def card(css: str, big: str, label: str) -> str:
        return f'<div class="card {css}"><div class="big">{big}</div><div class="lab">{label}</div></div>'

    body = f"""
<main>
<h1>Model Equality, m-met-1 : phase 2, rapport v2</h1>
<p class="meta">21/11/2026. E1 = 5 reponses ep par endpoint (165), balayage ref gratuit ({ref_new} reponses
nouvelles, store dedie). Aucune soumission faite par m-met-1.</p>
<div class="cards">
{card("us", _fr(our_score), f"score public E1 (soumis par l'equipe), rang {_esc(our_rank)}")}
{card("good", "+" + _fr(v2_5['mcc_one_class'] - v1_5['mcc_one_class']), f"gain de MCC modele sur les ref, v2 ({_fr(v2_5['mcc_one_class'])}) contre v1 ({_fr(v1_5['mcc_one_class'])}), N=5")}
{card("", f"{len(set(families))} / {len(set(models_v2))}", "familles / modeles trouves (candidat e1b)")}
{card("warn", "169", "reponses ep comptees pour 165 stockees (plafond 5 par ep ; 4 ep a 6)")}
{card("", str(ref_new), "reponses ref gratuites collectees (store m1)")}
</div>
{_fig(f_leader, "Classement public Model Equality", "la tete est a 0,40 contre 0,05 pour nous ; E1 separe des familles, pas des modeles, et le modele pese 0,9 du score.")}

<h2>1. Ce que E1 montre</h2>
{_fig(f_examples, "Exemples de reponses", "le createur cite et l'ecriture du oui/non (minuscule sans point chez Qwen, majuscule et point chez Llama) separent les familles ; ep-02, ep-16, ep-23 sont ambigus, classes en A par prudence.")}
{_fig(f_dendro, "Dendrogramme famille", f"un seul saut franc au-dela du maximum vu sur les ref ({_fr(tau_family, 3)}) : deux familles. Une troisieme apparait si l'on coupe entre 0,50 et 0,60.")}
{_fig(f_heat_family, "Carte de chaleur famille", "deux blocs carres bien distincts, un cluster ambigu de 3 a 8 endpoints dans le bloc A.")}

<h2>2. Une methode qui separe des modeles a petit N</h2>
<p>Le regroupement par distance (v1) dilue un LoRA qui ne bavarde que sur quelques prompts. La v2 note chaque
endpoint contre le reste de sa famille : ecart signe de longueur a la mediane des autres, en z robuste, coupe
fixee sur les ref a N=5. Les hors-normes deviennent chacun un modele ; le reste de la famille est un modele.</p>
{_fig(f_mcc, "MCC en fonction de N", f"sur les 9 ref (prompts de test disjoints de ceux de calibration) la v2 donne {_fr(v2_5['mcc_one_class'])} +- {_fr(v2_5['mcc_one_class_sd'])} a N=5 contre {_fr(v1_5['mcc_one_class'])} pour la v1 ; barres = ecart-type sur des tirages de prompts.")}
{_fig(f_detect, "Detection des variantes", "les LoRA forts sont detectes des N=5 ; fp8 et LoRA 0,3 montent avec N (AUC 0,68 a 0,9) mais ne reposent que sur 2 endpoints : a confirmer.")}
{_fig(f_heat_model, "Carte de chaleur modele", "la distance modele par paires a N=5 est quasi plate : c'est pourquoi la v1 ne coupait presque rien.")}

<h2>3. Gain de chaque palier ep et temps</h2>
{_fig(f_tiers, "Gain par palier", f"estimation sur les ref avec la v2 : {gain_text}. Barres = ecart-type de la difference (large). Duree = requetes / {_fr(rate)} requete par seconde mesures.")}
<p><b>E2 cible recommande :</b> {TARGETED_ANSWERS} nouvelles reponses sur {TARGETED_ENDPOINTS} endpoints
({targeted} requetes, {targeted / rate / 60:.0f} min) : les hors-normes de longueur proches du seuil et les
endpoints de famille ambigue ({_esc(', '.join(e for e, _ in low_margin))}), sur des prompts differents de E1.
Gain attendu : confirmer ou infirmer l'isolement de {len(flagged)} endpoints, soit la partie la plus risquee du MCC modele (estimation).</p>

<h2>4. Budget et incident</h2>
{_fig(f_budget, "Budget ep", "un ReadTimeout de 180 s a fait rejouer 4 requetes deja comptees (ep-09 a ep-12) ; le client ne rejoue plus apres un ReadTimeout sur un ep.")}

<h2>5. Candidats de soumission (non soumis)</h2>
<ul>
<li><b>submission_e1.csv</b> (v1, soumis, 0,05 public) : 2 familles, 2 modeles.</li>
<li><b>submission_e1b.csv</b> (v2) : {len(set(families))} familles, {len(set(models_v2))} modeles ; isole {len(flagged)} endpoints
hors-normes de longueur : {_esc(', '.join(flagged))}. Risque : si la composition des ep n'est pas celle des ref
(copies majoritaires), des endpoints sont isoles a tort ; la v2 n'ajoute que des separations, jamais de fusion.</li>
</ul>

<h2>Annexe</h2>
<details><summary>Methode et limites</summary>
<p>Famille : createur de l'identite (lexique : {_esc(dict(creators))}) et polarite plus style des deux oui/non, liaison moyenne,
coupe au maximum vu sur les ref. Modele v2 : z robuste de longueur relative dans la famille, coupe a
{_fr(scores_v2['threshold'])} ajustee sur la moitie des prompts des ref, MCC mesure sur l'autre moitie (3 coupures,
{level['units']} unites, {level['prompts']} prompts). Limites : 9 ref d'un seul modele de base ; MCC famille non mesurable ;
une seule composition (5 copies + 4 variantes) ; ecart-type 0,2 ; les 5 prompts E1 ont ete choisis sur ces ref.</p></details>
<details><summary>Table : MCC, AUC et seuil par statistique et N</summary>
{_table(["statistique", "N", "MCC", "ecart-type", "AUC fort", "AUC faible", "seuil"], cal_rows)}</details>
<details><summary>Table : classement des prompts (tirage 0 contre tirage 1)</summary>
{_table(["prompt", "tirage 0", "tirage 1", "retenu"], selection_table(ref_answers))}</details>
<details><summary>Table : les 5 reponses de chaque ep (tronquees)</summary>
{_table(["ep", *E1_PROMPT_IDS], answers_rows)}</details>
<details><summary>CSV v1 (soumis)</summary><pre>{_esc(submission_v1)}</pre></details>
<details><summary>CSV v2 (candidat e1b)</summary><pre>{_esc(submission_v2)}</pre></details>
<details><summary>Sensibilite</summary>
<p>Familles selon la coupe : {_esc({str(k): v for k, v in sens['family_counts'].items()})}. Prompt retire (familles, accord de Rand) :
{_esc({k: (int(v['n_families']), round(v['rand_vs_full'], 3)) for k, v in sens['dropped'].items()})}.</p></details>
<details open><summary>Reproduction</summary><pre>export PYTHONPATH=$PWD/src
python -m hackathon.model_equality budget
python -m hackathon.model_equality query-ep --phase ep-e1 --reserve 95 --yes     # 5 par ep, cap dur
python -m hackathon.model_equality query-ref --phase ref-m1 --answers artifacts/model_equality/reference_answers_m1.jsonl
python -m hackathon.model_equality.research.small_n_eval --out artifacts/model_equality/analysis_e1
python -m hackathon.model_equality.research.e1_run --calibration artifacts/model_equality/analysis_e1/calibration.json --out artifacts/model_equality/analysis_e1
python -m hackathon.model_equality.research.model_level --out artifacts/model_equality/analysis_e1
python -m hackathon.model_equality.research.e1_v2 --analysis artifacts/model_equality/analysis_e1 --model-level artifacts/model_equality/analysis_e1/model_level.json
python -m hackathon.model_equality make-submission --labels artifacts/model_equality/analysis_e1/labels_ep_v2.csv --out artifacts/model_equality/submissions/submission_e1b.csv
python -m hackathon.model_equality.research.report_v2 --analysis artifacts/model_equality/analysis_e1 --out reports/phase2/model_equality/m-met-1
node scripts/hackathon/report_to_pdf.mjs reports/phase2/model_equality/m-met-1/report.html</pre>
<p>Seeds : 20261121 (calibration, tirages de prompts). Reprise apres crash : stores append-only dedupliques
sur (endpoint, id, repeat).</p></details>
</main>"""
    page = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Model Equality m-met-1 v2</title>'
            f"<style>{STYLE}{EXTRA_CSS}</style></head><body>{body}</body></html>")
    report_path = out_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "leaderboard": {"top": top, "ours": our_score, "rank": our_rank},
        "tiers": tiers, "flagged": flagged, "request_rate": rate, "over_budget": over,
        "model_level_rows": rows, "calibration_rows": calibration["rows"],
    }, indent=1, default=str), encoding="utf-8")
    return report_path


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    args = parser.parse_args()
    print(build_report(args.analysis, args.out, args.store_dir))


if __name__ == "__main__":
    main()
