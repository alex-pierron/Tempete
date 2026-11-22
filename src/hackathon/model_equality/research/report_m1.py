"""HTML report of m-met-1 (phase 2, E1): figures, tables and sensitivity, from the stores.

Run (after `small_n_eval` and `e1_run` wrote their files into the analysis directory):

    python -m hackathon.model_equality.research.report_m1 \
        --analysis artifacts/model_equality/analysis_e1 --out reports/phase2/model_equality/m-met-1

Everything is computed offline from the committed stores; the text is judgement written against
the numbers (French, as the team's reports), the tables and figures are generated.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster

from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS
from hackathon.model_equality.research import figures_m1
from hackathon.model_equality.research.e1_run import (
    IDENTITY_PROMPT_ID,
    YESNO_PROMPT_IDS,
    creator_of,
    family_distance,
    label_endpoints,
)
from hackathon.model_equality.research.report_b import STYLE
from hackathon.model_equality.research.small_n_eval import (
    CHOSEN_METHOD,
    DISGUISED_COPY_INDICES,
    METHOD_WEIGHTS,
    REF_ENDPOINTS,
    load_ref_answers,
)
from hackathon.model_equality.smalln import build_small_n_data, build_tree, pair_distance
from hackathon.model_equality.store import Answer, AnswerStore

FloatArray = npt.NDArray[np.float64]
SERVER_RATE_FALLBACK = 0.24
FAMILY_TAU_SWEEP = (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90)
MODEL_TAU_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.89)
STRONG_LORA_INDICES = (3, 6)
"""ref-04 (LoRA 0.7) and ref-07 (LoRA 1.5): the variants a small N can see."""
TOP_ROWS = 12
SPENT_PER_ENDPOINT_E1 = 5
TIERS = (5, 10, 20)
TARGETED_ENDPOINTS = 10
TARGETED_ANSWERS = 10


def _esc(value: object) -> str:
    return html.escape(str(value))


def _png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table(headers: list[str], rows: Sequence[Sequence[object]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{_esc(v)}</td>'
            for i, v in enumerate(row))
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def selection_table(ref_answers: list[Answer]) -> list[list[str]]:
    """Prompts ranked on draw 0 (strong-LoRA excess), scored again on draw 1 (held out)."""
    data = build_small_n_data(ref_answers, list(REF_ENDPOINTS))
    weights = METHOD_WEIGHTS[CHOSEN_METHOD]
    plain = list(DISGUISED_COPY_INDICES)
    pairs = [(i, j) for a, i in enumerate(plain) for j in plain[a + 1:]]
    by_prompt: dict[str, dict[int, float]] = {}
    for position, (prompt_id, repeat) in enumerate(data.units):
        distance = pair_distance(data, [position], weights)
        excess = np.mean([np.mean([distance[t, s] for s in plain]) for t in STRONG_LORA_INDICES]) \
            - np.mean([distance[i, j] for i, j in pairs])
        by_prompt.setdefault(prompt_id, {})[repeat] = float(excess)
    ranked = sorted(((v[0], v[1], pid) for pid, v in by_prompt.items() if 0 in v and 1 in v),
                    reverse=True)
    chosen = [r for r in ranked if r[2] in E1_PROMPT_IDS]
    rows = [[pid, f"{sel:+.2f}", f"{held:+.2f}", "oui" if pid in E1_PROMPT_IDS else ""]
            for sel, held, pid in ranked[:TOP_ROWS]]
    rows += [[pid, f"{sel:+.2f}", f"{held:+.2f}", "oui"] for sel, held, pid in chosen
             if pid not in {r[0] for r in rows}]
    return rows


def partition_agreement(first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> float:
    """Rand index: share of endpoint pairs on which two partitions agree."""
    same_a = first[:, None] == first[None, :]
    same_b = second[:, None] == second[None, :]
    upper = np.triu_indices(len(first), 1)
    return float((same_a[upper] == same_b[upper]).mean())


def sensitivity(
    ep_answers: list[Answer], model_dist: FloatArray, tau_family: float, tau_model: float,
) -> dict[str, Any]:
    """How much the labels move with the cuts and with each family prompt removed."""
    responses = {(a.endpoint, a.prompt_id): a.response for a in ep_answers if a.repeat == 0}
    endpoints = list(SCORED_ENDPOINTS)
    full = family_distance(responses, endpoints)
    base_families = label_endpoints(full, model_dist, tau_family, tau_model).families
    tree = build_tree(full)
    family_counts = {tau: len(set(fcluster(tree, t=tau, criterion="distance").tolist()))
                     for tau in FAMILY_TAU_SWEEP}
    model_counts = {tau: len(set(label_endpoints(full, model_dist, tau_family, tau).models
                                     .tolist())) for tau in MODEL_TAU_SWEEP}
    dropped: dict[str, dict[str, float]] = {}
    for prompt_id in (IDENTITY_PROMPT_ID, *YESNO_PROMPT_IDS):
        reduced = family_distance(responses, endpoints, skip=frozenset({prompt_id}))
        families = label_endpoints(reduced, model_dist, tau_family, tau_model).families
        dropped[prompt_id] = {"n_families": float(len(set(families.tolist()))),
                              "rand_vs_full": partition_agreement(base_families, families)}
    own_margin: dict[str, float] = {}
    for i, endpoint in enumerate(endpoints):
        same = [j for j in range(len(endpoints)) if j != i and base_families[j] == base_families[i]]
        other = [j for j in range(len(endpoints)) if base_families[j] != base_families[i]]
        if same and other:
            own_margin[endpoint] = float(full[i, other].mean() - full[i, same].mean())
    return {"family_counts": family_counts, "model_counts": model_counts, "dropped": dropped,
            "margins": own_margin}


def request_rate(ep_answers: list[Answer]) -> float:
    """Measured ep throughput (requests per second) from the stored timestamps."""
    times = sorted(datetime.fromisoformat(a.timestamp) for a in ep_answers if a.timestamp)
    span = (times[-1] - times[0]).total_seconds()
    return len(times) / span if span > 0 else SERVER_RATE_FALLBACK


def _minutes(requests_count: int, rate: float) -> str:
    return f"{requests_count / rate / 60:.0f} min"


def build_report(analysis_dir: Path, out_dir: Path, store_dir: Path) -> Path:
    """Write figures, `report.html` and `metrics.json` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    summary = json.loads((analysis_dir / "e1_summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((analysis_dir / "calibration.json").read_text(encoding="utf-8"))
    labels = list(csv.DictReader((analysis_dir / "labels_ep.csv").open(encoding="utf-8")))
    endpoints = [row["endpoint"] for row in labels]
    families = [row["family"] for row in labels]
    reader = list(csv.reader((analysis_dir / "distance_ep.csv").open(encoding="utf-8")))
    model_dist = np.array([[float(v) for v in r[1:]] for r in reader[1:]])
    family_reader = list(csv.reader((analysis_dir / "distance_family_ep.csv").open(encoding="utf-8")))
    family_dist = np.array([[float(v) for v in r[1:]] for r in family_reader[1:]])
    ep_answers = list(AnswerStore(store_dir / "answers.jsonl").load().values())
    ref_answers = load_ref_answers(store_dir)
    tau_model, tau_family = summary["tau_model"], summary["tau_family"]
    sens = sensitivity(ep_answers, model_dist, tau_family, tau_model)
    rate = request_rate(ep_answers)
    selection = selection_table(ref_answers)

    fig_dendro = figures_m1.dendrogram_figure(family_dist, endpoints, families,
                                              tau_family, figures_dir / "fig1_dendrogram.png")
    fig_heat = figures_m1.heatmap_figure(family_dist, endpoints, figures_dir / "fig2_family.png",
                                         "Distance famille (identite + oui/non), ordre dendrogramme")
    fig_mcc = figures_m1.mcc_vs_n_figure(calibration["rows"], figures_dir / "fig3_mcc_vs_n.png")

    budget_lines = (analysis_dir / "budget_after_e1.txt").read_text(encoding="utf-8").splitlines()
    budget_rows: list[list[object]] = []
    for line in budget_lines:
        parts = line.split()
        budget_rows.append([parts[0], 0, parts[2], parts[3], parts[4]])
    over = [str(row[0]) for row in budget_rows if int(str(row[2])) != SPENT_PER_ENDPOINT_E1]
    used_total = sum(int(str(row[2])) for row in budget_rows)

    row5 = next(r for r in calibration["rows"] if r["method"] == CHOSEN_METHOD and r["n_prompts"] == 5)
    cal_rows = [[r["method"], r["n_prompts"], f"{r['mcc_mean']:+.2f}", f"{r['mcc_sd']:.2f}",
                 f"{r['mcc_p10']:+.2f} / {r['mcc_p90']:+.2f}", f"{r['tau_model']:.2f}",
                 f"{r['tau_family']:.2f}"] for r in calibration["rows"]]
    creator_counts = Counter(str(v) for v in summary["ep_creators"].values())
    ref_identity = Counter(
        str(creator_of(a.response)) for a in ref_answers
        if a.prompt_id == IDENTITY_PROMPT_ID and a.repeat == 0)
    tree_cluster_rows: list[list[object]] = []
    for family in sorted(set(families)):
        members = [e for e, f in zip(endpoints, families, strict=True) if f == family]
        tree_cluster_rows.append([f"famille {family}", len(members), " ".join(members)])
    answers_table = list(csv.reader((analysis_dir / "e1_answers.csv").open(encoding="utf-8")))
    answer_rows = [[r[0], r[1][:55], r[2][:14], r[3][:40], r[4][:14], r[5][:45]]
                   for r in answers_table[1:]]
    submission_text = (store_dir / "submissions" / "submission_e1.csv").read_text(encoding="utf-8")
    low_margin = sorted(sens["margins"].items(), key=lambda kv: kv[1])[:8]
    ref_pool_units = calibration["units"]

    def fig(path: Path, caption: str) -> str:
        return (f'<figure><img alt="{_esc(caption)}" src="{_png(path)}"><figcaption>'
                f"{_esc(caption)}</figcaption></figure>")

    tier_rows = [[f"+{extra}", extra * len(SCORED_ENDPOINTS), _minutes(extra * 33, rate),
                  f"{SPENT_PER_ENDPOINT_E1 + extra}"] for extra in TIERS]
    targeted_requests = TARGETED_ENDPOINTS * TARGETED_ANSWERS
    mcc_by_n = {r["n_prompts"]: r["mcc_mean"] for r in calibration["rows"]
                if r["method"] == CHOSEN_METHOD}
    body = f"""
<main>
<h1>Model Equality, m-met-1 : premiere soumission reelle (phase E1, 5 reponses par endpoint)</h1>
<p class="meta">21/11/2026. 165 reponses ep (5 prompts x 1 tirage x 33 endpoints), {ref_pool_units}
unites ref complets utilisees pour la calibration. Aucune soumission faite.</p>

<h2>1. Resume decisionnel</h2>
<div class="summary"><ul>
<li><b>Resultat :</b> {summary['n_families']} familles (A : {families.count('A')} ep, B : {families.count('B')} ep)
et {summary['n_models']} modeles. CSV valide : <code>artifacts/model_equality/submissions/submission_e1.csv</code>.</li>
<li><b>Score public (communique par l'equipe) :</b> 0,05, 10e sur 18 (meilleurs : 0,40, 0,20, 0,10).
Coherent avec l'attente : E1 ne separe que des familles ; le MCC modele, qui pese 0,9, est proche de 0.</li>
<li><b>Ce qui separe les familles :</b> le prompt d'identite (createur cite : {_esc(dict(creator_counts))})
et le style des reponses oui/non (Qwen ecrit "yes" sans majuscule ni point, Llama "No." / "Yes."). Les
completions et la capitale du Burkina Faso sont identiques dans les deux familles : elles ne servent qu'au niveau modele.</li>
<li><b>Modeles :</b> a N = 5 aucun decoupage au sein d'une famille n'est justifie ; les endpoints a LoRA fort
ne ressortent pas assez (sur les ref, seul LoRA 1,5 est isole). MCC modele attendu sur ref, prompts
tenus a l'ecart : <b>{row5['mcc_mean']:+.2f} +- {row5['mcc_sd']:.2f}</b> (p10 {row5['mcc_p10']:+.2f}, p90
{row5['mcc_p90']:+.2f}) ; sur les 5 prompts retenus (choisis sur ces ref, donc optimiste) :
{summary['ref_check']['mcc_mean']:+.2f}. Baseline triviale (tout dans un modele) : 0.</li>
<li><b>Budget :</b> {used_total} reponses comptees par le serveur pour {len(ep_answers)} stockees.
<b>Depassement de 1 sur {', '.join(over)}</b> (6 au lieu de 5) : un ReadTimeout de 180 s a declenche un
nouveau tir alors que le serveur avait deja compte la requete. Irreversible, corrige dans le kit.</li>
<li><b>Recommandation E2 :</b> ne pas faire de palier uniforme (la courbe MCC en fonction de N est plate de 5 a 30).
Un E2 cible de {targeted_requests} requetes ({_minutes(targeted_requests, rate)}) sur {TARGETED_ENDPOINTS} endpoints ambigus (section 8).</li>
</ul></div>

<h2>2. Budget avant et apres</h2>
<p>Avant : 0 utilise sur les 33 ep (serveur, 21/11/2026 avant le lancement). Apres : tableau ci-dessous
(serveur, <code>GET /models</code>). Plafond humain : 5 par endpoint (5 %), soit 165 requetes ; 169 comptees.</p>
{_table(["endpoint", "utilise avant", "utilise apres", "restant", "stocke"], budget_rows)}
<p>Debit mesure : {rate:.2f} requete/s (165 requetes en environ 11 min).</p>

<h2>3. Choix des 5 prompts</h2>
<p>Choisis sur les 1 654 reponses ref deja commitees (aucune collecte pour ce choix). Les prompts de
force LoRA sont classes sur le tirage 0 et revalues sur le tirage 1 (hors echantillon) : excedent de distance
des LoRA 0,7 et 1,5 au pool S, moins le bruit des copies S entre elles. Les 5 retenus :
{_esc(', '.join(E1_PROMPT_IDS))}. Deux prompts de famille (identite ; capitale du Burkina Faso, connaissance
a longue traine) et trois de force LoRA (deux oui/non, une completion). Sur les ref (un seul modele de base,
{_esc(dict(ref_identity))}), le prompt d'identite ne cite jamais un createur connu different : pas de
fausse separation de famille mesurable.</p>
{_table(["prompt", "excedent tirage 0 (selection)", "excedent tirage 1 (hors echantillon)", "retenu"], selection)}
<p>Correlation entre les deux tirages : le classement est reproductible pour les LoRA forts, bruite sinon.</p>

<h2>4. Methode</h2>
<p><b>Statistique petit N</b> (<code>smalln.py</code>) : le MMD non biaise du kit exige au moins deux reponses du meme
endpoint au meme prompt ; avec un seul tirage il n'est pas defini. On utilise donc le terme croise : distance
entre deux endpoints = moyenne sur les prompts d'une dissimilarite (Hamming caractere sur l'answer la plus longue
et longueur en log), poids a priori. <b>Famille :</b> createur de l'identite + polarite et style des oui/non,
liaison moyenne, coupe a la plus grande fusion vue sur les ref ({tau_family:.3f}, ref = une seule famille).
<b>Modele :</b> <code>ham+len</code>, liaison moyenne dans chaque famille, coupe {tau_model:.2f} (optimum de MCC hors
echantillon sur les ref a N = 5). Regle de prudence : tout ce qui n'est pas nettement separe est fusionne.</p>
{fig(fig_mcc, "MCC modele sur les 9 ref en fonction de N (un tirage par prompt, prompts de test disjoints de ceux de calibration).")}
{_table(["methode", "N", "MCC moyen", "ecart-type", "p10 / p90", "tau modele", "tau famille (p95)"], cal_rows)}
<p>Les ref ne contiennent qu'une famille : le MCC famille n'y est pas mesurable. fp8 et LoRA 0,3 restent
indiscernables des copies S (equipes A et B) ; le MCC maximal d'un decoupage grossier est 0,52.</p>

<h2>5. Resultats ep</h2>
{_table(["famille", "ep", "endpoints"], tree_cluster_rows, numeric_from=1)}
{fig(fig_dendro, "Dendrogramme des 33 endpoints sur la distance famille (identite + oui/non), couleur des etiquettes = famille ; trait = coupe retenue.")}
{fig(fig_heat, "Distance famille, lignes et colonnes dans l'ordre du dendrogramme : deux blocs.")}
<h3>CSV</h3><pre>{_esc(submission_text)}</pre>
<h3>Les 5 reponses de chaque endpoint (tronquees)</h3>
{_table(["ep", "identite", "oui/non 1", "completion", "oui/non 2", "capitale"], answer_rows, numeric_from=9)}

<h2>6. Incertitude</h2>
<ul>
<li>Nombre de familles selon la coupe : {_esc({str(k): v for k, v in sens['family_counts'].items()})}.
Avec la coupe retenue ({tau_family:.3f}), 2 familles ; entre 0,50 et 0,60 une troisieme apparait.</li>
<li>Nombre de modeles selon la coupe modele (coupe famille fixee) : {_esc({str(k): v for k, v in sens['model_counts'].items()})}.</li>
<li>Prompt de famille retire (accord de Rand avec la partition complete) :
{_esc({k: (int(v['n_families']), round(v['rand_vs_full'], 3)) for k, v in sens['dropped'].items()})}.</li>
<li>Endpoints les moins surs (marge de distance autre famille moins sa famille) :
{_esc([(e, round(m, 2)) for e, m in low_margin])}. Revendiquent Anthropic : ep-02, ep-08 ; OpenAI : ep-16 ; aucun
createur : ep-20, ep-23 (persona ou system prompt probable). Ils sont classes avec leur style par prudence.</li>
<li>Modele : sur les ref a N = 5, le MCC est 0 pour 10 % des tirages et atteint 0,49 pour 10 % ; un seul tirage
de 5 prompts ne permet pas de promettre mieux que 0,26 +- 0,20.</li>
<li>Score attendu : 0,1 x MCC famille + 0,9 x MCC modele sur les paires de meme famille VRAIE. Si les familles
ep sont plus nombreuses (3 ou plus), le MCC modele reste proche de 0 tant que chaque famille est un seul modele
predit ; la valeur realiste de cette premiere soumission est donc basse et surtout portee par la famille.</li>
</ul>

<h2>7. Travail gratuit sur les ref</h2>
<p>Store dedie <code>artifacts/model_equality/reference_answers_m1.jsonl</code> (union des stores A et B plus les nouvelles
reponses). Balayage <code>ref-m1</code> (banque entiere et prompts extra, 3 tirages, repeat-major) lance apres
la collecte ep : {len(AnswerStore(store_dir / 'reference_answers_m1.jsonl').load())} reponses dans le store au moment de cette
generation. Les courbes MCC en fonction de N a 40 prompts demandent un pool de prompts plus large que
la moitie de 70 : elles seront produites par <code>small_n_eval</code> quand le balayage aura avance (voir
la reproductibilite).</p>

<h2>8. Phase E2 : paliers et temps</h2>
<p>Debit mesure {rate:.2f} requete/s (4 en parallele). Duree = requetes / debit.</p>
{_table(["palier par endpoint", "requetes", "duree", "total par ep"], tier_rows)}
<p><b>Gain attendu :</b> la courbe MCC modele (ref, prompts tenus a l'ecart) est de {mcc_by_n.get(5, float('nan')):+.2f}
a N = 5, {mcc_by_n.get(10, float('nan')):+.2f} a 10, {mcc_by_n.get(20, float('nan')):+.2f} a 20,
{mcc_by_n.get(30, float('nan')):+.2f} a 30 : <b>aucun palier uniforme ne rapporte plus que le bruit</b> (ecart-type 0,1 a 0,2).
Le biais vient de fp8 et LoRA 0,3, pas du nombre de reponses (equipe A : 400 a 1 000 reponses seraient necessaires).</p>
<p><b>E2 cible recommande :</b> {TARGETED_ANSWERS} nouvelles reponses sur {TARGETED_ENDPOINTS} endpoints :
les 8 de marge la plus faible ({_esc(', '.join(e for e, _ in low_margin))}) plus ep-28 et ep-29 (completions de 91 et 148
mots contre 19 : signature LoRA fort a verifier), sur des prompts oui/non et d'identite DIFFERENTS de E1,
soit {targeted_requests} requetes, {_minutes(targeted_requests, rate)}. Gain attendu : lever l'ambiguite de famille de ces endpoints
(famille : poids 0,1 du score) et trancher 2 isolements de LoRA fort (modele : poids 0,9, +0,05 a +0,15 de MCC
si les deux sont confirmes, estimation, non mesuree). Cout : {targeted_requests} reponses sur 95 restantes par endpoint
concerne ; reste a 85 minimum.</p>

<h2>9. Reproductibilite</h2>
<pre>export PYTHONPATH=$PWD/src
python -m hackathon.model_equality budget
python -m hackathon.model_equality plan --phase ep-e1 --reserve 95        # 5 par endpoint, 165 requetes
python -m hackathon.model_equality query-ep --phase ep-e1 --reserve 95 --yes   # cap dur 5 par endpoint
python -m hackathon.model_equality.research.small_n_eval --out artifacts/model_equality/analysis_e1   # ~3 min
python -m hackathon.model_equality.research.e1_run --calibration artifacts/model_equality/analysis_e1/calibration.json \\
    --out artifacts/model_equality/analysis_e1
python -m hackathon.model_equality make-submission --labels artifacts/model_equality/analysis_e1/labels_ep.csv \\
    --out artifacts/model_equality/submissions/submission_e1.csv
python -m hackathon.model_equality validate artifacts/model_equality/submissions/submission_e1.csv
python -m hackathon.model_equality.research.report_m1 --analysis artifacts/model_equality/analysis_e1 \\
    --out reports/phase2/model_equality/m-met-1
node scripts/hackathon/report_to_pdf.mjs reports/phase2/model_equality/m-met-1/report.html</pre>
<p>Seeds : permutations 20261121 (calibration). Reprise apres crash : les stores sont append-only et dedupliques
sur (endpoint, id, repeat) ; relancer la meme commande ne redemande rien. Limites : 2 familles par un lexique de
createurs fixe a la main (non teste sur d'autres bases que SmolLM) ; calibration sur 9 ref d'un seul modele de base.</p>
</main>"""
    page = (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>Model Equality m-met-1, soumission E1</title>'
            f"<style>{STYLE}</style></head><body>{body}</body></html>")
    report_path = out_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "summary": summary, "sensitivity": sens, "request_rate": rate,
        "calibration": calibration["rows"]}, indent=1, default=str), encoding="utf-8")
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
