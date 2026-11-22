"""Narrative of the team-B report (v2, final data of the session).

Numbers are read from `metrics.json` wherever possible; the judgement sentences were written
against the final data (1654 answers, 184 per endpoint, 79 prompts) and say so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from hackathon.model_equality.research.candidates import candidate_bank

HEADLINE_KERNEL = "hamming_char"
BASE_ENDPOINTS: tuple[str, ...] = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
HARD_ENDPOINTS: tuple[str, ...] = ("ref-02", "ref-09")


def auc(res: Mapping[str, Any], role: str, repeats: int, statistic: str, budget: int,
        kernel: str = HEADLINE_KERNEL) -> str:
    """AUC of a curve point as text, `n/a` when the point does not exist."""
    for point in res["repeat_vs_diversity"].get(f"{kernel}/{role}/r{repeats}/{statistic}", []):
        if point["budget"] == budget:
            return f'{point["auc_mean"]:.2f}'
    return "n/a"


def style_t(res: Mapping[str, Any], subset: str, feature: str, endpoint: str) -> str:
    """t statistic of one endpoint on one style feature, as text."""
    for row in res["style"][subset]:
        if row["feature"] == feature:
            value = row["t"].get(endpoint)
            return "n/a" if value is None or math.isnan(value) else f"{value:.1f}"
    return "n/a"


def agree(res: Mapping[str, Any], kernel: str, key: str) -> str:
    """Mean agreement of one pair kind for one kernel."""
    return f'{res["agreement"][kernel][key]:.3f}'


def counts(table: Mapping[str, list[int]], endpoints: tuple[str, ...]) -> str:
    """`hits/total` summed over endpoints."""
    hits = sum(table.get(e, [0, 0])[0] for e in endpoints)
    total = sum(table.get(e, [0, 0])[1] for e in endpoints)
    return f"{hits}/{total}"


def _verdict(kind: str, label: str) -> str:
    return f'<span class="verdict-{kind}">{label}</span>'


def sections(res: Mapping[str, Any], provenance: Mapping[str, str]) -> dict[str, str]:
    """Every narrative block of the report."""
    per_endpoint = res["answers_per_endpoint"]
    total = sum(per_endpoint.values())
    per_ep = f"{min(per_endpoint.values())}-{max(per_endpoint.values())}"
    n_prompts = len(res["prompts"])
    n_candidates = len(candidate_bank())
    reliability = res["reliability"]
    rel = reliability[HEADLINE_KERNEL]
    rel_best = max(v["spearman_r0_r1"] for v in reliability.values())
    adaptive = res.get("adaptive") or {}
    echo = res["echo_compliance"]["strict"]
    echo_t1, echo_t2 = counts(echo, ("ref-02",)), counts(echo, ("ref-09",))
    echo_t3, echo_t4 = counts(echo, ("ref-04",)), counts(echo, ("ref-07",))
    echo_s_all = counts(echo, BASE_ENDPOINTS)
    yesno, numeric = res["adherence"]["yesno"], res["adherence"]["numeric"]
    t = {e: style_t(res, "all prompts", "log_words", e) for e in ("ref-03", "ref-02", "ref-09", "ref-04", "ref-07")}
    clean = {e: style_t(res, "all prompts", "ends_cleanly", e)
             for e in ("ref-03", "ref-02", "ref-09", "ref-04")}
    short_len = {e: style_t(res, "short-answer prompts", "log_words", e) for e in ("ref-03", "ref-09")}
    nulls = [style_t(res, "all prompts", "log_words", e) for e in BASE_ENDPOINTS if e != "ref-03"]

    summary = f"""<ul>
<li><b>Marche : LoRA 0,7 et 1,5.</b> Séparés de S dès 12 réponses par endpoint (AUC
{auc(res, "easy", 1, "pair", 12)}), 0,98 à 40 réponses (paires) et 1,00 contre le pool S ; longueur
de réponse : t = {t["ref-04"]} (0,7) et {t["ref-07"]} (1,5). Les prompts oui/non et choix multiple
suffisent (T4 : 0 réponse courte sur 12 oui/non ; T3 : 7 sur 12).</li>
<li><b>Ne marche pas : fp8 et LoRA 0,3.</b> AUC {auc(res, "hard", 1, "pair", 60)} (paires) et
{auc(res, "hard", 2, "pool", 120)} (contre le pool S) jusqu'à 120 réponses par endpoint ; accord
Hamming S-T1 {agree(res, HEADLINE_KERNEL, "S_T1")} et S-T2 {agree(res, HEADLINE_KERNEL, "S_T2")}
contre {agree(res, HEADLINE_KERNEL, "S_S")} entre endpoints S : l'écart (au plus 0,02) est du même ordre
que le bruit des déguisements, fp8 et LoRA 0,3 ne sont pas plus éloignés de S que deux S entre eux. Fiabilité entre tirages du score par prompt :
Spearman {rel["spearman_r0_r1"]:.2f} sur {rel["prompts_used"]} prompts (meilleur noyau
{rel_best:.2f}) : aucun prompt « dur » à recommander.</li>
<li><b>« Here is the sentence: » (LoRA 0,3) : réfuté.</b> Sur 12 prompts de copie exacte, 2 à 3 tirages,
la conformité stricte est de {echo_t2} pour LoRA 0,3 (fp8 : {echo_t1}) contre {echo_s_all} pour les cinq S ;
le préambule est un événement aléatoire, présent aussi chez watermark et température 1,0.
Seuls les LoRA forts décrochent (0,7 : {echo_t3} ; 1,5 : {echo_t4}).</li>
<li><b>Plan <code>ep</code>.</b> Dépenser au plus 54 réponses par endpoint en deux phases
(24 prompts courts contraints x 1 tirage, puis 30 adaptatifs), car la courbe plafonne à N ~ 40 pour
les LoRA forts et ne décolle pas pour fp8 / LoRA 0,3 ; garder 36 réponses en réserve (section 7).</li>
<li><b>Décisions demandées.</b> (1) Acter que fp8 et LoRA 0,3 resteront probablement classés avec S
(perte assumée sur MCC modèle) ou financer une autre statistique ; (2) valider le plan à 54 réponses
plutôt que 90 ; (3) intégrer <code>yesno</code> et <code>choice</code> (pas <code>echo</code>) à la
banque officielle ; (4) dire si l'on cherche un signal fp8 / LoRA 0,3 hors texte (JURECA).</li>
</ul>"""

    leads = f"""<table><thead><tr><th>Piste</th><th>Hypothèse et source</th><th>Fait</th>
<th>Résultat chiffré</th><th>Verdict</th></tr></thead><tbody>
<tr><td>1. Pouvoir discriminant par prompt et catégorie</td>
<td>Les perturbations fp8 / LoRA changent peu les logits ; certains prompts les amplifient (kit,
papier MET).</td>
<td>AUC (T contre S, bruit = paires S-S) par groupe et noyau, bootstrap sur les prompts ; score par
prompt ; split-half entre tirages ; {n_prompts} prompts, {total} réponses ({per_ep} par endpoint).</td>
<td>LoRA 0,7 / 1,5 : AUC 1,0 sur yesno, choice, style, completion, fact, code (Hamming). fp8 / LoRA 0,3 :
AUC 0,40-0,48 à tous les budgets ; sur 80 combinaisons groupe x noyau une seule dépasse 0,5 de façon
significative (bank:completion / longueur, 0,85 [0,52 ; 0,95]), ce qui est le nombre attendu par hasard (~4).
Fiabilité par prompt {rel["spearman_r0_r1"]:.2f}.</td><td>{_verdict("keep", "garder")} pour LoRA forts ;
{_verdict("drop", "abandonner")} pour fp8 / LoRA 0,3 au niveau du texte</td></tr>
<tr><td>2. Nouveaux prompts ciblés</td>
<td>Le modèle échantillonne chaud (même <code>487 * 639</code> donne un résultat différent par
endpoint) : viser faible cardinalité (oui/non, lettre) ou signal de style ; copie exacte (echo)
après un premier indice sur LoRA 0,3.</td>
<td>{n_candidates} candidats en 9 familles, interrogés sur les 9 ref (2 à 3 tirages).</td>
<td>yesno / choice : les LoRA forts décrochent, T1 / T2 non. echo : T2 {echo_t2} contre S {echo_s_all},
aucun effet. recite, arith diffus, style libre : bruit.</td>
<td>{_verdict("keep", "garder")} yesno, choice ; {_verdict("drop", "abandonner")} echo, recite,
arith diffus</td></tr>
<tr><td>3. Répétitions contre diversité</td>
<td>L'estimateur MMD non biaisé exige &gt;= 2 tirages par prompt ; la diversité apporte de
l'information indépendante.</td>
<td>À budget égal N, courbes « N x 1 », « N/2 x 2 », « N/3 x 3 », deux statistiques, 300
rééchantillonnages de prompts.</td>
<td>LoRA forts : indifférent (N = 40 : {auc(res, "easy", 1, "pair", 40)} contre
{auc(res, "easy", 2, "pair", 40)} ; N = 12 : {auc(res, "easy", 1, "pair", 12)} contre
{auc(res, "easy", 2, "pair", 12)}). fp8 / LoRA 0,3 : rien à départager. À 3 tirages seuls 15 prompts
sont éligibles : intervalles non concluants.</td><td>{_verdict("keep", "garder")} : 1 tirage par
prompt tant que le kit n'exige pas le terme intra-endpoint</td></tr>
<tr><td>4. Stratégie de budget ep</td>
<td>Allocation adaptative vers les endpoints ambigus avec critère d'arrêt.</td>
<td>Simulation sur les ref : seuil calibré sur des prompts disjoints, paire « résolue » à 2
erreurs-types du seuil.</td>
<td>Dépense moyenne {adaptive.get("adaptive_mean_spend", float("nan")):.0f} prompts contre
{adaptive.get("cap_prompts", float("nan"))} ; erreur sur paires difficiles
{adaptive.get("adaptive_hard_pair_error", float("nan")):.2f} contre
{adaptive.get("uniform_hard_error_at_same_spend", float("nan")):.2f} en dépense uniforme : pas de gain de
précision, économie de ~20 % de dépense.</td><td>{_verdict("keep", "garder")} comme règle d'arrêt</td></tr>
<tr><td>5. Canaux de style</td>
<td>Un LoRA, même faible, décale le <i>comment</i> sur chaque prompt ; une petite dérive répétée
s'additionne.</td>
<td>t à un échantillon sur 8 traits, unité = prompt, contre le pool S ; les déguisements sont les
témoins nuls.</td>
<td>Longueur : T4 {t["ref-07"]}, T3 {t["ref-04"]}, T2 {t["ref-09"]}, T1 {t["ref-02"]}. Mais le system prompt
sort aussi (longueur {t["ref-03"]} ; {short_len["ref-03"]} sur prompts courts) : les « témoins » ne sont pas
tous nuls. T2 : ponctuation finale propre t = {clean["ref-09"]}, comparable à system ({clean["ref-03"]}).</td>
<td>{_verdict("keep", "garder")} T3 / T4 ; {_verdict("dig", "creuser")} T2 (confondu avec system prompt)</td></tr>
<tr><td>6. Neutraliser persona et system prompt</td>
<td>« Réponds uniquement par... » doit écraser les déguisements.</td>
<td>Familles <code>neutral</code>, <code>echo</code>, <code>yesno</code>, <code>numeric</code> ; accord moyen
entre endpoints.</td>
<td>Persona invisible sur ces prompts. Le <b>system prompt</b> fuit : numeric court {numeric["ref-03"][0]}/{numeric["ref-03"][1]}
contre 6-8/12 pour les autres S. Watermark : préambule ; température 1,0 : dérive. Sur le premier mot,
l'accord S-S ({agree(res, "onehot_first", "S_S")}) est plus bas qu'au sein d'un même endpoint
({agree(res, "onehot_first", "within_S")}) : les déguisements changent bien le premier mot.</td>
<td>{_verdict("keep", "garder")} yesno, choice (S 7-8/8 contre T4 0/8)</td></tr>
</tbody></table>"""

    metrics_intro = f"""<p>Les <code>ref-*</code> servent de vérité terrain (S : plain, system prompt, persona,
watermark, température 1,0 ; T1 fp8 ; T2 LoRA 0,3 ; T3 LoRA 0,7 ; T4 LoRA 1,5). Données : {total}
réponses, {per_ep} par endpoint, {n_prompts} prompts : mes {n_candidates} candidats (2 à 3 tirages) et
les prompts de la banque interrogés par l'équipe A (lecture seule). Les métriques sont des <b>proxys
locaux</b> (AUC de séparation sur 9 endpoints : 10 paires « même modèle », 10 paires « difficiles »), pas le
score officiel. Intervalles : 5-95 % sur le tirage des prompts, pas sur les endpoints.</p>"""

    group_intro = """<p>AUC = probabilité qu'une paire (T, S) soit plus éloignée qu'une paire (S, S) ; les
paires (S, S) portent le bruit des déguisements. 0,5 = hasard ; au-dessous de 0,5, les endpoints S déguisés
sont plus éloignés entre eux que T1 / T2 ne le sont de S. Groupes <code>bank:*</code> : prompts officiels
(équipe A). 16 groupes x 5 noyaux = 80 intervalles à 90 % : environ 4 passeront 0,5 par hasard.</p>"""

    agreement_intro = f"""<p>Accord moyen du noyau (1 = identique) entre deux tirages d'un même endpoint S,
entre deux endpoints S différents (bruit des déguisements) et entre S et chaque modèle T. Sur Hamming :
même endpoint {agree(res, HEADLINE_KERNEL, "within_S")}, S-S {agree(res, HEADLINE_KERNEL, "S_S")}, S-T1
{agree(res, HEADLINE_KERNEL, "S_T1")}, S-T2 {agree(res, HEADLINE_KERNEL, "S_T2")}, S-T3
{agree(res, HEADLINE_KERNEL, "S_T3")}, S-T4 {agree(res, HEADLINE_KERNEL, "S_T4")}. L'effet fp8 / LoRA 0,3
est nul à l'échelle de l'accord, tandis que le coût des déguisements (même endpoint moins S-S) est de
{res["agreement"][HEADLINE_KERNEL]["within_S"] - res["agreement"][HEADLINE_KERNEL]["S_S"]:.3f} : un test fondé sur
l'accord de réponses ne peut pas voir fp8 / LoRA 0,3.</p>"""

    style_intro = f"""<p>t d'un échantillon sur les prompts (écart à la moyenne des autres endpoints S).
T4 et T3 ressortent sur la longueur et le type-token ratio. Le <b>system prompt</b> n'est pas un témoin nul :
longueur t = {t["ref-03"]}, {short_len["ref-03"]} sur les prompts courts ; il se confond avec un LoRA faible.
Les quatre autres S restent dans [{min(nulls)}, {max(nulls)}] sur la longueur. Huit traits x deux
sous-ensembles x neuf endpoints : des t de 2 à 3 apparaissent par hasard.</p>"""

    adherence_intro = f"""<p>Part des réponses de 2 mots au plus sur les prompts « répondre uniquement par
oui/non / une lettre / un nombre / OK ». oui/non : T4 {yesno["ref-07"][0]}/{yesno["ref-07"][1]}, T3
{yesno["ref-04"][0]}/{yesno["ref-04"][1]}, T2 {yesno["ref-09"][0]}/{yesno["ref-09"][1]}, fp8
{yesno["ref-02"][0]}/{yesno["ref-02"][1]}, S 9 à 12 sur 12. numeric : le system prompt tombe à
{numeric["ref-03"][0]}/{numeric["ref-03"][1]}, comme les LoRA forts.</p>"""

    budget_intro = """<p>À budget égal (N réponses par endpoint), « N x 1 », « N/2 x 2 » ou « N/3 x 3 » ?
Deux statistiques : paires (T-S contre S-S) et endpoint contre pool S. Pour N = 120 il faut 60 prompts à
2 tirages ; les marqueurs creux signalent que les rééchantillons se recouvrent.</p>"""

    reliability_text = f"""<p>Corrélation de Spearman, sur les {rel["prompts_used"]} prompts qui ont au moins deux
tirages, entre l'excès « T1/T2 contre S » du tirage 0 et du tirage 1 (tirages indépendants) :
{", ".join(f"{k} {v['spearman_r0_r1']:.2f}" for k, v in reliability.items())}. Tout est proche de 0 : le
classement des prompts sur les cibles difficiles est du bruit d'échantillonnage. Le classement
« LoRA forts » en 4.2 est plus fiable parce que l'effet est grand, mais il reste mesuré sur les
mêmes tirages.</p>"""

    interpretability = f"""<p><b>Pourquoi les LoRA forts se voient.</b> Ils changent le registre : réponses longues et
explicatives après un « uniquement » (T4 : 0 réponse courte sur 12 oui/non, 0 sur 8 choix), type-token ratio plus
bas, moins de ponctuation propre. Ce décalage est visible dès un prompt, d'où l'AUC élevée à faible budget et
le plateau vers N = 40.</p>
<p><b>Pourquoi fp8 et LoRA 0,3 échappent.</b> Le modèle échantillonne chaud : sur Hamming, deux tirages d'un
même endpoint s'accordent à {agree(res, HEADLINE_KERNEL, "within_S")} seulement, et un endpoint fp8 s'accorde
avec S à {agree(res, HEADLINE_KERNEL, "S_T1")}, c'est-à-dire autant que S avec lui-même. La perturbation de
logits d'une quantification fp8 ou d'un LoRA à 0,3 est donc noyée dans l'entropie d'un tirage, et l'on ne voit que
le texte, jamais les probabilités.</p>
<p><b>Déguisements.</b> Le system prompt se comporte comme un LoRA faible (réponses plus longues sur les
prompts courts, ponctuation) ; watermark ajoute un préambule (« The sentence is: ») ; température 1,0 dérive
(« The Snapchat Experience... » pour une copie de phrase) ; persona est quasi invisible. Erreur typique à
prévoir sur ep : un endpoint S à system prompt classé avec les LoRA 0,7.</p>
<p><b>Sensibilité.</b> Le choix du noyau compte : Hamming sépare les LoRA forts, « réponse exacte » ne sépare rien
sur les prompts à réponse diffuse, longueur ne sépare que T4. Aucun noyau ne sépare fp8 / LoRA 0,3.</p>"""

    ranking = """<table><thead><tr><th>Piste</th><th>Gain attendu</th><th>Coût</th><th>Risque</th></tr></thead><tbody>
<tr><td>Prompts oui/non, choix, style court pour les LoRA forts</td><td>élevé : AUC 0,98 à N = 40</td><td>faible
(réponses de 1 à 40 tokens)</td><td>faible, mais confusion avec system prompt</td></tr>
<tr><td>Règle d'arrêt adaptative</td><td>~20 % de budget économisé, précision inchangée</td><td>nul (calcul)</td><td>faible</td></tr>
<tr><td>Pooling des endpoints d'un même cluster</td><td>moyen : plus de tirages par référence</td><td>nul</td><td>moyen (un
mauvais cluster contamine la référence)</td></tr>
<tr><td>Autre statistique pour fp8 / LoRA 0,3 (hors accord de texte)</td><td>inconnu</td><td>élevé (temps humain, calcul)</td>
<td>élevé : peut rester indécidable</td></tr>
<tr><td>Contenu factuel, arithmétique, copie exacte</td><td>nul mesuré</td><td>moyen</td><td>élevé (bruit)</td></tr>
</tbody></table>
<p>Sur-ajustement au classement public à 30 % : sans objet ici (aucune soumission). Le risque réel est de
choisir des prompts sur 9 endpoints ref qui ne se transfèrent pas aux 33 ep : la banque doit rester diverse.</p>"""

    plan = f"""<p>Budget par endpoint ep : 100, dont 10 de réserve. <b>Recommandation : ne dépenser que 54 de 90.</b></p>
<table><thead><tr><th>Phase</th><th>Réponses</th><th>Cumul</th><th>Contenu</th><th>Critère d'arrêt</th></tr></thead><tbody>
<tr><td>E1</td><td>24</td><td>24</td><td>prompts oui/non (6), choix (4), numeric (4), style court (4), recite ou count (6)
x 1 tirage, rejoués à l'identique sur les 33 endpoints</td><td>base (famille) fixée ; LoRA forts isolés (longueur,
conformité) ; clusters</td></tr>
<tr><td>E2</td><td>30 (adaptatif)</td><td>54</td><td>seulement les endpoints dont une paire est à moins de 2
erreurs-types du seuil calibré sur ref au même N ; comparaison au <b>pool du cluster</b></td>
<td>un endpoint s'arrête quand toutes ses paires sont résolues ; arrêt global à 54</td></tr>
<tr><td>E3</td><td>0 par défaut</td><td>54</td><td>réserve de 36 : à n'ouvrir que sur décision humaine, uniquement
pour des paires LoRA fort contre system prompt encore ambiguës</td><td>-</td></tr>
</tbody></table>
<p>Pourquoi 54 : l'AUC LoRA forts atteint {auc(res, "easy", 1, "pool", 40)} contre le pool à N = 40 (0,99 à
N = 30) ; fp8 / LoRA 0,3 reste à {auc(res, "hard", 1, "pool", 60)} à N = 60 et ne bouge pas ensuite
({auc(res, "hard", 2, "pool", 120)} à N = 120), donc dépenser plus ne les sépare pas. Un tirage par prompt
suffit (diversité = répétition pour les LoRA forts), sous réserve que l'analyse officielle du kit
n'exige pas de paires intra-endpoint : dans ce cas, 2 tirages sur 12 prompts.</p>
<p><b>Limites.</b> Calibrer le seuil sur ref au même N (<code>--ref-like</code>) ; les ref n'ont qu'un
modèle de base, les ep en ont plusieurs : la famille se détectera par le style (distance sur tous les canaux),
pas par ces courbes.</p>"""

    prompts = ("<p>À intégrer à la banque officielle (ajout en fin de catégorie, ids stables) : "
               "<code>cand-yesno-000..005</code> et <code>cand-choice-000..003</code> (consigne de format, "
               "décrochage des LoRA forts, S obéissants). À ne <b>pas</b> intégrer : echo, recite, arith diffus, "
               "numeric (le system prompt y fuit). Code : "
               "<code>src/hackathon/model_equality/research/candidates.py</code> "
               f"({n_candidates} prompts, familles et justifications en docstring).</p>")

    in_progress = (
        "<p>Collecte <code>ref</code> arrêtée à 184 réponses par endpoint (la limite de 300 n'a pas été atteinte : "
        "le serveur répond environ 0,2 à 0,5 requête par seconde pour toute l'équipe). Non fait : tirages 3 et 4 "
        "sur les prompts courts (seuls 15 prompts ont 3 tirages), test d'un autre type de statistique "
        "(probabilités de tokens indisponibles via l'API), transfert à des modèles de base différents. "
        "Suite proposée : si l'équipe veut tout de même fp8 / LoRA 0,3, la seule piste non épuisée est la "
        "distribution de la longueur et du premier token sur de très nombreux tirages d'un petit nombre de prompts "
        "(plusieurs centaines de réponses ref sur 3 prompts), à valider sur ref avant tout ep.</p>")

    top_intro = (
        "<p>Score = moyenne des excès T (distance au pool S, noyau adapté à la famille) moins le plus grand excès "
        "d'un déguisement. <b>Lecture :</b> classement dominé par l'excès des LoRA forts (colonne T3/T4) ; l'excès "
        "T1/T2 est ~0 partout (aucun prompt n'est recommandé pour eux). Les prompts de complétion de la "
        "banque (équipe A) et <code>cand-yesno-000</code> sortent en tête pour les LoRA forts. Classement brut sur les "
        "mêmes tirages : optimiste, à lire comme une liste de candidats.</p>")

    repro = f"""<pre>export PYTHONPATH=$PWD/src
MET_REFERENCE_ANSWERS_FILE=artifacts/model_equality/reference_answers_b.jsonl \\
  python -m hackathon.model_equality.research.ref_runner --repeat 0 --parallel 4
python -m hackathon.model_equality.research.build_phase1 \\
  --answers artifacts/model_equality/reference_answers_b.jsonl \\
  --team-answers &lt;reference_answers.jsonl de l'équipe A, copie du 21/11 17h10 UTC&gt; \\
  --practice data/hackathon/hf/model_equality/practice_answers.csv \\
  --out reports/phase1/model_equality/team-b
python -m hackathon.model_equality.research.report_b reports/phase1/model_equality/team-b provenance.json
node scripts/hackathon/report_to_pdf.mjs reports/phase1/model_equality/team-b/report.html</pre>
<p>Seeds constantes : permutations 20261121, bootstrap 20261122, simulation adaptative 20261123. Durées : collecte
~70 min, construction de l'analyse ~6 min (machine chargée). Aucune requête <code>ep-*</code>. Tests :
<code>tests/hackathon/test_hk_met_research.py</code> (14). SHA : {provenance.get("sha", "n/a")}.
Anomalie serveur : <code>GET /models</code> annonce <code>used</code> ~ 5,5e7 pour une limite de 1e6 sur les ref,
donc <code>left</code> = 0 et <code>query-ref</code> du kit planifie 0 requête ; mon runner applique un quota local.</p>"""

    return {
        "title": "Model Equality, équipe B : prompts discriminants et dépense ep (v2)",
        "meta": (f"21/11/2026. {total} réponses ref lues ({per_ep} par endpoint, {n_prompts} prompts), "
                 "dont les réponses de l'équipe A en lecture seule. Aucune requête ep-*."),
        "summary": summary, "leads": leads, "metrics_intro": metrics_intro,
        "group_intro": group_intro, "style_intro": style_intro, "adherence_intro": adherence_intro,
        "budget_intro": budget_intro, "reliability": reliability_text,
        "interpretability": interpretability, "ranking": ranking, "plan": plan,
        "prompts": prompts, "repro": repro, "in_progress": in_progress,
        "agreement_intro": agreement_intro, "top_intro": top_intro,
    }


def captions(res: Mapping[str, Any]) -> dict[str, str]:
    """What to conclude from each figure."""
    return {
        "fig1": ("Carte de chaleur prompts x endpoints (distance au pool S). Les cinq colonnes S doivent "
                 "ressembler au pool ; LoRA 0,7 et 1,5 sont nettement plus sombres sur les prompts "
                 "contraints ; fp8 et LoRA 0,3 se fondent dans S, et certains déguisements (system "
                 "prompt, watermark) ressortent aussi."),
        "fig2": ("AUC par groupe de prompts et par noyau. À droite (LoRA forts) presque tout sépare ; à "
                 "gauche (fp8, LoRA 0,3) tout reste autour de 0,5 avec des intervalles larges."),
        "fig3": ("Budget vers qualité à budget égal. LoRA forts : AUC 0,95 vers N = 20-30 puis plateau, "
                 "répartition prompts x tirages indifférente. fp8 / LoRA 0,3 : plat sous 0,5 quel que "
                 "soit N jusqu'à 120."),
        "fig4": ("Classement brut des prompts pour T1/T2. Sélection et mesure sur les mêmes tirages, "
                 "fiabilité split-half proche de 0 : ne pas s'y fier."),
        "fig5": ("Longueur des réponses par endpoint. Sur les yes/no, S répond en un mot, LoRA 0,7 et 1,5 "
                 "bavardent ; LoRA 0,3 à la marge."),
        "fig6": ("Dépense adaptative : l'erreur sur paires difficiles reste au hasard quel que soit le "
                 "budget ; l'arrêt anticipé ne coûte rien en précision et économise ~20 %."),
        "fig7": ("t de style par endpoint : T4 et T3 ressortent ; le system prompt ressort aussi (colonne "
                 "S system), donc T2 ne se distingue pas d'un déguisement."),
        "fig8": ("Conformité aux contraintes de format (réponses de 2 mots au plus, intervalle de Wilson "
                 "95 %). Les LoRA forts décrochent ; le system prompt décroche sur les numériques."),
        "fig9": ("Copie exacte (famille echo) : LoRA 0,3 et fp8 sont au niveau des S ; seuls 0,7 et 1,5 "
                 "chutent. Le signal « Here is the sentence: » n'est pas spécifique du LoRA 0,3."),
    }
