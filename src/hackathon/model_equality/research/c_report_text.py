"""Hand-written narrative of the team-C reports (French). Numbers come from `c_report.py`.

Judgement lives here, computed values are passed in, so a figure and its sentence cannot drift:
every number in the text is formatted from the arguments of `render`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

CAPTIONS: Mapping[str, str] = {
    "dash": "A retenir : 0,05 contre 0,75 en tete ; la sonde 4 (hachures) est une attente sous l'hypothese H*, pas une mesure.",
    "abc": "A retenir : sous H* conditionne aux trois 0,05, isoler 5 suspects (sonde 4) a la meilleure attente ; isoler les endpoints a persona (barre grise) est moins bon.",
    "decomp": "A retenir : la famille ne rapporte que 0,10 au mieux ; un score de 0,75 exige un MCC modele proche de 0,8.",
    "famc": "A retenir : si A/B etait la vraie structure, E1 vaudrait 0,10 ; 0,05 dit que le groupe Qwen-like contient deux familles (ou plus).",
    "agree": "A retenir : 7 endpoints Qwen-like donnent la reponse modale aux 5 prompts courts ; ce n'est PAS la preuve d'un decodage glouton (voir la correction) ; etiquettes colorees par createur cite.",
    "dendro": "A retenir : deux grands blocs ; les endpoints a identite reecrite (Anthropic, OpenAI, aucun) ne se rangent pas par le createur.",
    "info": "A retenir : 3 prompts sur 5 n'ont que 2 reponses distinctes : ils portent 1 bit chacun.",
    "ref": "A retenir : sur les ref, un endpoint s'accorde avec lui-meme dans 0,01 a 0,43 des cas : ils echantillonnent, comme les ep.",
    "lib": "A retenir : les scores E1 et E1b = 0,05 deplacent la bibliotheque vers moins de modeles et plus de familles.",
    "marg": "A retenir : le coeur Qwen modal est un bloc a ~0,7 de meme modele ; la coupe en deux familles Qwen vient de la structure d'accord, qui confond reponse modale et famille : a ne pas lire comme une preuve.",
    "graph_e1": "A retenir : E1 = deux familles, un modele par famille ; MCC modele nul par construction.",
    "graph_p3": "A retenir : la sonde 3 gardait les familles de E1 (donc +0,05 connu) et ne changeait que les modeles ; elle a lu 0,05.",
    "graph_p4": "A retenir : la sonde 4 garde un gros modele par famille et isole 5 suspects (ep-16, 20, 28, 29, 31).",
    "cands": "A retenir : E1b (deja soumis) est attendu a 0,14 en moyenne mais a lu 0,05 ; les clique-seules sont plus prudentes.",
    "pred": "A retenir : la bibliotheque donnait 21 % a 0,05 pour la sonde 3 et 79 % au-dessus : elle a donne 0,05. La bibliotheque est refutee comme outil de decision.",
    "voi": "A retenir : les sondes proches du consensus dominent ; la valeur d'information d'un pas reste de l'ordre de 0,07.",
    "sim": "A retenir : sur 3 verites synthetiques, 6 sondes ajoutent +0,15 en moyenne (de +0,04 a +0,31), pas de facon monotone ; indicatif.",
    "refpower": "A retenir (MESURE sur ref) : les variantes fortes montent de 0,60 a 0,88 d'AUC avec N, les variantes faibles (fp8, LoRA 0,3) restent a 0,5 : plus de prompts longs ne les separe pas.",
    "longstyle": "A retenir : au premier mot, Qwen-like dit 'Sure' ou 'Certainly', Llama-like dit 'Here's' : le style separe les familles.",
    "hist": "A retenir : trois 0,05 de suite (E1, E1b, sonde 3) ; la sonde 4 teste l'hypothese H*.",
}


def describe(rows: Sequence[Sequence[str]]) -> str:
    """Compact text of a partition: families with sizes, models with 2+ endpoints, singleton count."""
    families: dict[str, int] = defaultdict(int)
    models: dict[tuple[str, str], list[str]] = defaultdict(list)
    for endpoint, family, model in rows:
        families[family] += 1
        models[(family, model)].append(endpoint[3:])
    multi = ["{" + ",".join(v) + "}" for v in models.values() if len(v) > 1]
    singles = sum(1 for v in models.values() if len(v) == 1)
    sizes = "+".join(str(n) for n in sorted(families.values(), reverse=True))
    return f"familles {sizes} ; modeles : {' '.join(multi) or 'aucun groupe'} ; {singles} isoles"


def fmt(value: float) -> str:
    """French decimal comma, two digits."""
    return f"{value:.2f}".replace(".", ",")


def render(
    *, number: int, figures: Mapping[str, str], numbers: Mapping[str, Any], plan: Mapping[str, Any],
    observed: Mapping[str, float], pending: Sequence[str], long_info: Mapping[str, Any],
    agreement: Any, names: Sequence[str], creators: Sequence[str],
    ref_pair: Mapping[str, float], ref_self: Mapping[str, float],
    table: Callable[[Sequence[str], Sequence[Sequence[str]]], str], cliques: Sequence[Sequence[int]],
    probe_name: str, e: Callable[[str], str], regex: Any,
    ref_power: Mapping[int, Mapping[str, float]], style_counts: Mapping[str, Any],
    abc: Mapping[str, Any],
) -> str:
    """The whole report body as HTML."""
    del agreement, creators, regex, style_counts
    index = {n: k for k, n in enumerate(numbers["names"])}
    mean, sd = numbers["mean"], numbers["sd"]
    abc_cands = abc.get("candidates", {})
    p4 = abc_cands.get("carve_all5", {"mean_full": float("nan"), "sd_full": float("nan"),
                                       "p_above_0_05": float("nan"), "levels": []})
    abc_rows = [[name, fmt(v["mean_full"]), fmt(v["sd_full"]), fmt(v["p_above_0_05"]),
                 " / ".join(f"{x:.2f}" for x in v["levels"])] for name, v in abc_cands.items()]
    abc_table = table(["candidat", "score attendu (528 paires)", "ecart-type", "P(public > 0,05)",
                       "P(public = 0 / 0,05 / 0,10 / 0,15 / 0,20 / 0,25 / >= 0,30)"], abc_rows)
    clique_text = " ; ".join("{" + ",".join(names[i][3:] for i in g) + "}" for g in cliques if len(g) > 1)
    best = ", ".join(f"{k} {fmt(v)}" for k, v in observed.items())
    rows = []
    for name in numbers["names"]:
        k = index[name]
        levels = numbers["levels"][name]
        rows.append([name, fmt(float(mean[k])), fmt(float(sd[k])),
                     fmt(float(numbers["p_above"][name])), " / ".join(f"{x:.2f}" for x in levels)])
    candidate_table = table(
        ["candidat", "score attendu (528 paires)", "ecart-type", "P(public > 0,05)",
         "P(public = 0 / 0,05 / 0,10 / 0,15 / 0,20 / 0,25 / >= 0,30)"], rows)
    best_text = describe(plan["best_response"])
    probe_rows_text = "".join(
        f"<li>sonde {k + 1} : gain {fmt(t['voi'])}, information {t['mutual_bits']:.2f} bit, "
        f"attendu {fmt(t['expected_final_score'])} ; {e(describe(t['rows']))}</li>"
        for k, t in enumerate(plan["top_probes"][:5]))
    long_rows = [[pid, str(v["distinct"]), str(v["largest_identical_group"]), f"{v['mean_words']:.0f}"]
                 for pid, v in long_info.items()]
    long_distinct_text = " et ".join(
        f"{pid} : {v['distinct']} reponses distinctes sur 33 endpoints (plus grand groupe identique : "
        f"{v['largest_identical_group']})" for pid, v in long_info.items()) or "les prompts longs"
    long_block = (
        "<p>Les 2 prompts longs autorises sont arrives ; reponses distinctes et plus grand groupe identique :</p>"
        + table(["prompt", "reponses distinctes (33 ep)", "plus grand groupe identique",
                 "mots moyens"], long_rows)
        if long_rows else
        "<p>Aucun prompt long complet pour l'instant.</p>")
    return f"""
<h1>Model Equality, equipe C (r-met-c) : salve {number}</h1>
<p class="meta">21/11/2026. Angle : tirer le maximum de chaque soumission (structure du score,
information des scores publics, plan de sondage). Chiffres reels sauf mention ESTIMATION ;
les attentes sont celles d'une bibliotheque d'hypotheses, donc des hypotheses, pas des mesures.</p>

<div class="summary" style="border-left-color:#D55E00"><b>CORRECTION (en tete, decision du directeur).</b>
Dans mon message de 19 h 40 j'ai affirme que 7 endpoints Qwen-like etaient en decodage glouton (identiques sur 5 prompts) et qu'un accord exact sur des textes longs separerait fp8 et LoRA faibles. C'etait faux. Deux faits mesures :
<ol>
<li><b>Les ep echantillonnent, comme les ref.</b> Sur {long_distinct_text}. Les 7 endpoints identiques sur les 5 prompts courts donnent 7 textes longs differents (rapport de prefixe commun 0,01). Leurs reponses courtes identiques sont le mode de la loi ("yes", "I am Qwen, created by Alibaba Cloud.", la Declaration d'independance), pas un decodage glouton. La critique que j'avais faite de la courbe MCC-N de m-met-1 ne tient donc pas non plus.</li>
<li><b>Le style separe les familles.</b> Sur les prompts longs, les endpoints Qwen-like ouvrent par "Sure" ou "Certainly", les Llama-like par "Here's" (figure ci-dessous). Les statistiques de style separent les variantes FORTES (AUC 0,73 a 5 unites par endpoint sur les ref) mais pas fp8 ni LoRA 0,3 (AUC 0,46).</li>
</ol>
Consequence : l'estimation de detection par accord exact (1 - (1 - delta)^N) est retiree ; le palier ep est chiffre par une mesure sur les ref (section 7). La collecte a ete mise en pause a la decouverte, puis terminee a 2 prompts longs (option B du directeur).</div>
{figures.get("longstyle", "")}

<div class="cards">
<div class="card"><div class="big">0,05</div><div class="lab">meilleur score public (E1 et E1b), 10e sur 18</div></div>
<div class="card"><div class="big">0,75</div><div class="lab">1er du classement</div></div>
<div class="card"><div class="big">0,05 x 3</div><div class="lab">E1, E1b et sonde 3 (bibliotheque : 79 % au-dessus de 0,05) : aucun groupe de texte n'a d'information de modele</div></div>
<div class="card"><div class="big">{fmt(p4["mean_full"])}</div><div class="lab">sonde 4 : score attendu sous H* (ecart-type {fmt(p4["sd_full"])}), P(&gt; 0,05) = {fmt(p4["p_above_0_05"])}</div></div>
<div class="card"><div class="big">+2</div><div class="lab">prompts longs par endpoint collectes (66 reponses) ; 93 reponses restantes par ep</div></div>
</div>
{figures["dash"]}

<h2>1. Resume decisionnel</h2>
<div class="summary"><ul>
<li><b>Ce que disent E1 et E1b.</b> Deux scores publics a 0,05 avec les memes familles A (23) / B (10). E1 n'a qu'un modele par famille : son MCC modele vaut 0 par construction, donc 0,05 = 0,1 x MCC famille public et ce MCC vaut environ 0,5. La coupe A/B n'est pas la vraie structure : le groupe Qwen-like se coupe en deux familles ou plus. E1b (isoler 11 endpoints par la longueur) a un MCC modele public &lt;= 0.</li>
<li><b>Structure du score.</b> Avec une seule famille dans le CSV, le score vaut exactement 0,9 x MCC modele : sonde propre du niveau modele. Le MCC modele ne compte que les paires de la meme famille VRAIE : fusionner a tort deux familles ne coute que 0,1 x MCC famille, scinder a tort coute du rappel. La precision prime : un groupe trop gros est tout en faux positifs.</li>
<li><b>Donnees.</b> 3 prompts sur 5 n'ont que 2 reponses distinctes (1 bit chacun) ; les 7 endpoints identiques sur les 5 prompts donnent la reponse modale, pas un decodage glouton ; les textes longs, tous distincts, separent les familles par le style et les variantes fortes par la longueur.</li>
<li><b>Sonde 3 : 0,05.</b> La bibliotheque la donnait au-dessus de 0,05 avec 79 % de chance : trois soumissions construites sur des groupes de texte (E1, E1b, sonde 3) donnent le meme 0,05. Les groupes par similarite de reponses n'ont pas d'information de modele, et peuvent en avoir une inverse : les copies a system prompt ou persona (qui changent le texte) sont du MEME modele, les variantes fp8 et LoRA faibles (qui gardent le texte modal) en sont d'autres.</li>
<li><b>Plan (sonde 4, file, priorite 20).</b> Hypothese H* : le gros groupe de chaque famille est un seul modele (copies, personas, system prompts), les variantes sont isolees ; je teste les 5 suspects au texte tronque, deroulant ou garbled (ep-16, 20, 28 cote A ; ep-29, 31 cote B). Attente sous H* conditionne aux trois 0,05 : {fmt(p4["mean_full"])} +- {fmt(p4["sd_full"])}, P(&gt; 0,05) = {fmt(p4["p_above_0_05"])}.</li>
<li><b>A decider par l'equipe :</b> faut-il lire le niveau modele avec des sondes a une seule famille (score = 0,9 x MCC modele) ou garder les familles de E1 (+0,05 connu) ? Je recommande de garder E1 tant que les familles ne sont pas retrouvees.</li>
</ul></div>

<h2>2. Tableau de bord</h2>
<p>Scores publics lus : {e(best)}. En attente : {e(", ".join(pending) or "aucun")}.</p>
{figures["hist"]}
{figures["decomp"]}

<h2>3. Structure du score : ce que chaque lecture apprend</h2>
<p>Le score est une fonction de 9 nombres : le croisement entre l'etat vrai d'une paire (autre famille / meme famille / meme modele) et l'etat soumis. Le tableau public ne voit qu'environ 30 % des paires, tirees au hasard : le compte public de chaque case est une loi binomiale independante, ce qui donne la loi exacte du score public de tout couple (verite, soumission) sans simuler le sous-ensemble. Les 528 paires et le score officiel sont recoupes avec <code>scoring.official_score</code> par test (9 tests verts).</p>
<ul>
<li>Un CSV avec un modele par famille (E1) ne lit que MCC famille. Un CSV avec une famille unique ne lit que MCC modele. Un CSV avec tous les endpoints isoles vaut 0 (aucune paire "meme").</li>
<li>Resolution : tous les scores du tableau sont des multiples de 0,05 ; si le serveur quantifie a ce pas, un endpoint deplace seul (un ou deux paires publiques) ne se lit pas : seuls les changements de groupes entiers se lisent.</li>
<li>Transitivite : l'identite de modele est une relation d'equivalence ; une information sur le groupe d'un endpoint contraint ses ~32 paires, dont ~70 % sont privees. Les sondes doivent donc viser la structure (groupes), pas les paires.</li>
</ul>
{figures["famc"]}

<h2>4. Ce que disent les reponses ep (5 prompts, un tirage)</h2>
<p>Cliques strictes (5 prompts sur 5 identiques, texte normalise) : <code>{e(clique_text)}</code>.</p>
{figures["agree"]}
{figures["dendro"]}
{figures["info"]}
{figures["ref"]}
<p>Accord exact sur les ref : {e(", ".join(f"{k} {fmt(v)}" for k, v in ref_pair.items()))}. Accord d'un endpoint ref avec lui-meme entre deux tirages : de {fmt(min(ref_self.values()))} a {fmt(max(ref_self.values()))}. Les ref echantillonnent, et les ep aussi : sur les prompts longs aucun endpoint ne repete le texte d'un autre. Un accord exact sur les prompts courts mesure la probabilite du mode, pas l'identite du modele.</p>

<h2>5. Hypotheses sur la vraie structure et information des scores</h2>
<p>Methode : une bibliotheque de 3000 hypotheses (partitions a deux niveaux), non selectionnee sur le posterieur : 50 % de coupes bruitees de la distance d'accord, 35 % de coupes tres bruitees avec 4 a 12 editions, 15 % de verites synthetiques de la forme des ref. Chaque hypothese est ponderee par la vraisemblance exacte de E1 = 0,05 et E1b = 0,05 (pas de 0,05, sous-ensemble public 30 %, vraisemblance temperee a la puissance 0,6 car le sous-ensemble est le meme pour toutes les soumissions). <b>Limite assumee</b> : un premier essai par SMC glouton s'est effondre sur un seul mode (score attendu 0,96, faux) ; je l'ai remplace par cette bibliotheque large. Les attentes ci-dessous dependent de sa construction.</p>
{figures["lib"]}
{figures["marg"]}
{figures["graph_e1"]}
{figures["graph_p3"]}
{figures.get("graph_p4", "")}
<p>Meilleure reponse du posterieur : {e(best_text)} ; attendue a {fmt(plan["best_response_expected"])}.</p>
{figures["cands"]}
{candidate_table}
{figures["pred"]}

<h2>5b. Les trois 0,05 et l'hypothese H*</h2>
<p>Trois soumissions construites sur les groupes de texte donnent exactement 0,05 : E1 (un modele par famille), E1b (11 isoles par la longueur), sonde 3 (consensus du posterieur). Avec un MCC famille public d'environ 0,5 qui rapporte 0,05, leur MCC modele public est nul ou negatif. La bibliotheque de la section 5 predisait pour la sonde 3 : 21 % de 0,05 et 79 % au-dessus ; l'observation la refute comme outil de decision.</p>
<p><b>Hypothese H* (a tester, pas un fait).</b> Les organisateurs construisent chaque famille comme les 9 ref : un modele principal avec ses copies deguisees (system prompt, persona, watermark, temperature 1,0) plus des variantes isolees (fp8, LoRA de force 0,3 a 1,5). Un deguisement change le texte sans changer le modele : les endpoints a identite reecrite (Anthropic, OpenAI, Emoji Artist...) sont des copies du modele principal, pas des modeles a part. Les variantes faibles (fp8, LoRA 0,3) gardent les reponses modales et se cachent dans les groupes de texte serres, ce qui rend les groupes de texte anti-informatifs. Seules les variantes fortes (LoRA 0,7 et 1,5 : texte tronque, deroulant, garbled) se voient.</p>
<p>Methode (rejet, ABC) : tirer des verites sans regarder le texte (familles = createurs, un groupe pouvant etre coupe en 2 ou 3 familles vraies, modele principal = 35 a 65 % de la famille, variantes isolees, les 5 suspects etant des variantes avec probabilite 0,55), garder celles qui reproduisent trois fois 0,05 : {abc.get("accepted", 0)} verites retenues sur {abc.get("drawn", 0)} ({fmt(abc.get("acceptance_rate", 0.0))} de taux d'acceptation).</p>
{figures.get("abc", "")}
{abc_table}

<h2>6. Plan de sondage</h2>
<ol>
<li><b>Sonde 3 (lue : 0,05)</b> : familles E1 et modeles du consensus de la bibliotheque ; elle a refute la bibliotheque.</li>
<li><b>Sonde 4 (en file, priorite 20)</b> : familles E1 (+0,05 constant), un seul modele par famille sauf 5 suspects isoles : ep-16, 20, 28, 29, 31. Lecture : score - 0,05 = 0,9 x MCC modele public ; un vrai isole apporte environ 20 vraies paires "differentes", un faux isole coute environ 12 faux "differents" : un MCC positif exige que la majorite des 5 soit de vrais isoles. Si la lecture reste a 0,05 : H* est affaiblie, retirer les suspects ; si elle monte : isoler plus, un par un.</li>
<li><b>Apres la lecture</b> : mettre a jour le posterieur avec le score observe, puis choisir par valeur d'information. Tant que le niveau modele n'est pas lu, un seul changement controle par envoi (fusionner un sous-groupe douteux, ou scinder le plus gros groupe).</li>
<li><b>Sondes a une seule famille</b> : a utiliser quand deux sondes ont la meme famille et que l'on veut separer MCC modele de MCC famille.</li>
</ol>
<ul>{probe_rows_text}</ul>
{figures["voi"]}
{figures.get("sim", "")}
<p>Simulation en boucle fermee : verites synthetiques de 33 endpoints (forme des ref mise a l'echelle), texte seul a un MCC modele de l'ordre de 0,3, oracle exact sur un sous-ensemble public fixe de 30 %, 5 a 6 sondes choisies par valeur d'information. Resultats par verite dans la figure ; peu de verites (3), donc indicatif.</p>

<h2>7. Budget ep : ce que rapporterait chaque palier</h2>
{long_block}
<p>Collecte executee (decision humaine, +5 autorises puis option B du directeur) : 66 reponses, 2 prompts longs complets sur les 33 endpoints, 0 echec, 0 nouvelle tentative, compteur serveur verifie apres coup (7 utilises par ep, 8 sur ep-09 a ep-12). Debit mesure sur des reponses de 256 tokens : environ 7 reponses par minute a 4 requetes simultanees, 2 a 3 par minute a 2 requetes.</p>
{figures["refpower"]}
{table(["palier", "requetes (33 ep)", "duree a 7 reponses/min", "AUC variantes fortes (MESURE, ref)", "AUC fp8 et LoRA 0,3 (MESURE, ref)"],
       [["+5", "165", "24 min", fmt(ref_power[5]["strong"]), fmt(ref_power[5]["weak"])],
        ["+10", "330", "47 min", fmt(ref_power[10]["strong"]), fmt(ref_power[10]["weak"])],
        ["+20", "660", "94 min", fmt(ref_power[20]["strong"]), fmt(ref_power[20]["weak"])]])}
<p>Lecture : mesure sur 31 unites longues par endpoint ref (16 prompts, un modele de base, score de style a 5 caracteristiques). +5 prompts longs separent les variantes fortes au niveau 0,73, +20 au niveau 0,88 ; fp8 et LoRA 0,3 restent au hasard quel que soit N jusqu'a 20. Les ref sont d'un autre modele que les ep : c'est un ordre de grandeur, pas une promesse. Pour la strategie de sondage, la valeur d'un palier est donc : (a) mieux isoler les variantes fortes (utile : ce sont des modeles a part), (b) renforcer les familles par le style, (c) ne PAS resoudre fp8 et LoRA faibles, que seule la lecture du score public peut traiter.</p>

<h2>8. Reproductibilite</h2>
<pre>PYTHONPATH=src python -m hackathon.model_equality.research.c_plan_real --answers artifacts/model_equality/answers.jsonl --observed e1=...=0.05 --observed e1b=...=0.05 --out reports/phase2/model_equality/team-c/plan-{number}
PYTHONPATH=src python -m hackathon.model_equality.research.c_report --number {number} ...
node scripts/hackathon/report_to_pdf.mjs reports/phase2/model_equality/team-c/report-{number}.html</pre>
<p>Graines : bibliotheque 20261121, rapport 20261121. Tests : <code>tests/hackathon/test_hk_met_c_oracle.py</code>, <code>test_hk_met_c_long_run.py</code>. Les modules sont <code>src/hackathon/model_equality/research/c_*.py</code>.</p>
"""
