# Tache Model Equality (`33-model-equality`) : kit de l'equipe

Pour l'equipe de 3, pendant les 24 h. Tout est dans `src/hackathon/model_equality/` ; la CLI est
`python -m hackathon.model_equality`. Rien dans ce kit n'a ete execute contre le vrai serveur
(pas de cle disponible au moment de l'ecrire) : la section "Verifie / non verifie" dit ce qui est
prouve et ce qui reste a confirmer sur les `ref-*`.

## Ce que la tache demande (rappel chiffre)

- 33 endpoints `ep-01..ep-33` ; meme modele = meme checkpoint + meme quantization + meme LoRA.
  Watermark de decodage, system prompt du fournisseur, strategie de decodage (temperature...) ne
  changent PAS l'identite. Meme famille = meme modele de base.
- Score = 0,1 x max(0, MCC_famille sur les 528 paires) + 0,9 x max(0, MCC_modele sur les paires de
  la meme famille VRAIE). Le score public est un sous-ensemble : ne pas optimiser dessus.
- Budget : 100 reponses par endpoint ep POUR TOUTE LA COMPETITION et pour toute l'equipe, 256
  tokens max generes, 4 requetes simultanees. Les ref (`ref-01..ref-09`) sont illimites et jamais
  notes. Soumission : 1 toutes les 30 min, un echec compte aussi.

## Methode

1. **Reponses a des prompts identiques** (`prompts/bank.py`, 92 prompts en 8 categories). Les
   perturbations a detecter (fp8, LoRA) sont de petites variations de logits : on vise les prompts
   ou elles changent souvent le texte tire (completions quasi deterministes, arithmetique, faits de
   longue traine, comptage/repetition, code) et ou les deguisements (system prompt, persona,
   watermark, temperature) changent peu le contenu (formats contraints, nombre extrait).
2. **Statistique par paire et par canal** (`pairwise.py`) : MMD non biaise (noyau de Hamming a la
   MET, plus noyaux spectre de n-grammes, egalite exacte de la reponse extraite, 1er mot,
   longueur), null par permutation des etiquettes DANS chaque prompt, resume en z-score. Un canal =
   (categorie de prompts, noyau). Sous H0 le z est ~ N(0,1) quel que soit le modele : c'est ce qui
   rend les seuils comparables entre ref et ep.
3. **Nettoyage de boilerplate** (`boilerplate.py`) : une phrase d'ouverture/cloture frequente
   chez un endpoint, repartie sur >= 3 categories (persona, system prompt), est retiree avant
   comparaison. Elle est listee dans le rapport. `--no-strip-boilerplate` la desactive.
4. **Calibration sur les ref** (`clustering.py`) : poids de chaque canal = ce qui separe les
   modeles differents (T1..T4 vs S) sans separer les deguisements de S (system prompt, persona,
   watermark, temperature 1.0), puis seuil de coupe du modele qui maximise le MCC sur les ref.
5. **Regroupement a 2 niveaux** : familles par un dendrogramme sur la distance tous-canaux
   (un autre modele de base se voit partout), puis modeles DANS chaque famille avec la distance
   calibree. Les etiquettes sont emboitees par construction (meme modele => meme famille).
6. **Score local exact** (`scoring.py`) et equivalent de `score_practice.py`.

## Commandes (depuis la racine du depot, venv du repo)

```bash
export PYTHONPATH=$PWD/src
PY=.venv/bin/python
# cle dans .env : HACKATHON_API_KEY=...  (jamais commitee)

# 0. etat du serveur et du store (GET /models, ne coute rien)
$PY -m hackathon.model_equality budget

# 1. calibrer sur les ref (illimites). Mesurer d'abord la vitesse avec ref-1 (360 requetes).
$PY -m hackathon.model_equality plan       --phase ref-1
$PY -m hackathon.model_equality query-ref  --phase ref-1
$PY -m hackathon.model_equality query-ref  --phase ref-full      # 2484 requetes, reprise sure
$PY -m hackathon.model_equality analyze --target ref --practice-answers data/hackathon/hf/model_equality/practice_answers.csv

# 2. le cout exact des ep AVANT de payer, puis la phase 1 (exige --yes)
$PY -m hackathon.model_equality plan      --phase ep-1
$PY -m hackathon.model_equality query-ep  --phase ep-1 --yes

# 3. analyser (calibration sur les ref a la MEME forme d'echantillon que les ep)
$PY -m hackathon.model_equality analyze --calibrate --ref-like ep-1 \
    --practice-answers data/hackathon/hf/model_equality/practice_answers.csv
#    -> artifacts/model_equality/analysis/{report_ep.txt, distance_ep.csv, labels_ep.csv, ...}

# 4. soumission : construire, valider, puis soumettre (exige --yes)
$PY -m hackathon.model_equality make-submission --labels artifacts/model_equality/analysis/labels_ep.csv
$PY -m hackathon.model_equality validate artifacts/model_equality/submissions/submission.csv
$PY -m hackathon.model_equality submit   artifacts/model_equality/submissions/submission.csv        # essai a blanc
$PY -m hackathon.model_equality submit   artifacts/model_equality/submissions/submission.csv --yes

# utilitaire : score d'une reponse sur les ref, comme score_practice.py
$PY -m hackathon.model_equality score-practice my_ref_answer.csv --answers data/hackathon/hf/model_equality/practice_answers.csv
```

Options utiles : `--answers PATH` (store) ou `MET_ANSWERS_FILE` / `MET_REFERENCE_ANSWERS_FILE` ;
`--endpoints ep-03 ep-07` et `--categories arith format` (restreindre une phase) ;
`--reserve N` (defaut 10) ; `analyze --tau-model X --tau-family Y` (forcer les seuils),
`--permutations N`, `--ref-like ep-1 ep-2` (forme d'echantillon des ref pour la calibration),
`--units-of ...` (n'analyser que les unites de certaines phases). `make-submission --distances
distance_ep.csv --tau-model X` recoupe le dendrogramme sans rien recalculer : c'est la
facon de regarder l'effet d'un seuil en secondes.

## Compatibilite avec l'outil officiel

Le store lit et ecrit le format ligne a ligne de `task_template.py` / `query_reference.py`
(`endpoint, id, repeat, prompt, response`, `timestamp` optionnel). Un fichier produit par l'outil
officiel s'analyse tel quel ; l'outil officiel reprend un fichier ecrit par nous (il deduplique sur
`(endpoint, id, repeat)`). Convention d'equipe : `model_equality/answers.jsonl` (ep) et
`reference_answers.jsonl` (ref) sur `main`. Pour l'analyse du fichier d'un coequipier :
`--answers model_equality/answers.jsonl`. Details et regles de fusion git :
`artifacts/model_equality/README.md`.

## Plan de budget (par endpoint ep, limite 100, reserve 10 : 90 depensables)

| Phase | Reponses | Cumul | Contenu |
|---|---|---|---|
| `ep-1` | 40 | 40 (40 %) | 20 prompts x 2 repetitions, toutes categories : premier regard, familles, calibration de forme |
| `ep-2` | 30 | 70 | 3 nouveaux prompts x 2 repetitions dans arith, completion, fact, format ; 1 en count, code, list |
| `ep-3` | 20 | 90 | repetitions 3 et 4 des meilleurs prompts de la phase 1 (arith, completion, format) : ADAPTATIF |
| reserve | 10 | 100 | non depensee ; `--reserve 0` seulement pour la fin de partie |

Pourquoi >= 2 repetitions par prompt : le MMD non biaise a besoin de paires de reponses du MEME
endpoint au MEME prompt pour estimer le terme intra-endpoint ; avec 1 seule, il n'est pas defini
(le canal est alors NaN, signale, jamais remplace en silence).

Le budget est PAR ENDPOINT : la phase 3 doit etre lancee seulement sur les endpoints ambigus
(`report_ep.txt`, section "ambiguous same-family pairs", ligne "endpoints to feed first"), avec
`--endpoints`. Les endpoints deja bien places s'arretent a 70 ; les ambigus peuvent aller a 90,
puis a 100 en fin de partie. Les cumuls et la reserve sont imposes par `plan` et par le runner
(plafond dur verifie avant chaque envoi) ; `plan` donne le cout exact par endpoint, y compris ce
qui est deja dans le store.

## Pieges

- `query-ep` sans `--yes` ne fait rien (affiche le plan, code de sortie 2). Avec `--yes`, il lit
  `/models` et applique le plus strict de : limite - stocke - reserve, et restant serveur - reserve.
  Un `used` serveur superieur au store (`budget` le signale) veut dire qu'un coequipier a depense
  sans pousser : `git pull` AVANT de continuer.
- Une seule personne interroge les ep a la fois. Le store en union-merge n'evite pas la double
  depense, il evite seulement le conflit git.
- Ne jamais modifier un prompt existant de la banque (les ids sont les cles du store) : ajouter a
  la fin d'une categorie. `plan` signale la derive.
- Tester sur ref d'abord que les repetitions donnent bien des reponses differentes (diagonale de
  `exact_match_ref.csv` < 1 sur des prompts non deterministes) : si le serveur renvoyait la meme
  reponse en cache, les repetitions seraient perdues.
- La calibration sur les ref est EN ECHANTILLON (9 endpoints, un seul modele de base, petit) :
  le MCC annonce sur les ref est optimiste. Le rapport affiche aussi le MCC avec les poids a priori
  et le seuil par defaut, qui est plus honnete. Les seuils ne se transferent que si la forme
  d'echantillon est la meme (`--ref-like`).
- Les z augmentent avec la taille d'echantillon : ne pas comparer des seuils entre des analyses de
  tailles differentes sans recalibrer.
- Persona et system prompt sont le cas dur : surveiller les canaux `onehot_first` et `length`
  (poids a priori 0) et la section "boilerplate stripped". Un endpoint persona isole dans son
  propre modele est le symptome typique.
- Jamais de cle API dans le code, les logs ou un commit. Une soumission echouee relance quand meme le
  delai de 30 min : `validate` d'abord, toujours.
- Ne pas optimiser pour le score public (sous-ensemble) : decider avec les ref et les marges du
  rapport.

## Verifie / non verifie

Verifie (tests, `tests/hackathon/test_hk_met_*.py`, zero reseau) : format du store et reprise
avec l'outil officiel, plafond de budget et reserve, 429/5xx, 4 requetes max en parallele, cout
exact du plan, score officiel et equivalence sortie-pour-sortie avec `score_practice.py` (40
etiquetages aleatoires compares au vrai script), validation de soumission, statistique rapide =
reference brute-force, z ~ N(0,1) sous H0, workflow complet CLI contre un faux serveur local.

Non verifie (depend du serveur ou de vraies donnees) : la puissance reelle de la methode sur ep et
ref (les tests utilisent un modele jouet : ils prouvent que le logiciel marche, pas que la methode
separe fp8/LoRA), les formats exacts de reponse du serveur au-dela du template, l'effet reel d'un
persona/system prompt sur chaque canal, la justesse des seuils (`DEFAULT_TAU_MODEL`, fraction du
plancher de famille) hors calibration, le temps de reponse du serveur (donc la duree de
`ref-full`), le comportement de la convention de zero denominateur du MCC cote serveur.

## Dry run sans serveur

```python
from hackathon.model_equality.simulate import scored_endpoints, simulate_answers
from hackathon.model_equality.prompts import expand_phase, get_phase
eps = scored_endpoints()
answers = simulate_answers(eps, expand_phase(get_phase("ep-1")))   # donnees SYNTHETIQUES
```
Ecrire ces reponses dans un store temporaire (`AnswerStore(path).append(...)`) permet d'essayer
`analyze` de bout en bout ; ce n'est pas une mesure de la methode.
