# artifacts/model_equality

Reponses collectees aupres des endpoints de la tache `33-model-equality`, versionnees dans git :
le budget (100 reponses par endpoint ep pour toute la competition, commun a l'equipe) est
irremplacable, donc chaque reponse payee doit etre partagee.

Convention d'equipe : les fichiers de reference sont `model_equality/answers.jsonl` (ep) et
`model_equality/reference_answers.jsonl` (ref) sur la branche `main` du depot d'equipe. Le defaut
de l'outil est `artifacts/model_equality/answers.jsonl` et `.../reference_answers.jsonl` ; on le
change avec `--answers PATH` ou les variables d'environnement `MET_ANSWERS_FILE` (ep) et
`MET_REFERENCE_ANSWERS_FILE` (ref).

## Fichiers

| Fichier | Contenu | Ecrit par |
|---|---|---|
| `answers.jsonl` | reponses des `ep-01..ep-33` (store ep) | `query-ep` ou `task_template.py` officiel |
| `reference_answers.jsonl` | reponses des `ref-01..ref-09` (store ref) | `query-ref` ou `query_reference.py` officiel |
| `analysis/` | sorties de `analyze` (rapport, matrices de distances, labels, calibration) | `analyze` |
| `submissions/` | CSV prets a soumettre | `make-submission` |

## Format du store (identique a l'outil officiel)

Une ligne JSON par reponse, UTF-8, `ensure_ascii=False` :

```json
{"endpoint": "ep-07", "id": "arith-003", "repeat": 0, "prompt": "Compute 918 * 76. Answer with only the number.", "response": "69768", "timestamp": "2026-11-21T10:12:03+00:00"}
```

- `endpoint`, `id` (chaine), `repeat` (entier), `prompt`, `response` : exactement les cinq champs
  du `task_template.py` officiel. `timestamp` (UTC, ISO-8601) est optionnel et ignore par l'outil
  officiel. A la lecture, les champs supplementaires sont toleres et `prompt_id` est accepte
  comme alias de `id`.
- Cle d'identite : `(endpoint, id, repeat)`. C'est la cle de deduplication de l'outil officiel ET
  de la notre : un fichier ecrit par l'un est repris par l'autre sans doublon. Notre store ne
  re-demande jamais une cle presente et ne l'ecrit jamais deux fois ; si une fusion git cree des
  doublons, la premiere occurrence gagne (a la lecture).
- Append-only : on ne reecrit ni ne supprime jamais une ligne. Une derniere ligne tronquee par un
  crash est ignoree a la lecture et nettoyee a la prochaine ecriture.
- `id` : les prompts de notre banque ont des ids stables `<categorie>-<index>` (ex. `arith-003`).
  Ne jamais modifier le texte d'un prompt existant : `plan` signale toute derive entre la banque et
  le store.

## Fusions git

`.gitattributes` de ce dossier declare `*.jsonl merge=union` : deux branches qui ajoutent des
lignes se fusionnent sans conflit. Cela ne protege PAS le budget : si deux personnes interrogent le
meme endpoint en parallele, les deux paient. Regle : une seule personne interroge les `ep-*` a la
fois, apres `git pull`, et pousse le fichier tout de suite apres. `budget` compare ce que le serveur
dit consomme a ce que le store contient et signale l'ecart.

Rien de secret ici : jamais de cle API, jamais de donnees lourdes.
