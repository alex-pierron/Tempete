# Model Equality Testing (tâche `33-model-equality`)

Sujet complet : [../Sujets/Model Equality Testing.pdf](../Sujets/Model%20Equality%20Testing.pdf)

## Objectif

33 endpoints LLM (`ep-01` à `ep-33`). Pour chaque paire, il faut dire :

- **même famille ?** : construits sur le même modèle de base ;
- **même modèle ?** : même checkpoint, **même quantization** et **même adapter LoRA**.

Un watermark de décodage, un system prompt ajouté par le fournisseur ou une autre stratégie de décodage **ne changent
pas** le modèle. Une autre base, une autre quantization ou un autre LoRA en font **un modèle différent**. Être de la même
famille ne suffit pas.

## Score

MCC (corrélation de Matthews, plancher à 0) sur les 528 paires :

```
score = 0.1 × MCC_famille (toutes les paires) + 0.9 × MCC_modèle (paires d'une même famille uniquement)
```

Trouver les familles rapporte donc au plus 0.1. L'essentiel du score vient de la distinction des variantes
quantifiées et LoRA à l'intérieur d'une famille.

> **Ne pas optimiser pour le score public.** Le leaderboard n'utilise qu'une partie des paires. Une soumission calée
> dessus peut tomber à presque 0 au classement final. Valider sur les endpoints de référence.

## Budget et limites (pour toute la compétition)

- **100 réponses par endpoint et par équipe, au total.** Chaque réponse coûte une requête.
- ≤ 256 tokens générés par réponse, ≤ 1 024 tokens de prompt, **4 requêtes en parallèle** au maximum.
- `temperature` et `seed` sont ignorés, les system messages sont rejetés, les requêtes invalides ne sont pas décomptées.
- **Une soumission toutes les 30 minutes** (une erreur déclenche aussi le délai).

⚠️ Le budget est partagé par toute l'équipe : on coordonne avant d'interroger les `ep-*`, et on **commite
`answers.jsonl`** pour que tout le monde réutilise les réponses déjà payées.

## Endpoints d'entraînement

`ref-01` à `ref-09` : un autre petit modèle, avec les mêmes déguisements et pièges. Ils sont **illimités** et jamais
scorés. Les réponses attendues sont dans `practice_answers.csv` :

| Endpoint | Modèle | Variante |
|---|---|---|
| ref-01, 03, 05, 06, 08 | S | plain, system prompt, persona, watermark, température 1.0 |
| ref-02 | T1 | quantization fp8 |
| ref-09 / ref-04 / ref-07 | T2 / T3 / T4 | LoRA à force 0.3 / 0.7 / 1.5 |

## Format de soumission

Un CSV, une ligne par endpoint, colonnes `endpoint,family,model` (en-tête optionnel, labels libres sans virgule). Deux
endpoints avec le même label `model` doivent aussi avoir le même label `family`.

```csv
endpoint,family,model
ep-01,A,A1
ep-02,B,B1
ep-03,A,A2
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `task_template.py` | Officiel. Budget restant, `query()` (reprenable, ajoute dans `answers.jsonl`), `solve()` à remplir, écrit `submission.csv`. Mettre `QUERY = True` pour interroger. |
| `query_reference.py` | Officiel. Interroge `ref-*` avec les prompts de `reference_prompts.txt` et écrit `reference_answers.jsonl`. |
| `score_practice.py` | Officiel. Calcule le MCC local sur les `ref-*` : `python score_practice.py mon_answer.csv`. |
| `submission.py` | Officiel. Envoie le CSV. |
| `practice_answers.csv` | Vérité terrain des `ref-*`. |

Seule modification apportée aux scripts officiels : la clé API est lue depuis `HACKATHON_API_KEY`, et `submission.py`
prend le chemin du fichier en argument.

## Démarrage

```bash
cd model_equality
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
export HACKATHON_API_KEY="..."          # clé de l'équipe, jamais dans un fichier commité

python task_template.py                 # affiche le budget restant par endpoint (ne consomme rien)
# 1. mettre au point la méthode sur ref-* : écrire reference_prompts.txt, puis
python query_reference.py
python score_practice.py mon_answer_ref.csv
# 2. seulement ensuite : écrire prompts.txt, passer QUERY = True dans task_template.py et le relancer
python submission.py submission.csv
```

## Références

- Gao, Liang, Guestrin, *Model Equality Testing: Which Model Is This API Serving?* (MET), ICLR 2025 — <https://arxiv.org/pdf/2410.20247>
- Zhu et al., *Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test* (RUT), ICLR 2026 — <https://arxiv.org/pdf/2506.06975>
- Wang et al., *AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes*, ICLR 2026 — <https://arxiv.org/pdf/2609.00052>
