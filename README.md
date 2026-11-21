# Tempête : European AI Championship Paris 2026 (CISPA)

Dépôt de l'équipe pour les trois sujets du hackathon. Chaque tâche a son dossier, avec son README, le code de démarrage
officiel et ses dépendances.

| Tâche | Dossier | À rendre | Métrique | Limites clés |
|---|---|---|---|---|
| Unlearning | [`unlearning/`](unlearning/) | `state_dict` ViT-B/16 en float32 (`.pt`, ≤ 450 Mo) | `forget × retain` | < 70 % de précision en validation ⇒ score 0 |
| Model Equality Testing | [`model_equality/`](model_equality/) | CSV `endpoint,family,model` | `0.1·MCC_famille + 0.9·MCC_modèle` | 100 réponses par endpoint **pour toute la compétition**, 1 soumission / 30 min |
| Watermark Localization | [`watermark_localization/`](watermark_localization/) | `.pt` `{"image_names", "masks"[N,128,128]}` (≤ 20 Mo) | mIoU par image | dataset HF à accès restreint |

Les sujets originaux sont dans [`Sujets/`](Sujets/). Pour chaque tâche, le leaderboard public ne couvre qu'environ 30 %
des données de scoring. Le classement final se fait sur le reste, il ne faut donc pas sur-optimiser le score public.

- Leaderboard : <http://34.147.229.55/leaderboard_page>
- Repos officiels : [unlearning](https://huggingface.co/datasets/SprintML/unlearning),
  [model_equality](https://huggingface.co/datasets/SprintML/model_equality),
  [watermark](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task),
  [setup JURECA](https://huggingface.co/datasets/SprintML/hackathon)

## Organisation du dépôt

```text
.
├── README.md
├── .env.example            # variables à définir : HACKATHON_API_KEY, HF_TOKEN
├── slurm_job.sh            # job GPU JURECA générique
├── Sujets/                 # PDF des sujets
├── unlearning/             # task_template.py, submission.py, requirements.txt (+ données ignorées par git)
├── model_equality/         # task_template.py, query_reference.py, score_practice.py, submission.py, practice_answers.csv
└── watermark_localization/ # README + starter kit à décompresser (données dans data/, ignorées par git)
```

Chaque dossier a son propre environnement (`.venv`), comme sur JURECA. Les données, poids, archives et sorties
(`*.pt`, `*.npz`, `*.zip`, `data/`, `output/`) ne sont pas versionnés.

## Installation locale

```bash
cp .env.example .env        # remplir la clé API d'équipe (et HF_TOKEN pour le watermark)
set -a; source .env; set +a

cd <dossier-de-la-tâche>
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Le téléchargement des données propre à chaque tâche est décrit dans son README.

## Sur JURECA

Suivre le [README officiel](https://huggingface.co/datasets/SprintML/hackathon) : le chef d'équipe lance
`hackathon_setup.sh` une seule fois, puis chaque coéquipier lance `teammate.sh`. Les données arrivent dans
`/p/scratch/training2666/<chef>/<équipe>/`. Cloner ce dépôt au même endroit, créer le `.venv` de la tâche (le cache uv
est partagé), puis :

```bash
cd unlearning && mkdir -p output && sbatch ../slurm_job.sh task_template.py
squeue -u "$USER"
```

⚠️ Le `teammate.sh` officiel contient `PROJECT_NAME="training2625"`, alors que le README et `hackathon_setup.sh`
utilisent `training2666`. Corriger cette ligne avant de le lancer.

## Règles d'équipe

- **Jamais de clé API ni de token dans un fichier commité.** Les scripts lisent `HACKATHON_API_KEY` dans
  l'environnement.
- **Model Equality** : le budget de requêtes est commun et ne se recharge pas. On se coordonne avant d'interroger les
  `ep-*`, et on commite `model_equality/answers.jsonl`.
- Les scripts officiels ne sont modifiés qu'au minimum (lecture de la clé API, chemin de soumission en argument). Le code
  de chaque méthode va dans des fichiers à part.
- Branches : `main` = avancée du projet (ce qui est validé par l'équipe) ; `victor` = branche de travail de Victor
  (recherche, essais), fusionnée dans `main` quand c'est validé.

## Kit d'outils de l'équipe (`src/hackathon`, `wmkit`)

En plus du code officiel par dossier, le dépôt contient un kit typé et testé (`make hk-test`) qui tourne en local et sur
JURECA avec les `.venv` officiels (`PYTHONPATH=$PWD/src`).

| Besoin | Où |
|---|---|
| Point d'entrée, règles, check-lists | [`docs/hackathon/GUIDE.md`](docs/hackathon/GUIDE.md) |
| Présentation de synthèse (HTML + PDF) | [`docs/hackathon/presentation/`](docs/hackathon/presentation/) |
| GPU JURECA pas à pas (accès, MFA, clé `from=`, Slurm) | [`docs/hackathon/jureca.md`](docs/hackathon/jureca.md) |
| Processus idée → expérience → job → soumission | [`docs/hackathon/process.md`](docs/hackathon/process.md), `python -m hackathon.lab` |
| Boucle de recherche et format des rapports | [`docs/hackathon/research_loop.md`](docs/hackathon/research_loop.md), [`reports/`](reports/) |
| Unlearning | `python -m hackathon.unlearning` ([doc](docs/hackathon/unlearning.md)) |
| Model Equality | `python -m hackathon.model_equality` ([doc](docs/hackathon/model_equality.md)) |
| Watermark Localization | `python -m hackathon.wmloc` ([doc](docs/hackathon/wmloc.md)) |
| Keynote Meta résumée | [`docs/hackathon/keynote.md`](docs/hackathon/keynote.md) |

Installation du kit en local : `make setup` (crée `.venv` Python 3.12 avec torch, timm, videoseal...), puis `make hk-test`.
Toute soumission passe par la validation locale et exige `--yes` ; chaque tentative est journalisée.
