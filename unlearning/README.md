# Unlearning (tâche `31-unlearning`)

Sujet complet : [../Sujets/Unlearning.pdf](../Sujets/Unlearning.pdf)

## Objectif

On reçoit un ViT-B/16 (timm `vit_base_patch16_224`, 100 classes, LayerScale init `1e-4`) entraîné sur un dataset privé, et
3 500 images à lui faire « oublier ». Le modèle soumis doit se comporter comme un modèle **réentraîné de zéro sans ces
images** (le modèle « gold »), sans abîmer le reste.

- Pas de labels pour le forget set, et aucune autre donnée d'entraînement : on ne travaille qu'avec `model.pt` et les
  images à oublier.
- Il ne suffit pas de faire chuter la précision sur le forget set (gradient ascent naïf) : il faut reproduire la
  **distribution des pertes** et **la mauvaise classe choisie** par le modèle gold, image par image, y compris sur des
  versions altérées des images (cachées).

## Score

| Élément | Calcul |
|---|---|
| Porte d'utilité | < 70 % de précision sur un set de validation caché ⇒ **score 0** |
| `mean_std_score` | `1 - clip(d_sub / d_orig, 0, 1)`, avec `d = \|mean(L) - mean(L_gold)\| + \|std(L) - std(L_gold)\|` (cross-entropy par image) |
| `prediction_score` | `1 - clip(mean KL(p_gold ‖ p_sub) / mean KL(p_gold ‖ p_orig), 0, 1)`, softmax à **température 2** |
| `forget_score` | `mean_std_score × prediction_score`, chacun moyenné sur images exactes + copies altérées |
| `retain_score` | `clip(acc_sub / acc_orig, 0, 1) × 1 / (1 + mean KL(p_orig ‖ p_sub))` sur un set de contrôle caché |
| **Final** | `forget_score × retain_score` |

Leaderboard public = ~30 % des données de scoring, classement final = les 70 % restants.

## Format de soumission

- Un `state_dict` `.pt`/`.pth` qui se charge en `strict=True` dans `build_model()` de `task_template.py`.
- Poids **float32** uniquement, aucun NaN/Inf, **≤ 450 Mo** (un ViT-B float32 fait ~345 Mo).
- LoRA, adapters ou autre couche PEFT : à **fusionner** dans les poids d'origine avant de soumettre.
- Le serveur évalue en `eval()` sous autocast float16.

## Fichiers

| Fichier | Rôle |
|---|---|
| `task_template.py` | Officiel. Construit le modèle, charge les données, `preprocess()` identique au serveur, `predict()`, et le stub `unlearn()` à remplir. L'exécuter écrit `submission.pt` (baseline ≈ 0). |
| `submission.py` | Officiel. Envoie un `.pt` au serveur. Seule modification : clé API lue depuis `HACKATHON_API_KEY` et chemin passé en argument. |
| `requirements.txt` | Dépendances officielles (torch, timm, numpy, requests). |
| `model.pt`, `forget_images.npz` | Données à télécharger, ignorées par git. |

## Démarrage

```bash
cd unlearning
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# Données (~870 Mo) : model.pt + forget_images.npz, à côté des scripts
uvx --from huggingface_hub hf download SprintML/unlearning model.pt forget_images.npz --repo-type dataset --local-dir .

python task_template.py                 # vérifie le chargement, écrit submission.pt
export HACKATHON_API_KEY="..."          # clé de l'équipe, jamais dans un fichier commité
python submission.py submission.pt      # soumission
```

Sur JURECA, lancer sur GPU depuis ce dossier : `mkdir -p output && sbatch ../slurm_job.sh task_template.py`.

## Pièges

- Images en uint8 `[N, 224, 224, 3]`, canaux en dernier. Passer par `preprocess()` (normalisation `[-1, 1]`), sans
  redimensionner ni renormaliser.
- Sauvegarder avec `{k: v.detach().cpu().float() ...}` comme le fait `main()`, sinon le dtype peut être rejeté.
- Surveiller la précision globale : la porte des 70 % annule tout.

## Références

- Kurmanji et al., *Towards Unbounded Machine Unlearning* (SCRUB), NeurIPS 2023 — <https://arxiv.org/abs/2302.09880>
- Chundawat et al., *Zero-Shot Machine Unlearning*, IEEE TIFS 2023 — <https://arxiv.org/abs/2201.05629>
- Fan et al., *SalUn*, ICLR 2024 — <https://arxiv.org/abs/2310.12508>
- Zhang et al., *Negative Preference Optimization*, COLM 2024 — <https://arxiv.org/abs/2404.05868>
