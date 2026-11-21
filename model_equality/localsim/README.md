# Mini-compétition locale (nœud de connexion JURECA)

Reproduit la tâche sur nos propres modèles, avec un accès illimité et les probabilités des tokens.
Pour chaque modèle de base, les 9 mêmes déploiements que les références :

| Variante | Modèle (vérité) | Comment |
|---|---|---|
| plain, system prompt, persona, watermark, température 1,0 | S | message système / watermark rouge-vert de `wmkit` (gamma 0,25, delta 2) / T = 1,0 au lieu de 0,7 |
| fp8 | T1 | poids arrondis en fp8 (e4m3, une échelle par ligne) puis remis en fp16 ; la tête de sortie n'est pas touchée |
| LoRA 0,3 / 0,7 / 1,5 | T2 / T3 / T4 | un LoRA (rang 8, q/k/v/o) entraîné ici sur des instructions Alpaca, appliqué à 3 forces |

Modèles par défaut : `qwen05` (Qwen2.5-0.5B-Instruct), `llama1b` (Llama-3.2-1B-Instruct, miroir sans
restriction), `smol360` (SmolLM2-360M-Instruct, proche des références). Aussi : `qwen15` (deux tailles d'une même
lignée). Les endpoints sont anonymes (`sim-01`...), la vérité est dans `endpoints.csv`.

## Lancer (sur le nœud de connexion)

```bash
cd temp-hackhaton && source .venv/bin/activate
nvidia-smi                      # choisir une RTX 8000 libre
export CUDA_VISIBLE_DEVICES=0   # une seule carte, les nœuds sont partagés
python model_equality/localsim/localsim.py \
    --prompts model_equality/constitution/prompts_v0_C_fp8.jsonl \
    --repeats 5 --out model_equality/localsim/run_C
```

- Premier lancement : télécharge les modèles (le nœud de connexion a Internet) et entraîne un LoRA par modèle
  (150 pas, quelques minutes). Le LoRA est gardé dans `lora_<base>.pt` et réutilisé ensuite.
- Le script reprend là où il s'est arrêté (réponses déjà écrites sautées).
- Rester court sur le nœud partagé : `--max-new-tokens 64` par défaut suffit pour nos réponses courtes.
  Si ça dépasse ~15 minutes, passer par un job Slurm.

## Sorties

- `answers.jsonl` : même format que le serveur, donc nos scripts d'analyse marchent tels quels.
- `firsttok.jsonl` : les 20 premiers tokens les plus probables pour la première position, avec leur probabilité,
  sous la vraie distribution de tirage (température et watermark compris). N'existe pas sur le vrai serveur :
  sert seulement à **choisir** les questions (où le modèle hésite, ce que fp8 ou le watermark déplacent).
- `endpoints.csv` : la vérité (endpoint, famille, modèle, variante).

## Testé

Test à vide sur CPU avec un modèle minuscule (`--bases tiny`) : toutes les variantes tournent, les fichiers sont au
bon format, fp8 arrondit les poids (erreur relative ≈ 2,6 %), le LoRA produit des deltas non nuls, le watermark
déplace la distribution du premier token. Pas encore testé sur GPU ni sur les vrais modèles.
