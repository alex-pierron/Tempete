# Guide d'équipe : European AI Championship Paris 2026 (hackathon 24 h)

> Point d'entrée unique. Rédigé le 21/11/2026 à partir des 3 sujets (PDF du dépôt `Tempete`,
> dossier `Sujets/`), des dépôts officiels Hugging Face `SprintML/*` et de la keynote de Pierre
> Fernandez (Meta FAIR). Tout ce qui n'est pas vérifié est marqué **[À VÉRIFIER]**.

Sommaire : 1. Les 3 tâches en un tableau · 2. Règles communes · 3. Mise en route (local + JURECA)
· 4. Fiches par tâche · 5. Organisation de l'équipe · 6. Check-lists · 7. Sources.

Documents liés :
- `docs/hackathon/jureca.md` : connexion GPU (JURECA, Jülich), scripts officiels, Slurm.
- `docs/hackathon/keynote.md` : résumé complet des 57 slides de la keynote.
- `docs/hackathon/model_equality.md`, `unlearning.md`, `wmloc.md` : fiches détaillées et commandes.

## 1. Les 3 tâches

Organisateurs : CISPA (Adam Dziedzic, Franziska Boenisch), équipe SprintML. Serveur commun
`http://34.147.229.55`, leaderboard `http://34.147.229.55/leaderboard_page` (meilleur score par
équipe ; public = 30 %, classement final = 70 % caché).

| | Model Equality | Unlearning | Watermark Localization |
|---|---|---|---|
| Task id serveur | `33-model-equality` | `31-unlearning` | inconnu **[À VÉRIFIER]** |
| Entrée | 33 endpoints LLM `ep-01..ep-33` (+ 9 `ref-*` d'entraînement) | ViT-B/16 100 classes + 3 500 images à oublier (sans labels) | 1 530 images 512×512 (train 510 avec masques, test 1 020) |
| À rendre | CSV `endpoint,family,model` (33 lignes) | state dict float32 ≤ 450 MB | `.pt` `{"image_names", "masks"}` masks [N,128,128] 0/1, ≤ 20 MB |
| Métrique | 0,1 × MCC famille + 0,9 × MCC modèle (paires) | forget × retain, porte 70 % d'accuracy | mIoU 2 classes sur grille 128×128, moyenne par image |
| Ressource rare | **100 réponses par endpoint pour toute la compétition** | aucune donnée de rétention | dataset en accès restreint (token HF autorisé) |
| Limite de soumission | 1 toutes les 30 min (échec compris) | non documentée | non documentée |
| Données HF | `SprintML/model_equality` (public) | `SprintML/unlearning` (public, 344 Mo) | `SprintML/Hackathon_Watermark_Localisation_Task(_Paris)` (restreint) |
| Lien avec nos 3 papiers | MET (Gao, Liang, Guestrin) : déjà dans `wmkit.audit.met` | SCRUB : déjà dans `wmkit.unlearn` | Pixel Seal : déjà dans `wmkit` (poids locaux) |

## 2. Règles communes

- Clé API d'équipe : header `X-API-Key`. À mettre **uniquement** dans `.env` (`HACKATHON_API_KEY=`),
  jamais dans le code, un commit, un chat ou un log.
- Soumission : `POST {serveur}/submit/{task_id}`, champ multipart `file`. Le serveur peut mettre
  jusqu'à ~500 s pour noter. Réponse : score public, meilleur score précédent, amélioration.
- **Ne pas optimiser pour le leaderboard public** (30 %) : les sujets insistent, le privé peut
  s'effondrer. Juger une méthode sur nos évaluations locales (ref-endpoints, proxys, split train).
- Toute soumission passe par nos CLI : validation locale du format **avant** envoi, journal
  `data/hackathon/submissions/log.jsonl`, refus sans `--yes`, garde du délai de 30 min (Model Equality).

## 3. Mise en route

### 3.1 Local (Mac)

```bash
git clone git@github.com:alex-pierron/Tempete.git && cd Tempete && git checkout victor
make setup            # .venv Python 3.12 (torch, timm, videoseal...) ; PAS un simple uv sync
cp .env.example .env  # puis remplir HACKATHON_API_KEY, HF_TOKEN (compte autorisé), JURECA_*
make test             # suite rapide (doit être verte)
```

Code de la compétition : package `src/hackathon/` (un sous-package par tâche + `common/`). Il ne
dépend que de ce que contiennent les environnements officiels de JURECA (torch, timm, numpy,
requests, pandas, scipy, scikit-learn) : le même code tourne en local et sur le cluster avec
`PYTHONPATH=<repo>/src`.

### 3.2 GPU : JURECA (Jülich)

Résumé (détails et commandes exactes : `docs/hackathon/jureca.md`) :
1. Chaque membre : compte JuDoor, MFA (TOTP, demandé à chaque connexion), projet
   `training2666` rejoint, accord JURECA signé, clé **ed25519** déposée avec une clause
   `from="..."` couvrant le réseau du hackathon (active sous 15 min) ; connexion `ssh juelich`.
   Matériel : 1 nœud entier par job = 4 × A100 40 GB, 128 cœurs, 512 GB de RAM.
2. Le **chef d'équipe** lance une seule fois `hackathon_setup.sh` (projet `training2666`) :
   dossier partagé `/p/scratch/training2666/<chef>/<équipe>/`, ACL, `uv`, téléchargement des 3
   datasets, un `.venv` par tâche. Il faut `HF_TOKEN` exporté (dataset localisation restreint).
3. Les **autres** lancent `teammate.sh` (jamais les deux scripts depuis le même compte).
4. Jobs GPU : `sbatch` avec `--account=training2666 --reservation=cispahack --partition=dc-gpu
   --gres=gpu:1 --cpus-per-task=8` ; nos modèles prêts dans `jureca/*.sbatch`.
5. À chaque connexion : `jutil env activate -p training2666` puis `source ~/.bashrc`.

## 4. Fiches par tâche (synthèse ; détails dans les fiches dédiées)

### 4.1 Model Equality (`33-model-equality`)

- **Question** : pour chaque paire d'endpoints, même famille (même modèle de base) ? même modèle
  (même checkpoint + même quantization + même LoRA) ? Watermark, system prompt du fournisseur,
  persona, température ou stratégie de décodage **ne changent pas** l'identité ; quantization
  (ex. fp8) et LoRA (même à une autre force) **la changent**. Même famille ne suffit pas.
- **Score** : MCC (Matthews), sur 528 paires pour la famille (10 %), sur les paires de même famille
  vraie pour le modèle (90 %), chacun borné à 0. Reconnaître les familles seules rapporte au plus 0,1.
- **Budget** : 100 réponses par endpoint pour TOUTE la compétition et toute l'équipe ; 256 tokens
  générés max, 1 024 tokens de prompt max, 4 requêtes simultanées max, température/seed ignorés,
  system messages refusés, requêtes invalides non décomptées.
- **Entraînement gratuit** : `ref-01..ref-09` (illimités, jamais notés), réponses publiées :
  un modèle S décliné en plain / system prompt / persona / watermark / température 1,0, et 4
  variantes T1 fp8, T2 LoRA 0,3, T3 LoRA 0,7, T4 LoRA 1,5. Calibrer toute la méthode dessus avant
  de dépenser une seule requête `ep-*`.
- **Stratégie** : MMD noyau de Hamming (papier MET), empreintes sur prompts quasi déterministes,
  regroupement à 2 niveaux ; réponses stockées et versionnées (`artifacts/model_equality/`), jamais
  redemandées.

### 4.2 Unlearning (`31-unlearning`)

- **Entrée** : `model.pt` (timm `vit_base_patch16_224`, 100 classes, LayerScale 1e-4) et
  `forget_images.npz` (uint8 [3500, 224, 224, 3], sans labels). Aucune autre donnée.
- **Cible** : se comporter comme le modèle « gold » réentraîné sans ces images (moyenne de
  plusieurs réentraînements) : distribution des pertes (moyenne + écart-type) ET distribution
  prédite image par image (KL, softmax à température 2), sur les images exactes et sur des copies
  altérées cachées. La montée de gradient naïve est explicitement pénalisée.
- **Score** : `forget × retain`, avec retain = accuracy relative sur un control set caché ×
  1/(1 + KL(orig‖sub)) ; **porte : < 70 % d'accuracy sur un validation set caché ⇒ 0**.
- **Format** : state dict float32 seulement, chargé en `strict=True`, sans NaN/Inf, ≤ 450 MB,
  LoRA/adapters refusionnés. Le serveur évalue en float16 autocast.
- **Stratégie** : partir de SCRUB (papier recommandé, `wmkit.unlearn`), viser le comportement du
  gold plutôt que l'oubli brutal ; proxys locaux (pseudo-labels de l'original, copies altérées
  simulées, dérive sur images hors forget) ; garde-fou sur la rétention avant chaque envoi.

### 4.3 Watermark Localization

- **Entrée** : images 512×512 tatouées par **PixelSeal, TrustMark ou StegaStamp** sur un
  rectangle (aspect ≤ 16:1 ; ≤ 2:1 pour TrustMark), puis éditées : outpainting, retrait d'objet,
  ajout d'objet, transplantation d'objet. Le test fournit pour chaque image **la méthode et le
  message embarqué**.
- **Sortie** : masque binaire 128×128 (cellule 4×4 px ; vérité = 1 si ≥ 50 % de pixels tatoués).
- **Score** : mIoU des classes 0 et 1 par image (union vide ⇒ 1), moyenne sur les images.
- **Format** : `.pt` `{"image_names": names, "masks": tensor [N,128,128]}`, noms uniques, valeurs
  0/1, **≤ 20 MB** : stocker en uint8/bool (en float32, 1 020 masques font 67 MB ⇒ rejet).
- **Kit officiel** (restreint) : wrappers unifiés `decode(images) -> bits` + cartes de features
  intermédiaires + embedders pour les 3 méthodes, downsampling exact, mIoU local, template.
- **Stratégie** : les extracteurs Meta sortent une carte par pixel (1 + nbits) avant la moyenne
  spatiale ; comparer chaque position au message connu donne une carte de présence du tatouage.
  Ensuite : seuils calibrés par méthode sur le train, puis tête de segmentation entraînée sur des
  données synthétiques illimitées (on a les embedders). C'est exactement le problème que la
  keynote présente (slide 24 : « les tatouages globaux ratent les éditions locales »).

## 5. Organisation de l'équipe (3 personnes, 24 h)

- Dépôt `alex-pierron/Tempete` :
  - `main` : un dossier par tâche (`unlearning/`, `model_equality/`, `watermark_localization/`)
    avec le code de démarrage officiel, `slurm_job.sh` générique, `.env.example`
    (`HACKATHON_API_KEY`, `HF_TOKEN`). Les méthodes vont dans des fichiers à part ; les scripts
    officiels ne sont modifiés qu'au minimum.
  - `victor` : historique séparé ; boîte à outils `wmkit` (Pixel Seal, attaques, MET, SCRUB) et kit
    `src/hackathon` (validation, scores locaux, soumission journalisée, banc synthétique, jobs Slurm).
  - Les deux utilisent la même variable `HACKATHON_API_KEY` et le même format de réponses Model
    Equality (format officiel `{"endpoint","id","repeat","prompt","response"}`, fichier d'équipe
    `model_equality/answers.jsonl` sur `main`).
- Branches : chacun sa branche, `main` = sujets et version stable partagée.
- Un responsable par tâche au départ, rotation possible ; une seule personne soumet par tâche
  (le journal de soumission et le délai de 30 min sont par équipe, pas par personne).
- Réponses Model Equality : **toujours committées** dans `artifacts/model_equality/` (le budget est
  commun et ne se rachète pas).
- Toutes les 2 h : point de 5 min (scores locaux, soumissions faites, budget restant).

## 6. Check-lists

Avant de commencer :
- [ ] Comptes Jülich actifs pour les 3, clés SSH déposées, connexion testée.
- [ ] Chef d'équipe désigné, `hackathon_setup.sh` passé, `teammate.sh` passé pour les 2 autres.
- [ ] Clé API d'équipe dans `.env` (local) et dans l'environnement JURECA, jamais committée.
- [ ] Token HF d'un compte **autorisé** pour le dataset localisation (sinon demander l'accès).
- [ ] Task id serveur de la localisation (`HACKATHON_TASK_WMLOC`).

Avant chaque soumission :
- [ ] Validation locale OK (`python -m hackathon.<tâche> validate ...`).
- [ ] Score local / proxy meilleur que la soumission précédente, pour une raison comprise.
- [ ] Pas de sur-ajustement au public (30 %).
- [ ] Délai respecté (Model Equality : 30 min, un échec compte).

## 7. Sources

- Sujets : `Sujets/*.pdf` (dépôt Tempete, branche `main`).
- Dépôts officiels : `https://huggingface.co/datasets/SprintML/hackathon` (setup JURECA),
  `SprintML/model_equality`, `SprintML/unlearning`, `SprintML/Hackathon_Watermark_Localisation_Task_Paris`.
- Keynote : `https://pierrefdz.github.io/assets/slides/2026-11-21_image_watermarking_cispa.html`.
- Papiers cités par les sujets : MET (arXiv 2410.20247), RUT (2506.06975), AgentProv (2609.00052) ;
  SCRUB (2302.09880), Zero-Shot Unlearning (2201.05629), SalUn (2310.12508), NPO (2404.05868) ;
  Pixel Seal (2512.16874), TrustMark (ICCV 2025), StegaStamp (CVPR 2020), Watermark Anything (ICLR 2025).
