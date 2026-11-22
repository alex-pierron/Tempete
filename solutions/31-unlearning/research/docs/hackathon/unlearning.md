# Tâche Unlearning (`31-unlearning`) : kit de l'équipe

Modèle : ViT-B/16 timm (`vit_base_patch16_224`, 100 classes, `init_values=1e-4`), 85 893 988
paramètres. Forget set : 3500 images uint8 `[3500, 224, 224, 3]`, sans labels. Rien d'autre
n'est fourni : pas de données de rétention, pas de modèle gold, pas de control set.

Code : `src/hackathon/unlearning/`. Tests : `tests/hackathon/test_hk_unlearning_*.py`.
Job JURECA : `jureca/unlearning.sbatch`.

## 1. Ce que dit le score, et ce qu'on a mesuré

```
final  = forget x retain
forget = mean_std_score x prediction_score    (chacun moyenné sur : images exactes, copies altérées cachées)
retain = clip(acc_sub / acc_orig, 0, 1) x 1 / (1 + mean KL(p_orig || p_sub))      (T = 2, control set caché)
gate   : acc < 70 % sur un validation set caché => score 0
```

Conséquences directes :

- `sub = orig` donne 0 (baseline identité). Le score est une **distance au gold**, pas un taux
  d'oubli : être "plus faux" que le gold est aussi pénalisé que ne pas bouger.
- Les deux composantes du forget se multiplient ET se multiplient par retain : un modèle
  agressif perd plus sur retain qu'il ne gagne sur forget.
- Le gold répond sur une image oubliée comme sur n'importe quelle image jamais vue : confiance
  modérée, **classe fausse choisie** quand il se trompe (c'est ce que mesure `prediction_score`).

Mesures sur le vrai `model.pt` (128 premières images oubliées, M5/MPS, script jetable) :

| mesure | valeur |
|---|---|
| confiance moyenne de l'original sur le forget set | 0,9997 (min 0,988 sur 64 images) |
| marge médiane top1 - top2 (logits) | 12,2 (logit max moyen 22,9) |
| confiance sur flip / crop 60-90 % / JPEG q50 | 1,000 / 1,000 / 0,999 (copies douces : aucun effet) |
| confiance sur crop ~35 % de l'aire | 0,926 ; accord de l'argmax avec l'image exacte : 87,5 % |
| confiance sur crop ~15 % de l'aire | 0,808 ; accord de l'argmax : 43,8 % |
| confiance sur images hors distribution (COCO/DIV2K/Kodak) | 0,75 |

L'original est **mémorisé à l'extrême** sur le forget set : loss quasi nulle, y compris sur les
copies altérées légères. Un gold (jamais vu ces images) aura au contraire une loss moyenne et
surtout un écart-type non négligeables (quelques images mal classées). Toute la difficulté est de
reproduire cette hétérogénéité, image par image.

## 2. Stratégie

Six méthodes derrière `unlearn(model, forget_images, config) -> model` (`methods.py`) :

| méthode | idée | risque |
|---|---|---|
| `identity` | copie intacte | score ~0 ; sert à tester la chaîne complète |
| `scrub` | SCRUB sans labels : max-step sur le forget (KL(teacher||student), **plafonné** par `scrub_kl_cap`, donc pas la montée de gradient naïve) alternée avec min-step (KL vers l'enseignant) sur des images de rétention proxy | dérive plus forte sur retain (mesuré, §5) ; vise "être différent", pas "être le gold" |
| `gold_soft` | **recommandée** : cible = comportement du gold | la température `target_temperature` doit être calibrée sur le leaderboard |
| `trust_region` | région de confiance : l'élève est tiré vers une cible sur les images exactes du forget set (lots stratifiés), arrêt dès que KL(original ‖ élève) à T = 1 atteint le rayon (`kl_radius`, ou `radius_fraction` × KL de la cible) | rayon = budget d'oubli ; détail : `rapport/rapport.tex` §3 |
| `band_ascent` (V5) | montée de gradient bornée par image (bande à deux charnières sur la perte du pseudo-label), réservée aux images dont l'original doute sur des vues fortes, dosée par niveaux choisis sur un pli témoin | **abandonnée** : sélection au hasard sous le masquage (étape 0) ; locale (audit/exact 0,25-0,39) mais sans score (≤ 0,046) ; détail : `rapport/rapport.tex` §3 (V5) et §6 (campagne V5) |
| `band_ascent` V5.1 | V5 avec la bande sur la marge des logits, la retenue des autres images en KL à T = 2 et la dose choisie sous un retain estimé de 0,8 | **abandonnée** : la marge fait dépasser la bande (arrêt dur au premier niveau à lr 1e-4 et 3e-5) ; à 1e-5, 0,080 mais retain 0,725 et audit/exact 0,84 ; détail : `rapport/rapport.tex` §3 (V5.1) et §6 (campagne V5.1) |
| `dumb` (V_Dumb) | la plus simple : la perte de chaque image oubliée (sur l'argmax de l'original) monte jusqu'à `dumb_target_loss` (0,45, perte mesurée de l'original sur une vue forte d'aire 0,5) et pas au-delà (charnière), une vue forte de chaque image reste à la sortie de l'original (KL, `retain_weight`) | **abandonnée** : immobile avec l'ancre, effondrée sans ; aucune dose utilisable (meilleur 0,019) ; détail : `rapport/rapport.tex` §3 (V_Dumb) et §6 (campagne V_Dumb) |
| V_Fusion (`scrub` + synthétiques) | SCRUB conforme dont l'ancre devient des images synthétiques tirées du seul original (Plug-In Inversion, confiance plafonnée à 0,9), en deux lots : classes du forget set (A) et autres classes (B), 9 classes mises à part (C) ; portes de qualité a, b, c, c', d | **abandonnée sous cette génération** : images saturées (confiance ≈ 1), le modèle cassé reste d'accord avec l'original sur elles ; les 4 bras d'ancre s'effondrent (retain ≤ 0,025) ; détail : `rapport/rapport.tex` §3 (V_Fusion) et §6 (campagne V_Fusion) |

`trust_region` a trois cibles (`trust_target`) :

- `tempered` (V1, défaut) : `softmax(z_orig / target_temperature)` sur l'image exacte, le même
  adoucissement pour toutes les images ;
- `views` (V2) : pour chaque image, la moyenne sur `target_views` vues fortes
  (`strong_view`, aire `target_view_min_area`) de `softmax(z_orig(vue) / target_temperature)`,
  calculée une fois avant l'entraînement, sans gradient. Chaque image garde sa confiance et sa
  classe fausse préférée (le gold est hétérogène) ; le rayon fixe la quantité d'oubli. Les vues
  sont tirées d'une graine issue de `seed`, distincte de celles du banc. Spec :
  `experiments/unlearning/trust-region-views.toml`. Mesure de la cible sur le vrai modèle : fiche
  de la spec ;
- `calibrated` (V3) : le score compare deux objets du gold, des statistiques de pertes modèle par
  modèle (`mean_std`) et le mélange de leurs distributions (`prediction`). Sur les mêmes vues
  fortes : direction à T = 2 = mélange des vues `p̂ = moy softmax(z_v/2)`, couvert par l'original
  adouci, `p_dir = (1 − b)·p̂ + b·softmax(z_orig/target_temperature)` (b = `trust_hedge`) ; puis
  le seul logit de la pseudo-étiquette y est déplacé, en forme close
  (`z_y ← A − log(expm1(ℓ*))`, A = logsumexp des autres logits), pour que la perte à T = 1 vaille
  `ℓ* = ℓ_mix + λ·(ℓ̄ − ℓ_mix)` (ℓ̄ : perte moyenne par vue, ℓ_mix : celle de `p_dir`,
  λ = `trust_lambda`). Les rapports entre classes fausses de `p_dir` sont gardés exactement.
  Spec : `experiments/unlearning/trust-region-calibrated.toml` ; justification :
  `rapport/rapport.tex` §3.

Deux options valent pour toutes les cibles (défauts : comportement V1/V2 inchangé, au bit près) :

- `radius_fraction` (None) : si fixé, le rayon devient `radius_fraction × KL(original ‖ cible)`
  au lieu de `kl_radius` (≈ fraction de la perte moyenne visée) ; `final.kl_radius_effective`
  le donne ;
- `altered_fraction` (0) : part de chaque lot remplacée par un retournement ou un recadrage
  70-100 % de la même image, avec la cible de l'image source. Le jitter et le JPEG ne sont jamais
  vus : sur le banc, `forget_jitter` et `forget_jpeg` mesurent le transfert à des familles
  inédites.

V4 (22/11, codée, **aucun run**) : ~88 % du mouvement de la V1 est commun à toutes les images
mémorisées (le fold d'audit bouge 0,80 à 0,96 fois l'image exacte), et la V2 et la V3 retenues font
pareil (0,86 à 0,91) : changer la cible a amélioré la direction, pas la localité. Trois options indépendantes, toutes neutres à leur défaut (V1/V2
identiques au bit près) ; détail et justification : `rapport/rapport.tex` §3 (V4) :

- `anchor_folds` (0) : K ≥ 2 plis stratifiés, oubliés l'un après l'autre ; les autres plis sont
  tenus à leurs sorties du début de la phase (copie figée de l'élève, KL à T = 2,
  `anchor_weight`), une part `anchor_views` en vues fortes (aire 35-45 %). Proxy de rétention tiré
  du forget set seul ; `anchor_kl` (historique) mesure la fuite de chaque phase ;
- `radius_dual_lr` (0) : > 0, le rayon devient une contrainte active (montée duale) au lieu d'un
  arrêt au premier franchissement ; la phase parcourt tout son calendrier ;
- `retain_source` (`proxy`) pour `scrub` et `gold_soft` : `forget_views` remplace les images proxy
  externes par des vues fortes du forget set (forme conforme au sujet) ; `synthetic` (V_Fusion) les
  remplace par les images gardées d'un fichier de `synth-gates` (`synthetic_path`), hors thermomètre,
  dans les lots `synthetic_pools` (`A`, `B` ou `AB`), augmentées comme le proxy. Mesuré : effondrement
  avec les images de la première génération (campagne V_Fusion).

Avec une option V4, le rayon `radius_fraction` est pris sur la sonde de 256 images où l'arrêt est
testé ; V1-V3 gardent leur calcul (tout le set), pour que les runs mesurés se rejouent exactement.
Specs (sur la V3 retenue, appariées à ses runs mesurés) : `scrub-forget-views`,
`trust-region-anchored`, `trust-region-dual` ; fiches au même nom.

V5 (22/11, `band_ascent`, codée, **abandonnée** : sélection réfutée à l'étape 0, locale mais sans score ; §10) : une nouvelle méthode, pas une option de
`trust_region` (V1 à V4 inchangées au bit près). Les `band_release` (0,25) images dont la perte de
l'original sur `target_views` vues fortes est la plus haute montent jusqu'à t = ρ × cette perte,
sous h = `band_tolerance` × t + `band_floor` ; les autres restent sous `band_floor` (0,05 ; `none`
retire cette borne). Perte : deux charnières softplus de raideur `band_sharpness`. La dose ρ monte
par `band_levels` niveaux (AdamW et calendrier neufs à chaque niveau, `epochs` passes chacun).
`cv_folds` = K ≥ 2 : un sous-run sur K − 1 plis garde le dernier niveau dont la dérive du pli
témoin, × (K/(K−1))^`cv_scale_exponent`, reste sous `cv_max_drift` avec un accord ≥
`cv_min_agreement` ; le run final monte ces niveaux sur toutes les images. Arrêt dur (retour au
dernier niveau achevé) au double du budget, ou si plus de la moitié d'une sonde dépasse sa borne.
`final.cv_monitor_kl` × `final.cv_scale` estime la dérive ; à comparer à `bench.audit_kl`.
Étape 0 sans entraînement : `$HK diagnose` (spec `band-ascent-step0`) ; puis spec `band-ascent`.

V5.1 (22/11, options de `band_ascent`, **abandonnée** après 11 runs ; rapport §3, V5.1 et §6). Objectif : terme
retain ≥ 0,8, budget dépensé vers le gold. Trois options, neutres par défaut (V5 au bit près) :
`band_space=margin` écrit les mêmes bandes sur la marge m = z_y − logsumexp(autres logits)
(CE = softplus(−m)), une force d'ordre 1 au lieu d'une force qui s'éteint avec la perte ;
`band_hold_kl=δ` tient les images non poussées par une charnière sur leur KL(original ‖ élève) à
T = 2 au-delà de δ (au lieu de la borne `band_floor` en perte) ; `cv_min_retain=r` garde un niveau
si accord / (1 + échelle × dérive) du pli témoin ≥ r (remplace `cv_max_drift` et
`cv_min_agreement`). `diagnose` lit en plus, sans entraînement, un vote kNN ajusté par plis sur le
jeton CLS de chaque bloc (sélection et direction : clés `selection_by.knn_*`, `encoder.best`) et
les trois défauts de la V5 (`defects.*`). Specs : `band-ascent-budget` (sans code), puis
`band-ascent-v51` (ablation, un composant par bras), puis `band-ascent-v51-lr` (lr 3e-5 et 1e-5).
Mesuré : la marge s'arrête net au premier niveau à 1e-4 et 3e-5 ; à 1e-5, retain 0,725.

V_Fusion (22/11, rapport §3 et §6) : étape 0 = `synth-generate` puis `synth-gates`, lancées sous
`scripts/gpu_guard.sh` (le lab ne les admet pas dans une spec) ; une tranche sur la RTX 5080 =
34 min pour 2000 images. Puis `scrub` avec `retain_source=synthetic` (spec `v-fusion-anchor`).

### `gold_soft` : pourquoi et comment

Le gold n'a jamais vu l'image : il se comporte comme sur une image inédite de la distribution
(confiance modérée, classes voisines en tête, même comportement sur les copies altérées). On n'a
que l'original, donc on le **sonde là où il n'est pas mémorisé** :

1. pour chaque image oubliée `x`, `target_views` vues fortes (crop de ~35 % de l'aire + color
   jitter, `strong_view`) passées dans le teacher ; ses softmax (divisés par
   `target_temperature`) sont moyennés. Moyenner = l'équivalent de la moyenne de plusieurs gold
   (le sujet moyenne leurs distributions) ; la vue forte fait baisser la confiance **surtout
   sur les images fragiles**, ce qui donne une hétérogénéité par image (donc un écart-type de loss
   non nul) ; la température retire la sur-confiance en gardant le **classement** des classes
   (quelle classe fausse est préférée) ;
2. le student est entraîné à sortir cette cible sur l'image exacte **et** sur une copie altérée
   fraîche (flip, crop doux, jitter ou JPEG), car l'autre moitié du forget score porte sur des
   copies altérées cachées qui suivent leur image source ;
3. la perte est une KL bornée vers une cible raisonnable, à T = 1 (loss par échantillon, donc
   mean/std score) et à T = 2 (température du `prediction_score`). Aucune montée de gradient ;
4. ancre de rétention : KL(teacher||student) à T = 2 sur des images de rétention proxy
   (`retain_weight`) ; AdamW lr 1e-5, warmup puis cosinus, clip du gradient à 1, bf16 sur CUDA
   (poids en float32).

Réglages mesurés ou à balayer :

- `target_temperature` (défaut 2,0) : le réglage décisif. Sur l'original, avec des vues fortes
  (crop ~35 %), T = 2 donne une confiance moyenne de 0,81 ; T = 4 de 0,41. À balayer
  {1,5 ; 2 ; 3} sur le leaderboard ; 2 est un point de départ, pas une valeur prouvée.
- `target_view_min_area` (0,35) : plus petit = cibles plus floues et plus hétérogènes.
- `lr`, `epochs`, `retain_weight` : montent la dérive sur retain. Surveiller `retain-proxy KL`.
- `trainable_blocks` (-1 = tout) : n'entraîner que les N derniers blocs réduit la dérive.
- `altered_weight` : poids des copies altérées dans la loss de forget.

Ordre d'expériences conseillé (chaque run complet sur A100 : estimation 3 à 6 min, **non
mesurée**, extrapolée de 21 s pour 24 pas de batch 8 sur M5) :

1. `identity` : doit scorer ~0 sur le leaderboard (valide chaîne, format, cooldown).
2. `gold_soft`, `CONFIG=full`, `target_temperature` dans {1,5 ; 2 ; 3}.
3. Garder le meilleur ; balayer `lr` {5e-6 ; 1e-5 ; 3e-5}, puis `target_view_min_area`.
4. Comparer `scrub` une fois, pour mémoire.

Le leaderboard public n'utilise que ~30 % des données de scoring (classement final : les 70 %
restants). Ne pas sur-régler sur ces 30 % : préférer un réglage au milieu d'un plateau.

## 3. Commandes

Local (M5, MPS) depuis la racine du dépôt. `HACKATHON_DATA_DIR` doit pointer vers le dossier de
données partagé, car dans un worktree `data/` n'est pas celui du dépôt principal :

```bash
export HACKATHON_DATA_DIR=<dossier de données partagé>/data/hackathon
HK="env PYTHONPATH=$PWD/src <checkout principal>/.venv/bin/python -m hackathon.unlearning"

$HK fetch                                    # model.pt (sha256 vérifié) + forget_images.npz
$HK info
$HK proxy-eval --forget-limit 64 --retain-count 32 --batch-size 16     # l'original (identité)
$HK run --method gold_soft --config smoke    # < 2 min sur M5 : 64 images, 4 pas de batch 8
$HK run --method scrub --config smoke
$HK run --method gold_soft --config smoke --set max_steps_per_epoch=24 --set lr=5e-5
$HK proxy-eval --candidate <fichier.pt> --forget-limit 64
$HK validate <fichier.pt>
$HK submit <fichier.pt>                      # dry run : valide, n'envoie rien
$HK submit <fichier.pt> --yes                # envoie (clé HACKATHON_API_KEY dans .env)
```

`run` écrit `<task-dir>/output/<méthode>_<horodatage>.pt` (validé) et un `.json` (config,
historique, tableau proxy). `--config` accepte `full`, `smoke` ou un fichier JSON ; `--set
clé=valeur` surcharge un champ (liste des champs : `configs.py`, docstring de `UnlearnConfig`).

JURECA (nœud de login, une fois) : les nœuds de calcul n'ont en général pas Internet, donc tout
ce qui se télécharge se prépare au login.

```bash
# 1. dossier d'équipe : le script du lead a déjà téléchargé SprintML/unlearning dans
#    $TEAM_DIR/unlearning (model.pt, forget_images.npz, .venv, output/). Sinon : fetch au login.
# 2. code : copier src/ (rsync depuis le portable)
rsync -av --exclude __pycache__ src/ <user>@jureca:$TEAM_DIR/challenge/src/
# 3. images de rétention proxy (hors distribution). coco_val_subset (79 Mo, 500 images) + kodak
#    (15 Mo, 24 images) suffisent ; div2k_valid (429 Mo, 100 images) est optionnel : un dossier
#    absent est simplement ignoré, et retain_count est plafonné au nombre d'images trouvées.
rsync -av data/coco_val_subset data/kodak \
    <user>@jureca:$TEAM_DIR/unlearning/proxy_images/
# 4. lancer (depuis le dossier de la tâche)
cd $TEAM_DIR/unlearning && mkdir -p output
TEAM_DIR=$TEAM_DIR METHOD=gold_soft CONFIG=full SETS="target_temperature=2.0" \
    sbatch $TEAM_DIR/challenge/jureca/unlearning.sbatch
squeue -u "$USER"; tail -f output/<jobid>.out
# 5. récupérer le fichier, valider et soumettre DEPUIS LE PORTABLE (la clé reste locale)
rsync <user>@jureca:$TEAM_DIR/unlearning/output/gold_soft_<jobid>.pt* ./
```

Le `.pt` fait ~344 Mo ; on peut aussi le soumettre depuis JURECA si l'accès sortant vers
du serveur d'évaluation est autorisé (non vérifié).

## 4. Limites des métriques proxy (à lire avant de se fier au tableau)

`proxy-eval` ne calcule PAS le score. Ce qui manque localement : les modèles gold (donc
`mean(L_gold)`, `std(L_gold)`, `p_gold`), les vrais labels, les copies altérées réelles, le control
set et le validation set. Donc :

1. **Pseudo-labels = argmax de l'original.** L'original étant mémorisé, c'est presque sûrement le
   vrai label, mais c'est justement la classe que le gold n'est pas tenu de reproduire. Les
   colonnes `L_*`, `acc_*`, `shift` mesurent "de combien on s'est éloigné de l'original", pas "à
   quel point on est proche du gold". Un `shift` grand n'est pas bon en soi.
2. **Aucune mesure de `prediction_score` ni de `mean_std_score`.** Impossible sans `p_gold`. Le seul
   oracle est le leaderboard (30 % des données), pas un tableau local.
3. **Rétention mesurée hors distribution.** Les 624 images locales (500 + 100 + 24) (COCO, DIV2K, Kodak redimensionnées
   en 224) ne viennent pas du dataset 100 classes. `retain_proxy KL` borne la dérive sur des
   images naturelles quelconques ; elle ne dit RIEN de l'accuracy du control set, ni de la KL
   réelle de `retain_kl_similarity` (le control set vient de la distribution d'entraînement, là où
   un fine-tuning sur le forget set dérive le plus). La confiance de l'original y est de 0,75, donc
   ces images sont déjà "difficiles" pour lui : la dérive y est probablement sous-estimée ou
   différente.
4. **Copies altérées simulées** (flip, crop 60-90 %, jitter, JPEG q50) : les vraies sont inconnues.
   Le sujet dit seulement "modified versions". Une méthode qui n'a vu que ces quatre altérations
   peut échouer sur une cinquième.
5. **Pas de gate d'utilité local** (70 % d'accuracy sur un validation set caché). Garde-fou
   indirect : `retain_proxy` accord d'argmax avec l'original proche de 1 et KL faible.
6. Évaluation sous **fp32** sur MPS/CPU et fp16 autocast sur CUDA, comme le serveur ; l'entraînement
   utilise bf16 : l'écart est faible mais non nul.

Lecture utile malgré tout : une bonne méthode montre un mouvement net sur forget ET sur ses copies
altérées, avec une dérive de rétention proche de zéro (`movement ratio` élevé). L'identité donne
0 partout.

## 5. Résultats smoke mesurés (M5, MPS, 64 images oubliées, 32 de rétention)

Tableaux complets dans la sortie des commandes du §3. Résumé (`KL(o||s)` à T = 2, moyenne) :

| candidat | pas | KL forget (moy. groupes) | KL retain proxy | accord argmax forget exact | confiance forget exact |
|---|---|---|---|---|---|
| original (identité) | 0 | 0,0000 | 0,0000 | 1,000 | 1,000 |
| `gold_soft` smoke | 4 (lr 1e-5) | 0,0012 | 0,0004 | 1,000 | 1,000 |
| `gold_soft` | 24 (lr 5e-5) | 0,1434 | 0,0278 | 0,953 | 0,958 |
| `scrub` | 12 (lr 5e-5) | 0,1073 | 0,0532 | 0,969 | 0,982 |

À mouvement comparable, `gold_soft` dérive moitié moins sur la rétention proxy que `scrub`. Ce
sont des essais à 12-24 pas sur 64 images : ils prouvent que la chaîne fonctionne et que les
méthodes bougent, pas qu'elles battent quoi que ce soit au score réel.

## 6. Pièges

- **Worktree et données** : `common.config.REPO_ROOT` est relatif au fichier ; dans un worktree,
  `data/hackathon` pointe vers le worktree. Exporter `HACKATHON_DATA_DIR` (ou `--task-dir`).
- **Format** : state dict float32 uniquement (un seul tenseur float16/bf16 = rejet), pas de
  NaN/Inf, mêmes clés et formes, <= 450 Mo (limite lue en Mo décimaux, 450 000 000 octets ;
  fichier attendu : 343 617 075). `save_submission` caste et valide avant d'écrire sous le nom
  final ; ne jamais enregistrer `model` (le module) mais `state_dict()`.
- **LoRA/adapters** : à refusionner avant envoi (non utilisé ici : on entraîne les poids directs).
- **Serveur** : forward en eval + fp16 autocast. Un modèle numériquement fragile en fp16 (logits
  énormes) peut se comporter autrement que sous fp32 ; la marge médiane de l'original est déjà 12.
- **Mémoire CUDA** : A100 (40 ou 80 Go) : batch 32 passe sans checkpointing ; réduire `batch_size`
  ou activer `grad_checkpointing=true` si OOM. M5 16 Go partagés : batch 8 maximum.
- **Cooldown** : aucun délai documenté pour cette tâche (30 min pour model equality). Chaque
  tentative, même rejetée, est journalisée dans `<data_dir>/submissions/log.jsonl`.
- **Pas d'ancre sur le validation set** : un gold_soft trop long ou trop fort peut passer sous
  70 % d'accuracy (score 0). Garder `lr` bas et surveiller l'accord d'argmax sur la rétention.
- **Soumission** : `submit` sans `--yes` ne fait que valider. Aucune clé n'est stockée dans le code.

## 7. Cibles Makefile proposées (non ajoutées : fichier hors périmètre)

```make
HK_UNL := env PYTHONPATH=$(CURDIR)/src $(PY) -m hackathon.unlearning
hk-unl-fetch:  ; $(HEAVY) $(HK_UNL) fetch
hk-unl-info:   ; $(HK_UNL) info
hk-unl-smoke:  ; $(HEAVY) $(HK_UNL) run --method gold_soft --config smoke
hk-unl-proxy:  ; $(HEAVY) $(HK_UNL) proxy-eval --forget-limit 64 --retain-count 32
hk-unl-test:   ; $(PY) -m pytest -q tests/hackathon -k unlearning
```

## 8. Banc local (suivi sans leaderboard)

Pour juger les idées sans soumettre : `src/hackathon/unlearning/bench.py`, protocole `bench-v4`
depuis la V3 (détail et formules dans la docstring du module). La campagne à pleine échelle (§10,
54 runs) est notée en `bench-v3`, qui n'a que la famille des recadrages : ses chiffres ne se
comparent pas à ceux de `bench-v4`. Le banc écrit `bench.protocol_version` au registre (absent =
`bench-v3`), et `make hk-unl-scores` fait une section par protocole, sans jamais les classer
ensemble. Tout run avec `holdout_fraction > 0` est noté
automatiquement. Sur le forget set complet, les chiffres vont dans `bench.*` (`candidate.json`,
registre, `make hk-board`, métrique principale `bench.score_min` de toutes les specs) ; un smoke
(sous-ensemble) les écrit dans `bench_smoke.*`, hors classement. Tout fichier se recalcule :

```bash
$HK bench --candidate <fichier.pt> [--forget-subset 64]   # sans --candidate : l'original (identité)
```

- **Fold d'audit** : 20 % du forget set, stratifié par l'argmax de l'original, graine fixe (31),
  jamais entraîné par les runs de réglage. Il joue le control set : retain = accord × 1/(1 + KL),
  porte à 70 % d'accord.
- **Doublures du gold, deux familles** : l'original sur 4 vues perturbées de chaque image. Un gold
  traite une image oubliée comme une image inédite : sa précision y est sa précision de test,
  inconnue mais > 70 %.
  - Recadrages (`s40`, `s50`, `s60`) : vues fortes (crop + jitter). Mesuré sur le vrai modèle
    (256 images) : accord avec l'argmax 33 % pour un crop de 15 % de l'aire, 73 % à 35 %, 83 % à
    45 %, 89 % à 55 % ; les trois forces (40, 50, 60 %) encadrent un gold à ~78, 86 et 92 %.
  - Masquages (`m30`, `m20`, `m15`) : 30, 20, 15 % des patchs 16×16 remplacés par du gris
    (`mask_patches`). Mesuré (256 images, 4 vues, M2) : accord 0,944 à 10 %, 0,917 à 15 %, 0,856
    à 20 %, 0,766 à 30 %. Même plage ~77-92 % que les recadrages.
  - Biais connu : les recadrages favorisent les méthodes qui visent ce type de cible
    (`gold_soft`, `trust_region` avec `views` ou `calibrated`) ; aucune méthode n'entraîne sur des
    masquages, c'est la famille non circulaire.
- **Formules exactes du sujet** (`Sujets/Unlearning.pdf`) : `mean_std_score = 1 − clip(d_sub /
  d_orig, 0, 1)` et `prediction_score = 1 − clip(KL(gold‖sub) / KL(gold‖orig), 0, 1)`, chacun moyenné
  sur images exactes et copies altérées. Normalisées par l'original : l'identité vaut exactement 0.
- **Chiffre suivi** : `score_min` = ce score sous la pire des six doublures (0 = identité, 1 = la
  doublure atteinte). Il ne monte que si le candidat progresse quelle que soit l'hypothèse, dans
  les deux familles ; `score_min_family.crop|mask` donne le pire cas de chacune.
- **Dérive hors score** : `audit_view_kl` = KL(orig ‖ sub) à T = 2 sur une vue forte (aire 0,50,
  graine fixe) de chaque image du fold d'audit, des entrées « comme inédites » (le fold exact est
  mémorisé). Prévu comme seuil (au-delà de 0,01, ancre de rétention) avant la campagne ; celle-ci
  mesure déjà 0,23 de dérive sur le fold d'audit exact pour la V1 (§10) : l'ancre est la piste
  suivante, et ce chiffre dira si la dérive sur des entrées non mémorisées est plus forte.
- **Ce qui reste une doublure** : le gold (vues fortes), les labels (argmax de l'original), les
  copies altérées (simulées), le control set (fold d'audit) et la porte (accord sur ce fold).
- **Bases de comparaison** : `bench-baselines` (gold_soft, scrub) donne les références ;
  `bench-seed-noise` mesure le bruit d'entraînement (3 graines). Une différence ne compte que si
  elle dépasse ce bruit. Comparer seulement à protocole égal (`bench.protocol` dans le JSON).
- Coût : les logits de l'original sont en cache (`<task-dir>/cache/bench_orig_logits.pt`, écriture
  atomique, partagé par les variantes ; 24 passes des doublures sur les images entraînées, une
  fois par protocole) ; ensuite ~15 500 passes du candidat (dont une vue du fold d'audit). Smoke
  M2 : 6 s pour l'identité en `bench-v4`, cache chaud.

## 9. Lancer sur sa propre machine (GPU CUDA, ex. RTX 5080)

Les specs `experiments/unlearning/*.toml` passent `--config auto` en local : config `full` si
CUDA est disponible, `smoke` sinon (Mac). Sur JURECA : `full`, une variante par A100, 4 au plus
(un nœud, en file derrière les jobs de l'équipe).

Prérequis : Linux (ou WSL2), pilote NVIDIA ≥ 580 (torch 2.15 du verrou = CUDA 13, qui couvre la 5080),
`git`, [`uv`](https://docs.astral.sh/uv/), ~10 Go de disque (torch + bibliothèques CUDA ~5 Go, données ~1 Go,
~344 Mo par run). Aucune clé : `.env` est facultatif pour un run local.

```bash
git clone https://github.com/alex-pierron/Tempete.git && cd Tempete && git checkout alex
bash scripts/setup_unlearning_gpu.sh      # une commande : uv sync, contrôle GPU, données, smoke CUDA
make hk-run SPEC=trust-region-kl-radius    # 4 variantes l'une après l'autre, config full
make hk-run SPEC=bench-baselines
make hk-run SPEC=bench-seed-noise
make hk-run SPEC=trust-region-views        # V2 (cible par vues), rayons 0,1-0,3, testé à chaque pas
make hk-run SPEC=trust-region-calibrated   # V3 (cible calibrée), lambda x copies douces
make hk-board TASK=unlearning              # classement sur bench.score_min
make hk-unl-scores                         # tableau comparatif -> experiments/unlearning/SCOREBOARD.md
```

Les trois premières ont tourné sur la 5080 (registre, §10) ; leurs conclusions ont été reprises
par la campagne. `trust-region-views` n'a pas tourné à pleine échelle : voir l'avertissement de sa
fiche (rayon à tester à chaque pas).

Le script est idempotent (le relancer ne retélécharge rien) et s'arrête avec un message clair si
`uv`, le pilote ou CUDA manquent. Il vérifie que torch voit la carte et que son architecture est
compilée, puis fait tourner `trust_region` et `gold_soft` en smoke sur la carte : c'est le premier
passage du code CUDA (bf16, AdamW fusionné, augmentations sur GPU), à surveiller.

- Sous Windows, torch PyPI est une build CPU : utiliser WSL2, ou installer torch depuis l'index
  CUDA de pytorch.org. `torch.cuda.is_available()` doit répondre `True`, sinon `auto` lance le smoke.
- **Dimensionnement mesuré** (RTX 5080 16 Go partagée avec l'affichage, Ryzen 9 9950X, 30 Go de
  RAM ; 21/11, config `full` avec `holdout_fraction=0.2`, soit 2800 images entraînées, batch 32,
  bf16) :

  | run | entraînement | total (chargement + banc) | VRAM du job | GPU max | puissance moy. / max |
  |---|---|---|---|---|---|
  | `gold_soft` (3 époques, 264 pas) | 64 s (0,24 s/pas) | 1 min 47 s | ~7,3 Go | 62 °C | 266 / 303 W |
  | `trust_region` (ε = 0,6, arrêt à 90 pas) | 9 s | 27 s | ~7,3 Go | 57 °C | 205 / 296 W |

  RAM : 4,4 Go au plus ; CPU : ~2 cœurs. Aucun bridage thermique ni matériel. La config `full`
  tient donc largement : on la garde **identique à JURECA** (batch 32, mêmes réglages), pour que les
  chiffres des deux machines restent comparables. Le premier run construit le cache des logits de
  l'original (`cache/bench_orig_logits.pt`), les suivants le relisent. `scrub` n'est pas mesuré.
- **Garde GPU** (`scripts/gpu_guard.sh`, mise devant `make hk-run` et les cibles lourdes dès que
  `nvidia-smi` est présent et `jarvis-heavy` absent) : un seul job sur la carte à la fois (les
  suivants attendent leur tour, comme la file JURECA) ; refus de démarrer sous 9 Go de VRAM libre
  (`GPU_GUARD_MIN_FREE_MIB`) ; job suspendu à 83 °C et repris à 75 °C (`GPU_GUARD_PAUSE_C`,
  `GPU_GUARD_RESUME_C` ; la carte se bride seule vers 88 °C) ; priorité CPU et disque basses ;
  télémétrie toutes les 5 s dans `runs/gpu/<date>.csv` et résumé en fin de job. Ctrl-C passe au job.
  Pour un run hors lab : `PYTHONPATH=$PWD/src ./scripts/gpu_guard.sh .venv/bin/python -m hackathon.unlearning run ...`.
- Si un jour OOM (autre programme sur la carte) : `--set grad_checkpointing=true` dans `args` de
  la spec, plutôt que changer `batch_size` (ce qui change l'expérience).
- Disque : ~344 Mo par run dans `runs/unlearning/` (ignoré par git) ; supprimer les runs inutiles.
- Nouvelle idée : `make hk-new TASK=unlearning IDEA="..."` donne une spec qui tourne en `full` sur
  la carte et que le banc note (`holdout_fraction=0.2`, `bench.score_min`), comme les trois prêtes.
- Tableau comparatif de tous les runs : `make hk-unl-scores` → `experiments/unlearning/SCOREBOARD.md`
  (réglages, score sous chaque doublure, retain, ratio, moyenne ± écart-type des graines).

## 10. Résultats à pleine échelle (RTX 5080, 21-22/11)

> **Reprise des expériences** (autre GPU, nouvelle version) : lire d'abord
> `experiments/unlearning/ETAT.md` — état mesuré, réglages retenus et specs à rejouer (`ref4-*`),
> règles du protocole, pièges mesurés, recette pour juger une nouvelle version.

Deux campagnes, 84 runs, sur le banc `bench-v3` (fiche : `experiments/unlearning/trust-region-campaign.md`,
chiffres : `SCOREBOARD.md`, analyse : rapport §6). Résumé, `score_min` sur 3 graines :

| méthode | réglage | score_min | retain | ratio |
|---|---|---|---|---|
| `trust_region` V2 | `trust_target=views`, τ = 1, aire 0,45, 3 vues, ε = 0,15, `radius_check_every=1` | 0,0831 ± 0,0042 | 0,78 | 1,4 |
| `trust_region` V1 | ε = 0,15, τ = 6, `radius_check_every=1`, `epochs=10` | 0,0637 ± 0,0025 | 0,77 | 1,4 |
| `scrub` | `lr=1e-4` | 0,098 ± 0,022 | 0,93 | 2,0 |
| `gold_soft` | `epochs=1`, `lr=1e-5` | 0,059 ± 0,012 | 0,88 | 1,6 |
| `scrub` / `gold_soft` | défauts | 0,0025 / 0 | 0,998 / 0,40 | 4,9 / 1,1 |

- `trust_region` V1 : tester le rayon à chaque pas est indispensable (`radius_check_every=1`) ; le
  score ne dépend alors que de ε (plateau 0,1-0,2) ; lr, warmup, τ et lot sont dans le bruit ;
  entraîner tout le réseau (`trainable_blocks=-1`).
- Les défauts de `gold_soft` (3 époques) et de `scrub` (lr 1e-5) sont à proscrire : le premier va
  trop loin, le second ne bouge pas. `scrub` dosé est le meilleur en moyenne mais bruité, et
  s'effondre à `lr=3e-4`.
- Le §2 ci-dessus (« `gold_soft` recommandée ») date d'avant ces mesures.
- Ce qui précède mesure la **V1** de `trust_region` (`trust_target=tempered`).

### V2 (`trust_target=views`), campagne de 30 runs

Fiche : `experiments/unlearning/trust-region-v2-campaign.md` (étapes posées avant les runs) ;
mesure des cibles avant entraînement : `scripts/unlearning_v2_targets.py` →
`experiments/unlearning/measurements/v2_targets.json` ; analyse : rapport, campagne V2. Banc
`bench-v3`, comme la V1 : les runs V2 précèdent `bench-v4`.

| réglage V2 (aire 0,45, 3 vues, ε = 0,15) | score_min (3 graines) | V1 mêmes graines | retain |
|---|---|---|---|
| **τ = 1** | **0,0831 ± 0,0042** | 0,0637 ± 0,0025 | 0,78 |
| τ = 0,5 | 0,0831 ± 0,0108 (rayon parfois jamais atteint) | | 0,77 |

- **La V2 améliore la V1 de 30 %**, graine par graine (+0,016, +0,027, +0,015), à quantité d'oubli
  et retain égaux : le gain vient de la direction (composante des prédictions).
- **τ décide** (de 0,066 à τ = 6 à 0,081 à τ = 1, graine 0) ; l'aire des vues (0,25 à 0,65) et le
  nombre de vues (≥ 3) ne changent presque rien. Le gain ne suit donc pas le recouvrement avec les
  doublures du gold du banc.
- **À éviter** : la spec d'origine `trust-region-views` telle quelle (test du rayon tous les 10
  pas) ; τ ≥ 4 (gain faible) ; τ ≤ 0,5 (le rayon n'est parfois jamais atteint : 880 pas, autre
  régime) ; une seule vue.
- **Contrôle en `bench-v4`** (après la fusion de la V3 ; fichiers recalculés sans réentraînement,
  `experiments/unlearning/measurements/bench_v4_rebench.json`) : V2 0,0786 ± 0,0030 contre V1
  0,0637 ± 0,0025 (+23 %, pire des six doublures). Sous la seule famille de masquage, étrangère à
  la cible de la V2, le gain tombe à +12 % (contre +30 % sous recadrage) : une partie du gain
  `bench-v3` est probablement liée à la famille de vues commune à la cible et aux doublures.
- Commande du réglage retenu (run de réglage, à ne pas soumettre) :

```bash
PYTHONPATH=$PWD/src ./scripts/gpu_guard.sh .venv/bin/python -m hackathon.unlearning run \
    --method trust_region --config full --set trust_target=views --set target_temperature=1 \
    --set target_view_min_area=0.45 --set target_views=3 --set kl_radius=0.15 \
    --set radius_check_every=1 --set epochs=10 --set holdout_fraction=0.2
```

### V3 (`trust_target=calibrated`), campagne de 23 runs en `bench-v4`

Fiche : `experiments/unlearning/trust-region-v3-campaign.md` ; cibles mesurées avant
entraînement : `scripts/unlearning_v3_targets.py` → `measurements/v3_targets.json` ; références
relancées en `bench-v4` : `ref4-v1`, `ref4-v2`, `ref4-scrub`.

| `bench-v4`, 3 graines | score_min | pire masquage | retain |
|---|---|---|---|
| **V3 retenue** (b = 0, λ 0,25, ρ 0,9, aire 0,5, 4 vues) | **0,0798 ± 0,0012** | 0,0798 | 0,74 |
| V2 retenue (τ = 1, aire 0,45, 3 vues, ε = 0,15) | 0,0786 ± 0,0030 | 0,0786 | 0,78 |
| V1 (ε = 0,15) | 0,0637 ± 0,0025 | 0,0702 | 0,77 |
| `scrub` lr 1e-4 (non conforme) | 0,098 ± 0,022 | 0,101 | 0,93 |

- **La V3 fait jeu égal avec la V2** (+0,0012 apparié), plus stable, un peu plus de dérive.
- La spec d'origine (couverture b 0,25) fait moins bien (0,0761) ; la couverture nuit (b = 0
  meilleur) ; λ ≈ 0,25 ; copies douces sans effet ; 1-2 vues : rayon jamais atteint.
- Le rayon étant tiré de la cible (ρ × distance), tout réglage de la cible agit par le rayon ;
  l'optimum retombe vers 0,15-0,18, comme V1 et V2. À rayon fixe 0,15, V3 = V2.
- **Conformité.** `scrub` et `gold_soft` tirent leur terme de rétention d'images proxy (COCO,
  DIV2K, Kodak), extérieures au forget set. Ils servent de références sur le banc, mais le sujet
  exige une méthode qui marche « from the original model and the forget set alone » : une
  méthode soumise ne doit pas en dépendre. L'ancre de rétention proposée pour `trust_region` est
  conforme si elle s'appuie sur des données dérivées du forget set (vues fortes des images
  oubliées, rapport §3), pas sur ces images proxy. Parmi les méthodes **conformes** mesurées
  jusqu'à la V3, la meilleure était la V2 de `trust_region` (0,0831 ± 0,0042 en `bench-v3`), à
  égalité avec la V3 en `bench-v4` (0,0786 ± 0,0030 et 0,0798 ± 0,0012) ; `scrub` dosé aux images
  proxy (0,098 ± 0,022) n'est qu'une référence de banc. Depuis la V4, `scrub` à ancre conforme
  (`retain_source=forget_views`) les dépasse (sous-section suivante).
- Le registre (`experiments/registry.jsonl`) reçoit les runs de la machine : committer et pousser
  pour les partager. Le test du registre livré vérifie que chaque ligne se relit (plus qu'il est
  vide), et le registre n'entre pas dans le drapeau `dirty` d'un run (il change à chaque run).

### V4 (ancre tirée du forget set, contrainte active, `scrub` conforme), 26 runs en `bench-v4`

Campagne : `experiments/unlearning/trust-region-v4-campaign.md` (contrôle de non-régression,
règle des suivis posée avant) ; fiches : `scrub-forget-views.md`, `trust-region-dual.md`,
`trust-region-anchored.md` ; analyse : rapport, campagne V4. Options neutres à leur défaut :
`ref4-v3` et `ref4-scrub` rejoués au bit près sur le code fusionné.

| `bench-v4` | réglage | score_min | pire masquage | retain | audit/exact | conforme |
|---|---|---|---|---|---|---|
| **`scrub` conforme** | `retain_source=forget_views`, lr 3e-4 (3 graines) | **0,1085 ± 0,0068** | 0,1085 | 0,86 | 0,85 | oui |
| `scrub` conforme | lr 4e-4 (3 graines ; retain sous le plancher de 0,80) | 0,121 ± 0,007 | 0,121 | 0,73 | 1,05 | oui |
| V3 retenue (référence) | lr 1e-5 (3 graines) | 0,0798 ± 0,0012 | 0,0798 | 0,74 | 0,89 | oui |
| V3 + ancre croisée | `anchor_folds=4`, w 3, lr 3e-5 (1 graine) | 0,070 | 0,070 | 0,82 | 0,69 | oui |
| V3 + contrainte active | `radius_dual_lr=0.25` (2 graines) | 0,042 | 0,042 | 0,73 | 0,98 | oui |

- **`scrub` conforme est la meilleure méthode conforme mesurée** : +36 % sur la V3, apparié sur trois
  graines, sous les deux familles ; au niveau du `scrub` proxy. Sa dose n'est pas celle du proxy :
  aux lr 1e-4 et 2e-4 de la fiche, l'ancre défait l'oubli (0,017 et 0,061) ; à 3e-4, le proxy
  s'effondre, la version conforme non. Elle gagne par l'efficacité de son oubli, pas par la localité.
- **Contrainte active : abandonnée.** μ retombe à 0 dès l'époque 3 (la contrainte ne lie pas) ;
  poursuivre la V3 sur tout son calendrier égalise les pertes et divise le score par deux.
- **Ancre croisée : abandonnée.** Les phases n'atteignent pas leur rayon (même à lr 3e-5, remède prévu
  par la spec) ; l'ancre freine l'oubli sur le fold d'audit presque autant que sur les images
  poussées (audit/exact 0,67-0,70, témoin 0,86), et ce qu'elle gagne sous le recadrage se perd sous
  le masquage.
- V3 à lr 3e-5 (témoin du suivi) : 0,0845 ± 0,0028, pas mieux que lr 1e-5.
- Commande du réglage retenu (run de réglage, à ne pas soumettre ; spec `ref4-scrub-fv`) :

```bash
PYTHONPATH=$PWD/src ./scripts/gpu_guard.sh .venv/bin/python -m hackathon.unlearning run \
    --method scrub --config full --set retain_source=forget_views --set lr=3e-4 \
    --set holdout_fraction=0.2
```

- Avant toute soumission (décision d'équipe) : run final sans `holdout_fraction` (3500 images au lieu
  de 2800 : plus de pas par époque, la dose peut bouger) ; arbitrage entre 3e-4 et 4e-4 (score
  contre retain).

### V5 (`band_ascent`) : sélection réfutée à l'étape 0, localité sans score

Campagne : `experiments/unlearning/trust-region-v5-campaign.md` ; fiches : `band-ascent-step0.md`,
`band-ascent.md` ; analyse : rapport, campagne V5. Run `20261122T024729Z-unl-band-ascent-step0`
(`$HK diagnose`, 2800 images entraînées, 51 s sur la RTX 5080).

| doublure | précision φ = 0,25 (taux de base) | part de la perte (oracle) | direction `runner_ups` / `views` |
|---|---|---|---|
| m30 | 0,183 (0,190) | 0,236 (0,919) | 0,752 / 0,637 |
| m20 | 0,109 (0,118) | 0,231 (0,976) | 0,765 / 0,609 |
| m15 | 0,087 (0,086) | 0,238 (0,991) | 0,768 / 0,594 |
| s40 / s50 / s60 (circulaires) | 2,1 à 2,6 fois le hasard | 0,41 à 0,47 (0,74 à 0,92) | `views` devant |

- **Réfutée sur les deux critères posés avant** (précision < 0,35 et part de la perte < 0,5, sous les
  trois masquages) : sous le masquage, la sélection de la V5 vaut un tirage au hasard. `band-ascent`
  n'a pas été lancé ; la borne haute (localité) et le choix hors pli restent non testés.
- Le pari de direction tient sous le masquage (les secondes de l'original devant les vues).
- Mesure annexe (`scripts/unlearning_v5_families.py`, `measurements/v5_families.json`) : les deux
  familles de doublures ne s'accordent pas sur les images difficiles (corrélation de rang des pertes
  par image de −0,04 à +0,05 ; 0,89 à 0,93 entre masquages, 0,28 à 0,37 entre recadrages). Un
  sélecteur par masquage (graines propres) fait 3,7 à 4,0 fois le hasard sous les masquages et le
  hasard sous les recadrages : toute sélection tirée de vues de l'original ne vaut que sous sa
  propre famille, où elle est circulaire.
- **Mesure de la localité seule** (lancée sur décision de l'utilisateur ; `band-ascent`, puis
  `band-locality` et `band-ascent-s1` ; 8 runs, ~7 min) :

| bras | graine | niveaux | KL exacte | audit/exact | score_min | retain |
|---|---|---|---|---|---|---|
| défaut (validation croisée) | 0 / 1 | 1 / 2 | 0,074 / 0,119 | **0,246 / 0,393** | 0,000 / 0,005 | 0,98 / 0,95 |
| φ = 1 (sans sélection) | 0 | 1 ; 4 | 0,107 ; 0,418 | 0,354 ; 0,897 | 0,042 ; 0,040 | 0,96 ; 0,70 |
| `cv_folds=0` (4 niveaux) | 0 / 1 | 4 | 0,213 / 0,271 | 0,538 / 0,739 | 0,017 / 0,046 | 0,88 / 0,81 |
| sans borne haute, 4 niveaux | 0 | 4 | 2,71 | 1,01 (effondrée) | 0 (porte) | 0,14 |

- Localité obtenue (la plus forte mesurée à KL exacte égale : SCRUB conforme à lr 2e-4 est à 0,49 vers
  0,08), mais **sans score** : S_pred 0,00 à 0,13 aux niveaux gardés. Les doublures sont moins sûres que
  l'original sur toutes les images ; la localité refuse ce mouvement commun. À KL exacte égale, φ = 1
  est aussi local que φ = 0,25 (la retenue n'y est pour rien) ; la borne haute évite l'effondrement.
  Sur des vues des images d'audit, la bande n'est pas plus locale que les autres. Estimateur de dérive
  hors pli 1,5 à 2,2 fois trop haut.
- Suite : V_Dumb, sous-section suivante.
- Tableau : le bras `band_floor=none` (option à `None`, absente du registre) se distingue désormais
  du bras par défaut grâce au JSON du run (`scoreboard.py`, test ajouté).

### V_Dumb (`dumb`) : aucune dose utilisable, 11 runs

Campagne : `experiments/unlearning/trust-region-vdumb-campaign.md` ; fiches : `v-dumb.md`,
`v-dumb-dose.md`, `v-dumb-dose2.md` ; analyse : rapport, campagne V_Dumb. Graine 0, t = 0,45 sauf mention.

| réglage | KL exacte | `final.reached` | score_min | retain | porte |
|---|---|---|---|---|---|
| fiche : lr 1e-5, w = 1, t ∈ {0,31 ; 0,45 ; 0,63} | 0,0005 | ≤ 0,001 | ≈ 0 | 1,000 | 1 |
| w = 0,3 / 0,1 / 0,03 | 0,006 / 0,017 / 0,030 | ≤ 0,02 | ≤ 0,0035 | 1,000 | 1 |
| lr 1e-4, w = 1 | 0,009 | 0,007 | 0,003 | 0,998 | 1 |
| **w = 0,01** (meilleur) | 0,067 | 0,033 | **0,019** | 0,987 | 1 |
| w = 0,003 / 0,001 / 0 | 3,5 / 5,5 / 6,5 | 0,8-0,98 | 0 | ≤ 0,05 | **0** |

- **Réfutée à sa fiche** (meilleur bras sous V1 − 2σ = 0,0577) et **sans dose utilisable** : entre
  l'immobilité (ancre) et l'effondrement (sans ancre), une falaise entre w = 0,01 et 0,003.
- La charnière ne borne rien : avec l'ancre, aucune image n'atteint t ; sans elle, la perte dépasse t
  de 13 à 27 fois. La cible t est sans effet (bras t = 0,31 et 0,45 identiques au bit près).
- SCRUB conforme, même ancre de vues, alterne montée et ancre au lieu de les sommer : c'est la
  différence de conception, non une cause mesurée.
