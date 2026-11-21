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

Trois méthodes derrière `unlearn(model, forget_images, config) -> model` (`methods.py`) :

| méthode | idée | risque |
|---|---|---|
| `identity` | copie intacte | score ~0 ; sert à tester la chaîne complète |
| `scrub` | SCRUB sans labels : max-step sur le forget (KL(teacher||student), **plafonné** par `scrub_kl_cap`, donc pas la montée de gradient naïve) alternée avec min-step (KL vers l'enseignant) sur des images de rétention proxy | dérive plus forte sur retain (mesuré, §5) ; vise "être différent", pas "être le gold" |
| `gold_soft` | **recommandée** : cible = comportement du gold | la température `target_temperature` doit être calibrée sur le leaderboard |

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
export HACKATHON_DATA_DIR=/Users/nowais/Documents/code/challenge/data/hackathon
HK="env PYTHONPATH=$PWD/src /Users/nowais/Documents/code/challenge/.venv/bin/python -m hackathon.unlearning"

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
`34.147.229.55` est autorisé (non vérifié).

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
