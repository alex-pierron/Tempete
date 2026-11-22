# Unlearning : état des expériences (fiche de reprise)

Pour qui reprend les expériences sur un autre GPU (JURECA, Mac, RTX) : ce qui est mesuré, avec quels
réglages, comment le rejouer, et comment juger une nouvelle version (V5…) sans refaire nos erreurs.
Mise à jour : 2026-11-22 (après les campagnes V5, V_Dumb, V5.1 et V_Fusion). Détails et analyse : `rapport/rapport.pdf` §6
(vue d'ensemble §6.1) ; tous les runs : `SCOREBOARD.md` (`make hk-unl-scores`).

## 1. Où on en est (banc `bench-v4`, 3 graines appariées, RTX 5080)

| méthode, réglage retenu | spec à rejouer | score_min | pire recadrage | pire masquage | retain | conforme |
|---|---|---|---|---|---|---|
| V1 `tempered`, ε = 0,15 | `ref4-v1` | 0,0637 ± 0,0025 | 0,0637 | 0,0702 | 0,768 | oui |
| V2 `views`, τ = 1, aire 0,45, 3 vues, ε = 0,15 | `ref4-v2` | 0,0786 ± 0,0030 | 0,0831 | 0,0786 | 0,778 | oui |
| V3 `calibrated`, b = 0, λ 0,25, ρ 0,9, aire 0,5, 4 vues | `ref4-v3` | **0,0798 ± 0,0012** | 0,0875 | 0,0798 | 0,744 | oui |
| **`scrub` conforme** (V4, `retain_source=forget_views`), lr 3e-4 | `ref4-scrub-fv` | **0,1085 ± 0,0068** | 0,1155 | 0,1085 | 0,859 | **oui** |
| `scrub`, lr 1e-4 (référence de banc, images proxy) | `ref4-scrub` | 0,098 ± 0,022 | 0,098 | 0,101 | 0,931 | **non** |

- **La meilleure méthode conforme mesurée est `scrub` à ancre conforme** (V4) : ses min-steps
  tiennent des vues fortes du forget set aux sorties de l'original, aucune image externe. À lr 3e-4,
  +0,029 sur la V3 sous le masquage, apparié sur trois graines (≈ 10σ), retain 0,86 ; au niveau du
  `scrub` proxy (non conforme). Elle ne gagne pas par la localité (audit/exact 0,85, comme la V3) mais
  par l'efficacité de son oubli. À lr 4e-4 : 0,121 ± 0,007, mais retain 0,73 (sous le plancher de 0,80
  posé avant). Fiche `scrub-forget-views.md` ; campagne `trust-region-v4-campaign.md`.
- V2 et V3 sont **à égalité** (écart apparié +0,0012) et battent la V1 sous les deux familles de
  doublures. Leur avance est plus nette sous le recadrage, famille que leurs cibles partagent avec
  les doublures (sous le seul masquage : +12 % pour la V2, +14 % pour la V3).
- **Conformité** (sujet : « from the original model and the forget set alone ») : à leur défaut
  (`retain_source=proxy`), `scrub` et `gold_soft` ancrent leur rétention sur des images proxy (COCO,
  DIV2K, Kodak) ; ce sont des références de banc, pas des candidats à la soumission. Avec
  `retain_source=forget_views`, `scrub` n'utilise que l'original et le forget set : conforme.
- Défaut commun aux trois versions : l'oubli déborde sur les images voisines (fold d'audit : KL
  ~0,23-0,26, ratio 1,3-1,4, contre 0,07 et 2,0 pour `scrub`).
- **V4 mesurée** (26 runs + 6 rejeux, campagne `trust-region-v4-campaign.md`) : options neutres à leur
  défaut (`ref4-v3` et `ref4-scrub` rejoués au bit près). `scrub` conforme : **confirmé** à lr 3e-4
  (réfuté aux doses de la fiche, 1e-4 et 2e-4, où l'ancre défait l'oubli). Contrainte active
  (`radius_dual_lr`) : **réfutée**, elle ne lie pas (μ = 0 dès l'époque 3) et le calendrier complet
  divise le score par deux (0,042). Ancre croisée (`anchor_folds`) : **réfutée** à lr 1e-5 et 3e-5
  (0,035 à 0,070 ; audit/exact 0,67 à 0,98, contre 0,86 pour le témoin) ; elle freine l'oubli partout.
  La localité reste non obtenue à une dose qui score. La V3 à lr 3e-5 : 0,0845 ± 0,0028, pas mieux.
- **V5** `band_ascent` : **abandonnée** (campagne `trust-region-v5-campaign.md`, 1 diagnostic et
  8 runs). Étape 0 (sans entraînement) : sa sélection (les 25 % d'images où l'original doute sur des
  vues fortes) vaut un tirage au hasard sous les trois masquages (précision égale au taux de base,
  0,23-0,24 de leur perte pour un seuil de 0,5). Lancée ensuite comme mesure de la localité seule
  (décision de l'utilisateur) : **localité obtenue** (audit/exact 0,246 et 0,393, graines 0 et 1, au
  niveau gardé par la validation croisée ; la plus locale des méthodes mesurées à KL exacte égale),
  **mais sans score** (`score_min` 0,000 à 0,046 sur huit runs ; S_pred 0,00 à 0,13). À KL exacte égale,
  φ = 1 est aussi local que φ = 0,25 : la localité vient de la montée bornée, pas de la retenue (modèle
  linéaire réfuté sur ce point) ; sans borne haute, effondrement. Sur des vues des images d'audit, la
  bande n'est pas plus locale que les autres. Mesure annexe : les deux familles de doublures ne
  s'accordent pas sur les images difficiles (corrélation de rang −0,04 à +0,05 ; 0,89 à 0,93 entre
  masquages, 0,28 à 0,37 entre recadrages).
- **V_Dumb** `dumb` : **abandonnée** (campagne `trust-region-vdumb-campaign.md`, 11 runs). À sa fiche
  (lr 1e-5, ancre w = 1, t ∈ {0,31 ; 0,45 ; 0,63}), le modèle ne bouge pas (KL exacte 0,0005, score ≈ 0) ;
  sans ancre, il s'effondre (perte 12, 27 fois la cible). Deux suivis de dose posés avant : meilleur run
  0,019 (w = 0,01) ; de w = 0,01 à 0,003, on passe d'une KL exacte de 0,067 à l'effondrement. Aucune
  dose utilisable ; la charnière ne lie jamais (avec ancre) et ne borne rien (sans) ; t est sans effet.
- **V5.1** (options de `band_ascent`) : **abandonnée** (campagne `trust-region-v51-campaign.md`,
  1 diagnostic et 11 runs, lancée à la demande de l'utilisateur malgré l'abandon de sa base).
  - Étape 0 rejouée : les trois défauts de la V5 sont confirmés (famine : 10 % des images portent 87 %
    de la force ; borne δ0 à KL 0,37 pour un budget de 0,064 ; poussée au départ à 1,94 fois la
    montée). La sélection par l'encodeur (vote kNN à l'entrée de la tête) fait 1,4 fois le hasard
    sous les masquages, mais ne porte que 0,35-0,40 de la perte : elle n'est pas codée.
  - `band-ascent-budget` (V5, budget 0,1 / 0,2 / 0,3) : le budget est consommé (1, 3, 4 niveaux), le
    retain tient (0,98 à 0,88), mais le score reste nul (0,017 au mieux). Le bras 0,1 redonne la V5
    au bit près.
  - V5.1 en marge : arrêt dur au niveau 1, en 15 à 40 pas, à lr 1e-4 et 3e-5 (la marge fait dépasser
    la bande). À 1e-5 seulement, un niveau est gardé : 0,0796, mais retain 0,725 et audit/exact 0,84,
    donc réfutée. Le témoin y est optimiste de 0,136. La retenue KL ne tient pas à pleine échelle
    (KL des images tenues 0,30, pour une tolérance de 0,128).
  - Meilleur bras sous le plancher de retain : la bande en perte avec retenue KL à 3e-5, 0,0385 à
    retain 0,975, sous la V1.
- **V_Dumb2** (codée et testée, **aucun run** ; rapport §3, V_Dumb2) : SCRUB conforme dont le
  max-step devient le niveau par image de V_Dumb, en bande sur la marge, borné des deux côtés
  (`scrub_target_loss=0.45`, neutre par défaut : V4 rejouée au bit près). Porte sans entraînement :
  `band-ascent-step0`, clé `level_ceiling.t045.min_mask` ≥ 0,155, sinon ne pas lancer. Puis
  `v-dumb2` (lr 3e-4 / 4e-4 × avec / sans rappel) ; seuils dans `v-dumb2.md`. Risque mesuré depuis
  (campagne V5.1) : la bande en marge dépasse son plafond en 15 à 40 pas dès lr 3e-5 ; ici, ni arrêt
  dur ni niveaux, seulement le rappel et les passes d'ancre de SCRUB.
- Rien n'a été soumis. Tous ces fichiers ont un fold d'audit (20 % du forget set non oublié) : ne
  pas les soumettre ; le run final se fait sans `holdout_fraction`.
- **V_Fusion** (SCRUB conforme ancré sur des images synthétiques) : **abandonnée sous cette
  génération** (campagne `trust-region-vfusion-campaign.md`, étape 0 hors registre et 4 runs).
  - 2000 images Plug-In Inversion en 34 min sur la RTX 5080. Le plafond de confiance n'agit pas : les
    images sont saturées (0,998 à 1, au lieu de 0,9).
  - Portes a, b, c, c' passées pour A et B. La porte d est ininterprétable : le témoin reste à 0,92,
    même à 1000 pas. Les images réparent trois fois moins que les vues : le modèle bruité est déjà
    d'accord avec l'original sur elles.
  - Option `retain_source=synthetic` de `scrub` (commit ba05fcb). Les 4 bras (AB, A, B à lr 4e-4 ;
    AB à 3e-4) **s'effondrent** : retain 0,01 à 0,025, KL exacte autour de 4. La perte de l'ancre
    tombe à 0,03 : le modèle cassé par la montée reste d'accord avec l'original sur ces images.
  - Suite possible, non lancée : un plafond actif dès le premier étage, et une porte de sensibilité
    (perte du modèle bruité sur l'ancre, comparée aux vues) avant tout bras.

## 2. Rejouer

```bash
make hk-run SPEC=ref4-v3                    # local (RTX : ~45 s par graine), derrière la garde GPU
make hk-run SPEC=ref4-scrub-fv              # meilleur conforme (RTX : ~45 s par graine)
make hk-run SPEC=ref4-v3 WHERE=jureca       # JURECA : 3 variantes sur un nœud (règle d'équipe : un seul nœud)
make hk-unl-scores                          # tableau comparatif, une section par protocole
```

L'entraînement est déterministe à graine égale : un run rejoué sur la même machine redonne les
mêmes chiffres (vérifié : `ref4-*` = recalculs ; V2 réentraînée après fusion = même score ;
`ref4-v3` et `ref4-scrub` rejoués après la fusion de la V4 = mêmes 159 métriques au bit près). Sur
un autre GPU (A100), de petites différences numériques sont possibles : comparer à protocole et
machine égaux, ou rejouer les références sur la machine qui fait les nouveaux runs.

## 3. Règles du protocole (à garder)

1. **Banc `bench-v4`** pour toute nouvelle mesure (6 doublures : recadrage s40/s50/s60, masquage
   m30/m20/m15) ; ne jamais comparer un chiffre `bench-v3` à un chiffre `bench-v4`.
2. **`holdout_fraction=0.2`** dans tout run de réglage (fold d'audit = control set du banc).
3. **`radius_check_every=1`** pour `trust_region` : tous les 10 pas, des rayons différents donnent
   le même modèle (la KL saute de 0,07 à 0,65 en 10 pas à la fin du warmup).
4. **Comparaisons appariées** à graine égale (mêmes lots, même fold) ; une différence compte si
   elle dépasse **2σ, σ ≈ 0,003** (écart-type entre graines des références).
5. **Hypothèses et seuils écrits avant de lancer**, dans une fiche (`*-campaign.md`) : ce qui
   confirmerait, ce qui réfuterait.
6. **Lire les deux familles** : `bench.score_min_family.crop` et `.mask`. Un gain qui ne tient que
   sous le recadrage est suspect (les cibles par vues partagent cette famille avec les doublures).
7. **Une sélection d'images se juge sous la famille qu'elle ne partage pas** (mesuré à l'étape 0 de
   la V5) : les familles ne s'accordent pas sur les images difficiles, si bien qu'un sélecteur tiré de
   vues de l'original fait 2 à 4 fois le hasard sous sa propre famille (circulaire) et le hasard sous
   l'autre.

## 4. Pièges mesurés

| piège | symptôme | parade |
|---|---|---|
| rayon testé tous les 10 pas | ε 0,2 / 0,4 / 0,6 donnent le même modèle | `radius_check_every=1` |
| cible trop piquée (V2 τ ≤ 0,5 ; V3 à 1-2 vues) | rayon jamais atteint, 880 pas, `final.radius_reached` = 0, autre régime (S_ms ~0,1, S_pred ~0,6) | τ ≥ 1, ≥ 3 vues ; vérifier `final.radius_reached` |
| rayon au-delà de la cible | run refusé (`ConfigError` : cible à KL < rayon) | lire `final.target_kl` d'un run voisin avant de fixer ε |
| rayon tiré de la cible (`radius_fraction`) | tout réglage de la cible change aussi le rayon : effets mêlés | comparer des cibles à **rayon fixe** (`kl_radius`, sans `radius_fraction`) |
| rayon trop grand (> ~0,2) | chute par s60 et la composante des prédictions ; KL d'audit > 0,34 | optimum mesuré 0,15-0,18 pour V1, V2, V3 |
| « plafond » d'une cible mesuré avant entraînement | prometteur, puis rayon jamais atteint | un plafond ne vaut que si le rayon est atteint |
| cache du banc | un seul protocole par fichier (`cache/bench_orig_logits.pt`) : un essai réduit l'écrase | recalcul automatique (~15 s), sans effet sur les chiffres |
| objectif des V1-V3 poursuivi au-delà de l'arrêt (`radius_dual_lr`, phases de `anchor_folds` qui n'atteignent pas leur rayon) | KL T = 1 retombe sous le rayon, KL T = 2 monte, composante des pertes ÷ 2 à 3, score ÷ 2 | vérifier `final.radius_ratio` et `final.reached_p*` ; ce régime n'est pas celui des V1-V3 |
| dose de `scrub` reprise d'une autre ancre | l'optimum dépend de l'ancre : proxy à 1e-4 (effondré à 3e-4), conforme à 3e-4 | redoser quand l'ancre change ; plancher de retain posé avant |
| sélection d'images tirée de vues de l'original (V5) | 2 à 4 fois le hasard sous sa famille, le hasard sous l'autre | règle 7 ; `diagnose` (étape 0) ou `scripts/unlearning_v5_families.py` avant tout entraînement |
| localité comparée à des doses différentes | audit/exact croît avec la dose pour toute méthode (SCRUB conforme 0,23 → 1,06 de lr 1e-4 à 4e-4 ; bande 0,25 → 0,90) | comparer à KL exacte égale (`bench.groups.forget_exact.kl_orig_sub`) ; lire aussi `bench.audit_view_kl` |
| localité sans score (V5) | audit/exact 0,25, retain 0,98, mais S_pred ≈ 0 : les doublures bougent sur toutes les images | la localité ne paie que par le retain ; vérifier S_pred avant de viser la localité |
| montée et ancre sommées dans le même pas (V_Dumb) | tout ou rien : l'ancre fige le modèle, ou la montée s'emballe (falaise entre w = 0,01 et 0,003) | une charnière sur la perte ne borne pas l'oubli (le mouvement commun emporte les images arrivées) ; SCRUB conforme alterne ses pas |
| option mise à `None` (`band_floor=none`) | absente du registre (nombres seuls) : le bras ressemblait au défaut | le tableau lit le JSON du run (corrigé, testé) ; régénérer là où sont les dossiers de runs |
| fichier suivi modifié pendant une campagne (test, code) | runs marqués `dirty` | n'éditer que la doc (`*.md`, `docs`, `rapport`) et de nouvelles specs non suivies pendant les runs ; le registre ne se retouche pas (trace d'audit) |
| force d'ordre 1 sans frein (V5.1, `band_space=margin`) | arrêt dur au premier niveau en 15 à 40 pas (44-74 % de la sonde au-dessus de sa borne) ; à lr plus bas, le run final dérive bien plus que le témoin | lire `final.cv_stop_level` (0 = modèle intact) ; l'estimateur `cv_retain` n'est valable qu'à ±0,03 du banc, vérifier |
| images synthétiques saturées (V_Fusion) | plafond de confiance sans effet (0,998-1) ; le modèle cassé reste d'accord avec l'original sur elles : l'ancre ne voit pas les dégâts, et SCRUB s'effondre (retain 0,02) | avant tout bras, mesurer la perte du modèle bruité de la porte d sur l'ancre, comparée aux vues du forget set (ici 0,04-0,06 contre 2,17) |
| porte d dont le témoin échoue | le budget de réparation est court : la porte ne dit rien, même à 1000 pas (témoin 0,92) | décider sur les autres portes en le notant, et lire l'écart au témoin |
| sous-commande hors de la liste du lab (`synth-generate`, `synth-gates`) | spec refusée au lancement (rien ne tourne) | lancer sous `scripts/gpu_guard.sh` et noter la provenance (commit, arbre) dans le dossier de sortie, ou ajouter la sous-commande à `src/hackathon/lab/tasks.py` |

## 5. Où sont les preuves

- Registre : `experiments/registry.jsonl` (191 runs terminés, dont les deux étapes 0 de la V5, 8 runs
  de la V5, 11 de V_Dumb, 11 de la V5.1 et 4 de V_Fusion, et 1 refusé ; commit et drapeau `dirty`
  par run. L'étape 0 de V_Fusion, générée hors spec, n'y est pas : sa provenance est dans son dossier
  de run. 7 runs sont `dirty`, tous de la campagne V4 : ils ont été lancés avec un fichier de test
  modifié, puis rejoués propres
  au bit près, voir `trust-region-v4-campaign.md`) ; tableau : `SCOREBOARD.md` (une graine compte une
  fois : rejeux et doublons `dirty` ne doublent pas l'effectif).
- Fiches de campagne (hypothèses posées avant, résultats, décisions) : `trust-region-campaign.md`
  (V1 et références, `bench-v3`), `trust-region-v2-campaign.md` (V2, `bench-v3`, contrôle
  `bench-v4`), `trust-region-v3-campaign.md` (V3, `bench-v4`), `trust-region-v4-campaign.md` (V4,
  `bench-v4`, avec les fiches `scrub-forget-views.md`, `trust-region-dual.md`,
  `trust-region-anchored.md`), `trust-region-v5-campaign.md` (V5 : étape 0, puis mesure de localité ; fiches
  `band-ascent-step0.md`, `band-ascent.md`, `band-locality.md`, `band-ascent-s1.md`),
  `trust-region-vdumb-campaign.md` (V_Dumb ; fiches `v-dumb.md`, `v-dumb-dose.md`, `v-dumb-dose2.md`),
  `trust-region-v51-campaign.md` (V5.1 ; fiches `band-ascent-step0.md`, `band-ascent-budget.md`,
  `band-ascent-v51.md`, `band-ascent-v51-lr.md`), `trust-region-vfusion-campaign.md` (V_Fusion ;
  fiches `v-fusion-step0.md`, `v-fusion-anchor.md`).
- Mesures hors entraînement : `measurements/v2_targets.json`, `measurements/v3_targets.json`
  (cibles et plafonds ; scripts `scripts/unlearning_v2_targets.py`, `scripts/unlearning_v3_targets.py`),
  `measurements/bench_v4_rebench.json` (V1, V2, `scrub` recalculés en `bench-v4`),
  `measurements/v5_families.json` (difficulté par image selon la famille de doublures ; script
  `scripts/unlearning_v5_families.py`) ; l'étape 0 de la V5 est au registre (`step0.json` dans son
  dossier de run).
- Dossiers de runs (fichiers `.pt`, journaux) : sur la RTX 5080 seulement (`runs/`, non versionné),
  dont les images synthétiques de V_Fusion (`runs/unlearning/20261122-vfusion-step0/`).

## 6. Juger une nouvelle version (V5…)

0. **Si la méthode choisit des images** (sélection, poids par image) : mesurer d'abord, sans
   entraînement, si son choix désigne les erreurs des doublures de la famille qu'il ne partage pas
   (règle 7 ; la V5 y a échoué).
1. **Mesurer la cible avant d'entraîner** (adapter `scripts/unlearning_v3_targets.py`) : distance
   atteignable (le rayon doit rester en dessous), accord, plafond sous les deux familles.
2. **Spec à rayon fixe** d'abord (ε ∈ {0,1 ; 0,15 ; 0,2}, `radius_check_every=1`, graine 0), pour
   séparer la direction de la cible et la quantité d'oubli ; ensuite seulement ses réglages propres.
3. **Comparer apparié** à `ref4-v3` et, désormais, à `ref4-scrub-fv` (mêmes graines). Seuil
   d'amélioration sur la meilleure méthode conforme (`scrub` conforme, σ ≈ 0,007 entre graines) :
   moyenne sur 3 graines > 0,1085 + 2 × 0,007 ≈ **0,122**, et gain sous le masquage au-delà de 2σ ;
   sur la V3 seule (σ ≈ 0,003) : > 0,086.
4. Vérifier `final.radius_reached` = 1 (et `final.reached_p*`, `final.radius_ratio` pour les options
   V4), `bench.gate_passed` = 1, `bench.audit_kl` < 0,34, retain ≥ 0,80 si la fiche le pose.
   Mécanisme de localité : audit/exact = `bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`
   (≠ `bench.movement_ratio`, qui moyenne aussi les copies), **à KL exacte égale** avec les références
   (il croît avec la dose), et avec S_pred : la bande de la V5 est locale et ne score pas.
5. Écrire résultats et décision dans la fiche, puis `make hk-unl-scores`, rapport (§6) et cette
   fiche.
