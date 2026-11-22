# Campagne V_Dumb (`dumb`, la méthode la plus simple), RTX 5080

- Spécifications : `v-dumb` (4 bras, graine 0), puis les suivis justifiés plus bas (au plus 4
  variantes chacun)
- Tâche : `unlearning` ; banc : **`bench-v4`** (6 doublures : recadrages s40/s50/s60, masquages
  m30/m20/m15) ; fold d'audit `holdout_fraction=0.2`
- Créée le : 2026-11-22, avant tout run de V_Dumb ; responsable : Alex
- Méthode : `dumb` (rapport §3, V_Dumb ; code du commit 964e119, fusionné en 3a1f81e). Le nom de la
  fiche garde la série des campagnes (`trust-region-*-campaign.md`).
- Contrat : la fiche `v-dumb.md`, hypothèses et seuils posés avant tout run. Cette fiche ne les
  réécrit pas : elle ajoute les comparaisons appariées à SCRUB conforme et à la V5, les grandeurs lues
  et la règle des suivis, et consigne les résultats.
- Tableau : `make hk-unl-scores` → `SCOREBOARD.md` (section `bench-v4`)

## Grandeurs lues sur chaque run

- `bench.score_min`, `bench.score_min_family.crop` et `.mask` ; S_ms et S_pred sous chaque doublure
  (`bench.mean_std.*`, `bench.prediction.*`) : la fiche prévoit que l'écart-type des pertes borne S_ms.
- `final.reached` (part des images à la cible), `final.forget_loss_mean`, `final.forget_loss_std`,
  `final.forget_kl` ; `bench.groups.forget_exact.loss_mean_sub` et `.loss_std_sub`.
- Dérive : `bench.retain`, `bench.audit_kl`, audit/exact (`bench.audit_kl` /
  `bench.groups.forget_exact.kl_orig_sub`) **lu à KL exacte égale** avec les références (leçon de la
  V5 : il croît avec la dose), `bench.audit_view_kl`.
- Contrôles : `bench.gate_passed` = 1, `bench.audit_kl` < 0,34.

## Références appariées (graine 0, au registre, `bench-v4`, RTX 5080)

| référence | score_min | recadr. | masq. | retain | KL exacte | audit/exact | perte exacte (écart-type) |
|---|---|---|---|---|---|---|---|
| V1 (`ref4-v1` v1) | 0,0647 | 0,0647 | 0,0708 | 0,771 | 0,257 | 0,88 | 0,172 (0,624) |
| V3 (`ref4-v3` v1) | 0,0784 | 0,0865 | 0,0784 | 0,758 | 0,284 | 0,86 | 0,191 (0,635) |
| SCRUB conforme lr 3e-4 (`scrub-fv-dose` v1) | 0,1151 | 0,1229 | 0,1151 | 0,868 | 0,165 | 0,80 | 0,156 (1,164) |

Moyennes sur 3 graines : V1 0,0637 ± 0,0025 ; V3 0,0798 ± 0,0012 ; SCRUB conforme 0,1085 ± 0,0068.

## Seuils

- **Fiche `v-dumb` (tels quels)**, σ = 0,003 : réfutée si le meilleur bras est sous **0,0577**
  (V1 − 2σ), ou si son `bench.retain` est sous **0,8**, ou si le bras sans ancre a un retain « à 2σ
  près » de celui du bras t = 0,45. Le σ de la fiche est celui des scores ; appliqué au retain, il
  donne |Δ retain| ≤ 0,006 (lecture littérale, retenue ici ; l'écart-type du retain entre graines des
  références est plus grand, 0,012 à 0,023, et sera donné à côté). Lecture : au-dessus de **0,086**
  (V3 + 2σ), les V2-V3 n'apportent rien sur le banc.
- **Ajoutés ici, avant les runs** : sur SCRUB conforme apparié (même ancre de vues), moyenne sur trois
  graines > **0,122** (0,1085 + 2 × 0,007) et gain sous le masquage > 0,014 pour devenir la meilleure
  méthode conforme mesurée ; un réglage n'est retenu qu'avec un retain ≥ 0,80 (plancher de la V4).

## Règle des suivis (posée avant les runs)

Un suivi n'est lancé que pour une de ces raisons ; sa question, ses bras et ses seuils sont écrits ici
avant le lancement :

1. **Candidat** : meilleur bras > 0,0697 (V1 + 2σ) avec retain ≥ 0,8 → graines 1 et 2 du même réglage
   (trois graines appariées).
2. **Dose** : si les bras avec ancre n'atteignent pas leur cible (`final.reached` < 0,5) ou bougent
   peu (KL exacte < 0,1, sous toutes les références), le taux est trop bas → lr plus haut, choisi
   d'après la KL exacte mesurée (SCRUB conforme, même ancre, ne valait qu'à 3e-4). À l'inverse, si
   le meilleur score a un retain < 0,8 → lr plus bas ou ancre plus lourde.
3. **Cible** : si le score croît de t = 0,31 à 0,63 sans perdre le retain, t plus haut ; s'il décroît,
   t plus bas. La direction doit être mesurée, pas supposée.
4. Arrêt quand chaque composante (charnière, cible, ancre) est confirmée ou réfutée et que le meilleur
   réglage a trois graines appariées s'il score.

## Résultats

### `v-dumb`, telle quelle (groupe `20261122T032406Z-unl-v-dumb`, commit 525b0bf, propre, 3 min 21)

| bras (lr 1e-5, 3 passes, graine 0) | `score_min` | retain | KL exacte | perte exacte (écart-type) | `final.reached` | porte |
|---|---|---|---|---|---|---|
| t = 0,31 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,019) | 0,001 | 1 |
| t = 0,45 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,019) | 0,001 | 1 |
| t = 0,63 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,020) | 0 | 1 |
| t = 0,45 sans ancre (`retain_weight=0`) | 0 | 0,006 | 6,53 | 12,23 (5,38) | 0,977 | **0** |

- **Réfutée telle quelle** (meilleur bras 0,000003 < 0,0577). Avec l'ancre, le modèle ne bouge pas : la
  charnière reste à t pendant les trois passes (perte de l'élève ≈ 0,001, KL de l'ancre 0,002-0,005).
  Les bras t = 0,31 et 0,45 sont identiques au bit près : sous la cible, le gradient de la charnière
  ne dépend pas de t.
- **Sans ancre, effondrement** : la montée démarre à la fin de la première passe et s'emballe à la
  deuxième (KL des vues 9,9 puis 19,4) ; perte finale 12,2, 27 fois la cible. **La charnière ne borne
  rien** : elle annule le gradient des images arrivées, mais les mises à jour des autres images les
  emportent (le mouvement commun déjà mesuré à la V5).
- Critère de l'ancre (fiche) : le bras sans ancre n'est pas à 2σ du bras t = 0,45 (0,006 contre 1,000) :
  l'ancre sert, au sens de la fiche. Critère du retain : sans objet, aucun bras ne score.
- Règle 2 déclenchée (`final.reached` < 0,5, KL exacte < 0,1).

## Suivi D : la dose (posé le 2026-11-22, avant le run)

**Ce qui l'ouvre** : la règle 2. La mesure donne la direction : le poids de l'ancre fait passer de
l'immobilité (w = 1) à l'effondrement (w = 0) ; le taux, lui, est celui que la règle prévoyait (SCRUB
conforme, même ancre de vues, ne valait qu'à lr 3e-4).

**Bras** (`v-dumb-dose`, t = 0,45, graine 0) : `retain_weight` 0,03 ; 0,1 ; 0,3 (lr 1e-5) ; lr 1e-4
(w = 1). Pas de grille sur t : tant que la charnière ne lie pas, t ne change rien (mesuré).

**Critères (posés avant)** :
- ceux de la fiche : réfuté si le meilleur bras < 0,0577 ou son retain < 0,8 ; candidat (règle 1) si
  > 0,0697 avec retain ≥ 0,8 → graines 1 et 2 ;
- **la charnière borne** si la perte moyenne des images entraînées finit entre 0,5 t et 2 t (0,22 à 0,90)
  avec `final.reached` ≥ 0,5 ; au-delà de 2 t, elle ne borne pas ;
- **pas de fenêtre** si chaque bras est soit immobile (KL exacte < 0,02), soit effondré (porte non
  franchie) : V_Dumb n'a alors pas de dose utilisable à cette granularité, et on s'arrête (pas de
  grille plus fine sans autre indice).

### Résultat du suivi D (groupe `20261122T032841Z-unl-v-dumb-dose`, commit 525b0bf, propre, 3 min 20)

| bras (t = 0,45, graine 0) | KL exacte | perte exacte (écart-type) | `final.reached` | audit/exact | `score_min` | retain |
|---|---|---|---|---|---|---|
| w = 0,3, lr 1e-5 | 0,006 | 0,014 (0,17) | 0,009 | 0,02 | 0,0001 | 1,000 |
| w = 0,1, lr 1e-5 | 0,017 | 0,037 (0,45) | 0,013 | 0,02 | 0,0008 | 1,000 |
| w = 0,03, lr 1e-5 | 0,030 | 0,063 (0,55) | 0,021 | 0,04 | 0,0035 | 0,999 |
| w = 1, lr 1e-4 | 0,009 | 0,016 (0,23) | 0,007 | 0,23 | 0,0031 | 0,998 |
| (w = 0, lr 1e-5 : `v-dumb` v4) | 6,53 | 12,23 (5,38) | 0,977 | 0,99 | 0 (porte) | 0,006 |

- **Toujours réfutée** (meilleur 0,0035 < 0,0577). La charnière ne lie jamais : aucune image ou presque
  n'atteint t (`reached` ≤ 0,02), la perte moyenne finit à 0,01-0,06 (critère « borne » : 0,22-0,90).
- lr 1e-4 à w = 1 tend l'ancre (sa KL monte à 0,14-0,30 pendant l'entraînement) sans faire oublier : à
  taux égal pour les deux termes, l'ancre l'emporte.
- La direction est nette : la KL exacte croît quand w baisse (0,006 → 0,017 → 0,030 de w 0,3 à 0,03),
  lentement (× 5 pour w ÷ 10), puis saute à l'effondrement à w = 0. Le bras w = 0,03 n'est pas
  « immobile » au sens posé (0,030 > 0,02) : la règle d'arrêt du suivi D ne s'applique pas encore.

## Suivi D2 : la falaise entre w = 0,03 et w = 0 (posé le 2026-11-22, avant le run)

**Ce qui l'ouvre** : la direction mesurée au suivi D (moins d'ancre, plus d'oubli), prolongée au même pas
logarithmique. Dernier suivi de dose : la question est de savoir s'il existe un régime intermédiaire
entre l'immobilité et l'effondrement.

**Bras** (`v-dumb-dose2`, t = 0,45, lr 1e-5, graine 0) : `retain_weight` 0,01 ; 0,003 ; 0,001.

**Critères (posés avant)** :
- ceux de la fiche et du suivi D (réfuté < 0,0577 ; candidat > 0,0697 avec retain ≥ 0,8 → graines 1 et
  2 ; charnière « borne » si perte moyenne entre 0,22 et 0,90 avec `reached` ≥ 0,5) ;
- **falaise** : si la KL exacte passe de moins de 0,1 à l'effondrement (porte non franchie) entre deux
  poids voisins sans qu'aucun bras ne dépasse 0,0577, V_Dumb n'a pas de dose utilisable à lr 1e-5 :
  **arrêt de la campagne**, sans grille plus fine.

### Résultat du suivi D2 (groupe `20261122T033250Z-unl-v-dumb-dose2`, commit 525b0bf, propre, 2 min 31)

| bras (t = 0,45, lr 1e-5, graine 0) | KL exacte | perte exacte (écart-type) | `final.reached` | audit/exact | `score_min` (recadr. / masq.) | retain | porte |
|---|---|---|---|---|---|---|---|
| w = 0,01 | 0,067 | 0,122 (0,91) | 0,033 | 0,19 | 0,019 (0,019 / 0,032) | 0,987 | 1 |
| w = 0,003 | 3,456 | 5,94 (5,04) | 0,805 | 0,99 | 0 | 0,052 | **0** |
| w = 0,001 | 5,496 | 9,94 (5,22) | 0,954 | 0,99 | 0 | 0,012 | **0** |

- **Falaise** (critère posé) : de w = 0,01 à w = 0,003, la KL exacte passe de 0,067 à l'effondrement
  (3,5, accord d'audit 0,23), sans qu'aucun bras ne dépasse 0,0577. **Arrêt de la campagne.**
- La trajectoire le montre : à w = 0,003, la KL de l'ancre vaut 0,05 à la première passe, 2,5 à la
  deuxième, 9,3 à la troisième ; à w = 0,01, elle monte encore (0,012 → 0,10 → 0,15) au bout des trois
  passes. La montée de la perte s'auto-entretient (son gradient croît à mesure que l'image est
  oubliée) ; l'ancre la tient, ou la lâche.
- À w = 0,01, comme pour la bande de la V5 : oubli faible et local (audit/exact 0,19 à KL exacte
  0,067), sans score (S_pred 0,04 à 0,12, S_ms 0,29 à 0,48).

## Conclusion (2026-11-22)

| composante de V_Dumb | verdict | mesure |
|---|---|---|
| la méthode, à sa fiche (lr 1e-5, w = 1, t ∈ {0,31 ; 0,45 ; 0,63}) | **réfutée** | `score_min` ≈ 0 : le modèle ne bouge pas (KL exacte 0,0005) |
| la charnière (« monter jusqu'à t, pas au-delà ») | **réfutée** : elle ne borne rien | avec l'ancre, aucune image n'atteint t (`reached` ≤ 0,03) ; sans ancre ou presque, la perte dépasse t de 13 à 27 fois |
| la cible t | **sans effet mesurable** | bras t = 0,31 et 0,45 identiques au bit près : la charnière ne lie jamais |
| l'ancre sur les vues (conforme) | **nécessaire** (critère de la fiche), mais en tout ou rien | w ≥ 0,01 : KL exacte ≤ 0,07 ; w ≤ 0,003 : effondrement ; lr 1e-4 à w = 1 : l'ancre l'emporte |
| dose | **aucune utilisable** à lr 1e-5 | meilleur run 0,019 (w = 0,01) ; V1 − 2σ = 0,0577 |

- **V_Dumb est abandonnée.** La méthode la plus simple ne bat pas la V1 sur le banc : elle n'a pas de
  régime entre l'immobilité et l'effondrement. Les V1 à V3 et SCRUB conforme valent donc quelque chose
  sur cette mesure, au sens posé par la fiche.
- Ce qui la distingue de SCRUB conforme, qui a la même ancre de vues et score 0,1085 : SCRUB alterne des
  pas de montée (KL plafonnée) et des pas d'ancre, quand V_Dumb somme les deux termes dans chaque pas.
  C'est une différence de conception, pas une cause mesurée : une V_Dumb alternée le dirait.
- Rien n'a été soumis. Le meilleur conforme mesuré reste SCRUB conforme, 0,1085 ± 0,0068.

## Inventaire

11 runs d'entraînement, tous propres, RTX 5080, 2026-11-22 (≈ 9 min de GPU, 61 °C au plus, aucune pause
thermique) : `v-dumb` (4), `v-dumb-dose` (4), `v-dumb-dose2` (3). Registre : 175 runs terminés, 1 refusé.
