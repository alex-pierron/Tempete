# Campagne V5 (oubli en bande dosé hors pli, `band_ascent`), RTX 5080

- Spécifications : `band-ascent-step0` (sans entraînement), puis `band-ascent` (4 bras), puis les
  suivis justifiés plus bas (au plus 4 variantes chacun) ; `ref4-scrub-fv` rejoué en contrôle
- Tâche : `unlearning` ; banc : **`bench-v4`** (6 doublures : recadrages s40/s50/s60, masquages
  m30/m20/m15) ; fold d'audit `holdout_fraction=0.2`
- Créée le : 2026-11-22, avant tout run de la V5 ; responsable : Alex
- Méthode : `band_ascent` (rapport §3, V5 ; code du commit b9e9614, fusionné en 52fa1d3 ; V_Dumb
  fusionnée ensuite en 3a1f81e, sans toucher à `band_ascent`). Le nom de la fiche garde la série des
  campagnes (`trust-region-*-campaign.md`) ; la V5 n'est pas une option de `trust_region`.
- Contrat : les deux fiches V5 (`band-ascent-step0.md`, `band-ascent.md`), hypothèses et seuils posés
  avant tout run. Cette fiche ne les réécrit pas : elle fixe l'ordre, les contrôles, les comparaisons
  ajoutées et la règle des suivis, et consigne les résultats.
- Tableau : `make hk-unl-scores` → `SCOREBOARD.md` (section `bench-v4`)

## Grandeurs lues sur chaque run

- `bench.score_min` (pire des six doublures), `bench.score_min_family.crop` et `.mask`.
- **audit/exact** = `bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub` : le mécanisme visé
  (V1 retenue : 0,878 / 0,876 / 0,869 ; V3 : 0,857 / 0,897 / 0,909 ; SCRUB conforme lr 3e-4 :
  0,800 / 0,934 / 0,816 ; `scrub` proxy lr 1e-4 : 0,553 / 0,511 / 0,596, graines 0-2).
- Perte des images exactes : `bench.groups.forget_exact.loss_mean_sub` (la « perte comparable,
  ±30 % » de la fiche `band-ascent`).
- V5 : `final.cv_stop_level`, `final.cv_monitor_kl` × `final.cv_scale` contre `bench.audit_kl`
  (estimateur de la dérive, ±30 %), `final.hard_stop`, `final.band_levels_done`,
  `final.pushed_count`, et l'historique par niveau (`stdout.log` : dérive du témoin, accord, part de
  la sonde au-dessus de sa borne).
- Contrôles : `bench.gate_passed` = 1, `bench.audit_kl` < 0,34, `bench.retain` (≥ 0,85 attendu par la
  fiche ; plancher de 0,80 hérité de la V4 pour retenir un réglage).

## Références appariées (au registre, `bench-v4`, RTX 5080)

| référence | graine 0 | graine 1 | graine 2 | moyenne | pire masquage (moy.) |
|---|---|---|---|---|---|
| V1 retenue (`ref4-v1`) | 0,0647 | 0,0609 | 0,0656 | 0,0637 ± 0,0025 | 0,0702 |
| V3 retenue (`ref4-v3`) | 0,0784 | 0,0804 | 0,0806 | 0,0798 ± 0,0012 | 0,0798 |
| **SCRUB conforme lr 3e-4** (`scrub-fv-dose` v1-v2, `scrub-fv-seeds` v1) | 0,1151 | 0,1015 | 0,1090 | **0,1085 ± 0,0068** | 0,1085 |

## Seuils

- **Fiche `band-ascent` (tels quels)** : σ = max(écart-type V1, 0,0025) = 0,0025, donc le meilleur bras
  doit dépasser **0,0687** (V1 + 2σ) ; audit/exact du bras par défaut ≤ 0,5 et ≤ 0,5 × celui du bras
  `band_release=1` à perte exacte comparable (±30 %) ; `final.hard_stop` = 0 et
  `final.cv_stop_level` ≥ 1 ; `bench.gate_passed` = 1.
- **Comparaisons ajoutées (ETAT.md §6, posées ici avant les runs, hors fiche)** : sur la V3 appariée,
  moyenne > **0,086** et gain sous le masquage > 2σ (0,006) ; sur SCRUB conforme apparié, moyenne
  > **0,122** (0,1085 + 2 × 0,007) et gain sous le masquage > 0,014. La V5 ne change la meilleure
  méthode conforme mesurée que si elle passe ce dernier seuil.
- **Localité, lue pour elle-même** : un réglage qui score au-dessus de la V3 avec audit/exact ≤ 0,5
  serait le premier à obtenir la localité à une dose qui score (V1 à V4 : 0,80 à 0,93 à leur dose
  retenue).

## Ordre

1. `pytest tests/hackathon` (562 passent, 3 ignorés, le 2026-11-22 après la fusion 3a1f81e).
2. `band-ascent-step0` (sans entraînement), jugée contre sa fiche. **Si elle réfute la V5, on
   n'entraîne pas sous cette forme** : on documente et on s'arrête, avec une proposition.
3. `ref4-scrub-fv` (graines 0-2, ~2 min) : la spécification de rejeu n'a jamais tourné sous son nom ;
   elle doit redonner au bit près `scrub-fv-dose` v1-v2 et `scrub-fv-seeds` v1 (`score_min`,
   `retain`, `audit_kl`) après la fusion de V_Dumb (`methods.py`, `configs.py` modifiés). Un écart
   arrête la campagne.
4. `band-ascent` (4 bras, graine 0), jugée contre sa fiche.

## Règle des suivis (posée avant les runs)

Un suivi n'est lancé que pour une de ces raisons ; sa question, ses bras et ses seuils sont écrits ici
avant le lancement :

1. **Effondrement** (fiche : `final.hard_stop` = 1 ou `final.cv_stop_level` = 0 au bras par défaut) →
   les mêmes bras à lr 3e-5 ; abandon s'il s'effondre encore.
2. **Candidat** (meilleur bras > 0,0687 et contrôles passés) → graines 1 et 2 du même réglage,
   jusqu'à trois graines appariées ; le verdict final se lit sur la moyenne.
3. **Mécanisme tranché sur une seule graine** : chaque mécanisme (sélection, borne haute, choix hors
   pli) doit être jugé sur au moins deux graines s'il décide du verdict à moins de 2σ.
4. **Dose voisine** seulement si la mesure en donne la direction : par exemple, localité obtenue mais
   oubli insuffisant (validation croisée qui s'arrête tôt : `cv_max_drift`, ou ρ), ou casse à un niveau
   et pas au précédent (lr) ; `phi` seulement pour tester la prédiction « la localité croît quand φ
   baisse » si la borne haute est confirmée.
5. **Combinaison avec l'ancre conforme de `scrub`** : seulement si la V5 est locale mais n'oublie pas
   assez, et si l'ancre est défendable (elle n'entre pas dans `band_ascent` : il faudrait du code,
   donc une nouvelle version, hors de cette campagne).

Arrêt quand chaque mécanisme (sélection, borne haute, choix hors pli) est confirmé ou réfuté et que le
meilleur réglage a trois graines appariées s'il score.

## Résultats

### Ordre 1 : tests

`pytest tests/hackathon` : 562 passent, 3 ignorés (commit 3a1f81e, après la fusion de V_Dumb).

### Ordre 2 : étape 0, la V5 réfutée avant tout entraînement

Run `20261122T024729Z-unl-band-ascent-step0` (commit 3a1f81e, propre, 51 s). Détail : fiche
`band-ascent-step0.md`.

| | m30 | m20 | m15 | seuil de réfutation |
|---|---|---|---|---|
| précision φ = 0,25 (taux de base) | 0,183 (0,190) | 0,109 (0,118) | 0,087 (0,086) | < 0,35 sur les trois |
| `loss_share` (oracle) | 0,236 (0,919) | 0,231 (0,976) | 0,238 (0,991) | < 0,5 sur les trois |
| direction `runner_ups` / `views` | 0,752 / 0,637 | 0,765 / 0,609 | 0,768 / 0,594 | lecture : `views` > `runner_ups` + 0,05 |

- **Réfutée, sur les deux critères et les trois masquages.** Sous le masquage, la sélection de la V5
  vaut un tirage au hasard : précision égale au taux de base (0,92 à 1,01 fois), part de la perte
  égale à φ (0,23 à 0,24 pour 0,25). Même sous le recadrage, famille commune aux vues de la V5 et aux
  doublures, elle ne porte que 0,41 à 0,47 de la perte (oracle 0,74 à 0,92).
- Réserve, notée après coup : à φ = 0,25, la précision ne peut dépasser taux de base / φ ; sous m15
  ce plafond (0,344) est sous le seuil de 0,35. Le critère porte sur les trois masquages ; sous m30 et
  m20 (plafonds 0,76 et 0,47), la précision reste au taux de base, et le critère de masse tranche seul.
- Le pari de direction tient sous le masquage (`runner_ups` devant `views` de 0,11 à 0,17) : la
  montée pousse la masse vers les bonnes classes fausses, à condition de savoir quelles images
  pousser. C'est ce savoir qui manque.
- Conformément à la fiche, **`band-ascent` n'est pas lancé**. Les ordres 3 (`ref4-scrub-fv`) et 4
  (`band-ascent`) n'ont plus d'objet : le rejeu de contrôle ne servait qu'à apparier la V5.
- Mécanismes : **sélection réfutée** (étape 0) ; **borne haute** et **choix hors pli** non testés, car
  la fiche interdit d'entraîner la V5 sous cette forme. Le second ne vaut rien seul ; la première est
  la seule partie de la V5 qui vise la localité, d'où la proposition ci-dessous.

### Mesure exploratoire (posée le 2026-11-22, avant d'être lancée) : la difficulté par image est-elle propre à la famille ?

**Ce qui l'ouvre** : la sélection fait 2,1 à 2,6 fois le hasard sous le recadrage et exactement le
hasard sous le masquage. Deux lectures. (a) Les deux familles de doublures ne s'accordent pas sur les
images difficiles : aucune sélection tirée des vues de l'original ne peut alors être validée sur le
banc sans circularité. (b) Seules les vues de recadrage sont mauvaises pour prédire la difficulté,
et une sélection tirée d'une autre famille (le masquage, à d'autres graines que le banc) prédirait les
deux.

**Mesure** (sans entraînement, ~1 min, script `scripts/unlearning_v5_families.py`, sortie
`measurements/v5_families.json`) sur les 2800 images entraînées : corrélation de rang (Spearman) des
pertes par image entre les six doublures et deux sélecteurs (les vues de la V5 ; des masquages de
30 % tirés aux graines de la méthode, pas à celles du banc) ; et précision et part de la perte, à
φ = 0,25, de chaque sélecteur sous chaque doublure.

**Lecture, posée avant** : (a) si le sélecteur par masquage fait au plus 1,3 fois le hasard sous les
recadrages, alors que chaque sélecteur fait au moins 2 fois le hasard dans sa propre famille ; (b) si
le sélecteur par masquage fait au moins 2 fois le hasard sous les deux familles. Entre les deux : non
tranché. Cette mesure ne change pas le verdict de l'étape 0 ; elle ne sert qu'à la proposition.

### Résultat de la mesure exploratoire : lecture (a), la difficulté est propre à la famille

`scripts/unlearning_v5_families.py` (commit 3a1f81e, sans entraînement, 55 s), sortie
`measurements/v5_families.json`. Le sélecteur `v5` redonne exactement les chiffres de l'étape 0
(mêmes vues, même fold) : contrôle de reproductibilité.

Corrélation de rang (Spearman) des pertes par image, 2800 images entraînées :

| | `v5` | `mask_own` | s40 | s50 | s60 | m30 | m20 | m15 |
|---|---|---|---|---|---|---|---|---|
| `v5` (vues de la V5) | 1 | 0,00 | 0,35 | 0,37 | 0,35 | 0,00 | 0,00 | 0,01 |
| `mask_own` (masquages 30 %, graines propres) | 0,00 | 1 | −0,03 | 0,00 | 0,04 | 0,92 | 0,91 | 0,89 |
| s50 | 0,37 | 0,00 | 0,34 | 1 | 0,34 | 0,00 | −0,01 | 0,00 |
| m20 | 0,00 | 0,91 | −0,03 | −0,01 | 0,04 | 0,91 | 1 | 0,93 |

Sélection à φ = 0,25 (précision / taux de base ; part de la perte) :

| sélecteur | sous les recadrages s40 / s50 / s60 | sous les masquages m30 / m20 / m15 |
|---|---|---|
| `v5` | 2,13 / 2,41 / 2,62 ; 0,41 / 0,44 / 0,47 | 0,96 / 0,92 / 1,01 ; 0,24 / 0,23 / 0,24 |
| `mask_own` | 0,86 / 1,17 / 0,69 ; 0,23 / 0,23 / 0,22 | 3,71 / 3,93 / 3,98 ; 0,87 / 0,94 / 0,96 |

- **Lecture (a), posée avant.** Le sélecteur par masquage fait au plus 1,17 fois le hasard sous les
  recadrages (seuil 1,3), et chaque sélecteur fait au moins 2,1 fois le hasard dans sa famille. Les deux
  familles de doublures ne s'accordent pas sur les images difficiles : corrélation de rang de −0,04 à
  +0,05 entre tout recadrage et tout masquage.
- Dans une famille, l'accord diffère : sous le masquage, la difficulté est presque une propriété de
  l'image (0,89 à 0,93 entre tirages et forces) ; sous le recadrage, elle dépend surtout du tirage
  (0,28 à 0,37, les vues de la V5 comprises).
- Conséquence : toute sélection tirée des vues de l'original n'a d'effet que sous sa propre famille,
  où elle est circulaire, et vaut le hasard sous l'autre. Comme `score_min` retient la pire doublure, le
  banc ne peut ni la valider ni lui donner de gain. Le résultat de l'étape 0 ne tient donc pas au choix
  des recadrages pour les vues de la V5 : une V5 aux vues masquées échouerait sous les recadrages.

## Proposition pour la suite (posée après l'étape 0 ; la 1 a été lancée, suivis L et L2)

1. **Mesurer la localité seule (recommandé, ~6 min).** La borne haute des images tenues est la seule
   partie de la V5 qui vise le levier qu'aucune version n'a obtenu, et sa prédiction (éq. v5hold :
   audit/exact ≈ χ²φ ≈ 0,22, contre ~0,86 sans la borne) ne dépend pas de l'identité des images
   poussées. Sous le masquage, la sélection de la V5 vaut un tirage au hasard : `band-ascent` tel qu'il
   est spécifié mesure donc ce mécanisme sans biais de sélection. Ce serait une mesure de mécanisme,
   pas une candidate. Ses critères de localité restent ceux de la fiche, déjà posés : audit/exact du bras
   par défaut ≤ 0,5 et ≤ 0,5 fois celui de `band_release=1` à perte exacte comparable (±30 %) ; le
   bras `band_floor=none` aussi local que le bras par défaut réfute le modèle linéaire (fiche) ; seuil
   ajouté ici : le confirmer demande ce bras au-dessus de 0,7 (la V1 est à 0,87) ; `final.hard_stop` = 0 et
   `final.cv_stop_level` ≥ 1, sinon lr 3e-5 ; l'estimateur `cv_monitor_kl` × `cv_scale` à ±30 % de
   `bench.audit_kl`. Critères de score abandonnés, puisque la prémisse est réfutée : un score
   > 0,0687 serait une surprise à vérifier sur trois graines. Si la localité est obtenue, la suite
   devient une sélection jugée par une famille de doublures qu'elle ne partage pas (point 2) ; sinon,
   l'approche par sélection est close.
2. **Pour toute méthode qui choisit ses images** : un juge indépendant. Il faudrait une troisième
   famille de doublures dans le banc (un `bench-v5`, d'une altération ni recadrage ni masquage) et un
   sélecteur tiré d'une autre famille que ce juge. Laquelle des familles ressemble au vrai gold,
   personne ne le sait localement : seul le serveur le dirait, ce qui demande une décision d'équipe (rien
   n'est soumis, décision du 2026-11-21).
3. **Sinon, l'efficacité plutôt que la localité.** La meilleure méthode conforme (SCRUB conforme) gagne
   par l'efficacité de son terme d'oubli, sans localité. La spécification en attente, `v-dumb`
   (V_Dumb, seuils posés, même ancre que SCRUB conforme), est la campagne suivante naturelle, dans une
   session neuve.

## Suivi L : la localité de la bande seule (posé le 2026-11-22, avant le run)

**Ce qui l'ouvre** : la proposition 1 ci-dessus, acceptée par l'utilisateur (« Oui, lance-la »). La
spécification `band-ascent` est lancée **telle quelle** (4 bras, graine 0, lr 1e-4, rien modifié),
comme **mesure de mécanisme**, pas comme candidate : la sélection est réfutée, et sous le masquage elle
vaut un tirage au hasard, ce qui retire le biais de sélection de la mesure de localité.

**Critères (posés avant)**, sur audit/exact = `bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`
(exact = toutes les images entraînées, poussées et tenues) :

- **L1, localité** (fiche) : bras par défaut ≤ 0,5 → la fenêtre locale existe à 2800 images ; > 0,5 →
  réfutée.
- **L2, la retenue localise** (fiche) : défaut ≤ 0,5 × `band_release=1`, à perte exacte comparable
  (`bench.groups.forget_exact.loss_mean_sub` à ±30 %) ; si les pertes diffèrent de plus de 30 %, non
  interprétable tel quel (on le note, on compare alors à perte appariée sur l'historique des niveaux
  si c'est possible).
- **L3, modèle linéaire** (éq. v5hold) : bras `band_floor=none` au-dessus de 0,7 → la borne haute est
  ce qui localise (modèle confirmé) ; à moins de 20 % du bras par défaut → modèle réfuté (fiche :
  « aussi local ») ; entre les deux : non tranché.
- **L4, stabilité** (fiche) : bras par défaut avec `final.hard_stop` = 0 et `final.cv_stop_level` ≥ 1 ;
  sinon, les mêmes bras à lr 3e-5 avant de conclure, et abandon s'il s'effondre encore.
- **L5, choix hors pli** (fiche) : `final.cv_monitor_kl` × `final.cv_scale` à ±30 % de
  `bench.audit_kl` au bras par défaut ; le bras `cv_folds=0` (les 4 niveaux) dit ce que le choix évite
  (porte, KL d'audit > 0,34, retain).
- **Contrôles** : `bench.gate_passed`, `bench.audit_kl` < 0,34, retain.
- **Score** : lu sous les deux familles, pas jugé (prémisse réfutée). Un bras au-dessus de 0,0687
  (V1 + 2σ) serait une surprise, à confirmer sur trois graines (règle 2).
- **Graines** : un critère tranché sur une graine à moins de 20 % de son seuil reçoit la graine 1 des
  bras qui le décident (règle 3), au plus 4 variantes.

### Résultat du suivi L (groupe `20261122T030333Z-unl-band-ascent`, commit 4185ff2, propre, 2 min 50)

| bras (graine 0, lr 1e-4) | niveaux gardés | KL exacte | perte exacte | KL audit | **audit/exact** | `score_min` (recadr. / masq.) | retain |
|---|---|---|---|---|---|---|---|
| défaut (φ 0,25, CV) | 1 (CV : arrêt dur au niveau 2) | 0,074 | 0,109 | 0,018 | **0,246** | 0,000 (0,005 / 0,000) | 0,981 |
| `band_release=1` (φ = 1) | 1 (idem) | 0,107 | 0,161 | 0,038 | **0,354** | 0,042 (0,042 / 0,062) | 0,958 |
| `band_floor=none` | **0** (CV : arrêt dur au pas 40 du niveau 1) | 0 | — | 0 | — (modèle rendu intact) | 0 | 1 |
| `cv_folds=0` (4 niveaux) | 4 | 0,213 | 0,286 | 0,115 | **0,538** | 0,017 (0,077 / 0,017) | 0,878 |

- **L1, localité : confirmée** (0,246 ≤ 0,5), proche de la prédiction du modèle (χ²φ ≈ 0,22).
- **L2 : la retenue ne divise pas le rapport par deux.** 0,246 contre 0,354 (0,69 fois), mais à perte exacte
  différente de 32 % (0,109 contre 0,161), juste hors des ±30 % : non interprétable tel quel. À dose
  égale, l'écart est encore plus petit : la trajectoire du bras par défaut passe de 0,246 (KL exacte
  0,074, niveau 1) à 0,538 (0,213, niveau 4 ; mêmes pas au niveau 1, prequential identique), soit
  environ 0,31 à la KL exacte du bras φ = 1, contre 0,354.
- **L3 : non mesurable tel quel.** Sans borne haute, le sous-run de validation croisée s'arrête au pas 40
  du premier niveau : la dérive du pli témoin y vaut 0,154 (× 1,56 = 0,24 > 0,2), contre 0,026 au bout
  des 70 pas du niveau 1 avec la borne. La borne divise la dérive du témoin par six au moins, au même
  niveau ; mais le bras n'a pas de modèle, donc pas de rapport audit/exact.
- **L4, stabilité : tenue** au bras par défaut (`final.hard_stop` = 0, `final.cv_stop_level` = 1).
  Le sous-run s'arrête net au début du niveau 2 (pas 10 et 20) ; le run final à 4 niveaux, lui, ne
  s'arrête pas (sonde au-dessus de sa borne ≤ 5 %).
- **L5, estimateur : réfuté.** `cv_monitor_kl` × `cv_scale` = 0,041 contre `bench.audit_kl` 0,018
  (2,2 fois ; φ = 1 : 1,5 fois) : la correction d'échelle (5/4)² est trop prudente ; le rapport mesuré
  audit/témoin vaut 0,70 et 1,04. Le choix n'en dépend pas : au niveau 2, la dérive brute du témoin
  (0,15-0,16) dépasse déjà Λmax = 0,1.
- **Score : nul, par la composante des prédictions.** Au bras par défaut, S_ms vaut 0,41 à 0,68 (V1 :
  0,34 à 0,43), mais S_pred 0,000 à 0,048. Ne bouger que 25 % des images, choisies au hasard vis-à-vis
  du masquage, ne rapproche pas la distribution des doublures. Avec φ = 1, S_pred monte à 0,07-0,15 ;
  à 4 niveaux, 0,03 (m15) à 0,27 (s40). C'est le « prix de la localité » que la théorie annonçait
  (rapport §3, V5), mesuré : presque toute la composante des prédictions.
- **Lecture hors critères** : `bench.audit_view_kl` (une vue forte par image d'audit, hors score) vaut
  0,176 au bras par défaut, 9,7 fois sa KL d'audit (V1 : 2,2 fois ; SCRUB conforme : 1,9 fois). La
  localité mesurée porte sur les images d'audit **mémorisées** ; sur des vues de ces images, la bande
  déplace l'original presque autant que les autres méthodes à KL exacte égale.

## Suivi L2 : la borne haute et la sélection à calendrier égal, et une seconde graine (posé le 2026-11-22, avant le run)

**Ce qui l'ouvre** : L3 n'est pas mesurable tel quel (pas de modèle) et L2 est hors de la fenêtre de perte
comparable. Le bras `cv_folds=0` donne une trajectoire de référence à 4 niveaux sans validation croisée ;
on y apparie les deux ablations, au même calendrier. Et la localité, seul résultat positif, ne repose
que sur une graine.

**Bras** (deux spécifications de 2 variantes, le runner ne faisant que des produits cartésiens) :

- `band-locality` (graine 0, `cv_folds=0`, `cv_max_drift=1`) : `band_floor=none` ; `band_release=1`.
  `cv_max_drift` ne fixe, sans validation croisée, que le budget de l'arrêt dur préquentiel du niveau 1
  (2 × 0,1 = 0,2) ; le porter à 2 laisse le bras sans borne aller au bout pour qu'on mesure son rapport.
  Le témoin apparié est le bras `cv_folds=0` du suivi L, dont la fenêtre préquentielle n'a jamais
  dépassé 0,063 : sa trajectoire n'aurait pas changé.
- `band-ascent-s1` (graine 1) : bras par défaut et bras `cv_folds=0`.

**Critères (posés avant)** :

- **L3** : `band_floor=none` (4 niveaux) contre `cv_folds=0` (4 niveaux, graine 0, audit/exact 0,538) :
  au-dessus de 0,7, ou au moins 1,3 fois le témoin → **la borne haute est ce qui localise** ; à moins de
  20 % du témoin → modèle réfuté ; entre les deux : non tranché. Un effondrement (porte, accord) compte
  pour la borne.
- **L2** : `band_release=1` (4 niveaux) contre le même témoin : audit/exact du témoin ≤ 0,5 × celui de
  φ = 1, à perte exacte comparable (±30 %) → la sélection et la retenue localisent ; sinon réfuté. Si
  les pertes diffèrent de plus de 30 %, comparer à KL exacte égale sur les deux points de la
  trajectoire du témoin (niveaux 1 et 4), en le disant.
- **Graine 1** : L1 tient si le bras par défaut reste ≤ 0,5 ; le point à 4 niveaux est reproduit si son
  rapport reste à ±20 % de 0,538.
- Contrôles et score lus comme au suivi L, sans critère de score.

### Résultat du suivi L2 (groupes `20261122T031006Z-unl-band-locality`, `20261122T031151Z-unl-band-ascent-s1`, propres, 3 min 41)

| bras | graine | niveaux | KL exacte | perte exacte | **audit/exact** | `score_min` (recadr. / masq.) | retain |
|---|---|---|---|---|---|---|---|
| témoin `cv_folds=0` (suivi L) | 0 | 4 | 0,213 | 0,286 | **0,538** | 0,017 (0,077 / 0,017) | 0,878 |
| sans borne haute, `cv_folds=0` | 0 | 4 | 2,711 | 1,872 | **1,010** | 0 (porte non franchie) | 0,143 |
| φ = 1, `cv_folds=0` | 0 | 4 | 0,418 | 0,264 | **0,897** | 0,040 (0,122 / 0,040) | 0,698 |
| défaut (CV) | 1 | 2 (CV : arrêt dur au niveau 3) | 0,119 | 0,169 | **0,393** | 0,005 (0,038 / 0,005) | 0,950 |
| `cv_folds=0` | 1 | 4 | 0,271 | 0,288 | **0,739** | 0,046 (0,117 / 0,046) | 0,809 |

- **L3, borne haute : confirmée** au sens posé (1,01 ≥ 0,7, 1,9 fois le témoin, et effondrement : accord
  d'audit 0,53, porte non franchie, retain 0,14). Réserve : sans la borne, la dose explose aussi (KL
  exacte 2,7 contre 0,21) ; l'ablation ne sépare pas « la borne localise » de « la borne empêche
  l'effondrement ». Elle montre que la borne est indispensable.
- **L2, sélection et retenue : réfutées.** À perte exacte comparable (0,286 contre 0,264, −8 %), le
  témoin est à 0,60 fois φ = 1, pas 0,5. Plus net : rapportés à la KL exacte, les points φ = 1 (0,107 →
  0,354 ; 0,418 → 0,897) tombent sur la même courbe que les points φ = 0,25 (0,074 → 0,246 ; 0,119 →
  0,393 ; 0,213 → 0,538 ; 0,271 → 0,739). À KL exacte égale, le rapport ne dépend pas de φ : la localité
  de la bande ne vient pas de la retenue des 75 % d'images, contrairement au modèle linéaire (éq.
  v5hold), mais de la montée bornée elle-même.
- **Graine 1** : L1 tient (0,393 ≤ 0,5, à un niveau de plus que la graine 0, la validation croisée gardant
  2 niveaux) ; le point à 4 niveaux n'est pas reproduit à ±20 % (0,739 contre 0,538), mais la graine 1
  dérive plus (KL exacte 0,271 contre 0,213) et reste sur la même courbe.
- **L5, graine 1** : estimateur 1,8 fois trop haut (0,082 contre 0,047) ; rapport audit/témoin mesuré
  0,89. Sur trois bras, l'estimateur surestime la dérive finale de 1,5 à 2,2 fois.
- **Score** : 0,000 à 0,046 sur les huit runs, tous sous V1 + 2σ (0,0687) ; aucun ne demande de graines.
- **Comparaison aux autres méthodes, à KL exacte égale** (registre `bench-v4`) : la bande est la plus
  locale des méthodes mesurées. Vers 0,08 : bande 0,25, SCRUB conforme lr 2e-4 0,49. Vers 0,12 : bande
  0,35-0,39, `scrub` proxy lr 1e-4 0,55. Vers 0,17 à 0,27 : bande 0,54-0,74, SCRUB conforme lr 3e-4
  0,80-0,93, V1 0,87, SCRUB conforme lr 4e-4 1,03-1,09. Mais pas sur les vues : `bench.audit_view_kl`
  rapportée à la KL exacte vaut 1,6 à 2,4 pour la bande, contre 1,5 à 1,9 pour SCRUB conforme et la V1.

## Conclusion (2026-11-22)

| mécanisme de la V5 | verdict | mesure |
|---|---|---|
| sélection (25 % des images où les vues doutent) | **réfutée**, à l'étape 0 puis à l'entraînement | précision = taux de base sous le masquage, 0,23-0,24 de la perte ; à KL exacte égale, φ = 1 aussi local que φ = 0,25 |
| retenue des images non poussées (modèle linéaire, éq. v5hold) | **réfutée** comme source de la localité | même courbe audit/exact contre KL exacte pour φ = 0,25 et φ = 1 |
| borne haute des images tenues | **confirmée** comme indispensable | sans elle, effondrement à 4 niveaux (audit/exact 1,01, porte non franchie), arrêt dur au premier niveau du sous-run |
| choix de la dose hors pli | **tient la dérive, estimateur réfuté** | garde 1 à 2 niveaux (retain 0,95-0,98) ; estimateur 1,5 à 2,2 fois trop haut ; doses gardées sans score |
| direction (secondes de l'original) | **tient** sous le masquage, à logits libres | `runner_ups` 0,75-0,77 contre `views` 0,59-0,64 |
| localité (audit/exact ≤ 0,5) | **obtenue**, deux graines (0,246 ; 0,393), sur les images d'audit mémorisées | la plus locale à KL exacte égale ; pas sur des vues de ces images |

- **La V5 est abandonnée** : elle n'est candidate à rien (`score_min` ≤ 0,046 sur huit runs, sous la V1).
  Sa prémisse de sélection est fausse sur le banc, et la localité qu'elle obtient ne paie pas.
- **La localité ne paie pas, mesuré sur la bande** : au niveau gardé par la validation croisée,
  S_ms vaut 0,41 à 0,70 (au-dessus de la V1), mais S_pred 0,00 à 0,13. Les doublures sont moins sûres
  que l'original sur **toutes** les images, typiques comprises ; ce mouvement commun est celui que la
  localité refuse. Sur ce banc, la localité ne rapporte que par le retain (0,95-0,98).
- Un constat plus général, mesuré : les deux familles de doublures du banc ne s'accordent pas sur les
  images difficiles (corrélation de rang ≈ 0). Il est cohérent avec l'écart, déjà mesuré, des gains
  des V2 et V3 entre recadrage et masquage (leurs cibles sont tirées de vues par recadrage), et
  renforce la règle 6 de `ETAT.md` (lire les deux familles).
- Leçon de méthode : audit/exact croît avec la dose pour toutes les méthodes (SCRUB conforme : 0,23 à
  1,06 de lr 1e-4 à 4e-4 ; bande : 0,25 à 0,90). Une comparaison de localité ne vaut qu'à KL exacte
  égale.
- Rien n'a été soumis. Le meilleur conforme mesuré reste SCRUB conforme, 0,1085 ± 0,0068.

## Inventaire

RTX 5080, 2026-11-22, ≈ 10 min de GPU, 61 °C au plus, aucune pause thermique :
- 2 exécutions sans entraînement : `band-ascent-step0` (1 run au registre) et
  `scripts/unlearning_v5_families.py` (hors registre, sortie versionnée) ;
- 8 runs d'entraînement, tous propres : `band-ascent` (4), `band-locality` (2), `band-ascent-s1` (2).

Registre : 164 runs terminés, 1 refusé.

Outillage : `SCOREBOARD.md` ne distinguait pas un bras `band_floor=none` du bras par défaut (une option
à `None` ne laisse aucun nombre au registre ; à graine égale, les deux auraient compté pour une seule
graine). `scoreboard.py` lit désormais le JSON du run pour ce cas ; un test couvre les quatre bras de
`band-ascent` (563 tests passent). Vérifié sur les vrais runs : quatre lignes distinctes, `δ0=none` et
`K=0` affichés.
