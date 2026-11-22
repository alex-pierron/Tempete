# Campagne trust_region V3 (cible calibrée) à pleine échelle (RTX 5080)

- Spécifications : `ref4-*` (références en `bench-v4`), `trust-region-calibrated` (spec distante,
  lancée telle quelle), puis `tr3-*`
- Tâche : `unlearning` ; banc : **`bench-v4`** (6 doublures : recadrages s40/s50/s60, masquages
  m30/m20/m15)
- Créée le : 2026-11-22 ; responsable : Alex
- Méthode : `trust_region`, `trust_target=calibrated` (rapport, section V3 ; code du commit 4099f0f)
- Contrat de départ : la fiche distante `trust-region-calibrated.md` (hypothèses et critères de
  réfutation posés avant tout run) ; cette fiche les chiffre et pose les étapes suivantes.
- Tableau : `make hk-unl-scores` → `SCOREBOARD.md` (section `bench-v4`)

## Étape 0 : la cible V3 mesurée avant tout entraînement

`scripts/unlearning_v3_targets.py` → `measurements/v3_targets.json` (2800 images entraînées,
graine 0, mêmes vues et même fonction de cible que la méthode ; 560 cibles : aire 0,35-0,6, 1 à 8
vues, λ 0 à 1, couverture b 0 à 0,5, τ de couverture 2 ou 6). `plafond` : score d'oubli (images
exactes) d'un élève dont la sortie serait la cible, sous chaque doublure `bench-v4`.

**Différence clé avec la V2 : le rayon de la V3 est 0,9 × la distance de sa cible**
(`radius_fraction`). L'élève finit près de sa cible : le plafond y est donc plus prédictif que pour
la V2 (rayon fixe 0,15, loin de sa cible).

| cible | atteignable | accord | hétérog. | plafond s40 / s50 / s60 | plafond m30 / m20 / m15 | plafond min (6) |
|---|---|---|---|---|---|---|
| V1, τ = 6 | 1,217 | 1,000 | 0,27 | 0,000 / 0,000 / 0,000 | 0,000 / 0,000 / 0,000 | 0,000 |
| V2 retenue (τ = 1, aire 0,45, 3 vues) | 0,264 | 0,944 | 1,76 | 0,147 / 0,185 / 0,145 | 0,071 / 0,042 / 0,000 | 0,000 |
| **V3 spec : aire 0,5, 4 vues, λ 0,25, b 0,25** | 0,176 | 0,945 | 2,04 | 0,120 / 0,148 / 0,120 | 0,086 / 0,080 / 0,041 | 0,041 |
| V3, λ 0 | 0,073 | 0,984 | 2,71 | 0,063 / 0,086 / 0,088 | 0,047 / 0,052 / 0,046 | 0,046 |
| V3, λ 0,5 | 0,280 | 0,895 | 1,95 | 0,177 / 0,202 / 0,127 | 0,125 / 0,098 / 0,012 | 0,012 |
| V3, λ 1 | 0,487 | 0,823 | 1,91 | 0,282 / 0,275 / 0,046 | 0,184 / 0,085 / 0,000 | 0,000 |
| V3, b 0 | 0,192 | 0,940 | 2,22 | 0,132 / 0,187 / 0,191 | 0,070 / 0,063 / 0,032 | 0,032 |
| V3, b 0,1 | 0,185 | 0,941 | 2,14 | 0,132 / 0,178 / 0,172 | 0,089 / 0,090 / 0,061 | 0,061 |
| V3, b 0,5 | 0,165 | 0,960 | 1,87 | 0,095 / 0,094 / 0,029 | 0,072 / 0,050 / 0,000 | 0,000 |
| V3, b 0,25, τ couverture 2 | 0,149 | 0,958 | 1,99 | 0,096 / 0,138 / 0,147 | 0,054 / 0,055 / 0,040 | 0,040 |
| V3, 1 vue, aire 0,6, λ 0,5 | 0,246 | 0,924 | 3,68 | 0,217 / 0,286 / 0,268 | 0,166 / 0,170 / 0,125 | 0,125 |
| V3, 1 vue, aire 0,6, λ 0,25 | 0,215 | 0,926 | 3,51 | 0,188 / 0,248 / 0,235 | 0,145 / 0,153 / 0,118 | 0,118 |
| V3, 2 vues, aire 0,6, λ 0,5, b 0,1 | 0,200 | 0,920 | 3,02 | 0,157 / 0,222 / 0,235 | 0,110 / 0,117 / 0,094 | 0,094 |
| V3, 1 vue, aire 0,5, λ 0,25 | 0,309 | 0,893 | 2,86 | 0,234 / 0,291 / 0,234 | 0,173 / 0,163 / 0,091 | 0,091 |
| V3, 8 vues, aire 0,5, λ 0,25 | 0,164 | 0,966 | 1,59 | 0,097 / 0,114 / 0,080 | 0,069 / 0,060 / 0,020 | 0,020 |
| V3, 4 vues, aire 0,6, λ 0,25 | 0,113 | 0,973 | 2,48 | 0,085 / 0,113 / 0,112 | 0,065 / 0,068 / 0,051 | 0,051 |
| V3, 4 vues, aire 0,35, λ 0,25 | 0,499 | 0,817 | 1,31 | 0,232 / 0,175 / 0,000 | 0,150 / 0,045 / 0,000 | 0,000 |

Lecture :

- **La cible V2 a un plafond nul sous le masquage** (m15) : cohérent avec le contrôle `bench-v4`
  (gain V2 réduit à +12 % sous cette famille).
- **La cible de la spec V3 est positive sous les six doublures** (masquage ≥ 0,041) ; atteignable
  0,176, d'où un rayon effectif ≈ 0,158, dans le plateau mesuré de V1 et V2.
- λ monte la distance (0,07 à 0,49) et dégrade s60 et m15 au-delà de 0,5. b = 0,5 est mauvais ;
  b = 0,1 relève le masquage (0,061 contre 0,041). Une seule vue à l'aire 0,6 donne les meilleurs
  plafonds (0,12 sous les six doublures) et une hétérogénéité de 3,5-3,7, celle des doublures
  mesurées (1,7-3,8).

## Références en `bench-v4` (posées avant)

Recalculées sans réentraînement (`measurements/bench_v4_rebench.json`) : V1 ε = 0,15 :
0,0637 ± 0,0025 ; V2 retenue : 0,0786 ± 0,0030 ; `scrub` lr 1e-4 (non conforme) : 0,098 ± 0,022.
Relancées au lab (`ref4-v1`, `ref4-v2`, `ref4-scrub`, graines 0-2) pour figurer au registre ;
hypothèse : elles redonnent exactement ces chiffres (entraînement déterministe).

## Règles de décision (chiffrées avant les runs)

Les critères de la fiche distante, avec σ = 0,003 (plus grand écart-type des références V1/V2) :

- **Améliore la V2** si le meilleur réglage V3, sur 3 graines, dépasse 0,0786 + 2 × 0,003 =
  **0,085** en `bench.score_min` ; **et** si sa pire doublure de masquage dépasse celle de la V2
  (0,0786 ± 0,0030) de plus de 2σ (gain non circulaire).
- **Casse** si `bench.audit_kl` > 1,5 × celui de la V1 à rayon voisin (0,228 → 0,34).
- **Mécanisme** (fiche distante) : `mean_std` V3 > V2 sous au moins 3 des 6 doublures ; écart-type
  / moyenne des pertes de l'élève sur les images exactes > 1.
- **Copies douces** : jugées sur `kl(jitter)/kl(exact)` et `kl(jpeg)/kl(exact)` à λ égal.

## Étapes (posées avant de lancer ; 2 à 5 conditionnelles)

1. **Spec distante telle quelle** (`trust-region-calibrated`) : λ ∈ {0,25 ; 0,5} × copies douces ∈
   {0 ; 0,5} (aire 0,5, 4 vues, b 0,25, τ 6, ρ 0,9). Hypothèse distante : la V3 bat V1 et V2 ;
   prédiction de l'étape 0 : λ = 0,25 fait mieux que 0,5 (plafond 0,041 contre 0,012).
2. **λ et couverture** au meilleur réglage de copies : λ ∈ {0 ; 0,1}, b ∈ {0 ; 0,1}.
3. **Vues et aire** (pistes de l'étape 0) : 1 et 2 vues à l'aire 0,6 ; 8 vues à 0,5.
4. **Fraction du rayon** ρ ∈ {0,6 ; 0,75} (et un rayon fixe ε = 0,15, pour comparer à la V2 à
   rayon égal).
5. **Confirmation** : meilleur réglage sur 3 graines, apparié à `ref4-v2`.

## Résultats

Tous en `bench-v4`, comparaisons appariées à graine égale (graine 0 sauf mention).

### Références au lab (9 runs)

`ref4-v1`, `ref4-v2`, `ref4-scrub` redonnent **exactement** les chiffres recalculés : V1
0,0637 ± 0,0025 (masquage 0,0702) ; V2 0,0786 ± 0,0030 (masquage 0,0786) ; `scrub` 0,098 ± 0,022.
Graine 0 : V1 0,0647, V2 0,0757.

### Étape 1 : spec distante telle quelle (4 runs)

| V3 (aire 0,5, 4 vues, b 0,25, τ 6, ρ 0,9) | rayon effectif | score_min | pire recadr. | pire masq. | S_ms s60 | S_pred s60 | retain | KL audit | jitter/exact | JPEG/exact |
|---|---|---|---|---|---|---|---|---|---|---|
| λ 0,25, copies 0 | 0,159 | **0,0761** | 0,0773 | 0,0761 | 0,48 | 0,21 | 0,755 | 0,244 | 1,82 | 1,28 |
| λ 0,25, copies 0,5 | 0,159 | 0,0762 | 0,0774 | 0,0762 | 0,50 | 0,21 | 0,749 | 0,253 | 1,80 | 1,27 |
| λ 0,5, copies 0 | 0,252 | 0,0566 | 0,0566 | 0,0649 | 0,64 | 0,13 | 0,678 | **0,343** | 1,64 | 1,23 |
| λ 0,5, copies 0,5 | 0,252 | 0,0646 | 0,0646 | 0,0697 | 0,60 | 0,15 | 0,697 | 0,316 | 1,68 | 1,24 |
| *V2 retenue, graine 0* | 0,150 | 0,0757 | 0,0811 | 0,0757 | 0,49 | 0,22 | 0,762 | 0,238 | 1,84 | 1,26 |

- Hypothèse distante (« bat V1 et V2 ») **réfutée à ce stade** : le meilleur point (λ 0,25) fait
  jeu égal avec la V2 (+0,0004), loin du seuil de 0,085.
- Prédiction de l'étape 0 (λ 0,25 mieux que 0,5) **tenue**. À λ 0,5, la cible pousse le rayon à
  0,25 : on retrouve la chute de la V1 au-delà de 0,2 par s60, et la KL d'audit franchit le seuil de
  casse (0,343 > 0,34).
- Copies douces **réfutées** : à λ 0,25, ni le score (0,0762 contre 0,0761) ni le transfert aux
  familles jamais vues (jitter 1,80 contre 1,82 ; JPEG 1,27 contre 1,28) ne bougent. À λ 0,5 elles
  aident un peu (+0,008), en freinant la dérive.
- Mécanisme `mean_std` **réfuté** : l'écart-type des pertes de l'élève sur les images exactes vaut
  ~3,4 fois leur moyenne pour toutes les cibles (V1 3,6 ; V2 3,4 ; V3 3,35) : c'est le réseau, pas
  l*, qui fixe cette dispersion. S_ms de la V3 n'est pas au-dessus de la V2.

### Étapes 2 à 4 (posées avant de lancer, au vu de l'étape 1)

Base : V3 λ 0,25, b 0,25, τ 6, aire 0,5, 4 vues, ρ 0,9, **sans copies** (sans effet mesuré), graine 0
(0,0761). Un facteur à la fois.

- **2, λ et couverture** (`tr3-lambda`, `tr3-hedge`) : λ ∈ {0 ; 0,1} (rayons effectifs ~0,07 et
  ~0,11 : on teste aussi l'effet d'un rayon plus petit, lié à λ par ρ) ; b ∈ {0 ; 0,1} (étape 0 :
  b 0,1 relève le plafond de masquage, 0,061 contre 0,041). Hypothèse : b = 0,1 fait mieux sous le
  masquage que b = 0,25 ; λ < 0,25 fait moins bien (rayon trop petit, comme V1 et V2 sous 0,1).
- **3, vues et aire** (`tr3-views`) : 1 vue à l'aire 0,6 (λ 0,25 et 0,5), 2 vues à 0,6 (λ 0,5,
  b 0,1), 8 vues à 0,5. Hypothèse (plafonds de l'étape 0, 0,09-0,125 sous les six doublures) : une
  ou deux vues à l'aire 0,6 battent la base de plus de 2σ (0,006). Réfutée sinon ; et la V2 a déjà
  montré qu'une seule vue peut dégrader (0,077 contre 0,081 en `bench-v3`).
- **4, rayon** (`tr3-radius`) : ρ ∈ {0,6 ; 0,75} (rayons ~0,11 et ~0,13) et rayon fixe ε = 0,15
  (comparaison à la V2 à rayon égal). Hypothèse : plateau comme V1 et V2, optimum vers 0,15.

### Étapes 2 à 4 : résultats (11 runs, graine 0 ; base 0,0761, V2 à la même graine 0,0757)

| V3 (un facteur changé) | rayon effectif | pas | score_min | pire recadr. | pire masq. | S_pred s60 | retain | lecture |
|---|---|---|---|---|---|---|---|---|
| **b = 0** | 0,173 | 102 | **0,0784** | 0,0865 | 0,0784 | 0,24 | 0,758 | meilleur |
| b = 0,1 | 0,167 | 97 | 0,0774 | 0,0825 | 0,0774 | 0,24 | 0,766 | |
| λ = 0 | 0,066 | 88 | 0,0604 | 0,0647 | 0,0604 | 0,28 | 0,857 | rayon trop petit |
| λ = 0,1 | 0,103 | 90 | 0,0679 | 0,0679 | 0,0702 | 0,26 | 0,823 | |
| 1 vue, aire 0,6, λ 0,25 | 0,194 | **880** | 0,0376 | 0,0431 | 0,0376 | 0,43 | 0,739 | rayon jamais atteint |
| 1 vue, aire 0,6, λ 0,5 | 0,221 | **880** | 0,0406 | 0,0466 | 0,0406 | 0,45 | 0,752 | rayon jamais atteint |
| 2 vues, aire 0,6, λ 0,5, b 0,1 | 0,180 | **880** | 0,0465 | 0,0554 | 0,0465 | 0,49 | 0,776 | rayon jamais atteint |
| 8 vues | 0,147 | 89 | 0,0738 | 0,0766 | 0,0738 | 0,23 | 0,778 | |
| ρ = 0,6 | 0,106 | 91 | 0,0717 | 0,0721 | 0,0717 | 0,25 | 0,806 | |
| ρ = 0,75 | 0,132 | 92 | 0,0735 | 0,0767 | 0,0735 | 0,23 | 0,779 | |
| ε fixe 0,15 | 0,150 | 92 | 0,0735 | 0,0767 | 0,0735 | 0,23 | 0,779 | même pas d'arrêt que ρ 0,75 : même modèle |

- 2 : λ < 0,25 **fait moins bien** (tenue) ; b : **la couverture nuit**, monotone (0,25 → 0,1 → 0 :
  0,0761 → 0,0774 → 0,0784). L'hypothèse « b = 0,1 mieux sous le masquage » est tenue en signe mais
  sous 2σ (+0,0013).
- 3 **réfutée**, et par le mode de défaillance déjà vu sur la V2 (τ ≤ 0,5) : avec 1 ou 2 vues la
  cible est très piquée, la divergence à l'original plafonne vers 0,07 et le rayon n'est jamais
  atteint (880 pas, autre régime : S_ms ~0,1, S_pred ~0,6). Les plafonds de l'étape 0 ne valent que
  si le rayon est atteint. 8 vues : rien à gagner.
- 4 : le score **monte encore avec le rayon** jusqu'à 0,16-0,17 (ρ 0,6 → 0,9 : 0,0717 → 0,0761). À
  rayon fixe 0,15, la V3 (0,0735) fait jeu égal avec la V2 (0,0757) : à quantité d'oubli égale,
  la cible calibrée n'apporte rien de mesurable.

### Étapes 5 et 6 (posées avant de lancer)

- **5, direction contre rayon, à b = 0** (`tr3-b0-radius`, `tr3-b0-lambda`) : ρ = 0,95 ; λ = 0,35
  (rayon effectif plus grand) ; rayon fixe ε ∈ {0,15 ; 0,2}. Hypothèse : à b = 0 le gain sur la
  base vient surtout du rayon (0,173 contre 0,159) ; à ε fixe 0,15, b = 0 fait jeu égal avec la V2
  à 2σ près. Réfutée si b = 0 à ε = 0,15 dépasse la V2 (0,0757, graine 0) de plus de 0,006.
- **6, graines** (`tr3-seeds`) : le meilleur actuel (b = 0, λ 0,25, ρ 0,9) aux graines 1 et 2,
  apparié à `ref4-v2`. Décision (fixée plus haut) : la V3 améliore la V2 si sa moyenne sur 3
  graines dépasse 0,085 et si sa pire doublure de masquage dépasse celle de la V2 de plus de 2σ ;
  jeu égal si elle reste à moins de 2σ de 0,0786.

### Étapes 5 et 6 : résultats (7 runs, dont 1 refusé)

| V3, b = 0 (λ 0,25, aire 0,5, 4 vues) | graine | rayon | score_min | pire recadr. | pire masq. | retain | KL audit |
|---|---|---|---|---|---|---|---|
| **ρ 0,9** | 0 ; 1 ; 2 | 0,173 | 0,0784 ; 0,0804 ; 0,0806 → **0,0798 ± 0,0012** | 0,0876 (moy.) | **0,0798 ± 0,0012** | 0,744 | 0,261 |
| ρ 0,95 | 0 | 0,183 | 0,0782 | 0,0854 | 0,0782 | 0,741 | 0,266 |
| λ 0,35 | 0 | 0,208 | 0,0771 | 0,0813 | 0,0771 | 0,725 | 0,288 |
| ε fixe 0,15 | 0 | 0,150 | 0,0762 | 0,0830 | 0,0762 | 0,777 | 0,220 |
| ε fixe 0,2 | 0 | — | refusé : cible à 0,192 < 0,2 (erreur de plan : distance connue du run b = 0) | | | | |
| *V2 retenue (`ref4-v2`)* | 0 ; 1 ; 2 | 0,150 | 0,0757 ; 0,0817 ; 0,0785 → 0,0786 ± 0,0030 | 0,0831 | 0,0786 ± 0,0030 | 0,778 | 0,225 |

- 5 **tenue** : à rayon fixe 0,15, b = 0 fait jeu égal avec la V2 (0,0762 contre 0,0757) ; son gain
  sur la base (+0,0023) vient pour moitié de la direction sans couverture (0,0735 → 0,0762 à
  ε = 0,15) et pour moitié du rayon plus grand (0,173). Au-delà (ρ 0,95, λ 0,35), le score redescend.
- 6 : **jeu égal avec la V2**, apparié graine par graine (+0,0027 ; −0,0013 ; +0,0021 ; moyenne
  +0,0012, 0,4 écart-type). Loin du seuil d'amélioration (0,085) ; pire masquage 0,0798 contre
  0,0786, sous 2σ. La V3 est 2,5 fois plus stable, mais abîme un peu plus le fold d'audit (KL 0,26
  contre 0,23, retain 0,74 contre 0,78 ; sous le seuil de casse de 0,34). Son avance sous recadrage
  (0,0876 contre 0,0831) est sur la famille circulaire.

### Étape 7 (posée avant de lancer) : aire et vues au meilleur réglage

`tr3-b0-area` : aire ∈ {0,4 ; 0,6} (4 vues) ; `tr3-b0-views2` : 2 vues (aire 0,5), à b = 0, λ 0,25,
ρ 0,9, graine 0 (0,0784). Hypothèse : aucun ne dépasse 0,0784 de plus de 2σ (0,006) ; 2 vues
risquent le mode de défaillance (rayon jamais atteint) vu à 1-2 vues à l'aire 0,6.

### Étape 7 : résultats (3 runs, graine 0 ; meilleur actuel 0,0784)

| V3, b = 0, λ 0,25, ρ 0,9 | cible | rayon effectif | pas | score_min | pire masq. | retain | KL audit |
|---|---|---|---|---|---|---|---|
| aire 0,4 (4 vues) | 0,386 | 0,348 | 99 | 0,0417 | 0,0504 | 0,643 | **0,394** |
| aire 0,5 (4 vues) | 0,192 | 0,173 | 102 | **0,0784** | 0,0784 | 0,758 | 0,243 |
| aire 0,6 (4 vues) | 0,120 | 0,108 | 106 | 0,0723 | 0,0723 | 0,810 | 0,185 |
| 2 vues (aire 0,5) | 0,246 | 0,222 | **880** | 0,0535 | 0,0535 | 0,773 | 0,262 |

**Tenue** : rien ne dépasse l'aire 0,5. Comme le rayon est tiré de la cible, l'aire agit surtout par
le rayon : 0,4 le pousse à 0,35 (effondrement de s60, KL d'audit au-delà du seuil de casse), 0,6 le
ramène à 0,11. Deux vues : rayon jamais atteint.

## Décision

- **La V3 ne bat pas la V2 : jeu égal.** Meilleur réglage (b = 0, λ 0,25, ρ 0,9, aire 0,5, 4 vues,
  sans copies) : **0,0798 ± 0,0012** en `bench-v4` contre 0,0786 ± 0,0030 pour la V2 ; écart apparié
  +0,0012 (0,4 écart-type), sous le seuil d'amélioration (0,085) ; pire masquage 0,0798 contre
  0,0786, sous 2σ. Elle est 2,5 fois plus stable mais abîme un peu plus le fold d'audit (retain
  0,74 contre 0,78).
- **Le réglage de la spec d'origine fait moins bien** (b 0,25 : 0,0761 ; λ 0,5 : 0,0566, au-delà
  du seuil de casse) : la couverture par l'original adouci nuit, de façon monotone.
- **Ce qui a été réfuté** (critères de la fiche distante et de celle-ci) : la V3 « bat V1 et V2 »,
  y compris sous le masquage ; le mécanisme par `mean_std` (S_ms de la V3 sous celui de la V2 sous
  les six doublures ; la dispersion des pertes de l'élève, ~3,3 fois leur moyenne, ne dépend pas
  de la cible) ; les copies douces (aucun effet sur le score ni sur le transfert au jitter et au
  JPEG) ; les pistes « une ou deux vues » de l'étape 0 (rayon jamais atteint).
- **Ce qui tient** : λ ≈ 0,25 (prédiction de l'étape 0) ; un optimum de rayon vers 0,15-0,18, comme
  V1 et V2 ; à rayon fixe 0,15, V3 et V2 font jeu égal (0,0762 contre 0,0757).
- **Leçon de méthode** : avec un rayon tiré de la cible (ρ × distance), tous les réglages de la
  cible (λ, b, aire, vues) agissent d'abord par le rayon. Comparer des cibles demande un rayon
  fixe ; et un plafond mesuré avant entraînement ne vaut que si le rayon est atteint.
- **Suite** : la cible n'est plus le levier (V2 et V3 convergent vers ~0,08) ; le défaut commun de
  V1, V2 et V3 reste la localité (ratio 1,25-1,37). Prochaine itération : l'ancre de rétention
  conforme (vues fortes des images oubliées).
