# Campagne trust_region V2 (cible par vues) à pleine échelle (RTX 5080)

- Spécifications : `experiments/unlearning/tr2-*.toml` (4 variantes au plus chacune)
- Tâche : `unlearning`
- Créée le : 2026-11-21
- Responsable : Alex
- Méthode : `trust_region` avec `trust_target=views` (rapport §3.4) ; code du commit 3e5b6ab, repris tel quel à la fusion
- Point de comparaison : la campagne V1 (`trust-region-campaign.md`)
- Tableau à jour : `make hk-unl-scores` → `experiments/unlearning/SCOREBOARD.md`

## Question

La V2 (cible propre à chaque image, tirée de vues fortes) améliore-t-elle la V1 (cible adoucie
uniforme) sur le banc `bench-v3`, au-delà du bruit, et pour quelle part ce gain est-il mécanique
(la doublure du gold du banc est elle aussi l'original sur des vues fortes) ?

## Ce que la campagne V1 impose avant tout run V2

- Tester le rayon à chaque pas (`radius_check_every=1`) : tous les 10 pas, des rayons différents
  donnent le même modèle. La spec distante `trust-region-views` (test tous les 10 pas, ε de 0,2 à
  0,8) n'est donc **pas** lancée telle quelle : elle ne mesurerait rien.
- L'optimum V1 est à petit rayon (plateau 0,1-0,2 ; 0,0637 ± 0,0025 à ε = 0,15, 3 graines).
- Taux, warmup, lot : sans effet mesurable sur la V1 ; on les garde aux réglages V1 (lr 1e-5, lot
  32, `epochs=10`, tout le réseau) pour que seule la cible change.
- Références à battre : V1 ε = 0,15 (0,0637 ± 0,0025) ; `scrub` lr 1e-4 (0,098 ± 0,022).

## Étape 0 : la cible V2 mesurée avant tout entraînement

`scripts/unlearning_v2_targets.py` → `measurements/v2_targets.json` (2800 images du fold entraîné,
graine 0, mêmes vues que la méthode). `atteignable` : KL(original ‖ cible) à T = 1, le plus grand
rayon possible ; `accord` : part des images où la cible garde la pseudo-étiquette ;
`hétérog.` : écart-type / moyenne de −log q_y (hétérogénéité de la cible) ; `plafond sXX` :
score d'oubli (images exactes) d'un élève dont la sortie SERAIT la cible, sous chaque doublure du
gold. Le plafond dit combien de la doublure la cible contient : c'est la part potentiellement
mécanique d'un gain V2. Un run s'arrête au rayon, loin de sa cible : le plafond renseigne une
direction, pas un score attendu.

| aire | τ | vues | atteignable | accord | hétérog. | plafond s40 | plafond s50 | plafond s60 |
|---|---|---|---|---|---|---|---|---|
| 0,25 | 1 | 3 | 1,082 | 0,699 | 1,37 | 0,330 | 0,000 | 0,000 |
| 0,25 | 2 | 3 | 1,093 | 0,713 | 0,86 | 0,127 | 0,000 | 0,000 |
| 0,25 | 3 | 3 | 1,388 | 0,721 | 0,60 | 0,000 | 0,000 | 0,000 |
| 0,25 | 4 | 3 | 1,787 | 0,724 | 0,43 | 0,000 | 0,000 | 0,000 |
| 0,25 | 6 | 3 | 2,498 | 0,731 | 0,25 | 0,000 | 0,000 | 0,000 |
| 0,35 | 1 | 3 | 0,510 | 0,872 | 1,52 | 0,249 | 0,217 | 0,000 |
| 0,35 | 2 | 3 | 0,620 | 0,878 | 0,94 | 0,157 | 0,000 | 0,000 |
| 0,35 | 3 | 3 | 0,924 | 0,882 | 0,63 | 0,042 | 0,000 | 0,000 |
| 0,35 | 4 | 3 | 1,341 | 0,884 | 0,44 | 0,000 | 0,000 | 0,000 |
| 0,35 | 6 | 3 | 2,132 | 0,885 | 0,25 | 0,000 | 0,000 | 0,000 |
| 0,45 | 1 | 1 | 0,630 | 0,848 | 2,81 | 0,290 | 0,216 | 0,000 |
| 0,45 | 1 | 3 | 0,264 | 0,944 | 1,76 | 0,147 | 0,185 | 0,145 |
| 0,45 | 1 | 6 | 0,233 | 0,974 | 1,20 | 0,116 | 0,142 | 0,108 |
| 0,45 | 1 | 10 | 0,228 | 0,981 | 1,00 | 0,107 | 0,128 | 0,093 |
| 0,45 | 2 | 3 | 0,363 | 0,951 | 1,06 | 0,124 | 0,040 | 0,000 |
| 0,45 | 3 | 3 | 0,635 | 0,952 | 0,68 | 0,064 | 0,000 | 0,000 |
| 0,45 | 4 | 3 | 1,043 | 0,950 | 0,46 | 0,000 | 0,000 | 0,000 |
| 0,45 | 6 | 3 | 1,871 | 0,951 | 0,25 | 0,000 | 0,000 | 0,000 |
| 0,55 | 1 | 3 | 0,152 | 0,973 | 2,05 | 0,088 | 0,127 | 0,134 |
| 0,55 | 2 | 3 | 0,234 | 0,979 | 1,20 | 0,095 | 0,064 | 0,000 |
| 0,55 | 3 | 3 | 0,472 | 0,981 | 0,72 | 0,063 | 0,000 | 0,000 |
| 0,55 | 4 | 3 | 0,860 | 0,981 | 0,47 | 0,000 | 0,000 | 0,000 |
| 0,55 | 6 | 3 | 1,702 | 0,981 | 0,24 | 0,000 | 0,000 | 0,000 |
| 0,65 | 1 | 3 | 0,109 | 0,982 | 2,19 | 0,059 | 0,090 | 0,108 |
| 0,65 | 2 | 3 | 0,180 | 0,984 | 1,29 | 0,078 | 0,064 | 0,000 |
| 0,65 | 3 | 3 | 0,394 | 0,986 | 0,75 | 0,060 | 0,000 | 0,000 |
| 0,65 | 4 | 3 | 0,764 | 0,986 | 0,48 | 0,001 | 0,000 | 0,000 |
| 0,65 | 6 | 3 | 1,607 | 0,988 | 0,24 | 0,000 | 0,000 | 0,000 |
| V1 | 1 | - | 0,000 | 1,000 | 7,16 | 0,000 | 0,000 | 0,000 |
| V1 | 2 | - | 0,014 | 1,000 | 1,95 | 0,008 | 0,011 | 0,011 |
| V1 | 3 | - | 0,120 | 1,000 | 0,94 | 0,030 | 0,000 | 0,000 |
| V1 | 4 | - | 0,400 | 1,000 | 0,58 | 0,023 | 0,000 | 0,000 |
| V1 | 6 | - | 1,217 | 1,000 | 0,27 | 0,000 | 0,000 | 0,000 |

Lecture :

- **Le choix distant τ = 4 venait de la grille de rayons jusqu'à 0,8.** Au rayon utile (~0,15),
  τ = 1 devient atteignable (0,264 à l'aire 0,45, 3 vues) et garde l'hétérogénéité de la cible
  (1,76 contre 0,46 à τ = 4).
- **Seules les cibles à τ = 1, aire ≥ 0,45, ont un plafond positif sous les trois doublures**
  (0,147 / 0,185 / 0,145 à l'aire 0,45, 3 vues). À τ ≥ 3, et pour la V1 à tout τ, le plafond est nul
  sous s50 et s60 : atteinte, la cible serait plus loin des doublures que l'original.
- **Biais mesuré** : ces plafonds positifs viennent du recouvrement avec les doublures (mêmes
  vues, aires voisines). La V1 n'en a aucun. Un gain V2 dans cette zone sera en partie mécanique.
- Plus de vues : accord plus haut (0,94 → 0,98), cible moins hétérogène et moins loin, plafonds en
  baisse. Une seule vue : cible très bruitée (accord 0,85), plafond nul sous s60.
- Aire 0,25 : cible fausse sur 30 % des images (accord 0,70) ; aire 0,65 à τ = 1 : atteignable
  0,109, sous ε = 0,15.

## Règles de décision (fixées avant les runs)

- Métrique : `bench.score_min`, porte comprise ; comparaison à la V1 **au même rayon** ; une
  différence ne compte que si elle dépasse deux écarts-types du bruit entre graines.
- La V2 **améliore** la V1 si son meilleur réglage, sur 3 graines, dépasse 0,0637 + 2 × 0,0025 =
  0,069 ; elle fait **jeu égal** entre 0,059 et 0,069 ; elle **régresse** sous 0,059.
- **Part mécanique** : un gain qui n'apparaît que pour des aires de cible dans la plage des
  doublures (0,40-0,70) et disparaît loin d'elles (0,25-0,35) est noté comme probablement
  mécanique ; seul le classement public trancherait.

## Étapes et hypothèses (posées avant de lancer)

Base V2 : `trust_target=views`, `target_view_min_area=0.45`, `target_views=3`, lr 1e-5, lot 32,
`epochs=10`, `radius_check_every=1`, `holdout_fraction=0.2`, graine 0.

1. **Rayon, cible distante** (`tr2-radius-a`, `tr2-radius-b`) : τ = 4 (le choix distant),
   ε ∈ {0,05 ; 0,1 ; 0,15 ; 0,2 ; 0,3 ; 0,5}.
   - H1 : la V2 de la spec distante, avec un arrêt fin, fait mieux que la V1 au même rayon
     (hypothèse distante). Réfutée si elle ne dépasse la V1 (étape 1 de la campagne V1) à aucun ε
     de plus de 0,005 (le double de l'écart-type V1).
2. **Température de la cible** (`tr2-temperature`) : ε = 0,15, τ ∈ {1 ; 2 ; 3 ; 6}.
   - H2 : contrairement à la V1 (score plat de τ = 4 à 12), τ compte pour la V2, et τ = 1 est le
     meilleur (seule cible au plafond positif sous les trois doublures). Réfutée si le score est
     plat en τ (écart < 0,005) ou si τ = 1 n'est pas le meilleur.
3. **Aire des vues** (conditionnelle) : au meilleur τ, aires faisables parmi {0,25 ; 0,35 ; 0,55 ;
   0,65} (à τ = 1, 0,65 est infaisable et 0,55 limite : atteignable 0,152). Diagnostic du biais.
4. **Nombre de vues** (conditionnelle) : {1 ; 6 ; 10} au meilleur (τ, aire).
5. **Rayon au meilleur réglage** : ε autour de l'optimum (l'optimum V2 peut différer de la V1).
6. **Confirmation** : meilleur réglage V2 sur 3 graines (la graine change aussi les vues), contre
   V1 ε = 0,15 (3 graines).

## Résultats

Comparaison **appariée** : même graine (0), donc mêmes lots, même fold d'audit ; seule la cible
change entre V1 et V2. Chiffres complets : `SCOREBOARD.md`.

### Étapes 1 et 2 (10 runs)

| réglage V2 (aire 0,45, 3 vues) | ε | score_min | s40 | s50 | s60 | V1 même ε | écart | S_ms s60 V2/V1 | S_pred s60 V2/V1 | retain V2 | pas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| τ = 4 | 0,05 | 0,0564 | 0,056 | 0,059 | 0,057 | 0,0511 | +0,0053 | 0,26 / 0,24 | 0,25 / 0,25 | 0,876 | 80 |
| τ = 4 | 0,10 | 0,0610 | 0,070 | 0,061 | 0,066 | 0,0594 | +0,0016 | 0,33 / 0,35 | 0,24 / 0,22 | 0,835 | 82 |
| τ = 4 | 0,15 | 0,0678 | 0,083 | 0,070 | 0,068 | 0,0647 | +0,0031 | 0,44 / 0,46 | 0,20 / 0,18 | 0,777 | 84 |
| τ = 4 | 0,20 | 0,0641 | 0,088 | 0,075 | 0,064 | 0,0595 | +0,0046 | 0,51 / 0,53 | 0,17 / 0,15 | 0,747 | 85 |
| τ = 4 | 0,30 | 0,0394 | 0,094 | 0,076 | 0,039 | 0,0338 | +0,0056 | 0,68 / 0,68 | 0,09 / 0,07 | 0,670 | 87 |
| τ = 4 | 0,50 | 0,0023 | 0,091 | 0,056 | 0,002 | 0,0018 | +0,0005 | 0,72 / 0,71 | 0,01 / 0,00 | 0,573 | 89 |
| **τ = 1** | 0,15 | **0,0811** | 0,096 | 0,084 | 0,081 | 0,0647 | **+0,0164** | 0,49 / 0,46 | 0,22 / 0,18 | 0,762 | 96 |
| τ = 2 | 0,15 | 0,0723 | 0,088 | 0,076 | 0,072 | 0,0647 | +0,0076 | 0,47 / 0,46 | 0,20 / 0,18 | 0,768 | 86 |
| τ = 3 | 0,15 | 0,0698 | 0,083 | 0,070 | 0,070 | 0,0647 | +0,0051 | 0,43 / 0,46 | 0,21 / 0,18 | 0,787 | 84 |
| τ = 6 | 0,15 | 0,0661 | 0,082 | 0,069 | 0,066 | 0,0647 | +0,0014 | 0,44 / 0,46 | 0,19 / 0,18 | 0,773 | 84 |

- H1 **tenue, de justesse** : à τ = 4, la V2 dépasse la V1 aux six rayons (signe constant), de
  0,0016 à 0,0056 ; deux écarts franchissent le seuil de 0,005 (ε = 0,05 et 0,3). Chaque écart
  isolé reste de l'ordre d'un à deux écarts-types V1.
- Le gain passe par la **composante des prédictions** (S_pred plus haut sous les trois
  doublures), pas par celle des pertes (S_ms égal ou un peu plus bas) : l'inverse du mécanisme
  annoncé par la spec distante, et ce qu'avait déjà vu son essai réduit.
- H2 **tenue** : τ compte pour la V2 (0,0661 à 0,0811, écart 0,015 contre un V1 plat) et le score
  monte quand τ baisse, comme le prédisaient les plafonds de l'étape 0. τ = 1 : +0,0164 sur la V1
  au même rayon et à la même graine, soit environ six écarts-types V1. Le retain est le même
  (0,762 contre 0,771).
- L'optimum est **au bord** de la plage explorée (τ = 1) : il faut voir en dessous.

### Étapes 2 bis, 3 et 4 (posées avant de lancer, au vu des étapes 1-2)

Base : τ = 1, aire 0,45, 3 vues, ε = 0,15, graine 0 (0,0811).

- **2 bis, τ sous 1** (`tr2-temperature-low`) : τ ∈ {0,5 ; 0,75}. Hypothèse : la tendance continue
  ou sature ; l'optimum est à τ ≤ 1. Réfutée si les deux font moins bien que τ = 1 de plus de 0,005
  (l'optimum serait alors à 1, sans être un bord).
- **3, aire des vues** : à τ = 1, aires {0,25 ; 0,35} (`tr2-area-t1`) ; à τ = 2, aires {0,25 ;
  0,35 ; 0,55 ; 0,65} (`tr2-area-t2`, car à τ = 1 les aires 0,55 et 0,65 sont à peine ou pas
  atteignables : 0,152 et 0,109 ; à τ = 2 elles le sont : 0,234 et 0,180). Diagnostic du biais :
  si le gain sur la V1 tombe pour les aires hors de la plage des doublures (0,25-0,35) et monte
  pour celles qui la recouvrent (0,45-0,65), il est probablement mécanique. Les aires 0,25-0,35
  changent aussi l'accord de la cible (0,70-0,88) : diagnostic indicatif, pas une preuve.
- **4, nombre de vues** (`tr2-views`) : {1 ; 6 ; 10} à τ = 1, aire 0,45. Hypothèse (plafonds de
  l'étape 0) : 3 vues est un bon compromis ; une seule vue (accord 0,85) dégrade, 6-10 vues font
  au plus aussi bien. Réfutée si 6 ou 10 vues dépassent 3 vues de plus de 0,005.

### Étapes 2 bis, 3 et 4 : résultats (11 runs, ε = 0,15, graine 0 ; V1 au même rayon : 0,0647)

| balayage | valeur → score_min (écart à la V1) | lecture |
|---|---|---|
| τ (aire 0,45, 3 vues) | 6 → 0,0661 ; 4 → 0,0678 ; 3 → 0,0698 ; 2 → 0,0723 ; 1 → 0,0811 ; 0,75 → 0,0838 ; **0,5 → 0,0847 (+0,020)** | monte quand τ baisse, sature sous 1 |
| aire à τ = 1 (3 vues) | 0,25 → 0,0776 (+0,013) ; 0,35 → 0,0806 ; 0,45 → 0,0811 | presque plat |
| aire à τ = 2 (3 vues) | 0,25 → 0,0722 ; 0,35 → 0,0719 ; 0,45 → 0,0723 ; 0,55 → 0,0717 ; 0,65 → 0,0716 | plat (écart 0,0007) |
| vues (τ = 1, aire 0,45) | 1 → 0,0766 ; 3 → 0,0811 ; 6 → 0,0802 ; 10 → 0,0804 | une vue dégrade ; 3 suffisent |

- 2 bis **tenue** : l'optimum est à τ ≤ 1 ; sous 1 le gain continue mais sature (0,0811 → 0,0847,
  environ 1,5 écart-type V1).
- 3, diagnostic du biais : **le gain ne suit pas le recouvrement avec les doublures.** À τ = 2,
  les cinq aires, de 0,25 (aucun recouvrement) à 0,65, donnent le même score à 0,0007 près ; à
  τ = 1, l'aire 0,25 garde un gain de +0,013 sur la V1 alors que sa cible se trompe sur 30 % des
  images. L'hypothèse « gain surtout mécanique par les aires » est donc **réfutée** dans la forme
  testée ; reste un recouvrement plus large qu'on ne peut pas tester ici (même famille de vues :
  recadrage et couleur).
- 4 **tenue** : une seule vue fait moins bien (−0,0045) ; 6 et 10 vues n'apportent rien.
- Lecture (interprétation, cohérente avec ces chiffres, non démontrée) : le run s'arrête au rayon,
  loin de sa cible ; seule compte la **direction** du premier mouvement, p_θ − q, c'est-à-dire la
  façon dont la cible répartit la masse retirée à la classe en tête. Une τ basse la concentre sur
  les classes que les vues confondent réellement, ce que mesure `prediction_score` ; l'aire et le
  nombre de vues changent surtout combien de masse est retirée, ce que le rayon normalise. La V1
  ne peut pas descendre en τ : sa cible y est trop près de l'original (0,014 à τ = 2).

### Étapes 5 et 6 (posées avant de lancer)

Les points τ = 0,5, 0,75 et 1 sont à moins de 0,004 l'un de l'autre : on choisit entre τ = 0,5 et
τ = 1 sur trois graines, pas sur une.

- **5, rayon** (`tr2-radius-best`) : ε ∈ {0,1 ; 0,2} × τ ∈ {0,5 ; 1} (aire 0,45, 3 vues, graine 0).
  Hypothèse : comme pour la V1, plateau entre 0,1 et 0,2, optimum vers 0,15. Réfutée si ε = 0,1 ou
  0,2 dépasse ε = 0,15 de plus de 0,005.
- **5 bis** (`tr2-temperature-vlow`) : τ = 0,25. Hypothèse : plateau, à moins de 0,005 de τ = 0,5.
- **6, graines** (`tr2-seeds`) : τ ∈ {0,5 ; 1} × graines {1 ; 2} (la graine 0 est déjà là), ε =
  0,15. La graine change aussi les vues de la cible. Décision : la V2 améliore la V1 si la moyenne
  sur 3 graines du meilleur des deux dépasse **0,069** ; on retient le τ à la meilleure moyenne, et
  le plus petit écart-type en cas d'égalité à un écart-type près.

### Étapes 5 et 6 : résultats (9 runs)

| réglage V2 (aire 0,45, 3 vues) | graines | score_min | rayon atteint | pas | lecture |
|---|---|---|---|---|---|
| **τ = 1, ε = 0,15** | 0, 1, 2 | 0,0811 ; 0,0880 ; 0,0802 → **0,0831 ± 0,0043** | 3/3 | 96-98 | retenu |
| τ = 0,5, ε = 0,15 | 0, 1, 2 | 0,0847 ; 0,0930 ; 0,0716 → 0,0831 ± 0,0108 | 2/3 | 105-880 | même moyenne, 2,5 fois plus bruité |
| τ = 1, ε = 0,10 / 0,20 | 0 | 0,0735 / 0,0782 | oui | 94 / 97 | sous ε = 0,15 (0,0811) |
| τ = 0,5, ε = 0,10 / 0,20 | 0 | 0,0736 / 0,0732 | oui / **non** | 102 / 880 | |
| τ = 0,25, ε = 0,15 | 0 | 0,0765 | **non** | 880 | |

V1 au même rayon et aux mêmes graines (`tr-seeds`) : 0,0647 ; 0,0609 ; 0,0656 → 0,0637 ± 0,0025.

- 5 **tenue** : ni ε = 0,1 ni ε = 0,2 ne dépassent ε = 0,15 (à τ = 1 : −0,0076 et −0,0029) ;
  optimum vers 0,15, comme la V1. Une graine par voisin : un plateau 0,1-0,2 n'est pas exclu.
- 5 bis : τ = 0,25 **ne plafonne pas, il décroche** (0,0765) — et pour une raison nouvelle.
- **Mode de défaillance sous τ = 1 (mesuré, non prévu).** Trois runs à τ ≤ 0,5 (τ = 0,5 graine 2,
  τ = 0,5 à ε = 0,2, τ = 0,25) n'atteignent jamais le rayon : la divergence à l'original plafonne
  vers 0,085 et l'entraînement va au bout des 880 pas (quatre fois le calcul). Le modèle change
  alors de régime : composante des pertes effondrée (S_ms 0,14-0,21), composante des prédictions
  très haute (0,50-0,66). Cible et départ sont pourtant à la même distance qu'à τ = 1 (0,27 contre
  0,26) : c'est la forme de la cible (très piquée sur quelques classes) qui bloque. À τ = 1, les sept
  runs atteignent le rayon en 89 à 104 pas.
- 6 **tenue** : à τ = 1, la V2 vaut 0,0831 ± 0,0043 > 0,069. Comparaison appariée par graine (même
  graine, mêmes lots, seule la cible change) : +0,0164, +0,0271, +0,0146, moyenne +0,0194 (t ≈ 5,1
  à 2 degrés de liberté). Le meilleur τ à moyenne égale est τ = 1 (écart-type 0,0043 contre
  0,0108, rayon toujours atteint).

### Contrôle après la troisième fusion : seconde famille de doublures (`bench-v4`)

`bench-v4` (code distant, fusionné le 2026-11-22) ajoute aux trois recadrages une seconde famille
de doublures du gold : le masquage de patchs (30, 20 et 15 % des patchs), étrangère à la cible de
la V2. Les fichiers déjà entraînés ont été recalculés sans réentraînement (`bench --candidate`,
`measurements/bench_v4_rebench.json`) ; les scores sous recadrage sont identiques à `bench-v3`, et
la V2 réentraînée sur le code fusionné redonne exactement 0,0811 sous recadrage.

| 3 graines | pire recadrage | pire masquage | `bench-v4` score_min (pire des 6) |
|---|---|---|---|
| V1 ε = 0,15 | 0,0637 ± 0,0025 | 0,0702 ± 0,0013 | 0,0637 ± 0,0025 |
| V2 τ = 1 | 0,0831 ± 0,0042 | 0,0786 ± 0,0030 | **0,0786 ± 0,0030** |
| écart V2 − V1 (par graine) | +0,016 ; +0,027 ; +0,015 (+30 %) | +0,005 ; +0,011 ; +0,010 (**+12 %**) | +0,011 ; +0,021 ; +0,013 (+23 %) |
| `scrub` lr 1e-4 (référence) | 0,098 ± 0,022 | 0,101 ± 0,023 | 0,098 ± 0,022 |

- La V2 bat la V1 sous les **deux** familles, à chaque graine.
- Sous la famille étrangère à sa cible, le gain est **deux fois et demie plus petit** (+12 % contre
  +30 %) : environ la moitié du gain mesuré sous recadrage est probablement liée à la famille de
  vues commune à la cible et aux doublures (la part « mécanique » que l'étape 3 ne pouvait pas
  tester). `scrub`, dont la cible n'est pas faite de vues, ne bouge pas d'une famille à l'autre.
- En `bench-v4`, la V2 dépasse encore le seuil de décision (0,0637 + 2 × 0,0025 = 0,069).

## Décision

- **La V2 améliore la V1 sur le banc : 0,0831 ± 0,0043 contre 0,0637 ± 0,0025 (+30 %) en
  `bench-v3`, au même rayon, apparié graine par graine ; 0,0786 ± 0,0030 contre 0,0637 ± 0,0025
  (+23 %) en `bench-v4`.** Le gain passe par la composante des prédictions ; sous la seule famille
  de doublures étrangère à sa cible (masquage), il n'est plus que de +12 % (contrôle ci-dessus).
- **Réglage retenu** : `trust_target=views`, τ = 1, aire 0,45, 3 vues, ε = 0,15,
  `radius_check_every=1` (le reste comme la V1 : lr 1e-5, lot 32, `epochs=10`, tout le réseau).
- **Ce qui ne marche pas** : la spec distante telle quelle (test tous les 10 pas, ε ≥ 0,2) ; τ ≥ 4
  (gain faible, +0,002 à +0,006) ; τ ≤ 0,5 (même moyenne, mais rayon parfois jamais atteint) ; une
  seule vue.
- **Biais** : le gain ne dépend pas du recouvrement des aires avec les doublures (étape 3), mais
  il est deux fois et demie plus petit sous une famille de doublures étrangère à la cible (+12 %
  contre +30 %) : on retient le chiffre `bench-v4` (+23 %, pire des six doublures) et l'on tient le
  gain réel pour incertain entre ces bornes. Seul le classement public tranchera.
- **Conformité** (sujet : « from the original model and the forget set alone ») : la V2 n'utilise
  que le forget set ; `scrub` et `gold_soft` ancrent leur rétention sur des images proxy
  extérieures, ce sont des références de banc, pas des candidats à la soumission. **La V2 est la
  meilleure méthode conforme mesurée.**
- **Face aux références** : la V2 (0,083 ± 0,004) rejoint `scrub` dosé (0,098 ± 0,022) : l'écart
  est dans le bruit de `scrub`, et la pire graine de la V2 (0,080) dépasse celle de `scrub` (0,074).
  La V2 est cinq fois plus stable. Son défaut reste la localité (ratio 1,37, retain 0,76).
- **Suite** : V2 + ancre de rétention de `scrub` (le défaut commun à V1 et V2) ; run final sans
  fold d'audit du réglage retenu si l'équipe décide de soumettre.
