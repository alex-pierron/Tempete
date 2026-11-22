# Campagne trust_region à pleine échelle (RTX 5080)

- Spécifications : `experiments/unlearning/tr-*.toml` (une par balayage, 4 variantes au plus)
- Tâche : `unlearning`
- Créée le : 2026-11-21
- Responsable : Alex
- Tableau à jour : `make hk-unl-scores` → `experiments/unlearning/SCOREBOARD.md`

## Point de départ (mesuré avant la campagne)

Runs `bench-baselines`, `trust-region-kl-radius`, `bench-seed-noise` (pleine échelle : 2800
images entraînées, fold d'audit de 700, banc `bench-v3`) :

- références : `scrub` score_min = 0,0027 (retain 0,998, ratio 5,3, il bouge à peine) ;
  `gold_soft` 0 (retain 0,40, ratio 1,1) ; identité 0 par construction ;
- `trust_region` testé tous les 10 pas : ε = 0,2 / 0,4 / 0,6 s'arrêtent au **même pas** (90) et
  donnent le même modèle (score_min 0 ; s40 0,085) ; la KL passe de 0,07 à 0,65 en 10 pas à la
  fin du warmup. ε = 0,8 et les graines 1-2 s'arrêtent au pas 100 : KL ~1,8, porte cassée.
  Le « bruit de graine » mesuré là est surtout l'effet du pas d'arrêt.

D'où la campagne : rendre l'arrêt fin (`radius_check_every=1`), puis balayer les réglages un à
un autour du meilleur rayon, puis confirmer le meilleur sur plusieurs graines.

## Règles de décision (fixées avant les runs)

- Métrique : `bench.score_min` (pire des trois doublures du gold), moyenne sur les graines
  quand il y en a plusieurs ; `bench.gate_passed` = 1 exigé.
- Une différence ne compte que si elle dépasse l'écart-type entre graines du même réglage
  (étape 6). Avant l'étape 6, les classements sont provisoires.
- L'idée « trust_region » est jugée **mauvaise** si aucun réglage ne dépasse `scrub` (0,0027)
  au-delà du bruit ; **prometteuse** si un réglage le dépasse nettement sous les trois doublures.

## Étapes, hypothèses et critères de réfutation

Base commune : `target_temperature=6`, `lr=1e-5`, lots de 32, `epochs=10` (l'arrêt au rayon coupe
avant), `radius_check_every=1`, `holdout_fraction=0.2`, graine 0.

1. **Rayon** (`tr-radius-fine-small`, `tr-radius-fine-large`) : ε ∈ {0,01 ; 0,02 ; 0,05 ; 0,1 ;
   0,2 ; 0,3 ; 0,5 ; 0,8}.
   - H1 : la KL finale suit ε (`final.forget_kl` / ε entre 1 et 1,5). Réfutée si l'écart dépasse
     50 % pour la plupart des ε : l'arrêt ne contrôle toujours rien.
   - H2 : un optimum ε* existe, petit (s60 demande peu de mouvement), avec score_min > 0,0027.
     Réfutée si score_min ≤ 0,0027 pour tous les ε.
   - H3 : à petit ε, l'oubli reste localisé (ratio ≥ 2). Réfutée si ratio < 2 partout.
2. **Taux d'apprentissage** (`tr-lr`) : lr ∈ {3e-6 ; 3e-5 ; 1e-4} à ε*.
   - Hypothèse : un taux plus bas approche ε plus doucement et localise mieux (ratio plus haut à
     KL égale). Réfutée si ratio et score_min ne bougent pas au-delà du bruit.
3. **Température de la cible** (`tr-temperature`) : τ ∈ {2 ; 3 ; 4 ; 10} à ε*.
   - Hypothèse : τ fixe la direction du mouvement, donc le `prediction_score`. Réfutée si
     score_min est plat en τ.
4. **Blocs entraînés** (`tr-blocks`) : `trainable_blocks` ∈ {0 ; 2 ; 6} à ε* (0 = tête et normes).
   - Hypothèse : n'entraîner que la fin du réseau réduit la dérive sur le fold d'audit (ratio
     plus haut). Réfutée si le ratio ne monte pas.
5. **Taille de lot** (`tr-batch`) : lot ∈ {8 ; 16 ; 64} à ε*.
   - Hypothèse : à KL égale, la taille de lot change peu le résultat. Réfutée si score_min
     varie au-delà du bruit.
6. **Confirmation** : le meilleur réglage combiné, et `scrub` en référence, sur 3 graines.

## Résultats

(chiffres complets dans `SCOREBOARD.md`)

### Étape 1 : rayon (8 runs, graine 0)

| ε | KL finale / ε | pas d'arrêt | score_min | s40 | s50 | s60 | retain | ratio |
|---|---|---|---|---|---|---|---|---|
| 0,01 | 1,92 | 72 | 0,0236 | 0,0236 | 0,0435 | 0,0257 | 0,961 | 1,8 |
| 0,02 | 1,53 | 75 | 0,0328 | 0,0328 | 0,0514 | 0,0354 | 0,939 | 1,7 |
| 0,05 | 1,24 | 79 | 0,0511 | 0,0511 | 0,0564 | 0,0525 | 0,892 | 1,5 |
| 0,1 | 1,12 | 82 | **0,0594** | 0,0700 | 0,0594 | 0,0643 | 0,830 | 1,4 |
| 0,2 | 1,07 | 85 | **0,0595** | 0,0866 | 0,0726 | 0,0595 | 0,739 | 1,3 |
| 0,3 | 1,11 | 87 | 0,0338 | 0,0919 | 0,0727 | 0,0338 | 0,665 | 1,3 |
| 0,5 | 1,04 | 89 | 0,0018 | 0,0887 | 0,0500 | 0,0018 | 0,568 | 1,2 |
| 0,8 | 1,18 | 92 | 0 | 0,0660 | 0,0008 | 0 | 0,421 | 1,2 |

- H1 **tenue** : la KL finale suit ε (1,04 à 1,24 × ε dès ε ≥ 0,05 ; le dépassement d'un pas
  pèse plus sur les tout petits rayons).
- H2 **tenue** (graine 0) : optimum à ε ≈ 0,1-0,2, score_min ≈ 0,059, soit ~22 fois `scrub`.
  À confirmer contre le bruit (graines).
- H3 **réfutée** : ratio 1,2 à 1,8, jamais ≥ 2. L'oubli déborde sur le fold d'audit à tous les
  budgets ; le score reste positif parce que le gain sur l'oubli dépasse la perte de retain.
- Constat : tous les arrêts tombent entre les pas 72 et 92, **pendant le warmup** (88 pas avec
  `epochs=10`). La KL passe de 0,02 à 0,94 en 20 pas : la pente du taux pendant le warmup est
  un réglage caché, d'où l'étape 2 bis.
- Compromis lisible sur s60 : `prediction_score` culmine vers ε = 0,05 puis baisse,
  `mean_std_score` monte jusqu'à ε = 0,5 ; le retain baisse régulièrement.

### Choix pour les étapes 2 à 5 (fixé avant de les lancer)

- Point d'ancrage ε* = 0,15, entre les deux meilleurs rayons (0,1 et 0,2, à égalité).
- Bruit : graines 0, 1, 2 à ε* (`tr-seeds`), et `scrub` sur les graines 1 et 2 (`scrub-seeds`).
- Étape 3 modifiée : τ ∈ {4 ; 5 ; 8 ; 12}, car τ = 2 et 3 placent la cible sous ε* (KL
  atteignable mesurée sur les logits de l'original : 0,014 et 0,120 ; 0,40 à τ = 4, 1,22 à τ = 6).
- Étape 2 bis, **warmup** (`tr-warmup`) : `warmup_fraction` ∈ {0 ; 0,03 ; 0,3} à ε*.
  Hypothèse : approcher ε avec un taux plus bas (warmup long) localise mieux ; réfutée si ratio et
  score_min ne bougent pas au-delà du bruit.

### Étapes 2 à 5 et bruit (ε* = 0,15 ; 21 runs)

Bruit : graines 0, 1, 2 à ε* → score_min **0,0637 ± 0,0025** (0,0647 ; 0,0609 ; 0,0656).
`scrub` sur 3 graines : **0,0025 ± 0,0003**.

| balayage | valeurs → score_min | lecture |
|---|---|---|
| taux `lr` (pas d'arrêt) | 3e-6 → 0,064 (170) ; 1e-5 → 0,064 ± 0,002 (~84) ; 3e-5 → 0,067 (50) ; 1e-4 → 0,062 (29) | dans le bruit ; à 1e-4 l'arrêt dépasse ε (KL 0,22) |
| `warmup_fraction` | 0 → 0,065 ; 0,03 → 0,066 ; 0,1 → 0,064 ; 0,3 → 0,065 | plat |
| température τ | 4 → 0,065 ; 5 → 0,065 ; 6 → 0,064 ; 8 → 0,065 ; 12 → 0,065 | plat |
| `trainable_blocks` | 0 → 0,0006 (rayon jamais atteint) ; 2 → 0,025 ; 6 → 0,025 ; tous → 0,064 | **tout le réseau** : KL audit 0,20 contre 0,44-0,51 avec 2 ou 6 blocs |
| taille de lot | 8 → 0,063 ; 16 → 0,066 ; 32 → 0,064 ; 64 → 0,067 | plat |

- Taux (étape 2) et warmup (2 bis) : hypothèse de localisation **réfutée** (ratio 1,30 à 1,39,
  score dans le bruit).
- Température (étape 3) : **réfutée**, score plat de τ = 4 à 12. À KL égale, la direction de
  la cible ne change rien de mesurable.
- Blocs (étape 4) : **réfutée, et à l'envers** : restreindre l'entraînement à la fin du réseau
  déplace davantage le fold d'audit (ratio 1,19-1,28 contre 1,36).
- Lot (étape 5) : **tenue**, plat.
- Bilan : à pleine échelle, le score de `trust_region` ne dépend que de ε (et d'entraîner tout le
  réseau). Le reste se règle pour la vitesse : lr 3e-5 atteint ε* en 50 pas au lieu de ~84.

### Étape 6 : confirmation du rayon (fixée avant de lancer)

`tr-confirm-radius` : graines 1 et 2 pour ε = 0,1 et ε = 0,2 (la graine 0 est dans l'étape 1).
Hypothèse : ε = 0,1, 0,15 et 0,2 restent à moins d'un écart-type les uns des autres (plateau).
Réfutée si l'un d'eux s'écarte de plus de deux écarts-types.

### Étape 6 : résultats (3 graines par rayon)

| ε | score_min | s40 | s50 | s60 | retain | accord audit | KL audit | ratio | pas |
|---|---|---|---|---|---|---|---|---|---|
| 0,10 | 0,0609 ± 0,0029 | 0,071 | 0,061 | 0,064 | 0,825 | 0,964 | 0,169 | 1,42 | 83 |
| **0,15** | **0,0637 ± 0,0025** | 0,082 | 0,069 | 0,064 | 0,768 | 0,943 | 0,228 | 1,36 | 85 |
| 0,20 | 0,0583 ± 0,0061 | 0,087 | 0,073 | 0,058 | 0,739 | 0,934 | 0,266 | 1,33 | 86 |

Hypothèse du plateau **tenue** : les trois rayons restent à moins de deux écarts-types les uns des
autres ; ε = 0,15 a la meilleure moyenne et le plus petit écart. Transfert aux copies altérées : la
pire (recadrage) bouge 0,70 fois autant que les images exactes, le JPEG 1,3 fois.

### Étape 7 : références dosées (fixée avant de lancer)

Les références n'ont tourné qu'à leurs réglages par défaut : `gold_soft` va trop loin (KL 1,07,
retain 0,40), `scrub` ne bouge presque pas (KL 0,009). Pour juger l'idée contre une référence
réglée, pas contre un épouvantail :

- `gold-soft-dose` : `gold_soft`, 1 époque, lr ∈ {5e-7 ; 1e-6 ; 3e-6 ; 1e-5}.
- `scrub-dose` : `scrub`, lr ∈ {3e-5 ; 1e-4 ; 3e-4}.

Hypothèse : une référence dosée rejoint trust_region à ε* (0,0637 ± 0,0025). Réfutée (donc
avantage propre à trust_region) si aucune ne dépasse 0,0637 − 2 × 0,0025 = 0,059.

### Étapes 7 et 8 : références dosées, puis confirmées (13 runs)

Hypothèses de l'étape 8 écrites dans les specs (`scrub-dose-seeds`, `scrub-dose-fine`,
`gold-soft-dose-seeds`, champ `hypothesis`) avant leur lancement.

| référence | réglage | graines | score_min | s40 | s50 | s60 | retain | KL audit | ratio |
|---|---|---|---|---|---|---|---|---|---|
| `scrub` | défaut (lr 1e-5) | 3 | 0,0025 ± 0,0003 | 0,003 | 0,005 | 0,010 | 0,998 | 0,002 | 4,9 |
| `scrub` | lr 3e-5 | 1 | 0,014 | 0,014 | 0,023 | 0,025 | 0,995 | 0,005 | 5,5 |
| `scrub` | lr 5e-5 | 1 | 0,037 | 0,037 | 0,061 | 0,055 | 0,985 | 0,015 | 3,4 |
| `scrub` | **lr 1e-4** | 3 | **0,098 ± 0,022** | 0,098 | 0,147 | 0,136 | 0,931 | 0,067 | 2,0 |
| `scrub` | lr 2e-4 | 1 | 0,118 | 0,165 | 0,157 | 0,118 | 0,663 | 0,423 | 1,2 |
| `scrub` | lr 3e-4 | 1 | 0 | 0,106 | 0,033 | 0 | 0,405 | 0,960 | 1,1 |
| `gold_soft` | défaut (3 époques) | 1 | 0 | 0,073 | 0,014 | 0 | 0,401 | 1,022 | 1,1 |
| `gold_soft` | 1 époque, lr 5e-7 / 1e-6 / 3e-6 | 1 chacun | ≤ 0,001 | | | | ≥ 0,998 | | |
| `gold_soft` | **1 époque, lr 1e-5** | 3 | **0,059 ± 0,012** | 0,059 | 0,069 | 0,060 | 0,882 | 0,117 | 1,6 |
| `trust_region` | **ε = 0,15** | 3 | **0,0637 ± 0,0025** | 0,082 | 0,069 | 0,064 | 0,768 | 0,228 | 1,4 |

- Étape 7 : hypothèse **tenue**, et au-delà : `scrub` dosé dépasse trust_region. Les trois
  graines de `scrub` à lr 1e-4 (0,074 ; 0,102 ; 0,117) sont toutes au-dessus des trois de
  trust_region (0,061 ; 0,065 ; 0,066). `gold_soft` dosé fait jeu égal (0,059 ± 0,012).
- `scrub-dose-seeds` : **tenue** sur la moyenne (0,098 contre 0,064), mais `scrub` est bruité
  (écart-type 0,022, neuf fois celui de trust_region).
- `scrub-dose-fine` : **réfutée en partie** : 5e-5 tombe à 0,037 ; 2e-4 monte à 0,118 (une
  graine, retain 0,66) ; 3e-4 s'effondre (0). Pas un plateau large : une crête entre 1e-4 et 2e-4,
  avec une falaise juste après.
- `gold-soft-dose-seeds` : **tenue** (au niveau de trust_region, plus bruité).
- Mécanisme lisible dans les colonnes : à divergence d'oubli comparable (KL sur les images
  exactes 0,12 pour `scrub`, 0,14 pour `gold_soft`, 0,20 pour trust_region à ε = 0,1), `scrub`
  déplace deux à trois fois moins le fold d'audit (KL 0,067 contre 0,117 et 0,169) : son pas de
  rétention sur les images proxy protège mieux les images voisines.

## Décision

- **trust_region : pas abandonnée, mais pas la meilleure.** Contre les références à leurs
  réglages par défaut, elle gagne d'un facteur 25 ; contre des références dosées, `scrub` la
  dépasse d'environ 50 % en moyenne, `gold_soft` fait jeu égal. Ses atouts mesurés : un seul
  réglage utile (ε, plateau entre 0,1 et 0,2), une reproductibilité nettement meilleure (écart-type
  0,0025 contre 0,022 pour `scrub`) et une insensibilité au taux, au warmup, à la température et
  au lot. Son défaut mesuré : la localité (ratio 1,4, KL sur le fold d'audit 0,23). Relu le
  22/11 : le ratio moyenne images exactes et copies, et le jitter le gonfle ; sur les images
  exactes seules, le fold d'audit bouge 0,80 à 0,96 fois autant (ratio ~1,14), dans les 31 runs
  testés à chaque pas (rapport, « Diagnostic : un oubli générique »).
- **Réglage à garder si on la soumet** : ε = 0,15, τ = 6, `radius_check_every=1`, tout le réseau
  entraîné, lr 3e-5 pour aller plus vite (même score, 50 pas au lieu de ~85).
- **Itérer** (étape suivante proposée, non lancée) : ajouter à trust_region l'ancre de rétention
  de `scrub` (KL vers l'original sur les images proxy), qui est ce qui distingue `scrub` dans les
  mesures ; et, en parallèle, stabiliser `scrub` entre lr 1e-4 et 2e-4 sur plus de graines.
- Rien n'a été soumis ; tous ces fichiers ont un fold d'audit et ne doivent pas l'être.
