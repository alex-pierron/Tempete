# trust region views

- Identifiant : `trust-region-views` (spec : `trust-region-views.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-21
- Responsable : Alex

## Hypothese

V2 de `trust-region-kl-radius`. La V1 tire l'eleve vers l'original uniformement adouci,
`softmax(z0/6)` : toutes les images sont adoucies pareil, alors que le gold est heterogene
(quelques images fausses, vers une classe precise). Une perte moyenne juste ne donne alors ni le
bon ecart-type (`mean_std_score`) ni la bonne classe fausse image par image (`prediction_score`).
La V2 garde tout (region de confiance, arret au rayon, lots stratifies, fold d'audit) sauf la
cible : pour chaque image, la moyenne sur 3 vues fortes (`strong_view`, aire 0,45-0,55) de
`softmax(z_orig(vue)/4)`, calculee une fois avant l'entrainement, sans gradient
(`trust_target=views`). La cible donne la direction et l'heterogeneite, epsilon la quantite.

Rayons identiques a la V1 (epsilon in {0,2 ; 0,4 ; 0,6 ; 0,8}) et cible a distance comparable
(KL 1,08 a tau = 4 contre 1,22 pour la V1 a tau = 6, mesure ci-dessous) : une variante V2 et la
variante V1 de meme epsilon s'arretent au meme budget, a une fraction voisine du chemin vers leur
cible (74 % contre 66 % a epsilon = 0,8). Ce qui change entre elles est la cible (vues, tau).

On croit que : a epsilon egal, la V2 bat la V1 sur `bench.score_min`, surtout par
`bench.mean_std` (ecart-type des pertes plus proche de la doublure), sans casse supplementaire sur
le fold d'audit.

**Biais, a lire avant tout resultat** : la doublure du gold du banc est AUSSI l'original sur des
vues fortes (aires 0,40 / 0,50 / 0,60, 4 vues, graines derivees de `AUDIT_SEED` = 31). La V2 en
est proche par construction : ses vues sont tirees d'une autre graine (`seed` + 1 000 003) et a
une autre aire (0,45-0,55 contre 0,40-0,50 / 0,50-0,60 / 0,60-0,70), mais du meme generateur. Un
avantage local de la V2 sur la V1 est donc en partie mecanique ; il ne prouve pas un gain au score
reel. Seul un ecart qui tient sous la PIRE doublure (`score_min`) et au-dela du bruit compte.

**Refutee si**, pour les epsilon atteints par les deux versions :
- `bench.score_min` (V2) - `bench.score_min` (V1 de meme epsilon) <= 2 ecarts-types du plancher
  de bruit de `bench-seed-noise`, pour au moins 3 des 4 epsilon : pas mieux que la V1 ;
- ou `bench.score_min` (V2) sous la meilleure reference de `bench-baselines` (gold_soft, scrub)
  au meilleur epsilon : la V2 ne vaut pas mieux que les methodes existantes ;
- ou `bench.audit_kl` (V2) > 1,5 x `bench.audit_kl` (V1 de meme epsilon) : la cible par vues
  casse davantage les images voisines ;
- ou `bench.groups.forget_flip|crop|jpeg.kl_orig_sub` < 0,5 x `forget_exact.kl_orig_sub` (meme
  seuil que la V1) : l'oubli ne se transfere pas aux copies alterees.

Mecanisme refute (meme si le chiffre monte) si `bench.mean_std` (V2) <= `bench.mean_std` (V1 de
meme epsilon) sous au moins 2 des 3 doublures : le gain ne viendrait pas de l'heterogeneite des
pertes.

Seuils fixes a priori, le 2026-11-21, avant tout run complet (seul le smoke a tourne).
Revision du meme jour, toujours avant tout run complet : les runs a venir sont notes en
`bench-v4` (seconde famille de doublures, masquage de patchs ; `bench.score_min` = pire cas sur
les 6). Les criteres portant sur « les 3 doublures » se lisent sur les recadrages (s40, s50,
s60) ; sur les masquages (m30, m20, m15), que la cible par vues ne copie pas, on lit en plus
`bench.score_min_family.mask` : c'est la part non mecanique d'un gain.

## Choix de tau et d'epsilon (mesure sur le vrai modele)

KL(original || cible) a T = 1, moyenne sur 256 images du forget set (`select_subset` graine 0,
Apple M2, 3 vues, graine de vues 0 sauf mention), avec l'accord de l'argmax de la cible avec
celui de l'original et la perte de la cible sur ce pseudo-label (moyenne, ecart-type) :

| cible | KL | accord argmax | perte moy. | perte e.-t. |
|---|---|---|---|---|
| V1 tempered, tau = 6 | 1,218 | 1,000 | 1,222 | 0,345 |
| V2 vues, aire 0,45, tau = 1 | 0,257 | 0,945 | 0,256 | 0,319 |
| V2 vues, aire 0,45, tau = 1, graine 1 | 0,258 | 0,941 | 0,258 | 0,396 |
| V2 vues, aire 0,45, tau = 1,5 | 0,298 | 0,953 | 0,299 | 0,317 |
| V2 vues, aire 0,45, tau = 2 | 0,375 | 0,957 | 0,377 | 0,332 |
| V2 vues, aire 0,45, tau = 3 | 0,661 | 0,965 | 0,663 | 0,399 |
| **V2 vues, aire 0,45, tau = 4 (retenue)** | **1,079** | **0,965** | **1,082** | **0,455** |
| V2 vues, aire 0,45, tau = 6 | 1,911 | 0,965 | 1,914 | 0,436 |
| V2 vues, aire 0,45, tau = 6, 2 vues (smoke) | 1,920 | 0,941 | 1,923 | 0,489 |
| V2 vues, aire 0,55, tau = 1 | 0,147 | 0,980 | 0,146 | 0,234 |
| V2 vues, aire 0,55, tau = 1,5 | 0,172 | 0,984 | 0,173 | 0,228 |
| V2 vues, aire 0,55, tau = 2 | 0,229 | 0,984 | 0,230 | 0,239 |

Script jetable (scratchpad de la session, non versionne) ; il appelle `trust_target_logits`, le
code de la methode. ~20 s par cible de 256 images sur M2.

Lecture : a tau = 1, la cible par vues est bien plus pres de l'original qu'attendu (0,26 a l'aire
0,45, contre 0,5-0,8 extrapole de la perte par vue) : moyenner 3 vues remonte la probabilite du
pseudo-label (-log de la moyenne < moyenne des -log). Tous les epsilon de la V1 au-dessus de 0,2
seraient infaisables (le controle de faisabilite leve une `ConfigError`). On monte donc tau :

- tau = 4 : KL 1,08, au-dessus des 4 rayons (le plus grand, 0,8, a 74 % du chemin ; la marge
  0,28 depasse de loin l'erreur d'echantillonnage, ~0,03 = e.-t. 0,455 / racine de 256) ;
- tau = 6 (celui de la V1) ecarte, mesure a l'appui : l'heterogeneite relative de la cible
  (e.-t. / moyenne des pertes) tombe a 0,23, sous celle de la V1 (0,28) ; elle vaut 0,42 a
  tau = 4. Trop adoucie, la cible par vues redevient une cible uniforme ;
- tau = 3 (KL 0,66) laisserait epsilon = 0,6 a 91 % du chemin, approche asymptotique : rayon
  probablement pas atteint en 10 epoques.

Aire 0,45 plutot que 0,55 : a 0,55, la cible est juste sur 98 % des images (moins d'images
fausses que le gold plausible, 78-92 %, voir unlearning.md par. 8) ; a 0,45, 96,5 % a tau = 4.
Limite : meme a 0,45, la cible se trompe moins souvent que le gold plausible (3 vues moyennees se
trompent moins qu'une seule). L'aire et le nombre de vues sont les leviers de la meta-boucle a
venir (force de vue).

## Metrique locale attendue

- `final.target_kl` ~1,08 et `final.target_agreement` ~0,965 sur le forget set complet (a
  comparer aux 256 images ci-dessus ; un ecart > 0,1 sur la KL signale un echantillon atypique).
- `final.radius_reached` = 1 pour epsilon <= 0,6 en 10 epoques, comme la V1.
- `bench.mean_std` (V2) > `bench.mean_std` (V1) a epsilon egal sous les 3 doublures ;
  `bench.prediction` au moins egal.
- `bench.gate_passed` = 1, `bench.movement_ratio` >= 5.

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=trust-region-views                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=trust-region-views WHERE=jureca    # 4 epsilon sur 4 A100 (file d'equipe)
```

A lancer avec (ou apres) `trust-region-kl-radius`, `bench-baselines` et `bench-seed-noise`, qui
donnent les points de comparaison. Ces runs ont un holdout : ne pas soumettre leurs fichiers.

**Avertissement issu de la campagne V1 a pleine echelle (RTX 5080, 2026-11-21, avant tout run
complet de la V2 ; `trust-region-campaign.md`).** Avec le rayon teste tous les 10 pas (le defaut,
que cette spec garde), la V1 a donne le meme modele pour epsilon = 0,2, 0,4 et 0,6 : la KL passe de
0,07 a 0,65 en 10 pas a la fin du warmup, et l'ecart entre graines venait du pas d'arret. Pour que
la grille de rayons de cette spec mesure quelque chose, la lancer avec `radius_check_every=1`
(ajoute aux `args`) ; et comparer a la V1 au meme reglage, dont le plateau est entre 0,1 et 0,2
(0,0637 +- 0,0025 a epsilon = 0,15 sur 3 graines). Les references dosees a battre sont aussi dans
la campagne : `scrub` lr 1e-4 (0,098 +- 0,022), `gold_soft` 1 epoque lr 1e-5 (0,059 +- 0,012).

**Revision a la fusion (2026-11-21, avant tout run complet de la V2), qui applique
l'avertissement ci-dessus.** La spec teste le rayon a chaque pas (`radius_check_every=1`) et ses
rayons deviennent {0,1 ; 0,15 ; 0,2 ; 0,3}, les points de la V1 mesures autour de son plateau.
Les runs de la V2 sont notes en `bench-v4` et la campagne en `bench-v3` : les criteres « a epsilon
egal » se lisent contre la V1 recalculee en `bench-v4` (meme reglage, `bench --candidate` sur ses
fichiers ou nouveau run, moins d'une minute chacun sur la RTX 5080), jamais contre les chiffres
`bench-v3` de la campagne. Les seuils de refutation ne changent pas.

Run(s) :
- smoke local via le lab `20261121T203807Z-unl-trust-region-views` (v1-v4, `--config auto` ->
  smoke sur M2/MPS, 52 images entrainees, 12 tenues a l'ecart, 2 vues, 40 pas a lr 1e-5, ~170 s
  d'entrainement par variante). Lignes de registre mises de cote dans
  `runs/unlearning/registry-smoke-20261121.jsonl`.

## Resultat

Smoke seulement (aucun run complet) : chaine validee de bout en bout. Cible sur les 52 images :
KL 0,938, accord 0,923 (2 vues). Rayon non atteint en 40 pas (KL 0,036, attendu en smoke), donc
les 4 variantes ont fait les memes pas et donnent le meme banc (`bench_smoke.*`, bench-v3) :

| candidat (40 pas, lr 1e-5) | score_min | s40 / s50 / s60 | mean_std s40 / s50 / s60 | prediction s40 / s50 / s60 | audit KL | ratio |
|---|---|---|---|---|---|---|
| V1 (`trust-region-kl-radius` v1, recalcule en bench-v3) | 0,0287 | 0,029 / 0,059 / 0,048 | 0,096 / 0,177 / 0,186 | 0,299 / 0,332 / 0,260 | 0,0024 | 41 |
| V2 (ce smoke) | 0,0289 | 0,029 / 0,058 / 0,051 | 0,091 / 0,169 / 0,178 | 0,319 / 0,347 / 0,288 | 0,0018 | 52 |

Egalite sur `score_min` a cette echelle (une graine, 52 images, plancher de bruit non mesure).
Signal a surveiller : le petit avantage de la V2 vient de `prediction`, et `mean_std` est un peu
plus bas sous les 3 doublures, l'inverse du mecanisme annonce (critere de mecanisme ci-dessus, a
juger sur les runs complets au rayon atteint). V1 recalculee par
`python -m hackathon.unlearning bench --candidate <v1>/candidate.pt --forget-subset 64`.

## Decision

- [ ] Abandonner / [x] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

La version d'origine de cette spec (rayon teste tous les 10 pas, epsilon 0,2 a 0,8) n'a pas ete
lancee. Sa revision a la fusion (tau = 4, epsilon 0,1 / 0,15 / 0,2 / 0,3, rayon teste a chaque pas)
correspond a l'etape 1 de la campagne `trust-region-v2-campaign.md` (`tr2-radius-a`, `-b`), deja
mesuree a pleine echelle en `bench-v3` : gain faible, +0,002 a +0,006 sur la V1 au meme rayon. La
campagne (30 runs, 2026-11-21) retient tau = 1 : la V2 y bat la V1 de 30 % (0,0831 +- 0,0042 contre
0,0637 +- 0,0025, 3 graines appariees). Contre l'hypothese de cette fiche, le gain passe par la
composante `prediction`, pas par `mean_std`. Relancer cette spec en `bench-v4` ne recalculerait
que le point tau = 4 ; le reglage a porter en `bench-v4` est celui de la campagne (tau = 1, aire
0,45, 3 vues, epsilon = 0,15).

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
