# trust region anchored (V4, ancre croisee)

- Identifiant : `trust-region-anchored` (spec : `trust-region-anchored.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22 ; revue a la fusion avec la campagne V3 (meme jour, avant tout run)
- Responsable : Alex

## Hypothese

Les trois versions de `trust_region` oublient de facon generique : le fold d'audit (images
memorisees jamais entrainees) bouge 0,80 a 0,96 fois autant que les images exactes entrainees sur
les 31 runs V1 testes a chaque pas, 0,86 pour la V2 (`ref4-v2`, 3 graines), 0,86 a 0,91 pour la V3
retenue (3 graines), contre 0,51 a 0,60 pour `scrub` (`ref4-scrub`). Changer la cible a ameliore la
direction (V2, V3 : ~0,08 contre 0,064), pas ce mecanisme (rapport, « Diagnostic : un oubli
generique »).

La V4 tire du forget set un proxy de retention (`methods.py`, `anchor_folds`, `_trust_phase`,
`_held_outputs`) : les images entrainees sont coupees en K plis stratifies, oublies l'un apres
l'autre (chacun jusqu'a son rayon, 0,9 x KL de cible sur sa sonde) ; chaque pas ajoute un lot
d'ancre tire des autres plis, tenu aux sorties de l'original pour les plis jamais pousses et a
l'etat du debut de la phase pour les plis deja oublies (T^2 KL a T = 2, poids `anchor_weight`).
Un mouvement commun a toutes les images memorisees deplace les plis tenus, et l'ancre le fait payer.

**Base et reference.** La V3 retenue (`ref4-v3` : b = 0, lambda 0,25, rho 0,9, aire 0,5, 4 vues,
sans copies douces), graine 0. Reference appariee deja mesuree : `tr3-hedge` v1 (score_min 0,0784,
retain 0,758, accord 0,943, audit/exact 0,857, `bench.audit_kl` 0,243), rejouable par `ref4-v3` v1
(le code fusionne redonne la V3 au bit pres).

**Portee du gain, a lire avant tout resultat.** L'ancre vise le retain. A la graine 0, la V3 fait
oubli 0,103 x retain 0,758 ; une derive d'audit nulle donnerait au plus 0,097 (+24 %), au-dessus du
seuil d'amelioration (0,086) et au niveau de `scrub` dose. Un gain au-dela doit venir du terme
d'oubli : le verifier sur `bench.mean_std` et `bench.prediction`. Et l'ancre tient exactement la
population que mesure le retain du banc (images memorisees non entrainees) : un gain sur
`bench.retain` est en partie mecanique ; il ne vaut pour le score officiel que si le control set
est fait d'images d'entrainement memorisees, ce que le sujet ne precise pas.

Bras (graine 0 ; base : K = 4, poids 1, ancre exacte) :
- `anchor_views=0` : la base ;
- `anchor_folds=2` : K = 2 (la derniere phase n'a que des plis deja oublies dans son ancre) ;
- `anchor_views=1` : ancre sur vues fortes (aire 25-35 %, puis couleur) ;
- `anchor_weight=3` : l'ancre est-elle saturee ?

**Mecanisme** : audit/exact (`bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`) du bras
`anchor_views=0` <= 0,43 (moitie de 0,857), a KL exacte egale a +-20 % de la reference (0,284).

**Refutee (on arrete la V4, sans reglage supplementaire)** si :
- audit/exact > 0,60 (0,7 x la reference) dans les bras exacts (`anchor_views=0`,
  `anchor_folds=2`, `anchor_weight=3`) ;
- ou KL exacte < 0,7 x la reference dans tous les bras ancres (l'ancre empeche d'oublier) ;
- ou `bench.score_min` de chaque bras ancre <= 0,0784 + 2 sigma = 0,0844 (sigma = 0,003, ETAT.md).

Un bras au-dessus de 0,0844 est un candidat, pas une amelioration : il se confirme sur les graines
1 et 2 (seuil de la fiche ETAT : moyenne > 0,086 et gain sous le masquage au-dela de 2 sigma).

**Bras vues** : a retenir si `bench.audit_view_kl` <= 0,7 x celui du bras exact, sans que
`bench.audit_kl` depasse 1,2 x le sien. Les vues ne recouvrent aucune aire du banc, mais la couleur
est proche du jitter : trancher sur `bench.score_min_family.mask`.

**Auto-audit** : `final.pending_kl_p1..p3` est la derive (KL a T = 2 vers l'original) des plis
jamais pousses a la fin de chaque phase, mesuree par le run lui-meme ; `final.pending_kl` est celle
de la derniere phase qui en a. Si elle suit `bench.audit_kl` (meme ordre de grandeur, meme
classement des bras), elle servira a regler le run final, qui n'a pas de fold d'audit.

Seuils fixes a priori, le 2026-11-22, avant tout run de la V4.

## Reglages

Ceux de la V3 retenue, plus : un lot d'ancre de la taille du lot pousse, tire avec remise, nouvel
optimiseur, nouveau calendrier et nouveau flux de tirages a chaque phase. Cout : une passe
avant-arriere et une passe avant sans gradient de plus par pas ; +344 Mo (original) et +344 Mo
(copie figee, a partir de la phase 2).

## Lancement

```bash
make hk-run SPEC=trust-region-anchored                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=trust-region-anchored WHERE=jureca    # 4 variantes sur 4 A100 (file d'equipe)
```

Sur un autre GPU que la RTX 5080, rejouer d'abord `ref4-v3` sur la meme machine (ETAT.md §2).
Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Avant cette campagne : aucun run. Un essai intermediaire (M2, 320 images, version anterieure de
l'ancre) s'etait arrete sur une erreur propre au GPU d'Apple (pas de float64, corrigee).

Quatre runs, RTX 5080, `bench-v4`, graine 0, commit 52fa1d3, code propre (groupe
`20261122T015556Z-unl-trust-region-anchored`, 16 min). Reference appariee : V3 retenue (`ref4-v3`
v1 = `tr3-hedge` v1 au bit pres : 0,0784, KL exacte 0,284, audit/exact 0,857).

| bras | score_min | pire recadrage | pire masquage | retain | KL exacte (x V3) | KL audit | audit/exact | KL audit vue | pending_kl | phases au rayon |
|---|---|---|---|---|---|---|---|---|---|---|
| base K = 4, w 1, exacte | 0,0468 | 0,0589 | 0,0468 | 0,770 | 0,270 (0,95) | 0,264 | 0,98 | 0,430 | 0,078 | 0 / 4 |
| `anchor_folds=2` | 0,0454 | 0,0568 | 0,0454 | 0,797 | 0,239 (0,84) | 0,229 | 0,96 | 0,399 | 0,043 | 0 / 2 |
| `anchor_views=1` | 0,0553 | 0,0718 | 0,0553 | 0,736 | 0,367 (1,29) | 0,320 | 0,87 | 0,276 | 0,201 | 0 / 4 |
| `anchor_weight=3` | 0,0350 | 0,0437 | 0,0350 | 0,851 | 0,165 (0,58) | 0,161 | 0,97 | 0,310 | 0,032 | 0 / 4 |

**Aucune phase n'atteint son rayon** (`final.radius_reached` = 0, `final.reached_p*` = 0) : la KL de
la sonde a T = 1 finit chaque phase entre 0,02 et 0,07 (rayons 0,14-0,18), chaque phase parcourt ses 10 epoques
(220 pas a K = 4). L'ancre freine la montee pendant le warmup court de chaque phase (22 pas), puis,
comme dans `trust-region-dual`, l'objectif poursuivi se pose sous le rayon a T = 1. Signature commune
avec `dual` (V3 sans ancre, calendrier complet) : composante des pertes 0,08 a 0,23, composante des
predictions 0,33 a 0,68.

Lecture contre les seuils poses avant :

- **Mecanisme non confirme** : audit/exact du bras de base 0,98 (cible <= 0,43), a KL exacte 0,95 x
  la reference (dans les +-20 %).
- **Refutation 1 atteinte** : audit/exact > 0,60 dans les trois bras exacts (0,98 ; 0,96 ; 0,97).
- Refutation 2 non atteinte : seul `anchor_weight=3` freine l'oubli sous 0,7 x la reference (0,58).
- **Refutation 3 atteinte** : aucun bras au-dessus de 0,0844 (0,035 a 0,055).
- `anchor_weight=3` : l'ancre n'est pas saturee a w = 1, mais tripler son poids baisse d'autant la KL
  exacte et la KL d'audit (audit/exact inchange, 0,97) : elle freine l'oubli partout, sans le localiser.
- **Bras vues** : `bench.audit_view_kl` 0,276 = 0,64 x celui du bras exact (seuil 0,7, atteint), mais
  `bench.audit_kl` 1,21 x le sien (seuil 1,2, depasse) : non retenu.
- **Auto-audit** : `final.pending_kl` classe les quatre bras exactement comme `bench.audit_kl`
  (vues > base > K = 2 > w = 3), mais le sous-estime d'un facteur 1,6 a 5,3 (il est mesure a la fin de
  l'avant-derniere phase, et sur des images que l'ancre tient, alors que le fold d'audit n'est tenu par
  rien) : utilisable pour classer des reglages, pas pour mesurer une derive.

**Reserve, qui motive le suivi** : les seuils ont ete poses pour des phases arretees a leur rayon, au
regime de la V3 (arret en plein vol, au taux maximal). Ici aucune phase ne s'arrete ; or `dual` montre
que ce regime de calendrier complet donne a lui seul audit/exact 0,98-0,99 et le meme effondrement de
la composante des pertes, sans ancre. La specification a pose avant tout run le remede a ce cas :
« si `final.radius_reached` = 0 dans un bras, monter lr (3e-5), pas le nombre d'epoques ».

## Suivi : `tr4-anchored-lr` (pose le 2026-11-22, apres les quatre runs, avant le suivi)

**Question** : arretees a leur rayon (le regime pour lequel l'ancre a ete concue), les phases
ancrees localisent-elles l'oubli ? Remede prevu par la specification : lr 3e-5, rien d'autre ne change.

**Bras** (graine 0, lr 3e-5) : base (K = 4, w 1, ancre exacte) ; `anchor_folds=2` ; `anchor_weight=3` ;
et un temoin sans ancre (`anchor_folds=0` : la V3 retenue a lr 3e-5), pour separer l'effet de l'ancre
de celui du taux. Le bras vues n'est pas repris (quatre variantes au plus ; il ne fait que deplacer la
derive vers le fold exact, et a echoue a son propre critere).

**Seuils** (ceux de la fiche, plus le temoin) :
- prealable : `final.radius_reached` = 1 dans un bras ancre ; sinon le remede est epuise (pas
  d'autre lr, pas plus d'epoques) et la refutation ci-dessus vaut telle quelle ;
- **mecanisme confirme** si le bras de base atteint son rayon avec audit/exact <= 0,43, a KL exacte
  dans +-20 % de la reference (0,284), et <= 0,5 x l'audit/exact du temoin ;
- **refutee** (arret de l'ancre) si audit/exact > 0,60 dans les bras exacts qui atteignent leur rayon,
  ou si leur `bench.score_min` <= 0,0844 ;
- un bras > 0,0844 est un candidat : graines 1 et 2 (seuil ETAT.md).

### Resultat du suivi (groupe `20261122T021405Z-unl-tr4-anchored-lr`, code propre, 12 min)

| bras (lr 3e-5) | score_min | pire recadrage | pire masquage | retain | KL exacte (x V3) | KL audit | audit/exact | pending_kl | phases au rayon |
|---|---|---|---|---|---|---|---|---|---|
| base K = 4, w 1, exacte | 0,0583 | 0,1069 | 0,0583 | 0,790 | 0,344 (1,21) | 0,240 | 0,70 | 0,024 | 0 / 4 (0,82-0,89 x rayon) |
| `anchor_folds=2` | 0,0597 | 0,1097 | 0,0597 | 0,802 | 0,335 (1,18) | 0,225 | 0,67 | 0,019 | 1 / 2 |
| `anchor_weight=3` | 0,0703 | 0,1031 | 0,0703 | 0,820 | 0,293 (1,03) | 0,203 | 0,69 | 0,016 | 0 / 4 (0,54-0,89 x rayon) |
| temoin sans ancre (V3, lr 3e-5) | **0,0876** | 0,0904 | 0,0876 | 0,730 | 0,334 (1,18) | 0,287 | 0,86 | - | arret au pas 63 |

- **Prealable non rempli** : a lr 3e-5, les phases s'approchent de leur rayon (0,82 a 0,89 fois en fin
  de phase a w = 1 ; 0,54 a 0,89 a w = 3) sans l'atteindre, sauf la premiere a K = 2. Selon la regle posee avant, le remede est
  epuise et la refutation des quatre premiers runs vaut telle quelle.
- Lue quand meme, la mesure confirme la refutation : audit/exact 0,67 a 0,70 (> 0,60 ; 0,78 a 0,81 x le
  temoin, loin de 0,5), et aucun bras ancre au-dessus de 0,0844 (0,058 a 0,070).
- **Ce que fait l'ancre a regime egal** (contre le temoin au meme lr) : -20 % sur audit/exact, et KL
  d'audit -16 a -29 % ; mais le gain ne vaut que sous le recadrage (+0,013 a +0,019 sur le temoin)
  et se paie sous le masquage (-0,017 a -0,029) : le pire `bench.prediction.*` tombe a 0,23-0,27 sous
  m15. Un gain qui ne tient que sous la famille des vues est celui que la fiche juge suspect.
- Auto-audit : `pending_kl` (0,016 a 0,024) classe encore les bras comme `bench.audit_kl`
  (base > K = 2 > w = 3), et le sous-estime d'un facteur 10 a 13.
- **Question ouverte par le temoin** : la V3 retenue a lr 3e-5 vaut 0,0876 a la graine 0, +0,0092 sur
  la meme V3 a lr 1e-5 (3 sigma), sous les deux familles. Hors du perimetre de l'ancre : suivi pose
  dans `trust-region-v4-campaign.md` (« Suivi hors V4 »).

## Decision

- [x] Abandonner l'ancre croisee : refutee a lr 1e-5 et a lr 3e-5 par les criteres poses avant
  (audit/exact > 0,60 dans les bras exacts, aucun bras au-dessus de 0,0844). Elle freine l'oubli
  presque autant sur le fold d'audit que sur les images poussees ; la part qu'elle localise (-20 %
  sur audit/exact, a lr 3e-5) se paie sous le masquage. L'auto-audit `pending_kl` classe les
  reglages comme le fold d'audit mais en sous-estime la derive d'un facteur 1,6 a 13.
- [ ] Iterer / [ ] Candidate a la soumission
