# V_Fusion anchor (SCRUB conforme ancre sur des images synthetiques)

- Identifiant : `v-fusion-anchor` (spec : `v-fusion-anchor.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22, apres l'etape 0 (`v-fusion-step0.md`), avant tout run de bras
- Responsable : Alex

## Hypothese

V_Fusion (rapport, section V_Fusion). SCRUB conforme a lr 4e-4 oublie davantage (0,1279 a la graine 0)
mais son retain tombe a 0,734, sous le plancher de 0,8 : son ancre de vues fortes du forget set
protege mal a cette dose. Le pari : une ancre d'images des bonnes classes, qui ne sont pas les images
oubliees, garde la fonction sans retenir ces images. Option `retain_source=synthetic` de `scrub`
(codee pour ce run) : les min-steps distillent l'original a T = 2 sur les images d'ancre augmentees
comme le proxy (`random_augment`), lots `synthetic_pools`.

Images : etape 0, lots A (540 images d'ancre, 36 classes du forget set) et B (825, 55 autres
classes) ; portes a, b, c, c' passees, d ininterpretable (temoin 0,921 < 0,95 meme a 1000 pas), donc
admis par la regle de l'etape 0. Reserve mesuree, posee avant ce run : les images sont saturees
(confiance 0,998 a 1, le plafond de 0,9 n'a pas joue) et reparent trois fois moins qu'une ancre de
vues (porte d : 17-21 % des degats repares, contre 60 % pour le temoin).

Bras (graine 0, fold d'audit de 20 %) : AB, A, B a lr 4e-4 ; AB a lr 3e-4.

References appariees (graine 0, `bench-v4`, RTX 5080, ancre de vues `forget_views`) :

| reference | score_min | recadr. | masq. | retain | KL exacte | audit/exact |
|---|---|---|---|---|---|---|
| SCRUB conforme lr 3e-4 (`scrub-fv-dose` v1) | 0,1151 | 0,1229 | 0,1151 | 0,868 | 0,165 | 0,80 |
| SCRUB conforme lr 4e-4 (`scrub-fv-dose` v3) | 0,1279 | 0,1396 | 0,1279 | 0,734 | 0,277 | 1,03 |

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run de bras ; sigma = 0,003) :
- aucun des trois bras a lr 4e-4 n'a `bench.retain` >= 0,8 : l'ancre synthetique ne tient pas le
  plancher a la dose visee ;
- ou le meilleur bras de retain >= 0,8 (les quatre bras comptent) a `bench.score_min` <= **0,121**
  (SCRUB conforme a 3e-4, 0,1151, + 2 sigma) : rien de mieux que l'ancre de vues a sa dose retenue ;
- un bras a `bench.gate_passed` = 0 ou `bench.audit_kl` >= 0,34 est ecarte.

Lectures (sans seuil de refutation) :
- l'ancre defait-elle l'oubli ? KL exacte d'un bras a 4e-4 contre 0,277 (ancre de vues, meme dose) :
  sous 0,22 (-20 %), l'ancre retient les images oubliees (risque du lot A) ;
- localite : audit/exact a KL exacte egale avec la courbe de SCRUB conforme (0,23 -> 1,06 de lr 1e-4
  a 4e-4) ; `bench.audit_view_kl` ;
- A contre B : le lot A, des classes du forget set, doit proteger le fold d'audit mieux que B
  (KL d'audit plus basse a KL exacte comparable).

## Regle des suivis (posee avant)

1. **Candidat** (un bras de retain >= 0,8 et score > 0,121) : graines 1 et 2 du meme reglage ; il
   devient la meilleure methode conforme si sa moyenne sur trois graines depasse **0,122** (0,1085 +
   2 x 0,007, seuil des campagnes V4 et V_Dumb) avec un retain >= 0,8.
2. Aucun autre suivi sans hypothese ecrite ici avant. Les bras de protection des poids du rapport
   (gel des 2 premiers blocs, EWC, L2) ne sont lances que si l'ancre synthetique tient le plancher.

## Lancement

```bash
make hk-run SPEC=v-fusion-anchor
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Groupe `20261122T061143Z-unl-v-fusion-anchor` (RTX 5080, commit ba05fcb, arbre propre, 3 min 11 au
total, 61 degres au plus), graine 0.

| bras | lr | score_min | retain | KL exacte | perte exacte | audit/exact | `audit_view_kl` | porte | min-steps : perte (epoque 1 -> 3) |
|---|---|---|---|---|---|---|---|---|---|
| AB | 4e-4 | 0 | 0,025 | 3,85 | 6,00 | 0,99 | 3,36 | non | 0,513 -> 0,037 |
| A | 4e-4 | 0 | 0,018 | 4,18 | 6,53 | 1,00 | 3,65 | non | 0,440 -> 0,025 |
| B | 4e-4 | 0 | 0,010 | 4,59 | 7,25 | 0,99 | 3,99 | non | 0,376 -> 0,026 |
| AB | 3e-4 | 0 | 0,022 | 3,94 | 6,10 | 1,01 | 3,41 | non | 0,283 -> 0,032 |
| ref. : vues, 4e-4 (`scrub-fv-dose` v3) | 4e-4 | 0,1279 | 0,734 | 0,277 | -- | 1,03 | 0,409 | oui | 6,60 -> 1,81 |
| ref. : vues, 3e-4 (`scrub-fv-dose` v1) | 3e-4 | 0,1151 | 0,868 | 0,165 | -- | 0,80 | 0,249 | oui | 5,16 -> 1,28 |

- **Les quatre bras s'effondrent** : retain de 0,010 a 0,025, KL exacte de 3,8 a 4,6 sur les images
  oubliees comme sur le fold d'audit, porte du banc non franchie. Ils sont tous ecartes, et aucun bras
  n'atteint le plancher a 4e-4 : **refutee**.
- **Mecanisme** : la montee de la premiere epoque est la meme qu'avec l'ancre de vues (KL maximisee
  25,9 a 4e-4, 25,2 a 3e-4, au bit pres, puisqu'aucune ancre n'a encore agi). Les min-steps sur les
  vues voient ensuite une perte de 6,6 et la ramenent a 1,8. Sur les images synthetiques, ils ne
  voient que 0,28 a 0,51, puis 0,03 : le modele casse reste d'accord avec l'original sur ces images,
  qui ne contraignent donc pas son comportement sur des images reelles. C'est la lecture de la porte d
  (le modele bruite y etait deja d'accord), a pleine echelle.
- A contre B, l'ancre defait-elle l'oubli, localite : sans objet, tous les bras sont effondres.

## Decision

- [x] Abandonner V_Fusion sous cette generation (2026-11-22) : l'ancre synthetique ne retient rien,
  a 4e-4 comme a 3e-4. Pas de graines 1 et 2, pas de bras de protection des poids (regle 2 : ils
  supposaient une ancre qui tient le plancher).
- [ ] Graines 1 et 2 / [ ] Candidate a la soumission

Suite possible, non lancee : une generation dont le plafond agit des le premier etage, des images non
saturees, et une porte de **sensibilite** avant tout bras. Cette porte comparerait la perte de
distillation du modele bruite de la porte d sur les images d'ancre et sur des vues du forget set ;
ici, 0,04-0,06 contre 2,17. Elle se mesure en deux minutes, sans entrainement.
