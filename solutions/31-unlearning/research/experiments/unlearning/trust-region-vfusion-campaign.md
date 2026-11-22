# Campagne V_Fusion (SCRUB conforme ancré sur des images synthétiques), RTX 5080

- **Étapes** :
  1. `v-fusion-step0` : génération et portes, hors registre, voir la fiche ;
  2. `v-fusion-anchor` : 4 bras d'ancre ;
  3. suivis éventuels, selon la règle de chaque fiche.
- **Banc** : tâche `unlearning`, **`bench-v4`**, fold d'audit `holdout_fraction=0.2`, graine 0.
- **Création** : 2026-11-22, avant tout run de V_Fusion ; responsable : Alex.
- **Méthode** : rapport §3, V_Fusion. Le générateur et les portes viennent du commit ceae862, l'option d'ancre `retain_source=synthetic` du commit ba05fcb. Le nom de la feuille garde la série des campagnes.
- **Contrat** : les fiches `v-fusion-step0.md` (règle d'entrée des lots) et `v-fusion-anchor.md` (seuils et suivis). Leurs seuils ont été posés avant chaque run.

## Étape 0 : images et portes

- **Génération** : 2000 images en 34 min. Lots : A = 720 (36 classes), B = 1100 (55), C = 180 (9). Ancre : 540 (A) et 825 (B).
- **Plafond sans effet** : la confiance des images vaut 0,998 à 1, au lieu de 0,9. Le plafond n'est testé qu'au dernier étage.
- **Portes a, b, c, c'** : passées pour A et B.
- **Porte d** : ininterprétable, le témoin reste à 0,911 (300 pas), puis 0,921 (1000 pas), sous 0,95. Les deux lots entrent dans l'ancre, par la règle posée avant.
- **Lecture de d** : les images synthétiques réparent environ trois fois moins que les vues (17 à 21 % des dégâts, contre 55 à 60 %). Le modèle bruité est déjà d'accord avec l'original sur ces images : perte de distillation 0,04-0,06, contre 2,17 sur les vues.

## Bras d'ancre

| bras | lr | score_min | retain | KL exacte | porte | perte des min-steps (époque 1 → 3) |
|---|---|---|---|---|---|---|
| AB | 4e-4 | 0 | 0,025 | 3,85 | non | 0,51 → 0,04 |
| A | 4e-4 | 0 | 0,018 | 4,18 | non | 0,44 → 0,02 |
| B | 4e-4 | 0 | 0,010 | 4,59 | non | 0,38 → 0,03 |
| AB | 3e-4 | 0 | 0,022 | 3,94 | non | 0,28 → 0,03 |
| vues (SCRUB conforme) | 4e-4 | 0,1279 | 0,734 | 0,277 | oui | 6,60 → 1,81 |
| vues (SCRUB conforme) | 3e-4 | 0,1151 | 0,868 | 0,165 | oui | 5,16 → 1,28 |

Les quatre bras s'effondrent.

## Conclusion

**V_Fusion est abandonnée sous cette génération** (2026-11-22) :
- Une ancre qui ne voit pas les dégâts ne les répare pas.
- La montée de SCRUB casse le modèle sur les images réelles, mais pas sur ces images synthétiques saturées.
- Leur KL tombe à 0,03, alors que le retain tombe à 0,02.

Ce n'est pas une preuve contre toute ancre synthétique. La suite logique, non lancée :
- **Une génération dont le plafond agit dès le premier étage.**
- **Une porte de sensibilité, avant tout bras** : sur le modèle bruité de la porte d, la perte de distillation sur les images d'ancre, à comparer à celle sur des vues du forget set. Elle se mesure en deux minutes.

Le meilleur conforme reste SCRUB conforme : 0,1085 ± 0,0068.
