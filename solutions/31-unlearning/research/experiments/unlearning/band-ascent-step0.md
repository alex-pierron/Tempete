# band ascent step 0 (V5, sans entrainement)

- Identifiant : `band-ascent-step0` (spec : `band-ascent-step0.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

Etape 0 de la V5 (`band_ascent`, rapport section V5), avant tout entrainement. La V5 ne pousse que
la fraction phi = 0,25 des images dont l'original doute le plus sur 4 vues fortes (aire 0,50) et
parie que le gold se trompe vers les secondes classes de l'original. Sur les 2800 images
entrainees du banc (`diagnose`, `src/hackathon/unlearning/diagnose.py`), sous chacune des six
doublures :

1. **Selection** (`selection.<cle>.phi25`) : les images poussees sont-elles celles ou la doublure
   se trompe ? `precision` contre `base_rate` (8 a 23 % selon la doublure), `recall`.
2. **Masse** (`mass.<cle>.phi25`) : part de la perte moyenne de la doublure portee par ces images
   (`loss_share`) contre le meilleur choix de meme taille (`loss_share_oracle`), et part du
   denominateur de S_pred (`pred_share`). Ce que la V5 laisse sur la table en ne bougeant pas les
   75 % d'images typiques.
3. **Direction** (`direction.<cle>`) : S_pred atteint a logits libres quand l'eleve met sur la
   pseudo-etiquette exactement la masse de la doublure et forme les classes fausses comme
   l'original sur l'image exacte (`runner_ups`, ce que fait la montee au premier ordre) ou comme le
   melange des vues de la V5 (`views`).

Les cles `mXX` (masquages) font foi : la selection est tiree de recadrages, comme les cles `sXX`,
qui sont donc circulaires.

**Refutee si** (seuils fixes a priori, le 2026-11-22) :
- `selection.mXX.phi25.precision` < 0,35 pour les trois masquages : la selection ne designe pas les
  erreurs ; ne pas lancer `band-ascent` sous cette forme ;
- ou `mass.mXX.phi25.loss_share` < 0,5 pour les trois masquages : la queue des pertes ne porte pas
  l'essentiel du score, et la V5 laisserait trop sur la table.

Lecture : si `direction.mXX.views` depasse `direction.mXX.runner_ups` de plus de 0,05 sur les trois
masquages, le pari de direction de la V5 perd ; l'option de direction du rapport (non codee) devient
la suite.

**Extension V5.1** (ajoutee le 2026-11-22, avant tout run ; rapport section V5.1). Le meme run lit
en plus, sans entrainement :

4. **Encodeur** (`selection_by.knn_<bloc>`, `direction.<cle>.knn_<bloc>`, `encoder.*`) : un vote
   kNN (k = 10, tau = 0,1, fixes d'avance) sur le jeton CLS de l'original apres chaque bloc et a
   l'entree de la tete, ajuste par plis (les 5 plis de la validation croisee de la V5 : une image
   ne vote jamais pour elle-meme ni pour son pli). Le bloc est choisi d'avance par la regle
   `encoder.best` : la plus haute precision moyenne de selection a phi = 0,25 sur les trois
   masquages. **L'option encodeur n'est pas codee si**, pour tous les blocs, cette precision ne
   depasse pas celle de la selection V5 (`encoder.views_mask_precision`) de plus de 0,05 en restant
   sous 0,35, **et** si sa direction ne bat pas `runner_ups` sur au moins deux des trois masquages
   (`encoder.best_direction_wins` < 2). Selection et direction se jugent separement.
   `encoder.knn_agreement` proche de 1 dit qu'un bloc est effondre sur les classes memorisees.
5. **Defauts de la V5** (`defects.*`, sur l'original, au premier pas) :
   - famine : la bande en marge est inutile si `defects.force_participation` > 0,5 et
     `defects.force_top10_share` < 0,3 (la force de la montee en perte n'est pas concentree) ;
   - borne des images tenues : le defaut 2 est refute si `defects.floor_kl_median` < 0,064 (une
     image a la borne delta_0 deriverait moins que le budget brut du temoin a cv_max_drift 0,1) ;
   - poussee au depart : `defects.held_over_pushed` donne la force qui rend les images tenues plus
     sures, rapportee a la montee ; sous 0,05, elle est negligeable.

## Reglages

`--views 4 --view-min-area 0.5` (ceux de `band-ascent`), graine 0 ; fold d'audit de 20 % retire
comme dans le banc.

## Metrique locale attendue

Precision des masquages au moins deux fois le taux de base ; `loss_share` proche de
`loss_share_oracle`. Cout : ~30 passes avant de 2800 images, aucune retropropagation.

## Lancement

```bash
make hk-run SPEC=band-ascent-step0                 # RTX 5080
make hk-run SPEC=band-ascent-step0 WHERE=jureca    # 1 A100, file d'equipe
```

## Resultat

Run `20261122T024729Z-unl-band-ascent-step0` (RTX 5080, commit 3a1f81e, arbre propre, 2800 images
entrainees, 42 s de calcul, 51 s au total, 58 °C au plus). JSON : `runs/unlearning/<run>/step0.json`.

| doublure | precision phi25 (taux de base) | precision / base | `loss_share` (oracle) | `pred_share` | direction `runner_ups` / `views` |
|---|---|---|---|---|---|
| s40 | 0,144 (0,068) | 2,13 | 0,406 (0,741) | 0,328 | 0,714 / 0,802 |
| s50 | 0,050 (0,021) | 2,41 | 0,441 (0,860) | 0,357 | 0,705 / 0,797 |
| s60 | 0,027 (0,010) | 2,62 | 0,465 (0,924) | 0,391 | 0,712 / 0,775 |
| **m30** | **0,183 (0,190)** | 0,96 | **0,236 (0,919)** | 0,229 | **0,752 / 0,637** |
| **m20** | **0,109 (0,118)** | 0,92 | **0,231 (0,976)** | 0,224 | **0,765 / 0,609** |
| **m15** | **0,087 (0,086)** | 1,01 | **0,238 (0,991)** | 0,223 | **0,768 / 0,594** |

- **Selection : refutee.** Sous les trois masquages, la precision (0,087 a 0,183) est sous 0,35 et
  egale au taux de base (0,92 a 1,01 fois) : la selection de la V5 ne designe pas mieux les erreurs
  des masquages qu'un tirage au hasard. Le rappel vaut 0,23 a 0,25, celui d'un tirage de 25 %.
  Sous les recadrages (circulaires), elle fait 2,1 a 2,6 fois le hasard, mais reste sous 0,35 (les
  recadrages se trompent sur 1 a 7 % des images seulement).
- **Masse : refutee.** `loss_share` vaut 0,231 a 0,238 sous les masquages (seuil 0,5), soit la part
  d'un tirage au hasard (0,25) ; le meilleur choix de meme taille en porterait 0,92 a 0,99. Meme sous
  les recadrages, famille commune aux vues de la V5 et aux doublures, elle reste sous 0,5 (0,41 a
  0,47, contre 0,74 a 0,92 pour l'oracle).
- **Direction : le pari de la V5 tient sous les masquages.** `runner_ups` y depasse `views` de 0,11 a
  0,17 (la lecture ne bascule que si `views` depasse `runner_ups` de plus de 0,05) ; sous les
  recadrages, `views` gagne de 0,06 a 0,09, ce qui est circulaire. L'option de direction n'a donc pas
  lieu d'etre codee. Reserve : ce score suppose que l'eleve met sur la pseudo-etiquette exactement
  la masse de la doublure, image par image, c'est-a-dire qu'il sait quelles images la doublure doute ;
  c'est precisement ce que la selection ne sait pas.
- A noter : la perte des doublures est tres concentree. Les 25 % d'images les plus perdues portent
  74 a 92 % de la perte moyenne des recadrages et 92 a 99 % de celle des masquages (10 % en portent
  43 a 65 % et 65 a 89 %). Mais ce ne sont pas les memes images d'une famille a l'autre.

Les deux criteres de refutation fixes a priori sont atteints, chacun sur les trois masquages.

Reserve sur le seuil de precision, notee apres coup : a phi = 0,25, la precision d'un selecteur ne
peut pas depasser taux de base / phi (tous les cas d'erreur selectionnes). Sous m15 ce plafond vaut
0,344 : le seuil de 0,35 y etait inatteignable, meme pour un selecteur parfait. Le critere porte sur
les trois masquages a la fois ; sous m30 et m20, ou les plafonds valent 0,76 et 0,47, la precision
mesuree (0,183 et 0,109) reste au taux de base, et le critere de masse tranche seul de toute facon.

**Extension V5.1, rejouee** : run `20261122T050855Z-unl-band-ascent-step0` (RTX 5080, commit ceae862,
arbre propre, 50 s). Les lectures de la V5 y sont identiques au run precedent (precision m30 0,1829).

- Encodeur (vote kNN ajuste par plis) : la precision de selection monte avec la profondeur ; bloc
  retenu par la regle `encoder.best` : `pre_logits` (entree de la tete), precision 0,256 / 0,169 /
  0,137 sous m30 / m20 / m15 (taux de base 0,190 / 0,118 / 0,086, soit 1,35 a 1,59 fois), moyenne
  0,187 contre 0,126 pour la selection V5 : +0,061, au-dessus de la barre de +0,05 mais sous 0,35.
  Part de la perte 0,35 a 0,40 (critere de masse de la V5 : 0,5, toujours non atteint). Direction :
  perdante sur les trois masquages (`best_direction_wins` = 0 ; 0,46 contre 0,75 pour `runner_ups`
  sous m30). Les deux derniers niveaux sont effondres sur les classes memorisees
  (`knn_agreement` 0,994 et 0,995). Lettre de la regle : l'option encodeur n'est pas exclue pour la
  selection (+0,061 > 0,05), mais une V5.1 a selection par l'encodeur tomberait encore sous le
  critere de masse ; elle n'est pas codee (la V5.1 est abandonnee, voir `band-ascent-v51.md`).
- Defauts (sur l'original) : les trois sont **confirmes**.
  - famine : `force_participation` 0,030 (seuil 0,5) et `force_top10_share` 0,873 (seuil 0,3) : 10 %
    des images portent 87 % de la force de la montee en perte ;
  - borne des images tenues : `floor_kl_median` 0,370 (refute sous 0,064) : une image a la borne
    delta_0 deriverait de 5,8 fois le budget brut du temoin ;
  - poussee au depart : `held_over_pushed` 1,94 (negligeable sous 0,05) : la force qui rend les images
    tenues plus sures vaut presque deux fois la montee.

## Decision

- [x] Abandonner la V5 sous cette forme : `band-ascent` n'est pas lance (2026-11-22).
- [ ] Lancer `band-ascent` / [ ] Coder l'option de direction (le pari de direction tient : inutile)

Suite proposee : `trust-region-v5-campaign.md`, section Proposition.
- [ ] V5.1 : coder l'option encodeur (selection, direction) : sans objet depuis l'abandon de la V5
