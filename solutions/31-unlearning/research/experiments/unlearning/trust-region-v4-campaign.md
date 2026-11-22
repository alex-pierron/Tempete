# Campagne V4 (ancre tirée du forget set, contrainte active, SCRUB conforme), RTX 5080

- Spécifications : `ref4-v3` (rejeu de contrôle), `scrub-forget-views`, `trust-region-dual`,
  `trust-region-anchored`, puis les suivis justifiés plus bas (au plus 4 variantes chacun)
- Tâche : `unlearning` ; banc : **`bench-v4`** (6 doublures : recadrages s40/s50/s60, masquages
  m30/m20/m15) ; fold d'audit `holdout_fraction=0.2`
- Créée le : 2026-11-22, avant tout run de la V4 ; responsable : Alex
- Méthodes : options V4 de `trust_region` et de `scrub` (rapport §3, V4 ; code du commit ec682ec,
  fusionné en 52fa1d3)
- Contrat : les trois fiches V4 (`scrub-forget-views.md`, `trust-region-dual.md`,
  `trust-region-anchored.md`), hypothèses et seuils posés avant tout run. Cette fiche ne les
  réécrit pas : elle fixe l'ordre, les contrôles et la règle des suivis, et consigne les résultats.
- Tableau : `make hk-unl-scores` → `SCOREBOARD.md` (section `bench-v4`)

## Grandeurs lues sur chaque run

- `bench.score_min` (pire des six doublures), `bench.score_min_family.crop` et `.mask`.
- **audit/exact** = `bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub` : le mécanisme visé
  (V3 retenue : 0,857 / 0,897 / 0,909 aux graines 0-2 ; `scrub` proxy : 0,553 / 0,511 / 0,596).
  À ne pas confondre avec `bench.movement_ratio` (KL moyenne des groupes oubliés, copies comprises,
  sur la KL d'audit : 1,35 / 1,27 / 1,25 pour la V3), donné aussi.
- Contrôles : `final.radius_reached` = 1 (pour `dual`, lire `final.radius_ratio`, la KL finale
  pouvant finir juste sous le rayon), `bench.gate_passed` = 1, `bench.audit_kl` < 0,34.
- `dual` : `final.kl_radius_effective` des deux côtés (V3 : rayon sur toutes les images ; dual :
  sur la sonde de 256).

## Références appariées (déjà au registre, `bench-v4`, RTX 5080, commit 67d4806)

| référence | graine 0 | graine 1 | graine 2 | moyenne |
|---|---|---|---|---|
| V3 retenue (`tr3-hedge` v1, `tr3-seeds` v1-v2) | 0,0784 | 0,0804 | 0,0806 | 0,0798 ± 0,0012 |
| V2 retenue (`ref4-v2`) | 0,0757 | 0,0817 | 0,0785 | 0,0786 ± 0,0030 |
| `scrub` proxy lr 1e-4 (`ref4-scrub`, non conforme) | 0,1024 | 0,0739 | 0,1171 | 0,098 ± 0,022 |

## Étape 0 : contrôle de non-régression (posé avant)

Hypothèse du dépôt : les options V4 sont neutres à leur défaut. `ref4-v3` (graines 0-2) doit redonner
au bit près `tr3-hedge` v1 et `tr3-seeds` v1-v2 (`score_min`, `retain`, `audit_kl`). Un écart, même
au cinquième chiffre, arrête la campagne : les comparaisons appariées n'auraient plus de sens.

**Étape 0 bis (posée avant, ~2 min)** : rejouer `ref4-scrub` (graines 0-2), pour deux raisons.
(1) Vérifier que `retain_source` est neutre à son défaut sur `scrub` (même critère : au bit près).
(2) Donner à la référence proxy les composantes `bench.exact.*` (images exactes seules), que la fiche
`scrub-forget-views` demande de comparer (circularité des max-steps) et que les runs `ref4-scrub`
d'origine n'ont pas.

## Étapes 1-3 : les trois spécifications, telles quelles

Ordre par coût : `scrub-forget-views` (4 runs, ~35 s), `trust-region-dual` (4 runs, ~3 min),
`trust-region-anchored` (4 runs, 4 phases). Jugées chacune contre sa fiche.

## Règle des suivis (posée avant les runs)

Un suivi n'est lancé que pour une de ces raisons, et sa question, ses bras et ses seuils sont écrits
ici avant le lancement :

1. **Candidat à confirmer** : un bras passe le seuil « candidat » de sa fiche → graines manquantes du
   même réglage, jusqu'à trois graines appariées ; seuil final ETAT.md : moyenne > 0,086 et gain sous
   le masquage au-delà de 2σ.
2. **Non interprétable** (fiche `dual` : `final.radius_ratio` hors [0,85 ; 1,15]) → nouvelle valeur de
   η justifiée par la trajectoire mesurée du multiplicateur, pas une grille.
3. **Composante tranchée sur une seule graine** : le meilleur réglage conforme de la campagne doit
   avoir trois graines appariées, même s'il échoue (pour que la conclusion porte sur une moyenne).
4. **Dose voisine** seulement si la mesure en donne la direction (par exemple : mécanisme confirmé mais
   oubli insuffisant, ou casse à une dose et pas à l'autre).

Pas de suivi sur la V5. Arrêt quand chaque composante (ancre, contrainte duale, SCRUB conforme) est
confirmée ou réfutée et que le meilleur réglage conforme a trois graines appariées.

## Résultats

### Étape 0 : non-régression confirmée

`pytest tests/hackathon` : 558 tests passent (3 ignorés). `ref4-v3` (graines 0-2, commit 52fa1d3, code
propre, groupe `20261122T012512Z-unl-ref4-v3`) redonne au bit près `tr3-hedge` v1 et `tr3-seeds`
v1-v2 : sur les 159 métriques communes, seule `train_seconds` diffère (`score_min` 0,078382 ;
0,080411 ; 0,080563 ; `retain` et `audit_kl` identiques aux 16 chiffres). Les options V4 sont
neutres à leur défaut. Le rejeu ajoute `bench.exact.*` (images exactes seules, hors score), que les
runs de la V3 n'avaient pas : les comparaisons appariées à la V3 se font désormais sur `ref4-v3`.
2 min 11 s, 59 °C au plus.

### Étape 0 bis : `ref4-scrub` rejoué, neutre au bit près

Groupe `20261122T013637Z-unl-ref4-scrub` (propre) : identique aux runs d'origine sur toutes les
métriques communes (sauf `train_seconds`) ; `retain_source` est neutre à son défaut. Composantes sur
images exactes seules (`bench.exact.*`, graines 0-2) : pertes 0,24 à 0,71, prédictions 0,34 à 0,42.

**Incident de protocole, corrigé.** `scrub-forget-views` et ce rejeu ont d'abord tourné avec un
fichier de test modifié (`tests/hackathon/test_hk_unlearning_scoreboard.py`, drapeau `dirty`,
groupes `20261122T012752Z` et `20261122T013058Z`). Rejoués sur un arbre propre, ils redonnent les mêmes
chiffres au bit près. Les sept lignes `dirty` restent au registre (trace d'audit, signalées par une
étoile après le commit dans `SCOREBOARD.md`) ; seuls les runs propres sont cités. Le test n'est
réappliqué qu'après le dernier run.

### Étape 1 : `scrub-forget-views` et ses suivis (détail : fiche `scrub-forget-views.md`)

- Aux doses de la fiche (lr 1e-4, 2e-4), **réfutée** : 0,0607 en moyenne au meilleur lr, sous 0,086 ;
  mécanisme confirmé (audit/exact 0,22 à 0,51) mais l'ancre défait l'oubli (KL exacte 0,27 à 0,31 fois
  le proxy à 1e-4).
- Suivi 1 (`scrub-fv-dose`, lr 3e-4 et 4e-4, posé avant) : l'ancre conforme déplace la dose. À 3e-4,
  **candidat** : 0,1151 et 0,1015 (moyenne 0,108), retain 0,85-0,87, gain apparié sur la V3 +0,037 et
  +0,021 sous le masquage. À 4e-4 : 0,121 en moyenne mais retain 0,73, écarté par le plancher de 0,80.
  La localité ne tient plus à ces doses (audit/exact 0,80 à 1,09) : le gain vient du terme d'oubli.
- Suivi 2 (`scrub-fv-seeds`, graine 2 à 3e-4 et 4e-4, posé avant) : en file.

### Étape 2 : `trust-region-dual` (détail : fiche `trust-region-dual.md`)

Quatre runs (η 0,25 et 0,5, graines 0 et 1) : 0,042 partout (V3 appariée 0,078 et 0,080). La contrainte
ne lie pas : μ retombe à 0 dès l'époque 3, η n'y change rien (même modèle à 1e-4 près), et l'objectif
de la V3 poursuivi sur tout le calendrier se pose à KL T = 1 de 0,085, la moitié du rayon
(`radius_ratio` 0,53-0,54 : non interprétable au sens de la fiche, mais pas par η). Il ajuste sa cible
(perte 0,63 contre 2,1-2,2) : composante des prédictions doublée, composante des pertes divisée par
2 à 3, score divisé par deux, KL d'audit 1,25 à 1,39 fois la V3. **Abandonnée.**

### Étape 3 : `trust-region-anchored` et son suivi (détail : fiche `trust-region-anchored.md`)

- lr 1e-5 (4 bras, graine 0) : aucune phase n'atteint son rayon ; 0,035 à 0,055 ; audit/exact 0,87 à
  0,98 ; même signature de calendrier complet que `dual`.
- Suivi `tr4-anchored-lr` (remède prévu par la spec : lr 3e-5 ; plus un témoin sans ancre) : les phases
  approchent leur rayon (0,82-0,89 à w = 1) sans l'atteindre ; audit/exact 0,67 à 0,70 (témoin 0,86) ; 0,058 à
  0,070, gain sous le recadrage seulement. **Ancre réfutée.**
- `pending_kl` classe les réglages comme le fold d'audit, mais en sous-estime la dérive (facteur 1,6 à 13).

## Suivi hors V4 : la V3 retenue à lr 3e-5 (posé le 2026-11-22, avant le run)

**Ce qui l'ouvre** : le témoin sans ancre de `tr4-anchored-lr` (la V3 retenue, seul le taux change :
3e-5 au lieu de 1e-5) vaut 0,0876 à la graine 0, contre 0,0784 au même réglage à 1e-5 (+0,0092, 3σ,
sous le masquage comme sous le recadrage). Son rayon (0,173) est le même ; il l'atteint au pas 63,
pendant le warmup (88 pas), au lieu du pas 102 : l'élève est arrêté plus tôt dans son vol, à un taux de
2,1e-5. La V1 était insensible au taux (`tr-lr` : 0,062 à 0,067 de 3e-6 à 1e-4, `bench-v3`).

**Hypothèse** : à rayon égal, la V3 retenue gagne à lr 3e-5.

**Bras** (`tr3-lr-seeds`, grille lr × graine) : lr 3e-5, graines 1 et 2 (trois graines appariées à
`ref4-v3` avec le témoin) ; lr 1e-4, graines 1 et 2 (la direction, appariée aux mêmes graines : la
tendance continue-t-elle, ou 3e-5 est-il un point isolé ?).

**Seuils (ETAT.md)** : confirmée si la moyenne des trois graines à 3e-5 > 0,086 et si le gain apparié
sous le masquage dépasse 2σ (0,006) en moyenne ; réfutée sinon (le point de la graine 0 était du
bruit). Contrôles habituels : `final.radius_reached` = 1, `bench.gate_passed` = 1, `bench.audit_kl`
< 0,34. Le bras 1e-4 (deux graines) ne confirme rien seul : il dit s'il faut aller plus loin. Quel que
soit le résultat, il ne change pas la meilleure méthode conforme mesurée (`scrub` conforme, 0,1085), à
moins que la V3 à 3e-5 dépasse 0,1085 + 2σ.


### Résultat du suivi hors V4 (groupe `20261122T022731Z-unl-tr3-lr-seeds`, code propre)

| V3 retenue | graine 0 | graine 1 | graine 2 | moyenne | gain apparié (masquage) |
|---|---|---|---|---|---|
| lr 1e-5 (`ref4-v3`) | 0,0784 | 0,0804 | 0,0806 | 0,0798 ± 0,0012 | |
| lr 3e-5 | 0,0876 | 0,0837 | 0,0821 | 0,0845 ± 0,0028 | +0,0092 ; +0,0033 ; +0,0015 (moy. +0,0047) |
| lr 1e-4 | | 0,0841 | 0,0872 | | +0,0037 ; +0,0066 |

- **Réfutée** : à lr 3e-5, moyenne 0,0845 (≤ 0,086) et gain apparié moyen +0,0047 (< 2σ). La graine 0
  était le haut de la fourchette. La V3, comme la V1, est insensible au taux dans le bruit.
- lr 1e-4 casse les contrôles : graine 1, `bench.audit_kl` 0,382 (> 0,34) et retain 0,665 ; graine 2,
  `final.radius_reached` = 0 (arrêt au pas 34 sur la sonde, KL de toutes les images sous le rayon).
  Pas de tendance à suivre.

## Conclusion (2026-11-22)

| composante V4 | verdict | meilleur réglage mesuré | score_min | audit/exact |
|---|---|---|---|---|
| SCRUB conforme (`retain_source=forget_views`) | **confirmée**, à lr 3e-4 (réfutée aux doses de la fiche) | lr 3e-4, 3 graines | **0,1085 ± 0,0068** | 0,85 |
| contrainte active (`radius_dual_lr`) | **réfutée** : la contrainte ne lie pas ; le calendrier complet divise le score par deux | η 0,25, 2 graines | 0,042 | 0,98 |
| ancre croisée (`anchor_folds`) | **réfutée** à lr 1e-5 et 3e-5 : freine l'oubli partout, ne le localise pas | w 3, lr 3e-5, 1 graine | 0,070 | 0,69 |

- La meilleure méthode conforme mesurée est désormais `scrub` à ancre conforme, lr 3e-4 :
  +0,029 sur la V3 retenue sous le masquage, apparié sur trois graines (≈ 10σ), retain 0,86.
  Elle ne gagne pas par la localité (audit/exact 0,85, comme la V3) mais par l'efficacité de son
  oubli : à KL exacte moitié moindre (0,17 contre 0,29), des composantes d'oubli au moins égales et un
  retain plus haut de 0,12.
- Le levier visé par la V4, la localité, n'a été obtenu par aucune composante à une dose qui score :
  `scrub` conforme n'est local (0,22 à 0,51) qu'aux doses où l'ancre défait l'oubli ; l'ancre ne
  baisse audit/exact que de 20 % (0,86 → 0,69, lr 3e-5), au prix du masquage.
- Leçon commune à `dual` et à l'ancre : l'objectif des V1-V3 ne tient son score qu'arrêté en plein
  vol. Poursuivi (tout le calendrier), il se pose sous le rayon à T = 1, monte à T = 2, égalise les
  pertes (composante des pertes ÷ 2 à 3) et perd la moitié du score.
- Rien n'a été soumis. Tous ces runs ont un fold d'audit.

## Inventaire

39 runs, RTX 5080, 2026-11-22 (≈ 60 min de GPU, 63 °C au plus, aucune pause thermique) : 26 mesures
nouvelles (`scrub-forget-views` 4, `scrub-fv-dose` 4, `scrub-fv-seeds` 2, `trust-region-dual` 4,
`trust-region-anchored` 4, `tr4-anchored-lr` 4, `tr3-lr-seeds` 4), 6 rejeux propres de références
(`ref4-v3` 3, `ref4-scrub` 3), 7 doublons `dirty` (étape 0 bis). Registre : 155 runs terminés, 1 refusé.
