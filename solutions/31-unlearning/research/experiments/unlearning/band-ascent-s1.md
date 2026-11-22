# band ascent s1 (V5, suivi L2)

- Identifiant : `band-ascent-s1` (spec : `band-ascent-s1.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese, seuils, resultat

Suivi L2 de la campagne V5 : hypotheses et criteres poses avant le run dans
`trust-region-v5-campaign.md` (section « Suivi L2 »), resultats et decision consignes au meme endroit.
Mesure de mecanisme (localite), pas une candidate : la selection de la V5 est refutee a l'etape 0.

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat (groupe `20261122T031151Z-unl-band-ascent-s1`, commit 4185ff2, propre)

| bras (graine 1) | niveaux | KL exacte | audit/exact | `score_min` (recadr. / masq.) | retain |
|---|---|---|---|---|---|
| defaut (CV) | 2 (CV : arret dur au niveau 3) | 0,119 | 0,393 | 0,005 (0,038 / 0,005) | 0,950 |
| `cv_folds=0` | 4 | 0,271 | 0,739 | 0,046 (0,117 / 0,046) | 0,809 |

- Localite du bras par defaut **reproduite** (0,393 <= 0,5). Point a 4 niveaux **non reproduit** a +-20 %
  (0,739 contre 0,538), mais la graine 1 derive plus (KL exacte 0,271 contre 0,213) et reste sur la
  courbe audit/exact contre KL exacte de la graine 0.

## Decision

- [x] Mesure faite ; V5 abandonnee (`trust-region-v5-campaign.md`).
