# band locality (V5, suivi L2)

- Identifiant : `band-locality` (spec : `band-locality.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese, seuils, resultat

Suivi L2 de la campagne V5 : hypotheses et criteres poses avant le run dans
`trust-region-v5-campaign.md` (section « Suivi L2 »), resultats et decision consignes au meme endroit.
Mesure de mecanisme (localite), pas une candidate : la selection de la V5 est refutee a l'etape 0.

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat (groupe `20261122T031006Z-unl-band-locality`, commit 4185ff2, propre)

| bras (graine 0, 4 niveaux) | KL exacte | perte exacte | audit/exact | `score_min` | retain |
|---|---|---|---|---|---|
| temoin `cv_folds=0` (`band-ascent` v4) | 0,213 | 0,286 | 0,538 | 0,017 | 0,878 |
| `band_floor=none` | 2,711 | 1,872 | 1,010 | 0 (porte non franchie, accord d'audit 0,53) | 0,143 |
| `band_release=1` | 0,418 | 0,264 | 0,897 | 0,040 | 0,698 |

- Borne haute : **indispensable** (sans elle, effondrement). L'ablation ne separe pas localite et dose.
- Selection : **refutee** (temoin a 0,60 fois phi = 1 a perte comparable ; a KL exacte egale, meme courbe).

## Decision

- [x] Mesure faite ; V5 abandonnee (`trust-region-v5-campaign.md`).
