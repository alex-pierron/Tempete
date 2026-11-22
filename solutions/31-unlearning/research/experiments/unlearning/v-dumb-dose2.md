# v dumb dose2 (V_Dumb, suivi D2)

- Identifiant : `v-dumb-dose2` (spec : `v-dumb-dose2.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese, seuils, resultat

Suivi D2 de la campagne V_Dumb : hypotheses et criteres poses avant le run dans
`trust-region-vdumb-campaign.md` (section « Suivi D2 »), resultats et decision consignes au meme endroit.

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat (groupe `20261122T033250Z-unl-v-dumb-dose2`, commit 525b0bf, propre)

| bras (t = 0,45, lr 1e-5) | KL exacte | `final.reached` | `score_min` | retain | porte |
|---|---|---|---|---|---|
| w = 0,01 | 0,067 | 0,033 | 0,019 | 0,987 | 1 |
| w = 0,003 | 3,456 | 0,805 | 0 | 0,052 | 0 |
| w = 0,001 | 5,496 | 0,954 | 0 | 0,012 | 0 |

Falaise entre w = 0,01 et w = 0,003, aucun bras au-dessus de 0,0577 : arret de la campagne (critere pose).

## Decision

- [x] V_Dumb abandonnee (`trust-region-vdumb-campaign.md`).
