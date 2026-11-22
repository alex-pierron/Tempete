# v dumb dose (V_Dumb, suivi D)

- Identifiant : `v-dumb-dose` (spec : `v-dumb-dose.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese, seuils, resultat

Suivi D de la campagne V_Dumb : hypotheses et criteres poses avant le run dans
`trust-region-vdumb-campaign.md` (section « Suivi D »), resultats et decision consignes au meme endroit.

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat (groupe `20261122T032841Z-unl-v-dumb-dose`, commit 525b0bf, propre)

| bras (t = 0,45) | KL exacte | `final.reached` | `score_min` | retain |
|---|---|---|---|---|
| w = 0,3, lr 1e-5 | 0,006 | 0,009 | 0,0001 | 1,000 |
| w = 0,1, lr 1e-5 | 0,017 | 0,013 | 0,0008 | 1,000 |
| w = 0,03, lr 1e-5 | 0,030 | 0,021 | 0,0035 | 0,999 |
| w = 1, lr 1e-4 | 0,009 | 0,007 | 0,0031 | 0,998 |

Refutee (< 0,0577) ; la charniere ne lie pas. Direction : moins d'ancre, plus d'oubli -> suivi D2.

## Decision

- [x] Suivi D2 (`v-dumb-dose2`), pose avant dans la fiche de campagne.
