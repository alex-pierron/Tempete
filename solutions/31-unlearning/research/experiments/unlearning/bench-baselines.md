# bench baselines

- Identifiant : `bench-baselines` (spec : `bench-baselines.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-21
- Responsable : Alex

## Hypothese

Lignes de reference du banc local `bench-v3` : `gold_soft` et `scrub` avec leurs reglages par
defaut, notees sur le meme fold d'audit (20 % stratifie, graine 31) que toutes les idees. Sans
elles, un `bench.score_min` positif dit seulement « mieux que ne rien faire » (identite = 0).

On croit que `gold_soft` note mieux que `scrub` : le banc le favorise par construction (sa cible
est l'original sur des vues fortes, comme la doublure du gold). **Refutee si** `scrub` note
au-dessus : alors la doublure du gold ne biaise pas le classement autant qu'on le craint.

## Metrique locale attendue

- `bench.score_min` des deux methodes, `bench.audit_kl` (casse sur le fold d'audit),
  `bench.gate_passed` = 1.
- Aucune attente chiffree : ce sont les references, pas une idee a valider.

## Lancement

```bash
make hk-run SPEC=bench-baselines
```

Ces runs ont un holdout : leurs fichiers n'ont pas oublie 20 % du forget set, ne pas les soumettre.
`gold_soft` et `scrub` lisent les images proxy (COCO/DIV2K/Kodak) pour leur terme de retain :
sur une machine neuve, `scripts/fetch_data.py` (voir unlearning.md par. 9).

Run(s) : <ids du registre, `make hk-board`>

## Resultat

Pleine echelle, RTX 5080, 2026-11-21 (`20261121T205256Z-unl-bench-baselines`) :
- `gold_soft` (defauts : 3 epoques, lr 1e-5) : score_min 0, s40 0,073 ; retain 0,40 (accord 0,81,
  KL audit 1,02), ratio 1,1 : va beaucoup trop loin. 61 s d'entrainement.
- `scrub` (defauts) : score_min 0,0027 (positif sous les 3 doublures), retain 0,998, ratio 5,3 :
  bouge a peine (KL sur les images exactes 0,009).

Ces references a leurs defauts etaient des epouvantails : dosees (`gold-soft-dose`, `scrub-dose`,
voir `trust-region-campaign.md`), `gold_soft` atteint 0,059 +- 0,012 et `scrub` 0,098 +- 0,022
(3 graines chacun).

## Decision

- [ ] Abandonner / [x] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

Comparer desormais aux references dosees (`scrub` lr 1e-4, `gold_soft` 1 epoque lr 1e-5), pas aux
defauts.

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
