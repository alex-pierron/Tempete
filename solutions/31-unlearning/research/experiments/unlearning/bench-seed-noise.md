# bench seed noise

- Identifiant : `bench-seed-noise` (spec : `bench-seed-noise.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-21
- Responsable : Alex

## Hypothese

Plancher de bruit du banc : `trust_region` a `kl_radius = 0,6` relance avec les graines
d'entrainement 1 et 2 (la graine 0 est la variante `kl_radius=0.6` de `trust-region-kl-radius`).
Le fold d'audit, les alterations et les doublures du gold ont leur propre graine fixe : tout
l'ecart entre les trois runs vient de l'entrainement.

On croit que l'ecart-type de `bench.score_min` sur les 3 graines est petit devant l'ecart
entre methodes. **Refutee si** il est du meme ordre que l'ecart entre `gold_soft` et `scrub`
(`bench-baselines`) : il faudra alors comparer chaque idee sur plusieurs graines.

## Metrique locale attendue

- ecart-type de `bench.score_min` et de `bench.audit_kl` sur les graines 0, 1, 2.
- Regle de decision ensuite : une difference entre deux runs ne compte que si elle depasse
  2 ecarts-types de ce plancher.

## Lancement

```bash
make hk-run SPEC=bench-seed-noise
```

A lancer apres (ou avec) `trust-region-kl-radius`, qui fournit la graine 0. Ne pas soumettre.

Run(s) : <ids du registre, `make hk-board`>

## Resultat

Pleine echelle, RTX 5080, 2026-11-21 (`20261121T205647Z-unl-bench-seed-noise`, graines 1 et 2 ;
graine 0 = v3 de `trust-region-kl-radius`). Avec l'arret teste tous les 10 pas, les graines 1 et 2
s'arretent au pas 100 (KL ~1,8, porte cassee, score 0) et la graine 0 au pas 90 (score 0, retain
0,52) : l'ecart vient du pas d'arret, pas de l'entrainement. Mesure de bruit inutilisable telle
quelle.

Mesure refaite avec l'arret a chaque pas (`tr-seeds`, epsilon = 0,15) : score_min
0,0637 +- 0,0025 sur 3 graines ; `scrub` par defaut 0,0025 +- 0,0003 (`scrub-seeds`).

## Decision

- [x] Abandonner / [ ] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

Remplacee par `tr-seeds` (meme question, arret a chaque pas).

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
