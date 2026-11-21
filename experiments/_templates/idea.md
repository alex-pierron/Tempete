# {{idea_text}}

- Identifiant : `{{id}}` (spec : `{{id}}.toml`)
- Tache : `{{task}}`
- Creee le : {{date}}
- Responsable : <nom>

## Hypothese

Ce que l'on croit qui va changer, et pourquoi (une a trois phrases). Ecrire ce qui REFUTERAIT
l'idee avant de lancer.

## Metrique locale attendue

Quelle metrique locale (jamais le score public), de combien, par rapport a quelle reference.
Exemple : `proxy.retain_kl` < 0,02 sans perdre la confiance moyenne du forget set.

## Lancement

```bash
make hk-run SPEC={{id}}
```

Run(s) : <ids du registre, `make hk-board`>

## Resultat

Ce qui s'est passe, chiffres a l'appui (metriques locales des runs, anomalies, duree).

## Decision

- [ ] Abandonner / [ ] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
