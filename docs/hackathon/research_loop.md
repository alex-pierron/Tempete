# Boucle de recherche (22 h) : recherche → rapports → décision → implémentation

> Cadre commun à toutes les équipes d'agents. Le directeur (JARVIS) orchestre ; l'équipe humaine
> décide à l'étape 3. Version du 21/11/2026.

## Les 3 étapes, en boucle

1. **Recherche** : 2 agents par sujet, angles complémentaires. Chacun explore plusieurs idées,
   les teste pour de vrai (petite échelle en local, ou sur le serveur si le budget est autorisé),
   et mesure tout ce qui est mesurable.
2. **Rapports** : chaque agent livre un rapport HTML autonome + son PDF dans
   `reports/phase<N>/<tâche>/<équipe>/` (`report.html`, `report.pdf`, `figures/`, `metrics.json`).
   Le directeur en fait une synthèse d'une page par sujet.
3. **Décision humaine** : l'équipe choisit les pistes ; le directeur lance la phase
   d'implémentation (agents de code + QA), puis renourrit la recherche avec les résultats.

## Contenu obligatoire d'un rapport (dans cet ordre)

1. **Résumé décisionnel** (5 lignes max) : ce qui marche, ce qui ne marche pas, ce que l'on
   recommande, ce que l'on demande à l'équipe de décider.
2. **Pistes explorées** : pour chacune, hypothèse, source (papier, keynote, intuition), ce qui a
   été fait, résultat chiffré, verdict (garder / abandonner / creuser).
3. **Toutes les métriques possibles** : la métrique officielle (ou son proxy local, clairement
   étiqueté comme proxy), et les métriques de diagnostic utiles (par méthode, par type d'édition,
   par catégorie de prompt, distributions, intervalles de confiance quand c'est possible).
4. **Visuels** : figures PNG incluses dans le HTML (base64) ou dans `figures/` : distributions,
   matrices, cartes, exemples réussis ET ratés. Chaque figure a une légende qui dit quoi en conclure.
5. **Interprétabilité** : pourquoi la méthode marche ou échoue (cas d'erreur typiques, ce que le
   modèle « regarde », sensibilité aux hyperparamètres).
6. **Classement des pistes** : gain attendu, coût (temps humain, GPU, budget de requêtes), risque
   (surtout de sur-ajustement au leaderboard public 30 %).
7. **Reproductibilité** : commandes exactes, SHA git, seeds, durée, où sont les sorties.

## Règles

- Ne jamais lire ni afficher `.env` ; ne jamais écrire de clé ou de token.
- Aucune soumission au serveur pendant la recherche (c'est une décision humaine). Les requêtes
  `ep-*` du Model Equality sont interdites sans décision ; les `ref-*` sont limités à 100 par
  endpoint pour toute l'équipe : chaque équipe a un quota écrit dans son brief.
- Chiffres réels seulement ; un chiffre extrapolé est marqué « estimation ».
- Code de recherche dans `src/hackathon/<tâche>/research/` (typé, testé au minimum) ou dans
  `reports/.../scripts/` (jetable mais reproductible) ; jamais de données ou de poids commités.
- Commandes lourdes via `jarvis-heavy` (machine partagée, 16 Go) ; les gros calculs iront sur
  JURECA dès que l'accès sera ouvert.
- Export PDF : `node scripts/hackathon/report_to_pdf.mjs reports/.../report.html`.
- Visuels : matplotlib (dans `.venv`), pas de dépendance nouvelle sans accord.
