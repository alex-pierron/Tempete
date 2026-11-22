# Campagne V5.1 (`band_ascent` : bande en marge, retenue KL, plancher de retain), RTX 5080

- Spécifications, dans l'ordre de la fiche `band-ascent-v51.md` : `band-ascent-step0` (rejoué avec
  les lectures V5.1 : encodeur et défauts), `band-ascent-budget` (3 bras), `band-ascent-v51` (4 bras),
  puis les suivis justifiés plus bas (au plus 4 variantes chacun)
- Tâche : `unlearning` ; banc : **`bench-v4`** ; fold d'audit `holdout_fraction=0.2` ; graine 0
- Créée le : 2026-11-22, avant tout run de la V5.1 ; responsable : Alex
- Méthode : options de `band_ascent` (rapport §3, V5.1 ; commit 4a0956b, fusionné en ceae862), neutres
  par défaut. La V5.1 a été conçue avant le verdict de la V5 et retirée du plan à la fusion (sa base est
  abandonnée) ; elle est lancée ici à la demande de l'utilisateur (« expérimente avec les nouvelles
  méthodes »). Le verdict de l'étape 0 de la V5 (sélection au hasard sous les masquages) reste acquis :
  il vaut pour la V5.1, qui garde la même sélection.
- Contrat : les fiches `band-ascent-step0.md` (extension V5.1), `band-ascent-budget.md` et
  `band-ascent-v51.md`, seuils posés avant tout run. Cette fiche ne les réécrit pas : elle ajoute les
  comparaisons appariées à SCRUB conforme et à la V5, et la règle des suivis.

## Grandeurs lues sur chaque run

- `bench.score_min`, `.score_min_family.crop` et `.mask` ; S_ms et S_pred par doublure.
- Dérive : `bench.retain` (plancher 0,8), audit/exact (`bench.audit_kl` /
  `bench.groups.forget_exact.kl_orig_sub`) **lu à KL exacte égale** (il croît avec la dose),
  `bench.audit_view_kl`.
- V5.1 : `final.cv_stop_level`, `final.hard_stop`, `final.band_levels_done`, `final.pushed_in_band`,
  `final.held_kl`, `final.cv_retain` contre `bench.retain` (estimateur, ±0,03).
- Contrôles : `bench.gate_passed` = 1, `bench.audit_kl` < 0,34.

## Références appariées (graine 0, au registre, `bench-v4`, RTX 5080)

| référence | score_min | recadr. | masq. | retain | KL exacte | audit/exact |
|---|---|---|---|---|---|---|
| V5 par défaut (`band-ascent` v1) | 0,000 | 0,005 | 0,000 | 0,981 | 0,074 | 0,246 |
| V5 à 4 niveaux (`band-ascent` v4, `cv_folds=0`) | 0,017 | 0,077 | 0,017 | 0,878 | 0,213 | 0,538 |
| V3 (`ref4-v3` v1) | 0,0784 | 0,0865 | 0,0784 | 0,758 | 0,284 | 0,86 |
| SCRUB conforme lr 3e-4 (`scrub-fv-dose` v1) | 0,1151 | 0,1229 | 0,1151 | 0,868 | 0,165 | 0,80 |

Moyennes sur 3 graines : V1 0,0637 ± 0,0025 ; V3 0,0798 ± 0,0012 ; SCRUB conforme 0,1085 ± 0,0068.

## Seuils ajoutés ici, avant les runs

- Le bras 0,1 de `band-ascent-budget` est le bras par défaut de `band-ascent` : il doit en redonner
  les métriques au bit près (options V5.1 neutres par défaut). Sinon, la fusion a changé la V5.
- Pour compter comme une méthode qui score : meilleur bras > **0,0687** (V1 + 2σ, σ = 0,0025 comme dans
  les fiches) avec retain ≥ 0,8.
- Pour devenir la meilleure méthode conforme : à la graine 0, > **0,121** (SCRUB conforme apparié,
  0,1151, + 2σ), puis moyenne sur trois graines > **0,122** avec retain ≥ 0,8.

## Règle des suivis (posée avant les runs)

1. **Arrêt dur** du bras V5.1 complet (`final.hard_stop` = 1 ou `final.cv_stop_level` = 0) : la fiche
   prescrit de refaire à lr 3e-5 avant d'abandonner.
2. **Candidat** (bras > 0,0687 avec retain ≥ 0,8) : graines 1 et 2 du même réglage.
3. Aucun autre suivi : pas de nouveau réglage sans hypothèse écrite ici avant.

### Suivi S1 (règle 1), posé le 2026-11-22 après `band-ascent-v51`, avant son lancement

Les trois bras en marge de `band-ascent-v51` (lr 1e-4) font un arrêt dur au niveau 1 en 15 à 20 pas
(`final.cv_stop_level` = 0) : la règle 1 s'applique. Spec `band-ascent-v51-lr` (fiche au même nom) :
V5.1 complète, sans retenue KL et bande en perte à lr 3e-5, plus la V5.1 complète à lr 1e-5. Seuils de
`band-ascent-v51` reconduits ; abandon si les deux bras complets s'arrêtent encore sans niveau gardé.

## Résultats

Tous les runs ont tourné sur la RTX 5080, au commit ceae862, avec un arbre propre. En tout : 1 diagnostic et 11 runs d'entraînement, 14 min de GPU, 62 °C au plus.

### Étape 0 rejouée (`band-ascent-step0`, 50 s)

- **Lectures de la V5** : identiques au premier run.
- **Encodeur** : le vote kNN à l'entrée de la tête sélectionne 1,35 à 1,59 fois mieux que le hasard sous
  les masquages. Sa précision moyenne vaut 0,187, contre 0,126 pour la sélection V5. Mais la part de la perte reste à
  0,35-0,40 (critère de masse : 0,5), et la direction perd sur les trois masquages. L'option n'est pas codée.
- **Défauts de la V5** : les trois sont confirmés sur l'original. Il y a famine (10 % des images portent 87 % de
  la force). La borne δ0 ne serre pas au budget (KL médiane 0,37 à la borne, budget 0,064). Et la poussée
  au départ vaut 1,94 fois la montée.

### `band-ascent-budget` (V5, budget du témoin 0,1 / 0,2 / 0,3)

| budget | niveaux | KL exacte | audit/exact | score_min | retain | cv_retain − retain | pushed_in_band |
|---|---|---|---|---|---|---|---|
| 0,1 | 1 | 0,074 | 0,246 | 0,000 | 0,981 | −0,025 | 0,006 |
| 0,2 | 3 | 0,181 | 0,441 | 0,006 | 0,916 | −0,038 | 0,016 |
| 0,3 | 4 | 0,213 | 0,538 | 0,017 | 0,878 | −0,083 | 0,046 |

- **Contrôle** : le bras 0,1 redonne `band-ascent` v1 au bit près (192 métriques), et le bras 0,3 redonne
  `band-ascent` v4 (187 métriques).
- **Critères de la fiche** : aucun n'est atteint. Le budget est consommé et le retain tient, mais le
  score reste nul.

### `band-ascent-v51` (lr 1e-4)

Les trois bras en marge (le complet, sans retenue KL, et à une passe) s'arrêtent net au niveau 1, en
15 à 20 pas, sans garder de niveau. Le modèle reste intact. Le bras en perte avec retenue KL garde
4 niveaux : il score 0,011, avec un retain de 0,895 et un rapport audit/exact de 0,543. La règle 1
s'applique.

### Suivi S1 `band-ascent-v51-lr`

| bras | lr | niveaux | KL exacte | audit/exact | score_min (recadr. / masq.) | retain | cv_retain − retain |
|---|---|---|---|---|---|---|---|
| V5.1 complète | 3e-5 | 0 (arrêt dur, pas 40) | 0 | -- | 0 | 1 | -- |
| sans retenue KL | 3e-5 | 0 (témoin 0,70) | 0 | -- | 0 | 1 | -- |
| bande en perte | 3e-5 | 4 | 0,067 | 0,340 | 0,0385 (0,0555 / 0,0385) | 0,975 | −0,015 |
| V5.1 complète | 1e-5 | 1 | 0,339 | 0,839 | 0,0796 (0,0796 / 0,0898) | 0,725 | +0,136 |

Le bras jugé est la V5.1 complète à 1e-5. Il dépasse V1 + 2σ (0,0687), mais son retain de 0,725 est
sous le plancher de 0,8 et son rapport audit/exact de 0,84 dépasse 0,4. Il est donc réfuté sur deux
critères, et ce n'est pas un candidat (règle 2) : pas de graines 1 et 2. À retain comparable, SCRUB conforme à
lr 4e-4 fait 0,1279 à la graine 0, avec un retain de 0,734.

## Conclusion

**La V5.1 est abandonnée** (2026-11-22).

- **La marge** donne bien une force d'ordre 1, mais cette force fait dépasser la bande en
  quelques pas. À 1e-4 et 3e-5, les bras en marge s'arrêtent sur arrêt dur, sans garder de niveau.
- **À 1e-5**, seul taux où la V5.1 complète garde un niveau, le run final dérive bien plus que le
  sous-run du témoin. L'estimateur se trompe alors dans le sens optimiste (+0,136) : le plancher de
  retain n'est pas tenu (0,725), et l'oubli n'est pas local (0,84, comme la V3).
- **La retenue KL** ne tient pas les images tenues à pleine échelle (KL finale 0,299, pour une
  tolérance de 0,128).
- **Ce qui marche le moins mal** est la retenue KL en espace de perte à 3e-5 : un retain de 0,975 et un
  score de 0,0385, plus que la V5 à dérive comparable, mais toujours sous la V1.

Le meilleur conforme reste SCRUB conforme, à 0,1085 ± 0,0068.
