# band ascent V5.1 (ablation, un composant par bras)

- Identifiant : `band-ascent-v51` (spec : `band-ascent-v51.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

V5.1 (options de `band_ascent`, rapport section V5.1). Objectif du 2026-11-22 : terme retain
officiel >= 0,8, budget de derive depense vers le gold (bandes plafonnees a rho <= 1). La V5.1
corrige trois defauts de la V5 releves sans run : (1) la montee en perte est affamee (force
<= sqrt(2) L, perte de l'original 3,2e-4 en moyenne) -> `band_space=margin`, force d'ordre 1 ;
(2) la borne `band_floor` des images tenues ne serre pas au budget et sa force de rappel est elle
aussi affamee -> `band_hold_kl=0.128`, charniere sur la KL a T = 2 de chaque image tenue ;
(3) cette borne pousse les images tenues des le premier pas (poids 0,38 a beta = 10) -> la retenue
KL a un gradient nul a l'original. Dose choisie par `cv_min_retain=0.8` : accord / (1 + echelle x
derive) du pli temoin >= 0,8. Deux passes par niveau (`epochs=2`) : dans la simulation du modele
lineaire, la localite n'apparait qu'en consolidation.

Prediction de la simulation (tendance seulement, rapport tableau v51sim) : rapport audit/exact
final 0,20 (tolerance 0,1) a 0,33 (0,2), contre 0,46 pour la marge seule et 0,55 pour la V5 ;
toutes les images liberees dans leur bande avec la marge, 31 % en perte.

Bras (4 = un noeud) : V5.1 complete (`band_release=0.25`, la valeur par defaut) ;
`band_hold_kl=none` (borne `band_floor` ecrite en marge, sans retenue KL) ; `band_space=loss`
(bande en perte, retenue KL gardee) ; `epochs=1` (sans consolidation). Reference appariee : le bras
`cv_max_drift=0.2` de `band-ascent-budget` (V5 au meme plancher, sans code).

sigma = max(ecart-type entre graines de la reference en `bench-v4`, 0,0025) ; seuil 2 sigma ~ 0,006.

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run) :
- `bench.retain` < 0,8 au bras complet : le plancher n'est pas tenu par la selection ;
- ou rapport audit/exact (`bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`) du bras
  complet > 0,4 : la retenue KL ne localise pas a pleine echelle ;
- ou `bench.score_min` du bras complet <= celui du bras 0,2 de `band-ascent-budget` + 2 sigma : la
  V5.1 n'apporte rien sur le banc ;
- ou le bras complet termine en arret dur (`final.hard_stop` = 1) ou sans niveau retenu
  (`final.cv_stop_level` = 0) : refaire a lr 3e-5 avant d'abandonner.

Lecture des ablations (chaque composant juge separement) :
- marge : `final.pushed_in_band` du bras complet >= 0,8 et au moins 0,2 au-dessus du bras
  `band_space=loss` ; sinon la famine n'est pas le facteur limitant (voir aussi
  `defects.force_participation` de `band-ascent-step0`) ;
- retenue KL : le rapport audit/exact du bras `band_hold_kl=none` depasse celui du bras complet
  d'au moins 0,1 ; sinon la retenue KL est inutile sous cette forme ;
- consolidation : le rapport du bras `epochs=1` depasse celui du bras complet d'au moins 0,05 ;
  sinon une passe suffit (et coute deux fois moins).

Estimateur : `final.cv_retain` doit tomber a +-0,03 de `bench.retain` (sinon le retain du temoin ne
predit pas celui du banc, et `cv_min_retain` ne vaut rien sous cette forme).

## Reglages

Ceux de `band-ascent` (V5 par defaut : lr 1e-4, lots de 32 stratifies, 4 niveaux rho = 0,25 a 1,
phi = 0,25, kappa = 2, delta_0 = 0,05, beta = 10, 4 vues d'aire 0,50, K = 5), plus :
`band_space=margin` (beta garde 10, desormais par logit), `band_hold_kl=0.128`,
`cv_min_retain=0.8` (arret dur a 2 x (1/0,8 - 1) = 0,5), `epochs=2`.

## Metrique locale attendue

- `bench.retain` >= 0,8 ; rapport audit/exact <= 0,4 ; `final.pushed_in_band` >= 0,8 ;
- `bench.score_min` du bras complet > bras 0,2 de `band-ascent-budget` + 2 sigma, avec un gain
  sous le masquage (`bench.score_min_family.mask`) ;
- toujours lire `bench.audit_view_kl` (derive sur des entrees non memorisees).
- Cout : ~2,5 min par variante sur la RTX 5080 (deux passes par niveau ; estime, non mesure).

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=band-ascent-step0               # d'abord : sans entrainement (lit aussi les defauts)
make hk-run SPEC=band-ascent-budget              # la V5 au meme plancher, sans code
make hk-run SPEC=band-ascent-v51                 # RTX 5080 (full)
make hk-run SPEC=band-ascent-v51 WHERE=jureca    # 4 A100 sur un noeud (file d'equipe)
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Groupe `20261122T051321Z-unl-band-ascent-v51` (RTX 5080, commit ceae862, arbre propre, 4 min 06),
graine 0, lr 1e-4. Suivi a lr 3e-5 et 1e-5 : `band-ascent-v51-lr.md`. Detail :
`trust-region-v51-campaign.md`.

| bras | niveaux gardes | KL exacte | audit/exact | `score_min` (recadr. / masq.) | retain | `cv_retain` (ecart) | `pushed_in_band` | `held_kl` |
|---|---|---|---|---|---|---|---|---|
| V5.1 complete | 0 (arret dur au pas 20 du niveau 1) | 0 | -- | 0 (modele intact) | 1 | -- | -- | -- |
| `band_hold_kl=none` | 0 (arret dur au pas 20 du niveau 1) | 0 | -- | 0 (modele intact) | 1 | -- | -- | -- |
| `band_space=loss` | 4 | 0,190 | 0,543 | 0,011 (0,095 / 0,011) | 0,895 | 0,832 (-0,063) | 0,039 | 0,047 |
| `epochs=1` | 0 (arret dur au pas 15 du niveau 1) | 0 | -- | 0 (modele intact) | 1 | -- | -- | -- |

- Les trois bras en marge s'arretent au premier niveau en 15 a 20 pas : derive du temoin 0,5 a 1,0
  (KL brute), 44 a 74 % de la sonde au-dessus de sa borne. La force d'ordre 1 fait depasser la bande
  en quelques pas ; aucun niveau garde, le run final ne bouge pas le modele.
- Critere « arret dur ou sans niveau garde » : **atteint** au bras complet -> refaire a lr 3e-5 (fait,
  `band-ascent-v51-lr`).
- Ablations : illisibles a ce lr (un seul bras a entraine). Le bras en perte avec retenue KL n'est pas
  plus local que la V5 au meme ordre de dose (audit/exact 0,543 a KL exacte 0,190 ; V5 a 4 niveaux :
  0,538 a 0,213), et son estimateur se trompe de 0,063.

Apres le suivi (`band-ascent-v51-lr`) : la V5.1 complete ne garde un niveau qu'a lr 1e-5, ou elle
score 0,0796, mais avec un retain de 0,725 et un rapport audit/exact de 0,84 : refutee sur deux
criteres (retain < 0,8 ; audit/exact > 0,4).

## Decision

- [x] Abandonner la V5.1 (2026-11-22) : la marge depasse la bande (arret dur a 1e-4 et 3e-5) ; au
  seul lr ou elle garde un niveau (1e-5), le plancher de retain n'est pas tenu (0,725) et l'oubli n'est
  pas local (0,84). Le temoin, optimiste de 0,136 a ce lr, ne protege pas le run final.
- [ ] Iterer / [ ] Coder l'option encodeur : sans objet (selection par l'encodeur 1,4 fois le hasard
  seulement, part de la perte 0,35 a 0,40 sous les masquages, direction perdante ; voir
  `band-ascent-step0.md`) / [ ] Candidate a la soumission
