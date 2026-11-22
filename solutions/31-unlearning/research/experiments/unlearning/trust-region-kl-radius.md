# trust region kl radius

- Identifiant : `trust-region-kl-radius` (spec : `trust-region-kl-radius.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-21
- Responsable : Alex

## Hypothese

SCRUB+R sans donnees : le point de « rewind » (erreur sur le forget = erreur d'un set de
validation qu'on n'a pas) devient un rayon epsilon. On tire les sorties de l'eleve vers l'original
adouci (`softmax(z0/6)`) sur les images EXACTES du forget set (KL convexe, classement des classes
garde, aucune augmentation, aucune image externe), en batches stratifies par pseudo-label, et on
s'arrete des que KL(original || eleve) moyenne a T = 1 atteint epsilon. epsilon = budget d'oubli
(a T = 1, ~ perte moyenne du gold sur des images inedites ; a priori 0,3-1, non mesure).

Garde-fou mesure par validation croisee : 20 % du forget set, stratifie par pseudo-label, est tenu
a l'ecart. Ces images sont dans la distribution et memorisees ; un gold « sans le fold entraine »
les memoriserait aussi, donc toute KL qui y apparait est de la casse collaterale pure (meilleur
proxy du control set que COCO/DIV2K/Kodak, hors distribution).

On croit que : l'oubli reste localise sur le fold entraine (peu de KL sur le fold tenu a l'ecart)
et se transfere aux copies alterees bien qu'on n'entraine que sur les pixels exacts.

**Refutee si**, pour les epsilon atteints :
- `bench.movement_ratio` (KL forget moyenne sur exact + alterees / KL fold d'audit, T = 2)
  < 2 : l'oubli se repand sur les images memorisees voisines autant que sur les cibles ;
- ou `bench.groups.forget_flip|crop|jpeg.kl_orig_sub` < 0,5 x `bench.groups.forget_exact.kl_orig_sub` :
  on a creuse un trou sur les pixels exacts, la moitie du forget score (copies alterees cachees)
  est perdue ;
- ou `bench.score_min` = 0 pour tous les epsilon : pas mieux que l'identite sous la pire
  doublure du gold.

Seuils 2 et 0,5 fixes a priori, avant tout run. Revisions du 2026-11-21, avant tout lancement :
metriques renommees vers le banc local `bench-v3` (formules du sujet) ; epsilon in {0,2 ; 0,4 ; 0,6 ; 0,8} sur 4 A100.

## Metrique locale attendue

- `final.radius_reached` = 1 pour epsilon <= 0,6 en 10 epoques (sinon : lr trop bas, a monter).
- `bench.score_min` > 0 avec `bench.gate_passed` = 1, et au-dessus des references
  `bench-baselines` (gold_soft, scrub) d'au moins le plancher de bruit mesure par
  `bench-seed-noise` (sinon : egalite, pas une victoire).
- `bench.movement_ratio` >= 5 et `bench.audit_kl` qui croit avec epsilon.
- Mesure du smoke (M2, 52 images, 7 pas a lr 5e-5) : l'oubli est specifique a chaque image (les
  images pas encore entrainees restent a KL ~1e-6 apres une mise a jour) ; movement ratio 84.

Le score reel (distance au gold) n'est PAS mesure ici. epsilon se choisit sur le suivi local
(aucune soumission au leaderboard pour l'instant, decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=trust-region-kl-radius                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=trust-region-kl-radius WHERE=jureca    # 4 epsilon sur 4 A100 (file d'equipe)
```

Ces runs ont un holdout : leurs fichiers n'ont pas oublie 20 % du forget set, ne pas les soumettre.

Run(s) :
- smoke local `20261121T191133Z-unl-trust-region-kl-radius` (v1-v4, ancienne spec a 4 rayons,
  M2/MPS, 52 images entrainees, 12 tenues a l'ecart, 4 pas) : chaine validee, cible a KL 1,20
  (tau = 6), rayon non atteint en 4 pas (KL 1,4e-5, attendu), ~15 s d'entrainement par variante.
  Lignes de registre mises de cote dans `runs/unlearning/registry-smoke-20261121.jsonl`.
- smoke local via le lab `20261121T194457Z-unl-trust-region-kl-radius` (v1-v2, spec a 2 rayons,
  `--config auto` -> smoke sur M2, 40 pas a lr 1e-5) : chaine validee de bout en bout, KL 0,041,
  rayon non atteint (attendu en smoke), banc ecrit sous `bench_smoke.*` (hors classement).

## Resultat

Pleine echelle, RTX 5080, 2026-11-21 (`20261121T205507Z-unl-trust-region-kl-radius`, v1-v4,
~25 s par variante). Le rayon, teste tous les 10 pas, ne controle rien : la KL passe de 0,07 a 0,65
entre les pas 80 et 90 (fin du warmup), donc epsilon = 0,2, 0,4 et 0,6 s'arretent au pas 90 et
donnent le meme modele (score_min 0, s40 0,085, retain 0,52, ratio 1,2) ; epsilon = 0,8 s'arrete
au pas 100 (KL 1,9, porte cassee).

Criteres fixes avant lancement : ratio < 2 -> **atteint** (1,0-1,2) ; copies alterees < 0,5 x
exactes -> non (0,80-0,90) ; score_min = 0 pour tous les epsilon -> **atteint**.

Suite dans `trust-region-campaign.md` : avec le rayon teste a chaque pas
(`radius_check_every=1`), la KL suit epsilon et score_min devient positif (0,0637 +- 0,0025 a
epsilon = 0,15 sur 3 graines). Le ratio reste < 2 a tous les rayons.

## Decision

- [ ] Abandonner / [x] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

Iteree dans `trust-region-campaign.md` (arret a chaque pas, puis balayages). Cette spec reste comme
trace du defaut de l'arret tous les 10 pas ; ne plus la relancer telle quelle.

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
