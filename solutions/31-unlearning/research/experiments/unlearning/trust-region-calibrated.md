# trust region calibrated

- Identifiant : `trust-region-calibrated` (spec : `trust-region-calibrated.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-21
- Responsable : Alex

## Hypothese

V3 de `trust-region-kl-radius` (V1) et de `trust-region-views` (V2). Le score compare deux objets
differents du gold : des statistiques de pertes calculees modele par modele puis moyennees
(`mean_std`) et le MELANGE des distributions des modeles (`prediction`). Les cibles V1 et V2
n'en visent qu'un, et mal : leurs pertes ont un ecart-type de 0,2 a 0,4 fois leur moyenne, contre
1,7 a 3,8 pour toutes les doublures du gold mesurees (note de recherche du 2026-11-21, §1.1).
La V3 garde tout (region de confiance, lots stratifies, fold d'audit, lr 1e-5, 10 epoques) et
change trois choses (`src/hackathon/unlearning/methods.py`) :

1. **Cible calibree** (`trust_target=calibrated`), une fois, sans gradient, sur 4 vues fortes
   (aire 0,50-0,60) et la pseudo-etiquette y : direction a T = 2 = melange des vues
   p_hat = moy softmax(z_v/2), couvert par l'original adouci de la V1 :
   p_dir = 0,75 p_hat + 0,25 softmax(z_orig/6) ; puis le SEUL logit de y est deplace, en forme
   close, pour que la perte a T = 1 vaille l* = l_mix + lambda (l_bar - l_mix), ou l_mix est la
   perte de p_dir et l_bar la perte moyenne par vue. Les rapports entre classes fausses de p_dir
   sont gardes exactement.
2. **Rayon tire de la cible** : epsilon = 0,9 x KL(original || cible) (~ 0,9 x moy l*), plus de
   grille.
3. **Copies douces** (variantes `altered_fraction` = 0,5) : la moitie de chaque lot est remplacee
   par un retournement ou un recadrage 70-100 % de la meme image, avec la cible de la source. Le
   jitter et le JPEG ne sont jamais vus a l'entrainement.

Variantes (4 = un noeud) : `trust_lambda` in {0,25 ; 0,5} x `altered_fraction` in {0 ; 0,5}.

On croit que : (a) la V3 bat la meilleure V1 et la meilleure V2 sur `bench.score_min` (pire cas
des 6 doublures du banc `bench-v4`, deux familles), par la composante `mean_std` ; (b) le gain
tient sur la famille de doublures que la cible ne copie pas (masquage) ; (c) les copies douces
relevent l'oubli des copies alterees de familles jamais vues (jitter, JPEG).

**Plafonds, pas attentes.** La recherche donne, dans un modele a logits libres (chaque image garde
ses logits, flot de gradient de la perte du code jusqu'au rayon ; ce n'est pas un reseau), et sur
le seul groupe des images exactes : pire cas sur les 6 doublures 0,086 pour cette cible
(lambda = 0,25, b = 0,25) contre 0,030-0,037 pour la V1 et la V2, soit x2,3 a x2,9 ; au sein de
la seule famille des recadrages, x5 (circulaire). Pour les copies alterees, si l'eleve ne
transfere qu'une fraction r de son mouvement, forget ~ ((1+r)/2)^2 S_ms S_pred (D6) : passer de
r ~ 0,5-0,67 (rapport mesure en essai reduit) a r ~ 1 multiplierait l'oubli par 1,45 a 1,8. Ce
sont des bornes d'un modele simplifie : un reseau, un rayon non atteint ou un transfert partiel
donneront moins.

**Biais, a lire avant tout resultat** : la cible et les doublures par recadrage sont toutes deux
« l'original sur des vues fortes » (graines distinctes : `seed` + 1 000 003 contre 31000-31203).
Seules les doublures par masquage (m30, m20, m15) sont d'une famille que rien n'entraine : un gain
qui n'y tient pas est mecanique.

**Refutee si** (comparaisons a protocole egal `bench-v4`, sigma = ecart-type de `bench.score_min`
mesure par `bench-seed-noise`) :
- `bench.score_min` de la meilleure variante V3 <= max(`bench.score_min` des 4 epsilon de la V1,
  des 4 epsilon de la V2) + 2 sigma : pas mieux que la meilleure V1/V2 (comparaison genereuse
  envers elles : 8 essais contre 4) ;
- ou `bench.score_min_family.mask` de cette variante <= celui de la meilleure V1/V2 + 2 sigma :
  le gain ne tient pas sur la famille non circulaire ;
- ou `bench.score_min` de cette variante <= la meilleure reference de `bench-baselines` + 2 sigma ;
- ou `bench.audit_kl` > 1,5 x `bench.audit_kl` de la variante V1 dont epsilon est le plus proche
  de `final.kl_radius_effective` : la cible casse davantage les images voisines ;
- ou, pour les variantes `altered_fraction` = 0, `bench.groups.forget_flip|crop|jpeg.kl_orig_sub`
  < 0,5 x `forget_exact.kl_orig_sub` (meme seuil que la V1 et la V2).

Mecanisme refute (meme si le chiffre monte) si :
- `bench.mean_std` (V3) <= `bench.mean_std` (meilleure V2) sous au moins 4 des 6 doublures : le
  gain ne vient pas des statistiques de pertes ;
- ou l'ecart-type des pertes de l'eleve sur les images exactes reste sous sa moyenne
  (`bench.groups.forget_exact.loss_std_sub` / `loss_mean_sub` < 1 ; V1/V2 visent 0,2-0,4, les
  doublures 1,7-3,8).

Copies douces refutees (a lambda egal, `altered_fraction` 0,5 contre 0) si le rapport
`kl_orig_sub(jitter) / kl_orig_sub(exact)` et le meme pour `jpeg` n'augmentent pas, ou si
`bench.score_min` baisse de plus de 2 sigma. Seuls jitter et JPEG comptent : le retournement est
vu a l'entrainement et le recadrage du banc (60-90 %) recouvre celui de l'entrainement (70-100 %).
Le rayon etant mesure sur les images exactes, les deux variantes s'arretent a la meme KL exacte :
le rapport compare le transfert a oubli egal.

Regle de decision ajoutee (pas une refutation) : si `bench.audit_view_kl` (derive sur une vue
forte du fold d'audit, hors score) depasse 0,01, l'ancre de retention (rang 3 de la recherche)
devient la prochaine piste ; sinon elle est abandonnee. (Depassee a la fusion, voir la revision
ci-dessous : la campagne place deja l'ancre en tete.)

Seuils fixes a priori, le 2026-11-21, avant tout run complet (seul le smoke ci-dessous a tourne).

**Revision a la fusion avec la campagne V1 (2026-11-21, avant tout run de la V3).** Les seuils
ci-dessus ne changent pas ; leur lecture, si :
- **Protocole.** La campagne V1 (54 runs) est notee en `bench-v3`, la V3 le sera en `bench-v4` :
  les references (V1 a epsilon = 0,15, `scrub` lr 1e-4, `gold_soft` 1 epoque lr 1e-5, et le
  plancher de bruit) se recalculent en `bench-v4` avant toute comparaison (`bench --candidate` sur
  leurs fichiers, ou nouveaux runs : moins d'une minute chacun sur la RTX 5080). Jamais de
  comparaison entre un chiffre `bench-v3` et un chiffre `bench-v4`.
- **Une prediction de la recherche se verifie.** Formulee sans connaitre la campagne : avec la
  cible de la V1, `score_min` chute au-dela d'epsilon ~ 0,2-0,3, par la doublure la plus sure
  d'elle (s60) et la composante `prediction`. Mesure (`bench-v3`) : 0,034 a 0,3 ; 0,002 a 0,5 ; 0 a
  0,8, s60 etant la pire doublure a chaque fois. Le rayon que la V3 tire de sa cible (0,9 x KL,
  soit ~0,16 a 0,18 selon l'echantillon) tombe dans le plateau mesure de la V1 (0,1-0,2).
- **Une premisse s'affaiblit.** La recherche plafonnait `mean_std` d'un eleve qui copie une
  cible temperee vers 0,27-0,31 ; a pleine echelle, la V1 atteint 0,456 sous s60 (rapport,
  meilleure performance) : le reseau etale davantage les pertes que sa cible. Le gain attendu de
  la V3 par `mean_std` est donc moins sur ; la composante qui limite tous les runs mesures est
  `prediction` (0,18 a 0,38), que la direction de la V3 (melange des vues, couvert) vise. Le
  critere de mecanisme ci-dessus reste tel quel : il tranchera.
- **Copies douces.** A pleine echelle, la V1 transfere deja : la pire copie (recadrage) bouge
  0,70 fois autant que l'image exacte, le JPEG 1,3 fois. Le gain maximal des copies douces tombe a
  ((1+1)/(1+0,70))^2 ~ x1,38 (contre x1,45-1,8 estime d'apres le smoke).
- **Localite.** Le defaut mesure de la V1 est la derive sur le fold d'audit (KL 0,23 a
  epsilon = 0,15, ratio 1,4 ; 1,14 sur les images exactes seules), que la V3 ne traite pas. La regle « ancre si `bench.audit_view_kl`
  > 0,01 » est donc depassee : l'ancre de retention est deja la piste suivante. Sous la forme que
  propose la campagne (KL vers l'original sur les images proxy COCO/DIV2K/Kodak, comme `scrub`),
  elle utilise des images exterieures au forget set, ce que le sujet exclut pour une methode
  soumise (« from the original model and the forget set alone ») ; la forme conforme ancre l'eleve
  sur les vues fortes des images oubliees, dont la V3 calcule deja les sorties de l'original.
  `bench.audit_view_kl` dira si la derive sur des entrees non memorisees differe de celle du fold
  d'audit exact.

**Revision du 2026-11-22 (relecture des 54 runs V1 et du code, ecrite en parallele de la campagne
V3, sans la connaitre ; relue a la fusion).**
- Trois criteres ci-dessus ne sont pas operants tels quels : « ecart-type des pertes de l'eleve
  sous sa moyenne » ne peut pas se declencher (l'eleve generique de la V1 est deja a 3,6) ; le
  critere des copies douces recompense le mouvement du jitter, que le banc penalise (sous s60, S_ms
  du groupe jitter vaut 0 pour tout run qui bouge) ; la barre « bench-baselines + 2 sigma » compare
  a des references non dosees. La campagne V3 (`trust-region-v3-campaign.md`) les a chiffres en
  bench-v4 avec ses propres seuils.
- Pronostic de la relecture : la V3 retomberait sur le plateau de la V1. **Refute** par la
  campagne V3 : 0,0798 +- 0,0012 au meilleur reglage, au-dessus de la V1 par la direction de sa
  cible. Le mecanisme decrit, lui, se confirme : la dispersion des pertes est fixee par le reseau, et
  le fold d'audit bouge autant qu'avec la V1 (audit/exact 0,86 a 0,91 pour la V3 retenue).
- Rayon et sonde : le rayon `radius_fraction` est pris sur tout le set alors que l'arret est teste
  sur une sonde de 256 images (un eleve qui suivrait sa cible pourrait ne jamais s'arreter). Les
  options de la V4 prennent le rayon sur la sonde ; la V3 garde son calcul, pour que les runs
  mesures se rejouent exactement (`ref4-v3`).

## Reglages

- `trust_hedge` = 0,25 et `target_temperature` = 6 : la couverture est l'original adouci de la V1 ;
  elle achete de la robustesse en `prediction` quand la direction des vues est fausse (D3 :
  KL(p_gold || .) recompense de repartir la masse quand la direction est incertaine).
- `target_views` = 4, `target_view_min_area` = 0,50 : la moyenne harmonique des forces des
  doublures (2 x 0,954 x 0,333 / 1,287 = 0,494, D5) ; 2 a 4 vues a une force centrale font mieux
  dans le pire cas qu'un melange de forces (recherche §1.2).
- `radius_fraction` = 0,9 : l'eleve qui atteint la cible a K ~ moy l* (D5) ; 0,9 garde le
  controle de faisabilite (cible > rayon) avec une marge.
- lr 1e-5, 10 epoques, lots de 32 : ceux de la V1 et de la V2 ; `radius_check_every` = 1. La
  campagne V1 a pleine echelle (RTX 5080, `trust-region-campaign.md`) a mesure le contraire du
  risque craint ici d'apres le smoke : a lr 1e-5 la V1 atteint epsilon = 0,15 en ~85 pas, pendant
  le warmup, et le danger est de DEPASSER le rayon entre deux tests (la KL passe de 0,07 a 0,65 en
  10 pas) ; d'ou le test a chaque pas, ajoute a la spec a la fusion. Les variantes
  `altered_fraction` = 0,5 voient chaque image exacte deux fois moins : si `final.radius_reached`
  = 0, monter lr (la campagne : 3e-5 atteint le rayon en 50 pas, meme score pour la V1).

## Metrique locale attendue

- `final.target_kl` ~ `final.target_loss_mean` (moy l*) ~ 0,17-0,20 (smoke : 0,173 sur 52 images ;
  recherche : 0,20 sur 256) ; `final.target_loss_std` / `target_loss_mean` ~ 2,8 (smoke) ;
  `final.kl_radius_effective` = 0,9 x `final.target_kl`.
- `final.radius_reached` = 1 en 10 epoques (sinon : lr, voir ci-dessus).
- `bench.gate_passed` = 1. `bench.movement_ratio` et `bench.audit_kl` voisins de ceux de la V1
  (1,4 et 0,23 a epsilon = 0,15 dans la campagne, `bench-v3`) : la V3 ne vise pas la localite.
  (Ecrit avant la campagne : ratio >= 5 et `bench.audit_view_kl` < 0,01 ; la campagne a mesure
  que la V1 en est loin.)

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=trust-region-calibrated                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=trust-region-calibrated WHERE=jureca    # 4 variantes sur 4 A100 (file d'equipe)
```

A lancer avec (ou apres) `trust-region-kl-radius`, `trust-region-views`, `bench-baselines` et
`bench-seed-noise`, tous notes en `bench-v4`. Ces runs ont un holdout : ne pas soumettre leurs
fichiers.

Run(s) :
- smoke en ligne de commande (pas via le lab, donc rien au registre), une seule configuration :
  `lambda` = 0,25, `altered_fraction` = 0,5, M2/MPS, 52 images entrainees, 12 tenues a l'ecart,
  40 pas a lr 1e-5, 38 s d'entrainement. Commande :
  `python -m hackathon.unlearning run --config smoke --method trust_region` avec les `--set` de la
  spec (variante v2), sortie dans le scratchpad de la session (non versionnee).

## Resultat

Smoke seulement (aucun run complet) : chaine validee de bout en bout sur MPS (le premier essai a
revele que MPS n'a pas de float64 : la cible est desormais calculee sur le CPU, corrige avant ce
smoke). Cible sur les 52 images : KL 0,173, accord de l'argmax 0,962, l* = 0,173 +- 0,482 (rapport
ecart-type/moyenne 2,8, contre 0,2-0,4 pour les cibles V1/V2). Rayon effectif 0,156, non atteint
en 40 pas (KL 0,033, attendu en smoke).

Banc `bench-v4` (`bench_smoke.*`) :

| | score_min | pire recadrage / masquage | s40 / s50 / s60 | m30 / m20 / m15 | audit KL | audit view KL | ratio |
|---|---|---|---|---|---|---|---|
| identite | 0 | 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 | -- |
| V3 smoke | 0,0195 | 0,0224 / 0,0195 | 0,022 / 0,058 / 0,059 | 0,020 / 0,025 / 0,031 | 0,0006 | 0,0086 | 143 |

`mean_std` 0,076 / 0,180 / 0,188 (recadrages), 0,067 / 0,092 / 0,120 (masquages) ; `prediction`
0,29 / 0,32 / 0,31 et 0,29 / 0,28 / 0,26. KL(orig || sub) a T = 2 par groupe : exact 0,066,
flip 0,058, crop 0,047, jitter 0,126, JPEG 0,100.

Lecture : rien n'est interpretable a cette echelle (une graine, 52 images, rayon a 21 %, une
seule variante). Les scores par recadrage utilisent les memes graines que `bench-v3` ; a titre
indicatif seulement, s40 vaut 0,022 contre 0,029 pour les smokes V1 et V2 (40 pas aussi, mais
sans copies douces : chaque image exacte vue deux fois plus). `bench.audit_view_kl` (0,0086) vaut
15 fois la KL sur le fold d'audit exact (0,0006) : a cette echelle, le fold d'audit memorise
sous-estime la derive sur des entrees non memorisees ; elle reste sous le seuil de 0,01. Identite
recalculee par `python -m hackathon.unlearning bench --forget-subset 64` : 0 exactement.

## Resultat a pleine echelle (2026-11-22, RTX 5080, bench-v4)

Campagne complete : `trust-region-v3-campaign.md` (23 runs V3 + 9 references relancees en
bench-v4, etapes posees avant les runs). Cette spec, lancee telle quelle (graine 0) :

| variante | rayon effectif | score_min | pire recadrage | pire masquage | retain | KL audit |
|---|---|---|---|---|---|---|
| lambda 0,25, copies 0 | 0,159 | 0,0761 | 0,0773 | 0,0761 | 0,755 | 0,244 |
| lambda 0,25, copies 0,5 | 0,159 | 0,0762 | 0,0774 | 0,0762 | 0,749 | 0,253 |
| lambda 0,5, copies 0 | 0,252 | 0,0566 | 0,0566 | 0,0649 | 0,678 | 0,343 |
| lambda 0,5, copies 0,5 | 0,252 | 0,0646 | 0,0646 | 0,0697 | 0,697 | 0,316 |

References bench-v4 (3 graines) : V1 epsilon 0,15 : 0,0637 +- 0,0025 ; V2 retenue (tau 1) :
0,0786 +- 0,0030 ; sigma = 0,003. Meilleur reglage V3 trouve par la campagne (b = 0 au lieu de
0,25) : 0,0798 +- 0,0012.

Criteres de cette fiche, fixes avant tout run :
- score_min <= meilleure V1/V2 + 2 sigma (0,0846) : **atteint** (0,0761 pour cette spec, 0,0798 au
  mieux) -> pas mieux que la V2 ;
- masquage <= meilleure V1/V2 + 2 sigma : **atteint** (0,0798 contre 0,0786) ;
- <= meilleure reference de `bench-baselines` + 2 sigma : non (references par defaut : 0,0027) ;
- KL audit > 1,5 x V1 a rayon voisin (0,228) : non au meilleur reglage (0,26) ; **oui a lambda 0,5**
  (0,343) ;
- copies alterees < 0,5 x exactes (copies 0) : non (recadrage 0,76, JPEG 1,25).
- Mecanisme : S_ms (V3) <= S_ms (V2) sous les 6 doublures -> **refute** ; dispersion des pertes de
  l'eleve > 1 (3,3) mais identique a V1/V2 : elle ne vient pas de la cible.
- Copies douces : rapports jitter/exact (1,80 contre 1,82) et JPEG/exact (1,27 contre 1,28)
  inchanges -> **refutees**.

## Decision

- [ ] Abandonner / [x] Iterer (quoi, pourquoi) / [ ] Candidate a la soumission

La V3 fait jeu egal avec la V2 (meilleur reglage : couverture b = 0, pas 0,25) ; elle n'est pas
une amelioration. Le levier suivant n'est plus la cible mais la localite (ancre de retention
conforme, sur les vues fortes des images oubliees).

Si candidate : valider d'abord (`python -m hackathon.lab submit <RUN_ID>`), puis une seule
personne soumet pour cette tache (`--yes`). Rappel : le score public (30 %) informe, il ne decide pas.
