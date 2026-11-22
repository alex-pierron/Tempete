# V_Dumb (la methode la plus simple)

- Identifiant : `v-dumb` (spec : `v-dumb.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

V_Dumb (`dumb`, `methods.py`, rapport section V_Dumb). L'original a memorise les images oubliees
(perte ~0) ; un modele qui ne les a jamais vues y a une perte moderee. A chaque pas, sur un lot tire
au hasard : on fait monter l'entropie croisee de chaque image sur son pseudo-label (argmax de
l'original) jusqu'a une cible t, pas au-dela (charniere relu(t - CE)), et on garde une vue forte de
chaque image du lot (recadrage de 25-35 % de l'aire puis couleur) a la sortie de l'original (KL a
T = 2, poids 1). Pas de distribution cible, pas d'arret, pas de pli. Reglages par defaut : lr 1e-5,
3 passes, lots de 32, 10 % de montee puis cosinus.

Cible t : perte moyenne de l'original sur une vue forte, mesuree sur les 2800 images entrainees
(`measurements/v3_targets.json`) : 0,31 (aire 0,60), 0,45 (aire 0,50), 0,63 (aire 0,45), les trois
doublures du gold du banc.

Bras (4 = un noeud) : t = 0,31 ; t = 0,45 ; t = 0,63 ; t = 0,45 sans ancre (`retain_weight=0`).

sigma = 0,003 (ecart-type entre graines des references `bench-v4`) ; references V1 `ref4-v1`
(0,0637) et V3 `ref4-v3` (0,0798), meme graine 0.

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run) :
- `bench.score_min` du meilleur bras < V1 - 2 sigma (0,0577) : la methode simple ne suffit pas,
  les cibles de la V1 valent quelque chose ;
- ou le meilleur bras a `bench.retain` < 0,8 (plancher de la V5.1) ;
- ou le bras sans ancre a un `bench.retain` a 2 sigma pres de celui du bras t = 0,45 : l'ancre ne
  sert a rien, la retirer.

Lecture : si le meilleur bras depasse la V3 (> 0,0798 + 2 sigma ~ 0,086), les V2-V3 n'apportent rien
sur cette mesure ; s'il tombe entre V1 et V3, elles apportent cet ecart. La charniere donne la meme
perte a toutes les images : `final.forget_loss_std` doit rester bas, et c'est le terme d'ecart-type
de S_ms qui devrait limiter le score.

## Reglages

lr 1e-5, 3 passes, lots de 32, ancre `retain_weight` 1 a T = 2, vues de l'ancre d'aire 0,25-0,35
(hors des aires du banc), `holdout_fraction` 0,2.

## Metrique locale attendue

- `bench.score_min` >= 0,0637 pour au moins un bras ; `bench.retain` >= 0,8 ;
- `final.reached` proche de 1 (la cible est atteinte en 3 passes) ;
- Cout : ~1 min par variante sur la RTX 5080 (estime, non mesure).

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=v-dumb                      # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=v-dumb WHERE=jureca         # 4 variantes sur 4 A100 (file d'equipe)
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Groupe `20261122T032406Z-unl-v-dumb` (RTX 5080, commit 525b0bf, propre, 3 min 21), graine 0 :

| bras | `score_min` | retain | KL exacte | perte exacte (ecart-type) | `final.reached` | porte |
|---|---|---|---|---|---|---|
| t = 0,31 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,019) | 0,001 | 1 |
| t = 0,45 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,019) | 0,001 | 1 |
| t = 0,63 | 0,000003 | 1,000 | 0,0005 | 0,0013 (0,020) | 0 | 1 |
| t = 0,45 sans ancre | 0 | 0,006 | 6,53 | 12,23 (5,38) | 0,977 | 0 |

Contre les criteres de cette fiche :
- meilleur bras < V1 - 2 sigma (0,0577) : **oui, refutee** (0,000003) ;
- retain du meilleur bras < 0,8 : non (1,000), sans objet ;
- bras sans ancre a 2 sigma du retain du bras t = 0,45 : non (0,006 contre 1,000) : l'ancre sert.

Avec l'ancre, le modele ne bouge pas ; sans elle, il s'effondre (perte 27 fois la cible : la charniere
ne borne rien). Deux suivis de dose, poses avant (`trust-region-vdumb-campaign.md`) : poids d'ancre
0,3 / 0,1 / 0,03 et lr 1e-4 (`v-dumb-dose`), puis 0,01 / 0,003 / 0,001 (`v-dumb-dose2`). Meilleur run :
0,019 (w = 0,01, retain 0,987) ; de w = 0,01 a w = 0,003, on passe d'une KL exacte de 0,067 a
l'effondrement. Aucune dose utilisable a lr 1e-5. Le pronostic de la fiche (ecart-type des pertes bas)
ne se verifie pas : la charniere ne lie jamais, et l'ecart-type reste celui de l'oubli partiel.

## Decision

- [x] Abandonner : refutee a sa fiche, aucune dose utilisable (falaise entre l'immobilite et
  l'effondrement), la charniere ne borne pas l'oubli.
- [ ] Iterer
- [ ] Candidate a la soumission
