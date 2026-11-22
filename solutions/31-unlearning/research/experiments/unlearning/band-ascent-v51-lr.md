# band ascent V5.1 a lr 3e-5 (suivi S1)

- Identifiant : `band-ascent-v51-lr` (spec : `band-ascent-v51-lr.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22, apres `band-ascent-v51`, avant ce run
- Responsable : Alex

## Hypothese

Suivi S1 de `trust-region-v51-campaign.md`. A lr 1e-4 (`band-ascent-v51`), les trois bras en marge
font un arret dur au niveau 1 apres 15 a 20 pas : derive du temoin 0,5 a 1,0 (KL brute), 44 a 74 % de
la sonde au-dessus de sa borne, aucun niveau garde, modele inchange. La force d'ordre 1 de la marge
depasse la bande en quelques pas. La fiche `band-ascent-v51` prescrit de refaire a lr 3e-5 avant
d'abandonner.

Bras : V5.1 complete a lr 3e-5 ; sans retenue KL (`band_hold_kl=none`) et bande en perte
(`band_space=loss`) au meme lr, pour garder les lectures d'ablation a lr egal ; V5.1 complete a
lr 1e-5 (un cran plus bas, pour separer « trop rapide » de « instable a toute vitesse »). Le bras
`epochs=1` n'est pas repris (4 variantes au plus).

**Refutee si** (seuils de la fiche `band-ascent-v51`, reconduits tels quels, le 2026-11-22 avant ce
run) :
- le bras complet a lr 3e-5 termine en arret dur ou sans niveau garde : **abandon** de la V5.1 si le
  bras a lr 1e-5 fait de meme ; si seul le bras a 1e-5 garde un niveau, c'est lui qui est juge ;
- ou `bench.retain` < 0,8 au bras juge ;
- ou rapport audit/exact > 0,4 au bras juge ;
- ou `bench.score_min` du bras juge <= 0,0114 (bras 0,2 de `band-ascent-budget`, 0,0064, + 2 sigma).

Ajoute dans la feuille de campagne : pour compter comme methode qui score, > 0,0687 (V1 + 2 sigma)
avec retain >= 0,8 ; sinon pas de graines 1 et 2.

Lecture des ablations (comme dans `band-ascent-v51`) : `final.pushed_in_band` du bras complet >= 0,8
et >= bras `band_space=loss` + 0,2 ; audit/exact du bras sans retenue KL >= bras complet + 0,1.

## Reglages

Ceux de `band-ascent-v51` (marge, retenue KL 0,128, `cv_min_retain=0.8`, 2 passes par niveau, 4
niveaux, phi = 0,25, 4 vues d'aire 0,5, K = 5), lr 3e-5 (bras 4 : 1e-5).

## Lancement

```bash
make hk-run SPEC=band-ascent-v51-lr
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Groupe `20261122T055240Z-unl-band-ascent-v51-lr` (RTX 5080, commit ceae862, arbre propre, 5 min 11),
graine 0.

| bras | lr | niveaux gardes | KL exacte | audit/exact | `score_min` (recadr. / masq.) | retain | `cv_retain` (ecart) | `pushed_in_band` | `held_kl` |
|---|---|---|---|---|---|---|---|---|---|
| V5.1 complete | 3e-5 | 0 (arret dur au pas 40 du niveau 1, 57 % de la sonde au-dessus) | 0 | -- | 0 (modele intact) | 1 | -- | -- | -- |
| `band_hold_kl=none` | 3e-5 | 0 (niveau 1 acheve, retain du temoin 0,70 < 0,8) | 0 | -- | 0 (modele intact) | 1 | -- | -- | -- |
| `band_space=loss` | 3e-5 | 4 | 0,067 | 0,340 | 0,0385 (0,0555 / 0,0385) | 0,975 | 0,960 (-0,015) | 0,026 | 0,016 |
| V5.1 complete | 1e-5 | 1 (arret dur au niveau 2) | 0,339 | 0,839 | **0,0796** (0,0796 / 0,0898) | **0,725** | 0,861 (+0,136) | 0,071 | 0,299 |

Contre les seuils (poses avant ce run) :
- bras complet a 3e-5 sans niveau garde, bras a 1e-5 avec un niveau : c'est lui qui est juge ;
- `bench.retain` >= 0,8 : **non** (0,725) -> refutee ;
- audit/exact <= 0,4 : **non** (0,839, comme la V3 a 0,86) -> refutee ;
- `bench.score_min` > 0,0114 : tenu (0,0796) ; il depasse V1 + 2 sigma (0,0687) mais sous le plancher
  de retain : pas de graines 1 et 2 (regle de la feuille de campagne). A retain egal, SCRUB conforme
  fait bien mieux (lr 4e-4, graine 0 : 0,1279 a retain 0,734) ;
- lecture marge : `pushed_in_band` 0,071 (seuil 0,8) ; a lr egal (3e-5), la marge n'entraine pas
  (arret dur) quand la perte garde 4 niveaux : la famine n'est pas ce qui bloque, la marge depasse la
  bande ;
- lecture retenue KL : `held_kl` final 0,299, plus du double de la tolerance 0,128 : la charniere ne
  tient pas les images tenues a pleine echelle ;
- estimateur `cv_retain` a +-0,03 de `bench.retain` : **non** au bras juge (+0,136, optimiste ; le run
  final sur 2800 images derive bien plus que le sous-run sur 4 plis) ; tenu au bras en perte (-0,015).

Le bras en perte avec retenue KL a 3e-5 tient un retain de 0,975 et un score de 0,0385, plus que la V5
a KL exacte comparable (0,000 a 0,074, S_pred 0,00-0,05 contre 0,10-0,23 ici), mais reste sous la V1.

## Decision

- [x] Abandonner la V5.1 (2026-11-22)
- [ ] Graines 1 et 2 (retain < 0,8 : exclu par la regle posee avant) / [ ] Candidate a la soumission
