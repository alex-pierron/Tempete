# band ascent (V5)

- Identifiant : `band-ascent` (spec : `band-ascent.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

V5 (`band_ascent`, `methods.py`, rapport section V5). Au lieu de descendre vers une cible, une
montee de gradient bornee image par image : les 25 % d'images dont l'original doute le plus sur des
vues fortes montent jusqu'a t_i = rho x (leur perte sur les vues) et restent sous
h_i = 2 t_i + 0,05 ; les autres restent sous 0,05 (borne haute des images tenues). Perte : deux
charnieres softplus (beta = 10). La dose rho monte par 4 niveaux ; une validation croisee a 5 plis
(dans les 80 % entraines) garde le dernier niveau dont la derive du pli temoin, corrigee de
l'echelle ((5/4)^2), reste sous 0,1 avec un accord >= 0,97 ; le run final monte ces niveaux sur
toutes les images. Arret dur au double du budget ou si plus de la moitie d'une sonde passe
au-dessus de sa borne.

Le modele lineaire du rapport (eq. v5ratio, v5hold) predit : avec la borne haute, rapport
audit/exact ~ chi^2 phi ~ 0,22 ; sans elle, ~0,86 (presque la V1, 0,87). Mesure exploratoire
(M2, 320 images, une graine) : la montee plafonnee est ~15 fois plus locale que la V1 a perte egale,
mais sa fenetre locale se ferme en moins de 10 pas a lr 1e-4.

Bras (4 = un noeud) : V5 par defaut ; `band_release=1` (sans selection) ; `band_floor=none` (sans
borne haute des images tenues) ; `cv_folds=0` (les 4 niveaux, sans choix hors pli).

sigma = max(ecart-type entre graines de la reference en `bench-v4`, 0,0025) ; references V1
(epsilon = 0,15) et `scrub` lr 1e-4 rebenchees en `bench-v4`.

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run) :
- rapport audit/exact (`bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`) du bras par
  defaut > 0,5 : la fenetre locale n'existe pas a 2800 images ;
- ou ce rapport n'est pas au moins divise par deux par rapport au bras `band_release=1`, a perte
  moyenne des images exactes comparable (+-30 %) : la selection ne localise pas ;
- ou le bras par defaut termine en arret dur (`final.hard_stop` = 1) ou sans niveau retenu
  (`final.cv_stop_level` = 0) : la bande s'effondre ou derive a lr 1e-4 ; refaire a lr 3e-5 avant
  d'abandonner ; abandon si elle s'effondre encore ;
- ou `bench.score_min` du meilleur bras <= reference V1 + 2 sigma ;
- ou `bench.gate_passed` = 0.

Lecture du modele : si le bras `band_floor=none` est aussi local que le bras par defaut, c'est le
modele lineaire qui est refute, pas la methode. Estimateur : `final.cv_monitor_kl` x `final.cv_scale`
doit tomber a +-30 % de `bench.audit_kl` (sinon la derive du run final, qui n'a pas de fold d'audit,
est mal estimee).

## Reglages

lr 1e-4 (dix fois la V1, le regime de la fenetre locale de la sonde), lots de 32 stratifies, une
passe par niveau, 4 niveaux (rho = 0,25 a 1), phi = 0,25, kappa = 2, delta_0 = 0,05, beta = 10,
4 vues d'aire 0,50, K = 5, Lambda_max = 0,1, a_min = 0,97, nu = 2, controle tous les 5 pas.

## Metrique locale attendue

- rapport audit/exact du bras par defaut <= 0,5, et <= 0,5 x celui du bras `band_release=1` ;
- `final.cv_stop_level` >= 1 et `final.hard_stop` = 0 ;
- `bench.retain` >= 0,85.
- Cout : ~1,5 min par variante sur la RTX 5080 (estime, non mesure).

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=band-ascent-step0                # d'abord : sans entrainement
make hk-run SPEC=band-ascent                      # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=band-ascent WHERE=jureca         # 4 variantes sur 4 A100 (file d'equipe)
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

**Etape 0 (2026-11-22)** : `band-ascent-step0` a refute la V5 sur ses deux criteres, sous les trois
masquages (selection au hasard : precision egale au taux de base, part de la perte 0,23 a 0,24 pour un
seuil de 0,5). La regle de la fiche interdisait alors de lancer cette specification sous cette forme.

**Lancee ensuite comme mesure de la localite seule**, sur decision de l'utilisateur (criteres de
localite de cette fiche gardes, criteres de score abandonnes ; detail et suivis :
`trust-region-v5-campaign.md`, suivis L et L2). Groupe `20261122T030333Z-unl-band-ascent` (commit
4185ff2, propre, 2 min 50), graine 0 :

| bras | niveaux | KL exacte | perte exacte | audit/exact | `score_min` (recadr. / masq.) | retain |
|---|---|---|---|---|---|---|
| defaut | 1 (CV : arret dur au niveau 2) | 0,074 | 0,109 | 0,246 | 0,000 (0,005 / 0,000) | 0,981 |
| `band_release=1` | 1 | 0,107 | 0,161 | 0,354 | 0,042 (0,042 / 0,062) | 0,958 |
| `band_floor=none` | 0 (CV : arret dur au pas 40 du niveau 1) | 0 | -- | -- | 0 (modele intact) | 1 |
| `cv_folds=0` | 4 | 0,213 | 0,286 | 0,538 | 0,017 (0,077 / 0,017) | 0,878 |

Contre les criteres de cette fiche :
- audit/exact du bras par defaut <= 0,5 : **tenu** (0,246 ; graine 1 : 0,393) ;
- divise par deux par rapport a `band_release=1` a perte comparable : **non** (0,69 ici, a perte
  differente de 32 % ; au suivi L2, a perte comparable et 4 niveaux : 0,60) -> la selection ne localise
  pas ; a KL exacte egale, phi = 1 est aussi local que phi = 0,25 ;
- `final.hard_stop` = 0 et `final.cv_stop_level` >= 1 : **tenu** (niveau 1 ; graine 1 : niveau 2) ;
  pas de reprise a lr 3e-5 ;
- `bench.score_min` du meilleur bras > V1 + 2 sigma (0,0687) : **non** (0,042 ; meilleur des huit runs
  V5 : 0,046) ;
- `bench.gate_passed` = 1 : **tenu** (sauf le bras sans borne haute a 4 niveaux du suivi L2, effondre) ;
- lecture du modele : `band_floor=none` n'a pas de modele ici ; a 4 niveaux (suivi L2), il s'effondre
  (audit/exact 1,01) : la borne haute est indispensable ;
- estimateur `cv_monitor_kl` x `cv_scale` a +-30 % de `bench.audit_kl` : **non** (2,2 fois trop haut ;
  1,5 et 1,8 fois sur les deux autres bras qui en ont un) ;
- `bench.retain` >= 0,85 : tenu aux niveaux gardes par la validation croisee (0,95 a 0,98).

Le score est nul par la composante des predictions (S_pred 0,00 a 0,05 au bras par defaut, S_ms 0,41 a
0,68) : ne bouger qu'une part des images ne rapproche pas des doublures, moins sures que l'original sur
toutes les images.

## Decision

- [x] Abandonner la V5 : selection refutee (etape 0, puis phi sans effet sur la localite a KL egale),
  score sous la V1 sur les huit runs. Localite obtenue (la plus forte mesuree a KL exacte egale), mais
  sans score.
- [ ] Iterer
- [ ] Candidate a la soumission
