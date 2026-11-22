# scrub forget views (V4, SCRUB conforme)

- Identifiant : `scrub-forget-views` (spec : `scrub-forget-views.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22 ; revue a la fusion avec la campagne V3 (meme jour, avant tout run)
- Responsable : Alex

## Hypothese

`scrub` dose est la meilleure methode mesuree (`ref4-scrub`, lr 1e-4, `bench-v4` : 0,098 +- 0,022 ;
meilleur run 0,118 a lr 2e-4, `bench-v3`), et la plus locale (audit/exact 0,51 a 0,60, contre 0,86
a 0,91 pour les V2 et V3 retenues). Mais son ancre (min-steps) tient des images externes (COCO,
DIV2K, Kodak), que le sujet exclut pour une methode soumise : la meilleure methode conforme mesuree
reste la V2/V3 (0,0786 / 0,0798). L'option `retain_source=forget_views` (`methods.py`,
`_resolve_retain`, `_retain_batch`) remplace ces images par des vues fortes du forget set lui-meme
(recadrage de 25 a 35 % de l'aire, puis couleur), tenues aux sorties de l'original : l'ascension
plafonnee efface la pointe de memorisation sur l'image exacte, l'ancre garde la fonction autour
d'elle. On croit que la methode en tete reste en tete une fois conforme.

**References deja mesurees** (`bench-v4`, aucun nouveau run necessaire) : `ref4-scrub` (lr 1e-4,
graines 0-2 : 0,1024 ; 0,0739 ; 0,1171), V3 retenue (0,0798 +- 0,0012), V2 (0,0786 +- 0,0030) ;
sigma = 0,003 (ETAT.md) pour les references conformes, 0,022 pour `scrub` (bruite).

**Refutee si** (protocole `bench-v4`) :
- la moyenne des deux graines au meilleur lr <= 0,0798 + 2 x 0,003 = 0,086 : pas mieux que la V3
  retenue, la meilleure methode conforme ;
- ou une graine a `bench.retain` < 0,80 ou `bench.gate_passed` = 0.

**Mecanisme** : audit/exact (`bench.audit_kl` / `bench.groups.forget_exact.kl_orig_sub`) <= 0,7
(scrub proxy : 0,51-0,60) et `bench.audit_kl` <= 0,12.

**Risque a lire** : l'ancre sur des vues des memes images peut defaire l'oubli. Si la KL des images
exactes tombe sous 0,7 x celle de `ref4-scrub` a la meme graine (lr 1e-4 : 0,128 ; 0,088), c'est ce
qui se passe.

**Circularite** : les vues de l'ancre ne recouvrent aucune aire du banc (doublures de 0,40 a 0,70,
vue d'audit 0,50), mais la couleur est proche du jitter du banc, et les max-steps de scrub
s'entrainent sur les quatre familles d'alterations du banc (`random_augment`). Si le gain sur les
recadrages depasse celui sur les masquages de plus de 2 sigma, trancher sur
`bench.score_min_family.mask` ; comparer aussi `bench.exact.mean_std.*` et
`bench.exact.prediction.*` (images exactes seules, hors score).

Un positif se confirme sur trois graines avant d'entrer au rapport. Seuils fixes a priori, le
2026-11-22, avant tout run.

## Reglages

Ceux de `scrub` (`configs.py`) ; seuls changent `retain_source`, `lr` (les deux doses qui encadrent
le meilleur scrub proxy) et `seed` (deux graines : scrub est bruite).

## Lancement

```bash
make hk-run SPEC=scrub-forget-views                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=scrub-forget-views WHERE=jureca    # 4 variantes sur 4 A100 (file d'equipe)
```

Sur un autre GPU que la RTX 5080, rejouer d'abord `ref4-scrub` sur la meme machine (ETAT.md §2).
Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Quatre runs, RTX 5080, `bench-v4`, commit 52fa1d3, code propre (groupe
`20261122T013331Z-unl-scrub-forget-views`). Un premier passage (groupe `20261122T012752Z`) avait
tourne avec un fichier de test modifie (drapeau `dirty`) ; rejoue propre, il redonne les memes
chiffres au bit pres (seul `train_seconds` differe). Ses lignes restent au registre, marquees
`dirty` (trace d'audit) ; seuls les runs propres sont cites ici.
Reference proxy rejouee de meme (`ref4-scrub`, groupe `20261122T013637Z`) : identique au bit pres
aux runs d'origine, avec en plus `bench.exact.*`.

| lr | graine | score_min | pire recadrage | pire masquage | retain | KL audit | KL exacte | audit/exact | V3 meme graine |
|---|---|---|---|---|---|---|---|---|---|
| 1e-4 | 0 | 0,0211 | 0,0211 | 0,0217 | 0,992 | 0,008 | 0,035 | 0,23 | 0,0784 |
| 1e-4 | 1 | 0,0137 | 0,0142 | 0,0137 | 0,994 | 0,006 | 0,028 | 0,22 | 0,0804 |
| 2e-4 | 0 | 0,0698 | 0,0747 | 0,0698 | 0,956 | 0,041 | 0,088 | 0,47 | 0,0784 |
| 2e-4 | 1 | 0,0516 | 0,0539 | 0,0516 | 0,961 | 0,039 | 0,077 | 0,51 | 0,0804 |
| proxy 1e-4 (`ref4-scrub`) | 0 / 1 | 0,1024 / 0,0739 | 0,1024 / 0,0739 | 0,1055 / 0,0754 | 0,929 / 0,955 | 0,071 / 0,045 | 0,128 / 0,088 | 0,55 / 0,51 | |

Lecture contre les seuils poses avant :

- **Hypothese refutee aux doses de la fiche.** Au meilleur lr (2e-4), moyenne 0,0607 sur les deux
  graines, sous 0,086 et sous la V3 appariee (-0,0086 et -0,0288). Retain >= 0,956 et
  `bench.gate_passed` = 1 partout : pas d'effondrement.
- **Mecanisme confirme** : audit/exact 0,22 a 0,51 (<= 0,7) et `bench.audit_kl` <= 0,041 (<= 0,12).
  La version conforme est plus locale que le proxy (0,51 a 0,60).
- **Le risque se realise a 1e-4** : KL exacte 0,035 et 0,028, soit 0,27 et 0,31 fois le proxy de meme
  graine (seuil 0,7) : l'ancre sur les vues des memes images defait l'oubli. A 2e-4 : 0,69 et 0,87.
- **Circularite** : pas de gain propre au recadrage (ecart recadrage - masquage <= 0,005). Sur les
  images exactes seules, la composante des predictions vaut 0,22 a 0,27 a 2e-4 (`bench.exact.prediction.*`),
  contre 0,35 a 0,41 pour le proxy a 1e-4 : le deficit de la version conforme n'est pas un effet des
  copies, il porte sur la direction des erreurs.
- La question de dose reste ouverte (le score triple de 1e-4 a 2e-4, l'ancre tient encore) : suivi 1.

## Suivi 1 : `scrub-fv-dose` (pose le 2026-11-22, apres les quatre runs, avant le suivi)

**Ce que mesurent les quatre runs** (graines 0 et 1) : a lr 1e-4, `score_min` 0,021 et 0,014 ; a 2e-4,
0,070 et 0,052. Le score triple d'une dose a l'autre sur les deux graines, et a 2e-4 l'ancre tient
encore largement : retain 0,956 et 0,961, `bench.audit_kl` 0,041 et 0,039, quand le `scrub` proxy, au
meme lr 2e-4, en est deja a retain 0,663 et KL d'audit 0,423 (`scrub-dose-fine`, `bench-v3`), puis
s'effondre a 3e-4 (retain 0,405, KL d'audit 0,960). L'ancre sur vues du forget set deplace la
courbe de dose vers la droite : le meilleur lr de la version proxy n'est pas celui de la version
conforme, et l'hypothese de la fiche, posee aux doses du proxy, ne dit rien des doses superieures.

**Hypothese** : la version conforme a son optimum au-dela de 2e-4 et y depasse la V3 retenue.

**Bras** : lr 3e-4 (point d'effondrement du proxy) et 4e-4 (suite geometrique 1, 2, 4), graines 0 et
1, appariees aux quatre runs ci-dessus et a la V3 (`ref4-v3`).

**Seuils (ceux de la fiche, inchanges)** :
- candidat si la moyenne des deux graines a un lr > 0,086, avec retain >= 0,80, `bench.gate_passed` = 1
  et `bench.audit_kl` < 0,34 sur les deux graines ; il se confirme alors sur la graine 2 (moyenne des
  trois > 0,086, gain sous le masquage au-dela de 2 sigma sur la V3 appariee) ;
- refutee si la moyenne reste <= 0,086 aux deux lr (la version conforme ne bat pas la V3 entre 1e-4 et
  4e-4) ;
- effondrement (comme le proxy a 3e-4) si retain < 0,80 ou `bench.gate_passed` = 0 a une dose ; il
  borne alors la courbe.

**Mecanisme a suivre** : audit/exact <= 0,7 ; et la composante des predictions, qui limite la
version conforme (`bench.exact.prediction.*` 0,22 a 0,27 a 2e-4, contre 0,35 a 0,41 pour le proxy a
1e-4, rejeu `ref4-scrub`) : monte-t-elle avec la dose, ou seule la composante des pertes monte-t-elle ?


### Resultat du suivi 1 (groupe `20261122T013912Z-unl-scrub-fv-dose`, code propre)

| lr | graine | score_min | pire recadrage | pire masquage | retain | KL audit | KL exacte | audit/exact | V3 meme graine |
|---|---|---|---|---|---|---|---|---|---|
| 3e-4 | 0 | 0,1151 | 0,1229 | 0,1151 | 0,868 | 0,132 | 0,165 | 0,80 | 0,0784 |
| 3e-4 | 1 | 0,1015 | 0,1098 | 0,1015 | 0,850 | 0,155 | 0,166 | 0,93 | 0,0804 |
| 4e-4 | 0 | 0,1279 | 0,1396 | 0,1279 | 0,734 | 0,284 | 0,277 | 1,03 | 0,0784 |
| 4e-4 | 1 | 0,1141 | 0,1253 | 0,1141 | 0,731 | 0,286 | 0,261 | 1,09 | 0,0804 |

- **lr 3e-4 : candidat.** Moyenne 0,1083 > 0,086 ; retain 0,868 et 0,850 (>= 0,80) ; `gate_passed` = 1 ;
  KL d'audit 0,13 et 0,16 (< 0,34). Gain apparie sur la V3 : +0,037 et +0,021, sous le masquage comme
  sous le recadrage (pas de gain propre a la famille des vues). Il bat aussi le `scrub` proxy a son
  meilleur lr, aux memes graines (0,1024 et 0,0739).
- **lr 4e-4 : ecarte par la regle posee avant.** Meilleure moyenne (0,121), mais retain 0,734 et 0,731,
  sous le plancher de 0,80 de la fiche (niveau de la V3 : 0,74-0,76).
- **Le mecanisme de localite ne tient pas a ces doses** : audit/exact 0,80 et 0,93 a 3e-4 (seuil 0,7),
  1,03 et 1,09 a 4e-4 (le fold d'audit bouge autant que les images exactes, ou plus). Il ne tenait
  qu'aux doses ou l'ancre defait l'oubli (1e-4, 2e-4). Le gain sur la V3 passe par le terme d'oubli :
  composante des predictions 0,27 a 0,39 sous les six doublures (V3 : 0,23 a 0,38), composante des
  pertes 0,33 a 0,66 (V3 : 0,34 a 0,47), et par un retain plus haut a mouvement d'audit egal (KL exacte
  0,165 contre 0,28-0,30 pour la V3).
- La composante des predictions monte avec la dose : sur les images exactes, 0,31 a 0,41 a 3e-4,
  le niveau du proxy (0,35 a 0,41), contre 0,22 a 0,27 a 2e-4.

## Suivi 2 : `scrub-fv-seeds` (pose le 2026-11-22, apres le suivi 1, avant le run)

**Bras** : graine 2 a lr 3e-4 (confirmation du candidat) et graine 2 a lr 4e-4 (troisieme graine du
bras a la meilleure moyenne, ecarte par le plancher de retain : dit ce que coute ce plancher, sans
changer la decision).

**Seuils (ETAT.md et fiche, inchanges)** : le candidat lr 3e-4 est confirme si la moyenne des trois
graines > 0,086, le gain apparie sur la V3 sous le masquage depasse 2 sigma (0,006) en moyenne, et
retain >= 0,80, `bench.gate_passed` = 1 sur la graine 2. Infirme sinon. Le bras 4e-4 reste ecarte si
une graine a retain < 0,80, quelle que soit sa moyenne.

### Resultat du suivi 2 (groupe `20261122T015421Z-unl-scrub-fv-seeds`, code propre)

| lr | graine 2 : score_min | pire masquage | retain | KL audit | audit/exact | V3 graine 2 |
|---|---|---|---|---|---|---|
| 3e-4 | 0,1090 | 0,1090 | 0,859 | 0,140 | 0,82 | 0,0806 |
| 4e-4 | 0,1202 | 0,1202 | 0,732 | 0,284 | 1,03 | 0,0806 |

Trois graines appariees (0-2), `bench-v4` :

| reglage | score_min | pire recadrage | pire masquage | retain | KL audit | audit/exact | conforme |
|---|---|---|---|---|---|---|---|
| **`scrub`, `retain_source=forget_views`, lr 3e-4** | **0,1085 +- 0,0068** | 0,1155 | 0,1085 | 0,859 | 0,142 | 0,85 (0,80-0,93) | oui |
| `scrub`, `forget_views`, lr 4e-4 (retain < 0,80) | 0,1208 +- 0,0069 | 0,1317 | 0,1208 | 0,733 | 0,284 | 1,05 | oui |
| V3 retenue (`ref4-v3`) | 0,0798 +- 0,0012 | 0,0875 | 0,0798 | 0,744 | 0,261 | 0,89 | oui |
| `scrub` proxy lr 1e-4 (`ref4-scrub`) | 0,098 +- 0,022 | 0,098 | 0,101 | 0,931 | 0,067 | 0,55 | non |

- **Candidat confirme.** lr 3e-4 : moyenne 0,1085 > 0,086 ; gain apparie sur la V3 sous le masquage
  +0,037, +0,021, +0,028 (moyenne +0,029, soit pres de 10 sigma) ; retain >= 0,850 et
  `bench.gate_passed` = 1 sur les trois graines. **+36 % sur la V3 retenue**, sous les deux familles.
- Contre le `scrub` proxy (non conforme) aux memes graines : +0,013, +0,028, -0,008 (moyenne +0,011),
  dans le bruit du proxy (sigma 0,022) : la version conforme fait au moins aussi bien.
- lr 4e-4 : +0,012 de plus a chaque graine (0,1208 +- 0,0069), mais retain 0,73 sur les trois : reste
  ecarte par le plancher pose avant. Le plancher coute donc environ 11 % de score au banc ; il garde
  un retain superieur de 0,12 a celui de la V3.
- **Mecanisme** : a la dose retenue, audit/exact 0,85 (0,80 a 0,93), au niveau de la V3 (0,89) : la
  version conforme ne gagne pas par la localite. Elle gagne par l'efficacite de son oubli : a KL
  exacte moitie moindre (0,17 contre 0,29), composantes des pertes et des predictions au moins egales
  a celles de la V3, et un retain plus haut de 0,12.

## Decision

- [x] Candidate a la soumission, comme meilleure methode conforme mesuree : `scrub`,
  `retain_source=forget_views`, lr 3e-4 (0,1085 +- 0,0068 sur trois graines appariees, `bench-v4`).
  Avant soumission (decision d'equipe) : run final sans `holdout_fraction` (3500 images, donc plus de
  pas par epoque : la dose peut bouger) ; la dose entre 3e-4 et 4e-4 reste a trancher (score contre
  retain) ; seul le classement public dira ce qui se transfere au vrai gold.
- [ ] Abandonner / [ ] Iterer
