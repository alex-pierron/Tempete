# V_Dumb2 (la charniere de V_Dumb dans le squelette de SCRUB conforme)

- Identifiant : `v-dumb2` (spec : `v-dumb2.toml`) ; etape 0 : `band-ascent-step0` (cle `level_ceiling`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

V_Dumb (`v-dumb.md`) avait la bonne cible et la mauvaise dynamique. La bonne cible : sans savoir sur
quelles images le gold doute (la V5 a montre qu'aucune selection tiree de vues ne le sait), un meme
niveau de perte pour toutes les images est ce que demande la composante des predictions (KL directe
vers un melange : la meilleure prediction constante est la moyenne), et les doublures sont moins sures
sur *toutes* les images (V5). La mauvaise dynamique, mesuree : montee et ancre sommees dans le meme
pas (falaise entre l'immobilite et l'effondrement), charniere d'un seul cote (le mouvement commun
emporte les images arrivees, perte 27 fois la cible), force en perte qui s'eteint sur une image
memorisee (~3e-4) puis s'emballe, et un taux de 1e-5, trente fois sous la dose de SCRUB conforme.

SCRUB conforme (`scrub-forget-views.md`, 0,1085 +- 0,0068) a la dynamique qui marche (une passe de
montee plafonnee, puis des passes d'ancre sur des vues du forget set), mais son plafond porte sur la
KL *moyenne du lot* (2, jamais atteint) : il concentre l'oubli sur peu d'images (coefficient de
variation des pertes 7,5 a 3e-4, 5,2 a 4e-4, graine 0) et en laisse la plupart a l'original.

V_Dumb2 = SCRUB conforme dont le max-step est remplace par la charniere de V_Dumb, par image :
`scrub_target_loss` = t (`methods.py`, `level_band_loss`). Sur chaque image exacte et sa copie
alteree, la marge de la pseudo-etiquette m = z_y - logsumexp(autres) (CE = softplus(-m)) est poussee
avec une force d'ordre 1 jusqu'a CE = t, laissee libre jusqu'a `band_tolerance` x t, ramenee au-dela
(la bande de la V5.1 en marge, `band_sharpness` 10). Les min-steps (ancre sur des vues fortes du
forget set) et le calendrier sont ceux de SCRUB conforme. Rien hors du forget set : conforme au sujet.

On croit que, a dose egale, l'arret par image depense la derive sur les images en retard plutot que sur
celles deja au-dela du gold : oubli plus uniforme (coefficient de variation plus bas), composante des
predictions plus haute, et retain tenu a 4e-4 ou SCRUB conforme tombe a 0,73.

Risque connu, mesure depuis par la campagne V5.1 (`trust-region-v51-campaign.md`, avant tout run de
cette spec) : a pleine echelle, la bande en marge depasse son plafond en 15 a 40 pas des lr 3e-5, et la
V5.1 s'arrete au premier niveau sans en garder. Ici, ni arret dur ni niveaux : seuls le rappel et les
passes d'ancre la retiennent. Les seuils ci-dessous ne changent pas.

## Etape 0 (sans entrainement, porte de la spec)

`make hk-run SPEC=band-ascent-step0` : `diagnose` rend desormais `level_ceiling`, l'oubli du banc
(S_ms x S_pred) sur les seules images exactes quand chaque marge est ramenee exactement dans la bande
[t, 2t] par la seule pseudo-etiquette (classes fausses a la forme de l'original, rien d'autre ne bouge,
retain 1), pour t dans {0,10 ; 0,20 ; 0,31 ; 0,45 ; 0,63 ; 1,00}. C'est un plafond : ce que la bande
donnerait si elle etait atteinte sans derive.

Reference de meme portee : SCRUB conforme, graine 0, images exactes seules (`bench.exact.*`), pire des
six doublures : 0,155 a 3e-4 (m30), 0,198 a 4e-4 (m30).

**Ne pas lancer `v-dumb2` si** `level_ceiling.t045.min_mask` < 0,155 : meme atteinte exactement, la
bande oublierait moins, sous le pire masquage, que SCRUB conforme a sa dose retenue. Si un autre t
depasse t = 0,45 de plus de 20 % sous le pire masquage, le noter pour un suivi ; cette spec garde
t = 0,45 (la cible de V_Dumb, perte moyenne de l'original sur une vue d'aire 0,50).

## Bras (4 = un noeud), graine 0, t = 0,45

| bras | lr | `band_tolerance` | question |
|---|---|---|---|
| a | 3e-4 | 2 | meilleur que SCRUB conforme a sa dose retenue ? |
| b | 4e-4 | 2 | le retain tient-il la ou SCRUB conforme passe a 0,73 ? |
| c | 3e-4 | 1e6 (aucun rappel) | le rappel au-dela du plafond sert-il ? |
| d | 4e-4 | 1e6 (aucun rappel) | idem, a la dose forte |

References appariees, deja mesurees (`scrub-fv-dose`, graine 0, aucun nouveau run) :

| SCRUB conforme | `score_min` | retain | KL exacte (T=2) | KL d'audit | CV des pertes |
|---|---|---|---|---|---|
| lr 3e-4 | 0,1151 | 0,868 | 0,165 | 0,132 | 7,5 |
| lr 4e-4 | 0,1279 | 0,734 | 0,277 | 0,284 | 5,2 |

sigma = 0,0068 (ecart-type entre graines de SCRUB conforme a 3e-4, `ref4-scrub-fv`).

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run) :
- aucun bras a `bench.retain` >= 0,8 n'atteint `bench.score_min` >= 0,1151 + 2 sigma = 0,129 : pas
  mieux que SCRUB conforme apparie, la charniere n'apporte rien ;
- ou le bras b a `bench.retain` < 0,8 : la bande ne tient pas le retain a la dose ou SCRUB conforme
  le perd.

**Mecanisme refute si** le meilleur bras a un coefficient de variation des pertes des images exactes
(`bench.groups.forget_exact.loss_std_sub / loss_mean_sub`) au-dessus de la moitie de celui de SCRUB
conforme au meme lr (3,7 a 3e-4, 2,6 a 4e-4) : la bande n'egalise pas l'oubli, et un gain eventuel ne
vient pas de la raison avancee.

**Rappel inutile si** les bras c et d sont chacun a moins de 2 sigma (0,0136) du score et a moins de
0,02 du retain de leur bras avec plafond (a, b) : le retirer (`band_tolerance` 1e6 dans les suivis).

Suivis poses maintenant, avant tout run : (1) si un bras passe le premier critere, ses graines 1 et 2
(confirmation : moyenne > 0,122 sur trois graines, le seuil d'amelioration de `ref4-scrub-fv`, retain
>= 0,8 a chaque graine) ; (2) si le bras b tient le retain avec un score au-dessus du bras a, le meme
reglage a 5e-4 (une dose, la mesure en donnant la direction). Rien d'autre sans nouvelle fiche.

Lecture : `bench.score_min_family.mask` fait foi (aucune cible tiree de vues ici, mais t est calibree
sur s50) ; lire `final.in_band` (part des images dans [t, 2t] a la fin, apres les passes d'ancre) et
`bench.audit_kl / bench.groups.forget_exact.kl_orig_sub` (localite, a KL exacte egale avec SCRUB).

## Reglages

`scrub`, `retain_source=forget_views`, `scrub_target_loss` 0,45, `band_sharpness` 10, 3 passes dont
une de montee (`scrub_msteps` 1), copies alterees ponderees 1 (`altered_weight`), lots de 32, 10 % de
montee du taux puis cosinus, `holdout_fraction` 0,2. Option neutre par defaut : `scrub` sans
`scrub_target_loss` redonne SCRUB V4 au bit pres (verifie sur le petit modele de test, trois reglages).

## Metrique locale attendue

- `bench.score_min` >= 0,129 pour au moins un bras a retain >= 0,8 ;
- coefficient de variation des pertes exactes < 3 ;
- Cout : celui de SCRUB conforme (~45 s par variante sur la RTX 5080, estime, non mesure).

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=band-ascent-step0           # etape 0, sans entrainement (GPU CUDA ; lourd sur Mac)
make hk-run SPEC=v-dumb2                      # GPU CUDA (RTX 5080) : full ; Mac : smoke
make hk-run SPEC=v-dumb2 WHERE=jureca         # 4 variantes sur 4 A100 (file d'equipe, un noeud)
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

(a remplir apres le run)

## Decision

- [ ] Abandonner
- [ ] Iterer
- [ ] Candidate a la soumission
