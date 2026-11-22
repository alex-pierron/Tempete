# band ascent budget (V5 sous le plancher retain >= 0,8, premier essai de la V5.1)

- Identifiant : `band-ascent-budget` (spec : `band-ascent-budget.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

Objectif fixe le 2026-11-22 : tenir le terme retain officiel (accord x 1/(1 + KL) a T = 2) au-dessus
de 0,8 et depenser ce budget de derive pour oublier davantage, vers le gold et pas au-dela (rapport,
section V5.1). La V5 (`band_ascent`) par defaut vise un retain d'environ 0,88 (cv_max_drift 0,1,
accord 0,97). Sans code nouveau, ce run ne change que le budget du pli temoin : avec
cv_min_agreement = 0,97, le retain estime du temoin reste >= 0,97 / (1 + cv_max_drift), soit
0,88 / 0,81 / 0,75 pour 0,1 / 0,2 / 0,3. Les bandes restent plafonnees a rho <= 1.

Le rapport (V5.1, defaut 1) predit que la montee en perte est affamee (force <= sqrt(2) L, perte de
l'original 3,2e-4 en moyenne) : un budget plus large risque de ne pas etre consomme. Le bras 0,1
est le bras par defaut de `band-ascent` (meme graine, memes lots) : il sert d'appariement.

sigma = max(ecart-type entre graines de la reference en `bench-v4`, 0,0025) ; seuil 2 sigma ~ 0,006.

**Refutee si** (seuils fixes a priori, le 2026-11-22, avant tout run) :
- `bench.retain` < 0,8 au bras 0,2 : le temoin n'estime pas le retain du banc, la contrainte n'est
  pas tenue par la selection ;
- ou `bench.score_min` du bras 0,2 ne depasse pas celui du bras 0,1 de 2 sigma : depenser le
  budget ne rapporte rien sous cette forme ;
- ou `final.cv_stop_level` est le meme aux bras 0,1 et 0,2 : le budget n'est pas consomme ; lecture
  attendue si la famine domine, et la suite est `band_space=margin` (V5.1, correctif 1), pas un
  budget plus large.

Lecture de l'estimateur : `final.cv_agreement / (1 + final.cv_scale x final.cv_monitor_kl)` doit
tomber a +-0,03 de `bench.retain` (sinon le retain du temoin ne predit pas celui du banc, et
`cv_min_retain` ne vaut rien sous cette forme). Le bras 0,3 encadre la contrainte du cote agressif :
son `bench.retain` doit passer sous 0,8 ; s'il reste au-dessus avec un meilleur score, le facteur
d'echelle gamma_n est trop prudent.

## Reglages

Ceux de `band-ascent` (V5 par defaut) : lr 1e-4, lots de 32 stratifies, une passe par niveau,
4 niveaux (rho = 0,25 a 1), phi = 0,25, 4 vues d'aire 0,50, K = 5, accord minimal 0,97. Seul
`cv_max_drift` change : 0,1 / 0,2 / 0,3 (arret dur au double : 0,2 / 0,4 / 0,6).

## Metrique locale attendue

- `bench.retain` >= 0,8 au bras 0,2 et `final.cv_stop_level` plus haut qu'au bras 0,1 ;
- `bench.score_min` du bras 0,2 > bras 0,1 + 2 sigma, avec un gain sous le masquage
  (`bench.score_min_family.mask`) ;
- toujours lire `bench.audit_view_kl` : la derive sur des entrees non memorisees vaut environ deux
  fois celle du fold d'audit dans les runs mesures (V1 : 0,45-0,52 contre 0,20-0,26).
- Cout : ~1,5 min par variante sur la RTX 5080 (estime, non mesure).

Le score reel (distance au gold) n'est PAS mesure ici. Aucune soumission (decision du 2026-11-21).

## Lancement

```bash
make hk-run SPEC=band-ascent-budget                # RTX 5080 (full)
make hk-run SPEC=band-ascent-budget WHERE=jureca   # 3 A100 sur un noeud (file d'equipe)
```

Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Groupe `20261122T050945Z-unl-band-ascent-budget` (RTX 5080, commit ceae862, arbre propre, 3 min 36 au
total, 62 degres au plus), graine 0. Detail et suivis : `trust-region-v51-campaign.md`.

| `cv_max_drift` | niveaux gardes | KL exacte | perte exacte | audit/exact | `score_min` (recadr. / masq.) | retain | `cv_retain` (ecart au banc) | `pushed_in_band` |
|---|---|---|---|---|---|---|---|---|
| 0,1 | 1 | 0,074 | 0,109 | 0,246 | 0,000 (0,005 / 0,000) | 0,981 | 0,956 (-0,025) | 0,006 |
| 0,2 | 3 | 0,181 | 0,266 | 0,441 | 0,006 (0,038 / 0,006) | 0,916 | 0,878 (-0,038) | 0,016 |
| 0,3 | 4 | 0,213 | 0,286 | 0,538 | 0,017 (0,077 / 0,017) | 0,878 | 0,795 (-0,083) | 0,046 |

- Bras 0,1 = bras par defaut de `band-ascent` **au bit pres** (192 metriques communes identiques, hors
  `train_seconds` ; seules s'ajoutent les trois cles V5.1) : la fusion n'a pas change la V5. Le bras
  0,3 garde les 4 niveaux et redonne au bit pres le bras `cv_folds=0` de `band-ascent` (187 metriques).
- `bench.retain` >= 0,8 au bras 0,2 : **tenu** (0,916).
- `score_min` du bras 0,2 > bras 0,1 + 2 sigma (0,005) : **tenu de justesse** (0,0064), mais les deux
  sont presque nuls (V1 : 0,064) ; S_pred sous les masquages 0,01 a 0,10.
- `cv_stop_level` different aux bras 0,1 et 0,2 : **tenu** (1 contre 3) : le budget est consomme.
- Estimateur `cv_retain` a +-0,03 de `bench.retain` : tenu au bras 0,1 seulement ; le temoin est
  pessimiste, de plus en plus avec la dose (-0,038, puis -0,083).
- Bras 0,3 : retain 0,878, au-dessus de 0,8 (prevu dessous), avec un meilleur score : le facteur
  d'echelle gamma_n est trop prudent (lecture de la fiche).
- Famine : `pushed_in_band` 0,006 a 0,046 (part des images poussees dans leur bande) ; meme a 4
  niveaux, 95 % des images liberees n'atteignent pas leur bande. C'est la lecture qui renvoie au
  correctif 1 (`band_space=margin`).

Aucun critere de refutation n'est atteint, mais le score reste nul : depenser le budget sous cette
forme ne rapproche pas des doublures (0,017 au mieux, sous la V1 de 0,047).

## Decision

- [ ] Abandonner / [x] Coder la V5.1 (marge, retenue KL, cv_min_retain) : deja codee, mesuree dans
  `band-ascent-v51` / [ ] Candidate a la soumission
