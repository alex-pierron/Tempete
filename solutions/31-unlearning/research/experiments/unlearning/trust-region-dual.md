# trust region dual (V4, contrainte active)

- Identifiant : `trust-region-dual` (spec : `trust-region-dual.toml`)
- Tache : `unlearning`
- Creee le : 2026-11-22 ; revue a la fusion avec la campagne V3 (meme jour, avant tout run)
- Responsable : Alex

## Hypothese

Les arrets de la V1 tombent tous autour de la fin du warmup (pas 72-92 sur 88), modele fige au taux
maximal ; la campagne V3 retrouve un optimum de rayon vers 0,15-0,18 pour les trois versions, et des
reglages de cible qui n'agissent que par le rayon. Ici le run parcourt ses 10 epoques ; la perte
recoit mu x KL(original || eleve) a T = 1 sur le lot, et a chaque test de la sonde
mu <- max(0, mu + eta x log(KL / rayon)) (pas symetrique : un depassement se rattrape aussi vite
qu'il s'est paye). L'eleve continue d'apprendre la forme de sa cible sans depasser le budget. On
croit a un gain de `bench.score_min` au-dela de 2 sigma sur la V3 retenue de meme graine, et a une
hausse >= 0,03 du pire `bench.prediction.*`.

**Base et references.** La V3 retenue (`ref4-v3` : b = 0, lambda 0,25, rho 0,9, aire 0,5, 4 vues,
sans copies douces). References appariees deja mesurees : graine 0 `tr3-hedge` v1 (0,0784), graine 1
`tr3-seeds` v1 (0,0804), rejouables par `ref4-v3`. Difference de calcul a noter : en mode dual, le
rayon est pris sur la sonde de 256 images ou l'arret est teste (la V3 le prend sur toutes) ; lire
`final.kl_radius_effective` des deux cotes. lambda = 0,5 n'est pas repris : la campagne V3 l'a deja
mesure au-dela du seuil de casse (0,0566). Prediction a tester : converger vers la cible egalise les
pertes et fait baisser `bench.mean_std` ; le gain, s'il existe, passe par `bench.prediction`.

sigma = 0,003 (ETAT.md, ecart-type entre graines des references).

**Refutee si** :
- aucun bras ne gagne plus de 2 sigma (0,006) sur la V3 de meme graine ;
- ou `bench.audit_kl` d'un bras depasse 1,3 x celui de la V3 de meme graine (0,243 et 0,271).

**Non interpretable** (eta mal regle, pas une refutation) si `final.radius_ratio` (KL finale /
rayon) sort de [0,85 ; 1,15].

Un gain sur les deux graines se confirme sur la graine 2 (seuil ETAT : moyenne > 0,086, gain sous le
masquage au-dela de 2 sigma). Seuils fixes a priori, le 2026-11-22, avant tout run.

## Reglages

Ceux de la V3 retenue ; lr 1e-5, lots de 32, 10 epoques, rayon teste a chaque pas. eta = 0,25 et
0,5 : un depassement d'un facteur e (KL = 2,7 x rayon) monte mu de 0,25 ou 0,5 par pas.

## Lancement

```bash
make hk-run SPEC=trust-region-dual                 # Mac : smoke ; GPU CUDA (RTX 5080) : full
make hk-run SPEC=trust-region-dual WHERE=jureca    # 4 variantes sur 4 A100 (file d'equipe)
```

Sur un autre GPU que la RTX 5080, rejouer d'abord `ref4-v3` sur la meme machine (ETAT.md §2).
Ces runs ont un holdout : ne pas soumettre leurs fichiers.

## Resultat

Quatre runs, RTX 5080, `bench-v4`, commit 52fa1d3, code propre (groupe
`20261122T014218Z-unl-trust-region-dual`, 12 min). References appariees : `ref4-v3` (rejeu au bit
pres de `tr3-hedge` v1 et `tr3-seeds` v1).

| eta | graine | score_min | pire recadrage | pire masquage | retain | KL audit | audit/exact | KL T=1 finale | rayon (sonde) | radius_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,25 | 0 | 0,0423 | 0,0609 | 0,0423 | 0,729 | 0,339 | 0,99 | 0,085 | 0,157 | 0,54 |
| 0,25 | 1 | 0,0417 | 0,0596 | 0,0417 | 0,731 | 0,339 | 0,98 | 0,085 | 0,161 | 0,53 |
| 0,5 | 0 | 0,0422 | 0,0608 | 0,0422 | 0,729 | 0,339 | 0,99 | 0,085 | 0,157 | 0,54 |
| 0,5 | 1 | 0,0417 | 0,0596 | 0,0417 | 0,731 | 0,339 | 0,98 | 0,085 | 0,161 | 0,53 |
| V3 retenue (`ref4-v3`) | 0 / 1 | 0,0784 / 0,0804 | 0,0865 / 0,0901 | 0,0784 / 0,0804 | 0,758 / 0,735 | 0,243 / 0,271 | 0,86 / 0,90 | 0,189 / 0,193 | 0,173 / 0,172 (toutes) | |

Ce qui s'est passe (historique par epoque, `candidate.json`) : la KL de la sonde a T = 1 franchit le
rayon vers le pas 100, au taux maximal ; le multiplicateur monte pendant l'epoque 2 (penalite moyenne
0,02), ramene la KL sous le rayon, puis **retombe a 0 et y reste de l'epoque 3 a l'epoque 10**. Sans
contrainte active, l'objectif de la V3 poursuivi sur tout le calendrier se pose de lui-meme a une KL
T = 1 de 0,085, la moitie du rayon, tout en ajustant sa cible bien plus (perte d'ajustement 0,63 contre
2,1 a 2,2 a l'arret de la V3). A T = 2, l'inverse : KL exacte 0,34-0,35 contre 0,28-0,30.

Lecture contre les seuils poses avant :

- **Non interpretable au sens de la fiche** (`final.radius_ratio` 0,53-0,54, hors [0,85 ; 1,15]), mais
  pas pour la raison prevue (eta mal regle) : eta n'y est pour rien (mu = 0 sur 80 % du calendrier ;
  eta 0,25 et 0,5 donnent le meme modele a 1e-4 pres). La contrainte est une borne haute
  (mu >= 0 ne peut que ramener vers l'original) et elle ne lie pas : la question « a rayon egal, la
  contrainte active laisse-t-elle apprendre la forme de la cible ? » n'est pas testable avec ce code.
  Aucune valeur d'eta ne la rendrait testable : pas de suivi sur eta (regle 2 de la fiche de campagne).
- **Ce qui est mesure refute la premisse du mode** : laisser l'eleve apprendre la forme de sa cible sur
  tout le calendrier fait perdre la moitie du score (-0,036 et -0,039 appariees, au-dela de 2 sigma
  dans le mauvais sens, sous le masquage comme sous le recadrage).
- **Prediction de la fiche confirmee** : converger vers la cible egalise les pertes. Composante des
  pertes 0,12 a 0,19 (V3 : 0,34 a 0,47), composante des predictions 0,32 a 0,69 (V3 : 0,23 a 0,38) ;
  le pire `bench.prediction.*` monte de +0,09 (0,323 contre 0,234 a la graine 0), au-dela du +0,03
  attendu, mais ne compense pas la perte de `bench.mean_std`.
- **Derive** : `bench.audit_kl` 0,339 = 1,39 x la V3 a la graine 0 (seuil de refutation 1,3) et 1,25 x
  a la graine 1 ; juste sous la porte de casse (0,34). audit/exact 0,98-0,99 : plus generique que la V3.
- `final.kl_radius_effective` : 0,157 et 0,161 sur la sonde (dual) contre 0,173 et 0,172 sur toutes les
  images (V3) ; l'ecart (-9 %) ne change rien a la lecture, le rayon n'etant pas atteint.

## Decision

- [x] Abandonner : la contrainte duale telle que codee ne lie pas apres le warmup ; poursuivre la V3
  sur tout le calendrier divise son score par deux. Le test de la fiche (contrainte active au rayon)
  demanderait une contrainte d'egalite (mu de signe libre, qui pousse aussi loin de l'original), soit
  une modification de la methode, hors du perimetre de cette campagne ; rien dans ces runs ne laisse
  esperer un gain (`bench.mean_std` s'effondre des que l'eleve ajuste sa cible).
- [ ] Iterer / [ ] Candidate a la soumission
