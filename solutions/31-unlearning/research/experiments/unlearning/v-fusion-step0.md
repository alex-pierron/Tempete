# V_Fusion step 0 (images synthetiques et portes, sans entrainement de l'eleve)

- Identifiant : `v-fusion-step0` (sans spec : sous-commandes lancees directement, voir Lancement)
- Tache : `unlearning`
- Creee le : 2026-11-22
- Responsable : Alex

## Hypothese

V_Fusion (rapport, section V_Fusion ; `src/hackathon/unlearning/synthetic.py`) : SCRUB conforme dont
l'ancre (les min-steps) devient des images synthetiques tirees du seul original, pour tenir le retain
>= 0,8 a une dose ou l'ancre de vues du forget set le laisse tomber (lr 4e-4 : 0,121 +- 0,007, retain
0,73). Etape 0 : generer les images (Plug-In Inversion, 20 par classe dont 5 de thermometre, zoom
56 -> 112 -> 224, 100 pas par etage, Adam 0,05, 4 copies augmentees, arret a une confiance de 0,9) et
les passer aux portes du module, fixees avant toute lecture des resultats (rapport, Portes) :

- a. robustesse : sous retournement, recadrage de 60 a 90 % de l'aire et flou (sigma = 1),
  l'original garde la classe sur >= 80 % des images, pour chaque transformation ;
- b. appartenance (lots A et C) : le jeton de classe a l'entree de la tete est plus proche du centre
  de sa classe (vraies images du forget set) que de tout autre, pour >= 70 % des images ;
- c. variete : <= 5 % des paires d'une meme classe aussi semblables qu'une image et son retournement ;
- c'. distance au forget set (lot A) : <= 10 % des images retirees comme trop proches d'une image du
  forget set ;
- d. reparation : apres un bruit sur les poids qui fait changer d'avis l'original sur 20 % des vues
  fortes d'une moitie du forget set, 300 pas de distillation a T = 2 (lr 3e-5) sur les seules images
  d'ancre d'un lot ramenent l'accord a >= 0,95 ; le temoin (vues fortes de l'autre moitie, meme
  budget) doit lui-meme atteindre 0,95, sinon la porte ne dit rien du lot.

**Regle d'entree, fixee le 2026-11-22 avant le run** :
- le lot A entre dans une ancre s'il passe a, b, c, c' et d (d interpretable) ;
- le lot B entre s'il passe a, c et d (b n'est pas mesurable pour lui) ;
- le lot C n'entre dans aucune ancre (hypothese de l'equipe, rapport) ; ses portes sont lues pour
  memoire ;
- si le temoin de d reste sous 0,95, un seul suivi : les portes seules sur les memes images avec
  `--repair-steps 1000` ; si le temoin echoue encore, d est ininterpretable et la decision se prend sur
  a, b, c, c' seules, en le notant ;
- si aucun lot n'entre, V_Fusion s'arrete sous cette generation (pas de second reglage de la
  generation sans hypothese ecrite d'avance).

Lecture de la generation (sans seuil) : part des images qui atteignent le plafond 0,9
(`summary.<lot>.reached`) et confiance moyenne ; temps de calcul.

## Reglages

Defauts de `SynthConfig` et `RepairConfig` (proposes, non mesures), graine 0, une seule tranche
(`--shard 0/1`) sur la RTX 5080 : memes graines par image que les 4 tranches du job JURECA, mais
lots d'optimisation et augmentations differents, donc images non identiques au bit pres.

## Metrique locale attendue

Portes passees pour A et/ou B ; accord apres reparation proche de celui du temoin. Cout : estime a
~1 h sur la RTX 5080 (2000 images x 300 pas x 4 copies), non mesure.

## Lancement

Le lab n'admet pas `synth-generate` ni `synth-gates` dans une spec : la spec prevue a ete refusee au
lancement, sans rien executer. Les deux sous-commandes ont tourne directement, sous la garde GPU,
comme dans `jureca/synthetic.sbatch`. Elles sont donc **hors registre** ; la provenance est dans
`runs/unlearning/20261122-vfusion-step0/provenance.txt` (commit ceae862, arbre propre).

```bash
export PYTHONPATH=$PWD/src HK="./scripts/gpu_guard.sh .venv/bin/python -m hackathon.unlearning --device cuda"
OUT=runs/unlearning/20261122-vfusion-step0
$HK synth-generate --shard 0/1 --out $OUT/synthetic_0of1.pt
$HK synth-gates $OUT/synthetic_0of1.pt --out $OUT/synthetic_gated.pt --grid $OUT/grid.png
$HK synth-gates $OUT/synthetic_0of1.pt --repair-steps 1000 --out $OUT/synthetic_gated_r1000.pt
```

## Resultat

Generation : 2000 images en 34 min 15 s (RTX 5080, 63 degres au plus, VRAM 8,3 Go). Le forget set
compte 45 classes de pseudo-etiquettes ; 9 sont mises a part. Lots : A = 720 images (36 classes),
B = 1100 (55), C = 180 (9). Ancre (hors thermometre, toutes gardees) : A 540, B 825.

- **Plafond de confiance : sans effet.** Toutes les images l'atteignent (`reached` = 1), mais leur
  confiance finale, lue comme le serveur sur les images uint8, vaut 0,998 a 1 (mediane 0,999999 ;
  aucune sous 0,95), au lieu de 0,9. Le plafond n'est teste qu'au dernier etage ; les images sont
  deja saturees apres les etages a 56 et 112 px. A T = 2, elles ne dessinent presque plus les
  classes fausses : c'est un defaut de la generation, note avant les bras.

Portes (`synth-gates`, 1 min 52 s) :

| lot | a : flip / recadrage / flou | b : appartenance | c : quasi-copies | c' : retirees | d : accord apres reparation (300 pas ; 1000 pas) |
|---|---|---|---|---|---|
| A | 1,00 / 1,00 / 1,00 | 1,00 | 0,020 | 0,000 | 0,834 ; 0,842 |
| B | 1,00 / 1,00 / 1,00 | non mesurable | 0,034 | -- | 0,834 ; 0,833 |
| C | 1,00 / 1,00 / 1,00 | 1,00 | 0,010 | -- | -- |
| temoin (vues du forget set) | | | | | 0,911 ; 0,921 |

Le bruit sur les poids (sigma = 0,483 x l'ecart-type de chaque tenseur) ramene l'accord a 0,800.

- a, b, c, c' : **passees** pour A et B (et pour C, lu pour memoire).
- d : **ininterpretable**, le temoin reste sous 0,95, meme a 1000 pas (suivi prevu, fait). Selon la
  regle, la decision se prend sur a, b, c, c' seules : **A et B entrent dans l'ancre**.
- Lecture de d, sans seuil : les images synthetiques reparent environ trois fois moins que les vues.
  A ramene 17 a 21 % des degats, B 17 %, et le temoin 55 a 60 %. Un diagnostic (script hors depot)
  explique pourquoi. Sur les images synthetiques, la perte de distillation part a 0,04-0,06 et tombe
  a 0,003, contre 2,17 sur les vues : le modele bruite est deja d'accord avec l'original sur ces
  images saturees. Les accords egaux de A et B a 300 pas (1459 sur 1750) sont une coincidence : les
  deux eleves sont distincts (distance 2,5 entre leurs poids), avec des trajectoires differentes.
- Variete, en pixels (cosinus a 32 px, centre) : 0,20 en moyenne entre images d'une meme classe,
  0,03 entre classes, 0,25 entre une image et son propre retournement. Sur la grille, les images d'une
  classe partagent un motif sans etre des copies.

## Decision

- [x] Coder l'ancre synthetique de `scrub` (lots A et B admis par la regle) : commit ba05fcb, puis
  `v-fusion-anchor.md`. Reserve posee avant les bras : images saturees, reparation faible.
- [ ] Arreter V_Fusion sous cette generation
