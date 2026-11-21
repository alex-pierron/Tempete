# Watermark Localization : kit de la tache

Rendre, pour chaque image test (512x512), un masque 128x128 (1 = tatoue) ; score = mIoU par
image (classes 0 et 1, union vide = 1) moyenne sur les images. Code : `src/hackathon/wmloc/`,
CLI : `python -m hackathon.wmloc`. Tout ce qui suit a ete ecrit AVANT d'avoir acces au dataset
(HTTP 403 sans token) : voir "Verifie / non verifie".

## 1. Ce qu'on sait du sujet, et ce que montre le TRAIN reel

Dataset public (plus de token requis). Starter kit : `data/hackathon/wmloc/starter_kit/`, train :
`release/train` (510 images, 170 par methode, masques 512 + `masks_128.npz`), test : `release/test`
(1020). `metadata.json` donne `image_name`, `method`, `message` (256 bits PixelSeal, 100 bits
TrustMark Q et StegaStamp) et, sur le train, `full_watermark`. Pas de type d'edition dans les
metadonnees (outpainting / retrait / ajout / transplantation ne sont pas etiquetes).

Constat (statistiques sur les 510 masques train, voir section 2) : DEUX REGIMES, 50/50 dans chaque
methode.
- `full_watermark = true` (85/methode) : toute l'image est tatouee, puis une partie est restauree
  a l'original propre. Masque tres majoritairement a 1 (aire moyenne 0,90). Le decodage de l'image
  entiere reussit (accord moyen 0,98). Il faut trouver le TROU propre (~10 % de l'image).
- `full_watermark = false` (85/methode) : un petit sous-rectangle seulement (aire mediane 4-5 %,
  moyenne 7-10 % ; de 12x171 px a ~370x150 px, parfois des bandes de 32 px de large) est tatoue.
  Le decodage de l'image entiere est au hasard (accord moyen 0,52-0,55).
Consequence : le baseline "tout a zero" du template vaut 0,25 (0,05 sur le regime full, 0,45 sur
le regime rect) et "tout a un" 0,25 ; un simple aiguillage par decodage global (pas de
localisation) donne ~0,45 par methode. La tache se joue sur la localisation fine dans chaque regime.

Rappel du sujet : le test fournit la methode et le MESSAGE ; la question est "cette zone porte-t-elle
encore ce message ?" (accord bits decodes / bits connus : 0,5 = hasard, 1,0 = intact).

## 2. Ce qu'on a mesure (Pixel Seal sur MPS ; banc synthetique + train reel)

1. **La carte spatiale native est trop grossiere.** La tete de l'extracteur Pixel Seal est un
   `Linear` apres une moyenne spatiale ; appliquee par position elle donne une carte 8x8
   (cellules de 64 px sur 512) dont la moyenne est EXACTEMENT le logit global (test slow).
   Mais le champ receptif ConvNeXt melange tout : contraste dedans/dehors faible (voir bench).
   Entree 512 natif : le decodeur echoue (bit acc 0,57) : il faut redimensionner a 256.
2. **Le decodage d'une fenetre reussit si elle est (presque) entierement dans la zone tatouee**
   et echoue des qu'elle y melange du contenu etranger. D'ou le localiseur par fenetres glissantes
   avec agregation `max` (une fenetre qui decode prouve que ses cellules sont tatouees ; celles qui
   chevauchent le bord ne diluent rien).
3. **Mode "crop" : le crop EXACT decode a 1,00, l'image entiere a 0,50** (dilution). Une fenetre
   bien plus petite que le rectangle decode moins bien (changement d'echelle : 0,76 pour une
   fenetre de 128 px dans un rectangle de 300 px). Les fenetres doivent etre proches de l'echelle
   du rectangle.
4. **Recherche du rectangle** : l'accord binaire du crop reste a 1,00 sur un large plateau
   (+-10 % de la taille), donc il ne localise pas les bords ; la MARGE DOUCE (moyenne logit x signe
   du message) a un pic net au rectangle exact. `RectRefineLocalizer` place chaque bord au pic de
   marge (depart : boite des cellules marquees, retrecie uniformement, car le seed deborde).
5. **Transplantation** (tatouage seulement sur l'objet) : cas le plus dur, le signal est trop
   faible pour une fenetre ; resultat proche du hasard dans tous nos localiseurs.

## 3. Strategie recommandee (phase de recherche)

1. Aiguillage par regime avec le decodage global (`bit_agreement` de l'image entiere vs message
   connu ; seuil ~0,75-0,8 selon la methode, a calibrer) : c'est deja ~0,45 sans localiser.
2. Regime full : l'image est marquee sauf un trou restaure ; chercher le trou = zone ou le message
   n'est plus decode. Fenetres proches de l'echelle native du decodeur (PixelSeal : 256 px sur
   l'image 512 ; fenetres plus petites decodent mal), puis seuil calibre.
3. Regime rect : petit rectangle, souvent < 5 % de l'image ; `rect_refine_*` (fenetres + ajustement
   des bords sur la marge douce) fonctionne sur les rectangles de taille moyenne ; les plus petits
   (< 2 % de l'aire, bandes fines) sont au hasard pour tous nos decodeurs : s'y resoudre a "tout a
   zero" ou chercher mieux. Fenetres de 64 px a ajouter pour ce regime.
4. TrustMark / StegaStamp : memes localiseurs via `KitBackend` (decodage seul, deja branche) ;
   non evalues (voir section 8).
5. Tete apprise (section 6) si le temps le permet. Verifier le fichier (`validate --manifest`),
   soumettre une fois pour mesurer, iterer.

## 4. Commandes

Depuis la racine du depot (venv principal) ; en worktree, pointer les donnees :

```bash
export PYTHONPATH=$PWD/src HACKATHON_DATA_DIR=<repo>/data/hackathon
PY=<repo>/.venv/bin/python
$PY -m hackathon.wmloc fetch                 # sans token ; dezippe dans wmloc/{starter_kit,release}
# Modeles (une fois) : lire puis lancer le script officiel. TrustMark et StegaStamp ont besoin de
# trustmark==0.8.0, onnxruntime, onnx, pytorch-lightning : les installer dans un dossier a part
# (ex. uv pip install --target /tmp/kitlibs --no-deps ...) et l'ajouter a PYTHONPATH.
PYTHONPATH=/tmp/kitlibs $PY data/hackathon/wmloc/starter_kit/download_models.py \
    --models_dir data/hackathon/wmloc/models
$PY -m hackathon.wmloc prepare               # release/{train,test}/manifest.json
jarvis-heavy $PY -m hackathon.wmloc bench --manifest data/hackathon/wmloc/release/train/manifest.json \
    --methods pixelseal --per-edit 30 --localizers all_ones,all_zeros,rect_refine_96_128
$PY -m hackathon.wmloc predict --manifest data/hackathon/wmloc/release/test/manifest.json \
    --localizer rect_refine_96_128 --out data/hackathon/wmloc/sub.pt
$PY -m hackathon.wmloc validate data/hackathon/wmloc/sub.pt --manifest <test manifest>
$PY -m hackathon.wmloc submit data/hackathon/wmloc/sub.pt --manifest <test manifest>        # dry run
$PY -m hackathon.wmloc submit data/hackathon/wmloc/sub.pt --manifest <test manifest> --yes  # ENVOI
```

`submit --yes` exige `HACKATHON_API_KEY` ; l'id serveur est `34-watermark-localisation-image`
(template officiel ; surchargeable par `HACKATHON_TASK_WMLOC`). `fetch` essaie
`SprintML/Hackathon_Watermark_Localisation_Task` puis `..._Paris` (404 seulement). Le banc
synthetique (`synth`, Pixel Seal via wmkit, exige `WMKIT_WEIGHTS_DIR`) reste utile pour tester un
localiseur sans le train mais ne ressemble PAS au vrai (rectangles trop grands, un seul regime).
Le groupe `full_watermark` / `rect_watermark` du train est dans le rapport du `bench`.

JURECA (1 A100) : `sbatch jureca/wmloc.sbatch <commande> ...` depuis le dossier de la tache (voir
l'en-tete du script). Les noeuds de calcul n'ont pas internet : `fetch` et installations sur le login.

## 5. Integration du kit (faite)

- `kit_backend.py` charge `starter_kit/wmloc/watermarks.py` PAR CHEMIN (nom de module prive) : pas
  de collision avec le package `wmloc` du kit. Dossiers : `$WMLOC_STARTER_KIT_DIR` et
  `$WMLOC_MODELS` (defauts sous `<data_dir>/wmloc/`). `KitBackend` expose `logits` (> 0 = bit 1)
  et, pour PixelSeal, `features()["per_position_logits"]` ; c'est le backend par defaut de
  `bench`/`predict` (`--backend-factory module:fonction(method, device)` pour en changer).
- `manifest_from_release` + `prepare` : manifests des splits officiels (noms = fichier sans .png).
- Parite metrique : `tests/hackathon/test_hk_wmloc_metric.py` charge `starter_kit/wmloc/utils.py`
  par chemin et compare `downsample_mask`, `miou_image`, `score` et `masks_128.npz` du train :
  identiques (le test se saute proprement si le kit est absent).
- Limite de taille : le PDF dit 20 MB, le README du kit 25 MB. On valide a 20 000 000 octets (le
  plus strict) ; le fichier attendu fait ~17 Mo.

## 6. Plan : tete de segmentation apprise (non implemente)

Donnees synthetiques illimitees : les embedders sont fournis, donc par methode on peut generer
(image source, rectangle, message, edition simulee) avec masque exact (`synth.py` est le
generateur, a remplacer par les vrais outils d'edition du kit si dispo).
1. Geler l'extracteur ; sortie par fenetre = logits de bits alignes sur le message connu
   (`sign x logit`, [n_bits, h, w]) et la carte de features intermediaire.
2. Petite tete convolutionnelle (3-4 couches, ~100 k parametres) -> masque 128x128, perte BCE +
   Dice, entree = accord par bit (carte [n_bits,h,w] reduite en statistiques : moyenne, ecart,
   fraction > 0), eventuellement le crop 512 redimensionne en plus pour les contours.
3. Entrainement par methode, valide sur le train reel (510) ; seeds multiples ; baseline =
   `rect_refine`. Mesurer le transfert synthetique -> reel : c'est le risque principal (nos
   editions sont grossieres).
4. Execution : A100 JURECA, `sbatch`.

## 7. Pieges

- Le fichier de soumission fait 16,7 Mo en uint8 pour 1020 images, 67 Mo en float32 (rejete) :
  `save_submission` convertit et refuse toute valeur hors {0,1}. Limite appliquee : 20 000 000
  octets (lecture stricte de "20 MB").
- Un envoi, meme invalide, peut declencher un delai serveur (30 min pour model-equality ; inconnu
  ici) : on valide toujours avant, `submit` sans `--yes` ne fait rien.
- `huggingface_hub` v2 utilise httpx (plus requests) : les erreurs reseau sont enveloppees en
  `FetchError`.
- Les images du banc sont quantifiees 8 bits comme les PNG reels ; ne pas evaluer sur des float.
- Le chemin de donnees par defaut (`Settings.data_dir`) depend de la racine du depot : en worktree,
  exporter `HACKATHON_DATA_DIR`.

## 8. Verifie / non verifie

Voir le rapport de livraison (chiffres reels sur le train, parite metrique, ce qui reste a faire).
