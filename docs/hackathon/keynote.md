# Keynote : « Invisible image watermarking, between research and production »

> Pierre Fernandez (Meta FAIR, Superintelligence Labs), 21/11/2026, 57 slides.
> Source : `https://pierrefdz.github.io/assets/slides/2026-11-21_image_watermarking_cispa.html`.
> Lu en entier le 21/11/2026 : texte de chaque slide, animations dévoilées (slides 15 et 21) et
> figures chiffrées. Numéros de slides entre crochets.

## Ce qu'il faut en retenir pour le hackathon

1. **Localisation = le sujet de la tâche Watermark Localization.** Un tatouage global donne une
   seule réponse par image ; après inpainting, collage (splicing) ou fort recadrage, l'extracteur
   moyenne sur toute l'image et ne dit ni où est le tatouage ni ce qui a changé [24].
2. **Les extracteurs Meta produisent déjà une carte par pixel** : Watermark Anything sort
   256×256×(1+nbits), une carte de détection plus les bits par pixel, avant toute moyenne [29, 32] ;
   c'est aussi la structure de sortie des extracteurs Video Seal / Pixel Seal (B×(1+nbits)×H×W
   avant mise en commun, vérifié dans le code Meta).
3. **Pixel Seal travaille en 256×256** : l'image est redimensionnée en 256, l'embedder produit un
   résidu modulé par une carte JND, le résidu est remis à la taille d'origine ; l'extracteur
   redimensionne aussi en 256 [17]. Conséquence pour nous : la résolution effective de la carte de
   présence est celle de l'extracteur, pas celle de l'image.
4. **Détection = test statistique sur la bit accuracy** (loi binomiale sous H0, image naturelle =
   bits aléatoires) ; on règle un seuil τ selon le taux de faux positifs visé [14, 15].
5. **Sécurité ≠ robustesse** : en boîte blanche, enlever un tatouage est « extrêmement facile » ;
   les attaques par régénération (DiffPure, auto-encodeur / compression neuronale) effacent les
   tatouages post-hoc [37, 38, 52].

## Plan détaillé

### Partie 1 : pourquoi tatouer [3-11]
- Risques de l'IA générative : influence, arnaques, fraude, pollution d'Internet [4].
- Détection passive (classifieur « généré par IA ? ») : difficile et de plus en plus [5].
- Métadonnées : perdues très souvent (plateformes) [6]. Empreintes : il faut tout stocker, ne
  passe pas à l'échelle [7]. Tatouage : message binaire caché, révélé même après transformations [8].
- **Obligation légale** [9] : AI Act européen (contenus IA marqués et détectables dès août 2026),
  California SB 942 (« imperceptible and maximally indelible watermark » + détecteur public, août
  2026), AI Basic Act coréen (étiquette visible ou lisible par machine, en vigueur depuis le
  22/01/2026, art. 31).
- Trois critères [10] : imperceptibilité (distorsion faible), charge utile (message assez long,
  détection avec forte confiance), robustesse (message récupéré après édition).
- Où tatouer [11] : données d'entraînement, entraînement, inférence (bruit, échantillonnage),
  image de sortie. Familles : forensique (deviner), métadonnées, **post-hoc** (après génération :
  PixelSeal, WAM), génération hors modèle (bruit initial, échantillonnage : Tree-Ring, AR
  sampling), génération dans le modèle (poids, décodeur, latent : Stable Signature, DistSeal).

### Partie 2 : tatouage post-hoc [12-35]
- Recette commune (HiDDeN, ECCV 2018 ; ReDMark) [13] : embedder + extracteur entraînés ensemble,
  transformation aléatoire au milieu, perte perceptuelle (imperceptibilité) + perte message
  (robustesse).
- Détection [14] : exemple 31/32 bits corrects (image générée) contre 17/32 (image naturelle) ;
  test bit accuracy > τ ; p-valeur = somme binomiale. Évaluation TPR/FPR et courbe ROC [15] :
  τ = 0,80 → TPR 0,50, FPR 4·10⁻⁴ ; τ = 0,90 → TPR 0,16, FPR 4·10⁻⁶.
- Panorama [16] : HiDDeN ; **TrustMark** (résolution de travail fixe puis redimensionnement :
  toute taille d'image) ; SynthID-Image (post-hoc à l'échelle d'Internet, Google DeepMind) ;
  Video Seal (même recette, image par image) ; **Pixel Seal** (entraînement adversarial seul,
  modulation JND) ; **Watermark Anything** (messages localisés). Différences : résolution,
  imperceptibilité, stabilité d'entraînement, localisation.
- **Pixel Seal** [17-22] : pipeline resize 256 → embedder → JND → resize original → augmentations
  → resize 256 → extracteur ; pertes ℓ_msg + discriminateur ℓ_adv. JND = plus petite différence
  perceptible par l'œil ; le résidu est atténué là où il serait visible [18, 19]. Aucune perte
  perceptuelle : un discriminateur (type GAN) remplace l'imperceptibilité [20]. Entraînement en 4
  étapes [21] : (1) sans discriminateur, force α = 1,0 ; (2) discriminateur ajouté, α = 1,0 ;
  (3) α réduit à 0,2 ; (4) affinage sur d'autres attaques. Meilleur compromis invisible/robuste,
  code MIT `github.com/facebookresearch/videoseal` [22].
- **Limite des tatouages globaux** [24] : inpainting (une zone régénérée : le reste est-il encore
  tatoué ?), splicing (deux images de deux modèles combinées : quel message où ?), fort recadrage
  (petite zone restante moyennée).
- **Watermark Anything (WAM)** [25-35] : le tatouage comme segmentation ; l'extraction donne quels
  pixels sont tatoués et quel message ils portent. Embedder type auto-encodeur de Stable Diffusion
  (plus petit, sous-échantillonnage f = 16, message injecté par couche d'embedding) [27].
  Augmenteur : édite l'image et masque le tatouage par zones (masques de segmentation, formes
  aléatoires : traits, carrés) [28]. Extracteur ≈ SAM : encodeur ViT-base (patchs 16×16×d) +
  décodeur pixel → 256×256×(1+nbits) ; DINO/SAM pré-entraînés n'aident pas beaucoup [29]. Pertes :
  BCE de détection moyennée sur tous les pixels + BCE de décodage sur les pixels tatoués [30].
  Affinage multi-messages avec masques COCO [31]. Inférence : bits des pixels tatoués, regroupés
  par DBSCAN (nombre de messages inconnu), moyenne par groupe [32].
- Résultats WAM [33] (COCO, 10k images de validation ; TPR % / bit acc. %) :

  | Méthode | FPR | Aucune | Géométrique | Valeurs | Inpainting | Splicing |
  |---|---|---|---|---|---|---|
  | HiDDeN | 0,08 | 76,9 / 95,5 | 31,2 / 80,1 | 48,4 / 87,2 | 44,7 / 88,7 | 0,9 / 68,0 |
  | DWTDCT | 0,02 | 77,1 / 91,4 | 0,0 / 50,5 | 13,6 / 58,1 | 47,8 / 81,7 | 0,8 / 59,9 |
  | SSL | 0,00 | 99,9 / 100,0 | 14,3 / 76,5 | 70,5 / 92,1 | 67,7 / 91,1 | 0,4 / 58,9 |
  | FNNS | 0,10 | 99,6 / 99,9 | 62,4 / 86,6 | 82,1 / 93,9 | 89,4 / 97,3 | 38,7 / 88,5 |
  | TrustMark | 2,88 | 99,8 / 99,9 | 36,3 / 71,4 | 90,1 / 98,2 | 39,4 / 83,2 | 83,2 / 57,1 |
  | WAM | 0,04 | 100,0 / 100,0 | 99,3 / 91,8 | 100,0 / 99,9 | 97,9 / 99,2 | 100,0 / 95,3 |

  À noter pour la tâche : TrustMark a le FPR le plus élevé (2,88 %) et seulement 57 % de bits
  justes après splicing, même s'il détecte encore 83 % des images.
- Localisation qualitative [34, 35] (baseline localisée : EditGuard, CVPR 2024 ; démo HF
  `xiaoyao9184/watermark-anything`).

### Partie 3 : tatouage à la génération [36-45]
- Attaques [37, 38] : boîte blanche très facile, boîte noire difficile mais possible ; DiffPure
  (bruit puis débruitage par diffusion) et régénération VAE / compression neuronale effacent le
  tatouage.
- Tatouage sémantique [39] : les encodeurs/décodeurs post-hoc restent à PSNR élevé, avec peu de
  changement de pixels ; un tatouage sémantique (PSNR bas) est plus dur à retirer.
- Stable Signature (décodeur du modèle latent affiné pour tatouer chaque image) [40], DistSeal
  (tatouage dans l'espace latent) [41], Tree-Ring (motif dans le bruit initial, détecté par
  inversion de la diffusion) [42], modèles autorégressifs (biais d'échantillonnage des tokens,
  p-valeur 1,2·10⁻⁷ dans l'exemple ; 7 papiers 2025-2026 cités) [43, 44]. Les tatouages à la
  génération survivent à la régénération là où le post-hoc casse souvent [45].

### Partie 4 : discussion [46-57]
- En production, les images sont tatouées **post-hoc presque tout le temps** malgré la meilleure
  robustesse du tatouage à la génération [47] ; Content Seal en ligne sur `meta.ai/identification`
  [48]. Les annonces citées : tatouage du texte de Claude (Engadget, août 2026), vérification des
  images OpenAI (TechCrunch, mai 2026) [47].
- Nouveaux compromis : imperceptibilité, charge, robustesse, sécurité, vitesse, compacité,
  adaptabilité [49]. Obstacles organisationnels : ne pas toucher au modèle génératif, lenteur,
  inadapté aux données réelles → post-hoc [50, 51].
- Robustesse (traitements classiques, 99,9 %) ≠ sécurité (attaques intentionnelles, 0,1 %) [52].
- Problèmes ouverts : rétrocompatibilité entre versions de modèles (un extracteur unique pour
  PixelSeal v1 et v2) [53], détecteur commun à plusieurs entreprises sans partager les embedders
  [54], détecteurs publics : API protégée (limitation, authentification) ou extracteur open
  source (risque d'évasion plus élevé) [55].
- Conclusion [56] : tatouage de plus en plus important (régulateurs, grandes entreprises), utile
  pour toutes les modalités ; génération = plus sûr contre le retrait adversarial ; post-hoc =
  beaucoup plus souple.
- Références [57] : HiDDeN, TrustMark, SynthID-Image, Video Seal, Pixel Seal, Watermark Anything
  (`github.com/facebookresearch/watermark-anything`), EditGuard, Stable Signature, DistSeal,
  Tree-Ring, `facebookresearch.github.io/content-seal`, `synthid.com`.
