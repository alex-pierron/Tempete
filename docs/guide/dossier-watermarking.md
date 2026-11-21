---
title: "Watermarking : dossier de preparation"
subtitle: "European AI Championship Paris, 21-22 novembre 2026 — l'essentiel, les pistes de recherche, le playbook d'algorithmes et la boite a outils prete a l'emploi"
date: "19 novembre 2026"
---

# 0. L'essentiel en une page

**Le sujet.** Un watermark invisible est un signal cache dans un contenu (image, video, audio,
texte) qui permet plus tard de prouver son origine : « cette image a ete generee par tel modele »,
« cette video vient de tel compte ». Meta l'utilise en production pour la provenance des
contenus generes par IA (Meta AI, Movie Gen) et publie ses modeles en open source (famille
*Meta Seal* : Video Seal, Pixel Seal, Audio Seal, Watermark Anything...).

**Les trois papiers imposes dessinent un theme plus large que le watermarking : l'audit des
systemes d'IA.**

| Papier | Question d'audit | Outil statistique |
|---|---|---|
| Pixel Seal (Meta FAIR, 2025) | D'ou vient ce contenu ? | Test binomial sur des bits extraits |
| Model Equality Testing (ICLR 2025) | Quel modele sert reellement cette API ? | Test a deux echantillons (MMD) |
| Towards Unbounded Machine Unlearning (NeurIPS 2023) | Ce modele a-t-il vraiment oublie ces donnees ? | Membership inference (MIA) |

Les trois reposent sur la meme idee : **transformer une question de confiance en test
statistique avec une p-value**. C'est le fil rouge a defendre devant un jury.

**Les dix choses a retenir.**

1. Tout watermark est un compromis entre **imperceptibilite** (PSNR, LPIPS, JND), **robustesse**
   (bit accuracy apres attaque) et **capacite** (nombre de bits). On ne gagne sur un axe qu'en
   payant sur un autre, sauf changement de methode.
2. La detection se juge au **taux de faux positifs** : on rapporte $-\log_{10} p$ ou un TPR a
   FPR fixe ($10^{-6}$), jamais une simple accuracy.
3. Pour 256 bits, il suffit de lire correctement **167 bits sur 256 (65 %)** pour conclure avec
   $p < 10^{-6}$. Un watermark « a moitie detruit » reste donc detectable.
4. Pixel Seal (256 bits, PSNR ~49 dB) est l'etat de l'art : **91 a 94 % de bits corrects**
   apres l'attaque combinee luminosite x crop 50 % x JPEG 40, la ou presque tous les concurrents
   tombent a 50 % (hasard).
5. Son point faible mesure : la **video compressee H.264 combinee a un crop** (68-70 % de bits).
6. Ses poids sont **publics** : une attaque adverse en boite blanche (PGD sur l'extracteur) est
   un scenario realiste et peu couteux.
7. Les attaques les plus fortes en boite noire sont la **regeneration** (passer l'image dans un
   VAE ou un modele de diffusion) et les **transformations geometriques** ; les filtres de
   couleur sont quasi inoffensifs pour Pixel Seal (il ne marque que la luminance).
8. Model Equality Testing detecte qu'une API sert un modele quantifie, fine-tune ou **tatoue**
   a partir de quelques centaines de completions, avec un simple noyau de Hamming.
9. SCRUB fait oublier un sous-ensemble de donnees par un schema eleve-professeur : maximiser la
   divergence sur les donnees a oublier, la minimiser sur les donnees a garder.
10. Le kit `wmkit` du depot evalue n'importe quel watermark contre 54 attaques en une commande ;
    l'utiliser des la premiere heure pour mesurer avant d'optimiser.

**Ce que l'on sait de l'epreuve (renseignement, sources publiques).** La competition se tient au
campus Epitech du Kremlin-Bicetre et dure 24 h, du samedi 21 midi au dimanche 22 novembre midi ; un
talk d'introduction a lieu le vendredi 20 au soir. **Les epreuves sont revelees le samedi matin
pendant la keynote de Meta.** 200 participants, equipes de 3 ou 4, 10 000 EUR de prix ; co-organisee
par PoC Innovation et le CISPA (laboratoire SprintML de F. Boenisch et A. Dziedzic), avec Meta et
Epitech. C'est une etape du CISPA European Cybersecurity & AI Hackathon Championship : chaque etape
propose **trois epreuves** notees sur un jeu de test cache (leaderboard), puis un passage devant
jury. Le watermarking y revient presque a chaque etape (detection d'images a Paris 2025, suppression
a Vienne, detection de watermarks LLM a Varsovie, localisation dans du texte en finale). Deux
constantes : les metriques **punissent les faux positifs** (TPR a 1 % ou 0,1 % de FPR, variantes
$\mathrm{TPR}/(1+5\,\mathrm{FPR})$), et l'acces se fait souvent **par API a debit limite** (une
requete toutes les 15 min a Paris 2025). Les trois papiers imposes correspondent probablement a une
epreuve chacun ; le detail des formats plausibles est en section 6 et dans
`docs/research/competition-intel.md`.

# 1. Le watermarking en quinze minutes

## 1.1 Vocabulaire

- **Watermark invisible vs visible.** Le logo en surimpression est un watermark visible ; on parle
  ici exclusivement de signaux invisibles, ajoutes comme une perturbation de quelques niveaux de
  gris.
- **Zero-bit vs multi-bit.** Un watermark *zero-bit* repond seulement « marque / non marque ».
  Un watermark *multi-bit* cache un message de $k$ bits (identifiant d'utilisateur, de modele,
  horodatage). Le multi-bit sert aussi a la detection : on insere toujours la meme cle et on
  verifie sa presence.
- **Post-hoc vs a la generation.** *Post-hoc* : on tatoue une image deja produite (Pixel Seal,
  Video Seal, TrustMark). *A la generation* : le generateur lui-meme produit des images marquees
  (Stable Signature fine-tune le decodeur d'un modele de diffusion ; Tree-Ring marque le bruit
  initial ; KGW biaise l'echantillonnage d'un LLM).
- **Robuste vs fragile.** Un watermark robuste survit aux editions ; un watermark fragile est
  concu pour casser au moindre changement (detection d'alteration).
- **A ne pas confondre.** La *steganographie* cherche le secret (capacite maximale, pas de
  robustesse) ; le *fingerprinting* identifie un contenu par ses caracteristiques sans le modifier
  (hash perceptuel) ; les *metadonnees C2PA* sont signees mais supprimees par une simple capture
  d'ecran ; le *model watermarking* marque les poids d'un modele, pas ses sorties.

## 1.2 Le schema general d'un watermark neuronal

Un **embedder** prend une image $x \in [0,1]^{H \times W \times 3}$ et un message
$m \in \{0,1\}^{k}$, et produit une perturbation $w$ ajoutee a l'image :

$$x_w = x + \alpha \cdot w, \qquad w = \mathrm{Emb}(x, m)$$

ou $\alpha$ regle la force. Un **extracteur** lit, dans une version eventuellement attaquee
$\tilde{x}_w$, un message souple $\tilde{m} \in [0,1]^k$ que l'on seuille a 0,5. Les deux reseaux
sont entraines ensemble : on tatoue, on **attaque l'image pendant l'entrainement** (JPEG, crop,
flou, luminosite...), puis on demande a l'extracteur de retrouver le message. Les attaques non
differentiables (JPEG) passent par un *straight-through estimator* :
$\tilde{x}_w = x_w + \mathrm{nograd}(T(x_w) - x_w)$. C'est le schema de HiDDeN (2018), dont
descendent presque toutes les methodes actuelles.

## 1.3 Mesurer l'imperceptibilite

| Metrique | Definition courte | Reperes |
|---|---|---|
| PSNR | $10 \log_{10}(1/\mathrm{MSE})$ en dB | > 40 dB : invisible a l'oeil sur la plupart des images ; Pixel Seal ~49 dB |
| SSIM | similarite de structure locale, dans $[0,1]$ | > 0,98 |
| LPIPS | distance entre activations d'un reseau profond | < 0,01 excellent ; Pixel Seal 0,0013 |
| JND | « just noticeable difference » ; 1 JND = 75 % des observateurs voient une difference | Pixel Seal 0,34 |
| FID | distance entre distributions d'images | utile pour juger une attaque sur un lot |

Piege classique : le PSNR moyenne les erreurs. Un watermark concentre dans une petite tache tres
visible peut avoir un excellent PSNR (Video Seal 0.0 : 53,7 dB mais artefacts visibles). Toujours
**regarder les images** et la carte du watermark amplifiee.

## 1.4 Mesurer la robustesse et la detection

La **bit accuracy** est la fraction de bits correctement lus. Pour decider si une image est
marquee, on calcule la probabilite qu'une image non marquee donne par hasard au moins autant de
bits corrects (chaque bit est alors une piece equilibree) :

$$p = P\big(X \ge n_{\text{correct}}\big), \qquad X \sim \mathrm{Binomiale}(k, 1/2)$$

On rapporte $-\log_{10} p$ (plus c'est grand, plus la preuve est forte). Ce test donne un taux de
faux positifs **garanti par la theorie**, sans calibration sur des images propres. Seuils
calcules pour $p < 10^{-6}$ :

| Taille du message $k$ | Bits corrects requis | Bit accuracy minimale | $-\log_{10} p$ a 100 % |
|---|---|---|---|
| 32 (WAM) | 30 / 32 | 93,8 % | 9,6 |
| 96 (Video Seal 0.0) | 72 / 96 | 75,0 % | 28,9 |
| 256 (Pixel Seal) | 167 / 256 | 65,2 % | 77,1 |
| 1024 (Chunky Seal) | 589 / 1024 | 57,5 % | 308 |

Lecon : **plus de bits donne plus de marge de detection**, a qualite egale. Quand on teste
plusieurs hypotheses sur la meme image (plusieurs rotations, plusieurs crops), il faut corriger
la p-value (Bonferroni : $p \times$ nombre d'essais).

## 1.5 Les familles d'attaques

| Famille | Exemples | Ce qui casse |
|---|---|---|
| Valeurs (*valuemetric*) | luminosite, contraste, saturation, teinte, flou, bruit | rarement grave pour les methodes modernes |
| Compression | JPEG, WebP, H.264 / HEVC pour la video | les hautes frequences ; tres dur en video |
| Geometrique | crop, redimensionnement, rotation, perspective, miroir | la synchronisation spatiale ; la plupart des methodes s'effondrent |
| Combinee | luminosite + crop + JPEG | revele les vrais ecarts entre methodes |
| Regeneration | encodeur-decodeur VAE, diffusion img2img a faible force | efface les signaux pixel en gardant le contenu |
| Adverse | PGD sur l'extracteur (boite blanche) ou sur un substitut | invisible et tres efficace si les poids fuient |
| Moyennage / collusion | moyenne des residus de N images marquees par la meme cle | estime le motif, permet de l'effacer ou de le copier |
| Forgerie (*spoofing*) | faire detecter un watermark sur une image propre | attaque la confiance, pas la robustesse |

# 2. Panorama des methodes

**Methodes classiques (1994-2015).** Spread spectrum : on ajoute un motif pseudo-aleatoire
module par chaque bit et on decode par correlation. QIM (*quantization index modulation*) :
on quantifie certains coefficients sur deux grilles decalees selon le bit. On travaille dans un
domaine de frequences (DCT 8x8 comme JPEG, ondelettes DWT, SVD) pour viser les bandes moyennes,
qui survivent a la compression et restent peu visibles. Rapides et sans apprentissage, mais
fragiles aux transformations geometriques. Stable Diffusion embarque encore `invisible-watermark`
(DWT-DCT-SVD).

**Methodes neuronales post-hoc (2018-2025).** HiDDeN (2018) fonde le paradigme
embedder / attaques simulees / extracteur. Suivent MBRS (JPEG reel et simule), CIN (reseaux
inversibles), TrustMark (Adobe, resolution arbitraire), InvisMark, **WAM** (Meta, Watermark
Anything : localise plusieurs messages dans une meme image, donc robuste au collage et au crop),
**Video Seal** (Meta, image et video, 256 bits), **Pixel Seal** (decembre 2025, etat de l'art),
**Chunky Seal** (1024 bits).

**A la generation.** Stable Signature (Meta, 2023) fine-tune le decodeur latent d'un modele de
diffusion pour que toutes ses images portent une cle ; Tree-Ring et Gaussian Shading marquent le
bruit initial (detection par inversion DDIM). SynthID (Google) marque images et texte en
production. Pour le texte, KGW (2023) partitionne le vocabulaire en liste verte / rouge a chaque
pas et favorise les tokens verts ; la detection compte les tokens verts (z-score). Les details
sont en section 5.F.

# 3. Pixel Seal decortique (papier impose numero 1)

## 3.1 Les trois problemes identifies

1. **Les pertes perceptuelles sont de mauvais proxys de l'oeil humain.** La MSE traite pareil un
   ciel uni et une texture ; l'humain voit l'artefact dans le ciel, pas dans la texture.
2. **Les objectifs se contredisent.** L'optimum de la perte perceptuelle est un watermark nul ;
   l'optimum de la robustesse est un watermark visible. L'entrainement s'effondre souvent (bits
   a 50 %) et depend lourdement des hyperparametres.
3. **L'ecart de resolution.** Les modeles sont entraines en 256x256 ; en production on calcule
   le watermark en basse resolution puis on l'agrandit, ce qui cree des artefacts et place
   l'extracteur hors distribution.

## 3.2 Les reponses

**Perte adverse seule.** Aucune MSE ni LPIPS : seulement la perte de message (entropie croisee
binaire sur les bits) et un discriminateur par patchs (celui de Stable Diffusion) qui apprend a
distinguer images originales et tatouees (perte *dual-hinge*). L'embedder est recompense quand
le discriminateur se trompe. Le discriminateur « voit » ce qu'un humain verrait : des motifs non
naturels, surtout dans les zones plates.

**Boosting du watermark.** On montre au discriminateur une version amplifiee
$\hat{x}_w = x + \beta (x_w - x)$. Avec $\beta > 1$ les artefacts sont plus faciles a reperer et
l'embedder apprend a les supprimer : plus invisible, un peu moins robuste ($\beta = 2{,}5$ :
PSNR 50,1 dB, combinee 86 %). $\beta < 1$ fait l'inverse ($\beta = 0{,}5$ : 46,2 dB, 96 %). C'est
un **bouton de compromis** explicite.

**Attenuation JND.** La perturbation est multipliee par une carte *Just Noticeable Difference*
non apprise : $x_w = x + \alpha\,(w \odot \mathrm{JND}(x))$. La carte combine un masquage de
luminance (l'oeil tolere plus d'erreur dans les zones tres claires ou tres sombres) et un
masquage de contraste (gradients de Sobel : on cache dans les contours). Dans le code Meta :
$\mathrm{JND} = \max(0,\, LA + CM - 0{,}3 \min(LA, CM))$. Resultat : le watermark vit sur les
contours, pas dans le ciel.

**Entrainement en trois etapes.** (1) message seul avec une force elevee $\alpha_0 = 1$ : le
watermark est visible mais le modele apprend a coder ; (2) on ajoute le discriminateur et on
diminue $\alpha$ de $\alpha_0$ a $\alpha_1 = 0{,}2$ par un cosinus :
$\alpha(t) = \alpha_1 + (\alpha_0 - \alpha_1)\cos(\tfrac{\pi}{2}\varphi)$ avec
$\varphi = \mathrm{clip}\big(\tfrac{t - N_{start}}{N_{end} - N_{start}}, 0, 1\big)$ ;
(3) finetuning a $\alpha_1$ fixe, eventuellement sur d'autres donnees ou attaques pour une
application precise. Les ablations sont nettes : demarrer directement a $\alpha = 0{,}2$, ou
activer le discriminateur des le debut, ou ajouter une MSE de poids 1, donne **50 % de bits :
le modele n'apprend rien**.

**Adaptation haute resolution.** On simule l'inference pendant l'entrainement : watermark calcule
en 256x256, agrandi, multiplie par la JND calculee **a la resolution d'origine**, discriminateur
applique a la resolution d'origine, attaques a la resolution d'origine, puis reduction pour
l'extracteur. Les images d'entrainement ont des tailles aleatoires entre 256 et 768 pixels.
Gain : combinee de 68 % (resolution fixe 256) a 94 %.

**Video : *temporal watermark pooling*.** A l'inference seulement, on moyenne les
caracteristiques de haut niveau de $k = 4$ images consecutives apres le $d$-ieme bloc du U-Net,
puis on les repete a la remontee. Acceleration x2,2 ($d = 1$) sans perte de qualite ni de
robustesse, et sans entrainement video.

## 3.3 Architecture et chiffres a connaitre

- Embedder : U-Net 43,8 M parametres, **sur la seule luminance Y** (1 canal en entree et en
  sortie dans la carte `pixelseal.yaml`) ; extracteur ConvNeXt-v2 Tiny 33,4 M ; 256 bits ;
  entraine 600 k pas, batch 256, sur SA-1B ; AdamW, lr $5 \cdot 10^{-4}$ ;
  $\lambda_{msg} = 1$, $\lambda_{adv} = 0{,}1$, $\alpha_1 = 0{,}2$.
- Imperceptibilite (1000 images Meta AI) : PSNR 48,9 dB, SSIM 0,9905, LPIPS 0,0013, JND 0,34.
- Robustesse (bit accuracy, images Meta AI / photos SA-1B) : valeurs 0,97 / 0,98 ; compression
  0,98 / 0,99 ; geometrique 0,95 / 0,93 ; combinee 0,91 / 0,94. Les meilleurs concurrents sur la
  combinee : InvisMark 0,94-0,98 mais avec 94 bits seulement ; Video Seal 1.0 a 0,69.
- Video (MovieGen / SA-V) : compression H.264-HEVC 0,86 ; combinee avec H.264 0,68-0,70. **C'est
  la zone ou il reste le plus a gagner.**

## 3.4 Ce qu'on en tire pour la competition

- **Points forts a exploiter en defense** : le recette en trois etapes est reutilisable telle
  quelle (y compris sur un petit modele entraine sur Mac) ; l'etape 3 est explicitement prevue
  pour specialiser le modele a de nouvelles attaques ; $\alpha$, $\beta$ et la JND sont trois
  boutons de compromis documentes.
- **Angles d'attaque** : poids publics (PGD boite blanche) ; signal localise sur les contours
  (un debruiteur preservant les contours ou une regeneration le visent directement) ; luminance
  seule (inutile d'attaquer la couleur) ; la video compressee et recadree.

# 4. Les deux autres papiers et leur lien avec le watermarking

## 4.1 Model Equality Testing : quel modele sert cette API ?

**Probleme.** Un fournisseur expose un modele « Llama-3 70B » derriere une API. Sert-il vraiment
les poids publies, ou une version quantifiee, fine-tunee, tatouee, voire un autre modele ? On ne
voit que des completions. Les auteurs formulent un **test a deux echantillons** : $P$ est la
distribution de reference (on dispose des poids et on peut echantillonner localement), $Q$ celle
de l'API, sur des paires (prompt, completion) avec des prompts tires d'une distribution fixee.
$H_0 : P = Q$.

**Statistique.** Le *Maximum Mean Discrepancy* (MMD) avec un noyau tres simple : deux
echantillons ne se comparent que s'ils repondent au **meme prompt**, et leur similarite est la
fraction de positions ou les tokens coincident (noyau de Hamming sur les $L$ premiers tokens) :

$$k\big((x, y), (x', y')\big) = \mathbb{1}[x = x'] \cdot \frac{1}{L}\sum_{l=1}^{L} \mathbb{1}[y_l = y'_l]$$

$$\widehat{\mathrm{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \ne j} k(z_i, z_j) + \frac{1}{m(m-1)}\sum_{i \ne j} k(z'_i, z'_j) - \frac{2}{nm}\sum_{i,j} k(z_i, z'_j)$$

Comme on sait echantillonner $P$, la loi de la statistique sous $H_0$ s'obtient par **simulation**
(tirer deux echantillons de $P$ un grand nombre de fois) plutot que par permutation, d'ou une
p-value exacte.

**Resultats.** Quelques centaines de completions suffisent a detecter la quantification, le
fine-tuning et le watermarking avec une puissance elevee, et l'audit d'endpoints commerciaux de
Llama montre qu'une part notable d'entre eux ne sert pas la distribution de reference. Les
chiffres precis sont dans `docs/research/model-equality-testing.md`.

**Lien avec le watermarking.** Un watermark texte de type KGW **deforme la distribution** de
sortie ; MET peut donc reveler qu'une API tatoue ses sorties, sans connaitre la cle. Inversement,
un watermark est une facon *cooperative* de repondre a la meme question (« ce texte vient-il de
ce modele ? ») avec beaucoup moins d'echantillons, mais seulement si le fournisseur joue le jeu.

## 4.2 SCRUB : desapprendre sans limite de taille

**Probleme.** Apres entrainement, on doit faire oublier un sous-ensemble $D_f$ (*forget set*) :
donnees personnelles, etiquettes erronees, biais. La reference est le reentrainement sans $D_f$,
trop cher. On veut un modele proche de cette reference : bon sur $D_r$ (*retain set*) et sur le
test, et qui se comporte sur $D_f$ comme sur des donnees jamais vues.

**Methode.** Le modele original sert de **professeur** fige ; l'**eleve** part de ses poids.
Avec $d(x) = \mathrm{KL}\big(p_{\text{prof}}(x)\,\|\,p_{\text{eleve}}(x)\big)$ (avec temperature) :

$$\min_{w}\; \frac{\alpha}{|D_r|}\sum_{x \in D_r} d(x;w) + \frac{\gamma}{|D_r|}\sum_{(x,y) \in D_r} \ell\big(f(x;w), y\big) - \frac{1}{|D_f|}\sum_{x \in D_f} d(x;w)$$

En pratique on **alterne** des *max-steps* (monter la divergence sur $D_f$) et des *min-steps*
(coller au professeur et a la verite terrain sur $D_r$), avec quelques *min-steps*
supplementaires a la fin pour restaurer la performance. **SCRUB+R** (*rewinding*) : quand le but
est la confidentialite, un taux d'erreur trop eleve sur $D_f$ trahit lui-meme l'appartenance
(effet Streisand) ; on revient au point de controle ou l'erreur sur $D_f$ egale celle de donnees
jamais vues.

**Evaluation.** Erreur sur $D_f$, sur $D_r$, sur le test, et **attaque d'inference
d'appartenance** (MIA) : un classifieur qui distingue, a partir de la perte, les exemples de
$D_f$ de ceux du test. Ideal : 50 % (le modele ne laisse rien fuir). Trois scenarios :
suppression de biais (erreur maximale sur $D_f$), resolution de confusion (etiquettes
melangees), confidentialite (MIA a 50 %). Details et hyperparametres :
`docs/research/scrub-unlearning.md`.

**Liens avec le watermarking.** (1) **Verification d'unlearning** : des donnees marquees
(watermark ou *canari*) laissent une trace mesurable dans un modele qui s'en est nourri
(« radioactivite », Sander et al. 2024) ; apres un vrai unlearning, la trace doit disparaitre :
c'est un test plus fort qu'une MIA. (2) **Attaque** : desapprendre un watermark integre au
modele (Stable Signature, watermark par backdoor) revient a un probleme d'unlearning cible.
(3) **Defense** : concevoir des watermarks qui survivent a l'unlearning ou au fine-tuning.

## 4.3 Le fil rouge : auditer l'IA par des tests statistiques

| Question | Signal | Test | Faux positifs controles par |
|---|---|---|---|
| Ce contenu est-il genere par notre modele ? | bits caches | binomial | la theorie (bits uniformes) |
| Cette API sert-elle le bon modele ? | distribution des completions | MMD, simulation de $H_0$ | la simulation sous $P$ |
| Ce modele a-t-il oublie ces donnees ? | pertes sur $D_f$ vs test | MIA, radioactivite | des donnees jamais vues |

Une epreuve qui melange les trois est plausible : par exemple « une API pretend avoir desappris
des donnees tatouees ; prouvez-le ou refutez-le avec un budget de requetes ».

# 5. Playbook : quel algorithme selon l'epreuve

Chaque fiche : metrique probable, baseline en moins d'une heure, approche avancee, outil du kit.

## A. Rendre un watermark plus robuste sous contrainte de qualite (defense)

- **Metrique probable** : bit accuracy ou TPR a FPR fixe apres un jeu d'attaques cache, avec un
  PSNR ou un LPIPS minimal impose.
- **Baseline** : Pixel Seal tel quel ; regler la force (`scaling_w`) pour **consommer tout le
  budget de qualite** (PSNR juste au-dessus du seuil impose) : chaque dB inutilise est de la
  robustesse perdue.
- **Gains sans reentrainement, cote detection** : (1) augmentation au test : extraire sur
  plusieurs versions (miroir, petites rotations, echelles, recadrages inverses) et moyenner les
  logits ou garder le meilleur score avec correction de Bonferroni ; (2) **code correcteur** : si
  la charge utile reelle est courte (32 bits), la repeter ou la coder en BCH dans les 256 bits et
  decoder par vote majoritaire ; (3) pour la video, agreger les bits souples sur toutes les images
  (la precision par image de 70 % devient pres de 100 % sur 90 images si les erreurs sont peu
  correlees).
- **Avance** : etape 3 de Pixel Seal, c'est-a-dire finetuner sur les attaques de l'epreuve
  (`train.py` de videoseal, ou le mini-Pixel Seal du kit sur Mac) ; ajouter des attaques de
  regeneration et des exemples adverses a l'entrainement ; motif de synchronisation pour inverser
  les transformations geometriques avant extraction ; watermark localise par tuiles (idee WAM)
  pour survivre aux crops extremes.

## B. Effacer un watermark sous contrainte de qualite (attaque)

- **Metrique probable** : taux de detection apres attaque (plus bas = mieux) combine a une
  degradation de qualite (PSNR, SSIM, LPIPS, FID) ; c'etait le format de la competition NeurIPS
  2024 *Erasing the Invisible*.
- **Baseline boite noire** : balayer avec `wmkit eval --suite removal` puis garder l'attaque
  la moins degradante qui passe sous le seuil de detection ; essayer d'abord la geometrie
  (rotation de quelques degres + recadrage + redimensionnement), puis JPEG fort, puis
  debruitage (non-local means, median).
- **Regeneration** : encodeur-decodeur VAE (`regen_vae`), puis diffusion img2img a faible force
  (bruit a $t \approx 0{,}1$ a $0{,}3$, puis debruitage) ; tres efficace contre les signaux de
  niveau pixel, cout en fidelite mesurable par LPIPS.
- **Boite blanche** (Pixel Seal public) : PGD en norme $L_\infty$ ($\epsilon = 2$ a $4/255$, 50
  pas) qui pousse les logits de l'extracteur vers des bits aleatoires ; perturbation invisible,
  detection effondree. Si l'organisateur utilise un modele prive, attaquer le modele public
  comme **substitut** et mesurer le transfert.
- **Boite grise** : entrainer un detecteur substitut (marque / non marque) sur les exemples
  fournis, l'attaquer par PGD ; si de nombreuses images portent la **meme cle**, estimer le motif
  par moyennage des residus haute frequence puis le soustraire (`estimate_pattern`).

## C. Forger un watermark (spoofing)

- **Boite blanche** : PGD vers un message cible sur une image propre (`pgd_forge`).
- **Boite noire** : estimer le motif sur des images marquees et l'ajouter a des images propres
  (`add_pattern`) ; WmForger (Meta, NeurIPS 2025) apprend un forgeur generique, code dans
  `third_party/videoseal/wmforger`.
- **Defense associee** : lier le message au contenu (signature de l'image dans le message),
  detecter les incoherences.

## D. Detecter un watermark sans la cle, ou identifier la methode

- **Le plus rentable** : passer tous les detecteurs publics (Pixel Seal, Video Seal, Chunky
  Seal, WAM, TrustMark...) et calculer leur p-value ; un $-\log_{10} p > 6$ tranche.
- **Sans detecteur** : steganalyse, c'est-a-dire un classifieur sur les residus haute frequence
  (filtres SRM, image moins sa version debruitee) ou sur le spectre de Fourier ; au niveau d'un
  lot, un test a deux echantillons de type MMD entre residus du lot suspect et residus d'images
  propres (l'idee de MET transposee aux images).
- **Watermarks de diffusion** (Tree-Ring) : inversion DDIM puis test sur l'anneau de Fourier
  du bruit retrouve.

## E. Video

- Pixel Seal avec *temporal pooling* ($k = 4$) pour la vitesse ; extraction agregee sur toutes
  les images ; le point dur est H.264 combine a un crop : prevoir un finetuning avec compression
  video simulee (straight-through) et une detection par agregation et test de Bonferroni.

## F. Texte genere par LLM

- **KGW** : a chaque pas, une graine derivee du token precedent (et d'une cle) partitionne le
  vocabulaire en liste verte de proportion $\gamma$ ; on ajoute $\delta$ aux logits verts. Detection
  sur $T$ tokens dont $|s|_G$ verts :
  $z = \dfrac{|s|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$ ; $z > 4$ est un seuil usuel.
  Ecarter les n-grammes repetes (Three Bricks) pour que la p-value reste valide.
- **Aaronson / Gumbel** : echantillonnage deterministe $\arg\max_v r_v^{1/p_v}$ avec $r$
  pseudo-aleatoire, sans distorsion de la distribution ; score $\sum_t -\log(1 - r_{t})$ suivant
  une loi Gamma sous $H_0$.
- **Attaques** : paraphrase par un autre LLM (la plus efficace), substitutions de mots,
  *emoji attack* ; *watermark stealing* (estimer la liste verte par frequences, puis forger).
- **Radioactivite** : detecter qu'un modele a ete entraine sur des textes tatoues en mesurant
  l'exces de tokens verts dans ses sorties.

## G. Identifier le modele derriere une API

- **Baseline** : MMD avec noyau de Hamming, completions de 50 tokens, quelques prompts fixes,
  loi nulle simulee avec le modele de reference local (`wmkit.audit.met`).
- **Avance** : si l'API renvoie des log-probabilites, tests de vraisemblance bien plus puissants ;
  choisir des prompts a haute entropie (plus discriminants) ; plan sequentiel pour economiser les
  requetes ; tester plusieurs candidats (quantifie 4/8 bits, fine-tune, tatoue) et designer le
  plus proche.

## H. Unlearning

- **Baseline** : *fine-tune* sur $D_r$ seul, et *gradient ascent* sur $D_f$ (NegGrad), pour
  avoir des points de comparaison en dix minutes.
- **Methode cible** : SCRUB, avec rewinding si l'epreuve mesure la confidentialite (MIA).
- **Evaluation a produire** : erreurs sur $D_f$, $D_r$, test ; AUC de la MIA ; ecart a un
  modele reentraine si on peut se l'offrir ; et, si les donnees a oublier sont tatouees, test de
  radioactivite avant / apres.

# 6. Pistes de recherche pour se distinguer

1. **Detection a l'epreuve de la geometrie par recherche au test** : grille de rotations et
   d'echelles inverses, score maximal avec correction de Bonferroni ; gain gratuit, mesurable.
2. **Codes correcteurs et p-values** : comparer repetition, BCH et codes polaires dans les 256 bits
   de Pixel Seal a charge utile courte ; tracer la courbe TPR a FPR $10^{-6}$ en fonction de la
   force d'attaque.
3. **Robustesse adverse** : PGD contre Pixel Seal public, puis finetuning adverse de l'etape 3 ;
   mesurer le compromis securite / imperceptibilite (une contribution que le papier ne traite pas).
4. **Boosting $\beta$ et JND comme boutons d'une frontiere de Pareto** : reproduire la figure 1 du
   papier sur un mini-modele et proposer un $\beta$ adaptatif selon le contenu.
5. **MET pour les images** : detecter un watermark au niveau d'un lot, sans cle, par un test MMD
   sur des residus ; c'est le pont naturel entre les papiers 1 et 2.
6. **Watermark comme preuve d'unlearning** : marquer $D_f$, entrainer, desapprendre par SCRUB,
   mesurer la radioactivite residuelle ; montrer qu'une MIA a 50 % peut coexister avec une trace
   detectable (ou l'inverse).
7. **Video** : agregation temporelle ponderee par la confiance par image, pour attaquer le point
   faible H.264 + crop.

# 7. Plan de bataille du jour J

**Premiere heure.** Lire l'enonce deux fois ; ecrire noir sur blanc la metrique exacte, les
contraintes (qualite minimale, budget de requetes, temps de calcul, materiel), le format de
soumission. Reproduire la baseline fournie ; brancher le harnais `wmkit eval` sur des donnees
qui ressemblent a celles du jury ; soumettre une premiere version tot.

**Ensuite, boucle courte.** Une hypothese, une mesure, une decision. Garder un tableau des essais
(CSV du kit). Ne jamais optimiser contre le jeu d'attaques local au point de le surapprendre :
garder des attaques de validation que l'on n'utilise pas pour regler.

**Repartition a deux ou trois.** Une personne sur la baseline et la mesure, une sur la methode
avancee, une sur la presentation (figures de compromis, exemples visuels, recit « audit
statistique »).

**Pieges techniques.**

- Quantifier en 8 bits (enregistrement PNG) **avant** de mesurer : un watermark de tres faible
  amplitude peut perdre une partie de son signal a l'arrondi.
- Mesurer la qualite sur l'image complete en resolution d'origine, pas sur la version 256x256.
- Fixer les graines ; noter la version des poids (carte videoseal) ; verifier l'espace couleur.
- Sur Mac : MPS d'abord, repli CPU si un operateur manque ; sur GPU fourni : meme code, le kit
  choisit `cuda` automatiquement.
- Toute p-value issue de plusieurs essais doit etre corrigee.

# 8. La boite a outils prete (`wmkit`)

Le depot `challenge/` n'est pas qu'une documentation : c'est une boite a outils fonctionnelle,
testee et prete a lancer. Il contient un environnement Python 3.12 (torch 2.15 avec MPS), le code
officiel de Meta (`third_party/videoseal` : Pixel Seal, Video Seal, Chunky Seal, WmForger,
entrainement) et le paquet `wmkit` : 33 modules, ~5 700 lignes typees (mypy strict), plus de 1 500
tests verts.

## 8.1 Ce qu'on a : la carte du code

Chaque brique mathematique est deja ecrite et verifiee. Le jour J, on appelle, on ne re-derive pas.

| Module | Fonctions-cles | A quoi ca sert |
|---|---|---|
| `wmkit.seal` | `embed`, `extract` | Pixel Seal / Video Seal / Chunky Seal derriere une interface unique |
| `wmkit.classical` | DCT-QIM, spread spectrum, DWT-DCT-SVD | Baselines simples pour comparer |
| `wmkit.attacks` | 54 attaques, suites `quick` / `paper` / `removal` / `all` | Degrader une image et mesurer la survie du watermark |
| `wmkit.metrics` | `bit_accuracy`, `neg_log10_pvalue`, `DETECTION_NEG_LOG10_P` | Decider « tatoue ou non » avec une p-value |
| `wmkit.eval` | `evaluate`, `summarize` | Benchmark complet (bit accuracy, $-\log_{10} p$, PSNR, SSIM, LPIPS) |
| `wmkit.adversarial` | `pgd_remove`, `pgd_forge`, `estimate_pattern` | Casser ou forger un watermark (boite blanche + moyennage) |
| `wmkit.text` | `KGWLogitsProcessor`, `TextDetection.neg_log10_p` | Tatouer et detecter du texte genere par LLM (KGW, Gumbel) |
| `wmkit.audit.met` | `two_sample_test`, `mmd_hamming`, `power_curve` | Tester si deux API servent le meme modele |
| `wmkit.audit.api` | `CachedCompletionClient.sample`, `seconds_until_ready` | Interroger une API a debit limite sans gaspiller de requetes |
| `wmkit.unlearn` | `scrub`, `ScrubConfig.large`, `evaluate_unlearning`, `neurips2023_score` | Desapprendre des donnees (SCRUB) et scorer l'oubli (MIA) |

Les 8 methodes de watermarking sont listees par `python -m wmkit list`. Note d'honnetete : la
methode `minipixelseal` (entrainement local d'un mini Pixel Seal) est **cablee mais non entrainee**
(reportee) ; `chunkyseal` est supportee mais ses poids (13 Go) ne sont pas telecharges.

## 8.2 Les trois epreuves, cote code

| Epreuve probable | Ce qu'on te demande | Les fonctions a appeler |
|---|---|---|
| Image (Pixel Seal) | detecter / tatouer / **effacer** / forger un watermark | `seal.extract` + `metrics.neg_log10_pvalue` ; `seal.embed` ; `adversarial.pgd_remove` ; `adversarial.pgd_forge` |
| Audit d'API (MET) | dire quel modele sert reellement une API | `audit.api.CachedCompletionClient` + `audit.met.two_sample_test` |
| Unlearning (SCRUB) | faire **oublier** un sous-ensemble, sans casser le reste | `unlearn.scrub` (config `large()` pour une classe entiere) + `unlearn.evaluate_unlearning` |

Le fil rouge reste le meme (section 4.3) : chaque epreuve se termine par un test statistique et une
p-value. On a le code de scoring pour les trois.

## 8.3 Ce qui est fait vs ce qu'on ecrit le jour J

- **Fait, teste, reutilisable** : toute la mecanique (p-values, PGD, MMD, SCRUB, MIA, scores
  officiels, 54 attaques, client d'API a debit limite). On n'ecrit pas une formule sous pression.
- **A ecrire sur place** : la colle specifique au format d'entree de l'epreuve (leur arborescence
  de fichiers, leur endpoint, leur checkpoint). Compter 20 a 50 lignes par epreuve, qui appellent
  nos fonctions. Le jour J = branchement + strategie (quels prompts, quelle attaque, quel seuil).

## 8.4 Toutes les commandes : le `Makefile`

Un `Makefile` a la racine evite de tout retaper. `make` seul liste les cibles.

| Commande | Effet |
|---|---|
| `make test` | Suite de tests rapide (~40 s ; exclut les tests lents) |
| `make smoke` | Sanity check en 2 s (import + CLI) |
| `make demo` | Tatoue une image avec Pixel Seal, extrait, affiche bit accuracy + p-value |
| `make data` | Telecharge les images de test (Kodak) |
| `make list` | Liste methodes, attaques et suites |
| `make eval` | Benchmark Pixel Seal complet sur la suite du papier (lourd) |
| `make lint` | ruff + mypy strict |
| `make demo-text` / `make demo-met` / `make demo-unlearn` | Une demo par papier |
| `make test-slow` / `make test-all` | Suite lente / suite complete |

Message aleatoire : on **omet** `--message`, ou on ecrit `--message random` (les deux tirent un
message depuis `--seed`). Les resultats sortent sur la sortie standard, les avertissements (doublon
`cv2`/`av`, deprecation `torch.jit`) sur l'erreur standard et sont inoffensifs ; ajouter `2>/dev/null`
pour une sortie propre.

# 9. Lancer et accelerer : MPS natif, 128 Go, Docker

## 9.1 Le chemin rapide sur Mac : le GPU Apple (MPS), nativement

`wmkit` choisit automatiquement le meilleur appareil (`get_device` : CUDA, sinon MPS, sinon CPU).
Sur un Mac Apple Silicon, c'est **MPS**, le GPU integre, et il est deja exploite sans rien
configurer. Mesure sur un Apple M5 (Pixel Seal, embed) :

| Batch | MPS | CPU | Acceleration |
|---|---|---|---|
| 1 image | 42 ms | 203 ms | x4.8 |
| 16 images | 40 ms/image | 246 ms/image | x6.2 |

Aucun operateur ne retombe sur le CPU, aucune erreur. **Le natif est donc le chemin rapide sur
Mac**, et il n'y a rien a « reparer ». Pour forcer un appareil : variable d'environnement
`WMKIT_DEVICE` (par ex. `WMKIT_DEVICE=cpu` pour comparer).

## 9.2 Traiter un dossier entier d'un coup : le mode batch

Pour le travail de masse (« parmi ces 5 000 images, lesquelles sont tatouees ? »), deux commandes
traitent un dossier complet en un seul appel, avec le modele charge une seule fois :

```bash
# Tatouer / detecter un dossier entier, par paquets (taille auto selon la RAM, ou forcee)
.venv/bin/python -m wmkit batch-embed   --method pixelseal --images data/in --out data/out --messages-out m.txt
.venv/bin/python -m wmkit batch-extract --method pixelseal --images data/out --expected m.txt
```

Les images sont regroupees par taille, la taille de batch s'adapte a la memoire disponible (ou se
force avec `--batch-size`), et l'extraction en batch donne **exactement les memes bits** qu'une
extraction une par une (verifie bit a bit dans les tests).

**Honnetete sur le gain de vitesse.** Mesure faite : sur Pixel Seal, le GPU Apple est **deja sature
des le batch 1**, donc le temps par image reste plat de 1 a 16 images (~41 ms en tatouage, ~15 ms en
extraction) et il n'y a pas de multiplication du debit. Le batch apporte surtout : le **chargement
unique du modele** sur tout le dossier, une API propre, et un gain reel de **1,5 a 2,5x sur les
methodes legeres** (DCT-QIM). Sur 16 Go, pousser le batch a 64 fait au contraire **chuter** le debit
(pression memoire). Le benefice de la memoire unifiee de 128 Go, c'est de pouvoir monter le batch
sans cette chute : s'il augmente le debit de Pixel Seal reste **a mesurer sur la machine cible**, pas
a supposer. Ne pas promettre un « x5 » sans l'avoir mesure la-bas.

## 9.3 Docker : pour la portabilite, pas pour la vitesse

Docker rend le kit reproductible a l'identique sur n'importe quelle machine (celle d'un coequipier,
un serveur de la competition). **Piege important :** sur Mac, Docker tourne dans une VM Linux qui
n'a **aucun acces au GPU Apple (MPS)**. Metal est une API macOS ; un conteneur Linux ne peut pas y
toucher, et aucun runtime (« Docker Runner », krunkit...) n'y change rien pour PyTorch. Un conteneur
sur Mac est donc **CPU seulement, plus lent que le natif**. Docker n'est le chemin rapide que sur un
host **Linux + GPU NVIDIA**, via l'image CUDA.

| Cible | Vitesse | Quand l'utiliser |
|---|---|---|
| `.venv` natif (MPS) | Rapide | Chemin par defaut sur ton Mac |
| Image Docker CPU | Lente (portable) | Reproduire a l'identique ailleurs, machine d'un coequipier |
| Image Docker CUDA | Rapide | Uniquement sur un host Linux + NVIDIA fourni |

```bash
make docker-build        # image CPU portable
make docker-run          # lance le kit dans le conteneur
make docker-test         # suite de tests dans le conteneur
make docker-cuda-build   # image GPU (host Linux + NVIDIA uniquement)
```

L'image CPU (`wmkit:cpu`, ~2,2 Go) est construite et verifiee : `docker run` liste les methodes, la
suite de tests passe dans le conteneur, et un tatouage/extraction via volumes monte donne les memes
resultats que le natif. Poids et donnees se montent en volume, jamais copies dans l'image. L'image
CUDA est ecrite mais ne se construit que sur un host Linux + NVIDIA (jamais testee ici).

Resume : **sur ton Mac, reste en natif (MPS) ; garde Docker pour partager le kit.**

# Annexe A. Formulaire

| Objet | Formule |
|---|---|
| Watermark additif avec JND (Pixel Seal) | $x_w = x + \alpha\,(\mathrm{resize}_\uparrow(w) \odot \mathrm{JND}(x))$, $w = \mathrm{Emb}(\mathrm{resize}_\downarrow(x), m)$ |
| Perte de message | $L_{msg} = -\frac{1}{k}\sum_i m_i \log \tilde m_i + (1 - m_i)\log(1 - \tilde m_i)$ |
| Perte adverse (dual hinge) | $L_D = \tfrac12[\mathrm{ReLU}(1 - D(x)) + \mathrm{ReLU}(1 + D(\hat x_w))]$, $L_{adv} = -D(\hat x_w)$ |
| Boosting | $\hat{x}_w = x + \beta (x_w - x)$ |
| Planning de force | $\alpha(t) = \alpha_1 + (\alpha_0 - \alpha_1)\cos(\tfrac{\pi}{2}\varphi)$ |
| p-value multi-bit | $p = \sum_{j \ge n_c} \binom{k}{j} 2^{-k}$ |
| PSNR | $10 \log_{10}(1 / \mathrm{MSE})$ pour des images dans $[0,1]$ |
| KGW | $z = (|s|_G - \gamma T) / \sqrt{T\gamma(1-\gamma)}$ |
| MMD sans biais | voir section 4.1 |
| SCRUB | voir section 4.2 |

# Annexe B. References

- T. Soucek, P. Fernandez et al. *Pixel Seal: Adversarial-only training for invisible image and
  video watermarking.* arXiv 2512.16874, 2025. Code : github.com/facebookresearch/videoseal
- I. Gao, P. Liang, C. Guestrin. *Model Equality Testing: Which Model Is This API Serving?*
  ICLR 2025, arXiv 2410.20247.
- M. Kurmanji, P. Triantafillou, J. Hayes, E. Triantafillou. *Towards Unbounded Machine
  Unlearning.* NeurIPS 2023, arXiv 2302.09880.
- P. Fernandez et al. *Video Seal: Open and efficient video watermarking.* arXiv 2412.09492, 2024.
- T. Sander et al. *Watermark Anything with Localized Messages.* ICLR 2025.
- P. Fernandez et al. *The Stable Signature: Rooting watermarks in latent diffusion models.* ICCV 2023.
- J. Zhu et al. *HiDDeN: Hiding data with deep networks.* ECCV 2018.
- J. Kirchenbauer et al. *A watermark for large language models.* ICML 2023.
- P. Fernandez et al. *Three Bricks to Consolidate Watermarks for Large Language Models.* WIFS 2023.
- T. Sander et al. *Watermarking makes language models radioactive.* NeurIPS 2024.
- S. Gowal et al. *SynthID-Image: Image watermarking at internet scale.* arXiv 2510.09263, 2025.
