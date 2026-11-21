# Ecosysteme Meta Seal (renomme Content Seal) — notes de recherche

Etat au 2026-11-18. Sources primaires : PDF arXiv (lus en texte integral), README des depots
`facebookresearch/*`, API GitHub (licences), site du projet. Chaque chiffre porte sa source.

**Point d'alerte.** L'URL citee par Pixel Seal, `https://facebookresearch.github.io/meta-seal`, renvoie
**HTTP 404** au 2026-11-18 ; le depot `facebookresearch/meta-seal` redirige (HTTP 301) vers
`facebookresearch/content-seal` (MIT). Le site actif est `https://facebookresearch.github.io/content-seal/`.
Il indique que « Content Seal Image is deployed at scale for Muse Image with a custom proprietary
implementation » : la version de production n'est pas la version open source.

## 0. Vue d'ensemble

| Travail | Modalite / role | Charge utile | Code | Licence (code / poids) |
|---|---|---|---|---|
| Stable Signature (ICCV 2023, 2303.15435) | image, in-model (decodeur LDM) | 48 bits | `stable_signature` | CC-BY-NC 4.0 (sauf `src/ldm`, `src/taming` en MIT) |
| AudioSeal (ICML 2024, 2401.17264) | audio, post-hoc, localise | 16 bits | `audioseal` | MIT, poids inclus depuis 2024-04-02 |
| Video Seal (2412.09492) | image + video, post-hoc | 96 (v0.0), 256 (v1.0) | `videoseal` | MIT (code + poids) |
| WAM (ICLR 2025, 2411.07231) | image, localise, multi-messages | 32 bits | `watermark-anything` | MIT pour `wam_mit.pth` ; poids du papier non commerciaux |
| Pixel Seal (2512.16874) | image + video, post-hoc | 256 bits | `videoseal` (`pixelseal`) | MIT |
| ChunkySeal (2510.12812) | image, haute capacite | 1024 bits | `videoseal` (`chunkyseal`) | MIT |
| WmForger (NeurIPS 2025 spotlight, 2510.20468) | attaque : forge / retrait | - | `videoseal/wmforger` | MIT |
| SyncSeal (2509.15208) | synchronisation geometrique | - | `wmar/syncseal` | MIT (depot `wmar`) |
| DistSeal (2601.16140, 2026) | latent, distille dans le generateur | - | `distseal` | MIT |
| WMAR (2506.16349) | generation d'images autoregressive | - | `wmar` | MIT |
| Three Bricks (2308.00113) | texte LLM, tests statistiques | zero-bit / multi-bit | `three_bricks` | CC-BY-NC 4.0 |
| Radioactivite (2402.14904) | texte, detection d'entrainement | - | `radioactive-watermark` | CC-BY-NC 4.0 |
| Post-hoc rephrasing (2512.16904), TextSeal (2605.12456), TextSeal2 (2610.40613) | texte | - | `textseal` | Apache-2.0 |
| Benchmark contamination (2502.17259), WM-Basin (2610.37266), Public detection (2610.12106) | texte, radioactivite / securite | - | `textseal` (contamination) | Apache-2.0 |

Licences : API GitHub `api.github.com/repos/facebookresearch/<repo>` (champ `license`) et fichiers
`LICENSE`/README (CC-BY-NC pour `three_bricks`, `stable_signature`, `radioactive-watermark`).
Pour la competition : **tout le bloc image/video/audio est MIT**, donc reutilisable sans reserve.

## 1. Conventions communes a toute la famille (a reutiliser telles quelles)

- **Test de detection multi-bit.** Message $m\in\{0,1\}^k$, extrait $\hat m$. Sous $H_0$ (image non tatouee),
  les bits extraits sont i.i.d. $\mathcal{B}(1/2)$, donc le nombre de bits concordants $M\sim\mathrm{Bin}(k,1/2)$ et
  $\mathrm{FPR}(\tau)=P(M>\tau\mid H_0)=I_{1/2}(\tau+1,k-\tau)$ (fonction beta incomplete regularisee)
  [2303.15435, eq. 2]. Video Seal reformule en p-value : $p=\sum_{j\ge k\cdot\mathrm{acc}}\binom{k}{j}2^{-k}$ et
  rapporte $\log_{10}p$ ; seuil $\log_{10}p<-6$ pour FPR $<10^{-6}$ [2412.09492, sec. 3.1, eq. 10].
  Code : `scipy.stats.binom.sf(n_correct - 1, k, 0.5)`.
- **N utilisateurs** (identification) : $\mathrm{FPR}(\tau,N)=1-(1-\mathrm{FPR}(\tau))^N\approx N\cdot\mathrm{FPR}(\tau)$
  [2303.15435, eq. 3]. A FPR $10^{-6}$ et $k=48$ : $\tau=41$ pour $N=1$, $\tau=44$ pour $N=1000$ [2303.15435, sec. 5.2].
- **Comparer des methodes a charges utiles differentes** : ne jamais comparer des bit accuracies brutes ;
  utiliser $\log_{10}p$ ou la capacite $C=k\,(1-H_2(p_{err}))$. Exemple du papier : $k=64$, acc. 0,99 donne
  $C=58{,}8$ ; $k=128$, acc. 0,9 donne $C=68{,}0$ [2412.09492, note 2 et App. A.1].
- **Augmentations non differentiables** (JPEG, H.264) : straight-through,
  $x_{aug}=x_w+\mathrm{nograd}(T(x_w)-x_w)$ [2412.09492, eq. 8].
- **Imperceptibilite** : PSNR $=-10\log_{10}\mathrm{MSE}$ sur $[0,1]$ [2303.15435, sec. 6.1] ; SSIM, LPIPS, VMAF
  (video) [2412.09492, Tab. 5].

## 2. Stable Signature (Fernandez, Couairon, Jegou, Douze, Furon — ICCV 2023)

- **Idee.** Fine-tuner uniquement le decodeur $D$ du VAE d'un LDM pour que toute image generee contienne une
  signature fixe $m$ lisible par un extracteur $W$ pre-entraine (type HiDDeN). Perte
  $\mathcal{L}=\mathcal{L}_m+\lambda_i\mathcal{L}_i$, $\mathcal{L}_m=\mathrm{BCE}(\sigma(m'),m)$, $\mathcal{L}_i$ = Watson-VGG entre
  $D_m(z)$ et $D(z)$, $\lambda_i=2$ ; 100 iterations AdamW, batch 4, < 500 images, **environ 1 minute sur un GPU**
  [2303.15435, sec. 4.2].
- **Chiffres.** Sans modification : 99 % detectes a FPR $10^{-9}$ ; crop gardant 10 % : 84 % ;
  crop + couleur + JPEG : 65 % [sec. 5.1]. Identification parmi $N=10^5$ : 98 % (non modifiee), 40 % (combinee),
  zero fausse accusation [sec. 5.2]. PSNR environ 30 dB, SSIM environ 0,9, FID quasi inchange (-0,3) [Tab. 1].
  Bit acc. : JPEG 50 = 0,88, crop 0,1 = 0,95, resize 0,7 = 0,91 [Tab. 2].
- **Attaques etudiees** [sec. 7] : auto-encodeurs neuronaux = il faut PSNR < 26 pour ramener la bit acc. vers 0,5,
  **sauf** avec le meme VAE KL-f8 que le LDM, qui efface a PSNR > 29 ; attaque adverse boite blanche sur $W$ :
  efface a budget de distorsion plus faible ; « model purification » (re-fine-tuning sans $\mathcal{L}_m$) :
  difficile sans artefacts. Les bits de deux utilisateurs moyennes suivent la « marking assumption ».
- **Competition.** Recette la plus rapide pour tatouer un generateur (1 min GPU) ; attaque de reference :
  re-encoder par le VAE du generateur.

## 3. AudioSeal (San Roman, Fernandez, Elsahar, Defossez, Furon, Tran — ICML 2024)

- **Idee.** Generateur + **detecteur** (pas decodeur) entraines conjointement avec une perte de localisation
  $\mathcal{L}_{loc}=\mathrm{BCE}(D(s')_t,y_t)$ par echantillon, perte perceptuelle TF-Loudness (masquage auditif),
  16 bits de message ; resolution 1/16 000 s [2401.17264, sec. 3]. Entrainement : 4,5 k h VoxPopuli, 16 kHz,
  600 k pas, Adam, lr $10^{-4}$, batch 32 [sec. 3.4].
- **Chiffres.** Qualite vs WavMark : SI-SNR 26,00 vs 38,25 (volontairement plus bas), PESQ 4,470 vs 4,302,
  ViSQOL 4,829 vs 4,730, MUSHRA 77,07 vs 71,52 [Tab. 1]. Robustesse : AUC moyenne 0,97 vs 0,84 ;
  point faible = passe-haut (AUC 0,61) ; EnCodec AUC 1,00 [Tab. 3]. Detection 3,25 ms vs 1710 ms (**485x**) sur
  audio non tatoue, 35x sur audio tatoue ; generation 14x plus rapide (GPU Quadro GP100) [sec. 5.5, Tab. 5].
- **Code.** `pip install audioseal` ; `AudioSeal.load_generator("audioseal_wm_16bits")`,
  `AudioSeal.load_detector("audioseal_detector_16bits")`, `detector.detect_watermark(wav)` ; v0.2 (2024-12-12)
  ajoute le streaming (`audioseal_wm_streaming`, trames de 320 echantillons) [README audioseal].
- **Competition.** Seule brique audio de la famille ; utile si une epreuve est multimodale.

## 4. Video Seal (Fernandez, Elsahar, Yalniz, Mourachko — 2024)

- **Idee.** Embedder U-Net (16 M parametres) + extracteur ViT (24 M), 256x256, $n_{bits}=96$ [2412.09492, sec. 2.1] ;
  inference a toute resolution par downscale -> watermark -> upscale du residu ; **propagation temporelle** :
  $w_i=\mathrm{Embedder}(x_i,m)$ si $i \bmod k=0$, sinon $w_i=w_{i-1}$ [eq. 2] ; extraction = moyenne des bits
  souples sur les trames [sec. 2.2.3]. Entrainement multi-etapes (pre-entrainement image, post-entrainement
  hybride, fine-tuning extracteur), H.264/H.265 dans la boucle via straight-through.
- **Chiffres** (Tab. 4, bit acc. / $\log_{10}p$, SA-1b image) : combinee 0,91 / -18,4 vs WAM 0,86 / -5,9,
  TrustMark 0,53 / -0,8 ; geometrique 0,83 / -16,4. Video SA-V combinee : 0,73 / -8,1. H.264 CRF 30 +
  luminosite 0,5 + crop 50 % : $\log_{10}p=-6{,}1$ en moyenne [sec. 4.1]. Imperceptibilite : PSNR image 47,39,
  video 48,02, VMAF 93,77 [Tab. 5] ; 42,0 GFlops [Tab. 6].
- **Code.** `videoseal.load("videoseal")` (v1.0, 256 bits, checkpoint `y_256b_img.pth`), `model.embed(x)`,
  `model.detect(x)["preds"]` (canal 0 = detection, bits a partir de l'indice 1) ; demo `aidemos.meta.com/videoseal`.
- **Competition.** Socle d'entrainement et d'evaluation (augmentations, metriques, cartes YAML) partage par
  Pixel Seal et ChunkySeal : c'est le code a maitriser en priorite.

## 5. Watermark Anything, WAM (Sander, Fernandez, Durmus, Furon, Douze — ICLR 2025)

- **Idee.** Le watermarking comme **segmentation** : l'extracteur predit pour chaque pixel un bit « tatoue ou
  non » + 32 bits de message ; plusieurs messages separes par DBSCAN sur les vecteurs par pixel
  [2411.07231, sec. 3]. Embedder 1,1 M parametres, extracteur 96 M (ViT-base) ; phase 1 a 256x256 sans
  contrainte perceptuelle, phase 2 avec attenuation JND, $x_m=x+\alpha_{JND}\,\mathrm{JND}(x)\odot\delta_\theta(x,m)$,
  $\alpha_{JND}=2$, jusqu'a 3 messages par image [sec. 4, 5.1] ; JND fait passer le PSNR d'environ 24 a 36 dB [Fig. 3].
- **Chiffres.** COCO PSNR 38,3 [Tab. 1]. COCO, TPR / bit acc. (%) : aucune 100 / 100 ; geometrique 99,3 / 91,8 ;
  inpainting 97,9 / 99,2 ; **splicing 100 / 95,3** (TrustMark 83,2 / 57,1) ; FPR 0,04 % [Tab. 2a]. Avec 10 % de
  l'image tatoue : 31 bits sur 32 en moyenne [sec. 5.4] ; 5 messages de 32 bits dans 5 zones de 10 % : mIoU > 85 %,
  bit acc. > 95 % apres flip + contraste [sec. 1, 5.5].
- **Code.** `load_model_from_checkpoint(json, ckpt)`, `wam.embed(img, msg)`, `wam.detect(img)["preds"]`
  ([1, 33, H, W]), `multiwm_dbscan(...)` ; `wam.scaling_w` regle le compromis qualite/robustesse [README].
- **Competition.** Seule methode qui localise : indispensable si l'epreuve comporte du splicing, de l'inpainting
  ou une evaluation par masque (IoU).

## 6. Pixel Seal (rappel, deja lu)

Entrainement adversarial-only (pas de perte perceptuelle pixel, $\lambda_{msg}=1{,}0$, $\lambda_{adv}=0{,}1$), planning
en 3 etapes, adaptation haute resolution par JND, 256 bits ; `videoseal.load("pixelseal")` [2512.16874, sec. 4 et 5.1 ; README videoseal].

## 7. ChunkySeal, « We Can Hide More Bits » (Petrov, Fernandez, Soucek, Elsahar — 2025)

- **Idee.** Bornes de capacite sous contrainte de PSNR (et de robustesse lineaire : rotation, crop, JPEG
  linearise) : les capacites theoriques depassent de plusieurs ordres de grandeur les ~100-200 bits des
  methodes actuelles a PSNR > 40 dB [2510.12812, sec. 1, Fig. 1]. Preuve empirique : Video Seal grossi.
- **Chiffres** (Tab. 3, SA-1B, vs Video Seal 256 bits) : 1024 bits (0,0052 bpp vs 0,0013) ; embedder
  1022,7 M vs 11,0 M ; extracteur 773,7 M vs 33,0 M ; PSNR 45,32 vs 44,42 ; LPIPS 0,0085 vs 0,0019 ;
  bit acc. globale 99,15 % vs 99,31 % ; JPEG Q 50-80 : 98,79 % vs 99,74 %. Clipping de gradient a 0,01
  « critique » ; tatouage sur les 3 canaux au lieu de la seule luminance Y [sec. 5].
- **Limites.** Latence elevee, difficile a deployer [sec. 6].
- **Competition.** Si l'epreuve note la capacite ou le nombre de bits a PSNR fixe : reference a battre, et
  argument que la capacite n'est pas saturee.

## 8. WmForger (Soucek, Rebuffi, Fernandez, Jovanovic et al. — NeurIPS 2025)

- **Idee.** Un **modele de preference** $R$ (ConvNeXt V2-Tiny) entraine par perte de rang Bradley-Terry
  $p(x^+\succ x^-)=\frac{e^{R(x^+)}}{e^{R(x^+)}+e^{R(x^-)}}$ a preferer une image propre a une image corrompue par des
  artefacts **procéduraux** generes en Fourier (vagues, bruit, lignes) — aucun watermark reel n'est vu
  [2510.20468, sec. 3.1]. Paires adverses $\tilde x^-=x^-+\epsilon\nabla R(x^-)$ une batch sur deux.
  Attaque : $\hat w=\arg\max_\delta R(x_w-\delta)$ par montee de gradient a $k$ pas fixes, puis forge
  $y_{\hat w}=y+\hat w$ ; **une seule image tatouee, boite noire** [sec. 3.2]. Entrainement : 120 k pas,
  8 GPU V100, 60 h [sec. 4.1].
- **Chiffres** (bit acc. ; 100 images SA-1b, meme message). Forge [Tab. 1] : CIN 1,00, MBRS 0,83,
  TrustMark 0,61, **Video Seal 0,83** a PSNR 31,3 ; moyennage de 100 images : 1,00 / 0,91 / 0,61 / 0,59 a PSNR 26,2 ;
  Warfare (n=1000) : 0,93 / 0,50 / 0,53 / 0,74. Retrait [Tab. 2] : Video Seal 0,49 a PSNR 31,2 ; re-encodage VAE
  FLUX.1 : 0,99 (inefficace) ; DiffPure FLUX : 0,60 a PSNR 25,4.
- **Enseignement.** Le moyennage marche contre les watermarks independants du contenu (CIN, MBRS) mais pas
  contre Video Seal (embedder dependant de l'image) ; Video Seal reste forgeable car son extracteur « ignore
  l'image » ; TrustMark est le plus dur a forger [sec. 4.2].
- **Competition.** Outil d'attaque cle en main contre un modele Pixel-Seal-like ; defense a envisager :
  rendre l'extracteur dependant du contenu.

## 9. Autres briques image 2025-2026

- **SyncSeal** (Fernandez et al., 2509.15208) : embedder/extracteur dedies a predire la transformation
  geometrique (crop, rotation) subie, pour l'inverser avant de decoder un autre watermark ; « upgrade » des
  methodes vulnerables au geometrique [abstract]. Code `wmar/syncseal`.
- **DistSeal** (Rebuffi et al., 2601.16140, janvier 2026) : watermarkers entraines dans l'espace latent puis
  distilles dans le generateur ou le decodeur latent ; jusqu'a 20x plus rapide que le pixel [abstract].
- **WMAR** (Jovanovic, Labiad, Soucek, Vechev, Fernandez, 2506.16349) : watermark au niveau token pour la
  generation d'images autoregressive ; probleme de reverse cycle-consistency resolu par fine-tuning
  tokenizer/detokenizer + couche de synchronisation ; p-values exactes [abstract].

## 10. Branche texte (details et formules : `text-watermarking.md`)

- **Three Bricks** (2308.00113) et **Radioactivite** (2402.14904) : voir `text-watermarking.md`, sections 3 et 7.
- **Benchmark contamination** (Sander et al., 2502.17259) : reformuler un benchmark avec un LLM tatoue ;
  $p=10^{-3}$ pour +5 % sur ARC-Easy (modeles 1 B pre-entraines sur 10 B tokens) [abstract].
- **Post-hoc rephrasing** (Fernandez et al., 2512.16904) : tatouer un texte existant en le faisant reecrire ;
  Gumbel-max simple bat les alternatives sous nucleus sampling ; beam search aide ; echec sur le texte
  verifiable (code), ou de petits modeles font mieux que de grands [abstract].
- **TextSeal** (Sander et al., 2605.12456, mai 2026) : Gumbel-max a double cle, score pondere par l'entropie,
  localisation multi-regions, compatible speculative decoding, « domine strictement SynthID-Text » ;
  6000 comparaisons A/B humaines, 5 langues, sans difference percue ; radioactif via distillation [abstract].
- **TextSeal2** (Sander et al., 2610.40613, octobre 2026) : fusion de cles en une cle virtuelle (copule
  gaussienne d'AutoMark), +8 a +22 points de detection vs TextSeal a FPR $10^{-3}$ (reponses de 100 tokens) [abstract].
- **WM-Basin** (Chang et al., 2610.37266, octobre 2026) : detection boite noire de distillation via un
  « bassin » de gradient propre a la cle ; a FPR 0,1 % : 79 % (n-gram 1) et 45 % (n-gram 3) vs 29 % et 1,3 % pour
  la reference token-level ; survit a quantification, pruning, RL [abstract].
- **Public detection** (Milis, Sander, Soucek, Huang, Fernandez, 2610.12106) : cle scindee publique/privee ;
  un detecteur public aide surtout la **forge**, que la cle privee detecte [abstract].

## 11. Ce qui sert en competition (synthese)

1. **Metrique de reference Meta** : $\log_{10}p$ binomial (section 1) et seuil $10^{-6}$ ; l'implementer a
   l'identique (deja dans le contrat `wmkit.metrics.neg_log10_pvalue`).
2. **Modeles prets** (MIT) : `videoseal.load("videoseal" | "pixelseal" | "chunkyseal")`, WAM `wam_mit.pth`,
   AudioSeal. Prevoir le telechargement des poids avant l'epreuve (reseau incertain).
3. **Attaques reconnues par Meta** : re-encodage VAE (Stable Signature), DiffPure / CtrlRegen, moyennage
   (n=100), WmForger une image, adverse boite blanche sur l'extracteur. Points faibles publies de Video Seal :
   combinaisons geometriques + codec (SA-V combinee 0,73), forgeabilite (0,83).
4. **Leviers de robustesse** : JND (WAM, Pixel Seal), straight-through pour les codecs, propagation temporelle
   $k=4$, synchronisation (SyncSeal) contre le geometrique, localisation (WAM) contre le splicing.
5. **Texte** : tests exacts + dedoublonnage, radioactivite. Les auteurs Meta insistent sur des p-values
   valides a FPR tres bas ; hypothese de travail : un rendu sans p-value correcte sera penalise
   `[A VERIFIER : non publie comme regle]`.

## References

- Fernandez, Couairon, Jegou, Douze, Furon. *The Stable Signature: Rooting Watermarks in Latent Diffusion Models*. ICCV 2023. https://arxiv.org/abs/2303.15435
- San Roman, Fernandez, Elsahar, Defossez, Furon, Tran. *Proactive Detection of Voice Cloning with Localized Watermarking* (AudioSeal). ICML 2024. https://arxiv.org/abs/2401.17264
- Fernandez, Elsahar, Yalniz, Mourachko. *Video Seal: Open and Efficient Video Watermarking*. 2024. https://arxiv.org/abs/2412.09492
- Sander, Fernandez, Durmus, Furon, Douze. *Watermark Anything with Localized Messages*. ICLR 2025. https://arxiv.org/abs/2411.07231
- Soucek, Fernandez, Elsahar, Rebuffi, Lacatusu, Tran, Sander, Mourachko. *Pixel Seal: Adversarial-only training for invisible image and video watermarking*. 2025. https://arxiv.org/abs/2512.16874
- Petrov, Fernandez, Soucek, Elsahar. *We Can Hide More Bits: The Unused Watermarking Capacity in Theory and in Practice*. 2025. https://arxiv.org/abs/2510.12812
- Soucek, Rebuffi, Fernandez, Jovanovic, Elsahar, Lacatusu, Tran, Mourachko. *Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models*. NeurIPS 2025. https://arxiv.org/abs/2510.20468
- Fernandez, Soucek, Jovanovic, Elsahar, Rebuffi, Lacatusu, Tran, Mourachko. *Geometric Image Synchronization with Deep Watermarking*. 2025. https://arxiv.org/abs/2509.15208
- Rebuffi, Tran, Lacatusu, Fernandez, Soucek, Jovanovic, Sander, Elsahar, Mourachko. *Learning to Watermark in the Latent Space of Generative Models*. 2026. https://arxiv.org/abs/2601.16140
- Jovanovic, Labiad, Soucek, Vechev, Fernandez. *Watermarking Autoregressive Image Generation*. 2025. https://arxiv.org/abs/2506.16349
- Fernandez, Chaffin, Tit, Chappelier, Furon. *Three Bricks to Consolidate Watermarks for Large Language Models*. IEEE WIFS 2023. https://arxiv.org/abs/2308.00113
- Sander, Fernandez, Durmus, Douze, Furon. *Watermarking Makes Language Models Radioactive*. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2402.14904
- Sander, Fernandez, Mahloujifar, Durmus, Guo. *Detecting Benchmark Contamination Through Watermarking*. 2025. https://arxiv.org/abs/2502.17259
- Fernandez, Sander, Elsahar et al. *How Good is Post-Hoc Watermarking With Language Model Rephrasing?* 2025. https://arxiv.org/abs/2512.16904
- Sander, Chang, Soucek et al. *TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection*. 2026. https://arxiv.org/abs/2605.12456
- Sander, Fernandez, Gat et al. *TextSeal2: Distortion-Free LLM Watermarking with Copula Mixing and Memoryless Races*. 2026. https://arxiv.org/abs/2610.40613
- Chang, Elsahar, Sander et al. *Watermark Basins: Black-Box Detection of Distilled LLMs*. 2026. https://arxiv.org/abs/2610.37266
- Milis, Sander, Soucek, Huang, Fernandez. *Could LLM Watermark Detection be Public?* 2026. https://arxiv.org/abs/2610.12106
- Depots : https://github.com/facebookresearch/content-seal , https://github.com/facebookresearch/videoseal , https://github.com/facebookresearch/audioseal , https://github.com/facebookresearch/watermark-anything , https://github.com/facebookresearch/stable_signature , https://github.com/facebookresearch/three_bricks , https://github.com/facebookresearch/radioactive-watermark , https://github.com/facebookresearch/textseal , https://github.com/facebookresearch/wmar , https://github.com/facebookresearch/distseal
- Site : https://facebookresearch.github.io/content-seal/ (l'ancien https://facebookresearch.github.io/meta-seal renvoie 404 au 2026-11-18)
