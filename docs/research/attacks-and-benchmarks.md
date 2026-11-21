# Attaques sur watermarks d'images et benchmarks

Notes de preparation (w-research, 2026-11-18). Notation : image propre $x$, image tatouee $x_w$, extracteur
$D$, message $m \in \{0,1\}^k$, bit accuracy $BA = \frac{1}{k}\sum_i \mathbb{1}[D(x)_i = m_i]$. Les chiffres
portent leur source (arXiv id + table/section). Ce qui n'est pas verifie est marque `[A VERIFIER]`.

## 0. Ordres de grandeur a garder en tete (calcul local)

Le detecteur Pixel Seal / Video Seal teste $H_0$ : « les bits decodes sont independants de $m$ », avec
$p = P(X \ge n_{correct})$, $X \sim \mathcal{B}(k, 1/2)$ (2512.16874, eq. 2). Seuils de bits corrects
(calcul exact local de la queue binomiale) :

| $k$ bits | seuil $p<10^{-3}$ | seuil $p<10^{-6}$ |
|---|---|---|
| 32 | 26 (81 %) | 30 (94 %) |
| 96 | 64 (67 %) | 72 (75 %) |
| 256 (Pixel Seal, Video Seal 1.0) | 154 (60 %) | 167 (65 %) |
| 1024 (ChunkySeal) | 562 (55 %) | 589 (58 %) |

Consequences : (1) pour **effacer** un watermark 256 bits au seuil $10^{-6}$, il suffit de ramener $BA$
sous ~65 % ; $BA = 0.60$ donne $-\log_{10} p \approx 3.2$ (non detecte a $10^{-6}$, limite a $10^{-3}$) ;
(2) pour **forger**, $BA = 0.83$ sur 256 bits donne $-\log_{10} p \approx 27$ : une forgerie « imparfaite »
est massivement significative. Plus le payload est grand, plus les deux sont faciles.
(3) Viser $BA \approx 0.5$ et non $BA \approx 0$ : un detecteur « double-tail » flagge aussi les $BA$ trop
basses (Jiang et al., 2305.03807, sec. 1 et 5.1) ; le gagnant NeurIPS 2024 visait le message inverse et
le score officiel etait unilateral (2508.21072, sec. 3.1.1), a verifier sur le score de l'epreuve.

## 1. Modeles de menace

| Modele | Ce que l'attaquant a | Exemples |
|---|---|---|
| White-box | Poids de l'extracteur $D$ | WEvade-W (2305.03807) ; PGD sur $D$ |
| Beige-box / grey-box | Nom de l'algorithme, pas les parametres | NeurIPS 2024 beige-box (2508.21072) ; WAVES AdvEmbG (2401.08573) |
| Black-box avec oracle | Requetes au detecteur (API) | WEvade-B-Q / HopSkipJump (2305.03807) |
| No-box | Uniquement des images tatouees | Transfer attack (2403.15365), UnMarker (2405.08363), WmForger (2510.20468) |

Point competition : Pixel Seal est publie avec poids et code sous MIT (depot `facebookresearch/videoseal`,
cartes `pixelseal`, `videoseal`, `chunkyseal`). Si l'epreuve utilise le checkpoint public, on est en
**white-box** : l'attaque PGD sur l'extracteur domine tout le reste. Si les poids sont re-entraines ou
prives, on retombe en no-box et le checkpoint public devient un **surrogate** de transfert.

## 2. Regeneration (Zhao et al., 2306.01953, NeurIPS 2024)

- Attaque : $\hat{x} = \mathcal{A}\big(\phi(x_w) + \mathcal{N}(0, \sigma^2 I_d)\big)$ ; $\phi$ = encodeur (VAE,
  latent de diffusion), $\mathcal{A}$ = reconstruction (eq. 1). Bruit = destruction, reconstruction = qualite.
- Garantie (Thm 4.3) : pour un watermark $\Delta$-invisible en $\ell_2$ et $\phi$ $L_{x,w}$-localement
  Lipschitz, tout detecteur verifie $f(\epsilon_1) = \Phi\big(\Phi^{-1}(1-\epsilon_1) - L_{x,w}\Delta/\sigma\big)$
  (compromis TPR/FPR force). Exemple chiffre : $\sigma = 1.16\Delta$ (Fig. 2).
- Setup (sec. 5, experiences) : VAE CompressAI `Bmshj2018` et `Cheng2020`, facteurs de qualite 1 a 6 ; diffusion
  Stable Diffusion v2.1, noise steps {10, 30, 50, 100, 150, 200} soit $\sigma$ = {0.10 ... 0.57}, sampler PNDM.
  Detection : $p < 0.01$ (32 bits : >= 23 bits corrects, sec. 2.3).
- Resultats : RivaGAN efface a 98 % avec PSNR > 30 dB (sec. 1) ; DwtDctSvd, SSL, StegaStamp testes ;
  Tree-Ring (watermark semantique) **resiste** (Table 2). Code : `XuandongZhao/WatermarkAttacker`.

## 3. Purification par diffusion et regeneration controlee

- **Saberi et al.** (2310.00076, ICLR 2024), Thm 1 : apres purification au pas $t$,
  $e_0 + e_1 \ge 1 - \mathrm{erf}\Big(\frac{\sqrt{\bar\alpha_t}\, W(R,F)}{2\sqrt{2(1-\bar\alpha_t)}}\Big)$ ($e_0$ evasion,
  $e_1$ spoofing, $W$ Wasserstein $\ell_2$ entre images tatouees et non tatouees). Watermarks a faible
  perturbation : purification efficace ($t = 0.2$, sec. 3.2, Fig. 3). Forte perturbation (Tree-Ring, StegaStamp) :
  purification insuffisante, d'ou la *model substitution attack* (classifieur substitut + PGD) :
  AUROC Tree-Ring -> 0.14 avec $\ell_\infty$, $\epsilon = 2/255$ (sec. 1 et 3.3).
- **WAVES rinsing** (2401.08573, Table 2) : regenerations repetees `Rinse-2xDiff`, `Rinse-4xDiff` ;
  2x = meilleur compromis TPR/qualite sur Tree-Ring (sec. 3.2).
- **CtrlRegen** (Liu et al., 2410.05470, ICLR 2025) : regeneration depuis un bruit **propre** par diffusion
  controlee (adaptateur semantique + ControlNet spatial) ; CtrlRegen+ ajoute un bruit partiel reglable.
  Code `yepengliu/CtrlRegen`, poids HF `yepengliu/ctrlregen`.
- Chiffres sur des methodes post-hoc proches de Pixel Seal :

| Attaque (source) | Cible | $BA$ apres attaque | PSNR |
|---|---|---|---|
| VAE FLUX.1 [dev] (2510.20468, T2) | Video Seal 256 b | 0.99 (echec) | 34.3 |
| DiffPure FLUX.1, 3 derniers pas (2510.20468, T2) | Video Seal | 0.60 | 25.4 |
| CtrlRegen step 0.1 (2510.20468, T2) | Video Seal | 0.55 | 24.4 |
| DiffPure (2411.07231, T9) | WAM 32 b | 99.4 / 71.3 / 49.1 % | 28.4 / 27.0 / 24.9 |
| VAE attack (2411.07231, T9) | WAM ; MBRS ; TrustMark | 53.7 ; 97.7 ; 98.1 % | 25.5 |

Lecture : les post-hoc resistent a un VAE « doux » ; la diffusion les efface vers 25 dB, au prix d'hallucinations
de details (2510.20468, sec. 4.2). Pixel Seal n'est pas evalue contre la regeneration dans son papier
(2512.16874 : attaques valuemetric, compression, geometriques, combinees seulement) `[A VERIFIER]` en local.

## 4. Attaques adverses

- **White-box** (WEvade, 2305.03807, CCS 2023) : PGD sur $D$. `WEvade-W-I` fait tendre $BA$ vers 0 (contre
  par un detecteur double-tail) ; `WEvade-W-II` vise $BA \approx 0.5$ et atteint un taux d'evasion de 1
  (sec. 1). Variante black-box par requetes : `WEvade-B-Q` (HopSkipJump, label dur).
- **Surrogate detector** (Saberi ; WAVES `AdvCls-*`) : classifieur tatoue/non tatoue + PGD. WAVES : transfere
  mal vers Stable Signature et StegaStamp malgre une forte precision du substitut (sec. 3.2) ; Tree-Ring
  vulnerable. Les **embedding attacks** (`AdvEmbG/B`, perturber l'embedding d'un VAE ou de CLIP) cassent
  Tree-Ring, pas StegaStamp ni Stable Signature (sec. 3.2).
- **Transfer no-box** (Hu, Jiang, Guo, Gong, 2403.15365, ICLR 2025) : perturbation qui evade un ensemble de
  surrogates HiDDeN entraines soi-meme ; avec 100 surrogates, evasion > 85 % et perturbation moyenne < 0.25
  ($\ell_\infty$, echelle d'image `[A VERIFIER]`) ; 40 surrogates suffisent pour une cible CNN 30 bits
  (sec. 6.2). Code `hifi-hyp/Watermark-Transfer-Attack`.
- **UnMarker** (Kassis, Hengartner, 2405.08363, IEEE S&P 2025) : universel, black-box, data-free, query-free ;
  optimisations adverses sur les **amplitudes spectrales** (hautes puis basses frequences) + crop central
  leger (10 %). Detection des watermarks semantiques ramenee a 43 % au mieux (abstract). Cout : jusqu'a
  5 min/image sur A100 40 GB (discussion, apres sec. 6.2). Code `andrekassis/ai-watermark`.
- **WmForger en mode suppression** (2510.20468, T2 et T6) : gradient sur un modele de preference (sec. 3.2) ;
  Video Seal $BA$ 0.49 a 31.2 dB ($k$ = 500 pas), 0.85 a 38.8 dB ($k$ = 50). Meilleur PSNR que la diffusion.

## 5. Moyennage et collusion

- **Steganalyse par moyennage** (Yang, Ci, Song, Shou, 2406.09026 ; version publiee probable : « Can simple averaging
  defeat modern watermarks? », NeurIPS 2024 `[A VERIFIER]`) : hypothese additive $x_w = x_\emptyset + \delta_w$,
  $\hat\delta_w = \frac{1}{n}\big(\sum_i x_{w,i} - \sum_i x_{\emptyset,i}\big)$ (eq. 2) ; suppression
  $x_w - \hat\delta_w$, forgerie $x_\emptyset + \hat\delta_w$. Marche sur les methodes **content-agnostic**
  (DwtDctSvd, Tree-Ring : premiere attaque black-box reussie sur Tree-Ring) ; en grey-box, plus d'artefact
  visible au-dela de 50 images (sec. 4.3) ; jusqu'a $n$ = 5000 (sec. 4.4) ; contre-mesure : multi-cles et watermark content-adaptive.
- Contre Video Seal (content-adaptive), le moyennage ($n$ = 100, meme message) **echoue** : suppression
  $BA$ 0.89, forgerie 0.59 ; il reussit sur CIN (1.00) et MBRS (0.91) (2510.20468, T1-T2, sec. 4.2).
- Collusion au sens fingerprinting (moyenner des copies d'un meme contenu portant des messages differents) :
  pas de resultat publie sur Pixel Seal trouve `[A VERIFIER]`.

## 6. Forgerie et spoofing

- **Copy attack** (Kutter, Voloshynovskiy, Herrigel, SPIE 2000) : estimer le watermark par filtrage
  (denoising) puis le recoller sur une autre image, sans cle ni algorithme.
- **Spoofing par bruit tatoue** (Saberi, 2310.00076, sec. 3.2) : tatouer une image de bruit via l'API puis la
  melanger a une image reelle ; AUROC des images spoofees : RivaGAN 0.9965, DwtDctSvd 0.7533, Tree-Ring 0.7134 (Fig. 7).
- **Warfare** (Li et al., 2310.07726) : GAN entraine a ajouter/retirer le watermark ; 1000 images tatouees,
  6 jours sur 8 V100 par methode dans la reproduction de 2510.20468 (sec. 4.1).
- **Mueller et al.** (2412.03283, CVPR 2025) : *Imprinting* (rapprocher le latent d'une image propre de celui
  d'une image tatouee) et *Reprompting* (inverser puis regenerer avec un autre prompt) ; watermarks
  semantiques Tree-Ring et Gaussian Shading ; une seule image de reference ; modele de diffusion sans rapport.
- **WmForger** (Soucek, Rebuffi, Fernandez, Jovanovic et al., Meta FAIR, 2510.20468, NeurIPS 2025) :
  - modele de preference $R$ (ConvNeXt V2-Tiny) entraine avec une loss de ranking Bradley-Terry
    $p(x^+ \succ x^-) = \sigma\big(R(x^+) - R(x^-)\big)$ sur des artefacts **proceduraux** (vagues de Fourier,
    bruit, lignes), sans aucun vrai watermark, plus des paires adverses $\tilde{x}^- = x^- + \epsilon \nabla R(x^-)$
    pour avoir des gradients interpretables (sec. 3.1) ;
  - extraction : $\hat{w} = \arg\max_\delta R(x_w - \delta)$ par $k$ pas de gradient, puis $y_{\hat w} = y + \hat w$
    (eq. 3) ; une seule image tatouee, aucun acces au modele ;
  - forgerie (T1) : CIN 1.00, MBRS 0.83, TrustMark 0.61, Video Seal 0.83, PSNR 31.3 ; Warfare ($n$ = 1000)
    Video Seal 0.74 ; entrainement 120k pas, 8 GPU, 60 h V100 ; inference 6 s/image pour $k$ = 50 (App. C) ;
  - code et poids pre-entraines : `facebookresearch/videoseal/tree/main/wmforger` (MIT).

## 7. Attaques geometriques

- Desynchronisation classique : StirMark (Petitcolas, Anderson, Kuhn, Information Hiding 1998), resampling
  et petites distorsions geometriques qui « confondent la plupart des systemes du marche ».
- WAVES : `Dist-Rotation`, `Dist-RCrop`, `Dist-Erase`, combos `DistCom-Geo/All` (Table 2) ; les trois
  watermarks testes restent relativement robustes aux distorsions (sec. 4).
- Gagnant NeurIPS 2024 sur Tree-Ring : translation horizontale de 7 px puis restauration des 7 colonnes de
  gauche, sans modele ni entrainement (2508.21072, sec. 3.1.2).
- Pixel Seal est concu contre cela : $BA$ geometrique 0.95 (Meta AI) / 0.93 (SA-1b), combine
  crop 50 % + JPEG 40 + luminosite 0.91 / 0.94, contre 0.52 / 0.50 pour MBRS (2512.16874, Table 1).
  Rotations testees jusqu'a 90 degres, crops 0.32 a 1.0 (annexe) ; perspective, flips et warps non lus `[A VERIFIER]`.

## 8. Benchmarks

| Benchmark | Perimetre | Metrique detection | Qualite | Enseignements |
|---|---|---|---|---|
| WAVES (An, Ding, Rabbani et al., 2401.08573, ICML 2024) | 3 watermarks (Stable Signature, Tree-Ring, StegaStamp), 26 attaques (distorsion, regeneration, adverses), DiffusionDB / MS-COCO / DALL-E 3 | TPR@0.1%FPR | 8 metriques (PSNR, SSIM, NMI, FID, CLIP-FID, LPIPS, aesthetics, artifacts) normalisees : quantile 10 % -> 0.1, quantile 90 % -> 0.9, puis agregees | Tree-Ring tombe aux attaques adverses, Stable Signature a presque toutes les regenerations, StegaStamp le plus robuste ; classement d'attaque par Q@0.95P, Q@0.7P, Avg P, Avg Q (sec. 4) |
| W-Bench (Lu et al., 2410.18775, ICLR 2025) | 11 watermarks, edition par modeles T2I : regeneration stochastique/deterministe, global (Instruct-Pix2Pix, MagicBrush, UltraEdit), local (ControlNet-Inpainting, UltraEdit), image-to-video (SVD) ; 10 000 images UltraEdit | TPR@0.1%FPR (et @1%) | PSNR, SSIM, LPIPS, FID | La plupart des methodes echouent ; VINE-R (SDXL-Turbo, flou comme attaque substitut) : 99.66 % regen stochastique, 36.33 % SVD, PSNR 37.34 ; StegaStamp 91.09 % regen mais PSNR 29.65 (Table 1) |
| MarkDiffusion (Pan et al., 2509.10569, JMLR 2026) | Toolkit THU-BPM de watermarks **generatifs** de LDM (Tree-Ring, ROBIN, WIND, PRC, SEAL, VideoShield ...) | detection, attaques, qualite | 24 outils d'eval (v1) ; la doc annonce 11 algos / 33 outils `[A VERIFIER]` | Utile seulement si l'epreuve porte sur un watermark dans la generation, pas sur Pixel Seal (post-hoc) |
| Pixel Seal (2512.16874) | 9 methodes post-hoc, images Meta AI / SA-1b, videos MovieGen / SA-V | $BA$ et $-\log_{10} p$ | PSNR, SSIM, LPIPS, CVVDP, JND | Protocole probable des organisateurs Meta |

## 9. Tableau final : attaque -> cout -> efficacite attendue contre Pixel Seal-like (256 bits, post-hoc)

Efficacite : estimation a partir des chiffres ci-dessus, surtout de Video Seal (meme famille, 256 bits),
pas de mesures sur Pixel Seal sauf mention. A confirmer avec le harnais local `wmkit`.

| Attaque | Cout (temps, acces) | Efficacite attendue | Base |
|---|---|---|---|
| JPEG, luminosite, contraste, flou seuls | ~0, aucun acces | Nulle ($BA$ 0.97-0.98) | 2512.16874 T1 |
| Crop, rotation, resize seuls ou combo crop+JPEG+luminosite | ~0 | Faible ($BA$ 0.91-0.95, $-\log_{10}p$ > 50) | 2512.16874 T1 |
| Regeneration VAE (CompressAI, SD/FLUX VAE) | ms a 1 s/image GPU | Faible a qualite raisonnable (Video Seal 0.99) ; forte seulement vers 25 dB | 2510.20468 T2 ; 2411.07231 T9 |
| DiffPure / img2img diffusion (SD 2.1, FLUX, refiner) | 1-10 s/image GPU `[A VERIFIER]` | Elevee mais PSNR ~25 dB, hallucinations | 2510.20468 T2 |
| CtrlRegen / CtrlRegen+ | secondes/image, checkpoints HF | Elevee ($BA$ 0.55), PSNR 24.4 | 2510.20468 T2 |
| WmForger suppression ($k$ = 50 a 500) | 6 s a ~1 min/image, poids publics | Elevee a qualite meilleure ($BA$ 0.49, 31.2 dB) | 2510.20468 T2, T6 |
| PGD white-box sur l'extracteur public | secondes/image, poids MIT | Tres elevee si le checkpoint de l'epreuve = public ; viser $BA \approx 0.5$ | 2305.03807 ; hypothese |
| Transfert depuis surrogates (checkpoint public, ensemble) | heures a jours d'entrainement | Moderee, inconnue sur Pixel Seal `[A VERIFIER]` | 2403.15365 |
| UnMarker (spectral) | ~5 min/image A100 | Moderee a elevee, non mesuree sur Pixel Seal `[A VERIFIER]` | 2405.08363 |
| Moyennage $n \ge 100$ meme cle | ms, besoin de nombreuses images | Faible (suppression 0.89, forgerie 0.59 sur Video Seal) | 2510.20468 T1-T2 |
| Forgerie WmForger ($n$ = 1) | idem suppression | Elevee : $BA$ 0.83 sur 256 bits = $-\log_{10}p \approx 27$ | 2510.20468 T1 ; calcul sec. 0 |
| Combo gagnant NeurIPS 2024 (regen ciblee + restauration couleur CIELAB) | VAE fine-tune < 2 h A6000 | Meilleur compromis effacement/qualite connu | 2508.21072 sec. 3 |

## References

- Zhao, Zhang, Su, Vasan, Grishchenko, Kruegel, Vigna, Wang, Li. *Invisible Image Watermarks Are Provably Removable Using Generative AI*. NeurIPS 2024. https://arxiv.org/abs/2306.01953
- Saberi, Sadasivan, Rezaei, Kumar, Chegini, Wang, Feizi. *Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks*. ICLR 2024. https://arxiv.org/abs/2310.00076
- An, Ding, Rabbani et al. *WAVES: Benchmarking the Robustness of Image Watermarks*. ICML 2024. https://arxiv.org/abs/2401.08573
- Lu, Zhou, Lu, Zhu, Kong. *Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances* (W-Bench, VINE). ICLR 2025. https://arxiv.org/abs/2410.18775
- Pan, Guan, Fu, Si et al. *MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models*. 2025. https://arxiv.org/abs/2509.10569
- Jiang, Zhang, Gong. *Evading Watermark based Detection of AI-Generated Content* (WEvade). CCS 2023. https://arxiv.org/abs/2305.03807
- Hu, Jiang, Guo, Gong. *A Transfer Attack to Image Watermarks*. ICLR 2025. https://arxiv.org/abs/2403.15365
- Kassis, Hengartner. *UnMarker: A Universal Attack on Defensive Image Watermarking*. IEEE S&P 2025. https://arxiv.org/abs/2405.08363
- Yang, Ci, Song, Shou. *Steganalysis on Digital Watermarking: Is Your Defense Truly Impervious?* 2024. https://arxiv.org/abs/2406.09026
- Liu, Song, Ci, Zhang, Wang, Shou, Bu. *Image Watermarks are Removable Using Controllable Regeneration from Clean Noise*. ICLR 2025. https://arxiv.org/abs/2410.05470
- Li, Chen, Zhang et al. *Warfare: Breaking the Watermark Protection of AI-Generated Content*. 2023. https://arxiv.org/abs/2310.07726
- Mueller, Lukovnikov, Thietke, Fischer, Quiring. *Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models*. CVPR 2025. https://arxiv.org/abs/2412.03283
- Soucek, Rebuffi, Fernandez, Jovanovic, Elsahar, Lacatusu, Tran, Mourachko. *Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models* (WmForger). NeurIPS 2025. https://arxiv.org/abs/2510.20468
- Soucek, Fernandez, Elsahar et al. *Pixel Seal: Adversarial-only training for invisible image and video watermarking*. 2025. https://arxiv.org/abs/2512.16874
- Sander, Fernandez, Durmus, Furon, Douze. *Watermark Anything with Localized Messages*. ICLR 2025. https://arxiv.org/abs/2411.07231
- Shamshad et al. *First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge*. 2025. https://arxiv.org/abs/2508.21072
- Kutter, Voloshynovskiy, Herrigel. *Watermark copy attack*. SPIE 3971, 2000. https://doi.org/10.1117/12.384991
- Petitcolas, Anderson, Kuhn. *Attacks on Copyright Marking Systems*. Information Hiding 1998, LNCS 1525. https://www.cl.cam.ac.uk/~rja14/Papers/ih98-attacks.pdf
- Depot videoseal (Pixel Seal, ChunkySeal, WmForger, MIT) : https://github.com/facebookresearch/videoseal
