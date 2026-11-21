# Renseignement competition : European AI Championship Paris (20-22 novembre 2026)

Notes de preparation (w-research, 2026-11-18). Sources publiques uniquement ; l'information officielle
est maigre (page Luma, aucun reglement ni critere publie). Ce qui vient de sources non officielles ou
d'agregateurs est signale ; ce qui n'est pas confirme est marque `[A VERIFIER]`.

## 1. Ce qui est public sur l'evenement

| Element | Fait | Source |
|---|---|---|
| Nom, etape | « European AI Championship Paris », etape regionale d'une competition europeenne « cyber and AI » menee par le CISPA Helmholtz Center for Information Security | Luma |
| Organisateurs | Co-organise par PoC Innovation et CISPA ; hotes Alexandre Rousselle, Manmohit Singh Lal, Adam Dziedzic (CISPA) | Luma |
| Partenaires | « in partnership with Meta and Epitech » ; Fraunhofer AISEC (Franziska Boenisch) cite par des posts et un depot non officiel, absent de Luma `[A VERIFIER]` | Luma ; depot sinonchum |
| Dates | Ven. 20/11 19:00 talk d'introduction ; sam. 21/11 09:00 accueil, keynote Meta le matin **avec revelation des epreuves**, hacking a midi ; dim. 22/11 midi fin du hacking, demos devant jury, deliberation, fin 16:00 | Luma |
| Lieu | 24 rue Pasteur, 94276 Le Kremlin-Bicetre (campus Epitech) | Luma |
| Format | Hackathon de nuit de 24 h, 200 participants, equipes de 3 a 4, selection sur dossier | Luma |
| Dotation | 10 000 EUR de prix en especes offerts par Meta, repartition non publiee | Luma |
| Jury, criteres | « To be announced » ; aucun critere publie | Luma |
| Theme | Watermarking IA (« mark, detect and trace AI-generated content »), cadre EU AI Act art. 50 | posts PoC Innovation via recherche, depot sinonchum `[A VERIFIER]` |
| Lectures | E-mail CISPA du 17/11/2026 recommandant Pixel Seal, Model Equality Testing, SCRUB | brief directeur ; depot sinonchum |

Contexte reglementaire : l'art. 50(2) de l'AI Act impose aux fournisseurs de systemes generant audio, image,
video ou texte de marquer les sorties dans un format lisible par machine et detectable ; applicable a partir
du 2 aout 2026 (art. 113) selon artificialintelligenceact.eu.

Lecture : deux des trois papiers ne portent pas sur le watermarking (identification de modele, unlearning).
Inference (non officielle, partagee par le depot sinonchum) : les epreuves couvrent probablement trois
familles distinctes, une par papier, a l'image des editions precedentes a trois epreuves.

## 2. Lignee : CISPA European Cybersecurity & AI Hackathon Championship, saison 2025-2026

Format commun (page CISPA du championnat) : 6 etapes regionales de 24 h (jusqu'a 200 etudiants Bachelor,
Master, doctorants d'universites UE, equipes de 3-4), **trois epreuves** a l'interface IA / securite,
pitch devant jury apres 24 h ; top 3 de chaque etape en finale. Prix finale : 4 000 / 2 000 / 1 200 EUR,
400 EUR aux suivants. Organise par le SprintML Lab (F. Boenisch, A. Dziedzic). Boenisch : « The tasks are
completely different every time » (CISPA, Vienne).

| Etape | Epreuves connues | Evaluation connue | Source |
|---|---|---|---|
| Paris, 22-23/11/2025 (47 participants, 16 equipes) | (1) perturbation ciblee pour faire mal classer un modele ; (2) attribution d'images generees a leur modele ; (3) non decrite par CISPA | Modeles/soumissions envoyes a un serveur CISPA, leaderboard live, re-evaluation finale sur plus de donnees ; cle d'API par e-mail | CISPA Paris |
| Paris, detail cote equipe Loki (2e) | (1) adversarial black-box sur 100 images, API renvoyant les logits ; (2) attribution open-set : VAR-16d, RAR-XL, Taming Transformers, Stable Diffusion + « outlier », 250 images/classe ; (3) **detection de watermarks Tree-Ring, BitMark, Stable Signature**, score continu dans [0,1] | (1) L2 normalisee moyenne $\lVert\delta\rVert_2/\sqrt{CHW}$, echec = 1.0 ; requete API 1 / 15 min, soumission 1 / 5 min ; (2) $\mathrm{TPR}/(1 + 5\cdot\mathrm{FPR})$, macro sur 4 classes ; (3) **TPR@1%FPR** | github.com/Loki-Hackathons/cispa (le README date l'evenement de novembre 2024, incoherent avec CISPA : `[A VERIFIER]`) |
| Vienne, 13-14/12/2025 (~70) | dataset inference ; classification de texte ; **suppression de watermarks d'images** | Watermark residuel mesure par un « evaluator model » ; GPU du supercalculateur de Julich | CISPA Vienne |
| Stockholm, 14-15/02/2026 (~60) | model inversion (reconstruire les donnees d'entrainement) ; mauvaise classification ; attribution de modele | Inversion : proximite des donnees reconstruites aux vraies | CISPA Stockholm |
| Munich, 21-22/03/2026 | 3 epreuves « model robustness and trustworthy AI » ; keynotes Papernot (LLM dataset inference), Gunnemann | Leaderboard + presentation ou le jury questionne la methode | CISPA Munich |
| Barcelone, 11-12/04/2026 | property inference en black-box (shadow model, population de patients desequilibree) + 2 non decrites | - | CISPA Barcelone |
| Varsovie, 09-10/05/2026 (~100) | **LLM Watermark Detection** ; memorisation dans les LMM multimodaux ; Dataset Usage Cardinality Inference | - | NASK ; CISPA Varsovie |
| Finale, 02-03/07/2026, Sarrebruck / St. Ingbert (17 equipes, 50 participants) | SBA : **localisation de watermark dans du texte** ; mauvaise classification ciblee d'images ; reconstruction de visages depuis des gradients. CISPA : detection de watermarks dans du texte ; distinguer donnees d'entrainement et donnees generees ; reconstruction de donnees en apprentissage collaboratif | Scoreboard 24 h puis top 6 en presentation au jury ; vainqueur SeeWeedZ (Suede) | CISPA finale ; SBA Research |

Signaux forts :
1. **Le watermarking apparait a presque chaque etape** (detection image a Paris, suppression a Vienne,
   detection LLM a Varsovie, localisation texte en finale), toujours sous forme d'epreuve a leaderboard.
2. Metriques qui **punissent les faux positifs** : TPR@1%FPR, $\mathrm{TPR}/(1+5\,\mathrm{FPR})$. Calibrer
   les seuils sur des negatifs avant toute soumission.
3. Acces **black-box par API a debit limite** (1 requete / 15 min a Paris 2025) : budget de requetes a planifier.
4. Classement final = leaderboard sur test cache + **jury** (Munich, finale) : la methode doit etre expliquable.
5. La keynote de la finale 2026 etait donnee par Tomas Soucek (Meta FAIR Paris), premier auteur de Pixel Seal
   et de WmForger (CISPA finale) : forgerie et suppression Pixel Seal sont des sujets qu'il maitrise.
6. Les themes SprintML recurrents (dataset inference, memorisation, inversion, model stealing, attribution)
   recoupent MET (identification de modele) et SCRUB (unlearning, verifie par MIA).

## 3. Competitions de watermarking passees

### 3.1 NeurIPS 2024 « Erasing the Invisible: A Stress-Test Challenge for Image Watermarks »

- Organisation : UMD (Ding, Rabbani, An, Huang, Goldstein, Feizi et al.), inspiree de WAVES ; Codabench ;
  warm-up 31/07-15/09/2024, soumissions 16/09-07/11/2024 (site ; la page NeurIPS dit 18/07-31/10), atelier 15/12/2024 ; 7 000 USD de prix ;
  vainqueurs valides par juges humains pour eviter le « hacking » des metriques de qualite (site officiel).
- Pistes : **black-box** (methode inconnue) et **beige-box** (methode connue, pas ses parametres).
  Taille : 300 images tatouees par piste selon 2508.21072 (sec. 3) ; la page NeurIPS annonce 10k images en
  evaluation continue et 50k en finale : contradiction `[A VERIFIER]`.
- Score detection $D$ : TPR@0.1%FPR ; seuil = quantile 0.001 des distances $d(m, m')$ sur 10 000 images non
  tatouees ; $D$ = fraction d'images encore detectees (2508.21072, sec. 3).
- Score qualite $Q$ : combinaison ponderee normalisee de PSNR, SSIM, NMI, FID, CLIP-FID, LPIPS,
  $\Delta$Aesthetics, $\Delta$Artifacts (poids positifs pour « plus bas = mieux », negatifs sinon).
- Score total : les 6 lignes publiees verifient exactement $\sqrt{D^2 + Q^2}$ (plus bas = mieux), par exemple
  $\sqrt{0.043^2 + 0.136^2} = 0.143$ (calcul local sur 2508.21072, Tables 1-2) ; formule officielle `[A VERIFIER]`.
- Participation : 2 722 soumissions, 298 equipes ; le top 5 de chaque piste efface >= 89 % des watermarks
  (rapport technique NeurIPS 2025, Ding et al.).
- Podium : black-box MBZUAI 0.143, SHARIF 0.170, UFL 0.197 ; beige-box MBZUAI 0.157, SONY 0.183, SHARIF 0.256.
- Methode gagnante (MBZUAI, 2508.21072) :
  - beige-box StegaStamp modifie : 1 000 paires generees (SD 2.1) tatouees / message inverse, fine-tuning du
    VAE SDXL (MSE, Adam $10^{-5}$, 10 epochs, < 2 h A6000), optimisation au test (MSE + LPIPS + 0.5(1-SSIM)),
    transfert couleur/contraste en CIELAB ; Tree-Ring : translation de 7 px + restauration des 7 colonnes ;
  - black-box : clustering manuel des 300 images par artefacts spatiaux / spectre de Fourier (4 clusters :
    aucun artefact, bords, cercles, carres), puis img2img SD Refiner guide par captions ChatGPT, force par cluster.

### 3.2 Autres

- **BOWS-2** (Break Our Watermarking System 2, ECRYPT/Wavila, 17/07/2007-17/04/2008) : schema zero-bit
  « Broken Arrows » (Furon, Bas) ; 3 episodes : traitements courants, attaques par oracle, nombreuses images
  sous la meme cle. Metrique de classement (PSNR de l'image attaquee) `[A VERIFIER]`.
- **NeurIPS 2023 Machine Unlearning Challenge** (Kaggle, 1 121 equipes ; rapport 2406.09073 co-signe par
  E. Triantafillou, Kurmanji, Hayes et P. Triantafillou, soit les quatre auteurs de SCRUB) : oublier des visages d'un predicteur d'age ; score
  $F \times \frac{RA_u}{RA_r} \times \frac{TA_u}{TA_r}$ (eq. 2), $F = \frac{1}{|S|}\sum_s 2/2^{n(\varepsilon_s)}$,
  $n(\varepsilon) = \lfloor \varepsilon / 0.5 \rfloor$ (sec. 3.1.2), $\hat\varepsilon$ estime par attaques type MIA sur 512 modeles
  desappris vs 512 re-entraines (sec. 4), temps limite ~20 % du re-entrainement (sec. 3) ; plafonnement
  eventuel des ratios d'accuracy `[A VERIFIER]`. Precedent direct pour une epreuve SCRUB.

## 4. Formats d'epreuve plausibles (vu Pixel Seal, MET, SCRUB et les precedents)

| # | Epreuve | Metrique probable | Argument |
|---|---|---|---|
| 1 | **Effacer** un watermark Pixel Seal (256 bits) sur N images, qualite contrainte | Fraction encore detectee a $p < 10^{-6}$ ou TPR@0.1%FPR, combinee a une qualite composite (PSNR, LPIPS ...), type $\sqrt{D^2+Q^2}$ | Vienne 2025, NeurIPS 2024 ; Pixel Seal en MIT ; il suffit de passer sous ~65 % de bits corrects |
| 2 | **Forger** : apposer le watermark (meme message) sur des images propres a partir de 1 ou n images tatouees | Taux de detection des forgeries a $p < 10^{-6}$, sous contrainte PSNR | WmForger (Meta, auteur keynote 2026) ; securite du watermark, pas seulement robustesse |
| 3 | **Rendre robuste** : entrainer / regler un embedder (ou le `scaling_w` de Pixel Seal) sous contrainte PSNR ou JND | Moyenne de $BA$ ou $-\log_{10}p$ sur une suite d'attaques cachee, a qualite >= seuil | Sujet du papier Pixel Seal ; mais 24 h limitent l'entrainement : plutot reglage / fine-tuning |
| 4 | **Detecter** un watermark (zero-bit) parmi plusieurs methodes inconnues | TPR@1%FPR avec score continu | Paris 2025 (Tree-Ring, BitMark, Stable Signature) |
| 5 | **Decoder / attribuer** : retrouver le message ou l'utilisateur d'images tatouees et attaquees | Bit accuracy, exact match, ou identification top-1 | Pixel Seal 256 bits = tracabilite ; ChunkySeal 1024 bits |
| 6 | **Model Equality Testing** : quelles API servent le modele de reference (quantification, watermark, fine-tune, prompt systeme) | Accuracy / F1 de la decision a $\alpha$ fixe, puissance a budget de requetes fixe | Papier MET ; attribution de modele a Paris 2025 et Stockholm 2026 ; acces API limite |
| 7 | **Detecter un LLM tatoue** via son API ou des textes (KGW, Gumbel, SynthID) | TPR@1%FPR ou AUROC | Varsovie 2026 (LLM Watermark Detection) ; MET appliquee a un watermark |
| 8 | **Localiser** les segments tatoues dans un texte mixte | AUROC / F1 par token `[A VERIFIER]` | Finale 2026 (text watermark localization) |
| 9 | **Desapprendre** un forget set (SCRUB) en preservant l'utilite | Type NeurIPS 2023 : $F \times RA$ ratio $\times TA$ ratio, ou MIA proche du hasard + accuracy retain/test | Papier SCRUB ; ses 4 auteurs co-signent le rapport du challenge 2023 |
| 10 | **Verifier l'unlearning ou l'usage de donnees** : le modele a-t-il appris des donnees tatouees / backdoorees ? | TPR a FPR faible, p-value | Radioactivite (Sander et al.), dataset inference (Vienne, Varsovie) |
| 11 | **Desapprendre un watermark** : un modele entraine sur sorties tatouees ne doit plus le reproduire | $-\log_{10}p$ du watermark en sortie + utilite | Croise SCRUB et radioactivite ; plausible mais sans precedent public |
| 12 | **Detecter / attribuer** du contenu genere sans watermark (open-set) | $\mathrm{TPR}/(1+5\,\mathrm{FPR})$ macro | Paris 2025 (attribution VAR, RAR, Taming, SD) |

Priorites de preparation suggerees : 1, 2, 6, 9 (une par papier, et 1-2 couvrent Pixel Seal des deux cotes),
puis 4 et 7. Pour chaque format : baseline en < 1 h, scoring local qui reproduit la metrique, calibration
des seuils sur negatifs, et un recit methodologique court pour le jury.

## References

- Page Luma « European AI Championship Paris ». https://luma.com/iwlbokns
- CISPA, championnat (format, etapes, prix). https://www.cispa.de/en/cispa-hackathon-championship
- CISPA, etape de Paris 2025. https://cispa.de/en/cispa-hackathon-championship-paris
- CISPA, etape de Vienne 2025. https://cispa.de/en/cispa-hackathon-championship-vienna
- CISPA, etape de Stockholm 2026. https://cispa.de/en/cispa-hackathon-championship-stockholm
- CISPA, etape de Munich 2026. https://cispa.de/en/hackathon-munich
- CISPA, etape de Barcelone 2026. https://cispa.de/en/hackathon-barcelona
- CISPA, etape de Varsovie 2026. https://cispa.de/en/hackathon-warsaw
- CISPA, finale 2026. https://cispa.de/en/hackathon-final
- NASK, Varsovie 2026. https://science.nask.pl/en/news/14009
- SBA Research, finale 2026. https://www.sba-research.org/?p=35890
- Equipe Loki, notes de l'etape de Paris. https://github.com/Loki-Hackathons/cispa
- Depot non officiel de preparation (sinonchum). https://github.com/sinonchum/ai-watermark-hackathon-prep
- EU AI Act, art. 50. https://artificialintelligenceact.eu/article/50/
- Erasing the Invisible, site officiel. https://erasinginvisible.github.io/ ; page NeurIPS https://nips.cc/virtual/2024/competition/84795
- Ding et al. *A Technical Report on « Erasing the Invisible »: The 2024 NeurIPS Competition on Stress Testing Image Watermarks*. NeurIPS 2025. https://par.nsf.gov/biblio/10689521
- Shamshad et al. *First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge*. 2025. https://arxiv.org/abs/2508.21072
- Triantafillou et al. *Are we making progress in unlearning? Findings from the first NeurIPS unlearning competition*. 2024. https://arxiv.org/abs/2406.09073
- BOWS-2, IEEE SPS. https://signalprocessingsociety.org/publications-resources/data-challenges/break-our-watermarking-system-2nd-ed
- Soucek et al. *Pixel Seal*. 2025. https://arxiv.org/abs/2512.16874 ; *WmForger*. 2025. https://arxiv.org/abs/2510.20468
- Gao, Liang, Guestrin. *Model Equality Testing: Which Model Is This API Serving?* ICLR 2025. https://arxiv.org/abs/2410.20247
- Kurmanji, Triantafillou, Hayes, Triantafillou. *Towards Unbounded Machine Unlearning*. NeurIPS 2023. https://arxiv.org/abs/2302.09880
