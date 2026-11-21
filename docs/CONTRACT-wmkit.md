# CONTRACT wmkit (lot 1 + lot 2) — fige le 2026-11-18

Boite a outils de preparation au European AI Championship (Meta, Paris), sujet watermarking.
Code en anglais, docstrings Google, typage strict (mypy strict), aucun magic number
(constantes nommees en tete de module), Pydantic ou dataclasses figees pour les configs.

## 0. Conventions communes (deja commitees, ne pas modifier sans le directeur)

- `src/wmkit/types.py` : `Watermarker` (Protocol), `AttackSpec`, `AttackGroup`, `ATTACK_GROUPS`.
- `src/wmkit/config.py` : `get_device()`, `data_dir()`, `weights_dir()`, `results_dir()`.
- `src/wmkit/methods.py` : registre `METHODS`, `get_watermarker(name, n_bits=None, device="cpu", **kw)`.
  Chaque proprietaire de methode expose dans son module une fabrique
  `build(n_bits: int | None = None, device: str = "cpu", **kwargs) -> Watermarker`
  (sauf `wmkit.seal.load_seal(n_bits=None, device="cpu", card="pixelseal", **kw)`).
- Tenseurs : images `float32 [B,3,H,W]` RGB dans `[0,1]` ; messages `float32 [B,n_bits]`
  dans `{0,1}` ; bits souples `float32 [B,n_bits]` dans `[0,1]` = P(bit = 1).
  `extract` accepte une taille differente de celle de `embed` (crop, resize).
- Seuil de detection par defaut : `DETECTION_NEG_LOG10_P = 6.0` (p < 1e-6), dans `wmkit.metrics`.
- Environnement : `.venv` partage a la racine du depot principal. **Ne jamais lancer `uv sync`
  ni `uv add`** (videoseal est installe hors lock, voir `scripts/setup_env.sh`) ; une conda est
  active sur la machine : toute installation passe par
  `uv pip install --python /Users/nowais/Documents/code/challenge/.venv/bin/python <pkg>`
  et doit etre signalee dans le rapport (le directeur l'ajoute au pyproject).
- Lancer le code d'un worktree : `PYTHONPATH=$PWD/src /Users/nowais/Documents/code/challenge/.venv/bin/python ...`
- Tests : `pytest` ; tout test qui telecharge, charge des poids ou depasse ~10 s est marque
  `@pytest.mark.slow` (exclu par defaut, lance avec `-m slow`).
- Commandes lourdes (suite complete, entrainement > 1 min) : prefixer par `jarvis-heavy`.

## 1. Lot 1

### 1.1 `w-imgcore` — coeur image (proprietaire)

Fichiers : `src/wmkit/{io,messages,metrics,attacks,eval,cli,__main__}.py`,
`src/wmkit/classical/{__init__,dct_qim,spread_spectrum,dwt_dct_svd}.py`,
`tests/test_{io,messages,metrics,attacks,eval,classical,cli}.py`.

`wmkit.io`
- `load_image(path: str | Path) -> Tensor[3,H,W]` ; `save_image(img: Tensor[3,H,W], path) -> None`
  (PNG sans perte par defaut, JPEG si extension .jpg avec qualite 95) ;
  `list_images(directory, max_images: int | None = None) -> list[Path]` (tri lexical, png/jpg/jpeg/webp).

`wmkit.messages`
- `random_messages(batch: int, n_bits: int, generator: torch.Generator | None = None) -> Tensor[B,n_bits]`
- `bits_to_hex(bits: Tensor[n_bits]) -> str` ; `hex_to_bits(hex_str: str, n_bits: int) -> Tensor[n_bits]`

`wmkit.metrics`
- `bit_accuracy(soft_or_hard: Tensor[B,n], target: Tensor[B,n]) -> Tensor[B]` (seuil 0.5).
- `neg_log10_pvalue(n_bits: int, n_correct: Tensor[B]) -> Tensor[B]` :
  `-log10 P(X >= n_correct)`, `X ~ Binomial(n_bits, 0.5)` (eq. 2 du papier Pixel Seal),
  calcul stable en log-espace (scipy.stats.binom.logsf), jamais `inf` pour n_bits <= 1024.
- `psnr(x, y) -> Tensor[B]`, `ssim(x, y) -> Tensor[B]`, `lpips_distance(x, y) -> Tensor[B]`
  (lpips charge paresseusement, reseau `alex`).
- `tpr_at_fpr(scores_pos: ndarray, scores_neg: ndarray, fpr: float) -> float`.
- Constante `DETECTION_NEG_LOG10_P: float = 6.0`.

`wmkit.attacks`
- `get_attack(name: str) -> AttackSpec` (KeyError si inconnu), `list_attacks(group=None) -> list[str]`,
  `SUITES: dict[str, list[str]]`, `get_suite(name) -> list[AttackSpec]`,
  `straight_through(fn: AttackFn) -> AttackFn` (x + (fn(x) - x).detach()).
- Aleatoire (crop, perspective...) : deterministe via `torch.Generator` seme par `seed` du harnais ;
  signature des fn aleatoires : la graine est capturee par la fabrique, `fn(images)` reste unaire.
- Noms exacts (format `<op>_<param>`, parametre ecrit avec `format(p, "g")`) :

| groupe | noms |
|---|---|
| identity | `identity` |
| valuemetric | `brightness_{0.5,0.75,1.25,1.5,2}`, `contrast_{0.5,0.75,1.5,2}`, `saturation_{0.5,1.5}`, `hue_{-0.1,0.1,0.25}`, `grayscale`, `gaussian_blur_{3,9,17}`, `gaussian_noise_{0.02,0.05}` |
| compression | `jpeg_{40,50,60,70,80,90}`, `webp_{50,80}` |
| geometric | `crop_{0.33,0.5,0.71,0.9}` (fraction du COTE conservee, fenetre aleatoire, sortie plus petite), `resize_{0.5,0.75}` (sortie plus petite), `rotate_{5,10,30,90}` (taille conservee, coins noirs), `hflip`, `perspective_{0.1,0.3,0.5}` |
| combined | `combo_jpeg40_crop0.71_bright0.5`, `combo_jpeg60_crop0.71_bright0.5`, `combo_jpeg80_crop0.71_bright0.5` (ordre : brightness, crop, jpeg — attaque combinee du papier) |
| removal | `median_{3,5}`, `nlmeans_{5,10}` (cv2.fastNlMeansDenoisingColored, h), `jpeg_10`, `downup_{0.25,0.5}` (resize bas puis retour taille), `regen_vae` (VAE `stabilityai/sd-vae-ft-mse` via diffusers, charge paresseusement, marque slow dans les tests) |

- Suites : `quick` (identity, brightness_1.5, contrast_0.5, hue_0.1, gaussian_blur_9, jpeg_50,
  crop_0.71, resize_0.5, rotate_10, hflip, combo_jpeg40_crop0.71_bright0.5), `paper`
  (identity + valuemetric + compression + geometric + combined), `removal`, `all`.

`wmkit.eval`
- `evaluate(watermarker: Watermarker, images: Sequence[tuple[str, Tensor[3,H,W]]],
  attacks: Sequence[AttackSpec], *, seed: int = 0, device: str = "cpu",
  with_lpips: bool = False, include_clean: bool = True) -> pd.DataFrame`
- Une ligne par (image, attaque) ; colonnes EXACTES, dans cet ordre :
  `method, image_id, is_watermarked, attack, attack_group, n_bits, bit_acc, neg_log10_p, detected, psnr, ssim, lpips`
  (`psnr/ssim/lpips` = qualite de l'image tatouee AVANT attaque vs originale, repetee sur chaque ligne ;
  `lpips` = NaN si `with_lpips=False`). `include_clean=True` ajoute pour chaque image et chaque
  attaque une ligne `is_watermarked=False` (image originale attaquee, message attendu = celui
  qu'on aurait insere) : `detected` sur ces lignes = faux positif.
- `summarize(df) -> pd.DataFrame` : moyennes par `(method, is_watermarked, attack_group)`.
- `summarize_by_attack(df) -> pd.DataFrame` : moyennes par `(method, is_watermarked, attack)`.

`wmkit.cli` (`python -m wmkit ...` et script `wmkit`) — sous-commandes :
- `list` (methodes, attaques par groupe, suites)
- `embed --method M --in IMG --out IMG [--message HEX] [--seed N]` -> imprime le message hex
- `extract --method M --in IMG [--message HEX]` -> imprime bits hex, et si `--message` : bit_acc, neg_log10_p, detected
- `attack --attack NAME --in IMG --out IMG [--seed N]`
- `eval --method M[,M2...] --images DIR --suite quick|paper|removal|all [--max-images N]
  [--seed N] [--lpips] --out CSV` -> ecrit le CSV complet, imprime `summarize` en markdown.

`wmkit.classical` — trois `build(n_bits=None, device="cpu", strength=None, key=0)` :
- `dct_qim` : DCT 8x8 sur la luminance Y, QIM sur coefficients moyenne frequence, n_bits defaut 32.
- `spread_spectrum` : un motif pseudo-aleatoire +/-1 par bit (graine `key`), somme ponderee
  sur Y, decodage par correlation apres redimensionnement a la taille de reference ; n_bits defaut 32.
- `dwt_dct_svd` : DWT niveau 1 -> blocs DCT -> SVD (style `invisible-watermark`), n_bits defaut 32.
- Cible : PSNR >= 38 dB et bit_acc >= 0.95 sans attaque sur 256x256.

### 1.2 `w-seal` — Meta Seal, attaques adverses, donnees (proprietaire)

Fichiers : `src/wmkit/seal.py`, `src/wmkit/adversarial.py`, `src/wmkit/video.py`,
`scripts/fetch_data.py`, `scripts/fetch_weights.py`, `notebooks/01_seal_quickstart.ipynb`,
`tests/test_seal.py`, `tests/test_adversarial.py`, `tests/test_video.py`.
Ne pas importer `wmkit.metrics` ni `wmkit.attacks` (livres en parallele) : calcul inline.

`wmkit.seal`
- `load_seal(n_bits=None, device="cpu", card="pixelseal", scaling_w: float | None = None,
  lowres_attenuation: bool = True) -> SealWatermarker` ; cartes : `pixelseal`, `videoseal`
  (= `videoseal_1.0`), `videoseal_0.0`, `chunkyseal`. `n_bits` doit etre None ou egal a la taille native.
- `SealWatermarker.embed(images, messages)` / `.extract(images)` conformes au Protocol ;
  `extract` = sigmoide des logits de message de `model.detect(...)["preds"][:, 1:]`
  (verifier dans le code videoseal si le canal 0 est bien la detection pour chaque carte).
- `extract_logits(images) -> Tensor[B,n_bits]` differentiable (pour les attaques adverses).
- `embed_video(frames: Tensor[T,3,H,W], message: Tensor[n_bits]) -> Tensor[T,3,H,W]` et
  `extract_video(frames) -> Tensor[n_bits]` (agregation videoseal).
- `scaling_w` surcharge la force du watermark (compromis robustesse / imperceptibilite).

`wmkit.adversarial`
- `pgd_remove(extract_logits: Callable[[Tensor], Tensor], images, *, eps: float = 4/255,
  step: float = 1/255, steps: int = 50, target_bits: Tensor | None = None) -> Tensor`
  (L-inf ; sans cible : pousse les logits vers 0, i.e. bits aleatoires ; avec cible : vers les bits inverses).
- `pgd_forge(extract_logits, images, target_message, *, eps, step, steps) -> Tensor`
  (fait lire `target_message` sur une image propre).
- `estimate_pattern(watermarked: Tensor[N,3,H,W], *, denoise: str = "gaussian") -> Tensor[3,H,W]`
  (moyenne des residus haute frequence ; attaque par moyennage / collusion, meme taille requise),
  `subtract_pattern(images, pattern, strength=1.0)`, `add_pattern(images, pattern, strength=1.0)` (forgerie).

`wmkit.video` : `read_video(path) -> (Tensor[T,3,H,W], fps)`, `write_video(frames, path, fps, codec="h264", crf=23)`
via PyAV (pas de ffmpeg systeme sur la machine), `h264_attack(frames, crf) -> frames`.

`scripts/fetch_data.py` : `--dataset kodak|coco_val_subset|div2k_valid --n N` vers `data/<dataset>/`,
idempotent (ne retelecharge pas), affiche le nombre d'images. Kodak (24 images) obligatoire.
Disque libre ~26 Go : COCO limite par `--n` (defaut 500), pas d'archive complete conservee.

`scripts/fetch_weights.py` : precharge les checkpoints des 4 cartes (cache videoseal) et affiche leurs tailles.

### 1.3 `w-research` — notes de recherche (proprietaire)

Fichiers : `docs/research/*.md` uniquement (voir brief). Aucune modification de code.

## 2. Lot 2 (lance apres les merges du lot 1)

### 2.2 `w-textmet` — watermark texte LLM + Model Equality Testing (amende le 2026-11-18, apres docs/research)
Fichiers : `src/wmkit/text/{__init__,kgw,gumbel,attacks,generate,localize}.py`, `src/wmkit/audit/{__init__,met,api}.py`,
`scripts/demo_text_watermark.py`, `scripts/demo_met.py`, `tests/test_text.py`, `tests/test_met.py`.
Modele local par defaut : `HuggingFaceTB/SmolLM2-135M-Instruct` (constante `DEFAULT_LM`), surchargeable.
- `KGWConfig(gamma=0.25, delta=2.0, context_width=1, hash_key=15485863, seeding="lefthash")` (dataclass figee),
  `KGWLogitsProcessor(vocab_size, config)` (transformers LogitsProcessor),
  `kgw_detect(token_ids, vocab_size, config, *, dedup_ngrams=True) -> TextDetection` (z-score ET p-value binomiale exacte).
- `GumbelConfig(hash_key, context_width=1)`, `GumbelSampler`, `gumbel_detect(token_ids, vocab_size, config) -> TextDetection`
  (score $\sum -\log(1-r_t)$, p-value Gamma exacte via scipy, dedup des contextes repetes).
- `TextDetection(n_scored: int, n_green: int | None, score: float, z_score: float | None, p_value: float, detected: bool)` ;
  `detected = p_value < TEXT_DETECTION_P` avec `TEXT_DETECTION_P = 1e-3`.
- `wmkit.text.localize` : `token_scores(token_ids, vocab_size, config) -> ndarray[T]` (1 si vert, 0 sinon ; ou score Gumbel)
  et `localize_segments(token_ids, vocab_size, config, *, window=32, p_threshold=1e-3) -> list[tuple[int, int, float]]`
  (segments (debut, fin exclusive, p-value) de texte tatoue dans un texte mixte).
- `wmkit.text.attacks` : `substitute_tokens(ids, rate, vocab_size, rng)`, `delete_tokens(ids, rate, rng)`,
  `insert_tokens(ids, rate, vocab_size, rng)`.
- `wmkit.text.generate` : `generate(model, tokenizer, prompt, *, scheme: Literal["none","kgw","gumbel"], config, max_new_tokens, seed) -> list[int]`.
- `wmkit.audit.met` :
  - echantillon = `Sample(prompt_id: int, completion_ids: tuple[int, ...])` (dataclass figee) ;
  - `mmd_hamming(samples_p, samples_q, *, completion_len=50) -> float` (noyau Hamming, paires de meme prompt seulement,
    normalisation identique au code officiel `model-equality-testing`, documentee) ;
  - `two_sample_test(samples_p, samples_q, *, completion_len=50, pvalue="permutation"|"simulation", n_resamples=1000,
    reference_pool: Sequence[Sample] | None = None, seed=0) -> METResult(statistic, p_value, n_p, n_q)` ;
  - `power_curve(sample_p_fn, sample_q_fn, budgets: Sequence[int], *, n_trials, alpha=0.05, seed) -> list[tuple[int, float]]`.
- `wmkit.audit.api` : `CachedCompletionClient(fn: Callable[[str], str], cache_path, min_interval_s: float = 0.0)`
  (cache disque JSONL rejouable, compteur de requetes, respect d'un debit limite) — prevu pour une API a 1 requete / 15 min.

### 2.3 `w-unlearn` — SCRUB, baselines d'unlearning, scores de competition (amende le 2026-11-18)
Fichiers : `src/wmkit/unlearn/{__init__,scrub,baselines,mia,data,score}.py`, `scripts/demo_scrub_cifar.py`,
`tests/test_unlearn.py`.
- `ScrubConfig` (Pydantic) : valeurs par defaut du depot officiel (variante « small » : lr 5e-4, msteps 2, epochs 3,
  alpha 0.001, gamma 0.99, kd_T 4, wd 5e-4) et preset `ScrubConfig.large()` (msteps 3, epochs 10, alpha 0.5, gamma 1, T 2, wd 0.1).
- `scrub(student, teacher, retain_loader, forget_loader, config, device, *, rewind_val_loader=None)
  -> tuple[nn.Module, list[dict[str, float]]]` (SCRUB+R si `rewind_val_loader`).
- Baselines : `finetune_retain`, `negative_gradient`, `neggrad_plus`, `random_labels`.
- `evaluate_unlearning(model, retain_loader, forget_loader, test_loader, device) -> UnlearnReport(
  retain_acc, forget_acc, test_acc, mia_auc)` (MIA : regression logistique sur la perte, forget vs test, AUC en validation croisee).
- `wmkit.unlearn.score` : `neurips2023_score(forget_quality: float, ra_u, ra_r, ta_u, ta_r) -> float`
  et `forget_quality_from_epsilons(eps: Sequence[float]) -> float` (formules de docs/research/competition-intel.md 3.2).

### 2.4 `w-neural` — reporte (lot 3 optionnel)
Le mini Pixel Seal entrainable est reporte : la competition dure 24 h, le reglage et le finetuning du modele
public (`third_party/videoseal/train.py`) priment. L'entree `minipixelseal` du registre reste non implementee.
