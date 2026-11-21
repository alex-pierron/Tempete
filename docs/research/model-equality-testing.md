# Model Equality Testing (MET) — « Which Model Is This API Serving? »

Gao, Liang, Guestrin (Stanford), ICLR 2025, arXiv 2410.20247 (v2, 8 avril 2025). Notes rédigées
à partir du PDF arXiv et du code officiel (lus le 2026-11-18). Convention : `§x` / `Tab. x` /
`App. x` renvoient au papier ; « code » renvoie au dépôt officiel.

## 1. Fiche
| Élément | Valeur | Source |
|---|---|---|
| Problème | Détecter si une API LLM boîte noire sert la même distribution qu'un modèle de référence | §1-2 |
| Statistique retenue | MMD avec noyau de Hamming sur les complétions, à prompt égal | §3, éq. 7 |
| Résultat phare | puissance médiane 77,4 % avec en moyenne 10 échantillons par prompt (N = 10m, m = 25) | Abstract, §4.1 |
| Audit réel | 11 endpoints sur 31 (9 fournisseurs, 4 Llama, été 2024) servent une distribution différente | Abstract, §5, Tab. 16 |
| Coût d'un audit | < 1 $ ; ex. Azure Llama-3 70B : Wikipedia 0,14 $, HumanEval 0,83 $, UltraChat 0,93 $ | §1, §5 |
| Code | `github.com/i-gao/model-equality-testing` (redirige vers `irenasaracay/model-equality-testing`), licence MIT, PyPI `model-equality-testing` 0.0.2 | README, `model_equality_testing/LICENSE`, pypi.org |
| Données | 1,1 M complétions dans le papier (App. B.1) ; le README annonce 1,6 M complétions, 540 prompts, `dataset.zip` de 37,1 GB | App. B.1, README |

## 2. Problème formel (§2)
- Tâche = distribution $\pi$ sur $m$ prompts, longueur max $L$ tokens. Référence $P$ et API $Q$ partagent
  vocabulaire $\mathcal V$ et paramètres de décodage.
- Échantillons (éq. 1) : $D_P=\{z^{(i)}=(x^{(i)},y^{(i)}) : x^{(i)}\sim\pi,\ y^{(i)}\sim P(\cdot\mid x^{(i)})\}_{i=1}^N$, idem $D_Q$ avec $Q$.
- Hypothèses (éq. 2) : $H_0:\ \pi(x)P(y\mid x)=\pi(x)Q(y\mid x)$ contre $H_1:\ \pi(x)P(y\mid x)\neq\pi(x)Q(y\mid x)$.
  Erreur de type I contrôlée à $\alpha$ ; on maximise la puissance contre des $Q$ inconnus.
- Difficulté : l'espace des paires a une taille $m|\mathcal V|^L$ (Llama-3 : $|\mathcal V|=128\,256$, $L=250$ usuel, note 4).
  Notation : $n_s\{S\}$ = nombre d'occurrences de $s$ dans $S$.

## 3. Statistique : MMD et noyaux (§3)
- Définition (éq. 3), noyau $k$ de norme unité, feature map $\phi$ :
  $\mathrm{MMD}_k(\pi P,\pi Q)=\lVert \mathbb E_{z\sim\pi P}\phi(z)-\mathbb E_{z\sim\pi Q}\phi(z)\rVert^2=\mathbb E_{P,P}k+\mathbb E_{Q,Q}k-2\,\mathbb E_{P,Q}k$.
- Noyau factorisé : $k(z,z')=\mathbb 1\{x=x'\}\,\tilde k(y,y')$, d'où $\mathrm{MMD}_k=\mathbb E_{x\sim\pi}\,\mathrm{MMD}_{\tilde k}(P(\cdot\mid x),Q(\cdot\mid x))$.
- Estimateur (éq. 4) : $\widehat{\mathrm{MMD}}=\frac{1}{N(N-1)}\big(\sum_{z\neq z'\in D_P}k+\sum_{z\neq z'\in D_Q}k\big)-\frac{2}{N^2}\sum_{z\in D_P,z'\in D_Q}k$.
- **Ce que fait réellement le code** (`tests.py::_mmd`) : (i) masque des paires de prompts différents ;
  (ii) normalisation $k/\sqrt{k(z,z)k(z',z')}$ (pour Hamming avec padding, $k(z,z)=L$, donc noyau $\in[0,1]$) ;
  (iii) diagonales mises à zéro ; (iv) chaque bloc est moyenné sur le **nombre de paires de même prompt**
  (et non sur $N(N-1)$ ou $N^2$) : $\widehat{\mathrm{MMD}}=\overline{K_{PP}}-2\,\overline{K_{PQ}}+\overline{K_{QQ}}$.
- Noyaux comparés :
  - **Hamming** (éq. 7) : $\tilde k_{\text{ham}}(y,y')=\sum_{i=1}^{L}\mathbb 1\{y_i=y'_i\}$, complétions plus courtes complétées à droite par un token spécial (le code utilise `-1`, donc deux paddings « matchent »).
  - all-substrings (éq. 5) : $\tilde k_{\text{all}}=\sum_{s\in\mathcal V^{\le L}}n_s\{y\}\,n_s\{y'\}$ (universel, très lent) ; one-hot (éq. 6) : $\mathbb 1\{y=y'\}$ (universel, test multinomial) ;
    Levenshtein $\exp(-\mathrm{Lev}(y,y'))$ (App. C.1, Tab. 9, moins puissant que Hamming).
  - Baselines multinomiales : $L_1=\sum_{z}|n_z\{D_P\}-n_z\{D_Q\}|/N$ (éq. 8) et
    $\chi^2=N^{-2}\sum_z\big[(n_z\{D_P\}-n_z\{D_Q\})^2-n_z\{D_P\}-n_z\{D_Q\}\big]/(n_z\{D_P\}+n_z\{D_Q\})$ (éq. 9).
- Propriété clé (App. A.1) : la mean embedding Hamming empile les **marginales par position** $P(y_i=v)$.
  Le MMD-Hamming est donc une pseudo-métrique : aveugle à deux lois de mêmes marginales positionnelles mais de jointures différentes.

## 4. Loi nulle et p-value (§3, App. A.2)
- **Algorithme 1 (simulation sous $H_0$)** : tirer $B$ fois $D_P^{(i)},D_Q^{(i)}\sim P$, stocker $\widehat{\mathrm{MMD}}$ ;
  $p=\#\{T_i\ge \widehat{\mathrm{MMD}}_{\text{obs}}\}/B$. Exige un accès en échantillonnage à $P$ (poids publiés).
  Dans le papier : $B=1000$ (100 pour all-substrings), distribution nulle réutilisée pour toutes les alternatives (App. B.2).
- **Algorithme 2 (permutation)** : concaténer $D_P\cup D_Q$, permuter, re-scinder, recalculer ; même règle $\ge$.
  Pas d'échantillons supplémentaires, puissance potentiellement plus faible ; validé en App. C.7 (100 permutations,
  tendances identiques). Le code calcule `np.mean(stats >= obs)` sans correction ; utiliser plutôt
  $(1+\#\{T_i\ge T_{\text{obs}}\})/(B+1)$ pour une p-value exactement valide (recommandation standard, pas du papier).
- **Null composite** (§5, éq. 10-11) : référence fp32 ($P_1$) OU fp16 ($P_2$) ; rejet ssi $p_1<\alpha\wedge p_2<\alpha$ ;
  le niveau reste $\le\alpha$ (preuve §5), au prix de puissance. Plusieurs distributions de prompts : Bonferroni $\alpha=0{,}01/3$ (note 8).

## 5. Complexité d'échantillon et puissance (§4, App. C)
Protocole (§4) : $\pi$ = 25 chaînes Wikipedia de 100 caractères (EN, DE, ES, FR, RU), $L=50$ tokens, température 1,
sans top-k/top-p, $\alpha=0{,}05$, 100 simulations Monte Carlo, 10 distributions de prompts, $N$ de $10m$ à $100m$.

| Résultat | Chiffre | Source |
|---|---|---|
| Hamming MMD, médiane sur alternatives (nf4, int8, fp16, watermark), N = 10m | 77,4 % | §4.1, Fig. 2 |
| Point faible | fp16 vs fp32 : puissance initiale faible, croît lentement | Fig. 2 droite, Fig. 5 |
| Longueur | Hamming et all-substrings gardent la puissance quand $L$ croît de 1 à 50 ; L1/χ² s'effondrent après $L=5$ | Fig. 2 milieu, Fig. 7 |
| Fine-tuning Llama-3 8B (lr $10^{-6}$, AdamW, 1000 exemples) | > 50 % de puissance dès 1 epoch (42 steps), même fine-tuning hors distribution (code) | §4.2, Fig. 3 |
| Changement de modèle (espace caractères, $L\approx1000$) | 100 % sauf intra-famille Llama : 3.1 8B vs 3.1 70B 75 %, 3 8B vs 3.1 8B 76 % | §4.3, Tab. 10 |
| Température 0,5 ou 1,5 vs 1 | 1,00 (Mistral à 1,5 : 0,97) | App. C.3, Tab. 12 |
| Espace caractères, Llama-3 8B, N = 10m → 50m | watermark 0,62 → 1,00 ; nf4 0,38 → 1,00 ; int8 0,30 → 0,96 ; audit < 5 $ | App. C.2, Tab. 11 |
| Nombre de prompts $m$ (1 à 100, N = 10m) | la puissance croît avec $m$ | App. C.4, Fig. 8 |
| Type de tâche | ne pèse que sur la détection de quantization, pas de watermark | App. C.4, Tab. 13 |
| Hamming vs one-hot | Hamming meilleur sur toutes les distributions de prompts (one-hot ≤ 0,06 sur les distributions Wikipedia) | Tab. 13 vs Tab. 14 |
| Accès aux logprobs (tests goodness-of-fit approchés) | meilleurs seulement pour N < 1000, puis battus par le two-sample | App. C.5, Fig. 9 |
| Coûts asymétriques ($|D_Q|=250$ fixe, $|D_P|$ jusqu'à $400m$) | gains de puissance légers, L1 gagne le plus | App. C.6, Fig. 10 |

Puissance Hamming, espace tokens, $L=50$, $N=10m$ (Tab. 8, moyenne sur 10 distributions de prompts) :

| Modèle | watermark (KGW) | nf4 | int8 | FPR fp32 |
|---|---|---|---|---|
| Llama-3 8B | 1,00 | 0,93 | 0,79 | 0,05 |
| Llama-3.1 8B | 0,62 | 1,00 | 0,08 | 0,05 |
| Llama-3 70B | 0,97 | 1,00 | 1,00 | 0,07 |
| Llama-3.1 70B | 0,54 | 1,00 | 1,00 | 0,06 |
| Mistral 7B | 0,72 | 0,89 | 0,30 | 0,04 |
| OLMo 7B / Gemma-2 9B / Phi-3 Mini | 0,47 / 0,43 / 0,75 | 0,99 / 0,12 / 0,63 | 0,36 / 0,06 / 0,24 | 0,06 |

Distances (Fig. 3 droite, $N=100m$, 10 bootstraps) : les modèles se regroupent par famille plutôt que par taille ;
ex. Llama-3 8B vs gpt-3.5-turbo-0125 ≈ 0,03.

## 6. Audit des API réelles (§5, App. B.1, C.8)
- 9 fournisseurs (Amazon Bedrock, Anyscale, Azure AI Studio, Deepinfra, Fireworks, Groq, Perplexity, Replicate,
  Together), 4 modèles (Llama-3 8B/70B, Llama-3.1 8B/70B), 31 endpoints, échantillons de juillet-août 2024.
- Tâches : Wikipedia ($m=25$, 50 tokens ≈ 1000 caractères) pour tous ; HumanEval et UltraChat ($m=20$,
  250 tokens ≈ 3000 caractères) pour les 8B. Tests **en espace caractères** (les API renvoient du texte),
  $T=1$, $\alpha=0{,}01$, $N=10m$, 10 répétitions, échec si taux de rejet moyen $\ge0{,}5$.
- Verdict global (Fig. 1 / Tab. 16, Bonferroni) : **11/31 échouent** — Amazon 4/4, Perplexity 4/4, Groq 2
  (3.1 8B, 3.1 70B), Fireworks 1 (3.1 8B). Le Tab. 1 (par tâche, sans Bonferroni) diffère légèrement
  (Replicate 3 70B rejeté sur Wikipedia, Amazon 3 70B accepté) mais totalise aussi 11.
- Seuls deux fournisseurs déclaraient une optimisation altérant la distribution : Fireworks (semantic caching),
  Together (quantization) (note 7).
- Taille d'effet : Perplexity Llama-3 8B vs fp32 = 0,03 en MMD, comparable à Llama-3 8B vs GPT-3.5-Turbo (0,03),
  Phi-3 mini (0,04), OLMo 7B (0,04) (§5). Perplexity semble servir une distribution de plus faible entropie (Box 2).
- MMD vs écart de précision HumanEval : corrélation modérée, $R^2=0{,}392$ dans le texte, 0,394 dans la Fig. 4.
- Piège pratique : le chat template compte. Celui de Llama-3.1 dans Transformers insère la date et ne collait
  pas au nombre de tokens de prompt facturés ; les auteurs ont utilisé le template Llama-3 (App. B.1).

## 7. Limites et suites
- Pseudo-métrique (App. A.1) ; faible contre fp16 ; puissance réduite en espace caractères ($|\mathcal V|=155\,063$) ;
  null composite pessimiste ; l'Algorithme 1 suppose de pouvoir échantillonner $P$ ; seulement $T=1$ testé.
- Pas d'adversaire : un fournisseur peut reconnaître les requêtes d'audit (prompts répétés N fois) et router vers le
  vrai modèle. **Non testés dans le papier** : system prompt caché, jailbreak, fine-tuning nuisible (grep négatif sur le PDF).
- **RUT** (Zhu et al., ICLR 2026, arXiv 2506.06975) : un seul appel API par prompt ; rang randomisé
  $r=F_{\text{ref}}(s^-)+U\cdot P(s=s_{\text{tgt}})$ d'un score (log-rank moyen des tokens, retenu parmi 5 candidats) de la
  réponse API dans la loi du modèle local, puis test de Cramér-von Mises d'uniformité ; évalué sur quantization,
  harmful fine-tuning, jailbreak prompts, substitution complète ; annonce une meilleure puissance que MMD et KS à budget égal.
- Cai et al. 2025 (arXiv 2504.04715) : les tests statistiques sur texte sont gourmands en requêtes et échouent contre
  des substitutions subtiles ; les méthodes à logprobs sont cassées par le non-déterminisme d'inférence ; proposent des TEE.
- Zhang et al. 2026 (arXiv 2603.01919, CCS 2026) : audit de « shadow APIs » utilisant MET + fingerprinting.

## 8. Code officiel (API)
```python
from model_equality_testing.utils import tokenize_unicode, pad_to_length   # pad = -1
from model_equality_testing.distribution import CompletionSample
from model_equality_testing.algorithm import run_two_sample_test
s1 = CompletionSample(prompts=np.array([0, 1, 0]), completions=pad_to_length(tokenize_unicode(c1), L=200), m=2)
pvalue, stat = run_two_sample_test(s1, s2, pvalue_type="permutation_pvalue", stat_type="mmd_hamming", b=100)
```
- Attention : `stat_type` vaut `"two_sample_L2"` **par défaut** ; `pvalue_type` ∈ {`permutation_pvalue`,
  `parametric_bootstrap`, `dummy`}. Autres statistiques : `mmd_kspectrum`, `mmd_all_subsequences`, `two_sample_L1`,
  `two_sample_chi_squared` ; goodness-of-fit : `g_squared`, `chi_squared`, `truncated_chi_squared`, `L1`, `L2`.
- Données : `model_equality_testing.dataset.download_dataset(root_dir)` (via gdown) puis
  `load_distribution(model=..., prompt_ids={"wikipedia_ru": [0, 3]}, L=1000, source="fp32"|"nf4"|"int8"|"watermark"|"amazon"|..., load_in_unicode=True)`.

## 9. Implémentation en 50 lignes (numpy, conforme au code officiel)
```python
import numpy as np
PAD = -1  # padding token; equal paddings count as matches, as in the official code

def encode(texts: list[str], L: int) -> np.ndarray:
    """Unicode code points, truncated / right-padded to L (character space, as for API audits)."""
    out = np.full((len(texts), L), PAD, dtype=np.int64)
    for i, t in enumerate(texts):
        cps = [ord(c) for c in t[:L]]
        out[i, :len(cps)] = cps
    return out

def hamming_mmd(X, px, Y, py) -> float:
    """X:[n,L], Y:[m,L] encoded completions; px, py: prompt ids. Kernel = matches / L, same prompt only."""
    L = X.shape[1]
    def block(A, pa, B, pb, same_set: bool) -> float:
        K = (A[:, None, :] == B[None, :, :]).sum(-1) / L      # normalized Hamming kernel
        mask = pa[:, None] == pb[None, :]                     # k = 0 across prompts
        if same_set:
            np.fill_diagonal(mask, False)                     # drop k(z, z)
        return K[mask].sum() / max(mask.sum(), 1)             # mean over same-prompt pairs
    return block(X, px, X, px, True) - 2 * block(X, px, Y, py, False) + block(Y, py, Y, py, True)

def permutation_pvalue(X, px, Y, py, B: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = hamming_mmd(X, px, Y, py)
    Z, pz, n = np.concatenate([X, Y]), np.concatenate([px, py]), len(X)
    hits = 0
    for _ in range(B):
        idx = rng.permutation(len(Z))
        a, b = idx[:n], idx[n:]
        hits += hamming_mmd(Z[a], pz[a], Z[b], pz[b]) >= obs
    return obs, (1 + hits) / (B + 1)

def simulated_null_pvalue(obs: float, sample_P, B: int = 1000) -> float:
    """sample_P() -> (X, px, Y, py) with BOTH samples drawn from the reference P (Algorithm 1)."""
    T = np.array([hamming_mmd(*sample_P()) for _ in range(B)])
    return float((1 + (T >= obs).sum()) / (B + 1))

def audit(p_fp32: float, p_fp16: float, alpha: float = 0.01) -> bool:
    """Composite null (eq. 11): flag the API only if it differs from BOTH references."""
    return p_fp32 < alpha and p_fp16 < alpha
```
Coût : $O(N^2L)$ mémoire si vectorisé ; au-delà de ~$10^4$ échantillons, boucler par prompt (seules les paires de
même prompt comptent). Réglages du papier : $N=10m$ (audit) à $50m$ (puissance quasi totale), $L$ = 1000 caractères.

## 10. Liens avec le watermarking
- **Le papier traite explicitement le watermark comme une substitution à détecter** : KGW (Kirchenbauer 2023) via
  Transformers, « default bias of 2.5 » et context width 1 (§4.1, App. B.1). Le $\gamma$ (greenlist ratio) n'est pas
  donné ; le défaut actuel de `WatermarkingConfig` est `greenlist_ratio=0.25`, `bias=2.0`, `seeding_scheme="lefthash"`
  (source Transformers) — l'écart 2,5 vs 2,0 est à [A VERIFIER] selon la version utilisée.
- Verdict : un KGW à $\delta=2{,}5$ **est détectable** sans connaître la clé : 0,43 à 1,00 de puissance en espace tokens à
  $N=10m$ (Tab. 8) ; 0,23 à 0,62 en espace caractères (Tab. 10) ; 1,00 pour Llama-3 8B à $N=50m$ (Tab. 11) ; mais 0,06 à
  0,50 seulement sous null composite avec Bonferroni (Tab. 17). Le watermark est plus dur à voir que nf4 et que
  les changements de modèle (§4.3).
- Pourquoi ça marche : KGW ajoute $\delta$ aux logits d'une liste verte dépendant du token précédent, ce qui déplace les
  marginales par position, exactement ce que capte la mean embedding Hamming (App. A.1).
- Watermarks « distortion-free » (Aaronson/Gumbel, Kuditipudi) : non testés par MET. Analyse (non vérifiée
  expérimentalement) : ils préservent la loi **en moyenne sur la clé**, mais à clé fixe et contexte fixe le
  Gumbel-max est déterministe, donc N tirages du même prompt se ressemblent beaucoup, ce qui donne un
  excès de collisions intra-échantillon que Hamming et même one-hot détectent. Kuditipudi prévoit `shift-generate`
  (décalage aléatoire de la clé, arXiv 2307.15593, Algorithme 4) pour limiter ce problème.
- Travail dédié : Gloaguen, Jovanović, Staab, Vechev, « Black-Box Detection of Language Model Watermarks » (ICLR 2025,
  arXiv 2405.20777) : tests statistiques qui détectent la présence et estiment les paramètres des trois familles
  de watermarks avec peu de requêtes, validés sur des API réelles.
- Pour une épreuve image (type Pixel Seal) : le cadre MET se transpose (la conclusion du papier le suggère pour la
  génération d'images). On remplace le noyau Hamming par un noyau sur des features (résidu haute fréquence, embedding
  d'un extracteur) pour tester « cette API d'images tatoue-t-elle ? » ou « est-ce le même générateur ? ». C'est une
  proposition de notre part, pas un résultat publié.
- En compétition, sur « quel modèle sert cette API ? » : (1) si un watermark à clé connue est attendu, son
  score de détection (z-score KGW, p-value) est le test le plus puissant ; (2) sinon MET-Hamming à $N\approx 10m$-$50m$,
  plusieurs langues et prompts, $T=1$, en espace caractères ; (3) contrôler template et system prompt avant de conclure.

## Références
- I. Gao, P. Liang, C. Guestrin. Model Equality Testing: Which Model Is This API Serving? ICLR 2025. https://arxiv.org/abs/2410.20247 ; code https://github.com/i-gao/model-equality-testing
- A. Gretton et al. A Kernel Two-Sample Test. JMLR 2012. https://jmlr.org/papers/v13/gretton12a.html
- J. Kirchenbauer et al. A Watermark for Large Language Models. ICML 2023. https://arxiv.org/abs/2301.10226
- R. Kuditipudi, J. Thickstun, T. Hashimoto, P. Liang. Robust Distortion-free Watermarks for Language Models. 2023. https://arxiv.org/abs/2307.15593
- X. Zhu, Y. Ye, T. Qiu, H. Zhu, S. Tan, A. Mannan et al. Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test. ICLR 2026. https://arxiv.org/abs/2506.06975
- W. Cai, T. Shi, X. Zhao, D. Song. Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs. 2025. https://arxiv.org/abs/2504.04715
- Y. Zhang, Y. Jiang, Z. Chen, M. Backes, X. Shen, Y. Zhang. Real Money, Fake Models: Deceptive Model Claims in Shadow APIs. CCS 2026. https://arxiv.org/abs/2603.01919
- T. Gloaguen, N. Jovanović, R. Staab, M. Vechev. Black-Box Detection of Language Model Watermarks. ICLR 2025. https://arxiv.org/abs/2405.20777
- Hugging Face Transformers, `WatermarkingConfig`. https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
