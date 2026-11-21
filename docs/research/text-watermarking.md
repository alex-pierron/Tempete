# Watermarking de texte LLM — notes de recherche

Etat au 2026-11-18. Sources : PDF arXiv lus en texte integral (2301.10226, 2308.00113, 2307.15593,
2402.14904, 2402.19361), article Nature de SynthID-Text (HTML), README des depots. Notations :
vocabulaire $V$, texte $x^{(1..T)}$, cle secrete $k$, fenetre de contexte $h$.

## 0. Cadre commun

- **Graine par contexte.** $k^{(t)}=H(x^{(t-1)},\dots,x^{(t-h)},k)$ avec $H$ un hash cryptographique ; elle initialise
  le RNG qui produit la liste verte ou le vecteur aleatoire du pas $t$ [2308.00113, sec. II-B]. $h$ petit = robuste
  aux editions mais graines repetees ; $h=0$ = cle fixe, la plus robuste [ibid.].
- **Test.** $H_0$ : « texte naturel / non tatoue » contre $H_1$ : « genere avec la cle ». On fixe un FPR et on
  signale si p-value < FPR [2308.00113, sec. II-E].
- **Regle de dedoublonnage (indispensable).** Ne scorer un token que si le $(h+1)$-uplet {contexte + token}
  n'a pas deja ete score (« scoring v2 » de `three_bricks`) ; sans cela, les FPR empiriques depassent largement
  les FPR theoriques (texte formate, listes a puces) [2308.00113, sec. III-C, Fig. 2c].

## 1. KGW / liste verte (Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein — ICML 2023)

- **Generation (soft).** A chaque pas, la graine (hash du token precedent dans l'article original) partitionne
  $V$ en liste verte $G$ de taille $\gamma|V|$ et liste rouge ; on ajoute $\delta$ aux logits verts puis softmax :
  $\hat p_k=\frac{e^{l_k+\delta}}{\sum_{i\in R}e^{l_i}+\sum_{i\in G}e^{l_i+\delta}}$ pour $k\in G$ [2301.10226, Alg. 2].
- **Detection (z-score).** $|s|_G$ = nombre de tokens verts sur $T$ tokens scores :
  $z=\dfrac{|s|_G-\gamma T}{\sqrt{T\gamma(1-\gamma)}}$ [2301.10226, eq. 3]. Seuil $z>4$ : FPR unilateral
  $3\times10^{-5}$ ; la liste rouge dure est detectable des 16 tokens [sec. 2].
- **p-value exacte (a preferer).** $S_T\sim\mathrm{Bin}(T,\gamma)$ sous $H_0$ :
  $p=P(S_T\ge s\mid H_0)=I_\gamma(s,T-s+1)$ [2308.00113, eq. 6] = `scipy.stats.binom.sf(s-1, T, gamma)`.
  Le z-test (approximation normale) n'est pas fiable aux FPR bas [2308.00113, sec. III-A].
- **Chiffres.** $\gamma=0{,}5$, $\delta=2$, $T=200$ : 98,4 % des generations detectees a $z=4$ en echantillonnage
  multinomial, 99,6 % avec beam search a 4 faisceaux [2301.10226, sec. 4.1]. AUC 0,998 a $\delta=2$, 0,985-0,989 a
  $\delta=1$ [Fig. 4]. Variante « KGW2-SelfHash » ($h=3$) : introduite dans 2306.04634, cible de 2402.19361 (sec. 5).

## 2. Aaronson / Gumbel-max (Aaronson & Kirchner 2023, formalise dans Three Bricks)

- **Generation.** Vecteur secret $r\in[0,1]^{|V|}$ tire depuis la graine ; token choisi
  $x^{(t)}=\arg\max_v r_v^{1/p_v}$ ; non biaise en esperance sur $r$ ($P(X=v)=p_v$) [2308.00113, sec. II-B2, App. A].
  Equivalent au Gumbel-max trick : $\arg\max_v(\log p_v+G_v)$ avec $G_v=-\log(-\log r_v)$.
- **Score.** $S_T=-\sum_{t=1}^{T}\ln\!\big(1-r^{(t)}_{x^{(t)}}\big)$ [eq. 3]. Sous $H_0$, $r\sim\mathcal{U}(0,1)$ donc
  $S_T\sim\Gamma(T,1)$ et $p=\Gamma(T,s)/\Gamma(T)$ (gamma incomplete superieure regularisee)
  [2308.00113, eq. 7, App. A] = `scipy.special.gammaincc(T, s)`.
- **Score Neyman-Pearson** (acces au LLM) : $S_T=\sum_t\big(\tfrac{1}{p_{x^{(t)}}}-1\big)\ln r_{x^{(t)}}$ ; pas de p-value
  fermee (borne de Chernoff seulement), fragile aux attaques [sec. V-A].

## 3. Three Bricks (Fernandez, Chaffin, Tit, Chappelier, Furon — WIFS 2023)

- **Constat.** 100 k textes Wikipedia x 10 cles = 1 M tests sous $H_0$ ($T=256$, $\gamma=0{,}25$) : les FPR empiriques
  des z-tests sont tres au-dessus des FPR theoriques ; il faudrait $h\gg8$ [2308.00113, sec. III-A].
- **Correctifs.** Tests exacts (sections 1 et 2) + dedoublonnage v2 : FPR empiriques = theoriques, sauf
  $h=0$ legerement sous-estime [sec. III-C].
- **Multi-bit.** $M$ messages via decalages cycliques $r^{(m)}=\mathrm{CyclicShift}(r,m)$, $d\ge|V|$ ;
  $\hat m=\arg\min_m p_m$ ; p-value globale $1-(1-p_{\min})^M$ [sec. V-B, Alg. 1].
- **Chiffres** [Tab. II, TPR a FPR $10^{-5}$, Guanaco-7b, 1 k prompts Alpaca] : KGW $\delta=3$, $h=4$ : TPR 0,76,
  mais 0,03 apres remplacement aleatoire de 30 % des tokens ; a $h=0$ : 0,58 -> 0,20. Aaronson $\theta=1{,}0$, $h=4$ :
  0,71. Identification [Tab. III, FPR $10^{-6}$] : KGW 0,69 / 0,64 / 0,59 / 0,55 pour $M=10/10^2/10^3/10^4$
  utilisateurs ; zero fausse accusation.
- **Code** `three_bricks` (CC-BY-NC) : `--method openai|maryland`, `--method_detect same|openaiz|marylandz|openainp`,
  `--scoring_method none|v1|v2`, `--ngram 4` par defaut [README].

## 4. Distortion-free robuste (Kuditipudi, Thickstun, Hashimoto, Liang — 2023)

- **Principe.** Suite de cles $\xi=(\xi_1..\xi_n)$ (pas de hash du contexte) ; token $i$ genere par
  $y_i=\Gamma(\xi_{i\bmod n},p(\cdot\mid y_{<i}))$ [2307.15593, Alg. 1]. EXP : $\Gamma(\xi,\mu)=\arg\min_i -\log(\xi_i)/\mu(i)$
  (Gumbel, eq. 4) ; ITS : inverse transform sampling avec permutation $\pi$ [eq. 1].
- **Statistique.** Cout d'alignement EXP $d(y,\xi)=\sum_{i=1}^{k}\log(1-\xi_{i,y_i})$ [eq. 6] ; ITS
  $d=\sum_i|u_i-\eta(\pi_i(y_i))|$ [eq. 3]. $\varphi(y,\xi)=\min_{i,j}d(y_{i:i+k},\xi_{j:j+k\,(\bmod n)})$ sur blocs de
  longueur $k$ [Alg. 3]. Version « -edit » : Levenshtein souple $d_\gamma$ (insertion/suppression a cout $\gamma$)
  [Def. 5], cout $O(mnk^2)$.
- **p-value par permutation.** Tirer $T$ cles $\xi^{(t)}\sim\nu$ :
  $\hat p=\frac{1}{T+1}\big(1+\sum_{t=1}^{T}\mathbb{1}\{\varphi(y,\xi^{(t)})\le\varphi(y,\xi)\}\big)$, valide sans hypothese
  [Alg. 2]. Reglages : $T=5000$, $n=256$, $\gamma=0{,}4$ (ITS-edit), $\gamma=0{,}0$ (EXP-edit) [sec. 3].
- **Chiffres.** OPT-1.3B et LLaMA-7B : $p\le0{,}01$ des 35 tokens meme apres 40-50 % d'editions aleatoires ;
  Alpaca-7B : environ 25 % des reponses (mediane environ 100 tokens) detectables a $p\le0{,}01$ [abstract].
- **Limite.** La plus petite p-value atteignable est $1/(T+1)$ : inadapte a un FPR $10^{-6}$ sans $T$ enorme.

## 5. SynthID-Text (Dathathri, See, Ghaisas, Huang, McAdam et al., Google DeepMind — Nature 634, 2024)

- **Tournament sampling.** Graine = hash des $H=4$ derniers tokens + cle. $m$ fonctions $g_\ell(x,r)$ donnent des
  g-values pseudo-aleatoires ($\mathrm{Bernoulli}(0{,}5)$ par defaut). On tire $M=2^m$ candidats de $p_{LM}$, on
  les apparie, le meilleur selon $g_1$ passe, puis $g_2$..., le vainqueur de la couche $m$ est emis ; $m=30$ dans
  les experiences [Nature, sec. Tournament sampling, Methods].
- **Score.** $\mathrm{Score}(x)=\frac{1}{mT}\sum_{t=1}^{T}\sum_{\ell=1}^{m}g_\ell(x_t,r_t)$ [eq. 1] ; variantes ponderees et
  detecteur bayesien appris [Suppl. A, I.5.2]. **p-value (derivation propre, non publiee)** : avec g-values
  Bernoulli(0,5) independantes et contextes dedoublonnes, $mT\cdot\mathrm{Score}\sim\mathrm{Bin}(mT,1/2)$ sous $H_0$
  `[A VERIFIER : independance inter-couches dans l'implementation]`.
- **Faits.** Configurable non-distortionary (par sequence, via repeated context masking) ou distortionary ;
  evaluation live sur environ 20 millions de reponses Gemini ; deploye dans Gemini [Nature]. Code
  `google-deepmind/synthid-text` (Apache-2.0), detecteurs frequentiste et bayesien, mixin Transformers (README).
  Implementation sans tirer $2^{30}$ candidats (mise a jour analytique par couche) `[A VERIFIER dans le code]`.

## 6. Ajouts Meta 2025-2026 (voir `meta-seal-ecosystem.md`)

TextSeal (2605.12456) : Gumbel-max double cle + score pondere par l'entropie + localisation ; TextSeal2
(2610.40613) : cle virtuelle par copule, compatible speculative decoding, detection par test Gamma sur la cle
virtuelle avec dedoublonnage (README `textseal`) ; post-hoc par reecriture (2512.16904) ; detection publique a
cle scindee (2610.12106). Bibliotheque `pip install textseal` (Apache-2.0) : Green/Red list, Gumbel-max, DipMark,
SynthID, MorphMark, WaterMax [README textseal].

## 7. Radioactivite (Sander, Fernandez, Durmus, Douze, Furon — NeurIPS 2024)

- **Question.** Bob a-t-il fine-tune son modele $B$ sur des sorties du modele tatoue $A$ d'Alice ?
  $\rho=|D_A|/|D|$ (part tatouee des donnees de Bob), $d$ = degre de supervision (Alice connait-elle ces textes),
  acces open-model (logits) ou closed-model (API) [2402.14904, sec. 3].
- **Test.** $H_0$ : $S(X_N)\sim\mathrm{Bin}(N,\gamma)$, valide si (1) les tokens scores sont deux a deux distincts
  (dedoublonnage {fenetre + token}) et (2) aucun token n'apparait dans son propre contexte [Prop. 1] ;
  $p=I_\gamma(t+1,N-t)$. Amplification : « reading mode » (open-model, on fait passer les textes tatoues de $A$
  dans $B$ et on score ses predictions next-token), filtre $\phi$ sur les k-grams presents dans les sorties de $A$,
  millions de tokens scores (225 k open, 600 k closed) [sec. 4.2, 5].
- **Chiffres** (Llama-7B fine-tune). Non supervise, open-model : $p<10^{-5}$ avec $\rho=5\%$ ; supervise ($d=1$) :
  $p<10^{-30}$ (open) et $<10^{-10}$ (closed) des $\rho=1\%$ [sec. 5, Fig. 5]. Le MIA classique ne marche qu'en
  supervise open-model [Tab. 1]. Extensions : contamination de benchmark (2502.17259), distillation (WM-Basin).

## 8. Attaques

| Attaque | Mecanisme | Effet chiffre | Parade |
|---|---|---|---|
| Remplacement aleatoire | 30 % des tokens remplaces | KGW $h=4$ : TPR 0,76 -> 0,03 ; $h=0$ : 0,58 -> 0,20 [2308.00113, Tab. II] | $h$ petit, edit-distance |
| Editions aleatoires | substitutions/insertions/suppressions | EXP-edit tient a 40-50 % d'editions (35 tokens, $p\le0{,}01$) [2307.15593] | alignement Levenshtein |
| Paraphrase LLM | DIPPER 11 B (Krishna et al.) | evite watermark, GPTZero, DetectGPT (70,3 % -> 4,6 % a FPR 1 %) [2303.13408] | retrieval des generations |
| Paraphrase humaine forte | reecriture | KGW redetecte apres environ 800 tokens a FPR $10^{-5}$ [2306.04634] | detection sur fenetres |
| Paraphrase recursive | paraphraser plusieurs fois | baisse forte de detection, qualite peu degradee [2303.11156] | - |
| Emoji attack | faire inserer puis retirer des emojis | casse le contexte de hash [2301.10226, sec. 7] | $h$ petit, normalisation |
| Watermark stealing | 30 000 requetes API (< 50 USD) -> estimer $s^\star(T,\text{ctx})$ | spoofing et scrubbing > 80 % sur KGW2-SelfHash ; scrubbing DIPPER de ~0 % a > 80 %, baselines < 25 % [2402.19361] | $h$ grand, cles multiples |
| Spoofing | produire du texte detecte comme tatoue | demontre via stealing [2402.19361] et via detecteur public [2610.12106] | cle privee de verification |

## 9. Formules pretes a coder

```python
import math
from scipy.stats import binom
from scipy.special import gammaincc
def dedup(tokens, h):                      # garde (contexte, token) jamais vus
    seen, keep = set(), []
    for t in range(h, len(tokens)):
        key = tuple(tokens[t - h:t + 1])
        if key not in seen: seen.add(key); keep.append(t)
    return keep
def kgw_pvalue(n_green, T, gamma):  return binom.sf(n_green - 1, T, gamma)          # I_gamma(s, T-s+1)
def kgw_z(n_green, T, gamma):       return (n_green - gamma*T) / (T*gamma*(1-gamma))**0.5
def gumbel_pvalue(r_selected):                                                       # r du token choisi
    s = -sum(math.log(1 - r) for r in r_selected); return gammaincc(len(r_selected), s)
def multibit_global_pvalue(p_min, M): return 1 - (1 - p_min) ** M
def synthid_pvalue(g_sum, m, T):    return binom.sf(g_sum - 1, m*T, 0.5)            # [A VERIFIER]
def permutation_pvalue(phi_obs, phi_resampled):                                      # Kuditipudi Alg. 2
    return (1 + sum(p <= phi_obs for p in phi_resampled)) / (len(phi_resampled) + 1)
```
Pour des FPR tres bas, travailler en log : `binom.logsf` ; `gammaincc` sort un float64 qui passe a 0 sous
environ $10^{-308}$ : au-dela, utiliser `mpmath.gammainc(T, s, mpmath.inf, regularized=True)`.

## References

- Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein. *A Watermark for Large Language Models*. ICML 2023. https://arxiv.org/abs/2301.10226
- Kirchenbauer, Geiping, Wen et al. *On the Reliability of Watermarks for Large Language Models*. ICLR 2024. https://arxiv.org/abs/2306.04634
- Aaronson, Kirchner. *Watermarking GPT Outputs* (slides). 2023. https://www.scottaaronson.com/talks/watermark.ppt
- Fernandez, Chaffin, Tit, Chappelier, Furon. *Three Bricks to Consolidate Watermarks for Large Language Models*. IEEE WIFS 2023. https://arxiv.org/abs/2308.00113
- Kuditipudi, Thickstun, Hashimoto, Liang. *Robust Distortion-free Watermarks for Language Models*. TMLR 2024. https://arxiv.org/abs/2307.15593
- Dathathri, See, Ghaisas, Huang, McAdam et al. *Scalable watermarking for identifying large language model outputs*. Nature 634, 818-823, 2024. https://www.nature.com/articles/s41586-024-08025-4 ; code https://github.com/google-deepmind/synthid-text
- Sander, Fernandez, Durmus, Douze, Furon. *Watermarking Makes Language Models Radioactive*. NeurIPS 2024. https://arxiv.org/abs/2402.14904
- Jovanovic, Staab, Vechev. *Watermark Stealing in Large Language Models*. ICML 2024. https://arxiv.org/abs/2402.19361
- Krishna, Song, Karpinska, Wieting, Iyyer. *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*. NeurIPS 2023. https://arxiv.org/abs/2303.13408
- Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-Generated Text be Reliably Detected?* TMLR 2023-2024. https://arxiv.org/abs/2303.11156
- Sander et al. *TextSeal*. 2026. https://arxiv.org/abs/2605.12456 ; Sander, Fernandez et al. *TextSeal2*. 2026. https://arxiv.org/abs/2610.40613 ; code https://github.com/facebookresearch/textseal
- Milis, Sander, Soucek, Huang, Fernandez. *Could LLM Watermark Detection be Public?* 2026. https://arxiv.org/abs/2610.12106
