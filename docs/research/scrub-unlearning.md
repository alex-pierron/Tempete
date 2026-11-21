# SCRUB — Towards Unbounded Machine Unlearning

Kurmanji, P. Triantafillou, Hayes, E. Triantafillou (Warwick, Google DeepMind), NeurIPS 2023, arXiv 2302.09880
(v3, 30 oct. 2023). Notes rédigées à partir du PDF arXiv et du dépôt officiel (lus le 2026-11-18).
`§x` / `Tab. x` renvoient au papier (annexes incluses dans la v3).

## 1. Fiche
| Élément | Valeur | Source |
|---|---|---|
| Méthode | SCRUB : distillation teacher-student où l'étudiant **désobéit** au teacher sur le forget set et lui obéit sur le retain set | §3.1 |
| Variante | SCRUB+R : « rewinding » vers le checkpoint dont l'erreur forget est « juste assez haute » (défense MIA) | §3.2 |
| Scénarios | Removing Biases (RB), Resolving Confusion (RC), User Privacy (UP), chacun avec sa métrique d'oubli | §2, §5 |
| Données / modèles | CIFAR-10, Lacuna-10 (VGG-Faces) ; ResNet-18, All-CNN ; pré-entraînement CIFAR-100 / Lacuna-100 ; extension VGG16+BN (~138M) sur CIFAR-100 | §5.1, §13 |
| Code | `github.com/Meghdad92/SCRUB` (redirige vers `meghdadk/SCRUB`), notebooks Jupyter, base RepDistiller + SelectiveForgetting | §3.1 note 2, README |
| Licence | « Academic Software License », **pas d'usage commercial** (héritée de SelectiveForgetting, UCLA) ; GitHub la classe « Other » | `License.md` |

## 2. Définitions (§2)
- $D=\{(x_i,y_i)\}_{i=1}^N$, modèle original $f(\cdot;w^o)$ entraîné par ERM (cross-entropy $l$).
- **Forget set** $D_f\subset D$ ($N_f$ ex.), **retain set** $D_r=D\setminus D_f$ ($N_r$ ex.), **test set** $D_t$ (hors $D$).
  Objectif : produire $w^u$ qui a « oublié » $D_f$ sans dégrader l'utilité (performance sur $D_r$ et $D_t$), en étant
  nettement moins cher que le ré-entraînement.
- **Unlearning exact** : ré-entraîner sur $D_r$ (« Retrain », oracle de référence, coûteux). **Approché** : modifier
  $w^o$ pour s'approcher de l'exact, souvent avec garanties sous hypothèses (Golatkar et al., Fisher, NTK) (§1, §4).
- Définition formelle utilisée ensuite par la compétition NeurIPS 2023 (arXiv 2406.09073, Déf. 2.1), dite
  $(\varepsilon,\delta)$-unlearning : $\Pr[A(D\setminus S)\in R]\le e^\varepsilon\Pr[U(A(D),S,D)\in R]+\delta$ et réciproquement.
- Thèse du papier : **l'oubli dépend de l'application**. RB : erreur forget maximale. RC : supprimer la confusion
  induite par des labels faux. UP : erreur forget **aussi haute que Retrain, pas plus**, sinon les exemples oubliés
  deviennent identifiables par MIA (§2).

## 3. Formulation teacher-student et pertes exactes (§3.1)
- Teacher = $w^o$ gelé ; student $w^u$ initialisé à $w^o$. Distance : $d(x;w^u)=D_{\mathrm{KL}}\big(p(f(x;w^o))\,\Vert\,p(f(x;w^u))\big)$
  (softmax du teacher contre softmax du student).
- Max seul (éq. 1) : $\min_{w^u}-\frac{1}{N_f}\sum_{x_f\in D_f}d(x_f;w^u)$ : oublie, mais dégrade aussi $D_r$.
- Contrastif (éq. 2) : $\min_{w^u}\frac{1}{N_r}\sum_{x_r}d(x_r;w^u)-\frac{1}{N_f}\sum_{x_f}d(x_f;w^u)$.
- **Objectif final (éq. 3)** :
  $\min_{w^u}\ \frac{\alpha}{N_r}\sum_{x_r\in D_r}d(x_r;w^u)+\frac{\gamma}{N_r}\sum_{(x_r,y_r)\in D_r}l(f(x_r;w^u),y_r)-\frac{1}{N_f}\sum_{x_f\in D_f}d(x_f;w^u)$.
- **Optimisation alternée** (le joint oscille, comme un GAN) :
  - *max-step* = une epoch sur $D_f$ : $w^u\leftarrow w^u+\epsilon\nabla\frac{1}{|b|}\sum_{x_f\in b}d(x_f;w^u)$ (Alg. 2) ;
  - *min-step* = une epoch sur $D_r$ : $w^u\leftarrow w^u-\epsilon\nabla\frac{1}{|b|}\sum_{b}\big[\alpha d(x_r;w^u)+\gamma l(f(x_r;w^u),y_r)\big]$ (Alg. 3) ;
  - on alterne max puis min pendant MAX-STEPS epochs, puis on enchaîne des min-steps seuls pour restaurer le retain.
    Arrêt quand l'erreur forget a monté sans abîmer le retain (Alg. 1, §3.1). Le pseudo-code de l'Alg. 1 publié
    n'incrémente jamais `i` : se fier au notebook officiel (ci-dessous).
- Implémentation officielle : KL tempérée `DistillKL(T)` = $T^2\cdot\mathrm{KL}(\mathrm{softmax}(z_t/T)\Vert\mathrm{softmax}(z_s/T))$ moyennée sur
  le batch ; `minimize` : `gamma*CE + alpha*KL (+ beta*0)` ; `maximize` : `-KL` (`thirdparty/repdistiller/helper/loops.py`).
- Ablations (§12, Fig. 5-10) : max seul détruit le retain ; trop peu de max-steps fait redescendre l'erreur forget ;
  sans la phase finale de min-steps, pas d'équilibre ; le terme CE apporte une protection « petite mais constante »
  de l'utilité ; peu sensible à $\alpha,\gamma\in\{0.1,0.5,1,2,3\}$ ; décroissance du LR « cruciale » contre les oscillations (§9).

## 4. Hyperparamètres
Notebooks officiels (`small_scale_unlearning.ipynb`, `large_scale_unlearning.ipynb`, cellule `args`) :

| Paramètre | Small-scale (CIFAR-5 / Lacuna-5) | Large-scale (CIFAR-10 / Lacuna-10) |
|---|---|---|
| optimiseur | Adam | SGD (momentum 0,9) |
| `sgda_learning_rate` | 5e-4 | 5e-4 |
| `lr_decay_epochs`, `lr_decay_rate` | [5, 8, 9], 0,1 | [3, 5, 9], 0,1 |
| `sgda_epochs` (total, min-steps) | 10 | 3 |
| `msteps` (epochs avec max-step) | 3 | 2 |
| `gamma` (poids CE) / `alpha` (poids KL retain) | 1 / 0,5 | 0,99 / 0,001 |
| `kd_T` (température KL) | 2 | 4 |
| `sgda_weight_decay` | 0,1 | 5e-4 |
| `beta` (KD auxiliaire), `smoothing` | 0, 0,5 | 0, 0,0 |

Papier (§9, Tab. 3) : LR initial 0,0005 décru par 0,1 ; Adam meilleur en small-scale, Adam ou SGD en large-scale ;
batch sizes distincts forget/retain pour régler le nombre d'itérations de chaque côté. Exemples (forget bs, retain bs,
max, min) : ResNet CIFAR-10 class (512, 128, 2, 3) ; ResNet CIFAR-10 selective (16, 64, 5, 5) ;
ResNet CIFAR-5 selective (32, 32, 10, 10) ; All-CNN Lacuna-10 selective (8, 32, 2, 4).
Baselines (§9) : Finetune 10 epochs, LR 0,01 ; NegGrad+ $L=\beta\,\overline{l}_{D_r}-(1-\beta)\,\overline{l}_{D_f}$ avec
$\beta=0{,}95$ (small, 10 epochs) ou $0{,}9999$ (large, 5 epochs), « petit β = explosion ».

## 5. SCRUB+R (rewinding, §3.2)
1. Construire un **validation set de même distribution que le forget set** (ex. forget = classe 0, alors validation = classe 0 seulement).
2. Lancer SCRUB en sauvegardant un checkpoint par epoch.
3. À la fin (erreur forget maximale), mesurer l'erreur sur ce validation set : c'est la **référence** (proxy de
   l'erreur qu'aurait un modèle n'ayant jamais vu ces exemples).
4. Revenir au checkpoint dont l'erreur forget est la plus proche de cette référence.
- Utile surtout en *selective* unlearning (une partie seulement de la classe est oubliée) ; déclenché une seule fois
  dans le Tab. 2 ; plus souvent pour de petits forget sets (Fig. 3, 100 à 200 exemples).
- Limite symétrique (§8) : si l'erreur forget finale est **trop basse** (arrêt prématuré), le rewinding n'aide pas.

## 6. Les trois scénarios et les métriques
- Erreurs (§10, éq. 5) : $\mathrm{Err}(D)=1-\frac{1}{|D|}\sum_{(x,y)\in D}\mathbb 1[\arg\max f(x;w)=y]$ ; **forget error** $=\mathrm{Err}(D_f)$
  (haut = mieux en RB), **retain error** $=\mathrm{Err}(D_r)$, **test error** $=\mathrm{Err}(D_t)$ (bas = mieux).
- **RB** (§5.2) : forget error maximale à utilité constante. Small-scale : 5 classes, 100 train / 25 val / 100 test
  par classe, forget = 25 ex. de la classe 0 (5 %). Large-scale : *class* (toute la classe 5, 10 %) et *selective*
  (100 ex. de la classe 5 : 0,25 % de CIFAR, 2 % de Lacuna) (§5.1).
- **RC** (§5.3, §10) : 50 ex. de la classe 0 étiquetés 1 et 50 de la classe 1 étiquetés 0 ; forget set = exactement
  les ex. mal étiquetés. Métriques sur la matrice de confusion $C$ :
  $\mathrm{IC\text{-}ERR}(D,A,B)=\frac{\sum_{k\neq A}C_{A,k}+\sum_{k'\neq B}C_{B,k'}}{|D_A|+|D_B|}$ (éq. 6) et
  $\mathrm{FGT\text{-}ERR}(D,A,B)=C_{A,B}+C_{B,A}$ (éq. 7, un **compte**, pas un taux).
- **UP / MIA basique** (§11.1) : attaquant = `LogisticRegression()` scikit-learn par défaut, entrée = perte CE du
  modèle désappris, clippée à [-400, 400] ; classes « in » = forget, « out » = test **de même classe** ;
  équilibré, validation croisée 5 splits ; score = accuracy moyenne ; idéal = **50 %**.
- **UP / LiRA-for-unlearning** (§11.2) : 256 shadow models sur des moitiés aléatoires de CIFAR-10, puis pour chacun
  des runs d'unlearning sur 10K forget sets (200 ex. de la classe 5) ; pour chaque exemple cible, deux gaussiennes
  sur la confiance logit-scalée $\phi(\mathrm{softmax}(f(x))_y)$ : « oublié » contre « jamais vu » ; test de rapport
  de vraisemblance. SCRUB+R y est implémenté par filtrage (|erreur forget - erreur validation| ≤ 0,1).

## 7. Baselines et résultats chiffrés
Baselines (§5.1) : Retrain, Original, Finetune, NegGrad+ (version renforcée de NegGrad proposée par les auteurs),
Fisher Forgetting, NTK Forgetting (small-scale seulement), CF-k et EU-k (Goel et al. : geler les k premières couches,
fine-tuner ou ré-entraîner le reste), Bad-T (Chundawat et al., ré-implémenté).

| Mesure (moyenne sur 3 seeds) | Retrain | Original | NegGrad+ | EU-k | Bad-T | SCRUB | SCRUB+R | Source |
|---|---|---|---|---|---|---|---|---|
| RB, CIFAR-10 class, ResNet : forget err. / test err. | 100 / 14,72 | 0 / 16,56 | 91,26 / 17,84 | 98,79 / 18,73 | 94,67 / 19,56 | **100 / 15,73** | – | Tab. 9 |
| RB, CIFAR-10 selective, ResNet : forget / test | 29,67 / 17,4 | 0 / 17,36 | 53,7 / 22,7 | 23,7 / 21,8 | 34,67 / 23,47 | **70,33 / 18,04** | – | Tab. 11 |
| RC, CIFAR-5, ResNet : IC test err. / Fgt test | 24,0 / 18,3 | 56,0 / 92,0 | 47,5 / 69,0 | 47,0 / 63,3 | 47,3 / 27,0 | **19,0 / 19,7** | – | Tab. 1 |
| UP, MIA basique CIFAR-10 ResNet class / selective (%) | 49,33 / 54,00 | 71,10 / 65,33 | 69,57 / 66,67 | 54,20 / 53,00 | 54,00 / 59,67 | 52,20 / 78,00 | **52,20 / 58,67** | Tab. 2 |
| UP, LiRA-unlearning, AUC (5 cibles) | – | 0,92 | 0,79-0,80 | 0,81-0,84 | – | 0,72-0,74 | 0,72-0,74 | Fig. 4 |
| Vitesse vs Retrain, CIFAR-10 ResNet class | 1× | – | 3,4× | 1,41× | 19,07× | 7,84× | – | Tab. 6 |

Lecture : SCRUB est le seul « top performer » constant en oubli (Fig. 2) ; en *selective* son erreur forget (70 %)
dépasse largement celle de Retrain (≈ 30 %), d'où la MIA à 78 % sans rewinding et 58,67 % avec. Fisher est plus lent
que Retrain (0,08×, Tab. 6). Large-scale VGG16 / CIFAR-100, classe 0 (Tab. 4) : SCRUB+R test 37,60 / forget 100 / MIA 57,33
contre Retrain 36,92 / 100 / 55,67.

Compétition NeurIPS 2023 sur l'unlearning (Triantafillou, Kurmanji et al., arXiv 2406.09073) : 1 121 équipes,
1 923 soumissions, Kaggle, prédicteur d'âge sur visages CASIA-SURF ; score
$F\times\frac{\mathrm{Acc}(D\setminus S,\theta^u)}{\mathrm{Acc}(D\setminus S,\theta^r)}\times\frac{\mathrm{Acc}(D_{test},\theta^u)}{\mathrm{Acc}(D_{test},\theta^r)}$ avec
$F=\frac{1}{|S|}\sum_s 2/2^{n(\varepsilon_s)}$, $n(\varepsilon)=\lfloor\varepsilon/0{,}5\rfloor$,
$\hat\varepsilon=\max\{\log\frac{1-\delta-\widehat{FPR}}{\widehat{FNR}},\log\frac{1-\delta-\widehat{FNR}}{\widehat{FPR}}\}$
estimé par exemple sur 512 checkpoints (8 h max) ; dans leur analyse, SCRUB garde une très bonne utilité mais un
F-score faible face aux meilleures soumissions (§A.8). Gabarit probable si l'épreuve porte sur l'unlearning.

## 8. Limites (§6, §8, §11)
- Aucune garantie théorique ; tuning min-max instable (oscillations) ; rewinding inutile si l'erreur forget est trop basse.
- Petite échelle (CIFAR 32×32, sans data augmentation, donc modèle original sur-appris et MIA facile) ; pas de
  transformers ni d'auto-supervisé ; classification seulement.
- Évaluation par proximité à Retrain contestée (Thudi et al. 2022, Goel et al. 2022, cités §4) ; la vérification
  d'unlearning elle-même est contournable (Zhang et al., ICML 2024, arXiv 2408.00929).

## 9. Implémentation en 60 lignes (PyTorch, fidèle au notebook officiel)
```python
import copy, numpy as np, torch, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def kl_teacher_student(z_s, z_t, T: float) -> torch.Tensor:
    """d(x; w_u) = KL(p_teacher || p_student), tempered, T^2-scaled, batch mean (= DistillKL)."""
    return F.kl_div(F.log_softmax(z_s / T, 1), F.softmax(z_t / T, 1), reduction="batchmean") * T * T

def epoch(student, teacher, loader, opt, mode: str, alpha: float, gamma: float, T: float, dev: str) -> None:
    student.train(); teacher.eval()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        z_s = student(x)
        with torch.no_grad():
            z_t = teacher(x)
        kl = kl_teacher_student(z_s, z_t, T)
        loss = -kl if mode == "max" else alpha * kl + gamma * F.cross_entropy(z_s, y)
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def error(model, loader, dev: str) -> float:
    model.eval(); wrong = n = 0
    for x, y in loader:
        wrong += (model(x.to(dev)).argmax(1).cpu() != y).sum().item(); n += len(y)
    return wrong / n

def scrub(original, forget_dl, retain_dl, val_same_dist_dl=None, epochs=10, msteps=3, lr=5e-4,
          decay_at=(5, 8, 9), alpha=0.5, gamma=1.0, T=2.0, wd=0.1, dev="cuda"):
    teacher = copy.deepcopy(original).to(dev).eval()
    student = copy.deepcopy(original).to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)  # SGD(momentum=0.9) at large scale
    ckpts = []
    for ep in range(1, epochs + 1):
        for g in opt.param_groups:                       # step decay x0.1 after each milestone
            g["lr"] = lr * 0.1 ** sum(ep > m for m in decay_at)
        if ep <= msteps:
            epoch(student, teacher, forget_dl, opt, "max", alpha, gamma, T, dev)
        epoch(student, teacher, retain_dl, opt, "min", alpha, gamma, T, dev)
        ckpts.append((error(student, forget_dl, dev), copy.deepcopy(student.state_dict())))
    if val_same_dist_dl is not None:                     # SCRUB+R: rewind
        target = error(student, val_same_dist_dl, dev)   # reference forget error
        student.load_state_dict(min(ckpts, key=lambda c: abs(c[0] - target))[1])
    return student

@torch.no_grad()
def basic_mia(model, forget_dl, test_same_class_dl, dev="cuda") -> float:
    """Accuracy of a logistic regression on per-example CE losses (forget=1, test=0). Ideal: 0.5."""
    def losses(dl):
        model.eval()
        return torch.cat([F.cross_entropy(model(x.to(dev)), y.to(dev), reduction="none").cpu() for x, y in dl])
    lf, lt = losses(forget_dl), losses(test_same_class_dl)
    k = min(len(lf), len(lt))                            # class-balanced
    X = torch.cat([lf[:k], lt[:k]]).clamp(-400, 400).numpy()[:, None]
    y = np.r_[np.ones(k), np.zeros(k)]
    return cross_val_score(LogisticRegression(), X, y, cv=5, scoring="accuracy").mean()
```

## 10. Liens avec le watermarking
- **Vérifier un unlearning par backdoor ou watermark** : Sommer et al. (arXiv 2003.04247, publié sous le titre « Athena » à PoPETs 2022) :
  les utilisateurs insèrent un backdoor dans leurs données avant de les confier ; après la demande de suppression, le
  trigger ne doit plus marcher. Le cadre est un test d'hypothèse. Avec 5 % d'utilisateurs participants, la moitié de leurs
  données marquées et 30 requêtes, FPR et FNR < $10^{-3}$ (abstract). Limite : un fournisseur malhonnête peut tromper
  ces vérifications (Zhang et al., ICML 2024).
- **Radioactivité = test d'appartenance à grande échelle** : Sablayrolles et al. (Meta, ICML 2020, arXiv 2002.00937)
  détectent l'usage d'images marquées à l'entraînement avec $p<10^{-4}$ quand seulement 1 % des données sont
  « radioactives ». Sander, Fernandez et al. (arXiv 2402.14904) font la même chose pour les LLM : un fine-tuning sur
  des instructions tatouées est détecté avec $p<10^{-5}$ même quand seulement 5 % du texte est tatoué (modèle ouvert).
  Pour l'unlearning, la conséquence est directe : un SCRUB réussi sur des données tatouées devrait ramener le test de
  radioactivité au niveau de $H_0$ (p-value uniforme). C'est une métrique d'oubli plus forte qu'une MIA basique, et
  sa p-value est calibrée.
- **Désapprendre un watermark** (attaque) : (i) un watermark appris par distillation se perd quand on fine-tune
  sur du texte normal (Gu et al., ICLR 2024, arXiv 2312.04469) ; (ii) la radioactivité LLM s'efface par paraphrase ciblée
  des données ou par « watermark neutralization » à l'inférence (Pan et al., ACL 2025, arXiv 2502.11598) ;
  (iii) les watermarks de modèle par backdoor (Adi et al., arXiv 1802.04633) tombent par fine-tuning (REFIT,
  arXiv 1911.07205) ; (iv) Stable Signature (watermark dans les poids du décodeur VAE) résiste à la « model purification »
  par fine-tuning : la bit accuracy ne baisse pas nettement sans artefacts visibles (arXiv 2303.15435, §7.2, Fig. 8). En
  revanche, moyenner deux modèles rend aléatoires les bits où ils diffèrent (« model collusion », même section).
- **SCRUB comme outil d'attaque** : pour un générateur tatoué dans ses poids, on prend comme forget set des sorties
  tatouées et comme retain set des sorties propres ou une cible perceptuelle. Le max-step KL pousse à s'écarter du
  teacher sur les sorties tatouées, le min-step préserve la qualité. Proposition non testée dans la littérature lue :
  à évaluer contre la bit accuracy et la qualité (PSNR, LPIPS).
- **Lien MET** : MET montre qu'un fine-tuning, même hors distribution, déplace la distribution de sortie de façon
  détectable (arXiv 2410.20247, §4.2). Un modèle « désappris » reste donc probablement distinguable de l'original par un
  test à deux échantillons. La cible pertinente est l'indistinguabilité avec Retrain, pas avec l'original.

## Références
- M. Kurmanji, P. Triantafillou, J. Hayes, E. Triantafillou. Towards Unbounded Machine Unlearning. NeurIPS 2023. https://arxiv.org/abs/2302.09880 ; code https://github.com/meghdadk/SCRUB
- E. Triantafillou, P. Kairouz, F. Pedregosa, J. Hayes, M. Kurmanji, K. Zhao et al. Are we making progress in unlearning? Findings from the first NeurIPS unlearning competition. 2024. https://arxiv.org/abs/2406.09073
- D. M. Sommer, L. Song, S. Wagh, P. Mittal. Athena: Probabilistic Verification of Machine Unlearning. PoPETs 2022 (preprint 2020 « Towards Probabilistic Verification of Machine Unlearning »). https://petsymposium.org/popets/2022/popets-2022-0072.php ; https://arxiv.org/abs/2003.04247
- B. Zhang, Z. Chen, C. Shen, J. Li. Verification of Machine Unlearning is Fragile. ICML 2024. https://arxiv.org/abs/2408.00929
- A. Sablayrolles, M. Douze, C. Schmid, H. Jégou. Radioactive data: tracing through training. ICML 2020. https://arxiv.org/abs/2002.00937
- T. Sander, P. Fernandez, A. Durmus, M. Douze, T. Furon. Watermarking Makes Language Models Radioactive. NeurIPS 2024. https://arxiv.org/abs/2402.14904
- C. Gu, X. L. Li, P. Liang, T. Hashimoto. On the Learnability of Watermarks for Language Models. ICLR 2024. https://arxiv.org/abs/2312.04469
- L. Pan, A. Liu, S. Huang, Y. Lu, X. Hu, L. Wen, I. King, P. S. Yu. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation? ACL 2025. https://arxiv.org/abs/2502.11598
- Y. Adi, C. Baum, M. Cisse, B. Pinkas, J. Keshet. Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring. USENIX Security 2018 [A VERIFIER : venue]. https://arxiv.org/abs/1802.04633
- X. Chen, W. Wang, C. Bender, Y. Ding, R. Jia, B. Li et al. REFIT: A Unified Watermark Removal Framework for Deep Learning Systems with Limited Data. AsiaCCS 2021. https://arxiv.org/abs/1911.07205
- P. Fernandez, G. Couairon, H. Jégou, M. Douze, T. Furon. The Stable Signature: Rooting Watermarks in Latent Diffusion Models. ICCV 2023. https://arxiv.org/abs/2303.15435
- N. Carlini et al. Membership Inference Attacks From First Principles (LiRA). IEEE S&P 2022 (cité ainsi par arXiv 2302.09880). https://arxiv.org/abs/2112.03570
