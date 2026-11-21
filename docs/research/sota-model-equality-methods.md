# Model equality côté utilisateur : les méthodes expliquées, et leurs angles morts

But : comprendre chaque méthode assez bien pour trouver un angle mort utile à notre tâche.
Écrit le 2026-11-21. Pour chaque méthode, on indique ce qui a été lu (texte complet / résumé seul / mémoire).
Les chiffres viennent des papiers tels que lus : les vérifier dans le papier avant de les citer.

## 0. Notre problème en un paragraphe

33 endpoints, sortie texte uniquement, 100 réponses par endpoint pour toute la compétition, 256 tokens max,
`temperature` et `seed` ignorés (chaque endpoint a ses propres réglages de génération), system messages refusés.
Pour chaque paire : même famille (même modèle de base) ? même modèle (même checkpoint + même quantification + même LoRA) ?
Watermark, system prompt, persona, stratégie de génération : **à ignorer**. Quantification (fp8) ou adaptateur LoRA :
**à détecter**. Score = 0,1 × MCC famille + 0,9 × MCC modèle, et le MCC modèle ne compte que les paires de la **vraie**
même famille. 9 références (`ref-01..09`, un seul modèle de base SmolLM, illimitées) montrent tous les déguisements :
plain, system prompt, persona, watermark, température 1,0, fp8, LoRA de force 0,3 / 0,7 / 1,5.

Conséquence du score : fusionner à tort deux familles coûte peu (seulement les 10 %) ; couper à tort une vraie famille
peut coûter cher (ses paires sortent des 90 % comme « différentes »). **En cas de doute sur les familles, fusionner.**

## 1. Les perturbations vues de la sortie

Comprendre *comment* chaque déguisement ou changement déplace les probabilités du prochain token, c'est la clé pour
les séparer.

| Changement | Effet sur les scores des tokens (logits) | Dépend de |
|---|---|---|
| Température T | Divise tous les scores par T : plus net (T < 1) ou plus plat (T > 1), **ordre des tokens inchangé** | Rien : pareil pour tout contexte |
| Watermark rouge-vert (type Kirchenbauer) | Ajoute +delta à un sous-ensemble « vert » du vocabulaire (fraction gamma). Liste verte tirée par un hash des h tokens précédents et d'une clé secrète | **Seulement les h tokens précédents et la clé** (h = 1 pour le « LeftHash » courant), pas le sens du prompt |
| System prompt / persona | Change le contexte lu par le modèle, donc tout : sujet, style, identité, parfois longueur | Toute la conversation |
| Quantification (fp8) | Petit bruit sur chaque score. Ne change le token gagnant que là où deux tokens sont presque à égalité | Tout le contexte, de façon pseudo-aléatoire |
| Adaptateur LoRA | Décalage appris : faible à force 0,3, fort à 1,5. Change style ou contenu dans une direction cohérente | Tout le contexte, avec du sens |

D'après le papier de détection de watermark (Gloaguen et al., Éq. 4) : le biais effectif d'un watermark rouge-vert est
+delta/T sur un token vert quand tous les autres candidats sont rouges, −delta/T sur un token rouge quand un candidat est
vert, 0 sinon. Le watermark n'agit donc **que là où le modèle hésite**, exactement comme fp8, mais avec une poussée plus
forte (delta typique = 2). C'est pour ça qu'un test naïf « les réponses diffèrent-elles ? » confond watermark et vrai
changement de modèle.

## 2. Méthodes pour les familles (étape 1, facile)

### 2.1 LLMmap (Pasquini, Kornaropoulos, Ateniese, USENIX Security 2025). Source : mémoire, pas relu
- **Idée.** Une huitaine de questions fixes choisies pour séparer les modèles : questions de bannière (« qui t'a créé ? »),
  questions d'alignement (requête limite), requêtes mal formées. Les paires (question, réponse) passent dans un réseau
  entraîné qui sort une étiquette (ensemble fermé) ou un vecteur (ensemble ouvert, comparé par distance).
- **Pourquoi ça colle aux familles.** Entraîné avec beaucoup de system prompts, de réglages de génération et de
  frameworks justement pour que l'empreinte y résiste.
- **Angles morts.** Entraîné sur sa propre liste de modèles : sur des modèles inconnus, on n'a que le vecteur.
  Sépare des versions de modèles, pas des variantes quantifiées ou LoRA d'un même checkpoint. Ses questions de bannière
  sont justement ce qu'un system prompt réécrit (section 5 : notre endpoint à system prompt ne dit jamais « SmolLM »).

### 2.2 AdaptPrint (arXiv 2608.22213, 2026). Source : extraits de recherche seulement
- **Idée.** Comme LLMmap, mais la question suivante dépend des réponses précédentes.
- **Rapporté.** Évalué avec 18 system prompts différents, à température 0,7 / top-p 0,95, contre MET, LLMmap et ZeroPrint.
  Méthode non lue en détail.
- **Angle mort.** Même granularité que LLMmap : identité du modèle, pas la quantification.

### 2.3 One Token Is Enough (Bruckner, arXiv 2607.10252, 2026). Source : résumé seul
- **Idée.** Questions triviales à un mot (« donne un nombre entre 1 et 100 », en quatre langues). Chaque réponse coûte
  un token. La distribution des réponses est très peu uniforme et propre à chaque modèle. On compare deux empreintes par
  la divergence de Jensen-Shannon.
- **Rapporté.** 165 modèles sur OpenRouter ; famille retrouvée dans 59,5 % des cas contre 18,4 % au hasard ;
  taux d'erreur de vérification 7,3 % avec toute la batterie, 10,6 % avec environ 120 requêtes d'un token.
- **Angles morts pour nous.** Les réponses sont très variées (haute entropie) : c'est exactement là que le watermark
  pousse le plus fort et que la température change le plus la distribution. Bon pour les familles, mauvais pour
  « même modèle ».

### 2.4 Idiosyncrasies in LLMs (Sun et al., arXiv 2502.12150, 2025). Source : mémoire
- **Idée.** Un classifieur de texte entraîné sur des sorties reconnaît le modèle source avec une bonne précision,
  surtout par le choix des mots et le formatage.
- **Angle mort.** Il faut des exemples étiquetés par modèle ; Cai et al. (section 3.8) montrent que ce type de
  classifieur est au hasard (environ 50 %) entre un modèle et sa version quantifiée.

### 2.5 En pratique pour les familles
Deux modèles de base différents diffèrent presque partout (connaissances, découpage en tokens, formatage). Utiliser ce
que les déguisements ne touchent pas : complétions quasi certaines, questions factuelles, formatage de listes ou de code.
Ne pas se fier à l'auto-présentation. Le seul cas dur : deux tailles d'une même lignée (deux SmolLM), qui sont deux
familles mais répondent pareil.

## 3. Méthodes pour les modèles (étape 2, difficile)

### 3.1 Model Equality Testing, MET (Gao, Liang, Guestrin, ICLR 2025). Source : texte complet, voir `docs/research/model-equality-testing.md`
- **Idée.** Test à deux échantillons. On collecte des réponses des deux endpoints sur les mêmes prompts et on calcule la
  MMD : similarité moyenne à l'intérieur de chaque endpoint moins similarité moyenne entre les deux. Similarité = noyau
  de Hamming (nombre de positions où les deux suites de tokens coïncident). La p-valeur vient de permutations des
  étiquettes d'endpoint.
- **Ce qu'il faut.** Seulement des échantillons des deux endpoints : pas de modèle local, pas de logits. Colle à notre cadre.
- **Rapporté.** Voir la note de l'équipe (section 5) : la puissance contre int8 / nf4 / watermark augmente avec le nombre
  d'échantillons ; le watermark est le plus dur à voir de leurs alternatives.
- **Angles morts.**
  1. Il traite le watermark comme une différence à détecter. Il nous faut l'inverse.
  2. Puissance faible contre fp8 avec notre budget.
  3. Tous les prompts pèsent pareil ; rien ne concentre l'effort sur les prompts informatifs.
  4. Le noyau de Hamming compare position par position : un persona qui ajoute une phrase d'ouverture décale toutes les
     positions et paraît très différent. Notre kit retire les phrases répétitives pour limiter ça.

### 3.2 Rank-based Uniformity Test, RUT (Zhu et al., ICLR 2026). Source : résumé et notes du projet
- **Idée.** Pour chaque prompt, une réponse de l'API ; on tire beaucoup de réponses d'une copie locale du modèle de
  référence ; on regarde où la réponse de l'API se classe (par log-probabilité) parmi elles. Même modèle : rangs
  uniformes ; on teste l'uniformité (Cramér-von Mises). Prompts naturels, donc le fournisseur ne repère pas l'audit.
- **Pourquoi pas utilisable tel quel.** Il faut un modèle de référence local qui donne des log-probabilités. On n'en a pas.
- **À garder.** Le schéma asymétrique (un côté pas cher, un côté riche). Nos références sont illimitées mais servent un
  autre petit modèle : elles ne peuvent pas être le côté riche pour les `ep-*`.
- **Limite annoncée par eux.** Pas testé contre un adversaire qui connaît le test.

### 3.3 B3IT : Token-Efficient Change Detection in LLM APIs (Chauvin, Lalanne, Le Merrer, Loubes, Taïani, Tredan, ICML 2026). Source : résumé et extraits (texte complet non lu)
- **Idée.** « Entrées frontières » : prompts où les deux premiers tokens les plus probables sont presque à égalité.
  À basse température, la réponse devient alors extrêmement sensible à de petits changements de poids. On les trouve en
  essayant des prompts candidats et en gardant ceux où l'endpoint hésite entre deux réponses, puis on surveille la
  distribution des réponses sur eux.
- **Ce qu'il faut.** Le texte seul. Coût rapporté environ 30 fois plus bas que les autres méthodes boîte noire,
  détection au niveau des méthodes qui utilisent les log-probabilités. Validé sur 131 endpoints de 34 fournisseurs.
  Entrées frontières faciles à trouver pour les modèles qui ne raisonnent pas.
  Code : `github.com/timothee-chauvin/token-efficient-change-detection-llm-apis`.
- **Pourquoi c'est important pour nous.** C'est la seule méthode qui **cherche des entrées discriminantes** : la plus
  proche d'une approche adversariale, et le signal le plus sensible à fp8.
- **Angles morts pour nous.**
  1. Fait pour détecter qu'un même endpoint change dans le temps, pas pour comparer deux endpoints. Une entrée frontière
     du modèle A n'en est pas forcément une du modèle B.
  2. Deux endpoints à températures différentes se répartissent différemment sur une même entrée frontière, même avec le
     même modèle. Mais l'ordre ne change pas avec la température : le token *le plus probable* des deux reste le même.
  3. Un watermark ajoute +delta à l'un des deux tokens à égalité une fois sur deux environ : le jumeau watermarké paraît
     changé sur beaucoup d'entrées frontières. Il faut une variante qui tient compte du watermark (section 6).

### 3.4 KBF : Knowledge Boundary as Fingerprint (Fang, Feng, Li, Zhou, arXiv 2605.29524, 2026). Source : résumé et extraits
- **Idée.** Questions factuelles numériques au bord de ce que le modèle sait. En deçà il a raison, au-delà il a tort, et
  près du bord il donne des **mauvaises réponses constantes** propres au checkpoint. Ces sondes sont choisies pour
  résister à la configuration de déploiement.
- **Rapporté.** Contre une API officielle de référence : signale les 155 substitutions économiquement intéressantes sur
  16 endpoints réels sans rejeter les contrôles ; 95 % de détection quand 15 % du trafic est détourné ; un audit réel
  signale 7 endpoints sur 28. Comparé à MET, LLMmap, ZeroPrint.
- **Pourquoi c'est important pour nous.** Une réponse factuelle courte a peu d'entropie : watermark et température la
  bougent peu, et un system prompt change rarement un nombre. C'est le bon type d'invariance.
- **Angles morts.** Les substitutions testées sont surtout d'autres modèles ou tailles. On ne sait pas si fp8 déplace
  assez le bord des connaissances. Un LoRA entraîné sur d'autres données le déplace probablement : ça aiderait plus contre
  LoRA que contre fp8.

### 3.5 IRIS (arXiv 2607.20860, juillet 2026). Source : sections 1-5 lues en entier, le reste via un résumé
- **Idée.** Sondes de génération sans bonne réponse : pile ou face, un chiffre, un entier dans un intervalle, une courte
  suite de chiffres ou de bits. Chaque réponse devient un vecteur de 179 nombres décrivant la chaîne visible (respect du
  format, fréquences des symboles, transitions, séries, équilibre des positions). Un classifieur par sonde donne un
  score ; la moyenne de m scores donne un test dont l'erreur décroît exponentiellement avec m
  (`log(1 − AUROC(m)) ≈ a − I·m`), ce qui fixe le budget de requêtes. Estime aussi la part de trafic détournée.
- **Rapporté (résumé).** AUROC 0,99 sur une échelle de 6 tailles de Qwen3 ; signale 14 paires sur 15 de même modèle
  chez différents fournisseurs, à cause de vraies différences de quantification et de noyaux de calcul ; détecte 30 % de
  détournement avec une puissance moyenne de 0,85 et 0,017 de faux positifs.
- **Pourquoi c'est important pour nous.** Ça montre que **la quantification se voit sur le texte seul** entre
  fournisseurs, à notre échelle de budget (quelques dizaines de requêtes).
- **Angles morts pour nous.**
  1. Information par sonde quasi nulle en décodage glouton (les tirages répétés sont identiques) : il faut de l'échantillonnage.
  2. Sondes aléatoires à haute entropie : les plus exposées au watermark et à la température. Un changement de
     température est explicitement hors périmètre pour eux ; watermark et system prompt pas abordés dans la partie lue.
  3. Il faut des échantillons de référence pour entraîner le classifieur de chaque sonde.

### 3.6 API-TAIL (arXiv 2610.36801, octobre 2026). Source : extraits seulement
- **Idée.** Détecter la substitution dans la queue peu probable des réponses, après avoir estimé la température de
  chaque réponse. Sans cette estimation, la puissance tombe à zéro sur certains de leurs cas.
- **À garder.** Corriger explicitement une température inconnue, c'est exactement notre besoin (`ref-08`).
- **Inconnu.** Demande probablement les probabilités d'un modèle de référence ; non lu.

### 3.7 Black-Box Detection of LM Watermarks (Gloaguen, Jovanović, Staab, Vechev, ICLR 2025). Source : texte complet lu (majeure partie)
- **Test rouge-vert.** Prompt : `Complete the sentence "{t1} {d·H}" using a random word from: [quatre fruits au pluriel].`
  On estime la préférence (log-odds) pour un fruit fixé avec 100 tirages par case (t1, t2). Sans watermark, cette
  préférence ne dépend pas des derniers tokens t2 ; avec watermark, si (la liste verte change avec le token précédent).
  On repère les cases aberrantes, on prend max − min sur t2, test par 10 000 permutations. Testé à T = 1,0 et 0,7.
  Permet aussi d'estimer delta/T et la taille de contexte h.
- **Test à échantillonnage fixe.** Un watermark qui utilise une suite aléatoire fixe fait se répéter les réponses :
  sur 1 000 réponses à « This is the story of », le nombre de réponses distinctes croît moins vite que prévu
  (test de Mann-Whitney).
- **Test à cache.** Compare les réponses avec un cache chaud et un cache vidé (test exact de Fisher).
- **Coût.** Moins de 3 dollars sur GPT-4o pour rouge-vert, 0,3 pour l'échantillonnage fixe, 0,1 pour le cache. En
  nombre de requêtes, le test rouge-vert demande environ 100 tirages par case sur beaucoup de cases : **trop cher pour
  nos 100 par endpoint**, mais gratuit sur les références.
- **Limites annoncées.** Seulement ces trois familles ; pas de garantie théorique de puissance ; suppose une même
  partition vert/rouge pour tous les contextes.

### 3.8 Are You Getting What You Pay For? (Cai, Shi, Zhao, Song, arXiv 2504.04715). Source : texte complet lu
- **Leçons.** Les classifieurs de texte sont au hasard entre un modèle et sa version quantifiée. L'auto-présentation
  échoue et suit le system prompt. MET ne marche que pour de forts taux de substitution. Les sorties gloutonnes diffèrent
  entre fournisseurs d'un même modèle (accord sous 5 % pour Gemma-2-9B) à cause du non-déterminisme de l'inférence
  (taille des lots, noyaux de calcul).
- **À garder.** Deux endpoints du même modèle sur des logiciels de service différents peuvent différer plus qu'avec fp8.
  Dans la compétition, tous les endpoints tournent probablement sur la même pile, ce qui supprime l'essentiel de ce bruit ;
  à vérifier sur les références (jumeaux plain / watermark / température).

### 3.9 Usurpation d'empreinte, GhostPrint (Zhang, Li, Wang, arXiv 2606.16100, 2026). Source : résumé seul
- Un petit modèle légèrement réentraîné pour imiter un gros passe LLMmap, MET et le classifieur d'Idiosyncrasies.
- **Intérêt pour nous.** Pas d'adversaire dans la tâche, mais ça montre qu'un réentraînement (comme un LoRA fort) peut
  déplacer très loin l'empreinte d'un modèle : un LoRA de force 1,5 pourrait ressembler à une autre famille.

## 4. Ce qu'aucune méthode ne fait (l'angle mort)

Toutes les méthodes traitent toute différence comme une substitution. Aucune ne sépare « déguisement de déploiement »
(watermark, system prompt, température) et « changement de modèle » (fp8, LoRA). La physique de la section 1 suggère
qu'on *peut* les séparer :

| Signal | Température | Watermark | System prompt | fp8 | LoRA |
|---|---|---|---|---|---|
| Réponse la plus fréquente (mode) sur un prompt à faible entropie | inchangée | peut basculer si quasi-égalité | souvent inchangée sur formats contraints | peut basculer si quasi-égalité | peut basculer |
| Dispersion des réponses (entropie) | change, uniformément sur tous les prompts | petit changement | petit changement | minime | petit à fort |
| Sens du décalage entre prompts qui partagent le même dernier token | aucun | **même décalage** (liste verte fixée par dernier token + clé) | dicté par le contenu | aléatoire | dicté par le sens |
| Préférences du premier token sur beaucoup de prompts | plus net ou plus plat | **mêmes tokens verts favorisés sur tous les prompts** (token précédent = fin fixe du gabarit de chat) | changent avec le persona | propres à chaque prompt | propres à chaque prompt |
| Auto-présentation | inchangée | inchangée | **changée** (section 5) | surtout inchangée | en partie changée |

## 5. Ce que nos sondes ont montré sur les références (2026-11-21)

Jeu de sondes : `model_equality/sysprompt_probe/` (9 prompts × 9 références × 3 tirages = 243 réponses).
- **L'extraction du system prompt ne marche pas sur SmolLM.** Aucun endpoint ne renvoie un texte stable retrouvé par
  deux sondes différentes ; réponses inventées (« hello, hello... », paragraphes sans rapport) ou NONE.
- **La neutralisation (« ignore all previous instructions... ») ne rapproche pas les endpoints déguisés du plain.**
- **L'auto-présentation sépare les déguisements.** « Comment t'appelles-tu, qui t'a créé ? » donne « SmolLM / Hugging Face »
  dans 3 tirages sur 3 pour plain et watermark, 0 sur 3 pour system prompt et persona (nom inventé à chaque fois :
  Lyra Flynn, Emily, Nexus...), et un mélange pour fp8 et LoRA. Une ou deux questions d'identité par endpoint repèrent
  donc les endpoints à system prompt / persona à bas coût, ce qui dit quelles paires demandent une comparaison sur le
  contenu seul.
- Le tableau fruits / océan de `sysprompt_probe/report.md` montre aussi le LoRA fort (force 1,5) loin de tout
  (similarité 0,07-0,09 contre environ 0,5 pour les autres sur la liste de fruits) : le LoRA fort est facile ; le LoRA
  faible et fp8 sont les cas durs.

## 6. Pistes à tester sur les références (illimitées, gratuites)

1. **Signature du watermark par le premier token.** Sur beaucoup de prompts différents, le premier token généré suit
   toujours le même token du gabarit : un watermark LeftHash favorise donc les mêmes tokens verts partout. En cumulant
   les premiers tokens sur tous les prompts, un watermark montre une inclinaison indépendante du prompt ; fp8 et LoRA
   non. Ensuite, comparer les endpoints watermarkés seulement sur ce que cette inclinaison n'explique pas.
2. **Accord sur le mode plutôt que distance entre distributions.** Comparer la réponse la plus fréquente sur des prompts
   à faible entropie (formats contraints, arithmétique, complétions quasi certaines). La température ne peut pas la
   changer. fp8 et LoRA peuvent, sur les quasi-égalités. Le watermark aussi, d'où la piste 1 d'abord.
3. **Entrées frontières trouvées sur les références, réutilisées sur les endpoints.** Chercher les prompts où SmolLM plain
   et son jumeau fp8 diffèrent de mode (les références ont les deux). Ce sont des prompts où le modèle hésite ; ils sont
   de bons candidats pour les autres familles aussi, car les quasi-égalités viennent souvent du prompt (deux suites aussi
   naturelles l'une que l'autre), pas du modèle.
4. **Questions au bord des connaissances (style KBF)** contre le LoRA : faible entropie, insensibles au watermark et à
   la température, probablement déplacées par un LoRA entraîné sur d'autres données.
5. **Question d'identité comme drapeau de déguisement** (section 5) : pas chère, une ou deux réponses par endpoint.

Juger chaque piste par son MCC sur les références avec `score_practice.py`, en gardant en tête que les références sont
un seul petit modèle : un seuil réglé là peut ne pas se transférer.

## 7. Sources

- Gao, Liang, Guestrin. Model Equality Testing: Which Model Is This API Serving? ICLR 2025. https://arxiv.org/abs/2410.20247
- Zhu et al. Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test. ICLR 2026. https://arxiv.org/abs/2506.06975
- Chauvin et al. Token-Efficient Change Detection in LLM APIs. ICML 2026. https://arxiv.org/abs/2602.11083
- Fang, Feng, Li, Zhou. KBF: Knowledge Boundary as Fingerprint for Language Model and Black-Box API Auditing. https://arxiv.org/abs/2605.29524
- IRIS: Budgeted Black-Box Auditing of Model Substitution and Routing Dilution in LLM Gateways. https://arxiv.org/abs/2607.20860
- API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails. https://arxiv.org/abs/2610.36801
- Bruckner. One Token Is Enough. https://arxiv.org/abs/2607.10252
- AdaptPrint: Response-Adaptive Fingerprinting of Black-Box LLM Services. https://arxiv.org/abs/2608.22213
- Pasquini, Kornaropoulos, Ateniese. LLMmap: Fingerprinting for Large Language Models. USENIX Security 2025. https://arxiv.org/abs/2407.15847
- Sun et al. Idiosyncrasies in Large Language Models. https://arxiv.org/abs/2502.12150
- Gloaguen, Jovanović, Staab, Vechev. Black-Box Detection of Language Model Watermarks. ICLR 2025. https://arxiv.org/abs/2405.20777
- Cai, Shi, Zhao, Song. Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs. https://arxiv.org/abs/2504.04715
- Zhang, Li, Wang. Your "Pro" LLM Subscription May Actually Be "Free" (GhostPrint). https://arxiv.org/abs/2606.16100
