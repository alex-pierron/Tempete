# Processus d'équipe : d'une idée à une soumission notée

> Chaîne automatique du hackathon (24 h) : une idée devient une spécification TOML, une
> expérience tracée (local ou JURECA), une ligne du registre versionné, puis une soumission
> validée. Code : `src/hackathon/lab/`, CLI `python -m hackathon.lab`, raccourcis `make hk-*`.
> Tout ce qui n'a pas pu être vérifié sans accès à JURECA est marqué **[À VÉRIFIER]**.

Sommaire : 1. Règle d'or · 2. Vue d'ensemble · 3. Mise en place (une fois) · 4. Le pas à pas ·
5. Exemple Unlearning · 6. Exemple Model Equality · 7. Exemple Watermark Localization ·
8. JURECA en détail · 9. Garde-fous · 10. Dépannage.

## 1. Règle d'or

1. **Validation locale d'abord.** Aucun fichier ne part au serveur sans avoir passé le `validate` de
   sa tâche (le lab l'impose) et sans qu'une mesure locale justifie l'envoi (ref-* pour Model
   Equality, proxys pour Unlearning, split d'entraînement pour la localisation).
2. **On n'optimise pas sur le score public.** Il porte sur 30 % des données cachées ; le classement
   final utilise les 70 % restants et les sujets préviennent qu'il peut s'effondrer. Le score
   public est enregistré à titre d'information ; la décision (garder, abandonner une idée) se prend
   sur la métrique locale. Le tableau `board` sépare les deux « meilleurs » pour cette raison.
3. **Une seule personne soumet par tâche.** Le journal de soumission et le délai de 30 minutes de
   Model Equality sont par équipe, pas par personne. Le lab lit le registre partagé : si un
   coéquipier a soumis il y a moins de 30 minutes (et que vous avez fait `git pull`), il refuse.
4. **`--yes` est un geste explicite, jamais automatisé.** Aucune cible `make` ne soumet, aucune
   spécification ne peut contenir `submit`, `--yes` ni `query-ep` (refus à la lecture du TOML). Les
   100 réponses par endpoint `ep-*` sont irremplaçables : leur dépense reste une commande manuelle.

## 2. Vue d'ensemble

```
make hk-new            experiments/<tâche>/<id>.toml  (spec)   +  <id>.md (fiche d'idée)
      |
make hk-run SPEC=<id>  ----- local ------------------------------------> runs/<tâche>/<run_id>/
      |                 \--- jureca : rsync, 1 job, 1 variante / A100 --->  (stdout, stderr, sorties)
      |                                                                      |
      +--> experiments/registry.jsonl (versionné) <--------------------------+
      |       run_id, tâche, idée, SHA git, où, statut, métriques locales, fichier, score public
make hk-fetch          (JURECA seulement : rapatrie métriques + fichier de soumission)
make hk-board          tableau terminal + runs/board.html (meilleur local / meilleur public)
make hk-submit RUN=    valide le fichier, n'envoie RIEN
python -m hackathon.lab submit RUN --yes     valide, soumet, inscrit le score public au registre
```

Ce qui est versionné (à committer) : `experiments/**` (specs, fiches, `registry.jsonl`). Ce qui ne
l'est pas : `runs/` (logs, fichiers de soumission, `board.html`), gitignoré.

Le registre est un journal en ajout seul : chaque changement d'état d'un run ajoute un instantané
complet, le plus récent (`updated_at`) fait foi. `experiments/.gitattributes` le déclare
`merge=union` : un `git merge` entre deux coéquipiers garde les lignes des deux côtés.

## 3. Mise en place (une fois par personne)

### 3.1 `.env` (jamais committé)

```bash
cp .env.example .env
# HACKATHON_API_KEY, HF_TOKEN (compte autorisé), JURECA_USER, JURECA_TEAM_LEAD, JURECA_TEAM_FOLDER
# optionnel : JURECA_SSH_HOST (alias ssh, défaut "juelich"),
#             MET_ANSWERS_FILE=model_equality/answers.jsonl (convention d'équipe, voir model_equality.md)
```

`JURECA_TEAM_LEAD` et `JURECA_TEAM_FOLDER` donnent le dossier partagé
`/p/scratch/training2666/<chef>/<équipe>/`. Ils sont validés (lettres, chiffres, `. _ -`) : tout
autre caractère est refusé, car ils finissent dans des lignes de commande.

### 3.2 `~/.ssh/config` : ne taper le code TOTP qu'une seule fois

JURECA demande un code TOTP à **chaque** connexion. Avec une connexion maître persistante, le code
n'est demandé qu'à la première commande : toutes les suivantes (`ssh`, `rsync`) la réutilisent
pendant 8 h. Bloc à ajouter (en complément de celui de `docs/hackathon/jureca.md`, §2.7) :

```text
Host juelich
  Hostname jureca.fz-juelich.de
  User <user-juelich>
  IdentityFile ~/.ssh/id_ed25519_juelich
  IdentitiesOnly yes
  MACs hmac-sha2-256-etm@openssh.com
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
```

`ls -ld ~/.ssh` doit répondre `drwx------` (le dossier doit exister, sinon la socket ne se crée pas).
Vérifier : `make hk-login` (demande le code TOTP) puis `ssh -O check juelich` répond
`Master running`. Fermer la session en fin de journée : `ssh -O exit juelich`.

Si `ControlMaster` n'est pas configuré, le lab fonctionne mais avertit que le code sera redemandé
à chaque commande. Même dans ce cas, `make hk-login` ouvre la session avec le terminal attaché :
c'est ce qui permet à ssh d'afficher l'invite TOTP.

### 3.3 Vérifier l'installation

```bash
make hk-test          # ~380 tests du kit, aucun réseau (dont ~160 du lab)
make help | grep hk-  # les raccourcis
```

## 4. Le pas à pas

### 4.1 Créer l'idée

```bash
make hk-new TASK=unlearning IDEA="gold soft, température 2"
# créé experiments/unlearning/gold-soft-temperature-2.toml  (la spec)
# créé experiments/unlearning/gold-soft-temperature-2.md    (la fiche : hypothèse, métrique, résultat, décision)
```

`TASK` accepte `unlearning`, `model_equality`, `wmloc` (et les alias `unl`, `met`, `wm`). L'identifiant
est le slug de l'idée ; il ne peut pas exister deux fois, même sous deux tâches.

**Écrire l'hypothèse et la métrique attendue dans la fiche `.md` avant de lancer.** Une idée qu'on ne
peut pas réfuter n'apprend rien.

### 4.2 La spécification TOML

```toml
id = "gold-soft-temperature-2"          # = le nom du fichier
task = "unlearning"
idea = "gold soft, température 2"
hypothesis = "..."
where = "local"                         # ou "jureca" ; `run --where` l'emporte

[resources]                             # optionnel
gpus = 3                                # >= nb de variantes, 4 au plus ; défaut = nb de variantes
time_limit = "01:00:00"                 # --time Slurm ; défaut : celui de la partition

[sweep]                                 # optionnel : produit cartésien, 4 variantes AU PLUS
target_temperature = [1.5, 2.0, 3.0]    # chaque clé devient le placeholder {target_temperature}

[[steps]]                               # une ou plusieurs étapes, dans l'ordre
name = "train"
args = ["run", "--method", "gold_soft", "--set", "target_temperature={target_temperature}",
        "--out", "{out}/candidate.pt"]  # une liste : un élément = un argument, jamais de shell
args_local = ["--config", "smoke"]      # ajoutés en local seulement
args_jureca = ["--config", "full", "--proxy-root", "{task_dir}/proxy_images"]  # JURECA seulement

[expect]
submission = "candidate.pt"             # relatif au dossier de sortie du run
metrics_file = "candidate.json"         # JSON ; les feuilles numériques deviennent des métriques
primary_metric = "proxy.retain_kl"      # la métrique locale qui classe les runs sur le board
higher_is_better = false
[expect.metrics_regex]                  # optionnel : un flottant capturé dans les logs (dernier match)
mcc = 'Practice score \(MCC[^)]*\): (-?[0-9.]+)'
```

Placeholders : `{out}` (dossier de sortie du run), `{run_id}`, `{variant}` (`v1`..`v4`),
`{variant_index}`, les clés du `[sweep]`, et en `args_jureca` seulement `{task_dir}` (dossier de la
tâche sur JURECA) et `{team_dir}`. Un placeholder inconnu est refusé à la lecture de la spec.

Une étape doit nommer au moins une sous-commande connue de la tâche (`run`, `analyze`, `bench`...).
Sur JURECA, l'adaptateur de la tâche place devant la sous-commande les options globales utiles
(Unlearning : `--task-dir <dossier de la tâche>`, comme `jureca/unlearning.sbatch`).

Statuts d'un run : `queued` (en file Slurm) → `running` → `finished` (JURECA : terminé, pas encore
rapatrié) → `done` (fichiers collectés ; **seul statut soumettable**) ou `failed` (avec la raison
dans la colonne `note` du registre). Un fichier attendu manquant fait échouer le run ; une métrique
introuvable n'est qu'un avertissement (on ne jette pas un bon fichier pour une faute de frappe).

### 4.3 Lancer

```bash
make hk-run SPEC=gold-soft-temperature-2                  # local
make hk-run SPEC=gold-soft-temperature-2 WHERE=jureca     # JURECA : rsync + un job Slurm
```

En local, les variantes tournent l'une après l'autre ; stdout et stderr s'affichent en direct
(préfixe `  | `) et sont écrits dans `runs/<tâche>/<run_id>/stdout.log` et `stderr.log`. Ce dossier contient
aussi `spec.toml` (copie figée), `run.json` (commandes exactes, SHA git, drapeau `dirty`, version de
Python ; aucun secret) et les sorties. Une ligne `running` est écrite au registre **avant** le
lancement : un run interrompu laisse une trace. Ctrl-C le marque `failed`.

Le SHA git enregistré ne reproduit l'expérience que si l'arbre était propre : un `*` (ou
`dirty: true`) signale une modification non committée. Committer avant de lancer une expérience
qu'on veut défendre.

### 4.4 Suivre, rapatrier, comparer

```bash
make hk-status                       # JURECA : met à jour les jobs en file / en cours
make hk-fetch RUN=<run_id|group_id>  # rapatrie métriques + fichier ; un group_id = toutes ses variantes
make hk-board                        # tableau terminal + runs/board.html ; TASK=unlearning pour filtrer
```

### 4.5 Soumettre (une personne par tâche)

```bash
make hk-submit RUN=<run_id>                       # valide, n'envoie rien (sort 2 : « --yes absent »)
git pull                                          # voir les soumissions des coéquipiers (délai 30 min)
PYTHONPATH=$PWD/src .venv/bin/python -m hackathon.lab submit <run_id> --yes    # le geste explicite
git add experiments && git commit -m "chore(lab): score public de <run_id>"   # partager le registre
```

`submit` vérifie, dans l'ordre : le run est `done` ; le fichier est sous `runs/`, existe, et son
SHA-256 est celui enregistré (un fichier modifié à la main est refusé) ; le `validate` de la tâche
passe ; le run n'a pas déjà été soumis (`--again` pour passer outre) ; le délai du serveur est
respecté d'après le registre partagé. Puis il lance le `submit --yes` de la tâche (donc le journal
`data/hackathon/submissions/log.jsonl` et la garde de délai du kit sont inchangés), lit le score
dans la réponse (JSON ou ligne `ok=... score=...`, sinon dans ce journal) et l'inscrit au registre.
Un refus **local** du kit (clé absente, délai local) n'est pas une soumission : rien n'est enregistré
et aucun délai ne démarre. Un rejet **par le serveur** compte (il relance le délai de 30 minutes).

## 5. Exemple complet : Unlearning

Objectif : trouver la température de la cible `gold_soft` (réglage décisif d'après `unlearning.md`)
sans surapprendre le score public.

```bash
make hk-unl-fetch                                      # une fois : model.pt (sha256 vérifié) + forget_images.npz
make hk-unl-smoke                                      # hors lab : la méthode tourne-t-elle ? (< 2 min sur M5)

make hk-new TASK=unlearning IDEA="gold soft temperature"
$EDITOR experiments/unlearning/gold-soft-temperature.toml    # sweep target_temperature = [1.5, 2.0, 3.0]
$EDITOR experiments/unlearning/gold-soft-temperature.md      # hypothèse AVANT de lancer
make hk-run SPEC=gold-soft-temperature                 # 3 variantes smoke sur MPS : valide la chaîne

# images de rétention proxy, une fois, vers JURECA (voir unlearning.md §3 étape 3) :
rsync -av data/coco_val_subset data/kodak juelich:/p/scratch/training2666/<chef>/<équipe>/unlearning/proxy_images/

make hk-login                                          # code TOTP, une seule fois
make hk-run SPEC=gold-soft-temperature WHERE=jureca    # rsync + 1 job, 3 variantes sur 3 A100, config full
make hk-status                                         # queued -> running -> finished
make hk-fetch RUN=<group_id>                           # 3 runs -> done (3 × ~344 Mo rapatriés)
make hk-board TASK=unlearning                          # comparer proxy.retain_kl, la confiance, la dérive
make hk-submit RUN=<run_id retenu>                     # validation seule
PYTHONPATH=$PWD/src .venv/bin/python -m hackathon.lab submit <run_id retenu> --yes   # une personne
```

Ce que mesure le board ici : `proxy.retain_kl` (dérive sur des images hors forget, **plus bas =
mieux**) est un garde-fou, **pas** le score : le vrai score est une distance à un modèle « gold »
que nous n'avons pas (voir `unlearning.md` §1 et §4). Choisir un plateau, pas un pic.

## 6. Exemple complet : Model Equality

Tout est hors ligne tant qu'on n'interroge pas les `ep-*`. Ce qui coûte du budget reste manuel.

```bash
make met-budget                          # état du serveur (GET /models, gratuit) vs le store local
make met-plan PHASE=ep-1                 # coût exact de la phase, hors ligne
git pull                                 # AVANT de dépenser : un coéquipier a peut-être interrogé
PYTHONPATH=$PWD/src .venv/bin/python -m hackathon.model_equality query-ep --phase ep-1 --yes   # MANUEL, une personne
git add artifacts/model_equality && git commit -m "chore(met): réponses ep-1"    # le budget ne se rachète pas

make hk-new TASK=model_equality IDEA="seuil modele 3"
$EDITOR experiments/model_equality/seuil-modele-3.toml
make hk-run SPEC=seuil-modele-3          # analyze --calibrate + make-submission, hors ligne, sans GPU
make hk-board TASK=model_equality
make met-validate FILE=runs/model_equality/<run_id>/submission.csv
make hk-submit RUN=<run_id>              # validate seulement
PYTHONPATH=$PWD/src .venv/bin/python -m hackathon.lab submit <run_id> --yes
```

Pour balayer un seuil (4 valeurs au plus), décommenter `[sweep]` dans la spec et ajouter
`"--tau-model", "{tau_model}"` aux arguments de l'étape `analyze`. Pour une mesure locale honnête,
ajouter une étape `score-practice` sur les ref-* et déclarer `practice_mcc` en `primary_metric`
(modèle commenté dans le gabarit) : les comptes de familles et de modèles du gabarit ne sont que
descriptifs. Fichier de réponses d'un coéquipier : variable `MET_ANSWERS_FILE` (voir `.env.example`).

Model Equality ne gagne rien à tourner sur JURECA (CPU) ; `WHERE=jureca` fonctionne mais inutilement.

## 7. Exemple : Watermark Localization (squelette)

Le CLI `python -m hackathon.wmloc` (`fetch|synth|bench|predict|validate|submit`) est livré par un
autre chantier. L'adaptateur est déjà enregistré dans `src/hackathon/lab/tasks.py` (sous-commandes
connues, dossier `watermark_localization` sur JURECA, identifiant serveur lu dans
`HACKATHON_TASK_WMLOC`) et un gabarit existe :

```bash
make hk-new TASK=wmloc IDEA="seuil par methode"
# remplacer dans la spec les arguments de `bench` et `predict` par les vrais, déclarer la métrique
# (primary_metric = nom de la métrique mIoU écrite par `bench`), puis comme pour les autres tâches :
make hk-run SPEC=seuil-par-methode
make hk-submit RUN=<run_id>
```

Points d'intégration à la fusion : voir §9 (liste « À brancher »).

## 8. JURECA en détail

| Commande | Effet |
|---|---|
| `make hk-login` | `ssh -O check juelich` ; sinon `ssh juelich true` avec le terminal attaché (invite TOTP) |
| `make hk-sync` | `rsync -az` du dépôt vers `/p/scratch/training2666/<chef>/<équipe>/Tempete/` |
| `lab jureca submit SPEC` | génère le job depuis `jureca/lab_job.sbatch.tmpl`, l'envoie, `sbatch --parsable` |
| `make hk-status` | `squeue` + `sacct` + fichiers `exit_code`, un appel ssh par job |
| `make hk-fetch RUN=` | `rsync` du dossier du run, puis métriques et fichier collectés |

`make hk-run WHERE=jureca` enchaîne sync et submit (`--no-sync` pour sauter le rsync ; `--dry-run`
affiche le script de job et la commande distante, sans rien envoyer ni écrire).

**Exclusions du rsync** : `.venv`, `data`, `weights`, `third_party`, `runs`, `.env` (jamais envoyé),
`.git`, caches, `*.pt|*.pth|*.ckpt|*.safetensors`. Aucun `--delete` : rien n'est jamais supprimé sur le
cluster. Les dossiers du script officiel (`<tâche>/.venv`, jeux de données) sont voisins du dépôt,
hors de portée du rsync.

**Le job** (un seul par lancement, quel que soit le nombre de variantes) : en-tête imposé par les
organisateurs (`--account=training2666 --reservation=cispahack --partition=dc-gpu --gres=gpu:1
--cpus-per-task=128`, sorties `output/%j.out|err`). Slurm n'isole pas les GPU : le nœud entier est
visible, et chaque variante est épinglée sur « son » A100 par `CUDA_VISIBLE_DEVICES=0..3`. Si le nœud
montre moins de GPU que de variantes (Slurm se met à isoler **[À VÉRIFIER]**), les variantes tournent
l'une après l'autre au lieu de viser un indice inexistant. Chaque variante écrit `stdout.log`,
`stderr.log`, ses sorties, puis `exit_code` **en dernier** : sa présence signifie « terminé ».
Sorties distantes : `<équipe>/lab_runs/<tâche>/<group_id>/v1..v4/` (hors du dépôt synchronisé).

Le job ne soumet **jamais** : les nœuds de calcul n'ont en général pas d'accès Internet, et la clé
d'API reste sur le portable. Toujours : `fetch` sur le Mac, puis `submit`.

## 9. Garde-fous (ce que le lab refuse ou fait pour vous)

- Spec : `submit`, `query-ep`, `--yes` interdits partout (y compris venant d'une valeur de balayage) ;
  clés inconnues refusées ; plus de 4 variantes refusé ; `args` doit être une liste.
- Hôte et noms issus de `.env` validés (`-oProxyCommand=...` ou `a;b` refusés) ; `ssh`/`rsync` lancés
  par listes d'arguments, jamais par un shell local ; chaque morceau dynamique de la commande
  distante est protégé par `shlex.quote` ; le script de job est généré avec des valeurs entre guillemets.
- Un envoi JURECA qui échoue (code non synchronisé, venv absent, ssh coupé) ne laisse ni dossier de
  run ni ligne de registre.
- `submit` : voir §4.5. Le registre est partagé par git : un fichier hors de `runs/` désigné par une
  ligne de registre n'est jamais envoyé.
- Jamais de commande distante hors de ce qui est listé au §8 ; jamais de `docker volume rm`, de `push`.

**À brancher à la fusion du chantier localisation** (le lab ne dépend pas de son code) :
1. confirmer les sous-commandes dans `WMLOC.subcommands` (`tasks.py`) ;
2. confirmer que `validate FILE` et `submit FILE --yes` suivent la même forme que les autres tâches
   (sinon surcharger `validate_args` / `submit_args` de l'adaptateur) ;
3. aligner `experiments/_templates/wmloc.toml` (arguments de `bench` et `predict`, nom de la métrique) ;
4. renseigner `HACKATHON_TASK_WMLOC` dans `.env` pour la garde de délai et le repli sur le journal ;
5. confirmer le nom du dossier de tâche sur JURECA (`watermark_localization`, d'après le guide) et si
   la tâche attend des options globales (équivalent de `--task-dir`) : `remote_global_args`.

## 10. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `ssh login to 'juelich' failed` | code TOTP expiré/erroné, clé non active, IP hors de la clause `from=` | `make hk-login` à nouveau ; revoir `jureca.md` §2 |
| `connection failed: TOTP expired?` | la session maître est tombée | `make hk-login` |
| `code not synced: run jureca sync` | rien dans `Tempete/` côté cluster | `make hk-sync` (ou ne pas passer `--no-sync`) |
| `no .venv in the task folder` | le script du chef n'a pas créé le venv de cette tâche | demander au chef de relancer `hackathon_setup.sh` |
| `JURECA_TEAM_LEAD is not set` | `.env` incomplet | compléter `.env` |
| run `failed`, `note` : `expected submission file not found` | la commande n'a pas écrit ce que `[expect]` annonce | lire `runs/<tâche>/<run_id>/stderr.log` |
| `submit` : `changed since the run recorded it` | le fichier a été modifié après le run | relancer l'expérience ; ne pas soumettre un fichier édité |
| `submit` : `cooldown, wait N more minute(s)` | une soumission (la vôtre ou celle d'un coéquipier, via le registre) est trop récente | attendre ; un échec compte aussi |
| `registry.jsonl:N: not valid JSON` | conflit de merge laissé dans le registre | garder les deux côtés, retirer les marqueurs `<<<<` `====` `>>>>` |
| `rsync failed` | réseau, ou clé/ControlMaster | `ssh -O check juelich`, sinon `make hk-login` |
| lancé par erreur, il faut annuler | job en file ou en cours | `ssh juelich scancel <job_id>` (le `job_id` est dans `make hk-status`) ; le run restera `failed` au prochain `hk-status` |
