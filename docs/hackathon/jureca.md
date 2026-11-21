# GPU : JURECA (Jülich Supercomputing Centre)

> Sources : README et scripts officiels `https://huggingface.co/datasets/SprintML/hackathon`
> (`hackathon_setup.sh`, `teammate.sh`) et le support organisateur « Compute Infrastructure »
> (`GPU-usage-paris.pdf`, 23 slides), lus le 21/11/2026. Les points encore marqués
> **[À VÉRIFIER]** n'apparaissent dans aucun des deux.

## 0. Matériel (support organisateur, slide 2 et 23)

| | Nœuds de calcul (jobs) | Nœuds de connexion (édition, petits tests) |
|---|---|---|
| Nombre | 40 nœuds, ~200 GPU au total | 12 |
| GPU | 4 × NVIDIA A100-SXM4 40 GB (HBM2e) | 2 × Quadro RTX 8000 |
| CPU | 2 × AMD EPYC 7742 (128 cœurs) | 2 × AMD EPYC 7742 (128 cœurs) |
| RAM | 512 GB | 1 024 GB |
| Pilote | 580.95.05, CUDA 13.0 | |

- **Un nœud entier est réservé par job** (slide 2), et le message d'accueil de JURECA prévient
  que, à cause d'un bug Slurm, les GPU ne sont pas isolés par cgroups (slide 9) : un job demandé
  avec `--gres=gpu:1` voit en pratique les 4 A100. Bon usage : lancer jusqu'à 4 expériences dans
  le même job, chacune épinglée sur un GPU (`CUDA_VISIBLE_DEVICES=0..3`), plutôt que 4 jobs.
  **[À VÉRIFIER]** au premier `nvidia-smi` dans un job.
- Le modèle de job organisateur demande `--cpus-per-task=128` (slide 21), le README HF en demande
  8 : avec un nœud réservé par job, 128 est le bon réglage pour les chargements de données.

## 1. Ce qui est fixé par les organisateurs

| Paramètre | Valeur |
|---|---|
| Machine | JURECA (`jureca.fz-juelich.de`) |
| Projet | `training2666` |
| Réservation Slurm | `cispahack` |
| Partition | `dc-gpu` |
| Ressources par job | 1 nœud entier réservé ; `--gres=gpu:1` ; 8 CPU (README HF) ou 128 (support organisateur) |
| Dossier d'équipe partagé | `/p/scratch/training2666/<chef>/<équipe>/` |
| Python | 3.12 via `uv`, un `.venv` par tâche, créé par le script du chef |

## 2. Accès (chacun, une fois ; support organisateur, slides 3 à 14)

1. **Compte** : `https://judoor.fz-juelich.de/login` → « Register », puis valider le mail.
2. **MFA** : activer l'authentification à deux facteurs (TOTP : FreeOTP, Google Authenticator...).
   Le code TOTP est demandé à **chaque** connexion SSH.
3. **Projet et accord** : rejoindre le projet du hackathon (`training2666`) et **signer l'accord
   d'utilisation de JURECA** (ligne « JURECA-DC_GPU ») ; sans signature, pas d'accès.
4. **Clé SSH** : ed25519 obligatoire (les clés `id-rsa` sont refusées). Une clé dédiée :
   `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_juelich -C "<email>"`.
5. **Dépôt de la clé** dans JuDoor (« Manage SSH-keys » sur la ligne JURECA) : la clé publique
   **doit** porter une clause `from="..."` qui limite les adresses IP sources. JuDoor affiche
   l'IP actuelle et propose des plages pour le réseau courant : les copier dans « Additional
   public key options ». Piège : la clause ne couvre que le réseau d'où l'on dépose la clé ; si
   l'on change de réseau (Wi-Fi du lieu, partage 4G, VPN), redéposer une clé ou élargir la clause.
   **Déposer la clé depuis le réseau qu'on utilisera pendant le hackathon.**
6. **Attendre jusqu'à 15 min** que la clé soit active.
7. **Configuration SSH** (`~/.ssh/config`, utilisable aussi par VS Code Remote-SSH) :

   ```text
   Host juelich
     Hostname jureca.fz-juelich.de
     User <user-juelich>
     IdentityFile ~/.ssh/id_ed25519_juelich
     IdentitiesOnly yes
     MACs hmac-sha2-256-etm@openssh.com
   ```

   puis `ssh juelich` (code TOTP demandé). VS Code : « Connect to... » → `juelich`.
8. Les nœuds de calcul n'ont en général pas d'accès Internet : téléchargements, `uv add` et
   soumissions au serveur depuis un nœud de connexion. **[À VÉRIFIER]**

Variante sans les scripts d'équipe (support organisateur, slides 16 à 20) : dossier personnel
dans le projet (`jutil env activate -p training2666 ; cd $PROJECT ; mkdir $USER ; chmod o-r $USER`),
`uv` installé dans `~/.local/bin`, caches `UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR`, `UV_TOOL_DIR`
sous `$PROJECT/$USER/uv/` dans `~/.bashrc`, puis `exec bash`. Les scripts d'équipe de la section 3
font l'équivalent dans `/p/scratch` avec un dossier partagé : les préférer.

## 3. Installation de l'équipe (ordre impératif)

### Chef d'équipe (une seule fois)

```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
# éditer la section CONFIGURATION de hackathon_setup.sh :
#   TEAM_LEAD="pierron1"  TEAMMATE_1="castiel1"  TEAMMATE_2="biechy1"  TEAM_FOLDER="tempete"   (équipe Tempête)
export HF_TOKEN=<token d'un compte AUTORISÉ sur le dataset localisation>   # sinon ce dataset échoue
source hackathon_setup.sh      # à SOURCER, pas exécuter ; attendre la fin avant les coéquipiers
```

Le script : active le projet, crée le dossier d'équipe, pose les ACL (accès réservé à l'équipe),
installe `uv`, écrit dans `~/.bashrc` les caches partagés (`UV_CACHE_DIR`, `HF_HOME`... dans le
dossier d'équipe), télécharge les 3 datasets et crée un `.venv` par tâche depuis son
`requirements.txt`.

Piège connu : le script télécharge `SprintML/Hackathon_Watermark_Localisation_Task` (sans `_Paris`,
alors que le PDF cite la version `_Paris`) et ce dépôt est **restreint** : sans `HF_TOKEN` d'un
compte autorisé, l'étape échoue pour cette tâche seulement (« Download failed - skipping venv
setup ») et les deux autres continuent. Relancer après avoir obtenu l'accès : le script saute ce
qui existe déjà.

### Coéquipiers (chacun, après le chef)

```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
# BUG du script officiel (vérifié le 21/11/2026, ligne 48) : PROJECT_NAME="training2625".
# Le README et hackathon_setup.sh utilisent training2666 : corriger AVANT de lancer.
sed -i 's/^PROJECT_NAME="training2625"/PROJECT_NAME="training2666"/' teammate.sh
grep -n '^PROJECT_NAME=' teammate.sh      # doit afficher training2666
# éditer : TEAM_LEAD="<user du chef>"  TEAM_FOLDER="<nom d'équipe>"   (PAS son propre user)
source teammate.sh
```

Ne pas retélécharger les datasets ni recréer les `.venv`.

### À chaque connexion

```bash
jutil env activate -p training2666
source ~/.bashrc
cd /p/scratch/training2666/<chef>/<équipe>/<tâche> && source .venv/bin/activate
```

## 4. Notre code sur JURECA

Le dépôt de l'équipe est privé. Deux options, au choix :
- `rsync` depuis le Mac (le plus simple, aucune clé à déposer sur le cluster) :
  `rsync -az --exclude .venv --exclude data --exclude weights --exclude third_party ./ <user>@jureca.fz-juelich.de:/p/scratch/training2666/<chef>/<équipe>/Tempete/`
- ou `git clone` sur le nœud de connexion avec une clé de déploiement du dépôt.

Exécution : le code `src/hackathon` n'a besoin que des paquets des `.venv` officiels :
`PYTHONPATH=/p/scratch/training2666/<chef>/<équipe>/Tempete/src python -m hackathon.<tâche> ...`.
Si un paquet manque, se coordonner avant tout `uv add` (environnement partagé par l'équipe).

## 5. Jobs GPU (Slurm)

Modèles prêts : `jureca/unlearning.sbatch`, `jureca/wmloc.sbatch` (Model Equality tourne sur CPU).

```bash
mkdir -p output          # Slurm exige le dossier de logs AVANT sbatch
sbatch jureca/unlearning.sbatch
squeue -u "$USER"        # suivi
tail -f output/<jobid>.out
scancel <jobid>          # arrêt
```

Règles partagées (README officiel) : ne pas supprimer les dossiers ou datasets partagés, ne pas
toucher aux ACL (sauf chef), ne pas recréer les `.venv` sans accord, sorties et logs dans
`<tâche>/output/`.
