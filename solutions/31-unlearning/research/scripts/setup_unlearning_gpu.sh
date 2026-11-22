#!/usr/bin/env bash
# Installation de la tâche Unlearning sur une machine Linux avec GPU NVIDIA (ex. RTX 5080), en une
# commande, depuis la racine d'un clone du dépôt (branche alex) :
#
#     bash scripts/setup_unlearning_gpu.sh
#
# Idempotent : relancer ne retélécharge rien. Étapes :
#   1. prérequis : uv, nvidia-smi, pilote NVIDIA >= 580 (torch 2.15 du verrou = CUDA 13) ;
#   2. environnement Python : `uv sync` (.venv, Python 3.12, torch CUDA du uv.lock) ;
#   3. contrôle GPU : torch voit la carte, son architecture est compilée, un calcul bf16 passe ;
#   4. données : model.pt + forget_images.npz (sha256 vérifié), images proxy COCO + Kodak ;
#   5. smoke sur la carte : trust_region (banc compris) et gold_soft (augmentations sur GPU).
# Aucune clé n'est nécessaire (.env facultatif pour un run local) ; rien n'est soumis au serveur.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$*"; }
fail() { printf '\nÉCHEC : %s\n' "$*" >&2; exit 1; }

say "1/5 Prérequis"
command -v uv >/dev/null || fail "uv absent. Installer : curl -LsSf https://astral.sh/uv/install.sh | sh (puis rouvrir le terminal)."
command -v nvidia-smi >/dev/null || fail "nvidia-smi absent : pilote NVIDIA non installé (sous Windows, utiliser WSL2)."
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
printf 'GPU : %s ; pilote %s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" "$driver"
[ "${driver%%.*}" -ge 580 ] || fail "pilote $driver trop ancien : torch 2.15 (CUDA 13) demande un pilote NVIDIA >= 580."

say "2/5 Environnement Python (uv sync)"
uv sync
PY=.venv/bin/python
UNL=(env "PYTHONPATH=$PWD/src" "$PY" -m hackathon.unlearning)

say "3/5 Contrôle GPU"
"$PY" - <<'EOF'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(f"torch {torch.__version__} ne voit aucun GPU CUDA (pilote, ou build CPU de torch ?).")
major, minor = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
if f"sm_{major}{minor}" not in arches and f"compute_{major}{minor}" not in arches:
    sys.exit(f"architecture sm_{major}{minor} absente de ce torch ({arches}).")
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float()).all()
total = torch.cuda.mem_get_info()[1] / 1e9
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)} "
      f"(sm_{major}{minor}), {total:.1f} Go : OK")
EOF

say "4/5 Données"
"${UNL[@]}" fetch
"$PY" scripts/fetch_data.py --dataset coco_val_subset
"$PY" scripts/fetch_data.py --dataset kodak

say "5/5 Smoke sur la carte (config smoke : 64 images, quelques pas)"
out=runs/unlearning/setup-check
mkdir -p "$out"
"${UNL[@]}" run --method trust_region --config smoke --set target_temperature=6 \
    --set holdout_fraction=0.2 --out "$out/trust_region.pt" | grep -E "device=|score_min|retain|finished"
"${UNL[@]}" run --method gold_soft --config smoke --set holdout_fraction=0.2 \
    --out "$out/gold_soft.pt" | grep -E "score_min|retain|finished"

say "Prêt. Lancer les expériences (config full sur CUDA, variantes l'une après l'autre) :"
echo "    (make passe chaque run par scripts/gpu_guard.sh : un job à la fois, pause à 83 °C, télémétrie runs/gpu/)"
cat <<'EOF'
    make hk-run SPEC=trust-region-kl-radius
    make hk-run SPEC=bench-baselines
    make hk-run SPEC=bench-seed-noise
    make hk-board TASK=unlearning
EOF
