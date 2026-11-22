#!/bin/bash
# Vérification du cluster en ~2 min, à lancer une fois (ou après un changement d'environnement) :
#   - GPU réellement visibles par un job (le guide annonce 4 A100 malgré --gres=gpu:1) et driver ;
#   - torch de chaque .venv de tâche : version, CUDA du build, calcul sur chaque GPU ;
#   - onnxruntime sur GPU pour le watermark (StegaStamp) ;
#   - accès internet et au serveur d'évaluation depuis un nœud de calcul.
# En-tête identique à jureca/lab_job.sbatch.tmpl : si ce job passe, celui du lab passe aussi.
#
# Depuis un nœud de connexion :
#   cd /p/scratch/trainingXXXX/<chef>/<équipe>/checks && mkdir -p output
#   bash ../Tempete/jureca/team_sbatch.sh gpu_check.sbatch     # jamais sbatch direct : 1 nœud max pour l'équipe
#   cat output/<job>.out

#SBATCH --account=trainingXXXX
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.out
#SBATCH --job-name=gpu-check

TEAM_DIR="${TEAM_DIR:-$(cd .. && pwd)}"
echo "== nœud $(hostname), job ${SLURM_JOB_ID:-?}, $(nproc) CPU, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-non défini}"
echo "== nvidia-smi (GPU physiques du nœud)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

for task in unlearning model_equality Hackathon_Watermark_Localisation_Task; do
    echo "== $task"
    py="$TEAM_DIR/$task/.venv/bin/python"
    [ -x "$py" ] || { echo "   pas de .venv"; continue; }
    # CUDA_VISIBLE_DEVICES retiré : on teste ce que le lab utilise (il épingle lui-même 0..3).
    (cd "$TEAM_DIR/$task" && env -u CUDA_VISIBLE_DEVICES TASK="$task" "$py" - <<'PY'
import importlib.util, os
if importlib.util.find_spec("torch") is None:
    print("   torch absent (attendu pour model_equality, qui tourne sur CPU)")
    raise SystemExit
import torch
n = torch.cuda.device_count()
print(f"   torch {torch.__version__} | CUDA du build {torch.version.cuda} | cuda dispo {torch.cuda.is_available()} | GPU vus {n}")
for i in range(n):
    x = torch.randn(4096, 4096, device=f"cuda:{i}")
    y = float((x @ x).abs().mean())
    torch.cuda.synchronize(i)
    print(f"   cuda:{i} {torch.cuda.get_device_name(i)} : calcul OK ({y:.1f})")
if os.environ["TASK"].startswith("Hackathon_Watermark"):
    import onnxruntime as ort   # après torch : onnxruntime-gpu réutilise ses bibliothèques CUDA 12
    s = ort.InferenceSession("models/stegastamp/stega_stamp_logits.onnx",
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"   onnxruntime {ort.__version__} | session StegaStamp sur {s.get_providers()[0]}")
PY
    )
done

echo "== réseau depuis le nœud de calcul (000 = pas d'accès)"
echo "   huggingface.co : $(curl -s -m 10 -o /dev/null -w '%{http_code}' https://huggingface.co)"
echo "   serveur d'évaluation : $(curl -s -m 10 -o /dev/null -w '%{http_code}' ${HACKATHON_SERVER_URL:-http://hackathon-server.invalid}/leaderboard_page)"
echo "== fin"
