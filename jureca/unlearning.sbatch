#!/bin/bash
# Unlearning (task 31-unlearning) on JURECA: one A100, official task venv.
#
# Header = the model of the official README (section 5). The account, reservation, partition and
# GPU request must match what the organizers assigned; do not change them.
#
# Submit FROM THE TASK DIRECTORY (the relative --output path resolves against it) and create the
# log directory first:
#
#   cd "$TEAM_DIR/unlearning" && mkdir -p output
#   TEAM_DIR=/p/scratch/training2666/<lead>/<team> REPO=<path of this repo's checkout> \
#   METHOD=gold_soft CONFIG=full SETS="target_temperature=2.0 epochs=3" \
#   sbatch "$REPO/jureca/unlearning.sbatch"
#
# Variables (all read from the environment of the sbatch call, thanks to the default --export=ALL):
#   TEAM_DIR   REQUIRED. /p/scratch/training2666/<team-lead>/<team-folder>
#   REPO       checkout of this repo on JURECA (default: $TEAM_DIR/challenge); only src/ is used.
#   TASK_DIR   folder holding model.pt, forget_images.npz, .venv, output (default: $TEAM_DIR/unlearning).
#   PROXY_ROOT folder holding coco_val_subset / div2k_valid / kodak (default: $TASK_DIR/proxy_images;
#              rsync them from the laptop once, see docs/hackathon/unlearning.md).
#   METHOD     identity | scrub | gold_soft   (default gold_soft)
#   CONFIG     preset (full | smoke) or a JSON file path (default full)
#   SETS       space separated key=value overrides of the config, e.g. "lr=2e-5 epochs=4"
#   RUN_TAG    suffix of the output file (default: the Slurm job id)
#
# The job writes $TASK_DIR/output/<METHOD>_<RUN_TAG>.pt (validated) and a .json sidecar with the
# config, the history and the proxy table. It NEVER submits to the evaluation server.

#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --job-name=unlearning

set -euo pipefail

: "${TEAM_DIR:?set TEAM_DIR=/p/scratch/training2666/<team-lead>/<team-folder>}"
TASK_DIR="${TASK_DIR:-$TEAM_DIR/unlearning}"
REPO="${REPO:-$TEAM_DIR/challenge}"
PROXY_ROOT="${PROXY_ROOT:-$TASK_DIR/proxy_images}"
METHOD="${METHOD:-gold_soft}"
CONFIG="${CONFIG:-full}"
SETS="${SETS:-}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"

cd "$TASK_DIR" || exit 1
mkdir -p output

# Shared task environment (torch, timm, numpy, requests); nothing else is needed.
source .venv/bin/activate
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
# The submission log and defaults of hackathon.common live here, away from the shared task folder.
export HACKATHON_DATA_DIR="${HACKATHON_DATA_DIR:-$TEAM_DIR/hackathon_data}"

OUTPUT_FILE="$TASK_DIR/output/${METHOD}_${RUN_TAG}.pt"

SET_ARGS=()
for assignment in $SETS; do
    SET_ARGS+=(--set "$assignment")
done

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} method=$METHOD config=$CONFIG sets='$SETS'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, timm; print('torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"

srun python -m hackathon.unlearning --task-dir "$TASK_DIR" --device cuda info
srun python -m hackathon.unlearning --task-dir "$TASK_DIR" --device cuda run \
    --method "$METHOD" --config "$CONFIG" --proxy-root "$PROXY_ROOT" \
    --out "$OUTPUT_FILE" ${SET_ARGS[@]+"${SET_ARGS[@]}"}
# Independent check in a fresh process: the file is what the server will load.
srun python -m hackathon.unlearning validate "$OUTPUT_FILE"

echo "done! $OUTPUT_FILE"
