#!/bin/bash
# Watermark Localization on JURECA (Julich): 1 A100, 8 CPUs, partition dc-gpu, reservation cispahack.
# Resources are copied from the official main.sh (README of SprintML/hackathon): do not change them.
#
# Usage, from the TASK directory (so that output/ exists and receives the Slurm logs):
#   export WMLOC_TASK_DIR=/p/scratch/training2666/<team-lead>/<team-folder>/Hackathon_Watermark_Localisation_Task
#   export HACKATHON_REPO=<path of this repository on JURECA>
#   cd "$WMLOC_TASK_DIR" && mkdir -p output
#   sbatch "$HACKATHON_REPO/jureca/wmloc.sbatch" bench --manifest <train manifest> --localizers sw_max_96_128
#   sbatch "$HACKATHON_REPO/jureca/wmloc.sbatch" predict --manifest <test manifest> --out output/sub.pt
#
# Everything after the script name is passed to `python -m hackathon.wmloc`. Compute nodes have no
# internet access: run `fetch` (and any pip install) on the login node, not in a job. This script
# never submits to the leaderboard: `submit --yes` is a separate, deliberate step.
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

set -euo pipefail

: "${WMLOC_TASK_DIR:?set WMLOC_TASK_DIR to the Hackathon_Watermark_Localisation_Task folder}"
: "${HACKATHON_REPO:?set HACKATHON_REPO to the checkout of this repository}"

# The official task venv carries torch and the kit's dependencies; our package is pure Python on
# top of it, hence PYTHONPATH rather than an install (do not modify the shared venv).
cd "$WMLOC_TASK_DIR" || exit 1
source .venv/bin/activate
export PYTHONPATH="$HACKATHON_REPO/src:${PYTHONPATH:-}"
# Data, submission log and .env live with the team folder, not in the repository checkout.
export HACKATHON_DATA_DIR="${HACKATHON_DATA_DIR:-$WMLOC_TASK_DIR/data}"

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch wmloc.sbatch <bench|predict|validate|...> [args]" >&2
    exit 2
fi

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -u +%FT%TZ)"
srun python -m hackathon.wmloc "$@"
echo "done $(date -u +%FT%TZ)"
