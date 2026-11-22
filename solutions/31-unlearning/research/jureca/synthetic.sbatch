#!/bin/bash
# V_Fusion synthetic images on JURECA: 4 class shards in parallel (one per A100 of the node), then
# the quality gates a, b, c, c', d on the merged set. Header copied from unlearning.sbatch; the
# node shows its 4 A100 even with --gres=gpu:1 (see lab_job.sbatch.tmpl), each process is pinned.
#
#   cd "$TEAM_DIR/unlearning" && mkdir -p output
#   TEAM_DIR=/p/scratch/trainingXXXX/<lead>/<team> REPO=<checkout> \
#   bash "$REPO/jureca/team_sbatch.sh" "$REPO/jureca/synthetic.sbatch"   # never plain sbatch
#
# Variables: TEAM_DIR (required), REPO, TASK_DIR as in unlearning.sbatch; GEN_ARGS and GATE_ARGS,
# extra flags of synth-generate and synth-gates (e.g. GEN_ARGS="--per-class 30").
# Writes $TASK_DIR/synthetic/synthetic_{k}of4.pt, synthetic_gated.pt (+ .json) and grid.png.

#SBATCH --account=trainingXXXX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --job-name=synthetic

set -euo pipefail

: "${TEAM_DIR:?set TEAM_DIR=/p/scratch/trainingXXXX/<team-lead>/<team-folder>}"
TASK_DIR="${TASK_DIR:-$TEAM_DIR/unlearning}"
REPO="${REPO:-$TEAM_DIR/challenge}"
GEN_ARGS="${GEN_ARGS:-}"
GATE_ARGS="${GATE_ARGS:-}"
SHARDS=4

cd "$TASK_DIR" || exit 1
mkdir -p output synthetic
source .venv/bin/activate
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1
export HACKATHON_DATA_DIR="${HACKATHON_DATA_DIR:-$TEAM_DIR/hackathon_data}"
CLI=(python -m hackathon.unlearning --task-dir "$TASK_DIR" --device cuda)

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gen='$GEN_ARGS' gates='$GATE_ARGS'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

pids=()
for ((k = 0; k < SHARDS; k++)); do
    # shellcheck disable=SC2086  # GEN_ARGS is a list of flags
    CUDA_VISIBLE_DEVICES=$k "${CLI[@]}" synth-generate --shard "$k/$SHARDS" $GEN_ARGS \
        > "output/synthetic_${SLURM_JOB_ID:-manual}_shard$k.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done  # set -e: a failed shard stops the job here

shards=()
for ((k = 0; k < SHARDS; k++)); do shards+=("synthetic/synthetic_${k}of${SHARDS}.pt"); done
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 "${CLI[@]}" synth-gates "${shards[@]}" --grid synthetic/grid.png $GATE_ARGS

echo "done! $TASK_DIR/synthetic/synthetic_gated.json"
