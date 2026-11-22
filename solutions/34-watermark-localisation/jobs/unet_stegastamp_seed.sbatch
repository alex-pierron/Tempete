#!/bin/bash
# StegaStamp seed 19 of the u1 recipe, for the seed mean; full-resolution network, PUSHED recipe (synthetic pre-training + two fine-tuning
# recipes, both cross-validated out of fold), one GPU, 30 min. Same body as the Pixel Seal
# variant; method, seed, tag and budgets come from the export lines below.
# Shared inode quota: ONE file written on scratch, $WORK/output/<tag>.npz (20 tries, 15 s apart).
#SBATCH --time=00:30:00
export WLS_METHOD=stegastamp WLS_TAG=x17_ss_s19 WLS_SEED=19
export WLS_TRAIN_SECONDS=660 WLS_FT_B_SECONDS=60 WLS_FT_B_SHARE=0.5
# Environment (cluster): JURECA_TEAM_ROOT = team folder on the cluster scratch; WMLOC_HEADS_DIR = work
# folder of the heads (src/ copy of this repository, data/ with the official dataset and feature caches);
# WMLOC_UNET_DIR = work folder of the full-resolution network (src/, output/).
TEAM_ROOT="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
WORK="${WMLOC_UNET_DIR:?set WMLOC_UNET_DIR}"
PY=$TEAM_ROOT/Hackathon_Watermark_Localisation_Task/.venv/bin/python
METHOD=${WLS_METHOD:-stegastamp}
TAG=${WLS_TAG:-x_$METHOD}
SEED=${WLS_SEED:-11}
export HACKATHON_DATA_DIR="${WMLOC_HEADS_DIR:?set WMLOC_HEADS_DIR}/data"
export HF_HOME="$WMLOC_HEADS_DIR/hf" HF_HUB_OFFLINE=1
export PYTHONPATH=$WORK/src PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU=${GPU%%,*}
SHM=/dev/shm/x2lg17_$$
OUT=$WORK/output
LOG=$SHM/log
GEN=hackathon.wmloc.research.small_gen
TRAIN=hackathon.wmloc.research.small_train
START=$(date +%s)
GEN_SECONDS=360
GEN_GRACE=120
TRAIN_SECONDS=${WLS_TRAIN_SECONDS:-840}
FINETUNE_SECONDS=30
HARD_STOP=$((START + 29 * 60))
mkdir -p "$SHM/data" "$LOG"
trap 'rm -rf "$SHM"' EXIT
trap 'rm -rf "$SHM"; exit 143' TERM INT
echo "host=$(hostname) start=$(date -u +%FT%TZ) task=x-small-pushed method=$METHOD seed=$SEED" \
  "gpu=$GPU TASK_GPUS=${TASK_GPUS:-unset} cpus=$(nproc) train=$TRAIN_SECONDS" \
  "ftb=$WLS_FT_B_SECONDS/$WLS_FT_B_SHARE"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
gen() {
  CUDA_VISIBLE_DEVICES=$1 timeout $((GEN_SECONDS + GEN_GRACE)) $PY -W ignore -m $GEN \
    --method "$METHOD" --worker "$3" --out-dir "$SHM/data" --seconds $GEN_SECONDS \
    --max-pairs 700 --max-rects 2400 --device "$2" > "$LOG/gen_w$3.log" 2>&1 &
}
for w in 0 1 2; do gen "$GPU" cuda $w; done
if [ "$METHOD" = stegastamp ]; then
  for w in 3 4; do OMP_NUM_THREADS=4 gen "" cpu $w; done
else
  gen "$GPU" cuda 3
fi
wait
echo "generation done $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
tail -n 2 "$LOG"/gen_w*.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU timeout $((HARD_STOP - $(date +%s))) $PY -W ignore -m $TRAIN \
  --method "$METHOD" --in-dir "$SHM/data" --out-dir "$OUT" --tag "$TAG" --seed "$SEED" \
  --train-seconds $TRAIN_SECONDS --finetune-seconds $FINETUNE_SECONDS \
  --finetune-b-seconds ${WLS_FT_B_SECONDS:-0} --finetune-b-real-share ${WLS_FT_B_SHARE:-0.5} \
  --batch 32 2>&1 | grep --line-buffered -v "^  step [0-9]*[1-9]00 "
echo "training done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
ls -la "$OUT/$TAG.npz"
echo "done $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
