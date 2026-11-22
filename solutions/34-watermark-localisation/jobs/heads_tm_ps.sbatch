#!/bin/bash
# x23b-heads-iso (18 min on one A100: x23 rerun with 48 shards per method, syn8 and syn3): the Pixel Seal and
# TrustMark HEADS (b2_train, the public PS d_syn3 / TM syn8 recipe) retrained on synthetic samples
# made with the chain we measured on the real images: full regime = central crop of a cover
# magnified isotropically by m ~ U(1.02, 1.15) and watermarked whole; rect regime = bounding box
# enlarged by m ~ U(1.02, 1.10) (b2_iso_launch.py wraps b2_synth_features).
# Criterion: out-of-fold head >= +0.005 over 0.5812 (PS d_syn3) and 0.5764 (TM d_syn8).
# Everything new lives in /dev/shm (fresh synth/ and cv/; the real caches are SYMLINKED from
# the heads work folder), so the old synthetic shards are not mixed in and scratch gets ONE file at the end:
#   $OUT_DIR/x23_heads_iso.npz  (keys <result file stem>/<array>).
#SBATCH --time=00:18:00
set -uo pipefail
# Environment (cluster): JURECA_TEAM_ROOT = team folder on the cluster scratch; WMLOC_HEADS_DIR = work
# folder of the heads (src/ copy of this repository, data/ with the official dataset and feature caches);
# WMLOC_UNET_DIR = work folder of the full-resolution network (src/, output/).
TEAM_ROOT="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
AGENT_DIR="${WMLOC_HEADS_DIR:?set WMLOC_HEADS_DIR}"
TASK_DIR=$TEAM_ROOT/Hackathon_Watermark_Localisation_Task
OUT_DIR="${WMLOC_UNET_DIR:?set WMLOC_UNET_DIR}/output"
TAG=x23b_heads_iso
SHARDS_PER_PROCESS=24
GEN_SECONDS=$((5 * 60))
SHARD_SIZE=128
SYNTHETIC_PER_REAL=3
EPOCHS=20
DROPOUT=0.5
FIT_SEED=8000
GPU=${CUDA_VISIBLE_DEVICES:-0}
GPU=${GPU%%,*}
START=$(date +%s)
HARD_STOP=$((START + 17 * 60))
SHM=/dev/shm/x2lg23b_$$
ROOT=$SHM/data/wmloc/team_b2
mkdir -p "$ROOT/synth" "$SHM/data/wmloc/team_b2/cv" "$SHM/log"
trap 'rm -rf "$SHM"' EXIT
trap 'rm -rf "$SHM"; exit 143' TERM INT
for item in feat dino global rgb128 clean_images.npz stegastamp_feats.onnx; do
  [ -e "$AGENT_DIR/data/wmloc/team_b2/$item" ] && ln -s "$AGENT_DIR/data/wmloc/team_b2/$item" "$ROOT/$item"
done
source "$TASK_DIR/.venv/bin/activate"
export PYTHONPATH="$AGENT_DIR/src"
export HACKATHON_DATA_DIR="$AGENT_DIR/data"
export WMLOC_B2_CACHE="$ROOT/feat"
export WMLOC_CLEAN_IMAGES="$AGENT_DIR/clean"
export HF_HOME="$AGENT_DIR/hf" HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WMLOC_ISO_RANGE="1.02,1.15" WMLOC_ISO_RECT_RANGE="1.02,1.10"
echo "host=$(hostname) start=$(date -u +%FT%TZ) task=x23b-heads-iso gpu=$GPU shards/process=$SHARDS_PER_PROCESS"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
ls -la "$ROOT"

# Stage 1: iso synthetic shards: 2 processes per method on the one GPU, 2 threads
# each (8 in total), 6 shards of 128 asked per process under a 13 min timeout; finished shards
# are enough (estimate ~12 per method at 4 threads, so ~6-8 here).
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
gen() {  # gen <method> <first shard> <seed>
  CUDA_VISIBLE_DEVICES=$GPU WMLOC_ISO_SEED=$3 timeout $GEN_SECONDS \
    python -W ignore ~/x2lg/b2_iso_launch.py --method "$1" --first-shard "$2" \
    --shards $SHARDS_PER_PROCESS --shard-size $SHARD_SIZE > "$SHM/log/gen_$1_$2.log" 2>&1
}
gen pixelseal 0 300 &
gen pixelseal $SHARDS_PER_PROCESS 324 &
gen trustmark 0 400 &
gen trustmark $SHARDS_PER_PROCESS 424 &
wait
echo "generation done $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
tail -n 2 "$SHM"/log/gen_*.log | cut -c1-200
ls "$ROOT/synth/pixelseal" "$ROOT/synth/trustmark" 2>&1 | head -n 40

# Stage 2: x23 (12 shards, syn3) gave PS 0.5808 / TM 0.5718 (kit syn3 0.5812 / 0.5641): TM gained.
# Now 48 shards per method (6144 samples, as d_syn3/d_syn8), and both public recipes:
#   A) syn8 (public TM head recipe): cv + fit in parallel; B) syn3 (public PS head): cv + fit.
# Two processes at a time on the GPU (14 GB of maps each), 4 threads each.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WMLOC_GPU_DATA_GB=14
run_pair() {  # run_pair <synthetic per real> <tag> <fit seed>
  local common="--epochs $EPOCHS --dropout $DROPOUT --threads 4 --synthetic --synthetic-per-real $1"
  CUDA_VISIBLE_DEVICES=$GPU timeout $((HARD_STOP - $(date +%s))) python -W ignore -m \
    hackathon.wmloc.research.b2_train cv --methods pixelseal trustmark $common --tag "$2" \
    > "$SHM/log/cv_$2.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=$GPU timeout $((HARD_STOP - $(date +%s))) python -W ignore -m \
    hackathon.wmloc.research.b2_train fit --methods pixelseal trustmark $common --tta --seed "$3" \
    --tag "fit_$2_s0" > "$SHM/log/fit_$2.log" 2>&1 &
  wait
  echo "pair $2 done $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
  for f in "cv_$2" "fit_$2"; do echo "-- $f"; grep -vE "epoch [0-9]*[1-9] " "$SHM/log/$f.log" | tail -n 4 | cut -c1-300; done
}
run_pair 8 i_syn8 8010
run_pair 3 i_syn3b 8020
ls -la "$SHM/data/wmloc/team_b2/cv"

# ONE container on scratch: every npz/json of cv/ packed as <stem>/<array> (json as a string).
python - "$SHM/data/wmloc/team_b2/cv" "$OUT_DIR/$TAG.npz" <<'PYEOF'
import json, sys, time
from pathlib import Path
import numpy as np
source, target = Path(sys.argv[1]), Path(sys.argv[2])
payload = {}
for path in sorted(source.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            payload[f"{path.stem}/{key}"] = data[key]
for path in sorted(source.glob("*.json")):
    payload[f"{path.stem}/json"] = np.array(path.read_text())
if not payload:
    sys.exit("nothing to write")
for attempt in range(20):
    try:
        with open(target, "wb") as handle:
            np.savez(handle, **payload)
        print(f"wrote {target} ({len(payload)} arrays)", flush=True)
        break
    except OSError as error:
        print(f"write failed ({error}), try {attempt + 1}/20", flush=True)
        time.sleep(15)
PYEOF
ls -la "$OUT_DIR/$TAG.npz"
echo "done $(date -u +%FT%TZ) ($(( $(date +%s) - START )) s)"
