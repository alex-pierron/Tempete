#!/bin/bash
# X-LOC-B xb1 (LOC lane, prio 045, 40 min): END-TO-END fine-tuning of the official Pixel Seal / TrustMark
# extractors for localization on the REAL train images (5 folds, 512 px, FPN decoder + known-message agreement
# maps). Two runs share one GPU: PS / TM at lr 1e-4 (xb1b: lr 5e-5, first 2 stages frozen).
# Stop rule (written before the run): keep only if OOF mIoU >= heads + 0.05 (PS 0.636, TM 0.633), else stop.
#SBATCH --time=00:40:00
source ~/xlocb/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 XLB_BUDGET_SECONDS=2000
echo "start $(date -u +%FT%TZ) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
run() {  # run <tag> <method> [VAR=value ...]
  local tag=$1 method=$2; shift 2
  env "$@" timeout 2280 $PY -W ignore ~/xlocb/e2e_loc.py $method $W/data/pack_$method.npz $W/output/$tag \
    > $W/output/$tag.log 2>&1
}
run xb1_ps pixelseal XLB_LR_BACKBONE=1e-4 &
run xb1_tm trustmark XLB_LR_BACKBONE=1e-4 &
sleep 240; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
wait
for t in xb1_ps xb1_tm; do echo "== $t"; grep -E "flip|fold [0-9]:|done|Error|error" $W/output/$t.log | tail -n 8; done
echo "done $(date -u +%FT%TZ)"
