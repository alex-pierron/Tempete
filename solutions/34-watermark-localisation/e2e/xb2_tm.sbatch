#!/bin/bash
# X-LOC-B xb2-tm (LOC lane, 30 min): end-to-end fine-tuning of the official trustmark extractor on a MIX of real train
# images (4 per batch) and synthetic composites with known messages (4 per batch, b2_synth recipe as the deployed
# heads, generated on CPU before this task; shapes of held-out images excluded via the b2 folds).
# Stop rule (written before the run): OOF >= heads + 0.05 (PS 0.636 / TM 0.633), or blend with heads >= heads
# + 0.02, else the e2e line stops (second and last try).
#SBATCH --time=00:35:00
source ~/xlocb/env.sh
B=$T/agents/r-wml-b2
export PYTHONPATH=$B/src HACKATHON_DATA_DIR=$B/data
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 XLB_BUDGET_SECONDS=1400 XLB_SYNTH_DIR=$W/synth XLB_B2_FOLDS=1 XLB_LR_BACKBONE=1e-4
# The CPU generation writes READY when its files are complete: wait for it (at most 5 min).
for i in $(seq 60); do [ -f $W/synth/READY ] && break; sleep 5; done
echo "start $(date -u +%FT%TZ) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} ready=$(ls $W/synth/READY 2>&1)"
timeout 1680 $PY -W ignore ~/xlocb/e2e_loc.py trustmark $W/data/pack_trustmark.npz $W/output/xb2_tm > $W/output/xb2_tm.log 2>&1
grep -E "synthetic|flip|fold [0-9]:|done|Error|error" $W/output/xb2_tm.log | tail -n 12
echo "done $(date -u +%FT%TZ)"
