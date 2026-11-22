#!/bin/bash
# X-LOC-B CPU step (inside ONE srun --overlap -c 96 on the relay node): PROCS generators per method, 2 threads
# each, CUDA hidden. Output $W/synth/<method>/s_<k>.npz, then $W/synth/READY.
source ~/xlocb/env.sh
B=$T/agents/r-wml-b2
export PYTHONPATH=$B/src HACKATHON_DATA_DIR=$B/data WMLOC_B2_CACHE=$B/data/wmloc/team_b2/feat
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CLEAN=$B/data/wmloc/team_b2/clean_images.npz
MB=$(( $(stat -c %s $CLEAN) / 1000000 ))
# Every process loads the whole clean pool: keep the total under ~150 GB of RAM.
PROCS=$(( 150000 / (MB + 1500) )); [ $PROCS -gt 24 ] && PROCS=24; [ $PROCS -lt 2 ] && PROCS=2
SECONDS_BUDGET=${SECONDS_BUDGET:-330}
echo "gen start $(date -u +%FT%TZ) host=$(hostname) clean=${MB} MB procs/method=$PROCS free=$(free -g | awk '/Mem/{print $7}') GB"
rm -f $W/synth/READY
for m in pixelseal trustmark; do
  mkdir -p $W/synth/$m
  for k in $(seq 0 $((PROCS - 1))); do
    $PY -W ignore ~/xlocb/gen_synth.py $m 400 $((3000 + k)) $SECONDS_BUDGET $W/synth/$m/s_$k.npz > $W/synth/$m/log_$k.txt 2>&1 &
  done
done
wait
for m in pixelseal trustmark; do echo "$m: $(ls $W/synth/$m/s_*.npz 2>/dev/null | wc -l) files"; tail -qn 1 $W/synth/$m/log_*.txt | head -3; done
touch $W/synth/READY
echo "gen done $(date -u +%FT%TZ)"
