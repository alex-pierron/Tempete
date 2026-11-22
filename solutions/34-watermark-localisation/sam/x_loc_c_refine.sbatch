#!/bin/bash
# X-LOC-C (LOC lane, prio 038, directeur): refine c_rect boxes (25 train detections, 109 test) to the exact
# embedding crop by coordinate ascent on the soft margin of the known message -> agents/x-loc-c/refined.json.
#SBATCH --time=00:15:00
TEAM_ROOT="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
W=$TEAM_ROOT/agents/x-loc-c
PY=$TEAM_ROOT/Hackathon_Watermark_Localisation_Task/.venv/bin/python
export WMLOC_ROOT=$TEAM_ROOT/agents/r-wml-b2/data/wmloc
export WMLOC_KIT=$TEAM_ROOT/Hackathon_Watermark_Localisation_Task
export HF_HOME=$TEAM_ROOT/agents/r-wml-b2/hf HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "start $(date -u +%FT%TZ) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
timeout 840 $PY -W ignore $W/box_refine.py $W/boxes.json $W/refined.json 2>&1 | grep -v -i -E "warn|state_dict|torch.load|Model loaded"
echo "done status=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
