#!/bin/bash
# X-LOC-C (LOC lane, prio 050): SAM ViT-H automatic object masks on the train images of the rect regime
# (transplantation track) -> agents/x-loc-c/sam_train.npz (128 x 128 bit-packed candidates + SAM scores).
#SBATCH --time=00:20:00
TEAM_ROOT="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
W=$TEAM_ROOT/agents/x-loc-c
PY=$TEAM_ROOT/Hackathon_Watermark_Localisation_Task/.venv/bin/python
export SAM_CKPT=$W/sam_vit_h_4b8939.pth SAM_WHEEL=$W/segment_anything-1.0-py3-none-any.whl
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "start $(date -u +%FT%TZ) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
timeout 1020 $PY -W ignore $W/sam_amg.py $TEAM_ROOT/agents/r-wml-b2/data/wmloc/release/train $W/train_rect_names.txt $W/sam_train.npz 900
echo "done status=$? $(date -u +%FT%TZ)"
