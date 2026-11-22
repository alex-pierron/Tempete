#!/bin/bash
# X-LOC-B dn4 (LOC lane, 10 min): map denoiser trained on the 170 StegaStamp images ONLY (split7 OOF + SAM samGx13
# channels), no RGB and RGB, seed 0; plus a no-SAM control. Asked by X-LOC: SS is where dn adds a clean gain.
#SBATCH --time=00:10:00
source ~/xlocb/env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 XLB_DN_ONLY=stegastamp XLB_DN_EPOCHS=60
echo "start $(date -u +%FT%TZ) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
PACK=$W/data/dn_pack_sep3sam.npz
XLB_DN_RGB=0 XLB_DN_FIRST_SEED=0 timeout 540 $PY -W ignore ~/xlocb/denoise.py $PACK $W/output/dn4_ss_norgb_s0 > $W/output/dn4_ss_norgb_s0.log 2>&1 &
XLB_DN_RGB=0 XLB_DN_FIRST_SEED=1 timeout 540 $PY -W ignore ~/xlocb/denoise.py $PACK $W/output/dn4_ss_norgb_s1 > $W/output/dn4_ss_norgb_s1.log 2>&1 &
XLB_DN_RGB=1 XLB_DN_FIRST_SEED=0 timeout 540 $PY -W ignore ~/xlocb/denoise.py $PACK $W/output/dn4_ss_rgb_s0 > $W/output/dn4_ss_rgb_s0.log 2>&1 &
wait
tail -n 4 $W/output/dn4_*.log
echo "done $(date -u +%FT%TZ)"
