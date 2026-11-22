#!/bin/bash
# (08 h 05) explore the per-row head scale found by gridK (removed rows x2.5 / kept x1.5: KL_removed 0.137 at
# ms 0.980, ka_pred 0.237): sharper removed rows (x3, x4 = harder gate: class kept only on strong evidence), kept rows
# x1.25-1.75, body alpha 1.50-1.72. Best KL+accord combos at ms >= 0.97, files x_ua_Ck2_*.

#SBATCH --gres=gpu:1
#SBATCH --time=00:14:00

set -euo pipefail
AGENT_DIR="${UNL_WORK_DIR:?set UNL_WORK_DIR}"
export GRID_DEVICE=cuda GRID_OUT_PREFIX=x_ua_Ck2_ WRITE_TOP=3 GRID_MS_FLOOR=0.97
export GRID_SCALES='[2.5, 3.0, 4.0]' GRID_KEPT_SCALES='[1.25, 1.5, 1.75]'
export GRID_BODIES='[["f2l5", "v4l_f2_l5", [1.50, 1.64, 1.72]]]'
exec bash "$AGENT_DIR/jureca/x_unlc_grid.sh"   # = solutions/31-unlearning/build_head_gate_grid.sh
