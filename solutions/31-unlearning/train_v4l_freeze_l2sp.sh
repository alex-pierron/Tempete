#!/bin/bash
# Alex's directive "1 and 2 combined" (05 h 20): his V4 with the FIRST BLOCKS FROZEN plus an
# L2-SP penalty that holds the trained weights near the original, on the setting of A
# (v4d_k4_w01_lr4_r95_s1 = 0.1826 public: K 4, anchor 0.1, lr 4e-5, rho 0.95, 15 epochs, seed 1).
#   frozen through block 2 (trainable_blocks 9) or through block 3 (trainable_blocks 8)
#   L2-SP lambda 5 / 20 / 60 (weak / medium / strong)
# Freezing uses Alex's own `trainable_blocks`. L2-SP is not in his CLI: a decoupled pull toward the
# original weights is patched into `_apply_update` of the node-local copy of his code (tmpfs, never
# written back): after each AdamW step, p -= lr * lambda * (p - p_original). With the trust region, a
# run still stops at its KL radius, so the same forgetting is reached with weights nearer the original.
#
# Conformity: every arm is a SUBMITTABLE run (holdout_fraction 0, --no-eval); original + forget only.
# Output (inode quota): ONE file $OUT_DIR/x_ua_v4l_<tag>.pt = {arm: {"delta", "card"}}; replayed and built
# by x_ua_v4l_replay (task 2).

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:16:00

set -euo pipefail

# Environment (cluster): JURECA_TEAM_ROOT = team folder on the cluster scratch (holds
# unlearning/.venv); UNL_WORK_DIR = work folder (data/unlearning/{model.pt,forget_images.npz},
# jureca/alex_src_52fa1d3.zip, output/). See solutions/31-unlearning/README.md.
SHARED="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
AGENT_DIR="${UNL_WORK_DIR:?set UNL_WORK_DIR}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "x-ua-v4l: needs exactly one GPU in CUDA_VISIBLE_DEVICES, got '${CUDA_VISIBLE_DEVICES:-}'" >&2
    exit 2
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LANES="${AG_LANES:-3}"
# 19 min of arms, then ~2 min to pack: under the 25 min hard limit of the runner.
DEADLINE_SECONDS=$((12 * 60))
ARM_BUDGET_SECONDS=300
START=$(date +%s)
TAG="${SLURM_JOB_ID:-manual}_$(date +%H%M%S)"
OUT_DIR="$AGENT_DIR/output"
OUT="$OUT_DIR/x_ua_v4l_${TAG}.pt"

LOCAL_ROOT="${TMPDIR:-/tmp}"
case "$LOCAL_ROOT" in /p/*) LOCAL_ROOT=/tmp ;; esac
WORK="$(mktemp -d "$LOCAL_ROOT/x-ua-v4l.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/lanes.log"
# shellcheck disable=SC1091
source "$SHARED/unlearning/.venv/bin/activate"
python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$AGENT_DIR/jureca/alex_src_52fa1d3.zip" "$WORK/src_zip"
export PYTHONPATH="$WORK/src_zip/src"

# L2-SP patch of the node-local copy (see header). Fails loudly if the target text moved.
python - "$WORK/src_zip/src/hackathon/unlearning/methods.py" <<'PATCHEOF'
import sys
from pathlib import Path
path = Path(sys.argv[1])
source = path.read_text()
old = """    torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
"""
new = """    torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_CLIP_NORM)
    if _L2SP_LAMBDA > 0 and id(student) not in _L2SP_REFERENCE:
        _L2SP_REFERENCE[id(student)] = [p.detach().clone() for p in student.parameters()]
    optimizer.step()
    if _L2SP_LAMBDA > 0:
        with torch.no_grad():
            for p, p0 in zip(student.parameters(), _L2SP_REFERENCE[id(student)]):
                if p.requires_grad:
                    p.add_(p - p0, alpha=-learning_rate * _L2SP_LAMBDA)
    optimizer.zero_grad(set_to_none=True)
"""
if source.count(old) != 1:
    raise SystemExit("L2-SP patch: target text not found exactly once")
source = source.replace(old, new)
source = source.replace("def _apply_update(", "import os as _l2sp_os\n_L2SP_LAMBDA = float(_l2sp_os.environ.get(\"L2SP_LAMBDA\", \"0\"))\n_L2SP_REFERENCE: dict = {}\n\n\ndef _apply_update(", 1)
path.write_text(source)
print("L2-SP patch applied", flush=True)
PATCHEOF
mkdir -p "$WORK/arms"
echo "x-ua-v4l: gpu=$CUDA_VISIBLE_DEVICES lanes=$LANES work=$WORK out=$OUT"

# The V3 base Alex retained (ref4-v3: b 0, lambda 0.25, rho 0.9, area 0.5, 4 views, no copies),
# here with rho 0.95 and 15 epochs (see header).
V3_BASE="--set trust_target=calibrated --set trust_lambda=0.25 --set trust_hedge=0 \
--set target_temperature=6 --set target_views=4 --set target_view_min_area=0.5 \
--set radius_check_every=1 --set altered_fraction=0 --set epochs=15"
V4="--method trust_region $V3_BASE --set anchor_folds=4 --set anchor_views=0"
# name|arguments, in priority order: the lead and its second seed first.
ARMS=(
    "v4l_f2_l20|20|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=9"
    "v4l_f2_l5|5|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=9"
    "v4l_f2_l60|60|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=9"
    "v4l_f3_l20|20|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=8"
    "v4l_f3_l5|5|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=8"
    "v4l_f3_l60|60|$V4 --set seed=1 --set lr=4e-5 --set anchor_weight=0.1 --set radius_fraction=0.95 --set trainable_blocks=8"
)
NEXT_FILE="$WORK/next"
echo 0 > "$NEXT_FILE"

lane() {
    local lane_id=$1
    while true; do
        local index
        # flock: one lane at a time takes the next arm.
        index=$(flock "$NEXT_FILE" bash -c 'i=$(cat "$1"); echo $((i + 1)) > "$1"; echo "$i"' _ "$NEXT_FILE")
        [ "$index" -lt "${#ARMS[@]}" ] || return 0
        local left=$((START + DEADLINE_SECONDS - $(date +%s)))
        if [ "$left" -lt "$ARM_BUDGET_SECONDS" ]; then
            echo "[lane$lane_id] ${left}s left, stop" >> "$LOG"
            return 0
        fi
        local name="${ARMS[$index]%%|*}" rest="${ARMS[$index]#*|}"
        local lam="${rest%%|*}" args="${rest#*|}"
        echo "[lane$lane_id] start $name at $(( $(date +%s) - START ))s: $args" >> "$LOG"
        # shellcheck disable=SC2086
        if L2SP_LAMBDA="$lam" timeout "$left" python -m hackathon.unlearning --task-dir "$AGENT_DIR/data/unlearning" \
            run --config full $args --set holdout_fraction=0 --no-eval \
            --out "$WORK/arms/$name.pt" > "$WORK/$name.log" 2>&1; then
            echo "[lane$lane_id] done $name at $(( $(date +%s) - START ))s" >> "$LOG"
        else
            echo "[lane$lane_id] FAILED $name (code $?)" >> "$LOG"
            rm -f "$WORK/arms/$name.pt"
        fi
        grep -E "finished in|radius|Error|Traceback|error" "$WORK/$name.log" | tail -4 \
            | cut -c1-900 | sed "s/^/[lane$lane_id] $name | /" >> "$LOG"
    done
}

pids=()
for lane_id in $(seq 1 "$LANES"); do
    lane "$lane_id" &
    pids+=($!)
    sleep 20  # staggered start: the model and forget set load one at a time
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
cat "$LOG"
echo "x-ua-v4l: arms over at $(( $(date +%s) - START ))s, packing"

python - "$WORK/arms" "$AGENT_DIR/data/unlearning/model.pt" "$OUT" <<'PYEOF'
import glob
import json
import os
import sys

import torch

arms_dir, original_path, out_path = sys.argv[1:4]
original = torch.load(original_path, map_location="cpu", weights_only=True)
result = {"original": original_path, "arms": {}}
for path in sorted(glob.glob(f"{arms_dir}/*.pt")):
    name = os.path.basename(path)[: -len(".pt")]
    if name.endswith(".partial"):  # a save cut by the lane timeout (Alex's PARTIAL_MARKER)
        continue
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != set(original):
        print(f"x-ua-v4l: {name} keys differ from the original, skipped", flush=True)
        continue
    delta = {}
    for key, value in original.items():
        d = state[key].float() - value.float()
        delta[key] = d if key.startswith("head.") else d.half()
    card_path = f"{arms_dir}/{name}.json"
    card = json.load(open(card_path, encoding="utf-8")) if os.path.exists(card_path) else None
    history = (card or {}).get("history") or [{}]
    last = history[-1] if isinstance(history, list) and history else {}
    keys = ("forget_kl", "kl_radius_effective", "radius_ratio", "radius_reached", "pending_kl",
            "target_kl", "target_agreement")
    print(f"x2-v4d card {name} " + json.dumps({k: last.get(k) for k in keys}), flush=True)
    result["arms"][name] = {"delta": delta, "card": card}
torch.save(result, out_path + ".tmp")
os.replace(out_path + ".tmp", out_path)
print(f"x-ua-v4l: saved {out_path} arms {list(result['arms'])}", flush=True)
PYEOF
echo "x-ua-v4l: done at $(( $(date +%s) - START ))s"
