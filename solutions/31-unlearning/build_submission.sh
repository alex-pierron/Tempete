#!/bin/bash
# BUILD V4 submissions on the node from arms OR same-setting soups (mean of the deltas of
# the listed members, taken from the v4d / v4e / v4f packs or the v4c raw states), team C recipe, same
# code as x_build_v4.sh (which built B = 0.1768 public). BUILDS below: (members, alpha, head scale, out).
# build10 (06 h 20): ua_L_f3l20_a146 = 0.1959 public (freeze pays, k model blind to it): the other
# trained freeze arms of the v4l pack, alpha interpolated from the v4l-replay probes (accord ~0.77, offset -0.04
# observed on f3_l20), plus f3_l20 at head x1.6 and the f3 l20+l60 soup.
# Previous instance (build9): Alex's frozen-blocks + L2-SP arms (task x-ua-v4l), alpha interpolated for a final
# accord of ~0.79 like A: v4l_f3_l20 (frozen 0-3, lambda 20) at 1.46, v4l_f2_l5 (frozen 0-2, lambda 5) at 1.40.
# Each file: blend -> sibling routing 2 -> bias of the 9 removed classes -= delta (max ms) -> head x s;
# measured again after writing (ms, accord, drift rule), validated by the official validator, sha256.
# Conformity: original + forget set + Alex's V4 states (trained on them only).
# Output: TWO files (one per submission) in $AGENT_DIR/output.

#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00

set -euo pipefail
# Environment (cluster): JURECA_TEAM_ROOT = team folder on the cluster scratch (holds
# unlearning/.venv); UNL_WORK_DIR = work folder (data/unlearning/{model.pt,forget_images.npz},
# jureca/unl_alex_src2.tar = tar of the release src/, output/). See solutions/31-unlearning/README.md.
SHARED="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
AGENT_DIR="${UNL_WORK_DIR:?set UNL_WORK_DIR}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "x-ua-build: needs exactly one GPU, got '${CUDA_VISIBLE_DEVICES:-}'" >&2
    exit 2
fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# shellcheck disable=SC1091
source "$SHARED/unlearning/.venv/bin/activate"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/x_ua_build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
tar -xf "$AGENT_DIR/jureca/unl_alex_src2.tar" -C "$WORK"
export PYTHONPATH="$WORK/src"
echo "x-ua-build: host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
python - "$SHARED/unlearning" "$AGENT_DIR" <<'PYEOF'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hackathon.unlearning.alterations import random_crop_resize
from hackathon.unlearning.constants import DEFAULT_SEED
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.model import build_model, load_state_dict_file, preprocess
from hackathon.unlearning.research.c_classes import (
    centred_mean_direction,
    class_counts,
    cosine_groups,
    fully_forgotten_classes,
)
from hackathon.unlearning.validate import validate_submission

START = time.time()
TASK_DIR, AGENT_DIR = sys.argv[1:3]
import glob
PACKS = sorted(glob.glob(f"{AGENT_DIR}/output/x_ua_v4l_*.pt"))
V4C_DIR = f"{AGENT_DIR}/data/unlearning/candidates/alex_v45"
BUILDS = [
    (["v4l_f3_l60"], 1.50, 1.75, f"{AGENT_DIR}/output/x_ua_L_f3l60_a150.pt"),
    (["v4l_f3_l20", "v4l_f3_l60"], 1.48, 1.75, f"{AGENT_DIR}/output/x_ua_L_f3soup_a148.pt"),
    (["v4l_f2_l60"], 1.47, 1.75, f"{AGENT_DIR}/output/x_ua_L_f2l60_a147.pt"),
    (["v4l_f3_l20"], 1.46, 1.60, f"{AGENT_DIR}/output/x_ua_L_f3l20_a146_s160.pt"),
    (["v4l_f3_l20"], 1.46, 1.50, f"{AGENT_DIR}/output/x_ua_L_f3l20_a146_s150.pt"),
    (["v4l_f3_l5"], 1.45, 1.75, f"{AGENT_DIR}/output/x_ua_L_f3l5_a145.pt"),
]
EVAL_SUBSET, CROP_35_AREAS, CROP_35_OFFSET, BATCH = 1000, (0.33, 0.37), 11, 128
GOLD_MEAN, GOLD_STD, METRIC_T = 2.25, 3.0, 2.0
EXPECTED_REMOVED = [0, 6, 7, 8, 29, 30, 31, 33, 38]
ROUTING = 2.0
DELTAS = tuple(x / 4 for x in range(4, 41))
KEPT_WEIGHT, REMOVED_WEIGHT, PRED_AT_BEST, BEST_AGREEMENT, PRED_SLOPE = 0.91, 0.09, 0.353, 0.735, 0.81

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")


def log(*parts):
    print(f"[{time.time() - START:7.1f}s]", *parts, flush=True)


def log_softmax(z, t=1.0):
    a = z / t
    a = a - a.max(1, keepdims=True)
    return a - np.log(np.exp(a).sum(1, keepdims=True))


def law(z, labels):
    loss = -log_softmax(z)[np.arange(len(labels)), labels]
    mean, std = float(loss.mean()), float(loss.std())
    return abs(mean - GOLD_MEAN) + abs(std - GOLD_STD), mean, std


forget = load_forget_images(Path(TASK_DIR) / "forget_images.npz")
order = torch.randperm(len(forget), generator=torch.Generator().manual_seed(DEFAULT_SEED))
subset = forget[order[:EVAL_SUBSET].sort().values]
views = {"exact": subset,
         "crop_35": random_crop_resize(subset, torch.Generator().manual_seed(DEFAULT_SEED + CROP_35_OFFSET), *CROP_35_AREAS)}
model = build_model().to(dev).eval()
original = load_state_dict_file(Path(TASK_DIR) / "model.pt")


@torch.no_grad()
def features(state, names):
    model.load_state_dict(state, strict=True)
    out = {}
    for name in names:
        images = views[name] if name != "all" else forget
        chunks = []
        for i in range(0, len(images), BATCH):
            x = preprocess(images[i : i + BATCH], dev)
            chunks.append(model.forward_head(model.forward_features(x), pre_logits=True).float().cpu().double())
        out[name] = torch.cat(chunks).numpy()
    return out


W0 = original["head.weight"].double().numpy()
B0 = original["head.bias"].double().numpy()
F_ORIG = features(original, ["exact", "crop_35", "all"])
Z_ORIG = {v: f @ W0.T + B0 for v, f in F_ORIG.items()}
SATURATED = fully_forgotten_classes(class_counts(Z_ORIG["all"].argmax(1)))
assert SATURATED == EXPECTED_REMOVED, SATURATED
LABELS = Z_ORIG["exact"].argmax(1)
KEPT = ~np.isin(LABELS, SATURATED)
D_ORIG = law(Z_ORIG["exact"], LABELS)[0]
P_ORIG = np.exp(log_softmax(Z_ORIG["crop_35"], METRIC_T))
GROUPS = cosine_groups(W0)


def routed_weight(weight, feats_exact):
    weight = weight.copy()
    for members in GROUPS:
        saturated = [c for c in members if c in SATURATED]
        present = [c for c in members if c not in SATURATED]
        if not saturated or not present:
            continue
        selected = feats_exact[np.isin(LABELS, saturated)]
        direction = centred_mean_direction(selected, feats_exact)
        scale = ROUTING / float((selected @ direction).mean()) / len(present) ** 0.5
        weight[present] += scale * direction
    return weight


def measure(z_exact, z_crop):
    distance, mean, std = law(z_exact, LABELS)
    ms = 1.0 - min(1.0, distance / D_ORIG)
    accord = float((z_exact.argmax(1) == LABELS)[KEPT].mean())
    kl = (P_ORIG * (np.log(P_ORIG + 1e-30) - log_softmax(z_crop, METRIC_T))).sum(1)
    dk, dr = float(kl[KEPT].mean()), float(kl[~KEPT].mean())
    retain = 1.0 / (1.0 + KEPT_WEIGHT * dk + REMOVED_WEIGHT * dr)
    pred = max(0.0, PRED_AT_BEST - PRED_SLOPE * (accord - BEST_AGREEMENT))
    return {"ms": ms, "loss_mean": mean, "loss_std": std, "accord_final": accord, "drift_kept": dk,
            "drift_removed": dr, "R_d": retain, "pred": pred, "rule_final": ms * pred * retain}


arms = {}
for pack in PACKS:
    loaded = torch.load(pack, map_location="cpu", weights_only=False)
    for arm, entry in loaded["arms"].items():
        arms[arm] = entry["delta"]
    del loaded
log("arms available", sorted(arms))
for members, alpha, scale, out_path in BUILDS:
    deltas = []
    for arm in members:
        if arm.startswith("v4c_"):
            raw = load_state_dict_file(Path(f"{V4C_DIR}/{arm}.pt"))
            deltas.append({k: raw[k].float() - original[k].float() for k in original})
            del raw
        else:
            deltas.append({k: v.float() for k, v in arms[arm].items()})
    delta = {k: sum(d[k] for d in deltas) / len(deltas) for k in original}
    del deltas
    arm = "+".join(members)
    state = {k: (original[k].float() + alpha * delta[k]).contiguous() for k in original}
    feats = features(state, ("exact", "crop_35"))
    weight, bias = state["head.weight"].double().numpy(), state["head.bias"].double().numpy()
    accord_blend = float(((feats["exact"] @ weight.T + bias).argmax(1) == LABELS)[KEPT].mean())
    routed = routed_weight(weight, feats["exact"])
    best = None
    for d in DELTAS:
        b = bias.copy()
        b[SATURATED] -= d
        distance = law(scale * (feats["exact"] @ routed.T + b), LABELS)[0]
        if best is None or distance < best[0]:
            best = (distance, d)
    d = best[1]
    b = bias.copy()
    b[SATURATED] -= d
    state["head.weight"] = torch.from_numpy(scale * routed).float().contiguous()
    state["head.bias"] = torch.from_numpy(scale * b).float().contiguous()
    check = features(state, ("exact", "crop_35"))
    hw, hb = state["head.weight"].double().numpy(), state["head.bias"].double().numpy()
    metrics = measure(check["exact"] @ hw.T + hb, check["crop_35"] @ hw.T + hb)
    metrics.update({"arm": arm, "alpha": alpha, "scale": scale, "delta": d, "accord_blend": accord_blend,
                    "rule_blend_accord": metrics["ms"] * max(0.0, PRED_AT_BEST - PRED_SLOPE * (accord_blend - BEST_AGREEMENT)) * metrics["R_d"]})
    torch.save(state, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    digest = hashlib.sha256()
    with open(out_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    report = validate_submission(Path(out_path))
    log("BUILT", out_path, "sha256", digest.hexdigest(), json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}))
    log("VALID", report)
log("done")
PYEOF
echo "x-ua-build: done"
