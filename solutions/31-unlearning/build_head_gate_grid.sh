#!/bin/bash
# X-UNL-C bet (a), 22/11 07 h 05: rebuild the record (ua_L_f3l60_a150 = 0.1973, ua_L_f2l5_a140 = 0.1971) with
# LESS of the team C recipe on the 9 removed classes. Lesson of 06 h 54 (W-UNL): public ~ 0.238 - 0.083 x
# KL_removed (KL(original || file) at T = 2 on the gen2 synthetic images of the 9 removed classes, r = -0.79 on
# 9 reads). The recipe's sibling routing and bias shift act on the CLASS (every image of it, control included),
# not on the forget IMAGES. This grid measures, for each body (arm, alpha), every head variant
# (routing x bias delta, head x1.75) with BOTH predictors, then writes the best few files.
#
# v4 (07 h 29): ka_pred = 0.3728 - 0.0830 x KL_removed - 0.1707 x accord_final, least squares on the 11 read files
# (rms 0.0025; leave-one-out on ua_C_f2l5_a156 0.2168 for 0.2149 read). GRID_MS_FLOOR picks now maximise ka_pred.
# v3 (07 h 24, after ua_C_f2l5_a156_r20_d300 read 0.2149): GRID_WRITE_LIST (JSON [[body, alpha, routing, delta],
# ...]) writes those exact combos; GRID_MS_FLOOR (e.g. 0.97) writes, per (body, alpha), the routing / delta with the
# LOWEST KL_removed among rows with ms >= floor (the direction that paid: forget by the body, not the class bias).
# v2 (07 h 05): GRID_DEVICE=cuda|cpu, GRID_BODIES (JSON [[tag, arm, [alphas]], ...]), GRID_PACK_GLOB and
# GRID_OUT_PREFIX from the environment, so the same grid builds bet (b)'s anchored bodies on a GPU.
# CPU (srun --overlap -c 32 on the running job) or GPU: the head variants do not change the body features, so
# each body is forwarded once (forget exact + crop 35 % subsets as the builds, gen2 1000 images) and every
# head variant is numpy.
#   k model x freeze factor (freeze_predict.py, 6 public V4 points + 2 freeze reads)
#   KL rule  0.238 - 0.083 x KL_removed(gen2)
# Sanity: the routing 2 / best-ms delta / alpha 1.50 row of f3l60 must give back ms 0.964, accord 0.774,
# R_d 0.462 and KL_removed 0.567 (drift probe value).
# Output: GRID / BEST lines in the log; WRITE_TOP files x_ua_C_<tag>.pt in $AGENT_DIR/output (validated, sha256).
# Conformity: original + forget set + Alex's V4 arms (trained on them) + gen2 (made from the original alone).
#
# Usage (login node):
#   nohup srun --jobid=15905178 --overlap -N1 -n1 -c 32 bash x_unlc_grid_cpu.sh > LOG 2>&1 < /dev/null & disown

#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
SHARED="${JURECA_TEAM_ROOT:?set JURECA_TEAM_ROOT}"
AGENT_DIR="${UNL_WORK_DIR:?set UNL_WORK_DIR}"
# shellcheck disable=SC1091
source "$SHARED/unlearning/.venv/bin/activate"
THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORK="$(mktemp -d "${TMPDIR:-/tmp}/x_unlc_grid.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
tar -xf "$AGENT_DIR/jureca/unl_alex_src2.tar" -C "$WORK"
export PYTHONPATH="$WORK/src"
echo "x-unlc-grid: host=$(hostname) threads=$THREADS"
if [ "${GRID_DEVICE:-cpu}" = cpu ]; then export CUDA_VISIBLE_DEVICES=""; fi
python - "$SHARED/unlearning" "$AGENT_DIR" <<'PYEOF'
import glob
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
from hackathon.unlearning.research.c_classes import centred_mean_direction, cosine_groups
from hackathon.unlearning.validate import validate_submission

START = time.time()
TASK_DIR, AGENT_DIR = sys.argv[1:3]
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
PACK_GLOB = os.environ.get("GRID_PACK_GLOB", f"{AGENT_DIR}/output/x_ua_v4l_15903437_*.pt")
DEVICE = torch.device(os.environ.get("GRID_DEVICE", "cpu"))
OUT_PREFIX = os.environ.get("GRID_OUT_PREFIX", "x_ua_C_")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SYN_GLOB = f"{AGENT_DIR}/output/x_ul_inv2_*.npz"
# (tag, arm, alphas): the two record bodies; alpha above the record compensates a weaker routing.
BODIES = [tuple(b) for b in json.loads(os.environ.get("GRID_BODIES", '[["f3l60", "v4l_f3_l60", [1.50, 1.58]], ["f2l5", "v4l_f2_l5", [1.40]]]'))]
ROUTINGS = (0.0, 0.5, 1.0, 1.5, 2.0)
DELTAS = tuple(x / 4 for x in range(0, 41))  # 0 included: no bias shift at all
HEAD_SCALE = 1.75                              # default head scale
# v6 (07 h 50): GRID_SCALES (JSON list) sweeps the head scale too (pred probe: the gold is a smooth model average)
HEAD_SCALES = tuple(json.loads(os.environ.get("GRID_SCALES", "[1.75]")))
# v7 (07 h 56): GRID_HEAD_ALPHAS (JSON list, null = the body alpha) blends the HEAD delta with its own alpha:
# the body alpha moves the memorised images, the head delta also carries class-level (concept) changes.
HEAD_ALPHAS = tuple(json.loads(os.environ.get("GRID_HEAD_ALPHAS", "[null]")))
# v9 (08 h 02): GRID_KEPT_SCALES (JSON list, null = same as the scale): the 91 KEPT-class rows of the head get their
# own scale (softer: moderate confidence on correct kept images, like the smooth gold; less sharpening on the
# control), the 9 removed rows keep `scale` (their high losses carry the ms law).
KEPT_SCALES = tuple(json.loads(os.environ.get("GRID_KEPT_SCALES", "[null]")))
REMOVED_MASK = np.isin(np.arange(100), [0, 6, 7, 8, 29, 30, 31, 33, 38])


def row_scales(scale, kept_scale):
    """Per-row head scale: `scale` on the removed rows, `kept_scale` (or `scale`) on the kept rows."""
    return np.where(REMOVED_MASK, scale, scale if kept_scale is None else kept_scale)
WRITE_TOP = int(os.environ.get("WRITE_TOP", "3"))
RECORD_PUBLIC = 0.1973
EVAL_SUBSET, CROP_35_AREAS, CROP_35_OFFSET, BATCH = 1000, (0.33, 0.37), 11, 125
GOLD_MEAN, GOLD_STD, METRIC_T = 2.25, 3.0, 2.0
REMOVED = [0, 6, 7, 8, 29, 30, 31, 33, 38]
KEPT_WEIGHT, REMOVED_WEIGHT, PRED_AT_BEST, BEST_AGREEMENT, PRED_SLOPE = 0.91, 0.09, 0.353, 0.735, 0.81
KL_RULE_INTERCEPT, KL_RULE_SLOPE = 0.238, 0.083
# v5 (07 h 38): quadratic in accord, 12 reads (rms 0.0024, accord optimum ~0.68)
KA_INTERCEPT, KA_KL, KA_ACCORD, KA_ACCORD2 = -0.0285, -0.0957, 0.8304, -0.6103  # 14 reads incl. Ck2
PUBLIC_V4_POINTS = ((0.757, 0.9585, 0.4631, 0.1768), (0.7819, 0.9775, 0.4556, 0.1826), (0.8131, 0.9748, 0.4641, 0.1690),
                    (0.8583, 0.9178, 0.4925, 0.1654), (0.7866, 0.9816, 0.4573, 0.1803), (0.8037, 0.9640, 0.4680, 0.1753))
FREEZE_POINTS = ((0.7726, 0.9475, 0.4685, 0.1959), (0.7928, 0.9364, 0.4693, 0.1971))
K_COEF = np.polyfit([p[0] for p in PUBLIC_V4_POINTS], [p[3] / (p[1] * p[2]) for p in PUBLIC_V4_POINTS], 2)
FREEZE_FACTOR = float(np.mean([p[3] / (np.polyval(K_COEF, p[0]) * p[1] * p[2]) for p in FREEZE_POINTS]))


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
syn_path = sorted(f for f in glob.glob(SYN_GLOB) if not f.endswith(".tmp.npz"))[-1]
syn = np.load(syn_path)
views = {"exact": subset,
         "crop_35": random_crop_resize(subset, torch.Generator().manual_seed(DEFAULT_SEED + CROP_35_OFFSET), *CROP_35_AREAS),
         "syn": torch.from_numpy(syn["images"])}
SYN_REMOVED = np.isin(syn["labels"], REMOVED)
log("views", {k: tuple(v.shape) for k, v in views.items()}, "syn", syn_path, "syn removed", int(SYN_REMOVED.sum()))
del forget
model = build_model().to(DEVICE).eval()
original = load_state_dict_file(Path(TASK_DIR) / "model.pt")


@torch.no_grad()
def features(state):
    model.load_state_dict(state, strict=True)
    out = {}
    for name, images in views.items():
        chunks = []
        for i in range(0, len(images), BATCH):
            x = preprocess(images[i : i + BATCH], DEVICE)
            chunks.append(model.forward_head(model.forward_features(x), pre_logits=True).float().cpu().double())
        out[name] = torch.cat(chunks).numpy()
    return out


W0 = original["head.weight"].double().numpy()
B0 = original["head.bias"].double().numpy()
F_ORIG = features(original)
Z_ORIG = {v: f @ W0.T + B0 for v, f in F_ORIG.items()}
LABELS = Z_ORIG["exact"].argmax(1)
KEPT = ~np.isin(LABELS, REMOVED)
D_ORIG = law(Z_ORIG["exact"], LABELS)[0]
P_ORIG = np.exp(log_softmax(Z_ORIG["crop_35"], METRIC_T))
LOGP_SYN = log_softmax(Z_ORIG["syn"], METRIC_T)
P_SYN = np.exp(LOGP_SYN)
GROUPS = cosine_groups(W0)
log("original done; D_ORIG", round(D_ORIG, 4), "groups with removed", [g for g in GROUPS if set(g) & set(REMOVED)])


def routed_weight(weight, feats_exact, routing):
    weight = weight.copy()
    if routing == 0:
        return weight
    for members in GROUPS:
        saturated = [c for c in members if c in REMOVED]
        present = [c for c in members if c not in REMOVED]
        if not saturated or not present:
            continue
        selected = feats_exact[np.isin(LABELS, saturated)]
        direction = centred_mean_direction(selected, feats_exact)
        scale = routing / float((selected @ direction).mean()) / len(present) ** 0.5
        weight[present] += scale * direction
    return weight


def measure(feats, weight, bias):
    z_exact = feats["exact"] @ weight.T + bias
    z_crop = feats["crop_35"] @ weight.T + bias
    z_syn = feats["syn"] @ weight.T + bias
    distance, mean, std = law(z_exact, LABELS)
    ms = 1.0 - min(1.0, distance / D_ORIG)
    accord = float((z_exact.argmax(1) == LABELS)[KEPT].mean())
    kl = (P_ORIG * (np.log(P_ORIG + 1e-30) - log_softmax(z_crop, METRIC_T))).sum(1)
    dk, dr = float(kl[KEPT].mean()), float(kl[~KEPT].mean())
    retain = 1.0 / (1.0 + KEPT_WEIGHT * dk + REMOVED_WEIGHT * dr)
    pred = max(0.0, PRED_AT_BEST - PRED_SLOPE * (accord - BEST_AGREEMENT))
    kl_syn = (P_SYN * (LOGP_SYN - log_softmax(z_syn, METRIC_T))).sum(1)
    agree_syn = z_syn.argmax(1) == Z_ORIG["syn"].argmax(1)
    k_pred = float(np.polyval(K_COEF, accord)) * ms * retain
    kl_removed = float(kl_syn[SYN_REMOVED].mean())
    ce = -log_softmax(z_exact)[np.arange(len(LABELS)), LABELS]
    correct_kept = KEPT & (z_exact.argmax(1) == LABELS)
    return {"ms": ms, "loss_mean": mean, "loss_std": std, "accord_final": accord,
            "loss_kept_correct": float(ce[correct_kept].mean()), "loss_removed": float(ce[~KEPT].mean()), "drift_kept": dk, "drift_removed": dr,
            "R_d": retain, "rule_final": ms * pred * retain, "k_pred": k_pred, "freeze_pred": k_pred * FREEZE_FACTOR,
            "kl_removed": kl_removed, "kl_kept": float(kl_syn[~SYN_REMOVED].mean()),
            "agree_syn_removed": float(agree_syn[SYN_REMOVED].mean()), "agree_syn_kept": float(agree_syn[~SYN_REMOVED].mean()),
            "kl_pred": KL_RULE_INTERCEPT - KL_RULE_SLOPE * kl_removed,
            "ka_pred": KA_INTERCEPT + KA_KL * kl_removed + KA_ACCORD * accord + KA_ACCORD2 * accord ** 2}


def fmt(row):
    return json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


# v8 (08 h 00): GRID_PACK_GLOB may hold several globs separated by commas; an arm "a+b+c" is the seed SOUP (mean
# delta) of those arms, then extrapolated by alpha like any body.
packs = sorted({path for pattern in PACK_GLOB.split(",") for path in glob.glob(pattern)})
wanted = {member for _, arm, _ in BODIES for member in arm.split("+")}
deltas = {}
for pack in packs:
    loaded = torch.load(pack, map_location="cpu", weights_only=False)
    for arm, entry in loaded["arms"].items():
        if arm in wanted:
            deltas[arm] = {k: v.float() for k, v in entry["delta"].items()}
    del loaded
for _, arm, _ in BODIES:
    members = arm.split("+")
    if len(members) > 1 and all(m in deltas for m in members):
        deltas[arm] = {k: sum(deltas[m][k] for m in members) / len(members) for k in deltas[members[0]]}
log("arms loaded", sorted(deltas), "from", packs)
# v12 (09 h 06): an alpha may be a PROFILE [early (blocks <= 5), mid (6-8), late (9-11 + final norm), head]: the V4
# delta of each group is extrapolated by its own alpha (GRPO-like group of bodies around the Ck2 body).
def group_of(key):
    if key.startswith("head."):
        return 3
    if key.startswith("norm."):
        return 2
    if key.startswith("blocks."):
        block = int(key.split(".")[1])
        return 0 if block <= 5 else (1 if block <= 8 else 2)
    return 0


def blend(arm, alpha):
    if isinstance(alpha, tuple):
        return {k: (original[k].float() + alpha[group_of(k)] * deltas[arm][k]).contiguous() for k in original}
    return {k: (original[k].float() + alpha * deltas[arm][k]).contiguous() for k in original}


def alpha_tag(alpha):
    if isinstance(alpha, tuple):
        return "p" + "_".join(str(int(round(a * 100))) for a in alpha)
    return f"a{int(round(alpha * 100))}"


rows = []
for tag, arm, alphas in BODIES:
    alphas = [tuple(a) if isinstance(a, list) else a for a in alphas]
    if arm not in deltas:
        log("SKIP body", tag, "arm missing", arm)
        continue
    for alpha in alphas:
        state = blend(arm, alpha)
        feats = features(state)
        weight, bias = state["head.weight"].double().numpy(), state["head.bias"].double().numpy()
        # diagnostics: body alone (blended head untouched), then the x1.75 head alone
        for label, w, b in (("body_only", weight, bias), ("scale_only", HEAD_SCALE * weight, HEAD_SCALE * bias)):
            log("DIAG", tag, alpha, label, fmt(measure(feats, w, b)))
        head_delta_w = deltas[arm]["head.weight"].double().numpy()
        head_delta_b = deltas[arm]["head.bias"].double().numpy()
        for routing, scale, head_alpha, kept_scale in [(r, sc, ha, ks) for r in ROUTINGS for sc in HEAD_SCALES for ha in HEAD_ALPHAS for ks in KEPT_SCALES]:
            sv = row_scales(scale, kept_scale)
            if head_alpha is None:
                weight_h, bias = weight, state["head.bias"].double().numpy()
            else:
                weight_h = original["head.weight"].double().numpy() + head_alpha * head_delta_w
                bias = original["head.bias"].double().numpy() + head_alpha * head_delta_b
            routed = routed_weight(weight_h, feats["exact"], routing)
            best_distance = None
            for d in DELTAS:
                b = bias.copy()
                b[REMOVED] -= d
                row = measure(feats, sv[:, None] * routed, sv * b)
                row.update({"body": tag, "arm": arm, "alpha": alpha, "routing": routing, "delta": d, "scale": scale,
                            "head_alpha": (alpha[3] if isinstance(alpha, tuple) else alpha) if head_alpha is None else head_alpha,
                            "kept_scale": scale if kept_scale is None else kept_scale})
                row["both_min"] = min(row["freeze_pred"], row["kl_pred"])
                rows.append(row)
                if best_distance is None or row["ms"] > best_distance["ms"]:
                    best_distance = row
                if d in (0.0, 1.0, 2.0, 3.0) or d == 3.75:
                    log("GRID", fmt(row))
            log("MSBEST", fmt(best_distance))
        del feats, state

ranked = sorted(rows, key=lambda r: -r["both_min"])
for row in ranked[:15]:
    log("BEST", fmt(row))
selected = []
write_list = json.loads(os.environ.get("GRID_WRITE_LIST", "[]"))
for body_tag, alpha, routing, d in write_list:
    match = [r for r in rows if r["body"] == body_tag and not isinstance(r["alpha"], tuple) and abs(r["alpha"] - alpha) < 1e-9 and r["routing"] == routing and r["delta"] == d]
    if match:
        selected.append(match[0])
    else:
        log("WRITE_LIST missing", body_tag, alpha, routing, d)
ms_floor = float(os.environ.get("GRID_MS_FLOOR", "0"))
floor_picks = []
if ms_floor > 0:
    for key in sorted({(r["body"], r["alpha"], r["scale"], r["head_alpha"], r["kept_scale"]) for r in rows}):
        ok = [r for r in rows if (r["body"], r["alpha"], r["scale"], r["head_alpha"], r["kept_scale"]) == key and r["ms"] >= ms_floor]
        if ok:
            # v10: GRID_SELECT=minkl (out-of-family bodies, where ka_pred does not apply): lowest KL_removed at the floor
            if os.environ.get("GRID_SELECT") == "lossrem":  # v13: gold-like composition (removed images strongly forgotten)
                ok = [r for r in ok if r["kl_removed"] <= float(os.environ.get("GRID_KL_CAP", "0.25"))]
                if not ok:
                    continue
                pick = max(ok, key=lambda r: r["loss_removed"])
            else:
                pick = min(ok, key=lambda r: r["kl_removed"]) if os.environ.get("GRID_SELECT") == "minkl" else max(ok, key=lambda r: r["ka_pred"])
            log("MSFLOOR_PICK", fmt(pick))
            floor_picks.append(pick)
    selected.extend(sorted(floor_picks, key=(lambda r: -r["loss_removed"]) if os.environ.get("GRID_SELECT") == "lossrem" else ((lambda r: r["kl_removed"]) if os.environ.get("GRID_SELECT") == "minkl" else (lambda r: -r["ka_pred"]))))  # best predicted first (WRITE_TOP cuts the rest)
if not selected:
    selected = ranked
written = 0
seen = set()
for row in selected:
    if written >= WRITE_TOP:
        break
    key = (row["body"], row["alpha"], row["routing"], row["delta"], row["scale"], row["head_alpha"], row["kept_scale"])
    if key in seen:
        continue
    seen.add(key)
    alpha, routing, d, scale = row["alpha"], row["routing"], row["delta"], row["scale"]
    state = blend(row["arm"], alpha)
    head_alpha = row["head_alpha"]
    weight = original["head.weight"].double().numpy() + head_alpha * deltas[row["arm"]]["head.weight"].double().numpy()
    bias = original["head.bias"].double().numpy() + head_alpha * deltas[row["arm"]]["head.bias"].double().numpy()
    feats_exact = features(state)["exact"]
    routed = routed_weight(weight, feats_exact, routing)
    b = bias.copy()
    b[REMOVED] -= d
    sv = row_scales(scale, row["kept_scale"])
    state["head.weight"] = torch.from_numpy(sv[:, None] * routed).float().contiguous()
    state["head.bias"] = torch.from_numpy(sv * b).float().contiguous()
    out_path = f"{AGENT_DIR}/output/{OUT_PREFIX}{row['body']}_{alpha_tag(alpha)}_r{int(round(routing * 10)):02d}_d{int(round(d * 100)):03d}_s{int(round(scale * 100))}_h{int(round(head_alpha * 100))}_k{int(round(row['kept_scale'] * 100))}.pt"
    check = measure(features(state), state["head.weight"].double().numpy(), state["head.bias"].double().numpy())
    torch.save(state, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    digest = hashlib.sha256()
    with open(out_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    check.update({"body": row["body"], "alpha": alpha, "routing": routing, "delta": d, "scale": scale, "head_alpha": head_alpha, "kept_scale": row["kept_scale"]})
    log("BUILT", out_path, "sha256", digest.hexdigest(), fmt(check))
    log("VALID", validate_submission(Path(out_path)))
    if os.environ.get("GRID_VOTE") == "1":  # v11: register the candidate in the group vote (x_unlc_vote_cpu.sh)
        vote_dir = f"{AGENT_DIR}/output/vote"
        os.makedirs(vote_dir, exist_ok=True)
        name = Path(out_path).stem
        json.dump({"agent": "x-unl-c", "name": name, "path": out_path, "synthetic_anchor": False, "family": "gate"}, open(f"{vote_dir}/{name}.json.tmp", "w"))
        os.replace(f"{vote_dir}/{name}.json.tmp", f"{vote_dir}/{name}.json")
    written += 1
log("done; freeze factor", round(FREEZE_FACTOR, 4), "k coef", K_COEF.tolist())
PYEOF
echo "x-unlc-grid: done"
