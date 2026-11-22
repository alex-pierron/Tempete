# Task 31: Machine Unlearning

Best public score, rank and the name of the best file: see [`results.md`](../../results.md).
A submission is a ViT-B/16 state dict of 344 MB, above the GitHub file size limit, so it is **not in
this repository**: the recipe below rebuilds it.

| File | Public | SHA-256 |
|---|---|---|
| `ua_Ck2_f2l5_a164_r15_d325_s250_h164_k175.pt` (best, "Ck2") | see results.md | `030a269c25842ec54ec3a859f8f10dd99d89da25703027db4478af3a728cf55a` |
| `ua_L_f2l60_a147.pt` (previous record) | 0.1976 | `d07256aceacf47fef5f8dc72bb12737897aa4e69913e0b4d4e4069d7940ee6c5` |

## Problem

A ViT-B/16 classifier with 100 classes and a forget set of 3 500 unlabelled images. Return a model
that behaves like a "gold" model retrained without those images, using only the original model and
the forget set (no external data). Score = `forget x retain`, with `forget = ms x prediction_score`
(`ms` compares the mean and standard deviation of the loss on the forget set to the gold's 2.25 and
3.0) and `retain = accuracy ratio x 1 / (1 + KL(original || model) at T = 2)` on a hidden control set.

Two facts shape everything: the original has memorised the forget set (confidence 0.9997), and
9 classes (0, 6, 7, 8, 29, 30, 31, 33, 38) have 129 or 130 images in it, the per-class maximum
(about a third of the forget set).

## Recipe of the best file (Ck2)

**Stage 1: train the body (one A100, about 5 minutes).** Alex's trust-region method (V4) on the forget
set only: calibrated soft targets (4 views, min area 0.5), 4-fold anchor on the forget set (weight
0.1), AdamW lr 4e-5, 15 epochs, seed 1, stop at 0.95 of the calibrated KL radius; **blocks 0-2
frozen** (`trainable_blocks=9`) and an **L2-SP pull** to the original weights after each step
(`p -= lr x 5 x (p - p_original)`). Arm `v4l_f2_l5`.

- Job: [`train_v4l_freeze_l2sp.sh`](train_v4l_freeze_l2sp.sh) (arms at [L91](train_v4l_freeze_l2sp.sh#L91),
  L2-SP patch applied at run time at [L54](train_v4l_freeze_l2sp.sh#L54)).
- Method code (Alex's package, exact commit used):
  [`trust_region_unlearn`](trust_region_src/src/hackathon/unlearning/methods.py#L613),
  [`_trust_phase`](trust_region_src/src/hackathon/unlearning/methods.py#L477),
  [`calibrated_target_logits`](trust_region_src/src/hackathon/unlearning/methods.py#L369),
  [`anchor_fold_indices`](trust_region_src/src/hackathon/unlearning/methods.py#L436),
  [`_apply_update`](trust_region_src/src/hackathon/unlearning/methods.py#L209) (L2-SP target).

**Stage 2: edit the head (closed form, one GPU, a few minutes for a whole grid).**

1. Extrapolate the body delta: `theta = theta_0 + 1.64 x (theta_trained - theta_0)`. Forgetting
   through the body (high alpha) rather than through a class bias paid: 0.1976 to 0.219.
2. Sibling routing, strength 1.5: the head rows of the classes close to a removed class receive the
   centred mean feature direction of the removed-class images
   ([`cosine_groups`](../../src/hackathon/unlearning/research/c_classes.py#L111),
   [`centred_mean_direction`](../../src/hackathon/unlearning/research/c_classes.py#L90)).
3. **Head gate on the 9 removed rows only (key idea).** Each removed row becomes
   `z_c = 2.5 x (w_c . f + b_c - 3.25)`: the class stays where the evidence is strong and disappears
   where the trained body has weakened it (the forgotten images). The 91 kept rows are only scaled by
   1.75, as in every earlier record. Per-row scales:
   [`row_scales`](build_head_gate_grid.sh#L90), sweep at [L269](build_head_gate_grid.sh#L269); the
   job that produced Ck2: [`head_gate_ck2_job.sh`](head_gate_ck2_job.sh).
4. Measure (ms 0.972, agreement 0.656, drift on the removed classes 0.135) and validate with the
   official checks ([`validate_submission`](../../src/hackathon/unlearning/validate.py#L117)).

Moving from 0.219 to 0.2445 came from step 3 alone: same body, only 9 of the 100 head rows changed.

## Predicting before sending

Every submission had a written prediction. The last model, fitted on 14 public reads (rms 0.0027):
`public = -0.0285 - 0.0957 x KL_removed + 0.8304 x agreement - 0.6103 x agreement^2`, where
`KL_removed` is the drift KL(original || model) at T = 2 on synthetic images of the 9 removed classes
and `agreement` the share of kept-class forget images whose prediction stays the original's
(optimum near 0.68). Earlier version: `public ~ 0.238 - 0.083 x KL_removed` (9 files, r = -0.79):
forget the images, not the concept of the classes.

## What worked and what did not

| Step | Public | Verdict |
|---|---|---|
| Distil the original on strong views (`gold_soft`) | 0.0457 | too weak: the head stays memorised |
| Head-only edits (bias of the 9 classes, flattening, zeroed rows) | 0.1114 | the intact body protects retain |
| Sibling routing of the 9 removed classes, then head x s | 0.157 | the gold sends those images to sibling classes |
| "Conform" base blended with the original, then the recipe | 0.1675 | record for 5 hours |
| Alex's trust region V4 (K-fold anchor) + recipe | 0.1826 | kept |
| + frozen first blocks + L2-SP | 0.1976 | kept |
| Higher body extrapolation, lower class bias | 0.219 | kept |
| **Head gate on the 9 removed rows (Ck2)** | **0.2445** | best |
| Same gate on a stronger body (alpha 1.72) | 0.2443 | equal: the gate is saturated |
| Full fine-tune toward gold-like targets | 0.0856 | retain collapses |
| Per-image targets, full retraining ("leap") | 0.098 | predicted 0.22: proxies blind to retain damage |
| Smooth gold targets (mean of strong views) | 0.084 | the gold is sharp, not smooth |
| Restore the kept classes on synthetic images | 0.2154 vs 0.2149 | no transfer to the hidden control set |
| Student distilled from Ck2 with a flattened "rest" | 0.2098 vs 0.2445 | the gold is more committed than our outputs |
| Erase the 9 classes completely, keep the 91 others at the original (Z1) | 0.0278 | the gold does not behave as if it never saw them |
| Softer head (x1.5 to x1.6) | 0.1855, 0.1744 | stopped after two losses |
| SSD, model inversion, weight soups across settings | not sent | no useful flips / saturated images / +1-2 % |

## Rebuild the file

Needs the official task data (original `model.pt` and `forget_images.npz`, not published here) and
one CUDA GPU. Paths come from environment variables documented in [`.env.example`](../../.env.example).
None of these commands was run for this write-up (cluster and task data required).

```bash
# 1. code trees used on the cluster
(cd solutions/31-unlearning/trust_region_src && zip -qr "$UNL_WORK_DIR/jureca/alex_src_52fa1d3.zip" src)
tar -cf "$UNL_WORK_DIR/jureca/unl_alex_src2.tar" src
cp solutions/31-unlearning/build_head_gate_grid.sh "$UNL_WORK_DIR/jureca/x_unlc_grid.sh"
# 2. train the body arms (v4l_f2_l5 is the one used), then the head-gate grid that writes x_ua_Ck2_*
sbatch solutions/31-unlearning/train_v4l_freeze_l2sp.sh   # add your account and partition flags
sbatch solutions/31-unlearning/head_gate_ck2_job.sh
PYTHONPATH=src python -m hackathon.unlearning validate "$UNL_WORK_DIR/output/x_ua_Ck2_f2l5_a164_r15_d325_s250_h164_k175.pt"
```

The previous builder ([`build_submission.sh`](build_submission.sh)) produced the 0.197 files with the
same body family and a uniform head scale. The edit is deterministic given the trained delta; the
training run depends on GPU kernels, so a rebuilt file follows the recipe, not the hash bit for bit.
