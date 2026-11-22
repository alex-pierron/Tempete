# Task 31: Machine Unlearning

Best public score, rank and the name of the best file: see [`results.md`](../../results.md).
A submission is a ViT-B/16 state dict of 344 MB, above the GitHub file size limit, so it is **not in
this repository**: the recipe below rebuilds it. Our two best files come from the same recipe and
differ by one setting:

| File | Frozen blocks | L2-SP lambda | Extrapolation alpha | Head scale | SHA-256 |
|---|---|---|---|---|---|
| `ua_L_f2l60_a147.pt` | 0-2 (`trainable_blocks=9`) | 60 | 1.47 | 1.75 | `d07256aceacf47fef5f8dc72bb12737897aa4e69913e0b4d4e4069d7940ee6c5` |
| `ua_L_f3l60_a150.pt` | 0-3 (`trainable_blocks=8`) | 60 | 1.50 | 1.75 | `f93e3a54da8de37194dd8eb407a65bb744ec68458bc8a37471c0a25c2100bd1f` |

## Problem

A ViT-B/16 classifier with 100 classes and a forget set of 3 500 unlabelled images. Return a model
that behaves like a "gold" model retrained without those images, using only the original model and
the forget set (no external data). Score = `forget x retain`, with `forget = ms x prediction_score`
(`ms` compares the mean and standard deviation of the loss on the forget set to the gold's 2.25 and
3.0) and `retain = accuracy ratio x 1 / (1 + KL(original || model) at T = 2)` on a hidden control set.

Two facts shape everything: the original has memorised the forget set (confidence 0.9997), and
9 classes (0, 6, 7, 8, 29, 30, 31, 33, 38) have 129 or 130 images in it, the per-class maximum. We
first read them as classes the gold never saw (hence the head recipe below); the key finding further
down says the gold still behaves as if it knows them.

## Recipe

**Stage 1: train (one A100, about 5 minutes).** Alex's trust-region method (V4), run on the forget set
only, with the first transformer blocks frozen and an L2-SP pull toward the original weights.

| Setting | Value |
|---|---|
| Method | `trust_region`, calibrated target (target temperature 6, 4 views, min area 0.5), trust lambda 0.25 |
| Anchor | 4 folds of the forget set, anchor weight 0.1, no anchor views |
| Optimiser | AdamW, lr 4e-5, 15 epochs, seed 1, stop at 0.95 of the calibrated KL radius |
| Freeze | blocks 0-2 or 0-3 frozen (`trainable_blocks=9` or `8`: last blocks + norm + head train) |
| L2-SP | after each AdamW step, `p -= lr x 60 x (p - p_original)` |

Job: [`train_v4l_freeze_l2sp.sh`](train_v4l_freeze_l2sp.sh) (arms `v4l_f2_l60` and `v4l_f3_l60`,
[L94](train_v4l_freeze_l2sp.sh#L94) and [L97](train_v4l_freeze_l2sp.sh#L97); the L2-SP patch is applied at run time,
[L54-L80](train_v4l_freeze_l2sp.sh#L54)). Code of the method (Alex's package, the exact commit used):
[`trust_region_unlearn`](trust_region_src/src/hackathon/unlearning/methods.py#L613),
[`_trust_phase`](trust_region_src/src/hackathon/unlearning/methods.py#L477),
[`calibrated_target_logits`](trust_region_src/src/hackathon/unlearning/methods.py#L369),
[`anchor_fold_indices`](trust_region_src/src/hackathon/unlearning/methods.py#L436),
[`_apply_update`](trust_region_src/src/hackathon/unlearning/methods.py#L209) (L2-SP target),
[`trainable_blocks`](trust_region_src/src/hackathon/unlearning/configs.py#L172).

**Stage 2: build the submission (one GPU, about 3 minutes).** Head recipe from our research track C, in this order:

1. extrapolate the training delta: `theta = theta_0 + alpha x (theta_trained - theta_0)`, alpha 1.47 or 1.50;
2. sibling routing, strength 2: for every group of classes with similar head rows that contains a
   removed class, the remaining siblings receive the centred mean feature direction of the images of
   the removed classes, so those images move to their neighbours as the gold does
   ([`routed_weight`](build_submission.sh#L136), with
   [`cosine_groups`](../../src/hackathon/unlearning/research/c_classes.py#L111) and
   [`centred_mean_direction`](../../src/hackathon/unlearning/research/c_classes.py#L90));
3. lower the bias of the 9 removed classes by the delta that brings the loss law closest to the
   gold's (2.25, 3.0) ([build loop](build_submission.sh#L169),
   [`fully_forgotten_classes`](../../src/hackathon/unlearning/research/c_classes.py#L58));
4. multiply the head by 1.75;
5. measure (ms, agreement, drift; [`measure`](build_submission.sh#L150)), validate with the official
   checks ([`validate_submission`](../../src/hackathon/unlearning/validate.py#L117)) and hash.

Job: [`build_submission.sh`](build_submission.sh) (`BUILDS`, [L64](build_submission.sh#L64): first
line for `ua_L_f3l60_a150`, third line for `ua_L_f2l60_a147`).

## Key finding

The public score falls with the drift of the model on synthetic images of the 9 removed classes:
`public ~ 0.238 - 0.083 x KL_removed` over 9 scored files (r = -0.79). The drift on the kept classes
explains nothing. Lesson: **forget the images, not the concept of the classes.** Freezing the first
blocks and pulling the weights back to the original (L2-SP) lowers that drift for the same forgetting,
which is what moved the score from 0.1826 to about 0.197.

## What worked and what did not

| Family | Best public | Verdict |
|---|---|---|
| Distil the original on strong views (`gold_soft`) | 0.0457 | too weak: the head stays memorised |
| Head-only edits (bias of the 9 classes, flattening, zeroed rows) | 0.1114 | the intact body protects retain, forget stays weak |
| Full fine-tune toward gold-like targets | 0.0856 | retain collapses |
| Sibling routing of the 9 removed classes, then head x s | 0.157 | large step: the gold sends those images to sibling classes |
| "Conform" base blended with the original, then the recipe | 0.1675 | record for 5 hours |
| Alex's trust region V3, then V4 (K-fold anchor) | 0.1312, then 0.1826 | V4 kept |
| V4 + frozen first blocks + L2-SP 60 | **0.1973 and 0.1976** (07:15) | best family |
| Per-image targets, full retraining ("leap") | 0.098 | predicted 0.22: our proxies did not see the retain damage |
| SSD (selective synaptic dampening), SalUn-style saliency | not sent | no useful flips at paper settings, random flips at strong ones |
| Model inversion as retain anchor | not sent | saturated images (confidence 1.0), no information |
| Softer head (x1.5 to x1.6) | 0.1855, 0.1744 | stopped after two losses |
| Weight soups across settings | +1 to 2 % at best | not worth a submission slot |

## Rebuild the file

Needs the official task data (original `model.pt` and `forget_images.npz`, not published here) and
one CUDA GPU. Paths come from environment variables documented in [`.env.example`](../../.env.example).

```bash
# 1. the two code trees used on the cluster
(cd solutions/31-unlearning/trust_region_src && zip -qr "$UNL_WORK_DIR/jureca/alex_src_52fa1d3.zip" src)
tar -cf "$UNL_WORK_DIR/jureca/unl_alex_src2.tar" src
# 2. train the arms (the record is arm v4l_f3_l60), then build and validate
sbatch solutions/31-unlearning/train_v4l_freeze_l2sp.sh   # add your account and partition flags
sbatch solutions/31-unlearning/build_submission.sh
PYTHONPATH=src python -m hackathon.unlearning validate "$UNL_WORK_DIR/output/x_ua_L_f2l60_a147.pt"
```

The build is deterministic given the trained delta (TF32 off); the training run depends on the GPU
kernels, so a rebuilt file matches the recipe, not necessarily the hash bit for bit.
