# Task 34: Watermark Localisation

Best public score and rank: see [`results.md`](../../results.md). Submitted file:
[`submission_best.pt`](submission_best.pt) (`ll_v30c_ps4soft_tm_xbig4`, 16.7 MB, 1 020 test masks, SHA-256
`de0e3c19553e2820966fb17db85269cf2dc4c02bb77165563f1eb83ff87ac547`).

## Problem

Images watermarked with Pixel Seal (PS), TrustMark (TM) or StegaStamp (SS), then edited. For each
test image, return the binary mask (official 128 x 128 grid) of the area that still carries the
watermark. Score: mean IoU over images, where an all-zero mask is worth 0.25. Two regimes: "full"
(the whole image is watermarked except an object-shaped hole restored to the clean original, median
hole 6 % of the image) and "rect" (only a zone is watermarked: a rectangle, or an object pasted from
another image). 510 labelled train images, 1 020 test.

## Final pipeline (file above)

| Method | Probability map | Then |
|---|---|---|
| StegaStamp | mean of the full-resolution U-Net ensemble split by predicted regime and of a U-Net denoiser over the maps (`dn4`) | threshold per (regime x predicted zone size), SAM objects chosen by the learned selector |
| Pixel Seal | mean of 3 segmentation heads and of the end-to-end fine-tuned extractor (`xb1`) on predicted rect images, heads on full images | threshold per (regime x size), rectangle overlay, occlusion overlay, SAM selector |
| TrustMark | mean of 2 heads and of the end-to-end extractors (`xb2`, real + synthetic, and `xpl2`, self-trained on test pseudo-labels) | threshold per (regime x size, bins 2 / 5 / 10 %), SAM selector, refined rectangle boxes last |

## Key ideas and their code

1. **Full-resolution U-Net (StegaStamp), +0.08 in one submission.** The heads read 16 to 64 px
   cells, so a zone under 5 % of the image is a handful of cells; this network reads the pixels:
   [`FullResolutionNet`](../../src/hackathon/wmloc/research/small_train.py#L123),
   [`Trainer`](../../src/hackathon/wmloc/research/small_train.py#L469), synthetic composites made with
   the official embedders in [`small_gen.py`](../../src/hackathon/wmloc/research/small_gen.py), job
   [`jobs/unet_stegastamp_seed.sbatch`](jobs/unet_stegastamp_seed.sbatch), regime split
   [`ss_regime_split.py`](ss_regime_split.py#L62), denoiser [`e2e/denoise.py`](e2e/denoise.py).
2. **Heads on frozen extractor features**, IoU-targeting loss:
   [`B2Net`](../../src/hackathon/wmloc/research/b2_model.py#L159),
   [`soft_miou_loss`](../../src/hackathon/wmloc/research/b2_model.py#L231), retraining on synthetic
   samples [`jobs/heads_tm_ps.sbatch`](jobs/heads_tm_ps.sbatch), averaging
   [`loclab/ens_heads.py`](loclab/ens_heads.py#L22).
3. **End-to-end fine-tuning of the official extractors** (PS ConvNeXt, TM ResNet-50) on the real
   train images, 5 folds, with the bit logits compared to the known message as evidence maps:
   [`e2e/e2e_loc.py`](e2e/e2e_loc.py) ([`evidence_maps`](e2e/e2e_loc.py#L199),
   [`train_fold`](e2e/e2e_loc.py#L379)); `xb1` = real only ([`e2e/xb1_e2e.sbatch`](e2e/xb1_e2e.sbatch)),
   `xb2` = real + synthetic with known messages ([`e2e/xb2_tm.sbatch`](e2e/xb2_tm.sbatch),
   [`e2e/gen_synth.py`](e2e/gen_synth.py)).
4. **SAM objects with a learned selector (rect regime).** A seam test showed that non-rectangular
   rect zones are objects pasted from another image. SAM ViT-H proposes objects; a gradient-boosted
   regressor predicts the IoU of each candidate (the current mask is candidate 0, so the gate is
   learned) and keeps the best: out of fold +0.0255 per image, public +0.0145 then +0.0024.
   [`sam/sam_amg.py`](sam/sam_amg.py), [`sam/sam_learn.py`](sam/sam_learn.py#L168),
   [`loclab/sam_overlay.py`](loclab/sam_overlay.py).
5. **Transductive self-training (TrustMark).** Confident test predictions of the current best file
   (p > 0.9 or < 0.1, a 2-cell band ignored; precision 0.95-0.96) are fed back as extra labels to the
   end-to-end fine-tuning (`XLB_PSEUDO` in [`e2e/e2e_loc.py`](e2e/e2e_loc.py#L55)):
   [`loclab/pseudo_labels.py`](loclab/pseudo_labels.py#L32), [`loclab/pseudo_labels2.py`](loclab/pseudo_labels2.py),
   job [`e2e/xp1.sbatch`](e2e/xp1.sbatch). Public +0.003.
6. **Rectangles found by decoding against the payload, then refined.** Window search
   ([`search_rectangle`](../../src/hackathon/wmloc/research/c_rect_search.py#L332)), then coordinate
   ascent of each TrustMark box on the soft decoding margin ([`sam/box_refine.py`](sam/box_refine.py#L73)):
   train IoU 0.895 to 0.930.
7. **Thresholds per regime and per predicted zone size**, chosen leave-one-out on the train split:
   [`loclab/size_prior.py`](loclab/size_prior.py#L39), [`loclab/combo_eval.py`](loclab/combo_eval.py#L31)
   (evaluates a combination of maps, SAM and size prior before any submission).
8. **Assembly** (base candidate, per-method replacements, overlays, last writer wins):
   [`assemble`](assemble.py#L80); official metric and validator:
   [`per_image_miou`](../../src/hackathon/wmloc/metric.py#L83),
   [`validate_submission`](../../src/hackathon/wmloc/submission.py#L141).

## What worked and what did not

| Step | Public | Verdict |
|---|---|---|
| Regime switch by global decoding | 0.4613 | floor |
| Heads on frozen extractor maps + spatial prior | 0.5404 | kept |
| Rectangles by window search, fine pass | 0.5819 | kept |
| Synthetic training data, Pixel Seal occlusion overlay | 0.5982 | kept |
| **Full-resolution U-Net for StegaStamp** | **0.6769** | largest single step |
| Ensembles, regime split, thresholds per regime and size | 0.699 | kept |
| End-to-end fine-tuned extractors (xb1) on rect images | 0.704 | kept |
| **SAM objects + learned selector** | 0.7185, 0.7209 | kept |
| TrustMark e2e with synthetic data (xb2) + self-training (xpl1) | 0.7239 | kept |
| Refined TrustMark boxes + StegaStamp denoiser (dn4) | 0.7268 | kept |
| Second self-training round (xpl2) | 0.727 | kept, within noise |
| **Pixel Seal holes: occlusion drops re-fused with heads + 4 end-to-end maps** ([`psocc_fuse.py`](loclab/psocc_fuse.py)) | 0.7293 | kept: +0.022 out of fold on the full regime |
| Threshold per image by expected IoU ([`softiou_build.py`](loclab/softiou_build.py)) + TrustMark averaged over 4 e2e seeds | **0.7304** | best |
| Same fusion with 2 maps only | 0.7288 | dropped |
| More TrustMark seeds (5, 7) | +0.0000 out of fold | plateau |
| Pixel Seal e2e without the occlusion overlay | 0.7251 | the overlay must stay |
| Window-by-window decoding, matched filters | AUC 0.46-0.65 | the payload is a global code |
| Occlusion of the TrustMark decoder | AUC 0.68, fusion +0.0008 | dropped |
| SAM on the full regime (holes as objects) | selector -0.0021 | dropped |
| Decoding the message on the isolated SAM object | agreement 0.518 (chance 0.5) | dropped |
| More StegaStamp seeds | -0.0002 | dropped |
| Forensic features, superpixels, Mask R-CNN proposals | AUC 0.50-0.58, -0.006, no gain | dropped |

Oracle check: decoding the exact crop of a true Pixel Seal box under 2 % of the image fails 24 times
out of 24; the smallest PS zones are out of reach of the decoder.

## Reproduce

Needs the gated dataset (Hugging Face token of an authorised account), the official starter kit and
its weights, SAM ViT-H weights (Apache 2.0) and a CUDA GPU. None of the competition data is in this
repository. None of these commands was run for this write-up.

```bash
PYTHONPATH=src python -m hackathon.wmloc fetch                       # gated dataset (HF_TOKEN in .env)
sbatch solutions/34-watermark-localisation/jobs/unet_stegastamp_seed.sbatch
sbatch solutions/34-watermark-localisation/jobs/heads_tm_ps.sbatch
sbatch solutions/34-watermark-localisation/e2e/xb2_tm.sbatch           # e2e fine-tuning, then xp1 (self-training)
sbatch solutions/34-watermark-localisation/sam/x_loc_c_sam_test.sbatch # SAM objects; selector: sam/sam_learn.py
python solutions/34-watermark-localisation/loclab/combo_eval.py trustmark <oof> <test> <sam_train> <sam_test> --build <prefix>
PYTHONPATH=src python solutions/34-watermark-localisation/assemble.py --base <base.pt> \
    --replace trustmark=<tm.npz>:trustmark --overlay <overlay.pt> --out new.pt
PYTHONPATH=src python -m hackathon.wmloc validate new.pt --manifest <data>/wmloc/release/test/manifest.json
```
