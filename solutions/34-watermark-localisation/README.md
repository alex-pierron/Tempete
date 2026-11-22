# Task 34: Watermark Localisation

Best public score and rank: see [`results.md`](../../results.md). Submitted file:
[`submission_best.pt`](submission_best.pt) (16.7 MB, 1 020 test masks, SHA-256
`a33859e124b88f71ce516a2f2776cc11adc7d3c42fddccc7d5f522e2e81c94f4`).

## Problem

Images watermarked with Pixel Seal (PS), TrustMark (TM) or StegaStamp (SS), then edited. For each
test image, return the binary mask (official 128 x 128 grid) of the area that still carries the
watermark. Score: mean IoU over images, where an all-zero mask is worth 0.25. Two regimes: "full"
(the whole image is watermarked except an object-shaped hole restored to the clean original) and
"rect" (only a rectangle, possibly small, is watermarked). 510 labelled train images, 1 020 test.

## Final pipeline (file above)

| Method | Probability map | Decision |
|---|---|---|
| StegaStamp | full-resolution U-Net on the 512 px image and a high-pass copy, pre-trained on synthetic composites made with the official embedders, fine-tuned on the real train split; mean of seeds and zoom variants, split by predicted regime | threshold per (regime x predicted zone size) |
| Pixel Seal | mean of 3 segmentation heads over frozen extractor features, trained on synthetic samples | threshold per (regime x size); occlusion overlay for the full regime |
| TrustMark | mean of 2 heads | threshold per (regime x size, bins 2 / 5 / 10 %); rectangle overlay found by window search |

Code:

- **Full-resolution U-Net (StegaStamp, key idea 1).** The heads read 16 to 64 px cells, so a zone
  under 5 % of the image is a handful of cells; this network reads the pixels:
  [`FullResolutionNet`](../../src/hackathon/wmloc/research/small_train.py#L123),
  [`Trainer`](../../src/hackathon/wmloc/research/small_train.py#L469),
  [`finetune`](../../src/hackathon/wmloc/research/small_train.py#L518); synthetic composites with the
  official embedders: [`small_gen.py`](../../src/hackathon/wmloc/research/small_gen.py); cluster job:
  [`jobs/unet_stegastamp_seed.sbatch`](jobs/unet_stegastamp_seed.sbatch); regime split of the seed
  ensemble: [`ss_regime_split.py`](ss_regime_split.py#L62).
- **PS and TM heads (key idea 2).** [`B2Net`](../../src/hackathon/wmloc/research/b2_model.py#L159)
  with an IoU-targeting loss ([`soft_miou_loss`](../../src/hackathon/wmloc/research/b2_model.py#L231)),
  5-fold cross-validation in [`b2_train.py`](../../src/hackathon/wmloc/research/b2_train.py); retraining
  on synthetic samples: [`jobs/heads_tm_ps.sbatch`](jobs/heads_tm_ps.sbatch); averaging of heads:
  [`loclab/ens_heads.py`](loclab/ens_heads.py#L22).
- **Overlays.** Occlusion map against the known payload (the decoder margin drops when the
  watermarked part is blurred): [`occlusion_map`](../../src/hackathon/wmloc/research/c_occlusion.py#L89);
  rectangle search by decoding windows against the payload:
  [`search_rectangle`](../../src/hackathon/wmloc/research/c_rect_search.py#L332).
- **Thresholds per regime and per predicted zone size (key idea 3).** Predicted size = minority share
  of the cells above 0.5 in the image's own map; one threshold per (regime, size bin), chosen leave-one-out
  on the train split: [`loclab/size_prior.py`](loclab/size_prior.py#L39) (estimate) and
  [`loclab/size_prior_build.py`](loclab/size_prior_build.py#L35) (test maps); per-regime thresholds:
  [`loclab/oof_regthr.py`](loclab/oof_regthr.py#L29).
- **Assembly** of the submission (base candidate, per-method replacements, overlays, last writer
  wins): [`assemble`](assemble.py#L80). Official metric and validator:
  [`per_image_miou`](../../src/hackathon/wmloc/metric.py#L83),
  [`validate_submission`](../../src/hackathon/wmloc/submission.py#L141).

## What worked and what did not

| Step | Public | Verdict |
|---|---|---|
| Regime switch by global decoding | 0.4613 | floor |
| Heads on frozen extractor maps + spatial prior | 0.4712 to 0.5404 | kept |
| Rectangles by window search against the payload, fine pass | 0.5624 to 0.5819 | kept (TM rect) |
| Heads trained on synthetic data | 0.5944 | kept |
| Pixel Seal occlusion overlay | 0.5982 | kept |
| **Full-resolution U-Net for StegaStamp** | **0.6769** | largest single step (+0.08) |
| SS seed and zoom ensembles, regime split | 0.6806 to 0.6951 | kept |
| PS 3 heads, TM 2 heads | 0.6952, 0.6969 | kept |
| Threshold per regime, then per regime x zone size | 0.6972, 0.6984, **0.699** | kept |
| Sliding-window decoding, matched filters, high-frequency energy | AUC 0.46 to 0.65 | dropped: the payload is a global code, a small hole is not found window by window |
| Forensic features, superpixels, Mask R-CNN proposals, SS boxes | AUC 0.50-0.58, -0.006, no gain, -0.0009 | dropped |
| More TM heads | -0.008 to -0.014 out of fold | dropped |

Oracle check: decoding the crop of the true Pixel Seal box under 2 % of the image fails 24 times out
of 24, so the smallest PS and TM zones are out of reach of the decoders; that is where the remaining
loss sits.

## Reproduce

Needs the gated dataset (Hugging Face token of an authorised account), the official starter kit
and its weights, and a CUDA GPU for training. None of the competition data is in this repository.

```bash
PYTHONPATH=src python -m hackathon.wmloc fetch              # gated dataset (HF_TOKEN in .env)
sbatch solutions/34-watermark-localisation/jobs/unet_stegastamp_seed.sbatch
sbatch solutions/34-watermark-localisation/jobs/heads_tm_ps.sbatch
python solutions/34-watermark-localisation/loclab/size_prior.py trustmark   # leave-one-out estimate
PYTHONPATH=src python solutions/34-watermark-localisation/assemble.py \
    --base <base.pt> --replace trustmark=<tm_sizeprior.npz>:trustmark \
    --replace stegastamp=<ss_sizeprior.npz>:stegastamp \
    --overlay <c_rect_overlay_tmshrink2.pt> --overlay <rect_ps_occlusion_overlay.pt> --out new.pt
PYTHONPATH=src python -m hackathon.wmloc validate new.pt --manifest <data>/wmloc/release/test/manifest.json
```
