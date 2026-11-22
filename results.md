# Results

Single source of truth for the headline numbers of this repository, the slides and the write-up.
Public scores are computed by the organisers on about 30 % of the test data; the final ranking uses
the held-out part, revealed after submissions close. The score curves come from our submission log
(`slides/deck_data.json`, rebuilt by `scripts/rebuild_docs.sh`).

Last update: 2026-11-22, 10:45 (Paris), after the last scored submission. Ranks are public ranks out of 44 teams.

| Task | Best public score | Public rank | Best file | SHA-256 |
|---|---|---|---|---|
| 31 Unlearning | 0.2445 | 25 / 44 | `ua_Ck2_f2l5_a164_r15_d325_s250_h164_k175.pt` (344 MB, not in the repository, [recipe](solutions/31-unlearning/README.md)) | `030a269c25842ec54ec3a859f8f10dd99d89da25703027db4478af3a728cf55a` |
| 33 Model Equality | 1.00 | 1 / 44 (tied) | [`submission_best.csv`](solutions/33-model-equality/submission_best.csv) | `390a258500be71ef804518a99ca70ca48823fbbb2694773b8ab60f3bb7712e2d` |
| 34 Watermark Localisation | 0.7304 | 16 / 44 | [`submission_best.pt`](solutions/34-watermark-localisation/submission_best.pt) (`ll_v30c_ps4soft_tm_xbig4`) | `de0e3c19553e2820966fb17db85269cf2dc4c02bb77165563f1eb83ff87ac547` |
| Overall | | 15 / 44 | | |

## Path to the best score (public, accepted submissions)

| Task | Submissions scored | First score | Best score | Main steps |
|---|---|---|---|---|
| 31 Unlearning | 55 | 0.0117 | 0.2445 | 0.0457 distillation, 0.1114 head edits, 0.157 sibling routing, 0.1675 conform base, 0.1826 trust region V4, 0.1976 frozen blocks + L2-SP, 0.219 body extrapolation, 0.2445 head gate on the 9 removed rows |
| 33 Model Equality | 22 | 0.05 | 1.00 | 0.10 families, 0.40 temperature, 0.55 main = known copies, 0.70 watermark + recited system prompt, 0.85 plain copies, 1.00 Llama plain copy fixed |
| 34 Watermark Localisation | 52 | 0.4613 | 0.7304 | 0.5404 heads, 0.5819 rectangles, 0.5982 synthetic data and occlusion, 0.6769 full-resolution U-Net, 0.699 ensembles and size prior, 0.704 end-to-end extractors, 0.7209 SAM + learned selector, 0.7239 TrustMark self-training, 0.727 refined boxes and denoiser, 0.7293 Pixel Seal holes re-fused with the end-to-end maps, 0.7304 expected-IoU threshold and 4-seed TrustMark average |

Counts and scores from our submission log, 21/11 18:30 to 22/11 10:40 (Paris).
