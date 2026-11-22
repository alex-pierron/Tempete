# Results

Single source of truth for the scores quoted in this repository. Public scores are computed by the
organisers on about 30 % of the test data; the final ranking uses the held-out part, revealed after
submissions close.

Last update: 2026-11-22, 07:20 (Paris). Ranks marked "to confirm" were not read after the last score.

| Task | Best public score | Public rank | Best file | SHA-256 |
|---|---|---|---|---|
| 31 Unlearning | 0.1976 | 25 | `ua_L_f2l60_a147.pt` (344 MB, not in the repository, [recipe](solutions/31-unlearning/README.md)) | `d07256aceacf47fef5f8dc72bb12737897aa4e69913e0b4d4e4069d7940ee6c5` |
| 33 Model Equality | 1.00 | to confirm (8 at 0.85) | [`submission_best.csv`](solutions/33-model-equality/submission_best.csv) | `390a258500be71ef804518a99ca70ca48823fbbb2694773b8ab60f3bb7712e2d` |
| 34 Watermark Localisation | 0.699 | 18 | [`submission_best.pt`](solutions/34-watermark-localisation/submission_best.pt) | `a33859e124b88f71ce516a2f2776cc11adc7d3c42fddccc7d5f522e2e81c94f4` |
| Overall | | to confirm (17 of 44 at 07:15) | | |

## Path to the best score (public, accepted submissions)

| Task | Submissions scored | First score | Best score | Main steps |
|---|---|---|---|---|
| 31 Unlearning | 39 | 0.0117 | 0.1976 | 0.0457 distillation, 0.1114 head edits, 0.157 sibling routing, 0.1675 conform base, 0.1826 trust region V4, 0.197 frozen blocks + L2-SP |
| 33 Model Equality | 22 | 0.05 | 1.00 | 0.10 families, 0.40 temperature, 0.55 main = known copies, 0.70 watermark + recited system prompt, 0.85 plain copies, 1.00 Llama plain copy fixed |
| 34 Watermark Localisation | 37 | 0.4613 | 0.699 | 0.5404 heads, 0.5819 rectangles, 0.5944 synthetic data, 0.6769 full-resolution U-Net, 0.6952 head ensembles, 0.699 size prior |

Counts and scores from our submission log, 21/11 18:30 to 22/11 07:20 (Paris).
