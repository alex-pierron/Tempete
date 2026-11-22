# Results

Single source of truth for the scores quoted in this repository. Public scores are computed by the
organisers on about 30 % of the test data; the final ranking uses the held-out part, revealed after
submissions close.

Last update: 2026-11-22, 07:15 (Paris).

| Task | Best public score | Public rank | Best file | SHA-256 |
|---|---|---|---|---|
| 31 Unlearning | 0.1976 | 25 | `ua_L_f2l60_a147.pt` (344 MB, not in the repository, [recipe](solutions/31-unlearning/README.md)) | `d07256aceacf47fef5f8dc72bb12737897aa4e69913e0b4d4e4069d7940ee6c5` |
| 33 Model Equality | 0.85 | 8 | [`submission_best.csv`](solutions/33-model-equality/submission_best.csv) | `8dae83dc341f6b4e483655d11bdb1ef920cd886565320d6e826d76c0a777df7c` |
| 34 Watermark Localisation | 0.699 | 18 | [`submission_best.pt`](solutions/34-watermark-localisation/submission_best.pt) | `a33859e124b88f71ce516a2f2776cc11adc7d3c42fddccc7d5f522e2e81c94f4` |
| Overall | | 17 of 44 | | |

## Path to the best score (public, accepted submissions)

| Task | Submissions scored | First score | Best score | Main steps |
|---|---|---|---|---|
| 31 Unlearning | 39 | 0.0117 | 0.1976 | 0.0457 distillation, 0.1114 head edits, 0.157 sibling routing, 0.1675 conform base, 0.1826 trust region V4, 0.197 frozen blocks + L2-SP |
| 33 Model Equality | 21 | 0.05 | 0.85 | 0.10 families, 0.40 temperature, 0.55 main = known copies, 0.70 watermark + recited system prompt, 0.85 plain copies |
| 34 Watermark Localisation | 37 | 0.4613 | 0.699 | 0.5404 heads, 0.5819 rectangles, 0.5944 synthetic data, 0.6769 full-resolution U-Net, 0.6952 head ensembles, 0.699 size prior |

Counts and scores from our submission log, 21/11 18:30 to 22/11 07:15 (Paris).
