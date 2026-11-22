# Tempete

Team **Tempete** (Tempête) at the European AI Championship Paris 2026, watermarking track
(Epitech Kremlin-Bicêtre, 21-22 November 2026). Team: Victor, Alex, Lucas.

We worked on the three tasks of the track: **31 Unlearning**, **33 Model Equality** and
**34 Watermark Localisation**. Scores, ranks and the exact best files are in one place:
[`results.md`](results.md). Slides for the 13:00 presentation: [`slides/`](slides/). Full write-up:
[`docs/writeup.md`](docs/writeup.md) ([PDF](docs/writeup.pdf)).

## The three tasks at a glance

| Task | Presenter | Our best solution in one line | Details and code |
|---|---|---|---|
| 33 Model Equality | Lucas | Families by likelihood under open models; inside a family, the 5 copies of the main model found one by one: watermarked copy by exact repetition, system prompt recited on request, plain copy by likelihood distance to the watermarked copy | [`solutions/33-model-equality/`](solutions/33-model-equality/README.md) |
| 31 Unlearning | Alex | Trust-region unlearning of the body (first blocks frozen, L2-SP), extrapolated, then a closed-form head edit: sibling routing and a gate on the 9 removed-class rows only | [`solutions/31-unlearning/`](solutions/31-unlearning/README.md) |
| 34 Watermark Localisation | Victor | Full-resolution U-Net (StegaStamp), heads and end-to-end fine-tuned extractors (Pixel Seal, TrustMark), SAM objects with a learned selector, transductive self-training, thresholds per regime and zone size | [`solutions/34-watermark-localisation/`](solutions/34-watermark-localisation/README.md) |

## 33 Model Equality

**Problem.** 33 black-box chat endpoints, 100 answers each: decide for every pair whether they share
the base model and whether they serve the same model (watermark, system prompt, persona and
temperature do not change the model; fp8 and each LoRA strength do). Score = 0.1 MCC(family) +
0.9 MCC(model).

**Best solution.** The practice endpoints show the construction: per family, 5 copies of one model
(plain, watermark, system prompt, persona, T 1.0) plus 6 variants. We found the 3 families of 11 by
likelihood, then the 5 copies in each: disguised copies by identity questions, the watermarked copy
because it is deterministic (identical answers to a repeated prompt), the hidden system prompt of
ep-29 recited on request, and the plain copy as the endpoint closest to the watermarked copy in
likelihood under the base checkpoint. Every other endpoint is its own model.

**What worked.** Exact facts read from the answers (repetition, recitation), and likelihood under the
right base model. **What did not.** Text distances (MMD, Hamming: 0.05), temperature alone (stuck at
0.40 for 8 submissions), API side channels (all ignored), the weak LoRA 0.3 (not separable within the
budget).

**Code.** [`prefix.py#L32`](solutions/33-model-equality/prefix.py#L32) (repetition),
[`rounds/round7_prompts.json`](solutions/33-model-equality/rounds/round7_prompts.json) (recitation),
[`plain_rank.py#L36`](solutions/33-model-equality/plain_rank.py#L36) (plain copy),
[`e_loglik.py#L349`](src/hackathon/model_equality/research/e_loglik.py#L349) and
[`e_matrix.py#L148`](src/hackathon/model_equality/research/e_matrix.py#L148) (families),
[`Fp8Simulator`](src/hackathon/model_equality/research/e_loglik.py#L441) and
[`analyse_paired`](src/hackathon/model_equality/research/x_relect_analysis.py#L160) (fp8),
[`client.py#L251`](src/hackathon/model_equality/client.py#L251) (budgeted API client).

**Reproduce.** `PYTHONPATH=src python -m hackathon.model_equality validate solutions/33-model-equality/submission_best.csv`
checks the submitted partition offline; the full pipeline needs the endpoints (see
[`solutions/33-model-equality/`](solutions/33-model-equality/README.md#reproduce)).

## 31 Unlearning

**Problem.** A ViT-B/16 with 100 classes and 3 500 unlabelled forget images: return a model that
behaves like a gold model retrained without them, using only the original model and the forget set.
Score = forget quality (loss law and predictions close to the gold) x retain (accuracy and KL drift
on a hidden control set).

**Best solution (Ck2).** Body: Alex's trust-region method (calibrated soft targets, K-fold anchor on
the forget set, stop at a KL radius) with blocks 0-2 frozen and an L2-SP pull to the original,
extrapolated by alpha 1.64. Head, in closed form: route the images of the 9 classes saturated by the
forget set (129-130 images each) to their sibling classes, then gate only those 9 rows,
`z_c = 2.5 x (w_c . f + b_c - 3.25)`, so a removed class survives only on strong evidence; the 91 kept
rows are scaled by 1.75.

**What worked.** Moving the network as little as possible (head edits, small trust-region updates,
frozen blocks and L2-SP), then forgetting through the body and gating only the removed rows (0.219 to
0.2445 with the same body). **What did not.** Full retraining toward per-image targets (0.098 for
0.22 predicted), smooth gold targets (0.084), restoring kept classes on synthetic images (no transfer),
flattening the outputs (0.2098), erasing the 9 classes completely (0.0278), SSD, model inversion.

**Code.**
[`trust_region_unlearn`](solutions/31-unlearning/trust_region_src/src/hackathon/unlearning/methods.py#L613),
L2-SP patch [`train_v4l_freeze_l2sp.sh#L54`](solutions/31-unlearning/train_v4l_freeze_l2sp.sh#L54),
head gate [`build_head_gate_grid.sh#L90`](solutions/31-unlearning/build_head_gate_grid.sh#L90) and
[`head_gate_ck2_job.sh`](solutions/31-unlearning/head_gate_ck2_job.sh),
[`cosine_groups`](src/hackathon/unlearning/research/c_classes.py#L111),
[`validate_submission`](src/hackathon/unlearning/validate.py#L117).

**Reproduce.** The 344 MB file is not in the repository (GitHub limit). Rebuild it with the jobs of
[`solutions/31-unlearning/`](solutions/31-unlearning/README.md#rebuild-the-file) on one GPU.

## 34 Watermark Localisation

**Problem.** Images watermarked with Pixel Seal, TrustMark or StegaStamp, then edited: return the
128 x 128 mask of the area that still carries the watermark ("full" regime with a restored hole, or a
"rect" regime with a small marked zone). Score = mean IoU (all-zero = 0.25).

**Best solution.** Per method, the best probability map: a full-resolution U-Net with a denoiser
(StegaStamp); segmentation heads averaged with end-to-end fine-tuned official extractors (Pixel Seal,
TrustMark), the TrustMark one self-trained on confident test pseudo-labels; SAM objects chosen by a
learned selector in the rect regime; rectangle and occlusion overlays from the known payload; one
threshold per (predicted regime x predicted zone size), chosen leave-one-out.

**What worked.** Reading pixels at full resolution for StegaStamp (+0.08 in one submission), SAM with
a learned selector (+0.017), end-to-end fine-tuning and self-training on TrustMark. **What did not.**
Window-by-window decoding (the payload is a global code), occlusion of the TrustMark decoder (AUC
0.68), more Pixel Seal changes (saturated at the public read), forensic features, Mask R-CNN.

**Code.** [`FullResolutionNet`](src/hackathon/wmloc/research/small_train.py#L123),
[`B2Net`](src/hackathon/wmloc/research/b2_model.py#L159),
[`e2e_loc.py`](solutions/34-watermark-localisation/e2e/e2e_loc.py),
[`sam_learn.py`](solutions/34-watermark-localisation/sam/sam_learn.py#L168),
[`pseudo_labels.py`](solutions/34-watermark-localisation/loclab/pseudo_labels.py#L32),
[`box_refine.py`](solutions/34-watermark-localisation/sam/box_refine.py#L73),
[`size_prior.py`](solutions/34-watermark-localisation/loclab/size_prior.py#L39),
[`assemble`](solutions/34-watermark-localisation/assemble.py#L80),
[`per_image_miou`](src/hackathon/wmloc/metric.py#L83).

**Reproduce.** The submitted file is in the repository
([`submission_best.pt`](solutions/34-watermark-localisation/submission_best.pt)); training needs the
gated dataset and a GPU (see [`solutions/34-watermark-localisation/`](solutions/34-watermark-localisation/README.md#reproduce)).

## How we worked

- **Three people, a team of AI coding agents.** Each of us steered one task (Lucas: model equality,
  Alex: unlearning, Victor: localisation and coordination). Research tracks ran in parallel as coding
  agents (Claude Code), one git worktree each, with a written prediction before every submission and
  a rule to stop a track after two losses.
- **One GPU node.** JURECA (Jülich), one node of 4 A100 for the whole team, kept by a runner job that
  executed a priority queue of short tasks (two GPUs for localisation, two for the other tasks).
- **Submission discipline.** One queue per task, one submission per server cooldown, expected score
  written before reading the result; the last submission of each task is its best file.

## Repository layout

```
src/hackathon/common/          configuration, server client, submission helpers
src/hackathon/unlearning/      unlearning kit (data, model, proxy evaluation, validator) and research code
src/hackathon/model_equality/  API client, answer store, likelihood re-reading, partitions, validator
src/hackathon/wmloc/           localisation kit (metric, validator, synthetic data) and research code
src/wmkit/                     our watermarking toolkit (Pixel Seal wrapper, attacks, metrics)
solutions/<task>/              scripts that produced the best submission of each task, with a README
tests/hackathon/               unit tests of the packages
slides/                        presentation (PPTX and PDF) and its generators
docs/                          final write-up (Markdown and PDF) and its generator
results.md                     scores, ranks, best files and hashes (single source of numbers)
```

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a CUDA GPU for any training, a Hugging
Face token with access to the gated localisation dataset, and the team API key of the event for the
Model Equality endpoints and submissions. No competition data, answer of the API or credential is
stored in this repository.

```bash
uv sync                                   # dependencies, including pytest and mypy
cp .env.example .env                      # fill in the values (every variable is documented)
PYTHONPATH=src uv run python -m hackathon.model_equality validate solutions/33-model-equality/submission_best.csv
```

Tests: `uv run pytest -q` (fast tests only; tests marked `slow` need weights or data).

Slides, report and their numbers: `bash scripts/rebuild_docs.sh <log.jsonl>` regenerates
`slides/deck_data.json`, the deck (PPTX, PDF) and the write-up (Markdown, PDF) from `results.md` and our
submission log (needs python-pptx, lxml, markdown and a Chromium).

## Known pitfalls

- The Model Equality budget is per endpoint and for the whole event; the server ignores
  `temperature`, `seed`, `n` and `logprobs`, and allows 4 requests at a time.
- The Unlearning rules forbid external data, even for tuning; synthetic images made from the original
  model alone are allowed.
- The localisation starter kit (TrustMark, StegaStamp ONNX) runs in its own Python 3.11 environment.
- Public scores cover about 30 % of the test data and are rounded for Model Equality (0.05): a single
  read moves within that noise.
