# Tempete

Team **Tempete** (Tempête) at the European AI Championship Paris 2026, watermarking track
(Epitech Kremlin-Bicêtre, 21-22 November 2026). Team: Victor, Alex, Lucas.

We worked on the three tasks of the track: **31 Unlearning**, **33 Model Equality** and
**34 Watermark Localisation**. Scores, ranks and the exact best files are in one place:
[`results.md`](results.md). Slides for the 13:00 presentation: [`slides/`](slides/).

## The three tasks at a glance

| Task | Our best solution in one line | Details and code |
|---|---|---|
| 31 Unlearning | Trust-region unlearning on the forget set with the first ViT blocks frozen and an L2-SP pull to the original weights, then a head recipe (extrapolation, sibling routing of the 9 removed classes, bias shift, head scale) | [`solutions/31-unlearning/`](solutions/31-unlearning/README.md) |
| 33 Model Equality | Families by likelihood under open models; inside a family, the 5 copies of the main model found one by one: watermarked copy by exact repetition, system prompt recited on request, plain copy by likelihood distance to the watermarked copy | [`solutions/33-model-equality/`](solutions/33-model-equality/README.md) |
| 34 Watermark Localisation | Full-resolution U-Net for StegaStamp, ensembles of segmentation heads for Pixel Seal and TrustMark, occlusion and rectangle overlays, thresholds per regime and per predicted zone size | [`solutions/34-watermark-localisation/`](solutions/34-watermark-localisation/README.md) |

## 31 Unlearning

**Problem.** A ViT-B/16 with 100 classes and 3 500 unlabelled forget images: return a model that
behaves like a gold model retrained without them, using only the original model and the forget set.
Score = forget quality (loss law and predictions close to the gold) x retain (accuracy and KL drift
on a hidden control set).

**Best solution.** Stage 1: Alex's trust-region method (calibrated soft targets, K-fold anchor on the
forget set, stop at a KL radius), blocks 0-2 or 0-3 frozen, L2-SP lambda 60. Stage 2: extrapolate the
delta (alpha about 1.5), route the images of the 9 classes saturated by the forget set (129-130
images each) to their sibling classes, shift their bias to match the gold loss law (mean 2.25, std 3.0), scale the head by
1.75, validate.

**What worked.** Moving the network as little as possible: head edits, then small trust-region
updates, then freezing and L2-SP. Measured: the score falls with the drift on synthetic images of the
removed classes (`public ~ 0.238 - 0.083 x KL`, r = -0.79): forget the images, not the classes.
**What did not.** Full retraining toward per-image targets (0.098 for 0.22 predicted), SSD, model
inversion as an anchor, softer heads, soups across settings.

**Code.**
[`trust_region_unlearn`](solutions/31-unlearning/trust_region_src/src/hackathon/unlearning/methods.py#L613),
[`_trust_phase`](solutions/31-unlearning/trust_region_src/src/hackathon/unlearning/methods.py#L477),
L2-SP patch [`train_v4l_freeze_l2sp.sh#L54`](solutions/31-unlearning/train_v4l_freeze_l2sp.sh#L54),
head recipe [`build_submission.sh#L136`](solutions/31-unlearning/build_submission.sh#L136) and
[`#L169`](solutions/31-unlearning/build_submission.sh#L169),
[`cosine_groups`](src/hackathon/unlearning/research/c_classes.py#L111),
[`validate_submission`](src/hackathon/unlearning/validate.py#L117).

**Reproduce.** The 344 MB file is not in the repository (GitHub limit). Rebuild it with the two jobs
of [`solutions/31-unlearning/`](solutions/31-unlearning/README.md#rebuild-the-file) on one GPU.

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

## 34 Watermark Localisation

**Problem.** Images watermarked with Pixel Seal, TrustMark or StegaStamp, then edited: return the
128 x 128 mask of the area that still carries the watermark ("full" regime with a restored hole, or a
"rect" regime with a small marked rectangle). Score = mean IoU (all-zero = 0.25).

**Best solution.** Per method, the best probability map: a full-resolution U-Net trained on synthetic
composites made with the official embedders (StegaStamp), ensembles of segmentation heads over the
frozen extractor features (Pixel Seal, TrustMark); overlays from occlusion of the decoder margin and
from a rectangle search against the payload; one threshold per (predicted regime x predicted zone
size), chosen leave-one-out on the train split.

**What worked.** Reading pixels at full resolution for StegaStamp (+0.08 in one submission), synthetic
training data, regime- and size-aware thresholds. **What did not.** Window-by-window decoding and
matched filters (the payload is a global code), forensic features, superpixels, Mask R-CNN proposals.
The smallest Pixel Seal and TrustMark zones (under 2 % of the image) do not decode even when cropped
exactly: that is the remaining loss.

**Code.** [`FullResolutionNet`](src/hackathon/wmloc/research/small_train.py#L123),
[`small_gen.py`](src/hackathon/wmloc/research/small_gen.py),
[`B2Net`](src/hackathon/wmloc/research/b2_model.py#L159),
[`occlusion_map`](src/hackathon/wmloc/research/c_occlusion.py#L89),
[`search_rectangle`](src/hackathon/wmloc/research/c_rect_search.py#L332),
[`size_prior.py`](solutions/34-watermark-localisation/loclab/size_prior.py#L39),
[`assemble`](solutions/34-watermark-localisation/assemble.py#L80),
[`per_image_miou`](src/hackathon/wmloc/metric.py#L83).

**Reproduce.** The submitted file is in the repository
([`submission_best.pt`](solutions/34-watermark-localisation/submission_best.pt)); training needs the
gated dataset and a GPU (see [`solutions/34-watermark-localisation/`](solutions/34-watermark-localisation/README.md#reproduce)).

## How we worked

- **Three people, a team of AI coding agents.** Each of us owned a task direction (Alex: unlearning,
  Lucas: model equality, Victor: coordination and localisation). Research tracks ran in parallel as
  coding agents (Claude Code), one git worktree each, with a written prediction before every
  submission and a rule to stop a track after two losses.
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
slides/                        presentation (PPTX and PDF)
results.md                     scores, ranks, best files and hashes
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

## Known pitfalls

- The Model Equality budget is per endpoint and for the whole event; the server ignores
  `temperature`, `seed`, `n` and `logprobs`, and allows 4 requests at a time.
- The Unlearning rules forbid external data, even for tuning; synthetic images made from the original
  model alone are allowed.
- The localisation starter kit (TrustMark, StegaStamp ONNX) runs in its own Python 3.11 environment.
- Public scores cover about 30 % of the test data and are rounded for Model Equality (0.05): a single
  read moves within that noise.
