"""Evaluation context: fixed image subsets, cached original logits, surrogate gold views, metrics.

One `ResearchContext` is built per process. It owns:

* subset A (training/eval images of a run) and subset B (other forget images, never trained on in
  exploration runs): B measures whether the effect of a method is specific to the images it was
  trained on or leaks to every in-distribution image the original memorised (which would hurt the
  hidden control set if that set was part of the original training data);
* simulated altered copies of A (the kit's four alterations, fixed seed);
* a retain proxy: out-of-distribution natural images (COCO / DIV2K / Kodak), see the kit docs;
* original logits of every group and hypothetical-gold view logits (cached on disk).

`evaluate` returns a JSON-able dict plus per-sample arrays (for histograms).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.alterations import ALTERATIONS, apply_alteration, strong_view
from hackathon.unlearning.constants import (
    DEFAULT_SEED,
    FORGET_IMAGES_FILENAME,
    MODEL_FILENAME,
    SCORING_TEMPERATURE,
)
from hackathon.unlearning.data import load_forget_images, load_proxy_images
from hackathon.unlearning.model import load_model, predict
from hackathon.unlearning.research.surrogate import (
    GoldHypothesis,
    hypothesis_loss_statistics,
    per_sample_loss,
    score_group,
)

LOGGER = logging.getLogger("hackathon.unlearning.research")

SUBSET_SEED = DEFAULT_SEED
"""Named seed of the A/B split and of the retain draw."""
SUBSET_A_COUNT = 256
SUBSET_B_COUNT = 256
RETAIN_EVAL_COUNT = 128
HOLDOUT_EVAL_COUNT = 128
"""Only the first images of subset B are evaluated (halves the cost of the leak measurement)."""
ALL_PROXY_IMAGES = 10_000
"""Larger than the folder: `load_proxy_images` then returns every available image."""
EVALUATED_ALTERATIONS = ("crop", "jitter")
"""Flip and JPEG q50 leave the memorised original at ~0 loss (measured): they cost time, say nothing."""
GOLD_VIEW_AREAS = (0.20, 0.35, 0.60)
GOLD_TEMPERATURES = (1.0, 2.0, 3.0)
GOLD_VIEW_COUNT = 4
GOLD_VIEW_SEED = 10_000
"""Seed of the hypothetical-gold views; training views use `training seed + step`, never this."""
ALTERATION_SEED_OFFSET = 100
EVAL_BATCH_SIZE = 32
CONFIDENCE_BINS = 10
DISTRIBUTION_CACHE_NAME = "context_cache.pt"
GOLD_HYPOTHESES = tuple(
    GoldHypothesis(area, temperature)
    for area in GOLD_VIEW_AREAS
    for temperature in GOLD_TEMPERATURES
)


@dataclass
class ResearchContext:
    """Everything an evaluation needs, built once.

    Attributes:
        device: Compute device of the original model.
        original: The provided model, eval mode, on `device`.
        forget_all: All 3500 forget images (uint8 NHWC, CPU).
        indices_a: Positions of subset A inside `forget_all`.
        indices_b: Positions of subset B (disjoint from A).
        retain: Retain-proxy images used ONLY to measure drift (uint8 NHWC).
        retain_train: Disjoint retain-proxy images used as the KL anchor during training (the
            kit evaluates on the same images it trains on, which understates the drift).
        altered_a: Alteration name -> altered copy of subset A.
        orig_logits: Group name -> original logits.
        gold_view_logits: Area -> [K, |A|, C] logits of the original on K strong views of A.
        labels_a: Pseudo-labels of A (argmax of the original on the exact images).
    """

    device: torch.device
    original: nn.Module
    forget_all: Tensor
    indices_a: Tensor
    indices_b: Tensor
    retain: Tensor
    retain_train: Tensor
    altered_a: dict[str, Tensor]
    orig_logits: dict[str, Tensor] = field(default_factory=dict)
    gold_view_logits: dict[float, Tensor] = field(default_factory=dict)
    labels_a: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    @property
    def forget_a(self) -> Tensor:
        """Subset A images."""
        return self.forget_all[self.indices_a]

    @property
    def forget_b(self) -> Tensor:
        """Subset B images."""
        return self.forget_all[self.indices_b]


def _split_indices(total: int, count_a: int, count_b: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    return order[:count_a].sort().values, order[count_a : count_a + count_b].sort().values


def _group_images(context: ResearchContext) -> dict[str, Tensor]:
    groups: dict[str, Tensor] = {"exact": context.forget_a}
    for name, images in context.altered_a.items():
        groups[f"alt_{name}"] = images
    groups["holdout_b"] = context.forget_b[:HOLDOUT_EVAL_COUNT]
    groups["retain"] = context.retain
    return groups


def build_context(
    task_dir: Path, proxy_root: Path, device: torch.device, cache_dir: Path
) -> ResearchContext:
    """Load data and the original, compute (or load from `cache_dir`) the reference logits."""
    forget_all = load_forget_images(task_dir / FORGET_IMAGES_FILENAME)
    original = load_model(task_dir / MODEL_FILENAME, device)
    indices_a, indices_b = _split_indices(
        len(forget_all), SUBSET_A_COUNT, SUBSET_B_COUNT, SUBSET_SEED
    )
    everything = load_proxy_images(proxy_root, ALL_PROXY_IMAGES, SUBSET_SEED)
    proxy_order = torch.randperm(
        len(everything), generator=torch.Generator().manual_seed(SUBSET_SEED)
    )
    retain = everything[proxy_order[:RETAIN_EVAL_COUNT]]
    retain_train = everything[proxy_order[RETAIN_EVAL_COUNT:]]
    exact_a = forget_all[indices_a]
    altered = {
        name: apply_alteration(name, exact_a, SUBSET_SEED + ALTERATION_SEED_OFFSET + offset)
        for offset, name in enumerate(ALTERATIONS, start=1)
        if name in EVALUATED_ALTERATIONS
    }
    context = ResearchContext(
        device, original, forget_all, indices_a, indices_b, retain, retain_train, altered
    )
    cache_path = cache_dir / DISTRIBUTION_CACHE_NAME
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context.orig_logits = dict(payload["orig_logits"])
        context.gold_view_logits = {float(k): v for k, v in payload["gold_view_logits"].items()}
        context.orig_logits["holdout_b"] = context.orig_logits["holdout_b"][:HOLDOUT_EVAL_COUNT]
    else:
        for name, images in _group_images(context).items():
            LOGGER.info("original logits: %s (%d images)", name, len(images))
            context.orig_logits[name] = predict(original, images, EVAL_BATCH_SIZE)
        generator = torch.Generator().manual_seed(GOLD_VIEW_SEED)
        for area in GOLD_VIEW_AREAS:
            views = []
            for _ in range(GOLD_VIEW_COUNT):
                view = strong_view(exact_a, generator, area)
                views.append(predict(original, view, EVAL_BATCH_SIZE))
            context.gold_view_logits[area] = torch.stack(views)
            LOGGER.info("hypothetical gold views at area %.2f done", area)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "orig_logits": context.orig_logits,
                "gold_view_logits": {str(k): v for k, v in context.gold_view_logits.items()},
            },
            cache_path,
        )
    context.labels_a = context.orig_logits["exact"].argmax(dim=1)
    return context


def _group_stats(orig_logits: Tensor, sub_logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Proxy diagnostics of one group (all against the ORIGINAL, see the kit docs)."""
    loss_orig = per_sample_loss(orig_logits, labels)
    loss_sub = per_sample_loss(sub_logits, labels)
    log_p = torch_f.log_softmax(orig_logits.double() / SCORING_TEMPERATURE, dim=1)
    log_q = torch_f.log_softmax(sub_logits.double() / SCORING_TEMPERATURE, dim=1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=1)
    log_p1 = torch_f.log_softmax(orig_logits.double(), dim=1)
    log_q1 = torch_f.log_softmax(sub_logits.double(), dim=1)
    kl_native = (log_p1.exp() * (log_p1 - log_q1)).sum(dim=1)
    return {
        "loss_mean_orig": float(loss_orig.mean()),
        "loss_std_orig": float(loss_orig.std(unbiased=False)),
        "loss_mean_sub": float(loss_sub.mean()),
        "loss_std_sub": float(loss_sub.std(unbiased=False)),
        "loss_shift": abs(float(loss_sub.mean()) - float(loss_orig.mean()))
        + abs(float(loss_sub.std(unbiased=False)) - float(loss_orig.std(unbiased=False))),
        "kl_t2_mean": float(kl.mean()),
        "kl_t1_mean": float(kl_native.mean()),
        "argmax_agreement": float((orig_logits.argmax(1) == sub_logits.argmax(1)).double().mean()),
        "pseudo_accuracy_sub": float((sub_logits.argmax(1) == labels).double().mean()),
        "confidence_orig": float(torch_f.softmax(orig_logits.double(), dim=1).amax(1).mean()),
        "confidence_sub": float(torch_f.softmax(sub_logits.double(), dim=1).amax(1).mean()),
    }


@dataclass
class EvalResult:
    """Metrics (JSON-able) and per-sample arrays (for histograms) of one evaluation."""

    metrics: dict[str, object]
    arrays: dict[str, Tensor]


def evaluate(context: ResearchContext, model: nn.Module, full: bool = True) -> EvalResult:
    """Evaluate `model`; `full=False` skips the altered copies and subset B (cheaper, for curves)."""
    model.eval()
    groups = _group_images(context)
    wanted = list(groups) if full else ["exact", "retain"]
    logits = {name: predict(model, groups[name], EVAL_BATCH_SIZE) for name in wanted}
    labels = context.labels_a
    metrics: dict[str, object] = {}
    arrays: dict[str, Tensor] = {}
    group_stats: dict[str, dict[str, float]] = {}
    for name in wanted:
        group_labels = (
            context.orig_logits[name].argmax(1) if name in ("retain", "holdout_b") else labels
        )
        group_stats[name] = _group_stats(context.orig_logits[name], logits[name], group_labels)
    metrics["groups"] = group_stats
    arrays["loss_exact"] = per_sample_loss(logits["exact"], labels).float()
    arrays["confidence_exact"] = torch_f.softmax(logits["exact"].double(), 1).amax(1).float()
    arrays["argmax_exact"] = logits["exact"].argmax(1)
    arrays["logits_exact"] = logits["exact"].float()
    metrics["retain_kl_t2"] = group_stats["retain"]["kl_t2_mean"]
    metrics["retain_factor_proxy"] = 1.0 / (1.0 + group_stats["retain"]["kl_t2_mean"])
    metrics["retain_argmax_flip_rate"] = 1.0 - group_stats["retain"]["argmax_agreement"]
    metrics["forget_kl_t2"] = group_stats["exact"]["kl_t2_mean"]
    if full:
        metrics["holdout_kl_t2"] = group_stats["holdout_b"]["kl_t2_mean"]
        metrics["holdout_argmax_flip_rate"] = 1.0 - group_stats["holdout_b"]["argmax_agreement"]
        denominator = max(group_stats["exact"]["kl_t2_mean"], 1e-12)
        metrics["leak_ratio_holdout_over_trained"] = (
            group_stats["holdout_b"]["kl_t2_mean"] / denominator
        )
        metrics["surrogate"] = _surrogate_table(context, logits)
    return EvalResult(metrics, arrays)


def _surrogate_table(context: ResearchContext, logits: dict[str, Tensor]) -> dict[str, object]:
    """Official forget sub-scores against each hypothetical gold (exact and altered groups)."""
    table: dict[str, dict[str, float]] = {}
    for hypothesis in GOLD_HYPOTHESES:
        views = context.gold_view_logits[hypothesis.area]
        exact = score_group(
            logits["exact"], context.orig_logits["exact"], context.labels_a, views, hypothesis
        )
        altered = [
            score_group(
                logits[f"alt_{name}"],
                context.orig_logits[f"alt_{name}"],
                context.labels_a,
                views,
                hypothesis,
            )
            for name in context.altered_a
        ]
        mean_std = 0.5 * (
            exact.mean_std_score + sum(a.mean_std_score for a in altered) / len(altered)
        )
        prediction = 0.5 * (
            exact.prediction_score + sum(a.prediction_score for a in altered) / len(altered)
        )
        table[hypothesis.name] = {
            "mean_std_score": mean_std,
            "prediction_score": prediction,
            "forget_score": mean_std * prediction,
            "forget_score_exact_only": exact.forget,
        }
    forget_scores = [row["forget_score"] for row in table.values()]
    return {
        "by_hypothesis": table,
        "forget_mean_over_hypotheses": sum(forget_scores) / len(forget_scores),
        "forget_min_over_hypotheses": min(forget_scores),
    }


def hypothesis_statistics(context: ResearchContext) -> dict[str, dict[str, float]]:
    """Loss mean/std/accuracy/confidence implied by each hypothesis (what each one stands for)."""
    return {
        h.name: hypothesis_loss_statistics(context.gold_view_logits[h.area], context.labels_a, h)
        for h in GOLD_HYPOTHESES
    }
