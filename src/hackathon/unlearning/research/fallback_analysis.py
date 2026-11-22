"""Which WRONG class would a model that never saw the image pick? Agreement between hypotheses.

Every hypothesis names, for each forget image of subset A, a "fallback class" different from the
original (memorised) prediction `y`. None is ground truth: they are all derived from the original
model, so this measures how much the hypotheses AGREE with one another (a stable fallback class is
a safer training target) and how many images are fragile at all (views disagree with `y`).

Hypotheses:
    rank2          second class of the original on the exact image
    view_mode_<a>  class != y with the largest summed probability over 8 strong views of area `a`
    strongest      same, on 8 views of area 0.15 (the hardest crops)
    jitter_rank2   second class of the original on the colour-jittered copy of the image
"""

from __future__ import annotations

import json
import logging

import torch
from torch import Tensor

from hackathon.common.config import load_settings
from hackathon.unlearning.alterations import strong_view
from hackathon.unlearning.constants import default_task_dir
from hackathon.unlearning.model import predict, select_device
from hackathon.unlearning.research.context import EVAL_BATCH_SIZE, LOGGER, build_context
from hackathon.unlearning.research.warmup import RESEARCH_SUBDIR

VIEW_COUNT = 8
VIEW_AREAS = (0.25, 0.35, 0.15)
ANALYSIS_SEED = 31_000
TOP_RANK_FOR_OVERLAP = 3
OUTPUT_NAME = "fallback_analysis.json"


def wrong_class_from_mass(mass: Tensor, original_class: Tensor) -> Tensor:
    """Argmax of `mass` [N, C] excluding the original class of each row."""
    masked = mass.clone()
    masked[torch.arange(len(masked)), original_class] = float("-inf")
    return masked.argmax(dim=1)


def pairwise_agreement(classes: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    """Fraction of images on which two hypotheses name the same fallback class."""
    return {
        a: {b: float((classes[a] == classes[b]).double().mean()) for b in classes} for a in classes
    }


def main() -> None:
    """Compute the hypotheses on subset A and write the JSON next to the run results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = load_settings()
    task_dir = default_task_dir(settings.data_dir)
    research_dir = task_dir / RESEARCH_SUBDIR
    context = build_context(task_dir, settings.data_dir.parent, select_device(None), research_dir)
    exact = context.forget_a
    original_class = context.labels_a
    exact_logits = context.orig_logits["exact"]
    classes: dict[str, Tensor] = {
        "rank2": wrong_class_from_mass(exact_logits, original_class),
        "jitter_rank2": wrong_class_from_mass(context.orig_logits["alt_jitter"], original_class),
    }
    generator = torch.Generator().manual_seed(ANALYSIS_SEED)
    agreement_fraction: dict[str, Tensor] = {}
    for area in VIEW_AREAS:
        probabilities = []
        for _ in range(VIEW_COUNT):
            logits = predict(context.original, strong_view(exact, generator, area), EVAL_BATCH_SIZE)
            probabilities.append(torch.softmax(logits, dim=1))
        stacked = torch.stack(probabilities)
        name = "strongest" if area == min(VIEW_AREAS) else f"view_mode_{area}"
        classes[name] = wrong_class_from_mass(stacked.sum(dim=0), original_class)
        agreement_fraction[name] = (
            (stacked.argmax(dim=2) == original_class.unsqueeze(0)).double().mean(dim=0)
        )
        LOGGER.info("views at area %.2f done", area)
    ranking = (
        exact_logits.masked_fill(
            torch.nn.functional.one_hot(original_class, exact_logits.shape[1]).bool(),
            float("-inf"),
        )
        .topk(TOP_RANK_FOR_OVERLAP, dim=1)
        .indices
    )
    in_top = {
        name: float((ranking == value.unsqueeze(1)).any(dim=1).double().mean())
        for name, value in classes.items()
    }
    fragile = {
        name: {
            "fraction_with_any_disagreeing_view": float((fraction < 1.0).double().mean()),
            "mean_agreement": float(fraction.mean()),
            "histogram_of_agreement": torch.histc(
                fraction.float(), bins=VIEW_COUNT + 1, min=0, max=1
            ).tolist(),
        }
        for name, fraction in agreement_fraction.items()
    }
    payload = {
        "pairwise_agreement": pairwise_agreement(classes),
        "fallback_in_top3_of_rank2_ranking": in_top,
        "fragility": fragile,
        "classes": {name: value.tolist() for name, value in classes.items()},
        "original_class": original_class.tolist(),
    }
    (research_dir / OUTPUT_NAME).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    LOGGER.info("written %s", research_dir / OUTPUT_NAME)


if __name__ == "__main__":
    main()
