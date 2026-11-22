"""Surrogate scores: the official formulas evaluated against HYPOTHETICAL gold models.

The real score is a distance to the gold models, which we do not have. What we can do is write
down plausible hypotheses about the gold and compute the exact official formulas against each.
A hypothesis is a pair (view crop area, logit temperature): a "gold model" k is the ORIGINAL model
queried on a fresh strong view k of each image, with its logits divided by the temperature. This
is the same family `gold_soft` is trained to imitate, so scoring a `gold_soft` candidate against
it is partly circular. The value of the table is therefore the CROSS-HYPOTHESIS spread (how much
a candidate loses when its training hypothesis is wrong), never the absolute level.

All functions are pure and operate on logits, so they are easy to test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_f
from torch import Tensor

from hackathon.unlearning.constants import SCORING_TEMPERATURE

RATIO_CLIP_LOW = 0.0
RATIO_CLIP_HIGH = 1.0
LOG_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class GoldHypothesis:
    """One hypothetical gold: strong views of crop area `area`, logits divided by `temperature`."""

    area: float
    temperature: float

    @property
    def name(self) -> str:
        """Stable identifier used as a JSON key and a figure label."""
        return f"a{self.area:.2f}_t{self.temperature:.1f}"


@dataclass(frozen=True)
class GroupScore:
    """The two official forget sub-scores of one image group against one hypothesis."""

    mean_std_score: float
    prediction_score: float

    @property
    def forget(self) -> float:
        """Product of the two sub-scores (the official combination inside a group)."""
        return self.mean_std_score * self.prediction_score


def per_sample_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-sample cross-entropy at temperature 1 (float64), shape [N]."""
    return torch_f.cross_entropy(logits.double(), labels, reduction="none")


def loss_distance(
    sub_loss: Tensor, gold_means: Sequence[float], gold_stds: Sequence[float]
) -> float:
    """|mean(L) - mean(L_gold)| + |std(L) - std(L_gold)|, gold statistics averaged over models."""
    gold_mean = sum(gold_means) / len(gold_means)
    gold_std = sum(gold_stds) / len(gold_stds)
    return abs(float(sub_loss.mean()) - gold_mean) + abs(
        float(sub_loss.std(unbiased=False)) - gold_std
    )


def clipped_ratio_score(distance_sub: float, distance_orig: float) -> float:
    """1 - clip(d_sub / d_orig, 0, 1); a zero normaliser yields 0 (nothing to improve on)."""
    if distance_orig <= 0:
        return 0.0
    return 1.0 - min(RATIO_CLIP_HIGH, max(RATIO_CLIP_LOW, distance_sub / distance_orig))


def gold_distribution(gold_view_logits: Tensor, hypothesis: GoldHypothesis) -> Tensor:
    """Average predicted distribution p_gold at the scoring temperature, shape [N, C].

    Args:
        gold_view_logits: [K, N, C] logits of the original on K views (one per hypothetical gold).
        hypothesis: Provides the logit temperature of the hypothetical gold models.
    """
    scaled = gold_view_logits.double() / (hypothesis.temperature * SCORING_TEMPERATURE)
    return torch_f.softmax(scaled, dim=2).mean(dim=0)


def mean_kl_from_distribution(log_p_gold: Tensor, logits: Tensor) -> float:
    """Mean KL(p_gold || softmax(logits / 2)); `log_p_gold` is log p_gold at T = 2, shape [N, C]."""
    log_q = torch_f.log_softmax(logits.double() / SCORING_TEMPERATURE, dim=1)
    return float((log_p_gold.exp() * (log_p_gold - log_q)).sum(dim=1).mean())


def score_group(
    sub_logits: Tensor,
    orig_logits: Tensor,
    labels: Tensor,
    gold_view_logits: Tensor,
    hypothesis: GoldHypothesis,
) -> GroupScore:
    """Official mean/std and prediction sub-scores of one group against one hypothesis.

    Args:
        sub_logits: [N, C] logits of the candidate.
        orig_logits: [N, C] logits of the original model (the normaliser of both scores).
        labels: [N] pseudo-labels (argmax of the original on the exact image).
        gold_view_logits: [K, N, C] hypothetical gold logits before temperature.
        hypothesis: The hypothesis (applies its temperature to `gold_view_logits`).
    """
    scaled = gold_view_logits.double() / hypothesis.temperature
    gold_losses = [per_sample_loss(scaled[k], labels) for k in range(len(scaled))]
    gold_means = [float(loss.mean()) for loss in gold_losses]
    gold_stds = [float(loss.std(unbiased=False)) for loss in gold_losses]
    d_sub = loss_distance(per_sample_loss(sub_logits, labels), gold_means, gold_stds)
    d_orig = loss_distance(per_sample_loss(orig_logits, labels), gold_means, gold_stds)

    log_p_gold = (
        gold_distribution(gold_view_logits, hypothesis).clamp_min(LOG_PROBABILITY_FLOOR).log()
    )
    kl_sub = mean_kl_from_distribution(log_p_gold, sub_logits)
    kl_orig = mean_kl_from_distribution(log_p_gold, orig_logits)
    return GroupScore(
        mean_std_score=clipped_ratio_score(d_sub, d_orig),
        prediction_score=clipped_ratio_score(kl_sub, kl_orig),
    )


def hypothesis_loss_statistics(
    gold_view_logits: Tensor, labels: Tensor, hypothesis: GoldHypothesis
) -> dict[str, float]:
    """Mean/std of the hypothetical gold loss and its accuracy on the pseudo-labels (diagnostic)."""
    scaled = gold_view_logits.double() / hypothesis.temperature
    losses = torch.stack([per_sample_loss(scaled[k], labels) for k in range(len(scaled))])
    accuracy = (scaled.argmax(dim=2) == labels.unsqueeze(0)).double().mean()
    confidence = torch_f.softmax(scaled, dim=2).amax(dim=2).mean()
    return {
        "loss_mean": float(losses.mean(dim=1).mean()),
        "loss_std": float(losses.std(dim=1, unbiased=False).mean()),
        "accuracy_vs_pseudo": float(accuracy),
        "confidence": float(confidence),
    }
