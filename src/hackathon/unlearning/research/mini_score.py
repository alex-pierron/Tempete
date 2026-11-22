"""The OFFICIAL unlearning score (subject formulas), on logits, in pure numpy.

Everything here is a transcription of `Unlearning.pdf`, with no model and no I/O, so it can be unit
tested against hand-computed values and reused on any scale (the mini benchmark of
`mini_bench.py`, or the real ViT once gold models exist):

    final  = forget x retain
    forget = mean_std_score x prediction_score        (each averaged over exact images and altered copies)
    retain = clip(acc_sub / acc_orig, 0, 1) x 1 / (1 + mean KL(p_orig || p_sub))   (control set)
    gate   : accuracy on a validation set below a threshold => final = 0

Choices the subject leaves open, fixed here and documented:

* `std` is the population standard deviation (ddof = 0);
* the gold reference is the average over gold models of the per-model mean and std of the loss,
  and the average of the per-model predicted distributions (softmax at temperature 2);
* a zero denominator (`d_orig == 0` or `KL(p_gold || p_orig) == 0`) gives score 1 when the numerator
  is also zero and 0 otherwise (sub = orig exactly, and orig = gold exactly, are degenerate);
* the group average of each sub-score is the plain mean of the exact-image and altered-copy values.

The proxy readings of the kit (`proxy_eval`) are computed here too, from the same logits, so that
"does the local proxy predict the real score" is answered on identical inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

SCORING_TEMPERATURE = 2.0
"""Temperature of every predicted distribution p_m in the official score."""
CROSS_ENTROPY_TEMPERATURE = 1.0
"""Per-sample loss L_m is the plain cross-entropy (temperature 1)."""
ZERO_TOLERANCE = 1e-12
"""Below this a distance or KL counts as exactly zero (degenerate denominator)."""
BOOTSTRAP_RESAMPLES = 2000
"""Resamples of the bootstrap confidence interval of a rank correlation."""
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 0

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SetLogits = Mapping[str, FloatArray]
"""Logits [N, C] of one model, keyed by evaluation set name."""

EXACT_SET = "forget_exact"
ALTERED_SET = "forget_altered"
CONTROL_SET = "control"
VALIDATION_SET = "validation"


class ScoreError(ValueError):
    """The logits or labels handed to the scorer are inconsistent (shape, empty, no gold)."""


def log_softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Row-wise log-softmax of `logits / temperature` in float64 (stable)."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return np.asarray(shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True)))


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Row-wise softmax of `logits / temperature`."""
    return np.exp(log_softmax(logits, temperature))


def cross_entropy(logits: FloatArray, labels: IntArray) -> FloatArray:
    """Per-sample cross-entropy loss L_m(x) with the true label, shape [N]."""
    if logits.shape[0] != labels.shape[0]:
        raise ScoreError(f"{logits.shape[0]} logits rows for {labels.shape[0]} labels")
    log_p = log_softmax(logits, CROSS_ENTROPY_TEMPERATURE)
    return np.asarray(-log_p[np.arange(len(labels)), labels])


def per_sample_kl(reference_probs: FloatArray, other_logits: FloatArray) -> FloatArray:
    """Per-sample KL(reference || softmax(other / T)) with T = 2, shape [N]."""
    log_q = log_softmax(other_logits, SCORING_TEMPERATURE)
    reference = np.clip(reference_probs, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(reference > 0, reference * (np.log(reference) - log_q), 0.0)
    return np.asarray(terms.sum(axis=1))


def accuracy(logits: FloatArray, labels: IntArray) -> float:
    """Top-1 accuracy of `logits` on `labels`."""
    if len(labels) == 0:
        raise ScoreError("accuracy of an empty set")
    return float((logits.argmax(axis=1) == labels).mean())


def clip_unit(value: float) -> float:
    """clip(value, 0, 1)."""
    return float(min(1.0, max(0.0, value)))


def ratio_score(numerator: float, denominator: float) -> float:
    """1 - clip(numerator / denominator, 0, 1), with the degenerate zero denominator handled."""
    if denominator <= ZERO_TOLERANCE:
        return 1.0 if numerator <= ZERO_TOLERANCE else 0.0
    return 1.0 - clip_unit(numerator / denominator)


@dataclass(frozen=True)
class GoldReference:
    """Gold statistics on one group of images: what the sub and orig models are compared to."""

    loss_mean: float
    loss_std: float
    probabilities: FloatArray
    """Average over gold models of softmax(logits / 2), shape [N, C]."""


def build_gold_reference(gold_logits: Sequence[FloatArray], labels: IntArray) -> GoldReference:
    """Average the per-model loss mean / std and the per-model T = 2 distributions.

    Raises:
        ScoreError: no gold model, or gold logits of different shapes.
    """
    if not gold_logits:
        raise ScoreError("at least one gold model is required")
    shapes = {tuple(z.shape) for z in gold_logits}
    if len(shapes) != 1:
        raise ScoreError(f"gold logits have different shapes: {sorted(shapes)}")
    losses = [cross_entropy(z, labels) for z in gold_logits]
    return GoldReference(
        loss_mean=float(np.mean([loss.mean() for loss in losses])),
        loss_std=float(np.mean([loss.std() for loss in losses])),
        probabilities=np.mean([softmax(z, SCORING_TEMPERATURE) for z in gold_logits], axis=0),
    )


def loss_distance(logits: FloatArray, labels: IntArray, gold: GoldReference) -> float:
    """d_m = |mean(L_m) - mean(L_gold)| + |std(L_m) - std(L_gold)|."""
    loss = cross_entropy(logits, labels)
    return abs(float(loss.mean()) - gold.loss_mean) + abs(float(loss.std()) - gold.loss_std)


def mean_std_score(
    sub_logits: FloatArray, orig_logits: FloatArray, labels: IntArray, gold: GoldReference
) -> float:
    """1 - clip(d_sub / d_orig, 0, 1) on one group."""
    return ratio_score(
        loss_distance(sub_logits, labels, gold), loss_distance(orig_logits, labels, gold)
    )


def prediction_score(sub_logits: FloatArray, orig_logits: FloatArray, gold: GoldReference) -> float:
    """1 - clip(mean KL(p_gold || p_sub) / mean KL(p_gold || p_orig), 0, 1) on one group."""
    kl_sub = float(per_sample_kl(gold.probabilities, sub_logits).mean())
    kl_orig = float(per_sample_kl(gold.probabilities, orig_logits).mean())
    return ratio_score(kl_sub, kl_orig)


@dataclass(frozen=True)
class ScoreBreakdown:
    """Every component of the official score for one candidate (all floats, JSON friendly)."""

    mean_std_exact: float
    mean_std_altered: float
    prediction_exact: float
    prediction_altered: float
    mean_std_score: float
    prediction_score: float
    forget_score: float
    accuracy_sub_control: float
    accuracy_orig_control: float
    retain_accuracy_score: float
    retain_kl: float
    retain_kl_similarity: float
    retain_score: float
    validation_accuracy: float
    utility_threshold: float
    gate_passed: bool
    final_score: float

    def to_dict(self) -> dict[str, float | bool]:
        """Plain dict for JSON."""
        return asdict(self)


@dataclass(frozen=True)
class ScoringContext:
    """Everything the scorer needs that does not depend on the candidate."""

    orig: SetLogits
    golds: Sequence[SetLogits]
    forget_labels: IntArray
    control_labels: IntArray
    validation_labels: IntArray
    utility_threshold: float
    """Minimum validation accuracy (the real gate is 0.70 on the hidden validation set)."""


def official_score(sub: SetLogits, context: ScoringContext) -> ScoreBreakdown:
    """The exact subject score of `sub` given the original and the gold models.

    Sets expected in every logits mapping: `forget_exact`, `forget_altered` (one altered copy per
    forget image, same order), `control`, `validation`.

    Raises:
        ScoreError: a set is missing or label counts do not match.
    """
    for name in (EXACT_SET, ALTERED_SET, CONTROL_SET, VALIDATION_SET):
        if name not in sub or name not in context.orig:
            raise ScoreError(f"missing logits set '{name}'")
    labels = context.forget_labels
    group_scores: dict[str, tuple[float, float]] = {}
    for name in (EXACT_SET, ALTERED_SET):
        gold = build_gold_reference([gold_set[name] for gold_set in context.golds], labels)
        group_scores[name] = (
            mean_std_score(sub[name], context.orig[name], labels, gold),
            prediction_score(sub[name], context.orig[name], gold),
        )
    mean_std = 0.5 * (group_scores[EXACT_SET][0] + group_scores[ALTERED_SET][0])
    prediction = 0.5 * (group_scores[EXACT_SET][1] + group_scores[ALTERED_SET][1])
    forget = mean_std * prediction

    control = context.control_labels
    acc_orig = accuracy(context.orig[CONTROL_SET], control)
    acc_sub = accuracy(sub[CONTROL_SET], control)
    retain_acc = clip_unit(acc_sub / acc_orig) if acc_orig > 0 else 0.0
    orig_probs = softmax(context.orig[CONTROL_SET], SCORING_TEMPERATURE)
    retain_kl = float(per_sample_kl(orig_probs, sub[CONTROL_SET]).mean())
    retain_similarity = 1.0 / (1.0 + retain_kl)
    retain = retain_acc * retain_similarity

    validation_accuracy = accuracy(sub[VALIDATION_SET], context.validation_labels)
    gate_passed = validation_accuracy >= context.utility_threshold
    return ScoreBreakdown(
        mean_std_exact=group_scores[EXACT_SET][0],
        mean_std_altered=group_scores[ALTERED_SET][0],
        prediction_exact=group_scores[EXACT_SET][1],
        prediction_altered=group_scores[ALTERED_SET][1],
        mean_std_score=mean_std,
        prediction_score=prediction,
        forget_score=forget if gate_passed else 0.0,
        accuracy_sub_control=acc_sub,
        accuracy_orig_control=acc_orig,
        retain_accuracy_score=retain_acc,
        retain_kl=retain_kl,
        retain_kl_similarity=retain_similarity,
        retain_score=retain if gate_passed else 0.0,
        validation_accuracy=validation_accuracy,
        utility_threshold=context.utility_threshold,
        gate_passed=gate_passed,
        final_score=forget * retain if gate_passed else 0.0,
    )


# ---------------------------------------------------------------------------------------------
# Proxy readings of the kit (what a participant can compute WITHOUT gold, labels or hidden sets)
# ---------------------------------------------------------------------------------------------

KIT_FORGET_GROUPS = ("forget_exact", "forget_flip", "forget_crop", "forget_jitter", "forget_jpeg")
KIT_RETAIN_GROUP = "retain_proxy"
PROXY_NAMES = (
    "forget_kl_mean",
    "retain_proxy_kl",
    "movement_ratio",
    "forget_loss_shift",
    "forget_disagreement",
    "kit_composite",
)


def proxy_readings(orig: SetLogits, sub: SetLogits, pseudo_labels: IntArray) -> dict[str, float]:
    """The kit's proxy summary (same definitions as `proxy_eval`), computed from logits.

    Pseudo-labels are the ORIGINAL model's argmax on the exact forget images (the kit has no true
    labels). `kit_composite` is NOT a kit metric: it is the simplest way to turn "move on forget,
    stay on retain" into one number, added so the proxy can be correlated with a single score.

    Returns:
        forget_kl_mean (mean over groups of KL(orig||sub) at T = 2), retain_proxy_kl,
        movement_ratio, forget_loss_shift (mean over groups of |d mean| + |d std| of the
        pseudo-label loss), forget_disagreement (1 - argmax agreement with orig, mean over groups),
        kit_composite.
    """
    kls: list[float] = []
    shifts: list[float] = []
    disagreements: list[float] = []
    for name in KIT_FORGET_GROUPS:
        orig_z, sub_z = orig[name], sub[name]
        orig_probs = softmax(orig_z, SCORING_TEMPERATURE)
        kls.append(float(per_sample_kl(orig_probs, sub_z).mean()))
        loss_orig = cross_entropy(orig_z, pseudo_labels)
        loss_sub = cross_entropy(sub_z, pseudo_labels)
        shifts.append(
            abs(float(loss_sub.mean() - loss_orig.mean()))
            + abs(float(loss_sub.std() - loss_orig.std()))
        )
        disagreements.append(float((orig_z.argmax(axis=1) != sub_z.argmax(axis=1)).mean()))
    retain_kl = float(
        per_sample_kl(
            softmax(orig[KIT_RETAIN_GROUP], SCORING_TEMPERATURE), sub[KIT_RETAIN_GROUP]
        ).mean()
    )
    forget_kl = float(np.mean(kls))
    if retain_kl > 0:
        movement = forget_kl / retain_kl
    else:
        movement = float("inf") if forget_kl > 0 else float("nan")
    return {
        "forget_kl_mean": forget_kl,
        "retain_proxy_kl": retain_kl,
        "movement_ratio": movement,
        "forget_loss_shift": float(np.mean(shifts)),
        "forget_disagreement": float(np.mean(disagreements)),
        "kit_composite": forget_kl / (1.0 + forget_kl) / (1.0 + retain_kl),
    }


# ---------------------------------------------------------------------------------------------
# Rank correlations
# ---------------------------------------------------------------------------------------------


def average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks with ties replaced by their average rank (as scipy.stats.rankdata)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman(x: FloatArray, y: FloatArray) -> float:
    """Spearman rank correlation; nan when either side is constant or fewer than 3 points."""
    if len(x) != len(y):
        raise ScoreError("spearman needs two vectors of the same length")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    rx, ry = average_ranks(x[finite]), average_ranks(y[finite])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: FloatArray, y: FloatArray) -> float:
    """Pearson correlation on the finite pairs; nan when degenerate."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or x[finite].std() == 0 or y[finite].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def bootstrap_spearman(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Percentile bootstrap interval of the Spearman correlation (resampling candidates)."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(x)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = generator.integers(0, count, size=count)
        value = spearman(x[index], y[index])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    tail = 100.0 * (1.0 - CONFIDENCE_LEVEL) / 2.0
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail))


def top_k_overlap(
    x: FloatArray, y: FloatArray, k: int, *, x_higher_is_better: bool = True
) -> float:
    """Fraction of the k best candidates by `y` (higher = better) also among the k best by `x`."""
    if len(x) != len(y) or k < 1 or k > len(x):
        raise ScoreError("top_k_overlap needs equal lengths and 1 <= k <= len")
    x_key = x if x_higher_is_better else -x
    best_x = set(np.argsort(-np.nan_to_num(x_key, nan=-np.inf), kind="mergesort")[:k].tolist())
    best_y = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    return len(best_x & best_y) / k


# ---------------------------------------------------------------------------------------------
# Surrogate gold: a LOCAL stand-in for the gold models, built from the original model only
# ---------------------------------------------------------------------------------------------


def surrogate_reference(
    view_logits: Sequence[FloatArray], pseudo_labels: IntArray, temperature: float
) -> GoldReference:
    """Gold surrogate of the `gold_soft` recipe: tempered softmax of the original on strong views.

    The target is the mean over views of softmax(logits / temperature) (the kit's `gold_soft`
    target). Its loss statistics use the pseudo-labels, its T = 2 distribution is the target
    tempered once more (`softmax(log target / 2)`), exactly as the kit's `_matching_loss` does.

    Raises:
        ScoreError: no view given.
    """
    if not view_logits:
        raise ScoreError("at least one view is required")
    target = np.mean([softmax(z, temperature) for z in view_logits], axis=0)
    safe = np.clip(target, 1e-12, None)
    loss = -np.log(safe[np.arange(len(pseudo_labels)), pseudo_labels])
    tempered = softmax(np.log(safe), SCORING_TEMPERATURE)
    return GoldReference(
        loss_mean=float(loss.mean()), loss_std=float(loss.std()), probabilities=tempered
    )


def surrogate_forget_score(
    sub: SetLogits, orig: SetLogits, pseudo_labels: IntArray, reference: GoldReference
) -> float:
    """Official forget score with the surrogate in place of the gold (exact + altered groups).

    The altered copies are compared with the SAME reference, since an altered copy follows its
    source image (the best a participant can assume without gold models).
    """
    mean_std: list[float] = []
    prediction: list[float] = []
    for name in (EXACT_SET, ALTERED_SET):
        mean_std.append(mean_std_score(sub[name], orig[name], pseudo_labels, reference))
        prediction.append(prediction_score(sub[name], orig[name], reference))
    return float(np.mean(mean_std) * np.mean(prediction))
