"""Local bench: one fixed, versioned protocol that turns a candidate into an auditable score.

The server score is `forget x retain` against gold models we do not have. The bench keeps that
shape and replaces each missing piece by a DECLARED stand-in, so every number can be recomputed
and challenged:

* audit fold: a fixed stratified share of the forget set (`audit_split`, seed AUDIT_SEED) that
  tuning runs never train on. In the distribution and memorised by the original, like the images
  a gold model trained on; it plays the control set. Labels = argmax of the original.
* retain = clip(agreement with the original on the audit fold) / (1 + KL(orig || sub) at T = 2),
  the server formula; gate: agreement < 70 % gives a score of 0 (the server's accuracy gate).
* gold stand-ins: the original on GOLD_VIEWS perturbed views of each trained forget image, i.e.
  off its memorised pixels, in TWO families: strong views (crop + colour jitter, keys sXX) and
  16x16 patch masking (keys mXX). A gold model treats a forget image like an unseen one, so its
  accuracy there is its test accuracy: unknown, but above the 70 % gate. Each family has three
  strengths, calibrated on the real model to bracket a plausible gold accuracy (GOLD_STRENGTHS,
  MASK_FRACTIONS). Known bias: the crops favour methods that train toward such views
  (gold_soft, trust_region views/calibrated); the masks are a family no method trains on.
* drift on "as if unseen" images: KL(orig || sub) at T = 2 on one strong view (area
  AUDIT_VIEW_AREA, fixed seed) of each audit image, reported as `audit_view_kl`, outside the
  score (the audit fold itself is memorised, so it may under-state the drift on unseen images).
* forget = mean_std x prediction, each averaged over the exact images and their altered copies
  (the stand-in of an altered copy is the one of its source image), with the EXACT formulas of
  the subject (Sujets/Unlearning.pdf): mean_std = 1 - clip(d_sub / d_orig, 0, 1) where
  d = |mean(L) - mean(L_gold)| + |std(L) - std(L_gold)| (losses at T = 1, here on the
  pseudo-labels), and prediction = 1 - clip(mean KL(p_gold || p_sub) / mean KL(p_gold || p_orig),
  0, 1) at T = 2. Both are normalised by the original: the identity scores exactly 0.

Tracked number: `score_min` = the server-shaped score under the WORST of the six gold
stand-ins, so it only rises when a candidate improves under every gold hypothesis of both
families; `score_min_family` gives the worst case of each family. Scores are comparable only
under the same `protocol` string.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.alterations import (
    ALTERATIONS,
    apply_alteration,
    mask_patches,
    strong_view,
)
from hackathon.unlearning.constants import DEFAULT_BATCH_SIZE, SCORING_TEMPERATURE
from hackathon.unlearning.data import stratified_split
from hackathon.unlearning.model import model_device, predict
from hackathon.unlearning.proxy_eval import (
    cache_fingerprint,
    compute_group_metrics,
    load_logits_cache,
    save_logits_cache,
    tempered_kl,
)

PROTOCOL_NUMBER = 4
PROTOCOL_VERSION = f"bench-v{PROTOCOL_NUMBER}"
"""`report["protocol_version"]` carries the number, so the registry and the scoreboard keep
scores of different protocols apart."""
AUDIT_SEED = 31
"""Seed of the audit fold, of every alteration and gold view: never taken from a run's config."""
GOLD_STRENGTHS = (0.40, 0.50, 0.60)
"""Crop area (lower bound, +0.1 spread) of the gold views. Measured on the real model (256 forget
images, 2026-11-21): argmax agreement with the exact image 73 % at 0.35, 83 % at 0.45, 89 % at
0.55; so these three stand for a gold of roughly 78, 86 and 92 % accuracy on the forget set."""
MASK_FRACTIONS = (0.30, 0.20, 0.15)
"""Share of 16x16 patches grayed out (`mask_patches`) in the second family, ordered like
GOLD_STRENGTHS (least to most accurate). Measured on the real model (256 forget images, 4 views,
Apple M2, 2026-11-21): argmax agreement 0.944 at 0.10, 0.917 at 0.15, 0.856 at 0.20, 0.766 at
0.30; 0.661 at 0.40 (research note, per-patch Bernoulli mask), out of range. These three bracket
about the same 77-92 % as the crops."""
FAMILY_PREFIXES = {"crop": "s", "mask": "m"}
"""Gold stand-in family -> prefix of its keys (s40 ... crops, m30 ... masks)."""
GOLD_VIEWS = 4
GATE_AGREEMENT = 0.70
RATIO_FLOOR = 1e-12
EXACT = "forget_exact"
AUDIT = "audit"
AUDIT_VIEW = "audit_view"
AUDIT_VIEW_AREA = 0.50
AUDIT_VIEW_SEED = AUDIT_SEED * 1000 + 900
"""One strong view of each audit image, apart from the gold views' seeds (31000-31503)."""
HELD_OUT = (AUDIT, AUDIT_VIEW)
"""Groups outside the forget score: their labels are their own original argmax."""
ALTERED_PREFIX = "forget_"


def audit_split(
    original: nn.Module, forget: Tensor, fraction: float, batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[Tensor, Tensor]:
    """(kept, audit) indices of `forget`, stratified by the original's argmax, fixed seed."""
    labels = predict(original, forget, batch_size).argmax(dim=1)
    return stratified_split(labels, fraction, AUDIT_SEED)


def protocol_id(trained: int, audit: int) -> str:
    """Everything a bench number depends on besides the two models."""
    strengths = "-".join(f"{s:g}" for s in GOLD_STRENGTHS)
    masks = "-".join(f"{f:g}" for f in MASK_FRACTIONS)
    return (
        f"{PROTOCOL_VERSION}|trained={trained}|audit={audit}|seed={AUDIT_SEED}|views={GOLD_VIEWS}"
        f"|strengths={strengths}|masks={masks}|audit_view={AUDIT_VIEW_AREA:g}"
        f"|alterations={'-'.join(ALTERATIONS)}"
    )


GoldView = Callable[[Tensor, torch.Generator, float], Tensor]


def _gold_stand_ins() -> list[tuple[str, GoldView, float]]:
    """(key, view function, strength) of the six gold stand-ins, crops first."""
    crops: list[tuple[str, GoldView, float]] = [
        (f"s{round(s * 100):02d}", strong_view, s) for s in GOLD_STRENGTHS
    ]
    return crops + [(f"m{round(f * 100):02d}", mask_patches, f) for f in MASK_FRACTIONS]


def _losses(logits: Tensor, labels: Tensor) -> Tensor:
    return torch_f.cross_entropy(logits.double(), labels, reduction="none")


def _stats(losses: Tensor) -> tuple[float, float]:
    return float(losses.mean()), float(losses.std(unbiased=False))


def _distance(stats: tuple[float, float], gold: tuple[float, float]) -> float:
    """d = |mean(L) - mean(L_gold)| + |std(L) - std(L_gold)| (subject, loss-distribution score)."""
    return abs(stats[0] - gold[0]) + abs(stats[1] - gold[1])


def _one_minus_clipped_ratio(numerator: float, denominator: float) -> float:
    return 1.0 - min(1.0, max(0.0, numerator / max(denominator, RATIO_FLOOR)))


def _mean_std_score(
    sub_losses: Tensor, orig_losses: Tensor, gold_stats: tuple[float, float]
) -> float:
    """Subject: 1 - clip(d_sub / d_orig, 0, 1)."""
    return _one_minus_clipped_ratio(
        _distance(_stats(sub_losses), gold_stats), _distance(_stats(orig_losses), gold_stats)
    )


def _mean_kl(gold_probs: Tensor, logits: Tensor) -> float:
    """mean_x KL(p_gold(x) || p_m(x)), p_m = softmax(logits / 2)."""
    log_m = torch_f.log_softmax(logits.double() / SCORING_TEMPERATURE, dim=1)
    gold = gold_probs.double()
    return float((gold * (gold.clamp_min(1e-12).log() - log_m)).sum(dim=1).mean())


def _prediction_score(gold_probs: Tensor, sub_logits: Tensor, orig_logits: Tensor) -> float:
    """Subject: 1 - clip(mean KL(p_gold || p_sub) / mean KL(p_gold || p_orig), 0, 1)."""
    return _one_minus_clipped_ratio(
        _mean_kl(gold_probs, sub_logits), _mean_kl(gold_probs, orig_logits)
    )


def _halves(exact: float, altered: Sequence[float]) -> float:
    """Mean of the exact-image value and of the altered copies' mean (the server's two halves)."""
    return 0.5 * (exact + (sum(altered) / len(altered) if altered else exact))


def _scores(
    orig: Mapping[str, Tensor],
    sub: Mapping[str, Tensor],
    gold_views: Mapping[str, Sequence[Tensor]],
) -> dict[str, object]:
    labels = orig[EXACT].argmax(dim=1)
    agreement = float((sub[AUDIT].argmax(dim=1) == orig[AUDIT].argmax(dim=1)).double().mean())
    audit_kl = float(tempered_kl(orig[AUDIT], sub[AUDIT]).mean())
    retain = min(1.0, max(0.0, agreement)) / (1.0 + audit_kl)
    gate = float(agreement >= GATE_AGREEMENT)
    groups = [name for name in sub if name not in HELD_OUT]  # EXACT, then the altered copies
    score: dict[str, float] = {}
    forget: dict[str, float] = {}
    mean_std: dict[str, float] = {}
    prediction: dict[str, float] = {}
    # The exact images alone (not in the score): a method trained on the bench's alteration
    # families (scrub's max-steps draw from them) sees part of the altered half before scoring.
    exact_only: dict[str, dict[str, float]] = {"mean_std": {}, "prediction": {}}
    for key, views in gold_views.items():
        # Several gold models: mean and std of their losses averaged model by model (subject).
        per_view = [_stats(_losses(view, labels)) for view in views]
        gold_stats = (
            sum(m for m, _ in per_view) / len(per_view),
            sum(s for _, s in per_view) / len(per_view),
        )
        gold_probs = torch.stack(
            [torch.softmax(view.double() / SCORING_TEMPERATURE, dim=1) for view in views]
        ).mean(dim=0)
        ms = {
            g: _mean_std_score(_losses(sub[g], labels), _losses(orig[g], labels), gold_stats)
            for g in groups
        }
        pred = {g: _prediction_score(gold_probs, sub[g], orig[g]) for g in groups}
        mean_std[key] = _halves(ms[EXACT], [ms[g] for g in groups if g != EXACT])
        prediction[key] = _halves(pred[EXACT], [pred[g] for g in groups if g != EXACT])
        exact_only["mean_std"][key], exact_only["prediction"][key] = ms[EXACT], pred[EXACT]
        forget[key] = mean_std[key] * prediction[key]
        score[key] = gate * forget[key] * retain
    return {
        "score": score, "forget": forget, "mean_std": mean_std, "prediction": prediction,
        "retain": retain, "audit_agreement": agreement, "audit_kl": audit_kl, "gate_passed": gate,
        "exact": exact_only,
    }  # fmt: skip


def score_from_logits(
    orig: Mapping[str, Tensor],
    sub: Mapping[str, Tensor],
    gold_views: Mapping[str, Sequence[Tensor]],
) -> dict[str, object]:
    """Bench report from logits: groups EXACT, AUDIT, AUDIT_VIEW and altered copies; gold views
    by stand-in key (sXX crops, mXX masks).

    Pure function (no model), so the arithmetic is tested on hand-made logits. With the
    subject's normalisation by the original, the identity scores exactly 0.
    """
    report = _scores(orig, sub, gold_views)
    score: dict[str, float] = report["score"]  # type: ignore[assignment]
    labels = {name: orig[EXACT].argmax(dim=1) for name in sub if name not in HELD_OUT}
    labels |= {name: orig[name].argmax(dim=1) for name in HELD_OUT}
    groups = {
        name: {k: v for k, v in asdict(compute_group_metrics(name, orig[name], logits,
                                                             labels[name])).items() if k != "name"}
        for name, logits in sub.items()
    }  # fmt: skip
    forget_kl = [groups[name]["kl_orig_sub"] for name in groups if name not in HELD_OUT]
    audit_kl = float(report["audit_kl"])  # type: ignore[arg-type]
    worst: dict[str, float] = {}
    for family, prefix in FAMILY_PREFIXES.items():
        values = [value for key, value in score.items() if key.startswith(prefix)]
        if values:
            worst[family] = min(values)
    report.update(
        score_min=min(score.values()),
        score_min_family=worst,
        groups=groups,
        audit_view_kl=groups[AUDIT_VIEW]["kl_orig_sub"],
        movement_ratio=(sum(forget_kl) / len(forget_kl)) / max(audit_kl, RATIO_FLOOR),
    )
    return report


def _iter_groups(trained: Tensor, audit: Tensor) -> Iterator[tuple[str, Tensor]]:
    """Lazily, so only one altered copy of the trained images lives in memory."""
    yield EXACT, trained
    yield AUDIT, audit
    view_generator = torch.Generator().manual_seed(AUDIT_VIEW_SEED)
    yield AUDIT_VIEW, strong_view(audit, view_generator, AUDIT_VIEW_AREA)
    for offset, name in enumerate(ALTERATIONS, start=1):
        yield ALTERED_PREFIX + name, apply_alteration(name, trained, AUDIT_SEED + offset)


def gold_view_logits(
    original: nn.Module,
    trained: Tensor,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache: dict[str, Tensor] | None = None,
) -> dict[str, list[Tensor]]:
    """The original's logits on the GOLD_VIEWS views of `trained` of each gold stand-in, by key
    (sXX crops, mXX masks), fixed seeds; entries `gold_<key>_<view>` of `cache` are reused and
    filled in place."""
    cache = {} if cache is None else cache
    gold: dict[str, list[Tensor]] = {}
    for index, (key, draw, strength) in enumerate(_gold_stand_ins()):
        for view in range(GOLD_VIEWS):
            name = f"gold_{key}_{view}"
            if name not in cache:
                generator = torch.Generator().manual_seed(AUDIT_SEED * 1000 + index * 100 + view)
                cache[name] = predict(original, draw(trained, generator, strength), batch_size)
        gold[key] = [cache[f"gold_{key}_{view}"] for view in range(GOLD_VIEWS)]
    return gold


def run_bench(
    original: nn.Module,
    candidate: nn.Module,
    trained: Tensor,
    audit: Tensor,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: Path | None = None,
) -> dict[str, object]:
    """Bench report of `candidate`, trained on `trained` with `audit` held out (uint8 NHWC).

    Images are moved to the original's device, so alterations and gold views run on the GPU. The
    original's logits (groups and gold views) are cached in `cache_path` per protocol and data:
    re-benching another candidate only runs the candidate (exact, audit, audit view, altered).
    """
    if len(trained) == 0 or len(audit) == 0:
        raise ValueError("The bench needs trained AND audit images (holdout_fraction > 0).")
    device = model_device(original)
    protocol = protocol_id(len(trained), len(audit))
    data = cache_fingerprint(trained, audit, tuple(ALTERATIONS), AUDIT_SEED, device)
    fingerprint = hashlib.sha256(f"{protocol}|{data}".encode()).hexdigest()[:16]
    cache = load_logits_cache(cache_path, fingerprint) if cache_path else {}
    known = set(cache)
    trained, audit = trained.to(device), audit.to(device)
    sub: dict[str, Tensor] = {}
    for name, images in _iter_groups(trained, audit):
        if name not in cache:
            cache[name] = predict(original, images, batch_size)
        sub[name] = predict(candidate, images, batch_size)
    gold = gold_view_logits(original, trained, batch_size, cache)
    if cache_path and set(cache) != known:
        save_logits_cache(cache_path, fingerprint, cache)
    report = score_from_logits({name: cache[name] for name in sub}, sub, gold)
    report["protocol"] = protocol
    report["protocol_version"] = PROTOCOL_NUMBER
    report["trained_count"], report["audit_count"] = len(trained), len(audit)
    return report


def format_bench(report: Mapping[str, object]) -> str:
    """Short text summary of a bench report."""
    scores = "  ".join(f"{k}={v:.4f}" for k, v in report["score"].items())  # type: ignore[attr-defined]
    forget = "  ".join(f"{k}={v:.4f}" for k, v in report["forget"].items())  # type: ignore[attr-defined]
    family = ", ".join(f"{k} {v:.4f}" for k, v in report["score_min_family"].items())  # type: ignore[attr-defined]
    return (
        f"bench {report['protocol']}\n"
        f"score_min {report['score_min']:.4f} (worst per family: {family})\n"
        f"score: {scores}\nforget: {forget}\n"
        f"retain {report['retain']:.4f} (audit agreement {report['audit_agreement']:.3f}, "
        f"KL {report['audit_kl']:.4f}, gate {'ok' if report['gate_passed'] else 'FAILED'}) | "
        f"audit view KL {report['audit_view_kl']:.4f} (not scored) | "
        f"movement ratio {report['movement_ratio']:.1f}\n"
        "LOCAL BENCH: subject formulas; gold = original on strong views (sXX) and patch masks "
        "(mXX), pseudo-labels."
    )
