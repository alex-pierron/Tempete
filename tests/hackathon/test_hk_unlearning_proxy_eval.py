"""Tests of the local proxy metrics (pure maths on logits, plus an end-to-end tiny run)."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.constants import NUM_CLASSES, SCORING_TEMPERATURE
from hackathon.unlearning.proxy_eval import (
    cache_fingerprint,
    compute_group_metrics,
    evaluate_proxy,
    format_report,
    load_logits_cache,
    save_logits_cache,
    tempered_kl,
)

SEED = 7


def _logits(count: int = 16, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, NUM_CLASSES, generator=generator) * 3.0


def test_kl_of_identical_logits_is_zero_and_otherwise_positive() -> None:
    z = _logits()
    assert float(tempered_kl(z, z).abs().max()) < 1e-12
    assert float(tempered_kl(z, _logits(seed=1)).min()) > 0.0


def test_kl_matches_a_hand_computed_two_class_value_at_temperature_two() -> None:
    ref = torch.tensor([[2.0, 0.0]])
    other = torch.tensor([[0.0, 0.0]])
    p = torch.softmax(ref / SCORING_TEMPERATURE, dim=1)
    q = torch.tensor([[0.5, 0.5]])
    expected = float((p * (p / q).log()).sum())
    assert math.isclose(float(tempered_kl(ref, other)), expected, rel_tol=1e-6)


def test_kl_is_asymmetric() -> None:
    a, b = _logits(seed=0), _logits(seed=1) * 0.2
    assert not math.isclose(float(tempered_kl(a, b).mean()), float(tempered_kl(b, a).mean()))


def test_group_metrics_of_identity_have_zero_movement() -> None:
    z = _logits()
    metrics = compute_group_metrics("g", z, z, z.argmax(dim=1))
    assert metrics.loss_shift == 0.0 and metrics.kl_orig_sub < 1e-12
    assert metrics.argmax_agreement == 1.0 and metrics.pseudo_acc_sub == 1.0
    assert metrics.count == 16


def test_group_metrics_flag_a_softened_model() -> None:
    z = _logits()
    labels = z.argmax(dim=1)
    soft = compute_group_metrics("g", z, z / 4.0, labels)
    assert soft.loss_mean_sub > soft.loss_mean_orig
    assert soft.confidence_sub < soft.confidence_orig
    assert soft.argmax_agreement == 1.0  # temperature scaling never changes the ranking
    assert soft.kl_orig_sub > 0.0


def test_group_metrics_loss_std_is_the_population_std() -> None:
    z = _logits(count=5)
    labels = torch.zeros(5, dtype=torch.long)
    metrics = compute_group_metrics("g", z, z, labels)
    losses = torch.nn.functional.cross_entropy(z.double(), labels, reduction="none")
    assert math.isclose(metrics.loss_std_orig, float(losses.std(unbiased=False)))


def test_evaluate_proxy_identity_is_all_zero_and_groups_are_complete() -> None:
    model = tiny_model()
    forget, retain = random_images(4), random_images(3, seed=99)
    report = evaluate_proxy(
        model, copy.deepcopy(model), forget, retain, variants=("flip", "jpeg"), seed=SEED,
        batch_size=2,
    )  # fmt: skip
    names = [g.name for g in report.groups]
    assert names == ["forget_exact", "forget_flip", "forget_jpeg", "retain_proxy"]
    assert report.forget_kl_mean < 1e-9 and report.retain_kl < 1e-9
    assert math.isnan(report.movement_ratio)
    assert [g.count for g in report.groups] == [4, 4, 4, 3]


def test_evaluate_proxy_detects_a_perturbed_model_and_reuses_the_cache() -> None:
    original = tiny_model()
    perturbed = copy.deepcopy(original)
    with torch.no_grad():
        perturbed.head.weight.add_(torch.randn_like(perturbed.head.weight))  # type: ignore[union-attr]
    forget, retain = random_images(4), random_images(3, seed=99)
    cache: dict[str, torch.Tensor] = {}
    first = evaluate_proxy(original, perturbed, forget, retain, variants=("flip",), seed=SEED,
                           original_logits_cache=cache)  # fmt: skip
    assert set(cache) == {"forget_exact", "forget_flip", "retain_proxy"}
    assert first.forget_kl_mean > 0.0 and first.retain_kl > 0.0
    again = evaluate_proxy(original, perturbed, forget, retain, variants=("flip",), seed=SEED,
                           original_logits_cache=cache)  # fmt: skip
    assert again == first


def test_report_serialises_and_formats_without_emoji() -> None:
    model = tiny_model()
    report = evaluate_proxy(model, model, random_images(2), random_images(2, 5), variants=("flip",))
    as_dict = report.to_dict()
    assert len(as_dict["groups"]) == 3  # type: ignore[arg-type]
    text = format_report(report, "title")
    assert text.startswith("title") and "forget_exact" in text and "PROXY ONLY" in text
    assert text.isascii()


def test_cache_round_trip_and_fingerprint_mismatch(tmp_path: Path) -> None:
    forget, retain = random_images(2), random_images(2, 5)
    device = torch.device("cpu")
    fingerprint = cache_fingerprint(forget, retain, ("flip",), SEED, device)
    path = tmp_path / "cache" / "logits.pt"
    assert load_logits_cache(path, fingerprint) == {}
    logits = {"forget_exact": _logits(2)}
    save_logits_cache(path, fingerprint, logits)
    loaded = load_logits_cache(path, fingerprint)
    assert torch.equal(loaded["forget_exact"], logits["forget_exact"])
    other = cache_fingerprint(forget, retain, ("flip", "crop"), SEED, device)
    assert other != fingerprint and load_logits_cache(path, other) == {}
    assert cache_fingerprint(forget, retain, ("flip",), SEED + 1, device) != fingerprint


@pytest.mark.parametrize("retain_kl", [0.0])
def test_movement_ratio_is_inf_when_only_forget_moved(retain_kl: float) -> None:
    from hackathon.unlearning.proxy_eval import _movement_ratio

    assert math.isinf(_movement_ratio(0.5, retain_kl))
    assert math.isnan(_movement_ratio(0.0, retain_kl))
    assert _movement_ratio(1.0, 0.25) == 4.0
