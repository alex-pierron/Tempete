"""Tests of the local bench: arithmetic on hand-made logits, then one end-to-end pass (tiny ViT)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.bench import (
    AUDIT,
    AUDIT_VIEW,
    EXACT,
    GOLD_VIEWS,
    audit_split,
    format_bench,
    run_bench,
    score_from_logits,
)
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.proxy_eval import load_logits_cache

CLASSES = 100


def _peaked(labels: torch.Tensor, margin: float = 20.0) -> torch.Tensor:
    """Memorised-looking logits: the label `margin` above every other class."""
    logits = torch.zeros(len(labels), CLASSES)
    logits[torch.arange(len(labels)), labels] = margin
    return logits


def _setup() -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    generator = torch.Generator().manual_seed(0)
    trained_labels = torch.randint(CLASSES, (10,), generator=generator)
    audit_labels = torch.randint(CLASSES, (5,), generator=generator)
    orig = {
        EXACT: _peaked(trained_labels),
        "forget_flip": _peaked(trained_labels),
        AUDIT: _peaked(audit_labels),
        AUDIT_VIEW: _peaked(audit_labels, margin=5.0),
    }
    gold = {
        key: [torch.randn(10, CLASSES, generator=generator) * 2 for _ in range(GOLD_VIEWS)]
        for key in ("s40", "s50", "m20")
    }
    return orig, gold


def test_identity_scores_exactly_zero_and_keeps_full_retain() -> None:
    orig, gold = _setup()
    report = score_from_logits(orig, orig, gold)
    assert report["score_min"] == 0.0 and report["forget"]["s50"] == 0.0  # subject: sub = orig
    assert report["score_min_family"] == {"crop": 0.0, "mask": 0.0}
    assert report["retain"] == 1.0 and report["gate_passed"] == 1.0
    assert report["groups"][AUDIT]["kl_orig_sub"] == 0.0 and report["audit_view_kl"] == 0.0


def test_matching_the_gold_distribution_scores_its_prediction_fully() -> None:
    orig, gold = _setup()
    target = torch.stack([torch.softmax(v / SCORING_TEMPERATURE, dim=1) for v in gold["s50"]]).mean(
        dim=0
    )
    matched = SCORING_TEMPERATURE * target.log()  # softmax(matched / T) == target exactly
    sub = {EXACT: matched, "forget_flip": matched, AUDIT: orig[AUDIT], AUDIT_VIEW: orig[AUDIT_VIEW]}
    report = score_from_logits(orig, sub, gold)
    assert report["prediction"]["s50"] == pytest.approx(1.0)  # KL to that gold is 0
    assert report["prediction"]["s50"] > report["prediction"]["s40"]
    assert report["exact"]["prediction"]["s50"] == pytest.approx(1.0)  # exact images alone
    assert set(report["exact"]["mean_std"]) == set(report["mean_std"])
    assert 0.0 < report["score"]["s50"] <= 1.0
    assert report["score_min"] == min(report["score"].values())
    assert report["score_min_family"]["mask"] == report["score"]["m20"]


def test_the_audit_view_drift_is_reported_outside_the_score() -> None:
    orig, gold = _setup()
    drifted = {**orig, AUDIT_VIEW: orig[AUDIT_VIEW].roll(1, dims=1)}
    report = score_from_logits(orig, drifted, gold)
    assert report["audit_view_kl"] > 0.0
    assert report["score_min"] == 0.0 and report["retain"] == 1.0  # not part of the score


def test_the_gate_zeroes_every_score() -> None:
    orig, gold = _setup()
    wrong = orig[AUDIT].roll(1, dims=1)  # every audit argmax moved: agreement 0
    report = score_from_logits(orig, {**orig, AUDIT: wrong}, gold)
    assert report["gate_passed"] == 0.0
    assert all(value == 0.0 for value in report["score"].values())


def test_run_bench_end_to_end_and_cache(tmp_path: Path) -> None:
    original = tiny_model()
    forget = random_images(10)
    kept, held = audit_split(original, forget, 0.2, batch_size=4)
    assert len(held) == 2 and len(kept) == 8
    cache = tmp_path / "bench.pt"
    candidate = tiny_model(seed=1)
    first = run_bench(original, candidate, forget[kept], forget[held], batch_size=4,
                      cache_path=cache)  # fmt: skip
    assert cache.is_file() and str(first["protocol"]).startswith("bench-v4|trained=8|audit=2")
    assert set(first["score"]) == {"s40", "s50", "s60", "m30", "m20", "m15"}  # 6 gold stand-ins
    assert first["protocol_version"] == 4  # numeric: reaches the registry and the scoreboard
    assert first["score_min"] == min(first["score_min_family"].values())
    assert first["audit_view_kl"] > 0.0 and "audit view KL" in format_bench(first)
    second = run_bench(original, candidate, forget[kept], forget[held], batch_size=4,
                       cache_path=cache)  # fmt: skip
    assert second["score"] == first["score"]
    identity = run_bench(original, original, forget[kept], forget[held], batch_size=4,
                         cache_path=cache)  # fmt: skip
    assert identity["score_min"] == 0.0 and identity["audit_view_kl"] == 0.0
    cache.write_bytes(b"half-written")  # a crashed writer must not break the next reader
    assert load_logits_cache(cache, "any") == {}
    with pytest.raises(ValueError, match="holdout_fraction"):
        run_bench(original, candidate, forget, forget[:0])
