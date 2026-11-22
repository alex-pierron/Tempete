"""Tests of the batch-scale memory lever and the out-of-memory fallback of the pool workers."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch

from hackathon.unlearning.research import a3_generate
from hackathon.unlearning.research.a3_generate import MIN_BATCH, run_pool, scale_variant
from hackathon.unlearning.research.a3_spec import MAX_BATCH, VariantSpec

STEPS = 400
LEARNING_RATE = 4e-5


def _variant(**changes: object) -> VariantSpec:
    base = {"name": "v", "steps": STEPS, "lr": LEARNING_RATE, "batch_size": 16,
            "retain_batch_size": 16}  # fmt: skip
    return VariantSpec(**{**base, **changes})  # type: ignore[arg-type]


def test_scale_one_is_the_variant_itself() -> None:
    variant = _variant()
    assert scale_variant(variant, 1.0) is variant


def test_scale_keeps_the_epochs_and_applies_the_square_root_rule() -> None:
    scaled = scale_variant(_variant(), 2.0)
    assert (scaled.batch_size, scaled.retain_batch_size) == (32, 32)
    assert scaled.steps == STEPS // 2
    assert scaled.lr == pytest.approx(LEARNING_RATE * math.sqrt(2.0))
    assert scaled.batch_size * scaled.steps >= 16 * STEPS  # never fewer samples than the base


def test_scale_down_doubles_the_steps_and_lowers_the_learning_rate() -> None:
    scaled = scale_variant(_variant(), 0.5)
    assert scaled.batch_size == 8 and scaled.steps == 2 * STEPS
    assert scaled.lr == pytest.approx(LEARNING_RATE / math.sqrt(2.0))


def test_scale_is_clipped_to_the_batch_limits_and_the_budget_follows_the_real_batch() -> None:
    big = scale_variant(_variant(), 100.0)
    assert big.batch_size == MAX_BATCH and big.retain_batch_size == MAX_BATCH
    assert big.steps == math.ceil(STEPS / (MAX_BATCH / 16))
    small = scale_variant(_variant(), 0.001)
    assert small.batch_size == MIN_BATCH == small.retain_batch_size
    assert small.steps == STEPS * (16 // MIN_BATCH)


def test_a_variant_can_opt_out_of_the_batch_scale() -> None:
    variant = _variant(no_batch_scale=True)
    assert scale_variant(variant, 4.0) is variant


def test_scale_never_exceeds_the_learning_rate_limit() -> None:
    assert scale_variant(_variant(lr=9e-4), 4.0).lr <= 1e-3


def _write_spec(path: Path, *names: str) -> None:
    variants = [{"name": name, "steps": STEPS, "lr": LEARNING_RATE} for name in names]
    path.write_text(json.dumps({"generation": "t", "variants": variants}), encoding="utf-8")


def test_pool_halves_the_batch_after_an_out_of_memory_and_keeps_the_lower_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "spec.json"
    _write_spec(spec, "first", "second")
    seen: list[tuple[str, int, int]] = []

    def fake_run(variant: VariantSpec, *args: object) -> Path:
        seen.append((variant.name, variant.batch_size, variant.steps))
        if variant.name == "first" and variant.batch_size == 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return tmp_path / "result.json"

    monkeypatch.setattr(a3_generate, "run_variant", fake_run)
    monkeypatch.setattr(a3_generate, "build_view_probe", lambda *a, **k: object())
    done = run_pool(spec, None, tmp_path, time.time() + 10_000, "tag", "gpu0", batch_scale=4.0)  # type: ignore[arg-type]
    assert done == 2
    assert seen == [
        ("first", 64, STEPS // 4),  # scale 4: out of memory
        ("first", 32, STEPS // 2),  # retried at half the scale
        ("second", 32, STEPS // 2),  # the reduced scale stays for the next variant
    ]


def test_pool_gives_up_on_a_variant_that_does_not_fit_at_the_minimum_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "spec.json"
    _write_spec(spec, "huge", "next")

    def fake_run(variant: VariantSpec, *args: object) -> Path:
        if variant.name == "huge":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return tmp_path / "result.json"

    monkeypatch.setattr(a3_generate, "run_variant", fake_run)
    monkeypatch.setattr(a3_generate, "build_view_probe", lambda *a, **k: object())
    done = run_pool(spec, None, tmp_path, time.time() + 10_000, "tag", "gpu0", batch_scale=1.0)  # type: ignore[arg-type]
    assert done == 1  # "huge" is abandoned after the halvings, "next" still runs
