"""Tests for wmkit.eval."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
from conftest import load_sample

from wmkit.attacks import get_attack, get_suite
from wmkit.classical.dct_qim import build as build_dct_qim
from wmkit.eval import (
    RESULT_COLUMNS,
    EvaluationError,
    evaluate,
    summarize,
    summarize_by_attack,
)
from wmkit.types import AttackSpec

EXPECTED_COLUMNS = [
    "method", "image_id", "is_watermarked", "attack", "attack_group", "n_bits", "bit_acc",
    "neg_log10_p", "detected", "psnr", "ssim", "lpips",
]
FAKE_BITS = 32  # 32 matching bits give -log10 p = 9.6, above the detection threshold
FAKE_LEVEL_LOW = 0.25
FAKE_LEVEL_SPAN = 0.5


class PixelWatermarker:
    """Fake method: writes the message in the first pixels of the red channel."""

    def __init__(self, name: str = "pixel", n_bits: int = FAKE_BITS) -> None:
        self.name = name
        self.n_bits = n_bits
        self.seen_messages: list[torch.Tensor] = []

    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        self.seen_messages.append(messages.clone())
        marked = images.clone()
        marked[:, 0, 0, : self.n_bits] = FAKE_LEVEL_LOW + FAKE_LEVEL_SPAN * messages
        return marked

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        row = images[:, 0, 0, : self.n_bits]
        return ((row - FAKE_LEVEL_LOW) / FAKE_LEVEL_SPAN).clamp(0.0, 1.0)


class WrongShapeExtractor(PixelWatermarker):
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        return super().extract(images)[:, :-1]


class ResizingEmbedder(PixelWatermarker):
    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        return images[..., :-1]


class RecordingAttack:
    """Deterministic identity attack that records the seeds it is reseeded with."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def reseed(self, seed: int) -> None:
        self.seeds.append(seed)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images


def sample_images(count: int = 2, side: int = 64) -> list[tuple[str, torch.Tensor]]:
    names = ["astronaut", "coffee", "chelsea", "rocket"][:count]
    return [(name, load_sample(name, (side, side))) for name in names]


def attack_specs(*names: str) -> list[AttackSpec]:
    return [get_attack(name) for name in names]


def test_columns_are_exactly_those_of_the_contract() -> None:
    assert RESULT_COLUMNS == EXPECTED_COLUMNS
    frame = evaluate(PixelWatermarker(), sample_images(), attack_specs("identity", "jpeg_50"))
    assert list(frame.columns) == EXPECTED_COLUMNS


def test_one_row_per_image_attack_and_variant() -> None:
    images = sample_images(3)
    attacks = attack_specs("identity", "brightness_1.5", "hflip", "crop_0.5")
    both = evaluate(PixelWatermarker(), images, attacks)
    assert len(both) == 3 * 4 * 2
    assert both["is_watermarked"].sum() == 3 * 4
    only_marked = evaluate(PixelWatermarker(), images, attacks, include_clean=False)
    assert len(only_marked) == 3 * 4
    assert only_marked["is_watermarked"].all()
    assert set(both["attack_group"]) == {"identity", "valuemetric", "geometric"}
    assert set(both["image_id"]) == {"astronaut", "coffee", "chelsea"}


def test_dtypes_and_value_ranges() -> None:
    frame = evaluate(PixelWatermarker(), sample_images(), attack_specs("identity", "jpeg_50"))
    assert frame["is_watermarked"].dtype == bool
    assert frame["detected"].dtype == bool
    assert frame["n_bits"].eq(FAKE_BITS).all()
    assert frame["bit_acc"].between(0, 1).all()
    assert (frame["neg_log10_p"] >= 0).all()
    assert frame["method"].eq("pixel").all()


def test_clean_rows_measure_false_positives_and_marked_rows_detections() -> None:
    frame = evaluate(PixelWatermarker(), sample_images(4), attack_specs("identity"))
    marked = frame[frame["is_watermarked"]]
    clean = frame[~frame["is_watermarked"]]
    assert (marked["bit_acc"] == 1.0).all()
    assert marked["detected"].all()
    assert (marked["neg_log10_p"] > 6.0).all()
    assert not clean["detected"].any()
    assert (clean["bit_acc"] < 1.0).all()


def test_quality_columns_describe_the_watermarked_image_and_repeat_on_every_row() -> None:
    images = sample_images(2)
    frame = evaluate(PixelWatermarker(), images, attack_specs("identity", "hflip"))
    for image_id, _ in images:
        rows = frame[frame["image_id"] == image_id]
        assert rows["psnr"].nunique() == 1
        assert rows["ssim"].nunique() == 1
        assert 0 < rows["psnr"].iloc[0] < 100
        assert 0.9 < rows["ssim"].iloc[0] <= 1.0
    assert frame["lpips"].isna().all()


def test_same_seed_reproduces_and_different_seeds_change_messages() -> None:
    images = sample_images(2)
    attacks = ("identity", "crop_0.71", "gaussian_noise_0.05", "rotate_10")
    first = evaluate(PixelWatermarker(), images, attack_specs(*attacks), seed=3)
    second = evaluate(PixelWatermarker(), images, attack_specs(*attacks), seed=3)
    pd.testing.assert_frame_equal(first, second)
    other = evaluate(PixelWatermarker(), images, attack_specs(*attacks), seed=4)
    assert not first[["bit_acc", "neg_log10_p"]].equals(other[["bit_acc", "neg_log10_p"]])


def test_result_does_not_depend_on_attack_order() -> None:
    images = sample_images(2)
    forward = evaluate(PixelWatermarker(), images, attack_specs("crop_0.5", "rotate_10", "hflip"))
    backward = evaluate(PixelWatermarker(), images, attack_specs("hflip", "rotate_10", "crop_0.5"))
    keys = ["image_id", "attack", "is_watermarked"]
    pd.testing.assert_frame_equal(
        forward.sort_values(keys).reset_index(drop=True),
        backward.sort_values(keys).reset_index(drop=True),
    )


def test_messages_are_shared_across_methods_for_one_seed() -> None:
    first, second = PixelWatermarker("a"), PixelWatermarker("b")
    evaluate(first, sample_images(3), attack_specs("identity"), seed=11)
    evaluate(second, sample_images(3), attack_specs("identity"), seed=11)
    assert len(first.seen_messages) == 3
    for left, right in zip(first.seen_messages, second.seen_messages, strict=True):
        assert torch.equal(left, right)
    assert not torch.equal(first.seen_messages[0], first.seen_messages[1])


def test_watermarked_image_and_clean_twin_get_the_same_attack_seed() -> None:
    recorder = RecordingAttack()
    spec = AttackSpec(name="probe", group="geometric", fn=recorder, differentiable=True)
    evaluate(PixelWatermarker(), sample_images(2), [spec], seed=5)
    assert len(recorder.seeds) == 4  # 2 images x (watermarked, clean)
    assert recorder.seeds[0] == recorder.seeds[1]
    assert recorder.seeds[2] == recorder.seeds[3]
    assert recorder.seeds[0] != recorder.seeds[2]


def test_real_method_on_the_quick_suite() -> None:
    method = build_dct_qim()
    frame = evaluate(method, sample_images(2, side=128), get_suite("quick"))
    assert len(frame) == 2 * 11 * 2
    identity = frame[(frame["attack"] == "identity")]
    marked = identity[identity["is_watermarked"]]
    clean = identity[~identity["is_watermarked"]]
    assert (marked["bit_acc"] >= 0.95).all() and marked["detected"].all()
    assert not clean["detected"].any()
    assert (marked["psnr"] >= 38.0).all()


def test_empty_inputs_give_an_empty_typed_frame() -> None:
    frame = evaluate(PixelWatermarker(), [], attack_specs("identity"))
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert frame.empty


def test_contract_violations_are_reported() -> None:
    images, attacks = sample_images(1), attack_specs("identity")
    with pytest.raises(EvaluationError, match="extract"):
        evaluate(WrongShapeExtractor(), images, attacks)
    with pytest.raises(EvaluationError, match="embed"):
        evaluate(ResizingEmbedder(), images, attacks)
    with pytest.raises(EvaluationError, match="shape"):
        evaluate(PixelWatermarker(), [("bad", torch.rand(1, 64, 64))], attacks)
    with pytest.raises(EvaluationError, match="floating"):
        evaluate(PixelWatermarker(), [("bad", torch.zeros(3, 64, 64, dtype=torch.uint8))], attacks)


def make_frame() -> pd.DataFrame:
    rows = []
    for method, accuracies in {"m": (1.0, 0.5, 0.75, 0.25)}.items():
        for is_watermarked, attack, group, accuracy, detected in [
            (True, "identity", "identity", accuracies[0], True),
            (True, "jpeg_50", "compression", accuracies[1], False),
            (True, "jpeg_70", "compression", accuracies[2], True),
            (False, "jpeg_50", "compression", accuracies[3], False),
        ]:
            rows.append({
                "method": method, "image_id": "x", "is_watermarked": is_watermarked,
                "attack": attack, "attack_group": group, "n_bits": 8, "bit_acc": accuracy,
                "neg_log10_p": accuracy * 10, "detected": detected, "psnr": 40.0, "ssim": 0.9,
                "lpips": float("nan"),
            })
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_summarize_means_per_method_watermark_flag_and_group() -> None:
    summary = summarize(make_frame())
    assert list(summary.columns) == [
        "method", "is_watermarked", "attack_group", "bit_acc", "neg_log10_p", "detected", "psnr",
        "ssim", "lpips",
    ]
    compression = summary[(summary["attack_group"] == "compression") & summary["is_watermarked"]]
    assert len(summary) == 3
    assert compression["bit_acc"].item() == pytest.approx(0.625)
    assert compression["detected"].item() == pytest.approx(0.5)
    assert math.isnan(compression["lpips"].item())
    clean = summary[~summary["is_watermarked"]]
    assert clean["bit_acc"].item() == pytest.approx(0.25)


def test_summarize_by_attack_groups_per_attack_name() -> None:
    summary = summarize_by_attack(make_frame())
    assert list(summary.columns)[:3] == ["method", "is_watermarked", "attack"]
    assert len(summary) == 4
    row = summary[(summary["attack"] == "jpeg_70") & summary["is_watermarked"]]
    assert row["bit_acc"].item() == pytest.approx(0.75)


def test_summaries_reject_foreign_frames() -> None:
    with pytest.raises(EvaluationError):
        summarize(pd.DataFrame({"method": ["m"]}))
    with pytest.raises(EvaluationError):
        summarize_by_attack(pd.DataFrame({"method": ["m"]}))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_evaluate_on_mps_matches_cpu_for_a_classical_method() -> None:
    images, attacks = sample_images(1, side=128), attack_specs("identity", "jpeg_50", "hflip")
    cpu = evaluate(build_dct_qim(), images, attacks, device="cpu")
    mps = evaluate(build_dct_qim(), images, attacks, device="mps")
    assert (cpu["bit_acc"] - mps["bit_acc"]).abs().max() <= 0.1


@pytest.mark.slow
def test_lpips_column_is_filled_on_request() -> None:
    frame = evaluate(PixelWatermarker(), sample_images(1), attack_specs("identity"),
                     with_lpips=True)
    assert frame["lpips"].notna().all()
    assert (frame["lpips"] >= 0).all()
