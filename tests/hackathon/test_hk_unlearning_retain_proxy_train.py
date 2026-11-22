"""Tests of dataset assembly and fine-tuning of the generated retain proxy (tiny ViT, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.model import build_model
from hackathon.unlearning.research.retain_proxy.concepts import sister_classes
from hackathon.unlearning.research.retain_proxy.finetune import (
    FinetuneConfig,
    FinetuneError,
    ProxyData,
    augment_batch,
    forget_statistics,
    holdout_statistics,
    run_finetune,
)


def test_augment_batch_is_deterministic_and_bounded() -> None:
    images = torch.randint(0, 256, (6, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8)
    first = augment_batch(images, torch.Generator().manual_seed(1), 0.5)
    second = augment_batch(images, torch.Generator().manual_seed(1), 0.5)
    other = augment_batch(images, torch.Generator().manual_seed(2), 0.5)
    assert first.shape == (6, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(first, second) and not torch.equal(first, other)
    assert first.min() >= -1.0 and first.max() <= 1.0


def test_forget_statistics_identity_and_shift() -> None:
    original = np.random.default_rng(0).normal(size=(40, NUM_CLASSES)) * 5
    same = forget_statistics(original, original)
    assert same.agreement_with_original == 1.0 and same.kl_original_to_model_t2 == pytest.approx(
        0.0, abs=1e-9
    )
    flat = forget_statistics(np.zeros_like(original), original)
    assert (
        flat.mean_confidence == pytest.approx(1 / NUM_CLASSES) and flat.kl_original_to_model_t2 > 0
    )
    top = original.argmax(axis=1)
    pushed = original.copy()
    pushed[np.arange(len(top)), top] -= 50.0  # the original's class is removed everywhere
    assert forget_statistics(pushed, original).agreement_with_original == 0.0


def test_forget_statistics_loss_matches_the_cross_entropy_definition() -> None:
    original = np.zeros((4, NUM_CLASSES))
    original[np.arange(4), [1, 6, 7, 8]] = 20.0  # classes 6, 7, 8 are fully forgotten, 1 is not
    same = forget_statistics(original, original)
    assert same.loss_mean < 1e-6 and same.loss_std < 1e-6
    assert same.loss_target_distance == pytest.approx(2.5 + 3.0, abs=1e-3)
    uniform = forget_statistics(np.zeros_like(original), original)
    assert uniform.loss_mean == pytest.approx(np.log(NUM_CLASSES)) and uniform.loss_std < 1e-9
    split = original.copy()
    split[1:, :] = 0.0  # the three forgotten-class images become uniform, the first stays sharp
    stats = forget_statistics(split, original)
    assert stats.loss_mean_partial_classes < 1e-6
    assert stats.loss_mean_forgotten_classes == pytest.approx(np.log(NUM_CLASSES))
    assert stats.loss_std > 1.0


def test_sister_classes_and_routing_statistics() -> None:
    assert sister_classes(0) == [1, 2, 3, 4]
    assert (
        sister_classes(29) == [32]
        and sister_classes(6) == [5]
        and sister_classes(38) == [34, 35, 36, 37]
    )
    assert sister_classes(70) == [] and sister_classes(1) == [2, 3, 4], "class 0 was forgotten"
    original = np.zeros((4, NUM_CLASSES))
    original[np.arange(4), [6, 6, 29, 38]] = 20.0  # four images of removed classes
    routed = np.zeros_like(original)
    routed[np.arange(4), [5, 5, 32, 34]] = 20.0  # every one answered by a sister class
    stats = forget_statistics(routed, original)
    assert stats.sister_mass_forgotten_images == pytest.approx(1.0, abs=1e-6)
    assert stats.sister_argmax_share_forgotten_images == 1.0
    elsewhere = np.zeros_like(original)
    elsewhere[:, 90] = 20.0
    assert forget_statistics(elsewhere, original).sister_argmax_share_forgotten_images == 0.0
    # Identity: the model equals the original, so its distance to the target is the original's own.
    assert forget_statistics(original, original).mean_std_score_estimate == pytest.approx(
        0.0, abs=1e-6
    )
    # A model with a heavy-tailed loss close to (2.5, 3.0) scores high.
    heavy = np.zeros((100, NUM_CLASSES))
    heavy_original = np.zeros((100, NUM_CLASSES))
    heavy_original[np.arange(100), 1] = 20.0  # class 1: a partial class
    heavy[:, 1] = 20.0
    heavy[:30, 1] = 20.0 - 8.0  # 30 images get a loss of roughly 8 against the 99 other classes
    estimate = forget_statistics(heavy, heavy_original).mean_std_score_estimate
    assert 0.0 <= estimate <= 1.0


def test_group_routing_entropy_and_tail_statistics() -> None:
    original = np.zeros((4, NUM_CLASSES))
    original[np.arange(4), [6, 7, 8, 29]] = 20.0
    routed = np.zeros_like(original)
    routed[np.arange(4), [5, 5, 5, 32]] = 20.0
    stats = forget_statistics(routed, original)
    assert stats.lantern_to_5_argmax_share == 1.0 and stats.boats_to_32_argmax_share == 1.0
    assert stats.lantern_to_5_mass == pytest.approx(1.0, abs=1e-6)
    assert stats.mean_entropy_t2 < 0.1 and stats.tail_mass_beyond_top5_t1 < 1e-6
    flat = forget_statistics(np.zeros_like(original), original)
    assert flat.mean_entropy_t2 == pytest.approx(np.log(NUM_CLASSES))
    assert flat.tail_mass_beyond_top5_t1 == pytest.approx(1.0 - 5.0 / NUM_CLASSES)
    assert flat.lantern_to_5_mass == pytest.approx(1.0 / NUM_CLASSES)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(embed_dim=32, depth=2, num_heads=2)


def test_run_finetune_end_to_end_on_a_tiny_vit(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    dataset = tmp_path / "proxy.npz"
    np.savez(
        dataset,
        images=rng.integers(0, 255, (8, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        labels=np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64),
    )
    base = _tiny_model()
    before = {key: value.clone() for key, value in base.state_dict().items()}
    forget = torch.from_numpy(rng.integers(0, 255, (6, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    config = FinetuneConfig(
        name="tiny",
        dataset_path=dataset,
        original_path=tmp_path / "unused.pt",
        forget_path=tmp_path / "unused.npz",
        out_dir=tmp_path / "out",
        lr=1e-3,
        epochs=2,
        effective_batch=4,
        kd_weight=0.5,
        start_micro_batch=2,
        device="cpu",
        save_model=True,
        export_logits_dir=tmp_path / "team_logits",
        export_alphas=(1.0, 1.1),
        export_subset_size=4,
        holdout_fraction=0.25,
    )
    result = run_finetune(config, original=base, forget_images=forget)
    history = result["history"]
    assert isinstance(history, list) and len(history) == 2
    exported = result["exported_logits"]
    assert isinstance(exported, list) and len(exported) == 2
    assert all("holdout" in entry for entry in history), "2 of the 8 proxy images are held out"
    blends = result["blends"]
    assert isinstance(blends, list) and [b["alpha"] for b in blends] == [1.0, 1.1]
    assert all({"forget", "holdout"} <= set(b) for b in blends)
    assert (tmp_path / "out" / "tiny" / "blends.json").is_file()
    with np.load(tmp_path / "team_logits" / "d_tiny_a1.10.npz") as team_c:
        assert team_c["subset_indices"].shape == (4,)
        assert team_c["logits_strong_view"].shape == (4, NUM_CLASSES)
        assert team_c["feats_exact"].shape == (4, 32) and team_c["feats_exact"].dtype == np.float16
    run_dir = tmp_path / "out" / "tiny"
    assert (run_dir / "history.json").is_file() and (run_dir / "model.pt").is_file()
    with np.load(run_dir / "forget_logits.npz") as archive:
        assert archive["model"].shape == (6, NUM_CLASSES) == archive["original"].shape
    assert json.loads((run_dir / "config.json").read_text())["name"] == "tiny"
    unchanged = all(torch.equal(before[key], value) for key, value in base.state_dict().items())
    assert unchanged, "the original module handed in must not be modified"
    trained = torch.load(run_dir / "model.pt", weights_only=True)
    assert any(not torch.equal(before[key], trained[key]) for key in before), "weights must move"


def test_proxy_data_rejects_bad_files(tmp_path: Path) -> None:
    with pytest.raises(FinetuneError):
        ProxyData(tmp_path / "missing.npz", torch.device("cpu"))
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        images=np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8),
        labels=np.array([0, NUM_CLASSES], dtype=np.int64),
    )
    with pytest.raises(FinetuneError):
        ProxyData(bad, torch.device("cpu"))


def test_holdout_split_is_seeded_and_disjoint() -> None:
    images = torch.randint(0, 255, (10, 4, 4, 3), dtype=torch.uint8)
    labels = torch.arange(10)
    first = ProxyData.__new__(ProxyData)
    first.images, first.labels = images.clone(), labels.clone()
    second = ProxyData.__new__(ProxyData)
    second.images, second.labels = images.clone(), labels.clone()
    _, labels_a = first.hold_out(0.3, seed=7)
    _, labels_b = second.hold_out(0.3, seed=7)
    assert torch.equal(labels_a, labels_b) and len(labels_a) == 3 and len(first) == 7
    assert set(labels_a.tolist()).isdisjoint(first.labels.tolist())
    assert sorted(labels_a.tolist() + first.labels.tolist()) == list(range(10))
    none_held, _ = first.hold_out(0.0, seed=7)
    assert len(none_held) == 0 and len(first) == 7


def test_holdout_statistics_identity_and_drift() -> None:
    original = np.random.default_rng(0).normal(size=(30, NUM_CLASSES)) * 6
    labels = original.argmax(axis=1)
    same = holdout_statistics(original, original, labels)
    assert same.kl_original_to_model_t2 == pytest.approx(0.0, abs=1e-9)
    assert same.accuracy == 1.0 and same.retain_estimate == pytest.approx(1.0)
    flat = holdout_statistics(np.zeros_like(original), original, labels)
    assert flat.kl_original_to_model_t2 > 0.5 and flat.retain_estimate < 0.6


def test_true_probability_shape_is_bimodal_for_sure_or_sure_not_models() -> None:
    original = np.zeros((6, NUM_CLASSES))
    original[np.arange(6), [1, 1, 2, 2, 6, 7]] = 20.0  # four partial-class images, two removed
    model = np.zeros_like(original)
    model[[0, 1], [1, 1]] = 20.0  # two partial images answered with certainty
    model[[2, 3], [90, 90]] = 20.0  # two partial images certainly answered wrong
    model[[4, 5], [5, 5]] = 20.0  # removed-class images routed to class 5
    stats = forget_statistics(model, original)
    assert stats.true_prob_above_0_9_partial == pytest.approx(0.5)
    assert stats.true_prob_below_0_1_partial == pytest.approx(0.5)
    assert stats.true_prob_middle_partial == pytest.approx(0.0)
    assert stats.true_prob_below_0_1_forgotten == pytest.approx(1.0)
    blurred = forget_statistics(np.zeros_like(original), original)
    assert blurred.true_prob_middle_partial == pytest.approx(0.0), (
        "1/100 is below 0.1: flat != middle"
    )
    assert blurred.mean_entropy_partial_classes == pytest.approx(np.log(NUM_CLASSES))
