"""Tests of the band_ascent (V5) step-0 diagnostic on hand-made logits."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as torch_f
from test_hk_unlearning_support import TINY_EMBED_DIM, random_images, tiny_model

from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.diagnose import (
    PRE_LOGITS,
    _wrong_classes,
    band_defects,
    best_encoder_block,
    capture_cls,
    format_step0,
    knn_vote,
    level_ceiling,
    run_step0,
    step0_report,
)
from hackathon.unlearning.model import predict

IMAGES, CLASSES, HARD = 40, 10, 10


def _logits() -> tuple[torch.Tensor, torch.Tensor]:
    """(original, views): sure originals; the views lose the label on the first HARD images."""
    torch.manual_seed(0)
    labels = torch.arange(IMAGES) % CLASSES
    original = 0.5 * torch.randn(IMAGES, CLASSES)
    original[torch.arange(IMAGES), labels] += 10.0
    views = original + torch.randn(IMAGES, CLASSES)  # the views reshape the wrong classes too
    drop = torch.full((IMAGES,), 2.0)
    drop[:HARD] = 12.0
    views[torch.arange(IMAGES), labels] -= drop
    return original, views


def test_step0_finds_the_stand_in_errors_and_scores_each_direction() -> None:
    original, views = _logits()
    report = step0_report(original, [views, views], {"m30": [views, views]})
    assert report["selection"]["m30"]["phi25"] == {"precision": 1.0, "recall": 1.0,
                                                   "base_rate": 0.25}  # fmt: skip
    mass = report["mass"]["m30"]["phi25"]
    assert mass["loss_share"] == pytest.approx(mass["loss_share_oracle"])  # same ranking
    assert 0.0 < mass["pred_share"] <= 1.0
    direction = report["direction"]["m30"]
    assert direction["views"] == pytest.approx(1.0)  # the stand-in IS the views' mixture
    assert direction["runner_ups"] < direction["views"]
    assert set(report["mass"]["m30"]) == {"phi10", "phi25", "phi50"}
    assert "m30" in format_step0(report)


# ----- V5.1: encoder k-NN vote and the V5 defects ---------------------------------------------


def test_capture_cls_reads_the_class_tokens_without_changing_the_logits() -> None:
    model, images = tiny_model(), random_images(5)
    plain = predict(model, images)
    with capture_cls(model, [0, 1, PRE_LOGITS]) as store:
        hooked = predict(model, images)
    features = {block: torch.cat(parts) for block, parts in store.items()}
    assert torch.equal(plain, hooked)
    assert features[0].shape == (5, TINY_EMBED_DIM)
    with torch.no_grad():
        assert torch.allclose(model.head(features[PRE_LOGITS]), plain, atol=1e-5)
    assert not model.head._forward_pre_hooks and not model.blocks[0]._forward_hooks


def test_knn_vote_takes_its_neighbours_from_the_other_folds() -> None:
    features = torch.tensor([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.05, 1.0]])
    fold = torch.tensor([0, 1, 0, 1])
    exclude = fold[:, None] == fold[None, :]  # 0 and 2 share a fold, so do 1 and 3
    vote = knn_vote(features, torch.eye(4), exclude, k=1)
    assert vote.argmax(dim=1).tolist() == [1, 0, 3, 2]  # each one's twin, from the other fold
    assert torch.allclose(vote.sum(dim=1), torch.ones(4, dtype=torch.float64))
    assert float(vote[0, 2]) == 0.0  # same fold: never a neighbour


def test_selection_by_and_best_block_follow_the_fixed_rule() -> None:
    original, views = _logits()
    labels = original.argmax(dim=1)
    loss = torch_f.cross_entropy(views, labels, reduction="none")
    shape = _wrong_classes(torch.softmax(views / 2.0, dim=1), labels)
    report = step0_report(
        original,
        [views, views],
        {"m30": [views, views], "s40": [views, views]},
        scores={"knn_b0": loss, "knn_b1": -loss},
        shapes={"knn_b0": shape},
    )
    by = report["selection_by"]
    assert (
        by["knn_b0"]["m30"]["phi25"]["precision"]
        == report["selection"]["m30"]["phi25"]["precision"]
    )
    assert by["knn_b1"]["m30"]["phi25"]["precision"] == 0.0
    assert report["direction"]["m30"]["knn_b0"] == pytest.approx(1.0)
    best = best_encoder_block(report)
    assert best["best"] == "knn_b0" and best["masks"] == 1.0 and best["best_direction_wins"] == 1.0
    assert best_encoder_block(step0_report(original, [views], {"m30": [views]})) == {}


def test_band_defects_read_the_three_v5_defects_on_the_original() -> None:
    labels = torch.arange(8) % CLASSES
    margins = torch.tensor([12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 6.0, 9.0])  # pushed: the last two
    original = torch.zeros(8, CLASSES)
    original[torch.arange(8), labels] = margins
    view_loss = torch.arange(8, dtype=torch.float64)
    config = UnlearnConfig(method="band_ascent", band_release=0.25, band_floor=0.05)
    defects = band_defects(original, view_loss, config)
    loss = torch_f.cross_entropy(original.double(), labels, reduction="none")
    force = (torch.softmax(original.double(), 1) - torch_f.one_hot(labels, CLASSES)).norm(dim=1)
    held = torch.sigmoid(10.0 * (loss[:6] - 0.05)) * force[:6]
    assert defects["held_over_pushed"] == pytest.approx(float(held.sum() / force[6:].sum()))
    assert float(torch.sigmoid(10.0 * (loss[0] - 0.05))) == pytest.approx(0.38, abs=0.01)
    assert 0.5 < defects["force_participation"] < 1.0  # two pushed images, unequal forces
    assert defects["floor_kl_mean"] > 0.0
    tight = band_defects(original, view_loss, UnlearnConfig(method="band_ascent", band_floor=1e-9))
    assert tight["floor_kl_mean"] == pytest.approx(0.0)  # every loss already above that floor


def test_run_step0_reports_the_encoder_and_the_defects_on_a_tiny_model() -> None:
    config = UnlearnConfig(method="band_ascent", target_views=2, cv_folds=2, seed=0)
    report = run_step0(tiny_model(), random_images(8), config, batch_size=4)
    assert set(report["encoder"]["knn_agreement"]) == {"knn_b0", "knn_b1", "knn_pre_logits"}
    assert report["encoder"]["best"] in report["selection_by"]
    assert set(report["defects"]) >= {"force_top10_share", "floor_kl_mean"}
    assert "encoder (V5.1)" in format_step0(report) and "defects (V5.1)" in format_step0(report)
    assert set(report["level_ceiling"]) == {"t010", "t020", "t031", "t045", "t063", "t100"}
    assert "level ceiling (V_Dumb2) t045" in format_step0(report)


def test_level_ceiling_scores_the_band_at_free_logits() -> None:
    original, views = _logits()
    gold = {"s40": [views, views], "m30": [views + 0.1, views]}
    report = level_ceiling(original, gold, tolerance=2.0, levels=(1e-7, 0.45, 2.0))
    assert set(report) == {"t000", "t045", "t200"}
    tiny = report["t000"]  # the band sits at the original's own loss: nothing forgotten
    assert tiny["min_crop"] == pytest.approx(0.0, abs=1e-3) and tiny["s40"] == tiny["min_crop"]
    middle = report["t045"]
    assert 0.0 < middle["min_mask"] == middle["m30"] <= 1.0 and middle["min_crop"] > 0.0
    free = level_ceiling(original, gold, tolerance=float("inf"), levels=(0.45,))["t045"]
    assert free == middle  # no original sits past the ceiling: the pull never acts here
