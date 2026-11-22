"""Tests of the research team A modules (pure numpy parts; no decoder, no data needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hackathon.wmloc.metric import downsample_mask_to_grid, per_image_miou
from hackathon.wmloc.research.crossval import MethodData
from hackathon.wmloc.research.data import Record
from hackathon.wmloc.research.evaluate import (
    EvaluationError,
    miou_per_image,
    roc_auc,
    stratified_folds,
)
from hackathon.wmloc.research.extract import (
    ExtractionError,
    WindowSpec,
    parse_window_pass,
    window_boxes,
)
from hackathon.wmloc.research.maps import robust_z, to_grid
from hackathon.wmloc.research.postprocess import (
    PostprocessError,
    bounding_box_mask,
    evidence_to_mask,
    largest_component,
    top_fraction_mask,
)
from hackathon.wmloc.research.rectangles import (
    box_from_grid,
    box_iou,
    box_to_mask,
    is_valid,
)
from hackathon.wmloc.research.rules import ImageFeatures, RuleError, SwitchRule, ThresholdRule
from hackathon.wmloc.research.scoring import (
    agreement,
    dense_position_maps,
    dense_to_grid,
    margin,
)

RNG = np.random.default_rng(7)


def test_miou_matches_the_official_metric_implementation() -> None:
    pred = RNG.integers(0, 2, size=(6, 128, 128)).astype(np.uint8)
    gold = RNG.integers(0, 2, size=(6, 128, 128)).astype(np.uint8)
    ours = miou_per_image(pred, gold)
    official = per_image_miou(torch.from_numpy(pred), torch.from_numpy(gold)).numpy()
    np.testing.assert_allclose(ours, official, atol=1e-12)


def test_miou_empty_union_counts_as_one() -> None:
    zeros = np.zeros((1, 128, 128), dtype=np.uint8)
    assert miou_per_image(zeros, zeros)[0] == pytest.approx(1.0)


def test_miou_shape_mismatch_raises() -> None:
    with pytest.raises(EvaluationError):
        miou_per_image(np.zeros((1, 4, 4)), np.zeros((1, 5, 5)))


def test_stratified_folds_are_balanced_and_reproducible() -> None:
    strata = ["a"] * 20 + ["b"] * 10
    first = stratified_folds(strata, 5, seed=1)
    second = stratified_folds(strata, 5, seed=1)
    np.testing.assert_array_equal(first, second)
    for fold in range(5):
        assert int(((first == fold) & (np.array(strata) == "a")).sum()) == 4
        assert int(((first == fold) & (np.array(strata) == "b")).sum()) == 2


def test_roc_auc_extremes_and_ties() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(np.ones(4), labels) == pytest.approx(0.5)
    with pytest.raises(EvaluationError):
        roc_auc(scores, np.ones(4, dtype=bool))


def test_window_boxes_cover_the_image_and_end_flush() -> None:
    boxes = window_boxes(WindowSpec(size=128, stride=96))
    assert boxes[:, 2].max() == 512 and boxes[:, 3].max() == 512
    assert set((boxes[:, 2] - boxes[:, 0]).tolist()) == {128}
    with pytest.raises(ExtractionError):
        window_boxes(WindowSpec(size=600, stride=32))


def test_parse_window_pass() -> None:
    assert parse_window_pass("w128s64") == WindowSpec(128, 64)
    assert parse_window_pass("global") is None


def test_agreement_and_margin_follow_the_message_sign() -> None:
    message = np.array([1, 0, 1, 0], dtype=np.float32)
    logits = np.array([2.0, -1.0, 3.0, 4.0], dtype=np.float32)  # last bit wrong
    assert float(agreement(logits, message)) == pytest.approx(0.75)
    assert float(margin(logits, message)) == pytest.approx((2 + 1 + 3 - 4) / 4)


def test_dense_position_maps_places_each_shift_on_its_own_slot() -> None:
    n_bits = 3
    message = np.ones(n_bits, dtype=np.float32)
    maps = np.zeros((4, n_bits, 8, 8), dtype=np.float32)
    for shift in range(4):
        maps[shift] = shift + 1  # a distinct constant per shift
    dense = dense_position_maps(maps, message)
    assert dense.shape == (16, 16)
    # Slot (r, c) comes from shift (1 - r % 2, 1 - c % 2): row steps 0 for odd r, 1 for even r.
    assert dense[1, 1] == 1.0  # shift (0, 0)
    assert dense[1, 2] == 2.0  # shift (0, 1)
    assert dense[2, 1] == 3.0  # shift (1, 0)
    assert dense[2, 2] == 4.0  # shift (1, 1)
    assert np.count_nonzero(dense) == 256


def test_dense_to_grid_preserves_a_constant() -> None:
    grid = dense_to_grid(np.full((16, 16), 3.0, dtype=np.float32))
    assert grid.shape == (128, 128)
    np.testing.assert_allclose(grid, 3.0, atol=1e-5)


def test_robust_z_is_zero_at_the_median_and_survives_a_flat_map() -> None:
    flat = np.full((8, 8), 2.0, dtype=np.float32)
    assert np.isfinite(robust_z(flat)).all()
    noisy = RNG.normal(size=(64, 64)).astype(np.float32)
    assert abs(float(np.median(robust_z(noisy)))) < 1e-5


def test_to_grid_upsamples_to_128() -> None:
    assert to_grid(np.ones((32, 32), dtype=np.float32)).shape == (128, 128)


def test_box_to_mask_agrees_with_the_official_downsampling() -> None:
    for box in [(8, 12, 200, 340), (1, 3, 99, 101), (0, 0, 512, 512), (100, 100, 102, 400)]:
        pixels = torch.zeros(512, 512, dtype=torch.uint8)
        pixels[box[0] : box[2], box[1] : box[3]] = 1
        official = downsample_mask_to_grid(pixels).numpy()
        np.testing.assert_array_equal(box_to_mask(box), official)


def test_box_from_grid_roundtrip_and_iou() -> None:
    grid = np.zeros((128, 128), dtype=np.float32)
    grid[10:20, 30:60] = 1.0
    assert box_from_grid(grid, 0.5) == (40, 120, 80, 240)
    assert box_from_grid(grid, 2.0) is None
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou(None, (0, 0, 1, 1)) == 0.0


def test_is_valid_enforces_aspect_and_size() -> None:
    assert is_valid((0, 0, 100, 100), 2.0)
    assert not is_valid((0, 0, 100, 300), 2.0)
    assert is_valid((0, 0, 100, 300), 16.0)
    assert not is_valid((0, 0, 8, 100), 16.0)
    assert not is_valid((0, 0, 600, 100), 16.0)


def test_largest_component_and_bbox() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    mask[50:70, 50:80] = 1
    kept = largest_component(mask)
    assert kept.sum() == 20 * 30
    assert bounding_box_mask(kept).sum() == 20 * 30
    assert bounding_box_mask(np.zeros((128, 128), dtype=np.uint8)).sum() == 0


def test_evidence_to_mask_rejects_unknown_mode() -> None:
    with pytest.raises(PostprocessError):
        evidence_to_mask(np.zeros((128, 128)), 0.0, "nope")


def test_top_fraction_mask_area() -> None:
    evidence = RNG.normal(size=(128, 128))
    mask = top_fraction_mask(evidence, 0.1)
    assert abs(int(mask.sum()) - round(0.1 * 128 * 128)) <= 1
    assert top_fraction_mask(evidence, 0.0).sum() == 0


def test_switch_rule_fits_a_separating_cut() -> None:
    agreements = np.concatenate([RNG.uniform(0.45, 0.6, 50), RNG.uniform(0.9, 1.0, 50)])
    is_full = np.array([False] * 50 + [True] * 50)
    switch = SwitchRule()
    with pytest.raises(RuleError):
        switch.is_full(0.9)
    switch.fit(agreements, is_full)
    assert switch.cut is not None and 0.6 <= switch.cut <= 0.9
    assert switch.is_full(0.95) and not switch.is_full(0.5)


def test_threshold_rule_needs_fit_and_picks_the_better_cut() -> None:
    truth = np.zeros((4, 128, 128), dtype=np.uint8)
    truth[:, 40:80, 40:80] = 1
    feats = []
    for index in range(4):
        posmap = np.zeros((128, 128), dtype=np.float32)
        posmap[40:80, 40:80] = 10.0
        posmap += RNG.normal(scale=0.1, size=posmap.shape).astype(np.float32)
        feats.append(ImageFeatures(f"i{index}", 0.5, 0.0, posmap, None))
    rule = ThresholdRule("posmap_z", [1.0, 3.0, 5.0, 1e9])
    with pytest.raises(RuleError):
        rule.predict(feats[0])
    rule.fit(feats, truth)
    assert rule.cut in (1.0, 3.0, 5.0)
    score = miou_per_image(np.stack([rule.predict(f) for f in feats]), truth).mean()
    assert score > 0.95


def _synthetic_method_data(count: int = 20) -> MethodData:
    truth = np.zeros((count, 128, 128), dtype=np.uint8)
    features: list[ImageFeatures] = []
    records: list[Record] = []
    is_full = np.array([index % 2 == 0 for index in range(count)])
    for index in range(count):
        records.append(
            Record(f"i{index}", "pixelseal", np.ones(4, dtype=np.float32), bool(is_full[index]))
        )
        posmap = RNG.normal(scale=0.1, size=(128, 128)).astype(np.float32)
        if is_full[index]:
            truth[index] = 1
            truth[index, 10:30, 10:30] = 0
            posmap += 5.0
            posmap[10:30, 10:30] = 0.0
        else:
            truth[index, 40:80, 40:90] = 1
            posmap[40:80, 40:90] += 8.0
        agreement_value = 0.99 if is_full[index] else 0.52
        features.append(
            ImageFeatures(
                f"i{index}", agreement_value, 5.0 if is_full[index] else 0.3, posmap, None
            )
        )
    folds = stratified_folds([str(flag) for flag in is_full], 5, seed=3)
    return MethodData("pixelseal", records, truth, is_full, features, folds)


def test_cross_validation_recovers_the_hole_and_the_rectangle() -> None:
    from hackathon.wmloc.research.crossval import (
        combine,
        evaluate_everywhere,
        evaluate_rule,
        evaluate_switch,
    )
    from hackathon.wmloc.research.rules import (
        AreaMatchedRule,
        ConstantRule,
    )

    data = _synthetic_method_data()
    switch = evaluate_switch(data)
    assert switch.accuracy == pytest.approx(1.0)
    hole_rule = evaluate_rule(
        data, lambda: ThresholdRule("posmap_rel", [0.2, 0.5, 0.8], keep_ones_candidate=True), True
    )
    ones = evaluate_rule(data, lambda: ConstantRule(1), True)
    assert hole_rule.mean > ones.mean
    rect_rule = evaluate_rule(data, lambda: ThresholdRule("posmap_z", [2.0, 5.0, 1e9]), False)
    zeros = evaluate_rule(data, lambda: ConstantRule(0), False)
    assert rect_rule.mean > 0.9 > zeros.mean
    matched = evaluate_rule(data, lambda: AreaMatchedRule("posmap_z"), False)
    assert matched.mean > 0.8
    full_all = evaluate_everywhere(data, lambda: ConstantRule(1), True)
    rect_all = evaluate_everywhere(data, lambda: ThresholdRule("posmap_z", [2.0, 5.0, 1e9]), False)
    mean, per_image = combine(data, switch, full_all, rect_all)
    assert per_image.shape == (20,) and 0.0 <= mean <= 1.0


def test_area_matched_rule_needs_fit() -> None:
    from hackathon.wmloc.research.rules import AreaMatchedRule

    data = _synthetic_method_data(4)
    rule = AreaMatchedRule("posmap_z")
    with pytest.raises(RuleError):
        rule.predict(data.features[0])
