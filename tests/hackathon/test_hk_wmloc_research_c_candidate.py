"""Team C candidate merge: only detected rectangles of the chosen method replace base masks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hackathon.wmloc.research.c_candidate import (
    CandidateError,
    accepted_box,
    collect_overlay,
    load_base,
    merge_rectangles,
    write_overlay,
)
from hackathon.wmloc.research.c_rect_search import Box, is_detection

DETECTED_PAYLOAD = {
    "searched": True,
    "detected": True,
    "agreement": 1.0,
    "box": {"top": 40, "left": 60, "bottom": 120, "right": 180},
}


def test_accepted_box_requires_search_detection_agreement_and_valid_shape() -> None:
    assert accepted_box(DETECTED_PAYLOAD, "trustmark") == Box(40, 60, 120, 180)
    assert accepted_box({**DETECTED_PAYLOAD, "detected": False}, "trustmark") is None
    assert accepted_box({**DETECTED_PAYLOAD, "searched": False}, "trustmark") is None
    assert accepted_box({**DETECTED_PAYLOAD, "agreement": 0.7}, "trustmark") is None
    assert accepted_box({**DETECTED_PAYLOAD, "box": None}, "trustmark") is None
    tall = {**DETECTED_PAYLOAD, "box": {"top": 0, "left": 0, "bottom": 400, "right": 40}}
    assert accepted_box(tall, "trustmark") is None  # aspect 10 > TrustMark limit 2
    assert accepted_box(tall, "stegastamp") is not None


def test_accepted_box_rejects_a_whole_image_decode() -> None:
    whole = {**DETECTED_PAYLOAD, "box": {"top": 4, "left": 4, "bottom": 504, "right": 508}}
    assert accepted_box(whole, "trustmark") is None  # area 98 % > 50 % cap
    assert is_detection(Box(0, 0, 370, 370), 0.95) is False  # 52 % of the image
    assert is_detection(Box(0, 0, 360, 360), 0.95) is True  # 49.4 %
    assert is_detection(Box(0, 0, 300, 300), 0.95) is True
    assert is_detection(Box(0, 0, 300, 300), 0.79) is False


def test_merge_replaces_only_detected_images_of_the_method(tmp_path: Path) -> None:
    names = ["a", "b", "c", "d"]
    masks = torch.ones((4, 128, 128), dtype=torch.uint8)
    (tmp_path / "a.json").write_text(json.dumps({**DETECTED_PAYLOAD, "name": "a"}))
    (tmp_path / "b.json").write_text(
        json.dumps({**DETECTED_PAYLOAD, "name": "b", "detected": False})
    )
    merged, report = merge_rectangles(names, masks, {"a", "b", "c"}, tmp_path, "trustmark")
    assert report.n_replaced == 1 and report.n_missing == 1 and report.n_searched == 2
    assert int(merged[0].sum()) == 20 * 30  # 80 x 120 px on 4 px cells
    assert torch.equal(merged[1], masks[1]) and torch.equal(merged[2], masks[2])
    assert torch.equal(merged[3], masks[3])  # image of another method: untouched
    assert torch.equal(masks, torch.ones_like(masks))  # input not modified in place


def test_overlay_collects_detections_with_confidence_and_writes_atomically(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(
        json.dumps({**DETECTED_PAYLOAD, "name": "a", "agreement": 0.93})
    )
    (results / "b.json").write_text(
        json.dumps({**DETECTED_PAYLOAD, "name": "b", "detected": False})
    )
    entries = collect_overlay({"a", "b", "c"}, results, "trustmark")
    assert set(entries) == {"a"} and entries["a"][1] == pytest.approx(0.93)
    target = tmp_path / "overlays" / "c_rect_overlay.pt"
    write_overlay(target, entries)
    assert [p.name for p in target.parent.iterdir()] == ["c_rect_overlay.pt"]  # no .part left
    loaded = torch.load(target, weights_only=True)
    assert loaded["image_names"] == ["a"]
    assert loaded["masks"].shape == (1, 128, 128) and loaded["masks"].dtype == torch.uint8
    assert int(loaded["masks"][0].sum()) == 20 * 30
    assert loaded["confidence"].dtype == torch.float32 and float(
        loaded["confidence"][0]
    ) == pytest.approx(0.93)


def test_empty_overlay_is_a_valid_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.pt"
    write_overlay(target, {})
    loaded = torch.load(target, weights_only=True)
    assert loaded["image_names"] == [] and loaded["masks"].shape == (0, 128, 128)


def test_merge_rejects_a_name_missing_from_the_base(tmp_path: Path) -> None:
    empty = torch.zeros((1, 128, 128), dtype=torch.uint8)
    with pytest.raises(CandidateError):
        merge_rectangles(["a"], empty, {"z"}, tmp_path, "trustmark")


def test_load_base_rejects_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"weights": torch.zeros(3)}, path)
    with pytest.raises(CandidateError):
        load_base(path)
