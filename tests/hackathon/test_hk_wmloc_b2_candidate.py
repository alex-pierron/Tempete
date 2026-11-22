"""Tests of the B2 candidate builder: seed averaging, fallback coverage, validated output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hackathon.wmloc.research import b2_candidate
from hackathon.wmloc.research.b2_candidate import (
    CandidateBuildError,
    average_fits,
    build_candidate,
    load_overlay,
)

GRID = 128
TEST_NAMES = [f"img{index:02d}" for index in range(6)]
LEARNED_NAMES = TEST_NAMES[:4]
METHOD = "pixelseal"


def _write_fit(directory: Path, tag: str, value: float, names: list[str]) -> None:
    probabilities = np.full((len(names), GRID, GRID), value, dtype=np.float16)
    np.savez(
        directory / f"{tag}_test.npz",
        **{METHOD: probabilities, f"names_{METHOD}": np.array(names)},  # type: ignore[arg-type]
    )


@pytest.fixture()
def fallback(tmp_path: Path) -> Path:
    masks = torch.zeros(len(TEST_NAMES), GRID, GRID, dtype=torch.uint8)
    path = tmp_path / "fallback.pt"
    torch.save({"image_names": TEST_NAMES, "masks": masks}, path)
    return path


@pytest.fixture(autouse=True)
def fake_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        b2_candidate, "load_records", lambda split: [SimpleNamespace(name=n) for n in TEST_NAMES]
    )


def test_average_fits_is_the_mean_over_seeds(tmp_path: Path) -> None:
    _write_fit(tmp_path, "s0", 0.2, LEARNED_NAMES)
    _write_fit(tmp_path, "s1", 0.8, LEARNED_NAMES)
    names, probabilities = average_fits(["s0", "s1"], METHOD, tmp_path)
    assert names == LEARNED_NAMES
    assert np.allclose(probabilities, 0.5, atol=1e-3)


def test_average_fits_rejects_fits_over_different_images(tmp_path: Path) -> None:
    _write_fit(tmp_path, "s0", 0.2, LEARNED_NAMES)
    _write_fit(tmp_path, "s1", 0.8, list(reversed(LEARNED_NAMES)))
    with pytest.raises(CandidateBuildError):
        average_fits(["s0", "s1"], METHOD, tmp_path)


def test_build_candidate_replaces_only_the_learned_images(tmp_path: Path, fallback: Path) -> None:
    _write_fit(tmp_path, "s0", 0.9, LEARNED_NAMES)
    out = tmp_path / "candidate.pt"
    summary = build_candidate(["s0"], [METHOD], fallback, out, directory=tmp_path)
    assert summary["images_in"] == summary["images_out"] == len(TEST_NAMES)
    assert summary["replaced_by_learned_head"] == {METHOD: len(LEARNED_NAMES)}
    loaded = torch.load(out, weights_only=True)
    masks = loaded["masks"]
    assert masks.shape == (len(TEST_NAMES), GRID, GRID)
    index = {name: i for i, name in enumerate(loaded["image_names"])}
    assert int(masks[index[LEARNED_NAMES[0]]].min()) == 1
    assert int(masks[index[TEST_NAMES[5]]].max()) == 0


def _submission(path: Path, value_by_name: dict[str, int]) -> Path:
    masks = torch.zeros(len(TEST_NAMES), GRID, GRID, dtype=torch.uint8)
    for index, name in enumerate(TEST_NAMES):
        masks[index] = value_by_name.get(name, 0)
    torch.save({"image_names": TEST_NAMES, "masks": masks}, path)
    return path


def test_overlay_with_a_base_keeps_only_the_images_that_differ(tmp_path: Path) -> None:
    base = _submission(tmp_path / "base.pt", {})
    candidate = _submission(tmp_path / "c.pt", {"img02": 1, "img05": 1})
    overlay = load_overlay(candidate, base)
    assert sorted(overlay) == ["img02", "img05"]
    assert int(overlay["img02"].min()) == 1


def test_overlay_is_applied_last_and_counted(tmp_path: Path, fallback: Path) -> None:
    _write_fit(tmp_path, "s0", 0.9, LEARNED_NAMES)
    base = _submission(tmp_path / "base.pt", {})
    rectangles = _submission(tmp_path / "c.pt", {"img00": 0, "img05": 1})
    summary = build_candidate(
        ["s0"],
        [METHOD],
        fallback,
        tmp_path / "out.pt",
        directory=tmp_path,
        overlays=[(rectangles, base)],
    )
    assert summary["overlay_masks_applied"] == 1  # img00 equals the base (all zeros), img05 differs
    loaded = torch.load(tmp_path / "out.pt", weights_only=True)
    index = {name: i for i, name in enumerate(loaded["image_names"])}
    assert int(loaded["masks"][index["img05"]].min()) == 1
    assert int(loaded["masks"][index["img00"]].min()) == 1  # learned head kept where not overlaid


def test_overlay_refuses_a_base_missing_an_overlay_image(tmp_path: Path) -> None:
    candidate = _submission(tmp_path / "c.pt", {"img02": 1})
    short = tmp_path / "short.pt"
    torch.save(
        {"image_names": TEST_NAMES[:2], "masks": torch.zeros(2, GRID, GRID, dtype=torch.uint8)},
        short,
    )
    with pytest.raises(CandidateBuildError):
        load_overlay(candidate, short)


def test_build_candidate_refuses_a_fallback_missing_an_image(tmp_path: Path) -> None:
    short = tmp_path / "short.pt"
    torch.save(
        {"image_names": TEST_NAMES[:-1], "masks": torch.zeros(5, GRID, GRID, dtype=torch.uint8)},
        short,
    )
    _write_fit(tmp_path, "s0", 0.9, LEARNED_NAMES)
    with pytest.raises(CandidateBuildError):
        build_candidate(["s0"], [METHOD], short, tmp_path / "x.pt", directory=tmp_path)


def test_overlay_drops_detections_below_the_confidence_threshold(tmp_path: Path) -> None:
    masks = torch.ones(3, GRID, GRID, dtype=torch.uint8)
    path = tmp_path / "overlay.pt"
    torch.save(
        {
            "image_names": ["img00", "img01", "img02"],
            "masks": masks,
            "confidence": torch.tensor([0.99, 0.80, 0.90]),
        },
        path,
    )
    assert sorted(load_overlay(path)) == ["img00", "img02"]
    assert sorted(load_overlay(path, min_confidence=0.5)) == ["img00", "img01", "img02"]
