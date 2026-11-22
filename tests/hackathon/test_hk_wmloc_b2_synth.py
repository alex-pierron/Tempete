"""Tests of the real-shape synthetic generator (pure numpy, fake embedder)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from hackathon.wmloc.research.b2_synth import (
    GRID,
    IMAGE,
    MAX_AREA_FRACTION,
    ShapeLibrary,
    SynthError,
    augment_shape,
    build_library,
    compose_full,
    compose_rect,
    downsample_mask,
    sample_pair,
    upsample_mask,
)

SEED = 11
STAMP = 5


def fake_embed(image: NDArray[np.uint8], bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Deterministic 'watermark': add STAMP to every channel (easy to detect in the tests)."""
    return np.clip(image.astype(np.int16) + STAMP, 0, 255).astype(np.uint8)


def _disc(radius: int, centre: tuple[int, int] = (64, 64)) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[:GRID, :GRID]
    return np.asarray((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2, dtype=np.bool_)


@pytest.fixture()
def library() -> ShapeLibrary:
    masks = np.stack([1 - _disc(20).astype(np.uint8), _disc(25).astype(np.uint8)])
    return build_library(masks, np.array([True, False]))


def test_build_library_splits_holes_and_regions(library: ShapeLibrary) -> None:
    assert library.holes.shape == (1, GRID, GRID)
    assert library.regions.shape == (1, GRID, GRID)
    assert library.holes[0, 64, 64] and library.regions[0, 64, 64]


def test_build_library_needs_both_regimes() -> None:
    masks = np.ones((2, GRID, GRID), dtype=np.uint8)
    with pytest.raises(SynthError):
        build_library(masks, np.array([True, True]))


def test_upsample_then_downsample_keeps_a_large_shape() -> None:
    shape = _disc(30)
    back = downsample_mask(upsample_mask(shape))
    overlap = float((back.astype(bool) & shape).sum() / (back.astype(bool) | shape).sum())
    assert overlap > 0.95


def test_augment_shape_keeps_the_area_in_range() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(25):
        shape = augment_shape(_disc(20), rng)
        assert shape.shape == (GRID, GRID)
        assert 0.0 < float(shape.mean()) <= MAX_AREA_FRACTION


def test_compose_full_restores_the_hole_and_marks_the_rest() -> None:
    clean = np.full((IMAGE, IMAGE, 3), 100, dtype=np.uint8)
    image, label = compose_full(clean, np.zeros(8, dtype=np.uint8), _disc(20), fake_embed)
    assert image[0, 0, 0] == 100 + STAMP
    assert image[256, 256, 0] == 100
    assert label[0, 0] == 1 and label[64, 64] == 0
    hole_fraction = 1.0 - float(label.mean())
    assert 0.05 < hole_fraction < 0.12


def test_compose_rect_marks_only_inside_the_region() -> None:
    clean = np.full((IMAGE, IMAGE, 3), 100, dtype=np.uint8)
    image, label = compose_rect(clean, np.zeros(8, dtype=np.uint8), _disc(25), fake_embed)
    assert image[256, 256, 0] == 100 + STAMP
    assert image[0, 0, 0] == 100
    assert label[64, 64] == 1 and label[0, 0] == 0


def test_compose_rect_refuses_an_empty_region() -> None:
    clean = np.zeros((IMAGE, IMAGE, 3), dtype=np.uint8)
    with pytest.raises(SynthError):
        compose_rect(
            clean, np.zeros(8, dtype=np.uint8), np.zeros((GRID, GRID), dtype=bool), fake_embed
        )


def test_shapes_keep_the_fold_of_their_source_mask() -> None:
    masks = np.stack([1 - _disc(20).astype(np.uint8), _disc(25).astype(np.uint8)])
    library = build_library(masks, np.array([True, False]), np.array([3, 4]))
    rng = np.random.default_rng(SEED)
    clean = np.full((IMAGE, IMAGE, 3), 100, dtype=np.uint8)
    for _ in range(10):
        _, _, is_full, fold = sample_pair(
            clean, np.zeros(8, dtype=np.uint8), library, fake_embed, rng
        )
        assert fold == (3 if is_full else 4)


def test_compose_rect_refuses_an_elongated_region_when_the_embedder_needs_it() -> None:
    clean = np.zeros((IMAGE, IMAGE, 3), dtype=np.uint8)
    strip = np.zeros((GRID, GRID), dtype=bool)
    strip[60:68, 10:110] = True  # about 32 x 400 px: aspect 12
    with pytest.raises(SynthError):
        compose_rect(clean, np.zeros(8, dtype=np.uint8), strip, fake_embed, max_aspect=2.0)
    image, _ = compose_rect(clean, np.zeros(8, dtype=np.uint8), strip, fake_embed)
    assert image.shape == clean.shape


def test_sample_pair_label_matches_the_pixels(library: ShapeLibrary) -> None:
    rng = np.random.default_rng(SEED)
    clean = np.full((IMAGE, IMAGE, 3), 100, dtype=np.uint8)
    seen_full, seen_rect = False, False
    for _ in range(20):
        image, label, is_full, _fold = sample_pair(
            clean, np.zeros(8, dtype=np.uint8), library, fake_embed, rng
        )
        marked_cells = downsample_mask(image[..., 0] == 100 + STAMP)
        agreement = float((marked_cells == label).mean())
        assert agreement > 0.97
        seen_full, seen_rect = seen_full or is_full, seen_rect or not is_full
    assert seen_full and seen_rect


def test_select_shards_spreads_over_the_whole_range() -> None:

    from hackathon.wmloc.research.b2_synth_features import select_shards

    shards = [Path(f"shard_{i:04d}.npz") for i in range(96)]
    kept = select_shards(shards, per_shard=128, max_samples=6144)
    assert len(kept) == 48
    assert kept[0] == shards[0] and kept[-1] == shards[-1]
    assert select_shards(shards[:10], per_shard=128, max_samples=6144) == shards[:10]


def test_load_synthetic_ignores_temporary_files_and_caps_the_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hackathon.wmloc.research import b2_synth_features as module

    folder = tmp_path / "synth" / "pixelseal"
    folder.mkdir(parents=True)
    for index in range(6):
        np.savez(folder / f"shard_{index:04d}.npz", truth=np.zeros((4, 2, 2), np.uint8))
    np.savez(folder / "shard_0009.npz.123.tmp.npz", truth=np.zeros((4, 2, 2), np.uint8))
    monkeypatch.setattr(module, "synth_dir", lambda method: folder)
    everything = module.load_synthetic("pixelseal", max_samples=1000)
    assert everything["truth"].shape[0] == 24  # six real shards, the temporary file is skipped
    capped = module.load_synthetic("pixelseal", max_samples=12)
    assert capped["truth"].shape[0] == 12
