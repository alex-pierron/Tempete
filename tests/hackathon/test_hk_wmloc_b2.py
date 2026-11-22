"""Tests of the team B2 learned segmenter (pure torch / numpy parts; no extractor, no data)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hackathon.wmloc.research.b2_features import evidence_maps, pool_to
from hackathon.wmloc.research.b2_model import (
    DIHEDRAL_ELEMENTS,
    DINO_KEY,
    GRID,
    OUT_SIDE,
    SOURCES_PIXELSEAL,
    B2Net,
    HeadConfig,
    combined_loss,
    count_parameters,
    dihedral,
    inverse_dihedral,
    lovasz_hinge,
    lovasz_miou_loss,
    soft_miou_loss,
)
from hackathon.wmloc.research.b2_train import breakdown

LARGE_LOGIT = 20.0
BATCH = 2
N_BITS = 12


def _target() -> torch.Tensor:
    target = torch.zeros(BATCH, OUT_SIDE, OUT_SIDE)
    target[:, 20:60, 30:90] = 1.0
    return target


def test_soft_miou_loss_is_near_zero_for_a_perfect_prediction() -> None:
    target = _target()
    logits = (2 * target - 1) * LARGE_LOGIT
    assert float(soft_miou_loss(logits, target)) < 1e-3


def test_soft_miou_loss_is_large_for_an_inverted_prediction() -> None:
    target = _target()
    logits = (1 - 2 * target) * LARGE_LOGIT
    assert float(soft_miou_loss(logits, target)) > 0.9


def test_lovasz_hinge_is_zero_when_margin_is_met_and_positive_otherwise() -> None:
    labels = (torch.rand(BATCH, 400) > 0.5).float()
    separated = (2 * labels - 1) * LARGE_LOGIT
    assert float(lovasz_hinge(separated, labels).max()) == pytest.approx(0.0, abs=1e-6)
    assert float(lovasz_hinge(-separated, labels).min()) > 0.0


def test_lovasz_miou_loss_orders_good_below_bad() -> None:
    target = _target()
    good = (2 * target - 1) * LARGE_LOGIT
    assert float(lovasz_miou_loss(good, target)) < float(lovasz_miou_loss(-good, target))


def test_lovasz_handles_images_with_a_single_class() -> None:
    target = torch.zeros(1, OUT_SIDE, OUT_SIDE)
    loss = lovasz_miou_loss(torch.randn(1, OUT_SIDE, OUT_SIDE), target)
    assert torch.isfinite(loss)


def test_combined_loss_skips_zero_weight_terms() -> None:
    target = _target()
    logits = torch.randn(BATCH, OUT_SIDE, OUT_SIDE)
    only_soft = combined_loss(logits, target, 0.0, 1.0, 0.0)
    assert float(only_soft) == pytest.approx(float(soft_miou_loss(logits, target)))


def test_dihedral_has_eight_distinct_elements_on_an_asymmetric_tensor() -> None:
    tensor = torch.arange(12.0).reshape(1, 3, 4)[:, :, :3]
    seen = {tuple(dihedral(tensor, e).flatten().tolist()) for e in range(DIHEDRAL_ELEMENTS)}
    assert len(seen) == DIHEDRAL_ELEMENTS


@pytest.mark.parametrize("element", range(DIHEDRAL_ELEMENTS))
def test_dihedral_commutes_with_nearest_upsampling(element: int) -> None:
    """A mask and a coarser feature map stay aligned after the same symmetry."""
    coarse = torch.rand(1, 1, 8, 8)
    fine = torch.nn.functional.interpolate(coarse, scale_factor=4, mode="nearest")
    left = torch.nn.functional.interpolate(
        dihedral(coarse, element), scale_factor=4, mode="nearest"
    )
    assert torch.equal(left, dihedral(fine, element))


@pytest.mark.parametrize("element", range(DIHEDRAL_ELEMENTS))
def test_inverse_dihedral_undoes_dihedral(element: int) -> None:
    tensor = torch.rand(2, 3, 8, 8)
    assert torch.equal(inverse_dihedral(dihedral(tensor, element), element), tensor)


def test_net_forward_shape_and_parameter_budget() -> None:
    names = ("n_st2", "n_ev", "u_ev")
    config = HeadConfig(sources={"pixelseal": names}, use_dino=True, use_rgb=True)
    net = B2Net(config)
    sources = {
        name: torch.randn(
            BATCH,
            SOURCES_PIXELSEAL[name][0],
            SOURCES_PIXELSEAL[name][1],
            SOURCES_PIXELSEAL[name][1],
        )
        for name in names
    }
    sources[DINO_KEY] = torch.randn(BATCH, 384, GRID, GRID)
    logits = net(
        "pixelseal", sources, torch.zeros(BATCH, 3), torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE)
    )
    assert logits.shape == (BATCH, OUT_SIDE, OUT_SIDE)
    assert count_parameters(net) < 400_000


def test_net_runs_without_dino_and_without_rgb() -> None:
    config = HeadConfig(sources={"pixelseal": ("n_ev",)}, use_dino=False, use_rgb=False)
    net = B2Net(config)
    logits = net(
        "pixelseal",
        {"n_ev": torch.randn(BATCH, 4, 8, 8)},
        torch.zeros(BATCH, 3),
        torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE),
    )
    assert logits.shape == (BATCH, OUT_SIDE, OUT_SIDE)


def test_evidence_maps_marks_agreement_with_the_message() -> None:
    signs = torch.tensor([1.0, -1.0] * (N_BITS // 2))
    agreeing = signs[:, None, None].expand(N_BITS, 4, 4) * 3.0
    evidence = evidence_maps(agreeing, signs)
    assert evidence.shape == (3, 4, 4)
    assert float(evidence[0].mean()) == pytest.approx(3.0)
    assert float(evidence[1].mean()) == pytest.approx(1.0)
    assert float(evidence[2].mean()) == pytest.approx(0.0, abs=1e-6)


def test_evidence_maps_is_chance_level_for_the_opposite_message() -> None:
    signs = torch.tensor([1.0, -1.0] * (N_BITS // 2))
    evidence = evidence_maps(-signs[:, None, None].expand(N_BITS, 4, 4) * 3.0, signs)
    assert float(evidence[1].mean()) == pytest.approx(0.0)


def test_pool_to_handles_non_divisible_sizes() -> None:
    pooled = pool_to(torch.rand(5, 31, 31), 16)
    assert pooled.shape == (5, 16, 16)


def test_breakdown_splits_regimes_and_bins() -> None:
    truth = np.ones((4, OUT_SIDE, OUT_SIDE), dtype=np.uint8)
    truth[0, :10] = 0  # full regime, small hole
    truth[1, :] = 0
    truth[1, 5:15, 5:15] = 1  # rectangle regime, tiny
    truth[2, :] = 0
    truth[2, :64, :64] = 1  # rectangle regime, 25 percent
    per_image = np.array([0.5, 0.4, 0.6, 0.9])
    is_full = np.array([True, False, False, True])
    result = breakdown(per_image, truth, is_full)
    assert result["all"] == pytest.approx(0.6)
    assert result["full"] == pytest.approx(0.7)
    assert result["rect"] == pytest.approx(0.5)
    rect_bins = result["by_size_rect"]
    assert isinstance(rect_bins, list)
    assert sum(item["n"] for item in rect_bins) == 2


def test_window_spec_matches_the_window_cache_layout() -> None:
    from hackathon.wmloc.research.b2_model import WINDOW_SPEC
    from hackathon.wmloc.research.b2_windows import (
        CHANNELS_PER_SIZE,
        TRUNK_GRID,
        WINDOW_SIZES,
    )

    assert WINDOW_SPEC == (len(WINDOW_SIZES) * CHANNELS_PER_SIZE, TRUNK_GRID)


def test_every_window_size_tiles_the_image() -> None:
    from hackathon.wmloc.research.b2_windows import WINDOW_SIZES, WindowError, grid_side

    assert [grid_side(size) for size in WINDOW_SIZES] == [15, 14, 13, 12, 11, 9, 7]
    with pytest.raises(WindowError):
        grid_side(100)


def _constant_cache(value: float) -> dict[str, np.ndarray]:
    from hackathon.wmloc.research.b2_windows import WINDOW_SIZES, grid_side

    cache: dict[str, np.ndarray] = {}
    for size in WINDOW_SIZES:
        side = grid_side(size)
        cache[f"w{size}_m"] = np.full((side, side), value, dtype=np.float16)
        cache[f"w{size}_a"] = np.full((side, side), value, dtype=np.float16)
    return cache


def test_window_maps_of_constant_decodes_are_constant_on_the_trunk_grid() -> None:
    from hackathon.wmloc.research.b2_windows import TRUNK_GRID, WINDOW_SIZES, window_maps

    maps = window_maps(_constant_cache(0.75))
    assert maps.shape == (len(WINDOW_SIZES) * 2, TRUNK_GRID, TRUNK_GRID)
    assert np.allclose(maps, 0.75, atol=1e-3)


def test_window_maps_place_a_single_good_window_under_its_centre() -> None:
    """The 128 px window at grid position (4, 6) is centred on pixel (4*32+64, 6*32+64)."""
    from hackathon.wmloc.research.b2_windows import CELL_PX, WINDOW_SIZES, window_maps

    cache = _constant_cache(0.0)
    cache["w128_a"][4, 6] = 1.0
    maps = window_maps(cache)
    channel = WINDOW_SIZES.index(128) * 2 + 1
    row, col = np.unravel_index(int(np.argmax(maps[channel])), maps[channel].shape)
    assert abs((row + 0.5) * CELL_PX - (4 * 32 + 64)) <= CELL_PX
    assert abs((col + 0.5) * CELL_PX - (6 * 32 + 64)) <= CELL_PX


def test_net_with_the_pixel_stem_reads_full_resolution_images() -> None:
    from hackathon.wmloc.research.b2_model import PIXEL_KEY, count_parameters

    plain = B2Net(HeadConfig(sources={"pixelseal": ("n_ev",)}, use_dino=False))
    with_pixels = B2Net(
        HeadConfig(sources={"pixelseal": ("n_ev",)}, use_dino=False, use_pixels=True)
    )
    assert count_parameters(with_pixels) > count_parameters(plain)
    sources = {
        "n_ev": torch.randn(BATCH, 4, 8, 8),
        PIXEL_KEY: torch.randint(0, 255, (BATCH, 3, 512, 512), dtype=torch.uint8),
    }
    logits = with_pixels(
        "pixelseal", sources, torch.zeros(BATCH, 3), torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE)
    )
    assert logits.shape == (BATCH, OUT_SIDE, OUT_SIDE)
    # The stem must actually change the output: same weights elsewhere, different pixels.
    other = dict(sources)
    other[PIXEL_KEY] = torch.zeros_like(sources[PIXEL_KEY])
    with_pixels.eval()
    zeros = torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE)
    assert not torch.allclose(
        with_pixels("pixelseal", sources, torch.zeros(BATCH, 3), zeros),
        with_pixels("pixelseal", other, torch.zeros(BATCH, 3), zeros),
    )


def test_stegastamp_source_specs_match_the_tapped_graph() -> None:
    from hackathon.wmloc.research.b2_model import SOURCES_STEGASTAMP
    from hackathon.wmloc.research.b2_stegastamp import FEATURE_GRIDS, FEATURE_TENSORS

    assert set(SOURCES_STEGASTAMP) == set(FEATURE_TENSORS) == set(FEATURE_GRIDS)
    assert {name: side for name, (_, side) in SOURCES_STEGASTAMP.items()} == FEATURE_GRIDS


def test_net_runs_on_stegastamp_feature_maps() -> None:
    from hackathon.wmloc.research.b2_model import SOURCES_STEGASTAMP

    names = tuple(SOURCES_STEGASTAMP)
    net = B2Net(HeadConfig(sources={"stegastamp": names}, use_dino=True, use_rgb=True))
    sources = {
        name: torch.randn(BATCH, channels, side, side)
        for name, (channels, side) in SOURCES_STEGASTAMP.items()
    }
    sources[DINO_KEY] = torch.randn(BATCH, 384, GRID, GRID)
    logits = net(
        "stegastamp", sources, torch.zeros(BATCH, 3), torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE)
    )
    assert logits.shape == (BATCH, OUT_SIDE, OUT_SIDE)


def test_net_runs_with_content_prior_only() -> None:
    """The ablation without any extractor evidence still produces a mask (DINO + RGB + scalars)."""
    net = B2Net(HeadConfig(sources={"pixelseal": ()}, use_dino=True, use_rgb=True))
    logits = net(
        "pixelseal",
        {DINO_KEY: torch.randn(BATCH, 384, GRID, GRID)},
        torch.zeros(BATCH, 3),
        torch.zeros(BATCH, 3, OUT_SIDE, OUT_SIDE),
    )
    assert logits.shape == (BATCH, OUT_SIDE, OUT_SIDE)
