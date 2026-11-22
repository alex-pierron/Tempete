"""Tests of the localizers and of threshold calibration, on a fake decoder (no weights needed).

The fake "decodes" the known message wherever channel 0 of the (resized) input is bright, and the
exact opposite elsewhere: the agreement is then 1 inside the marked region and 0 outside, so the
tests check geometry (cell placement, shift-and-stitch, window painting) and not model quality.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from hackathon.wmloc.calibrate import (
    bootstrap_mean_ci,
    cross_validated_per_image,
    sweep_thresholds,
)
from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, METHOD_PIXELSEAL, METHOD_TRUSTMARK
from hackathon.wmloc.errors import LocalizerError, UnsupportedMethodError
from hackathon.wmloc.local_backend import DecodeBackend
from hackathon.wmloc.localizers import (
    ConstantLocalizer,
    GlobalSwitchLocalizer,
    PixelMapLocalizer,
    RectRefineLocalizer,
    RoutingLocalizer,
    SlidingWindowLocalizer,
    apply_threshold,
    bit_agreement,
    gaussian_smooth,
    otsu_threshold,
)
from hackathon.wmloc.metric import downsample_mask_to_grid, miou_score

N_BITS = 16
NATIVE = 256
STRIDE = 32
BRIGHT = 0.5
WINDOW_PURITY_LOGIT = 0.8
"""Mean polarity above which a window decodes: (1 + 0.8) / 2 = 90 % marked cells."""


class FakeBackend:
    """Decoder that reads the known message where channel 0 is bright, its inverse elsewhere."""

    n_bits = N_BITS
    native_size = NATIVE
    map_stride: int | None = STRIDE

    def __init__(self, message: torch.Tensor) -> None:
        self.signs = message * 2.0 - 1.0

    def logit_map(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[-2:] == (NATIVE, NATIVE)
        cells = functional.avg_pool2d(images[:, 0:1], STRIDE)[:, 0]  # [B, 8, 8]
        polarity = torch.where(cells > BRIGHT, 1.0, -1.0)
        return self.signs[None, :, None, None] * polarity[:, None]

    def bit_logits(self, images: torch.Tensor) -> torch.Tensor:
        resized = functional.interpolate(
            images, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False, antialias=True
        )
        # A real decoder needs the window to lie almost entirely in the marked area (bit
        # accuracy falls fast with outside content): decode the message only above 90 % marked.
        polarity = self.logit_map(resized).mean(dim=(2, 3)) * self.signs[None]
        return self.signs[None] * (polarity - WINDOW_PURITY_LOGIT)


def make_message() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randint(0, 2, (N_BITS,), generator=generator).to(torch.float32)


def image_with_region(
    top: int, left: int, bottom: int, right: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image bright on [top:bottom, left:right] (pixels) and its 128-grid truth mask."""
    image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    image[0, top:bottom, left:right] = 1.0
    mask = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    mask[top:bottom, left:right] = 1
    return image, downsample_mask_to_grid(mask)


def iou_score(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    return miou_score(prediction, truth)


def test_bit_agreement_counts_matching_signs() -> None:
    message = torch.tensor([1.0, 0.0, 1.0, 0.0])
    logits = torch.tensor([[3.0, -2.0, -1.0, 4.0]])  # bits 1,0,0,1 -> 2 of 4 match
    assert bit_agreement(logits, message, bit_dim=1).item() == 0.5


def test_pixel_map_localizes_an_aligned_region() -> None:
    message = make_message()
    image, truth = image_with_region(128, 128, 384, 384)
    localizer = PixelMapLocalizer(
        FakeBackend(message), shifts_per_axis=1, smoothing_sigma_cells=0.0, threshold=0.5
    )
    mask = localizer.predict(image, METHOD_PIXELSEAL, message)
    assert mask.shape == (GRID_SIZE, GRID_SIZE) and mask.dtype == torch.uint8
    assert iou_score(mask, truth) > 0.95


def test_shift_and_stitch_refines_an_unaligned_region() -> None:
    message = make_message()
    image, truth = image_with_region(80, 80, 336, 336)  # edges are not multiples of 64 px
    coarse = PixelMapLocalizer(FakeBackend(message), 1, 0.0, 0.5)
    fine = PixelMapLocalizer(FakeBackend(message), 4, 0.0, 0.5)
    coarse_score = iou_score(coarse.predict(image, METHOD_PIXELSEAL, message), truth)
    fine_score = iou_score(fine.predict(image, METHOD_PIXELSEAL, message), truth)
    assert fine_score > coarse_score
    assert fine_score > 0.9


def test_pixel_map_rejects_bad_inputs() -> None:
    message = make_message()
    localizer = PixelMapLocalizer(FakeBackend(message))
    image, _ = image_with_region(0, 0, 10, 10)
    with pytest.raises(UnsupportedMethodError):
        localizer.predict(image, METHOD_TRUSTMARK, message)
    with pytest.raises(LocalizerError, match="message"):
        localizer.predict(image, METHOD_PIXELSEAL, message[:4])
    with pytest.raises(LocalizerError, match="image"):
        localizer.predict(image[:, :100, :100], METHOD_PIXELSEAL, message)
    with pytest.raises(ValueError, match="divide"):
        PixelMapLocalizer(FakeBackend(message), shifts_per_axis=3).predict(
            image, METHOD_PIXELSEAL, message
        )


def test_pixel_map_needs_a_spatial_backend() -> None:
    backend = DecodeBackend(lambda images: torch.zeros(images.shape[0], N_BITS), N_BITS, NATIVE)
    image, _ = image_with_region(0, 0, 10, 10)
    with pytest.raises(UnsupportedMethodError, match="spatial"):
        PixelMapLocalizer(backend).predict(image, METHOD_PIXELSEAL, make_message())


@pytest.mark.parametrize("aggregation", ["mean", "max"])
def test_sliding_windows_localize_a_region(aggregation: str) -> None:
    message = make_message()
    image, truth = image_with_region(128, 96, 416, 352)
    # The mean of overlapping windows needs a lower cut: border cells are covered by windows
    # that straddle the edge and decode to 0.
    threshold = 0.5 if aggregation == "max" else 0.25
    localizer = SlidingWindowLocalizer(
        FakeBackend(message), (64, 128), 32, threshold, aggregation=aggregation
    )
    mask = localizer.predict(image, METHOD_PIXELSEAL, message)
    assert iou_score(mask, truth) > 0.8


def test_sliding_window_validation() -> None:
    backend = FakeBackend(make_message())
    with pytest.raises(ValueError, match="multiples"):
        SlidingWindowLocalizer(backend, (62,), 32)
    with pytest.raises(ValueError, match="aggregation"):
        SlidingWindowLocalizer(backend, aggregation="median")
    with pytest.raises(ValueError, match="window sizes"):
        SlidingWindowLocalizer(backend, (1024,), 32)


def test_sliding_window_works_with_a_decode_only_backend() -> None:
    message = make_message()
    fake = FakeBackend(message)
    decode_only = DecodeBackend(fake.bit_logits, N_BITS, NATIVE)
    image, truth = image_with_region(0, 0, 256, 512)
    mask = SlidingWindowLocalizer(decode_only, (128,), 64, 0.5).predict(
        image, METHOD_PIXELSEAL, message
    )
    assert iou_score(mask, truth) > 0.9


def test_decode_backend_shifts_probabilities_to_logits() -> None:
    backend = DecodeBackend(lambda images: torch.tensor([[0.9, 0.1]]), 2, NATIVE, bit_threshold=0.5)
    logits = backend.bit_logits(torch.zeros(1, 3, 8, 8))
    assert logits[0, 0] > 0 > logits[0, 1]


def test_constant_and_routing_localizers() -> None:
    image, _ = image_with_region(0, 0, 10, 10)
    message = make_message()
    ones, zeros = ConstantLocalizer(1), ConstantLocalizer(0)
    assert bool((ones.predict(image, METHOD_PIXELSEAL, message) == 1).all())
    assert bool((zeros.predict(image, METHOD_PIXELSEAL, message) == 0).all())
    with pytest.raises(ValueError):
        ConstantLocalizer(2)
    router = RoutingLocalizer({METHOD_PIXELSEAL: zeros}, fallback=ones)
    assert bool((router.predict(image, METHOD_PIXELSEAL, message) == 0).all())
    assert bool((router.predict(image, METHOD_TRUSTMARK, message) == 1).all())
    assert "pixelseal=constant_0" in router.name


def test_otsu_splits_a_bimodal_map_and_handles_flat_maps() -> None:
    score = torch.full((GRID_SIZE, GRID_SIZE), 0.5)
    score[:, 64:] = 0.95
    cut = otsu_threshold(score)
    assert cut is not None and 0.5 < cut <= 0.95
    mask = apply_threshold(score, None)
    assert int(mask[:, 64:].sum()) == GRID_SIZE * 64 and int(mask[:, :64].sum()) == 0
    flat_chance = torch.full((GRID_SIZE, GRID_SIZE), 0.5)
    flat_marked = torch.full((GRID_SIZE, GRID_SIZE), 0.95)
    assert otsu_threshold(flat_chance) is None
    assert int(apply_threshold(flat_chance, None).sum()) == 0
    assert int(apply_threshold(flat_marked, None).sum()) == GRID_SIZE * GRID_SIZE


def test_gaussian_smooth_preserves_a_constant_and_is_a_noop_at_zero() -> None:
    constant = torch.full((GRID_SIZE, GRID_SIZE), 0.7)
    assert torch.allclose(gaussian_smooth(constant, 2.0), constant, atol=1e-5)
    spiky = torch.zeros(GRID_SIZE, GRID_SIZE)
    spiky[10, 10] = 1.0
    assert torch.equal(gaussian_smooth(spiky, 0.0), spiky)
    blurred = gaussian_smooth(spiky, 1.5)
    assert blurred[10, 10] < 1.0 and blurred.sum().item() == pytest.approx(1.0, abs=1e-4)


def noisy_scores(count: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    truth = (torch.rand(count, GRID_SIZE, GRID_SIZE, generator=generator) > 0.6).to(torch.uint8)
    noise = torch.randn(count, GRID_SIZE, GRID_SIZE, generator=generator) * 0.05
    return truth.float() * 0.4 + 0.5 + noise, truth


def test_sweep_finds_a_threshold_between_the_two_modes() -> None:
    scores, truth = noisy_scores()
    result = sweep_thresholds(scores, truth)
    assert 0.55 <= result.best_threshold <= 0.85
    assert result.best_score > 0.95
    assert result.best_score == max(result.scores)


def test_cross_validation_is_reproducible_and_honest() -> None:
    scores, truth = noisy_scores()
    first = cross_validated_per_image(scores, truth)
    second = cross_validated_per_image(scores, truth)
    assert torch.equal(first, second)
    assert float(first.mean()) > 0.95
    assert first.shape == (scores.shape[0],)


def test_bootstrap_interval_contains_the_mean_and_is_seeded() -> None:
    values = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.5, 0.7, 0.3, 0.9])
    mean, low, high = bootstrap_mean_ci(values)
    assert low <= mean <= high
    assert (mean, low, high) == bootstrap_mean_ci(values)
    with pytest.raises(ValueError):
        bootstrap_mean_ci(torch.empty(0))


def test_rect_refine_recovers_the_rectangle_and_cuts_holes() -> None:
    message = make_message()
    image, truth = image_with_region(128, 96, 416, 352)
    inner = SlidingWindowLocalizer(FakeBackend(message), (64, 128), 32, 0.5)
    localizer = RectRefineLocalizer(inner, seed_threshold=0.5, threshold=0.5)
    mask = localizer.predict(image, METHOD_PIXELSEAL, message)
    assert iou_score(mask, truth) > 0.9
    # Nothing marked anywhere: no rectangle, an all-clean mask.
    blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    assert int(localizer.predict(blank, METHOD_PIXELSEAL, message).sum()) == 0
    # A large hole inside the rectangle is removed by the inner score.
    holed = image.clone()
    holed[0, 224:320, 160:288] = 0.0
    hole_truth = truth.clone()
    hole_truth[224 // 4 : 320 // 4, 160 // 4 : 288 // 4] = 0
    with_hole = localizer.predict(holed, METHOD_PIXELSEAL, message)
    assert iou_score(with_hole, hole_truth) > iou_score(with_hole, truth)
    assert iou_score(with_hole, hole_truth) > 0.8


def test_rect_refine_leaves_edges_alone_when_nothing_decodes() -> None:
    message = make_message()
    inner = SlidingWindowLocalizer(FakeBackend(message), (64,), 64, 0.5)
    localizer = RectRefineLocalizer(inner)
    blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)  # nothing decodes anywhere: agreement 0
    box = (128, 128, 384, 384)
    assert localizer._move_edge(blank, message, box, 0, 8, 48) == box


def test_global_switch_is_all_ones_when_the_image_decodes_and_all_zeros_otherwise() -> None:
    message = make_message()
    marked, _ = image_with_region(0, 0, IMAGE_SIZE, IMAGE_SIZE)
    blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    localizer = GlobalSwitchLocalizer(FakeBackend(message), threshold=0.8)
    assert bool((localizer.predict(marked, METHOD_TRUSTMARK, message) == 1).all())
    assert bool((localizer.predict(blank, METHOD_TRUSTMARK, message) == 0).all())
    assert localizer.score_map(marked, METHOD_TRUSTMARK, message).unique().numel() == 1
