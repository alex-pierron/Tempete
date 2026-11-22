"""Team C rectangle search: geometry, candidate selection and the full search on a planted rectangle."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from hackathon.wmloc.research.c_occlusion import (
    OcclusionError,
    gaussian_blur,
    occlusion_map,
    to_grid,
)
from hackathon.wmloc.research.c_occlusion import window_origins as occlusion_origins
from hackathon.wmloc.research.c_rect_search import (
    DETECTION_AGREEMENT,
    Box,
    RectSearchError,
    SearchConfig,
    WindowScores,
    coarse_boxes,
    geometric_sizes,
    refine_box,
    search_rectangle,
    select_candidates,
    window_origins,
)

PLANTED = Box(top=120, left=200, bottom=232, right=320)
PEAK_MARGIN = 9.0
SCALE_SIGMA = 0.2
SHIFT_SIGMA = 0.3
EVIDENCE_PER_MARGIN = 3.0


class PlantedScorer:
    """Margin peaks at a planted box and decays with scale error (log) and shift (fraction of size)."""

    def __init__(self, planted: Box = PLANTED) -> None:
        self.planted = planted
        self.n_decoded = 0

    def score(self, boxes: list[Box]) -> WindowScores:
        self.n_decoded += len(boxes)
        margins = []
        for box in boxes:
            scale = (
                np.log(box.height / self.planted.height) ** 2
                + np.log(box.width / self.planted.width) ** 2
            ) / (2 * SCALE_SIGMA**2)
            dy = (
                ((box.top + box.bottom) - (self.planted.top + self.planted.bottom))
                / 2
                / self.planted.height
            )
            dx = (
                ((box.left + box.right) - (self.planted.left + self.planted.right))
                / 2
                / self.planted.width
            )
            shift = (dy**2 + dx**2) / (2 * SHIFT_SIGMA**2)
            margins.append(PEAK_MARGIN * np.exp(-(scale + shift)))
        margin = np.array(margins, dtype=np.float32)
        agreement = np.where(margin > 1.0, 1.0, 0.5).astype(np.float32)
        return WindowScores(margin, agreement, margin * EVIDENCE_PER_MARGIN)


def test_geometric_sizes_are_increasing_multiples_of_four() -> None:
    sizes = geometric_sizes(24, 512, 1.55)
    assert sizes[0] == 24 and sizes[-1] == 512
    assert all(b > a for a, b in itertools.pairwise(sizes))
    assert all(size % 4 == 0 for size in sizes)


def test_geometric_sizes_reject_bad_ratio() -> None:
    with pytest.raises(RectSearchError):
        geometric_sizes(24, 512, 1.0)


def test_window_origins_cover_the_axis_with_a_flush_last_window() -> None:
    origins = window_origins(512, 100, 0.55)
    assert origins[0] == 0 and origins[-1] == 412
    assert all(b > a for a, b in itertools.pairwise(origins))
    assert window_origins(512, 512, 0.55) == [0]


def test_coarse_grid_respects_the_aspect_limit_and_the_image() -> None:
    config = SearchConfig(max_aspect=2.0)
    boxes = coarse_boxes(config)
    assert boxes
    assert all(b.is_valid(2.0) for b in boxes)


def test_box_geometry_and_mask_snap_to_the_official_grid() -> None:
    box = Box(top=10, left=20, bottom=50, right=100)
    assert (box.height, box.width, box.area) == (40, 80, 3200)
    assert box.aspect == pytest.approx(2.0)
    mask = box.to_mask()
    assert mask.shape == (128, 128) and mask.dtype == np.uint8
    # Rows 10..50 cover 2 of 4 pixel rows of the first and last cell: the official rule keeps them
    # (2 rows x 4 columns = half of the cell), hence 11 cell rows x 20 cell columns.
    assert int(mask.sum()) == 11 * 20


def test_box_iou_identity_disjoint_and_half() -> None:
    box = Box(0, 0, 100, 100)
    assert box.iou(box) == pytest.approx(1.0)
    assert box.iou(Box(200, 200, 300, 300)) == 0.0
    assert box.iou(Box(0, 50, 100, 150)) == pytest.approx(1 / 3)


def test_select_candidates_suppresses_duplicates_and_noise() -> None:
    boxes = [
        Box(0, 0, 100, 100),
        Box(4, 4, 104, 104),
        Box(300, 300, 400, 400),
        Box(400, 0, 500, 100),
    ]
    evidence = np.array([9.0, 8.8, 6.0, 0.01], dtype=np.float32)
    chosen = select_candidates(boxes, evidence, top_k=3)
    assert chosen == [boxes[0], boxes[2]]


def test_select_candidates_length_mismatch_raises() -> None:
    with pytest.raises(RectSearchError):
        select_candidates([Box(0, 0, 10, 10)], np.zeros(2, dtype=np.float32), 1)


def test_refinement_climbs_to_the_planted_box() -> None:
    config = SearchConfig(max_aspect=2.0)
    start = Box(top=136, left=216, bottom=252, right=340)
    best, scores = refine_box(PlantedScorer(), start, config)
    assert best.iou(PLANTED) > 0.9
    assert float(scores.margin[0]) > 5.0


def test_search_finds_the_planted_rectangle_and_declares_it_detected() -> None:
    scorer = PlantedScorer()
    result = search_rectangle(scorer, SearchConfig(max_aspect=2.0))
    assert result.detected
    assert result.agreement >= DETECTION_AGREEMENT
    assert result.box is not None and result.box.iou(PLANTED) > 0.85
    assert result.n_decodes == scorer.n_decoded > 0


def test_search_reports_no_detection_when_nothing_decodes() -> None:
    far = Box(top=0, left=0, bottom=24, right=24)

    class Silent(PlantedScorer):
        def score(self, boxes: list[Box]) -> WindowScores:
            self.n_decoded += len(boxes)
            zeros = np.zeros(len(boxes), dtype=np.float32)
            return WindowScores(zeros, zeros + 0.5, zeros)

    result = search_rectangle(Silent(far), SearchConfig(max_aspect=2.0))
    assert not result.detected and result.box is None and result.margin == 0.0


# --- occlusion ---------------------------------------------------------------------------------

FAKE_RESOLUTION = 64
FAKE_BITS = 4
BLOCK_CELL_PX = 8


class EnergyDecoder:
    """Logit of every bit = mean absolute deviation of the image: blurring energy lowers it."""

    method = "trustmark"
    resolution = FAKE_RESOLUTION

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        deviation = (images - images.mean(dim=(1, 2, 3), keepdim=True)).abs().mean(dim=(1, 2, 3))
        return deviation[:, None].repeat(1, FAKE_BITS)


def _textured_left_half() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    grid = torch.rand(3, 64, 64, generator=generator)
    texture = grid.repeat_interleave(BLOCK_CELL_PX, dim=1).repeat_interleave(BLOCK_CELL_PX, dim=2)
    image = torch.full((3, 512, 512), 0.5)
    image[:, :, :256] = texture[:, :, :256]
    return image


def test_gaussian_blur_keeps_a_flat_image_and_rejects_bad_sigma() -> None:
    flat = torch.full((3, 64, 64), 0.3)
    assert torch.allclose(gaussian_blur(flat, 4.0), flat, atol=1e-5)
    with pytest.raises(OcclusionError):
        gaussian_blur(flat, 0.0)


def test_occlusion_origins_cover_the_image_flush() -> None:
    origins = occlusion_origins(32, 16)
    assert origins[0] == 0 and origins[-1] == 480 and len(origins) == 31
    with pytest.raises(OcclusionError):
        occlusion_origins(0, 16)


def test_occlusion_drop_is_larger_where_the_signal_lives() -> None:
    message = np.ones(FAKE_BITS, dtype=np.float32)
    found = occlusion_map(
        EnergyDecoder(), _textured_left_half(), message, sigma=12.0, block=64, stride=64
    )
    assert found.drop.shape == (8, 8)
    left, right = found.drop[:, :4].mean(), found.drop[:, 4:].mean()
    assert left > 10 * max(abs(right), 1e-6)
    grid = to_grid(found)
    assert grid.shape == (128, 128)
    assert grid[:, :64].mean() > grid[:, 64:].mean()


# --- decoder-backed scorer ---------------------------------------------------------------------


class MeanDecoder:
    """Every bit logit is the mean pixel of the (resized) crop minus one half."""

    method = "trustmark"
    resolution = 16
    device = "cpu"

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return (images.mean(dim=(1, 2, 3)) - 0.5)[:, None].repeat(1, FAKE_BITS)


def test_decoder_scorer_keeps_the_order_of_boxes_of_mixed_shapes() -> None:
    from hackathon.wmloc.research.c_rect_search import DecoderWindowScorer, resize_crop

    generator = torch.Generator().manual_seed(1)
    image = torch.rand(3, 512, 512, generator=generator)
    boxes = [Box(0, 0, 40, 40), Box(100, 100, 180, 140), Box(0, 100, 40, 140), Box(200, 0, 280, 40)]
    message = np.ones(FAKE_BITS, dtype=np.float32)
    scorer = DecoderWindowScorer(MeanDecoder(), image, message)
    scores = scorer.score(boxes)
    expected = [
        float(
            resize_crop(image[:, b.top : b.bottom, b.left : b.right], "trustmark", 16).mean() - 0.5
        )
        for b in boxes
    ]
    assert scorer.n_decoded == len(boxes)
    assert scores.margin == pytest.approx(expected, abs=1e-5)
    assert scores.agreement.shape == scores.evidence.shape == (len(boxes),)
