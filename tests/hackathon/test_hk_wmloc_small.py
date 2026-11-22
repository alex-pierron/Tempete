"""Tests of the full-resolution localization code (w-loc-small): pure parts, no release data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.wmloc.research import small_candidate, small_gen, small_train
from hackathon.wmloc.research.data import Record


class FakeWrapper:
    """Embedder stand-in: adds one grey level everywhere, any crop size."""

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse

    def embed(self, image: np.ndarray, bits: list[int]) -> np.ndarray:
        if self.refuse:
            raise RuntimeError("crop too small")
        return np.clip(image.astype(np.int64) + 1, 0, 255).astype(np.uint8)


def rectangle(top: int, bottom: int, left: int, right: int) -> np.ndarray:
    mask = np.zeros((512, 512), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def test_official_label_follows_the_half_cell_rule() -> None:
    mask = np.zeros((512, 512), dtype=bool)
    mask[0:2, 0:4] = True  # exactly half of cell (0, 0)
    mask[4:5, 0:4] = True  # a quarter of cell (1, 0)
    label = small_gen.official_label(mask)
    assert label.shape == (128, 128)
    assert label[0, 0] == 1
    assert label[1, 0] == 0
    assert label.sum() == 1


def test_official_grid_matches_official_label() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((512, 512)) > 0.5
    expected = small_gen.official_label(mask)
    got = small_train.official_grid(torch.from_numpy(mask)[None, None])
    assert np.array_equal(got[0, 0].numpy().astype(np.uint8), expected)


@pytest.mark.parametrize("element", range(8))
def test_symmetry_keeps_the_area_and_is_a_bijection(element: int) -> None:
    mask = rectangle(10, 50, 100, 300)
    moved = small_gen.symmetry(mask, element)
    assert moved.shape == mask.shape
    assert moved.sum() == mask.sum()


def test_augment_shape_keeps_the_area_in_range() -> None:
    rng = np.random.default_rng(1)
    shape = rectangle(200, 300, 200, 320)
    for _ in range(20):
        area = float(small_gen.augment_shape(shape, rng).mean())
        assert small_gen.MIN_AREA_FRACTION <= area <= small_gen.MAX_AREA_FRACTION


def test_compose_rect_marks_only_the_region() -> None:
    clean = np.full((512, 512, 3), 100, dtype=np.uint8)
    region = rectangle(64, 128, 32, 96)
    bits = np.zeros(8, dtype=np.uint8)
    image = small_gen.compose_rect(clean, bits, region, FakeWrapper(), max_aspect=16.0)
    assert image is not None
    assert (image[region] == 101).all()
    assert (image[~region] == 100).all()


def test_compose_rect_rejects_thin_elongated_and_refused_crops() -> None:
    clean = np.zeros((512, 512, 3), dtype=np.uint8)
    bits = np.zeros(8, dtype=np.uint8)
    thin = rectangle(0, 8, 0, 100)
    long = rectangle(0, 32, 0, 200)
    fine = rectangle(0, 64, 0, 64)
    assert small_gen.compose_rect(clean, bits, thin, FakeWrapper(), 16.0) is None
    assert small_gen.compose_rect(clean, bits, long, FakeWrapper(), 2.0) is None
    assert small_gen.compose_rect(clean, bits, fine, FakeWrapper(refuse=True), 16.0) is None
    assert (
        small_gen.compose_rect(clean, bits, np.zeros((512, 512), bool), FakeWrapper(), 16.0) is None
    )


def test_augment_shapes_is_binary_and_keeps_the_area_in_range() -> None:
    shapes = torch.from_numpy(np.stack([rectangle(200, 300, 200, 320)] * 16))[:, None].float()
    generator = torch.Generator().manual_seed(0)
    moved = small_train.augment_shapes(shapes, generator)
    assert moved.dtype == torch.bool
    assert moved.shape == shapes.shape
    area = moved.float().mean(dim=(1, 2, 3))
    assert bool((area >= small_train.MIN_AREA_FRACTION).all())
    assert bool((area <= small_train.MAX_AREA_FRACTION).all())
    # Two draws differ: the augmentation is not the identity.
    assert not torch.equal(moved[0], moved[1]) or not torch.equal(moved[2], moved[3])


def test_miou_per_image_reference_values() -> None:
    truth = np.zeros((3, 128, 128), dtype=bool)
    truth[0, :64] = True
    truth[1, :64] = True
    prediction = truth.copy()
    prediction[1] = False  # class 1 missed: IoU 1 = 0, IoU 0 = 0.5
    scores = small_train.miou_per_image(prediction, truth)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.25)
    assert scores[2] == pytest.approx(1.0)  # both empty for class 1: counted as 1
    assert np.allclose(scores, small_candidate.miou_per_image(prediction, truth))


def test_learning_rate_warms_up_then_decays() -> None:
    assert small_train.learning_rate(0.0) == 0.0
    peak = small_train.learning_rate(small_train.WARMUP_FRACTION)
    assert peak == pytest.approx(small_train.PEAK_LEARNING_RATE)
    assert small_train.learning_rate(0.5) < peak
    floor = small_train.PEAK_LEARNING_RATE * small_train.FINAL_LEARNING_RATE_FRACTION
    assert small_train.learning_rate(1.0) == pytest.approx(floor)
    assert small_train.learning_rate(2.0) == pytest.approx(floor)


def test_segmentation_loss_prefers_the_truth() -> None:
    labels = torch.zeros(2, 1, 128, 128)
    labels[:, :, :40] = 1.0
    right = (labels * 2 - 1) * 10
    assert float(small_train.segmentation_loss(right, labels)) < 0.01
    assert float(small_train.segmentation_loss(-right, labels)) > 5.0


def test_network_maps_a_512_image_to_the_128_grid() -> None:
    network = small_train.FullResolutionNet(small_train.PLAIN_CHANNELS).eval()
    with torch.no_grad():
        logits = network(torch.zeros(1, small_train.PLAIN_CHANNELS, 512, 512))
    assert logits.shape == (1, 1, 128, 128)


def test_featurizer_without_template_has_six_channels() -> None:
    featurizer = small_train.Featurizer("stegastamp", "cpu", template=False)
    images = torch.randint(0, 256, (2, 512, 512, 3), dtype=torch.uint8)
    inputs = featurizer(images, None)
    assert inputs.shape == (2, small_train.PLAIN_CHANNELS, 512, 512)
    assert featurizer.channels == small_train.PLAIN_CHANNELS


def test_template_is_refused_outside_pixelseal() -> None:
    with pytest.raises(small_train.SmallTrainError):
        small_train.Featurizer("stegastamp", "cpu", template=True)


def test_zoomed_pair_cuts_the_same_512_window_of_a_larger_marked_base() -> None:
    rng = np.random.default_rng(0)
    clean = np.tile(np.arange(512, dtype=np.uint8)[None, :, None] // 4, (512, 1, 3))
    bits = np.zeros(8, dtype=np.uint8)
    zoomed_clean, zoomed_marked = small_gen.zoomed_pair(clean, bits, FakeWrapper(), rng, (1.1, 1.3))
    assert zoomed_clean.shape == zoomed_marked.shape == (512, 512, 3)
    # The fake embedder adds one grey level everywhere: same window in both images.
    difference = zoomed_marked.astype(np.int64) - zoomed_clean.astype(np.int64)
    assert np.all(difference == 1)


def test_weaken_scales_the_residual_and_stays_in_range() -> None:
    clean = np.array([[[10, 250, 0]]], dtype=np.uint8)
    marked = np.array([[[20, 255, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(small_gen.weaken(clean, marked, 1.0), marked)
    np.testing.assert_array_equal(small_gen.weaken(clean, marked, 0.0), clean)
    np.testing.assert_array_equal(small_gen.weaken(clean, marked, 0.5), [[[15, 252, 0]]])


def test_extractor_maps_are_refused_for_stegastamp_before_any_model_load() -> None:
    # StegaStamp has no per-position output: the refusal must come before the kit import.
    with pytest.raises(small_train.SmallTrainError):
        small_train.ExtractorMapper("stegastamp", "cpu")


def test_featurizer_stacks_the_extractor_channels() -> None:
    channels = small_train.EXTRACTOR_CHANNELS["trustmark"]
    featurizer = small_train.Featurizer("trustmark", "cpu", template=False, extra_channels=channels)
    images = torch.randint(0, 256, (2, 512, 512, 3), dtype=torch.uint8)
    extra = torch.zeros((2, channels, 128, 128), dtype=torch.float16)
    inputs = featurizer(images, None, extra)
    assert inputs.shape == (2, small_train.PLAIN_CHANNELS + channels, 512, 512)
    with pytest.raises(small_train.SmallTrainError):
        featurizer(images, None, None)


def test_parse_replacement() -> None:
    parsed = small_candidate.parse_replacement("stegastamp=a/b.npz:test_finetuned:0.4")
    assert parsed == ("stegastamp", [Path("a/b.npz")], "test_finetuned", 0.4)
    two = small_candidate.parse_replacement("stegastamp=a.npz+b.npz:k")
    assert two[1] == [Path("a.npz"), Path("b.npz")]
    assert small_candidate.parse_replacement("trustmark=x.npz:k")[3] == 0.5
    with pytest.raises(small_candidate.SmallCandidateError):
        small_candidate.parse_replacement("unknown=x.npz:k")
    with pytest.raises(small_candidate.SmallCandidateError):
        small_candidate.parse_replacement("stegastamp=x.npz")


def test_build_replaces_one_method_then_applies_confident_rectangles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        Record("a", "stegastamp", np.zeros(1, np.float32), None),
        Record("b", "stegastamp", np.zeros(1, np.float32), None),
        Record("c", "pixelseal", np.zeros(1, np.float32), None),
    ]
    monkeypatch.setattr(small_candidate, "load_records", lambda split: records)
    base = tmp_path / "base.pt"
    torch.save(
        {"image_names": ["a", "b", "c"], "masks": torch.zeros(3, 128, 128, dtype=torch.uint8)}, base
    )
    probability = np.zeros((2, 128, 128), dtype=np.float16)
    probability[0, :10] = 0.9
    probability[1, :10] = 0.45
    maps = tmp_path / "maps.npz"
    np.savez(maps, names_test=np.array(["a", "b"]), test_finetuned=probability)
    overlay = tmp_path / "overlay.pt"
    boxes = torch.zeros(2, 128, 128, dtype=torch.uint8)
    boxes[:, 100:] = 1
    torch.save(
        {"image_names": ["c", "b"], "masks": boxes, "confidence": torch.tensor([0.9, 0.5])}, overlay
    )
    out = tmp_path / "out.pt"
    changed = small_candidate.build(
        base, [("stegastamp", [maps, maps], "test_finetuned", 0.5)], overlay, out
    )
    assert changed == {"stegastamp": 1, "overlay": 1}
    result = torch.load(out, weights_only=True)
    assert result["image_names"] == ["a", "b", "c"]
    assert int(result["masks"][0].sum()) == 10 * 128  # network map
    assert int(result["masks"][1].sum()) == 0  # under the threshold, weak rectangle ignored
    assert int(result["masks"][2].sum()) == 28 * 128  # confident rectangle


def test_build_refuses_a_map_of_another_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [Record("a", "pixelseal", np.zeros(1, np.float32), None)]
    monkeypatch.setattr(small_candidate, "load_records", lambda split: records)
    base = tmp_path / "base.pt"
    torch.save({"image_names": ["a"], "masks": torch.zeros(1, 128, 128, dtype=torch.uint8)}, base)
    maps = tmp_path / "maps.npz"
    np.savez(maps, names_test=np.array(["a"]), k=np.zeros((1, 128, 128), np.float16))
    with pytest.raises(small_candidate.SmallCandidateError):
        small_candidate.build(base, [("stegastamp", [maps], "k", 0.5)], None, tmp_path / "o.pt")
