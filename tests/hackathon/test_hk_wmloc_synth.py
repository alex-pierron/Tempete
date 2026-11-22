"""Tests of the synthetic bench and of the manifest, with a trivial injected embedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from PIL import Image

from hackathon.wmloc.constants import GRID_SIZE, IMAGE_SIZE, METHOD_PIXELSEAL
from hackathon.wmloc.errors import ManifestError
from hackathon.wmloc.manifest import (
    ManifestSample,
    bits_to_string,
    parse_message_bits,
    read_image_tensor,
    read_manifest,
    read_truth_mask128,
    write_manifest,
)
from hackathon.wmloc.metric import downsample_mask_to_grid
from hackathon.wmloc.synth import (
    EDIT_ADDITION,
    EDIT_OUTPAINTING,
    EDIT_REMOVAL,
    EDIT_TRANSPLANT,
    EDIT_TYPES,
    EMBED_CROP,
    EMBED_FULL,
    RECT_MAX_ASPECT,
    RECT_MIN_AREA_FRACTION,
    RECT_MIN_SIDE,
    SynthSample,
    generate_samples,
    list_source_images,
    make_sample,
    rect_mask,
    sample_blob,
    sample_rect,
    write_synth_dataset,
)

SEED = 5
N_BITS = 8
EMBED_DELTA = 0.05


def fake_embed(images: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """A visible, deterministic 'watermark': brighten by a constant."""
    return (images + EMBED_DELTA).clamp(0.0, 1.0)


@pytest.fixture
def sources(tmp_path: Path) -> list[Path]:
    folder = tmp_path / "photos"
    folder.mkdir()
    generator = np.random.default_rng(0)
    for index in range(4):
        pixels = generator.integers(20, 200, (300 + 50 * index, 420, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"p{index}.png")
    return list_source_images([folder])


def test_rect_sampling_respects_the_limits() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        y0, x0, height, width = sample_rect(rng)
        assert min(height, width) >= RECT_MIN_SIDE
        assert max(height, width) / min(height, width) <= RECT_MAX_ASPECT
        assert height * width / IMAGE_SIZE**2 >= RECT_MIN_AREA_FRACTION
        assert 0 <= y0 and y0 + height <= IMAGE_SIZE and 0 <= x0 and x0 + width <= IMAGE_SIZE


def test_blob_area_is_close_to_the_target() -> None:
    rng = np.random.default_rng(SEED)
    for target in (2_000.0, 10_000.0, 30_000.0):
        areas = [int(sample_blob(rng, (256, 256), target).sum()) for _ in range(20)]
        assert 0.4 * target < float(np.median(areas)) < 1.6 * target


def make(edit: str, mode: str = EMBED_CROP, seed: int = SEED) -> tuple[SynthSample, torch.Tensor]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.8
    other = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.8
    message = torch.ones(N_BITS)
    return make_sample("s", edit, base, other, message, fake_embed, rng, mode), base


@pytest.mark.parametrize("mode", [EMBED_CROP, EMBED_FULL])
def test_truth_geometry_per_edit_type(mode: str) -> None:
    for seed in range(5):
        outpaint, _ = make(EDIT_OUTPAINTING, mode, seed)
        rect = rect_mask(outpaint.rect)
        assert np.array_equal(outpaint.mask512.numpy().astype(bool), rect)
        for edit in (EDIT_REMOVAL, EDIT_ADDITION):
            sample, _ = make(edit, mode, seed)
            truth = sample.mask512.numpy().astype(bool)
            inside = rect_mask(sample.rect)
            assert (truth & ~inside).sum() == 0
            assert truth.sum() < inside.sum()  # a hole was cut
        transplant, _ = make(EDIT_TRANSPLANT, mode, seed)
        truth = transplant.mask512.numpy().astype(bool)
        assert (truth & ~rect_mask(transplant.rect)).sum() == 0


def test_outpainting_keeps_the_watermark_inside_and_foreign_content_outside() -> None:
    sample, base = make(EDIT_OUTPAINTING)
    image = sample.image.float() / 255
    inside = torch.from_numpy(rect_mask(sample.rect))
    expected_inside = (base + EMBED_DELTA).clamp(0, 1)
    assert torch.allclose(image[:, inside], expected_inside[:, inside], atol=1 / 255)
    assert not torch.allclose(image[:, ~inside], base[:, ~inside], atol=1 / 255)


def test_sample_is_deterministic_and_typed() -> None:
    first, _ = make(EDIT_REMOVAL)
    second, _ = make(EDIT_REMOVAL)
    assert torch.equal(first.image, second.image)
    assert first.image.dtype == torch.uint8
    assert first.image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert first.mask512.shape == (IMAGE_SIZE, IMAGE_SIZE)
    with pytest.raises(ValueError, match="edit"):
        make("teleport")
    with pytest.raises(ValueError, match="embed_mode"):
        make(EDIT_REMOVAL, "bogus")


def test_generate_is_seeded_balanced_and_writes_a_readable_dataset(
    sources: list[Path], tmp_path: Path
) -> None:
    samples = list(generate_samples(sources, fake_embed, 2, seed=SEED, n_bits=N_BITS))
    again = list(generate_samples(sources, fake_embed, 2, seed=SEED, n_bits=N_BITS))
    assert [s.edit for s in samples] == [e for e in EDIT_TYPES for _ in range(2)]
    assert all(torch.equal(a.image, b.image) for a, b in zip(samples, again, strict=True))
    out = tmp_path / "synth"
    assert write_synth_dataset(out, samples) == 8
    loaded = read_manifest(out / "manifest.json")
    assert [s.name for s in loaded] == [s.name for s in samples]
    assert all(s.method == METHOD_PIXELSEAL and s.edit for s in loaded)
    first = loaded[0]
    image = read_image_tensor(out / first.image)
    assert torch.allclose(image, samples[0].image.float() / 255)
    assert first.mask512 is not None
    truth = read_truth_mask128(out / first.mask512)
    assert truth.shape == (GRID_SIZE, GRID_SIZE)
    assert torch.equal(truth, downsample_mask_to_grid(samples[0].mask512))
    assert parse_message_bits(first.message).numel() == N_BITS


def test_generate_needs_two_sources(sources: list[Path]) -> None:
    with pytest.raises(ManifestError, match="two"):
        list(generate_samples(sources[:1], fake_embed, 1))


def test_list_sources_errors(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        list_source_images([tmp_path / "nope"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifestError, match="no image"):
        list_source_images([tmp_path / "empty"])


def test_message_bits_round_trip_and_validation() -> None:
    bits = parse_message_bits("0110")
    assert bits.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert bits_to_string(bits) == "0110"
    for bad in ("", "01a0", "  "):
        with pytest.raises(ManifestError):
            parse_message_bits(bad)


def test_manifest_validation(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)).save(tmp_path / "a.png")
    good = ManifestSample("a", "a.png", METHOD_PIXELSEAL, "0101")
    path = tmp_path / "m.json"
    write_manifest(path, [good])
    assert read_manifest(path) == [good]
    write_manifest(path, [good, good])
    with pytest.raises(ManifestError, match="duplicate"):
        read_manifest(path)
    write_manifest(path, [ManifestSample("a", "a.png", "weird", "01")])
    with pytest.raises(ManifestError, match="unknown method"):
        read_manifest(path)
    write_manifest(path, [ManifestSample("a", "missing.png", METHOD_PIXELSEAL, "01")])
    with pytest.raises(ManifestError, match="image not found"):
        read_manifest(path)
    path.write_text("{not json")
    with pytest.raises(ManifestError, match="malformed"):
        read_manifest(path)
    with pytest.raises(ManifestError, match="not found"):
        read_manifest(tmp_path / "absent.json")


def test_wrong_size_image_is_rejected(tmp_path: Path) -> None:
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(tmp_path / "small.png")
    with pytest.raises(ManifestError, match="expected 512x512"):
        read_image_tensor(tmp_path / "small.png")
