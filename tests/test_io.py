"""Tests for wmkit.io."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from wmkit.io import ImageIOError, list_images, load_image, save_image

UINT8_STEP = 1.0 / 255.0


def test_png_round_trip_is_lossless_up_to_quantisation(tmp_path: Path) -> None:
    image = torch.rand(3, 20, 31)
    save_image(image, tmp_path / "x.png")
    loaded = load_image(tmp_path / "x.png")
    assert loaded.shape == (3, 20, 31)
    assert loaded.dtype == torch.float32
    assert (loaded - image).abs().max() <= UINT8_STEP / 2 + 1e-6


def test_save_png_is_idempotent_on_quantised_values(tmp_path: Path) -> None:
    image = torch.rand(3, 16, 16)
    save_image(image, tmp_path / "a.png")
    first = load_image(tmp_path / "a.png")
    save_image(first, tmp_path / "b.png")
    assert torch.equal(first, load_image(tmp_path / "b.png"))


def test_jpeg_extension_writes_a_jpeg_with_high_quality(tmp_path: Path) -> None:
    image = torch.full((3, 32, 32), 0.5)
    save_image(image, tmp_path / "x.jpg")
    with Image.open(tmp_path / "x.jpg") as handle:
        assert handle.format == "JPEG"
    assert (load_image(tmp_path / "x.jpg") - image).abs().max() < 0.02


def test_save_clamps_out_of_range_values(tmp_path: Path) -> None:
    image = torch.tensor([-1.0, 2.0]).repeat(3, 4, 2)
    save_image(image, tmp_path / "x.png")
    loaded = load_image(tmp_path / "x.png")
    assert loaded.min() == 0.0
    assert loaded.max() == 1.0


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    save_image(torch.zeros(3, 8, 8), tmp_path / "deep" / "er" / "x.png")
    assert (tmp_path / "deep" / "er" / "x.png").is_file()


def test_load_expands_grayscale_and_drops_alpha(tmp_path: Path) -> None:
    Image.new("L", (10, 6), 128).save(tmp_path / "gray.png")
    Image.new("RGBA", (10, 6), (255, 0, 0, 10)).save(tmp_path / "alpha.png")
    gray = load_image(tmp_path / "gray.png")
    alpha = load_image(tmp_path / "alpha.png")
    assert gray.shape == (3, 6, 10)
    assert torch.allclose(gray[0], gray[2])
    assert alpha.shape == (3, 6, 10)
    assert torch.allclose(alpha[0], torch.ones(6, 10))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_corrupt_file_raises_typed_error(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ImageIOError):
        load_image(tmp_path / "bad.png")


@pytest.mark.parametrize("shape", [(4, 8, 8), (8, 8), (1, 3, 8, 8)])
def test_save_rejects_wrong_shapes(tmp_path: Path, shape: tuple[int, ...]) -> None:
    with pytest.raises(ImageIOError):
        save_image(torch.zeros(shape), tmp_path / "x.png")


def test_save_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ImageIOError):
        save_image(torch.zeros(3, 8, 8), tmp_path / "x.bmp")


def test_list_images_sorted_filtered_and_limited(tmp_path: Path) -> None:
    for name in ["b.png", "a.JPG", "c.jpeg", "d.webp", "notes.txt", "e.gif"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "z.png").write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "b.png", "c.jpeg", "d.webp"]
    assert [p.name for p in list_images(tmp_path, max_images=2)] == ["a.JPG", "b.png"]
    assert list_images(tmp_path, max_images=0) == []


def test_list_images_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list_images(tmp_path / "missing")
    with pytest.raises(ValueError):
        list_images(tmp_path, max_images=-1)
