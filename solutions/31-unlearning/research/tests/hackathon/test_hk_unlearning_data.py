"""Tests of forget-set loading and retain-proxy image loading (tiny synthetic files only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_hk_unlearning_support import random_images

from hackathon.unlearning.data import (
    DataError,
    is_full_forget_set,
    list_proxy_image_files,
    load_forget_images,
    load_proxy_images,
    load_resized_image,
    select_subset,
)


def _save_npz(path: Path, array: np.ndarray, key: str = "images") -> Path:
    np.savez(path, **{key: array})
    return path


def test_load_forget_images_round_trip_and_limit(tmp_path: Path) -> None:
    array = random_images(4).numpy()
    path = _save_npz(tmp_path / "f.npz", array)
    assert np.array_equal(load_forget_images(path).numpy(), array)
    assert load_forget_images(path, limit=2).shape == (2, 224, 224, 3)


def test_channels_first_input_is_tolerated_like_the_official_loader(tmp_path: Path) -> None:
    array = random_images(2).numpy().transpose(0, 3, 1, 2)
    loaded = load_forget_images(_save_npz(tmp_path / "cf.npz", array))
    assert loaded.shape == (2, 224, 224, 3)


@pytest.mark.parametrize(
    ("array", "key", "match"),
    [
        (np.zeros((1, 224, 224, 3), dtype=np.uint8), "pixels", "no 'images'"),
        (np.zeros((1, 224, 224, 3), dtype=np.float32), "images", "uint8"),
        (np.zeros((224, 224, 3), dtype=np.uint8), "images", "4 dimensions"),
        (np.zeros((1, 100, 100, 3), dtype=np.uint8), "images", "Expected"),
    ],
)
def test_load_forget_images_rejects_malformed_archives(
    tmp_path: Path, array: np.ndarray, key: str, match: str
) -> None:
    with pytest.raises(DataError, match=match):
        load_forget_images(_save_npz(tmp_path / "bad.npz", array, key))


def test_missing_forget_file_points_to_fetch(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="fetch"):
        load_forget_images(tmp_path / "missing.npz")


def test_is_full_forget_set_only_for_3500() -> None:
    assert not is_full_forget_set(random_images(3))


def test_select_subset_is_reproducible_sorted_and_clipped() -> None:
    images = torch.arange(20, dtype=torch.uint8).view(20, 1, 1, 1).expand(20, 2, 2, 3)
    first = select_subset(images, 5, seed=1)
    assert torch.equal(first, select_subset(images, 5, seed=1))
    ids = first[:, 0, 0, 0].tolist()
    assert ids == sorted(ids) and len(set(ids)) == 5
    assert select_subset(images, 99, seed=1) is images


def _make_proxy_tree(root: Path) -> None:
    for folder, count, size in (("kodak", 2, (300, 200)), ("div2k_valid", 3, (500, 400))):
        (root / folder).mkdir()
        for index in range(count):
            array = np.random.default_rng(index).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
            Image.fromarray(array).save(root / folder / f"{index}.png")
    (root / "kodak" / "notes.txt").write_text("not an image")


def test_resized_image_is_224_square_uint8(tmp_path: Path) -> None:
    _make_proxy_tree(tmp_path)
    image = load_resized_image(tmp_path / "kodak" / "0.png")
    assert tuple(image.shape) == (224, 224, 3) and image.dtype == torch.uint8


def test_undecodable_image_raises_a_typed_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(DataError, match="Cannot read"):
        load_resized_image(broken)


def test_proxy_listing_skips_missing_folders_and_non_images(tmp_path: Path) -> None:
    _make_proxy_tree(tmp_path)
    names = [p.name for p in list_proxy_image_files(tmp_path)]
    assert len(names) == 5 and "notes.txt" not in names


def test_load_proxy_images_is_reproducible_and_bounded(tmp_path: Path) -> None:
    _make_proxy_tree(tmp_path)
    first = load_proxy_images(tmp_path, count=3, seed=0)
    assert first.shape == (3, 224, 224, 3)
    assert torch.equal(first, load_proxy_images(tmp_path, count=3, seed=0))
    assert load_proxy_images(tmp_path, count=50, seed=0).shape[0] == 5


def test_load_proxy_images_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="No proxy images"):
        load_proxy_images(tmp_path, count=2, seed=0)
    with pytest.raises(DataError, match="at least one"):
        load_proxy_images(tmp_path, count=0, seed=0)
