"""Tests of the simulated altered copies: shape, dtype, determinism and what each one does."""

from __future__ import annotations

import pytest
import torch
from test_hk_unlearning_support import random_images

from hackathon.unlearning.alterations import (
    ALTERATIONS,
    apply_alteration,
    color_jitter,
    horizontal_flip,
    jpeg_compress,
    mask_patches,
    random_augment,
    random_crop_resize,
    strong_view,
)

COUNT = 6


@pytest.mark.parametrize("name", sorted(ALTERATIONS))
def test_every_alteration_keeps_shape_and_dtype(name: str) -> None:
    images = random_images(COUNT)
    altered = apply_alteration(name, images, seed=0)
    assert altered.shape == images.shape and altered.dtype == torch.uint8


@pytest.mark.parametrize("name", sorted(ALTERATIONS))
def test_alterations_are_deterministic_for_a_seed_and_do_not_mutate_input(name: str) -> None:
    images = random_images(COUNT)
    snapshot = images.clone()
    first = apply_alteration(name, images, seed=3)
    second = apply_alteration(name, images, seed=3)
    assert torch.equal(first, second)
    assert torch.equal(images, snapshot)


@pytest.mark.parametrize("name", ["crop", "jitter"])
def test_random_alterations_depend_on_the_seed(name: str) -> None:
    images = random_images(COUNT)
    assert not torch.equal(
        apply_alteration(name, images, seed=1), apply_alteration(name, images, seed=2)
    )


@pytest.mark.parametrize("name", sorted(ALTERATIONS))
def test_every_alteration_changes_the_pixels(name: str) -> None:
    images = random_images(COUNT)
    assert not torch.equal(apply_alteration(name, images, seed=0), images)


def test_flip_is_an_involution_and_mirrors_columns() -> None:
    images = random_images(COUNT)
    generator = torch.Generator().manual_seed(0)
    flipped = horizontal_flip(images, generator)
    assert torch.equal(flipped[:, :, 0], images[:, :, -1])
    assert torch.equal(horizontal_flip(flipped, generator), images)


def test_full_frame_crop_is_close_to_identity() -> None:
    smooth = torch.linspace(0, 255, 224).view(1, 1, 224, 1).expand(2, 224, 224, 3).to(torch.uint8)
    generator = torch.Generator().manual_seed(0)
    cropped = random_crop_resize(smooth, generator, min_area=1.0, max_area=1.0)
    difference = (cropped.float() - smooth.float()).abs().mean()
    assert float(difference) < 2.0


def test_jitter_leaves_a_gray_image_gray_under_saturation_only_changes() -> None:
    gray = torch.full((2, 224, 224, 3), 128, dtype=torch.uint8)
    jittered = color_jitter(gray, torch.Generator().manual_seed(0))
    channel_spread = jittered.float().amax(dim=3) - jittered.float().amin(dim=3)
    assert float(channel_spread.max()) <= 1.0


def test_jpeg_round_trip_is_lossy_but_close() -> None:
    smooth = torch.linspace(0, 255, 224).view(1, 224, 1, 1).expand(2, 224, 224, 3).to(torch.uint8)
    compressed = jpeg_compress(smooth, torch.Generator())
    mean_error = (compressed.float() - smooth.float()).abs().mean()
    assert 0.0 < float(mean_error) < 10.0
    assert jpeg_compress(smooth[:0], torch.Generator()).shape[0] == 0


def test_strong_view_is_stronger_than_the_mild_crop() -> None:
    gradient = torch.linspace(0, 255, 224).view(1, 1, 224, 1).expand(4, 224, 224, 3).to(torch.uint8)
    mild = random_crop_resize(gradient, torch.Generator().manual_seed(0))
    strong = strong_view(gradient, torch.Generator().manual_seed(0), min_area=0.1)
    assert strong.shape == gradient.shape
    assert (strong.float() - gradient.float()).abs().mean() > (
        mild.float() - gradient.float()
    ).abs().mean() * 0.5


@pytest.mark.parametrize("fraction", [0.15, 0.2, 0.3])
def test_mask_patches_grays_out_exactly_the_share_of_patches(fraction: float) -> None:
    images = random_images(COUNT)
    masked = mask_patches(images, torch.Generator().manual_seed(0), fraction)
    assert masked.shape == images.shape and masked.dtype == torch.uint8
    gray = (masked == 128).all(dim=3).view(COUNT, 14, 16, 14, 16).all(dim=4).all(dim=2)
    assert gray.sum(dim=(1, 2)).tolist() == [round(fraction * 196)] * COUNT
    untouched = ~gray.repeat_interleave(16, 1).repeat_interleave(16, 2)
    assert torch.equal(masked[untouched], images[untouched])


def test_mask_patches_is_deterministic_for_a_generator_and_varies_per_image() -> None:
    images = random_images(COUNT)
    snapshot = images.clone()
    first = mask_patches(images, torch.Generator().manual_seed(3), 0.2)
    assert torch.equal(first, mask_patches(images, torch.Generator().manual_seed(3), 0.2))
    assert not torch.equal(first, mask_patches(images, torch.Generator().manual_seed(4), 0.2))
    assert torch.equal(images, snapshot)
    same = images[:1].expand(2, -1, -1, -1)
    twice = mask_patches(same, torch.Generator().manual_seed(3), 0.2)
    assert not torch.equal(twice[0], twice[1])  # each image draws its own patches


def test_random_augment_returns_a_valid_batch() -> None:
    images = random_images(COUNT)
    out = random_augment(images, torch.Generator().manual_seed(5))
    assert out.shape == images.shape and out.dtype == torch.uint8


def test_unknown_alteration_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError, match="flip"):
        apply_alteration("blur", random_images(1), seed=0)
