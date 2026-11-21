"""Unit tests (QA pass) of the classical watermarkers on small synthetic images.

`tests/test_classical.py` uses skimage photographs at 96 px and above, one message per case.
Here the images are synthetic (gradient + texture + flat regions, built from a seeded generator),
several random messages are tried per case, PSNR is recomputed with numpy, and the size range is
pushed below 96 px, which exposes a limit of two of the three methods.

Finding pinned with `xfail(strict=True)`:
    F-CLS-1  `dct_qim` and `spread_spectrum` do not decode images of 48 px (bit accuracy 0.4 to
             0.7), whereas `Watermarker` promises "arbitrary H and W". `dwt_dct_svd` decodes
             down to 32 px.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from wmkit.messages import random_messages
from wmkit.metrics import bit_accuracy, psnr
from wmkit.types import Watermarker

METHOD_NAMES = ["dct_qim", "spread_spectrum", "dwt_dct_svd"]
MIN_PSNR_DB = 38.0
"""Same acceptance floor as the contract's 40 dB target minus clipping losses."""
MIN_BIT_ACCURACY = 0.95
N_MESSAGES = 6
SYNTHETIC_SIDES = (96, 128, 192)
BATCH = 3
CHANCE_BAND = 0.2


def build(name: str, **kwargs: object) -> Watermarker:
    """Instantiate a method through its module factory."""
    factory = importlib.import_module(f"wmkit.classical.{name}").build
    method: Watermarker = factory(**kwargs)
    return method


def synthetic_images(batch: int, height: int, width: int, seed: int) -> torch.Tensor:
    """Smooth colour gradients, a bright disc, a flat block and mild texture noise, in [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, height), torch.linspace(0.0, 1.0, width), indexing="ij"
    )
    images = []
    for _ in range(batch):
        phase = torch.rand(3, generator=generator) * 6.28
        channels = torch.stack(
            [0.5 + 0.35 * torch.sin(3.0 * xs + phase[0]),
             0.5 + 0.35 * torch.cos(2.0 * ys + phase[1]),
             0.5 + 0.35 * torch.sin(2.0 * (xs + ys) + phase[2])]
        )
        disc = ((xs - 0.3) ** 2 + (ys - 0.6) ** 2 < 0.04).float()
        channels = channels * (1.0 - disc) + disc * 0.95
        channels[:, : height // 5, -(width // 5) :] = 0.5  # flat block
        channels = channels + 0.04 * torch.randn(channels.shape, generator=generator)
        images.append(channels.clamp(0.0, 1.0))
    return torch.stack(images)


def messages_for(method: Watermarker, batch: int, seed: int) -> torch.Tensor:
    """Seeded random payloads for `method`."""
    return random_messages(batch, method.n_bits, torch.Generator().manual_seed(seed))


def numpy_psnr(marked: torch.Tensor, original: torch.Tensor) -> np.ndarray:
    """Per-image PSNR in float64 numpy (peak 1.0), independent of `wmkit.metrics`."""
    difference = marked.numpy().astype(np.float64) - original.numpy().astype(np.float64)
    mse = (difference**2).reshape(difference.shape[0], -1).mean(axis=1)
    return -10.0 * np.log10(np.maximum(mse, 1e-10))


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("side", SYNTHETIC_SIDES)
def test_round_trip_is_exact_for_several_random_messages_on_synthetic_images(
    name: str, side: int
) -> None:
    method = build(name)
    images = synthetic_images(BATCH, side, side, seed=side)
    for seed in range(N_MESSAGES):
        messages = messages_for(method, BATCH, seed)
        marked = method.embed(images, messages)
        accuracy = bit_accuracy(method.extract(marked), messages)
        assert (accuracy >= MIN_BIT_ACCURACY).all(), (name, side, seed, accuracy.tolist())


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_psnr_agrees_between_wmkit_and_numpy_and_meets_the_floor(name: str) -> None:
    method = build(name)
    images = synthetic_images(BATCH, 128, 160, seed=1)
    marked = method.embed(images, messages_for(method, BATCH, 0))
    reference = numpy_psnr(marked, images)
    assert psnr(marked, images).numpy() == pytest.approx(reference, abs=1e-3)
    assert (reference >= MIN_PSNR_DB).all(), reference
    assert np.isfinite(reference).all() and (reference < 80.0).all(), "the watermark must be present"


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_embed_keeps_shape_dtype_range_and_does_not_touch_the_input(name: str) -> None:
    method = build(name)
    images = synthetic_images(2, 100, 140, seed=2)
    snapshot = images.clone()
    marked = method.embed(images, messages_for(method, 2, 1))
    assert marked.shape == images.shape and marked.dtype == torch.float32
    assert marked.min() >= 0.0 and marked.max() <= 1.0
    assert torch.equal(images, snapshot)
    assert not torch.equal(marked, images)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_each_message_of_a_batch_is_read_back_on_its_own_image(name: str) -> None:
    """Swapped or averaged batch handling would show as cross-talk between rows."""
    method = build(name)
    images = synthetic_images(4, 128, 128, seed=3)
    messages = messages_for(method, 4, 2)
    decoded = (method.extract(method.embed(images, messages)) > 0.5).float()
    assert torch.equal(decoded, messages)
    rolled = torch.roll(messages, 1, dims=0)
    assert (bit_accuracy(decoded, rolled) < 0.9).all(), "distinct messages must stay distinct"


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_complementary_messages_decode_to_complementary_bits(name: str) -> None:
    method = build(name)
    image = synthetic_images(1, 128, 128, seed=4)
    message = messages_for(method, 1, 3)
    direct = method.extract(method.embed(image, message))
    inverse = method.extract(method.embed(image, 1.0 - message))
    assert bit_accuracy(direct, message).item() >= MIN_BIT_ACCURACY
    assert bit_accuracy(inverse, 1.0 - message).item() >= MIN_BIT_ACCURACY
    assert bit_accuracy(inverse, message).item() <= 1.0 - MIN_BIT_ACCURACY


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_all_zero_and_all_one_payloads_round_trip(name: str) -> None:
    method = build(name)
    image = synthetic_images(1, 128, 128, seed=5)
    for fill in (0.0, 1.0):
        message = torch.full((1, method.n_bits), fill)
        assert bit_accuracy(method.extract(method.embed(image, message)), message).item() >= (
            MIN_BIT_ACCURACY
        )


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_decoding_clean_synthetic_images_is_chance_level_for_every_message(name: str) -> None:
    method = build(name)
    images = synthetic_images(BATCH, 128, 128, seed=6)
    accuracies = []
    for seed in range(N_MESSAGES):
        accuracies.extend(bit_accuracy(method.extract(images), messages_for(method, BATCH, seed)).tolist())
    assert abs(float(np.mean(accuracies)) - 0.5) < CHANCE_BAND
    assert max(accuracies) < MIN_BIT_ACCURACY, "an unmarked image must never decode as marked"


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_embedding_the_same_payload_twice_does_not_corrupt_it(name: str) -> None:
    method = build(name)
    image = synthetic_images(1, 128, 128, seed=7)
    message = messages_for(method, 1, 4)
    twice = method.embed(method.embed(image, message), message)
    assert bit_accuracy(method.extract(twice), message).item() >= MIN_BIT_ACCURACY


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("level", (0.0, 0.5, 1.0))
def test_flat_and_saturated_images_at_the_reference_size_round_trip(name: str, level: float) -> None:
    """Clipping on white / black must be absorbed by the closed loop (measured: exact)."""
    method = build(name)
    image = torch.full((1, 3, 256, 256), level)
    for seed in range(3):
        message = messages_for(method, 1, seed)
        assert bit_accuracy(method.extract(method.embed(image, message)), message).item() >= (
            MIN_BIT_ACCURACY
        )


def test_dwt_dct_svd_decodes_small_images_down_to_32_pixels() -> None:
    method = build("dwt_dct_svd")
    for side in (32, 48, 64):
        images = synthetic_images(2, side, side, seed=side)
        messages = messages_for(method, 2, side)
        accuracy = bit_accuracy(method.extract(method.embed(images, messages)), messages)
        assert (accuracy >= MIN_BIT_ACCURACY).all(), (side, accuracy.tolist())


@pytest.mark.xfail(
    strict=True,
    reason="F-CLS-1: dct_qim / spread_spectrum lose the payload below ~96 px (accuracy 0.4-0.7 at 48 px)",
)
@pytest.mark.parametrize("name", ["dct_qim", "spread_spectrum"])
def test_small_images_of_48_pixels_round_trip(name: str) -> None:
    method = build(name)
    images = synthetic_images(N_MESSAGES, 48, 48, seed=48)
    messages = messages_for(method, N_MESSAGES, 48)
    accuracy = bit_accuracy(method.extract(method.embed(images, messages)), messages)
    assert (accuracy >= MIN_BIT_ACCURACY).all(), accuracy.tolist()


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_one_pixel_and_tiny_images_keep_their_shape_and_range(name: str) -> None:
    """Even where decoding is hopeless, `embed` must honour the shape / range contract."""
    method = build(name)
    for side in (1, 2, 8):
        image = synthetic_images(1, side, side, seed=side)
        marked = method.embed(image, messages_for(method, 1, 0))
        assert marked.shape == image.shape
        assert torch.isfinite(marked).all() and marked.min() >= 0.0 and marked.max() <= 1.0
        soft = method.extract(marked)
        assert soft.shape == (1, method.n_bits) and soft.min() >= 0.0 and soft.max() <= 1.0
