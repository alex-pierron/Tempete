"""Tests for the classical baselines (wmkit.classical.*)."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pytest
import torch
from conftest import load_sample

from wmkit.attacks import get_attack
from wmkit.classical import (
    ClassicalWatermarkError,
    balanced_assignment,
    block_dct,
    block_idct,
    dct_matrix,
    luma,
    pool_by_bit,
    qim_embed,
    qim_margin,
)
from wmkit.classical.dwt_dct_svd import haar_ll, inverse_haar_ll
from wmkit.messages import random_messages
from wmkit.methods import get_watermarker
from wmkit.metrics import bit_accuracy, psnr
from wmkit.types import Watermarker

METHOD_NAMES = ["dct_qim", "spread_spectrum", "dwt_dct_svd"]
SAMPLE_NAMES = ["astronaut", "coffee", "chelsea", "rocket"]
MIN_PSNR_DB = 38.0
MIN_BIT_ACCURACY = 0.95
DEFAULT_BITS = 32
CHANCE_BAND = 0.2


def build_method(name: str, **kwargs: object) -> Watermarker:
    module = importlib.import_module(f"wmkit.classical.{name}")
    factory: Callable[..., Watermarker] = module.build
    return factory(**kwargs)


def message_for(n_bits: int, batch: int = 1, seed: int = 0) -> torch.Tensor:
    return random_messages(batch, n_bits, torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("side", [256, 512])
@pytest.mark.parametrize("sample", SAMPLE_NAMES)
def test_no_attack_meets_the_acceptance_targets(name: str, side: int, sample: str) -> None:
    method = build_method(name)
    image = load_sample(sample, (side, side))[None]
    message = message_for(method.n_bits)
    marked = method.embed(image, message)
    assert marked.shape == image.shape
    assert psnr(marked, image).item() >= MIN_PSNR_DB
    assert bit_accuracy(method.extract(marked), message).item() >= MIN_BIT_ACCURACY


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_registry_builds_each_method_with_32_bits(name: str) -> None:
    method = get_watermarker(name)
    assert isinstance(method, Watermarker)
    assert method.name == name
    assert method.n_bits == DEFAULT_BITS


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_soft_bits_are_probabilities(name: str) -> None:
    method = build_method(name)
    image = load_sample("coffee", (256, 256))[None]
    soft = method.extract(method.embed(image, message_for(method.n_bits)))
    assert soft.shape == (1, method.n_bits)
    assert soft.dtype == torch.float32
    assert soft.min() >= 0.0 and soft.max() <= 1.0


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("shape", [(96, 96), (200, 300), (301, 257), (640, 427)])
def test_arbitrary_sizes_round_trip(name: str, shape: tuple[int, int]) -> None:
    method = build_method(name)
    image = load_sample("rocket", shape)[None]
    message = message_for(method.n_bits)
    marked = method.embed(image, message)
    assert marked.shape == image.shape
    assert marked.min() >= 0.0 and marked.max() <= 1.0
    assert bit_accuracy(method.extract(marked), message).item() >= MIN_BIT_ACCURACY


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_extract_survives_rescaling_and_moderate_jpeg(name: str) -> None:
    method = build_method(name)
    image = load_sample("chelsea", (512, 512))[None]
    message = message_for(method.n_bits)
    marked = method.embed(image, message)
    for attack in ("resize_0.5", "resize_0.75", "jpeg_70", "gaussian_blur_3"):
        attacked = get_attack(attack).fn(marked)
        accuracy = bit_accuracy(method.extract(attacked), message).item()
        assert accuracy >= MIN_BIT_ACCURACY, (name, attack, accuracy)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_unwatermarked_images_decode_at_chance_level(name: str) -> None:
    method = build_method(name)
    accuracies = []
    for index, sample in enumerate(SAMPLE_NAMES):
        image = load_sample(sample, (256, 256))[None]
        accuracies.append(
            bit_accuracy(method.extract(image), message_for(method.n_bits, seed=index)).item()
        )
    assert abs(float(np.mean(accuracies)) - 0.5) < CHANCE_BAND


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_wrong_key_cannot_read_the_message(name: str) -> None:
    owner = build_method(name, key=1)
    intruder = build_method(name, key=2)
    image = load_sample("astronaut", (256, 256))[None]
    message = message_for(owner.n_bits)
    marked = owner.embed(image, message)
    assert bit_accuracy(owner.extract(marked), message).item() >= MIN_BIT_ACCURACY
    assert abs(bit_accuracy(intruder.extract(marked), message).item() - 0.5) < 0.3


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_embedding_is_deterministic_and_batch_independent(name: str) -> None:
    method = build_method(name)
    images = torch.stack([load_sample("astronaut", (128, 128)), load_sample("coffee", (128, 128))])
    messages = message_for(method.n_bits, batch=2)
    batched = method.embed(images, messages)
    assert torch.equal(batched, method.embed(images, messages))
    for index in range(2):
        single = method.embed(images[index:index + 1], messages[index:index + 1])
        assert torch.allclose(single, batched[index:index + 1], atol=1e-6)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_only_luminance_is_modified(name: str) -> None:
    method = build_method(name)
    image = load_sample("coffee", (256, 256))[None].clamp(0.1, 0.9)  # keep clear of clipping
    marked = method.embed(image, message_for(method.n_bits))
    difference = marked - image
    assert difference.abs().max() > 0.0
    assert torch.allclose(difference[:, 0], difference[:, 1], atol=1e-5)
    assert torch.allclose(difference[:, 1], difference[:, 2], atol=1e-5)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_strength_trades_quality_for_robustness(name: str) -> None:
    image = load_sample("rocket", (256, 256))[None]
    message = message_for(DEFAULT_BITS)
    default = build_method(name)
    default_psnr = psnr(default.embed(image, message), image).item()
    weak_scale, strong_scale = 0.5, 2.0
    base = {"dct_qim": 0.124, "spread_spectrum": 0.01, "dwt_dct_svd": 0.277}[name]
    weak = build_method(name, strength=base * weak_scale)
    strong = build_method(name, strength=base * strong_scale)
    weak_psnr = psnr(weak.embed(image, message), image).item()
    strong_psnr = psnr(strong.embed(image, message), image).item()
    assert weak_psnr > default_psnr > strong_psnr


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("n_bits", [8, 64])
def test_custom_payload_sizes(name: str, n_bits: int) -> None:
    method = build_method(name, n_bits=n_bits)
    assert method.n_bits == n_bits
    image = load_sample("chelsea", (256, 256))[None]
    message = message_for(n_bits)
    assert bit_accuracy(method.extract(method.embed(image, message)), message).item() >= 0.95


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_invalid_configurations_and_inputs_raise(name: str) -> None:
    with pytest.raises(ClassicalWatermarkError):
        build_method(name, strength=0.0)
    with pytest.raises(ClassicalWatermarkError):
        build_method(name, n_bits=0)
    with pytest.raises(ClassicalWatermarkError):
        build_method(name, n_bits=10**6)
    method = build_method(name)
    image = load_sample("coffee", (64, 64))[None]
    with pytest.raises(ClassicalWatermarkError):
        method.embed(image, torch.zeros(1, DEFAULT_BITS + 1))
    with pytest.raises(ClassicalWatermarkError):
        method.embed(image, torch.zeros(2, DEFAULT_BITS))
    with pytest.raises(ClassicalWatermarkError):
        method.embed(image[0], torch.zeros(1, DEFAULT_BITS))
    with pytest.raises(ClassicalWatermarkError):
        method.extract(torch.zeros(1, 1, 64, 64))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
@pytest.mark.parametrize("name", METHOD_NAMES)
def test_outputs_follow_the_input_device(name: str) -> None:
    method = build_method(name, device="mps")
    image = load_sample("astronaut", (256, 256))[None].to("mps")
    message = message_for(method.n_bits).to("mps")
    marked = method.embed(image, message)
    assert marked.device.type == "mps"
    assert method.extract(marked).device.type == "mps"


# --- shared building blocks -------------------------------------------------------------


@pytest.mark.parametrize("size", [4, 8])
def test_dct_matrix_is_orthonormal(size: int) -> None:
    matrix = dct_matrix(size)
    assert torch.allclose(matrix @ matrix.T, torch.eye(size, dtype=torch.float64), atol=1e-12)


def test_block_dct_round_trip_and_energy_preservation() -> None:
    plane = torch.rand(2, 32, 48, dtype=torch.float64)
    blocks = block_dct(plane, 8)
    assert blocks.shape == (2, 4, 6, 8, 8)
    assert torch.allclose(block_idct(blocks), plane, atol=1e-12)
    assert blocks.pow(2).sum().item() == pytest.approx(plane.pow(2).sum().item())
    with pytest.raises(ClassicalWatermarkError):
        block_dct(torch.rand(1, 30, 32, dtype=torch.float64), 8)


def test_block_dct_dc_is_eight_times_the_block_mean() -> None:
    plane = torch.full((1, 8, 8), 0.5, dtype=torch.float64)
    assert block_dct(plane, 8)[0, 0, 0, 0, 0].item() == pytest.approx(4.0)


def test_luma_is_bt601() -> None:
    red = torch.zeros(1, 3, 2, 2)
    red[:, 0] = 1.0
    assert luma(red)[0, 0, 0].item() == pytest.approx(0.299)


def test_qim_lattices_and_margins() -> None:
    step = 0.5
    values = torch.tensor([0.37, -1.12, 2.9], dtype=torch.float64)
    zero = qim_embed(values, torch.zeros(3, dtype=torch.float64), step)
    one = qim_embed(values, torch.ones(3, dtype=torch.float64), step)
    assert torch.allclose(zero / step, torch.round(zero / step), atol=1e-9)
    assert torch.allclose(one / step - 0.5, torch.round(one / step - 0.5), atol=1e-9)
    assert ((zero - values).abs() <= step / 2 + 1e-12).all()
    assert ((one - values).abs() <= step / 2 + 1e-12).all()
    assert (qim_margin(zero, step) < -0.999).all()
    assert (qim_margin(one, step) > 0.999).all()
    assert qim_margin(torch.tensor([step / 4], dtype=torch.float64), step).item() == pytest.approx(0)


def test_haar_ll_matches_pywt() -> None:
    pywt = pytest.importorskip("pywt")
    plane = torch.rand(1, 16, 16, dtype=torch.float64)
    reference, _ = pywt.dwt2(plane[0].numpy(), "haar")
    assert np.allclose(haar_ll(plane)[0].numpy(), reference)
    # LL-only reconstruction equals pywt's with zeroed detail bands
    detail = (np.zeros((8, 8)),) * 3
    expected = pywt.idwt2((reference, detail), "haar")
    assert np.allclose(inverse_haar_ll(haar_ll(plane))[0].numpy(), expected)


def test_balanced_assignment_and_pooling() -> None:
    assignment = balanced_assignment(100, 8, np.random.default_rng(0))
    counts = torch.bincount(assignment, minlength=8)
    assert counts.max() - counts.min() <= 1
    pooled = pool_by_bit(torch.ones(2, 100, dtype=torch.float64), assignment, 8)
    assert torch.allclose(pooled, torch.ones(2, 8, dtype=torch.float64))
    with pytest.raises(ClassicalWatermarkError):
        balanced_assignment(4, 8, np.random.default_rng(0))
