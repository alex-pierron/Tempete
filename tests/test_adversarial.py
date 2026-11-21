"""Tests of wmkit.adversarial on a fake differentiable extractor (no pretrained weights)."""

from __future__ import annotations

import pytest
import torch

from wmkit.adversarial import (
    AdversarialInputError,
    add_pattern,
    estimate_pattern,
    pgd_forge,
    pgd_remove,
    subtract_pattern,
)

SEED = 0
N_BITS = 16
IMAGE_SIZE = 32
BATCH = 3
EPS = 8.0 / 255.0
ATTACK_EPS = 3 * EPS  # attacker budget well above the mark's margin
STEP = 2.0 / 255.0
STEPS = 30
FLOAT_TOLERANCE = 1e-6
PATTERN_AMPLITUDE = 0.04
CONTENT_AMPLITUDE = 0.3
COLLUSION_IMAGES = 32
MIN_CORRELATION = 0.8


class FakeExtractor:
    """Bit logits = fixed random linear projection of the image (sign = bit). Differentiable."""

    def __init__(self, n_bits: int = N_BITS, size: int = IMAGE_SIZE) -> None:
        generator = torch.Generator().manual_seed(SEED)
        self.weights = torch.randn(n_bits, 3 * size * size, generator=generator) / size

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        return (images - 0.5).flatten(1) @ self.weights.T


def _bit_acc(logits: torch.Tensor, bits: torch.Tensor) -> float:
    return float(((logits > 0).float() == bits).float().mean())


def _images(batch: int = BATCH) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.rand(batch, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.6 + 0.2


def _message(batch: int = BATCH) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 2)
    return (torch.rand(batch, N_BITS, generator=generator) > 0.5).float()


def _sign_message_images(extractor: FakeExtractor, images: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Craft images whose logits already read `bits` (stand-in for a watermarked image)."""
    return pgd_forge(extractor.logits, images, bits, eps=EPS, step=STEP, steps=STEPS)


def test_pgd_forge_reaches_target_and_respects_budget() -> None:
    extractor, images, target = FakeExtractor(), _images(), _message()
    before = _bit_acc(extractor.logits(images), target)
    forged = pgd_forge(extractor.logits, images, target, eps=EPS, step=STEP, steps=STEPS)
    after = _bit_acc(extractor.logits(forged), target)
    assert after > before
    assert after >= 0.95
    assert float((forged - images).abs().max()) <= EPS + FLOAT_TOLERANCE
    assert float(forged.min()) >= 0.0 and float(forged.max()) <= 1.0
    assert forged.shape == images.shape


def test_pgd_forge_accepts_single_message_broadcast() -> None:
    extractor, images = FakeExtractor(), _images()
    target = _message(1)[0]
    forged = pgd_forge(extractor.logits, images, target, eps=EPS, step=STEP, steps=STEPS)
    assert _bit_acc(extractor.logits(forged), target.expand(BATCH, -1)) >= 0.95


def test_pgd_remove_untargeted_shrinks_logits() -> None:
    extractor, images, bits = FakeExtractor(), _images(), _message()
    marked = _sign_message_images(extractor, images, bits)
    attacked = pgd_remove(extractor.logits, marked, eps=ATTACK_EPS, step=STEP, steps=STEPS)
    before = float(extractor.logits(marked).abs().mean())
    after = float(extractor.logits(attacked).abs().mean())
    assert after < 0.5 * before
    assert float((attacked - marked).abs().max()) <= ATTACK_EPS + FLOAT_TOLERANCE
    assert float(attacked.min()) >= 0.0 and float(attacked.max()) <= 1.0


def test_pgd_remove_targeted_flips_bits() -> None:
    extractor, images, bits = FakeExtractor(), _images(), _message()
    marked = _sign_message_images(extractor, images, bits)
    assert _bit_acc(extractor.logits(marked), bits) >= 0.95
    attacked = pgd_remove(
        extractor.logits, marked, eps=ATTACK_EPS, step=STEP, steps=STEPS, target_bits=bits
    )
    assert _bit_acc(extractor.logits(attacked), bits) < 0.5


def test_zero_steps_returns_input_unchanged() -> None:
    extractor, images = FakeExtractor(), _images()
    assert torch.equal(pgd_remove(extractor.logits, images, steps=0), images)
    assert torch.equal(pgd_forge(extractor.logits, images, _message(), steps=0), images)


def test_pgd_does_not_mutate_input_and_is_deterministic() -> None:
    extractor, images = FakeExtractor(), _images()
    snapshot = images.clone()
    first = pgd_remove(extractor.logits, images, eps=EPS, step=STEP, steps=5)
    second = pgd_remove(extractor.logits, images, eps=EPS, step=STEP, steps=5)
    assert torch.equal(images, snapshot)
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("eps", "step", "steps"), [(-0.1, 0.01, 1), (0.1, 0.0, 1), (0.1, 0.01, -1)]
)
def test_invalid_budget_is_rejected(eps: float, step: float, steps: int) -> None:
    extractor = FakeExtractor()
    with pytest.raises(AdversarialInputError):
        pgd_remove(extractor.logits, _images(), eps=eps, step=step, steps=steps)


def test_invalid_shapes_are_rejected() -> None:
    extractor = FakeExtractor()
    with pytest.raises(AdversarialInputError, match="images"):
        pgd_forge(extractor.logits, torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), _message(1)[0])
    with pytest.raises(AdversarialInputError, match="target_message"):
        pgd_forge(extractor.logits, _images(), _message(2))  # 2 messages for 3 images
    with pytest.raises(AdversarialInputError, match="target_bits"):
        pgd_remove(extractor.logits, _images(), target_bits=_message(2))


# --- pattern averaging ---


def _collusion_set(n_images: int, smooth: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """N smooth images plus the same fixed high-frequency pattern; returns (images, pattern)."""
    generator = torch.Generator().manual_seed(SEED + 3)
    coarse = torch.rand(n_images, 3, 4, 4, generator=generator)
    base = torch.nn.functional.interpolate(coarse, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")
    pattern = (torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) - 0.5) * 2
    pattern = pattern * PATTERN_AMPLITUDE
    content = base * CONTENT_AMPLITUDE + 0.35
    return (content + pattern).clamp(0, 1), pattern


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((a_flat @ b_flat) / (a_flat.norm() * b_flat.norm()))


@pytest.mark.parametrize("denoise", ["gaussian", "median"])
def test_estimate_pattern_recovers_shared_pattern(denoise: str) -> None:
    images, pattern = _collusion_set(COLLUSION_IMAGES)
    estimated = estimate_pattern(images, denoise=denoise)
    assert estimated.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert _correlation(estimated, pattern) > MIN_CORRELATION


def test_estimate_pattern_improves_with_more_images() -> None:
    images, pattern = _collusion_set(COLLUSION_IMAGES)
    few = _correlation(estimate_pattern(images[:2]), pattern)
    many = _correlation(estimate_pattern(images), pattern)
    assert many >= few


def test_subtract_pattern_moves_images_towards_the_unmarked_content() -> None:
    images, pattern = _collusion_set(COLLUSION_IMAGES)
    unmarked = images - pattern  # oracle: what the images would be without the pattern
    cleaned = subtract_pattern(images, estimate_pattern(images))
    assert float((cleaned - unmarked).abs().mean()) < float((images - unmarked).abs().mean())


def test_add_then_subtract_round_trips_inside_range() -> None:
    images = _images() * 0.5 + 0.25
    pattern = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.01
    restored = subtract_pattern(add_pattern(images, pattern), pattern)
    assert torch.allclose(restored, images, atol=1e-6)


def test_pattern_strength_scales_linearly_and_clamps() -> None:
    images = torch.full((1, 3, 4, 4), 0.5)
    pattern = torch.full((3, 4, 4), 0.1)
    assert torch.allclose(add_pattern(images, pattern, 2.0), torch.full_like(images, 0.7))
    assert float(add_pattern(images, pattern, 100.0).max()) == 1.0
    assert float(subtract_pattern(images, pattern, 100.0).min()) == 0.0


def test_pattern_validation() -> None:
    images = _images()
    with pytest.raises(AdversarialInputError, match="denoise"):
        estimate_pattern(images, denoise="bilateral")
    with pytest.raises(AdversarialInputError, match="pattern"):
        subtract_pattern(images, torch.zeros(3, 8, 8))
    with pytest.raises(AdversarialInputError, match="images"):
        estimate_pattern(torch.rand(3, 8, 8))


# --- real Pixel Seal extractor (slow) ----------------------------------------------------------

REAL_IMAGES = 2
REAL_EPS = 4.0 / 255.0
REAL_STEP = 1.0 / 255.0
REAL_STEPS = 50
REAL_REMOVE_MAX_BIT_ACC = 0.75
REAL_FORGE_MIN_BIT_ACC = 0.9
REAL_CLEAN_MAX_BIT_ACC = 0.6
REAL_MIN_PSNR_DB = 35.0


def _real_setup() -> tuple[object, torch.Tensor, torch.Tensor]:
    import numpy as np
    from PIL import Image

    from wmkit.config import data_dir
    from wmkit.seal import load_seal

    paths = sorted((data_dir() / "kodak").glob("*.png"))[:REAL_IMAGES]
    if len(paths) < REAL_IMAGES:
        pytest.skip("Kodak not fetched: run scripts/fetch_data.py --dataset kodak")
    arrays = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths]
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    watermarker = load_seal(card="pixelseal")
    generator = torch.Generator().manual_seed(SEED)
    messages = (torch.rand(REAL_IMAGES, watermarker.n_bits, generator=generator) > 0.5).float()
    return watermarker, images, messages


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(torch.tensor(1.0 / float(((a - b) ** 2).mean()))))


@pytest.mark.slow
def test_real_pixelseal_pgd_remove_and_forge() -> None:
    from wmkit.seal import SealWatermarker

    watermarker, images, messages = _real_setup()
    assert isinstance(watermarker, SealWatermarker)
    marked = (watermarker.embed(images, messages) * 255).round() / 255
    assert _bit_acc(watermarker.extract_logits(marked), messages) > 0.95

    removed = pgd_remove(
        watermarker.extract_logits, marked, eps=REAL_EPS, step=REAL_STEP, steps=REAL_STEPS
    )
    removed = (removed * 255).round() / 255
    assert _bit_acc(watermarker.extract_logits(removed), messages) < REAL_REMOVE_MAX_BIT_ACC
    assert float((removed - marked).abs().max()) <= REAL_EPS + 1.0 / 255.0
    assert _psnr(marked, removed) > REAL_MIN_PSNR_DB

    assert _bit_acc(watermarker.extract_logits(images), messages) < REAL_CLEAN_MAX_BIT_ACC
    forged = pgd_forge(
        watermarker.extract_logits, images, messages, eps=REAL_EPS, step=REAL_STEP, steps=REAL_STEPS
    )
    forged = (forged * 255).round() / 255
    assert _bit_acc(watermarker.extract_logits(forged), messages) > REAL_FORGE_MIN_BIT_ACC
    assert _psnr(images, forged) > REAL_MIN_PSNR_DB
