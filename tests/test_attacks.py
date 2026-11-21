"""Tests for wmkit.attacks."""

from __future__ import annotations

import sys

import pytest
import torch

from wmkit import attacks
from wmkit.attacks import (
    SUITES,
    AttackUnavailableError,
    attack_seed,
    get_attack,
    get_suite,
    list_attacks,
    reseed_attack,
    straight_through,
)
from wmkit.types import ATTACK_GROUPS, AttackSpec

EXPECTED_NAMES: dict[str, list[str]] = {
    "identity": ["identity"],
    "valuemetric": [
        "brightness_0.5", "brightness_0.75", "brightness_1.25", "brightness_1.5", "brightness_2",
        "contrast_0.5", "contrast_0.75", "contrast_1.5", "contrast_2",
        "saturation_0.5", "saturation_1.5",
        "hue_-0.1", "hue_0.1", "hue_0.25",
        "grayscale",
        "gaussian_blur_3", "gaussian_blur_9", "gaussian_blur_17",
        "gaussian_noise_0.02", "gaussian_noise_0.05",
    ],
    "compression": [
        "jpeg_40", "jpeg_50", "jpeg_60", "jpeg_70", "jpeg_80", "jpeg_90", "webp_50", "webp_80",
    ],
    "geometric": [
        "crop_0.33", "crop_0.5", "crop_0.71", "crop_0.9", "resize_0.5", "resize_0.75",
        "rotate_5", "rotate_10", "rotate_30", "rotate_90", "hflip",
        "perspective_0.1", "perspective_0.3", "perspective_0.5",
    ],
    "combined": [
        "combo_jpeg40_crop0.71_bright0.5", "combo_jpeg60_crop0.71_bright0.5",
        "combo_jpeg80_crop0.71_bright0.5",
    ],
    "removal": [
        "median_3", "median_5", "nlmeans_5", "nlmeans_10", "jpeg_10", "downup_0.25",
        "downup_0.5", "regen_vae",
    ],
}

QUICK_NAMES = [
    "identity", "brightness_1.5", "contrast_0.5", "hue_0.1", "gaussian_blur_9", "jpeg_50",
    "crop_0.71", "resize_0.5", "rotate_10", "hflip", "combo_jpeg40_crop0.71_bright0.5",
]

RANDOM_NAMES = [
    "gaussian_noise_0.02", "gaussian_noise_0.05", "crop_0.33", "crop_0.71", "rotate_10",
    "rotate_90", "perspective_0.1", "perspective_0.5", "combo_jpeg40_crop0.71_bright0.5",
]

FAST_NAMES = [name for name in list_attacks() if name != "regen_vae"]
PINNED_SEED = 0x28F889F1B4B1A48B


def all_expected_names() -> list[str]:
    return [name for group in ATTACK_GROUPS for name in EXPECTED_NAMES[group]]


def test_registry_names_and_groups_match_the_contract_exactly() -> None:
    for group in ATTACK_GROUPS:
        assert list_attacks(group) == EXPECTED_NAMES[group]
    assert list_attacks() == all_expected_names()
    assert len(list_attacks()) == len(set(list_attacks()))


def test_every_spec_carries_its_registry_name_and_group() -> None:
    for group in ATTACK_GROUPS:
        for name in list_attacks(group):
            spec = get_attack(name)
            assert (spec.name, spec.group) == (name, group)
            assert callable(spec.fn)


def test_suites_are_exactly_those_of_the_contract() -> None:
    assert list(SUITES) == ["quick", "paper", "removal", "all"]
    assert SUITES["quick"] == QUICK_NAMES
    assert SUITES["paper"] == [
        name for group in ATTACK_GROUPS if group != "removal" for name in EXPECTED_NAMES[group]
    ]
    assert SUITES["removal"] == EXPECTED_NAMES["removal"]
    assert SUITES["all"] == all_expected_names()
    assert [spec.name for spec in get_suite("quick")] == QUICK_NAMES


def test_unknown_names_raise_helpful_errors() -> None:
    with pytest.raises(KeyError, match="jpeg_50"):
        get_attack("jpeg_5O")
    with pytest.raises(KeyError):
        get_suite("nope")
    with pytest.raises(ValueError):
        list_attacks("nope")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", FAST_NAMES)
def test_attack_output_is_a_valid_image_batch(name: str, small_batch: torch.Tensor) -> None:
    output = get_attack(name).fn(small_batch)
    assert output.dtype == torch.float32
    assert output.ndim == 4 and output.shape[:2] == (2, 3)
    assert torch.isfinite(output).all()
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_output_sizes_follow_the_contract(small_batch: torch.Tensor) -> None:
    side = small_batch.shape[-1]
    assert get_attack("crop_0.5").fn(small_batch).shape[-2:] == (round(0.5 * side),) * 2
    assert get_attack("crop_0.9").fn(small_batch).shape[-2:] == (round(0.9 * side),) * 2
    assert get_attack("resize_0.75").fn(small_batch).shape[-2:] == (round(0.75 * side),) * 2
    for kept in ["rotate_30", "perspective_0.3", "downup_0.25", "hflip", "median_5", "jpeg_10"]:
        assert get_attack(kept).fn(small_batch).shape == small_batch.shape


def test_non_square_images_keep_their_aspect_ratio() -> None:
    images = torch.rand(1, 3, 60, 100)
    assert get_attack("crop_0.5").fn(images).shape == (1, 3, 30, 50)
    assert get_attack("resize_0.5").fn(images).shape == (1, 3, 30, 50)
    assert get_attack("rotate_10").fn(images).shape == images.shape
    assert get_attack("perspective_0.3").fn(images).shape == images.shape


def test_identity_and_flip(small_batch: torch.Tensor) -> None:
    assert torch.equal(get_attack("identity").fn(small_batch), small_batch)
    assert torch.equal(get_attack("hflip").fn(small_batch), small_batch.flip(-1))


def test_input_is_never_modified(small_batch: torch.Tensor) -> None:
    before = small_batch.clone()
    for name in ["identity", "brightness_2", "crop_0.5", "jpeg_40", "rotate_10"]:
        get_attack(name).fn(small_batch)
    assert torch.equal(before, small_batch)


def test_brightness_scales_and_clamps() -> None:
    images = torch.full((1, 3, 16, 16), 0.4)
    assert torch.allclose(get_attack("brightness_0.5").fn(images), torch.full_like(images, 0.2))
    assert torch.allclose(get_attack("brightness_2").fn(images), torch.full_like(images, 0.8))
    assert get_attack("brightness_2").fn(images * 2).max() == 1.0


def test_hue_parameter_is_a_fraction_of_the_colour_wheel() -> None:
    red = torch.zeros(1, 3, 8, 8)
    red[:, 0] = 1.0
    shifted = get_attack("hue_0.25").fn(red)[0, :, 0, 0]
    # red (hue 0) + 90 degrees = chartreuse (0.5, 1, 0)
    assert torch.allclose(shifted, torch.tensor([0.5, 1.0, 0.0]), atol=1e-4)
    backwards = get_attack("hue_-0.1").fn(red)[0, :, 0, 0]
    assert backwards[2] > 0.0  # negative shift moves red towards magenta (blue appears)


def test_grayscale_equalises_channels(small_batch: torch.Tensor) -> None:
    gray = get_attack("grayscale").fn(small_batch)
    assert torch.allclose(gray[:, 0], gray[:, 1]) and torch.allclose(gray[:, 1], gray[:, 2])


def test_blur_reduces_high_frequency_energy(small_batch: torch.Tensor) -> None:
    def energy(batch: torch.Tensor) -> float:
        return float((batch[..., 1:] - batch[..., :-1]).abs().mean())

    energies = [energy(get_attack(f"gaussian_blur_{k}").fn(small_batch)) for k in (3, 9, 17)]
    assert energy(small_batch) > energies[0] > energies[1] > energies[2]


def test_gaussian_noise_has_the_requested_standard_deviation() -> None:
    gray = torch.full((1, 3, 128, 128), 0.5)
    noise = get_attack("gaussian_noise_0.05").fn(gray) - gray
    assert noise.std().item() == pytest.approx(0.05, rel=0.05)


def test_jpeg_quality_orders_the_distortion(small_batch: torch.Tensor) -> None:
    errors = [
        (get_attack(f"jpeg_{q}").fn(small_batch) - small_batch).abs().mean().item()
        for q in (90, 70, 40, 10)
    ]
    assert errors == sorted(errors)
    assert errors[0] > 0.0


def test_webp_changes_the_image(small_batch: torch.Tensor) -> None:
    assert (get_attack("webp_50").fn(small_batch) - small_batch).abs().mean() > 0.0


def test_crop_is_a_window_of_the_input(small_image: torch.Tensor) -> None:
    batch = small_image[None]
    cropped = get_attack("crop_0.5").fn(batch)[0]
    crop_side = cropped.shape[-1]
    found = any(
        torch.equal(cropped, small_image[:, top:top + crop_side, left:left + crop_side])
        for top in range(small_image.shape[1] - crop_side + 1)
        for left in range(small_image.shape[2] - crop_side + 1)
    )
    assert found


def test_rotate_keeps_size_with_black_corners_and_90_degrees_is_a_quarter_turn() -> None:
    images = torch.ones(1, 3, 40, 40)
    rotated = get_attack("rotate_30").fn(images)
    assert rotated[0, :, 0, 0].sum() == 0.0  # corner pixel comes from outside the source
    assert rotated[0, :, 20, 20].min() == 1.0
    quarter = get_attack("rotate_90").fn(torch.rand(1, 3, 40, 40))
    reference = torch.rot90(torch.rand(1, 3, 40, 40), 1, (2, 3))
    assert quarter.shape == reference.shape


def test_median_removes_isolated_outliers() -> None:
    images = torch.full((1, 3, 32, 32), 0.5)
    images[:, :, 16, 16] = 1.0
    assert torch.allclose(get_attack("median_3").fn(images), torch.full_like(images, 0.5))


def test_nlmeans_reduces_noise() -> None:
    clean = torch.full((1, 3, 64, 64), 0.5)
    noisy = (clean + 0.08 * torch.randn(clean.shape, generator=torch.Generator().manual_seed(3))
             ).clamp(0, 1)
    denoised = get_attack("nlmeans_10").fn(noisy)
    assert (denoised - clean).abs().mean() < 0.5 * (noisy - clean).abs().mean()


def test_downup_loses_high_frequencies_more_when_smaller(small_batch: torch.Tensor) -> None:
    errors = [
        (get_attack(f"downup_{s}").fn(small_batch) - small_batch).abs().mean().item()
        for s in (0.5, 0.25)
    ]
    assert 0.0 < errors[0] < errors[1]


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_random_attacks_are_deterministic_at_equal_seed(
    name: str, small_batch: torch.Tensor
) -> None:
    first, second, third = get_attack(name), get_attack(name), get_attack(name)
    assert reseed_attack(first, 123) and reseed_attack(second, 123) and reseed_attack(third, 124)
    out_first, out_second, out_third = (s.fn(small_batch) for s in (first, second, third))
    assert torch.equal(out_first, out_second)
    assert not torch.equal(out_first, out_third)


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_reseeding_restarts_the_stream(name: str, small_batch: torch.Tensor) -> None:
    spec = get_attack(name)
    reseed_attack(spec, 5)
    first = spec.fn(small_batch)
    spec.fn(small_batch)  # advance the stream
    reseed_attack(spec, 5)
    assert torch.equal(first, spec.fn(small_batch))


def test_fresh_specs_start_from_the_same_default_seed(small_batch: torch.Tensor) -> None:
    assert torch.equal(get_attack("crop_0.5").fn(small_batch), get_attack("crop_0.5").fn(small_batch))


def test_random_crop_windows_differ_between_images_of_a_batch() -> None:
    images = torch.rand(1, 3, 64, 64).repeat(8, 1, 1, 1)
    crops = get_attack("crop_0.5").fn(images)
    assert any(not torch.equal(crops[0], crops[i]) for i in range(1, 8))


def test_deterministic_attacks_are_not_reseedable() -> None:
    assert reseed_attack(get_attack("jpeg_50"), 1) is False
    assert reseed_attack(get_attack("identity"), 1) is False


def test_attack_seed_is_stable_and_discriminating() -> None:
    assert attack_seed(0, "img", "crop_0.5") == attack_seed(0, "img", "crop_0.5")
    seeds = {attack_seed(s, i, a) for s in (0, 1) for i in ("a", "b") for a in ("x", "y")}
    assert len(seeds) == 8
    assert 0 <= attack_seed(0, "img", "crop_0.5") < 2**63
    # pinned value: guards against an accidental change that would invalidate past results
    assert attack_seed(0, "img", "crop_0.5") == PINNED_SEED


DIFFERENTIABLE_NAMES = [n for n in FAST_NAMES if get_attack(n).differentiable]
NON_DIFFERENTIABLE_NAMES = [n for n in FAST_NAMES if not get_attack(n).differentiable]


@pytest.mark.parametrize("name", DIFFERENTIABLE_NAMES)
def test_differentiable_attacks_propagate_finite_gradients(name: str) -> None:
    images = (0.25 + 0.5 * torch.rand(1, 3, 48, 48)).requires_grad_(True)
    output = get_attack(name).fn(images)
    assert output.requires_grad
    output.sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


@pytest.mark.parametrize("name", NON_DIFFERENTIABLE_NAMES)
def test_non_differentiable_attacks_break_the_graph_and_straight_through_repairs_it(
    name: str,
) -> None:
    spec = get_attack(name)
    shape_preserving = spec.fn(torch.rand(1, 3, 48, 48)).shape == (1, 3, 48, 48)
    if not shape_preserving:
        pytest.skip("combined attacks crop: straight-through needs equal shapes")
    images = torch.rand(1, 3, 48, 48, requires_grad=True)
    assert not spec.fn(images).requires_grad
    repaired = straight_through(spec.fn)(images)
    # x + (y - x) equals y only up to float rounding
    assert torch.allclose(repaired.detach(), spec.fn(images.detach()), atol=1e-6)
    repaired.sum().backward()
    assert images.grad is not None
    assert torch.allclose(images.grad, torch.ones_like(images))


def test_straight_through_rejects_size_changing_attacks_and_forwards_reseed() -> None:
    wrapped = straight_through(get_attack("crop_0.5").fn)
    with pytest.raises(ValueError):
        wrapped(torch.rand(1, 3, 32, 32))
    noisy = get_attack("gaussian_noise_0.05")
    spec = AttackSpec(name="st_noise", group="valuemetric", fn=straight_through(noisy.fn),
                      differentiable=True)
    images = torch.rand(1, 3, 16, 16)
    assert reseed_attack(spec, 9)
    first = spec.fn(images)
    assert reseed_attack(spec, 9)
    assert torch.equal(first, spec.fn(images))


@pytest.mark.parametrize(
    "bad",
    [torch.rand(3, 16, 16), torch.rand(1, 1, 16, 16), torch.zeros(1, 3, 16, 16, dtype=torch.int64)],
)
def test_attacks_reject_invalid_inputs(bad: torch.Tensor) -> None:
    for name in ("identity", "jpeg_50", "crop_0.5"):
        with pytest.raises(ValueError):
            get_attack(name).fn(bad)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
@pytest.mark.parametrize("name", ["brightness_1.5", "gaussian_blur_9", "resize_0.5", "crop_0.5",
                                  "rotate_10", "jpeg_50", "gaussian_noise_0.05", "median_3"])
def test_attacks_run_on_mps_and_stay_on_the_input_device(
    name: str, small_batch: torch.Tensor
) -> None:
    on_mps = small_batch.to("mps")
    spec = get_attack(name)
    reseed_attack(spec, 4)
    output = spec.fn(on_mps)
    assert output.device.type == "mps"
    reseed_attack(spec, 4)
    assert torch.allclose(spec.fn(small_batch), output.cpu(), atol=1e-4)


def test_regen_vae_reports_missing_diffusers(
    monkeypatch: pytest.MonkeyPatch, small_batch: torch.Tensor
) -> None:
    monkeypatch.setitem(sys.modules, "diffusers", None)
    monkeypatch.setattr(attacks, "_VAE_CACHE", {})
    with pytest.raises(AttackUnavailableError, match="diffusers"):
        get_attack("regen_vae").fn(small_batch)


def test_regen_vae_reports_unreachable_weights(
    monkeypatch: pytest.MonkeyPatch, small_batch: torch.Tensor
) -> None:
    diffusers = pytest.importorskip("diffusers")

    def offline(*_args: object, **_kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", offline)
    monkeypatch.setattr(attacks, "_VAE_CACHE", {})
    with pytest.raises(AttackUnavailableError, match="sd-vae-ft-mse"):
        get_attack("regen_vae").fn(small_batch)


@pytest.mark.slow
def test_regen_vae_regenerates_a_plausible_image(small_batch: torch.Tensor) -> None:
    output = get_attack("regen_vae").fn(small_batch)
    assert output.shape == small_batch.shape
    assert (output - small_batch).abs().mean() < 0.1
    assert (output - small_batch).abs().mean() > 0.0
