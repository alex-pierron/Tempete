"""Unit tests (QA pass) of `wmkit.attacks`: shape rules, determinism, gradients, tiny inputs.

The existing `tests/test_attacks.py` checks sizes for a handful of attacks and determinism for
a hand-picked list of random ones. Here the rules are derived from the registry itself, so a new
attack added to the registry is covered automatically:
    * every attack keeps the image size, except crop_* / resize_* / combo_* (see `expected_shape`);
    * every attack that exposes `reseed` is deterministic at equal seed, every other is a pure function;
    * straight-through is the identity in the backward pass, whatever the loss weights.

Finding pinned with `xfail(strict=True)`:
    F-ATK-1  `gaussian_blur_*` and `perspective_*` raise a raw RuntimeError / LinAlgError on
             images smaller than their kernel, although `_validate_images` accepts any H, W >= 1.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from wmkit.attacks import (
    attack_seed,
    get_attack,
    list_attacks,
    reseed_attack,
    straight_through,
)
from wmkit.types import AttackFn, AttackSpec

ALL_NAMES = [name for name in list_attacks() if name != "regen_vae"]
"""regen_vae needs diffusers and network weights: out of scope for unit tests."""
RECT_SHAPE = (2, 3, 40, 56)
"""Non-square, so that a swapped height / width shows up."""
ODD_SHAPE = (1, 3, 33, 47)
RESEED_SEED = 1234
SEED_PROBES = 6
GRADIENT_LOSS_SEED = 0
TINY_SHAPES = ((1, 3, 1, 1), (1, 3, 2, 3), (2, 3, 5, 5))
RANDOM_NAMES = [name for name in ALL_NAMES if hasattr(get_attack(name).fn, "reseed")]
DETERMINISTIC_NAMES = [name for name in ALL_NAMES if name not in RANDOM_NAMES]
CROP_FRACTION_BY_PREFIX = {"crop_": None, "combo_": 0.71}
"""`combo_*` ends with a crop of 0.71 of each side; `crop_<f>` crops by its own parameter."""


def fixed_batch(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """Reproducible uniform batch in [0, 1]."""
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def expected_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Output shape implied by the registry name (contract section 1.1), computed independently."""
    batch, channels, height, width = shape
    if name.startswith(("crop_", "resize_")):
        fraction = float(name.split("_", 1)[1])
    elif name.startswith("combo_"):
        fraction = CROP_FRACTION_BY_PREFIX["combo_"] or 1.0
    else:
        return shape
    return (batch, channels, max(1, round(fraction * height)), max(1, round(fraction * width)))


def spec_with_seed(name: str, seed: int) -> AttackSpec:
    """A fresh spec of `name`, reseeded."""
    spec = get_attack(name)
    assert reseed_attack(spec, seed)
    return spec


# ---------------------------------------------------------------------------
# Output size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", (RECT_SHAPE, ODD_SHAPE))
@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_size_is_kept_except_for_crop_resize_and_combo(
    name: str, shape: tuple[int, ...]
) -> None:
    output = get_attack(name).fn(fixed_batch(shape))
    assert tuple(output.shape) == expected_shape(name, shape)
    height, width = shape[-2:]
    assert output.shape[-2] <= height and output.shape[-1] <= width, "an attack never enlarges"


def test_the_size_changing_attacks_are_exactly_the_registered_geometric_reducers() -> None:
    reducers = {n for n in ALL_NAMES if expected_shape(n, RECT_SHAPE) != RECT_SHAPE}
    actual = {n for n in ALL_NAMES if get_attack(n).fn(fixed_batch(RECT_SHAPE)).shape != RECT_SHAPE}
    assert reducers == actual
    assert any(n.startswith("crop_") for n in reducers) and any(n.startswith("resize_") for n in reducers)
    assert {n for n in reducers if n.startswith("combo_")} == {n for n in ALL_NAMES if n.startswith("combo_")}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_random_attack_list_is_what_the_contract_says() -> None:
    expected_prefixes = ("gaussian_noise_", "crop_", "rotate_", "perspective_", "combo_")
    assert RANDOM_NAMES, "registry exposes no random attack"
    assert all(name.startswith(expected_prefixes) for name in RANDOM_NAMES)
    assert all(
        name.startswith(expected_prefixes) for name in ALL_NAMES if name in RANDOM_NAMES
    )


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_equal_seed_gives_bitwise_equal_output_and_a_new_seed_a_new_one(name: str) -> None:
    batch = fixed_batch((8, 3, 40, 56), seed=1)
    first = spec_with_seed(name, RESEED_SEED).fn(batch)
    again = spec_with_seed(name, RESEED_SEED).fn(batch)
    assert torch.equal(first, again)
    others = [spec_with_seed(name, RESEED_SEED + 1 + i).fn(batch) for i in range(SEED_PROBES)]
    if first.shape == batch.shape or name.startswith(("crop_", "combo_")):
        assert any(not torch.equal(first, other) for other in others)


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_consecutive_calls_consume_the_stream_but_reseeding_replays_it(name: str) -> None:
    batch = fixed_batch((4, 3, 40, 56), seed=2)
    spec = spec_with_seed(name, RESEED_SEED)
    first, second = spec.fn(batch), spec.fn(batch)
    assert reseed_attack(spec, RESEED_SEED)
    assert torch.equal(spec.fn(batch), first)
    assert torch.equal(spec.fn(batch), second), "the stream continues exactly as before"


@pytest.mark.parametrize("name", DETERMINISTIC_NAMES)
def test_deterministic_attacks_are_pure_functions(name: str) -> None:
    batch = fixed_batch((2, 3, 40, 56), seed=3)
    spec = get_attack(name)
    snapshot = batch.clone()
    first, second = spec.fn(batch), get_attack(name).fn(batch)
    assert torch.equal(first, second)
    assert torch.equal(batch, snapshot), "the input must not be modified in place"
    assert reseed_attack(spec, 1) is False


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_random_attacks_do_not_modify_their_input_either(name: str) -> None:
    batch = fixed_batch((2, 3, 40, 56), seed=4)
    snapshot = batch.clone()
    get_attack(name).fn(batch)
    assert torch.equal(batch, snapshot)


def test_reseeding_one_spec_does_not_affect_another_spec_of_the_same_attack() -> None:
    batch = fixed_batch((2, 3, 32, 32), seed=5)
    left, right = get_attack("gaussian_noise_0.05"), get_attack("gaussian_noise_0.05")
    reseed_attack(left, 99)
    reference = right.fn(batch)
    assert torch.equal(get_attack("gaussian_noise_0.05").fn(batch), reference)
    assert not torch.equal(left.fn(batch), reference)


def test_attack_seed_has_no_collisions_over_a_realistic_grid_and_fits_63_bits() -> None:
    seeds = {
        attack_seed(run, f"img{index:04d}", name)
        for run in range(3)
        for index in range(60)
        for name in ALL_NAMES
    }
    assert len(seeds) == 3 * 60 * len(ALL_NAMES)
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert attack_seed(0, "a", "b") != attack_seed(0, "b", "a")
    assert attack_seed(1, "2", "x") != attack_seed(12, "", "x")


# ---------------------------------------------------------------------------
# Value range and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_attack_clamps_to_unit_range_even_for_out_of_range_input(name: str) -> None:
    wild = fixed_batch((1, 3, 40, 40), seed=6) * 3.0 - 1.0  # values in [-1, 2]
    output = get_attack(name).fn(wild)
    assert torch.isfinite(output).all()
    assert output.min() >= 0.0 and output.max() <= 1.0


@pytest.mark.parametrize("name", ["identity", "brightness_1.5", "jpeg_50", "resize_0.5", "hflip"])
def test_float64_input_is_accepted_and_returned_as_float32(name: str) -> None:
    batch = fixed_batch((1, 3, 40, 40), seed=7).to(torch.float64)
    output = get_attack(name).fn(batch)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("name", ["identity", "hflip", "grayscale", "brightness_1.25", "contrast_0.5"])
def test_extreme_constant_images_stay_finite_and_in_range(name: str) -> None:
    for level in (0.0, 1.0):
        output = get_attack(name).fn(torch.full((1, 3, 16, 16), level))
        assert torch.isfinite(output).all() and output.min() >= 0.0 and output.max() <= 1.0


# ---------------------------------------------------------------------------
# Algebraic properties of simple attacks
# ---------------------------------------------------------------------------


def test_hflip_is_an_involution_and_grayscale_is_idempotent() -> None:
    batch = fixed_batch((2, 3, 24, 30), seed=8)
    flip = get_attack("hflip").fn
    assert torch.equal(flip(flip(batch)), batch)
    gray = get_attack("grayscale").fn
    # torchvision's BT.601 weights sum to 0.9999, so a second pass shrinks by 1e-4 at most.
    assert torch.allclose(gray(gray(batch)), gray(batch), atol=2e-4)
    assert torch.equal(gray(batch)[:, 0], gray(batch)[:, 1])


def test_brightness_is_monotone_in_its_factor_before_clamping() -> None:
    batch = fixed_batch((1, 3, 16, 16), seed=9) * 0.5
    outputs = [get_attack(f"brightness_{f}").fn(batch) for f in ("0.5", "0.75", "1.25", "1.5", "2")]
    for darker, brighter in itertools.pairwise(outputs):
        assert (brighter >= darker - 1e-7).all()


def test_noise_attack_leaves_a_mean_preserving_zero_mean_perturbation() -> None:
    batch = torch.full((4, 3, 64, 64), 0.5)
    for name, std in (("gaussian_noise_0.02", 0.02), ("gaussian_noise_0.05", 0.05)):
        difference = spec_with_seed(name, 5).fn(batch) - batch
        assert abs(difference.mean().item()) < 5 * std / math.sqrt(difference.numel())
        assert difference.std().item() == pytest.approx(std, rel=0.05)


def test_compression_distortion_increases_as_quality_drops() -> None:
    batch = fixed_batch((2, 3, 48, 48), seed=10)
    errors = [
        (get_attack(f"jpeg_{q}").fn(batch) - batch).abs().mean().item() for q in (90, 70, 50, 40)
    ]
    assert errors == sorted(errors)
    assert errors[0] > 0.0


COMBO_ORDER_SEED = 21
"""Any seed: the combined attack and the hand-made pipeline draw the same crop window from it."""


@pytest.mark.parametrize("quality", ["40", "60", "80"])
def test_combined_attack_applies_jpeg_then_crop_then_brightness_like_meta(quality: str) -> None:
    """Regression: wmkit used to run brightness, crop, JPEG; Meta's Sequential runs the reverse.

    The oracle is built from the public single attacks, so it does not share code with `_combo`.
    The reversed pipeline must differ, otherwise this test could not tell the two orders apart.
    """
    batch = fixed_batch((2, 3, 48, 64), seed=11)
    combined = spec_with_seed(f"combo_jpeg{quality}_crop0.71_bright0.5", COMBO_ORDER_SEED).fn(batch)

    jpeg = get_attack(f"jpeg_{quality}").fn
    brightness = get_attack("brightness_0.5").fn
    meta_order = brightness(spec_with_seed("crop_0.71", COMBO_ORDER_SEED).fn(jpeg(batch)))
    reversed_order = jpeg(spec_with_seed("crop_0.71", COMBO_ORDER_SEED).fn(brightness(batch)))

    assert combined.shape == meta_order.shape
    assert torch.allclose(combined, meta_order, atol=1e-6)
    assert not torch.allclose(combined, reversed_order, atol=1e-2)


# ---------------------------------------------------------------------------
# straight_through
# ---------------------------------------------------------------------------

SHAPE_PRESERVING_NON_DIFFERENTIABLE = [
    name
    for name in ALL_NAMES
    if not get_attack(name).differentiable and expected_shape(name, RECT_SHAPE) == RECT_SHAPE
]


def test_there_are_non_differentiable_attacks_to_exercise() -> None:
    assert {"jpeg_50", "webp_50", "nlmeans_5"} <= set(SHAPE_PRESERVING_NON_DIFFERENTIABLE)


@pytest.mark.parametrize("name", SHAPE_PRESERVING_NON_DIFFERENTIABLE)
def test_straight_through_gradient_equals_the_loss_weights(name: str) -> None:
    """With loss = sum(w * y), identity backward means d loss / d x = w exactly."""
    batch = fixed_batch((1, 3, 24, 24), seed=11).requires_grad_(True)
    weights = fixed_batch((1, 3, 24, 24), seed=GRADIENT_LOSS_SEED + 12) * 4.0 - 2.0
    attacked = straight_through(get_attack(name).fn)(batch)
    (attacked * weights).sum().backward()  # type: ignore[no-untyped-call]  # torch stub gap
    assert batch.grad is not None
    assert torch.allclose(batch.grad, weights, atol=1e-6)


@pytest.mark.parametrize("name", ["jpeg_50", "median_3", "gaussian_noise_0.05", "hflip"])
def test_straight_through_forward_matches_the_wrapped_attack(name: str) -> None:
    batch = fixed_batch((2, 3, 24, 24), seed=13)
    plain, wrapped = get_attack(name), get_attack(name)
    if reseed_attack(plain, 3):
        reseed_attack(wrapped, 3)
    expected = plain.fn(batch)
    got = straight_through(wrapped.fn)(batch)
    assert torch.allclose(got, expected, atol=1e-6)
    assert got.min() >= -1e-6 and got.max() <= 1.0 + 1e-6


def test_straight_through_replaces_the_true_gradient_of_a_differentiable_attack() -> None:
    """Documented behaviour: backward is the identity even when the attack has a gradient."""
    batch = fixed_batch((1, 3, 16, 16), seed=14).requires_grad_(True)
    straight_through(get_attack("brightness_0.5").fn)(batch).sum().backward()  # type: ignore[no-untyped-call]  # torch stub gap
    assert batch.grad is not None
    assert torch.allclose(batch.grad, torch.ones_like(batch))  # the true gradient would be 0.5


def test_straight_through_does_not_alias_or_mutate_the_input() -> None:
    batch = fixed_batch((1, 3, 16, 16), seed=15)
    snapshot = batch.clone()
    out = straight_through(get_attack("jpeg_50").fn)(batch)
    assert torch.equal(batch, snapshot)
    assert out.data_ptr() != batch.data_ptr()


def test_straight_through_forwards_reseeding_to_a_random_attack() -> None:
    wrapped = AttackSpec(
        "a", "valuemetric", straight_through(get_attack("gaussian_noise_0.05").fn), True
    )
    batch = fixed_batch((1, 3, 16, 16), seed=16)
    reseed_attack(wrapped, 1)
    first = wrapped.fn(batch)
    reseed_attack(wrapped, 2)
    assert not torch.equal(first, wrapped.fn(batch))
    reseed_attack(wrapped, 1)
    assert torch.equal(first, wrapped.fn(batch))


@pytest.mark.xfail(
    strict=True,
    reason="F-ATK-2: the straight_through wrapper always defines `reseed`, so reseed_attack "
    "reports True (random) for a wrapped deterministic attack, contradicting its docstring",
)
def test_reseed_attack_reports_false_for_a_wrapped_deterministic_attack() -> None:
    wrapped = AttackSpec("b", "compression", straight_through(get_attack("jpeg_50").fn), False)
    assert reseed_attack(wrapped, 1) is False


def test_straight_through_rejects_every_size_changing_registry_attack() -> None:
    changing = [n for n in ALL_NAMES if expected_shape(n, RECT_SHAPE) != RECT_SHAPE]
    assert changing
    for name in changing:
        with pytest.raises(ValueError, match="shape-preserving"):
            straight_through(get_attack(name).fn)(fixed_batch(RECT_SHAPE))


# ---------------------------------------------------------------------------
# Tiny images (F-ATK-1)
# ---------------------------------------------------------------------------


def run_or_raise_typed(function: AttackFn, batch: torch.Tensor) -> torch.Tensor | None:
    """Return the output, or None if the attack declines the input with a ValueError."""
    try:
        return function(batch)
    except ValueError:
        return None


@pytest.mark.parametrize("shape", TINY_SHAPES)
@pytest.mark.parametrize(
    "name",
    [
        name
        for name in ALL_NAMES
        if not name.startswith(("gaussian_blur_", "perspective_", "nlmeans_", "median_"))
    ],
)
def test_attacks_without_a_kernel_support_any_image_down_to_one_pixel(
    name: str, shape: tuple[int, ...]
) -> None:
    output = run_or_raise_typed(get_attack(name).fn, fixed_batch(shape))
    assert output is None or (output.min() >= 0.0 and output.max() <= 1.0)


@pytest.mark.xfail(
    strict=True,
    reason="F-ATK-1: blur / perspective raise an untyped RuntimeError or LinAlgError on tiny images",
)
@pytest.mark.parametrize(
    "name,shape",
    [
        ("gaussian_blur_17", (2, 3, 5, 5)),
        ("gaussian_blur_9", (1, 3, 2, 3)),
        ("gaussian_blur_3", (1, 3, 1, 1)),
        ("perspective_0.1", (1, 3, 1, 1)),
    ],
)
def test_kernel_attacks_decline_tiny_images_with_a_typed_error_or_succeed(
    name: str, shape: tuple[int, ...]
) -> None:
    output = run_or_raise_typed(get_attack(name).fn, fixed_batch(shape))
    assert output is None or output.shape == torch.Size(shape)
