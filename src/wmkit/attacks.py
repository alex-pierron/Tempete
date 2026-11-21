"""Registry of image attacks used to stress watermarks, with named evaluation suites.

Every attack is a function on batches [B, 3, H, W] float32 in [0, 1], run on the device
of its input, returning a batch in [0, 1] (smaller for crop/resize). Names follow
`<op>_<param>` with the parameter written by `format(p, "g")` (see
docs/CONTRACT-wmkit.md, section 1.1).

Random attacks (gaussian noise, crop window, rotation sign, perspective corners) draw
from a private CPU `torch.Generator`, so results do not depend on the device. The seed is
captured by the attack object: `fn(images)` stays unary and a fresh `get_attack(name)`
always starts from `DEFAULT_ATTACK_SEED`. The evaluation harness calls `reseed_attack`
with a seed derived from (run seed, image id, attack name) before every application, so
a watermarked image and its clean twin undergo exactly the same random transformation.
"""

from __future__ import annotations

import difflib
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

import cv2
import kornia.enhance
import kornia.filters
import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_f

from wmkit.io import tensor_to_uint8
from wmkit.types import ATTACK_GROUPS, AttackFn, AttackGroup, AttackSpec

DEFAULT_ATTACK_SEED: int = 0
SEED_BYTES: int = 8
SEED_MASK: int = (1 << 63) - 1
RGB_CHANNELS: int = 3
TWO_PI: float = 2.0 * math.pi
UINT8_MAX: float = 255.0

NLMEANS_TEMPLATE_WINDOW: int = 7
NLMEANS_SEARCH_WINDOW: int = 21
MIN_SIDE_AFTER_RESIZE: int = 1
VAE_DOWNSAMPLE_FACTOR: int = 8
VAE_REPO_ID: str = "stabilityai/sd-vae-ft-mse"
VAE_INSTALL_HINT: str = (
    "uv pip install --python /Users/nowais/Documents/code/challenge/.venv/bin/python diffusers"
)
IMAGE_VALUE_SCALE: float = 2.0
"""The VAE works on images in [-1, 1]: x * 2 - 1."""

BRIGHTNESS_FACTORS: tuple[float, ...] = (0.5, 0.75, 1.25, 1.5, 2)
CONTRAST_FACTORS: tuple[float, ...] = (0.5, 0.75, 1.5, 2)
SATURATION_FACTORS: tuple[float, ...] = (0.5, 1.5)
HUE_SHIFTS: tuple[float, ...] = (-0.1, 0.1, 0.25)
BLUR_KERNEL_SIZES: tuple[int, ...] = (3, 9, 17)
NOISE_STDS: tuple[float, ...] = (0.02, 0.05)
JPEG_QUALITIES: tuple[int, ...] = (40, 50, 60, 70, 80, 90)
WEBP_QUALITIES: tuple[int, ...] = (50, 80)
CROP_FRACTIONS: tuple[float, ...] = (0.33, 0.5, 0.71, 0.9)
RESIZE_SCALES: tuple[float, ...] = (0.5, 0.75)
ROTATION_DEGREES: tuple[float, ...] = (5, 10, 30, 90)
PERSPECTIVE_SCALES: tuple[float, ...] = (0.1, 0.3, 0.5)
COMBO_JPEG_QUALITIES: tuple[int, ...] = (40, 60, 80)
COMBO_CROP_FRACTION: float = 0.71
COMBO_BRIGHTNESS: float = 0.5
MEDIAN_KERNEL_SIZES: tuple[int, ...] = (3, 5)
NLMEANS_STRENGTHS: tuple[int, ...] = (5, 10)
REMOVAL_JPEG_QUALITY: int = 10
DOWNUP_SCALES: tuple[float, ...] = (0.25, 0.5)

_RandomApply = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


class AttackError(RuntimeError):
    """Base class of attack failures that are not caller mistakes."""


class AttackUnavailableError(AttackError):
    """An optional dependency or its weights are missing (e.g. diffusers, the VAE)."""


@runtime_checkable
class Reseedable(Protocol):
    """An attack function whose random stream can be restarted from a seed."""

    def reseed(self, seed: int) -> None:
        """Restart the random stream from `seed`."""
        ...


class SeededAttack:
    """Callable attack owning a CPU `torch.Generator`; deterministic for a given seed."""

    def __init__(self, apply: _RandomApply, seed: int = DEFAULT_ATTACK_SEED) -> None:
        """Wrap `apply(images, generator)` and seed its generator."""
        self._apply = apply
        self._generator = torch.Generator(device="cpu")
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        """Restart the random stream from `seed`."""
        self._generator.manual_seed(seed & SEED_MASK)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Attack a batch [B, 3, H, W]; consumes random numbers from the generator."""
        _validate_images(images)
        return _finalise(self._apply(images, self._generator), images)


class _StraightThrough:
    """Forward = attacked image, backward = identity (see `straight_through`)."""

    def __init__(self, fn: AttackFn) -> None:
        self._fn = fn

    def reseed(self, seed: int) -> None:
        """Forward the reseeding to the wrapped attack when it supports it."""
        if isinstance(self._fn, Reseedable):
            self._fn.reseed(seed)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        attacked = self._fn(images)
        if attacked.shape != images.shape:
            raise ValueError(
                "straight_through needs a shape-preserving attack, got "
                f"{tuple(images.shape)} -> {tuple(attacked.shape)}"
            )
        return images + (attacked - images).detach()


def straight_through(fn: AttackFn) -> AttackFn:
    """Make a non-differentiable attack usable in training (straight-through estimator).

    The forward pass returns `fn(x)`; the backward pass behaves as the identity,
    computed as `x + (fn(x) - x).detach()`.

    Args:
        fn: A shape-preserving attack.

    Returns:
        A callable with the same signature; raises ValueError at call time if `fn`
        changes the image size (crop, resize), since the identity gradient is then undefined.
    """
    return _StraightThrough(fn)


def reseed_attack(spec: AttackSpec, seed: int) -> bool:
    """Restart the random stream of `spec`, if it has one.

    Args:
        spec: The attack to reseed.
        seed: New seed.

    Returns:
        True if the attack is random and was reseeded, False if it is deterministic.
    """
    if isinstance(spec.fn, Reseedable):
        spec.fn.reseed(seed)
        return True
    return False


def attack_seed(run_seed: int, image_id: str, attack_name: str) -> int:
    """Derive a stable 63-bit seed from a run seed, an image id and an attack name.

    SHA-256 (not `hash()`, which is salted per process) keeps results reproducible
    across runs and independent of evaluation order.
    """
    digest = hashlib.sha256(f"{run_seed}\x00{image_id}\x00{attack_name}".encode()).digest()
    return int.from_bytes(digest[:SEED_BYTES], "big") & SEED_MASK


def _validate_images(images: torch.Tensor) -> None:
    """Raise ValueError unless `images` is a float [B, 3, H, W] tensor."""
    if images.ndim != 4 or images.shape[1] != RGB_CHANNELS:
        raise ValueError(f"expected images of shape [B, 3, H, W], got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise ValueError(f"expected a floating point tensor, got {images.dtype}")
    if images.shape[2] < 1 or images.shape[3] < 1:
        raise ValueError(f"empty image: {tuple(images.shape)}")


def _finalise(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Clamp to [0, 1], cast to float32 and put on the device of `reference`."""
    return output.to(device=reference.device, dtype=torch.float32).clamp(0.0, 1.0)


def _deterministic(function: AttackFn) -> AttackFn:
    """Wrap a stateless attack with input validation and output normalisation."""

    def attack(images: torch.Tensor) -> torch.Tensor:
        _validate_images(images)
        return _finalise(function(images), images)

    return attack


def _draw_int(low: int, high: int, generator: torch.Generator) -> int:
    """Uniform integer in [low, high) from `generator`."""
    return int(torch.randint(low, high, (1,), generator=generator).item())


# ---------------------------------------------------------------------------
# Elementary transformations
# ---------------------------------------------------------------------------


def _identity(images: torch.Tensor) -> torch.Tensor:
    return images


def _grayscale(images: torch.Tensor) -> torch.Tensor:
    gray: torch.Tensor = tv_f.rgb_to_grayscale(images, num_output_channels=RGB_CHANNELS)
    return gray


def _gaussian_noise(std: float) -> _RandomApply:
    def apply(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        noise = torch.randn(images.shape, generator=generator, dtype=torch.float32)
        return images + std * noise.to(images.device)

    return apply


def _pil_round_trip(images: torch.Tensor, image_format: str, quality: int) -> torch.Tensor:
    """Encode then decode every image with Pillow (lossy); non-differentiable."""
    decoded: list[torch.Tensor] = []
    for image in images:
        buffer = BytesIO()
        Image.fromarray(tensor_to_uint8(image), mode="RGB").save(
            buffer, format=image_format, quality=quality
        )
        buffer.seek(0)
        with Image.open(buffer) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        decoded.append(torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / UINT8_MAX)
    return torch.stack(decoded).to(images.device)


def _jpeg(quality: int) -> AttackFn:
    return lambda images: _pil_round_trip(images, "JPEG", quality)


def _webp(quality: int) -> AttackFn:
    return lambda images: _pil_round_trip(images, "WEBP", quality)


def _crop(fraction: float) -> _RandomApply:
    """Keep a random window whose sides are `fraction` of the image sides."""

    def apply(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        height, width = images.shape[-2:]
        crop_h = max(MIN_SIDE_AFTER_RESIZE, round(fraction * height))
        crop_w = max(MIN_SIDE_AFTER_RESIZE, round(fraction * width))
        windows: list[torch.Tensor] = []
        for image in images:
            top = _draw_int(0, height - crop_h + 1, generator)
            left = _draw_int(0, width - crop_w + 1, generator)
            windows.append(image[:, top:top + crop_h, left:left + crop_w])
        return torch.stack(windows)

    return apply


def _resize_to(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize, anti-aliased when shrinking."""
    return torch_f.interpolate(
        images, size=size, mode="bilinear", align_corners=False, antialias=True
    )


def _resize(scale: float) -> AttackFn:
    def apply(images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        size = (
            max(MIN_SIDE_AFTER_RESIZE, round(scale * height)),
            max(MIN_SIDE_AFTER_RESIZE, round(scale * width)),
        )
        return _resize_to(images, size)

    return apply


def _downup(scale: float) -> AttackFn:
    """Shrink to `scale` then return to the original size (loses high frequencies)."""

    def apply(images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        small = (
            max(MIN_SIDE_AFTER_RESIZE, round(scale * height)),
            max(MIN_SIDE_AFTER_RESIZE, round(scale * width)),
        )
        return _resize_to(_resize_to(images, small), (height, width))

    return apply


def _rotate(degrees: float) -> _RandomApply:
    """Rotate by +degrees or -degrees (sign drawn per image), size kept, black corners."""

    def apply(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        rotated: list[torch.Tensor] = []
        for image in images:
            sign = 1.0 if _draw_int(0, 2, generator) == 1 else -1.0
            rotated.append(
                tv_f.rotate(image, sign * degrees, interpolation=InterpolationMode.BILINEAR,
                            expand=False)
            )
        return torch.stack(rotated)

    return apply


def _perspective(distortion: float) -> _RandomApply:
    """Random corner displacement, same sampling as torchvision's RandomPerspective."""

    def apply(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        height, width = images.shape[-2:]
        max_dx = int(distortion * (width // 2))
        max_dy = int(distortion * (height // 2))
        start = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        warped: list[torch.Tensor] = []
        for image in images:
            end = [
                [_draw_int(0, max_dx + 1, generator), _draw_int(0, max_dy + 1, generator)],
                [_draw_int(width - max_dx - 1, width, generator),
                 _draw_int(0, max_dy + 1, generator)],
                [_draw_int(width - max_dx - 1, width, generator),
                 _draw_int(height - max_dy - 1, height, generator)],
                [_draw_int(0, max_dx + 1, generator),
                 _draw_int(height - max_dy - 1, height, generator)],
            ]
            warped.append(
                tv_f.perspective(image, start, end, interpolation=InterpolationMode.BILINEAR)
            )
        return torch.stack(warped)

    return apply


def _median(kernel_size: int) -> AttackFn:
    return lambda images: kornia.filters.median_blur(images, (kernel_size, kernel_size))


def _nlmeans(strength: int) -> AttackFn:
    """OpenCV non-local-means denoising (colour), `strength` = filter strength h."""

    def apply(images: torch.Tensor) -> torch.Tensor:
        denoised: list[torch.Tensor] = []
        for image in images:
            bgr = np.ascontiguousarray(tensor_to_uint8(image)[..., ::-1])
            cleaned = cv2.fastNlMeansDenoisingColored(
                bgr, None, strength, strength, NLMEANS_TEMPLATE_WINDOW, NLMEANS_SEARCH_WINDOW
            )
            rgb = np.ascontiguousarray(cleaned[..., ::-1])
            denoised.append(torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / UINT8_MAX)
        return torch.stack(denoised).to(images.device)

    return apply


_VAE_CACHE: dict[str, Any] = {}


def _load_vae(device: torch.device) -> Any:
    """Load `stabilityai/sd-vae-ft-mse` once per device.

    Raises:
        AttackUnavailableError: diffusers is not installed, or the weights cannot be
            fetched (no network and nothing in the Hugging Face cache).
    """
    key = str(device)
    if key in _VAE_CACHE:
        return _VAE_CACHE[key]
    try:
        from diffusers import AutoencoderKL
    except ImportError as error:
        raise AttackUnavailableError(
            f"attack 'regen_vae' needs the `diffusers` package; install it with: {VAE_INSTALL_HINT}"
        ) from error
    try:
        # diffusers ships an untyped `from_pretrained`, hence the targeted ignore.
        vae = AutoencoderKL.from_pretrained(VAE_REPO_ID)  # type: ignore[no-untyped-call]
    except (OSError, ValueError) as error:
        raise AttackUnavailableError(
            f"attack 'regen_vae' cannot load {VAE_REPO_ID} (no network or empty Hugging Face "
            f"cache?): {error}"
        ) from error
    vae = vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    _VAE_CACHE[key] = vae
    return vae


def _regen_vae(images: torch.Tensor) -> torch.Tensor:
    """Regeneration attack: encode then decode through the Stable Diffusion VAE."""
    vae = _load_vae(images.device)
    height, width = images.shape[-2:]
    pad_h = -height % VAE_DOWNSAMPLE_FACTOR
    pad_w = -width % VAE_DOWNSAMPLE_FACTOR
    scaled = images.detach() * IMAGE_VALUE_SCALE - 1.0
    padded = torch_f.pad(scaled, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.no_grad():
        latents = vae.encode(padded).latent_dist.mode()
        decoded = vae.decode(latents).sample
    restored = decoded[..., :height, :width]
    regenerated: torch.Tensor = (restored + 1.0) / IMAGE_VALUE_SCALE
    return regenerated


def _brightness(factor: float) -> AttackFn:
    def apply(images: torch.Tensor) -> torch.Tensor:
        adjusted: torch.Tensor = tv_f.adjust_brightness(images, factor)
        return adjusted

    return apply


def _contrast(factor: float) -> AttackFn:
    def apply(images: torch.Tensor) -> torch.Tensor:
        adjusted: torch.Tensor = tv_f.adjust_contrast(images, factor)
        return adjusted

    return apply


def _saturation(factor: float) -> AttackFn:
    def apply(images: torch.Tensor) -> torch.Tensor:
        adjusted: torch.Tensor = tv_f.adjust_saturation(images, factor)
        return adjusted

    return apply


def _hue(turn_fraction: float) -> AttackFn:
    """Shift hue by a fraction of the colour wheel, in [-0.5, 0.5].

    Uses kornia (matches torchvision's `adjust_hue` to 1e-6) because torchvision's
    implementation modifies tensors in place and cannot be back-propagated.
    """
    radians = turn_fraction * TWO_PI

    def apply(images: torch.Tensor) -> torch.Tensor:
        adjusted: torch.Tensor = kornia.enhance.adjust_hue(images, radians)
        return adjusted

    return apply


def _gaussian_blur(kernel_size: int) -> AttackFn:
    def apply(images: torch.Tensor) -> torch.Tensor:
        blurred: torch.Tensor = tv_f.gaussian_blur(images, [kernel_size, kernel_size])
        return blurred

    return apply


def _combo(jpeg_quality: int) -> _RandomApply:
    """Paper combined attack, in Meta's order: JPEG, then crop (random window), then brightness.

    Mirrors `Sequential(JPEG(), Crop(), Brightness())` in the pinned videoseal
    (`videoseal/augmentation/__init__.py`; `Sequential` applies its transforms in list
    order). The order is not cosmetic: on the 24 Kodak images, Pixel Seal keeps 0.881 of
    its bits at JPEG 40 in this order but only 0.574 in the reversed one (brightness
    first), so any other order makes `combo_*` incomparable with the paper's figures.
    """
    crop = _crop(COMBO_CROP_FRACTION)
    jpeg = _jpeg(jpeg_quality)

    def apply(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        cropped = crop(jpeg(images.clamp(0.0, 1.0)), generator)
        darkened: torch.Tensor = tv_f.adjust_brightness(cropped, COMBO_BRIGHTNESS)
        return darkened.clamp(0.0, 1.0)

    return apply


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """How to build a fresh AttackSpec: its group and a zero-argument factory."""

    group: AttackGroup
    build: Callable[[], AttackSpec]


_REGISTRY: dict[str, _Entry] = {}


def _label(value: float) -> str:
    """Parameter as written in attack names: `format(p, "g")`."""
    return format(value, "g")


def _register(
    name: str,
    group: AttackGroup,
    make_fn: Callable[[], AttackFn],
    *,
    differentiable: bool,
    **params: float | str,
) -> None:
    """Add an attack to the registry; `make_fn` is called on every `get_attack`."""
    if name in _REGISTRY:
        raise ValueError(f"duplicate attack name {name!r}")

    def build() -> AttackSpec:
        return AttackSpec(name=name, group=group, fn=make_fn(), differentiable=differentiable,
                          params=dict(params))

    _REGISTRY[name] = _Entry(group, build)


def _plain(function: AttackFn) -> Callable[[], AttackFn]:
    """Factory returning a validated deterministic attack."""
    return lambda: _deterministic(function)


def _random(apply: _RandomApply) -> Callable[[], AttackFn]:
    """Factory returning a fresh seeded random attack (generator at DEFAULT_ATTACK_SEED)."""
    return lambda: SeededAttack(apply)


def _build_registry() -> None:
    """Populate `_REGISTRY` in the order of the contract table."""
    _register("identity", "identity", _plain(_identity), differentiable=True)

    for factor in BRIGHTNESS_FACTORS:
        _register(f"brightness_{_label(factor)}", "valuemetric",
                  _plain(_brightness(factor)), differentiable=True, factor=factor)
    for factor in CONTRAST_FACTORS:
        _register(f"contrast_{_label(factor)}", "valuemetric",
                  _plain(_contrast(factor)),
                  differentiable=True, factor=factor)
    for factor in SATURATION_FACTORS:
        _register(f"saturation_{_label(factor)}", "valuemetric",
                  _plain(_saturation(factor)),
                  differentiable=True, factor=factor)
    for shift in HUE_SHIFTS:
        _register(f"hue_{_label(shift)}", "valuemetric",
                  _plain(_hue(shift)),
                  differentiable=True, turn_fraction=shift)
    _register("grayscale", "valuemetric", _plain(_grayscale), differentiable=True)
    for kernel in BLUR_KERNEL_SIZES:
        _register(f"gaussian_blur_{kernel}", "valuemetric",
                  _plain(_gaussian_blur(kernel)),
                  differentiable=True, kernel_size=kernel)
    for std in NOISE_STDS:
        _register(f"gaussian_noise_{_label(std)}", "valuemetric", _random(_gaussian_noise(std)),
                  differentiable=True, std=std)

    for quality in JPEG_QUALITIES:
        _register(f"jpeg_{quality}", "compression", _plain(_jpeg(quality)),
                  differentiable=False, quality=quality)
    for quality in WEBP_QUALITIES:
        _register(f"webp_{quality}", "compression", _plain(_webp(quality)),
                  differentiable=False, quality=quality)

    for fraction in CROP_FRACTIONS:
        _register(f"crop_{_label(fraction)}", "geometric", _random(_crop(fraction)),
                  differentiable=True, side_fraction=fraction)
    for scale in RESIZE_SCALES:
        _register(f"resize_{_label(scale)}", "geometric", _plain(_resize(scale)),
                  differentiable=True, scale=scale)
    for degrees in ROTATION_DEGREES:
        _register(f"rotate_{_label(degrees)}", "geometric", _random(_rotate(degrees)),
                  differentiable=True, degrees=degrees)
    _register("hflip", "geometric", _plain(lambda x: x.flip(-1)), differentiable=True)
    for distortion in PERSPECTIVE_SCALES:
        _register(f"perspective_{_label(distortion)}", "geometric",
                  _random(_perspective(distortion)), differentiable=True,
                  distortion_scale=distortion)

    for quality in COMBO_JPEG_QUALITIES:
        name = (f"combo_jpeg{quality}_crop{_label(COMBO_CROP_FRACTION)}"
                f"_bright{_label(COMBO_BRIGHTNESS)}")
        _register(name, "combined", _random(_combo(quality)), differentiable=False,
                  jpeg_quality=quality, crop_side_fraction=COMBO_CROP_FRACTION,
                  brightness=COMBO_BRIGHTNESS)

    for kernel in MEDIAN_KERNEL_SIZES:
        _register(f"median_{kernel}", "removal", _plain(_median(kernel)),
                  differentiable=True, kernel_size=kernel)
    for strength in NLMEANS_STRENGTHS:
        _register(f"nlmeans_{strength}", "removal", _plain(_nlmeans(strength)),
                  differentiable=False, h=strength)
    _register(f"jpeg_{REMOVAL_JPEG_QUALITY}", "removal", _plain(_jpeg(REMOVAL_JPEG_QUALITY)),
              differentiable=False, quality=REMOVAL_JPEG_QUALITY)
    for scale in DOWNUP_SCALES:
        _register(f"downup_{_label(scale)}", "removal", _plain(_downup(scale)),
                  differentiable=True, scale=scale)
    _register("regen_vae", "removal", _plain(_regen_vae), differentiable=False,
              model=VAE_REPO_ID)


_build_registry()

QUICK_SUITE: list[str] = [
    "identity", "brightness_1.5", "contrast_0.5", "hue_0.1", "gaussian_blur_9", "jpeg_50",
    "crop_0.71", "resize_0.5", "rotate_10", "hflip", "combo_jpeg40_crop0.71_bright0.5",
]


def list_attacks(group: AttackGroup | None = None) -> list[str]:
    """Names of the registered attacks, in registry order.

    Args:
        group: Restrict to one family; None lists all.

    Raises:
        ValueError: `group` is not one of ATTACK_GROUPS.
    """
    if group is None:
        return list(_REGISTRY)
    if group not in ATTACK_GROUPS:
        raise ValueError(f"unknown attack group {group!r}; known: {', '.join(ATTACK_GROUPS)}")
    return [name for name, entry in _REGISTRY.items() if entry.group == group]


def get_attack(name: str) -> AttackSpec:
    """Build a fresh `AttackSpec` (random attacks start at seed 0).

    Raises:
        KeyError: Unknown attack name; the message suggests close names.
    """
    if name not in _REGISTRY:
        close = difflib.get_close_matches(name, _REGISTRY, n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown attack {name!r}{hint} (list_attacks() gives all names)")
    return _REGISTRY[name].build()


SUITES: dict[str, list[str]] = {
    "quick": QUICK_SUITE,
    "paper": [name for group in ATTACK_GROUPS if group != "removal"
              for name in list_attacks(group)],
    "removal": list_attacks("removal"),
    "all": list_attacks(),
}


def get_suite(name: str) -> list[AttackSpec]:
    """Build fresh `AttackSpec`s for a named suite.

    Raises:
        KeyError: Unknown suite name.
    """
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; known: {', '.join(SUITES)}")
    return [get_attack(attack_name) for attack_name in SUITES[name]]
