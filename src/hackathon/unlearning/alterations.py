"""Simulated "altered copies" of images (flip, crop, color jitter, JPEG).

The server scores the forget set on HIDDEN altered copies of the forget images; their exact
transformations are unknown. These four are plausible stand-ins used for local proxy metrics and
for augmentation inside the unlearning methods (a model that only fixes the exact pixels would be
caught by the altered group, which counts for half of the forget score).

All functions take and return uint8 [N, H, W, 3] tensors on CPU and are deterministic given the
torch.Generator, so a proxy run is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from hackathon.unlearning.constants import IMAGE_SIZE, PIXEL_MAX

CROP_MIN_AREA_FRACTION = 0.6
CROP_MAX_AREA_FRACTION = 0.9
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.6, 1.4)
JPEG_QUALITY = 50
"""Aggressive enough to change high-frequency content, mild enough to keep the class visible."""
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
ROI_SAMPLING_RATIO = 2

Alteration = Callable[[Tensor, torch.Generator], Tensor]


def _to_float(images: Tensor) -> Tensor:
    """uint8 NHWC -> float32 NCHW in [0, 1]."""
    return images.permute(0, 3, 1, 2).float() / PIXEL_MAX


def _to_uint8(pixels: Tensor) -> Tensor:
    """float32 NCHW in [0, 1] -> uint8 NHWC (clamped, rounded)."""
    return (pixels.clamp(0.0, 1.0) * PIXEL_MAX).round().to(torch.uint8).permute(0, 2, 3, 1)


def _uniform(count: int, bounds: tuple[float, float], generator: torch.Generator) -> Tensor:
    """`count` samples of U(bounds) as a float32 [count, 1, 1, 1] tensor (broadcastable)."""
    low, high = bounds
    return (low + (high - low) * torch.rand(count, generator=generator)).view(count, 1, 1, 1)


def horizontal_flip(images: Tensor, generator: torch.Generator) -> Tensor:
    """Mirror every image left-right (deterministic; the generator is unused but kept for the API)."""
    del generator
    return images.flip(dims=(2,)).contiguous()


def random_crop_resize(
    images: Tensor,
    generator: torch.Generator,
    min_area: float = CROP_MIN_AREA_FRACTION,
    max_area: float = CROP_MAX_AREA_FRACTION,
) -> Tensor:
    """Crop a random square covering 60-90 % of the area, resized back to the input size.

    Vectorised with roi_align (one call for the whole batch).
    """
    from torchvision.ops import roi_align

    count, height, width, _ = images.shape
    area_fraction = min_area + (max_area - min_area) * torch.rand(count, generator=generator)
    side_fraction = area_fraction.sqrt()
    crop_h = side_fraction * height
    crop_w = side_fraction * width
    top = torch.rand(count, generator=generator) * (height - crop_h)
    left = torch.rand(count, generator=generator) * (width - crop_w)
    boxes = torch.stack(
        [torch.arange(count).float(), left, top, left + crop_w, top + crop_h], dim=1
    )
    cropped = roi_align(
        _to_float(images),
        boxes,
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
        spatial_scale=1.0,
        sampling_ratio=ROI_SAMPLING_RATIO,
        aligned=True,
    )
    return _to_uint8(cropped)


def color_jitter(images: Tensor, generator: torch.Generator) -> Tensor:
    """Random brightness, contrast and saturation per image (each factor drawn independently)."""
    count = images.shape[0]
    pixels = _to_float(images)
    luma_weights = torch.tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    pixels = pixels * _uniform(count, BRIGHTNESS_RANGE, generator)
    gray_mean = (pixels * luma_weights).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    pixels = (pixels - gray_mean) * _uniform(count, CONTRAST_RANGE, generator) + gray_mean
    gray = (pixels * luma_weights).sum(dim=1, keepdim=True)
    pixels = (pixels - gray) * _uniform(count, SATURATION_RANGE, generator) + gray
    return _to_uint8(pixels)


def jpeg_compress(images: Tensor, generator: torch.Generator) -> Tensor:
    """JPEG round trip at quality 50 (torchvision codec, CPU; one call per image)."""
    del generator
    from torchvision.io import decode_jpeg, encode_jpeg

    decoded: list[Tensor] = []
    for image in images:
        encoded = encode_jpeg(image.permute(2, 0, 1).contiguous(), quality=JPEG_QUALITY)
        decoded.append(decode_jpeg(encoded).permute(1, 2, 0))
    if not decoded:
        return images.clone()
    return torch.stack(decoded)


STRONG_VIEW_AREA_SPREAD = 0.1
"""Width of the crop-area interval of a strong view (min_area .. min_area + spread)."""


def strong_view(images: Tensor, generator: torch.Generator, min_area: float) -> Tensor:
    """Aggressive view (small crop + color jitter) used to probe how fragile a prediction is.

    The original model is memorised on the forget images: a mild alteration leaves it at 100 %
    confidence, but a crop of about a third of the image makes it hesitate on the images a model
    that never saw them would also find hard. Used only to build soft targets, never as a
    stand-in for the (unknown) hidden altered copies.
    """
    cropped = random_crop_resize(images, generator, min_area, min_area + STRONG_VIEW_AREA_SPREAD)
    return color_jitter(cropped, generator)


ALTERATIONS: dict[str, Alteration] = {
    "flip": horizontal_flip,
    "crop": random_crop_resize,
    "jitter": color_jitter,
    "jpeg": jpeg_compress,
}
"""Name -> alteration, in the order the proxy tables list them."""


def apply_alteration(name: str, images: Tensor, seed: int) -> Tensor:
    """Apply the named alteration to a uint8 NHWC batch with a generator seeded by `seed`.

    Raises:
        KeyError: unknown alteration name (the message lists the valid ones).
    """
    if name not in ALTERATIONS:
        raise KeyError(f"Unknown alteration '{name}'; choose from {sorted(ALTERATIONS)}")
    generator = torch.Generator().manual_seed(seed)
    return ALTERATIONS[name](images, generator)


def random_augment(images: Tensor, generator: torch.Generator) -> Tensor:
    """Apply one randomly chosen alteration per call (whole batch), for training augmentation."""
    name = list(ALTERATIONS)[int(torch.randint(len(ALTERATIONS), (1,), generator=generator))]
    return ALTERATIONS[name](images, generator)


__all__ = [
    "ALTERATIONS",
    "Alteration",
    "apply_alteration",
    "color_jitter",
    "horizontal_flip",
    "jpeg_compress",
    "random_augment",
    "random_crop_resize",
    "strong_view",
]
