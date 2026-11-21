"""White-box adversarial robustness probes for watermark extractors.

These are the standard "adversarial robustness" measurements of watermarking benchmarks
(WAVES, the NeurIPS 2024 "Erasing the Invisible" challenge): given the extractor of a
publicly released model, how much does a bounded L-inf perturbation change what it reads?
They are evaluation tools; the scope is exactly the functions below.

* `pgd_remove`: projected gradient descent that drives the extractor's bit logits towards 0
  (reads as random bits) or, with `target_bits`, towards the opposite of those bits.
* `pgd_forge`: PGD that makes the extractor read a chosen message on an unmarked image.
* `estimate_pattern` / `subtract_pattern` / `add_pattern`: averaging ("collusion") analysis,
  which measures how much of a watermark is a message-dependent, image-independent pattern.

Tensor conventions are those of `wmkit.types`: images `[B, 3, H, W]` in `[0, 1]`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor], torch.Tensor]
"""Differentiable extractor: images [B, 3, H, W] -> bit logits [B, n_bits] (sign = bit)."""

DEFAULT_EPS = 4.0 / 255.0
DEFAULT_STEP = 1.0 / 255.0
DEFAULT_STEPS = 50
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

DenoiseMethod = Literal["gaussian", "median"]
GAUSSIAN_SIGMA = 1.0
GAUSSIAN_RADIUS_IN_SIGMAS = 3
MEDIAN_KERNEL_SIZE = 3


class AdversarialInputError(ValueError):
    """Inputs of an adversarial routine have an unsupported shape, range or value."""


# --- PGD ---------------------------------------------------------------------------------------


def _check_budget(eps: float, step: float, steps: int) -> None:
    if eps < 0 or step <= 0 or steps < 0:
        raise AdversarialInputError(
            f"need eps >= 0, step > 0, steps >= 0; got eps={eps}, step={step}, steps={steps}"
        )


def _check_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise AdversarialInputError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")


def _expand_bits(bits: torch.Tensor, batch: int, n_bits: int, name: str) -> torch.Tensor:
    """Broadcast [n_bits] or [1, n_bits] or [B, n_bits] bits to [B, n_bits] float32."""
    if bits.ndim == 1:
        bits = bits.unsqueeze(0)
    if bits.ndim != 2 or bits.shape[1] != n_bits or bits.shape[0] not in (1, batch):
        raise AdversarialInputError(
            f"{name} must have shape [{n_bits}], [1, {n_bits}] or [{batch}, {n_bits}], "
            f"got {tuple(bits.shape)}"
        )
    return bits.expand(batch, -1).to(torch.float32)


def _projected_descent(
    extract_logits: LogitsFn,
    images: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float,
    step: float,
    steps: int,
) -> torch.Tensor:
    """Minimise `loss_fn(extract_logits(x))` over x in the L-inf ball of radius eps around images."""
    original = images.detach()
    adversarial = original.clone()
    with torch.enable_grad():
        for _ in range(steps):
            adversarial.requires_grad_(True)
            loss = loss_fn(extract_logits(adversarial))
            (gradient,) = torch.autograd.grad(loss, adversarial)
            with torch.no_grad():
                adversarial = adversarial.detach() - step * gradient.sign()
                adversarial = original + (adversarial - original).clamp(-eps, eps)
                adversarial = adversarial.clamp(PIXEL_MIN, PIXEL_MAX)
    return adversarial.detach()


def pgd_remove(
    extract_logits: LogitsFn,
    images: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    step: float = DEFAULT_STEP,
    steps: int = DEFAULT_STEPS,
    target_bits: torch.Tensor | None = None,
) -> torch.Tensor:
    """L-inf PGD that degrades what the extractor reads from watermarked `images`.

    Without `target_bits`, the loss is the mean squared logit: minimising it pushes every logit
    towards 0, i.e. the extractor reads random bits. With `target_bits` (the embedded message,
    [n_bits] or [B, n_bits]), the loss is the cross-entropy towards the opposite bits.

    Args:
        extract_logits: Differentiable extractor, e.g. `SealWatermarker.extract_logits`.
        images: Watermarked images [B, 3, H, W] in [0, 1].
        eps: L-inf radius (in [0, 1] pixel units).
        step: Signed-gradient step.
        steps: Number of iterations.
        target_bits: Optional ground-truth bits to flip.

    Returns:
        Attacked images, same shape as `images`, inside the eps-ball and within [0, 1].

    Raises:
        AdversarialInputError: bad shapes or budget.
    """
    _check_images(images)
    _check_budget(eps, step, steps)
    if target_bits is None:

        def loss_fn(logits: torch.Tensor) -> torch.Tensor:
            return logits.pow(2).mean()

    else:
        n_bits = int(target_bits.shape[-1])
        inverted = 1.0 - _expand_bits(target_bits, images.shape[0], n_bits, "target_bits")

        def loss_fn(logits: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(logits, inverted.to(logits.device))

    return _projected_descent(extract_logits, images, loss_fn, eps, step, steps)


def pgd_forge(
    extract_logits: LogitsFn,
    images: torch.Tensor,
    target_message: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    step: float = DEFAULT_STEP,
    steps: int = DEFAULT_STEPS,
) -> torch.Tensor:
    """L-inf PGD that makes the extractor read `target_message` on (clean) `images`.

    Args:
        extract_logits: Differentiable extractor.
        images: Images [B, 3, H, W] in [0, 1].
        target_message: Bits to be read, [n_bits] or [B, n_bits], values in {0, 1}.
        eps: L-inf radius.
        step: Signed-gradient step.
        steps: Number of iterations.

    Returns:
        Perturbed images, same shape, inside the eps-ball and within [0, 1].

    Raises:
        AdversarialInputError: bad shapes or budget.
    """
    _check_images(images)
    _check_budget(eps, step, steps)
    n_bits = int(target_message.shape[-1])
    wanted = _expand_bits(target_message, images.shape[0], n_bits, "target_message")

    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, wanted.to(logits.device))

    return _projected_descent(extract_logits, images, loss_fn, eps, step, steps)


# --- Pattern averaging -------------------------------------------------------------------------


def _gaussian_kernel(device: torch.device) -> torch.Tensor:
    radius = int(GAUSSIAN_RADIUS_IN_SIGMAS * GAUSSIAN_SIGMA)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(offsets**2) / (2 * GAUSSIAN_SIGMA**2))
    return kernel / kernel.sum()


def _denoise(images: torch.Tensor, method: DenoiseMethod) -> torch.Tensor:
    """Low-pass version of `images`; the residual `images - denoise(images)` is high-frequency."""
    if method == "gaussian":
        kernel = _gaussian_kernel(images.device)
        radius = kernel.numel() // 2
        padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
        horizontal = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        blurred = F.conv2d(padded, horizontal, groups=3)
        return F.conv2d(blurred, vertical, groups=3)
    if method == "median":
        radius = MEDIAN_KERNEL_SIZE // 2
        padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
        windows = padded.unfold(2, MEDIAN_KERNEL_SIZE, 1).unfold(3, MEDIAN_KERNEL_SIZE, 1)
        return windows.reshape(*windows.shape[:4], -1).median(dim=-1).values
    raise AdversarialInputError(f"unknown denoise method {method!r}; use 'gaussian' or 'median'")


def estimate_pattern(watermarked: torch.Tensor, *, denoise: str = "gaussian") -> torch.Tensor:
    """Estimate the shared watermark pattern of N same-size images carrying the SAME message.

    Each image's high-frequency residual (image minus its low-pass version) is averaged over
    the N images: content cancels out as 1/sqrt(N) while any message-dependent,
    image-independent component remains. Content-adaptive watermarks (JND-modulated) leave
    little of it, which is exactly what this probe measures.

    Args:
        watermarked: [N, 3, H, W] in [0, 1], N >= 1, same size, same embedded message.
        denoise: `gaussian` or `median`.

    Returns:
        Pattern [3, H, W] (signed, in pixel units).

    Raises:
        AdversarialInputError: bad shape or unknown `denoise`.
    """
    _check_images(watermarked)
    if denoise not in ("gaussian", "median"):
        raise AdversarialInputError(f"unknown denoise method {denoise!r}; use 'gaussian' or 'median'")
    method: DenoiseMethod = "gaussian" if denoise == "gaussian" else "median"
    residual = watermarked - _denoise(watermarked, method)
    return residual.mean(dim=0)


def _apply_pattern(images: torch.Tensor, pattern: torch.Tensor, signed_strength: float) -> torch.Tensor:
    _check_images(images)
    if pattern.shape != images.shape[1:]:
        raise AdversarialInputError(
            f"pattern {tuple(pattern.shape)} does not match image size {tuple(images.shape[1:])}"
        )
    shifted = images + signed_strength * pattern.to(images.device, images.dtype).unsqueeze(0)
    return shifted.clamp(PIXEL_MIN, PIXEL_MAX)


def subtract_pattern(
    images: torch.Tensor, pattern: torch.Tensor, strength: float = 1.0
) -> torch.Tensor:
    """Return `clamp(images - strength * pattern)`: remove an estimated pattern."""
    return _apply_pattern(images, pattern, -strength)


def add_pattern(images: torch.Tensor, pattern: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Return `clamp(images + strength * pattern)`: transplant an estimated pattern."""
    return _apply_pattern(images, pattern, strength)
