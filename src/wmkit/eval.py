"""Evaluation harness: run a watermarker against a battery of attacks, get a long DataFrame.

One row per (image, attack, watermarked-or-clean). The clean rows (the original image put
through the same attack and read with the same expected message) measure the false-positive
rate; the watermarked rows measure robustness. Images are processed one by one (batch of
one) because real datasets mix sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from tqdm import tqdm

from wmkit.attacks import attack_seed, reseed_attack
from wmkit.messages import random_messages
from wmkit.metrics import (
    DETECTION_NEG_LOG10_P,
    bit_accuracy,
    lpips_distance,
    neg_log10_pvalue,
    psnr,
    ssim,
)
from wmkit.types import AttackSpec, Watermarker

RESULT_COLUMNS: list[str] = [
    "method", "image_id", "is_watermarked", "attack", "attack_group", "n_bits", "bit_acc",
    "neg_log10_p", "detected", "psnr", "ssim", "lpips",
]
"""Columns of the frame returned by `evaluate`, in this exact order."""

SUMMARY_KEYS_BY_GROUP: list[str] = ["method", "is_watermarked", "attack_group"]
SUMMARY_KEYS_BY_ATTACK: list[str] = ["method", "is_watermarked", "attack"]
SUMMARY_METRICS: list[str] = ["bit_acc", "neg_log10_p", "detected", "psnr", "ssim", "lpips"]
RGB_CHANNELS: int = 3


class EvaluationError(ValueError):
    """Raised when evaluation inputs or a watermarker's outputs violate the contract."""


def _validate_image(image_id: str, image: torch.Tensor) -> None:
    """Images must be [3, H, W] float tensors."""
    if image.ndim != 3 or image.shape[0] != RGB_CHANNELS:
        raise EvaluationError(
            f"image {image_id!r} must have shape [3, H, W], got {tuple(image.shape)}"
        )
    if not image.is_floating_point():
        raise EvaluationError(f"image {image_id!r} must be floating point, got {image.dtype}")


def _score(
    watermarker: Watermarker, attacked: torch.Tensor, message: torch.Tensor
) -> tuple[float, float, bool]:
    """Extract from `attacked` and return (bit accuracy, -log10 p, detected)."""
    with torch.no_grad():
        soft_bits = watermarker.extract(attacked)
    expected_shape = (1, watermarker.n_bits)
    if tuple(soft_bits.shape) != expected_shape:
        raise EvaluationError(
            f"{watermarker.name}.extract returned {tuple(soft_bits.shape)}, expected {expected_shape}"
        )
    accuracy = bit_accuracy(soft_bits.to(message.device), message)
    n_correct = torch.round(accuracy * watermarker.n_bits)
    score = neg_log10_pvalue(watermarker.n_bits, n_correct)
    value = float(score[0])
    return float(accuracy[0]), value, value > DETECTION_NEG_LOG10_P


def evaluate(
    watermarker: Watermarker,
    images: Sequence[tuple[str, torch.Tensor]],
    attacks: Sequence[AttackSpec],
    *,
    seed: int = 0,
    device: str = "cpu",
    with_lpips: bool = False,
    include_clean: bool = True,
) -> pd.DataFrame:
    """Embed a random message in every image, attack it, extract, and score.

    Args:
        watermarker: Method under test.
        images: `(image_id, tensor[3, H, W] in [0, 1])` pairs.
        attacks: Attacks to apply (see `wmkit.attacks.get_suite`).
        seed: Seeds the per-image random messages (one generator, drawn in image order,
            so two methods evaluated with the same seed read the same messages) and, mixed
            with the image id and attack name, every random attack.
        device: Torch device on which images are embedded, attacked and read.
        with_lpips: Also compute LPIPS (downloads AlexNet weights on first use).
        include_clean: Also attack the original image (`is_watermarked=False`); `detected`
            on those rows is a false positive.

    Returns:
        DataFrame with one row per (image, attack[, clean]) and columns `RESULT_COLUMNS`.
        `psnr`, `ssim` and `lpips` describe the watermarked image before attack against
        the original and are repeated on every row of that image (NaN LPIPS if disabled).

    Raises:
        EvaluationError: A malformed image, or a watermarker output of the wrong shape.
    """
    torch_device = torch.device(device)
    message_generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, object]] = []
    progress = tqdm(total=len(images) * len(attacks), desc=watermarker.name, leave=False,
                    disable=None, unit="attack")
    for image_id, image in images:
        _validate_image(image_id, image)
        original = image.detach().to(torch_device, torch.float32)[None]
        message = random_messages(1, watermarker.n_bits, message_generator).to(torch_device)
        with torch.no_grad():
            watermarked = watermarker.embed(original, message)
        if watermarked.shape != original.shape:
            raise EvaluationError(
                f"{watermarker.name}.embed changed the image size: {tuple(original.shape)} -> "
                f"{tuple(watermarked.shape)}"
            )
        watermarked = watermarked.detach()
        quality_psnr = float(psnr(watermarked, original)[0])
        quality_ssim = float(ssim(watermarked, original)[0])
        quality_lpips = (
            float(lpips_distance(watermarked, original)[0]) if with_lpips else float("nan")
        )
        for spec in attacks:
            variants: list[tuple[bool, torch.Tensor]] = [(True, watermarked)]
            if include_clean:
                variants.append((False, original))
            for is_watermarked, source in variants:
                # Same seed for the watermarked image and its clean twin: identical crop window.
                reseed_attack(spec, attack_seed(seed, image_id, spec.name))
                with torch.no_grad():
                    attacked = spec.fn(source)
                accuracy, score, detected = _score(watermarker, attacked, message)
                rows.append({
                    "method": watermarker.name, "image_id": image_id,
                    "is_watermarked": is_watermarked, "attack": spec.name,
                    "attack_group": spec.group, "n_bits": watermarker.n_bits,
                    "bit_acc": accuracy, "neg_log10_p": score, "detected": detected,
                    "psnr": quality_psnr, "ssim": quality_ssim, "lpips": quality_lpips,
                })
            progress.update(1)
    progress.close()
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return frame.astype({
        "is_watermarked": bool, "n_bits": int, "bit_acc": float, "neg_log10_p": float,
        "detected": bool, "psnr": float, "ssim": float, "lpips": float,
    })


def _summarise(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean of the metric columns per group of `keys`; `detected` becomes a rate."""
    missing = [column for column in keys + SUMMARY_METRICS if column not in frame.columns]
    if missing:
        raise EvaluationError(f"frame lacks columns {missing}; expected those of evaluate()")
    numeric = frame.assign(detected=frame["detected"].astype(float))
    summary = numeric.groupby(keys, sort=False)[SUMMARY_METRICS].mean()
    return summary.reset_index()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Means per `(method, is_watermarked, attack_group)`; `detected` is the detection rate."""
    return _summarise(df, SUMMARY_KEYS_BY_GROUP)


def summarize_by_attack(df: pd.DataFrame) -> pd.DataFrame:
    """Means per `(method, is_watermarked, attack)`; `detected` is the detection rate."""
    return _summarise(df, SUMMARY_KEYS_BY_ATTACK)
