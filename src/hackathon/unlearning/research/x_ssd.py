"""Forget-only Selective Synaptic Dampening (LFSSD variant) for the 31-unlearning task.

References (read for this work):
- Foster, Schoepf, Brintrup, "Fast Machine Unlearning Without Retraining Through Selective
  Synaptic Dampening", AAAI 2024 (SSD): parameter importances on the full training set D and on
  the forget set D_f; parameters with Omega_f > alpha * Omega_D are multiplied by
  beta = min(lambda * Omega_D / Omega_f, 1).
- Foster, Schoepf, Brintrup, "Loss-Free Machine Unlearning", ICLR 2024 Tiny Paper (LFSSD,
  arXiv 2402.19308): the Fisher diagonal is replaced by the sensitivity of the squared l2 norm of
  the output, so no label and no loss are needed.

The task rule is "the unlearning method has to work from the original model and the forget set
alone": the paper's Omega_D (full training set, precomputed) is not available. It is replaced by
the importance on STRONG VIEWS of the forget images (crop of ~35 % of the area + colour jitter).
Why this is the right stand-in: the original is memorised on the exact forget images (confidence
0.9997) but behaves like on unseen images on strong views (confidence 0.81-0.93, argmax changes on
the hard images). A parameter far more important for the exact pixels than for their views
carries image-specific memorisation, which is exactly what a model retrained without the image
lacks; class knowledge, shared by both, is kept. Everything is computed from model.pt and
forget_images.npz only.

CLI (run from the repo root with PYTHONPATH=src):
  python -m hackathon.unlearning.research.x_ssd importance --out FILE [--limit N] [--batch-size B]
  python -m hackathon.unlearning.research.x_ssd dampen --importance FILE --alpha A --out STATE.pt
  python -m hackathon.unlearning.research.x_ssd quantiles --importance FILE
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hackathon.unlearning.alterations import strong_view
from hackathon.unlearning.constants import (
    DEFAULT_SEED,
    FORGET_IMAGES_FILENAME,
    IMAGE_CHANNELS,
    MODEL_FILENAME,
)
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.model import load_model, load_state_dict_file, preprocess, select_device

LOGGER = logging.getLogger("x_ssd")

STRONG_VIEW_MIN_AREA = 0.35
"""Crop area of the non-memorised views (same value as the team's targets and anchors)."""
STRONG_VIEW_SEED_OFFSET = 31
"""Seed offset of the strong views (distinct from the extraction offsets 1-4, 11 and 20)."""
DEFAULT_IMPORTANCE_BATCH = 8
"""Small micro-batches: the gradient of a batch mean averages out per-image directions, and the
memorisation we want to locate is per image."""
DEFAULT_LAMBDA = 1.0
"""SSD/LFSSD fix lambda = 1 and search alpha only (paper, section 4)."""
RATIO_EPSILON = 1e-20
"""Guards Omega_view = 0 (a parameter no view uses): its ratio is then huge, i.e. selected."""
LOG_EVERY_BATCHES = 50
IMAGE_NDIM = 4
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
ALPHA_PROBES = (2.0, 3.0, 5.0, 10.0, 20.0, 50.0)

IMPORTANCE_EXACT_KEY = "omega_exact"
IMPORTANCE_VIEW_KEY = "omega_view"
META_KEY = "meta"


class DampeningError(ValueError):
    """Invalid SSD hyperparameters or importance file."""


def output_norm_sensitivity(
    model: nn.Module, batches: Iterable[Tensor], device: torch.device
) -> dict[str, Tensor]:
    """LFSSD importance: mean over batches of |d mean_b ||z_b||^2 / d theta| (z = logits).

    Args:
        model: The network, in eval mode, on `device`.
        batches: uint8 NHWC image batches.
        device: Compute device.

    Returns:
        Parameter name -> importance tensor (float32, on CPU), same shape as the parameter.
    """
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    totals = {name: torch.zeros_like(p, device=device) for name, p in params.items()}
    count = 0
    start = time.monotonic()
    for batch in batches:
        model.zero_grad(set_to_none=True)
        logits = model(preprocess(batch, device))
        logits.pow(2).sum(dim=1).mean().backward()
        for name, p in params.items():
            if p.grad is not None:
                totals[name] += p.grad.detach().abs()
        count += 1
        if count % LOG_EVERY_BATCHES == 0:
            LOGGER.info("  %d batches, %.1f s", count, time.monotonic() - start)
    if count == 0:
        raise DampeningError("No batch: cannot compute an importance.")
    model.zero_grad(set_to_none=True)
    return {name: (total / count).float().cpu() for name, total in totals.items()}


def batched(images: Tensor, batch_size: int) -> Iterable[Tensor]:
    """Consecutive slices of `images` of at most `batch_size` rows."""
    for begin in range(0, len(images), batch_size):
        yield images[begin : begin + batch_size]


def strong_views(images: Tensor, seed: int, batch_size: int) -> Iterable[Tensor]:
    """Strong views of `images`, generated batch by batch with one seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    for batch in batched(images, batch_size):
        yield strong_view(batch, generator, STRONG_VIEW_MIN_AREA)


def importance_ratio(omega_exact: Tensor, omega_view: Tensor) -> Tensor:
    """Omega_exact / Omega_view, with Omega_view = 0 mapped to a huge ratio (always selected)."""
    return omega_exact / omega_view.clamp_min(RATIO_EPSILON)


def dampen_state(
    state: Mapping[str, Tensor],
    omega_exact: Mapping[str, Tensor],
    omega_view: Mapping[str, Tensor],
    alpha: float,
    lam: float = DEFAULT_LAMBDA,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """SSD rule: theta_i *= min(lam * Omega_view_i / Omega_exact_i, 1) where ratio > alpha.

    Args:
        state: Original state dict (float32).
        omega_exact: Importance on the exact forget images.
        omega_view: Importance on the strong views (stand-in for Omega_D).
        alpha: Selection threshold (> 0).
        lam: Dampening constant (> 0).

    Returns:
        (new state, statistics: selected fraction, mean beta of the selected parameters).

    Raises:
        DampeningError: Non-positive alpha or lam, or importances missing for a parameter.
    """
    if alpha <= 0 or lam <= 0:
        raise DampeningError(f"alpha and lambda must be > 0 (got {alpha}, {lam}).")
    out: dict[str, Tensor] = {}
    selected_total = 0
    param_total = 0
    beta_sum = 0.0
    for key, value in state.items():
        if key not in omega_exact or key not in omega_view:
            out[key] = value.clone()
            continue
        ratio = importance_ratio(omega_exact[key].double(), omega_view[key].double())
        selected = ratio > alpha
        beta = torch.where(selected, (lam / ratio).clamp(max=1.0), torch.ones_like(ratio))
        out[key] = (value.double() * beta).to(value.dtype)
        selected_total += int(selected.sum())
        param_total += selected.numel()
        beta_sum += float(beta[selected].sum())
    if param_total == 0:
        raise DampeningError("No parameter has an importance: wrong importance file.")
    stats = {
        "selected_fraction": selected_total / param_total,
        "selected_count": float(selected_total),
        "mean_beta_selected": beta_sum / selected_total if selected_total else 1.0,
    }
    return out, stats


def ratio_quantiles(
    omega_exact: Mapping[str, Tensor], omega_view: Mapping[str, Tensor]
) -> dict[str, float]:
    """Quantiles of the importance ratio and the fraction selected at a few alphas."""
    ratios = torch.cat(
        [importance_ratio(omega_exact[k], omega_view[k]).flatten() for k in omega_exact]
    )
    sample = ratios[torch.randperm(len(ratios), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    result = {f"q{q}": float(torch.quantile(sample.double(), q)) for q in QUANTILES}
    for alpha in ALPHA_PROBES:
        result[f"frac_gt_{alpha}"] = float((ratios > alpha).double().mean())
    return result


def compute_importances(
    task_dir: Path, limit: int | None, batch_size: int, device_name: str | None
) -> dict[str, object]:
    """Both importances of the original model on the forget set (exact and strong views)."""
    device = select_device(device_name)
    model = load_model(task_dir / MODEL_FILENAME, device)
    images = load_forget_images(task_dir / FORGET_IMAGES_FILENAME, limit)
    LOGGER.info("device=%s images=%d batch=%d", device, len(images), batch_size)
    start = time.monotonic()
    omega_exact = output_norm_sensitivity(model, batched(images, batch_size), device)
    LOGGER.info("exact done in %.1f s", time.monotonic() - start)
    omega_view = output_norm_sensitivity(
        model, strong_views(images, DEFAULT_SEED + STRONG_VIEW_SEED_OFFSET, batch_size), device
    )
    LOGGER.info("views done in %.1f s", time.monotonic() - start)
    return {
        IMPORTANCE_EXACT_KEY: omega_exact,
        IMPORTANCE_VIEW_KEY: omega_view,
        META_KEY: {
            "images": len(images),
            "batch_size": batch_size,
            "view_min_area": STRONG_VIEW_MIN_AREA,
            "seconds": time.monotonic() - start,
            "device": str(device),
        },
    }


def load_npz_images(path: Path, key: str) -> Tensor:
    """uint8 NHWC [N, 224, 224, 3] images of an npz (e.g. inversion images of the original).

    Raises:
        DampeningError: missing key or wrong dtype/shape (a normalised float array would silently
            give a meaningless importance).
    """
    with np.load(path) as payload:
        if key not in payload.files:
            raise DampeningError(f"{path} has no key '{key}' (keys: {payload.files}).")
        array = payload[key]
    if array.dtype != np.uint8 or array.ndim != IMAGE_NDIM or array.shape[-1] != IMAGE_CHANNELS:
        raise DampeningError(f"{path}:{key} must be uint8 NHWC, got {array.dtype} {array.shape}.")
    return torch.from_numpy(np.ascontiguousarray(array))


def compute_view_importance(
    task_dir: Path, images_path: Path, key: str, batch_size: int, device_name: str | None
) -> dict[str, object]:
    """Omega_D stand-in on external-free images given as an npz (no strong view applied)."""
    device = select_device(device_name)
    model = load_model(task_dir / MODEL_FILENAME, device)
    images = load_npz_images(images_path, key)
    start = time.monotonic()
    omega_view = output_norm_sensitivity(model, batched(images, batch_size), device)
    return {
        IMPORTANCE_VIEW_KEY: omega_view,
        META_KEY: {
            "images": len(images),
            "source": str(images_path),
            "batch_size": batch_size,
            "seconds": time.monotonic() - start,
        },
    }


def load_importances(
    path: Path, view_path: Path | None = None
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Read an importance file written by `importance`; `view_path` overrides its Omega_view."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or IMPORTANCE_EXACT_KEY not in loaded:
        raise DampeningError(f"{path} is not an x_ssd importance file.")
    if view_path is None:
        return loaded[IMPORTANCE_EXACT_KEY], loaded[IMPORTANCE_VIEW_KEY]
    other = torch.load(view_path, map_location="cpu", weights_only=True)
    if not isinstance(other, dict) or IMPORTANCE_VIEW_KEY not in other:
        raise DampeningError(f"{view_path} is not an x_ssd view-importance file.")
    return loaded[IMPORTANCE_EXACT_KEY], other[IMPORTANCE_VIEW_KEY]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="x_ssd")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("importance")
    imp.add_argument("--task-dir", type=Path, required=True)
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--limit", type=int, default=None)
    imp.add_argument("--batch-size", type=int, default=DEFAULT_IMPORTANCE_BATCH)
    imp.add_argument("--device", default=None)
    damp = sub.add_parser("dampen")
    damp.add_argument("--task-dir", type=Path, required=True)
    damp.add_argument("--importance", type=Path, required=True)
    damp.add_argument("--alpha", type=float, required=True)
    damp.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    damp.add_argument("--out", type=Path, required=True)
    quant = sub.add_parser("quantiles")
    quant.add_argument("--importance", type=Path, required=True)
    quant.add_argument("--view-importance", type=Path, default=None)
    view = sub.add_parser("view-importance")
    view.add_argument("--task-dir", type=Path, required=True)
    view.add_argument("--images", type=Path, required=True)
    view.add_argument("--key", default="images")
    view.add_argument("--out", type=Path, required=True)
    view.add_argument("--batch-size", type=int, default=DEFAULT_IMPORTANCE_BATCH)
    view.add_argument("--device", default=None)
    damp.add_argument("--view-importance", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "view-importance":
        payload = compute_view_importance(
            args.task_dir, args.images, args.key, args.batch_size, args.device
        )
        tmp = args.out.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(args.out)
        LOGGER.info("wrote %s %s", args.out, payload[META_KEY])
        return 0
    if args.command == "importance":
        payload = compute_importances(args.task_dir, args.limit, args.batch_size, args.device)
        tmp = args.out.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(args.out)
        LOGGER.info("wrote %s %s", args.out, payload[META_KEY])
        return 0
    if args.command == "quantiles":
        exact, view_omega = load_importances(args.importance, args.view_importance)
        print(ratio_quantiles(exact, view_omega))
        return 0
    exact, view_omega = load_importances(args.importance, args.view_importance)
    state = load_state_dict_file(args.task_dir / MODEL_FILENAME)
    new_state, stats = dampen_state(state, exact, view_omega, args.alpha, args.lam)
    tmp = args.out.with_suffix(".tmp")
    torch.save(new_state, tmp)
    tmp.replace(args.out)
    print({"out": str(args.out), "alpha": args.alpha, "lambda": args.lam, **stats})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
