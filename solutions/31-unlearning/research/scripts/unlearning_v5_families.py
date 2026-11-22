"""After the V5 step 0: is per-image difficulty specific to the family of gold stand-ins?

The step 0 of band_ascent found its selection (top 25 % of the original's mean loss on 4 strong
crop views) at 2.1-2.6x chance under the bench's crops and exactly at chance under its masks. On
the training fold of the tuning runs (2800 images, same audit split as `run`), no training:

  spearman   rank correlation of the per-image loss (mean over 4 views, on the pseudo-label)
             between the six stand-ins of bench-v4 (crops s40/s50/s60, masks m30/m20/m15, drawn
             exactly as `run_bench` does) and two selectors: `v5` (the views band_ascent draws,
             seed 0, area 0.5) and `mask_own` (30 % patch masks drawn at seeds the bench never
             uses, so not the bench's own masks);
  select     for each selector (the two above and the stand-ins s50 and m20 themselves) and each
             stand-in, at phi = 0.25: precision on the stand-in's errors (argmax of its view
             mixture at T = 2 != pseudo-label), its base rate, and the share of its mean loss
             carried by the selected images (`diagnose` definitions).

Usage (repo root, CUDA; ~1 min on an RTX 5080):
    PYTHONPATH=src ./scripts/gpu_guard.sh .venv/bin/python scripts/unlearning_v5_families.py \\
        experiments/unlearning/measurements/v5_families.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hackathon.unlearning.alterations import mask_patches
from hackathon.unlearning.bench import GOLD_VIEWS, audit_split, gold_view_logits
from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.methods import view_logits
from hackathon.unlearning.model import load_model, predict

TASK_DIR = Path("data/hackathon/unlearning")
PHI = 0.25
BATCH = 64
OWN_MASK_FRACTION = 0.30
OWN_MASK_SEED = 7_000_000
"""Far from the bench's stand-in seeds (AUDIT_SEED * 1000 + index * 100 + view)."""


def per_image_loss(views: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [F.cross_entropy(v.double().cpu(), labels, reduction="none") for v in views]
    ).mean(0)


def errors_of(views: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    mix = torch.stack([torch.softmax(v.double().cpu() / SCORING_TEMPERATURE, 1) for v in views])
    return mix.mean(0).argmax(1) != labels


def ranks(values: torch.Tensor) -> torch.Tensor:
    out = torch.empty(len(values), dtype=torch.float64)
    out[torch.argsort(values)] = torch.arange(len(values), dtype=torch.float64)
    return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def top(values: torch.Tensor, share: float) -> torch.Tensor:
    mask = torch.zeros(len(values), dtype=torch.bool)
    mask[torch.topk(values, max(1, round(share * len(values)))).indices] = True
    return mask


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original = load_model(TASK_DIR / "model.pt", device)
    forget = load_forget_images(TASK_DIR / "forget_images.npz")
    kept, _ = audit_split(original, forget, 0.2, BATCH)
    trained = forget[kept].to(device)
    labels = predict(original, trained, BATCH).cpu().argmax(1)
    config = UnlearnConfig(method="band_ascent", seed=0, target_views=4, target_view_min_area=0.5)
    v5_views = [v.cpu() for v in view_logits(original, trained, config, device)]
    own_masks = []
    for view in range(GOLD_VIEWS):
        generator = torch.Generator().manual_seed(OWN_MASK_SEED + view)
        own_masks.append(
            predict(original, mask_patches(trained, generator, OWN_MASK_FRACTION), BATCH).cpu()
        )
    gold = {
        k: [v.cpu() for v in views]
        for k, views in gold_view_logits(original, trained, BATCH).items()
    }
    losses = {"v5": per_image_loss(v5_views, labels), "mask_own": per_image_loss(own_masks, labels)}
    losses |= {key: per_image_loss(views, labels) for key, views in gold.items()}
    errors = {key: errors_of(views, labels) for key, views in gold.items()}
    names = list(losses)
    report: dict[str, object] = {
        "trained": len(trained),
        "phi": PHI,
        "spearman": {a: {b: spearman(losses[a], losses[b]) for b in names} for a in names},
        "select": {},
    }
    for selector in ("v5", "mask_own", "s50", "m20"):
        chosen = top(losses[selector], PHI)
        rows = {}
        for key in gold:
            err = errors[key]
            precision = float(err[chosen].double().mean())
            base = float(err.double().mean())
            rows[key] = {
                "precision": precision,
                "base_rate": base,
                "lift": precision / base if base > 0 else float("nan"),
                "loss_share": float(losses[key][chosen].sum() / losses[key].sum()),
                "loss_share_oracle": float(
                    losses[key][top(losses[key], PHI)].sum() / losses[key].sum()
                ),
            }
        report["select"][selector] = rows  # type: ignore[index]
    report["seconds"] = time.monotonic() - started
    print("spearman (rows/cols: " + " ".join(names) + ")")
    for a in names:
        print(f"{a:9}" + " ".join(f"{report['spearman'][a][b]:+.2f}" for b in names))  # type: ignore[index]
    for selector, rows in report["select"].items():  # type: ignore[union-attr]
        print(f"selector {selector}: " + " | ".join(
            f"{k} {r['lift']:.2f}x {r['loss_share']:.2f}" for k, r in rows.items()
        ))  # fmt: skip
    print(f"{report['seconds']:.1f} s")
    if out:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
