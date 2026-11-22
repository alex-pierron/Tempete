"""Pre-flight of the V3 campaign: the trust_region "calibrated" target measured on the real model.

On the training fold of the tuning runs (2800 images, same audit split as `run`), with the views
the method draws (`_view_logits`: generator seeded with seed + VIEW_TARGET_SEED_OFFSET) and the
method's own target function (`calibrated_target_logits`), for each crop area, number of views V,
trust_lambda, trust_hedge and hedge temperature tau:

  reach      KL(p_orig || target) at T = 1, mean: the largest reachable radius (the method's
             radius is radius_fraction x reach);
  agree      share of images where the target keeps the pseudo-label;
  het        std / mean of the target loss l* over images (the gold stand-ins: 1.7 to 3.8);
  ceil       bench-v4 forget score (exact images only, mean_std x prediction) of a student whose
             output WAS the target, under each of the six gold stand-ins of the bench (crops
             s40/s50/s60, patch masks m30/m20/m15), drawn exactly as `run_bench` does. The crops
             share their view family with the target (a ceiling there is partly circular); the
             masks do not. V1 and V2 targets are measured the same way for reference.

Usage (repo root, CUDA; ~2 min on an RTX 5080):
    PYTHONPATH=src ./scripts/gpu_guard.sh .venv/bin/python scripts/unlearning_v3_targets.py \\
        experiments/unlearning/measurements/v3_targets.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hackathon.unlearning.alterations import strong_view
from hackathon.unlearning.bench import AUDIT_SEED, GOLD_VIEWS, _gold_stand_ins, audit_split
from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.methods import VIEW_TARGET_SEED_OFFSET, calibrated_target_logits
from hackathon.unlearning.model import load_model, predict

TASK_DIR = Path("data/hackathon/unlearning")
AREAS = (0.35, 0.45, 0.50, 0.60)
VIEWS = (1, 2, 4, 8)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
HEDGES = (0.0, 0.1, 0.25, 0.5)
HEDGE_TAUS = (2.0, 6.0)
SEED = 0
BATCH = 64
T_SCORE = 2.0
FAMILIES = {"crop": ("s40", "s50", "s60"), "mask": ("m30", "m20", "m15")}


def ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.double(), labels, reduction="none")


def kl_mean(p: torch.Tensor, logq: torch.Tensor) -> float:
    p = p.double()
    return float((p * (p.clamp_min(1e-12).log() - logq.double())).sum(1).mean())


class Ceiling:
    """bench-v4 forget score on exact images of candidate logits, per gold stand-in."""

    def __init__(self, orig: torch.Tensor, labels: torch.Tensor, gold: dict) -> None:
        self.labels = labels
        self.orig = orig.double()
        self.gold = {}
        for key, views in gold.items():
            stats = [(ce(v, labels).mean(), ce(v, labels).std(unbiased=False)) for v in views]
            gm = sum(float(m) for m, _ in stats) / len(stats)
            gs = sum(float(s) for _, s in stats) / len(stats)
            gp = torch.stack([torch.softmax(v.double() / T_SCORE, 1) for v in views]).mean(0)
            d_orig = self._distance(self.orig, gm, gs)
            kl_orig = kl_mean(gp, torch.log_softmax(self.orig / T_SCORE, 1))
            self.gold[key] = (gm, gs, gp, d_orig, kl_orig)

    def _distance(self, logits: torch.Tensor, gm: float, gs: float) -> float:
        loss = ce(logits, self.labels)
        return abs(float(loss.mean()) - gm) + abs(float(loss.std(unbiased=False)) - gs)

    def __call__(self, logits: torch.Tensor) -> dict:
        out = {}
        for key, (gm, gs, gp, d_orig, kl_orig) in self.gold.items():
            ms = 1 - min(1.0, max(0.0, self._distance(logits, gm, gs) / max(d_orig, 1e-12)))
            kl_c = kl_mean(gp, torch.log_softmax(logits.double() / T_SCORE, 1))
            pred = 1 - min(1.0, max(0.0, kl_c / max(kl_orig, 1e-12)))
            out[key] = round(ms * pred, 5)
        out["crop_min"] = min(out[k] for k in FAMILIES["crop"])
        out["mask_min"] = min(out[k] for k in FAMILIES["mask"])
        out["min"] = min(out["crop_min"], out["mask_min"])
        return out


def describe(logits: torch.Tensor, p: torch.Tensor, labels: torch.Tensor, ceil: Ceiling) -> dict:
    lq = torch.log_softmax(logits.double(), 1)
    loss = -lq.gather(1, labels[:, None]).squeeze(1)
    return {
        "reach": round(kl_mean(p, lq), 5),
        "agree": round(float((lq.argmax(1) == labels).double().mean()), 5),
        "loss_mean": round(float(loss.mean()), 5),
        "het": round(float(loss.std(unbiased=False) / loss.mean()), 4),
        "ceil": ceil(logits),
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    device = torch.device("cuda")
    original = load_model(TASK_DIR / "model.pt", device)
    forget = load_forget_images(TASK_DIR / "forget_images.npz")
    kept, _ = audit_split(original, forget, 0.2)
    images = forget[kept].to(device)
    z = predict(original, images, BATCH)
    labels = z.argmax(1)
    p = torch.softmax(z.double(), 1)
    gold = {}
    for index, (key, draw, strength) in enumerate(_gold_stand_ins()):
        gold[key] = []
        for view in range(GOLD_VIEWS):
            generator = torch.Generator().manual_seed(AUDIT_SEED * 1000 + index * 100 + view)
            gold[key].append(predict(original, draw(images, generator, strength), BATCH))
    ceil = Ceiling(z, labels, gold)
    results: dict = {"n_images": len(images), "seed": SEED, "reference": {}, "v3": {}}
    results["reference"]["v1_tau6"] = describe(z / 6.0, p, labels, ceil)
    view_cache: dict[float, list[torch.Tensor]] = {}
    for area in AREAS:
        t0 = time.monotonic()
        generator = torch.Generator().manual_seed(SEED + VIEW_TARGET_SEED_OFFSET)
        view_cache[area] = [
            predict(original, strong_view(images, generator, area), BATCH)
            for _ in range(max(VIEWS))
        ]
        print(f"views at area {area}: {time.monotonic() - t0:.1f} s", flush=True)
    v2 = torch.stack([torch.softmax(v.double() / 1.0, 1) for v in view_cache[0.45][:3]]).mean(0)
    results["reference"]["v2_tau1_a045_V3"] = describe(v2.clamp_min(1e-12).log(), p, labels, ceil)
    grid = itertools.product(AREAS, VIEWS, LAMBDAS, HEDGES, HEDGE_TAUS)
    for area, n, lam, hedge, tau in grid:
        if hedge == 0.0 and tau != HEDGE_TAUS[-1]:
            continue  # the hedge temperature is irrelevant without a hedge
        config = UnlearnConfig(
            method="trust_region", trust_target="calibrated", trust_lambda=lam,
            trust_hedge=hedge, target_temperature=tau, target_views=n, target_view_min_area=area,
        )  # fmt: skip
        logits = calibrated_target_logits(z, view_cache[area][:n], config)
        key = f"a{area:g}_V{n}_l{lam:g}_b{hedge:g}_t{tau:g}"
        results["v3"][key] = {"area": area, "views": n, "lambda": lam, "hedge": hedge, "tau": tau}
        results["v3"][key].update(describe(logits, p, labels, ceil))
    out_path.write_text(json.dumps(results, indent=1))
    print(f"{len(results['v3'])} V3 targets -> {out_path}")


if __name__ == "__main__":
    main()
