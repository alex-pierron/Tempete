"""Pre-flight of the V2 campaign: what the trust_region "views" target looks like on the real model.

On the training fold of the tuning runs (2800 images, same audit split as `run`), for each crop
area a, draw 10 strong views exactly as `trust_target_logits` does (generator seeded with
seed + VIEW_TARGET_SEED_OFFSET, views drawn in sequence on the whole set), then for each
temperature tau and number of views V (prefixes of the 10 views):

  reach      KL(p_orig || q) at T = 1, mean: the largest reachable radius;
  agree      share of images where argmax q = argmax of the original (pseudo-label);
  het        std / mean of -log q_y over images (per-image heterogeneity of the target);
  ceil_sXX   bench forget score (exact images only) of a student whose output WAS q, under each
             gold stand-in of the bench: an upper bound of how much of the stand-in the target
             encodes (the mechanical part of a V2 gain). Same for the V1 target softmax(z / tau).

Usage (repo root, CUDA; ~2 min on an RTX 5080, ~7 GB of GPU memory):
    PYTHONPATH=src ./scripts/gpu_guard.sh .venv/bin/python scripts/unlearning_v2_targets.py \
        experiments/unlearning/measurements/v2_targets.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hackathon.unlearning.alterations import strong_view
from hackathon.unlearning.bench import AUDIT_SEED, GOLD_STRENGTHS, GOLD_VIEWS, audit_split
from hackathon.unlearning.data import load_forget_images
from hackathon.unlearning.methods import VIEW_TARGET_SEED_OFFSET
from hackathon.unlearning.model import load_model, predict

AREAS = (0.25, 0.35, 0.45, 0.55, 0.65)
TAUS = (1.0, 2.0, 3.0, 4.0, 6.0)
VIEWS = (1, 3, 6, 10)
SEED = 0
BATCH = 64
T_SCORE = 2.0


def ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.double(), labels, reduction="none")


def kl_mean(p: torch.Tensor, logq: torch.Tensor) -> float:
    p = p.double()
    return float((p * (p.clamp_min(1e-12).log() - logq.double())).sum(1).mean())


def loss_distance(logits, labels, gold_mean: float, gold_std: float) -> float:
    """d = |mean(L) - mean(L_gold)| + |std(L) - std(L_gold)| (the subject's mean_std distance)."""
    loss = ce(logits, labels)
    return abs(float(loss.mean()) - gold_mean) + abs(float(loss.std(unbiased=False)) - gold_std)


def ceiling(cand_logits, orig_logits, labels, gold_views):
    """Forget score on exact images (mean_std x prediction) of `cand_logits`, per stand-in."""
    out = {}
    for key, views in gold_views.items():
        stats = [(ce(v, labels).mean(), ce(v, labels).std(unbiased=False)) for v in views]
        gm = sum(float(m) for m, _ in stats) / len(stats)
        gs = sum(float(s) for _, s in stats) / len(stats)
        d_cand = loss_distance(cand_logits, labels, gm, gs)
        d_orig = loss_distance(orig_logits, labels, gm, gs)
        ms = 1 - min(1.0, max(0.0, d_cand / max(d_orig, 1e-12)))
        gold_p = torch.stack([torch.softmax(v.double() / T_SCORE, 1) for v in views]).mean(0)
        kl_c = kl_mean(gold_p, torch.log_softmax(cand_logits.double() / T_SCORE, 1))
        kl_o = kl_mean(gold_p, torch.log_softmax(orig_logits.double() / T_SCORE, 1))
        pred = 1 - min(1.0, max(0.0, kl_c / max(kl_o, 1e-12)))
        out[key] = {"mean_std": ms, "prediction": pred, "forget": ms * pred}
    return out


def main() -> None:
    out_path = Path(sys.argv[1])
    task_dir = Path("data/hackathon/unlearning")
    device = torch.device("cuda")
    original = load_model(task_dir / "model.pt", device)
    forget = load_forget_images(task_dir / "forget_images.npz")
    kept, _ = audit_split(original, forget, 0.2)
    images = forget[kept].to(device)
    z = predict(original, images, BATCH)
    labels = z.argmax(1)
    p = torch.softmax(z.double(), 1)
    # The bench's gold stand-ins, drawn exactly as run_bench does (same seeds, same strengths).
    gold = {}
    for index, strength in enumerate(GOLD_STRENGTHS):
        views = []
        for v in range(GOLD_VIEWS):
            generator = torch.Generator().manual_seed(AUDIT_SEED * 1000 + index * 100 + v)
            views.append(predict(original, strong_view(images, generator, strength), BATCH))
        gold[f"s{round(strength * 100)}"] = views
    results = {"n_images": len(images), "seed": SEED, "v1": {}, "v2": {}}
    for tau in TAUS:
        lq = torch.log_softmax(z.double() / tau, 1)
        loss = -lq.gather(1, labels[:, None]).squeeze(1)
        results["v1"][f"tau{tau:g}"] = {
            "reach": kl_mean(p, lq),
            "agree": 1.0,
            "het": float(loss.std(unbiased=False) / loss.mean()),
            "ceil": ceiling(lq, z, labels, gold),
        }
    torch.cuda.reset_peak_memory_stats()
    for area in AREAS:
        t0 = time.monotonic()
        generator = torch.Generator().manual_seed(SEED + VIEW_TARGET_SEED_OFFSET)
        view_logits = []
        for _ in range(max(VIEWS)):
            view = strong_view(images, generator, area)
            view_logits.append(predict(original, view, BATCH).double())
            del view
        for tau in TAUS:
            for n in VIEWS:
                q = torch.stack([torch.softmax(v / tau, 1) for v in view_logits[:n]]).mean(0)
                lq = q.clamp_min(1e-12).log()
                loss = -lq.gather(1, labels[:, None]).squeeze(1)
                results["v2"][f"a{area:g}_tau{tau:g}_V{n}"] = {
                    "area": area,
                    "tau": tau,
                    "views": n,
                    "reach": kl_mean(p, lq),
                    "agree": float((q.argmax(1) == labels).double().mean()),
                    "het": float(loss.std(unbiased=False) / loss.mean()),
                    "ceil": ceiling(lq, z, labels, gold),
                }
        print(f"area {area}: {time.monotonic() - t0:.1f} s", flush=True)
    results["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out_path.write_text(json.dumps(results, indent=1))
    print("peak GPU", results["peak_gpu_gb"])


if __name__ == "__main__":
    main()
