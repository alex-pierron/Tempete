"""Figures of the team-B report: learning curves, score bars, overlays, saliency, synth vs real.

Run in the kit venv after `pretrain`, `zeroshot` and `crossval` have written their outputs:
    python -m hackathon.wmloc.research.figures_b --work DIR --real DIR --out figures_dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from hackathon.wmloc.research.kit import METHODS
from hackathon.wmloc.research.model import Localizer, residual_input
from hackathon.wmloc.research.train import ArrayData, load_arrays, official_miou, pick_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELL = 4
EXAMPLES_PER_CELL = 2
SALIENCY_EXAMPLES = 2
VARIANTS = ("resid", "extras")


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def learning_curves(work: Path, out: Path) -> None:
    """Pretraining loss and held-out synthetic mIoU per epoch, by method and variant."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for method in METHODS:
        for variant in VARIANTS:
            path = work / "ckpt" / f"{method}_{variant}_pretrain.json"
            if not path.exists():
                continue
            history = _load(path)["history"]
            style = "-" if variant == "extras" else "--"
            axes[0].plot([h["train_loss"] for h in history], style, label=f"{method}/{variant}")
            axes[1].plot([h["val_miou"] for h in history], style, label=f"{method}/{variant}")
    axes[0].set_title("training loss (BCE + soft IoU), synthetic")
    axes[1].set_title("held-out SYNTHETIC mIoU")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=110)
    plt.close(fig)


def _overlay(axis: Any, image: Any, truth: Any, pred: Any, title: str) -> None:
    axis.imshow(image)
    axis.contour(np.kron(truth, np.ones((CELL, CELL))), [0.5], colors="lime", linewidths=1.3)
    axis.contour(np.kron(pred, np.ones((CELL, CELL))), [0.5], colors="red", linewidths=1.0)
    axis.set_title(title, fontsize=7)
    axis.axis("off")


def overlays(real: ArrayData, probs: np.ndarray, method: str, tag: str, out: Path) -> None:
    """Best and worst held-out examples per regime (green = truth, red = prediction)."""
    assert real.masks is not None and real.full is not None
    pred = probs >= 0.5
    scores = official_miou(pred, real.masks.astype(bool))
    fig, axes = plt.subplots(4, 2 * EXAMPLES_PER_CELL // 2 * 2, figsize=(11, 11))
    row = 0
    for regime_name, selector in (("full", real.full), ("region", ~real.full)):
        members = np.flatnonzero(selector)
        order = members[np.argsort(scores[members])]
        for label, picks in (
            ("worst", order[: EXAMPLES_PER_CELL * 2]),
            ("best", order[-EXAMPLES_PER_CELL * 2 :]),
        ):
            for column, index in enumerate(picks):
                _overlay(
                    axes[row, column],
                    real.images[index],
                    real.masks[index],
                    pred[index],
                    f"{regime_name} {label} mIoU={scores[index]:.2f}",
                )
            row += 1
    fig.suptitle(f"{method} ({tag}): green = truth, red = prediction (held-out folds)")
    fig.tight_layout()
    fig.savefig(out / f"overlays_{method}.png", dpi=80)
    plt.close(fig)


def saliency(work: Path, real: ArrayData, method: str, variant: str, out: Path) -> None:
    """Input-gradient saliency of the summed logits w.r.t. the image (what the model looks at)."""
    checkpoint = work / "ckpt" / f"{method}_{variant}.pt"
    if not checkpoint.exists():
        return
    device = pick_device()
    model = Localizer(use_extras=variant == "extras")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    assert real.masks is not None and real.full is not None
    picks = [int(np.flatnonzero(real.full)[0]), int(np.flatnonzero(~real.full)[0])][
        :SALIENCY_EXAMPLES
    ]
    fig, axes = plt.subplots(2, len(picks) * 2, figsize=(3.2 * len(picks) * 2, 6.4))
    for column, index in enumerate(picks):
        images = torch.from_numpy(np.ascontiguousarray(real.images[index : index + 1])).to(device)
        maps = torch.from_numpy(real.maps[index : index + 1].astype(np.float32)).to(device)
        scalars = torch.from_numpy(real.scalars[index : index + 1]).to(device)
        x = residual_input(images).detach().requires_grad_(True)
        prediction = model(images, maps, scalars, residual=x)
        torch.sigmoid(prediction).sum().backward()  # type: ignore[no-untyped-call]
        assert x.grad is not None
        grad = x.grad[0, 3:].abs().mean(dim=0).cpu().numpy()
        probs = torch.sigmoid(prediction)[0].detach().cpu().numpy()
        axes[0, 2 * column].imshow(real.images[index])
        axes[0, 2 * column].contour(
            np.kron(real.masks[index], np.ones((CELL, CELL))), [0.5], colors="lime"
        )
        axes[0, 2 * column].set_title(
            f"{'full' if real.full[index] else 'region'} (truth in green)", fontsize=8
        )
        axes[0, 2 * column + 1].imshow(probs, vmin=0, vmax=1, cmap="magma")
        axes[0, 2 * column + 1].set_title("predicted probability", fontsize=8)
        axes[1, 2 * column].imshow(grad, cmap="viridis", vmax=np.quantile(grad, 0.995))
        axes[1, 2 * column].set_title("|d prob / d high-pass residual|", fontsize=8)
        axes[1, 2 * column + 1].imshow(
            real.maps[index, 1].astype(np.float32), vmin=0.3, vmax=1, cmap="viridis"
        )
        axes[1, 2 * column + 1].set_title("extractor agreement map 8x8 (input)", fontsize=8)
    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(f"{method}/{variant}: what the model looks at")
    fig.tight_layout()
    fig.savefig(out / f"saliency_{method}.png", dpi=80)
    plt.close(fig)


def score_bars(work: Path, out: Path) -> dict[str, Any]:
    """Grouped bars: baselines, zero-shot synthetic, k-fold fine-tuned, by method and regime."""
    table: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, method in zip(axes, METHODS):
        series: dict[str, list[float]] = {}
        for path in sorted((work / "cv").glob(f"{method}_*_cv.json")):
            data = _load(path)
            label = path.stem.replace(f"{method}_", "").replace("_cv", "")
            summary = data["summary"]
            series[label] = [summary["all"], summary["full_regime"], summary["region_regime"]]
            table.setdefault(method, {})[label] = {
                **summary,
                "fold_std": data["fold_std"],
                "fold_means": data["fold_means"],
            }
            baselines = data["baselines"]
            table[method]["baselines"] = baselines
        for path in sorted((work / "zeroshot").glob(f"{method}_*_zeroshot.json")):
            data = _load(path)
            label = "zeroshot_" + path.stem.replace(f"{method}_", "").replace("_zeroshot", "")
            summary = data["summary"]
            series[label] = [summary["all"], summary["full_regime"], summary["region_regime"]]
            table.setdefault(method, {})[label] = summary
        width = 0.8 / max(1, len(series))
        for position, (label, values) in enumerate(series.items()):
            axis.bar(np.arange(3) + position * width, values, width, label=label)
        if method in table and "baselines" in table[method]:
            axis.axhline(
                table[method]["baselines"]["global_switch"],
                color="k",
                linestyle=":",
                label="global switch",
            )
            axis.axhline(
                table[method]["baselines"]["all_zeros"],
                color="gray",
                linestyle="--",
                label="all zeros",
            )
        axis.set_xticks(np.arange(3) + 0.3)
        axis.set_xticklabels(["all", "full", "region"])
        axis.set_title(method)
        axis.legend(fontsize=6)
    axes[0].set_ylabel("official mIoU (real train)")
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=110)
    plt.close(fig)
    return table


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--best-tag", default="extras_ft_seed2026")
    args = parser.parse_args(argv)
    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    learning_curves(work, out)
    table = score_bars(work, out)
    (out.parent / "scores_table.json").write_text(json.dumps(table, indent=1))
    for method in METHODS:
        real = load_arrays(Path(args.real) / "train" / method, mmap=False)
        cv_path = work / "cv" / f"{method}_{args.best_tag}_cv.npz"
        if cv_path.exists():
            overlays(real, np.load(cv_path)["probs"].astype(np.float32), method, args.best_tag, out)
        for variant in VARIANTS:
            if (work / "ckpt" / f"{method}_{variant}.pt").exists():
                saliency(work, real, method, variant, out)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
