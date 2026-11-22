"""Figures of the team-B2 reports: every figure states its conclusion in its title.

Inputs are the outputs of `b2_train cv` (`<tag>.npz` out-of-fold probabilities, `<tag>.json`
breakdown and learning curves). Reference numbers (all-zeros, global switch, team A, team B phase 1)
are read from the phase-1 reports, so the bars cannot drift from the published values.

Usage (kit venv or main venv):
    python -m hackathon.wmloc.research.b2_figures --tag ps_base --out reports/phase2/wmloc/team-b2/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray

from hackathon.wmloc.research.b2_train import DECISION_THRESHOLD, results_dir
from hackathon.wmloc.research.data import load_image, load_records, load_truth
from hackathon.wmloc.research.evaluate import miou_per_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_REPORTS = Path("reports/phase1/wmloc")
PALETTE: dict[str, str] = {
    "ours": "#0072B2",
    "ref_a": "#E69F00",
    "ref_b": "#009E73",
    "zeros": "#999999",
    "switch": "#56B4E9",
    "truth": "#009E73",
    "prediction": "#D55E00",
}
METHOD_COLOURS: dict[str, str] = {
    "pixelseal": "#0072B2",
    "trustmark": "#D55E00",
    "stegastamp": "#009E73",
}
FONT_SIZE = 11
EXAMPLES_PER_GRID = 6
DPI = 110


def _style() -> None:
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE + 1})


def load_references() -> dict[str, dict[str, float]]:
    """Per-method reference scores read from the phase-1 reports of teams A and B."""
    metrics_a = json.loads((REPO_REPORTS / "team-a" / "metrics.json").read_text())
    metrics_b = json.loads((REPO_REPORTS / "team-b" / "metrics.json").read_text())
    references: dict[str, dict[str, float]] = {}
    for method, block in metrics_a.items():
        if method not in metrics_b["results"]:
            continue
        references[method] = {
            "all_zeros": float(block["baselines"]["all_zeros"]),
            "global_switch": float(block["baselines"]["global_switch_cv"]),
            "team_a": float(block["deployable_cv_miou"]),
            "team_b_phase1": float(metrics_b["results"][method]["extras_ft_seed2026"]["all"]),
        }
    return references


def fold_means(per_image: NDArray[np.float64], folds: NDArray[np.int64]) -> list[float]:
    """Mean mIoU of each fold (the spread behind every error bar)."""
    return [float(per_image[folds == f].mean()) for f in sorted(set(folds.tolist()))]


def dashboard_bars(
    scores: dict[str, tuple[float, float]], references: dict[str, dict[str, float]], out: Path
) -> None:
    """Bars: our CV mIoU against every reference, per method."""
    methods = list(scores)
    labels = ["all zeros", "global switch", "team A", "team B ph.1", "B2 (this work)"]
    colours = [
        PALETTE["zeros"],
        PALETTE["switch"],
        PALETTE["ref_a"],
        PALETTE["ref_b"],
        PALETTE["ours"],
    ]
    fig, axis = plt.subplots(figsize=(9.5, 4.6))
    width = 0.15
    for k, label in enumerate(labels):
        values, errors = [], []
        for method in methods:
            ref = references.get(method, {})
            keys = ("all_zeros", "global_switch", "team_a", "team_b_phase1")
            if k < 4:
                values.append(ref.get(keys[k], 0.0))
                errors.append(0.0)
            else:
                values.append(scores[method][0])
                errors.append(scores[method][1])
        axis.bar(
            np.arange(len(methods)) + (k - 2) * width,
            values,
            width,
            yerr=errors,
            label=label,
            color=colours[k],
            capsize=3,
        )
    gain = {m: scores[m][0] - references.get(m, {}).get("team_a", 0.0) for m in methods}
    axis.set_xticks(np.arange(len(methods)), methods)
    axis.set_ylabel("mIoU, 5-fold out-of-fold (higher is better)")
    axis.set_ylim(0.2, 0.75)
    best = max(gain, key=lambda m: gain[m])
    axis.set_title(f"B2 reaches {scores[best][0]:.3f} on {best}: {gain[best]:+.3f} over team A")
    axis.legend(fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_dashboard_bars.png", dpi=DPI)
    plt.close(fig)


def learning_curves(curves: list[dict[str, Any]], out: Path) -> None:
    """Training loss and held-out mIoU per epoch, one line per fold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for curve in curves:
        losses = curve["losses"]
        axes[0].plot(np.arange(1, len(losses) + 1), losses, label=f"fold {curve['fold']}")
        monitor = np.array(curve["monitor"])
        if monitor.size:
            axes[1].plot(monitor[:, 0], monitor[:, 1], marker="o", label=f"fold {curve['fold']}")
    final = np.mean([c["monitor"][-1][1] for c in curves if c["monitor"]])
    axes[0].set_title("training loss falls steadily")
    axes[1].set_title(f"held-out mIoU ends near {final:.3f} (mean of folds)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (BCE + soft mIoU)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out mIoU")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_learning_curves.png", dpi=DPI)
    plt.close(fig)


def regime_and_size(reports: dict[str, dict[str, Any]], out: Path) -> None:
    """mIoU by regime and by minority-zone size, per method."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    methods = list(reports)
    axes[0].bar(
        np.arange(len(methods)) - 0.18,
        [reports[m]["full"] for m in methods],
        0.36,
        label="full regime (find the hole)",
        color=PALETTE["ours"],
    )
    axes[0].bar(
        np.arange(len(methods)) + 0.18,
        [reports[m]["rect"] for m in methods],
        0.36,
        label="rectangle regime",
        color=PALETTE["ref_a"],
    )
    axes[0].set_xticks(np.arange(len(methods)), methods)
    axes[0].set_ylabel("mIoU")
    axes[0].set_ylim(0.2, 0.8)
    axes[0].set_title("by regime")
    axes[0].legend(fontsize=8)
    for axis, key, title in (
        (axes[1], "by_size_rect", "rectangle regime: by rectangle area"),
        (axes[2], "by_size_full", "full regime: by hole area"),
    ):
        for method in methods:
            bins = reports[method][key]
            xs = [f"{b['low']:.0%}-{b['high']:.0%}\nn={b['n']}" for b in bins]
            ys = [b["mean"] if b["mean"] is not None else np.nan for b in bins]
            axis.plot(xs, ys, marker="o", label=method, color=METHOD_COLOURS.get(method))
        axis.axhline(0.47, color=PALETTE["zeros"], linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylim(0.2, 0.9)
        axis.set_ylabel("mIoU")
    axes[1].legend(fontsize=8)
    fig.suptitle("Where the learned segmenter gains: dashed line is the constant fallback (0.47)")
    fig.tight_layout()
    fig.savefig(out / "fig_regime_size.png", dpi=DPI)
    plt.close(fig)


def _contour(axis: Any, mask: NDArray[Any], colour: str) -> None:
    axis.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=1.6)


def example_grid(
    items: list[tuple[str, NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32], float]],
    title: str,
    path: Path,
) -> None:
    """Rows of `image | truth | prediction`; the IoU is in the caption of each row."""
    n = len(items)
    cols = 3
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 2.4, rows * 2.5))
    axes = np.atleast_2d(axes)
    for index, (name, image, truth, probability, score) in enumerate(items):
        row, col = divmod(index, 2)
        base = col * cols
        prediction = probability >= DECISION_THRESHOLD
        picture = np.moveaxis(image, 0, -1)
        axes[row, base].imshow(picture)
        axes[row, base].set_title(f"{name[:8]}", fontsize=9)
        axes[row, base + 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row, base + 1].set_title("truth", fontsize=9)
        axes[row, base + 2].imshow(probability, cmap="viridis", vmin=0, vmax=1)
        _contour(axes[row, base + 2], prediction, PALETTE["prediction"])
        _contour(axes[row, base + 2], truth.astype(bool), "white")
        axes[row, base + 2].set_title(f"pred, mIoU {score:.2f}", fontsize=9)
    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def per_image_hist(ours: NDArray[np.float64], constant: NDArray[np.float64], out: Path) -> None:
    """Per-image mIoU distribution of the learned head against the constant fallback."""
    fig, axis = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41).tolist()
    axis.hist(constant, bins=bins, alpha=0.6, color=PALETTE["zeros"], label="constant fallback")
    axis.hist(ours, bins=bins, alpha=0.7, color=PALETTE["ours"], label="B2")
    share = float((ours > constant + 0.1).mean())
    axis.set_title(f"B2 beats the fallback by more than 0.1 on {share:.0%} of images")
    axis.set_xlabel("per-image mIoU")
    axis.set_ylabel("images")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_per_image_hist.png", dpi=DPI)
    plt.close(fig)


def matched_filter(data: dict[str, Any], out: Path) -> None:
    """Negative result: re-embedding the known message gives no local detector."""
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    series = data["auc_by_window_px"]
    windows = ["16", "32", "64"]
    width = 0.2
    colours = [PALETTE["zeros"], PALETTE["switch"], PALETTE["ref_a"], PALETTE["ours"]]
    for k, (label, values) in enumerate(series.items()):
        axis.bar(
            np.arange(len(windows)) + (k - 1.5) * width,
            [values[w] - 0.5 for w in windows],
            width,
            bottom=0.5,
            label=label,
            color=colours[k],
        )
    axis.axhline(0.5, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(windows)), [f"{w} px window" for w in windows])
    axis.set_ylabel("pixel AUC vs truth mask (0.5 = chance)")
    axis.set_ylim(0.4, 0.7)
    axis.set_title("A matched filter on the re-embedded message stays near chance on real images")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_matched_filter.png", dpi=DPI)
    plt.close(fig)


def native_windows(data: dict[str, Any], out: Path) -> None:
    """Negative result: a watermarked window decodes the message only when it is large."""
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    sizes = data["window_px"]
    axis.plot(
        sizes,
        data["agreement_shrink_0.5"],
        marker="o",
        color=PALETTE["ours"],
        label="window shrunk by 2 (embedder scale)",
    )
    axis.plot(
        sizes,
        data["agreement_shrink_1.0"],
        marker="s",
        color=PALETTE["ref_a"],
        label="window kept at 512 scale",
    )
    axis.axhline(0.5, color="black", linewidth=1, linestyle="--")
    axis.text(sizes[0], 0.515, "chance", fontsize=9)
    axis.set_xlabel("window side (px of the 512 image), fully inside the watermarked zone")
    axis.set_ylabel("bit agreement with the known message")
    axis.set_ylim(0.45, 1.0)
    axis.set_title("The Pixel Seal code is global: windows under about 250 px barely decode")
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_native_windows.png", dpi=DPI)
    plt.close(fig)


def build_all(tag: str, methods_in_file: list[str], out: Path) -> dict[str, Any]:
    """Draw every figure for the CV run `tag`; returns the numbers used in the titles."""
    _style()
    out.mkdir(parents=True, exist_ok=True)
    directory = results_dir()
    payload = json.loads((directory / f"{tag}.json").read_text())
    probabilities = np.load(directory / f"{tag}.npz")
    reports = payload["reports"]
    references = load_references()
    scores: dict[str, tuple[float, float]] = {}
    summary: dict[str, Any] = {}
    for method in methods_in_file:
        records = [r for r in load_records("train") if r.method == method]
        names = [r.name for r in records]
        truth = load_truth(names)
        probability = probabilities[method].astype(np.float32)
        prediction = (probability >= DECISION_THRESHOLD).astype(np.uint8)
        per_image = miou_per_image(prediction, truth)
        is_full = np.array([bool(r.full_watermark) for r in records])
        constant = miou_per_image(
            np.where(is_full[:, None, None], 1, 0).astype(np.uint8) * np.ones_like(truth),
            truth,
        )
        from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, stratified_folds

        folds = stratified_folds(
            [f"{method}/{flag}" for flag in is_full], DEFAULT_FOLDS, DEFAULT_SEED
        )
        means = fold_means(per_image, folds)
        scores[method] = (float(per_image.mean()), float(np.std(means, ddof=1)))
        order = np.argsort(per_image)
        picks = {"success": order[::-1][:EXAMPLES_PER_GRID], "failure": order[:EXAMPLES_PER_GRID]}
        for kind, indices in picks.items():
            items = [
                (
                    names[i],
                    load_image("train", names[i]).numpy(),
                    truth[i],
                    probability[i],
                    float(per_image[i]),
                )
                for i in indices
            ]
            title = (
                f"{method}: best out-of-fold images (mIoU {per_image[indices].mean():.2f})"
                if kind == "success"
                else f"{method}: worst out-of-fold images (mIoU {per_image[indices].mean():.2f})"
            )
            example_grid(items, title, out / f"fig_grid_{kind}_{method}.png")
        per_image_hist(per_image, constant, out)
        summary[method] = {"cv": scores[method][0], "fold_std": scores[method][1]}
    mf_path = out.parent / "matched_filter.json"
    if mf_path.is_file():
        matched_filter(json.loads(mf_path.read_text()), out)
    nw_path = out.parent / "native_windows.json"
    if nw_path.is_file():
        native_windows(json.loads(nw_path.read_text()), out)
    dashboard_bars(scores, references, out)
    learning_curves(payload["curves"], out)
    regime_and_size(reports, out)
    return summary


def submission_timeline(done_dir: Path, out: Path) -> list[tuple[str, float, float]]:
    """Predicted against public score of every B2 candidate already scored by the server.

    Returns `(name, expected, public)` in submission order and draws the figure.
    """
    rows: list[tuple[str, float, float, str]] = []
    for path in sorted(done_dir.glob("b2_*.json")):
        record = json.loads(path.read_text())
        result = record.get("result") or {}
        if record.get("expected") is None or result.get("score") is None:
            continue
        rows.append(
            (
                path.stem.replace("b2_", ""),
                float(record["expected"]),
                float(result["score"]),
                str(result.get("submitted_at", "")),
            )
        )
    rows.sort(key=lambda row: row[3])
    triples = [(name, expected, score) for name, expected, score, _ in rows]
    if not triples:
        return triples
    fig, axis = plt.subplots(figsize=(9, 4.4))
    positions = np.arange(len(triples))
    axis.bar(
        positions - 0.19,
        [t[1] for t in triples],
        0.38,
        label="predicted before the result",
        color=PALETTE["zeros"],
    )
    axis.bar(
        positions + 0.19, [t[2] for t in triples], 0.38, label="public score", color=PALETTE["ours"]
    )
    for position, (_, expected, score) in zip(positions, triples, strict=True):
        axis.text(position + 0.19, score + 0.004, f"{score:.3f}", ha="center", fontsize=10)
        axis.text(position - 0.19, expected + 0.004, f"{expected:.3f}", ha="center", fontsize=9)
    axis.set_xticks(positions, [t[0] for t in triples], rotation=12, ha="right")
    axis.set_ylim(0.44, max(t[2] for t in triples) + 0.04)
    axis.set_ylabel("score on the public split (about 30 percent of the test)")
    errors = [abs(t[1] - t[2]) for t in triples]
    axis.set_title(
        f"Public score rose {triples[-1][2] - triples[0][2]:+.3f} over {len(triples)} candidates; "
        f"mean prediction error {np.mean(errors):.3f}"
    )
    axis.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_submission_timeline.png", dpi=DPI)
    plt.close(fig)
    return triples


def method_bars(
    entries: list[tuple[str, str, str]], references: dict[str, dict[str, float]], out: Path
) -> None:
    """CV mIoU per method: team A reference next to the B2 head of each method.

    Args:
        entries: `(method, label, run tag)`; the run must contain `method` in its CV output.
    """
    fig, axis = plt.subplots(figsize=(9, 4.4))
    methods = [m for m, _, _ in entries]
    scores = [oof_score(tag, method) for method, _, tag in entries]
    positions = np.arange(len(entries))
    axis.bar(
        positions - 0.19,
        [references[m]["team_a"] for m in methods],
        0.38,
        label="team A (phase 1)",
        color=PALETTE["ref_a"],
    )
    axis.bar(
        positions + 0.19,
        [s[0] for s in scores],
        0.38,
        yerr=[s[1] for s in scores],
        label="B2 learned head",
        color=PALETTE["ours"],
        capsize=3,
    )
    for position, (_, value, _) in zip(
        positions, [(m, s[0], t) for (m, _, t), s in zip(entries, scores, strict=True)], strict=True
    ):
        axis.text(position + 0.19, value + 0.008, f"{value:.3f}", ha="center", fontsize=10)
    axis.set_xticks(positions, [f"{m}\n{label}" for m, label, _ in entries], fontsize=9)
    axis.set_ylim(0.40, 0.62)
    axis.set_ylabel("5-fold out-of-fold mIoU")
    mean_a = float(np.mean([references[m]["team_a"] for m in methods]))
    mean_b = float(np.mean([s[0] for s in scores]))
    axis.set_title(
        f"Mean over methods: {mean_b:.3f} against {mean_a:.3f} for team A ({mean_b - mean_a:+.3f})"
    )
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_method_bars.png", dpi=DPI)
    plt.close(fig)


def oof_score(tag: str, method: str) -> tuple[float, float]:
    """Out-of-fold mIoU of the CV run `tag` for `method` and the std of its fold means."""
    from hackathon.wmloc.research.evaluate import DEFAULT_FOLDS, DEFAULT_SEED, stratified_folds

    records = [r for r in load_records("train") if r.method == method]
    truth = load_truth([r.name for r in records])
    probability = np.load(results_dir() / f"{tag}.npz")[method].astype(np.float32)
    per_image = miou_per_image((probability >= DECISION_THRESHOLD).astype(np.uint8), truth)
    is_full = np.array([bool(r.full_watermark) for r in records])
    folds = stratified_folds([f"{method}/{flag}" for flag in is_full], DEFAULT_FOLDS, DEFAULT_SEED)
    return float(per_image.mean()), float(np.std(fold_means(per_image, folds), ddof=1))


def ablation_figure(items: list[tuple[str, str]], method: str, reference: float, out: Path) -> None:
    """Bars of the CV score per (label, run tag), with the constant fallback as reference line."""
    scores = [(label, *oof_score(tag, method)) for label, tag in items]
    fig, axis = plt.subplots(figsize=(9, 4.4))
    colours = [PALETTE["ours"] if "full" in label else PALETTE["ref_a"] for label, *_ in scores]
    axis.bar(
        [s[0] for s in scores],
        [s[1] for s in scores],
        yerr=[s[2] for s in scores],
        color=colours,
        capsize=3,
    )
    axis.axhline(reference, color=PALETTE["zeros"], linestyle="--", linewidth=1.2)
    axis.text(
        len(scores) - 0.55,
        reference - 0.012,
        f"global switch {reference:.3f}",
        fontsize=9,
        ha="right",
    )
    for position, (_, value, _) in enumerate(scores):
        axis.text(position, value + 0.006, f"{value:.3f}", ha="center", fontsize=10)
    axis.set_ylim(0.40, 0.60)
    axis.set_ylabel(f"{method}: out-of-fold mIoU")
    best = max(scores, key=lambda s: s[1])
    worst = min(scores, key=lambda s: s[1])
    axis.set_title(
        f"The extractor features carry the gain: {worst[1]:.3f} without them, {best[1]:.3f} at best"
    )
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation_sources.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--methods", nargs="+", default=["pixelseal"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ablation", nargs="+", help="label=tag pairs, bars of one extra figure")
    parser.add_argument("--done-dir", type=Path, help="queue/wmloc/done: draws the timeline")
    parser.add_argument(
        "--method-bars", nargs="+", help="method:label:tag triples for the per-method bars"
    )
    args = parser.parse_args()
    _style()
    summary = build_all(args.tag, args.methods, args.out)
    if args.ablation:
        pairs = [tuple(item.split("=", 1)) for item in args.ablation]
        reference = load_references()[args.methods[0]]["global_switch"]
        ablation_figure([(a, b) for a, b in pairs], args.methods[0], reference, args.out)
    if args.done_dir:
        summary["timeline"] = submission_timeline(args.done_dir, args.out)
    if args.method_bars:
        triples = [tuple(item.split(":", 2)) for item in args.method_bars]
        method_bars([(m, label, tag) for m, label, tag in triples], load_references(), args.out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
