"""Phase 2 tools: weight extrapolation of a trained candidate and a one-line proxy card per file.

The first real leaderboard score (0.0117 for `cand_gold_soft_alpha0.75`) says the surrogate of
phase 1 does not predict the real score. The hypothesis tested here (a HYPOTHESIS, not a fact) is
that the candidates sit too close to the original: a gold model has never seen the forget images,
so on them it behaves like on test images (loss of the order of the validation loss, accuracy of
the order of the validation accuracy), far from a memorised model.

Two entry points:

* `extrapolate`: `theta_orig + alpha * (theta_trained - theta_orig)` for alpha > 1, free of any
  training. The trained state is a candidate file saved at alpha = 1.
* `card`: evaluate any candidate file and append one proxy line to a JSONL file.

Usage (from the repo root):

    python -m hackathon.unlearning.research.phase2 extrapolate \
        --base <task>/candidates/cand_gold_soft_alpha1.00.pt --alphas 1.5 2.0 3.0 --tag gs
    python -m hackathon.unlearning.research.phase2 card --files a.pt b.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from hackathon.common.config import load_settings
from hackathon.unlearning.constants import MODEL_FILENAME, default_task_dir
from hackathon.unlearning.model import build_model, load_model, select_device
from hackathon.unlearning.research.context import ResearchContext, build_context, evaluate
from hackathon.unlearning.research.experiments import REMOVED_CLASSES
from hackathon.unlearning.research.runner import RESEARCH_SUBDIR, THREAD_LIMIT, blend_state
from hackathon.unlearning.submission import save_submission

LOGGER = logging.getLogger("hackathon.unlearning.research")
PHASE2_CANDIDATES_SUBDIR = Path("candidates") / "phase2"
PHASE2_RESEARCH_SUBDIR = "phase2"
CARDS_FILENAME = "cards.jsonl"
ALTERED_GROUPS = ("alt_crop", "alt_jitter")


class Phase2Error(RuntimeError):
    """A phase 2 operation received an unusable input."""


@dataclass(frozen=True)
class CardRow:
    """One proxy line of a candidate (every number is a PROXY, never the official score)."""

    name: str
    values: dict[str, float]

    def to_json(self) -> str:
        """Serialise as one JSON object."""
        return json.dumps({"name": self.name, **self.values})


def load_state(path: Path) -> dict[str, Tensor]:
    """Load a float32 state dict saved by `save_submission`."""
    if not path.is_file():
        raise Phase2Error(f"State file not found: {path}")
    state: dict[str, Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return state


def card_values(metrics: dict[str, object]) -> dict[str, float]:
    """Flatten the evaluation metrics into the columns of a card line.

    The columns answer the phase 2 question: how far from the original is the candidate on the
    forget set (loss mean and std, accuracy against the pseudo-labels, confidence), how much does
    retention drift (KL on out-of-distribution images, argmax flips), and what the phase 1
    surrogate says (mean and minimum over the hypothetical golds; known not to predict the score).
    """
    groups = metrics["groups"]
    surrogate = metrics["surrogate"]
    assert isinstance(groups, dict) and isinstance(surrogate, dict)
    exact = groups["exact"]
    retain = groups["retain"]
    values: dict[str, float] = {
        "loss_mean": exact["loss_mean_sub"],
        "loss_std": exact["loss_std_sub"],
        "pseudo_acc": exact["pseudo_accuracy_sub"],
        "confidence": exact["confidence_sub"],
        "forget_kl_t2": float(metrics["forget_kl_t2"]),  # type: ignore[arg-type]
        "retain_kl_t2": float(metrics["retain_kl_t2"]),  # type: ignore[arg-type]
        "retain_flips": float(metrics["retain_argmax_flip_rate"]),  # type: ignore[arg-type]
        "retain_confidence": retain["confidence_sub"],
        "surrogate_mean": float(surrogate["forget_mean_over_hypotheses"]),
        "surrogate_min": float(surrogate["forget_min_over_hypotheses"]),
    }
    for name in ALTERED_GROUPS:
        if name in groups:
            values[f"{name}_loss_mean"] = groups[name]["loss_mean_sub"]
            values[f"{name}_pseudo_acc"] = groups[name]["pseudo_accuracy_sub"]
    return values


def _context(task_dir: Path, data_dir: Path) -> tuple[ResearchContext, Path]:
    research_dir = task_dir / RESEARCH_SUBDIR
    context = build_context(task_dir, data_dir.parent, select_device(None), research_dir)
    phase2_dir = task_dir / RESEARCH_SUBDIR / PHASE2_RESEARCH_SUBDIR
    phase2_dir.mkdir(parents=True, exist_ok=True)
    return context, phase2_dir


def _save_arrays(phase2_dir: Path, name: str, arrays: dict[str, Tensor]) -> None:
    """Per-image arrays (loss, confidence, argmax, logits on subset A) for the report histograms."""
    torch.save(arrays, phase2_dir / f"{name}.arrays.pt")


def _append_card(phase2_dir: Path, row: CardRow) -> None:
    with (phase2_dir / CARDS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(row.to_json() + "\n")


def extrapolate(
    context: ResearchContext,
    base_state: dict[str, Tensor],
    alphas: tuple[float, ...],
    tag: str,
    candidates_dir: Path,
    phase2_dir: Path,
) -> list[CardRow]:
    """Blend `theta_orig + alpha * (base - theta_orig)`, evaluate, save a validated file per alpha."""
    original_state = {k: v.detach().cpu().clone() for k, v in context.original.state_dict().items()}
    scratch: nn.Module = copy.deepcopy(context.original).eval()
    rows: list[CardRow] = []
    for alpha in alphas:
        started = time.time()
        scratch.load_state_dict(blend_state(original_state, base_state, alpha))
        result = evaluate(context, scratch, full=True)
        name = f"p2_{tag}_alpha{alpha:.2f}"
        path = candidates_dir / f"{name}.pt"
        report = save_submission(scratch, path)
        row = CardRow(name, {"alpha": alpha, **card_values(result.metrics)})
        _append_card(phase2_dir, row)
        _save_arrays(phase2_dir, name, result.arrays)
        rows.append(row)
        LOGGER.info(
            "%s validated (%d bytes) in %.0f s: %s",
            name, report.size_bytes, time.time() - started, row.to_json(),
        )  # fmt: skip
    return rows


def blend_only(
    task_dir: Path, base_state: dict[str, Tensor], alphas: tuple[float, ...], tag: str,
    candidates_dir: Path,
) -> list[Path]:
    """Blend and save validated files WITHOUT evaluating them (light: no forward pass).

    Used to prepare ladder candidates while the machine is busy; their proxies are read later with
    the `card` command, or interpolated from the neighbouring alphas.
    """
    original = load_model(task_dir / MODEL_FILENAME, torch.device("cpu"))
    original_state = {k: v.detach().clone() for k, v in original.state_dict().items()}
    scratch: nn.Module = copy.deepcopy(original).eval()
    paths: list[Path] = []
    for alpha in alphas:
        scratch.load_state_dict(blend_state(original_state, base_state, alpha))
        path = candidates_dir / f"p2_{tag}_alpha{alpha:.2f}.pt"
        save_submission(scratch, path)
        paths.append(path)
        LOGGER.info("%s saved and validated", path.name)
    return paths


HEAD_BIAS_KEY = "head.bias"


def offset_head_bias(
    state: dict[str, Tensor], classes: tuple[int, ...], delta: float
) -> dict[str, Tensor]:
    """Copy of `state` with `delta` subtracted from the head bias of `classes`.

    Only the logits of those classes move (by exactly `delta`, on every image): the other logits,
    hence the behaviour on images of other classes, are untouched.
    """
    if HEAD_BIAS_KEY not in state:
        raise Phase2Error(f"State has no '{HEAD_BIAS_KEY}'")
    edited = {key: value.clone() for key, value in state.items()}
    bias = edited[HEAD_BIAS_KEY]
    if max(classes) >= len(bias) or min(classes) < 0:
        raise Phase2Error(f"Class index out of range for a head of {len(bias)} classes")
    bias[list(classes)] -= delta
    return edited


def mask_and_save(
    base_state: dict[str, Tensor], classes: tuple[int, ...], delta: float, path: Path
) -> Path:
    """Apply `offset_head_bias`, then save and validate the file at `path`."""
    model = build_model()
    model.load_state_dict(offset_head_bias(base_state, classes, delta))
    save_submission(model, path)
    return path


def card(context: ResearchContext, files: list[Path], phase2_dir: Path) -> list[CardRow]:
    """Evaluate existing candidate files (state dicts) and append their proxy lines."""
    scratch: nn.Module = copy.deepcopy(context.original).eval()
    rows: list[CardRow] = []
    for path in files:
        scratch.load_state_dict(load_state(path))
        result = evaluate(context, scratch, full=True)
        row = CardRow(path.stem, card_values(result.metrics))
        _append_card(phase2_dir, row)
        _save_arrays(phase2_dir, path.stem, result.arrays)
        rows.append(row)
        LOGGER.info("%s", row.to_json())
    return rows


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extra = sub.add_parser("extrapolate", help="theta_orig + alpha (base - theta_orig)")
    extra.add_argument("--base", type=Path, required=True, help="trained state, saved at alpha 1")
    extra.add_argument("--alphas", type=float, nargs="+", required=True)
    extra.add_argument("--tag", required=True, help="name stem of the output files")
    extra.add_argument(
        "--also-card", type=Path, nargs="*", default=[], help="existing files to card afterwards"
    )
    blend = sub.add_parser("blend", help="like extrapolate, but save without evaluating (light)")
    blend.add_argument("--base", type=Path, required=True)
    blend.add_argument("--alphas", type=float, nargs="+", required=True)
    blend.add_argument("--tag", required=True)
    mask = sub.add_parser("mask", help="subtract delta from the head bias of the removed classes")
    mask.add_argument("--base", type=Path, required=True, help="state dict (candidate file)")
    mask.add_argument("--delta", type=float, required=True)
    mask.add_argument("--name", required=True, help="output file stem under candidates/phase2")
    cards = sub.add_parser("card", help="proxy line of existing candidate files")
    cards.add_argument("--files", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(THREAD_LIMIT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = load_settings()
    task_dir = default_task_dir(settings.data_dir)
    if args.command == "mask":
        out = mask_and_save(
            load_state(args.base), REMOVED_CLASSES, args.delta,
            task_dir / PHASE2_CANDIDATES_SUBDIR / f"{args.name}.pt",
        )  # fmt: skip
        LOGGER.info("saved and validated %s", out)
        return
    if args.command == "blend":
        blend_only(
            task_dir, load_state(args.base), tuple(args.alphas), args.tag,
            task_dir / PHASE2_CANDIDATES_SUBDIR,
        )  # fmt: skip
        return
    context, phase2_dir = _context(task_dir, settings.data_dir)
    if args.command == "extrapolate":
        extrapolate(
            context,
            load_state(args.base),
            tuple(args.alphas),
            args.tag,
            task_dir / PHASE2_CANDIDATES_SUBDIR,
            phase2_dir,
        )
        if args.also_card:
            card(context, list(args.also_card), phase2_dir)
    else:
        card(context, list(args.files), phase2_dir)


if __name__ == "__main__":
    main()
