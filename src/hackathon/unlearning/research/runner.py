"""Experiment runner: executes RunSpecs one after the other, one result file per run (replayable).

A run whose result file already exists is skipped, so a crashed batch is resumed by launching the
same command again. Outputs live under `<task dir>/research_a/runs/` (not versioned: data/).

Usage (from the repo root, see the report for the exact commands):

    python -m hackathon.unlearning.research.runner --group sweep_gold_soft
    python -m hackathon.unlearning.research.runner --spec-json my_specs.json
    python -m hackathon.unlearning.research.runner --group candidates --save-candidates
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from hackathon.common.config import load_settings
from hackathon.unlearning.constants import default_task_dir
from hackathon.unlearning.model import predict, select_device
from hackathon.unlearning.research.context import (
    ResearchContext,
    build_context,
    evaluate,
    hypothesis_statistics,
)
from hackathon.unlearning.research.experiments import GROUPS
from hackathon.unlearning.research.trainer import (
    TRAIN_SUBSET_ALL,
    RunSpec,
    calibrated_targets,
    precompute_view_cache,
    spec_to_dict,
    train,
)
from hackathon.unlearning.submission import save_submission

LOGGER = logging.getLogger("hackathon.unlearning.research")
RESEARCH_SUBDIR = "research_a"
RUNS_SUBDIR = "runs"
CANDIDATES_SUBDIR = "candidates"
REFERENCE_NAME = "_reference"
THREAD_LIMIT = 2
"""Shared machine: cap the intra-op CPU threads (coordinator rule)."""
VIEW_CACHE_BATCH = 32
EXACT_LOGITS_CACHE = "exact_logits_all.pt"


def blend_state(
    original_state: dict[str, Tensor], trained_state: dict[str, Tensor], alpha: float
) -> dict[str, Tensor]:
    """theta_original + alpha * (theta_trained - theta_original) for every floating tensor."""
    return {
        key: (
            original_state[key] + alpha * (value - original_state[key])
            if value.is_floating_point()
            else value
        )
        for key, value in trained_state.items()
    }


def interpolation_curve(
    context: ResearchContext,
    student: nn.Module,
    alphas: tuple[float, ...],
    name: str,
    candidates_dir: Path | None,
) -> list[dict[str, object]]:
    """Full evaluation of the weight interpolation between the original and `student`."""
    original_state = {k: v.detach().clone() for k, v in context.original.state_dict().items()}
    trained_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
    scratch = copy.deepcopy(student).eval()
    curve: list[dict[str, object]] = []
    for alpha in alphas:
        scratch.load_state_dict(blend_state(original_state, trained_state, alpha))
        result = evaluate(context, scratch, full=True)
        entry: dict[str, object] = {"alpha": alpha, "metrics": result.metrics}
        if candidates_dir is not None:
            path = candidates_dir / f"{name}_alpha{alpha:.2f}.pt"
            entry["candidate"] = str(path)
            save_submission(scratch, path)
        curve.append(entry)
        LOGGER.info("alpha %.2f done", alpha)
    return curve


def _load_view_cache(
    context: ResearchContext, spec: RunSpec, forget: Tensor, research_dir: Path
) -> Tensor:
    """Per-image teacher view distributions, computed once and shared by runs with equal settings."""
    path = research_dir / f"view_cache_a{spec.target_area}_v{spec.target_views}_n{len(forget)}.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
    started = time.time()
    cache = precompute_view_cache(context, spec, forget, VIEW_CACHE_BATCH)
    torch.save(cache, path)
    LOGGER.info("view cache %s built in %.0f s", path.name, time.time() - started)
    return cache


def _all_exact_logits(context: ResearchContext, research_dir: Path) -> Tensor:
    """Original logits on the 3500 exact forget images, computed once and cached on disk."""
    path = research_dir / EXACT_LOGITS_CACHE
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
    logits = predict(context.original, context.forget_all, VIEW_CACHE_BATCH)
    torch.save(logits, path)
    return logits


def run_one(
    context: ResearchContext, spec: RunSpec, runs_dir: Path, candidates_dir: Path | None
) -> None:
    """Train, evaluate and persist one run (skipped when its result file exists)."""
    result_path = runs_dir / f"{spec.name}.json"
    if result_path.is_file():
        LOGGER.info("skip %s (result exists)", spec.name)
        return
    if spec.init_from and not Path(spec.init_from).is_absolute():
        task_dir = runs_dir.parent.parent
        spec = replace(spec, init_from=str(task_dir / spec.init_from))
    LOGGER.info("run %s: %s", spec.name, spec_to_dict(spec))
    forget = context.forget_all if spec.train_on == TRAIN_SUBSET_ALL else context.forget_a
    view_cache = (
        _load_view_cache(context, spec, forget, runs_dir.parent) if spec.cache_views else None
    )
    target_cache: Tensor | None = None
    if spec.method == "calibrated":
        if view_cache is None or spec.train_on != TRAIN_SUBSET_ALL:
            raise ValueError("calibrated needs cache_views=True and train_on='all'")
        target_cache = calibrated_targets(
            view_cache, _all_exact_logits(context, runs_dir.parent), spec
        )
    outcome = train(context, spec, context.retain_train, forget, view_cache, target_cache)
    started = time.time()
    result = evaluate(context, outcome.student, full=True)
    payload: dict[str, object] = {
        "spec": spec_to_dict(spec),
        "train_seconds": outcome.seconds,
        "eval_seconds": time.time() - started,
        "history": outcome.history,
        "trajectory": outcome.trajectory,
        "metrics": result.metrics,
    }
    if spec.alphas:
        payload["alpha_curve"] = interpolation_curve(
            context, outcome.student, spec.alphas, spec.name, candidates_dir
        )
    if candidates_dir is not None and not spec.alphas:
        path = candidates_dir / f"{spec.name}.pt"
        report = save_submission(outcome.student, path)
        payload["candidate"] = {"path": str(path), "size_bytes": report.size_bytes}
    runs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.arrays, runs_dir / f"{spec.name}.arrays.pt")
    result_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    LOGGER.info("done %s in %.0f s", spec.name, outcome.seconds)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(GROUPS), help="named experiment group")
    parser.add_argument("--spec-json", type=Path, help="JSON list of RunSpec fields")
    parser.add_argument("--only", nargs="*", help="run only these spec names")
    parser.add_argument("--save-candidates", action="store_true", help="write validated .pt files")
    parser.add_argument(
        "--proxy-root", type=Path, default=None,
        help="folder holding coco_val_subset / div2k_valid / kodak (default: parent of the data dir)",
    )  # fmt: skip
    parser.add_argument(
        "--candidates-subdir", default=CANDIDATES_SUBDIR, help="under the task dir (phase 2: candidates/phase2)"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(THREAD_LIMIT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    specs: list[RunSpec] = []
    if args.group:
        specs.extend(GROUPS[args.group])
    if args.spec_json:
        for entry in json.loads(args.spec_json.read_text(encoding="utf-8")):
            entry["alphas"] = tuple(entry.get("alphas", ()))
            specs.append(RunSpec(**entry))
    if args.only:
        specs = [spec for spec in specs if spec.name in args.only]
    settings = load_settings()
    task_dir = default_task_dir(settings.data_dir)
    research_dir = task_dir / RESEARCH_SUBDIR
    proxy_root = args.proxy_root if args.proxy_root else settings.data_dir.parent
    context = build_context(task_dir, proxy_root, select_device(None), research_dir)
    reference = research_dir / RUNS_SUBDIR / f"{REFERENCE_NAME}.json"
    if not reference.is_file():
        reference.parent.mkdir(parents=True, exist_ok=True)
        identity = evaluate(context, context.original, full=True)
        reference.write_text(
            json.dumps(
                {
                    "identity": identity.metrics,
                    "hypotheses": hypothesis_statistics(context),
                    "labels_a": context.labels_a.tolist(),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
    candidates_dir = task_dir / args.candidates_subdir if args.save_candidates else None
    for spec in specs:
        run_one(context, spec, research_dir / RUNS_SUBDIR, candidates_dir)


if __name__ == "__main__":
    main()
