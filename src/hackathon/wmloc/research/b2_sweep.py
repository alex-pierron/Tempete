"""Hyper-parameter sweep of the head, one process per GPU, several seeds per configuration (team B2).

On one A100 a 5-fold CV of one method takes about a minute, so a node can score dozens of
configurations with several seeds each. The seeds matter more than the grid: the fold std of the
score is 0.016, and a single-seed difference of 0.005 between two configurations is noise.

Worker `k` of `W` scores the configurations whose index is `k` modulo `W` and appends one JSON line
per (configuration, seed, method) to `<results>/<tag>_w<k>.jsonl`. `summarise` merges the files and
ranks the configurations by their mean over seeds and methods.

Usage (kit venv):
    CUDA_VISIBLE_DEVICES=0 python -m hackathon.wmloc.research.b2_sweep run --worker 0 --workers 4 \
        --configs 24 --seeds 2 --tag sweep1
    python -m hackathon.wmloc.research.b2_sweep summarise --tag sweep1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.research.b2_model import SOURCES_BY_METHOD, HeadConfig
from hackathon.wmloc.research.b2_train import (
    MethodArrays,
    TrainConfig,
    cross_validate,
    load_arrays,
    release_device_data,
    results_dir,
)
from hackathon.wmloc.research.evaluate import DEFAULT_SEED

SWEEP_METHODS: tuple[str, ...] = ("pixelseal", "trustmark", "stegastamp")
BASE_CONFIG: dict[str, Any] = {
    "dropout": 0.5,
    "epochs": 20,
    "learning_rate": 3e-3,
    "weight_decay": 1e-2,
    "width": 48,
    "lovasz_weight": 0.0,
    "bce_weight": 0.5,
    "batch_size": 8,
}
"""The configuration of the submitted candidates: always scored, as the reference."""
SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "dropout": (0.3, 0.5, 0.7),
    "epochs": (12, 20, 30),
    "learning_rate": (1.5e-3, 3e-3, 6e-3),
    "weight_decay": (1e-2, 5e-2, 1e-1),
    "width": (32, 48, 64),
    "lovasz_weight": (0.0, 0.5, 1.0),
    "bce_weight": (0.0, 0.5, 1.0),
    "batch_size": (8, 16),
}
SYNTHETIC_SPACE: dict[str, tuple[Any, ...]] = {
    "synthetic_per_real": (2.0, 4.0, 8.0),
    "finetune_fraction": (0.15, 0.25, 0.4),
}
"""Extra axes when synthetic shards exist (Pixel Seal and TrustMark only)."""
DEVIATION_PROBABILITY = 0.4
"""Chance that a sampled configuration changes a given axis away from the base value."""
HEAD_KEYS = {"dropout", "width"}
SYNTHETIC_METHODS = {"pixelseal", "trustmark"}


def sample_configs(count: int, seed: int, with_synthetic: bool) -> list[dict[str, Any]]:
    """The base configuration followed by `count - 1` random neighbours of it (deterministic)."""
    rng = np.random.default_rng(seed)
    space = {**SEARCH_SPACE, **(SYNTHETIC_SPACE if with_synthetic else {})}
    configs: list[dict[str, Any]] = [dict(BASE_CONFIG, use_synthetic=False)]
    if with_synthetic:
        configs.append(dict(BASE_CONFIG, use_synthetic=True, synthetic_per_real=3.0))
    while len(configs) < count:
        config = dict(BASE_CONFIG, use_synthetic=with_synthetic and bool(rng.random() < 0.6))
        for key, values in space.items():
            if rng.random() < DEVIATION_PROBABILITY:
                config[key] = values[int(rng.integers(len(values)))]
        if config not in configs:
            configs.append(config)
    return configs[:count]


def split_config(config: dict[str, Any], method: str, seed: int) -> tuple[HeadConfig, TrainConfig]:
    """Build the head and training settings of one (configuration, method, seed)."""
    train_names = {field.name for field in fields(TrainConfig)}
    train_values = {k: v for k, v in config.items() if k in train_names}
    use_synthetic = bool(config.get("use_synthetic")) and method in SYNTHETIC_METHODS
    train_values.update(seed=DEFAULT_SEED + seed, use_synthetic=use_synthetic)
    head = HeadConfig(
        sources={method: tuple(SOURCES_BY_METHOD[method])},
        dropout=float(config["dropout"]),
        width=int(config["width"]),
    )
    return head, TrainConfig(**train_values)


def run_worker(
    worker: int,
    workers: int,
    configs: list[dict[str, Any]],
    seeds: int,
    tag: str,
    methods: tuple[str, ...] = SWEEP_METHODS,
) -> None:
    """Score this worker's share of the configurations; results are appended after every CV."""
    torch.set_num_threads(2)
    out = results_dir() / f"{tag}_w{worker}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, bool], dict[str, MethodArrays]] = {}
    current: tuple[str, bool] | None = None
    for index, config in enumerate(configs):
        if index % workers != worker:
            continue
        for seed in range(seeds):
            for method in methods:
                head, train = split_config(config, method, seed)
                key = (method, train.use_synthetic)
                if current is not None and current != key:
                    release_device_data(cache[current])
                current = key
                if key not in cache:
                    cache[key] = {
                        method: load_arrays("train", method, synthetic=train.use_synthetic)
                    }
                started = time.time()
                _, reports, _ = cross_validate(cache[key], head, train)
                report = reports[method]
                row = {
                    "config_id": index,
                    "config": config,
                    "seed": seed,
                    "method": method,
                    "all": report["all"],
                    "full": report["full"],
                    "rect": report["rect"],
                    "seconds": time.time() - started,
                }
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(f"config {index} seed {seed} {method}: {report['all']:.4f}", flush=True)


def summarise(tag: str, top: int = 10) -> list[dict[str, Any]]:
    """Rank configurations by mean score over seeds, then methods; prints the leaders."""
    scores: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    configs: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(results_dir()).glob(f"{tag}_w*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            scores[row["config_id"]][row["method"]].append(float(row["all"]))
            configs[row["config_id"]] = row["config"]
    ranked: list[dict[str, Any]] = []
    for config_id, by_method in scores.items():
        means = {m: float(np.mean(v)) for m, v in by_method.items()}
        ranked.append(
            {
                "config_id": config_id,
                "mean": float(np.mean(list(means.values()))),
                "by_method": means,
                "n_runs": sum(len(v) for v in by_method.values()),
                "config": configs[config_id],
            }
        )
    ranked.sort(key=lambda item: -item["mean"])
    for item in ranked[:top]:
        print(
            f"#{item['config_id']:3d} mean {item['mean']:.4f} "
            + " ".join(f"{m[:2]} {v:.4f}" for m, v in item["by_method"].items())
            + f" runs {item['n_runs']} {json.dumps(item['config'], sort_keys=True)}"
        )
    return ranked


CLI_FLAGS: dict[str, str] = {
    "dropout": "--dropout",
    "epochs": "--epochs",
    "learning_rate": "--lr",
    "weight_decay": "--weight-decay",
    "width": "--width",
    "lovasz_weight": "--lovasz",
    "bce_weight": "--bce-weight",
    "batch_size": "--batch-size",
    "synthetic_per_real": "--synthetic-per-real",
    "finetune_fraction": "--finetune-fraction",
}


def config_to_args(config: dict[str, Any], method: str) -> str:
    """The `b2_train` command-line flags of one swept configuration, for the given method."""
    flags = [f"{CLI_FLAGS[key]} {config[key]}" for key in CLI_FLAGS if key in config]
    if config.get("use_synthetic") and method in SYNTHETIC_METHODS:
        flags.append("--synthetic")
    return " ".join(flags)


def best_args(tag: str, method: str) -> str:
    """Flags of the configuration with the best mean score of `method` (ties: lowest id)."""
    best: tuple[float, int, dict[str, Any]] | None = None
    scores: dict[int, list[float]] = defaultdict(list)
    configs: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(results_dir()).glob(f"{tag}_w*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["method"] == method:
                scores[row["config_id"]].append(float(row["all"]))
                configs[row["config_id"]] = row["config"]
    for config_id, values in scores.items():
        candidate = (float(np.mean(values)), -config_id, configs[config_id])
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError(f"no sweep result for {method} under tag {tag!r}")
    return config_to_args(best[2], method)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--worker", type=int, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--configs", type=int, default=24)
    run.add_argument("--seeds", type=int, default=2)
    run.add_argument("--sample-seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--synthetic", action="store_true", help="include synthetic-data axes")
    run.add_argument("--methods", nargs="+", default=list(SWEEP_METHODS))
    run.add_argument("--tag", required=True)
    summary = sub.add_parser("summarise")
    summary.add_argument("--tag", required=True)
    best = sub.add_parser("best-args", help="print the b2_train flags of the best configuration")
    best.add_argument("--tag", required=True)
    best.add_argument("--method", required=True)
    args = parser.parse_args()
    if args.command == "run":
        configs = sample_configs(args.configs, args.sample_seed, args.synthetic)
        run_worker(args.worker, args.workers, configs, args.seeds, args.tag, tuple(args.methods))
    elif args.command == "best-args":
        print(best_args(args.tag, args.method))
    else:
        summarise(args.tag)


if __name__ == "__main__":
    main()
