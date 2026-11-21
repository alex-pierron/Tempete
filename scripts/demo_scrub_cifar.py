"""Unlearn a CIFAR-10 forget set with SCRUB / SCRUB+R and compare with the baselines.

For each scenario (whole-class forgetting, selective forgetting of 100 images) the script
1. trains the original ResNet-18 once (cached under `weights/unlearn/`),
2. runs SCRUB (one run gives both bare SCRUB and SCRUB+R: same trajectory, rewound or not),
   `finetune_retain`, `negative_gradient`, `neggrad_plus` and `random_labels`,
3. prints, per method, the forget error, retain / test accuracy, MIA AUC and accuracy, the error
   on a same-distribution validation set and its gap to the forget error, and the wall time.

Run it through the machine's heavy-job semaphore from the repository root::

    jarvis-heavy env PYTHONPATH=$PWD/src WMKIT_DATA_DIR=<data dir> \\
        <venv>/bin/python scripts/demo_scrub_cifar.py [--seeds 0,1,2] [--subset-per-class 2000]

Default is a 2000-image-per-class subset (20 000 training images) because a full ResNet-18 epoch
takes ~90 s on an Apple GPU; pass `--subset-per-class 0` for the full 50 000 images.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from wmkit.config import get_device, results_dir
from wmkit.unlearn import (
    BaselineConfig,
    ScrubConfig,
    UnlearnLoaders,
    build_loaders,
    evaluate_unlearning,
    finetune_retain,
    load_cifar10,
    load_or_train_original,
    make_split,
    negative_gradient,
    neggrad_plus,
    random_labels,
    scrub,
    seed_everything,
)
from wmkit.unlearn.data import (
    DEFAULT_FORGET_CLASS,
    ORIGINAL_EPOCHS,
    SELECTIVE_FORGET_SIZE,
)
from wmkit.unlearn.scrub import classification_error

DEFAULT_SUBSET_PER_CLASS = 2000
DEFAULT_FINETUNE_EPOCHS = 5
SELECTIVE_MSTEPS = 5
SELECTIVE_EPOCHS = 5
ALL_SCENARIOS = ("class", "selective")
BASELINE_METHODS = ("finetune_retain", "negative_gradient", "neggrad_plus", "random_labels")
ALL_METHODS = ("original", *BASELINE_METHODS, "scrub", "scrub_r")
PERCENT = 100.0
TABLE_COLUMNS = [
    "method",
    "forget_err %",
    "val_err %",
    "gap %",
    "retain_acc %",
    "test_acc %",
    "mia_auc",
    "mia_acc %",
    "seconds",
]


@dataclass(frozen=True)
class ResultRow:
    """One (scenario, seed, method) measurement, all rates in `[0, 1]`."""

    scenario: str
    seed: int
    method: str
    forget_err: float
    val_err: float
    gap: float
    retain_acc: float
    test_acc: float
    mia_auc: float
    mia_acc: float
    seconds: float


class ResultStore:
    """Rows measured so far, rewritten atomically to JSON after every `add`.

    A demo that dies (or is stopped) after an hour keeps every row measured before that point.
    """

    def __init__(self, path: Path) -> None:
        self.rows: list[ResultRow] = []
        self._path = path

    def add(self, row: ResultRow) -> None:
        """Append `row` and persist all rows (write to a temp file, then `os.replace`)."""
        self.rows.append(row)
        temporary = self._path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps([asdict(r) for r in self.rows], indent=2))
        os.replace(temporary, self._path)


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", default=",".join(ALL_SCENARIOS))
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--seeds", default="0", help="comma-separated; each draws a new forget set")
    parser.add_argument("--subset-per-class", type=int, default=DEFAULT_SUBSET_PER_CLASS)
    parser.add_argument("--original-epochs", type=int, default=ORIGINAL_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--forget-class", type=int, default=DEFAULT_FORGET_CLASS)
    parser.add_argument("--n-forget", type=int, default=SELECTIVE_FORGET_SIZE)
    parser.add_argument(
        "--output-name", default="unlearn_demo.json", help="file under results_dir()"
    )
    return parser.parse_args()


def say(message: str) -> None:
    """Print a progress line immediately (the job runs long and is usually piped)."""
    print(message, flush=True)


def measure(
    method: str,
    scenario: str,
    seed: int,
    model: nn.Module,
    loaders: UnlearnLoaders,
    device: torch.device,
    seconds: float,
) -> ResultRow:
    """Evaluate one model and pack the numbers into a row."""
    report = evaluate_unlearning(
        model, loaders.retain, loaders.forget, loaders.test, device, seed=seed
    )
    val_err = classification_error(model, loaders.val, device)
    return ResultRow(
        scenario=scenario,
        seed=seed,
        method=method,
        forget_err=report.forget_error,
        val_err=val_err,
        gap=abs(report.forget_error - val_err),
        retain_acc=report.retain_acc,
        test_acc=report.test_acc,
        mia_auc=report.mia_auc,
        mia_acc=report.mia_accuracy,
        seconds=seconds,
    )


def scrub_config_for(scenario: str) -> ScrubConfig:
    """Paper settings: class (512, 128, 2, 3) is the default config; selective (16, 64, 5, 5)."""
    if scenario == "class":
        return ScrubConfig()
    return ScrubConfig(msteps=SELECTIVE_MSTEPS, epochs=SELECTIVE_EPOCHS)


def run_scenario(
    scenario: str,
    seed: int,
    methods: tuple[str, ...],
    original: nn.Module,
    loaders: UnlearnLoaders,
    device: torch.device,
    finetune_epochs: int,
    store: ResultStore,
) -> None:
    """Run every requested method on one (scenario, seed), recording each row as it lands."""
    if "original" in methods:
        store.add(measure("original", scenario, seed, original, loaders, device, 0.0))
        say(f"[{scenario} s{seed}] original measured")

    if "scrub" in methods or "scrub_r" in methods:
        snapshots: dict[int, dict[str, torch.Tensor]] = {}
        config = scrub_config_for(scenario)

        def keep_last_epoch(epoch: int, student: nn.Module) -> None:
            if epoch == config.epochs:
                snapshots[epoch] = {k: v.detach().clone() for k, v in student.state_dict().items()}

        seed_everything(seed)
        started = time.perf_counter()
        rewound, history = scrub(
            copy.deepcopy(original), original, loaders.retain, loaders.forget, config, device,
            rewind_val_loader=loaders.val, epoch_callback=keep_last_epoch,
        )  # fmt: skip
        seconds = time.perf_counter() - started
        say(f"[{scenario} s{seed}] scrub done in {seconds:.0f}s; rewind -> {history[-1]}")
        if "scrub" in methods:
            bare = copy.deepcopy(original)
            bare.load_state_dict(snapshots[config.epochs])
            store.add(measure("scrub", scenario, seed, bare.to(device), loaders, device, seconds))
        if "scrub_r" in methods:
            store.add(measure("scrub_r", scenario, seed, rewound, loaders, device, seconds))

    baselines: dict[str, Callable[..., tuple[nn.Module, list[dict[str, float]]]]] = {
        "finetune_retain": finetune_retain,
        "negative_gradient": negative_gradient,
        "neggrad_plus": neggrad_plus,
        "random_labels": random_labels,
    }
    for name in BASELINE_METHODS:
        if name not in methods:
            continue
        baseline_config = BaselineConfig.for_method(name)  # type: ignore[arg-type]
        if name == "finetune_retain":
            baseline_config = baseline_config.model_copy(update={"epochs": finetune_epochs})
        seed_everything(seed)
        started = time.perf_counter()
        model, _ = baselines[name](
            original, loaders.retain, loaders.forget, baseline_config, device
        )
        seconds = time.perf_counter() - started
        store.add(measure(name, scenario, seed, model, loaders, device, seconds))
        say(f"[{scenario} s{seed}] {name} done in {seconds:.0f}s")


def format_cell(
    group: list[ResultRow], attribute: str, scale: float = PERCENT, digits: int = 1
) -> str:
    """Mean of `attribute` over the seeds of `group` (scaled), with +/- std when several seeds."""
    values = [getattr(row, attribute) * scale for row in group]
    mean = statistics.fmean(values)
    if len(values) > 1:
        return f"{mean:.{digits}f} +/- {statistics.stdev(values):.{digits}f}"
    return f"{mean:.{digits}f}"


def format_scenario_table(rows: list[ResultRow], scenario: str) -> str:
    """Markdown table of one scenario: mean over seeds (and +/- std when several seeds)."""
    grouped: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        if row.scenario == scenario:
            grouped[row.method].append(row)
    records: list[dict[str, str]] = []
    for method in ALL_METHODS:
        if method not in grouped:
            continue
        group = grouped[method]

        records.append(
            {
                "method": method,
                "forget_err %": format_cell(group, "forget_err"),
                "val_err %": format_cell(group, "val_err"),
                "gap %": format_cell(group, "gap"),
                "retain_acc %": format_cell(group, "retain_acc"),
                "test_acc %": format_cell(group, "test_acc"),
                "mia_auc": format_cell(group, "mia_auc", 1.0, 3),
                "mia_acc %": format_cell(group, "mia_acc"),
                "seconds": format_cell(group, "seconds", 1.0, 0),
            }
        )
    return str(pd.DataFrame(records, columns=TABLE_COLUMNS).to_markdown(index=False))


def main() -> None:
    """Entry point: train/load the original model, run every method, print the tables."""
    args = parse_arguments()
    scenarios = tuple(args.scenarios.split(","))
    methods = tuple(args.methods.split(","))
    seeds = [int(value) for value in args.seeds.split(",")]
    device = get_device()
    say(f"device: {device} (Apple GPU via MPS)" if device.type == "mps" else f"device: {device}")

    subset = args.subset_per_class or None
    arrays = load_cifar10(subset_per_class=subset)
    say(f"train images: {len(arrays.train_y)}  test images: {len(arrays.test_y)}")
    original, summary = load_or_train_original(arrays, device, epochs=args.original_epochs, log=say)
    say(
        f"original ResNet-18: {'loaded from cache' if summary.cached else 'trained'} "
        f"(training time {summary.seconds:.0f}s, {summary.epochs} epochs), "
        f"train acc {summary.train_acc * PERCENT:.1f}%, test acc {summary.test_acc * PERCENT:.1f}%"
    )

    output = results_dir() / args.output_name
    store = ResultStore(output)
    for scenario in scenarios:
        for seed in seeds:
            split = make_split(
                arrays,
                scenario=scenario,
                forget_class=args.forget_class,
                n_forget=args.n_forget,
                seed=seed,
            )
            loaders = build_loaders(arrays, split, device=device, seed=seed)
            say(
                f"[{scenario} s{seed}] retain={len(split.retain_idx)} forget={len(split.forget_idx)} "
                f"val={len(split.val_idx)} test={len(split.test_idx)}"
            )
            run_scenario(
                scenario, seed, methods, original, loaders, device, args.finetune_epochs, store
            )
    for scenario in scenarios:
        title = (
            f"class {args.forget_class} entirely"
            if scenario == "class"
            else f"{args.n_forget} images of class {args.forget_class}"
        )
        print(f"\n### Scenario {scenario}: forget {title} (seeds {seeds})\n")
        print(format_scenario_table(store.rows, scenario))
    print(f"\nraw rows: {output}")
    print(
        "val_err = error of the same model on held-out images of the forget class (the SCRUB+R "
        "reference); gap = |forget_err - val_err|; mia ideal = 0.5."
    )


if __name__ == "__main__":
    main()
