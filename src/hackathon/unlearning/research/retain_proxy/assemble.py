"""Turn generated chunks into the labelled retain-proxy dataset.

Selection rule: an image enters the proxy only if the ORIGINAL model recognises it confidently
(the real training images are recognised at 0.9997) and, depending on the labeling mode, agrees
with what the prompt was meant to produce. The label is always the original's own class, so the
proxy inherits the original's notion of every class, which is the only ground truth available.
Classes that the gold never saw (the nine fully forgotten ones) are excluded by default: a retain
set must not contain them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import IMAGE_SIZE, NUM_CLASSES
from hackathon.unlearning.research.retain_proxy.analyze import (
    concept_consensus,
    concept_statistics,
    load_logits,
    softmax,
)
from hackathon.unlearning.research.retain_proxy.concepts import FULLY_FORGOTTEN_CLASSES
from hackathon.unlearning.research.retain_proxy.generate import CHUNK_FILE_TEMPLATE
from hackathon.unlearning.research.retain_proxy.plan import PromptEntry, read_plan

DEFAULT_MIN_CONFIDENCE = 0.8
RUN_SEPARATOR = ":"
"""`--run <chunk_dir>:<plan.jsonl>`: paths on the cluster never contain a colon."""


class AssembleError(HackathonError):
    """No image passes the selection rule, or the inputs are inconsistent."""


class Labeling(StrEnum):
    """How strictly the original's answer must agree with what the prompt was meant to produce."""

    HINT = "hint"
    """Known-class images must get their intended class; probe images their concept's class."""
    CONSENSUS = "consensus"
    """The answer must be the class the original gives to most images of the same concept."""
    FREE = "free"
    """Any confident answer counts: the original's view of the image is the label."""


class SelectionRule(BaseModel):
    """Which generated images count as retain-proxy images."""

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(default=DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    excluded_classes: tuple[int, ...] = FULLY_FORGOTTEN_CLASSES
    per_class_cap: int | None = Field(default=None, ge=1)
    labeling: Labeling = Labeling.CONSENSUS


def select_samples(
    plan: Sequence[PromptEntry],
    indices: np.ndarray,
    logits: np.ndarray,
    rule: SelectionRule,
    consensus: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Positions (rows of `indices`/`logits`) and labels of the accepted images.

    The label is always the original's own answer; `rule.labeling` decides how much that answer
    must agree with the prompt: its intended class (HINT; probe images use `consensus`), the
    concept's consensus class (CONSENSUS), or nothing (FREE).
    """
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted: list[int] = []
    labels: list[int] = []
    per_class: Counter[int] = Counter()
    for position, plan_index in enumerate(indices):
        entry = plan[int(plan_index)]
        label = int(predicted[position])
        if confidence[position] < rule.min_confidence or label in rule.excluded_classes:
            continue
        if rule.labeling is Labeling.HINT and entry.class_hint is not None:
            if label != entry.class_hint:
                continue
        elif rule.labeling is not Labeling.FREE and consensus.get(entry.concept_key) != label:
            continue
        if rule.per_class_cap is not None and per_class[label] >= rule.per_class_cap:
            continue
        per_class[label] += 1
        accepted.append(position)
        labels.append(label)
    return np.array(accepted, dtype=np.int64), np.array(labels, dtype=np.int64)


def _collect_run(
    chunk_dir: Path, plan_path: Path, rule: SelectionRule
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(images, labels, orig_logits, generated count) of the accepted images of one run."""
    plan = read_plan(plan_path)
    indices, logits = load_logits(chunk_dir)
    consensus = concept_consensus(concept_statistics(plan, indices, logits))
    positions, labels = select_samples(plan, indices, logits, rule, consensus)
    wanted = {
        int(indices[position]): (int(label), position)
        for position, label in zip(positions, labels, strict=True)
    }
    row_of_plan_index = {plan_index: row for row, plan_index in enumerate(sorted(wanted))}
    images_out = np.empty((len(wanted), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels_out = np.empty(len(wanted), dtype=np.int64)
    logits_out = np.empty((len(wanted), NUM_CLASSES), dtype=np.float32)
    for path in sorted(chunk_dir.glob(CHUNK_FILE_TEMPLATE.replace("{first_index:07d}", "*"))):
        with np.load(path) as chunk:
            chunk_indices = chunk["indices"]
            keep = [i for i, plan_index in enumerate(chunk_indices) if int(plan_index) in wanted]
            if not keep:
                continue
            images = chunk["images"][keep]
            chunk_logits = chunk["logits"][keep]
        for local, i in enumerate(keep):
            plan_index = int(chunk_indices[i])
            row = row_of_plan_index[plan_index]
            images_out[row] = images[local]
            logits_out[row] = chunk_logits[local]
            labels_out[row] = wanted[plan_index][0]
    return images_out, labels_out, logits_out, len(indices)


def _global_cap(labels: np.ndarray, cap: int | None) -> np.ndarray:
    """Rows to keep so that no class has more than `cap` images (first come, first kept)."""
    if cap is None:
        return np.arange(len(labels))
    seen: Counter[int] = Counter()
    keep: list[int] = []
    for row, label in enumerate(labels.tolist()):
        if seen[label] < cap:
            seen[label] += 1
            keep.append(row)
    return np.array(keep, dtype=np.int64)


def assemble_dataset(
    runs: Sequence[tuple[Path, Path]], out_path: Path, rule: SelectionRule
) -> dict[str, object]:
    """Write `out_path` (npz: images, labels, orig_logits, run_ids) and return its summary.

    Args:
        runs: (chunk directory, plan file) of every generation run to draw from.
        out_path: destination `.npz`.
        rule: selection rule.

    Raises:
        AssembleError: no run given, or nothing passes the rule.
    """
    if not runs:
        raise AssembleError("At least one generation run is required.")
    images_parts, label_parts, logit_parts, run_parts = [], [], [], []
    generated = 0
    for run_id, (chunk_dir, plan_path) in enumerate(runs):
        images, labels, logits, count = _collect_run(chunk_dir, plan_path, rule)
        generated += count
        images_parts.append(images)
        label_parts.append(labels)
        logit_parts.append(logits)
        run_parts.append(np.full(len(labels), run_id, dtype=np.int64))
    labels_all = np.concatenate(label_parts)
    keep = _global_cap(labels_all, rule.per_class_cap)
    if len(keep) == 0:
        raise AssembleError(f"No image of {[str(c) for c, _ in runs]} passes {rule}.")
    images_all = np.concatenate(images_parts)[keep]
    labels_all = labels_all[keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            images=images_all,
            labels=labels_all,
            orig_logits=np.concatenate(logit_parts)[keep],
            run_ids=np.concatenate(run_parts)[keep],
        )
    temporary.replace(out_path)
    counts = Counter(int(label) for label in labels_all)
    return {
        "images": len(labels_all),
        "generated": generated,
        "acceptance_rate": float(len(labels_all) / generated),
        "classes": len(counts),
        "per_class_min": min(counts.values()),
        "per_class_max": max(counts.values()),
        "counts": {str(key): counts[key] for key in sorted(counts)},
        "rule": json.loads(rule.model_dump_json()),
        "runs": [str(chunk_dir) for chunk_dir, _ in runs],
    }


def parse_run(argument: str) -> tuple[Path, Path]:
    """`<chunk_dir>:<plan.jsonl>` -> (Path, Path).

    Raises:
        AssembleError: the separator is missing.
    """
    chunk_dir, separator, plan = argument.partition(RUN_SEPARATOR)
    if not separator or not plan:
        raise AssembleError(
            f"--run expects <chunk_dir>{RUN_SEPARATOR}<plan.jsonl>, got {argument!r}"
        )
    return Path(chunk_dir), Path(plan)


EVAL_SUBSET_COUNT = 1500
"""Images of the small evaluation file (~225 MB): enough for a stable KL, light enough to download."""
EVAL_SUBSET_SEED = 0
EVAL_SUBSET_SUFFIX = ".eval1500.npz"


def write_eval_subset(
    dataset_path: Path, out_path: Path, count: int = EVAL_SUBSET_COUNT, seed: int = EVAL_SUBSET_SEED
) -> int:
    """Write a seeded subset (`images`, `labels`) of an assembled dataset; returns its size.

    Lets a candidate that exists only on a laptop be scored with `proxy_eval --fraction 1.0`
    without copying the 1.4 GB dataset or the 344 MB candidate over a slow link.

    Raises:
        AssembleError: the dataset file is missing or empty.
    """
    if not dataset_path.is_file():
        raise AssembleError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path) as archive:
        images, labels = archive["images"], archive["labels"]
    if len(labels) == 0:
        raise AssembleError(f"Dataset is empty: {dataset_path}")
    kept = min(count, len(labels))
    order = np.random.default_rng(seed).permutation(len(labels))[:kept]
    order.sort()
    temporary = out_path.with_name(out_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, images=images[order], labels=labels[order])
    temporary.replace(out_path)
    return kept


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: assemble the dataset and write `<out>.summary.json` beside it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="CHUNK_DIR:PLAN")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--per-class-cap", type=int, default=None)
    parser.add_argument(
        "--labeling", type=Labeling, choices=list(Labeling), default=Labeling.CONSENSUS
    )
    parser.add_argument("--keep-forgotten-classes", action="store_true")
    args = parser.parse_args(argv)
    rule = SelectionRule(
        min_confidence=args.min_confidence,
        excluded_classes=() if args.keep_forgotten_classes else FULLY_FORGOTTEN_CLASSES,
        per_class_cap=args.per_class_cap,
        labeling=args.labeling,
    )
    summary = assemble_dataset([parse_run(run) for run in args.run], args.out, rule)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({key: value for key, value in summary.items() if key != "counts"}, indent=1))
    subset_path = args.out.with_name(args.out.stem + EVAL_SUBSET_SUFFIX)
    print(f"{subset_path}: {write_eval_subset(args.out, subset_path)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
