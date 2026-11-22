"""Calibration of the small-N statistic on the `ref-*` endpoints (phase 2, team m-met-1).

For each N, random sets of N prompts (one random draw of each) are scored on the 9 ref endpoints
with the practice truth (5 disguised copies of one model, plus fp8, LoRA 0.3, 0.7, 1.5 as four
distinct models: 10 same-model pairs out of 36). The model cut `tau_model` is chosen on one half
of the PROMPTS and the MCC measured on the other half, so the figure is an honest estimate of
what a new prompt set would give (a threshold fitted and scored on the same prompts is optimistic).

The family cut cannot be measured on ref (one base model); it is set from the height of the top
merge of the ref dendrogram: a merge higher than almost any same-family ref merge at the same N is
read as "another base model".

Run: `python -m hackathon.model_equality.research.small_n_eval --out DIR` (about 2 min, one
core per N).
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster

from hackathon.model_equality.scoring import confusion
from hackathon.model_equality.smalln import (
    COMPONENTS,
    SmallNData,
    build_small_n_data,
    build_tree,
)
from hackathon.model_equality.store import Answer, AnswerStore

REF_ENDPOINTS: tuple[str, ...] = tuple(f"ref-{index:02d}" for index in range(1, 10))
DISGUISED_COPY_INDICES: tuple[int, ...] = (0, 2, 4, 5, 7)
"""ref-01, 03, 05, 06, 08: plain, system prompt, persona, watermark, temperature 1.0."""
RANDOM_SEED = 20261121
SUBSETS_PER_HALF = 100
SPLITS = 3
SAMPLE_SIZES: tuple[int, ...] = (5, 10, 20, 30)
TAU_GRID: npt.NDArray[np.float64] = np.round(np.arange(0.04, 1.2, 0.02), 3)
FAMILY_QUANTILE = 0.95
EXCLUDED_PROMPT_MARK = "echo"
"""The `echo` family was refuted by team B: left out of the pool."""
METHOD_WEIGHTS: dict[str, dict[str, float]] = {
    "ham+len": {"ham": 1.0, "len": 1.0},
    "ham+spec+len": {"ham": 1.0, "spec": 1.0, "len": 1.0},
    "ham": {"ham": 1.0},
    "len": {"len": 1.0},
    "spec": {"spec": 1.0},
}
CHOSEN_METHOD = "ham+len"
REF_STORE_NAMES: tuple[str, ...] = (
    "reference_answers.jsonl", "reference_answers_b.jsonl", "reference_answers_m1.jsonl",
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationRow:
    """Held-out MCC and threshold of one (method, N)."""

    method: str
    n_prompts: int
    mcc_mean: float
    mcc_sd: float
    mcc_p10: float
    mcc_p90: float
    tau_model: float
    tau_model_sd: float
    tau_family: float


def load_ref_answers(directory: Path, names: Sequence[str] = REF_STORE_NAMES) -> list[Answer]:
    """Union of the ref stores present in `directory` (first occurrence of a key wins)."""
    merged: dict[tuple[str, str, int], Answer] = {}
    for name in names:
        path = directory / name
        if path.is_file():
            for answer in AnswerStore(path).load().values():
                if EXCLUDED_PROMPT_MARK not in answer.prompt_id:
                    merged.setdefault(answer.key, answer)
    return list(merged.values())


def ref_truth() -> tuple[list[tuple[int, int]], list[bool]]:
    """The 36 ref pairs and whether each is a same-model pair."""
    label = [0 if i in DISGUISED_COPY_INDICES else i + 1 for i in range(len(REF_ENDPOINTS))]
    pairs = list(itertools.combinations(range(len(REF_ENDPOINTS)), 2))
    return pairs, [label[i] == label[j] for i, j in pairs]


def _unit_distances(data: SmallNData, weights: Mapping[str, float]) -> FloatArray:
    """Per-unit weighted distance `[U, E, E]`."""
    total = sum(weights.values())
    return np.asarray(
        sum(weights.get(name, 0.0) * data.components[name] for name in COMPONENTS) / total,
        dtype=np.float64,
    )


def _mcc_curve(
    tree: FloatArray, pairs: Sequence[tuple[int, int]], truth: list[bool]
) -> FloatArray:
    """Model MCC of the tree cut at every threshold of `TAU_GRID`."""
    curve = np.empty(len(TAU_GRID))
    for position, tau in enumerate(TAU_GRID):
        labels = fcluster(tree, t=float(tau), criterion="distance")
        curve[position] = confusion(truth, [labels[i] == labels[j] for i, j in pairs]).mcc
    return curve


def _random_unit_sets(
    prompts: Sequence[str], positions_by_prompt: Mapping[str, list[int]], size: int, count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """`count` random sets of `size` prompts, one random draw (unit) of each."""
    sets: list[list[int]] = []
    for _ in range(count):
        chosen = rng.choice(len(prompts), size=size, replace=False)
        sets.append([int(rng.choice(positions_by_prompt[prompts[c]])) for c in chosen])
    return sets


def calibrate(data: SmallNData) -> list[CalibrationRow]:
    """Held-out MCC and thresholds of every method and sample size (see module docstring)."""
    rng = np.random.default_rng(RANDOM_SEED)
    pairs, truth = ref_truth()
    prompts = sorted({unit[0] for unit in data.units})
    positions_by_prompt = {
        prompt: [k for k, unit in enumerate(data.units) if unit[0] == prompt] for prompt in prompts
    }
    rows: list[CalibrationRow] = []
    for method, weights in METHOD_WEIGHTS.items():
        unit_distance = _unit_distances(data, weights)
        for size in SAMPLE_SIZES:
            held_out: list[FloatArray] = []
            taus: list[float] = []
            heights: list[float] = []
            for _ in range(SPLITS):
                order = rng.permutation(len(prompts))
                half = len(prompts) // 2
                calibration_prompts = [prompts[i] for i in order[:half]]
                test_prompts = [prompts[i] for i in order[half:2 * half]]
                if size > half:
                    continue
                calibration_curves = []
                for unit_set in _random_unit_sets(calibration_prompts, positions_by_prompt, size,
                                                  SUBSETS_PER_HALF, rng):
                    tree = build_tree(unit_distance[unit_set].mean(axis=0))
                    calibration_curves.append(_mcc_curve(tree, pairs, truth))
                best = int(np.argmax(np.mean(calibration_curves, axis=0)))
                test_curves = []
                for unit_set in _random_unit_sets(test_prompts, positions_by_prompt, size,
                                                  SUBSETS_PER_HALF, rng):
                    tree = build_tree(unit_distance[unit_set].mean(axis=0))
                    test_curves.append(_mcc_curve(tree, pairs, truth)[best])
                    heights.append(float(tree[-1, 2]))
                held_out.append(np.asarray(test_curves))
                taus.append(float(TAU_GRID[best]))
            if not held_out:
                continue
            values = np.concatenate(held_out)
            rows.append(CalibrationRow(
                method=method, n_prompts=size, mcc_mean=float(values.mean()),
                mcc_sd=float(values.std()), mcc_p10=float(np.percentile(values, 10)),
                mcc_p90=float(np.percentile(values, 90)), tau_model=float(np.mean(taus)),
                tau_model_sd=float(np.std(taus)),
                tau_family=float(np.quantile(heights, FAMILY_QUANTILE)),
            ))
    return rows


def main() -> None:
    """CLI: calibrate on the ref stores and write `calibration.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build_small_n_data(load_ref_answers(args.store_dir), list(REF_ENDPOINTS))
    rows = calibrate(data)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"units": len(data.units),
               "prompts": len({unit[0] for unit in data.units}),
               "rows": [asdict(row) for row in rows]}
    (args.out / "calibration.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for row in rows:
        print(f"{row.method:14s} N={row.n_prompts:2d} MCC {row.mcc_mean:+.3f} +- {row.mcc_sd:.3f} "
              f"[p10 {row.mcc_p10:+.2f}, p90 {row.mcc_p90:+.2f}] tau_model {row.tau_model:.2f} "
              f"tau_family {row.tau_family:.2f}")


if __name__ == "__main__":
    main()
