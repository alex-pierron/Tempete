"""Phase-1 analysis of team B: group / prompt discrimination, repeats vs diversity, budget curves.

`run_analysis` reads the `ref-*` answers of team B (and, read only, those of team A when given),
and returns a JSON-serialisable dictionary that `report_b.py` turns into figures and HTML. All
resampling uses constant seeds, so the same answers always give the same numbers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.pairwise import (
    KERNEL_HAMMING_CHAR,
    KERNEL_LENGTH,
    KERNEL_ONEHOT_FIRST,
    KERNEL_ONEHOT_KEY,
    KERNEL_SPECTRUM,
)
from hackathon.model_equality.research.budget_curve import (
    budget_curve,
    build_windows,
    evaluate_design,
)
from hackathon.model_equality.research.discrimination import (
    REF_ENDPOINTS,
    AgreementTensor,
    EndpointTruth,
    agreement_tensor,
    endpoint_sets,
    group_answers,
    group_label,
    pair_auc,
    pair_sets,
    pool_distance,
    pooled_pair_distance,
    prompt_scores,
    split_half_reliability,
)
from hackathon.model_equality.store import Answer, load_many

FloatArray = npt.NDArray[np.float64]
KERNELS_EXPLORED: tuple[str, ...] = (
    KERNEL_ONEHOT_KEY, KERNEL_HAMMING_CHAR, KERNEL_SPECTRUM, KERNEL_LENGTH, KERNEL_ONEHOT_FIRST,
)
BOOTSTRAP_DRAWS = 300
BOOTSTRAP_SEED = 20261122
CI_LOW = 0.05
CI_HIGH = 0.95
MIN_PROMPTS_FOR_GROUP = 2
CURVE_BUDGETS: tuple[int, ...] = (
    4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 40, 48, 60, 72, 84, 96, 120)
CURVE_KERNELS: tuple[str, ...] = (KERNEL_HAMMING_CHAR, KERNEL_SPECTRUM, KERNEL_ONEHOT_KEY)
MAX_REPEATS = 3
"""Draws per prompt used in the curves (windows of consecutive repeats 0..3)."""
HEADLINE_KERNEL = KERNEL_HAMMING_CHAR
"""Kernel of the budget curves: the only one with a perfect easy separation on the first data."""


def load_reference_answers(paths: Sequence[Path]) -> list[Answer]:
    """Concatenate stores, first file winning on duplicate keys (see `store.load_many`)."""
    return load_many(paths)


def _bootstrap_group_auc(
    tensor: AgreementTensor, rows: Sequence[int], truth: Mapping[str, EndpointTruth], role: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """AUC of a prompt group with a 5-95 % interval from resampling its prompts."""
    sets = endpoint_sets(truth, tensor.endpoints)
    pairs = pair_sets(sets)
    point = pair_auc(pooled_pair_distance(tensor, rows), pairs[role], pairs["ss"])
    draws = np.empty(BOOTSTRAP_DRAWS)
    for index in range(BOOTSTRAP_DRAWS):
        sample = [int(i) for i in rng.choice(np.asarray(rows), size=len(rows), replace=True)]
        draws[index] = pair_auc(pooled_pair_distance(tensor, sample), pairs[role], pairs["ss"])
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(finite, CI_LOW)), float(np.quantile(finite, CI_HIGH))


def group_table(
    tensors: Mapping[str, AgreementTensor], truth: Mapping[str, EndpointTruth]
) -> list[dict[str, Any]]:
    """One row per (group, kernel): AUC hard / easy with intervals and pool distances per endpoint."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows_out: list[dict[str, Any]] = []
    for kernel, tensor in tensors.items():
        by_group: dict[str, list[int]] = defaultdict(list)
        for index, prompt_id in enumerate(tensor.prompts):
            by_group[group_label(prompt_id)].append(index)
        pooled = pool_distance(tensor, truth)
        for group, rows in sorted(by_group.items()):
            if len(rows) < MIN_PROMPTS_FOR_GROUP:
                continue
            hard = _bootstrap_group_auc(tensor, rows, truth, "hard", rng)
            easy = _bootstrap_group_auc(tensor, rows, truth, "easy", rng)
            with np.errstate(all="ignore"):
                per_endpoint = np.nanmean(pooled[rows], axis=0)
            rows_out.append({
                "group": group, "kernel": kernel, "prompts": len(rows),
                "auc_hard": hard[0], "auc_hard_low": hard[1], "auc_hard_high": hard[2],
                "auc_easy": easy[0], "auc_easy_low": easy[1], "auc_easy_high": easy[2],
                "pool_distance": {e: float(v) for e, v in zip(tensor.endpoints, per_endpoint, strict=True)},
            })
    return rows_out


def prompt_table(
    tensors: Mapping[str, AgreementTensor], truth: Mapping[str, EndpointTruth]
) -> list[dict[str, Any]]:
    """One row per prompt: score per kernel, plus the best kernel (selection bias is flagged later)."""
    prompts = next(iter(tensors.values())).prompts
    scores = {kernel: prompt_scores(tensor, truth) for kernel, tensor in tensors.items()}
    pooled = {kernel: pool_distance(tensor, truth) for kernel, tensor in tensors.items()}
    out: list[dict[str, Any]] = []
    for index, prompt_id in enumerate(prompts):
        per_kernel = {
            kernel: {
                "hard": float(scores[kernel]["hard"][index]),
                "easy": float(scores[kernel]["easy"][index]),
                "excess_hard": float(scores[kernel]["excess_hard"][index]),
                "excess_easy": float(scores[kernel]["excess_easy"][index]),
                "disguise_leak": float(scores[kernel]["disguise_leak"][index]),
                "pool_distance": {
                    e: float(v) for e, v in zip(tensors[kernel].endpoints,
                                                pooled[kernel][index], strict=True)
                },
            }
            for kernel in tensors
        }
        out.append({"prompt": prompt_id, "group": group_label(prompt_id), "kernels": per_kernel})
    return out


def repeat_vs_diversity(
    answers: Sequence[Answer], prompts: Sequence[str], endpoints: Sequence[str],
    truth: Mapping[str, EndpointTruth], kernels: Sequence[str], budgets: Sequence[int],
    max_repeats: int,
) -> dict[str, list[dict[str, Any]]]:
    """Curves of the designs `N / r prompts x r draws` at equal budgets, per role and statistic."""
    out: dict[str, list[dict[str, Any]]] = {}
    for kernel in kernels:
        windows = build_windows(answers, prompts, endpoints, kernel, max_repeats)
        for role in ("hard", "easy"):
            for repeats in range(1, max_repeats + 1):
                for statistic in ("pair", "pool"):
                    curve = budget_curve(windows, truth, budgets, repeats, role,
                                         statistic=statistic)
                    out[f"{kernel}/{role}/r{repeats}/{statistic}"] = [asdict(p) for p in curve]
    return out


def agreement_summary(
    tensors: Mapping[str, AgreementTensor], truth: Mapping[str, EndpointTruth]
) -> dict[str, dict[str, float]]:
    """Mean agreement (kernel value) per kernel between groups of endpoints, prompts weighted equally.

    Keys: `within_S` (two draws of the same S endpoint), `S_S` (two different S endpoints, the
    disguise noise), `S_T1` .. `S_T4` (an S endpoint against each T model). The gap between `S_S`
    and `S_T1` / `S_T2` is the effect size that a test must detect; `within_S - S_S` is what the
    disguises cost.
    """
    out: dict[str, dict[str, float]] = {}
    for kernel, tensor in tensors.items():
        model = [truth[e].model for e in tensor.endpoints]
        base = [i for i, m in enumerate(model) if m == "S"]
        values = tensor.values
        rows: dict[str, list[float]] = {"within_S": [], "S_S": []}
        for name in ("T1", "T2", "T3", "T4"):
            rows[f"S_{name}"] = []
        for p_index in range(len(tensor.prompts)):
            block = values[p_index]
            within = [block[i, i] for i in base if np.isfinite(block[i, i])]
            cross = [block[i, j] for i in base for j in base if i < j and np.isfinite(block[i, j])]
            if within:
                rows["within_S"].append(float(np.mean(within)))
            if cross:
                rows["S_S"].append(float(np.mean(cross)))
            for name in ("T1", "T2", "T3", "T4"):
                targets = [k for k, m in enumerate(model) if m == name]
                mixed = [block[i, k] for i in base for k in targets if np.isfinite(block[i, k])]
                if mixed:
                    rows[f"S_{name}"].append(float(np.mean(mixed)))
        out[kernel] = {name: (float(np.mean(v)) if v else float("nan")) for name, v in rows.items()}
        out[kernel]["prompts_within"] = float(len(rows["within_S"]))
    return out


def top_prompts(per_prompt: Sequence[Mapping[str, Any]], kernel_for: Mapping[str, str],
                limit: int) -> list[dict[str, Any]]:
    """Prompts ranked by (mean T excess over S) minus the largest disguise excess.

    The score uses the kernel that suits the prompt family (`kernel_for[group]`). It is a raw,
    single-pool ranking: the report states its (low) split-half reliability next to it.
    """
    rows: list[dict[str, Any]] = []
    for entry in per_prompt:
        kernel = kernel_for.get(entry["group"], HEADLINE_KERNEL)
        stats = entry["kernels"][kernel]
        parts = [stats["excess_hard"], stats["excess_easy"]]
        finite = [x for x in parts if not math.isnan(x)]
        if not finite or math.isnan(stats["disguise_leak"]):
            continue
        rows.append({
            "prompt": entry["prompt"], "group": entry["group"], "kernel": kernel,
            "excess_hard": stats["excess_hard"], "excess_easy": stats["excess_easy"],
            "disguise_leak": stats["disguise_leak"],
            "score": float(np.mean(finite)) - stats["disguise_leak"],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def run_analysis(
    answers: Sequence[Answer],
    truth: Mapping[str, EndpointTruth],
    endpoints: Sequence[str] = REF_ENDPOINTS,
) -> dict[str, Any]:
    """Every metric of the report, as plain data."""
    grouped = group_answers(answers, endpoints)
    prompts = sorted(grouped)
    tensors = {k: agreement_tensor(grouped, prompts, endpoints, k) for k in KERNELS_EXPLORED}
    reliability = {
        kernel: dict(zip(("spearman_r0_r1", "prompts_used"),
                         split_half_reliability(grouped, prompts, endpoints, truth, kernel),
                         strict=True))
        for kernel in KERNELS_EXPLORED
    }
    answers_per_endpoint: dict[str, int] = defaultdict(int)
    for answer in answers:
        if answer.endpoint in endpoints:
            answers_per_endpoint[answer.endpoint] += 1
    return {
        "endpoints": list(endpoints),
        "truth": {e: asdict(t) for e, t in truth.items()},
        "prompts": prompts,
        "answers_per_endpoint": dict(answers_per_endpoint),
        "agreement": agreement_summary(tensors, truth),
        "groups": group_table(tensors, truth),
        "per_prompt": prompt_table(tensors, truth),
        "reliability": reliability,
        "repeat_vs_diversity": repeat_vs_diversity(
            answers, prompts, endpoints, truth, CURVE_KERNELS, CURVE_BUDGETS, MAX_REPEATS),
    }


__all__ = ["HEADLINE_KERNEL", "evaluate_design", "load_reference_answers", "run_analysis"]
