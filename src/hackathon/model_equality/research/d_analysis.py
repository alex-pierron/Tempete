"""Numbers of the team-D report: AUC vs N, dose response, effect sizes, growth, fresh validation.

Everything is computed from the `ref-*` answer stores and the published truth; the result is one
JSON-serialisable dict (`metrics.json`). Statistics are pre-registered by name (see `STATISTICS`)
so that no figure shows a winner picked after the fact; the selection experiment (`d_select`)
quantifies what picking a winner would have cost.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hackathon.model_equality.research.d_battery import BATTERY_FEATURES
from hackathon.model_equality.research.d_direction import fit_and_score, out_of_fold_scores
from hackathon.model_equality.research.d_eval import (
    BASE_MODEL,
    DEFAULT_SAMPLE_SIZES,
    EVAL_SEED,
    AucSummary,
    Design,
    EvalError,
    Roles,
    auc_curve_for_tensor,
    build_design,
    center_by_cell,
    draw_cells,
    feature_tensor,
    pair_values,
    restrict_design,
    roles_from_models,
    scalar_distances,
)
from hackathon.model_equality.research.d_exact import (
    NORMALISERS,
    answer_keys,
    collision_tensor,
    consensus_fraction,
    exact_auc_curve,
    mismatch_tensor,
)
from hackathon.model_equality.research.d_features import FEATURES, LONG_OUTPUT_FEATURES
from hackathon.model_equality.research.d_select import prompt_family, selection_experiment
from hackathon.model_equality.research.dataset import load_truth
from hackathon.model_equality.store import AnswerStore, load_many

SAMPLE_SIZES = (*DEFAULT_SAMPLE_SIZES, 80, 120, 160)
"""N = 5, 10, 20, 40 as asked, then 80 to 160 to see where a curve would cross 0.8 (the ep budget
is 100: beyond that is extrapolation)."""
GROWTH_SIZES = (10, 20, 40, 80, 120)
T_TARGET = 3.0
"""t statistic at which an endpoint is called different from S in the growth extrapolation."""
AUC_TARGET_Z = 1.68
"""z of the AUC 0.8 target: AUC = Phi(d sqrt(N) / 2), so N = (1.68 / d)^2 for AUC 0.8."""
BOOTSTRAPS = 300
MIN_SLOPE = 1e-3
SELECTION_SEED = 7
FRESH_MIN_CELLS_FOR_AUC = 10
SHORT_FAMILIES = frozenset({"arith", "numeric", "yesno", "choice", "completion"})
"""Families of wave d1 (short answers), as opposed to the d3 batteries."""
CONSENSUS_THRESHOLDS = (0.0, 0.34, 0.45, 0.67)
"""Consensus levels of the exact-agreement study (0 = every prompt, 0.45 = low-entropy prompts)."""
MATRIX_THRESHOLD = 0.45
EXAMPLE_COUNT = 5
EXAMPLE_CHARS = 150
TRANSFER_POSITIVES = ("ref-04", "ref-07")
"""LoRA 0.7 and 1.5: the strong adapters whose direction is learned (never the labels of T1, T2)."""
STATISTICS: Mapping[str, tuple[str, ...]] = {
    "length": ("log_chars",),
    "ends_period": ("ends_period",),
    "sentences": ("sentence_count",),
    "style22": tuple(FEATURES),
}
"""Pre-registered scalar statistics (equal-weight L2 distance of standardised feature means)."""
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs and sizes of one analysis run."""

    discovery_paths: tuple[Path, ...]
    fresh_path: Path
    truth_path: Path
    endpoints: tuple[str, ...]
    replicates: int
    selection_splits: int


def _curve_to_json(curve: Mapping[int, Mapping[str, AucSummary]]) -> dict[str, Any]:
    """AUC curve with string keys for JSON."""
    return {str(n): {k: asdict(v) for k, v in d.items()} for n, d in curve.items()}


def _paired_differences(values: FloatArray, endpoints: Sequence[str], base: Sequence[str]) -> dict[str, FloatArray]:
    """Per cell, endpoint value minus the mean of the OTHER base endpoints."""
    index = {e: i for i, e in enumerate(endpoints)}
    out: dict[str, FloatArray] = {}
    for endpoint in endpoints:
        others = [index[b] for b in base if b != endpoint]
        out[endpoint] = values[index[endpoint]] - values[others].mean(axis=0)
    return out


def _t_value(differences: FloatArray) -> float:
    """One-sample t of a vector of paired differences (0 when there is no spread)."""
    spread = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return 0.0 if spread == 0.0 else float(differences.mean() / (spread / math.sqrt(len(differences))))


def effect_matrix(design: Design, base: Sequence[str]) -> dict[str, dict[str, float]]:
    """t of every endpoint against the other S endpoints, for every scalar feature."""
    names = list(FEATURES)
    tensor = feature_tensor(design, names)
    result: dict[str, dict[str, float]] = {e: {} for e in design.endpoints}
    for k, name in enumerate(names):
        for endpoint, diffs in _paired_differences(tensor[k], design.endpoints, base).items():
            result[endpoint][name] = _t_value(diffs)
    return result


def t_growth(
    values: FloatArray, endpoints: Sequence[str], base: Sequence[str], seed: int, replicates: int
) -> dict[str, dict[str, float]]:
    """Mean t of each endpoint against S at several sample sizes, plus the fitted slope c of t = c sqrt(N).

    The slope is a least-squares fit through the origin; `n_for_target` is the extrapolated N where
    t reaches `T_TARGET` (an ESTIMATE: it assumes the effect keeps its size and growth in sqrt(N)).
    """
    diffs = _paired_differences(values, endpoints, base)
    result: dict[str, dict[str, float]] = {}
    for endpoint, d in diffs.items():
        means: dict[int, float] = {}
        for n in GROWTH_SIZES:
            if n > len(d):
                continue
            draws = draw_cells(len(d), n, replicates, seed)
            means[n] = float(np.mean([_t_value(d[row]) for row in draws]))
        sizes = np.asarray(sorted(means), dtype=np.float64)
        observed = np.asarray([means[int(n)] for n in sizes], dtype=np.float64)
        root = np.sqrt(sizes)
        slope = float((root * observed).sum() / (root * root).sum()) if len(sizes) else 0.0
        needed = (T_TARGET / slope) ** 2 if abs(slope) > MIN_SLOPE else float("inf")
        row = {f"t_at_{n}": v for n, v in means.items()}
        row["slope"] = slope
        row["n_for_target"] = needed
        result[endpoint] = row
    return result


def cohen_d_with_interval(
    values: FloatArray, endpoints: Sequence[str], base: Sequence[str], seed: int
) -> dict[str, dict[str, float]]:
    """Paired effect size d = mean(diff) / sd(diff) per endpoint, with a bootstrap 95 % interval, and N for AUC 0.8."""
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for endpoint, d in _paired_differences(values, endpoints, base).items():
        def effect(sample: FloatArray) -> float:
            spread = float(sample.std(ddof=1))
            return 0.0 if spread == 0.0 else float(sample.mean() / spread)

        boots = [effect(d[rng.integers(0, len(d), len(d))]) for _ in range(BOOTSTRAPS)]
        point = effect(d)
        result[endpoint] = {
            "d": point,
            "low": float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5)),
            "n_for_auc_08": (AUC_TARGET_Z / point) ** 2 if abs(point) > MIN_SLOPE else float("inf"),
        }
    return result


def mean_with_interval(values: FloatArray, seed: int) -> dict[str, float]:
    """Mean and bootstrap 95 % interval of a vector."""
    rng = np.random.default_rng(seed)
    boots = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(BOOTSTRAPS)]
    return {"mean": float(values.mean()), "low": float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5))}


def row_counts(paths: Sequence[Path], design: Design, merged_rows: int) -> dict[str, int]:
    """Rows in, after key de-duplication, used by the design and rejected (cell not answered by every endpoint)."""
    rows_in = sum(len(AnswerStore(path).load()) for path in paths)
    used = len(design.cells) * len(design.endpoints)
    return {"rows_in": rows_in, "after_dedupe": merged_rows, "duplicates_dropped": rows_in - merged_rows,
            "rows_used": used, "rows_rejected": merged_rows - used}


def mcc_value(true_positive: FloatArray, false_positive: FloatArray, false_negative: FloatArray,
              true_negative: FloatArray) -> FloatArray:
    """Matthews correlation per replicate from confusion counts (0 when a margin is empty)."""
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = np.sqrt((true_positive + false_positive) * (true_positive + false_negative)
                          * (true_negative + false_positive) * (true_negative + false_negative))
    return np.where(denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0)


def _mcc_at(same: FloatArray, different: FloatArray, threshold: float) -> FloatArray:
    """MCC per replicate when `distance <= threshold` means 'same model' ('same' is the positive class)."""
    tp = (same <= threshold).sum(axis=1).astype(np.float64)
    fn = (same > threshold).sum(axis=1).astype(np.float64)
    fp = (different <= threshold).sum(axis=1).astype(np.float64)
    tn = (different > threshold).sum(axis=1).astype(np.float64)
    return mcc_value(tp, fp, fn, tn)


def mcc_proxy(
    tensor: FloatArray,
    endpoints: Sequence[str],
    roles: Roles,
    target: Sequence[str],
    sample_size: int,
    replicates: int,
) -> dict[str, float]:
    """Held-out MCC of a threshold rule on the pairs (target, S) vs (S, S), at one sample size.

    The 10 S-S pairs are the 'same model' class, the pairs (target endpoint, S endpoint) the
    'different' class. The threshold is chosen on one half of the draws and scored on the other
    (and the reverse): the number is not tuned on the data it is reported on. It is a PROXY: it
    ignores the other models and the family level of the real score.
    """
    draws = draw_cells(tensor.shape[2], sample_size, replicates, EVAL_SEED)
    distances = scalar_distances(tensor, draws, np.ones(tensor.shape[0], dtype=np.float64))
    same = pair_values(distances, endpoints, roles.base, roles.base)
    different = pair_values(distances, endpoints, target, roles.base)
    grid = np.unique(np.percentile(np.concatenate([same.ravel(), different.ravel()]), np.arange(5, 100, 5)))
    halves = (np.arange(replicates) % 2 == 0, np.arange(replicates) % 2 == 1)
    scores: list[float] = []
    for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        best = max(grid, key=lambda th: float(_mcc_at(same[train], different[train], float(th)).mean()))
        scores.extend(_mcc_at(same[test], different[test], float(best)).tolist())
    values = np.asarray(scores, dtype=np.float64)
    return {"mean": float(values.mean()), "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)), "share_above_half": float((values >= 0.5).mean())}


def discovery_analysis(config: AnalysisConfig) -> tuple[dict[str, Any], Design, Roles]:
    """Metrics on the answers of teams A, B and m-met-1 (discovery)."""
    truth = load_truth(config.truth_path)
    answers = load_many(list(config.discovery_paths))
    design = build_design(answers, config.endpoints)
    roles = roles_from_models(truth.model)
    base = list(roles.base)
    out: dict[str, Any] = {
        "row_counts": row_counts(config.discovery_paths, design, len(answers)),
        "cells": len(design.cells),
        "prompts": len({p for p, _ in design.cells}),
        "answers_per_endpoint": len(design.cells),
        "families": sorted({prompt_family(p) for p, _ in design.cells}),
    }
    weights_one = np.ones(1, dtype=np.float64)
    curves: dict[str, Any] = {}
    for name, features in STATISTICS.items():
        tensor = center_by_cell(feature_tensor(design, features))
        weights = np.ones(len(features), dtype=np.float64)
        curve = auc_curve_for_tensor(tensor, design.endpoints, roles, weights, SAMPLE_SIZES, config.replicates)
        curves[name] = _curve_to_json(curve)
    transfer = out_of_fold_scores(design, TRANSFER_POSITIVES, base)
    curves["transfer"] = _curve_to_json(auc_curve_for_tensor(
        center_by_cell(transfer[None]), design.endpoints, roles, weights_one, SAMPLE_SIZES, config.replicates))
    oracle_fp8 = out_of_fold_scores(design, roles.fp8, base)
    oracle_lora = out_of_fold_scores(design, roles.lora_weak, base)
    curves["oracle_fp8"] = _curve_to_json(auc_curve_for_tensor(
        center_by_cell(oracle_fp8[None]), design.endpoints, roles, weights_one, SAMPLE_SIZES, config.replicates))
    curves["oracle_lora03"] = _curve_to_json(auc_curve_for_tensor(
        center_by_cell(oracle_lora[None]), design.endpoints, roles, weights_one, SAMPLE_SIZES, config.replicates))
    out["auc_curves"] = curves
    period_centered = center_by_cell(feature_tensor(design, ("ends_period",)))
    out["mcc_proxy"] = {
        label: {str(n): mcc_proxy(period_centered, design.endpoints, roles, target, n, config.replicates)
                for n in (10, 20, 40, 80)}
        for label, target in (("fp8", roles.fp8), ("lora_weak", roles.lora_weak))}
    index = {e: i for i, e in enumerate(design.endpoints)}
    out["transfer_mean_score"] = {
        e: mean_with_interval(transfer[index[e]], EVAL_SEED) for e in design.endpoints}
    out["oracle_mean_score"] = {
        "fp8": {e: mean_with_interval(oracle_fp8[index[e]], EVAL_SEED) for e in design.endpoints},
        "lora03": {e: mean_with_interval(oracle_lora[index[e]], EVAL_SEED) for e in design.endpoints},
    }
    out["effect_t"] = effect_matrix(design, base)
    growth_sources = {
        "transfer": transfer,
        "ends_period": feature_tensor(design, ("ends_period",))[0],
        "length": feature_tensor(design, ("log_chars",))[0],
    }
    out["t_growth"] = {
        name: t_growth(values, design.endpoints, base, EVAL_SEED, config.replicates)
        for name, values in growth_sources.items()}
    out["cohen_d"] = {
        "transfer": cohen_d_with_interval(transfer, design.endpoints, base, EVAL_SEED),
        "oracle_fp8": cohen_d_with_interval(oracle_fp8, design.endpoints, base, EVAL_SEED),
        "oracle_lora03": cohen_d_with_interval(oracle_lora, design.endpoints, base, EVAL_SEED),
    }
    if config.selection_splits > 0:
        selection = selection_experiment(
            design, roles, list(FEATURES), config.selection_splits, SELECTION_SEED)
        out["selection"] = {
            "selected_auc": list(selection.selected_auc),
            "held_out_auc": list(selection.held_out_auc),
            "winners": [list(w) for w in selection.winners],
            "mean_candidate_auc": selection.mean_candidate_auc,
            "splits_requested": config.selection_splits,
        }
    period_tensor = feature_tensor(design, ("ends_period",))[0]
    arith_cells = [i for i, (pid, _r) in enumerate(design.cells) if prompt_family(pid) == "arith"]
    out["period_rate"] = {e: float(period_tensor[index[e]].mean()) for e in design.endpoints}
    out["period_rate_arith"] = {
        e: float(period_tensor[index[e], arith_cells].mean()) for e in design.endpoints} if arith_cells else {}
    out["period_rate_arith_cells"] = len(arith_cells)
    repeat_stats = collision_tensor(design, NORMALISERS["normalised"])
    out["repeat_stats"] = {
        name: _curve_to_json(auc_curve_for_tensor(
            center_by_cell(repeat_stats[k:k + 1]), design.endpoints, roles, weights_one,
            tuple(n for n in SAMPLE_SIZES if n <= repeat_stats.shape[2]), config.replicates))
        for k, name in enumerate(("collisions", "diversity"))}
    out["repeat_prompts"] = int(repeat_stats.shape[2])
    out["exact"] = exact_analysis(design, roles, config.replicates)
    out["examples"] = example_rows(design, roles)
    out["truth"] = {e: {"model": truth.model[e], "kind": truth.kind[e]} for e in truth.endpoints}
    out["roles"] = {"base": list(roles.base), "fp8": list(roles.fp8), "lora_weak": list(roles.lora_weak)}
    return out, design, roles


def exact_analysis(design: Design, roles: Roles, replicates: int) -> dict[str, Any]:
    """Exact-agreement statistic on low-entropy cells: AUC table, mismatch matrix, agreement with the mode."""
    normaliser = NORMALISERS["normalised"]
    table: dict[str, Any] = {}
    for threshold in CONSENSUS_THRESHOLDS:
        count, curve = exact_auc_curve(design, roles, normaliser, threshold, SAMPLE_SIZES, replicates)
        table[f"{threshold:.2f}"] = {"cells": count, "curve": _curve_to_json(curve)}
    keys = answer_keys(design, normaliser)
    consensus = consensus_fraction(keys)
    selected = np.flatnonzero(consensus >= MATRIX_THRESHOLD)
    matrix = mismatch_tensor(keys)[selected].mean(axis=0)
    modes = [Counter(keys[e][c] for e in range(len(keys))).most_common(1)[0][0] for c in selected]
    agreement = {
        endpoint: float(np.mean([keys[i][c] == modes[n] for n, c in enumerate(selected)]))
        for i, endpoint in enumerate(design.endpoints)}
    return {
        "table": table,
        "matrix_threshold": MATRIX_THRESHOLD,
        "matrix_cells": len(selected),
        "endpoints": list(design.endpoints),
        "mismatch": matrix.tolist(),
        "mode_agreement": agreement,
    }


def example_rows(design: Design, roles: Roles) -> list[dict[str, Any]]:
    """Cells where LoRA 0.3 ends with a period and the S endpoints do not (and the converse trap)."""
    lora = roles.lora_weak[0]
    fp8 = roles.fp8[0]
    system_prompt = "ref-03"
    rows: list[dict[str, Any]] = []

    def period(endpoint: str, cell: int) -> bool:
        return design.responses[endpoint][cell].rstrip().endswith(".")

    plain = [e for e in roles.base if e != system_prompt]
    for cell, (prompt_id, _repeat) in enumerate(design.cells):
        if not prompt_id.startswith(("arith-", "cand-numeric", "cand-yesno")):
            continue
        sample = {e: design.responses[e][cell][:EXAMPLE_CHARS] for e in design.endpoints}
        if period(lora, cell) and not any(period(e, cell) for e in plain) and not period(fp8, cell):
            rows.append({"kind": "LoRA 0.3 ends with a period, the other S endpoints do not",
                         "prompt": design.prompts[prompt_id][:EXAMPLE_CHARS], "responses": sample})
        elif period(system_prompt, cell) and not period(lora, cell):
            rows.append({"kind": "trap: the system-prompt disguise ends with a period, LoRA 0.3 does not",
                         "prompt": design.prompts[prompt_id][:EXAMPLE_CHARS], "responses": sample})
    kept: list[dict[str, Any]] = []
    for kind in dict.fromkeys(r["kind"] for r in rows):
        kept.extend([r for r in rows if r["kind"] == kind][:EXAMPLE_COUNT])
    return kept


def fresh_analysis(
    config: AnalysisConfig, discovery: Design, roles: Roles
) -> dict[str, Any]:
    """Out-of-sample look at the answers collected by team D on prompts nobody had seen."""
    fresh_endpoints = tuple(e for e in config.endpoints if e in {*roles.base, *roles.fp8, *roles.lora_weak})
    answers = load_many([config.fresh_path])
    result: dict[str, Any] = {"endpoints": list(fresh_endpoints), "groups": {}}
    def is_short(prompt_id: str) -> bool:
        return prompt_id.startswith("d-") and prompt_family(prompt_id[2:]) in SHORT_FAMILIES

    def is_battery(prompt_id: str) -> bool:
        return prompt_id.startswith("d-b")

    def is_validation(prompt_id: str) -> bool:
        return prompt_id.startswith("d-varith")

    def is_facts(prompt_id: str) -> bool:
        return prompt_id.startswith("d-bfact2")

    groups: dict[str, Callable[[str], bool]] = {
        "d1_short": is_short, "d3_batteries": is_battery, "d4_validation": is_validation,
        "d5_facts": is_facts}
    base = list(roles.base)
    for group, selector in groups.items():
        subset = [a for a in answers if selector(a.prompt_id)]
        info: dict[str, Any] = {"answers": len(subset)}
        try:
            design = build_design(subset, fresh_endpoints)
        except EvalError:
            info["cells"] = 0
            result["groups"][group] = info
            continue
        info["cells"] = len(design.cells)
        info["cell_ids"] = [f"{p}#{r}" for p, r in design.cells]
        scores = fit_and_score(discovery, design, TRANSFER_POSITIVES, base)
        info["transfer_mean"] = {
            e: float(scores[i].mean()) for i, e in enumerate(design.endpoints)}
        tensor = feature_tensor(design, list(FEATURES))
        info["feature_means"] = {
            e: {name: float(tensor[k, i].mean()) for k, name in enumerate(FEATURES)}
            for i, e in enumerate(design.endpoints)}
        if group in {"d3_batteries", "d5_facts"}:
            info["family_means"] = _family_feature_means(design)
        if len(design.cells) >= FRESH_MIN_CELLS_FOR_AUC:
            sizes = tuple(n for n in (5, 10, 20, 40) if n <= len(design.cells))
            auc: dict[str, Any] = {}
            for name, features in STATISTICS.items():
                t = center_by_cell(feature_tensor(design, features))
                auc[name] = _curve_to_json(auc_curve_for_tensor(
                    t, design.endpoints, roles, np.ones(len(features)), sizes, config.replicates))
            auc["transfer"] = _curve_to_json(auc_curve_for_tensor(
                center_by_cell(scores[None]), design.endpoints, roles, np.ones(1), sizes, config.replicates))
            info["auc"] = auc
            period = center_by_cell(feature_tensor(design, ("ends_period",)))
            info["mcc_proxy"] = {
                label: {str(n): mcc_proxy(period, design.endpoints, roles, target, n, config.replicates)
                        for n in sizes}
                for label, target in (("fp8", roles.fp8), ("lora_weak", roles.lora_weak))}
        result["groups"][group] = info
    return result


def _family_feature_means(design: Design) -> dict[str, Any]:
    """Per prompt family, per endpoint, the mean of every battery / long-output feature that is not constant 0."""
    names = [*BATTERY_FEATURES, *LONG_OUTPUT_FEATURES]
    families = sorted({prompt_family(p.removeprefix("d-")) for p, _ in design.cells})
    out: dict[str, Any] = {}
    for family in families:
        keep = [i for i, (p, _r) in enumerate(design.cells) if prompt_family(p.removeprefix("d-")) == family]
        sub = restrict_design(design, keep)
        tensor = feature_tensor(sub, names)
        used = [k for k, _name in enumerate(names) if np.any(tensor[k] != 0.0)]
        out[family] = {
            "cells": len(keep),
            "cell_values": {
                f"{sub.cells[c][0]}#{sub.cells[c][1]}": {
                    e: {names[k]: float(tensor[k, i, c]) for k in used}
                    for i, e in enumerate(design.endpoints)}
                for c in range(len(sub.cells))},
            "features": [names[k] for k in used],
            "means": {e: {names[k]: float(tensor[k, i].mean()) for k in used}
                      for i, e in enumerate(design.endpoints)},
        }
    return out


def refresh_fresh(config: AnalysisConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    """Recompute only the fresh-prompt part of an earlier metrics dict (discovery numbers are kept)."""
    truth = load_truth(config.truth_path)
    design = build_design(load_many(list(config.discovery_paths)), config.endpoints)
    refreshed = dict(metrics)
    refreshed["fresh"] = fresh_analysis(config, design, roles_from_models(truth.model))
    return refreshed


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Full metrics dict (discovery + fresh)."""
    discovery_metrics, design, roles = discovery_analysis(config)
    return {
        "base_model": BASE_MODEL,
        "discovery": discovery_metrics,
        "fresh": fresh_analysis(config, design, roles),
        "sample_sizes": list(SAMPLE_SIZES),
        "replicates": config.replicates,
    }
