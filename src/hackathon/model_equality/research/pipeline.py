"""End-to-end phase-1 research run on the `ref-*` answers (team A).

`run_research` returns one JSON-serialisable dictionary (also written as `metrics.json`) and keeps
the raw z matrices in a `ResearchArtifacts` object for the figures. Every number in the report
comes from here; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.analysis import DEFAULT_TAU_MODEL, compute_channels
from hackathon.model_equality.boilerplate import BoilerplateNote, strip_boilerplate
from hackathon.model_equality.clustering import (
    build_linkage,
    calibrate_weights,
    combine_channels,
    cut_dendrogram,
    prior_weights,
    to_distance,
)
from hackathon.model_equality.features import tokenize_words
from hackathon.model_equality.pairwise import DEFAULT_PERMUTATIONS, KERNELS, Channel
from hackathon.model_equality.prompts import category_of
from hackathon.model_equality.research.dataset import ReferenceTruth
from hackathon.model_equality.research.evaluate import (
    METHOD_CALIBRATED,
    METHOD_PRIOR,
    METHOD_UNIFORM,
    evaluate_loeo,
    mcc_of,
    pairs_of,
    pairwise_threshold_mcc,
    score_channel,
)
from hackathon.model_equality.research.extra_stats import (
    EXTRA_KERNELS,
    extra_channels,
    pooled_channel,
)
from hackathon.model_equality.store import Answer

FloatMatrix = npt.NDArray[np.float64]

SWEEP_TARGETS: tuple[int, ...] = (12, 20, 30, 40, 50, 60, 70, 80, 90)
"""Answers per endpoint at which the MCC is measured (90 = the whole `ep` spendable budget)."""
SWEEP_DRAWS = 30
SWEEP_PERMUTATIONS = 100
SWEEP_SEED = 20261123
STABLE_CATEGORIES: tuple[str, ...] = ("arith", "completion", "fact", "format")
"""Near-deterministic categories (short, factual, constrained) as grouped by the kit."""
CHEAP_KERNELS: tuple[str, ...] = ("hamming_char", "hamming_word", "spectrum", "onehot_key")
"""Kernels the kit weights positively a priori (first word and length are disguise-prone)."""
ALL_KERNELS: tuple[str, ...] = KERNELS + EXTRA_KERNELS
Z_GROWTH_KERNELS: tuple[str, ...] = ("hamming_char", "spectrum", "length")
EXAMPLE_CHARS = 170
SECONDS_DECIMALS = 2


@dataclass
class ResearchArtifacts:
    """Raw matrices kept for plotting."""

    endpoints: list[str]
    channels: dict[Channel, FloatMatrix]
    pooled: dict[str, FloatMatrix]
    combined_prior: FloatMatrix
    combined_calibrated: FloatMatrix
    boilerplate: list[BoilerplateNote] = field(default_factory=list)
    sweep_raw: dict[str, dict[int, list[float]]] = field(default_factory=dict)


def _clean_float(value: float) -> float | None:
    """JSON-safe float (NaN becomes null)."""
    return None if not np.isfinite(value) else round(float(value), 4)


def data_summary(answers: Sequence[Answer], endpoints: Sequence[str]) -> dict[str, Any]:
    """Counts, determinism (are two repeats of a prompt different?) and length statistics."""
    per_endpoint: dict[str, Any] = {}
    for endpoint in endpoints:
        mine = [a for a in answers if a.endpoint == endpoint]
        by_prompt: dict[str, list[str]] = defaultdict(list)
        for answer in mine:
            by_prompt[answer.prompt_id].append(answer.response.strip())
        identical = [len(set(v)) == 1 for v in by_prompt.values() if len(v) >= 2]
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for prompt_id, texts in by_prompt.items():
            if len(texts) >= 2:
                by_cat[category_of(prompt_id)].append(len(set(texts)) == 1)
        words = [len(tokenize_words(a.response)) for a in mine]
        per_endpoint[endpoint] = {
            "n": len(mine),
            "prompts": len(by_prompt),
            "empty": sum(not a.response.strip() for a in mine),
            "identical_repeat_rate": _clean_float(float(np.mean(identical))) if identical else None,
            "identical_repeat_rate_by_category": {
                c: _clean_float(float(np.mean(v))) for c, v in sorted(by_cat.items())},
            "mean_words": _clean_float(float(np.mean(words))),
            "median_words": _clean_float(float(np.median(words))),
        }
    return per_endpoint


def examples_table(
    answers: Sequence[Answer], truth: ReferenceTruth, per_category: int = 1
) -> list[dict[str, Any]]:
    """For a few prompts, the raw reply of every endpoint (repeat 0), clipped for display."""
    seen: Counter[str] = Counter()
    chosen: list[str] = []
    for answer in sorted(answers, key=lambda a: a.prompt_id):
        category = category_of(answer.prompt_id)
        if answer.prompt_id not in chosen and seen[category] < per_category:
            chosen.append(answer.prompt_id)
            seen[category] += 1
    out: list[dict[str, Any]] = []
    for prompt_id in chosen:
        rows = {a.endpoint: a for a in answers if a.prompt_id == prompt_id and a.repeat == 0}
        if not rows:
            continue
        prompt = next(iter(rows.values())).prompt
        out.append({
            "prompt_id": prompt_id,
            "prompt": prompt[:EXAMPLE_CHARS],
            "replies": {e: rows[e].response.replace("\n", " ")[:EXAMPLE_CHARS]
                        for e in truth.endpoints if e in rows},
        })
    return out


def channel_scores_table(
    channels: Mapping[Channel, FloatMatrix], truth: ReferenceTruth
) -> list[dict[str, Any]]:
    """Per (category, kernel) separation of same-model vs different-model pairs."""
    rows = []
    for channel, matrix in channels.items():
        s = score_channel(channel, matrix, truth)
        rows.append({
            "category": channel[0], "kernel": channel[1], "auc": _clean_float(s.auc),
            "mean_same": _clean_float(s.mean_same), "mean_different": _clean_float(s.mean_different),
            "cohens_d": _clean_float(s.cohens_d), "n_same": s.n_same, "n_different": s.n_different,
        })
    return rows


def pooled_scores_table(
    pooled: Mapping[str, FloatMatrix], truth: ReferenceTruth
) -> list[dict[str, Any]]:
    """Same as `channel_scores_table` for the kernel pooled over categories."""
    rows = []
    for kernel, matrix in pooled.items():
        s = score_channel(("all", kernel), matrix, truth)
        rows.append({
            "kernel": kernel, "auc": _clean_float(s.auc), "mean_same": _clean_float(s.mean_same),
            "mean_different": _clean_float(s.mean_different), "cohens_d": _clean_float(s.cohens_d),
        })
    return rows


def trap_table(
    pooled: Mapping[str, FloatMatrix], truth: ReferenceTruth
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Pooled z of `ref-01` (plain S) against each other endpoint, per kernel.

    Disguises of S should stay small, T1..T4 should be large: this is the per-trap view.
    """
    anchor = truth.endpoints.index(next(e for e in truth.endpoints
                                        if truth.kind[e].startswith("plain")))
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for kernel, matrix in pooled.items():
        out[kernel] = {
            truth.endpoints[j]: {"z": _clean_float(float(matrix[anchor, j]))}
            for j in range(len(truth.endpoints)) if j != anchor
        }
    return out


def anchor_by_category(
    channels: Mapping[Channel, FloatMatrix], truth: ReferenceTruth,
    kernels: Sequence[str] = ("hamming_char", "spectrum"),
) -> dict[str, dict[str, dict[str, float | None]]]:
    """z of plain S against every other endpoint, per kernel and prompt category.

    Shows which categories carry the separation of fp8 / LoRA and which ones only reveal the
    disguises (input for the prompt-selection team).
    """
    anchor = next(i for i, e in enumerate(truth.endpoints) if truth.kind[e].startswith("plain"))
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for kernel in kernels:
        per_category: dict[str, dict[str, float | None]] = {}
        for (category, kern), matrix in sorted(channels.items()):
            if kern != kernel:
                continue
            per_category[category] = {
                truth.endpoints[j]: _clean_float(float(matrix[anchor, j]))
                for j in range(len(truth.endpoints)) if j != anchor
            }
        out[kernel] = per_category
    return out


def kit_default_mcc(
    channels: Mapping[Channel, FloatMatrix], truth: ReferenceTruth
) -> dict[str, float]:
    """MCC of the kit used as-is (prior weights, uncalibrated `DEFAULT_TAU_MODEL`), no fitting."""
    combined = combine_channels(channels, prior_weights(list(channels)))
    distance, _ = to_distance(combined)
    assigned = cut_dendrogram(build_linkage(distance), DEFAULT_TAU_MODEL)
    pairs = pairs_of(truth)
    flags = [same for _, _, same in pairs]
    guess = [bool(assigned[i] == assigned[j]) for i, j, _ in pairs]
    return {"mcc": mcc_of(flags, guess), "tau": DEFAULT_TAU_MODEL}


def loeo_table(
    sets: Mapping[str, tuple[Mapping[Channel, FloatMatrix], str]], truth: ReferenceTruth,
) -> list[dict[str, Any]]:
    """Leave-one-endpoint-out MCC (with a bootstrap interval) of several channel sets."""
    rows = []
    for name, (channels, method) in sets.items():
        result = evaluate_loeo(channels, truth, method)
        rows.append({
            "set": name, "method": method, "mcc": _clean_float(result.mcc),
            "ci95": [_clean_float(result.ci_low), _clean_float(result.ci_high)],
            "false_same": result.false_same, "false_different": result.false_different,
            "wrong_pairs": [f"{truth.endpoints[i]}~{truth.endpoints[j]}"
                            for i, j in result.wrong_pairs],
        })
    return rows


def named_sets(
    kit: Mapping[Channel, FloatMatrix], everything: Mapping[Channel, FloatMatrix],
    pooled: Mapping[str, FloatMatrix],
) -> dict[str, tuple[Mapping[Channel, FloatMatrix], str]]:
    """The channel sets compared in the leave-one-out table."""
    def of(kernels: Sequence[str]) -> dict[Channel, FloatMatrix]:
        return {("all", k): pooled[k] for k in kernels if k in pooled}

    stable = {c: m for c, m in kit.items() if c[0] in STABLE_CATEGORIES}
    sets: dict[str, tuple[Mapping[Channel, FloatMatrix], str]] = {
        "kit as shipped: prior weights, cut fitted": (kit, METHOD_PRIOR),
        "kit calibrated (weights + cut fitted)": (kit, METHOD_CALIBRATED),
        "kit calibrated, stable categories only": (stable, METHOD_CALIBRATED),
        "kit + extra channels, calibrated": (everything, METHOD_CALIBRATED),
        "pooled cheap kernels (4), uniform": (of(CHEAP_KERNELS), METHOD_UNIFORM),
        "pooled all kernels (14), calibrated": (of(ALL_KERNELS), METHOD_CALIBRATED),
    }
    for kernel in ALL_KERNELS:
        if kernel in pooled:
            sets[f"single channel: {kernel} (pooled)"] = (of([kernel]), METHOD_UNIFORM)
    return sets


def subset_answers(answers: Sequence[Answer], prompt_ids: set[str]) -> list[Answer]:
    """Keep the answers of the given prompts."""
    return [a for a in answers if a.prompt_id in prompt_ids]


def prompts_up_to(
    cleaned: Sequence[Answer], order: Sequence[str], target: int
) -> tuple[set[str], int]:
    """Prompts taken in `order` until each endpoint has at least `target` answers.

    Returns:
        The chosen prompt ids and the resulting number of answers per endpoint (the mean over
        endpoints, equal for a balanced store).
    """
    per_prompt: Counter[str] = Counter(a.prompt_id for a in cleaned)
    n_endpoints = len({a.endpoint for a in cleaned})
    chosen: set[str] = set()
    total = 0.0
    for prompt_id in order:
        chosen.add(prompt_id)
        total += per_prompt[prompt_id] / n_endpoints
        if total >= target:
            break
    return chosen, round(total)


def sweep_n(
    cleaned: Sequence[Answer], endpoints: Sequence[str], truth: ReferenceTruth,
    draws: int = SWEEP_DRAWS, targets: Sequence[int] = SWEEP_TARGETS,
    n_permutations: int = SWEEP_PERMUTATIONS, seed: int = SWEEP_SEED,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, list[float]]], list[dict[str, Any]]]:
    """MCC (leave-one-endpoint-out) and pooled z versus answers per endpoint, by subsampling.

    Besides the MCC rows, returns the mean pooled z of plain S (the first endpoint labelled
    `plain`) against every other endpoint, per kernel in `Z_GROWTH_KERNELS`: it shows whether the
    separation of a pair grows with n (a real difference) or not.

    Each draw shuffles the prompts and keeps the first ones until every endpoint has about
    `target` answers (all repeats of a kept prompt are kept, as in the real design); the same
    prompts are used for all endpoints. Targets above the stored size are skipped.
    """
    prompt_ids = sorted({a.prompt_id for a in cleaned})
    per_endpoint = len(cleaned) / len(endpoints)
    rng = np.random.default_rng(seed)
    raw: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    realised: dict[int, list[int]] = defaultdict(list)
    anchor = next(i for i, e in enumerate(truth.endpoints) if truth.kind[e].startswith("plain"))
    z_raw: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for target in targets:
        if target > per_endpoint:
            continue
        for _ in range(draws):
            order = [prompt_ids[i] for i in rng.permutation(len(prompt_ids))]
            chosen, actual = prompts_up_to(cleaned, order, target)
            realised[target].append(actual)
            sub = subset_answers(cleaned, chosen)
            kit = compute_channels(sub, endpoints, n_permutations)
            pooled = {k: m for k in KERNELS if (m := pooled_channel(kit, k)) is not None}

            for kernel in Z_GROWTH_KERNELS:
                for j, endpoint in enumerate(truth.endpoints):
                    if j != anchor and kernel in pooled and np.isfinite(pooled[kernel][anchor, j]):
                        z_raw[(kernel, endpoint, target)].append(float(pooled[kernel][anchor, j]))

            def of(kernels: Sequence[str], pooled: dict[str, FloatMatrix] = pooled
                   ) -> dict[Channel, FloatMatrix]:
                return {("all", k): pooled[k] for k in kernels if k in pooled}

            raw["kit default (tau 4, nothing fitted)"][target].append(
                kit_default_mcc(kit, truth)["mcc"])
            for name, channels, method in (
                ("kit prior", kit, METHOD_PRIOR), ("kit calibrated", kit, METHOD_CALIBRATED),
                ("pooled kit kernels (6), calibrated", of(KERNELS), METHOD_CALIBRATED),
                ("single: hamming_char", of(["hamming_char"]), METHOD_UNIFORM),
                ("single: length", of(["length"]), METHOD_UNIFORM),
            ):
                raw[name][target].append(
                    evaluate_loeo(channels, truth, method, with_interval=False).mcc)
    rows = []
    for name, per_target in raw.items():
        for target, values in sorted(per_target.items()):
            finite = [v for v in values if np.isfinite(v)]
            rows.append({
                "method": name, "answers_per_endpoint": int(np.mean(realised[target])),
                "target": target, "draws": len(finite),
                "mcc_mean": _clean_float(float(np.mean(finite))) if finite else None,
                "mcc_sd": _clean_float(float(np.std(finite))) if finite else None,
                "mcc_p10": _clean_float(float(np.percentile(finite, 10))) if finite else None,
                "mcc_p90": _clean_float(float(np.percentile(finite, 90))) if finite else None,
                "share_mcc_at_least_0_8": _clean_float(float(np.mean([v >= 0.8 for v in finite])))
                if finite else None,
            })
    z_rows = [{"kernel": k, "endpoint": e, "target": t, "z_mean": _clean_float(float(np.mean(v))),
               "z_sd": _clean_float(float(np.std(v))), "n_draws": len(v),
               "answers_per_endpoint": int(np.mean(realised[t]))}
              for (k, e, t), v in sorted(z_raw.items())]
    return rows, {n: {k: list(v) for k, v in d.items()} for n, d in raw.items()}, z_rows


def balanced_units(
    answers: Sequence[Answer], endpoints: Sequence[str]
) -> tuple[list[Answer], int]:
    """Keep only the `(prompt, repeat)` units answered by every endpoint.

    A collection that is still running leaves some units answered by a few endpoints only, which
    would bias pair statistics; those rows are dropped and counted.

    Returns:
        The balanced answers and the number of dropped rows.
    """
    seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    for answer in answers:
        seen[(answer.prompt_id, answer.repeat)].add(answer.endpoint)
    complete = {unit for unit, who in seen.items() if len(who) == len(endpoints)}
    kept = [a for a in answers if (a.prompt_id, a.repeat) in complete]
    return kept, len(answers) - len(kept)


def run_research(
    answers: Sequence[Answer], truth: ReferenceTruth, n_permutations: int = DEFAULT_PERMUTATIONS,
    sweep: bool = True, sweep_targets: Sequence[int] = SWEEP_TARGETS, sweep_draws: int = SWEEP_DRAWS,
) -> tuple[dict[str, Any], ResearchArtifacts]:
    """Run every analysis of the phase-1 report and return `(metrics, artifacts)`."""
    started = time.perf_counter()
    endpoints = list(truth.endpoints)
    all_rows = len(answers)
    answers, unbalanced = balanced_units(answers, endpoints)
    cleaned, notes = strip_boilerplate(list(answers))
    kit = compute_channels(cleaned, endpoints, n_permutations)
    extra = extra_channels(cleaned, endpoints, n_permutations)
    everything = {**kit, **extra}
    pooled = {k: m for k in ALL_KERNELS if (m := pooled_channel(everything, k)) is not None}
    stable_pooled = {k: m for k in ALL_KERNELS
                     if (m := pooled_channel({c: v for c, v in everything.items()
                                              if c[0] in STABLE_CATEGORIES}, k)) is not None}
    combined_prior = combine_channels(kit, prior_weights(list(kit)))
    weights, _ = calibrate_weights(kit, truth.labels)
    combined_calibrated = combine_channels(kit, weights)
    pair_names = [(truth.endpoints[i], truth.endpoints[j], same) for i, j, same in pairs_of(truth)]
    in_sample_pair_mcc, in_sample_tau = pairwise_threshold_mcc(combined_prior, truth)
    metrics: dict[str, Any] = {
        "endpoints": {e: truth.short_name(e) for e in endpoints},
        "rows": {"in": all_rows, "dropped_unbalanced_units": unbalanced, "used": len(answers),
                 "after_boilerplate_strip": len(cleaned)},
        "data_summary": data_summary(answers, endpoints),
        "boilerplate": [n.__dict__ for n in notes],
        "n_pairs": {"same_model": sum(1 for *_, s in pair_names if s),
                    "different_model": sum(1 for *_, s in pair_names if not s)},
        "channels": channel_scores_table(everything, truth),
        "pooled": pooled_scores_table(pooled, truth),
        "pooled_stable_categories": pooled_scores_table(stable_pooled, truth),
        "traps": trap_table(pooled, truth),
        "anchor_by_category": anchor_by_category(everything, truth),
        "kit_default": kit_default_mcc(kit, truth),
        "in_sample_pairwise_threshold_prior": {"mcc": in_sample_pair_mcc, "tau": in_sample_tau},
        "loeo": loeo_table(named_sets(kit, everything, pooled), truth),
        "calibrated_weights": {f"{c}/{k}": round(w, 4) for (c, k), w in weights.items() if w > 0},
        "baselines": {"all_same_mcc": 0.0, "all_different_mcc": 0.0, "note":
                      "a constant labelling has a zero MCC marginal, scored 0 by the kit"},
        "permutations": n_permutations,
    }
    if sweep:
        rows, raw, z_rows = sweep_n(cleaned, endpoints, truth, draws=sweep_draws,
                                    targets=sweep_targets)
        metrics["mcc_vs_n"] = rows
        metrics["z_vs_n"] = z_rows
    else:
        raw = {}
    metrics["duration_s"] = round(time.perf_counter() - started, SECONDS_DECIMALS)
    metrics["examples"] = examples_table(answers, truth)
    metrics["_raw_note"] = "z matrices are kept in memory for plotting, not serialised"
    return metrics, ResearchArtifacts(endpoints, everything, pooled, combined_prior,
                                      combined_calibrated, notes, raw)


def write_metrics(metrics: Mapping[str, Any], path: Path) -> None:
    """Write `metrics.json` (UTF-8, indented)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
