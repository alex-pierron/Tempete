"""Metrics of salvo 2 (team D): strong LoRA on d4, band rule, d6 echo test for fp8, d7 replication.

All thresholds below were written in `d_prompts.py` (commit 8ffdb75) BEFORE any d6 or d7 answer was
collected; this module only applies them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hackathon.model_equality.research.d_eval import (
    Design,
    EvalError,
    build_design,
    center_by_cell,
    draw_cells,
    feature_tensor,
)
from hackathon.model_equality.research.d_separability import separability_table
from hackathon.model_equality.store import Answer, load_many

D6_SUCCESS = 0.65
D6_FAILURE = 0.55
D7_SUCCESS = 0.65
D7_BAND_THRESHOLD = 0.6
D7_BAND_TPR_MIN = 0.7
D7_BAND_FPR_MAX = 0.05
VERDICT_SIZE = 20
BAND_THRESHOLDS = (0.5, 0.6, 0.7)
BAND_SIZES = (10, 20, 30)
SEED = 20261121
S_GROUP_MODEL = "S"
FloatArray = NDArray[np.float64]
FP8_ENDPOINT = "ref-02"
LORA_WEAK_ENDPOINT = "ref-09"
STRONG_LORA_ENDPOINTS = ("ref-04", "ref-07")


def _prefix_answers(answers: Sequence[Answer], prefix: str) -> list[Answer]:
    """Answers whose prompt id starts with `prefix`."""
    return [a for a in answers if a.prompt_id.startswith(prefix)]


def try_design(answers: Sequence[Answer], endpoints: Sequence[str]) -> Design | None:
    """The design on the cells common to `endpoints`, or None when there is none yet."""
    try:
        return build_design(answers, endpoints)
    except EvalError:
        return None


def band_rule(
    design: Design, thresholds: Sequence[float], sizes: Sequence[int], replicates: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Share of draws in which each endpoint is flagged (period rate >= threshold), per threshold and N.

    The rule uses no label: an endpoint is flagged when the share of its replies that end with a
    period, over N answers, reaches the threshold.
    """
    tensor = feature_tensor(design, ("ends_period",))[0]
    table: dict[str, dict[str, dict[str, float]]] = {}
    for threshold in thresholds:
        table[f"{threshold:.1f}"] = {}
        for n in sizes:
            if n > len(design.cells):
                continue
            full = n == len(design.cells)
            draws = draw_cells(len(design.cells), n, 1 if full else replicates, SEED)
            rates = tensor[:, draws].mean(axis=2)
            flagged = rates >= threshold - 1e-9
            table[f"{threshold:.1f}"][str(n)] = {
                e: float(flagged[i].mean()) for i, e in enumerate(design.endpoints)}
    return table


def period_and_length(design: Design) -> dict[str, FloatArray]:
    """Cell-centred `ends_period` and `log_chars` tensors, shape (1, endpoints, cells) each."""
    return {
        "ends_period": center_by_cell(feature_tensor(design, ("ends_period",))),
        "length": center_by_cell(feature_tensor(design, ("log_chars",))),
    }


def strong_lora_on_d4(
    answers: Sequence[Answer], model_of: Mapping[str, str], replicates: int
) -> dict[str, Any] | None:
    """AUC of the period-ending criterion for LoRA 0.7 and 1.5 on the d4 prompts, N = 5 and 10."""
    endpoints = ("ref-01", "ref-02", "ref-03", "ref-04", "ref-05", "ref-06", "ref-07", "ref-08", "ref-09")
    design = try_design(_prefix_answers(answers, "d-varith"), endpoints)
    if design is None:
        return None
    table = separability_table(design, model_of, list(STRONG_LORA_ENDPOINTS), period_and_length(design),
                               (5, 10, 20), replicates)
    return {"cells": len(design.cells), "auc": table}


def echo_test(answers: Sequence[Answer], model_of: Mapping[str, str], replicates: int) -> dict[str, Any] | None:
    """d6: fp8 against the five S endpoints on the fresh echo prompts, with every disguise as a control."""
    endpoints = ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08")
    design = try_design(_prefix_answers(answers, "d-echo-"), endpoints)
    if design is None:
        return None
    statistics = {
        **period_and_length(design),
        "echo_exact": center_by_cell(feature_tensor(design, ("echo_exact",))),
        "echo_loose": center_by_cell(feature_tensor(design, ("echo_loose",))),
    }
    targets = [FP8_ENDPOINT, "ref-03", "ref-05", "ref-06", "ref-08"]
    sizes = tuple(n for n in (5, 10, 20, 30, 40) if n <= len(design.cells))
    table = separability_table(design, model_of, targets, statistics, sizes, replicates)
    primary = table["length"][FP8_ENDPOINT].get(str(VERDICT_SIZE))
    verdict = "non évaluable (moins de 20 cellules)"
    if primary is not None:
        verdict = ("succès" if primary >= D6_SUCCESS else "échec" if primary <= D6_FAILURE else "indécis")
    means = {
        e: {name: float(feature_tensor(design, (feat,))[0, i].mean())
            for name, feat in (("echo_exact", "echo_exact"), ("echo_loose", "echo_loose"), ("log_chars", "log_chars"),
                               ("ends_period", "ends_period"))}
        for i, e in enumerate(design.endpoints)}
    return {"cells": len(design.cells), "auc": table, "primary_auc_n20": primary, "verdict": verdict,
            "means": means}


def replication_test(
    answers: Sequence[Answer], model_of: Mapping[str, str], replicates: int
) -> dict[str, Any] | None:
    """d7: LoRA 0.3 and fp8 on 30 new arithmetic prompts x 2 repeats; AUC and band rule."""
    endpoints = ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "ref-09")
    design = try_design(_prefix_answers(answers, "d-rarith-"), endpoints)
    if design is None:
        return None
    targets = [LORA_WEAK_ENDPOINT, FP8_ENDPOINT, "ref-03", "ref-05", "ref-06", "ref-08"]
    sizes = tuple(n for n in (5, 10, 20, 30, 40) if n <= len(design.cells))
    table = separability_table(design, model_of, targets, period_and_length(design), sizes, replicates)
    band = band_rule(design, BAND_THRESHOLDS, BAND_SIZES, replicates)
    auc20 = table["ends_period"][LORA_WEAK_ENDPOINT].get(str(VERDICT_SIZE))
    flagged = band[f"{D7_BAND_THRESHOLD:.1f}"].get(str(VERDICT_SIZE))
    verdict = "non évaluable (moins de 20 cellules)"
    tpr = fpr = None
    if auc20 is not None and flagged is not None:
        tpr = flagged[LORA_WEAK_ENDPOINT]
        s_endpoints = [e for e in design.endpoints if model_of[e] == S_GROUP_MODEL]
        fpr = float(np.mean([flagged[e] for e in s_endpoints]))
        verdict = ("succès" if auc20 >= D7_SUCCESS and tpr >= D7_BAND_TPR_MIN and fpr <= D7_BAND_FPR_MAX
                   else "échec")
    return {"cells": len(design.cells), "auc": table, "band": band, "auc_lora_n20": auc20,
            "band_tpr_n20": tpr, "band_fpr_n20": fpr, "verdict": verdict}


def salvo2_metrics(store: Path, model_of: Mapping[str, str], replicates: int) -> dict[str, Any]:
    """Everything of salvo 2 that is computed from the team-D store."""
    answers = load_many([store])
    d4 = [a for a in answers if a.prompt_id.startswith("d-varith")]
    d4_design = try_design(d4, ("ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "ref-09"))
    return {
        "d4_band": (band_rule(d4_design, BAND_THRESHOLDS, BAND_SIZES, replicates) if d4_design else None),
        "d4_strong": strong_lora_on_d4(answers, model_of, replicates),
        "d6_echo": echo_test(answers, model_of, replicates),
        "d7_replication": replication_test(answers, model_of, replicates),
    }
