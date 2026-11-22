"""Small-sample calibration of the likelihood features on the ref endpoints (known truth).

The ref endpoints run SmolLM2-360M with a known set of variants (plain, watermark, system prompt,
persona, temperature 1.0, fp8, LoRA 0.3 / 0.7 / 1.5) and about 340 stored answers each, so they can
play the role of 33 labelled ep endpoints. This module draws PSEUDO-ENDPOINTS from a ref endpoint
(random answers until a token budget is reached), computes the likelihood features of each draw,
and measures how well every feature separates a variant from the clean base endpoints at a given
number of tokens. The ep endpoints have about 560 tokens each: the first column of the result is
the honest answer to "what can we read today", the others to "what would more tokens buy".

Features of a draw (all under the identified base model):

* `mle_temperature`      sampling temperature fitted by maximum likelihood;
* `z_at_one`             calibration z at temperature 1 (0 for text sampled at T = 1 by the model);
* `mean_ll`              mean token log-likelihood at temperature 1;
* `llr_<variant>`        mean per-token log-likelihood ratio of a simulated fp8 copy against the base.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_analysis import (
    UNIT_TEMPERATURE,
    temperature_profile_from_sums,
)
from hackathon.model_equality.research.e_loglik import RowScores

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
DRAWS_PER_ENDPOINT = 400
CALIBRATION_SEED = 20261121
TOKEN_BUDGETS: tuple[int, ...] = (560, 1500, 4000, 10000)
"""560 tokens is what each ep endpoint has today (7 answers)."""
CLASS_OF: Mapping[str, str] = {
    "ref-01": "plain", "ref-06": "watermark", "ref-03": "system", "ref-05": "persona",
    "ref-08": "temp1", "ref-02": "fp8", "ref-09": "lora_0.3", "ref-04": "lora_0.7",
    "ref-07": "lora_1.5",
}
CLEAN_CLASSES: tuple[str, ...] = ("plain", "watermark")
TARGET_CLASSES: tuple[str, ...] = (
    "fp8", "lora_0.3", "lora_0.7", "lora_1.5", "system", "persona", "temp1",
)
MEAN_LL = "mean_ll"
MLE_TEMPERATURE = "mle_temperature"
Z_AT_ONE = "z_at_one"
AUC_FIELDS: tuple[str, ...] = (
    "target", "feature", "tokens", "auc", "tpr_at_5pct_fpr", "mean_target", "mean_clean", "sd_clean",
)
FALSE_POSITIVE_RATE = 0.05


class CalibrationError(HackathonError):
    """The row table is inconsistent (misaligned containers, unknown endpoint)."""


@dataclass(frozen=True)
class RowFeatureTable:
    """Per-row sums needed to build features for any subset of rows.

    Attributes:
        keys: `(endpoint, prompt id, repeat)` of every row.
        n_tokens: Reply tokens of each row under the base model.
        temperatures: Sorted temperature grid, `1.0` included.
        lp: `(rows, len(temperatures))` summed log-probabilities per temperature.
        ent: Same shape, summed predictive entropies.
        var: Same shape, summed predictive variances of log p.
        llr: Per fp8 variant, the per-row sum of `log p_variant - log p_base`.
    """

    keys: tuple[tuple[str, str, int], ...]
    n_tokens: IntArray
    temperatures: tuple[float, ...]
    lp: FloatArray
    ent: FloatArray
    var: FloatArray
    llr: Mapping[str, FloatArray]


def _row_sums_matrix(flat: FloatArray, offsets: IntArray) -> FloatArray:
    """Sum a `(k, tokens)` array over the token range of every row: `(rows, k)`."""
    cumulative = np.concatenate([np.zeros((flat.shape[0], 1)), np.cumsum(flat, axis=1)], axis=1)
    return np.asarray((cumulative[:, offsets[1:]] - cumulative[:, offsets[:-1]]).T, dtype=np.float64)


def build_row_table(
    keys: Sequence[tuple[str, str, int]], matrix_scores: RowScores,
    llr_by_variant: Mapping[str, FloatArray],
) -> RowFeatureTable:
    """Assemble the table from a matrix-stage score (tempered statistics) and per-row LLR sums.

    Raises:
        CalibrationError: The scores carry no tempered statistics, or an LLR array has the wrong length.
    """
    if matrix_scores.t_logp is None or matrix_scores.t_ent is None or matrix_scores.t_var is None:
        raise CalibrationError("matrix scores have no tempered statistics")
    n_rows = len(keys)
    for variant, values in llr_by_variant.items():
        if len(values) != n_rows:
            raise CalibrationError(f"{variant}: {len(values)} LLR rows for {n_rows} keys")
    offsets = matrix_scores.offsets()
    temperatures = [*matrix_scores.temperatures, UNIT_TEMPERATURE]
    lp = np.vstack([matrix_scores.t_logp, matrix_scores.logp[None, :]]).astype(np.float64)
    ent = np.vstack([matrix_scores.t_ent, matrix_scores.ent[None, :]]).astype(np.float64)
    var = np.vstack([matrix_scores.t_var, matrix_scores.var[None, :]]).astype(np.float64)
    order = np.argsort(temperatures)
    return RowFeatureTable(
        keys=tuple(keys),
        n_tokens=matrix_scores.n_tokens.astype(np.int64),
        temperatures=tuple(float(temperatures[i]) for i in order),
        lp=_row_sums_matrix(lp[order], offsets), ent=_row_sums_matrix(ent[order], offsets),
        var=_row_sums_matrix(var[order], offsets),
        llr={variant: np.asarray(values, dtype=np.float64) for variant, values in llr_by_variant.items()},
    )


def features_of_rows(table: RowFeatureTable, rows: IntArray) -> dict[str, float]:
    """Likelihood features of the union of `rows` (indices into the table)."""
    n_tokens = int(table.n_tokens[rows].sum())
    profile = temperature_profile_from_sums(
        table.temperatures, table.lp[rows].sum(axis=0), table.ent[rows].sum(axis=0),
        table.var[rows].sum(axis=0), n_tokens,
    )
    unit = profile.temperatures.index(UNIT_TEMPERATURE)
    features = {
        MLE_TEMPERATURE: profile.mle_temperature,
        Z_AT_ONE: float(profile.z[unit]),
        MEAN_LL: float(profile.total_logp[unit] / max(n_tokens, 1)),
    }
    for variant, values in table.llr.items():
        features[f"llr_{variant}"] = float(values[rows].sum() / max(n_tokens, 1))
    return features


def draw_rows(
    table: RowFeatureTable, candidates: IntArray, token_budget: int, rng: np.random.Generator
) -> IntArray:
    """Random rows among `candidates` (without replacement) until `token_budget` tokens are reached."""
    order = rng.permutation(candidates)
    cumulative = np.cumsum(table.n_tokens[order])
    count = int(np.searchsorted(cumulative, token_budget)) + 1
    return np.asarray(order[: min(count, len(order))], dtype=np.int64)


def draw_features(
    table: RowFeatureTable, endpoint: str, token_budget: int, draws: int, rng: np.random.Generator
) -> list[dict[str, float]]:
    """Features of `draws` pseudo-endpoints of `token_budget` tokens drawn from a ref endpoint.

    Raises:
        CalibrationError: The endpoint has no rows in the table.
    """
    candidates = np.array(
        [i for i, key in enumerate(table.keys) if key[0] == endpoint and table.n_tokens[i] > 0],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise CalibrationError(f"no rows for {endpoint}")
    return [
        features_of_rows(table, draw_rows(table, candidates, token_budget, rng))
        for _ in range(draws)
    ]


def auc(target: FloatArray, clean: FloatArray) -> float:
    """Probability that a random target draw has a larger value than a random clean draw (ties 0.5)."""
    if target.size == 0 or clean.size == 0:
        return float("nan")
    combined = np.concatenate([target, clean])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    # average the ranks of ties
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    u_stat = ranks[: target.size].sum() - target.size * (target.size + 1) / 2.0
    return float(u_stat / (target.size * clean.size))


def tpr_at_fpr(target: FloatArray, clean: FloatArray, fpr: float = FALSE_POSITIVE_RATE) -> float:
    """True-positive rate at a false-positive rate, the direction chosen by the class means.

    The threshold is the `1 - fpr` quantile of the clean draws when the target is above the clean
    draws on average, the `fpr` quantile otherwise (the feature then rejects on the low side).
    """
    if target.size == 0 or clean.size == 0:
        return float("nan")
    if float(target.mean()) >= float(clean.mean()):
        return float(np.mean(target > np.quantile(clean, 1.0 - fpr)))
    return float(np.mean(target < np.quantile(clean, fpr)))


def separation_table(
    table: RowFeatureTable, *, budgets: Sequence[int] = TOKEN_BUDGETS, draws: int = DRAWS_PER_ENDPOINT,
    seed: int = CALIBRATION_SEED, class_of: Mapping[str, str] = CLASS_OF,
    clean_classes: Sequence[str] = CLEAN_CLASSES, target_classes: Sequence[str] = TARGET_CLASSES,
) -> list[dict[str, float | str | int]]:
    """AUC and true-positive rate at 5 % false positives of every feature, per target class and budget.

    `class_of` maps an endpoint of the table to its class. The AUC is raw (0.5 means no separation,
    far from 0.5 either way means the feature separates the class; the sign of `mean_target -
    mean_clean` gives the direction). The true-positive rate is oriented by that sign.
    """
    endpoints = {klass: endpoint for endpoint, klass in class_of.items()}
    rows: list[dict[str, float | str | int]] = []
    for budget in budgets:
        rng = np.random.default_rng(seed + budget)
        by_class: dict[str, list[dict[str, float]]] = {}
        for klass in (*clean_classes, *target_classes):
            by_class[klass] = draw_features(table, endpoints[klass], budget, draws, rng)
        features = sorted(by_class[clean_classes[0]][0])
        for feature in features:
            clean = np.array([d[feature] for k in clean_classes for d in by_class[k]])
            valid_clean = clean[~np.isnan(clean)]
            for klass in target_classes:
                target = np.array([d[feature] for d in by_class[klass]])
                valid_target = target[~np.isnan(target)]
                rows.append({
                    "target": klass, "feature": feature, "tokens": budget,
                    "auc": round(auc(valid_target, valid_clean), 3),
                    "tpr_at_5pct_fpr": round(tpr_at_fpr(valid_target, valid_clean), 3),
                    "mean_target": round(float(valid_target.mean()), 5) if valid_target.size else float("nan"),
                    "mean_clean": round(float(valid_clean.mean()), 5) if valid_clean.size else float("nan"),
                    "sd_clean": round(float(valid_clean.std()), 5) if valid_clean.size else float("nan"),
                })
    return rows


def write_table(path: Path, rows: Sequence[Mapping[str, float | str | int]]) -> None:
    """Write the separation table as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
