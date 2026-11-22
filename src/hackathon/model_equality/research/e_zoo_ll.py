"""fp8 and mean-likelihood read-outs on the zoo from team F's own `ll_<family>.json.gz` (no GPU).

Team F scored every zoo answer under the base model (bf16) and under two simulated fp8 copies
(per-tensor weights; weights and activations), and stored per row `[sum, n_tokens]`: for `base` the
summed log-probability, for `fp8_w` and `fp8_w8a8` the log-likelihood RATIO against the base. From
these we get the two features that do not need the tempered statistics of `e_loglik`:

* `llr_fp8_w`, `llr_fp8_w8a8`  mean per-token log-likelihood ratio (the fp8 read-out);
* `mean_ll`                    mean per-token log-likelihood under the base (separates LoRA).

`e_zoo` has the full-feature version (temperature profile included), which needs our own GPU task.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_calib import auc, tpr_at_fpr
from hackathon.model_equality.research.e_zoo import (
    CLEAN_CLASSES,
    FP8_CLASSES,
    SIM_SEED,
    TARGET_CLASSES,
)

LL_FEATURES: tuple[str, ...] = ("llr_fp8_w", "llr_fp8_w8a8", "mean_ll")
LL_DRAWS = 300
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class ZooLlError(HackathonError):
    """A likelihood file of team F is incomplete."""


@dataclass(frozen=True)
class EndpointRows:
    """Per-row sums of one zoo endpoint."""

    n_tokens: IntArray
    base_logp: FloatArray
    llr_fp8_w: FloatArray
    llr_fp8_w8a8: FloatArray


def load_ll_rows(path: Path, family: str) -> dict[str, EndpointRows]:
    """Rows of a family's endpoints from team F's likelihood file (rows of other families skipped).

    Raises:
        ZooLlError: A row lacks one of the three scores.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        scores = json.load(handle)["scores"]
    prefix = f"zoo-{family}-"
    grouped: dict[str, list[tuple[int, float, float, float]]] = {}
    for key, (fp8_w_sum, n_tokens) in scores["fp8_w"].items():
        if not key.startswith(prefix):
            continue
        try:
            base_sum = scores["base"][key][0]
            w8a8_sum = scores["fp8_w8a8"][key][0]
        except KeyError as exc:
            raise ZooLlError(f"{key}: missing score") from exc
        grouped.setdefault(key.split("|", 1)[0], []).append(
            (int(n_tokens), float(base_sum), float(fp8_w_sum), float(w8a8_sum))
        )
    return {
        endpoint: EndpointRows(
            n_tokens=np.array([r[0] for r in rows], dtype=np.int64),
            base_logp=np.array([r[1] for r in rows]),
            llr_fp8_w=np.array([r[2] for r in rows]),
            llr_fp8_w8a8=np.array([r[3] for r in rows]),
        )
        for endpoint, rows in grouped.items()
    }


def ll_draw(rows: EndpointRows, budget: int, rng: np.random.Generator) -> dict[str, float]:
    """Per-token features of random rows reaching `budget` tokens."""
    order = rng.permutation(len(rows.n_tokens))
    count = min(int(np.searchsorted(np.cumsum(rows.n_tokens[order]), budget)) + 1, len(order))
    picked = order[:count]
    tokens = max(int(rows.n_tokens[picked].sum()), 1)
    return {
        "llr_fp8_w": float(rows.llr_fp8_w[picked].sum() / tokens),
        "llr_fp8_w8a8": float(rows.llr_fp8_w8a8[picked].sum() / tokens),
        "mean_ll": float(rows.base_logp[picked].sum() / tokens),
    }


def ll_separation_rows(
    rows_by_endpoint: Mapping[str, EndpointRows], class_of: Mapping[str, str],
    budgets: Sequence[int], *, draws: int = LL_DRAWS, seed: int = SIM_SEED,
    fp8_effect_scale: float = 1.0,
) -> list[dict[str, float | str | int]]:
    """AUC and true-positive rate at 5 % false positives of the fp8 and mean-likelihood read-outs.

    `fp8_effect_scale` below 1 shrinks the fp8 classes' ratio gap toward the clean endpoints (the
    real fp8 is milder than the simulated one), keeping the noise.
    """
    endpoint_of_class = {klass: endpoint for endpoint, klass in class_of.items()}
    out: list[dict[str, float | str | int]] = []
    for budget in budgets:
        rng = np.random.default_rng(seed + budget)
        by_class = {
            klass: [ll_draw(rows_by_endpoint[endpoint], budget, rng) for _ in range(draws)]
            for klass, endpoint in endpoint_of_class.items()
        }
        for feature in LL_FEATURES:
            clean = np.array([d[feature] for k in CLEAN_CLASSES for d in by_class[k]])
            clean_mean = float(clean.mean())
            for klass in TARGET_CLASSES:
                values = np.array([d[feature] for d in by_class[klass]])
                if klass in FP8_CLASSES and feature.startswith("llr") and fp8_effect_scale != 1.0:
                    values = values - (1.0 - fp8_effect_scale) * (float(values.mean()) - clean_mean)
                out.append({
                    "target": klass, "feature": feature, "tokens": budget,
                    "auc": round(auc(values, clean), 3),
                    "tpr_at_5pct_fpr": round(tpr_at_fpr(values, clean), 3),
                    "mean_target": round(float(values.mean()), 5),
                    "mean_clean": round(clean_mean, 5), "sd_clean": round(float(clean.std()), 5),
                })
    return out
