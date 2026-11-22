"""Validation of the likelihood chain on the labelled ZOO of team F.

Team F generated, from local copies of the three ep bases and of SmolLM2-360M, ten known variants
(plain, system prompt, persona, watermark, temperature 1.0, LoRA 0.3 / 0.7 / 1.5, fp8 weights,
fp8 weights and activations). Each variant is an endpoint with a known truth, about 860 answers.
This module measures, per family and per token budget:

1. how well each feature separates a variant from the clean endpoints (`e_calib.separation_table`:
   AUC and true-positive rate at 5 % false positives);
2. the OFFICIAL score (0.9 x MCC_model) of the partition rules of `e_partition` on simulated
   families of 11 endpoints (the ten variants plus a second plain copy), at 560, 1 500 and 3 500
   tokens per endpoint, with and without the identity read-out, for several fp8 margins.

Caveat carried into every table: the fp8 of the zoo is SIMULATED with the same E4M3 recipes as ours,
so detection rates are an upper bound on what a different real implementation allows. The control
(SmolLM2-360M zoo against the real ref endpoints) says how well the simulation matches reality.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.e_calib import (
    MLE_TEMPERATURE,
    Z_AT_ONE,
    RowFeatureTable,
    draw_features,
)
from hackathon.model_equality.research.e_partition import (
    TEMPERATURE_ONE_MAX_ABS_Z,
    EndpointEvidence,
    propose,
)
from hackathon.model_equality.scoring import MODEL_WEIGHT, official_score

ZOO_CLASS_OF_VARIANT: Mapping[str, str] = {
    "plain": "plain", "watermark": "watermark", "sysprompt": "system", "persona": "persona",
    "temp1": "temp1", "lora03": "lora_0.3", "lora07": "lora_0.7", "lora15": "lora_1.5",
    "fp8_w": "fp8", "fp8_w8a8": "fp8_w8a8",
}
CLEAN_CLASSES: tuple[str, ...] = ("plain", "watermark")
TARGET_CLASSES: tuple[str, ...] = (
    "fp8", "fp8_w8a8", "lora_0.3", "lora_0.7", "lora_1.5", "system", "persona", "temp1",
)
TRUE_MODEL_OF_CLASS: Mapping[str, str] = {
    "plain": "base", "watermark": "base", "system": "base", "persona": "base", "temp1": "base",
    "fp8": "fp8", "fp8_w8a8": "fp8_w8a8", "lora_0.3": "lora_0.3", "lora_0.7": "lora_0.7",
    "lora_1.5": "lora_1.5",
}
DISGUISED_CLASSES: tuple[str, ...] = ("system", "persona")
IDENTITY_BASE = "I am Qwen, created by Alibaba Cloud."
IDENTITY_DISGUISED = "I was created by Anthropic and my name is Claude."
FAMILY_LABEL = "zoo"
SIM_TRIALS = 200
SIM_SEED = 20261122
SIM_BUDGETS: tuple[int, ...] = (560, 1500, 3500)
FP8_MARGINS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 1e9)
"""Sigmas the best fp8 likelihood ratio must clear; 1e9 disables the fp8 rule."""
MAX_TOKEN_SHARE = 0.8
"""A budget above this share of an endpoint's tokens would reuse the same rows in every draw."""
PLAIN_COPY = "plain_copy"


class ZooError(HackathonError):
    """The zoo metadata or tables are inconsistent."""


def endpoint_classes(meta: Mapping[str, object]) -> dict[str, str]:
    """`{zoo endpoint: class}` from a `meta_<family>.json` of team F.

    Raises:
        ZooError: A variant name is unknown.
    """
    variants = meta["variants"]
    assert isinstance(variants, dict)
    out: dict[str, str] = {}
    for name, info in variants.items():
        if name not in ZOO_CLASS_OF_VARIANT:
            raise ZooError(f"unknown zoo variant {name!r}")
        out[str(info["endpoint"])] = ZOO_CLASS_OF_VARIANT[name]
    return out


def load_endpoint_classes(path: Path) -> dict[str, str]:
    """Read a `meta_<family>.json` and map its endpoints to classes."""
    return endpoint_classes(json.loads(path.read_text(encoding="utf-8")))


def attainable_budgets(
    table: RowFeatureTable, class_of: Mapping[str, str], budgets: Sequence[int]
) -> list[int]:
    """Budgets that stay under `MAX_TOKEN_SHARE` of the smallest endpoint's total tokens."""
    smallest = min(
        int(table.n_tokens[[i for i, k in enumerate(table.keys) if k[0] == endpoint]].sum())
        for endpoint in class_of
    )
    return [b for b in budgets if b <= MAX_TOKEN_SHARE * smallest]


@dataclass(frozen=True)
class SimResult:
    """Outcome of the partition rules on simulated families.

    Attributes:
        budget: Tokens per endpoint.
        oracle_identity: Whether the identity read-out was given (it flags system / persona).
        fp8_margin_sigma: Margin of the fp8 rule.
        mean_score: Mean official score (0.9 x MCC_model) over the trials.
        sd_score: Standard deviation of the score over the trials.
        median_score: Median of the score over the trials.
        p10_score: 10th percentile of the score (lower end of the 80 % interval).
        p90_score: 90th percentile of the score (upper end of the 80 % interval).
        isolated_share: Per class, the share of trials in which its endpoint was isolated from the
            main model (for the `base`-model classes this is a false positive).
    """

    budget: int
    oracle_identity: bool
    fp8_margin_sigma: float
    mean_score: float
    sd_score: float
    isolated_share: Mapping[str, float]
    median_score: float = 0.0
    p10_score: float = 0.0
    p90_score: float = 0.0


FP8_CLASSES: tuple[str, ...] = ("fp8", "fp8_w8a8")
REAL_FP8_EFFECT_SCALE = 0.37
"""Real fp8 (ref-02) over simulated fp8: the per-token likelihood-ratio gap of the real endpoint is
0.37 of the simulated one (per-tensor weights, SmolLM2-360M; `fp8_realism.csv`)."""


def shrink_fp8_effect(
    draws: dict[str, list[dict[str, float]]], feature: str, scale: float
) -> None:
    """Pull the fp8 classes' `feature` toward the clean classes by `1 - scale` of their mean shift.

    The noise of the draws is kept: only the mean gap is scaled, which is how a milder real fp8 would
    look. Changes `draws` in place; `scale = 1` does nothing.
    """
    if scale == 1.0:
        return
    clean = float(
        np.mean([d[feature] for k in CLEAN_CLASSES if k in draws for d in draws[k] if feature in d])
    )
    for klass in FP8_CLASSES:
        if klass not in draws:
            continue
        shift = float(np.mean([d[feature] for d in draws[klass]])) - clean
        for draw in draws[klass]:
            draw[feature] -= (1.0 - scale) * shift


def simulate_families(
    table: RowFeatureTable, class_of: Mapping[str, str], budget: int, *, oracle_identity: bool,
    fp8_feature: str, fp8_margin_sigma: float, trials: int = SIM_TRIALS, seed: int = SIM_SEED,
    fp8_effect_scale: float = 1.0, exclude: Sequence[str] = (), temperature_one_min: float | None = None,
    temperature_one_max_abs_z: float = TEMPERATURE_ONE_MAX_ABS_Z,
) -> SimResult:
    """Score the partition rules on `trials` simulated families at `budget` tokens.

    The family has one endpoint per class of `class_of` (minus `exclude`) plus a second plain copy.

    `fp8_effect_scale` below 1 shrinks the fp8 likelihood-ratio gap (see `shrink_fp8_effect`).
    """
    endpoint_of_class = {
        klass: endpoint for endpoint, klass in class_of.items() if klass not in exclude
    }
    rng = np.random.default_rng(seed + budget)
    draws: dict[str, list[dict[str, float]]] = {
        klass: draw_features(table, endpoint, budget, trials, rng)
        for klass, endpoint in endpoint_of_class.items()
    }
    draws[PLAIN_COPY] = draw_features(table, endpoint_of_class["plain"], budget, trials, rng)
    shrink_fp8_effect(draws, fp8_feature, fp8_effect_scale)
    classes = [*endpoint_of_class, PLAIN_COPY]
    scores: list[float] = []
    isolated = {klass: 0 for klass in classes}
    for trial in range(trials):
        evidence = []
        for klass in classes:
            features = draws[klass][trial]
            disguised = oracle_identity and klass in DISGUISED_CLASSES
            evidence.append(EndpointEvidence(
                endpoint=klass, family=FAMILY_LABEL, family_confidence="high",
                mle_temperature=features[MLE_TEMPERATURE], z_at_one=features[Z_AT_ONE],
                fp8_llr=features.get(fp8_feature),
                identity_answer=IDENTITY_DISGUISED if disguised else IDENTITY_BASE,
                full_stop_share=None,
            ))
        rows = propose(
            evidence, tokens=budget, fp8_margin_sigma=fp8_margin_sigma,
            temperature_one_min=temperature_one_min, temperature_one_max_abs_z=temperature_one_max_abs_z,
        )
        predicted = {str(r["endpoint"]): str(r["model_id"]) for r in rows}
        truth = {
            klass: (FAMILY_LABEL, TRUE_MODEL_OF_CLASS.get(klass, "base")) for klass in classes
        }
        guess = {klass: (FAMILY_LABEL, predicted[klass]) for klass in classes}
        scores.append(MODEL_WEIGHT * official_score(truth, guess).model.clipped_mcc)
        for klass in classes:
            if not predicted[klass].endswith(":main"):
                isolated[klass] += 1
    return SimResult(
        budget=budget, oracle_identity=oracle_identity, fp8_margin_sigma=fp8_margin_sigma,
        mean_score=float(np.mean(scores)), sd_score=float(np.std(scores)),
        median_score=float(np.median(scores)), p10_score=float(np.percentile(scores, 10)),
        p90_score=float(np.percentile(scores, 90)),
        isolated_share={klass: isolated[klass] / trials for klass in classes},
    )
