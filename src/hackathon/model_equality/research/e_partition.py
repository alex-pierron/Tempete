"""Proposed partition of the ep endpoints: families (exact base checkpoint) and model classes.

FAMILY = the identified base checkpoint (`e_matrix`), which is the unit of the "same family" label
of the task (same base model, whatever the quantisation or the LoRA).

MODEL CLASS inside a family, from three weak read-outs (each with the power measured on the ref
endpoints in `e_calib`, at the ~560 tokens the ep endpoints have today):

* disguise by identity: the answer to the identity prompt names none of the base-model names
  (rule v3 of team C, `c_identity.is_marked`) -> system prompt or persona, STAYS in the main class;
* temperature-1.0-like: the fitted temperature is ~1 and the calibration z at 1 is ~0 (ref-08 gives
  T = 1.003, z = -1.4) -> a decoding setting, STAYS in the main class;
* deviating and undisguised: the fitted temperature is well above the default 0.72 -> LoRA candidate
  (ref LoRA 0.7 / 1.5 give 0.81 / 0.94), ISOLATED;
* full-stop signature relative to the family (LoRA 0.3 of team D: a full stop at the end of short
  answers) -> isolated unless disguised;
* fp8 candidate: the best simulated-fp8 likelihood ratio of the family among the endpoints left in
  the main class, when it clears a margin -> separate class (weak: AUC 0.8 at 560 tokens).

Every threshold is a named constant with the ref number that motivates it. The output is a
PROPOSAL with a confidence label per endpoint; the score only rewards it when the labels are right.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.c_identity import is_marked
from hackathon.model_equality.store import Answer

DEFAULT_TEMPERATURE = 0.72
"""Fitted temperature of the plain ref endpoint (ref-01: 0.723, ref-06 watermark: 0.727)."""
REFERENCE_TOKENS = 560
"""Token count of an ep endpoint today (7 answers); the sigmas below are quoted at this size and
shrink as `1 / sqrt(tokens)`."""
TEMPERATURE_SIGMA_AT_560_TOKENS = 0.035
"""One-sigma error of the fitted temperature with ~560 tokens (from the likelihood curvature)."""
DEVIATION_SIGMAS = 2.0
"""A fitted temperature this many sigmas above the default counts as deviating."""
DEVIATION_MARGIN = DEVIATION_SIGMAS * TEMPERATURE_SIGMA_AT_560_TOKENS
TEMPERATURE_ONE_SIGMAS = 1.5
"""A temperature-1.0-like endpoint has a fitted temperature at most this many sigmas below 1.0 (0.9475
at 563 tokens, 0.979 at 3 500). ep-33 (0.930, z = 2.1) sits 2 sigma below 1.0, like the ref LoRA 1.5
(0.943), and is NOT temperature-1.0-like; ep-23, 30 and 31 (0.99 to 1.01) are."""
TEMPERATURE_ONE_MAX_ABS_Z = 1.5
"""Largest |z at 1| of a temperature-1.0-like endpoint (ref-08: 1.003, z = -1.4 over 9 900 tokens;
z is N(0, 1) at any sample size for text sampled at T = 1)."""
FP8_LLR_MARGIN_SIGMA = 2.5
"""Sigmas the best fp8 ratio of a family must clear above the family median. Frozen at 2.5 with team
C: on the real ref the score curve is flat between margins 1.0 and 3.0 (`partition_sim_ref.csv`), and
a higher margin cuts the false positives on clean endpoints."""
FP8_LLR_SIGMA_AT_560_TOKENS = 0.0058
"""Standard error of the per-token fp8 log-likelihood ratio at ~560 tokens (0.0013 at 10 900 tokens
on the ref, scaled by sqrt(10 900 / 560))."""
IDENTITY_PROMPT_ID = "fam-identity-000"
FP8_LLR_VARIANT = "w_tensor"
FLAG_DISGUISED = "disguised_identity"
FLAG_TEMP_ONE = "temperature_one_like"
FLAG_DEVIATING = "deviating_temperature"
FLAG_FP8 = "fp8_candidate"
FLAG_FULL_STOP = "full_stop_signature"
FULL_STOP_PROMPT_PREFIX = "cand-yesno"
FULL_STOP_SHARE_MIN = 0.5
MODEL_MAIN = "main"
MODEL_FP8 = "fp8"
MODEL_VARIANT = "variant"

PARTITION_FIELDS: tuple[str, ...] = (
    "endpoint", "family", "family_confidence", "model_id", "model_confidence", "flags",
    "n_tokens", "mle_temperature", "z_at_one", "fp8_llr_w_tensor", "fp8_llr_z_vs_family",
    "identity_marked_share", "identity_answer",
)


class PartitionError(HackathonError):
    """Missing input for the partition (no identification row, no temperature profile)."""


@dataclass(frozen=True)
class EndpointEvidence:
    """Everything the rules read for one endpoint.

    `n_tokens` is the number of reply tokens behind `mle_temperature`, `z_at_one` and `fp8_llr` (the
    sigmas of the rules scale with it; None means the `tokens` argument of `propose`).
    `identity_answers` lists every answer to an identity prompt; when present, the endpoint is
    disguised if strictly more than half of them name no base model (rule v3 of team C), and the
    single `identity_answer` is the fallback.
    """

    endpoint: str
    family: str
    family_confidence: str
    mle_temperature: float
    z_at_one: float
    fp8_llr: float | None
    identity_answer: str
    full_stop_share: float | None
    n_tokens: int | None = None
    identity_answers: tuple[str, ...] = ()


IDENTITY_PROMPT_PREFIXES: tuple[str, ...] = ("fam-identity", "d-ident")
"""Prompt ids of the identity questions: the E1 one and the three formulations of team D."""
DISGUISED_MARKED_SHARE = 0.5
"""Disguised iff the share of marked identity answers is STRICTLY above this (rule v3)."""


def is_identity_disguised(item: EndpointEvidence) -> bool:
    """Rule v3 on all the identity answers of an endpoint, or on the single one when there is no list."""
    if item.identity_answers:
        marked = sum(1 for answer in item.identity_answers if is_marked(answer))
        return marked / len(item.identity_answers) > DISGUISED_MARKED_SHARE
    return bool(item.identity_answer) and is_marked(item.identity_answer)


def all_identity_answers(answers: Sequence[Answer]) -> dict[str, tuple[str, ...]]:
    """`endpoint -> every answer to an identity prompt` (all repeats, stored order)."""
    grouped: dict[str, list[str]] = {}
    for answer in answers:
        if answer.prompt_id.startswith(IDENTITY_PROMPT_PREFIXES):
            grouped.setdefault(answer.endpoint, []).append(answer.response)
    return {endpoint: tuple(values) for endpoint, values in grouped.items()}


def identity_answers(answers: Sequence[Answer]) -> dict[str, str]:
    """`endpoint -> its answer to the identity prompt` (first repeat)."""
    return {
        a.endpoint: a.response for a in answers
        if a.prompt_id == IDENTITY_PROMPT_ID and a.repeat == 0
    }


def full_stop_shares(answers: Sequence[Answer]) -> dict[str, float]:
    """Share of the yes/no answers of each endpoint that end with a full stop (LoRA 0.3 signature)."""
    counts: dict[str, list[bool]] = {}
    for answer in answers:
        if answer.prompt_id.startswith(FULL_STOP_PROMPT_PREFIX):
            counts.setdefault(answer.endpoint, []).append(answer.response.strip().endswith("."))
    return {endpoint: float(np.mean(flags)) for endpoint, flags in counts.items()}


def token_scale(tokens: int) -> float:
    """`sqrt(560 / tokens)`: how the sigmas quoted at 560 tokens shrink with more tokens."""
    return math.sqrt(REFERENCE_TOKENS / max(tokens, 1))


def temperature_one_threshold(tokens: int = REFERENCE_TOKENS) -> float:
    """Smallest fitted temperature of a temperature-1.0-like endpoint at `tokens` tokens."""
    return 1.0 - TEMPERATURE_ONE_SIGMAS * TEMPERATURE_SIGMA_AT_560_TOKENS * token_scale(tokens)


def is_temperature_one_like(
    mle_temperature: float, z_at_one: float, tokens: int = REFERENCE_TOKENS,
    min_temperature: float | None = None, max_abs_z: float = TEMPERATURE_ONE_MAX_ABS_Z,
) -> bool:
    """Fitted temperature near 1 and a calibrated z at 1 (a decoding setting, same model).

    The temperature bound is `temperature_one_threshold(tokens)` unless `min_temperature` is given.
    """
    floor = temperature_one_threshold(tokens) if min_temperature is None else min_temperature
    return mle_temperature >= floor and abs(z_at_one) <= max_abs_z


def is_deviating(
    mle_temperature: float, z_at_one: float, tokens: int = REFERENCE_TOKENS,
    temperature_one_min: float | None = None,
    temperature_one_max_abs_z: float = TEMPERATURE_ONE_MAX_ABS_Z,
) -> bool:
    """Fitted temperature well above the default and not temperature-1.0-like."""
    margin = DEVIATION_SIGMAS * TEMPERATURE_SIGMA_AT_560_TOKENS * token_scale(tokens)
    return (
        mle_temperature - DEFAULT_TEMPERATURE >= margin
        and not is_temperature_one_like(
            mle_temperature, z_at_one, tokens, temperature_one_min, temperature_one_max_abs_z)
    )


def fp8_z_scores(
    evidence: Sequence[EndpointEvidence], eligible: Sequence[str], tokens: int = REFERENCE_TOKENS,
) -> dict[str, float]:
    """fp8 log-likelihood ratio of each eligible endpoint relative to the median of its family.

    Expressed in standard errors at ~560 tokens; the median is taken over the family's eligible
    endpoints (those still in the main class), so deviants do not drag the baseline.
    """
    by_family: dict[str, list[EndpointEvidence]] = {}
    for item in evidence:
        if item.endpoint in eligible and item.fp8_llr is not None:
            by_family.setdefault(item.family, []).append(item)
    scores: dict[str, float] = {}
    for items in by_family.values():
        median = float(np.median([i.fp8_llr for i in items if i.fp8_llr is not None]))
        for item in items:
            assert item.fp8_llr is not None
            scores[item.endpoint] = (item.fp8_llr - median) / (
                FP8_LLR_SIGMA_AT_560_TOKENS * token_scale(item.n_tokens or tokens)
            )
    return scores


def family_full_stop_medians(evidence: Sequence[EndpointEvidence]) -> dict[str, float]:
    """Median full-stop share of each family: the baseline of that base model's own style.

    Llama-3.2-1B ends almost every short answer with a full stop on its own, so a full stop only
    means something relative to the family.
    """
    by_family: dict[str, list[float]] = {}
    for item in evidence:
        if item.full_stop_share is not None:
            by_family.setdefault(item.family, []).append(item.full_stop_share)
    return {family: float(np.median(values)) for family, values in by_family.items()}


def propose(
    evidence: Sequence[EndpointEvidence], tokens: int = REFERENCE_TOKENS,
    fp8_margin_sigma: float = FP8_LLR_MARGIN_SIGMA,
    temperature_one_min: float | None = None,
    temperature_one_max_abs_z: float = TEMPERATURE_ONE_MAX_ABS_Z,
) -> list[dict[str, object]]:
    """Apply the rules and return one table row per endpoint (see the module docstring).

    Args:
        evidence: One item per endpoint.
        tokens: Tokens per endpoint, to scale the sigmas (560 for the ep endpoints today).
        fp8_margin_sigma: Sigmas above the family median the best fp8 likelihood ratio must clear.
        temperature_one_min: Smallest fitted temperature of a temperature-1.0-like endpoint
            (default: `temperature_one_threshold(tokens)`).
        temperature_one_max_abs_z: Largest |z at 1| of a temperature-1.0-like endpoint.
    """
    flags: dict[str, list[str]] = {item.endpoint: [] for item in evidence}
    stop_baseline = family_full_stop_medians(evidence)
    for item in evidence:
        if is_identity_disguised(item):
            flags[item.endpoint].append(FLAG_DISGUISED)
        if is_temperature_one_like(
            item.mle_temperature, item.z_at_one, item.n_tokens or tokens, temperature_one_min,
            temperature_one_max_abs_z,
        ):
            flags[item.endpoint].append(FLAG_TEMP_ONE)
        elif is_deviating(
            item.mle_temperature, item.z_at_one, item.n_tokens or tokens, temperature_one_min,
            temperature_one_max_abs_z,
        ):
            flags[item.endpoint].append(FLAG_DEVIATING)
        if (
            item.full_stop_share is not None
            and item.full_stop_share >= FULL_STOP_SHARE_MIN
            and stop_baseline.get(item.family, 1.0) < FULL_STOP_SHARE_MIN
        ):
            flags[item.endpoint].append(FLAG_FULL_STOP)
    eligible = [
        item.endpoint for item in evidence
        if not {FLAG_DISGUISED, FLAG_TEMP_ONE, FLAG_DEVIATING, FLAG_FULL_STOP} & set(flags[item.endpoint])
    ]
    z_fp8 = fp8_z_scores(evidence, eligible, tokens)
    best_per_family: dict[str, str] = {}
    for item in evidence:
        if item.endpoint in z_fp8 and (
            item.family not in best_per_family
            or z_fp8[item.endpoint] > z_fp8[best_per_family[item.family]]
        ):
            best_per_family[item.family] = item.endpoint
    for endpoint in best_per_family.values():
        if z_fp8[endpoint] >= fp8_margin_sigma:
            flags[endpoint].append(FLAG_FP8)
    rows: list[dict[str, object]] = []
    for item in evidence:
        item_flags = flags[item.endpoint]
        disguised = FLAG_DISGUISED in item_flags
        if FLAG_FP8 in item_flags:
            model_id, confidence = f"{item.family}:{MODEL_FP8}", "low"
        elif not disguised and {FLAG_DEVIATING, FLAG_FULL_STOP} & set(item_flags):
            model_id, confidence = f"{item.family}:{MODEL_VARIANT}:{item.endpoint}", "low"
        else:
            model_id, confidence = f"{item.family}:{MODEL_MAIN}", "medium"
        rows.append({
            "endpoint": item.endpoint, "family": item.family,
            "family_confidence": item.family_confidence, "model_id": model_id,
            "model_confidence": confidence, "flags": "|".join(item_flags),
            "n_tokens": item.n_tokens, "mle_temperature": round(item.mle_temperature, 3),
            "z_at_one": round(item.z_at_one, 2),
            "fp8_llr_w_tensor": None if item.fp8_llr is None else round(item.fp8_llr, 5),
            "fp8_llr_z_vs_family": round(z_fp8[item.endpoint], 2) if item.endpoint in z_fp8 else None,
            "identity_marked_share": (
                round(sum(1 for a in item.identity_answers if is_marked(a)) / len(item.identity_answers), 2)
                if item.identity_answers else None
            ),
            "identity_answer": item.identity_answer[:80],
        })
    return rows


def write_partition(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write the proposal as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PARTITION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
