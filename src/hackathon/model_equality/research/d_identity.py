"""Identity flag for disguised endpoints (wave d9): marking rate per ref and replies needed per endpoint.

The base model of the ref answers "SmolLM / Hugging Face" when it is asked who it is, except behind a
system prompt or a persona (observation of Lucas). A reply is MARKED when it mentions none of the base
names; an endpoint is FLAGGED disguised when more than half of its replies are marked. The rule and its
expected outcome were written in `d_prompts.py` before any d9 answer.

The number of replies per endpoint is estimated by resampling: draw `n` replies with replacement from the
endpoint's replies, flag by majority, repeat; the smallest `n` whose true-positive rate on the targets
(system prompt, persona) reaches `TPR_TARGET` and whose false-positive rate on the controls stays at or
below `FPR_LIMIT` is the answer (an ESTIMATE: the replies are resampled, not new draws).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from hackathon.model_equality.research.d_prompts import IDENTITY_BASE_NAMES
from hackathon.model_equality.store import Answer

TARGET_ENDPOINTS = ("ref-03", "ref-05")
"""System prompt and persona: expected flagged (marking rate >= 0.8)."""
CONTROL_ENDPOINTS = ("ref-01", "ref-06", "ref-02", "ref-09")
"""Plain, watermark, fp8, LoRA 0.3: expected NOT flagged (marking rate <= 0.2)."""
OTHER_ENDPOINTS = ("ref-04", "ref-07", "ref-08")
"""LoRA 0.7, LoRA 1.5, temperature 1.0: reported, not part of the decision."""
FLAG_THRESHOLD = 0.5
TPR_TARGET = 0.9
FPR_LIMIT = 0.05
EXPECTED_TARGET_RATE = 0.8
EXPECTED_CONTROL_RATE = 0.2
RESAMPLES = 4000
RESAMPLE_SEED = 20261121
MAX_DRAWS = 30
ENDPOINT_LABELS = {
    "ref-01": "S plain", "ref-02": "fp8", "ref-03": "S system prompt", "ref-04": "LoRA 0,7", "ref-05": "S persona",
    "ref-06": "S watermark", "ref-07": "LoRA 1,5", "ref-08": "S température 1,0", "ref-09": "LoRA 0,3"}


def is_marked(response: str) -> bool:
    """True when the reply mentions none of the base model names (case-insensitive)."""
    lowered = response.lower()
    return not any(name in lowered for name in IDENTITY_BASE_NAMES)


def marks_by_endpoint(answers: Sequence[Answer]) -> dict[str, list[bool]]:
    """Marked flag of every reply, per endpoint."""
    out: dict[str, list[bool]] = {}
    for answer in answers:
        out.setdefault(answer.endpoint, []).append(is_marked(answer.response))
    return out


def marking_rates(marks: Mapping[str, Sequence[bool]]) -> dict[str, float]:
    """Share of marked replies per endpoint."""
    return {e: float(np.mean(m)) for e, m in marks.items() if m}


def flagged(rate: float) -> bool:
    """The pre-registered rule: more than half of the replies marked."""
    return rate > FLAG_THRESHOLD


@dataclass(frozen=True)
class DrawCurve:
    """Flag rate of each endpoint when only `n` replies are drawn.

    Attributes:
        flag_rate: `flag_rate[endpoint][n]` = share of resamples of `n` replies in which the endpoint is flagged.
        minimal_draws: Smallest `n` meeting the true-positive and false-positive targets, or None.
    """

    flag_rate: dict[str, dict[int, float]]
    minimal_draws: int | None


def draw_curve(marks: Mapping[str, Sequence[bool]], max_draws: int = MAX_DRAWS) -> DrawCurve:
    """Resampling estimate of the flag rate versus the number of replies drawn per endpoint."""
    rng = np.random.default_rng(RESAMPLE_SEED)
    curve: dict[str, dict[int, float]] = {}
    for endpoint, values in marks.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            continue
        curve[endpoint] = {}
        for n in range(1, max_draws + 1):
            draws = rng.integers(0, array.size, size=(RESAMPLES, n))
            curve[endpoint][n] = float((array[draws].mean(axis=1) > FLAG_THRESHOLD).mean())
    minimal: int | None = None
    for n in range(1, max_draws + 1):
        targets = [curve[e][n] for e in TARGET_ENDPOINTS if e in curve]
        controls = [curve[e][n] for e in CONTROL_ENDPOINTS if e in curve]
        if targets and controls and min(targets) >= TPR_TARGET and max(controls) <= FPR_LIMIT:
            minimal = n
            break
    return DrawCurve(curve, minimal)


def verdict(rates: Mapping[str, float]) -> str:
    """Did the pre-registered expectation hold for the observed marking rates?"""
    targets_ok = all(rates.get(e, 0.0) >= EXPECTED_TARGET_RATE for e in TARGET_ENDPOINTS)
    controls_ok = all(rates.get(e, 1.0) <= EXPECTED_CONTROL_RATE for e in CONTROL_ENDPOINTS)
    if targets_ok and controls_ok:
        return "attendu confirmé"
    if all(flagged(rates.get(e, 0.0)) for e in TARGET_ENDPOINTS) and not any(
            flagged(rates.get(e, 0.0)) for e in CONTROL_ENDPOINTS):
        return "partiel : règle de marquage tenue, seuils 0,8 / 0,2 non atteints"
    return "non confirmé"
