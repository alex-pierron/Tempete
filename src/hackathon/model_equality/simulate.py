"""Synthetic endpoints for testing the pipeline offline (NOT evidence about the real server).

A toy "language model" assigns, per (family, prompt, position), a logit vector over a small
vocabulary and samples positions independently with the Gumbel-max trick. Variants mimic the
task's axes with arbitrary, documented strengths:

* different family       independent logits;
* quantization / LoRA    the family logits plus Gaussian noise of a given strength (a LoRA of
                         strength s perturbs more than fp8: the strengths below are guesses);
* system prompt, persona a text prefix (persona: always, system: sometimes) that does not change
                         the underlying distribution, except for the first word;
* watermark              a fixed logit bias on a per-position "green" subset of the vocabulary;
* temperature            logits divided by a temperature different from the default.

This exists so the client, store, analysis, clustering and submission code can be exercised end
to end without the server (and so the offline tooling can be dry-run by the team). Passing on
synthetic data proves the software works; it says nothing about the real power of the method.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hackathon.model_equality.prompts import PromptRepeat
from hackathon.model_equality.prompts.bank import (
    CATEGORY_ARITH,
    CATEGORY_COUNT,
    CATEGORY_FORMAT,
    CATEGORY_FREEFORM,
)
from hackathon.model_equality.store import Answer

WORLD_SEED = 7
LOGIT_SCALE = 2.0
"""Std of the base logits: large, so near-deterministic categories are peaked."""
DEFAULT_TEMPERATURE = 1.0
HOT_TEMPERATURE = 1.3
WATERMARK_BIAS = 0.4
GREEN_FRACTION = 0.5
SYSTEM_PREFIX_PROBABILITY = 0.3
PERSONA_PREFIX_PROBABILITY = 0.6
PERSONA_PREFIX = "arr matey"
SYSTEM_PREFIX = "sure"
STRENGTH_FP8 = 0.6
STRENGTH_LORA_LOW = 0.8
STRENGTH_LORA_MID = 1.2
STRENGTH_LORA_HIGH = 1.8
SMALL_VOCABULARY = 8
LARGE_VOCABULARY = 24
POSITIONS_BY_CATEGORY = {CATEGORY_ARITH: 1, CATEGORY_FORMAT: 1}
DEFAULT_POSITIONS = 24
SHORT_POSITIONS = 6
NUMBER_BASE = 100
NUMBER_STEP = 7

KIND_PLAIN = "plain"
KIND_SYSTEM = "system"
KIND_PERSONA = "persona"
KIND_WATERMARK = "watermark"
KIND_TEMPERATURE = "temperature"


@dataclass(frozen=True)
class SimEndpoint:
    """A synthetic endpoint: truth labels plus its deployment disguise.

    Attributes:
        name: Endpoint id.
        family: Base-model label (truth).
        model: Model label inside the family (truth: checkpoint + quantization + LoRA).
        strength: Logit-noise strength of the model variant relative to the family base (0 = base).
        kind: Disguise (`KIND_*`); it never changes the truth.
    """

    name: str
    family: str
    model: str
    strength: float = 0.0
    kind: str = KIND_PLAIN


def _seed(*parts: object) -> int:
    """Deterministic 32-bit seed from heterogeneous parts."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def _shape(category: str) -> tuple[int, int]:
    """`(positions, vocabulary size)` of a prompt category."""
    if category in POSITIONS_BY_CATEGORY:
        return POSITIONS_BY_CATEGORY[category], SMALL_VOCABULARY
    if category == CATEGORY_FREEFORM:
        return DEFAULT_POSITIONS, LARGE_VOCABULARY
    if category == CATEGORY_COUNT:
        return SHORT_POSITIONS, SMALL_VOCABULARY
    return DEFAULT_POSITIONS, SMALL_VOCABULARY


def generate_text(
    endpoint: SimEndpoint, category: str, prompt_id: str, rng: np.random.Generator
) -> str:
    """Sample one synthetic answer of `endpoint` to a prompt."""
    positions, vocabulary = _shape(category)
    base = np.random.default_rng(_seed(WORLD_SEED, endpoint.family, prompt_id)).normal(
        0.0, LOGIT_SCALE, (positions, vocabulary))
    variant = np.random.default_rng(_seed(WORLD_SEED, endpoint.family, endpoint.model, prompt_id)
                                    ).normal(0.0, 1.0, (positions, vocabulary))
    logits = base + endpoint.strength * variant
    temperature = HOT_TEMPERATURE if endpoint.kind == KIND_TEMPERATURE else DEFAULT_TEMPERATURE
    if endpoint.kind == KIND_WATERMARK:
        green = np.random.default_rng(_seed(WORLD_SEED, "green", prompt_id)).random(
            (positions, vocabulary)) < GREEN_FRACTION
        logits = logits + WATERMARK_BIAS * green
    tokens = np.argmax(logits / temperature + rng.gumbel(size=logits.shape), axis=1)
    if category == CATEGORY_ARITH:
        words = [str(NUMBER_BASE + NUMBER_STEP * int(t)) for t in tokens]
    else:
        words = [f"w{int(t)}" for t in tokens]
    prefix_probability = {KIND_PERSONA: PERSONA_PREFIX_PROBABILITY,
                          KIND_SYSTEM: SYSTEM_PREFIX_PROBABILITY}.get(endpoint.kind, 0.0)
    prefixed = rng.random() < prefix_probability
    if prefixed:
        words.insert(0, PERSONA_PREFIX if endpoint.kind == KIND_PERSONA else SYSTEM_PREFIX)
    return " ".join(words)


def simulate_answers(
    endpoints: Sequence[SimEndpoint], units: Sequence[PromptRepeat], seed: int = WORLD_SEED
) -> list[Answer]:
    """One answer per (endpoint, unit), in the round-robin order the real runner uses."""
    answers: list[Answer] = []
    for unit in units:
        for endpoint in endpoints:
            rng = np.random.default_rng(
                _seed(seed, endpoint.name, unit.prompt.id, unit.repeat))
            text = generate_text(endpoint, unit.prompt.category, unit.prompt.id, rng)
            answers.append(Answer(endpoint.name, unit.prompt.id, unit.repeat, unit.prompt.text,
                                  text))
    return answers


def reference_endpoints() -> list[SimEndpoint]:
    """Nine synthetic `ref-*` endpoints laid out like `practice_answers.csv`."""
    return [
        SimEndpoint("ref-01", "P", "S", 0.0, KIND_PLAIN),
        SimEndpoint("ref-02", "P", "T1", STRENGTH_FP8, KIND_PLAIN),
        SimEndpoint("ref-03", "P", "S", 0.0, KIND_SYSTEM),
        SimEndpoint("ref-04", "P", "T3", STRENGTH_LORA_MID, KIND_PLAIN),
        SimEndpoint("ref-05", "P", "S", 0.0, KIND_PERSONA),
        SimEndpoint("ref-06", "P", "S", 0.0, KIND_WATERMARK),
        SimEndpoint("ref-07", "P", "T4", STRENGTH_LORA_HIGH, KIND_PLAIN),
        SimEndpoint("ref-08", "P", "S", 0.0, KIND_TEMPERATURE),
        SimEndpoint("ref-09", "P", "T2", STRENGTH_LORA_LOW, KIND_PLAIN),
    ]


def scored_endpoints() -> list[SimEndpoint]:
    """33 synthetic `ep-*` endpoints in three families (14 + 10 + 9), mixed disguises."""
    layout: list[tuple[str, str, float, str]] = []
    disguises = (KIND_PLAIN, KIND_SYSTEM, KIND_PERSONA, KIND_WATERMARK, KIND_TEMPERATURE)
    families = {
        "A": [("A1", 0.0, 5), ("A2", STRENGTH_FP8, 3), ("A3", STRENGTH_LORA_MID, 3),
              ("A4", STRENGTH_LORA_HIGH, 3)],
        "B": [("B1", 0.0, 5), ("B2", STRENGTH_FP8, 3), ("B3", STRENGTH_LORA_LOW, 2)],
        "C": [("C1", 0.0, 5), ("C2", STRENGTH_LORA_MID, 4)],
    }
    for family, models in families.items():
        for model, strength, count in models:
            layout.extend((family, model, strength, disguises[i % len(disguises)])
                          for i in range(count))
    return [SimEndpoint(f"ep-{n:02d}", family, model, strength, kind)
            for n, (family, model, strength, kind) in enumerate(layout, start=1)]
