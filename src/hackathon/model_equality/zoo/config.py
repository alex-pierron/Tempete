"""Names, seeds, the variant table and the family registry of the zoo (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

ZOO_SEED: Final[int] = 20261122
"""Master seed of every random draw of the zoo (sampling, LoRA data order, persona choice)."""
ENDPOINT_PREFIX: Final[str] = "zoo"
"""Endpoint ids are `zoo-<family>-<slot>`; the slot is the ref number of the same variant."""

# --- sampling -------------------------------------------------------------------------------
DEFAULT_TEMPERATURE: Final[float] = 0.7
"""Server default temperature of the ref endpoints: 0.723 for ref-01 and 1.003 for ref-08 (the
check) as measured by team E under SmolLM2-360M, the exact ref model (the 0.9 estimated by team D
under the 135M copy was the wrong model). Applied to every variant but `temp1`."""
TEMPERATURE_ONE: Final[float] = 1.0
"""The `temperature 1.0` ref endpoint."""
TOP_P_OFF: Final[float] = 1.0
TOP_K_OFF: Final[int] = 0
"""The ref server exposes temperature only: pure temperature sampling, no nucleus, no top-k."""
MAX_NEW_TOKENS: Final[int] = 256
"""Task limit on generated tokens per answer."""
MIN_BATCH_SIZE: Final[int] = 4
"""Smallest generation / training micro-batch the out-of-memory fallback goes down to."""

# --- fp8 ------------------------------------------------------------------------------------
E4M3_MAX: Final[float] = 448.0
"""Largest finite value of float8 E4M3 (`torch.float8_e4m3fn`)."""
MIN_SCALE_DENOMINATOR: Final[float] = 1e-12
"""Floor of a quantisation scale so that an all-zero tensor does not divide by zero."""
LINEAR_TARGETS: Final[frozenset[str]] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
"""Attention and MLP projections: quantised by fp8 and carrying the LoRA; embeddings and lm_head
stay in bf16 (as vLLM `quantization=fp8` does)."""

# --- LoRA -----------------------------------------------------------------------------------
LORA_RANK: Final[int] = 16
LORA_ALPHA: Final[float] = 32.0
LORA_LEARNING_RATE: Final[float] = 2e-4
LORA_WARMUP_STEPS: Final[int] = 10
LORA_TRAIN_STEPS: Final[int] = 150
LORA_EFFECTIVE_BATCH: Final[int] = 32
LORA_MAX_TOKENS: Final[int] = 256
"""Examples longer than this (prompt + answer) are dropped from the training set."""
LORA_GRAD_CLIP: Final[float] = 1.0
LORA_DATASET_REPO: Final[str] = "yahma/alpaca-cleaned"
LORA_DATASET_FILE: Final[str] = "alpaca_data_cleaned.json"
LORA_SCALES: Final[tuple[float, ...]] = (0.3, 0.7, 1.5)
"""Strengths at which the ONE trained adapter is applied (the three LoRA ref endpoints)."""

# --- prompts --------------------------------------------------------------------------------
REPEATS_ARITHMETIC: Final[int] = 4
"""Draws per arithmetic prompt (the ref holdout wave has 2; the zoo models are small, so more)."""
IDENTITY_DRAWS: Final[int] = 20
"""Draws per identity formulation (the ref identity wave has 10 per formulation)."""
EXTRA_ARITHMETIC_PROMPTS: Final[int] = 120
"""New one-line arithmetic prompts (same template as the ref ones) on top of the 80 committed."""
PERSONA_POOL_SIZE: Final[int] = 11

# --- likelihood scoring ---------------------------------------------------------------------
SCORING_BATCH_SIZE: Final[int] = 64


@dataclass(frozen=True)
class VariantSpec:
    """One of the nine ref variants (plus the W8A8 bound), as a zoo endpoint.

    Attributes:
        slot: ref number of the same variant (`ref-0<slot>`); 10 is the extra W8A8 endpoint.
        key: short machine name.
        label: human label (English).
        truth_model: model label within the family (equal labels = same model = the truth).
        needs_lora: the answers come from the adapter at `lora_scale`.
        lora_scale: strength of the adapter, 0.0 when not used.
        temperature_one: sampled at temperature 1.0 instead of the server default.
    """

    slot: int
    key: str
    label: str
    truth_model: str
    needs_lora: bool = False
    lora_scale: float = 0.0
    temperature_one: bool = False


MODEL_MAIN: Final[str] = "S"
MODEL_FP8: Final[str] = "T1_fp8"
MODEL_LORA_03: Final[str] = "T2_lora03"
MODEL_LORA_07: Final[str] = "T3_lora07"
MODEL_LORA_15: Final[str] = "T4_lora15"

VARIANTS: Final[tuple[VariantSpec, ...]] = (
    VariantSpec(1, "plain", "plain", MODEL_MAIN),
    VariantSpec(2, "fp8_w", "fp8 (weights E4M3 per tensor)", MODEL_FP8),
    VariantSpec(3, "sysprompt", "system prompt", MODEL_MAIN),
    VariantSpec(4, "lora07", "LoRA 0.7", MODEL_LORA_07, needs_lora=True, lora_scale=0.7),
    VariantSpec(5, "persona", "persona", MODEL_MAIN),
    VariantSpec(6, "watermark", "watermark (KGW)", MODEL_MAIN),
    VariantSpec(7, "lora15", "LoRA 1.5", MODEL_LORA_15, needs_lora=True, lora_scale=1.5),
    VariantSpec(8, "temp1", "temperature 1.0", MODEL_MAIN, temperature_one=True),
    VariantSpec(9, "lora03", "LoRA 0.3", MODEL_LORA_03, needs_lora=True, lora_scale=0.3),
    VariantSpec(10, "fp8_w8a8", "fp8 W8A8 (bound)", MODEL_FP8),
)
BASE_SLOTS: Final[tuple[int, ...]] = tuple(range(1, 10))
"""The nine slots of the official-like endpoint set (the W8A8 bound is an alternative to slot 2)."""
VARIANT_BY_KEY: Final[dict[str, VariantSpec]] = {v.key: v for v in VARIANTS}
GENERATION_ORDER: Final[tuple[str, ...]] = (
    "plain", "sysprompt", "persona", "watermark", "temp1",
    "lora03", "lora07", "lora15", "fp8_w", "fp8_w8a8",
)
"""Cheap variants first, then the LoRA ones (after the adapter exists), fp8 last (it edits weights
in place and restores them from a CPU copy)."""


@dataclass(frozen=True)
class FamilySpec:
    """A model family of the zoo.

    Attributes:
        key: short machine name used in endpoint ids and file names (no dash).
        display: human name.
        hf_repo: public repository the weights come from (provenance only).
        cache_subdir: folder under the HF cache dir that holds the flat model files.
        generation_batch: sequences generated per `generate` call (halved on out-of-memory).
        scoring_batch: sequences per likelihood forward pass.
        train_micro_batch: LoRA micro-batch (halved on out-of-memory, accumulation compensates).
        is_control: the control family (the ref model itself), not an ep base model.
    """

    key: str
    display: str
    hf_repo: str
    cache_subdir: str
    generation_batch: int
    scoring_batch: int = SCORING_BATCH_SIZE
    train_micro_batch: int = 64
    is_control: bool = False


FAMILIES: Final[tuple[FamilySpec, ...]] = (
    FamilySpec("qwen15", "Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b",
               512),
    FamilySpec("qwen3", "Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3b", 384,
               train_micro_batch=48),
    FamilySpec("llama1", "Llama-3.2-1B-Instruct", "alpindale/Llama-3.2-1B-Instruct",
               "llama3.2-1b", 512),
    FamilySpec("smollm", "SmolLM2-360M-Instruct (ref model, control)",
               "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2-360m", 512, is_control=True),
)
FAMILY_BY_KEY: Final[dict[str, FamilySpec]] = {f.key: f for f in FAMILIES}


class ZooConfigError(ValueError):
    """An unknown family, variant or endpoint name."""


def endpoint_name(family_key: str, slot: int) -> str:
    """`zoo-<family>-<slot:02d>`, e.g. `zoo-qwen-03`."""
    if family_key not in FAMILY_BY_KEY:
        raise ZooConfigError(f"unknown family {family_key!r}")
    return f"{ENDPOINT_PREFIX}-{family_key}-{slot:02d}"


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """Inverse of `endpoint_name`: `(family key, slot)`.

    Raises:
        ZooConfigError: the id is not a zoo endpoint.
    """
    parts = endpoint.split("-")
    if len(parts) != 3 or parts[0] != ENDPOINT_PREFIX or parts[1] not in FAMILY_BY_KEY:
        raise ZooConfigError(f"not a zoo endpoint: {endpoint!r}")
    try:
        return parts[1], int(parts[2])
    except ValueError as error:
        raise ZooConfigError(f"not a zoo endpoint: {endpoint!r}") from error


def variant_for_slot(slot: int) -> VariantSpec:
    """The variant of a slot number.

    Raises:
        ZooConfigError: no variant has that slot.
    """
    for variant in VARIANTS:
        if variant.slot == slot:
            return variant
    raise ZooConfigError(f"no variant has slot {slot}")
