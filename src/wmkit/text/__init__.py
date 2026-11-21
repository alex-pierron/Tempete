"""LLM text watermarking: KGW and Gumbel-max schemes, detection, localization, attacks."""

from wmkit.text.attacks import AttackRateError, delete_tokens, insert_tokens, substitute_tokens
from wmkit.text.generate import (
    DEFAULT_LM,
    GenerationError,
    HFNextLogits,
    encode_text,
    generate,
    generate_many,
    load_lm,
    model_vocab_size,
    sample_sequences,
)
from wmkit.text.gumbel import (
    GumbelConfig,
    GumbelSampler,
    gumbel_detect,
    gumbel_token_scores,
    secret_vector,
)
from wmkit.text.kgw import (
    TEXT_DETECTION_P,
    KGWConfig,
    KGWLogitsProcessor,
    TextDetection,
    WatermarkConfigError,
    green_mask,
    kgw_detect,
    kgw_green_flags,
)
from wmkit.text.localize import Segment, localize_segments, segments_iou, token_scores

__all__ = [
    "DEFAULT_LM",
    "TEXT_DETECTION_P",
    "AttackRateError",
    "GenerationError",
    "GumbelConfig",
    "GumbelSampler",
    "HFNextLogits",
    "KGWConfig",
    "KGWLogitsProcessor",
    "Segment",
    "TextDetection",
    "WatermarkConfigError",
    "delete_tokens",
    "encode_text",
    "generate",
    "generate_many",
    "green_mask",
    "gumbel_detect",
    "gumbel_token_scores",
    "insert_tokens",
    "kgw_detect",
    "kgw_green_flags",
    "load_lm",
    "localize_segments",
    "model_vocab_size",
    "sample_sequences",
    "secret_vector",
    "segments_iou",
    "substitute_tokens",
    "token_scores",
]
