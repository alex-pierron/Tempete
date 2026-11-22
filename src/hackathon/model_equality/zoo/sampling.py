"""Chat prompts and chunked sampling for the zoo (HF `generate`, out-of-memory fallback, KGW).

Sampling mimics the ref server: pure temperature sampling (no top-p, no top-k, no repetition
penalty), at most 256 new tokens, the family chat template. The model's own `generation_config`
(for instance Qwen's top-k 20 and repetition penalty 1.05) is REPLACED, never merged, so that no
family silently samples differently from another.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from jinja2.exceptions import TemplateError
from transformers import GenerationConfig, LogitsProcessorList, TemperatureLogitsWarper

from hackathon.model_equality.zoo.config import (
    MAX_NEW_TOKENS,
    MIN_BATCH_SIZE,
    TOP_K_OFF,
    TOP_P_OFF,
)

SYSTEM_PREFIX_SEPARATOR = "\n\n"
"""Joins a system text to the user text when a chat template rejects the system role."""


class WatermarkUnavailableError(RuntimeError):
    """`wmkit.text.kgw` (the repository KGW) cannot be imported on this machine."""


def chat_text(tokenizer: Any, user_text: str, system_text: str | None = None) -> str:
    """The prompt string of one request: chat template with the generation prompt appended.

    A template that does not accept a system role (some Mistral templates) gets the system text
    prepended to the user message instead; the answer is still conditioned on it.
    """
    user_message = {"role": "user", "content": user_text}
    if system_text is None:
        messages = [user_message]
    else:
        messages = [{"role": "system", "content": system_text}, user_message]
    try:
        rendered: str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except TemplateError:
        if system_text is None:
            raise
        merged = [{"role": "user", "content": system_text + SYSTEM_PREFIX_SEPARATOR + user_text}]
        rendered = tokenizer.apply_chat_template(
            merged, add_generation_prompt=True, tokenize=False
        )
    return rendered


def load_kgw_processor(vocab_size: int) -> Any:
    """The repository KGW logits processor (`wmkit.text.kgw`, default gamma 0.25, delta 2.0).

    Raises:
        WatermarkUnavailableError: the module or its dependency (`scipy`) is missing.
    """
    try:
        from wmkit.text.kgw import KGWConfig, KGWLogitsProcessor
    except ImportError as error:
        raise WatermarkUnavailableError(
            f"cannot import wmkit.text.kgw ({error}); ship src/wmkit/text/kgw.py and scipy"
        ) from error
    return KGWLogitsProcessor(vocab_size=vocab_size, config=KGWConfig())


def eos_ids(model: Any, tokenizer: Any) -> list[int]:
    """Every end-of-turn id of the family (the model's generation config, else the tokenizer)."""
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if configured is None:
        raise ValueError("model and tokenizer define no end-of-sequence token")
    return [int(configured)] if isinstance(configured, int) else [int(i) for i in configured]


def padding_id(tokenizer: Any, eos: Sequence[int]) -> int:
    """The pad token id, falling back to the first end-of-sequence id."""
    return int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else int(eos[0])


@dataclass
class GenerationResult:
    """Answers in the order of the prompts, plus what the run cost."""

    texts: list[str]
    new_tokens: list[int]
    seconds: float
    oom_events: int = 0
    final_batch: int = 0
    notes: list[str] = field(default_factory=list)


def _generate_chunk(
    model: Any, tokenizer: Any, prompts: Sequence[str], gen_config: GenerationConfig,
    processors: LogitsProcessorList | None, seed: int, device: torch.device,
) -> tuple[list[str], list[int]]:
    """Sample one chunk of prompts; return `(answers, new token counts)`."""
    encoded = tokenizer(list(prompts), add_special_tokens=False, return_tensors="pt",
                        padding=True).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        output = model.generate(**encoded, generation_config=gen_config,
                                logits_processor=processors)
    fresh = output[:, encoded["input_ids"].shape[1]:]
    pad = gen_config.pad_token_id
    counts = [int((row != pad).sum()) for row in fresh]
    texts = [t.strip() for t in tokenizer.batch_decode(fresh, skip_special_tokens=True)]
    return texts, counts


def generate_texts(
    model: Any, tokenizer: Any, prompts: Sequence[str], temperature: float, watermark: bool,
    batch_size: int, seed: int, device: torch.device, log: Any = print,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> GenerationResult:
    """Sample one answer per prompt, in chunks, halving the chunk on CUDA out-of-memory.

    Prompts are processed longest first to keep padding small and the memory peak early; the
    answers are returned in the order of `prompts`. With `watermark`, the KGW bias is added to the
    raw logits and the temperature is applied after it (the order of the KGW paper).
    """
    tokenizer.padding_side = "left"
    eos = eos_ids(model, tokenizer)
    pad = padding_id(tokenizer, eos)
    builtin_temperature = 1.0 if watermark else temperature
    gen_config = GenerationConfig(  # type: ignore[no-untyped-call]  # transformers is untyped here
        do_sample=True, temperature=builtin_temperature, top_p=TOP_P_OFF, top_k=TOP_K_OFF,
        repetition_penalty=1.0, max_new_tokens=max_new_tokens, eos_token_id=eos,
        pad_token_id=pad,
    )
    processors: LogitsProcessorList | None = None
    if watermark:
        processors = LogitsProcessorList([
            load_kgw_processor(int(model.config.vocab_size)),
            TemperatureLogitsWarper(temperature),
        ])
    lengths = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    texts: list[str] = [""] * len(prompts)
    counts: list[int] = [0] * len(prompts)
    started = time.time()
    chunk_size = max(batch_size, MIN_BATCH_SIZE)
    oom_events = 0
    position = 0
    chunk_index = 0
    while position < len(order):
        indices = order[position:position + chunk_size]
        try:
            chunk_texts, chunk_counts = _generate_chunk(
                model, tokenizer, [prompts[i] for i in indices], gen_config, processors,
                seed + chunk_index, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_size <= MIN_BATCH_SIZE:
                raise
            chunk_size = max(MIN_BATCH_SIZE, chunk_size // 2)
            oom_events += 1
            log(f"generate: out of memory, chunk -> {chunk_size}")
            continue
        for slot, text, count in zip(indices, chunk_texts, chunk_counts, strict=True):
            texts[slot], counts[slot] = text, count
        position += len(indices)
        chunk_index += 1
    return GenerationResult(texts, counts, time.time() - started, oom_events, chunk_size)
