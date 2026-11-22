"""Watermarked (or plain) text generation with a Hugging Face causal LM.

The sampling loop is written by hand instead of calling `model.generate` for two reasons:
the Gumbel-max scheme replaces sampling (it is not a logits transform), and one loop gives
the same ordering for every scheme: logits -> temperature -> watermark bias -> softmax ->
draw. The loop only needs a function `ids -> next-token logits`, so it can be driven by a
fake language model in fast tests (`sample_sequences`) or by a real one (`generate`).

Limits: no top-k / top-p (MET studies pure temperature sampling); prompts are replicated, not
padded, so one call generates several sequences from ONE prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal

import torch

from wmkit.text.gumbel import GumbelConfig, GumbelSampler
from wmkit.text.kgw import KGWConfig, KGWLogitsProcessor, WatermarkConfigError

DEFAULT_LM: Final[str] = "HuggingFaceTB/SmolLM2-135M-Instruct"
"""Small local model used for every experiment (about 270 MB of weights)."""

DEFAULT_TEMPERATURE: Final[float] = 1.0
SCHEME_NONE: Final[str] = "none"
SCHEME_KGW: Final[str] = "kgw"
SCHEME_GUMBEL: Final[str] = "gumbel"

Scheme = Literal["none", "kgw", "gumbel"]
WatermarkConfig = KGWConfig | GumbelConfig
NextLogitsFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps the full prefix `[B, S]` (int64) to next-token logits `[B, V]` (float32, CPU)."""


class GenerationError(ValueError):
    """Raised when generation arguments are inconsistent."""


class HFNextLogits:
    """Adapter from a Hugging Face causal LM to `NextLogitsFn`, with a KV cache.

    One instance serves ONE generation: the cache grows with the prefix, so a new instance
    is required for a new prompt.
    """

    def __init__(self, model: Any) -> None:
        """Wrap a causal LM (any device)."""
        self._model = model
        self._cache: Any = None
        self._consumed = 0

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the logits after the last token of `input_ids`, feeding only new tokens."""
        device = next(self._model.parameters()).device
        new_tokens = input_ids[:, self._consumed :].to(device)
        with torch.inference_mode():
            output = self._model(input_ids=new_tokens, past_key_values=self._cache, use_cache=True)
        self._cache = output.past_key_values
        self._consumed = input_ids.shape[1]
        last_logits: torch.Tensor = output.logits[:, -1, :].float().cpu()
        return last_logits


def _resolve_config(scheme: str, config: WatermarkConfig | None) -> WatermarkConfig | None:
    """Check that `config` matches `scheme`, filling in the default config when absent."""
    if scheme == SCHEME_NONE:
        return None
    if scheme == SCHEME_KGW:
        if config is None:
            return KGWConfig()
        if not isinstance(config, KGWConfig):
            raise GenerationError(f"scheme 'kgw' needs a KGWConfig, got {type(config).__name__}")
        return config
    if scheme == SCHEME_GUMBEL:
        if config is None:
            return GumbelConfig()
        if not isinstance(config, GumbelConfig):
            raise GenerationError(
                f"scheme 'gumbel' needs a GumbelConfig, got {type(config).__name__}"
            )
        return config
    raise GenerationError(f"unknown scheme {scheme!r}; expected none, kgw or gumbel")


def sample_sequences(
    next_logits: NextLogitsFn,
    prompt_ids: Sequence[int],
    *,
    n_sequences: int,
    scheme: Scheme,
    config: WatermarkConfig | None,
    max_new_tokens: int,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    min_new_tokens: int = 0,
    eos_token_id: int | None = None,
) -> list[list[int]]:
    """Sample `n_sequences` continuations of one prompt from a next-token oracle.

    Args:
        next_logits: Maps the prefix `[B, S]` to logits `[B, V]`.
        prompt_ids: Prompt token ids (non-empty).
        n_sequences: Number of independent continuations.
        scheme: "none", "kgw" or "gumbel".
        config: Watermark parameters matching `scheme` (None gives the defaults).
        max_new_tokens: Maximum generated tokens per sequence.
        seed: Seed of the sampling generator (ignored by "gumbel", which is deterministic
            given key and context).
        temperature: Softmax temperature, strictly positive.
        min_new_tokens: EOS is forbidden before this many tokens (fixes the text length).
        eos_token_id: End-of-sequence token; generation of a row stops when it is drawn and
            the EOS itself is not returned.

    Returns:
        One list of generated token ids per sequence (prompt excluded).
    """
    if not prompt_ids:
        raise GenerationError("prompt_ids must not be empty")
    if n_sequences < 1 or max_new_tokens < 1:
        raise GenerationError("n_sequences and max_new_tokens must be >= 1")
    if temperature <= 0.0:
        raise GenerationError(f"temperature must be > 0, got {temperature}")
    resolved = _resolve_config(scheme, config)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    prefix = torch.tensor([list(prompt_ids)] * n_sequences, dtype=torch.long)
    finished = torch.zeros(n_sequences, dtype=torch.bool)
    generated: list[list[int]] = [[] for _ in range(n_sequences)]
    kgw_processor: KGWLogitsProcessor | None = None
    gumbel_sampler: GumbelSampler | None = None

    for step in range(max_new_tokens):
        logits = next_logits(prefix).clone()
        vocab_size = logits.shape[-1]
        if eos_token_id is not None and step < min_new_tokens:
            logits[:, eos_token_id] = -torch.inf
        logits = logits / temperature
        if isinstance(resolved, KGWConfig):
            kgw_processor = kgw_processor or KGWLogitsProcessor(vocab_size, resolved)
            logits = kgw_processor(prefix, logits)  # type: ignore[arg-type]
        probabilities = torch.softmax(logits, dim=-1)
        if isinstance(resolved, GumbelConfig):
            gumbel_sampler = gumbel_sampler or GumbelSampler(vocab_size, resolved)
            width = resolved.context_width
            probs64 = probabilities.double().numpy()
            chosen = torch.tensor(
                [
                    gumbel_sampler.select(prefix[row, -width:].tolist(), probs64[row])
                    for row in range(n_sequences)
                ],
                dtype=torch.long,
            )
        else:
            chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        for row in range(n_sequences):
            if finished[row]:
                continue
            token = int(chosen[row])
            if eos_token_id is not None and token == eos_token_id:
                finished[row] = True
            else:
                generated[row].append(token)
        if bool(finished.all()):
            break
        prefix = torch.cat([prefix, chosen.unsqueeze(1)], dim=1)
    return generated


def load_lm(model_name: str = DEFAULT_LM, device: str | None = None) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer from the standard Hugging Face cache.

    Args:
        model_name: Hub id.
        device: Torch device; None picks `wmkit.config.get_device()` (cuda, then mps, cpu).

    Returns:
        `(model, tokenizer)`, the model in eval mode on `device`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wmkit.config import get_device

    target = torch.device(device) if device else get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(target).eval()
    return model, tokenizer


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize a string without special tokens (the form expected by the detectors)."""
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def _prompt_to_ids(tokenizer: Any, prompt: str, chat_template: bool) -> list[int]:
    """Tokenize a prompt, optionally through the model's chat template."""
    if chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return [int(token) for token in tokenizer.encode(rendered, add_special_tokens=False)]
    return [int(token) for token in tokenizer.encode(prompt)]


def generate_many(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_sequences: int,
    *,
    scheme: Scheme,
    config: WatermarkConfig | None,
    max_new_tokens: int,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    min_new_tokens: int = 0,
    chat_template: bool = False,
) -> list[list[int]]:
    """Generate `n_sequences` continuations of one prompt in a single batch.

    Same arguments as `generate`, plus `n_sequences`. With "gumbel" all sequences are
    identical (the scheme is deterministic given key and context).

    Returns:
        One list of token ids per sequence (prompt excluded, EOS excluded).
    """
    prompt_ids = _prompt_to_ids(tokenizer, prompt, chat_template)
    return sample_sequences(
        HFNextLogits(model),
        prompt_ids,
        n_sequences=n_sequences,
        scheme=scheme,
        config=config,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    scheme: Scheme,
    config: WatermarkConfig | None,
    max_new_tokens: int,
    seed: int,
    temperature: float = DEFAULT_TEMPERATURE,
    min_new_tokens: int = 0,
    chat_template: bool = False,
) -> list[int]:
    """Generate one continuation of `prompt`.

    Args:
        model: Hugging Face causal LM.
        tokenizer: Matching tokenizer.
        prompt: Text to continue.
        scheme: "none" (plain sampling), "kgw" or "gumbel".
        config: Watermark parameters matching `scheme`; None gives the defaults.
        max_new_tokens: Maximum number of generated tokens.
        seed: Sampling seed (a Gumbel text depends on key and context only).
        temperature: Softmax temperature (> 0), applied before the watermark bias.
        min_new_tokens: Forbid EOS before this many tokens, to get fixed-length texts.
        chat_template: Wrap the prompt in the tokenizer's chat template.

    Returns:
        Generated token ids (prompt excluded). Detection needs the same `vocab_size` as the
        logits width, i.e. `model.config.vocab_size`.
    """
    return generate_many(
        model,
        tokenizer,
        prompt,
        1,
        scheme=scheme,
        config=config,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        min_new_tokens=min_new_tokens,
        chat_template=chat_template,
    )[0]


def model_vocab_size(model: Any) -> int:
    """Return the logits width of a model (the `vocab_size` detectors must use)."""
    size = getattr(model.config, "vocab_size", None)
    if not isinstance(size, int):
        raise WatermarkConfigError("model.config.vocab_size is missing")
    return size
