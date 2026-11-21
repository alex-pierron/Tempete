"""KGW green-list watermark (Kirchenbauer et al., ICML 2023) with exact detection.

Generation side: a `transformers`-compatible `LogitsProcessor` adds `delta` to the logits of a
pseudo-random "green" subset (a fraction `gamma` of the vocabulary) seeded by the previous
token. With `context_width=1` and `seeding="lefthash"` the green list is bit-identical to the
one produced by `transformers.WatermarkLogitsProcessor` on CPU (same seed formula, same
`torch.randperm`), so texts generated with Hugging Face's `watermarking_config` can be detected
here (a test pins this equivalence).

Detection side: a token is scored only once per distinct (context, token) tuple ("scoring v2"
of Three Bricks, Fernandez et al. 2023). Without that dedup the empirical false-positive rate
of the z-test explodes on repetitive text. Under H0 (text independent of the key) the number
of green tokens is exactly Binomial(n_scored, gamma), which gives an exact p-value.

Hardcoded-free: every numeric constant is named below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import gammaln, logsumexp
from transformers import LogitsProcessor

TEXT_DETECTION_P: Final[float] = 1e-3
"""A text is declared watermarked when its p-value is strictly below this threshold."""

DEFAULT_HASH_KEY: Final[int] = 15485863
"""The millionth prime: default key of `transformers.WatermarkingConfig`."""

SEEDING_LEFTHASH: Final[str] = "lefthash"
"""Seed = hash_key * previous token (Algorithm 2 of the paper)."""

TORCH_SEED_MODULUS: Final[int] = 2**64 - 1
"""Same modulus as transformers before `torch.Generator.manual_seed`."""

CONTEXT_MIXING_PRIME: Final[int] = 1_000_003
"""Multiplier used to fold several context tokens into one integer (context_width > 1)."""

GREEN_LIST_CACHE_SIZE: Final[int] = 1024
"""Max cached green lists (about 49 KB each for a 49k vocabulary): bounds memory near 50 MB."""

LN10: Final[float] = math.log(10.0)
_MIN_EXP_ARG: Final[float] = -745.0  # exp underflows to 0.0 below this; keep the log finite

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class WatermarkConfigError(ValueError):
    """Raised when a watermark configuration or its inputs are invalid."""


@dataclass(frozen=True)
class TextDetection:
    """Outcome of a watermark detection test on one text.

    Attributes:
        n_scored: Number of tokens that entered the test (after context and dedup rules).
        n_green: Number of green tokens (KGW only, None for Gumbel).
        score: KGW: number of green tokens as a float. Gumbel: sum of -log(1 - r_t).
        z_score: Normal approximation of the score under H0 (None when `n_scored == 0`).
        p_value: Exact p-value under H0 (binomial for KGW, Gamma for Gumbel).
        detected: `p_value < TEXT_DETECTION_P`.
        log10_p_value: log10 of the p-value, finite even when `p_value` underflows to 0.
    """

    n_scored: int
    n_green: int | None
    score: float
    z_score: float | None
    p_value: float
    detected: bool
    log10_p_value: float = 0.0

    @property
    def neg_log10_p(self) -> float:
        """Return -log10(p), robust to float underflow."""
        return -self.log10_p_value


@dataclass(frozen=True)
class KGWConfig:
    """Parameters of the KGW watermark.

    Attributes:
        gamma: Fraction of the vocabulary that is green, in (0, 1).
        delta: Bias added to green logits.
        context_width: Number of previous tokens that seed the green list. With 1 the scheme
            is the one of transformers; above 1 the tokens are folded with a fixed mixing
            prime (our own extension, not transformers-compatible).
        hash_key: Secret integer key multiplied into the seed.
        seeding: Only "lefthash" is implemented ("selfhash" is not).
    """

    gamma: float = 0.25
    delta: float = 2.0
    context_width: int = 1
    hash_key: int = DEFAULT_HASH_KEY
    seeding: Literal["lefthash"] = "lefthash"

    def __post_init__(self) -> None:
        """Validate ranges so that errors surface at construction, not mid-generation."""
        if not 0.0 < self.gamma < 1.0:
            raise WatermarkConfigError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.context_width < 1:
            raise WatermarkConfigError(f"context_width must be >= 1, got {self.context_width}")
        if self.hash_key < 0:
            raise WatermarkConfigError(f"hash_key must be >= 0, got {self.hash_key}")
        if self.seeding != SEEDING_LEFTHASH:
            raise WatermarkConfigError(
                f"seeding {self.seeding!r} is not implemented; only {SEEDING_LEFTHASH!r}"
            )


def green_list_size(vocab_size: int, gamma: float) -> int:
    """Return the number of green tokens, `int(vocab_size * gamma)` as in transformers."""
    return int(vocab_size * gamma)


def effective_gamma(vocab_size: int, gamma: float) -> float:
    """Return the exact green fraction after integer truncation of the list size."""
    return green_list_size(vocab_size, gamma) / vocab_size


def context_seed(context: Sequence[int], hash_key: int) -> int:
    """Fold a token context into the integer seed of a pseudo-random generator.

    For a single token this is exactly `hash_key * token`, the transformers formula.

    Args:
        context: The last `context_width` token ids, oldest first.
        hash_key: Secret key.

    Returns:
        A non-negative integer below `TORCH_SEED_MODULUS`.
    """
    folded = int(context[-1])
    for token in reversed(context[:-1]):
        folded = folded * CONTEXT_MIXING_PRIME + int(token)
    return (hash_key * folded) % TORCH_SEED_MODULUS


@lru_cache(maxsize=GREEN_LIST_CACHE_SIZE)
def _green_mask(seed: int, vocab_size: int, n_green: int) -> torch.Tensor:
    """Return the cached boolean green mask of one seed (CPU, so devices never change it)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(vocab_size, generator=generator)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[permutation[:n_green]] = True
    return mask


def green_mask(context: Sequence[int], vocab_size: int, config: KGWConfig) -> torch.Tensor:
    """Return the boolean green-list mask `[vocab_size]` for a token context."""
    seed = context_seed(context, config.hash_key)
    return _green_mask(seed, vocab_size, green_list_size(vocab_size, config.gamma))


class KGWLogitsProcessor(LogitsProcessor):
    """Add `delta` to the logits of the green tokens (usable in `model.generate`).

    Scores wider than `vocab_size` (padded embedding matrices) keep their extra columns
    untouched, so those padding tokens are never green.
    """

    def __init__(self, vocab_size: int, config: KGWConfig) -> None:
        """Create the processor.

        Args:
            vocab_size: Vocabulary size used to build the green lists. Detection must use
                the same value (for SmolLM2 it is `model.config.vocab_size`).
            config: Watermark parameters.
        """
        if vocab_size < 2:
            raise WatermarkConfigError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = vocab_size
        self.config = config

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Bias the scores `[B, V]` given the generated prefix `[B, S]`.

        When the prefix is shorter than the context width the scores are returned unchanged.
        """
        width = self.config.context_width
        if input_ids.shape[-1] < width:
            return scores
        n_columns = min(scores.shape[-1], self.vocab_size)
        rows = [
            green_mask(input_ids[row, -width:].tolist(), self.vocab_size, self.config)[:n_columns]
            for row in range(scores.shape[0])
        ]
        bias = torch.stack(rows).to(device=scores.device, dtype=scores.dtype) * self.config.delta
        out = scores.clone()
        out[:, :n_columns] += bias
        return out  # type: ignore[return-value]


def validate_token_ids(token_ids: Sequence[int], vocab_size: int) -> npt.NDArray[np.int64]:
    """Return token ids as an int64 array after checking they lie in [0, vocab_size)."""
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise WatermarkConfigError(f"token_ids must be 1-D, got shape {ids.shape}")
    if ids.size and (ids.min() < 0 or ids.max() >= vocab_size):
        raise WatermarkConfigError(
            f"token ids must lie in [0, {vocab_size}); got range [{ids.min()}, {ids.max()}]"
        )
    return ids


def scored_positions(
    ids: npt.NDArray[np.int64], context_width: int, *, dedup_ngrams: bool
) -> BoolArray:
    """Mark the positions that enter a detection test.

    A position is scored when it has a full context and, if `dedup_ngrams`, when its
    (context, token) tuple was not seen earlier in the text.

    Args:
        ids: Token ids `[T]`.
        context_width: Context size.
        dedup_ngrams: Apply the first-occurrence rule.

    Returns:
        Boolean array `[T]`.
    """
    scored = np.zeros(len(ids), dtype=np.bool_)
    seen: set[tuple[int, ...]] = set()
    for position in range(context_width, len(ids)):
        if dedup_ngrams:
            key = tuple(ids[position - context_width : position + 1].tolist())
            if key in seen:
                continue
            seen.add(key)
        scored[position] = True
    return scored


def kgw_green_flags(
    token_ids: Sequence[int],
    vocab_size: int,
    config: KGWConfig,
    *,
    dedup_ngrams: bool = True,
) -> FloatArray:
    """Return per-token green flags: 1.0 green, 0.0 red, NaN when the token is not scored."""
    ids = validate_token_ids(token_ids, vocab_size)
    flags = np.full(len(ids), np.nan, dtype=np.float64)
    width = config.context_width
    for position in np.flatnonzero(scored_positions(ids, width, dedup_ngrams=dedup_ngrams)):
        mask = green_mask(ids[position - width : position].tolist(), vocab_size, config)
        flags[position] = float(mask[ids[position]])
    return flags


def _log_binomial_sf(n_green: int, n_scored: int, gamma: float) -> float:
    """`log P(X >= n_green)` for `X ~ Binomial(n_scored, gamma)`, computed in log space.

    `scipy.stats.binom.logsf` is `log(sf)` of a discrete law and underflows to `-inf`
    once the tail probability drops below the float minimum (around 1e-308), which happens
    for a few hundred strongly green tokens. Summing the exact log-pmf terms with
    `logsumexp` keeps the log finite for any length, mirroring `metrics._log_binomial_tail`.
    """
    if n_green <= 0:
        return 0.0
    if n_green > n_scored:
        return -math.inf
    successes = np.arange(n_green, n_scored + 1, dtype=np.float64)
    log_pmf = (
        gammaln(n_scored + 1.0) - gammaln(successes + 1.0) - gammaln(n_scored - successes + 1.0)
        + successes * math.log(gamma) + (n_scored - successes) * math.log1p(-gamma)
    )
    return float(logsumexp(log_pmf))


def kgw_pvalue(n_green: int, n_scored: int, gamma: float) -> tuple[float, float]:
    """Exact upper-tail binomial p-value and its log10.

    Args:
        n_green: Observed green tokens.
        n_scored: Scored tokens.
        gamma: Green fraction under H0.

    Returns:
        `(p_value, log10_p_value)`; `(1.0, 0.0)` when nothing was scored. The log10 stays
        finite even when the p-value itself underflows to 0.0 for very long watermarked text.
    """
    if n_scored <= 0:
        return 1.0, 0.0
    log_p = _log_binomial_sf(n_green, n_scored, gamma)
    p_value = math.exp(log_p) if log_p > _MIN_EXP_ARG else 0.0
    return p_value, min(log_p / LN10, 0.0)


def kgw_detect(
    token_ids: Sequence[int],
    vocab_size: int,
    config: KGWConfig,
    *,
    dedup_ngrams: bool = True,
) -> TextDetection:
    """Test whether a token sequence carries the KGW watermark of `config`.

    Args:
        token_ids: Token ids of the text under test (no special tokens).
        vocab_size: Vocabulary size used at generation time.
        config: Watermark parameters (including the secret key).
        dedup_ngrams: Score each (context, token) tuple once (strongly recommended).

    Returns:
        A `TextDetection` with the z-score and the exact binomial p-value.
    """
    flags = kgw_green_flags(token_ids, vocab_size, config, dedup_ngrams=dedup_ngrams)
    observed = flags[~np.isnan(flags)]
    n_scored = int(observed.size)
    n_green = int(observed.sum())
    gamma = effective_gamma(vocab_size, config.gamma)
    p_value, log10_p = kgw_pvalue(n_green, n_scored, gamma)
    z_score: float | None = None
    if n_scored > 0:
        z_score = (n_green - gamma * n_scored) / math.sqrt(n_scored * gamma * (1.0 - gamma))
    return TextDetection(
        n_scored=n_scored,
        n_green=n_green,
        score=float(n_green),
        z_score=z_score,
        p_value=p_value,
        detected=p_value < TEXT_DETECTION_P,
        log10_p_value=log10_p,
    )
