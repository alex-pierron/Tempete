"""Gumbel-max (Aaronson) watermark with an exact Gamma detection test.

Generation: at each step a secret vector `r` in [0, 1)^V is derived from the previous tokens
and the key; the emitted token is `argmax_v r_v ** (1 / p_v)`, equivalently
`argmax_v (log p_v - log(-log r_v))` (Gumbel-max trick). Averaged over keys the token is
distributed exactly as `p` (distortion-free), but for a FIXED key and context the choice is
deterministic: the same prompt always yields the same text (see `docs/research`, Kuditipudi
et al., for the key-shift remedy; not implemented here).

Detection: with `r_t` the secret value of the emitted token, the score
`S = sum_t -log(1 - r_t)` follows Gamma(n, 1) under H0 (text independent of the key), so
`p = gammaincc(n, S)`. Each (context, token) tuple is scored once (Three Bricks, scoring v2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import gammaincc, gammaln
from transformers import LogitsProcessor

from wmkit.text.kgw import (
    DEFAULT_HASH_KEY,
    LN10,
    TEXT_DETECTION_P,
    FloatArray,
    TextDetection,
    WatermarkConfigError,
    scored_positions,
    validate_token_ids,
)

R_FLOOR: float = float(np.finfo(np.float64).tiny)
"""Lower clip of r so that log(r) stays finite."""

MAX_SCORE_TERM: float = 40.0
"""-log(1 - r) is at most ~36.7 for a float64 r < 1; this guards a degenerate r."""


@dataclass(frozen=True)
class GumbelConfig:
    """Parameters of the Gumbel-max watermark.

    Attributes:
        hash_key: Secret non-negative integer key (default: same prime as KGW; a default is
            an extension of the frozen contract, which lists the field without one).
        context_width: Number of previous tokens that seed the secret vector.
    """

    hash_key: int = DEFAULT_HASH_KEY
    context_width: int = 1

    def __post_init__(self) -> None:
        """Validate ranges at construction."""
        if self.hash_key < 0:
            raise WatermarkConfigError(f"hash_key must be >= 0, got {self.hash_key}")
        if self.context_width < 1:
            raise WatermarkConfigError(f"context_width must be >= 1, got {self.context_width}")


def secret_vector(context: Sequence[int], vocab_size: int, config: GumbelConfig) -> FloatArray:
    """Return the secret vector r in [0, 1)^V attached to a context.

    `numpy.random.default_rng` mixes `[hash_key, *context]` through `SeedSequence`, so
    nearby contexts give independent streams.
    """
    rng = np.random.default_rng([config.hash_key, *[int(token) for token in context]])
    return rng.random(vocab_size)


class GumbelSampler(LogitsProcessor):
    """Watermarked sampler: picks the Gumbel-max token for a given context.

    Two entry points: `select` (used by `wmkit.text.generate`) and `__call__`, which makes
    the object a `LogitsProcessor` that turns the scores into a one-hot (chosen token at 0,
    the rest at -inf) so that `model.generate(..., do_sample=False)` emits the chosen token.
    """

    def __init__(self, vocab_size: int, config: GumbelConfig) -> None:
        """Create the sampler.

        Args:
            vocab_size: Vocabulary size; detection must use the same value.
            config: Watermark parameters.
        """
        if vocab_size < 2:
            raise WatermarkConfigError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = vocab_size
        self.config = config

    def select(self, context: Sequence[int], probabilities: FloatArray) -> int:
        """Return the token `argmax_v log p_v - log(-log r_v)` for this context.

        Args:
            context: Previous token ids (the last `context_width` are used).
            probabilities: Next-token distribution `[V]` (float64, sums to 1).

        Returns:
            The selected token id.
        """
        width = self.config.context_width
        r = np.maximum(secret_vector(list(context)[-width:], self.vocab_size, self.config), R_FLOOR)
        with np.errstate(divide="ignore"):
            keyed = np.log(probabilities) - np.log(-np.log(r))
        return int(np.argmax(keyed))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Replace scores `[B, V]` by a one-hot of the Gumbel-max token of each row."""
        out = torch.full_like(scores, -math.inf)
        probabilities = torch.softmax(scores.double(), dim=-1).cpu().numpy()
        for row in range(scores.shape[0]):
            context = input_ids[row, -self.config.context_width :].tolist()
            chosen = self.select(context, probabilities[row, : self.vocab_size])
            out[row, chosen] = 0.0
        return out  # type: ignore[return-value]


def gumbel_token_scores(
    token_ids: Sequence[int],
    vocab_size: int,
    config: GumbelConfig,
    *,
    dedup_ngrams: bool = True,
) -> FloatArray:
    """Return `-log(1 - r_t)` per token, NaN where the token is not scored."""
    ids = validate_token_ids(token_ids, vocab_size)
    scores = np.full(len(ids), np.nan, dtype=np.float64)
    width = config.context_width
    for position in np.flatnonzero(scored_positions(ids, width, dedup_ngrams=dedup_ngrams)):
        vector = secret_vector(ids[position - width : position].tolist(), vocab_size, config)
        scores[position] = min(-math.log1p(-vector[ids[position]]), MAX_SCORE_TERM)
    return scores


def _log_gamma_sf(n_scored: float, total: float) -> float:
    """Return ln P(Gamma(n, 1) >= total) with an asymptotic fallback when it underflows."""
    p_value = float(gammaincc(n_scored, total))
    if p_value > 0.0:
        return math.log(p_value)
    # Leading terms of the upper incomplete gamma tail: s^(n-1) e^-s / Gamma(n) * 1/(1-(n-1)/s)
    correction = 1.0 - (n_scored - 1.0) / total
    return float(
        (n_scored - 1.0) * math.log(total) - total - gammaln(n_scored) - math.log(correction)
    )


def gumbel_pvalue(total_score: float, n_scored: int) -> tuple[float, float]:
    """Exact Gamma p-value and its log10.

    Args:
        total_score: Observed sum of `-log(1 - r_t)`.
        n_scored: Number of scored tokens.

    Returns:
        `(p_value, log10_p_value)`; `(1.0, 0.0)` when nothing was scored.
    """
    if n_scored <= 0:
        return 1.0, 0.0
    log_p = min(_log_gamma_sf(float(n_scored), total_score), 0.0)
    return float(math.exp(log_p)), log_p / LN10


def gumbel_detect(
    token_ids: Sequence[int],
    vocab_size: int,
    config: GumbelConfig,
    *,
    dedup_ngrams: bool = True,
) -> TextDetection:
    """Test whether a token sequence carries the Gumbel-max watermark of `config`.

    Args:
        token_ids: Token ids of the text under test (no special tokens).
        vocab_size: Vocabulary size used at generation time.
        config: Watermark parameters (including the secret key).
        dedup_ngrams: Score each (context, token) tuple once (strongly recommended).

    Returns:
        A `TextDetection`; `score` is the Gamma statistic, `z_score` its normal
        approximation `(S - n) / sqrt(n)`.
    """
    terms = gumbel_token_scores(token_ids, vocab_size, config, dedup_ngrams=dedup_ngrams)
    observed = terms[~np.isnan(terms)]
    n_scored = int(observed.size)
    total = float(observed.sum())
    p_value, log10_p = gumbel_pvalue(total, n_scored)
    z_score = (total - n_scored) / math.sqrt(n_scored) if n_scored > 0 else None
    return TextDetection(
        n_scored=n_scored,
        n_green=None,
        score=total,
        z_score=z_score,
        p_value=p_value,
        detected=p_value < TEXT_DETECTION_P,
        log10_p_value=log10_p,
    )
