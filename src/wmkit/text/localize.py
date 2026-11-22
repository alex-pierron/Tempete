"""Locate watermarked segments inside a longer, possibly mixed, text.

Pipeline (all vectorized with cumulative sums, O(T) per scan):
1. Per-token statistic: KGW green flag (1/0) or Gumbel term `-log(1 - r)`; tokens without a
   full context or already scored (dedup) are NaN and carry no information.
2. Sliding windows of `window`, `2 * window` and `4 * window` tokens are tested with the exact
   p-value of the scheme. Several scales are needed because real LM text has low entropy: a
   watermarked passage may be invisible in 32 tokens yet obvious in 128.
3. Overlapping or adjacent significant windows are merged into candidate regions.
4. Each region (plus a margin of one window on both sides) is refined with a maximum-sum
   subarray scan (Kadane) on the centred statistic minus half the observed excess, which is the
   change-point log-likelihood-ratio shape; this recovers boundaries to within a few tokens.
   The refined segment then absorbs adjacent unscored tokens (repetition loops, see
   `_extend_over_unscored`).
5. The exact p-value of the final segment is multiplied by the number of tokens (Bonferroni over
   the possible start positions, clipped at 1) before the `p_threshold` test; this keeps the
   false-alarm rate on unwatermarked text low (measured in the demo). The returned p-value is
   this adjusted, conservative value.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom

from wmkit.text.gumbel import GumbelConfig, gumbel_token_scores
from wmkit.text.kgw import (
    TEXT_DETECTION_P,
    BoolArray,
    FloatArray,
    KGWConfig,
    WatermarkConfigError,
    effective_gamma,
    kgw_green_flags,
)

DEFAULT_WINDOW: int = 32
"""Window length in tokens (contract default)."""

GUMBEL_NULL_MEAN: float = 1.0
"""E[-log(1 - r)] for r ~ U(0, 1): the per-token mean of the Gumbel statistic under H0."""

WINDOW_SCALES: tuple[int, ...] = (1, 2, 4)
"""Multiples of `window` scanned (multi-scale scan)."""

KADANE_ROUNDS: int = 4
"""Boundary refinements: the offset is re-estimated on the previous segment each round, since
the excess measured on the (diluted) flagged region underestimates the one inside the segment."""

KADANE_OFFSET_FRACTION: float = 0.4
"""Fraction of the observed mean excess subtracted per token before the boundary scan."""

Segment = tuple[int, int, float]
"""(start, end exclusive, p-value) in token positions."""


def token_scores(
    token_ids: Sequence[int],
    vocab_size: int,
    config: KGWConfig | GumbelConfig,
    *,
    dedup_ngrams: bool = True,
) -> FloatArray:
    """Return the per-token detection statistic.

    Args:
        token_ids: Token ids `[T]`.
        vocab_size: Vocabulary size used at generation time.
        config: KGW (statistic: 1.0 green / 0.0 red) or Gumbel (statistic: `-log(1 - r)`).
        dedup_ngrams: Mark repeated (context, token) tuples as unscored.

    Returns:
        Float array `[T]`; NaN where the token is not scored (first `context_width` tokens
        and repeated tuples).
    """
    if isinstance(config, KGWConfig):
        return kgw_green_flags(token_ids, vocab_size, config, dedup_ngrams=dedup_ngrams)
    if isinstance(config, GumbelConfig):
        return gumbel_token_scores(token_ids, vocab_size, config, dedup_ngrams=dedup_ngrams)
    raise WatermarkConfigError(f"unsupported config type {type(config).__name__}")


def _window_pvalues(
    config: KGWConfig | GumbelConfig,
    vocab_size: int,
    statistic_sum: FloatArray,
    n_scored: FloatArray,
) -> FloatArray:
    """Exact p-values for arrays of (sum of statistic, number of scored tokens)."""
    has_tokens = n_scored > 0
    safe_n = np.where(has_tokens, n_scored, 1.0)
    if isinstance(config, KGWConfig):
        gamma = effective_gamma(vocab_size, config.gamma)
        raw = binom.sf(statistic_sum - 1.0, safe_n, gamma)
    else:
        raw = gammaincc(safe_n, np.maximum(statistic_sum, 0.0))
    return np.where(has_tokens, raw, 1.0).astype(np.float64)


def _null_mean(config: KGWConfig | GumbelConfig, vocab_size: int) -> float:
    """Mean of the per-token statistic under H0."""
    if isinstance(config, KGWConfig):
        return effective_gamma(vocab_size, config.gamma)
    return GUMBEL_NULL_MEAN


def _max_subarray(values: FloatArray) -> tuple[int, int]:
    """Return `(start, end exclusive)` of the maximum-sum contiguous subarray (Kadane)."""
    best_sum = -np.inf
    best = (0, 0)
    current_sum = 0.0
    current_start = 0
    for index, value in enumerate(values.tolist()):
        if current_sum <= 0.0:
            current_sum = value
            current_start = index
        else:
            current_sum += value
        if current_sum > best_sum:
            best_sum = current_sum
            best = (current_start, index + 1)
    return best


def _extend_over_unscored(start: int, end: int, scored: BoolArray) -> tuple[int, int]:
    """Grow `[start, end)` over adjacent tokens that carry no evidence.

    Repeated (context, token) tuples are not scored (dedup) and the first tokens have no
    context; inside a watermarked passage such runs (typically a repetition loop) belong to
    the passage, whereas stopping at the last scored token would cut the segment short. The
    growth stops at the first scored token on each side, so evidence still decides the border.
    """
    while end < len(scored) and not scored[end]:
        end += 1
    while start > 0 and not scored[start - 1]:
        start -= 1
    return start, end


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching `[start, end)` intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def localize_segments(
    token_ids: Sequence[int],
    vocab_size: int,
    config: KGWConfig | GumbelConfig,
    *,
    window: int = DEFAULT_WINDOW,
    p_threshold: float = TEXT_DETECTION_P,
    dedup_ngrams: bool = True,
) -> list[Segment]:
    """Find the watermarked segments of a text.

    Args:
        token_ids: Token ids of the whole text.
        vocab_size: Vocabulary size used at generation time.
        config: Watermark parameters including the key.
        window: Smallest scan window in tokens; 2x and 4x are scanned too (all capped at the
            text length).
        p_threshold: A window must have an exact p-value below this, and the final
            segment an adjusted one.
        dedup_ngrams: Score each (context, token) tuple once.

    Returns:
        Disjoint segments `(start, end_exclusive, p_value)` sorted by position; empty when
        nothing is significant. The p-value is Bonferroni-adjusted by the text length.
    """
    if window < 1:
        raise WatermarkConfigError(f"window must be >= 1, got {window}")
    statistics = token_scores(token_ids, vocab_size, config, dedup_ngrams=dedup_ngrams)
    n_tokens = len(statistics)
    if n_tokens == 0:
        return []
    scored = ~np.isnan(statistics)
    values = np.where(scored, statistics, 0.0)
    cumulative_values = np.concatenate([[0.0], np.cumsum(values)])
    cumulative_counts = np.concatenate([[0.0], np.cumsum(scored.astype(np.float64))])
    spans = sorted({min(window * scale, n_tokens) for scale in WINDOW_SCALES})
    span = spans[0]

    def segment_p(start: int, end: int) -> float:
        total = cumulative_values[end] - cumulative_values[start]
        count = cumulative_counts[end] - cumulative_counts[start]
        return float(_window_pvalues(config, vocab_size, np.array([total]), np.array([count]))[0])

    regions_raw: list[tuple[int, int]] = []
    for current_span in spans:
        starts = np.arange(0, n_tokens - current_span + 1)
        window_p = _window_pvalues(
            config,
            vocab_size,
            cumulative_values[starts + current_span] - cumulative_values[starts],
            cumulative_counts[starts + current_span] - cumulative_counts[starts],
        )
        regions_raw += [
            (int(start), int(start) + current_span) for start in starts[window_p < p_threshold]
        ]
    if not regions_raw:
        return []
    regions = _merge_intervals(regions_raw)

    null_mean = _null_mean(config, vocab_size)
    refined: list[tuple[int, int]] = []
    for start, end in regions:
        lo, hi = max(0, start - span), min(n_tokens, end + span)
        current = (start, end)
        for _ in range(KADANE_ROUNDS):
            count = cumulative_counts[current[1]] - cumulative_counts[current[0]]
            total = cumulative_values[current[1]] - cumulative_values[current[0]]
            if count <= 0:
                break
            offset = KADANE_OFFSET_FRACTION * (total / count - null_mean)
            centred = np.where(scored[lo:hi], statistics[lo:hi] - null_mean - offset, 0.0)
            local_start, local_end = _max_subarray(centred)
            updated = (lo + local_start, lo + local_end)
            if updated == current:
                break
            current = updated
        refined.append(_extend_over_unscored(current[0], current[1], scored))

    segments: list[Segment] = []
    for start, end in _merge_intervals(refined):
        p_value = min(1.0, segment_p(start, end) * n_tokens)
        if p_value < p_threshold:
            segments.append((start, end, p_value))
    return segments


def segments_iou(predicted: Sequence[Segment], truth: tuple[int, int]) -> float:
    """Token-level intersection over union between predicted segments and one true span.

    Args:
        predicted: Output of `localize_segments`.
        truth: True watermarked span `(start, end_exclusive)`.

    Returns:
        IoU in [0, 1]; 0.0 when both sets are empty.
    """
    true_tokens = set(range(truth[0], truth[1]))
    predicted_tokens: set[int] = set()
    for start, end, _ in predicted:
        predicted_tokens.update(range(start, end))
    union = true_tokens | predicted_tokens
    if not union:
        return 0.0
    return len(true_tokens & predicted_tokens) / len(union)
