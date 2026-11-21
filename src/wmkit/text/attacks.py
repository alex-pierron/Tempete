"""Token-level attacks on watermarked text (substitution, deletion, insertion).

Attacks act on token ids, not on strings: decoding then re-encoding changes the tokens and
would conflate the attack with tokenizer round-trip noise. Every function is pure given its
`numpy.random.Generator`, so an experiment is reproducible from one seed.

The number of edited tokens is exactly `round(rate * len(ids))` (not a Bernoulli draw), so a
curve "detection vs rate" has no sampling noise on the rate itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class AttackRateError(ValueError):
    """Raised when an attack rate or vocabulary size is out of range."""


def _check_rate(rate: float) -> None:
    """Reject rates outside [0, 1]."""
    if not 0.0 <= rate <= 1.0:
        raise AttackRateError(f"rate must be in [0, 1], got {rate}")


def _edit_count(n_tokens: int, rate: float) -> int:
    """Return the number of tokens to edit for a given rate."""
    _check_rate(rate)
    return round(rate * n_tokens)


def substitute_tokens(
    ids: Sequence[int], rate: float, vocab_size: int, rng: np.random.Generator
) -> list[int]:
    """Replace a fraction of the tokens by uniformly random, different tokens.

    Args:
        ids: Token ids.
        rate: Fraction of positions to replace, in [0, 1].
        vocab_size: Vocabulary size (replacements are drawn in [0, vocab_size)).
        rng: Random generator.

    Returns:
        A new list with the same length as `ids`.
    """
    if vocab_size < 2:
        raise AttackRateError(f"vocab_size must be >= 2, got {vocab_size}")
    out = [int(token) for token in ids]
    n_edits = _edit_count(len(out), rate)
    if n_edits == 0:
        return out
    positions = rng.choice(len(out), size=n_edits, replace=False)
    # Draw in [0, V-1) then shift past the original token: uniform over the V-1 other tokens.
    draws = rng.integers(0, vocab_size - 1, size=n_edits)
    for position, draw in zip(positions.tolist(), draws.tolist(), strict=True):
        out[position] = draw + 1 if draw >= out[position] else draw
    return out


def delete_tokens(ids: Sequence[int], rate: float, rng: np.random.Generator) -> list[int]:
    """Delete a fraction of the tokens at random positions.

    Args:
        ids: Token ids.
        rate: Fraction of positions to delete, in [0, 1].
        rng: Random generator.

    Returns:
        A new, shorter list (length `len(ids) - round(rate * len(ids))`).
    """
    out = [int(token) for token in ids]
    n_edits = _edit_count(len(out), rate)
    if n_edits == 0:
        return out
    removed = set(rng.choice(len(out), size=n_edits, replace=False).tolist())
    return [token for position, token in enumerate(out) if position not in removed]


def insert_tokens(
    ids: Sequence[int], rate: float, vocab_size: int, rng: np.random.Generator
) -> list[int]:
    """Insert random tokens at random positions.

    Args:
        ids: Token ids.
        rate: Number of insertions as a fraction of `len(ids)`, in [0, 1].
        vocab_size: Vocabulary size (inserted tokens are uniform in [0, vocab_size)).
        rng: Random generator.

    Returns:
        A new, longer list (length `len(ids) + round(rate * len(ids))`).
    """
    if vocab_size < 1:
        raise AttackRateError(f"vocab_size must be >= 1, got {vocab_size}")
    out = [int(token) for token in ids]
    n_edits = _edit_count(len(out), rate)
    if n_edits == 0:
        return out
    # Insert from the right so that earlier insertion points stay valid.
    points = sorted(rng.integers(0, len(out) + 1, size=n_edits).tolist(), reverse=True)
    new_tokens = rng.integers(0, vocab_size, size=n_edits).tolist()
    for point, token in zip(points, new_tokens, strict=True):
        out.insert(point, token)
    return out
