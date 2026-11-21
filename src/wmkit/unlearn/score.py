"""NeurIPS 2023 Machine Unlearning Competition score (Triantafillou, Kurmanji et al., 2406.09073).

The official score of one submission is

    score = F * (RA_u / RA_r) * (TA_u / TA_r)                                  (eq. 2)

where `F` is the *forgetting quality*, `RA` the accuracy on the retain set and `TA` the accuracy on
the test set, for the unlearned model (subscript `u`) and the model retrained from scratch without
the forget set (subscript `r`). The forgetting quality averages a per-example term over the forget
set `S` (sec. 3.1.2):

    F = 1/|S| * sum_s  2 / 2**n(eps_s),        n(eps) = floor(eps / 0.5)

and the per-example privacy level `eps_s` is estimated from attack error rates (sec. 4):

    eps_hat = max( log((1 - delta - FPR) / FNR), log((1 - delta - FNR) / FPR) )

Note on the `F` range: with a `floor`, `n = 0` for `eps < 0.5` and the per-example term is `2`, so
`F` lives in `(0, 2]`. This reproduces `docs/research/competition-intel.md` section 3.2 verbatim;
whether the organisers cap the terms at 1 is marked `[A VERIFIER]` there, and is not assumed here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

EPSILON_BIN_WIDTH: float = 0.5
"""Width of one epsilon bin: `n(eps) = floor(eps / EPSILON_BIN_WIDTH)`."""

FORGET_QUALITY_NUMERATOR: float = 2.0
"""Numerator of the per-example forgetting quality `2 / 2**n(eps)`."""

DEFAULT_DELTA: float = 0.0
"""Default `delta` of the (epsilon, delta)-unlearning definition used by the epsilon estimator."""


class ScoreError(ValueError):
    """Raised when a score input is outside the domain of the competition formulas."""


def epsilon_bin(epsilon: float) -> int:
    """Return `n(eps) = floor(eps / 0.5)`, with negative estimates clamped to bin 0.

    A negative `eps_hat` means the attack did no better than chance; it carries no extra privacy
    credit, so it is treated as `eps = 0` rather than yielding a per-example term above 2.

    Args:
        epsilon: Estimated privacy level of one forget example (may be `inf`).

    Returns:
        The bin index (non-negative integer).

    Raises:
        ScoreError: If `epsilon` is NaN or `+inf` (an infinite epsilon has no bin: its quality
            term is exactly 0, handled by `forget_quality_from_epsilons`).
    """
    if math.isnan(epsilon):
        raise ScoreError("epsilon must not be NaN")
    if epsilon == math.inf:
        raise ScoreError("epsilon_bin is undefined for +inf; its quality term is exactly 0")
    if epsilon <= 0.0:
        return 0  # any non-positive epsilon (including -inf) lands in bin 0
    return max(0, math.floor(epsilon / EPSILON_BIN_WIDTH))


def forget_quality_from_epsilons(eps: Sequence[float]) -> float:
    """Average forgetting quality `F = mean_s 2 / 2**floor(eps_s / 0.5)`.

    Examples (checked by hand in the tests): `[0.0] -> 2.0`, `[0.5] -> 1.0`, `[1.0] -> 0.5`,
    `[0.0, 0.5, 1.0, 1.5] -> (2 + 1 + 0.5 + 0.25) / 4 = 0.9375`.

    Args:
        eps: One estimated epsilon per forget example. `+inf` (perfectly distinguishable) gives a
            term of exactly 0; negative values are clamped to bin 0.

    Returns:
        The mean per-example quality, a float in `(0, 2]` for finite inputs.

    Raises:
        ScoreError: If `eps` is empty or contains a NaN.
    """
    if len(eps) == 0:
        raise ScoreError("forget_quality_from_epsilons needs at least one epsilon")
    terms: list[float] = []
    for value in eps:
        if math.isnan(value):
            raise ScoreError("epsilon must not be NaN")
        if value == math.inf:
            terms.append(0.0)
            continue
        # ldexp(2, -n) == 2 / 2**n, exact in binary floating point and underflowing cleanly to 0.
        terms.append(math.ldexp(FORGET_QUALITY_NUMERATOR, -epsilon_bin(value)))
    return math.fsum(terms) / len(terms)


def neurips2023_score(
    forget_quality: float, ra_u: float, ra_r: float, ta_u: float, ta_r: float
) -> float:
    """Competition score `F * (RA_u / RA_r) * (TA_u / TA_r)`.

    Args:
        forget_quality: `F`, e.g. from `forget_quality_from_epsilons`.
        ra_u: Retain accuracy of the unlearned model.
        ra_r: Retain accuracy of the retrained reference model (must be > 0).
        ta_u: Test accuracy of the unlearned model.
        ta_r: Test accuracy of the retrained reference model (must be > 0).

    Returns:
        The product of the forgetting quality and the two utility ratios.

    Raises:
        ScoreError: If a reference accuracy is not strictly positive or an input is NaN/negative.
    """
    for name, value in (
        ("forget_quality", forget_quality),
        ("ra_u", ra_u),
        ("ra_r", ra_r),
        ("ta_u", ta_u),
        ("ta_r", ta_r),
    ):
        if math.isnan(value) or value < 0.0:
            raise ScoreError(f"{name} must be a non-negative number, got {value}")
    if ra_r == 0.0 or ta_r == 0.0:
        raise ScoreError("reference accuracies ra_r and ta_r must be strictly positive")
    return forget_quality * (ra_u / ra_r) * (ta_u / ta_r)


def _log_odds_term(numerator: float, denominator: float) -> float:
    """Return `log(numerator / denominator)` with the conventions `num <= 0 -> -inf`, `den == 0 -> +inf`."""
    if numerator <= 0.0:
        return -math.inf
    if denominator <= 0.0:
        return math.inf
    return math.log(numerator / denominator)


def estimate_epsilon(fpr: float, fnr: float, delta: float = DEFAULT_DELTA) -> float:
    """Empirical privacy level `eps_hat` of one attack operating point (competition sec. 4).

    `eps_hat = max(log((1 - delta - FPR)/FNR), log((1 - delta - FNR)/FPR))`, clamped to be `>= 0`.
    Perfect separation (`FPR == 0` or `FNR == 0` with a positive numerator) gives `+inf`.

    Args:
        fpr: False positive rate of the membership attack, in `[0, 1]`.
        fnr: False negative rate of the membership attack, in `[0, 1]`.
        delta: The `delta` of the (epsilon, delta)-unlearning definition, in `[0, 1)`.

    Returns:
        `eps_hat >= 0`, possibly `math.inf`.

    Raises:
        ScoreError: If a rate or `delta` is outside its range.
    """
    for name, value in (("fpr", fpr), ("fnr", fnr)):
        if not 0.0 <= value <= 1.0:
            raise ScoreError(f"{name} must lie in [0, 1], got {value}")
    if not 0.0 <= delta < 1.0:
        raise ScoreError(f"delta must lie in [0, 1), got {delta}")
    first = _log_odds_term(1.0 - delta - fpr, fnr)
    second = _log_odds_term(1.0 - delta - fnr, fpr)
    return max(0.0, first, second)
