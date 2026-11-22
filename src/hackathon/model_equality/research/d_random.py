"""IRIS-type test (wave d8): does the symbol distribution of random-digit / coin-flip replies separate fp8 from plain S?

Each reply is reduced to a vector of symbol counts (digits 0-9, or H and T). Two endpoints are compared
with a chi-square homogeneity statistic on the pooled counts, and its null distribution comes from a
permutation of the REPLIES (a reply is the unit: its symbols are dependent) between the two endpoints.

The decision rule was written in `d_prompts.py` before any d8 answer: fp8 is SEPARATED when its
permutation p-value against the plain endpoint is <= `ALPHA` on at least one prompt while every
disguise has a p-value above `ALPHA` on that same prompt.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hackathon.common.errors import HackathonError
from hackathon.model_equality.store import Answer

ALPHA = 0.01
PERMUTATIONS = 2000
PERMUTATION_SEED = 20261121
DIGIT_PATTERN = re.compile(r"\d")
COIN_PATTERN = re.compile(r"\b([HT])\b")
PLAIN_ENDPOINT = "ref-01"
FP8_ENDPOINT = "ref-02"
DISGUISE_ENDPOINTS = ("ref-03", "ref-05", "ref-06", "ref-08")
DIGIT_SYMBOLS = tuple("0123456789")
COIN_SYMBOLS = ("H", "T")
MIN_REPLIES = 5
FloatArray = NDArray[np.float64]


class RandomTestError(HackathonError):
    """Not enough replies to run the test."""


def symbol_counts(response: str, symbols: Sequence[str]) -> FloatArray:
    """Count of each symbol in one reply (digits, or the letters H and T standing alone)."""
    found = DIGIT_PATTERN.findall(response) if symbols[0].isdigit() else COIN_PATTERN.findall(response.upper())
    counts = np.zeros(len(symbols), dtype=np.float64)
    for symbol in found:
        if symbol in symbols:
            counts[symbols.index(symbol)] += 1.0
    return counts


def homogeneity_statistic(first: FloatArray, second: FloatArray) -> float:
    """Chi-square homogeneity statistic of two pooled count vectors (0 when a margin is empty)."""
    total = first + second
    n_first, n_second = float(first.sum()), float(second.sum())
    if n_first == 0.0 or n_second == 0.0:
        return 0.0
    grand = n_first + n_second
    keep = total > 0
    expected_first = total[keep] * n_first / grand
    expected_second = total[keep] * n_second / grand
    return float(((first[keep] - expected_first) ** 2 / expected_first
                  + (second[keep] - expected_second) ** 2 / expected_second).sum())


@dataclass(frozen=True)
class ComparisonResult:
    """One endpoint against the plain endpoint on one prompt.

    Attributes:
        statistic: Chi-square homogeneity statistic of the pooled symbol counts.
        p_value: Permutation p-value (replies as units).
        replies: Replies of the two endpoints (first, second).
        symbols_per_reply: Mean number of symbols per reply (first, second).
    """

    statistic: float
    p_value: float
    replies: tuple[int, int]
    symbols_per_reply: tuple[float, float]


def compare_endpoints(
    first: Sequence[str], second: Sequence[str], symbols: Sequence[str], permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> ComparisonResult:
    """Permutation test between two sets of replies.

    Raises:
        RandomTestError: Fewer than `MIN_REPLIES` replies on one side.
    """
    if len(first) < MIN_REPLIES or len(second) < MIN_REPLIES:
        raise RandomTestError(f"need at least {MIN_REPLIES} replies per endpoint, got {len(first)} and {len(second)}")
    matrix = np.stack([symbol_counts(r, symbols) for r in [*first, *second]])
    n_first = len(first)
    observed = homogeneity_statistic(matrix[:n_first].sum(axis=0), matrix[n_first:].sum(axis=0))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(len(matrix))
        statistic = homogeneity_statistic(matrix[order[:n_first]].sum(axis=0), matrix[order[n_first:]].sum(axis=0))
        exceed += statistic >= observed - 1e-12
    p_value = (exceed + 1) / (permutations + 1)
    return ComparisonResult(
        observed, p_value, (len(first), len(second)),
        (float(matrix[:n_first].sum(axis=1).mean()), float(matrix[n_first:].sum(axis=1).mean())))


def replies_by_endpoint(answers: Sequence[Answer], prompt_id: str) -> dict[str, list[str]]:
    """Replies of each endpoint to one prompt."""
    out: dict[str, list[str]] = {}
    for answer in answers:
        if answer.prompt_id == prompt_id:
            out.setdefault(answer.endpoint, []).append(answer.response)
    return out


def d8_verdict(per_prompt: Mapping[str, Mapping[str, ComparisonResult]]) -> str:
    """Apply the pre-registered rule: fp8 separated on some prompt while no disguise is."""
    for results in per_prompt.values():
        fp8 = results.get(FP8_ENDPOINT)
        if fp8 is None or fp8.p_value > ALPHA:
            continue
        if all(results[e].p_value > ALPHA for e in DISGUISE_ENDPOINTS if e in results):
            return "fp8 SEPARE"
    return "fp8 NON separe"


def run_d8(answers: Sequence[Answer], prompts: Mapping[str, Sequence[str]]) -> dict[str, dict[str, ComparisonResult]]:
    """Every endpoint against ref-01, per prompt (`prompts` maps prompt id to its symbol alphabet)."""
    out: dict[str, dict[str, ComparisonResult]] = {}
    for prompt_id, symbols in prompts.items():
        replies = replies_by_endpoint(answers, prompt_id)
        if PLAIN_ENDPOINT not in replies:
            continue
        out[prompt_id] = {
            endpoint: compare_endpoints(replies[PLAIN_ENDPOINT], texts, symbols)
            for endpoint, texts in replies.items()
            if endpoint != PLAIN_ENDPOINT and len(texts) >= MIN_REPLIES}
    return out
