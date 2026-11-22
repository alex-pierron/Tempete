"""Team C: configuration v6 likelihood chain (mean log-likelihood, fp8 ratio, identity), no network."""

from __future__ import annotations

import pytest

from hackathon.model_equality.research.c_v6 import (
    FP8_MARGIN,
    LL_RELATIVE_DROP,
    EndpointStats,
    RuleV6,
    decide_family,
    decide_v6,
    fp8_flags,
    lora_flags,
)
from hackathon.model_equality.research.c_v6_zoo import long_answers_for, tokens_for

FAMILY_SIZE = 11
CENTRE_LL = -0.20
CLEAN_FP8 = -0.020
TOKENS = 3000
LORA_LL = -0.60
FP8_VALUE = 0.030


def clean(k: int = 0) -> EndpointStats:
    """A clean endpoint with a tiny deterministic spread."""
    return EndpointStats(TOKENS, CENTRE_LL + 0.004 * (k % 5 - 2), CLEAN_FP8 + 0.0003 * (k % 3),
                         False)


def family(**changes: EndpointStats) -> list[EndpointStats]:
    """Eleven clean endpoints with `e<k>` replaced."""
    stats = [clean(k) for k in range(FAMILY_SIZE)]
    for key, value in changes.items():
        stats[int(key[1:])] = value
    return stats


def test_a_large_drop_of_the_mean_log_likelihood_flags_a_lora() -> None:
    stats = family(e3=EndpointStats(TOKENS, LORA_LL, CLEAN_FP8, False))
    assert [i for i, f in enumerate(lora_flags(stats, RuleV6())) if f] == [3]


def test_a_temperature_or_watermark_sized_shift_is_not_a_lora() -> None:
    shift = CENTRE_LL * (1.0 + 0.4)
    stats = family(e3=EndpointStats(TOKENS, shift, CLEAN_FP8, False))
    assert not any(lora_flags(stats, RuleV6()))


def test_a_disguised_endpoint_is_never_flagged_even_with_a_huge_drop() -> None:
    stats = family(e3=EndpointStats(TOKENS, -2.0, FP8_VALUE, True))
    loras, fp8s = decide_family(stats, RuleV6())
    assert not any(loras) and not any(fp8s)


def test_fp8_is_flagged_among_the_main_class_only() -> None:
    stats = family(e2=EndpointStats(TOKENS, CENTRE_LL, FP8_VALUE, False),
                   e5=EndpointStats(TOKENS, LORA_LL, FP8_VALUE, False))
    loras, fp8s = decide_family(stats, RuleV6())
    assert loras[5] and not fp8s[5]
    assert fp8s[2] and not loras[2]


def test_every_fp8_endpoint_joins_one_model() -> None:
    stats = family(e2=EndpointStats(TOKENS, CENTRE_LL, FP8_VALUE, False),
                   e6=EndpointStats(TOKENS, CENTRE_LL, FP8_VALUE, False))
    partition, roles = decide_v6([0] * FAMILY_SIZE, stats, RuleV6())
    assert roles[2] == roles[6] == "fp8"
    assert [2, 6] in [sorted(g) for g in partition.model_groups()]
    assert len(partition.model_groups()) == 2


def test_standard_error_scales_with_tokens() -> None:
    few = family(e2=EndpointStats(560, CENTRE_LL, CLEAN_FP8 + 0.006, False))
    many = family(e2=EndpointStats(4480, CENTRE_LL, CLEAN_FP8 + 0.006, False))
    assert not any(fp8_flags(few, [True] * FAMILY_SIZE, RuleV6()))
    assert fp8_flags(many, [True] * FAMILY_SIZE, RuleV6())[2]


def test_too_few_eligible_endpoints_means_no_decision() -> None:
    stats = [EndpointStats(TOKENS, LORA_LL, FP8_VALUE, False)] * 3
    assert not any(lora_flags(stats, RuleV6()))
    assert not any(fp8_flags(stats, [True] * 3, RuleV6()))


def test_three_families_get_their_own_centre() -> None:
    """Llama-like mean log-likelihoods (about -1.4) do not make a Qwen-like family look off."""
    stats: list[EndpointStats] = []
    for scale in (1.0, 7.0, 1.0):
        stats += [EndpointStats(TOKENS, CENTRE_LL * scale, CLEAN_FP8, False)
                  for _ in range(FAMILY_SIZE)]
    stats[15] = EndpointStats(TOKENS, LORA_LL * 7.0, CLEAN_FP8, False)
    partition, roles = decide_v6([k // FAMILY_SIZE for k in range(3 * FAMILY_SIZE)], stats,
                                 RuleV6())
    assert [i for i, r in enumerate(roles) if r != "main"] == [15]
    assert len(partition.family_groups()) == 3


def test_defaults_and_token_conversion() -> None:
    assert RuleV6().fp8_margin == pytest.approx(FP8_MARGIN)
    assert RuleV6().ll_relative_drop == pytest.approx(LL_RELATIVE_DROP)
    assert tokens_for(14) == 563 + 256 * 14
    assert long_answers_for(tokens_for(14)) == pytest.approx(14.0)
    assert long_answers_for(100) == 0.0
