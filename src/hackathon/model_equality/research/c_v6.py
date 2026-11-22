"""Configuration v6: likelihood chain (mean log-likelihood, fp8 ratio, identity guard-rail).

Why v6 replaces the full-stop chain. On the zoo of team F (three real base models, ten known
variants each) the full-stop signal of v2 to v5 does not exist for Qwen (0 % of the arithmetic
answers end with a full stop in all ten variants, the answers are almost deterministic at
T = 0.7) and depends on the adapter for Llama: those rules score 0.00, 0.00 and 0.30. The
likelihood of an endpoint's answers under the base model of its family does carry the signal:
an fp8 copy has a higher fp8 log-likelihood ratio, a LoRA a clearly lower mean log-likelihood.

Per family of endpoints (the three exact families of r-met-e), from the answers of each endpoint:

1. disguised (identity override of `c_identity`, k identity answers): never isolated;
2. LoRA candidate: among the other endpoints, the mean log-likelihood per token (under the
   family's base model) is lower than the family centre (median over the non-disguised
   endpoints) by at least `ll_relative_drop` x |centre| AND by at least `ll_mad_k` robust
   standard deviations (1.4826 x MAD of the non-disguised endpoints): each one is its own model;
   a temperature-1.0 or watermarked copy shifts the mean log-likelihood by 20 to 40 % of |centre|
   on the zoo, a LoRA of strength 0.7 or more by 70 % and above, hence the relative floor 0.5;
3. fp8 candidate: among the endpoints still in the main class, the fp8 log-likelihood ratio per
   token relative to the family median, in standard errors `fp8_sigma_560 x sqrt(560 / tokens)`
   (r-met-e's definition: 0.0058 at 560 tokens on the real ref); every endpoint with
   z >= `fp8_margin` joins ONE fp8 model (the zoo truth gives the weights-only and the W8A8 copies
   the same model);
4. every other endpoint is the main model of its family.

The fitted-temperature classes of `e_partition` (temperature ~ 1, deviating) need tempered
statistics that only the scoring pipeline has: they are an optional second read-out, not an input
of this module. Nothing here touches the network or a model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hackathon.model_equality.research.c_pairs import Partition

LL_RELATIVE_DROP = 0.5
LL_MAD_K = 3.0
FP8_MARGIN = 2.5
FP8_SIGMA_AT_560_TOKENS = 0.0058
REFERENCE_TOKENS = 560
MAD_TO_SIGMA = 1.4826
MIN_ELIGIBLE = 4
"""Fewer non-disguised endpoints than this in a family: no centre, nothing is isolated."""


@dataclass(frozen=True)
class RuleV6:
    """Frozen parameters of the likelihood chain."""

    ll_relative_drop: float = LL_RELATIVE_DROP
    ll_mad_k: float = LL_MAD_K
    fp8_margin: float = FP8_MARGIN
    fp8_sigma_at_560: float = FP8_SIGMA_AT_560_TOKENS


@dataclass(frozen=True)
class EndpointStats:
    """Pooled statistics of one endpoint: tokens, mean log-likelihood, fp8 ratio, disguise."""

    tokens: int
    mean_ll: float
    fp8_llr: float
    disguised: bool


def lora_flags(stats: Sequence[EndpointStats], rule: RuleV6) -> list[bool]:
    """LoRA candidates of ONE family: a large and significant drop of the mean log-likelihood."""
    eligible = [i for i, s in enumerate(stats) if not s.disguised]
    flags = [False] * len(stats)
    if len(eligible) < MIN_ELIGIBLE:
        return flags
    values = np.array([stats[i].mean_ll for i in eligible])
    centre = float(np.median(values))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(values - centre)))
    drop = max(rule.ll_relative_drop * abs(centre), rule.ll_mad_k * sigma)
    for i in eligible:
        flags[i] = stats[i].mean_ll <= centre - drop
    return flags


def fp8_flags(stats: Sequence[EndpointStats], eligible: Sequence[bool], rule: RuleV6) -> list[bool]:
    """fp8 candidates of ONE family among the `eligible` endpoints (z of the ratio vs the median)."""
    pool = [i for i, ok in enumerate(eligible) if ok]
    flags = [False] * len(stats)
    if len(pool) < MIN_ELIGIBLE:
        return flags
    median = float(np.median([stats[i].fp8_llr for i in pool]))
    for i in pool:
        sigma = rule.fp8_sigma_at_560 * math.sqrt(REFERENCE_TOKENS / max(stats[i].tokens, 1))
        flags[i] = (stats[i].fp8_llr - median) / sigma >= rule.fp8_margin
    return flags


def decide_family(stats: Sequence[EndpointStats], rule: RuleV6) -> tuple[list[bool], list[bool]]:
    """`(lora flags, fp8 flags)` of one family; a disguised endpoint is never flagged."""
    loras = lora_flags(stats, rule)
    eligible = [not (s.disguised or lora) for s, lora in zip(stats, loras, strict=True)]
    return loras, fp8_flags(stats, eligible, rule)


def decide_v6(
    family_of: Sequence[int], stats: Sequence[EndpointStats], rule: RuleV6
) -> tuple[Partition, list[str]]:
    """Partition (exact families) and a role per endpoint: `main`, `lora` or `fp8`."""
    roles = ["main"] * len(stats)
    groups: list[list[int]] = []
    family_of_group: list[int] = []
    for family in sorted(set(family_of)):
        members = [i for i, f in enumerate(family_of) if f == family]
        loras, fp8s = decide_family([stats[i] for i in members], rule)
        for position, lora, fp8 in zip(members, loras, fp8s, strict=True):
            roles[position] = "lora" if lora else "fp8" if fp8 else "main"
        main = [i for i in members if roles[i] == "main"]
        fp8_members = [i for i in members if roles[i] == "fp8"]
        for group in ([main] if main else []) + ([fp8_members] if fp8_members else []):
            groups.append(group)
            family_of_group.append(family)
        for position in members:
            if roles[position] == "lora":
                groups.append([position])
                family_of_group.append(family)
    return Partition.from_groups(groups, family_of_group), roles
