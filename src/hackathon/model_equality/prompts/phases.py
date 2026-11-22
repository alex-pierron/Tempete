"""Phased query plans: what to ask, in which order, and what it costs per endpoint.

Budget arithmetic for `ep-*` (100 answers per endpoint for the whole competition, default
reserve 10, so 90 spendable):

    ep-1   40 answers   (40 % of the budget; the brief's ceiling for phase 1)
    ep-2   30 answers   (cumulative 70)
    ep-3   20 answers   (cumulative 90, i.e. the reserve of 10 stays untouched)

Every selected prompt is asked at least twice (`repeats` >= 2): the unbiased MMD needs
same-prompt pairs *inside* one endpoint to estimate the within-endpoint kernel mean. With one
sample per prompt that term is undefined and the statistic degenerates.

Phase 3 is meant to be adaptive: run it on the ambiguous endpoints only (`--endpoints`). The
budget is per endpoint, so endpoints that are already clearly placed can stop at 70 while the
ambiguous ones receive all of the 90.

The `ref-*` phases mirror the `ep-*` shapes (to calibrate thresholds at the same sample size)
and add `ref-full`, a large sweep over the whole bank used to measure which categories separate
quantization / LoRA while ignoring the disguises (ref queries are unlimited).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hackathon.model_equality.prompts.bank import (
    CATEGORIES,
    CATEGORY_ARITH,
    CATEGORY_CODE,
    CATEGORY_COMPLETION,
    CATEGORY_COUNT,
    CATEGORY_FACT,
    CATEGORY_FORMAT,
    CATEGORY_FREEFORM,
    CATEGORY_LIST,
    Prompt,
    build_bank,
)
from hackathon.model_equality.prompts.extra import EXTRA_PROMPTS

MIN_REPEATS_PER_PROMPT = 2
"""Smallest useful number of samples per (endpoint, prompt) for the unbiased MMD."""
REF_FULL_REPEATS = 3
ALL_PROMPTS = 1_000_000
"""Upper bound meaning "to the end of the category" in a selection."""
PHASE_1_NAME = "ep-1"
PHASE_2_NAME = "ep-2"
PHASE_3_NAME = "ep-3"


class PhaseError(ValueError):
    """An unknown phase, or a selection that the bank cannot satisfy."""


@dataclass(frozen=True)
class Selection:
    """A block of the bank: prompts `[first, last)` of one category, repeats `[rep_from, rep_to)`."""

    category: str
    first: int
    last: int
    rep_from: int
    rep_to: int


@dataclass(frozen=True)
class Phase:
    """A named list of selections, or an explicit list of prompt ids (one draw each by default).

    Attributes:
        name: Phase name (`ep-*` / `ref-*` prefix decides which endpoints it targets).
        description: One-line description shown by `plan`.
        selections: Blocks of the bank (category slices) with repeat ranges.
        prompt_ids: Explicit prompt ids (bank or extra prompts); when set, `selections` is unused.
        repeats: Draws per explicit prompt id.
    """

    name: str
    description: str
    selections: tuple[Selection, ...] = ()
    prompt_ids: tuple[str, ...] = ()
    repeats: int = 1


@dataclass(frozen=True)
class PromptRepeat:
    """One (prompt, repeat) unit of a phase."""

    prompt: Prompt
    repeat: int


_PHASE_1 = Phase(
    PHASE_1_NAME,
    "Broad first look: 20 prompts x 2 repeats over every category (40 answers).",
    (
        Selection(CATEGORY_ARITH, 0, 4, 0, 2),
        Selection(CATEGORY_COMPLETION, 0, 4, 0, 2),
        Selection(CATEGORY_FACT, 0, 3, 0, 2),
        Selection(CATEGORY_COUNT, 0, 2, 0, 2),
        Selection(CATEGORY_CODE, 0, 2, 0, 2),
        Selection(CATEGORY_LIST, 0, 2, 0, 2),
        Selection(CATEGORY_FORMAT, 0, 2, 0, 2),
        Selection(CATEGORY_FREEFORM, 0, 1, 0, 2),
    ),
)
_PHASE_2 = Phase(
    PHASE_2_NAME,
    "New prompts in the near-deterministic categories (30 answers); re-weight with the "
    "calibration before running it.",
    (
        Selection(CATEGORY_ARITH, 4, 7, 0, 2),
        Selection(CATEGORY_COMPLETION, 4, 7, 0, 2),
        Selection(CATEGORY_FACT, 3, 6, 0, 2),
        Selection(CATEGORY_FORMAT, 2, 5, 0, 2),
        Selection(CATEGORY_COUNT, 2, 3, 0, 2),
        Selection(CATEGORY_CODE, 2, 3, 0, 2),
        Selection(CATEGORY_LIST, 2, 3, 0, 2),
    ),
)
_PHASE_3 = Phase(
    PHASE_3_NAME,
    "Adaptive: two more repeats of the best phase-1 prompts (20 answers); run on the ambiguous "
    "endpoints only.",
    (
        Selection(CATEGORY_ARITH, 0, 4, 2, 4),
        Selection(CATEGORY_COMPLETION, 0, 4, 2, 4),
        Selection(CATEGORY_FORMAT, 0, 2, 2, 4),
    ),
)
_REF_FULL = Phase(
    "ref-full",
    "The whole bank x 3 repeats, for per-category calibration on ref endpoints (unlimited).",
    tuple(Selection(category, 0, ALL_PROMPTS, 0, REF_FULL_REPEATS) for category in CATEGORIES),
)

FAMILY_CANDIDATE_IDS: tuple[str, ...] = tuple(
    prompt.id for prompt in EXTRA_PROMPTS if prompt.id.startswith("fam-"))
"""Nine prompts written to tell base-model families apart (measured on ref, never on ep)."""
FAMILY_CANDIDATES_REPEATS = 2
REF_FAM_NAME = "ref-fam"

E1_PROMPT_IDS: tuple[str, ...] = (
    "fam-identity-000",
    "cand-yesno-002",
    "completion-000",
    "cand-yesno-005",
    "fact-003",
)
"""Phase E1 (5 % of the budget, decided by the humans on 21/11/2026): 5 prompts x 1 draw.
Two family prompts (identity; long-tail knowledge) and three LoRA-strength prompts chosen on the
ref answers with a held-out draw (see the m-met-1 report, section selection)."""
EP_E1_NAME = "ep-e1"
REF_E1_NAME = "ref-e1"

REF_M1_NAME = "ref-m1"
REF_M1_REPEATS = 3
REF_M1_PROMPT_IDS: tuple[str, ...] = (
    tuple(prompt.id for prompt in build_bank()) + tuple(prompt.id for prompt in EXTRA_PROMPTS)
)
"""Free ref sweep of phase 2: the whole bank and every extra prompt, 3 draws, repeat-major so a
stopped run is still balanced (every prompt once before any second draw)."""

REF_DEEP_NAME = "ref-deep"
REF_DEEP_PROMPT_IDS: tuple[str, ...] = ("cand-yesno-002", "cand-choice-002", "cand-yesno-005")
REF_DEEP_REPEATS = 33
"""Few prompts, many draws on the free ref: does an answer-distribution test (first token, length)
see fp8 and LoRA 0.3, which the many-prompts design never did? Decides whether the next ep budget
should go to depth (30 draws of 3 prompts) or to breadth."""

PHASES: dict[str, Phase] = {
    phase.name: phase
    for phase in (
        _PHASE_1,
        _PHASE_2,
        _PHASE_3,
        Phase("ref-1", "Same shape as ep-1, for calibration at equal sample size.",
              _PHASE_1.selections),
        Phase("ref-2", "Same shape as ep-2.", _PHASE_2.selections),
        Phase("ref-3", "Same shape as ep-3.", _PHASE_3.selections),
        _REF_FULL,
        Phase(EP_E1_NAME, "E1: 5 prompts x 1 draw (2 family, 3 LoRA-strength), the same on every "
              "endpoint (5 answers per endpoint).", prompt_ids=E1_PROMPT_IDS),
        Phase(REF_E1_NAME, "Same 5 prompts as ep-e1 on the ref endpoints (calibration at N = 5).",
              prompt_ids=E1_PROMPT_IDS),
        Phase(REF_M1_NAME, "ref only: bank + extra prompts x 3 draws (free sweep for MCC-vs-N "
              "curves and prompt selection at small N).", prompt_ids=REF_M1_PROMPT_IDS,
              repeats=REF_M1_REPEATS),
        Phase(REF_DEEP_NAME, "ref only: 3 short prompts x 33 draws (depth instead of breadth).",
              prompt_ids=REF_DEEP_PROMPT_IDS, repeats=REF_DEEP_REPEATS),
        Phase(REF_FAM_NAME, "ref only: the 9 family-candidate prompts x 2 draws, to measure how "
              "stable they are across disguises and LoRA / fp8 variants.",
              prompt_ids=FAMILY_CANDIDATE_IDS, repeats=FAMILY_CANDIDATES_REPEATS),
    )
}


def get_phase(name: str) -> Phase:
    """Look a phase up by name.

    Raises:
        PhaseError: Unknown name (the message lists the valid ones).
    """
    try:
        return PHASES[name]
    except KeyError as exc:
        raise PhaseError(f"unknown phase {name!r}; valid: {', '.join(PHASES)}") from exc


def expand_phase(
    phase: Phase,
    bank: Sequence[Prompt] | None = None,
    categories: Sequence[str] | None = None,
) -> list[PromptRepeat]:
    """Expand a phase into ordered (prompt, repeat) units, repeat-major.

    The repeat-major order means an interrupted run has covered every prompt once before it
    starts second samples, so a partial run is still a balanced one.

    Args:
        phase: The phase to expand.
        bank: Prompt bank (the built-in one by default).
        categories: Optional filter keeping only these categories.

    Raises:
        PhaseError: A selection starts beyond the end of its category.
    """
    prompts = list(bank) if bank is not None else build_bank()
    if phase.prompt_ids:
        return _expand_explicit(phase, prompts, categories)
    by_category: dict[str, list[Prompt]] = {}
    for prompt in prompts:
        by_category.setdefault(prompt.category, []).append(prompt)
    units: list[PromptRepeat] = []
    for selection in phase.selections:
        if categories is not None and selection.category not in categories:
            continue
        available = by_category.get(selection.category, [])
        if selection.first > 0 and selection.first >= len(available):
            raise PhaseError(
                f"{phase.name}: {selection.category} has {len(available)} prompts, "
                f"selection starts at {selection.first}"
            )
        chosen = available[selection.first:selection.last]
        for repeat in range(selection.rep_from, selection.rep_to):
            units.extend(PromptRepeat(prompt, repeat) for prompt in chosen)
    units.sort(key=lambda unit: unit.repeat)  # stable: keeps bank order inside a repeat
    return units


def _expand_explicit(
    phase: Phase, bank: Sequence[Prompt], categories: Sequence[str] | None
) -> list[PromptRepeat]:
    """Units of a phase defined by prompt ids (bank prompts and extra prompts), repeat-major."""
    catalog = {prompt.id: prompt for prompt in (*bank, *EXTRA_PROMPTS)}
    missing = [pid for pid in phase.prompt_ids if pid not in catalog]
    if missing:
        raise PhaseError(f"{phase.name}: unknown prompt ids {missing}")
    units = [
        PromptRepeat(catalog[pid], repeat)
        for repeat in range(phase.repeats) for pid in phase.prompt_ids
        if categories is None or catalog[pid].category in categories
    ]
    return units
