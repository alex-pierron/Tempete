"""Exact-agreement statistic restricted to low-entropy prompts (team D).

Why: the `ep-*` endpoints are chat models that are close to deterministic (on the 5 prompts of the
first ep phase, 7 endpoints return the SAME string on all 5), whereas the `ref-*` model is tiny and
noisy: its answer to a trivial multiple-choice question changes from one endpoint to the next even
for the same model. On a noisy model, exact agreement is dominated by sampling noise; on a
near-deterministic one, a flip of a single token (fp8 rounding, a LoRA nudge) is visible as a
mismatch. The statistic must therefore be evaluated where the S endpoints agree among themselves.

Selection without labels: for each cell (prompt, repeat) the consensus is the share of the
endpoints that give the modal answer. Cells with a strong consensus are the low-entropy ones; the
rule uses no endpoint label, so it can be applied unchanged to the 33 scored endpoints.

Distance of two endpoints on a set of cells = mean exact-mismatch indicator.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from hackathon.model_equality.research.d_eval import (
    EVAL_SEED,
    AucSummary,
    Design,
    Roles,
    auc_for_roles,
    draw_cells,
    summarise,
)

WHITESPACE = re.compile(r"\s+")
TRAILING_PUNCTUATION = ".!?\"' "
FloatArray = NDArray[np.float64]
Normaliser = Callable[[str], str]


def exact_key(response: str) -> str:
    """The reply itself, surrounding whitespace removed."""
    return response.strip()


def normalised_key(response: str) -> str:
    """Lower case, collapsed whitespace, no trailing punctuation: `Yes.` and `yes` agree."""
    return WHITESPACE.sub(" ", response.strip().lower()).strip(TRAILING_PUNCTUATION)


NORMALISERS: dict[str, Normaliser] = {"exact": exact_key, "normalised": normalised_key}


def answer_keys(design: Design, normaliser: Normaliser) -> list[list[str]]:
    """`keys[endpoint][cell]` after normalisation."""
    return [[normaliser(r) for r in design.responses[e]] for e in design.endpoints]


def consensus_fraction(keys: Sequence[Sequence[str]]) -> FloatArray:
    """Per cell, the share of endpoints that give the modal answer."""
    endpoints = len(keys)
    cells = len(keys[0])
    return np.asarray(
        [Counter(keys[e][c] for e in range(endpoints)).most_common(1)[0][1] / endpoints
         for c in range(cells)], dtype=np.float64)


def mismatch_tensor(keys: Sequence[Sequence[str]]) -> FloatArray:
    """Mismatch indicators, shape (cells, endpoints, endpoints)."""
    endpoints = len(keys)
    cells = len(keys[0])
    tensor = np.zeros((cells, endpoints, endpoints), dtype=np.float64)
    for c in range(cells):
        column = [keys[e][c] for e in range(endpoints)]
        for a in range(endpoints):
            for b in range(endpoints):
                tensor[c, a, b] = float(column[a] != column[b])
    return tensor


def exact_auc_curve(
    design: Design,
    roles: Roles,
    normaliser: Normaliser,
    threshold: float,
    sample_sizes: Sequence[int],
    replicates: int,
) -> tuple[int, dict[int, dict[str, AucSummary]]]:
    """AUC vs N of the exact-mismatch distance on the cells whose consensus is >= `threshold`.

    Returns:
        `(selected cell count, curve)`; sample sizes above the selected count are skipped.
    """
    keys = answer_keys(design, normaliser)
    consensus = consensus_fraction(keys)
    selected = np.flatnonzero(consensus >= threshold)
    tensor = mismatch_tensor(keys)[selected]
    curve: dict[int, dict[str, AucSummary]] = {}
    for n in sample_sizes:
        if n > len(selected):
            continue
        draws = draw_cells(len(selected), n, replicates, EVAL_SEED)
        distances = tensor[draws].mean(axis=1)  # (R, E, E)
        aucs = auc_for_roles(distances, design.endpoints, roles)
        curve[n] = {name: summarise(values) for name, values in aucs.items()}
    return len(selected), curve


MIN_REPEATS_FOR_COLLISION = 2


def collision_tensor(design: Design, normaliser: Normaliser) -> FloatArray:
    """Repeat-level statistics per prompt, shape (2, endpoints, prompts).

    Feature 0 is the collision rate (share of pairs of repeats of one endpoint that give the same
    normalised answer), feature 1 the diversity (distinct answers over repeats), both measured
    prompt by prompt on the prompts that have at least `MIN_REPEATS_FOR_COLLISION` repeats in the
    design. A noisy endpoint (high temperature) has low collisions and high diversity.

    Raises:
        ValueError: No prompt has enough repeats.
    """
    cells_of: dict[str, list[int]] = {}
    for index, (prompt_id, _repeat) in enumerate(design.cells):
        cells_of.setdefault(prompt_id, []).append(index)
    prompts = [p for p, cells in cells_of.items() if len(cells) >= MIN_REPEATS_FOR_COLLISION]
    if not prompts:
        raise ValueError("no prompt with at least two repeats")
    keys = answer_keys(design, normaliser)
    tensor = np.zeros((2, len(design.endpoints), len(prompts)), dtype=np.float64)
    for e in range(len(design.endpoints)):
        for j, prompt_id in enumerate(prompts):
            answers = [keys[e][c] for c in cells_of[prompt_id]]
            pairs = [(a, b) for i, a in enumerate(answers) for b in answers[i + 1:]]
            tensor[0, e, j] = float(np.mean([a == b for a, b in pairs]))
            tensor[1, e, j] = len(set(answers)) / len(answers)
    return tensor
