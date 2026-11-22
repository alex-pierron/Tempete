"""Selection-honest estimate of what a screened statistic is worth (team D).

Screening 22 features over about 14 prompt families and keeping the best AUC is a search over
hundreds of candidates on 9 endpoints: the winner's AUC is optimistic. This module measures HOW
optimistic. In each of `splits` random halvings of the PROMPTS, the best (family, feature) pair is
chosen on the first half (at `selection_size` answers per endpoint) and its AUC is then measured on
the second half, whose prompts the choice never saw. A statistic that really works keeps its AUC
on the second half; a lucky one falls back to 0.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hackathon.model_equality.research.d_eval import (
    EVAL_SEED,
    Design,
    Roles,
    auc_for_roles,
    center_by_cell,
    draw_cells,
    feature_tensor,
    restrict_design,
    scalar_distances,
)

CANDIDATE_PREFIX = "cand-"
SELECTION_SIZE = 20
EVALUATION_SIZE = 20
MIN_CELLS_FOR_GROUP = 20
SELECTION_REPLICATES = 60
EVALUATION_REPLICATES = 200
WHOLE_SET_GROUP = "all"
FloatArray = NDArray[np.float64]


def prompt_family(prompt_id: str) -> str:
    """Family of a prompt id: `cand-arith-003` -> `arith`, `fact-010` -> `fact`."""
    stem = prompt_id.removeprefix(CANDIDATE_PREFIX)
    return stem.rsplit("-", 1)[0]


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of the repeated select-then-evaluate experiment.

    Attributes:
        selected_auc: AUC of the winner on the half it was chosen on (optimistic).
        held_out_auc: AUC of the same winner on the other half (honest).
        winners: `(group, feature)` chosen at each split.
        mean_candidate_auc: Mean held-out AUC over all candidates (the no-selection baseline).
    """

    selected_auc: tuple[float, ...]
    held_out_auc: tuple[float, ...]
    winners: tuple[tuple[str, str], ...]
    mean_candidate_auc: float


def _group_cells(design: Design, group: str, allowed_prompts: set[str]) -> list[int]:
    """Cell indices of a group (or of everything) restricted to the allowed prompts."""
    return [
        index
        for index, (prompt_id, _repeat) in enumerate(design.cells)
        if prompt_id in allowed_prompts
        and (group == WHOLE_SET_GROUP or prompt_family(prompt_id) == group)
    ]


def _auc_both(
    tensor: FloatArray, design: Design, roles: Roles, cells: Sequence[int], sample_size: int,
    replicates: int, feature_index: int,
) -> float:
    """Mean AUC (fp8 and LoRA 0.3 against S) of one feature on a cell subset."""
    sub = tensor[:, :, list(cells)]
    sub = center_by_cell(sub)
    weights = np.zeros(sub.shape[0], dtype=np.float64)
    weights[feature_index] = 1.0
    draws = draw_cells(sub.shape[2], min(sample_size, sub.shape[2]), replicates, EVAL_SEED)
    distances = scalar_distances(sub, draws, weights)
    return float(auc_for_roles(distances, design.endpoints, roles)["both"].mean())


def selection_experiment(
    design: Design,
    roles: Roles,
    feature_names: Sequence[str],
    splits: int,
    seed: int,
) -> SelectionResult:
    """Repeat select-on-half-one, evaluate-on-half-two `splits` times."""
    tensor = feature_tensor(design, feature_names)
    prompts = sorted({prompt_id for prompt_id, _ in design.cells})
    groups = [WHOLE_SET_GROUP, *sorted({prompt_family(p) for p in prompts})]
    rng = np.random.default_rng(seed)
    selected: list[float] = []
    held_out: list[float] = []
    winners: list[tuple[str, str]] = []
    baseline: list[float] = []
    for _ in range(splits):
        order = rng.permutation(len(prompts))
        half = {prompts[i] for i in order[: len(prompts) // 2]}
        other = set(prompts) - half
        best = (-1.0, WHOLE_SET_GROUP, 0)
        for group in groups:
            cells = _group_cells(design, group, half)
            if len(cells) < MIN_CELLS_FOR_GROUP:
                continue
            for k in range(len(feature_names)):
                value = _auc_both(
                    tensor, design, roles, cells, SELECTION_SIZE, SELECTION_REPLICATES, k
                )
                if value > best[0]:
                    best = (value, group, k)
        if best[0] < 0.0:
            continue
        auc_half, group, k = best
        held_cells = _group_cells(design, group, other)
        if len(held_cells) < MIN_CELLS_FOR_GROUP:
            continue
        selected.append(auc_half)
        held_out.append(
            _auc_both(tensor, design, roles, held_cells, EVALUATION_SIZE, EVALUATION_REPLICATES, k)
        )
        winners.append((group, feature_names[k]))
        if not baseline:
            # No-selection reference: every candidate (group, feature) on the unseen half, once.
            for g in groups:
                cells_g = _group_cells(design, g, other)
                if len(cells_g) < MIN_CELLS_FOR_GROUP:
                    continue
                for feature_index in range(len(feature_names)):
                    baseline.append(
                        _auc_both(tensor, design, roles, cells_g, EVALUATION_SIZE,
                                  SELECTION_REPLICATES, feature_index)
                    )
    return SelectionResult(
        tuple(selected), tuple(held_out), tuple(winners), float(np.mean(baseline)) if baseline else float("nan")
    )


def whole_design_subset(design: Design, prompts: set[str]) -> Design:
    """The design reduced to the cells of the given prompts."""
    keep = [i for i, (p, _r) in enumerate(design.cells) if p in prompts]
    return restrict_design(design, keep)
