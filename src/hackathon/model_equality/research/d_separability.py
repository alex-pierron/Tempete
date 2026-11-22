"""Separability of every ref endpoint from the base model S, by statistic and by prompt family (team D).

Question of the director: for N = 10, 20, 30, 50 answers per endpoint, how separable is S from fp8,
from the LoRA endpoints and from the system-prompt disguise, and which prompt categories carry the
signal? One number per cell of the tables: the AUC of the pairs (target, S endpoint) against the
pairs (S endpoint, S endpoint), the target being left out of the S group when it is itself an S
disguise. For a LoRA or fp8 target a high AUC is GOOD (the model change is seen); for a disguise
target (system prompt, persona, watermark, temperature) a high AUC is BAD (the statistic mistakes a
disguise for a model change).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from hackathon.model_equality.research.d_eval import (
    BASE_MODEL,
    Design,
    Roles,
    auc_curve_for_tensor,
    center_by_cell,
    feature_tensor,
    restrict_design,
)
from hackathon.model_equality.research.d_select import prompt_family

SEPARABILITY_SIZES = (10, 20, 30, 50)
FAMILY_MIN_CELLS = 20
FAMILY_SIZE = 10
FloatArray = NDArray[np.float64]


def roles_for_target(model_of: Mapping[str, str], target: str) -> Roles:
    """Roles where `target` is the only positive endpoint and the other S endpoints are the negatives."""
    base = tuple(sorted(e for e, m in model_of.items() if m == BASE_MODEL and e != target))
    return Roles(base=base, fp8=(target,), lora_weak=(target,))


def separability_table(
    design: Design,
    model_of: Mapping[str, str],
    targets: Sequence[str],
    statistics: Mapping[str, FloatArray],
    sizes: Sequence[int],
    replicates: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """`table[statistic][target][N]` = mean AUC of the target against the S group.

    Args:
        design: Answers on the common cells.
        model_of: `endpoint -> model label`.
        targets: Endpoints tested one by one.
        statistics: Named (1, endpoints, cells) tensors, ALREADY cell-centred.
        sizes: Sample sizes (clipped to the number of cells).
        replicates: Draws per size.
    """
    usable = tuple(n for n in sizes if n <= len(design.cells))
    table: dict[str, dict[str, dict[str, float]]] = {}
    for name, tensor in statistics.items():
        table[name] = {}
        for target in targets:
            roles = roles_for_target(model_of, target)
            curve = auc_curve_for_tensor(tensor, design.endpoints, roles, np.ones(1), usable, replicates)
            table[name][target] = {str(n): curve[n]["fp8"].mean for n in usable}
    return table


def family_table(
    design: Design,
    model_of: Mapping[str, str],
    targets: Sequence[str],
    feature_names: Sequence[str],
    replicates: int,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """`table[family][feature][target]` = AUC at N = min(FAMILY_SIZE, cells) on the cells of one family.

    Only families with at least `FAMILY_MIN_CELLS` cells are kept. The families are chosen after
    looking at the data: read the result as a lead, not as a validated number.
    """
    families = sorted({prompt_family(p) for p, _ in design.cells})
    table: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for family in families:
        keep = [i for i, (p, _r) in enumerate(design.cells) if prompt_family(p) == family]
        if len(keep) < FAMILY_MIN_CELLS:
            continue
        sub = restrict_design(design, keep)
        tensor = feature_tensor(sub, feature_names)
        entry: dict[str, dict[str, dict[str, float]]] = {"cells": {"n": {"cells": float(len(keep))}}}
        for k, feature in enumerate(feature_names):
            single = center_by_cell(tensor[k:k + 1])
            entry[feature] = {}
            for target in targets:
                roles = roles_for_target(model_of, target)
                curve = auc_curve_for_tensor(single, sub.endpoints, roles, np.ones(1), (FAMILY_SIZE,), replicates)
                entry[feature][target] = {str(FAMILY_SIZE): curve[FAMILY_SIZE]["fp8"].mean}
        table[family] = entry
    return table
