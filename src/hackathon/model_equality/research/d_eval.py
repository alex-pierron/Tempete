"""AUC of a pair statistic as a function of the number of answers per endpoint (team D).

Question answered: with N answers per endpoint, how well does a distance between two endpoints
rank the pairs "fp8 vs S" and "LoRA 0.3 vs S" (different models, positives) above the pairs
"S disguise vs S disguise" (same model, negatives)? An AUC of 0.5 is chance, 1.0 is perfect.

Design of the experiment
------------------------
* The `ref-*` endpoints answer the SAME cells (prompt id, repeat), like the scored endpoints will.
  A replicate draws N cells without replacement and gives every endpoint its answers on them.
* The negatives are the 10 pairs among the 5 endpoints of S (plain, system prompt, persona,
  watermark, temperature 1.0): they are all disguised, so a statistic that fires on a disguise is
  punished. The positives are the 5 pairs (fp8 endpoint, S endpoint) and the 5 pairs (LoRA 0.3
  endpoint, S endpoint).
* The interval is the 2.5 - 97.5 percentile of the AUC over replicates. It covers the sampling
  of the answers, NOT the choice of endpoints: there is one fp8 endpoint and one LoRA 0.3
  endpoint, so an effect specific to that endpoint (and not to the model) cannot be excluded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.d_features import ALL_FEATURES, FEATURES
from hackathon.model_equality.store import Answer

EVAL_SEED = 20261121
"""Seed of the cell draws (named, so that every figure of a report can be regenerated)."""
DEFAULT_REPLICATES = 300
DEFAULT_SAMPLE_SIZES = (5, 10, 20, 40)
"""Answers per endpoint asked for by the brief: N = 5, 10, 20, 40."""
INTERVAL_LOW_PERCENTILE = 2.5
INTERVAL_HIGH_PERCENTILE = 97.5
MIN_STANDARD_DEVIATION = 1e-9
BASE_MODEL = "S"
FP8_MODEL = "T1"
LORA_WEAK_MODEL = "T2"

FloatArray = NDArray[np.float64]


class EvalError(HackathonError):
    """The answers cannot support the requested evaluation."""


@dataclass(frozen=True)
class Roles:
    """Which endpoints play which role in the AUC.

    Attributes:
        base: Endpoints of the base model S, disguises included (the negatives).
        fp8: Endpoints of the fp8 model T1.
        lora_weak: Endpoints of the LoRA 0.3 model T2.
    """

    base: tuple[str, ...]
    fp8: tuple[str, ...]
    lora_weak: tuple[str, ...]


def roles_from_models(model_of: Mapping[str, str]) -> Roles:
    """Roles from `endpoint -> model label` (`S`, `T1`, `T2`, ...)."""
    def of(label: str) -> tuple[str, ...]:
        return tuple(sorted(e for e, m in model_of.items() if m == label))

    return Roles(base=of(BASE_MODEL), fp8=of(FP8_MODEL), lora_weak=of(LORA_WEAK_MODEL))


@dataclass(frozen=True)
class Design:
    """Answers laid out on the cells common to every endpoint.

    Attributes:
        endpoints: Endpoint order of the first axis of every array.
        cells: `(prompt_id, repeat)` answered by all endpoints.
        prompts: Prompt text per prompt id.
        responses: `responses[endpoint][cell_index]`.
    """

    endpoints: tuple[str, ...]
    cells: tuple[tuple[str, int], ...]
    prompts: Mapping[str, str]
    responses: Mapping[str, tuple[str, ...]]


def build_design(answers: Sequence[Answer], endpoints: Sequence[str]) -> Design:
    """Keep the cells answered by every endpoint.

    Raises:
        EvalError: No common cell.
    """
    by_endpoint: dict[str, dict[tuple[str, int], Answer]] = {e: {} for e in endpoints}
    for answer in answers:
        if answer.endpoint in by_endpoint:
            by_endpoint[answer.endpoint].setdefault((answer.prompt_id, answer.repeat), answer)
    common = set.intersection(*(set(cells) for cells in by_endpoint.values()))
    if not common:
        raise EvalError("no (prompt, repeat) cell answered by every endpoint")
    cells = tuple(sorted(common))
    prompts: dict[str, str] = {}
    for endpoint in endpoints:
        for cell in cells:
            prompts.setdefault(cell[0], by_endpoint[endpoint][cell].prompt)
    responses = {e: tuple(by_endpoint[e][c].response for c in cells) for e in endpoints}
    return Design(tuple(endpoints), cells, prompts, responses)


def restrict_design(design: Design, keep: Sequence[int]) -> Design:
    """The design reduced to the given cell indices."""
    cells = tuple(design.cells[i] for i in keep)
    responses = {e: tuple(design.responses[e][i] for i in keep) for e in design.endpoints}
    return Design(design.endpoints, cells, design.prompts, responses)


def feature_tensor(design: Design, names: Sequence[str]) -> FloatArray:
    """Feature values, shape (features, endpoints, cells)."""
    tensor = np.zeros((len(names), len(design.endpoints), len(design.cells)), dtype=np.float64)
    for k, name in enumerate(names):
        function = ALL_FEATURES[name]
        for e, endpoint in enumerate(design.endpoints):
            for c, (prompt_id, _repeat) in enumerate(design.cells):
                tensor[k, e, c] = function(design.prompts[prompt_id], design.responses[endpoint][c])
    return tensor


def center_by_cell(tensor: FloatArray) -> FloatArray:
    """Subtract, for every feature and cell, the mean over endpoints.

    Removes the prompt effect (some prompts make every endpoint answer long, others short) with
    unlabelled information only: at scoring time every endpoint answers the same prompts too.
    """
    return tensor - tensor.mean(axis=1, keepdims=True)


def standard_deviations(tensor: FloatArray) -> FloatArray:
    """Global standard deviation of each feature over endpoints and cells (floored)."""
    flat = tensor.reshape(tensor.shape[0], -1)
    return np.maximum(flat.std(axis=1), MIN_STANDARD_DEVIATION)


def draw_cells(cell_count: int, sample_size: int, replicates: int, seed: int) -> NDArray[np.int64]:
    """`replicates` draws of `sample_size` distinct cell indices, shape (replicates, sample_size).

    Raises:
        EvalError: `sample_size` larger than the number of cells.
    """
    if sample_size > cell_count:
        raise EvalError(f"sample size {sample_size} exceeds the {cell_count} common cells")
    rng = np.random.default_rng(seed + sample_size)
    return np.stack([rng.permutation(cell_count)[:sample_size] for _ in range(replicates)])


def scalar_distances(
    tensor: FloatArray, draws: NDArray[np.int64], weights: FloatArray
) -> FloatArray:
    """Pair distances of every replicate, shape (replicates, endpoints, endpoints).

    distance[r, a, b] = sqrt(sum_k w_k * ((mean_a - mean_b)_k / sd_k)^2), the means being taken on
    the replicate's cells. With a single feature of weight 1 it is the absolute standardised
    difference of means.
    """
    sd = standard_deviations(tensor)
    standardised = tensor / sd[:, None, None]
    sampled = standardised[:, :, draws]  # (K, E, R, N)
    means = sampled.mean(axis=3)  # (K, E, R)
    diff = means[:, :, None, :] - means[:, None, :, :]  # (K, E, E, R)
    squared = np.einsum("k,kabr->rab", weights, diff**2)
    return np.asarray(np.sqrt(squared), dtype=np.float64)


def roc_auc(positives: FloatArray, negatives: FloatArray) -> FloatArray:
    """AUC per replicate; inputs (replicates, pairs). Ties count one half."""
    greater = (positives[:, :, None] > negatives[:, None, :]).mean(axis=(1, 2))
    ties = (positives[:, :, None] == negatives[:, None, :]).mean(axis=(1, 2))
    return np.asarray(greater + 0.5 * ties, dtype=np.float64)


@dataclass(frozen=True)
class AucSummary:
    """AUC of one statistic at one sample size.

    Attributes:
        mean: Mean AUC over replicates.
        low: 2.5 percentile over replicates.
        high: 97.5 percentile over replicates.
    """

    mean: float
    low: float
    high: float


def summarise(values: FloatArray) -> AucSummary:
    """Mean and percentile interval of replicate AUCs."""
    return AucSummary(
        mean=float(values.mean()),
        low=float(np.percentile(values, INTERVAL_LOW_PERCENTILE)),
        high=float(np.percentile(values, INTERVAL_HIGH_PERCENTILE)),
    )


def pair_values(
    distances: FloatArray, endpoints: Sequence[str], left: Sequence[str], right: Sequence[str]
) -> FloatArray:
    """Distances of the pairs (l, r) for l in left, r in right (l != r), shape (R, pairs).

    Pairs inside one group are listed once (unordered).
    """
    index = {e: i for i, e in enumerate(endpoints)}
    seen: set[frozenset[str]] = set()
    columns: list[FloatArray] = []
    for a in left:
        for b in right:
            key = frozenset((a, b))
            if a == b or key in seen:
                continue
            seen.add(key)
            columns.append(distances[:, index[a], index[b]])
    return np.stack(columns, axis=1)


def auc_for_roles(
    distances: FloatArray, endpoints: Sequence[str], roles: Roles
) -> dict[str, FloatArray]:
    """Per-replicate AUC of `fp8`, `lora_weak` and `both` against the S-S pairs."""
    negatives = pair_values(distances, endpoints, roles.base, roles.base)
    fp8 = pair_values(distances, endpoints, roles.fp8, roles.base)
    weak = pair_values(distances, endpoints, roles.lora_weak, roles.base)
    return {
        "fp8": roc_auc(fp8, negatives),
        "lora_weak": roc_auc(weak, negatives),
        "both": roc_auc(np.concatenate([fp8, weak], axis=1), negatives),
    }


def auc_curve_for_features(
    design: Design,
    roles: Roles,
    feature_names: Sequence[str],
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    replicates: int = DEFAULT_REPLICATES,
) -> dict[int, dict[str, AucSummary]]:
    """AUC summary per sample size for the distance built on `feature_names` (equal weights)."""
    tensor = center_by_cell(feature_tensor(design, feature_names))
    weights = np.ones(len(feature_names), dtype=np.float64)
    return auc_curve_for_tensor(tensor, design.endpoints, roles, weights, sample_sizes, replicates)


def auc_curve_for_tensor(
    tensor: FloatArray,
    endpoints: Sequence[str],
    roles: Roles,
    weights: FloatArray,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    replicates: int = DEFAULT_REPLICATES,
) -> dict[int, dict[str, AucSummary]]:
    """AUC summary per sample size for a ready (features, endpoints, cells) tensor."""
    curve: dict[int, dict[str, AucSummary]] = {}
    for sample_size in sample_sizes:
        draws = draw_cells(tensor.shape[2], sample_size, replicates, EVAL_SEED)
        distances = scalar_distances(tensor, draws, weights)
        aucs = auc_for_roles(distances, endpoints, roles)
        curve[sample_size] = {name: summarise(values) for name, values in aucs.items()}
    return curve


def screen_features(
    design: Design,
    roles: Roles,
    sample_size: int,
    replicates: int = DEFAULT_REPLICATES,
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, AucSummary]]:
    """AUC of every single feature at one sample size (discovery screening)."""
    chosen = list(names) if names is not None else list(FEATURES)
    tensor = center_by_cell(feature_tensor(design, chosen))
    draws = draw_cells(len(design.cells), sample_size, replicates, EVAL_SEED)
    result: dict[str, dict[str, AucSummary]] = {}
    for k, name in enumerate(chosen):
        weights = np.zeros(len(chosen), dtype=np.float64)
        weights[k] = 1.0
        distances = scalar_distances(tensor, draws, weights)
        aucs = auc_for_roles(distances, design.endpoints, roles)
        result[name] = {key: summarise(values) for key, values in aucs.items()}
    return result
