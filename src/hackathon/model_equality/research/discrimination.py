"""Per-prompt and per-category discriminating power measured on the `ref-*` endpoints.

Ground truth (practice_answers.csv): ref-01 plain, ref-03 system prompt, ref-05 persona, ref-06
watermark, ref-08 temperature 1.0 are all the model S; ref-02 is T1 (fp8), ref-09 T2 (LoRA 0.3),
ref-04 T3 (LoRA 0.7), ref-07 T4 (LoRA 1.5). A good prompt separates T1 / T2 from S while the four
disguises stay close to S.

Everything derives from one tensor, the *agreement* between endpoints at one prompt:

    A[p, a, b] = mean over the draws i of a and j of b of kernel(y_i, y_j)      (a != b)

and the distance `D = 1 - A`. From it:

* `pool_distance`  distance of each endpoint to the pool of the S endpoints (itself excluded):
  the "prompt x endpoint" heat map. A disguise should read like S, T1 / T2 higher.
* `pair_auc`       probability that a (T1|T2, S) pair is farther apart than an (S, S) pair: 0.5 is
  chance, 1.0 is a perfect separation, and the S-S pairs carry the disguise noise.
* `info_score`     `AUC - 0.5`, the score used to rank prompts.

Draws of one prompt are independent samples, so selecting prompts on repeat 0 and validating on
repeat 1 is a clean test of how much of a per-prompt score is signal rather than sampling noise
(`split_half_reliability`).
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.features import build_views
from hackathon.model_equality.pairwise import (
    KERNEL_HAMMING_CHAR,
    KERNEL_ONEHOT_KEY,
    KERNEL_SPECTRUM,
    kernel_block,
)
from hackathon.model_equality.prompts import category_of
from hackathon.model_equality.prompts.bank import CATEGORY_ARITH, CATEGORY_FORMAT
from hackathon.model_equality.research.candidates import (
    FAMILY_ARITH,
    FAMILY_CHOICE,
    FAMILY_NUMERIC,
    FAMILY_YESNO,
    family_of,
)
from hackathon.model_equality.store import Answer

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

REF_ENDPOINTS: tuple[str, ...] = tuple(f"ref-{index:02d}" for index in range(1, 10))
DISGUISE_KINDS: frozenset[str] = frozenset(
    {"plain", "system prompt", "persona", "watermark", "sampling at temperature 1.0"}
)
HARD_MODELS: frozenset[str] = frozenset({"T1", "T2"})
EASY_MODELS: frozenset[str] = frozenset({"T3", "T4"})
BASE_MODEL = "S"
KEY_KERNEL_FAMILIES: Mapping[str, str] = {
    FAMILY_ARITH: CATEGORY_ARITH,
    FAMILY_NUMERIC: CATEGORY_ARITH,
    FAMILY_YESNO: CATEGORY_FORMAT,
    FAMILY_CHOICE: CATEGORY_FORMAT,
}
"""Candidate families whose "answer key" is a number / the first word (see features.answer_key)."""
DEFAULT_KERNELS: tuple[str, ...] = (KERNEL_ONEHOT_KEY, KERNEL_HAMMING_CHAR, KERNEL_SPECTRUM)
AUC_TIE_WEIGHT = 0.5
CHANCE_AUC = 0.5


class TruthError(ValueError):
    """The practice answers file is malformed."""


@dataclass(frozen=True)
class EndpointTruth:
    """Ground truth of one `ref-*` endpoint."""

    endpoint: str
    model: str
    kind: str

    @property
    def is_disguise_of_base(self) -> bool:
        """True for the five endpoints that serve the base model S."""
        return self.model == BASE_MODEL


def read_truth(path: Path) -> dict[str, EndpointTruth]:
    """Read `practice_answers.csv` (columns endpoint, family, model, kind).

    Raises:
        TruthError: Missing file or columns.
    """
    if not path.is_file():
        raise TruthError(f"practice answers not found: {path}")
    truth: dict[str, EndpointTruth] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                truth[row["endpoint"]] = EndpointTruth(row["endpoint"], row["model"], row["kind"])
            except KeyError as exc:
                raise TruthError(f"practice answers: missing column {exc}") from exc
    return truth


def kernel_category(prompt_id: str) -> str:
    """Category used by `features.answer_key` for this prompt (candidates map to bank rules)."""
    family = family_of(prompt_id)
    if family is not None:
        return KEY_KERNEL_FAMILIES.get(family, family)
    return category_of(prompt_id)


def group_label(prompt_id: str) -> str:
    """Reporting group of a prompt: candidate family, or `bank:<category>` for the official bank."""
    family = family_of(prompt_id)
    return family if family is not None else f"bank:{category_of(prompt_id)}"


@dataclass(frozen=True)
class AgreementTensor:
    """Agreement of every endpoint pair at every prompt, for one kernel.

    Attributes:
        prompts: Prompt ids, row order of `values`.
        endpoints: Endpoint ids, order of the two last axes.
        values: `[P, E, E]` mean kernel between the draws of two endpoints; the diagonal holds
            the within-endpoint agreement (NaN with fewer than 2 draws); NaN where an endpoint
            has no draw at that prompt.
        draws: `[P, E]` number of draws of each endpoint at each prompt.
        kernel: Kernel name.
    """

    prompts: tuple[str, ...]
    endpoints: tuple[str, ...]
    values: FloatArray
    draws: IntArray
    kernel: str

    def distance(self) -> FloatArray:
        """`1 - agreement`."""
        return np.asarray(1.0 - self.values, dtype=np.float64)


def group_answers(
    answers: Iterable[Answer],
    endpoints: Sequence[str] = REF_ENDPOINTS,
    repeats: Collection[int] | None = None,
) -> dict[str, dict[str, list[Answer]]]:
    """`{prompt id: {endpoint: [answers sorted by repeat]}}` restricted to the wanted endpoints."""
    wanted = set(endpoints)
    out: dict[str, dict[str, list[Answer]]] = defaultdict(lambda: defaultdict(list))
    for answer in answers:
        if answer.endpoint in wanted and (repeats is None or answer.repeat in repeats):
            out[answer.prompt_id][answer.endpoint].append(answer)
    for per_endpoint in out.values():
        for rows in per_endpoint.values():
            rows.sort(key=lambda a: a.repeat)
    return {pid: dict(per) for pid, per in out.items()}


def agreement_tensor(
    grouped: Mapping[str, Mapping[str, Sequence[Answer]]],
    prompts: Sequence[str],
    endpoints: Sequence[str],
    kernel: str,
) -> AgreementTensor:
    """Build the agreement tensor of `prompts` for one kernel."""
    n_prompts, n_endpoints = len(prompts), len(endpoints)
    values = np.full((n_prompts, n_endpoints, n_endpoints), np.nan)
    draws = np.zeros((n_prompts, n_endpoints), dtype=np.int64)
    for p_index, prompt_id in enumerate(prompts):
        per_endpoint = grouped.get(prompt_id, {})
        rows: list[Answer] = []
        owner: list[int] = []
        for e_index, endpoint in enumerate(endpoints):
            for answer in per_endpoint.get(endpoint, []):
                rows.append(answer)
                owner.append(e_index)
        if not rows:
            continue
        views = build_views(kernel_category(prompt_id), [a.response for a in rows])
        gram = kernel_block(kernel, views, np.arange(len(rows), dtype=np.int64))
        owner_array = np.array(owner, dtype=np.int64)
        for e_index in range(n_endpoints):
            draws[p_index, e_index] = int((owner_array == e_index).sum())
        for a_index, b_index in itertools.product(range(n_endpoints), repeat=2):
            rows_a = np.flatnonzero(owner_array == a_index)
            rows_b = np.flatnonzero(owner_array == b_index)
            if len(rows_a) == 0 or len(rows_b) == 0:
                continue
            block = gram[np.ix_(rows_a, rows_b)]
            if a_index == b_index:
                if len(rows_a) < 2:
                    continue
                total = block.sum() - np.trace(block)
                values[p_index, a_index, b_index] = total / (len(rows_a) * (len(rows_a) - 1))
            else:
                values[p_index, a_index, b_index] = block.mean()
    return AgreementTensor(tuple(prompts), tuple(endpoints), values, draws, kernel)


def endpoint_sets(
    truth: Mapping[str, EndpointTruth], endpoints: Sequence[str]
) -> dict[str, list[int]]:
    """Index sets of the endpoints: `base`, `hard` (T1, T2), `easy` (T3, T4)."""
    sets: dict[str, list[int]] = {"base": [], "hard": [], "easy": []}
    for index, endpoint in enumerate(endpoints):
        model = truth[endpoint].model
        if model == BASE_MODEL:
            sets["base"].append(index)
        elif model in HARD_MODELS:
            sets["hard"].append(index)
        elif model in EASY_MODELS:
            sets["easy"].append(index)
    return sets


def pair_auc(
    pair_distance: FloatArray, positives: Sequence[tuple[int, int]], negatives: Sequence[tuple[int, int]]
) -> float:
    """AUC of "positive pairs are farther apart than negative pairs" (ties count one half).

    Args:
        pair_distance: `[E, E]` distance matrix (NaN entries are skipped).
        positives: Index pairs of different-model pairs.
        negatives: Index pairs of same-model pairs (the disguise noise).

    Returns:
        The AUC, or NaN when either side has no finite value.
    """
    pos = np.array([pair_distance[a, b] for a, b in positives], dtype=np.float64)
    neg = np.array([pair_distance[a, b] for a, b in negatives], dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + AUC_TIE_WEIGHT * equal) / (len(pos) * len(neg)))


def pair_sets(sets: Mapping[str, Sequence[int]]) -> dict[str, list[tuple[int, int]]]:
    """Endpoint pairs of each role: `ss` (same model S), `hard` (T1|T2 x S), `easy` (T3|T4 x S)."""
    base = list(sets["base"])
    return {
        "ss": list(itertools.combinations(base, 2)),
        "hard": [(t, s) for t in sets["hard"] for s in base],
        "easy": [(t, s) for t in sets["easy"] for s in base],
    }


def endpoint_pool_auc(
    pair_distance: FloatArray, sets: Mapping[str, Sequence[int]], role: str
) -> float:
    """Endpoint-level AUC: distance to the S pool of `role` endpoints against S endpoints.

    Each endpoint is summarised by its mean distance to the *other* S endpoints (a pooled
    reference has more draws than any single partner, hence more power than pair distances).
    """
    base = list(sets["base"])

    def delta(endpoint: int) -> float:
        others = [s for s in base if s != endpoint]
        values = pair_distance[endpoint, others]
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else float("nan")

    positives = np.array([delta(e) for e in sets[role]])
    negatives = np.array([delta(e) for e in base])
    positives, negatives = positives[np.isfinite(positives)], negatives[np.isfinite(negatives)]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    greater = (positives[:, None] > negatives[None, :]).sum()
    equal = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + AUC_TIE_WEIGHT * equal) / (len(positives) * len(negatives)))


def pooled_pair_distance(tensor: AgreementTensor, prompt_rows: Sequence[int]) -> FloatArray:
    """Distance matrix `[E, E]` averaged over the selected prompts (NaN-aware)."""
    block = tensor.distance()[list(prompt_rows)]
    with np.errstate(all="ignore"):
        return np.asarray(np.nanmean(block, axis=0), dtype=np.float64)


def set_auc(tensor: AgreementTensor, prompt_rows: Sequence[int],
            truth: Mapping[str, EndpointTruth], role: str = "hard") -> float:
    """AUC of a prompt set: `role` pairs against S-S pairs on the set-averaged distances."""
    sets = endpoint_sets(truth, tensor.endpoints)
    pairs = pair_sets(sets)
    return pair_auc(pooled_pair_distance(tensor, prompt_rows), pairs[role], pairs["ss"])


def pool_distance(tensor: AgreementTensor, truth: Mapping[str, EndpointTruth]) -> FloatArray:
    """`[P, E]` mean distance of each endpoint to the S pool (the endpoint itself excluded)."""
    sets = endpoint_sets(truth, tensor.endpoints)
    distance = tensor.distance()
    out = np.full((len(tensor.prompts), len(tensor.endpoints)), np.nan)
    for e_index in range(len(tensor.endpoints)):
        others = [s for s in sets["base"] if s != e_index]
        with np.errstate(all="ignore"):
            out[:, e_index] = np.nanmean(distance[:, e_index, others], axis=1)
    return out


def prompt_scores(tensor: AgreementTensor, truth: Mapping[str, EndpointTruth]) -> dict[str, FloatArray]:
    """Per-prompt `info_score` (AUC - 0.5) for the hard and easy roles, and the mean pool excess.

    Returns:
        `{"hard", "easy", "excess_hard", "excess_easy", "disguise_leak"}`, each `[P]`:
        `excess_easy` is the same as `excess_hard` for T3 / T4; `excess_hard` is the mean pool distance of T1/T2 minus that of the S endpoints;
        `disguise_leak` is the largest pool distance among the S endpoints minus their mean
        (a disguise that sticks out shows up here).
    """
    sets = endpoint_sets(truth, tensor.endpoints)
    pairs = pair_sets(sets)
    distance = tensor.distance()
    n_prompts = len(tensor.prompts)
    keys = ("hard", "easy", "excess_hard", "excess_easy", "disguise_leak")
    out = {key: np.full(n_prompts, np.nan) for key in keys}
    pooled = pool_distance(tensor, truth)
    for p_index in range(n_prompts):
        for role in ("hard", "easy"):
            out[role][p_index] = pair_auc(distance[p_index], pairs[role], pairs["ss"]) - CHANCE_AUC
        base_values = pooled[p_index, sets["base"]]
        hard_values = pooled[p_index, sets["hard"]]
        if np.isfinite(base_values).any() and np.isfinite(hard_values).any():
            out["excess_hard"][p_index] = float(np.nanmean(hard_values) - np.nanmean(base_values))
            easy_values = pooled[p_index, sets["easy"]]
            if np.isfinite(easy_values).any():
                out["excess_easy"][p_index] = float(np.nanmean(easy_values) - np.nanmean(base_values))
            out["disguise_leak"][p_index] = float(np.nanmax(base_values) - np.nanmean(base_values))
    return out


def split_half_reliability(
    grouped_all: Mapping[str, Mapping[str, Sequence[Answer]]],
    prompts: Sequence[str],
    endpoints: Sequence[str],
    truth: Mapping[str, EndpointTruth],
    kernel: str,
) -> tuple[float, int]:
    """Spearman correlation of the per-prompt `excess_hard` between repeat 0 and repeat 1.

    Draws are independent, so a high correlation means the per-prompt score is a property of the
    prompt, and a low one means it is sampling noise. Only prompts with both repeats are used.

    Returns:
        `(correlation, number of prompts used)`; NaN when fewer than 4 prompts qualify.
    """
    from scipy import stats

    both = [
        pid for pid in prompts
        if all(
            any(a.repeat == r for per in grouped_all.get(pid, {}).values() for a in per)
            for r in (0, 1)
        )
    ]
    if len(both) < 4:
        return float("nan"), len(both)
    scores: list[FloatArray] = []
    for repeat in (0, 1):
        grouped = group_answers(
            (a for pid in both for per in grouped_all[pid].values() for a in per),
            endpoints, repeats=[repeat],
        )
        tensor = agreement_tensor(grouped, both, endpoints, kernel)
        scores.append(prompt_scores(tensor, truth)["excess_hard"])
    mask = np.isfinite(scores[0]) & np.isfinite(scores[1])
    if mask.sum() < 4:
        return float("nan"), int(mask.sum())
    rho = stats.spearmanr(scores[0][mask], scores[1][mask]).statistic
    return float(rho), int(mask.sum())
