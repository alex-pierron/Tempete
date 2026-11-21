"""Pairwise two-sample statistics between endpoints (MET-style unbiased MMD, permutation null).

For one *channel* (a prompt category and a kernel) and one pair of endpoints (a, b):

    MMD = mean_{i != j in a} k(y_i, y_j) + mean_{i != j in b} k(y_i, y_j) - 2 mean_{i in a, j in b} k
          (means over same-prompt pairs only: k = 0 across different prompts, as in the official
          `model_equality_testing` code, whose Hamming statistic this reproduces)

The null distribution is built by permuting the endpoint labels *inside each prompt* (the exact
conditional null given the prompt design: the number of a-labelled answers per prompt is
preserved, so the pair counts in the denominators are constants). All permutations are evaluated
with a single matrix product, which makes the 528 pairs x ~40 channels of a full analysis a matter
of seconds. The result per (pair, channel) is a z-score `(MMD - mean_null) / sd_null`: under equal
distributions it is approximately standard normal whatever the model, which is what makes
thresholds comparable between the reference endpoints and the scored ones.

A channel is undefined (NaN) for a pair when a within-endpoint term has no same-prompt pairs
(fewer than 2 answers per prompt on one side): that is reported, never silently replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.features import Views

KERNEL_HAMMING_CHAR = "hamming_char"
KERNEL_HAMMING_WORD = "hamming_word"
KERNEL_SPECTRUM = "spectrum"
KERNEL_ONEHOT_KEY = "onehot_key"
KERNEL_ONEHOT_FIRST = "onehot_first"
KERNEL_LENGTH = "length"
KERNELS: tuple[str, ...] = (
    KERNEL_HAMMING_CHAR, KERNEL_HAMMING_WORD, KERNEL_SPECTRUM, KERNEL_ONEHOT_KEY, KERNEL_ONEHOT_FIRST, KERNEL_LENGTH,
)
LENGTH_KERNEL_BANDWIDTH = 0.5
"""Bandwidth of the Gaussian kernel on log-length (a factor e of length, about 1.6x, per sigma)."""
DEFAULT_PERMUTATIONS = 200
"""Permutations per (pair, channel): p-value resolution 1/201 and a stable z-score."""
PERMUTATION_SEED = 20261121
"""Constant seed: the same answers always give the same z-scores."""
MIN_STD = 1e-12
"""Below this null spread the statistic is considered constant (z = 0)."""
P_VALUE_TOLERANCE = 1e-12

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]
Channel = tuple[str, str]
"""`(category, kernel)`."""


@dataclass(frozen=True)
class PairStat:
    """Statistic of one pair in one channel."""

    mmd: float
    z: float
    p_value: float


def hamming_block(matrix: npt.NDArray[np.int64]) -> FloatMatrix:
    """Normalised Hamming kernel (fraction of equal positions, padding matching padding)."""
    length = matrix.shape[1]
    equal = matrix[:, None, :] == matrix[None, :, :]
    return np.asarray(equal.sum(axis=2) / float(length), dtype=np.float64)


def onehot_block(ids: npt.NDArray[np.int64]) -> FloatMatrix:
    """1 where two responses have the same id, else 0."""
    return np.asarray(ids[:, None] == ids[None, :], dtype=np.float64)


def gaussian_block(values: npt.NDArray[np.float64], bandwidth: float) -> FloatMatrix:
    """Gaussian kernel on a scalar feature."""
    diff = values[:, None] - values[None, :]
    return np.asarray(np.exp(-(diff**2) / (2.0 * bandwidth**2)), dtype=np.float64)


def kernel_block(kernel: str, views: Views, rows: IntVector) -> FloatMatrix:
    """Gram block of `kernel` for the given rows of `views`."""
    if kernel == KERNEL_HAMMING_CHAR:
        return hamming_block(views.chars[rows])
    if kernel == KERNEL_HAMMING_WORD:
        return hamming_block(views.words[rows])
    if kernel == KERNEL_SPECTRUM:
        block = views.ngrams[rows]
        return np.asarray((block @ block.T).toarray(), dtype=np.float64)
    if kernel == KERNEL_ONEHOT_KEY:
        return onehot_block(views.key[rows])
    if kernel == KERNEL_ONEHOT_FIRST:
        return onehot_block(views.first[rows])
    if kernel == KERNEL_LENGTH:
        return gaussian_block(views.loglen[rows], LENGTH_KERNEL_BANDWIDTH)
    raise ValueError(f"unknown kernel {kernel!r}; valid: {', '.join(KERNELS)}")


def gram_matrix(kernel: str, views: Views, prompt_index: IntVector) -> FloatMatrix:
    """Full Gram matrix of one category: kernel inside a prompt, 0 across prompts, 0 diagonal.

    Args:
        kernel: One of `KERNELS`.
        views: Row-aligned views of the category's responses.
        prompt_index: Prompt id (dense integer) of each row.
    """
    size = len(prompt_index)
    gram = np.zeros((size, size), dtype=np.float64)
    for prompt in np.unique(prompt_index):
        rows = np.flatnonzero(prompt_index == prompt)
        gram[np.ix_(rows, rows)] = kernel_block(kernel, views, rows)
    np.fill_diagonal(gram, 0.0)
    return gram


def _indicators(
    group: IntVector, count_a: IntVector, n_permutations: int, rng: np.random.Generator
) -> FloatMatrix:
    """Label matrix `[rows, 1 + B]`: column 0 observed, others within-group permutations.

    Rows must be sorted by `group` (contiguous groups). `count_a[g]` rows of group g are
    labelled a (value 1.0) in every column.
    """
    rows = len(group)
    starts = np.searchsorted(group, np.arange(group.max() + 1))
    rank_in_group = np.arange(rows) - starts[group]
    is_a_by_position = (rank_in_group < count_a[group]).astype(np.float64)
    keys = rng.random((rows, n_permutations)) + group[:, None]
    order = np.argsort(keys, axis=0)
    indicators = np.empty((rows, 1 + n_permutations), dtype=np.float64)
    columns = np.arange(n_permutations)
    permuted = np.empty((rows, n_permutations), dtype=np.float64)
    permuted[order, columns[None, :]] = is_a_by_position[:, None]
    indicators[:, 1:] = permuted
    return indicators


def pair_statistic(
    gram: FloatMatrix,
    prompt_index: IntVector,
    rows_a: IntVector,
    rows_b: IntVector,
    n_permutations: int,
    rng: np.random.Generator,
) -> PairStat | None:
    """MMD, permutation z-score and p-value of endpoint a versus endpoint b in one channel.

    Returns:
        None when a within-endpoint term has no same-prompt pair (statistic undefined).
    """
    rows = np.concatenate([rows_a, rows_b])
    prompts = prompt_index[rows]
    order = np.argsort(prompts, kind="stable")
    rows = rows[order]
    is_a_observed = np.concatenate([np.ones(len(rows_a)), np.zeros(len(rows_b))])[order]
    prompts = prompts[order]
    _, group = np.unique(prompts, return_inverse=True)
    group = group.astype(np.int64)
    n_groups = int(group.max()) + 1
    count_a = np.bincount(group, weights=is_a_observed, minlength=n_groups).astype(np.int64)
    count_b = np.bincount(group, minlength=n_groups).astype(np.int64) - count_a
    pairs_a = float((count_a * (count_a - 1)).sum())
    pairs_b = float((count_b * (count_b - 1)).sum())
    pairs_ab = float((count_a * count_b).sum())
    if pairs_a == 0 or pairs_b == 0 or pairs_ab == 0:
        return None
    sub = gram[np.ix_(rows, rows)]
    indicators = _indicators(group, count_a, n_permutations, rng)
    indicators[:, 0] = is_a_observed
    in_a = sub @ indicators
    in_b = sub.sum(axis=1, keepdims=True) - in_a
    other = 1.0 - indicators
    sum_aa = (indicators * in_a).sum(axis=0)
    sum_bb = (other * in_b).sum(axis=0)
    sum_ab = (indicators * in_b).sum(axis=0)
    statistics = sum_aa / pairs_a + sum_bb / pairs_b - 2.0 * sum_ab / pairs_ab
    observed, null = float(statistics[0]), statistics[1:]
    spread = float(null.std())
    z_score = 0.0 if spread < MIN_STD else (observed - float(null.mean())) / spread
    p_value = (1.0 + float((null >= observed - P_VALUE_TOLERANCE).sum())) / (n_permutations + 1.0)
    return PairStat(mmd=observed, z=z_score, p_value=p_value)


def brute_force_mmd(
    gram: FloatMatrix, prompt_index: IntVector, rows_a: IntVector, rows_b: IntVector
) -> float:
    """Reference O(n^2) implementation of the statistic (used by tests to pin the fast one)."""
    def block_mean(first: IntVector, second: IntVector, same: bool) -> float:
        total, count = 0.0, 0
        for i in first:
            for j in second:
                if same and i == j:
                    continue
                if prompt_index[i] == prompt_index[j]:
                    total += gram[i, j]
                    count += 1
        return total / count if count else float("nan")

    return (block_mean(rows_a, rows_a, True) + block_mean(rows_b, rows_b, True)
            - 2.0 * block_mean(rows_a, rows_b, False))


def all_pair_stats(
    gram: FloatMatrix,
    prompt_index: IntVector,
    rows_by_endpoint: Mapping[int, IntVector],
    n_endpoints: int,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[FloatMatrix, FloatMatrix]:
    """z-score and MMD matrices `[E, E]` of one channel (NaN where undefined, symmetric)."""
    rng = np.random.default_rng(seed)
    z_matrix = np.full((n_endpoints, n_endpoints), np.nan)
    mmd_matrix = np.full((n_endpoints, n_endpoints), np.nan)
    indices: Sequence[int] = sorted(rows_by_endpoint)
    for position, first in enumerate(indices):
        for second in indices[position + 1:]:
            stat = pair_statistic(gram, prompt_index, rows_by_endpoint[first],
                                  rows_by_endpoint[second], n_permutations, rng)
            if stat is None:
                continue
            z_matrix[first, second] = z_matrix[second, first] = stat.z
            mmd_matrix[first, second] = mmd_matrix[second, first] = stat.mmd
    return z_matrix, mmd_matrix

