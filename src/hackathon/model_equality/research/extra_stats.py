"""Extra pair statistics compared with the kit's MMD channels (research, team A).

All of them reuse the kit's within-prompt label permutation to turn a raw statistic into a
z-score, so they are directly comparable with the MMD channels on the same pairs:

* `exact_text`   MMD with the "identical normalised reply" kernel (the simplest MMD).
* `determinism`  |within-endpoint agreement of a - of b| on identical replies: a pure
                 "does this endpoint repeat itself as much as that one" statistic (entropy of the
                 reply distribution in disguise).
* `loglen_rank`  difference of mean within-prompt rank of the reply length (a Mann-Whitney test).
* `hamming_early`  Hamming on the first `EARLY_CHARS` characters only; `lcp`  longest common
                 prefix; `char3`  character-trigram cosine (finer than word n-grams).
* `rut_char`, `rut_spectrum`   a RUT-style rank test (arXiv 2506.06975). RUT ranks the API reply
                 among reference-model samples with the reference model's likelihood. We have no
                 likelihood, so the score of a reply is its typicality, the mean similarity to the
                 other replies to the same prompt from all endpoints (a label-free mixture
                 reference); replies are rank-normalised inside each prompt and the two endpoints
                 are compared with a Mann-Whitney statistic. It is a PROXY of RUT, not RUT.

Scalar statistics are two-sided: the z returned is `|z|`, so it is half-normal under H0 (fine for
AUC and for ordering, not for a Stouffer sum with the signed MMD z's).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from hackathon.model_equality.features import (
    build_views,
    encode_labels,
    log_length,
    normalize_text,
)
from hackathon.model_equality.pairwise import (
    DEFAULT_PERMUTATIONS,
    KERNEL_HAMMING_CHAR,
    KERNEL_SPECTRUM,
    PERMUTATION_SEED,
    Channel,
    _indicators,
    gram_matrix,
    hamming_block,
    onehot_block,
    pair_statistic,
)
from hackathon.model_equality.prompts import category_of
from hackathon.model_equality.store import Answer

KERNEL_EXACT_TEXT = "exact_text"
KERNEL_DETERMINISM = "determinism"
KERNEL_LENGTH_RANK = "loglen_rank"
KERNEL_RUT_CHAR = "rut_char"
KERNEL_RUT_SPECTRUM = "rut_spectrum"
KERNEL_HAMMING_EARLY = "hamming_early"
KERNEL_LCP = "lcp"
KERNEL_CHAR_TRIGRAM = "char3"
EXTRA_KERNELS: tuple[str, ...] = (
    KERNEL_EXACT_TEXT, KERNEL_DETERMINISM, KERNEL_LENGTH_RANK, KERNEL_RUT_CHAR,
    KERNEL_RUT_SPECTRUM, KERNEL_HAMMING_EARLY, KERNEL_LCP, KERNEL_CHAR_TRIGRAM,
)
EARLY_CHARS = 40
"""Characters compared by `hamming_early`: the start of a reply is where a small logit shift
(quantization, a weak LoRA) is most likely to flip the first sampled tokens."""
TRIGRAM_SIZE = 3
EMPTY_TEXT_TOKEN = "<empty>"
MIN_STD = 1e-12
FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]
IntMatrix = npt.NDArray[np.int64]


def _prepared_groups(
    prompt_index: IntVector, rows_a: IntVector, rows_b: IntVector
) -> tuple[IntVector, IntVector, IntVector, IntVector] | None:
    """Rows sorted by prompt, dense group ids, per-group counts of a and b (None if undefined)."""
    rows = np.concatenate([rows_a, rows_b])
    prompts = prompt_index[rows]
    order = np.argsort(prompts, kind="stable")
    rows = rows[order]
    is_a = np.concatenate([np.ones(len(rows_a)), np.zeros(len(rows_b))])[order]
    _, group = np.unique(prompts[order], return_inverse=True)
    group = group.astype(np.int64)
    n_groups = int(group.max()) + 1
    count_a = np.bincount(group, weights=is_a, minlength=n_groups).astype(np.int64)
    count_b = np.bincount(group, minlength=n_groups).astype(np.int64) - count_a
    if (count_a * count_b).sum() == 0:
        return None
    return rows, group, is_a.astype(np.int64), np.stack([count_a, count_b])


def scalar_pair_z(
    values: npt.NDArray[np.float64],
    prompt_index: IntVector,
    rows_a: IntVector,
    rows_b: IntVector,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """|z| of the difference of means of a per-row scalar, null by within-prompt permutation.

    Returns:
        NaN when the pair shares no prompt.
    """
    prepared = _prepared_groups(prompt_index, rows_a, rows_b)
    if prepared is None:
        return float("nan")
    rows, group, is_a, counts = prepared
    v = values[rows]
    indicators = _indicators(group, counts[0], n_permutations, rng)
    indicators[:, 0] = is_a
    n_a = float(indicators[:, 0].sum())
    n_b = float(len(rows)) - n_a
    stats = (indicators.T @ v) / n_a - ((1.0 - indicators).T @ v) / n_b
    observed, null = float(stats[0]), stats[1:]
    spread = float(null.std())
    return 0.0 if spread < MIN_STD else abs(observed - float(null.mean())) / spread


def determinism_pair_z(
    gram: FloatMatrix,
    prompt_index: IntVector,
    rows_a: IntVector,
    rows_b: IntVector,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """|z| of (within-a mean kernel - within-b mean kernel), null by within-prompt permutation."""
    prepared = _prepared_groups(prompt_index, rows_a, rows_b)
    if prepared is None:
        return float("nan")
    rows, group, is_a, counts = prepared
    pairs_a = float((counts[0] * (counts[0] - 1)).sum())
    pairs_b = float((counts[1] * (counts[1] - 1)).sum())
    if pairs_a == 0 or pairs_b == 0:
        return float("nan")
    sub = gram[np.ix_(rows, rows)]
    indicators = _indicators(group, counts[0], n_permutations, rng)
    indicators[:, 0] = is_a
    in_a = sub @ indicators
    in_b = sub.sum(axis=1, keepdims=True) - in_a
    stats = (indicators * in_a).sum(axis=0) / pairs_a - ((1.0 - indicators) * in_b).sum(
        axis=0) / pairs_b
    observed, null = float(stats[0]), stats[1:]
    spread = float(null.std())
    return 0.0 if spread < MIN_STD else abs(observed - float(null.mean())) / spread


def within_prompt_rank(values: npt.NDArray[np.float64], prompt_index: IntVector) -> FloatMatrix:
    """Rank of each value among the rows of its prompt, scaled to (0, 1) as rank / (m + 1)."""
    out = np.zeros(len(values), dtype=np.float64)
    for prompt in np.unique(prompt_index):
        rows = np.flatnonzero(prompt_index == prompt)
        out[rows] = rankdata(values[rows]) / (len(rows) + 1.0)
    return out


def typicality(gram: FloatMatrix, prompt_index: IntVector) -> FloatMatrix:
    """Mean similarity of each row to the other rows of its prompt (all endpoints pooled)."""
    out = np.zeros(len(prompt_index), dtype=np.float64)
    for prompt in np.unique(prompt_index):
        rows = np.flatnonzero(prompt_index == prompt)
        if len(rows) > 1:
            out[rows] = gram[np.ix_(rows, rows)].sum(axis=1) / (len(rows) - 1)
    return out


def extra_channels(
    answers: Sequence[Answer],
    endpoints: Sequence[str],
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> dict[Channel, FloatMatrix]:
    """z matrices of the extra channels, per prompt category, keyed `(category, kernel)`."""
    index_of = {endpoint: position for position, endpoint in enumerate(endpoints)}
    by_category: dict[str, list[Answer]] = defaultdict(list)
    for answer in answers:
        by_category[category_of(answer.prompt_id)].append(answer)
    channels: dict[Channel, FloatMatrix] = {}
    size = len(endpoints)
    for category, rows_all in sorted(by_category.items()):
        rows_sorted = sorted(rows_all, key=lambda a: (a.prompt_id, a.endpoint, a.repeat))
        ids = sorted({a.prompt_id for a in rows_sorted})
        position = {pid: n for n, pid in enumerate(ids)}
        prompt_index = np.array([position[a.prompt_id] for a in rows_sorted], dtype=np.int64)
        endpoint_of = np.array([index_of[a.endpoint] for a in rows_sorted], dtype=np.int64)
        rows_of: Mapping[int, IntVector] = {
            int(e): np.flatnonzero(endpoint_of == e) for e in np.unique(endpoint_of)
        }
        texts = [a.response for a in rows_sorted]
        views = build_views(category, texts)
        exact_ids = encode_labels([normalize_text(t).lower() for t in texts])
        exact_gram = np.zeros((len(texts), len(texts)))
        for prompt in np.unique(prompt_index):
            members = np.flatnonzero(prompt_index == prompt)
            exact_gram[np.ix_(members, members)] = onehot_block(exact_ids[members])
        np.fill_diagonal(exact_gram, 0.0)
        length_rank = within_prompt_rank(log_length(texts), prompt_index)
        rut_values = {
            KERNEL_RUT_CHAR: within_prompt_rank(
                typicality(gram_matrix(KERNEL_HAMMING_CHAR, views, prompt_index), prompt_index),
                prompt_index),
            KERNEL_RUT_SPECTRUM: within_prompt_rank(
                typicality(gram_matrix(KERNEL_SPECTRUM, views, prompt_index), prompt_index),
                prompt_index),
        }
        mmd_grams = {
            KERNEL_EXACT_TEXT: exact_gram,
            KERNEL_HAMMING_EARLY: blockwise_gram(
                prompt_index, partial(early_hamming_block, views.chars)),
            KERNEL_LCP: blockwise_gram(prompt_index, partial(prefix_block, views.chars)),
            KERNEL_CHAR_TRIGRAM: char_trigram_gram(texts, prompt_index),
        }
        matrices = {kernel: np.full((size, size), np.nan) for kernel in EXTRA_KERNELS}
        rng = np.random.default_rng(seed)
        ordered = sorted(rows_of)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                ra, rb = rows_of[first], rows_of[second]
                results = {
                    kernel: _mmd_z(gram, prompt_index, ra, rb, n_permutations, rng)
                    for kernel, gram in mmd_grams.items()
                }
                results |= {
                    KERNEL_DETERMINISM: determinism_pair_z(exact_gram, prompt_index, ra, rb,
                                                           n_permutations, rng),
                    KERNEL_LENGTH_RANK: scalar_pair_z(length_rank, prompt_index, ra, rb,
                                                      n_permutations, rng),
                    KERNEL_RUT_CHAR: scalar_pair_z(rut_values[KERNEL_RUT_CHAR], prompt_index, ra,
                                                   rb, n_permutations, rng),
                    KERNEL_RUT_SPECTRUM: scalar_pair_z(rut_values[KERNEL_RUT_SPECTRUM],
                                                       prompt_index, ra, rb, n_permutations, rng),
                }
                for kernel, value in results.items():
                    matrices[kernel][first, second] = matrices[kernel][second, first] = value
        for kernel, matrix in matrices.items():
            if np.isfinite(matrix).any():
                channels[(category, kernel)] = matrix
    return channels


def blockwise_gram(
    prompt_index: IntVector, block_of: Callable[[IntVector], FloatMatrix]
) -> FloatMatrix:
    """Gram matrix with `block_of(rows)` inside each prompt, 0 across prompts and on the diagonal."""
    size = len(prompt_index)
    gram = np.zeros((size, size), dtype=np.float64)
    for prompt in np.unique(prompt_index):
        rows = np.flatnonzero(prompt_index == prompt)
        gram[np.ix_(rows, rows)] = block_of(rows)
    np.fill_diagonal(gram, 0.0)
    return gram


def early_hamming_block(chars: IntMatrix, rows: IntVector) -> FloatMatrix:
    """Hamming kernel on the first `EARLY_CHARS` characters of the selected rows."""
    return hamming_block(chars[rows][:, :EARLY_CHARS])


def prefix_block(chars: IntMatrix, rows: IntVector) -> FloatMatrix:
    """`lcp_block` of the selected rows."""
    return lcp_block(chars[rows])


def lcp_block(matrix: npt.NDArray[np.int64]) -> FloatMatrix:
    """Normalised longest-common-prefix kernel (padding matching padding, as in Hamming)."""
    equal = matrix[:, None, :] == matrix[None, :, :]
    prefix = np.cumprod(equal, axis=2).sum(axis=2)
    return np.asarray(prefix / float(matrix.shape[1]), dtype=np.float64)


def char_trigram_gram(texts: Sequence[str], prompt_index: IntVector) -> FloatMatrix:
    """Cosine kernel of lower-cased character trigram counts (finer than word n-grams)."""
    cleaned = [normalize_text(text).lower() or EMPTY_TEXT_TOKEN for text in texts]
    counts = CountVectorizer(analyzer="char", ngram_range=(TRIGRAM_SIZE, TRIGRAM_SIZE),
                             lowercase=False).fit_transform(cleaned)
    normalised = sk_normalize(counts)
    gram = np.asarray((normalised @ normalised.T).toarray(), dtype=np.float64)
    same_prompt = prompt_index[:, None] == prompt_index[None, :]
    gram = np.where(same_prompt, gram, 0.0)
    np.fill_diagonal(gram, 0.0)
    return gram


def _mmd_z(
    gram: FloatMatrix, prompt_index: IntVector, rows_a: IntVector, rows_b: IntVector,
    n_permutations: int, rng: np.random.Generator,
) -> float:
    """Signed MMD z of the kit for one pair (NaN when undefined)."""
    stat = pair_statistic(gram, prompt_index, rows_a, rows_b, n_permutations, rng)
    return float("nan") if stat is None else stat.z


def pooled_channel(
    channels: Mapping[Channel, FloatMatrix], kernel: str
) -> FloatMatrix | None:
    """Stouffer sum over categories of one kernel's z (`sum z / sqrt(k)`), NaN where undefined."""
    stack = [matrix for (_, k), matrix in channels.items() if k == kernel]
    if not stack:
        return None
    data = np.stack(stack)
    defined = np.isfinite(data)
    count = defined.sum(axis=0)
    total = np.where(defined, data, 0.0).sum(axis=0)
    out = np.full(data.shape[1:], np.nan)
    positive = count > 0
    out[positive] = total[positive] / np.sqrt(count[positive])
    return out
