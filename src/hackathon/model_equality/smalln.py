"""Small-N analysis: one answer per (endpoint, prompt), few prompts, no repeated draws.

The kit's unbiased MMD (`pairwise.py`) needs at least two answers of the SAME endpoint to the
SAME prompt (its within-endpoint term). With a budget of a handful of answers per endpoint
(phase E1: 5 prompts x 1 draw) that term is undefined, so this module uses the cross-endpoint
term only: for each pair of endpoints and each prompt, a dissimilarity between the two answers,
averaged over the prompts. There is no permutation null: instead the thresholds are measured on
the `ref-*` endpoints, which can be sampled at exactly the same shape (N prompts, one draw).

Components (all computed on the raw text, comparable between ref and ep when the prompts are the
same):

* `ham`    character Hamming over the longer answer (first `CHAR_LENGTH` characters); two short
           answers that differ ("Yes" / "No.") score high, and a one-word answer against a
           rambling one scores near 1, so it folds length and content together.
* `spec`   1 - cosine of the word uni+bigram bags (invariant to a shifted start).
* `first`  first words differ (0 or 1).
* `exact`  normalised answers (case, punctuation, spacing) differ (0 or 1).
* `len`    absolute difference of log(1 + words), divided by `LEN_SCALE` so one unit means
           "about a factor 3 in length".

The pair distance is a weighted mean of the component dissimilarities over the prompts; the
weights are fixed a priori (never fitted on the 9 `ref-*`: team A showed that fitting overfits).
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hackathon.model_equality.features import (
    CHAR_LENGTH,
    first_word,
    normalize_text,
    tokenize_words,
)
from hackathon.model_equality.store import Answer

FloatArray = npt.NDArray[np.float64]
UnitKey = tuple[str, int]
"""`(prompt id, repeat)`: one (prompt, draw) unit; every endpoint answers each unit once."""

COMPONENT_HAM = "ham"
COMPONENT_SPEC = "spec"
COMPONENT_FIRST = "first"
COMPONENT_EXACT = "exact"
COMPONENT_LEN = "len"
COMPONENTS: tuple[str, ...] = (
    COMPONENT_HAM, COMPONENT_SPEC, COMPONENT_FIRST, COMPONENT_EXACT, COMPONENT_LEN,
)
LEN_SCALE = math.log(3.0)
"""Log-length gap that counts as 1.0 for the `len` component (a factor 3)."""
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
LINKAGE_METHOD = "average"
"""Average linkage: a cluster is as far as its members are on average, so one noisy pair of a
tight group cannot split it (the prudent side of the cut)."""
DEFAULT_WEIGHTS: dict[str, float] = {
    COMPONENT_HAM: 1.0, COMPONENT_SPEC: 1.0, COMPONENT_FIRST: 0.0, COMPONENT_EXACT: 0.0,
    COMPONENT_LEN: 1.0,
}
"""A priori weights: the three components that carry a continuous signal (team A: Hamming
character, spectrum and length were the useful channels; one-hot keys were inverted)."""


class SmallNError(ValueError):
    """The answers do not form a complete (endpoints x units) grid."""


@dataclass(frozen=True)
class SmallNData:
    """Per-unit pairwise dissimilarities of a complete grid of answers.

    Attributes:
        endpoints: Endpoint ids, in matrix order.
        units: `(prompt id, repeat)` units, in array order.
        components: For each component, an array `[U, E, E]` of per-unit dissimilarities.
        log_length: `log(1 + words)` of every answer, `[U, E]` (signed length, for endpoint-level
            statistics that the pairwise `len` component cannot give).
        ends_cleanly: 1.0 when the answer ends with a sentence end or a closing quote, `[U, E]`.
        punctuation: Share of non-space characters that are punctuation, `[U, E]`.
    """

    endpoints: tuple[str, ...]
    units: tuple[UnitKey, ...]
    components: dict[str, FloatArray]
    log_length: FloatArray
    ends_cleanly: FloatArray
    punctuation: FloatArray

    def unit_positions(self, wanted: Collection[UnitKey]) -> list[int]:
        """Array positions of the wanted units (those absent from the grid are skipped)."""
        wanted_set = set(wanted)
        return [position for position, unit in enumerate(self.units) if unit in wanted_set]


def _comparable_text(text: str) -> str:
    """Lower-case text without punctuation, spacing collapsed."""
    return normalize_text(PUNCTUATION_PATTERN.sub("", text)).lower()


def _hamming_dissimilarity(texts: Sequence[str]) -> FloatArray:
    """Pairwise fraction of mismatching characters over the longer answer."""
    clipped = [normalize_text(text)[:CHAR_LENGTH] for text in texts]
    size = len(clipped)
    lengths = np.array([len(text) for text in clipped], dtype=np.float64)
    codes = np.full((size, CHAR_LENGTH), -1, dtype=np.int64)
    for row, text in enumerate(clipped):
        codes[row, : len(text)] = [ord(char) for char in text]
    equal = (codes[:, None, :] == codes[None, :, :]) & (codes[:, None, :] >= 0)
    longer = np.maximum(lengths[:, None], lengths[None, :])
    matches = equal.sum(axis=2).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        dissimilarity = np.where(longer > 0, 1.0 - matches / np.maximum(longer, 1.0), 0.0)
    return np.asarray(dissimilarity, dtype=np.float64)


def _spectrum_dissimilarity(texts: Sequence[str]) -> FloatArray:
    """1 - cosine of the word uni+bigram count vectors."""
    bags: list[dict[str, float]] = []
    for text in texts:
        tokens = tokenize_words(text) or ["<empty>"]
        grams = tokens + [a + " " + b for a, b in itertools.pairwise(tokens)]
        bag: dict[str, float] = {}
        for gram in grams:
            bag[gram] = bag.get(gram, 0.0) + 1.0
        bags.append(bag)
    norms = [math.sqrt(sum(v * v for v in bag.values())) for bag in bags]
    size = len(bags)
    out = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            dot = sum(v * bags[j].get(g, 0.0) for g, v in bags[i].items())
            # clip: rounding can push identical bags to -1e-16, which linkage rejects
            out[i, j] = out[j, i] = min(1.0, max(0.0, 1.0 - dot / (norms[i] * norms[j])))
    return out


SENTENCE_ENDS = (".", "!", "?", '"', ")")


def _ends_cleanly(text: str) -> float:
    """1.0 when the trimmed answer ends with a sentence end or a closing quote or bracket."""
    return float(text.strip().endswith(SENTENCE_ENDS))


def _punctuation_share(text: str) -> float:
    """Punctuation characters over non-space characters (0 for an empty answer)."""
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return sum(not char.isalnum() for char in compact) / len(compact)


def _unit_components(texts: Sequence[str]) -> dict[str, FloatArray]:
    """All component dissimilarity matrices `[E, E]` of one unit."""
    size = len(texts)
    firsts = [first_word(text) for text in texts]
    exact = [_comparable_text(text) for text in texts]
    loglen = np.array([math.log1p(len(tokenize_words(text))) for text in texts])
    return {
        COMPONENT_HAM: _hamming_dissimilarity(texts),
        COMPONENT_SPEC: _spectrum_dissimilarity(texts),
        COMPONENT_FIRST: np.array(
            [[float(firsts[i] != firsts[j]) for j in range(size)] for i in range(size)]),
        COMPONENT_EXACT: np.array(
            [[float(exact[i] != exact[j]) for j in range(size)] for i in range(size)]),
        COMPONENT_LEN: np.abs(loglen[:, None] - loglen[None, :]) / LEN_SCALE,
    }


def build_small_n_data(
    answers: Sequence[Answer],
    endpoints: Sequence[str],
    units: Collection[UnitKey] | None = None,
) -> SmallNData:
    """Pairwise dissimilarities of every unit that ALL the endpoints answered.

    Args:
        answers: Stored answers (any endpoints, any prompts).
        endpoints: Endpoints to compare, in matrix order.
        units: Optional restriction to these `(prompt id, repeat)` units.

    Raises:
        SmallNError: No unit is answered by every endpoint.
    """
    table: dict[UnitKey, dict[str, str]] = {}
    wanted = set(endpoints)
    for answer in answers:
        unit = (answer.prompt_id, answer.repeat)
        if answer.endpoint in wanted and (units is None or unit in units):
            table.setdefault(unit, {})[answer.endpoint] = answer.response
    complete = sorted(unit for unit, by_endpoint in table.items() if len(by_endpoint) == len(wanted))
    if not complete:
        raise SmallNError("no (prompt, repeat) unit is answered by every endpoint")
    per_component: dict[str, list[FloatArray]] = {name: [] for name in COMPONENTS}
    lengths: list[FloatArray] = []
    ends: list[FloatArray] = []
    punctuation: list[FloatArray] = []
    for unit in complete:
        texts = [table[unit][endpoint] for endpoint in endpoints]
        matrices = _unit_components(texts)
        for name in COMPONENTS:
            per_component[name].append(matrices[name])
        lengths.append(np.array([math.log1p(len(tokenize_words(text))) for text in texts]))
        ends.append(np.array([_ends_cleanly(text) for text in texts]))
        punctuation.append(np.array([_punctuation_share(text) for text in texts]))
    return SmallNData(
        endpoints=tuple(endpoints), units=tuple(complete),
        components={name: np.stack(stack) for name, stack in per_component.items()},
        log_length=np.stack(lengths),
        ends_cleanly=np.stack(ends),
        punctuation=np.stack(punctuation),
    )


def pair_distance(
    data: SmallNData,
    positions: Sequence[int],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> FloatArray:
    """Weighted mean (over components) of the per-prompt mean dissimilarity, `[E, E]`.

    Args:
        data: Per-unit dissimilarities.
        positions: Units to use (array positions in `data.units`).
        weights: Component weights; the result is divided by their sum so distances stay in
            comparable units whatever the weights.

    Raises:
        SmallNError: Empty selection or all weights zero.
    """
    total_weight = sum(weights.get(name, 0.0) for name in COMPONENTS)
    if not positions or total_weight <= 0:
        raise SmallNError("pair_distance needs at least one unit and one positive weight")
    index = list(positions)
    mixed = sum(
        weights.get(name, 0.0) * data.components[name][index].mean(axis=0) for name in COMPONENTS
    )
    return np.asarray(mixed / total_weight, dtype=np.float64)


def cluster_at(distance: FloatArray, tau: float) -> npt.NDArray[np.int64]:
    """Average-linkage clusters of a distance matrix cut at `tau` (labels 0..k-1)."""
    size = distance.shape[0]
    if size < 2:
        return np.zeros(size, dtype=np.int64)
    tree = linkage(squareform(distance, checks=False), method=LINKAGE_METHOD)
    return np.asarray(fcluster(tree, t=tau, criterion="distance") - 1, dtype=np.int64)


def build_tree(distance: FloatArray) -> FloatArray:
    """Average-linkage tree of a (clipped, non-negative) distance matrix."""
    clipped = np.clip(distance, 0.0, None)
    np.fill_diagonal(clipped, 0.0)
    return np.asarray(linkage(squareform(clipped, checks=False), method=LINKAGE_METHOD),
                      dtype=np.float64)


def top_merge_height(distance: FloatArray) -> float:
    """Height of the last merge of the tree: how far apart the two broadest groups are."""
    return float(build_tree(distance)[-1, 2])


def label_two_levels(
    distance: FloatArray, tau_model: float, tau_family: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Nested `(family ids, model ids)` from ONE tree cut at two heights.

    A single dendrogram cut at `tau_model <= tau_family` gives nested clusters by construction
    (every model cluster lies inside one family cluster), which is what the official
    "same model implies same family" rule needs.

    Raises:
        SmallNError: `tau_model` above `tau_family`.
    """
    if tau_model > tau_family:
        raise SmallNError(f"tau_model {tau_model} must not exceed tau_family {tau_family}")
    tree = build_tree(distance)
    families = np.asarray(fcluster(tree, t=tau_family, criterion="distance") - 1, dtype=np.int64)
    models = np.asarray(fcluster(tree, t=tau_model, criterion="distance") - 1, dtype=np.int64)
    return families, models


def restrict(data: SmallNData, members: Sequence[int]) -> SmallNData:
    """The same units, keeping only the endpoints at the given positions (e.g. one family)."""
    index = list(members)
    return SmallNData(
        endpoints=tuple(data.endpoints[i] for i in index),
        units=data.units,
        components={name: array[:, index][:, :, index] for name, array in data.components.items()},
        log_length=data.log_length[:, index],
        ends_cleanly=data.ends_cleanly[:, index],
        punctuation=data.punctuation[:, index],
    )
