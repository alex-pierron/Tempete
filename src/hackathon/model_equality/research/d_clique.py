"""Clustering of the scored endpoints by exact answer agreement (team D, read-only on the ep store).

The first ep phase (e1) holds a few answers per endpoint. Because the scored models are close to
deterministic, two endpoints that serve the same model without a disguise return the SAME string.
Strict string equality is transitive, so the endpoints that agree on every shared prompt form
cliques; looser cuts of an average-linkage dendrogram merge cliques that differ on one prompt.

No request is sent: the module only reads an answers file already stored by the team.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from hackathon.common.errors import HackathonError
from hackathon.model_equality.research.d_exact import Normaliser, normalised_key
from hackathon.model_equality.store import Answer

FloatArray = NDArray[np.float64]
LINKAGE_METHOD = "average"
STRICT_CUT = 0.0


class CliqueError(HackathonError):
    """The answers cannot support a clustering (no shared prompt, fewer than two endpoints)."""


@dataclass(frozen=True)
class AgreementResult:
    """Exact agreement of the endpoints of one answers file.

    Attributes:
        endpoints: Endpoint order of the matrices.
        prompts: Prompts answered by every endpoint.
        mismatch: Share of shared prompts on which two endpoints differ, shape (E, E).
        leaf_order: Dendrogram leaf order (for drawing).
        distinct_per_prompt: Number of distinct normalised answers of each prompt.
        modal_share_per_prompt: Share of endpoints giving the modal answer, per prompt.
    """

    endpoints: tuple[str, ...]
    prompts: tuple[str, ...]
    mismatch: FloatArray
    leaf_order: tuple[int, ...]
    distinct_per_prompt: tuple[int, ...]
    modal_share_per_prompt: tuple[float, ...]


def agreement(
    answers: Sequence[Answer], normaliser: Normaliser = normalised_key, repeat: int = 0
) -> AgreementResult:
    """Mismatch matrix over the prompts answered by every endpoint (repeat `repeat` only).

    Raises:
        CliqueError: Fewer than two endpoints or no prompt answered by all of them.
    """
    keyed: dict[str, dict[str, str]] = defaultdict(dict)
    for answer in answers:
        if answer.repeat == repeat:
            keyed[answer.endpoint][answer.prompt_id] = normaliser(answer.response)
    endpoints = tuple(sorted(keyed))
    if len(endpoints) < 2:
        raise CliqueError("need at least two endpoints")
    shared = sorted(set.intersection(*(set(keyed[e]) for e in endpoints)))
    if not shared:
        raise CliqueError("no prompt answered by every endpoint")
    keys = [[keyed[e][p] for p in shared] for e in endpoints]
    count = len(endpoints)
    mismatch = np.zeros((count, count), dtype=np.float64)
    for a in range(count):
        for b in range(count):
            mismatch[a, b] = float(np.mean([keys[a][c] != keys[b][c] for c in range(len(shared))]))
    order = tuple(int(i) for i in leaves_list(linkage(squareform(mismatch, checks=False), LINKAGE_METHOD)))
    distinct = tuple(len({keys[e][c] for e in range(count)}) for c in range(len(shared)))
    modal = tuple(
        Counter(keys[e][c] for e in range(count)).most_common(1)[0][1] / count for c in range(len(shared)))
    return AgreementResult(endpoints, tuple(shared), mismatch, order, distinct, modal)


def strict_cliques(result: AgreementResult) -> list[list[str]]:
    """Groups of endpoints with zero mismatch on every shared prompt, largest first."""
    return _cut(result, STRICT_CUT)


def cut_clusters(result: AgreementResult, max_mismatch: float) -> list[list[str]]:
    """Average-linkage clusters at a mismatch distance of at most `max_mismatch`, largest first."""
    return _cut(result, max_mismatch)


def _cut(result: AgreementResult, distance: float) -> list[list[str]]:
    """Clusters of the dendrogram cut at `distance` (0 cuts into strict cliques)."""
    tree = linkage(squareform(result.mismatch, checks=False), LINKAGE_METHOD)
    labels = fcluster(tree, t=distance + 1e-9, criterion="distance")
    groups: dict[int, list[str]] = defaultdict(list)
    for endpoint, label in zip(result.endpoints, labels, strict=True):
        groups[int(label)].append(endpoint)
    return sorted(groups.values(), key=lambda g: (-len(g), g))


IDENTITY_PROMPT_PREFIX = "fam-identity"
FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen"), ("alibaba", "Qwen"), ("llama", "Llama"), ("meta", "Llama"),
)
"""Substrings of an identity answer that hint at the base model, checked in order."""
UNKNOWN_FAMILY = "?"
FAMILY_CUT = 0.4
"""Mismatch distance at which the average-linkage tree is cut to read the families."""
MAX_DISTINCT_FOR_BINARY = 3
"""A prompt with at most this many distinct normalised answers is read as a binary fingerprint."""


def family_hints(answers: Sequence[Answer], repeat: int = 0) -> dict[str, str]:
    """Family hint of each endpoint from its answer to the identity prompt (`?` when unclear).

    This reads what the model SAYS about itself; a persona or a provider system prompt can lie,
    so the hint only labels the clusters, it never builds them.
    """
    hints: dict[str, str] = {}
    for answer in answers:
        if answer.repeat != repeat or not answer.prompt_id.startswith(IDENTITY_PROMPT_PREFIX):
            continue
        text = answer.response.lower()
        hints.setdefault(answer.endpoint, UNKNOWN_FAMILY)
        for needle, family in FAMILY_HINTS:
            if needle in text:
                hints[answer.endpoint] = family
                break
    return hints


def lcp_distance_matrix(
    answers: Sequence[Answer], prompt_id: str, endpoints: Sequence[str], repeat: int = 0
) -> FloatArray:
    """1 - (common prefix / longer reply), in characters, for one prompt; 0 = identical.

    With a (near-)deterministic decoder the position where two replies first part is a property
    of the two models: a late divergence means almost identical weights, an early one a larger
    change (or a disguise that alters the context).

    Raises:
        CliqueError: An endpoint has no answer for the prompt.
    """
    replies: dict[str, str] = {}
    for answer in answers:
        if answer.prompt_id == prompt_id and answer.repeat == repeat:
            replies[answer.endpoint] = answer.response.strip()
    missing = [e for e in endpoints if e not in replies]
    if missing:
        raise CliqueError(f"no answer to {prompt_id} for {missing}")
    count = len(endpoints)
    matrix = np.zeros((count, count), dtype=np.float64)
    for a in range(count):
        for b in range(count):
            left, right = replies[endpoints[a]], replies[endpoints[b]]
            longest = max(len(left), len(right))
            common = 0
            for x, y in zip(left, right, strict=False):
                if x != y:
                    break
                common += 1
            matrix[a, b] = 0.0 if longest == 0 else 1.0 - common / longest
    return matrix


def binary_patterns(
    answers: Sequence[Answer], prompt_ids: Sequence[str], repeat: int = 0
) -> dict[str, str]:
    """Normalised answers to some prompts joined into one string per endpoint (a fingerprint)."""
    parts: dict[str, dict[str, str]] = defaultdict(dict)
    for answer in answers:
        if answer.repeat == repeat and answer.prompt_id in prompt_ids:
            parts[answer.endpoint][answer.prompt_id] = normalised_key(answer.response)
    return {e: "/".join(p.get(pid, "-")[:6] for pid in prompt_ids) for e, p in parts.items()}


def structure_metrics(answers: Sequence[Answer]) -> dict[str, object]:
    """JSON-ready summary of the exact-agreement structure of one ep answers file."""
    result = agreement(answers)
    ordered = [result.endpoints[i] for i in result.leaf_order]
    completion = next((p for p in result.prompts if p.startswith("completion")), None)
    lcp = lcp_distance_matrix(answers, completion, ordered).tolist() if completion is not None else []
    index = [result.endpoints.index(e) for e in ordered]
    mismatch = result.mismatch[np.ix_(index, index)]
    binary = [
        p for p, d in zip(result.prompts, result.distinct_per_prompt, strict=True)
        if d <= MAX_DISTINCT_FOR_BINARY]
    return {
        "endpoints": ordered,
        "prompts": list(result.prompts),
        "distinct_per_prompt": list(result.distinct_per_prompt),
        "modal_share_per_prompt": list(result.modal_share_per_prompt),
        "mismatch": mismatch.tolist(),
        "lcp_distance": lcp,
        "lcp_prompt": completion,
        "strict_cliques": strict_cliques(result),
        "clusters_at_cut": cut_clusters(result, FAMILY_CUT),
        "clusters_at_0_2": cut_clusters(result, 0.2),
        "family_hints": family_hints(answers),
        "binary_patterns": binary_patterns(answers, binary),
        "binary_prompts": binary,
    }
