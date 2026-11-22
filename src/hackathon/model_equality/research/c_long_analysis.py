"""Long open prompts: exact agreement and common-prefix length between endpoints.

Two greedy endpoints serving the same model give the same text; a quantised or LoRA variant
diverges somewhere, and the position of the first divergence is a continuous, prompt-by-prompt
measure of how different two endpoints are. Sampled endpoints (temperature, watermark) diverge
early whatever the model, so they stay uninformative at this level.

* `prefix_ratio[p, i, j]`: length of the common prefix of the two normalised texts over the length
  of the longer one (1 = identical, 0 = different first character);
* `identical_count[i, j]`: prompts on which the two answers are identical after normalisation;
* `decoder_groups`: endpoints linked by `identical_count >= required` (default: every long prompt).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_data import AgreementData, clique_components, comparable

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
NO_OVERLAP_FLOOR = 1


@dataclass(frozen=True)
class LongForm:
    """Prefix statistics of a complete (prompts x endpoints) grid of long answers."""

    endpoints: tuple[str, ...]
    prompts: tuple[str, ...]
    prefix_ratio: FloatArray
    identical: npt.NDArray[np.bool_]

    @property
    def identical_count(self) -> IntArray:
        """`[E, E]` number of long prompts with identical normalised answers."""
        return np.asarray(self.identical.sum(axis=0), dtype=np.int64)

    @property
    def mean_prefix(self) -> FloatArray:
        """`[E, E]` mean over prompts of the common-prefix ratio."""
        return np.asarray(self.prefix_ratio.mean(axis=0), dtype=np.float64)


def prefix_length(left: str, right: str) -> int:
    """Number of leading characters two strings share."""
    return len(os.path.commonprefix([left, right]))


def build_long_form(data: AgreementData, prompt_ids: Sequence[str]) -> LongForm:
    """Prefix statistics over the long prompts of `data` that every endpoint answered.

    Raises:
        ValueError: none of `prompt_ids` is present and complete in `data`.
    """
    prompts = tuple(p for p in prompt_ids if p in data.texts)
    if not prompts:
        raise ValueError("no long prompt is complete yet")
    size = len(data.endpoints)
    ratio = np.zeros((len(prompts), size, size))
    same = np.zeros((len(prompts), size, size), dtype=bool)
    for p, prompt in enumerate(prompts):
        texts = [comparable(data.texts[prompt][e]) for e in data.endpoints]
        for i in range(size):
            for j in range(size):
                longer = max(len(texts[i]), len(texts[j]), NO_OVERLAP_FLOOR)
                ratio[p, i, j] = prefix_length(texts[i], texts[j]) / longer
                same[p, i, j] = texts[i] == texts[j]
    return LongForm(data.endpoints, prompts, ratio, same)


def decoder_groups(form: LongForm, required: int | None = None) -> list[list[int]]:
    """Endpoint groups identical on at least `required` long prompts (default: all of them)."""
    needed = len(form.prompts) if required is None else required
    return [g for g in clique_components(form.identical_count, needed) if len(g) > 1]


def split_clique(form: LongForm, members: Sequence[int], threshold: float) -> list[list[int]]:
    """Split a group of endpoints: single-link components of mean common-prefix ratio >= threshold."""
    mean = form.mean_prefix
    remaining = list(members)
    groups: list[list[int]] = []
    while remaining:
        group = [remaining.pop(0)]
        grew = True
        while grew:
            grew = False
            for other in list(remaining):
                if any(mean[member, other] >= threshold for member in group):
                    group.append(other)
                    remaining.remove(other)
                    grew = True
        groups.append(sorted(group))
    return groups


def log_evidence_long(
    form: LongForm, first: IntArray, second: IntArray,
    pmf: Mapping[str, tuple[float, ...]], power: float,
) -> FloatArray:
    """`[3, pairs]` log-likelihood of the identical-count under (cross, family, model)."""
    counts = form.identical_count[first, second]
    size = len(form.prompts)
    table = np.empty((3, len(first)))
    for state, name in enumerate(("cross", "family", "model")):
        probabilities = np.array(pmf[name])[: size + 1]
        probabilities = probabilities / probabilities.sum()
        table[state] = np.log(np.maximum(probabilities[counts], 1e-6))
    return power * table


def refinement_curve(
    data: AgreementData, prompts: Sequence[str], members: Sequence[int],
    rng: np.random.Generator, subsets: int = 200,
) -> list[float]:
    """Mean number of distinct answer signatures among `members` after k long prompts, k = 0..P.

    Two endpoints share a signature when their normalised answers are identical on every prompt
    of the subset. The curve rises while more prompts still split the group and flattens once
    the group is resolved: it prices the next prompts for this group. Averaged over random
    k-subsets of the available prompts (all subsets when P is small).
    """
    names = [data.endpoints[m] for m in members]
    keys = {p: [comparable(data.texts[p][n]) for n in names] for p in prompts}
    curve = [1.0]
    for k in range(1, len(prompts) + 1):
        counts = []
        for _ in range(subsets if k < len(prompts) else 1):
            chosen = list(prompts) if k == len(prompts) else list(
                rng.choice(list(prompts), size=k, replace=False))
            signatures = {tuple(keys[p][i] for p in chosen) for i in range(len(names))}
            counts.append(len(signatures))
        curve.append(float(np.mean(counts)))
    return curve


def extrapolate_refinement(curve: Sequence[float], horizon: Sequence[int]) -> dict[int, float]:
    """ESTIMATION: fit `n(k) = 1 + a (1 - r^k)` to the curve and predict it at `horizon` prompts."""
    values = np.asarray(curve, dtype=np.float64)
    ks = np.arange(len(values), dtype=np.float64)
    best: tuple[float, float, float] | None = None
    for ratio_value in np.linspace(0.05, 0.98, 94):
        ratio = float(ratio_value)
        basis = 1.0 - ratio**ks
        denominator = float((basis * basis).sum())
        if denominator <= 0:
            continue
        amplitude = float((basis * (values - 1.0)).sum() / denominator)
        error = float(((1.0 + amplitude * basis - values) ** 2).sum())
        if best is None or error < best[0]:
            best = (error, ratio, amplitude)
    if best is None:
        return {}
    _, ratio, amplitude = best
    return {int(k): float(1.0 + amplitude * (1.0 - ratio**k)) for k in horizon}
