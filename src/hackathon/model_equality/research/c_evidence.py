"""Text evidence as a pairwise log-likelihood table, and the synthetic truths used to test plans.

The text analysis (distances between endpoints' answers) enters the plan only through
`log_evidence[state, pair]`: the log-likelihood of the observed pair feature under each of the
three pair states. The log prior of a hypothetical truth is then the sum over pairs of the entry
of its own state, tempered by `power` (< 1 admits that the pair features are not independent).

`PairFeatureModel` is a Gaussian class-conditional model of ONE scalar pair distance. It is used
two ways: to turn the real distance matrices into `log_evidence`, and, with the same parameters,
to generate synthetic distances from a synthetic truth, so the closed-loop test of the probing
plan sees evidence of exactly the quality the model claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_pairs import (
    STATE_COUNT,
    IntArray,
    PairIndex,
    Partition,
    StateArray,
    pair_index,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PairFeatureModel:
    """Gaussian distance per pair state: `means[s]`, `sigmas[s]` for s in (cross, family, model)."""

    means: tuple[float, float, float]
    sigmas: tuple[float, float, float]

    def log_evidence(self, distance: FloatArray, index: PairIndex, power: float = 1.0) -> FloatArray:
        """`[3, pairs]` log-likelihood of the observed pair distances under each pair state."""
        values = distance[index.first, index.second]
        table = np.empty((STATE_COUNT, index.count))
        for state in range(STATE_COUNT):
            mean, sigma = self.means[state], self.sigmas[state]
            table[state] = -0.5 * ((values - mean) / sigma) ** 2 - np.log(sigma)
        return power * table

    def simulate(
        self, states: StateArray, size: int, rng: np.random.Generator
    ) -> FloatArray:
        """A symmetric `[size, size]` distance matrix drawn from the model for given pair states."""
        index = pair_index(size)
        means = np.array(self.means)[states]
        sigmas = np.array(self.sigmas)[states]
        values = np.maximum(rng.normal(means, sigmas), 0.0)
        matrix = np.zeros((size, size))
        matrix[index.first, index.second] = values
        matrix[index.second, index.first] = values
        return matrix


def pair_log_prior(states: StateArray, log_evidence: FloatArray) -> FloatArray:
    """`sum_p log_evidence[state_p, p]` for every row of `states` `[T, pairs]`."""
    gathered = np.take_along_axis(log_evidence.T[None, :, :], states[:, :, None].astype(np.int64), 2)
    return np.asarray(gathered[:, :, 0].sum(axis=1), dtype=np.float64)


def random_truth(
    rng: np.random.Generator,
    size: int = 33,
    family_count: int | None = None,
) -> Partition:
    """A synthetic truth shaped like the organisers' ref set, scaled to `size` endpoints.

    The ref set is one family of 9 endpoints: a base model with 5 disguised copies (system
    prompt, persona, watermark, temperature, plain) plus 4 singleton variants (fp8, three LoRA
    strengths). Here each family gets a main model (about half of its endpoints) and the rest
    are variants: singletons, with occasional pairs (two endpoints sharing a LoRA setting).
    """
    families = family_count or int(rng.integers(2, 5))
    cuts = np.sort(rng.choice(np.arange(3, size - 2), size=families - 1, replace=False))
    bounds = [0, *cuts.tolist(), size]
    groups: list[list[int]] = []
    family_of_group: list[int] = []
    order = rng.permutation(size)
    for family_id in range(families):
        members = order[bounds[family_id]:bounds[family_id + 1]].tolist()
        main_size = max(1, round(len(members) * rng.uniform(0.35, 0.65)))
        groups.append(members[:main_size])
        family_of_group.append(family_id)
        rest = members[main_size:]
        while rest:
            take = 2 if (len(rest) >= 2 and rng.random() < 0.2) else 1
            groups.append(rest[:take])
            family_of_group.append(family_id)
            rest = rest[take:]
    return Partition.from_groups(groups, family_of_group)


def evidence_estimate(
    log_evidence: FloatArray, seeds: Sequence[Partition], rng: np.random.Generator,
    sweeps: int = 12,
) -> Partition:
    """Greedy text-only estimate: best of `seeds` after pair-evidence hill climbing (diagnostics)."""
    from hackathon.model_equality.research.c_particles import propose_neighbour

    index = pair_index(seeds[0].size)
    best = max(seeds, key=lambda p: float(pair_log_prior(p.states(index)[None], log_evidence)[0]))
    best_value = float(pair_log_prior(best.states(index)[None], log_evidence)[0])
    for _ in range(sweeps * best.size):
        candidate = propose_neighbour(best, rng)
        value = float(pair_log_prior(candidate.states(index)[None], log_evidence)[0])
        if value > best_value:
            best, best_value = candidate, value
    return best


def family_of_model_array(partition: Partition) -> IntArray:
    """Family id of every model id (dense array indexed by model id)."""
    out = np.zeros(int(partition.model.max()) + 1, dtype=np.int64)
    out[partition.model] = partition.family
    return out
