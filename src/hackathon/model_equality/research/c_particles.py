"""Hypotheses on the true structure as a weighted particle set, with one-step value of information.

A particle is a nested two-level partition of the 33 endpoints (a candidate TRUTH). The set carries
a prior weight (from the text evidence) and the log-likelihood of every public score already
observed, so the posterior is `prior x prod_k P(y_k | truth, submission_k)`. Two questions are
answered from it, both in closed form over the particles:

* `expected_scores`: the exact expected final score (all pairs) of any candidate submission;
* `value_of_information`: for each candidate probe, the expected gain of the best final submission
  after seeing its public score (`voi`), and the mutual information between the score and the
  hypothesis (`mutual_information`, in bits).

Limitation (stated in the report): the public subset is one fixed random draw shared by every
submission, not a fresh draw per probe. The likelihood treats probes as independent draws, which
is exact for the first probe and a mild approximation afterwards (probes with the same family
labelling share the same MCC_family exactly).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_oracle import (
    DEFAULT_PUBLIC_FRACTION,
    cross_tabulate,
    cross_tabulate_many,
    official_score_from_cells,
    public_score_draws,
    quantize,
    score_bins,
)
from hackathon.model_equality.research.c_pairs import (
    IntArray,
    PairIndex,
    Partition,
    StateArray,
    pair_index,
    pair_states,
)

FloatArray = npt.NDArray[np.float64]
LIKELIHOOD_DRAWS = 64
LIKELIHOOD_FLOOR = 1e-4
"""Probability floor of an observed score under any particle: keeps one misspecified assumption
(public fraction, quantisation) from zeroing the whole posterior."""
KERNEL_BANDWIDTH = 0.012
"""Gaussian bandwidth (score units) of the likelihood kernel when the score is not quantised."""
MI_SMOOTHING = 1e-4


@dataclass
class ParticleSet:
    """Weighted nested partitions: labels `[T, n]`, pair states `[T, pairs]`, log weights `[T]`."""

    family: IntArray
    model: IntArray
    states: StateArray
    log_prior: FloatArray
    log_likelihood: FloatArray = field(default_factory=lambda: np.zeros(0))
    index: PairIndex = field(default_factory=pair_index)

    def __post_init__(self) -> None:
        if self.log_likelihood.shape[0] != self.family.shape[0]:
            self.log_likelihood = np.zeros(self.family.shape[0])

    @property
    def count(self) -> int:
        """Number of particles."""
        return int(self.family.shape[0])

    @staticmethod
    def from_partitions(
        partitions: Sequence[Partition], log_prior: FloatArray | None = None
    ) -> ParticleSet:
        """Stack partitions into a particle set (uniform prior by default)."""
        index = pair_index(partitions[0].size)
        family = np.stack([p.family for p in partitions])
        model = np.stack([p.model for p in partitions])
        prior = np.zeros(len(partitions)) if log_prior is None else log_prior
        return ParticleSet(family, model, pair_states(family, model, index), prior, index=index)

    def partition(self, position: int) -> Partition:
        """Particle `position` as a `Partition`."""
        return Partition(self.family[position].copy(), self.model[position].copy())

    def weights(self) -> FloatArray:
        """Normalised posterior weights."""
        log_w = self.log_prior + self.log_likelihood
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        return np.asarray(w / w.sum(), dtype=np.float64)

    def effective_size(self) -> float:
        """Kish effective sample size of the posterior weights."""
        w = self.weights()
        return float(1.0 / np.sum(w * w))

    def extend(self, other: ParticleSet) -> ParticleSet:
        """Concatenate two particle sets (priors and likelihoods are kept as given)."""
        return ParticleSet(
            np.concatenate([self.family, other.family]),
            np.concatenate([self.model, other.model]),
            np.concatenate([self.states, other.states]),
            np.concatenate([self.log_prior, other.log_prior]),
            np.concatenate([self.log_likelihood, other.log_likelihood]),
            self.index,
        )

    def select(self, keep: IntArray) -> ParticleSet:
        """The sub-set at positions `keep`."""
        return ParticleSet(
            self.family[keep], self.model[keep], self.states[keep],
            self.log_prior[keep], self.log_likelihood[keep], self.index,
        )


def observation_log_likelihood(
    states: StateArray,
    probe_states: StateArray,
    observed: float,
    step: float | None,
    rng: np.random.Generator,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION,
    draws: int = LIKELIHOOD_DRAWS,
    power: float = 1.0,
) -> FloatArray:
    """`power * log P(observed public score | truth)` for every truth in `states` `[T, pairs]`.

    `power < 1` tempers the likelihood: the public subset is one fixed draw shared by all the
    probes, so successive probes are not independent and the product of their likelihoods is
    over-confident.
    """
    cells = cross_tabulate(states, probe_states)
    simulated = public_score_draws(cells, draws, rng, public_fraction, step=None)
    if step:
        hit = np.abs(quantize(simulated, step) - observed) < step / 2.0 + 1e-9
        probability = hit.mean(axis=1)
    else:
        kernel = np.exp(-0.5 * ((simulated - observed) / KERNEL_BANDWIDTH) ** 2)
        probability = kernel.mean(axis=1)
    return np.asarray(power * np.log(np.maximum(probability, LIKELIHOOD_FLOOR)), dtype=np.float64)


def observe(
    particles: ParticleSet,
    probe: Partition,
    observed: float,
    step: float | None,
    rng: np.random.Generator,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION,
    power: float = 1.0,
) -> None:
    """Fold one observed public score into the particle weights (in place)."""
    probe_states = probe.states(particles.index)
    particles.log_likelihood = particles.log_likelihood + observation_log_likelihood(
        particles.states, probe_states, observed, step, rng, public_fraction, power=power
    )


def expected_scores(
    particles: ParticleSet, candidates: StateArray
) -> tuple[FloatArray, FloatArray]:
    """`(expected final score per candidate [C], score matrix [T, C])` over all pairs."""
    cells = cross_tabulate_many(particles.states, candidates)
    matrix = official_score_from_cells(cells)
    return np.asarray(particles.weights() @ matrix, dtype=np.float64), matrix


@dataclass(frozen=True)
class ProbeValue:
    """One-step value of a probe: expected score gain of the best submission, and information."""

    voi: FloatArray
    mutual_information_bits: FloatArray
    baseline: float
    outcome_entropy_bits: FloatArray


def value_of_information(
    particles: ParticleSet,
    probes: StateArray,
    decisions: StateArray,
    step: float | None,
    rng: np.random.Generator,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION,
    draws: int = 48,
) -> ProbeValue:
    """VOI and mutual information of each probe (rows of `probes`), decisions = candidate finals.

    `voi[c] = E_y[ max_d E[score_d | y] ] - max_d E[score_d]` with y the public score of probe c.
    """
    weights = particles.weights()
    cells = cross_tabulate_many(particles.states, probes)
    utility = official_score_from_cells(cross_tabulate_many(particles.states, decisions))
    baseline = float((weights @ utility).max())
    edges = score_bins(step)
    bin_count = len(edges) - 1
    voi = np.zeros(probes.shape[0])
    mutual = np.zeros(probes.shape[0])
    entropy = np.zeros(probes.shape[0])
    rows = np.arange(particles.count)[:, None] * bin_count
    for probe_position in range(probes.shape[0]):
        simulated = public_score_draws(cells[:, probe_position, :], draws, rng, public_fraction, step)
        bins = np.clip(np.searchsorted(edges, simulated, side="right") - 1, 0, bin_count - 1)
        counts = np.bincount((rows + bins).ravel(), minlength=particles.count * bin_count)
        likelihood = counts.reshape(particles.count, bin_count) / draws + MI_SMOOTHING
        likelihood /= likelihood.sum(axis=1, keepdims=True)
        weighted = likelihood * weights[:, None]
        joint_value = weighted.T @ utility
        voi[probe_position] = float(joint_value.max(axis=1).sum()) - baseline
        outcome = weighted.sum(axis=0)
        entropy[probe_position] = float(-(outcome * np.log2(outcome)).sum())
        mutual[probe_position] = float(
            (weighted * (np.log2(likelihood) - np.log2(outcome)[None, :])).sum()
        )
    return ProbeValue(voi=voi, mutual_information_bits=mutual, baseline=baseline,
                      outcome_entropy_bits=entropy)


def propose_neighbour(partition: Partition, rng: np.random.Generator) -> Partition:
    """One random local edit: move an endpoint, split a model, merge two models or two families."""
    family = partition.family.copy()
    model = partition.model.copy()
    size = partition.size
    next_model = int(model.max()) + 1
    kind = int(rng.integers(0, 5))
    if kind == 0:  # move one endpoint into another existing model (maybe another family)
        endpoint = int(rng.integers(0, size))
        target = int(rng.integers(0, size))
        model[endpoint], family[endpoint] = model[target], family[target]
    elif kind == 1:  # make one endpoint a model of its own, in its current family
        endpoint = int(rng.integers(0, size))
        model[endpoint] = next_model
    elif kind == 2:  # split a random model in two halves
        endpoint = int(rng.integers(0, size))
        members = np.flatnonzero(model == model[endpoint])
        if len(members) > 1:
            moved = members[rng.random(len(members)) < 0.5]
            model[moved] = next_model
    elif kind == 3:  # merge two models of the same family
        first, second = (int(rng.integers(0, size)) for _ in range(2))
        if family[first] == family[second]:
            model[model == model[second]] = model[first]
    else:  # merge two families (their models stay distinct)
        first, second = (int(rng.integers(0, size)) for _ in range(2))
        family[family == family[second]] = family[first]
    return Partition(family, model)
