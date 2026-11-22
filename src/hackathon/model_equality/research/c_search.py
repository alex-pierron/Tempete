"""Sequential probing: particle population, local expansion, best response, one-step planner.

The planner keeps a population of hypotheses (nested partitions) with weight
`exp(text log-evidence + sum of observation log-likelihoods)`. After each observed public score it
(1) re-weights, (2) proposes local edits of the heaviest hypotheses and adds the ones the data
like (a cheap sequential Monte Carlo with greedy move step: no Metropolis acceptance, but every
kept hypothesis is scored by the exact posterior), (3) picks the next probe by one-step value of
information among candidates built from the posterior (best response, its one-edit neighbours,
single-family variants that isolate the model MCC).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hackathon.model_equality.research.c_evidence import pair_log_prior
from hackathon.model_equality.research.c_oracle import DEFAULT_PUBLIC_FRACTION
from hackathon.model_equality.research.c_pairs import (
    IntArray,
    PairIndex,
    Partition,
    StateArray,
    canonical,
    pair_index,
)
from hackathon.model_equality.research.c_particles import (
    ParticleSet,
    expected_scores,
    observation_log_likelihood,
    propose_neighbour,
    value_of_information,
)

FloatArray = npt.NDArray[np.float64]
SEED_CUT_RANGE = (0.2, 1.0)
"""Heights between which model and family cuts of the seed dendrograms are drawn."""


@dataclass(frozen=True)
class SearchConfig:
    """Knobs of the sequential search (sizes chosen for a few seconds per round)."""

    population: int = 1500
    elite: int = 120
    neighbours_per_elite: int = 10
    max_edits: int = 3
    probe_candidates: int = 45
    voi_draws: int = 24
    response_rounds: int = 12
    response_neighbours: int = 90
    public_fraction: float = DEFAULT_PUBLIC_FRACTION
    step: float | None = 0.01
    likelihood_power: float = 0.6
    selection_temperature: float = 4.0


@dataclass
class Observation:
    """A submitted probe and its observed public score."""

    probe: StateArray
    score: float


@dataclass
class Planner:
    """Posterior over truths given text evidence and the observations so far."""

    log_evidence: FloatArray
    config: SearchConfig
    rng: np.random.Generator
    particles: ParticleSet
    history: list[Observation] = field(default_factory=list)
    extras: list[Partition] = field(default_factory=list)

    @property
    def index(self) -> PairIndex:
        """The pair index of the particles."""
        return self.particles.index

    def _likelihood_of_history(self, states: StateArray) -> FloatArray:
        total = np.zeros(states.shape[0])
        for item in self.history:
            total += observation_log_likelihood(
                states, item.probe, item.score, self.config.step, self.rng,
                self.config.public_fraction, power=self.config.likelihood_power,
            )
        return total

    def observe(self, probe: Partition, score: float) -> None:
        """Record an observed public score and re-weight every particle."""
        states = probe.states(self.index)
        self.history.append(Observation(states, score))
        self.particles.log_likelihood = self.particles.log_likelihood + observation_log_likelihood(
            self.particles.states, states, score, self.config.step, self.rng,
            self.config.public_fraction, power=self.config.likelihood_power,
        )

    def expand(self) -> None:
        """Add local edits of the heaviest particles, keep the best `population` by posterior."""
        weights = self.particles.weights()
        elite = np.argsort(weights)[::-1][: self.config.elite]
        children: list[Partition] = []
        for position in elite:
            parent = self.particles.partition(int(position))
            for _ in range(self.config.neighbours_per_elite):
                child = parent
                for _ in range(int(self.rng.integers(1, self.config.max_edits + 1))):
                    child = propose_neighbour(child, self.rng)
                children.append(child)
        fresh = ParticleSet.from_partitions(children)
        fresh.log_prior = pair_log_prior(fresh.states, self.log_evidence)
        fresh.log_likelihood = self._likelihood_of_history(fresh.states)
        merged = self.particles.extend(fresh)
        posterior = merged.log_prior + merged.log_likelihood
        self.particles = merged.select(self._select(posterior))

    def _select(self, posterior: FloatArray) -> IntArray:
        """Keep the best half outright, sample the other half by tempered posterior weight.

        Pure top-k selection collapses the population onto one mode and makes every expected
        score look certain; the sampled half keeps competing hypotheses alive.
        """
        count = self.config.population
        order = np.argsort(posterior)[::-1]
        sure = order[: count // 2]
        rest = order[count // 2:]
        if len(rest) <= count - len(sure):
            return order[:count]
        logits = (posterior[rest] - posterior[rest].max()) / self.config.selection_temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        drawn = self.rng.choice(rest, size=count - len(sure), replace=False, p=probabilities)
        return np.concatenate([sure, drawn])

    def best_response(self, start: Partition | None = None) -> tuple[Partition, float]:
        """Hill-climb the submission maximising the posterior-expected final score."""
        current = start or self.particles.partition(int(np.argmax(self.particles.weights())))
        value = float(expected_scores(self.particles, current.states(self.index)[None])[0][0])
        for _ in range(self.config.response_rounds):
            neighbours = [current] + [
                propose_neighbour(current, self.rng) for _ in range(self.config.response_neighbours)
            ]
            states = np.stack([p.states(self.index) for p in neighbours])
            scores = expected_scores(self.particles, states)[0]
            best = int(np.argmax(scores))
            if scores[best] <= value + 1e-12:
                break
            current, value = neighbours[best], float(scores[best])
        return current, value

    def candidates(self, best: Partition) -> list[Partition]:
        """Probe/decision candidates: best response, neighbours, heavy particles, extras, 1-family."""
        pool: list[Partition] = [best, *self.extras]
        weights = self.particles.weights()
        heavy = np.argsort(weights)[::-1][: self.config.probe_candidates // 3]
        pool += [self.particles.partition(int(k)) for k in heavy]
        while len(pool) < self.config.probe_candidates:
            pool.append(propose_neighbour(best, self.rng))
        single = [
            Partition(np.zeros_like(p.family), p.model.copy())
            for p in [best, *self.extras, *pool[len(self.extras) + 1: len(pool) // 2]]
        ]
        return pool + single

    def plan_table(self) -> PlanTable:
        """VOI, information and expected final score of every candidate probe."""
        best, _ = self.best_response()
        pool = self.candidates(best)
        states = np.stack([p.states(self.index) for p in pool])
        value = value_of_information(
            self.particles, states, states, self.config.step, self.rng,
            self.config.public_fraction, self.config.voi_draws,
        )
        expected, _ = expected_scores(self.particles, states)
        return PlanTable(pool, expected, value.voi, value.mutual_information_bits, value.baseline)

    def plan_next(self) -> tuple[Partition, float, float, Partition]:
        """Next probe by one-step VOI: `(probe, voi, mutual information bits, best response)`."""
        table = self.plan_table()
        winner = int(np.argmax(table.voi))
        return (table.pool[winner], float(table.voi[winner]),
                float(table.mutual_bits[winner]), table.pool[0])


@dataclass(frozen=True)
class PlanTable:
    """One-step plan: candidates (index 0 = best response) with their scores and information."""

    pool: list[Partition]
    expected: FloatArray
    voi: FloatArray
    mutual_bits: FloatArray
    baseline: float


def seed_partitions(
    distance: FloatArray, rng: np.random.Generator, count: int, noise: float = 0.08
) -> list[Partition]:
    """Random-cut average-linkage partitions of a noisy copy of `distance` (two nested cuts)."""
    out: list[Partition] = []
    low, high = SEED_CUT_RANGE
    for _ in range(count):
        jitter = rng.normal(0.0, noise, size=distance.shape)
        jitter = (jitter + jitter.T) / 2.0
        noisy = np.maximum(distance + jitter, 0.0)
        np.fill_diagonal(noisy, 0.0)
        tree = linkage(squareform(noisy, checks=False), method="average")
        cuts = np.sort(rng.uniform(low, high, size=2))
        model = canonical(fcluster(tree, t=cuts[0], criterion="distance"))
        family = canonical(fcluster(tree, t=cuts[1], criterion="distance"))
        out.append(Partition(_nest(family, model), model))
    return out


def _nest(family: IntArray, model: IntArray) -> IntArray:
    """Family ids made consistent with models (a model lies in the family of its first member)."""
    first_family: dict[int, int] = {}
    nested = np.empty_like(family)
    for position, (f_id, m_id) in enumerate(zip(family.tolist(), model.tolist(), strict=True)):
        nested[position] = first_family.setdefault(m_id, f_id)
    return nested


def initial_planner(
    log_evidence: FloatArray,
    seeds: Sequence[Partition],
    config: SearchConfig,
    rng: np.random.Generator,
) -> Planner:
    """Planner over seed hypotheses weighted by the text evidence alone."""
    index = pair_index(seeds[0].size)
    particles = ParticleSet.from_partitions(seeds)
    particles.index = index
    particles.log_prior = pair_log_prior(particles.states, log_evidence)
    return Planner(log_evidence, config, rng, particles)
