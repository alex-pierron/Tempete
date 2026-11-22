"""Rejection sampling (ABC) of organiser-like truths consistent with the observed public scores.

Why a second posterior. The data-driven library of `c_real` builds hypotheses from the text
distance; three submissions built from text groups (E1, E1b, probe 3) all scored exactly 0.05,
which says the text groups carry no model information. This module draws truths WITHOUT looking at
the text: the families follow the stated creator (Qwen-like, Llama-like), a family may be split in
2 or 3 true families, and inside each true family the model structure is the organisers' one (the
ref set): a main model holding 35-65 % of the endpoints (copies with system prompt, persona,
watermark, temperature) plus variants alone (fp8, LoRA). Which endpoints are variants is random,
except that endpoints listed as `suspects` (garbled or rambling text, the signature of a strong
LoRA on the ref) are variants with probability `suspect_outlier_probability`.

A drawn truth is kept when the public score it would have given to each observed submission
rounds to the observed one (same thinning model as `c_oracle`). The kept truths are then used to
predict the public score of any candidate: this is the posterior predictive of the organiser-like
hypothesis class, conditioned on every score read so far.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_oracle import (
    DEFAULT_PUBLIC_FRACTION,
    cross_tabulate_many,
    official_score_from_cells,
    public_score_draws,
)
from hackathon.model_equality.research.c_pairs import Partition, StateArray, pair_index

FloatArray = npt.NDArray[np.float64]
MAIN_MODEL_SHARE = (0.35, 0.65)
PAIR_VARIANT_PROBABILITY = 0.0
DEFAULT_DRAWS = 40000
PRIOR_ORGANISER = "organiser"
PRIOR_BALANCED = "balanced"
PUBLIC_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


@dataclass(frozen=True)
class AbcResult:
    """Kept truths and the predictive distribution of candidate public scores."""

    accepted: int
    drawn: int
    names: tuple[str, ...]
    mean_full: FloatArray
    sd_full: FloatArray
    level_probabilities: FloatArray
    """`[candidates, len(PUBLIC_LEVELS) + 1]`: P(public score = level) and P(>= 0.30)."""
    p_above_floor: FloatArray

    @property
    def acceptance_rate(self) -> float:
        """Share of drawn truths consistent with every observation."""
        return self.accepted / max(self.drawn, 1)


BALANCED_MODEL_COUNT = (2, 6)
"""Competing hypothesis: each true family holds 2 to 6 models of comparable size."""


def balanced_truth(
    rng: np.random.Generator, groups: Sequence[Sequence[int]], split_options: Sequence[Sequence[int]],
) -> Partition:
    """A truth with several medium-size models per family, roles random (no main model, no suspects).

    This is the competing hypothesis to the organiser-like one: if carving out suspect variants
    from a main model does not pay, models of comparable size are the next candidate.
    """
    models: list[list[int]] = []
    families: list[int] = []
    family_id = 0
    for members, options in zip(groups, split_options, strict=True):
        count = int(rng.choice(list(options)))
        order = [int(x) for x in rng.permutation(list(members))]
        cuts = sorted(rng.choice(np.arange(2, len(order) - 1), size=count - 1, replace=False)) \
            if count > 1 else []
        for part in np.split(np.array(order), cuts):
            chosen = [int(x) for x in part]
            model_count = int(rng.integers(BALANCED_MODEL_COUNT[0], BALANCED_MODEL_COUNT[1] + 1))
            model_count = max(1, min(model_count, len(chosen)))
            labels = rng.integers(0, model_count, size=len(chosen))
            for label in sorted(set(labels.tolist())):
                models.append([m for m, lab in zip(chosen, labels.tolist(), strict=True) if lab == label])
                families.append(family_id)
            family_id += 1
    return Partition.from_groups(models, families)


def organiser_truth(
    rng: np.random.Generator, groups: Sequence[Sequence[int]], suspects: frozenset[int],
    suspect_outlier_probability: float, split_options: Sequence[Sequence[int]],
    anchors: frozenset[int] = frozenset(),
) -> Partition:
    """One organiser-like truth over `groups` of endpoints (one group per stated creator).

    Args:
        rng: Random generator.
        groups: Endpoint positions of each creator group (the base family candidates).
        suspects: Positions that are variants with probability `suspect_outlier_probability`.
        suspect_outlier_probability: See module docstring.
        split_options: For each group, the allowed numbers of true families it is split into.
        anchors: Positions known to belong to the main model of their family (copies disguised by
            a system prompt or a persona: on the ref these are the same model as the plain one).
            They are never variants and are always part of the main model.
    """
    models: list[list[int]] = []
    families: list[int] = []
    family_id = 0
    for members, options in zip(groups, split_options, strict=True):
        count = int(rng.choice(list(options)))
        order = [int(x) for x in rng.permutation(list(members))]
        cuts = sorted(rng.choice(np.arange(2, len(order) - 1), size=count - 1, replace=False)) \
            if count > 1 else []
        for part in np.split(np.array(order), cuts):
            chosen = [int(x) for x in part]
            forced = [m for m in chosen if m in anchors]
            main_size = max(1, len(forced), round(len(chosen) * rng.uniform(*MAIN_MODEL_SHARE)))
            variants = {m for m in chosen if m in suspects and m not in anchors
                        and rng.random() < suspect_outlier_probability}
            eligible = [m for m in chosen if m not in variants and m not in forced]
            extra = [int(x) for x in rng.choice(
                eligible, size=min(main_size - len(forced), len(eligible)), replace=False)]
            main = [*forced, *extra]
            models.append(main)
            families.append(family_id)
            for member in chosen:
                if member not in main:
                    models.append([member])
                    families.append(family_id)
            family_id += 1
    return Partition.from_groups(models, families)


def abc_posterior(
    groups: Sequence[Sequence[int]], suspects: frozenset[int],
    observed: Mapping[str, tuple[StateArray, float]], candidates: Mapping[str, StateArray],
    rng: np.random.Generator, draws: int = DEFAULT_DRAWS, step: float = 0.05,
    suspect_outlier_probability: float = 0.55, split_options: Sequence[Sequence[int]] | None = None,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION, prior: str = PRIOR_ORGANISER,
    anchors: frozenset[int] = frozenset(),
) -> AbcResult:
    """Keep truths whose simulated public scores round to every observed score.

    `prior` is `PRIOR_ORGANISER` (main model plus variants) or `PRIOR_BALANCED` (several
    medium-size models per family).
    """
    options = split_options or [(1, 2, 3)] + [(1, 2)] * (len(groups) - 1)
    probes = np.stack([s for s, _ in observed.values()])
    targets = np.array([y for _, y in observed.values()])
    names = tuple(candidates)
    cand = np.stack(list(candidates.values()))
    kept_full: list[FloatArray] = []
    kept_levels: list[FloatArray] = []
    for _ in range(draws):
        truth = (balanced_truth(rng, groups, options) if prior == PRIOR_BALANCED
                 else organiser_truth(rng, groups, suspects, suspect_outlier_probability, options,
                                     anchors))
        states = truth.states(pair_index(truth.size))[None]
        cells = cross_tabulate_many(states, probes)[0]
        simulated = public_score_draws(cells, 1, rng, public_fraction, step=step)[:, 0]
        if np.all(np.abs(simulated - targets) < step / 2.0 + 1e-9):
            cand_cells = cross_tabulate_many(states, cand)[0]
            kept_full.append(official_score_from_cells(cand_cells))
            kept_levels.append(public_score_draws(cand_cells, 1, rng, public_fraction, step=step)[:, 0])
    if not kept_full:
        empty = np.zeros(len(names))
        return AbcResult(0, draws, names, empty, empty,
                         np.zeros((len(names), len(PUBLIC_LEVELS) + 1)), empty)
    full = np.array(kept_full)
    levels = np.array(kept_levels)
    probabilities = np.stack(
        [np.mean(np.abs(levels - lo) < step / 2.0 + 1e-9, axis=0) for lo in PUBLIC_LEVELS]
        + [np.mean(levels > PUBLIC_LEVELS[-1] + step / 2.0, axis=0)], axis=1)
    return AbcResult(len(kept_full), draws, names, full.mean(axis=0), full.std(axis=0),
                     probabilities, np.mean(levels > 0.05 + step / 2.0, axis=0))


def abc_sample(
    groups: Sequence[Sequence[int]], suspects: frozenset[int],
    observed: Mapping[str, tuple[StateArray, float]], rng: np.random.Generator,
    draws: int = DEFAULT_DRAWS, step: float = 0.05, suspect_outlier_probability: float = 0.55,
    split_options: Sequence[Sequence[int]] | None = None,
    public_fraction: float = DEFAULT_PUBLIC_FRACTION, prior: str = PRIOR_ORGANISER,
    anchors: frozenset[int] = frozenset(),
) -> list[Partition]:
    """The truths kept by rejection (the posterior sample itself, for planning on top of it)."""
    options = split_options or [(1, 2, 3)] + [(1, 2)] * (len(groups) - 1)
    probes = np.stack([s for s, _ in observed.values()])
    targets = np.array([y for _, y in observed.values()])
    kept: list[Partition] = []
    for _ in range(draws):
        truth = (balanced_truth(rng, groups, options) if prior == PRIOR_BALANCED
                 else organiser_truth(rng, groups, suspects, suspect_outlier_probability, options,
                                     anchors))
        cells = cross_tabulate_many(truth.states(pair_index(truth.size))[None], probes)[0]
        simulated = public_score_draws(cells, 1, rng, public_fraction, step=step)[:, 0]
        if np.all(np.abs(simulated - targets) < step / 2.0 + 1e-9):
            kept.append(truth)
    return kept
