"""What does "four times 0.05" say, for several shapes of the public split?

The public leaderboard scores an unknown subset of the 528 pairs. Three shapes are compared:

* `pairs30`: every pair is public independently with probability 0.3 (the working assumption);
* `endpoints_both`: 30 % of the endpoints are public and a pair is public when BOTH ends are
  (about 9 % of the pairs, 45 pairs for 10 endpoints);
* `endpoints_any`: 30 % of the endpoints are public and a pair is public when AT LEAST ONE end is
  (about 51 % of the pairs).

Unlike the independent-thinning shortcut of `c_oracle`, the mask drawn here is FIXED and shared by
all the submissions, which is what the server does. A truth is kept (ABC) when the public score of
every observed submission, computed on that one mask, rounds to the observed score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_abc import (
    PRIOR_BALANCED,
    balanced_truth,
    organiser_truth,
)
from hackathon.model_equality.research.c_oracle import cross_tabulate, official_score_from_cells
from hackathon.model_equality.research.c_pairs import (
    BoolArray,
    PairIndex,
    Partition,
    StateArray,
    pair_index,
)

SPLIT_PAIRS = "pairs30"
SPLIT_ENDPOINTS_BOTH = "endpoints_both"
SPLIT_ENDPOINTS_ANY = "endpoints_any"
SPLITS: tuple[str, ...] = (SPLIT_PAIRS, SPLIT_ENDPOINTS_BOTH, SPLIT_ENDPOINTS_ANY)
PUBLIC_SHARE = 0.3
STEP = 0.05
FloatArray = npt.NDArray[np.float64]


def sample_mask(kind: str, index: PairIndex, rng: np.random.Generator) -> BoolArray:
    """A public mask `[pairs]` of the given shape (see module docstring)."""
    if kind == SPLIT_PAIRS:
        return np.asarray(rng.random(index.count) < PUBLIC_SHARE, dtype=np.bool_)
    chosen = np.zeros(index.size, dtype=bool)
    chosen[rng.choice(index.size, size=round(PUBLIC_SHARE * index.size), replace=False)] = True
    both = chosen[index.first] & chosen[index.second]
    either = chosen[index.first] | chosen[index.second]
    return np.asarray(both if kind == SPLIT_ENDPOINTS_BOTH else either, dtype=np.bool_)


def masked_score(truth: StateArray, probe: StateArray, mask: BoolArray) -> float:
    """Official score of `probe` against `truth` restricted to the public pairs of `mask`."""
    return float(official_score_from_cells(cross_tabulate(truth[mask], probe[mask])))


def draw_truth(
    prior: str, rng: np.random.Generator, groups: Sequence[Sequence[int]],
    suspects: frozenset[int], suspect_probability: float,
) -> Partition:
    """One truth of the named prior (`organiser` uses `suspects`, `balanced` ignores them)."""
    options = [(1, 2, 3)] + [(1, 2)] * (len(groups) - 1)
    if prior == PRIOR_BALANCED:
        return balanced_truth(rng, groups, options)
    return organiser_truth(rng, groups, suspects, suspect_probability, options)


def acceptance_rate(
    prior: str, split: str, groups: Sequence[Sequence[int]], suspects: frozenset[int],
    observed: Mapping[str, tuple[StateArray, float]], rng: np.random.Generator, draws: int,
    suspect_probability: float = 0.55,
) -> float:
    """Share of drawn (truth, mask) couples reproducing every observed public score."""
    size = sum(len(g) for g in groups)
    index = pair_index(size)
    kept = 0
    for _ in range(draws):
        truth = draw_truth(prior, rng, groups, suspects, suspect_probability)
        states = truth.states(index)
        mask = sample_mask(split, index, rng)
        if all(abs(round(masked_score(states, probe, mask) / STEP) * STEP - target) < 1e-9
               for probe, target in observed.values()):
            kept += 1
    return kept / draws
