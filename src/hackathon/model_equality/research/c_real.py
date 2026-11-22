"""Real-data posterior for the ep endpoints: text evidence tables, seeds, observed public scores.

Text evidence (5 prompts, one answer each) enters through two pair features only:

* `k`, the number of prompts on which two endpoints gave the same normalised answer (0..5);
* `c`, whether they name the same creator in the identity answer.

Their distribution under each pair state is an a-priori table (below), anchored on the data in
two places that can be checked: the cross-creator pairs (Qwen against Llama, 180 pairs) give the
cross-family table directly, and the same-creator pairs (198) are a mixture of same-model and
same-family pairs whose weights the tables reproduce (about a quarter of same-family pairs are
same-model). The tables are tempered (`EVIDENCE_POWER`): the pair features are built from the same
33 x 5 answers, so they are far from independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_data import AgreementData
from hackathon.model_equality.research.c_long_analysis import LongForm, log_evidence_long
from hackathon.model_equality.research.c_pairs import PairIndex, Partition, pair_index
from hackathon.model_equality.research.c_search import seed_partitions

FloatArray = npt.NDArray[np.float64]
EVIDENCE_POWER = 0.5
AGREEMENT_PMF: dict[str, tuple[float, ...]] = {
    "cross": (0.01, 0.27, 0.56, 0.14, 0.01, 0.01),
    "family": (0.01, 0.05, 0.22, 0.52, 0.15, 0.05),
    "model": (0.005, 0.02, 0.08, 0.30, 0.30, 0.295),
}
"""P(k agreeing prompts of 5 | pair state); rows sum to 1."""
CREATOR_MATCH_RATE: dict[str, float] = {"cross": 0.03, "family": 0.80, "model": 0.80}
"""P(same stated creator | pair state): a provider system prompt can rewrite the creator."""
STATE_ORDER: tuple[str, str, str] = ("cross", "family", "model")
SEED_COUNT = 400
SEED_NOISE = 0.10
CREATOR_MISMATCH_DISTANCE = 0.5
LONG_EVIDENCE_POWER = 0.5
LONG_AGREEMENT_PMF: dict[str, tuple[float, ...]] = {
    "cross": (0.60, 0.20, 0.09, 0.06, 0.03, 0.02),
    "family": (0.40, 0.20, 0.12, 0.10, 0.10, 0.08),
    "model": (0.12, 0.10, 0.10, 0.12, 0.16, 0.40),
}
"""WITHDRAWN as evidence (kept for the record): this table assumed greedy endpoints, where two
copies of one model give identical long texts. The first long prompt showed 33 distinct answers
out of 33, the 7 "identical" short-prompt endpoints included: the ep endpoints sample, the
identical short answers are the mode of the distribution, and the identical-count of long answers
carries no model information. `c_plan_real` leaves it off unless `--long-evidence` is given."""


def log_evidence_from_data(
    data: AgreementData, index: PairIndex | None = None, power: float = EVIDENCE_POWER,
    long_form: LongForm | None = None,
) -> FloatArray:
    """`[3, pairs]` tempered log-likelihood of the (k, creator) features under each pair state.

    With `long_form`, the identical-count on the long prompts adds a second, independent-looking
    block of evidence (`LONG_AGREEMENT_PMF`, hand-set, tempered by `LONG_EVIDENCE_POWER`).
    """
    idx = index or pair_index(len(data.endpoints))
    counts = data.agreement_count()[idx.first, idx.second]
    creators = np.array(data.creators)
    same_creator = creators[idx.first] == creators[idx.second]
    table = np.empty((3, idx.count))
    for state, name in enumerate(STATE_ORDER):
        pmf = np.array(AGREEMENT_PMF[name])
        rate = CREATOR_MATCH_RATE[name]
        table[state] = np.log(pmf[counts]) + np.log(np.where(same_creator, rate, 1.0 - rate))
    table = power * table
    if long_form is not None:
        table = table + log_evidence_long(
            long_form, idx.first, idx.second, LONG_AGREEMENT_PMF, LONG_EVIDENCE_POWER)
    return table


def distance_for_seeds(data: AgreementData, long_form: LongForm | None = None) -> FloatArray:
    """`[E, E]` seed distance: unmatched prompts share plus a penalty for different creators.

    With `long_form`, half of the weight goes to `1 - mean common-prefix ratio` on the long prompts.
    """
    prompts = len(data.prompts)
    counts = data.agreement_count().astype(np.float64)
    creators = np.array(data.creators)
    mismatch = (creators[:, None] != creators[None, :]).astype(np.float64)
    distance = (prompts - counts) / prompts
    if long_form is not None:
        distance = 0.5 * distance + 0.5 * (1.0 - long_form.mean_prefix)
    distance = distance + CREATOR_MISMATCH_DISTANCE * mismatch
    np.fill_diagonal(distance, 0.0)
    return np.asarray(distance, dtype=np.float64)


def seeds_from_data(
    data: AgreementData, rng: np.random.Generator, long_form: LongForm | None = None
) -> list[Partition]:
    """Random-cut linkage partitions of the answer distance (the text-only hypotheses)."""
    return seed_partitions(distance_for_seeds(data, long_form), rng, SEED_COUNT, noise=SEED_NOISE)


LIBRARY_SIZE = 3000
LIBRARY_NOISE_RANGE = (0.03, 0.35)
LIBRARY_EDIT_RANGE = (0, 4)
LIBRARY_LOOSE_NOISE_RANGE = (0.35, 0.8)
LIBRARY_LOOSE_EDIT_RANGE = (4, 12)
LIBRARY_RANDOM_SHARE = 0.15
LIBRARY_LOOSE_SHARE = 0.35
LIBRARY_EVIDENCE_POWER = 0.03
"""The library is a deliberately broad, near-uniform prior over structures: the text evidence only
breaks ties (5 prompts cannot support a sharp posterior over 33 endpoints)."""


def library_hypotheses(
    data: AgreementData, rng: np.random.Generator, count: int = LIBRARY_SIZE,
    long_form: LongForm | None = None,
) -> list[Partition]:
    """A diverse library of candidate truths in three components.

    * 50 %: noisy linkage cuts of the answer distance (the data-driven structures), 0-4 edits;
    * 35 %: the same with a much larger noise (0.35-0.8) and 4-12 edits: structures that only
      loosely follow the 5-prompt distance, because the distance is weak evidence;
    * 15 %: synthetic truths shaped like the organisers' ref set, unrelated to the answers.

    Each hypothesis draws its own cut heights (number of families and models). Nothing is
    selected on the posterior, which keeps the weights honest when the evidence is weak.
    """
    from hackathon.model_equality.research.c_evidence import random_truth
    from hackathon.model_equality.research.c_particles import propose_neighbour

    distance = distance_for_seeds(data, long_form)
    size = distance.shape[0]
    out: list[Partition] = []
    for _ in range(count):
        draw = float(rng.random())
        if draw < LIBRARY_RANDOM_SHARE:
            out.append(random_truth(rng, size))
            continue
        loose = draw < LIBRARY_RANDOM_SHARE + LIBRARY_LOOSE_SHARE
        noise_range = LIBRARY_LOOSE_NOISE_RANGE if loose else LIBRARY_NOISE_RANGE
        edit_range = LIBRARY_LOOSE_EDIT_RANGE if loose else LIBRARY_EDIT_RANGE
        partition = seed_partitions(distance, rng, 1, noise=float(rng.uniform(*noise_range)))[0]
        for _ in range(int(rng.integers(edit_range[0], edit_range[1] + 1))):
            partition = propose_neighbour(partition, rng)
        out.append(partition)
    return out


@dataclass(frozen=True)
class ObservedScore:
    """A submission already scored on the public board."""

    name: str
    partition: Partition
    score: float


def read_observed(
    queue_done_dir: Path, names: list[str]
) -> list[tuple[str, float]]:
    """`(name, score)` of the queue `done/<name>.json` files that exist and carry a score."""
    import json

    out: list[tuple[str, float]] = []
    for name in names:
        path = queue_done_dir / f"{name}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        if result.get("ok") and isinstance(result.get("score"), (int, float)):
            out.append((name, float(result["score"])))
    return out
