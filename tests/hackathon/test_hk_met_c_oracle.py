"""Team C: pair algebra and oracle cross-checked against the official-score reimplementation."""

from __future__ import annotations

import numpy as np
import pytest

from hackathon.model_equality.research.c_oracle import (
    CELL_COUNT,
    cross_tabulate,
    cross_tabulate_many,
    full_score,
    mcc_from_counts,
    public_score_draws,
    quantize,
    thin_cells,
)
from hackathon.model_equality.research.c_pairs import (
    Partition,
    PartitionError,
    canonical,
    endpoint_names,
    pair_index,
)
from hackathon.model_equality.scoring import official_score

SIZE = 12
SEED = 20261121


def random_partition(rng: np.random.Generator, size: int = SIZE) -> Partition:
    """A random nested two-level partition."""
    families = rng.integers(0, 3, size=size)
    models = families * 10 + rng.integers(0, 3, size=size)
    return Partition(families.astype(np.int64), models.astype(np.int64))


def as_labeling(partition: Partition) -> dict[str, tuple[str, str]]:
    """The `Labeling` mapping `scoring.official_score` expects."""
    names = endpoint_names(partition.size)
    return {
        name: (str(f), str(m))
        for name, f, m in zip(names, partition.family.tolist(), partition.model.tolist(), strict=True)
    }


def test_full_score_matches_official_score_on_random_labelings() -> None:
    rng = np.random.default_rng(SEED)
    index = pair_index(SIZE)
    for _ in range(60):
        truth, guess = random_partition(rng), random_partition(rng)
        expected = official_score(as_labeling(truth), as_labeling(guess)).score
        got = float(full_score(truth.states(index), guess.states(index)))
        assert got == pytest.approx(expected, abs=1e-12)


def test_cross_tabulate_many_equals_single_calls() -> None:
    rng = np.random.default_rng(SEED)
    index = pair_index(SIZE)
    truths = np.stack([random_partition(rng).states(index) for _ in range(5)])
    guesses = np.stack([random_partition(rng).states(index) for _ in range(4)])
    many = cross_tabulate_many(truths, guesses)
    assert many.shape == (5, 4, CELL_COUNT)
    for t in range(5):
        for g in range(4):
            np.testing.assert_array_equal(many[t, g], cross_tabulate(truths[t], guesses[g]))
    assert np.all(many.sum(axis=-1) == index.count)


def test_zero_denominator_gives_zero_mcc() -> None:
    zero = np.zeros(1)
    one = np.ones(1)
    assert mcc_from_counts(one * 5, zero, zero, zero)[0] == 0.0
    assert mcc_from_counts(one * 3, one, one, one * 3)[0] == pytest.approx(0.5)


def test_single_model_per_family_scores_only_family_part() -> None:
    """The E1 shape: one model per family gives MCC_model = 0 whatever the truth."""
    rng = np.random.default_rng(SEED)
    index = pair_index(SIZE)
    truth = random_partition(rng)
    guess = Partition(truth.family.copy(), truth.family.copy())
    score = float(full_score(truth.states(index), guess.states(index)))
    assert score == pytest.approx(0.1, abs=1e-12)  # perfect family split, MCC_model forced to 0


def test_one_family_submission_isolates_the_model_question() -> None:
    """With every endpoint in one family the family MCC is 0 and the score is 0.9 * MCC_model."""
    rng = np.random.default_rng(SEED)
    index = pair_index(SIZE)
    truth = random_partition(rng)
    model = truth.model.copy()
    guess = Partition(np.zeros(SIZE, dtype=np.int64), model)
    states = guess.states(index)
    cells = cross_tabulate(truth.states(index), states)
    exact = official_score(as_labeling(truth), as_labeling(guess)).score
    assert float(full_score(truth.states(index), states)) == pytest.approx(exact, abs=1e-12)
    assert exact <= 0.9 + 1e-12
    assert cells.sum() == index.count


def test_thinning_keeps_expected_counts_and_full_fraction_is_identity() -> None:
    rng = np.random.default_rng(SEED)
    cells = np.array([30.0, 5.0, 0.0, 12.0, 40.0, 9.0, 3.0, 11.0, 20.0])
    np.testing.assert_array_equal(thin_cells(cells, 1.0, 3, rng)[0], cells)
    thinned = thin_cells(cells, 0.3, 4000, rng)
    np.testing.assert_allclose(thinned.mean(axis=0), cells * 0.3, rtol=0.05, atol=0.05)


def test_public_score_concentrates_on_full_score_for_large_counts() -> None:
    rng = np.random.default_rng(SEED)
    index = pair_index(33)
    truth = Partition.from_groups([list(range(12)), list(range(12, 24)), list(range(24, 33))])
    guess = Partition.from_groups([list(range(12)), list(range(12, 33))])
    cells = cross_tabulate(truth.states(index), guess.states(index))
    exact = float(full_score(truth.states(index), guess.states(index)))
    draws = public_score_draws(cells, 2000, rng, public_fraction=0.3)
    assert abs(float(draws.mean()) - exact) < 0.05
    assert float(draws.std()) > 0.0


def test_quantize_rounds_to_the_step() -> None:
    values = np.array([0.0, 0.024, 0.026, 0.49])
    np.testing.assert_allclose(quantize(values, 0.05), [0.0, 0.0, 0.05, 0.5])
    np.testing.assert_array_equal(quantize(values, None), values)


def test_partition_validation_and_rows() -> None:
    with pytest.raises(PartitionError):
        Partition(np.array([0, 1]), np.array([0, 0]))
    partition = Partition.from_labels(["A", "A", "B"], ["x", "y", "x"])
    assert partition.model.tolist() == [0, 1, 2]  # model "x" of B is not model "x" of A
    rows = Partition.from_groups([[0, 2], [1]], family_of_group=[0, 0]).to_rows()
    assert rows == [("ep-01", "F1", "F1M1"), ("ep-02", "F1", "F1M2"), ("ep-03", "F1", "F1M1")]
    assert canonical([5, 5, 2, 9, 2]).tolist() == [0, 0, 1, 2, 1]
