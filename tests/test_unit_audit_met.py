"""Unit tests (QA pass) of `wmkit.audit.met`: hand-computed statistics, invariances and calibration.

Complements `tests/test_met.py`, whose oracle for the statistic is a transcription of the same
definition. Here the oracle is arithmetic done by hand (tiny samples) and structural
invariances (relabelling, reordering, truncation) that any correct implementation must obey.

Note on the brief: "MMD is zero for identical samples" holds for *constant* samples only. The
estimator is the unbiased U-statistic of the official code (diagonal excluded inside P-P and
Q-Q, kept in P-Q), so feeding the same list as P and Q gives `2 (A - 1) / n <= 0` for one
prompt (A = mean off-diagonal kernel); the tests below pin that closed form instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from wmkit.audit import METError, Sample, mmd_hamming, power_curve, two_sample_test
from wmkit.audit.met import PAD_TOKEN_ID

FloatArray = npt.NDArray[np.float64]

CALIBRATION_TRIALS = 250
CALIBRATION_RESAMPLES = 99
CALIBRATION_SEED = 20260317
COMPLETION_LEN = 6
VOCAB = 7
N_PROMPTS = 4
ALPHA = 0.05
BAND_AT_ALPHA = (0.02, 0.09)
"""Type-I error acceptance band at 5 % over 250 trials (binomial sd = 1.4 %: about -2 / +3 sd)."""
BAND_AT_TWENTY = (0.14, 0.27)
BAND_AT_HALF = (0.42, 0.58)
SIM_BAND_AT_ALPHA = (0.01, 0.10)
SIM_BAND_AT_TWENTY = (0.12, 0.28)
SIM_BAND_AT_HALF = (0.40, 0.62)
SIMULATION_POOL_MARGIN = 300
"""Extra reference samples per prompt: a small pool adds its own sampling noise to the level."""
P_PER_PROMPT = (6, 14, 9, 3)
Q_PER_PROMPT = (11, 4, 8, 10)
POWER_PER_PROMPT = (30, 30, 30, 30)
"""Deliberately unbalanced designs: the stratified permutation must stay exact."""


def samples_of(prompt: int, *completions: tuple[int, ...]) -> list[Sample]:
    """Build the samples of one prompt."""
    return [Sample(prompt, tuple(completion)) for completion in completions]


def mean_off_diagonal_kernel(group: Sequence[Sample], completion_len: int) -> float:
    """Mean of k(y, y') over ordered pairs i != j of ONE prompt (own arithmetic, no padding)."""
    total, pairs = 0.0, 0
    for i, a in enumerate(group):
        for j, b in enumerate(group):
            if i != j:
                total += sum(
                    x == y for x, y in zip(a.completion_ids, b.completion_ids, strict=True)
                ) / completion_len
                pairs += 1
    return total / pairs


def ragged_law(rng: np.random.Generator) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Per-(prompt, position) categorical laws and per-prompt maximal completion length."""
    laws = rng.dirichlet(np.full(VOCAB, 0.8), size=(N_PROMPTS, COMPLETION_LEN))
    lengths = rng.integers(2, COMPLETION_LEN + 1, size=N_PROMPTS)
    return laws, lengths


def draw_ragged(
    laws: FloatArray,
    lengths: npt.NDArray[np.int64],
    per_prompt: Sequence[int],
    rng: np.random.Generator,
) -> list[Sample]:
    """Draw completions whose length varies (stop early with probability 0.3 per position)."""
    samples: list[Sample] = []
    for prompt, count in enumerate(per_prompt):
        for _ in range(count):
            tokens: list[int] = []
            for position in range(int(lengths[prompt])):
                tokens.append(int(rng.choice(VOCAB, p=laws[prompt, position])))
                if rng.random() < 0.3:
                    break
            samples.append(Sample(prompt, tuple(tokens)))
    return samples


# ---------------------------------------------------------------------------
# mmd_hamming: arithmetic done by hand
# ---------------------------------------------------------------------------


class TestHandComputedStatistic:
    """Each expected value is derived on paper in the comment above the assertion."""

    def test_padding_positions_match_each_other(self) -> None:
        # L=4, P = {(1,), (1,)}, Q = {(2,), (2,)}; padded rows are [1,-1,-1,-1] and [2,-1,-1,-1].
        # K_PP = K_QQ = 4/4 = 1 (same token + 3 paddings); K_PQ = 3/4 (only the 3 paddings agree).
        # MMD = 1 - 2 * 0.75 + 1 = 0.5.
        p, q = samples_of(0, (1,), (1,)), samples_of(0, (2,), (2,))
        assert mmd_hamming(p, q, completion_len=4) == pytest.approx(0.5, abs=1e-15)
        # L=1: no padding at all, K_PP = K_QQ = 1, K_PQ = 0 -> MMD = 2.
        assert mmd_hamming(p, q, completion_len=1) == pytest.approx(2.0, abs=1e-15)

    def test_pad_token_is_minus_one_and_cannot_be_a_real_token(self) -> None:
        assert PAD_TOKEN_ID == -1
        with pytest.raises(METError, match="padding"):
            mmd_hamming(samples_of(0, (1, PAD_TOKEN_ID)), samples_of(0, (1, 2)))

    def test_tokens_beyond_completion_len_are_ignored(self) -> None:
        # With L=2 the third token is cut: both samples collapse to identical rows [1, 2],
        # every kernel value is 1 and MMD = 1 - 2 + 1 = 0.
        p = samples_of(0, (1, 2, 9), (1, 2, 9))
        q = samples_of(0, (1, 2, 7), (1, 2, 7))
        assert mmd_hamming(p, q, completion_len=2) == pytest.approx(0.0, abs=1e-15)
        # With L=3 the third token counts: K_PP = K_QQ = 1, K_PQ = 2/3 -> MMD = 2 - 4/3 = 2/3.
        assert mmd_hamming(p, q, completion_len=3) == pytest.approx(2.0 / 3.0, abs=1e-15)

    def test_constant_samples_give_exactly_zero(self) -> None:
        for n_prompts in (1, 3):
            p = [s for prompt in range(n_prompts) for s in samples_of(prompt, (5, 5, 5), (5, 5, 5))]
            q = [s for prompt in range(n_prompts) for s in samples_of(prompt, (5, 5, 5), (5, 5, 5))]
            assert mmd_hamming(p, q, completion_len=3) == pytest.approx(0.0, abs=1e-15)

    def test_same_list_as_p_and_q_gives_two_a_minus_one_over_n(self) -> None:
        # One prompt, n = 4 samples, P = Q. K_PP = K_QQ = A; K_PQ keeps the diagonal:
        # (n (n-1) A + n * 1) / n^2. MMD = 2A - 2 ((n-1) A + 1) / n = 2 (A - 1) / n.
        group = samples_of(0, (1, 1), (1, 2), (2, 2), (3, 3))
        mean_a = mean_off_diagonal_kernel(group, completion_len=2)
        expected = 2.0 * (mean_a - 1.0) / len(group)
        assert mmd_hamming(group, list(group), completion_len=2) == pytest.approx(
            expected, abs=1e-12
        )
        assert expected < 0.0

    def test_samples_of_disjoint_prompts_never_compare_across_prompts(self) -> None:
        # No same-prompt P-Q pair exists: the cross block is 0, so MMD = A_P + A_Q.
        p = samples_of(0, (1, 2), (1, 2))
        q = samples_of(1, (1, 2), (1, 2))
        assert mmd_hamming(p, q, completion_len=2) == pytest.approx(2.0, abs=1e-15)

    def test_two_prompts_average_blocks_over_same_prompt_pairs_only(self) -> None:
        # Prompt 0: P={(1,1)x2}, Q={(2,2)x2}. Prompt 1: P={(3,3)x2}, Q={(3,3)x2}.
        # K_PP pairs (ordered, off-diagonal): 2 per prompt, all value 1 -> mean 1.
        # K_QQ likewise 1. K_PQ pairs: 4 per prompt: prompt 0 -> 0, prompt 1 -> 1; mean 0.5.
        # MMD = 1 - 2 * 0.5 + 1 = 1.
        p = samples_of(0, (1, 1), (1, 1)) + samples_of(1, (3, 3), (3, 3))
        q = samples_of(0, (2, 2), (2, 2)) + samples_of(1, (3, 3), (3, 3))
        assert mmd_hamming(p, q, completion_len=2) == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------------------------
# mmd_hamming: invariances
# ---------------------------------------------------------------------------


class TestStatisticInvariances:
    """Properties that hold for any data; seeded loops stand in for hypothesis."""

    @pytest.mark.parametrize("seed", range(8))
    def test_symmetric_and_independent_of_sample_order(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (4, 5, 3, 6), rng)
        q = draw_ragged(laws, lengths, (5, 2, 6, 4), rng)
        reference = mmd_hamming(p, q, completion_len=COMPLETION_LEN)
        assert mmd_hamming(q, p, completion_len=COMPLETION_LEN) == pytest.approx(reference, abs=1e-12)
        shuffled_p = [p[i] for i in rng.permutation(len(p))]
        shuffled_q = [q[i] for i in rng.permutation(len(q))]
        assert mmd_hamming(shuffled_p, shuffled_q, completion_len=COMPLETION_LEN) == pytest.approx(
            reference, abs=1e-12
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_invariant_to_relabelling_tokens_and_prompt_ids(self, seed: int) -> None:
        rng = np.random.default_rng(100 + seed)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (4, 5, 3, 6), rng)
        q = draw_ragged(laws, lengths, (5, 2, 6, 4), rng)
        token_map = rng.permutation(VOCAB) + 1000  # any injective map on the vocabulary
        prompt_map = {prompt: int(10 * prompt + 7) for prompt in range(N_PROMPTS)}

        def relabel(samples: Sequence[Sample]) -> list[Sample]:
            return [
                Sample(prompt_map[s.prompt_id], tuple(int(token_map[t]) for t in s.completion_ids))
                for s in samples
            ]

        assert mmd_hamming(relabel(p), relabel(q), completion_len=COMPLETION_LEN) == pytest.approx(
            mmd_hamming(p, q, completion_len=COMPLETION_LEN), abs=1e-12
        )

    def test_a_completely_different_vocabulary_on_each_side_maximises_the_distance(self) -> None:
        p = samples_of(0, *[(1, 2, 3)] * 5)
        q = samples_of(0, *[(4, 5, 6)] * 5)
        assert mmd_hamming(p, q, completion_len=3) == pytest.approx(2.0, abs=1e-12)

    def test_statistic_is_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (4, 5, 3, 6), rng)
        q = draw_ragged(laws, lengths, (5, 2, 6, 4), rng)
        assert mmd_hamming(p, q, completion_len=COMPLETION_LEN) == mmd_hamming(
            p, q, completion_len=COMPLETION_LEN
        )

    def test_a_single_sample_per_prompt_does_not_divide_by_zero(self) -> None:
        """No same-prompt P-P pair exists: the block mean falls back to 0 instead of NaN."""
        value = mmd_hamming(samples_of(0, (1, 2)), samples_of(0, (1, 3)), completion_len=2)
        # Blocks: K_PP = K_QQ = 0 (no pair), K_PQ = 1/2 -> MMD = -1.
        assert value == pytest.approx(-1.0, abs=1e-15)


# ---------------------------------------------------------------------------
# two_sample_test: structure of the p-value
# ---------------------------------------------------------------------------


class TestPValueStructure:
    """`p = (1 + #{T_i >= T_obs}) / (B + 1)` lives on a fixed grid."""

    @pytest.mark.parametrize("n_resamples", (1, 9, 49))
    def test_pvalue_lies_on_the_grid_k_over_b_plus_one(self, n_resamples: int) -> None:
        rng = np.random.default_rng(n_resamples)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (4, 4, 4, 4), rng)
        q = draw_ragged(laws, lengths, (4, 4, 4, 4), rng)
        result = two_sample_test(p, q, completion_len=COMPLETION_LEN, n_resamples=n_resamples)
        scaled = result.p_value * (n_resamples + 1)
        assert scaled == pytest.approx(round(scaled), abs=1e-9)
        assert 1.0 / (n_resamples + 1) - 1e-12 <= result.p_value <= 1.0

    def test_statistic_reported_equals_mmd_hamming_and_sizes_are_echoed(self) -> None:
        rng = np.random.default_rng(1)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (3, 4, 5, 6), rng)
        q = draw_ragged(laws, lengths, (6, 5, 4, 3), rng)
        result = two_sample_test(p, q, completion_len=COMPLETION_LEN, n_resamples=20)
        assert result.statistic == pytest.approx(mmd_hamming(p, q, completion_len=COMPLETION_LEN))
        assert (result.n_p, result.n_q) == (len(p), len(q))

    def test_seed_changes_the_resampling_but_not_the_statistic(self) -> None:
        rng = np.random.default_rng(2)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (5, 5, 5, 5), rng)
        q = draw_ragged(laws, lengths, (5, 5, 5, 5), rng)
        results = [
            two_sample_test(p, q, completion_len=COMPLETION_LEN, n_resamples=40, seed=seed)
            for seed in range(6)
        ]
        assert len({result.statistic for result in results}) == 1
        assert len({result.p_value for result in results}) > 1

    def test_pvalue_is_invariant_to_the_order_of_the_samples_inside_each_list(self) -> None:
        """Permutation tests condition on the pooled multiset; the draw order must not matter
        for the statistic (the p-value may differ through the rng stream, so compare T)."""
        rng = np.random.default_rng(4)
        laws, lengths = ragged_law(rng)
        p = draw_ragged(laws, lengths, (5, 5, 5, 5), rng)
        q = draw_ragged(laws, lengths, (5, 5, 5, 5), rng)
        reordered = two_sample_test(
            [p[i] for i in rng.permutation(len(p))], q, completion_len=COMPLETION_LEN, n_resamples=10
        )
        original = two_sample_test(p, q, completion_len=COMPLETION_LEN, n_resamples=10)
        assert reordered.statistic == pytest.approx(original.statistic, abs=1e-12)


# ---------------------------------------------------------------------------
# two_sample_test: calibration under H0 with unbalanced designs and ragged completions
# ---------------------------------------------------------------------------


class TestCalibrationUnderNull:
    """Type-I error of the permutation and simulation p-values on a toy law with padding."""

    @staticmethod
    def rejection_rates(p_values: Sequence[float]) -> tuple[float, float, float]:
        array = np.asarray(p_values)
        return (
            float(np.mean(array <= ALPHA)),
            float(np.mean(array <= 0.2)),
            float(np.mean(array <= 0.5)),
        )

    @pytest.mark.parametrize(
        "stratify",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="F-MET-1: the global (official) permutation mixes prompts, so with "
                    "unbalanced P/Q prompt designs the p-value is not uniform under H0",
                ),
            ),
        ],
    )
    def test_permutation_pvalue_is_calibrated_with_unbalanced_prompts_and_padding(
        self, stratify: bool
    ) -> None:
        rng = np.random.default_rng(CALIBRATION_SEED)
        laws, lengths = ragged_law(rng)
        p_values = [
            two_sample_test(
                draw_ragged(laws, lengths, P_PER_PROMPT, rng),
                draw_ragged(laws, lengths, Q_PER_PROMPT, rng),
                completion_len=COMPLETION_LEN,
                n_resamples=CALIBRATION_RESAMPLES,
                seed=trial,
                stratify_by_prompt=stratify,
            ).p_value
            for trial in range(CALIBRATION_TRIALS)
        ]
        at_alpha, at_twenty, at_half = self.rejection_rates(p_values)
        assert BAND_AT_ALPHA[0] <= at_alpha <= BAND_AT_ALPHA[1], at_alpha
        assert BAND_AT_TWENTY[0] <= at_twenty <= BAND_AT_TWENTY[1], at_twenty
        assert BAND_AT_HALF[0] <= at_half <= BAND_AT_HALF[1], at_half

    def test_simulation_pvalue_is_calibrated_with_unbalanced_prompts_and_padding(self) -> None:
        rng = np.random.default_rng(CALIBRATION_SEED + 100)
        laws, lengths = ragged_law(rng)
        pool_per_prompt = tuple(p + q + SIMULATION_POOL_MARGIN for p, q in zip(P_PER_PROMPT, Q_PER_PROMPT, strict=True))
        pool = draw_ragged(laws, lengths, pool_per_prompt, rng)
        p_values = [
            two_sample_test(
                draw_ragged(laws, lengths, P_PER_PROMPT, rng),
                draw_ragged(laws, lengths, Q_PER_PROMPT, rng),
                completion_len=COMPLETION_LEN,
                pvalue="simulation",
                reference_pool=pool,
                n_resamples=CALIBRATION_RESAMPLES,
                seed=trial,
            ).p_value
            for trial in range(CALIBRATION_TRIALS)
        ]
        at_alpha, at_twenty, at_half = self.rejection_rates(p_values)
        # Wider bands than the permutation test: the shared finite pool adds a random offset to
        # the level (observed across pool seeds: 0.016-0.076 / 0.17-0.22 / 0.44-0.59).
        assert SIM_BAND_AT_ALPHA[0] <= at_alpha <= SIM_BAND_AT_ALPHA[1], at_alpha
        assert SIM_BAND_AT_TWENTY[0] <= at_twenty <= SIM_BAND_AT_TWENTY[1], at_twenty
        assert SIM_BAND_AT_HALF[0] <= at_half <= SIM_BAND_AT_HALF[1], at_half

    def test_power_against_a_shifted_law_with_the_same_designs(self) -> None:
        """Guards against a test that never rejects: a different law must be caught."""
        rng = np.random.default_rng(CALIBRATION_SEED + 11)
        laws_p, lengths = ragged_law(rng)
        laws_q = rng.dirichlet(np.full(VOCAB, 0.2), size=(N_PROMPTS, COMPLETION_LEN))
        rejections = sum(
            two_sample_test(
                draw_ragged(laws_p, lengths, POWER_PER_PROMPT, rng),
                draw_ragged(laws_q, lengths, POWER_PER_PROMPT, rng),
                completion_len=COMPLETION_LEN,
                n_resamples=CALIBRATION_RESAMPLES,
                seed=trial,
            ).p_value
            <= ALPHA
            for trial in range(40)
        )
        assert rejections >= 36


# ---------------------------------------------------------------------------
# power_curve
# ---------------------------------------------------------------------------


class TestPowerCurveUnit:
    """Bookkeeping of the Monte Carlo loop (cheap settings)."""

    @staticmethod
    def draw(budget: int, rng: np.random.Generator) -> list[Sample]:
        return [Sample(i % 2, tuple(int(t) for t in rng.integers(0, 3, size=4))) for i in range(budget)]

    def test_equal_seed_gives_the_equal_curve_and_rates_are_multiples_of_one_over_trials(self) -> None:
        kwargs = {"n_trials": 8, "seed": 3, "completion_len": 4, "n_resamples": 19}
        first = power_curve(self.draw, self.draw, [6, 10], **kwargs)  # type: ignore[arg-type]
        assert first == power_curve(self.draw, self.draw, [6, 10], **kwargs)  # type: ignore[arg-type]
        assert [budget for budget, _ in first] == [6, 10]
        assert all(abs(rate * 8 - round(rate * 8)) < 1e-12 for _, rate in first)
        other = power_curve(self.draw, self.draw, [6, 10], **{**kwargs, "seed": 4})  # type: ignore[arg-type]
        assert isinstance(other, list)

    def test_empty_budget_list_gives_an_empty_curve(self) -> None:
        assert power_curve(self.draw, self.draw, [], n_trials=1, seed=0) == []

    def test_an_obvious_alternative_is_rejected_at_every_trial(self) -> None:
        def constant_one(budget: int, rng: np.random.Generator) -> list[Sample]:
            return [Sample(0, (1,) * 4)] * budget

        def constant_two(budget: int, rng: np.random.Generator) -> list[Sample]:
            return [Sample(0, (2,) * 4)] * budget

        curve = power_curve(
            constant_one, constant_two, [10], n_trials=5, seed=0, completion_len=4, n_resamples=99
        )
        assert curve == [(10, 1.0)]
