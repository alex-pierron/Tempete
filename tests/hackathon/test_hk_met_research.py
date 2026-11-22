"""Tests of the team-A research code (offline, synthetic data only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hackathon.model_equality.prompts import expand_phase, get_phase
from hackathon.model_equality.research.dataset import (
    ReferenceTruth,
    ResearchDataError,
    load_truth,
)
from hackathon.model_equality.research.evaluate import (
    METHOD_UNIFORM,
    auc_score,
    evaluate_loeo,
    mcc_of,
    pairs_of,
    pairwise_threshold_mcc,
)
from hackathon.model_equality.research.extra_stats import (
    EXTRA_KERNELS,
    char_trigram_gram,
    extra_channels,
    lcp_block,
    pooled_channel,
    scalar_pair_z,
    within_prompt_rank,
)
from hackathon.model_equality.research.pipeline import balanced_units, prompts_up_to
from hackathon.model_equality.simulate import reference_endpoints, simulate_answers
from hackathon.model_equality.store import Answer

PRACTICE_CSV = (
    "endpoint,family,model,kind\n"
    "ref-01,P,S,plain\nref-02,P,T1,quantized (fp8)\nref-03,P,S,system prompt\n"
    "ref-04,P,T3,LoRA at strength 0.7\nref-05,P,S,persona\nref-06,P,S,watermark\n"
    "ref-07,P,T4,LoRA at strength 1.5\nref-08,P,S,sampling at temperature 1.0\n"
    "ref-09,P,T2,LoRA at strength 0.3\n"
)
PERMUTATIONS = 60
SEED = 11


@pytest.fixture
def truth(tmp_path: Path) -> ReferenceTruth:
    path = tmp_path / "practice.csv"
    path.write_text(PRACTICE_CSV, encoding="utf-8")
    return load_truth(path)


def test_load_truth_reads_nine_endpoints(truth: ReferenceTruth) -> None:
    assert len(truth.endpoints) == 9
    assert sum(1 for e in truth.endpoints if truth.model[e] == "S") == 5
    assert truth.short_name("ref-05") == "ref-05 S/persona"


def test_load_truth_rejects_a_file_without_reference_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("endpoint,family,model,kind\nep-01,A,A1,x\n", encoding="utf-8")
    with pytest.raises(ResearchDataError):
        load_truth(path)


def test_auc_score_known_values() -> None:
    assert auc_score([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auc_score([1.0], [1.0]) == 0.5
    assert np.isnan(auc_score([], [1.0]))


def test_mcc_of_perfect_and_degenerate() -> None:
    assert mcc_of([True, False, True, False], [True, False, True, False]) == 1.0
    assert mcc_of([True, True, False], [True, True, True]) == 0.0


def test_scalar_pair_z_detects_a_shift_and_stays_small_under_h0() -> None:
    rng = np.random.default_rng(SEED)
    prompts = np.repeat(np.arange(20), 4)
    rows_a = np.flatnonzero(np.tile([True, True, False, False], 20))
    rows_b = np.flatnonzero(np.tile([False, False, True, True], 20))
    same = rng.normal(size=len(prompts))
    shifted = same + np.where(np.isin(np.arange(len(prompts)), rows_a), 2.0, 0.0)
    z_null = scalar_pair_z(same, prompts, rows_a, rows_b, PERMUTATIONS, np.random.default_rng(1))
    z_shift = scalar_pair_z(shifted, prompts, rows_a, rows_b, PERMUTATIONS, np.random.default_rng(1))
    assert z_shift > 4.0
    assert z_null < z_shift


def test_scalar_pair_z_is_nan_without_shared_prompt() -> None:
    prompts = np.array([0, 0, 1, 1])
    value = scalar_pair_z(np.arange(4.0), prompts, np.array([0, 1]), np.array([2, 3]), 20,
                          np.random.default_rng(1))
    assert np.isnan(value)


def test_within_prompt_rank_is_a_uniform_grid() -> None:
    ranks = within_prompt_rank(np.array([5.0, 1.0, 3.0, 9.0]), np.zeros(4, dtype=np.int64))
    assert sorted(ranks.tolist()) == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_extra_channels_are_symmetric_and_pool(truth: ReferenceTruth) -> None:
    answers = simulate_answers(reference_endpoints(), expand_phase(get_phase("ref-1")))
    channels = extra_channels(answers, list(truth.endpoints), PERMUTATIONS)
    assert {kernel for _, kernel in channels} <= set(EXTRA_KERNELS)
    for matrix in channels.values():
        finite = np.isfinite(matrix)
        assert np.array_equal(finite, finite.T)
        assert np.allclose(matrix[finite], matrix.T[finite])
    pooled = pooled_channel(channels, "exact_text")
    assert pooled is not None
    assert pooled.shape == (9, 9)


def test_pooled_channel_of_unknown_kernel_is_none() -> None:
    assert pooled_channel({}, "nope") is None


def test_loeo_is_perfect_on_perfectly_separated_scores(truth: ReferenceTruth) -> None:
    size = len(truth.endpoints)
    score = np.zeros((size, size))
    for i, j, same in pairs_of(truth):
        score[i, j] = score[j, i] = 0.2 if same else 9.0
    result = evaluate_loeo({("all", "oracle"): score}, truth, METHOD_UNIFORM)
    assert result.mcc == pytest.approx(1.0)
    assert result.wrong_pairs == []


def test_lcp_block_counts_the_common_prefix() -> None:
    chars = np.array([[1, 2, 3, 4], [1, 2, 9, 4], [7, 2, 3, 4]], dtype=np.int64)
    block = lcp_block(chars)
    assert block[0, 1] == pytest.approx(2 / 4)
    assert block[0, 2] == 0.0
    assert block[0, 0] == 1.0


def test_char_trigram_gram_is_zero_across_prompts_and_on_the_diagonal() -> None:
    gram = char_trigram_gram(["hello world", "hello word", "hello world"],
                             np.array([0, 0, 1], dtype=np.int64))
    assert gram[0, 1] > 0.5
    assert gram[0, 2] == 0.0
    assert np.all(np.diag(gram) == 0.0)


def test_prompts_up_to_stops_at_the_target() -> None:
    answers = [Answer(e, p, r, "q", "a") for e in ("ref-01", "ref-02")
               for p in ("p1", "p2", "p3") for r in (0, 1)]
    chosen, actual = prompts_up_to(answers, ["p3", "p1", "p2"], 4)
    assert chosen == {"p3", "p1"}
    assert actual == 4


def test_balanced_units_drops_units_missing_an_endpoint() -> None:
    answers = [Answer("ref-01", "p1", 0, "q", "a"), Answer("ref-02", "p1", 0, "q", "a"),
               Answer("ref-01", "p2", 0, "q", "a")]
    kept, dropped = balanced_units(answers, ["ref-01", "ref-02"])
    assert len(kept) == 2
    assert dropped == 1


def test_pairwise_threshold_mcc_on_oracle_scores(truth: ReferenceTruth) -> None:
    size = len(truth.endpoints)
    score = np.zeros((size, size))
    for i, j, same in pairs_of(truth):
        score[i, j] = score[j, i] = 0.2 if same else 9.0
    mcc, tau = pairwise_threshold_mcc(score, truth)
    assert mcc == pytest.approx(1.0)
    assert 0.2 < tau < 9.0
