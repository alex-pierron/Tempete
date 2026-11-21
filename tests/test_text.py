"""Tests for wmkit.text.

Fast tests use a fake language model (a random bigram table over a small vocabulary), so no
weights are downloaded. Tests that load SmolLM2 are marked `slow`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.special import gammaincc
from scipy.stats import binom

from wmkit.text import (
    TEXT_DETECTION_P,
    AttackRateError,
    GenerationError,
    GumbelConfig,
    GumbelSampler,
    KGWConfig,
    KGWLogitsProcessor,
    WatermarkConfigError,
    delete_tokens,
    generate,
    green_mask,
    gumbel_detect,
    insert_tokens,
    kgw_detect,
    localize_segments,
    sample_sequences,
    secret_vector,
    segments_iou,
    substitute_tokens,
    token_scores,
)
from wmkit.text.gumbel import gumbel_pvalue
from wmkit.text.kgw import kgw_pvalue

FAKE_VOCAB = 2000
FAKE_LOGIT_SCALE = 2.0
FAKE_LM_SEED = 11
TEXT_TOKENS = 200
N_WATERMARKED_TEXTS = 30
N_PLAIN_TEXTS = 60
KGW_CONFIG = KGWConfig(gamma=0.25, delta=2.0, hash_key=424243)
GUMBEL_CONFIG = GumbelConfig(hash_key=777)
H0_TRIALS = 400
LOCALIZE_SEGMENT = 100
MIN_LOCALIZATION_IOU = 0.8


def make_fake_lm() -> torch.Tensor:
    """Random bigram logits `[V, V]`: next-token logits depend on the last token only."""
    generator = torch.Generator().manual_seed(FAKE_LM_SEED)
    return torch.randn(FAKE_VOCAB, FAKE_VOCAB, generator=generator) * FAKE_LOGIT_SCALE


FAKE_TABLE = make_fake_lm()


def fake_next_logits(prefix: torch.Tensor) -> torch.Tensor:
    """`NextLogitsFn` of the fake bigram LM."""
    return FAKE_TABLE[prefix[:, -1]]


def fake_text(
    scheme: str,
    config: KGWConfig | GumbelConfig | None,
    index: int,
    length: int = TEXT_TOKENS,
) -> list[int]:
    """Generate one fake-LM text; the prompt token depends on `index` (Gumbel is deterministic)."""
    return sample_sequences(
        fake_next_logits,
        [(3 + 7 * index) % FAKE_VOCAB],
        n_sequences=1,
        scheme=scheme,  # type: ignore[arg-type]
        config=config,
        max_new_tokens=length,
        seed=index,
    )[0]


def random_text(rng: np.random.Generator, length: int = TEXT_TOKENS) -> list[int]:
    """I.i.d. uniform tokens: a text with no relation to any key."""
    return [int(token) for token in rng.integers(0, FAKE_VOCAB, size=length)]


class TestKGW:
    """Green lists, the logits processor and exact detection."""

    def test_green_list_matches_transformers_lefthash(self) -> None:
        from transformers import WatermarkLogitsProcessor

        config = KGWConfig(gamma=0.25, delta=2.0)
        reference = WatermarkLogitsProcessor(
            vocab_size=FAKE_VOCAB, device="cpu", greenlist_ratio=0.25, bias=2.0
        )
        ours = KGWLogitsProcessor(FAKE_VOCAB, config)
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, FAKE_VOCAB, (4, 9), generator=generator)
        scores = torch.randn(4, FAKE_VOCAB, generator=generator)
        assert torch.equal(ours(input_ids, scores), reference(input_ids, scores))  # type: ignore[arg-type]

    def test_green_list_has_exact_size_and_is_deterministic(self) -> None:
        mask = green_mask([5], FAKE_VOCAB, KGW_CONFIG)
        assert int(mask.sum()) == int(FAKE_VOCAB * KGW_CONFIG.gamma)
        assert torch.equal(mask, green_mask([5], FAKE_VOCAB, KGW_CONFIG))
        assert not torch.equal(mask, green_mask([6], FAKE_VOCAB, KGW_CONFIG))
        other_key = KGWConfig(hash_key=KGW_CONFIG.hash_key + 1)
        assert not torch.equal(mask, green_mask([5], FAKE_VOCAB, other_key))

    def test_processor_leaves_padding_columns_and_short_prefixes_alone(self) -> None:
        processor = KGWLogitsProcessor(FAKE_VOCAB, KGW_CONFIG)
        wide = torch.zeros(1, FAKE_VOCAB + 5)
        out = processor(torch.tensor([[1, 2]]), wide)  # type: ignore[arg-type]
        assert torch.equal(out[0, FAKE_VOCAB:], torch.zeros(5))
        assert out[0, :FAKE_VOCAB].max() == KGW_CONFIG.delta
        width_two = KGWLogitsProcessor(FAKE_VOCAB, KGWConfig(context_width=2))
        short = torch.zeros(1, FAKE_VOCAB)
        assert torch.equal(width_two(torch.tensor([[1]]), short), short)  # type: ignore[arg-type]

    def test_detection_matches_manual_binomial_and_z(self) -> None:
        tokens = fake_text("kgw", KGW_CONFIG, 0)
        detection = kgw_detect(tokens, FAKE_VOCAB, KGW_CONFIG)
        manual_green = sum(
            bool(green_mask([tokens[t - 1]], FAKE_VOCAB, KGW_CONFIG)[tokens[t]])
            for t in range(1, len(tokens))
        )
        no_dedup = kgw_detect(tokens, FAKE_VOCAB, KGW_CONFIG, dedup_ngrams=False)
        assert no_dedup.n_scored == len(tokens) - 1 and no_dedup.n_green == manual_green
        gamma = int(FAKE_VOCAB * 0.25) / FAKE_VOCAB
        assert no_dedup.p_value == pytest.approx(
            float(binom.sf(manual_green - 1, no_dedup.n_scored, gamma)), rel=1e-9
        )
        n = no_dedup.n_scored
        assert no_dedup.z_score == pytest.approx(
            (manual_green - gamma * n) / math.sqrt(n * gamma * (1 - gamma))
        )
        assert detection.n_scored <= no_dedup.n_scored

    def test_dedup_scores_each_context_token_pair_once(self) -> None:
        repeated = [1, 2] * 50
        assert kgw_detect(repeated, FAKE_VOCAB, KGW_CONFIG).n_scored == 2
        assert kgw_detect(repeated, FAKE_VOCAB, KGW_CONFIG, dedup_ngrams=False).n_scored == 99

    def test_degenerate_inputs(self) -> None:
        empty = kgw_detect([], FAKE_VOCAB, KGW_CONFIG)
        assert (empty.n_scored, empty.p_value, empty.detected, empty.z_score) == (
            0,
            1.0,
            False,
            None,
        )
        single = kgw_detect([4], FAKE_VOCAB, KGW_CONFIG)
        assert single.n_scored == 0 and not single.detected
        with pytest.raises(WatermarkConfigError):
            kgw_detect([FAKE_VOCAB], FAKE_VOCAB, KGW_CONFIG)
        with pytest.raises(WatermarkConfigError):
            kgw_detect([-1, 2], FAKE_VOCAB, KGW_CONFIG)

    def test_pvalue_log_is_finite_below_float_underflow(self) -> None:
        p_value, log10_p = kgw_pvalue(2000, 2000, 0.25)
        assert p_value == 0.0 and log10_p < -1000.0
        assert kgw_pvalue(0, 0, 0.25) == (1.0, 0.0)

    def test_config_validation(self) -> None:
        for bad in (0.0, 1.0, -0.1):
            with pytest.raises(WatermarkConfigError):
                KGWConfig(gamma=bad)
        with pytest.raises(WatermarkConfigError):
            KGWConfig(context_width=0)
        with pytest.raises(WatermarkConfigError):
            KGWConfig(seeding="selfhash")  # type: ignore[arg-type]

    def test_false_positive_rate_under_h0_is_at_most_nominal(self) -> None:
        rng = np.random.default_rng(1)
        p_values = [
            kgw_detect(random_text(rng), FAKE_VOCAB, KGW_CONFIG).p_value for _ in range(H0_TRIALS)
        ]
        assert float(np.mean(np.array(p_values) <= 0.05)) <= 0.09
        assert sum(p < TEXT_DETECTION_P for p in p_values) <= 2

    def test_watermarked_texts_are_detected_and_plain_ones_are_not(self) -> None:
        detected = [
            kgw_detect(fake_text("kgw", KGW_CONFIG, i), FAKE_VOCAB, KGW_CONFIG).detected
            for i in range(N_WATERMARKED_TEXTS)
        ]
        false_alarms = [
            kgw_detect(fake_text("none", None, i), FAKE_VOCAB, KGW_CONFIG).detected
            for i in range(N_PLAIN_TEXTS)
        ]
        assert all(detected)
        assert sum(false_alarms) / N_PLAIN_TEXTS <= 0.01

    def test_wrong_key_does_not_detect(self) -> None:
        wrong = KGWConfig(hash_key=KGW_CONFIG.hash_key + 99)
        results = [
            kgw_detect(fake_text("kgw", KGW_CONFIG, i), FAKE_VOCAB, wrong).detected
            for i in range(N_WATERMARKED_TEXTS)
        ]
        assert not any(results)

    def test_generation_is_reproducible_per_seed(self) -> None:
        first = fake_text("kgw", KGW_CONFIG, 3, length=50)
        assert first == fake_text("kgw", KGW_CONFIG, 3, length=50)
        other = sample_sequences(
            fake_next_logits,
            [3 + 21],
            n_sequences=1,
            scheme="kgw",
            config=KGW_CONFIG,
            max_new_tokens=50,
            seed=99,
        )[0]
        assert other != first


class TestGumbel:
    """Gumbel-max sampling and the Gamma test."""

    def test_pvalue_is_the_regularized_upper_gamma(self) -> None:
        for n_scored, total in ((50, 70.0), (200, 260.0), (10, 3.0)):
            p_value, log10_p = gumbel_pvalue(total, n_scored)
            assert p_value == pytest.approx(float(gammaincc(n_scored, total)), rel=1e-12)
            assert log10_p == pytest.approx(math.log10(p_value), rel=1e-9)

    def test_log_pvalue_stays_finite_when_the_float_underflows(self) -> None:
        p_value, log10_p = gumbel_pvalue(1500.0, 200)
        assert p_value == 0.0 and math.isfinite(log10_p) and log10_p < -300.0
        # The asymptotic branch is continuous with the exact one just above the underflow point.
        exact_edge = math.log10(float(gammaincc(200, 700.0)))
        assert gumbel_pvalue(700.0, 200)[1] == pytest.approx(exact_edge, rel=1e-9)

    def test_secret_vector_is_keyed_and_deterministic(self) -> None:
        first = secret_vector([5], 100, GUMBEL_CONFIG)
        assert np.array_equal(first, secret_vector([5], 100, GUMBEL_CONFIG))
        assert not np.array_equal(first, secret_vector([6], 100, GUMBEL_CONFIG))
        assert not np.array_equal(first, secret_vector([5], 100, GumbelConfig(hash_key=778)))
        assert 0.0 <= first.min() and first.max() < 1.0

    def test_sampling_is_unbiased_over_keys(self) -> None:
        """Averaged over contexts (keys) the chosen token follows the model distribution."""
        probabilities = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
        sampler = GumbelSampler(len(probabilities), GUMBEL_CONFIG)
        n_draws = 20000
        picks = np.bincount(
            [sampler.select([context], probabilities) for context in range(n_draws)],
            minlength=len(probabilities),
        )
        frequencies = picks / n_draws
        sigma = np.sqrt(probabilities * (1 - probabilities) / n_draws)
        assert np.all(np.abs(frequencies - probabilities) < 4.5 * sigma)

    def test_sampler_as_logits_processor_returns_a_one_hot(self) -> None:
        sampler = GumbelSampler(FAKE_VOCAB, GUMBEL_CONFIG)
        scores = FAKE_TABLE[[5, 6]]
        out = sampler(torch.tensor([[1, 5], [2, 6]]), scores)  # type: ignore[arg-type]
        for row, last in enumerate((5, 6)):
            probabilities = torch.softmax(scores[row].double(), dim=-1).numpy()
            assert int(out[row].argmax()) == sampler.select([last], probabilities)
            assert int((out[row] == 0).sum()) == 1

    def test_generation_is_deterministic_regardless_of_seed(self) -> None:
        a = sample_sequences(
            fake_next_logits,
            [9],
            n_sequences=1,
            scheme="gumbel",
            config=GUMBEL_CONFIG,
            max_new_tokens=40,
            seed=1,
        )[0]
        b = sample_sequences(
            fake_next_logits,
            [9],
            n_sequences=1,
            scheme="gumbel",
            config=GUMBEL_CONFIG,
            max_new_tokens=40,
            seed=2,
        )[0]
        assert a == b

    def test_false_positive_rate_under_h0(self) -> None:
        rng = np.random.default_rng(2)
        p_values = [
            gumbel_detect(random_text(rng), FAKE_VOCAB, GUMBEL_CONFIG).p_value
            for _ in range(H0_TRIALS)
        ]
        assert 0.02 <= float(np.mean(np.array(p_values) <= 0.05)) <= 0.09
        assert sum(p < TEXT_DETECTION_P for p in p_values) <= 2

    def test_watermarked_texts_are_detected_and_plain_ones_are_not(self) -> None:
        detected = [
            gumbel_detect(fake_text("gumbel", GUMBEL_CONFIG, i), FAKE_VOCAB, GUMBEL_CONFIG).detected
            for i in range(N_WATERMARKED_TEXTS)
        ]
        false_alarms = [
            gumbel_detect(fake_text("none", None, i), FAKE_VOCAB, GUMBEL_CONFIG).detected
            for i in range(N_PLAIN_TEXTS)
        ]
        assert all(detected)
        assert sum(false_alarms) / N_PLAIN_TEXTS <= 0.01

    def test_wrong_key_does_not_detect(self) -> None:
        wrong = GumbelConfig(hash_key=GUMBEL_CONFIG.hash_key + 1)
        results = [
            gumbel_detect(fake_text("gumbel", GUMBEL_CONFIG, i), FAKE_VOCAB, wrong).detected
            for i in range(N_WATERMARKED_TEXTS)
        ]
        assert not any(results)

    def test_degenerate_inputs_and_config(self) -> None:
        empty = gumbel_detect([], FAKE_VOCAB, GUMBEL_CONFIG)
        assert empty.p_value == 1.0 and empty.z_score is None and empty.n_green is None
        with pytest.raises(WatermarkConfigError):
            GumbelConfig(hash_key=-1)
        with pytest.raises(WatermarkConfigError):
            GumbelConfig(context_width=0)

    def test_context_width_two(self) -> None:
        config = GumbelConfig(hash_key=5, context_width=2)
        tokens = sample_sequences(
            fake_next_logits,
            [4, 8],
            n_sequences=1,
            scheme="gumbel",
            config=config,
            max_new_tokens=TEXT_TOKENS,
            seed=0,
        )[0]
        assert gumbel_detect(tokens, FAKE_VOCAB, config).detected
        assert not gumbel_detect(tokens, FAKE_VOCAB, GUMBEL_CONFIG).detected


class TestAttacks:
    """Token-level edits."""

    def test_substitution_edits_exactly_the_requested_count_with_other_tokens(self) -> None:
        rng = np.random.default_rng(0)
        ids = list(range(100))
        for rate in (0.0, 0.1, 0.3, 0.5, 1.0):
            out = substitute_tokens(ids, rate, FAKE_VOCAB, rng)
            changed = sum(a != b for a, b in zip(ids, out, strict=True))
            assert len(out) == len(ids) and changed == round(rate * len(ids))
            assert all(0 <= token < FAKE_VOCAB for token in out)

    def test_substitution_is_uniform_over_the_other_tokens(self) -> None:
        rng = np.random.default_rng(1)
        out = substitute_tokens([0] * 4000, 1.0, 5, rng)
        counts = np.bincount(out, minlength=5)
        assert counts[0] == 0 and np.all(np.abs(counts[1:] - 1000) < 150)

    def test_deletion_removes_the_requested_count_and_keeps_order(self) -> None:
        rng = np.random.default_rng(2)
        ids = list(range(100))
        out = delete_tokens(ids, 0.2, rng)
        assert len(out) == 80 and out == sorted(out) and set(out) <= set(ids)

    def test_insertion_adds_tokens_and_keeps_the_original_as_a_subsequence(self) -> None:
        rng = np.random.default_rng(3)
        ids = list(range(100, 200))
        out = insert_tokens(ids, 0.3, 50, rng)
        assert len(out) == 130
        iterator = iter(out)
        assert all(token in iterator for token in ids)

    def test_rate_zero_is_identity_and_inputs_are_not_mutated(self) -> None:
        rng = np.random.default_rng(4)
        ids = [3, 1, 4, 1, 5]
        snapshot = list(ids)
        assert substitute_tokens(ids, 0.0, 10, rng) == ids
        assert delete_tokens(ids, 0.0, rng) == ids
        assert insert_tokens(ids, 0.0, 10, rng) == ids
        substitute_tokens(ids, 0.6, 10, rng)
        assert ids == snapshot

    def test_same_seed_same_attack_and_invalid_rates(self) -> None:
        ids = list(range(50))
        first = substitute_tokens(ids, 0.4, 99, np.random.default_rng(7))
        assert first == substitute_tokens(ids, 0.4, 99, np.random.default_rng(7))
        rng = np.random.default_rng(0)
        for bad in (-0.1, 1.5):
            with pytest.raises(AttackRateError):
                substitute_tokens(ids, bad, 99, rng)
            with pytest.raises(AttackRateError):
                delete_tokens(ids, bad, rng)
            with pytest.raises(AttackRateError):
                insert_tokens(ids, bad, 99, rng)
        with pytest.raises(AttackRateError):
            substitute_tokens(ids, 0.1, 1, rng)

    def test_detection_degrades_with_the_substitution_rate(self) -> None:
        rng = np.random.default_rng(5)
        rates = (0.0, 0.1, 0.3, 0.5)
        mean_strength = []
        for rate in rates:
            strengths = []
            for index in range(12):
                tokens = fake_text("kgw", KGW_CONFIG, index)
                attacked = substitute_tokens(tokens, rate, FAKE_VOCAB, rng)
                strengths.append(kgw_detect(attacked, FAKE_VOCAB, KGW_CONFIG).neg_log10_p)
            mean_strength.append(float(np.mean(strengths)))
        assert mean_strength == sorted(mean_strength, reverse=True)
        assert mean_strength[-1] < 0.5 * mean_strength[0]


class TestLocalization:
    """Segment recovery in mixed texts."""

    @staticmethod
    def mixed_text(
        scheme: str, config: KGWConfig | GumbelConfig, index: int
    ) -> tuple[list[int], tuple[int, int]]:
        before = fake_text("none", None, 1000 + index, LOCALIZE_SEGMENT)
        marked = fake_text(scheme, config, 2000 + index, LOCALIZE_SEGMENT)
        after = fake_text("none", None, 3000 + index, LOCALIZE_SEGMENT)
        return before + marked + after, (LOCALIZE_SEGMENT, 2 * LOCALIZE_SEGMENT)

    @pytest.mark.parametrize(("scheme", "config"), [("kgw", KGW_CONFIG), ("gumbel", GUMBEL_CONFIG)])
    def test_the_marked_segment_is_found(
        self, scheme: str, config: KGWConfig | GumbelConfig
    ) -> None:
        ious = []
        for index in range(8):
            tokens, truth = self.mixed_text(scheme, config, index)
            segments = localize_segments(tokens, FAKE_VOCAB, config)
            ious.append(segments_iou(segments, truth))
            assert all(p < TEXT_DETECTION_P for _, _, p in segments)
        assert float(np.mean(ious)) >= MIN_LOCALIZATION_IOU

    @pytest.mark.parametrize(("scheme", "config"), [("kgw", KGW_CONFIG), ("gumbel", GUMBEL_CONFIG)])
    def test_plain_text_yields_no_segment(
        self, scheme: str, config: KGWConfig | GumbelConfig
    ) -> None:
        found = sum(
            bool(localize_segments(fake_text("none", None, 500 + i, 300), FAKE_VOCAB, config))
            for i in range(N_PLAIN_TEXTS)
        )
        assert found <= 2

    def test_token_scores_shapes_and_nan_rules(self) -> None:
        tokens = [1, 2, 1, 2, 1, 2, 9]
        flags = token_scores(tokens, FAKE_VOCAB, KGW_CONFIG)
        assert flags.shape == (7,)
        assert math.isnan(flags[0]) and math.isnan(flags[3])
        assert set(np.unique(flags[~np.isnan(flags)])) <= {0.0, 1.0}
        gumbel = token_scores(tokens, FAKE_VOCAB, GUMBEL_CONFIG)
        assert (gumbel[~np.isnan(gumbel)] >= 0).all()
        with pytest.raises(WatermarkConfigError):
            token_scores(tokens, FAKE_VOCAB, object())  # type: ignore[arg-type]

    def test_short_and_empty_inputs(self) -> None:
        assert localize_segments([], FAKE_VOCAB, KGW_CONFIG) == []
        assert localize_segments([5, 6, 7], FAKE_VOCAB, KGW_CONFIG) == []
        with pytest.raises(WatermarkConfigError):
            localize_segments([1, 2, 3], FAKE_VOCAB, KGW_CONFIG, window=0)

    def test_whole_text_marked_gives_one_covering_segment(self) -> None:
        tokens = fake_text("kgw", KGW_CONFIG, 77, 150)
        segments = localize_segments(tokens, FAKE_VOCAB, KGW_CONFIG)
        assert segments_iou(segments, (0, 150)) >= 0.9

    def test_segments_iou_edge_cases(self) -> None:
        assert segments_iou([], (3, 3)) == 0.0
        assert segments_iou([(0, 10, 1e-9)], (0, 10)) == 1.0
        assert segments_iou([(0, 10, 1e-9)], (5, 15)) == pytest.approx(5 / 15)
        assert segments_iou([], (0, 10)) == 0.0


class TestSampling:
    """The generic sampling loop."""

    def test_eos_stops_a_row_and_is_not_returned(self) -> None:
        eos = 7
        peaked = torch.full((FAKE_VOCAB,), -1e9)
        peaked[eos] = 0.0

        def always_eos(prefix: torch.Tensor) -> torch.Tensor:
            return peaked.repeat(prefix.shape[0], 1)

        out = sample_sequences(
            always_eos,
            [1],
            n_sequences=2,
            scheme="none",
            config=None,
            max_new_tokens=5,
            seed=0,
            eos_token_id=eos,
        )
        assert out == [[], []]

    def test_min_new_tokens_forbids_early_eos(self) -> None:
        eos = 7
        logits = torch.zeros(FAKE_VOCAB)
        logits[eos] = 50.0

        def eos_heavy(prefix: torch.Tensor) -> torch.Tensor:
            return logits.repeat(prefix.shape[0], 1)

        out = sample_sequences(
            eos_heavy,
            [1],
            n_sequences=1,
            scheme="none",
            config=None,
            max_new_tokens=10,
            min_new_tokens=4,
            seed=0,
            eos_token_id=eos,
        )[0]
        assert len(out) == 4 and eos not in out

    def test_batch_rows_are_independent_samples(self) -> None:
        out = sample_sequences(
            fake_next_logits,
            [3],
            n_sequences=4,
            scheme="none",
            config=None,
            max_new_tokens=30,
            seed=0,
        )
        assert len({tuple(row) for row in out}) == 4

    def test_temperature_controls_the_entropy(self) -> None:
        def distinct_ratio(temperature: float) -> float:
            out = sample_sequences(
                fake_next_logits,
                [3],
                n_sequences=1,
                scheme="none",
                config=None,
                max_new_tokens=300,
                seed=0,
                temperature=temperature,
            )[0]
            return len(set(out)) / len(out)

        assert distinct_ratio(0.1) < distinct_ratio(1.5)

    def test_argument_errors(self) -> None:
        kwargs = {"n_sequences": 1, "max_new_tokens": 3, "seed": 0}
        with pytest.raises(GenerationError):
            sample_sequences(fake_next_logits, [], scheme="none", config=None, **kwargs)
        with pytest.raises(GenerationError):
            sample_sequences(
                fake_next_logits,
                [1],
                scheme="none",
                config=None,
                temperature=0.0,
                **kwargs,
            )
        with pytest.raises(GenerationError):
            sample_sequences(
                fake_next_logits,
                [1],
                scheme="kgw",
                config=GUMBEL_CONFIG,
                **kwargs,
            )
        with pytest.raises(GenerationError):
            sample_sequences(
                fake_next_logits,
                [1],
                scheme="gumbel",
                config=KGW_CONFIG,
                **kwargs,
            )
        with pytest.raises(GenerationError):
            sample_sequences(fake_next_logits, [1], scheme="bogus", config=None, **kwargs)  # type: ignore[arg-type]

    def test_default_configs_are_used_when_none_is_given(self) -> None:
        tokens = sample_sequences(
            fake_next_logits,
            [3],
            n_sequences=1,
            scheme="kgw",
            config=None,
            max_new_tokens=TEXT_TOKENS,
            seed=0,
        )[0]
        assert kgw_detect(tokens, FAKE_VOCAB, KGWConfig()).detected


REAL_MODEL_TEXTS = 4
REAL_PROMPT = "The old lighthouse keeper looked out over the harbor and"


@pytest.fixture(scope="module")
def smollm() -> tuple[object, object, int]:
    """Load SmolLM2-135M-Instruct once (about 270 MB, from the HF cache)."""
    from wmkit.text import load_lm, model_vocab_size

    model, tokenizer = load_lm()
    return model, tokenizer, model_vocab_size(model)


@pytest.mark.slow
class TestRealModel:
    """End-to-end checks with the real language model."""

    def test_vocab_matches_tokenizer_and_kv_cache_is_exact(
        self, smollm: tuple[object, object, int]
    ) -> None:
        from wmkit.text.generate import HFNextLogits

        model, tokenizer, vocab_size = smollm
        assert vocab_size == len(tokenizer)  # type: ignore[arg-type]
        ids = torch.tensor([tokenizer.encode(REAL_PROMPT)])  # type: ignore[attr-defined]
        step = HFNextLogits(model)
        step(ids)
        extended = torch.cat([ids, torch.tensor([[11]])], dim=1)
        cached = step(extended)
        with torch.inference_mode():
            full = model(input_ids=extended.to(next(model.parameters()).device)).logits[:, -1]  # type: ignore[attr-defined,operator]
        assert torch.allclose(cached, full.float().cpu(), atol=1e-3)

    def test_kgw_and_gumbel_texts_are_detected_and_plain_text_is_not(
        self, smollm: tuple[object, object, int]
    ) -> None:
        model, tokenizer, vocab_size = smollm
        for index in range(REAL_MODEL_TEXTS):
            prompt = f"{REAL_PROMPT} number {index}"
            kgw = generate(
                model,
                tokenizer,
                prompt,
                scheme="kgw",
                config=KGWConfig(),
                max_new_tokens=TEXT_TOKENS,
                min_new_tokens=TEXT_TOKENS,
                seed=index,
            )
            gumbel = generate(
                model,
                tokenizer,
                prompt,
                scheme="gumbel",
                config=GumbelConfig(),
                max_new_tokens=TEXT_TOKENS,
                min_new_tokens=TEXT_TOKENS,
                seed=index,
            )
            plain = generate(
                model,
                tokenizer,
                prompt,
                scheme="none",
                config=None,
                max_new_tokens=TEXT_TOKENS,
                min_new_tokens=TEXT_TOKENS,
                seed=index,
            )
            assert len(kgw) == len(gumbel) == len(plain) == TEXT_TOKENS
            assert kgw_detect(kgw, vocab_size, KGWConfig()).detected
            assert gumbel_detect(gumbel, vocab_size, GumbelConfig()).detected
            assert not kgw_detect(plain, vocab_size, KGWConfig()).detected
            assert not gumbel_detect(plain, vocab_size, GumbelConfig()).detected

    def test_processor_works_inside_huggingface_generate(
        self, smollm: tuple[object, object, int]
    ) -> None:
        from transformers import LogitsProcessorList

        model, tokenizer, vocab_size = smollm
        encoded = tokenizer(REAL_PROMPT, return_tensors="pt").to(  # type: ignore[operator]
            next(model.parameters()).device  # type: ignore[attr-defined]
        )
        output = model.generate(  # type: ignore[attr-defined]
            **encoded,
            do_sample=True,
            max_new_tokens=TEXT_TOKENS,
            min_new_tokens=TEXT_TOKENS,
            logits_processor=LogitsProcessorList([KGWLogitsProcessor(vocab_size, KGWConfig())]),
            pad_token_id=tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )
        generated = output[0, encoded["input_ids"].shape[1] :].tolist()
        assert kgw_detect(generated, vocab_size, KGWConfig()).detected
