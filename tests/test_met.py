"""Tests for wmkit.audit (Model Equality Testing and the cached API client).

Fast tests use toy categorical laws, no language model. The real-model checks live in
tests/test_text.py (`slow`), which shares the SmolLM2 fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from wmkit.audit import (
    CachedCompletionClient,
    CompletionRequestError,
    METError,
    Sample,
    mmd_hamming,
    power_curve,
    two_sample_test,
)
from wmkit.audit.met import PAD_TOKEN_ID
from wmkit.text import KGWConfig

TOY_N_PROMPTS = 5
TOY_VOCAB = 12
TOY_LENGTH = 8
TOY_SEED = 20261121
H0_TRIALS = 300
H0_RESAMPLES = 199
ALPHA = 0.05
SIZE_LOW, SIZE_HIGH = 0.02, 0.09
"""Acceptance band of the empirical type-I error at alpha = 0.05 (criterion 1)."""

FloatArray = npt.NDArray[np.float64]


def toy_law(rng: np.random.Generator, concentration: float = 0.7) -> FloatArray:
    """Random per-(prompt, position) categorical laws `[prompts, positions, vocab]`."""
    return rng.dirichlet(np.full(TOY_VOCAB, concentration), size=(TOY_N_PROMPTS, TOY_LENGTH))


def draw_toy(law: FloatArray, n_samples: int, rng: np.random.Generator) -> list[Sample]:
    """Draw `n_samples` samples, prompts cycling so that every prompt gets the same count."""
    samples = []
    for index in range(n_samples):
        prompt = index % TOY_N_PROMPTS
        tokens = tuple(
            int(rng.choice(TOY_VOCAB, p=law[prompt, position])) for position in range(TOY_LENGTH)
        )
        samples.append(Sample(prompt, tokens))
    return samples


def naive_mmd(
    samples_p: Sequence[Sample], samples_q: Sequence[Sample], completion_len: int
) -> float:
    """Reference O(N^2 L) transcription of the official `_mmd` for the Hamming kernel."""

    def padded(sample: Sample) -> list[int]:
        row = list(sample.completion_ids[:completion_len])
        return row + [PAD_TOKEN_ID] * (completion_len - len(row))

    def block_mean(first: Sequence[Sample], second: Sequence[Sample], same: bool) -> float:
        total, pairs = 0.0, 0
        for i, a in enumerate(first):
            for j, b in enumerate(second):
                if same and i == j:
                    continue
                if a.prompt_id != b.prompt_id:
                    continue
                matches = sum(x == y for x, y in zip(padded(a), padded(b), strict=True))
                total += matches / completion_len
                pairs += 1
        return total / max(pairs, 1)

    return (
        block_mean(samples_p, samples_p, True)
        - 2.0 * block_mean(samples_p, samples_q, False)
        + block_mean(samples_q, samples_q, True)
    )


def ragged_samples(rng: np.random.Generator, n_samples: int, n_prompts: int) -> list[Sample]:
    """Samples with variable lengths (some longer than L, some empty) and few tokens."""
    samples = []
    for _ in range(n_samples):
        length = int(rng.integers(0, 14))
        samples.append(
            Sample(int(rng.integers(0, n_prompts)), tuple(rng.integers(0, 4, size=length).tolist()))
        )
    return samples


class TestStatistic:
    """The fast counting statistic equals the official definition."""

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_naive_implementation(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        samples_p = ragged_samples(rng, int(rng.integers(5, 30)), n_prompts=4)
        samples_q = ragged_samples(rng, int(rng.integers(5, 30)), n_prompts=5)
        for completion_len in (3, 10):
            fast = mmd_hamming(samples_p, samples_q, completion_len=completion_len)
            slow = naive_mmd(samples_p, samples_q, completion_len)
            assert fast == pytest.approx(slow, abs=1e-12)

    def test_identical_distributions_have_small_statistic(self) -> None:
        rng = np.random.default_rng(TOY_SEED)
        law = toy_law(rng)
        stat = mmd_hamming(draw_toy(law, 500, rng), draw_toy(law, 500, rng), completion_len=8)
        assert abs(stat) < 0.01

    def test_different_distributions_have_larger_statistic(self) -> None:
        rng = np.random.default_rng(TOY_SEED)
        law_p, law_q = toy_law(rng), toy_law(rng)
        same = mmd_hamming(draw_toy(law_p, 300, rng), draw_toy(law_p, 300, rng), completion_len=8)
        diff = mmd_hamming(draw_toy(law_p, 300, rng), draw_toy(law_q, 300, rng), completion_len=8)
        assert diff > same + 0.01

    def test_symmetric_in_its_arguments(self) -> None:
        rng = np.random.default_rng(1)
        a, b = ragged_samples(rng, 20, 3), ragged_samples(rng, 25, 3)
        assert mmd_hamming(a, b, completion_len=6) == pytest.approx(
            mmd_hamming(b, a, completion_len=6), abs=1e-12
        )

    def test_prompts_present_on_one_side_only_are_handled(self) -> None:
        a = [Sample(0, (1, 2, 3)), Sample(0, (1, 2, 4))]
        b = [Sample(1, (1, 2, 3)), Sample(1, (9, 9, 9))]
        assert mmd_hamming(a, b, completion_len=3) == pytest.approx(naive_mmd(a, b, 3))

    def test_padding_matches_padding(self) -> None:
        """Two short completions agree on every padded position, as in the official code."""
        a = [Sample(0, (1,)), Sample(0, (1,))]
        b = [Sample(0, (2,)), Sample(0, (2,))]
        assert mmd_hamming(a, b, completion_len=4) == pytest.approx(naive_mmd(a, b, 4))

    def test_invalid_inputs_raise_typed_errors(self) -> None:
        good = [Sample(0, (1, 2))]
        with pytest.raises(METError):
            mmd_hamming([], good)
        with pytest.raises(METError):
            mmd_hamming(good, [Sample(0, (-1, 2))])
        with pytest.raises(METError):
            mmd_hamming(good, good, completion_len=0)


class TestOfficialPackage:
    """Agreement with `model-equality-testing` (skipped when it is not installed)."""

    @pytest.mark.parametrize("seed", range(4))
    def test_statistic_equals_run_two_sample_test(self, seed: int) -> None:
        official = pytest.importorskip("model_equality_testing")
        from model_equality_testing.algorithm import run_two_sample_test
        from model_equality_testing.distribution import CompletionSample
        from model_equality_testing.utils import pad_to_length

        rng = np.random.default_rng(seed)
        law_p, law_q = toy_law(rng), toy_law(rng)
        samples_p = draw_toy(law_p, 60, rng)
        samples_q = draw_toy(law_q, 75, rng)

        def to_official(samples: Sequence[Sample]) -> object:
            prompts = np.array([s.prompt_id for s in samples])
            completions = pad_to_length(
                np.array([list(s.completion_ids) for s in samples]), L=TOY_LENGTH
            )
            return CompletionSample(prompts=prompts, completions=completions, m=TOY_N_PROMPTS)

        _, official_stat = run_two_sample_test(
            to_official(samples_p),
            to_official(samples_q),
            pvalue_type="dummy",
            stat_type="mmd_hamming",
        )
        ours = mmd_hamming(samples_p, samples_q, completion_len=TOY_LENGTH)
        assert official is not None
        assert ours == pytest.approx(float(official_stat), abs=1e-12)

    def test_permutation_pvalues_agree_on_a_clear_alternative(self) -> None:
        pytest.importorskip("model_equality_testing")
        from model_equality_testing.algorithm import run_two_sample_test
        from model_equality_testing.distribution import CompletionSample
        from model_equality_testing.utils import pad_to_length

        rng = np.random.default_rng(7)
        law_p, law_q = toy_law(rng), toy_law(rng)
        samples_p, samples_q = draw_toy(law_p, 100, rng), draw_toy(law_q, 100, rng)

        def to_official(samples: Sequence[Sample]) -> CompletionSample:
            prompts = np.array([s.prompt_id for s in samples])
            completions = pad_to_length(
                np.array([list(s.completion_ids) for s in samples]), L=TOY_LENGTH
            )
            return CompletionSample(prompts=prompts, completions=completions, m=TOY_N_PROMPTS)

        official_p, _ = run_two_sample_test(
            to_official(samples_p),
            to_official(samples_q),
            pvalue_type="permutation_pvalue",
            stat_type="mmd_hamming",
            b=100,
        )
        ours = two_sample_test(
            samples_p,
            samples_q,
            completion_len=TOY_LENGTH,
            n_resamples=199,
            stratify_by_prompt=False,
        )
        assert official_p <= 0.02
        assert ours.p_value <= 0.02


class TestPValues:
    """Type-I error and power on toy laws."""

    def test_permutation_pvalue_is_uniform_under_h0(self) -> None:
        """Criterion 1: rejection rate at alpha = 0.05 inside [0.02, 0.09] over >= 100 trials."""
        rng = np.random.default_rng(TOY_SEED)
        law = toy_law(rng)
        p_values = []
        for trial in range(H0_TRIALS):
            result = two_sample_test(
                draw_toy(law, 40, rng),
                draw_toy(law, 40, rng),
                completion_len=TOY_LENGTH,
                n_resamples=H0_RESAMPLES,
                seed=trial,
            )
            p_values.append(result.p_value)
        rejection_rate = float(np.mean(np.array(p_values) <= ALPHA))
        assert SIZE_LOW <= rejection_rate <= SIZE_HIGH, rejection_rate
        assert 0.4 < float(np.mean(p_values)) < 0.6

    def test_global_permutation_is_also_valid_under_h0(self) -> None:
        rng = np.random.default_rng(TOY_SEED + 1)
        law = toy_law(rng)
        rejections = 0
        for trial in range(H0_TRIALS):
            result = two_sample_test(
                draw_toy(law, 40, rng),
                draw_toy(law, 40, rng),
                completion_len=TOY_LENGTH,
                n_resamples=H0_RESAMPLES,
                seed=trial,
                stratify_by_prompt=False,
            )
            rejections += int(result.p_value <= ALPHA)
        assert SIZE_LOW <= rejections / H0_TRIALS <= SIZE_HIGH

    def test_simulation_pvalue_is_valid_under_h0(self) -> None:
        rng = np.random.default_rng(TOY_SEED + 2)
        law = toy_law(rng)
        pool = draw_toy(law, 1000, rng)
        rejections = 0
        n_trials = 150
        for trial in range(n_trials):
            result = two_sample_test(
                draw_toy(law, 40, rng),
                draw_toy(law, 40, rng),
                completion_len=TOY_LENGTH,
                pvalue="simulation",
                reference_pool=pool,
                n_resamples=H0_RESAMPLES,
                seed=trial,
            )
            rejections += int(result.p_value <= ALPHA)
        assert SIZE_LOW <= rejections / n_trials <= 0.10

    def test_power_is_high_for_a_clear_alternative(self) -> None:
        rng = np.random.default_rng(TOY_SEED + 3)
        law_p, law_q = toy_law(rng), toy_law(rng)
        result = two_sample_test(
            draw_toy(law_p, 100, rng),
            draw_toy(law_q, 100, rng),
            completion_len=TOY_LENGTH,
            n_resamples=H0_RESAMPLES,
        )
        assert result.p_value <= 0.01
        assert result.n_p == 100 and result.n_q == 100

    def test_minimum_pvalue_is_one_over_b_plus_one(self) -> None:
        a = [Sample(0, (1,) * 6)] * 20
        b = [Sample(0, (2,) * 6)] * 20
        result = two_sample_test(a, b, completion_len=6, n_resamples=99)
        assert result.p_value == pytest.approx(1.0 / 100.0)

    def test_identical_samples_give_pvalue_one(self) -> None:
        rng = np.random.default_rng(3)
        sample = draw_toy(toy_law(rng), 30, rng)
        result = two_sample_test(sample, list(sample), completion_len=TOY_LENGTH, n_resamples=50)
        assert result.p_value == pytest.approx(1.0)

    def test_same_seed_same_pvalue(self) -> None:
        rng = np.random.default_rng(4)
        law = toy_law(rng)
        a, b = draw_toy(law, 30, rng), draw_toy(law, 30, rng)
        first = two_sample_test(a, b, completion_len=TOY_LENGTH, n_resamples=60, seed=5)
        second = two_sample_test(a, b, completion_len=TOY_LENGTH, n_resamples=60, seed=5)
        assert first == second

    def test_simulation_needs_a_pool_and_enough_samples(self) -> None:
        rng = np.random.default_rng(5)
        law = toy_law(rng)
        a, b = draw_toy(law, 20, rng), draw_toy(law, 20, rng)
        with pytest.raises(METError, match="reference_pool"):
            two_sample_test(a, b, completion_len=TOY_LENGTH, pvalue="simulation")
        with pytest.raises(METError, match="needs"):
            two_sample_test(
                a,
                b,
                completion_len=TOY_LENGTH,
                pvalue="simulation",
                reference_pool=draw_toy(law, 25, rng),
            )

    def test_unknown_method_and_bad_resamples(self) -> None:
        sample = [Sample(0, (1, 2))]
        with pytest.raises(METError):
            two_sample_test(sample, sample, pvalue="bogus")  # type: ignore[arg-type]
        with pytest.raises(METError):
            two_sample_test(sample, sample, n_resamples=0)


class TestPowerCurve:
    """Monte Carlo curves."""

    def test_curve_is_near_alpha_under_h0_and_rises_under_alternative(self) -> None:
        rng = np.random.default_rng(TOY_SEED + 4)
        law_p, law_q = toy_law(rng), toy_law(rng)

        def draw_p(n: int, trial_rng: np.random.Generator) -> list[Sample]:
            return draw_toy(law_p, n, trial_rng)

        def draw_q(n: int, trial_rng: np.random.Generator) -> list[Sample]:
            return draw_toy(law_q, n, trial_rng)

        budgets = [10, 40, 150]
        size = power_curve(draw_p, draw_p, budgets, n_trials=120, seed=1, completion_len=TOY_LENGTH)
        power = power_curve(draw_p, draw_q, budgets, n_trials=60, seed=1, completion_len=TOY_LENGTH)
        assert [budget for budget, _ in size] == budgets
        assert all(rate <= 0.14 for _, rate in size)
        rates = [rate for _, rate in power]
        assert rates[-1] >= 0.9
        assert rates[-1] >= rates[0]

    def test_rejects_bad_trial_count(self) -> None:
        draw: Callable[[int, np.random.Generator], list[Sample]] = lambda n, r: [Sample(0, (1,))]
        with pytest.raises(METError):
            power_curve(draw, draw, [1], n_trials=0, seed=0)


class FakeClock:
    """Manual wall clock whose `sleep` advances time instead of blocking."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CountingApi:
    """Fake API recording when it was called."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, float]] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append((prompt, self.clock()))
        return f"answer-{len(self.calls)}-{prompt}"


RATE_LIMIT_S = 900.0
"""One request every 15 minutes."""


class TestCachedCompletionClient:
    """Cache, counters and rate limit, with a simulated clock (no real sleeping)."""

    def make(
        self, tmp_path: Path, interval: float = 0.0, clock: FakeClock | None = None
    ) -> tuple[CachedCompletionClient, CountingApi, FakeClock]:
        clock = clock or FakeClock()
        api = CountingApi(clock)
        client = CachedCompletionClient(
            api, tmp_path / "cache" / "api.jsonl", interval, clock=clock, sleep=clock.sleep
        )
        return client, api, clock

    def test_cache_hit_does_not_call_the_api(self, tmp_path: Path) -> None:
        client, api, _ = self.make(tmp_path)
        first = client("hello")
        second = client("hello")
        assert first == second
        assert len(api.calls) == 1
        assert (client.n_requests, client.n_cache_hits, client.n_cached) == (1, 1, 1)

    def test_replicates_are_cached_separately(self, tmp_path: Path) -> None:
        client, api, _ = self.make(tmp_path)
        answers = client.sample("p", 3)
        assert len(set(answers)) == 3
        assert client.sample("p", 3) == answers
        assert len(api.calls) == 3

    def test_cache_replays_across_instances(self, tmp_path: Path) -> None:
        client, _, clock = self.make(tmp_path)
        answers = client.sample("p", 2)
        reopened, api, _ = self.make(tmp_path, clock=clock)
        assert reopened.sample("p", 2) == answers
        assert api.calls == [] and reopened.n_requests == 0 and reopened.n_cache_hits == 2

    def test_min_interval_is_respected_with_a_simulated_clock(self, tmp_path: Path) -> None:
        client, api, clock = self.make(tmp_path, interval=RATE_LIMIT_S)
        for prompt in ("a", "b", "c", "d"):
            client(prompt)
        times = [when for _, when in api.calls]
        gaps = np.diff(times)
        assert (gaps >= RATE_LIMIT_S - 1e-9).all()
        assert clock.sleeps == [pytest.approx(RATE_LIMIT_S)] * 3

    def test_no_wait_when_enough_time_elapsed_by_itself(self, tmp_path: Path) -> None:
        client, _, clock = self.make(tmp_path, interval=RATE_LIMIT_S)
        client("a")
        clock.now += 2 * RATE_LIMIT_S
        client("b")
        assert clock.sleeps == []

    def test_cache_hits_are_never_throttled(self, tmp_path: Path) -> None:
        client, _, clock = self.make(tmp_path, interval=RATE_LIMIT_S)
        client("a")
        for _ in range(5):
            client("a")
        assert clock.sleeps == []

    def test_interval_survives_a_restart(self, tmp_path: Path) -> None:
        client, _, clock = self.make(tmp_path, interval=RATE_LIMIT_S)
        client("a")
        clock.now += 100.0
        restarted, _, _ = self.make(tmp_path, interval=RATE_LIMIT_S, clock=clock)
        assert restarted.seconds_until_ready() == pytest.approx(RATE_LIMIT_S - 100.0)
        restarted("b")
        assert clock.sleeps == [pytest.approx(RATE_LIMIT_S - 100.0)]

    def test_failed_call_is_not_cached_but_consumes_the_slot(self, tmp_path: Path) -> None:
        clock = FakeClock()
        outcomes = iter([RuntimeError("503"), "ok"])

        def flaky(prompt: str) -> str:
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            assert isinstance(outcome, str)
            return outcome

        client = CachedCompletionClient(
            flaky, tmp_path / "c.jsonl", RATE_LIMIT_S, clock=clock, sleep=clock.sleep
        )
        with pytest.raises(CompletionRequestError):
            client("p")
        assert client.n_cached == 0 and client.n_requests == 1
        assert client("p") == "ok"
        assert clock.sleeps == [pytest.approx(RATE_LIMIT_S)]

    def test_non_string_answer_is_rejected(self, tmp_path: Path) -> None:
        client = CachedCompletionClient(lambda p: 3, tmp_path / "c.jsonl")  # type: ignore[arg-type,return-value]
        with pytest.raises(CompletionRequestError):
            client("p")
        assert client.n_cached == 0

    def test_torn_last_line_is_skipped_and_the_file_stays_valid(self, tmp_path: Path) -> None:
        client, _, clock = self.make(tmp_path)
        client("a")
        path = tmp_path / "cache" / "api.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"prompt": "torn", "replic')
        reopened, _, _ = self.make(tmp_path, clock=clock)
        assert reopened.n_corrupt_lines == 1 and reopened.n_cached == 1
        reopened("b")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[-1])["prompt"] == "b"
        again, _, _ = self.make(tmp_path, clock=clock)
        assert again.n_cached == 2 and again.n_corrupt_lines == 1

    def test_unicode_and_empty_completions_roundtrip(self, tmp_path: Path) -> None:
        clock = FakeClock()
        answers = iter(["", "café 中文\nline2"])
        client = CachedCompletionClient(lambda p: next(answers), tmp_path / "c.jsonl", clock=clock)
        assert client("x") == ""
        assert client("y") == "café 中文\nline2"
        reopened = CachedCompletionClient(lambda p: "never", tmp_path / "c.jsonl", clock=clock)
        assert reopened("x") == "" and reopened("y").startswith("caf")
        assert reopened.n_requests == 0

    def test_negative_interval_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="min_interval_s"):
            CachedCompletionClient(lambda p: "", tmp_path / "c.jsonl", -1.0)


REAL_PROMPTS = (
    "The history of the Roman Empire began when",
    "Scientists announced yesterday that a new",
    "To install the package, first you need to",
    "Once upon a time, in a distant kingdom,",
)
REAL_SAMPLES_PER_PROMPT = 24
REAL_COMPLETION_LEN = 40


def real_samples(
    model: Any,
    tokenizer: Any,
    *,
    scheme: str,
    config: KGWConfig | None,
    temperature: float,
    seed: int,
) -> list[Sample]:
    """Sample completions of every REAL_PROMPTS prompt from the local model."""
    from wmkit.text import generate_many

    samples: list[Sample] = []
    for index, prompt in enumerate(REAL_PROMPTS):
        rows = generate_many(
            model,
            tokenizer,
            prompt,
            REAL_SAMPLES_PER_PROMPT,
            scheme=scheme,  # type: ignore[arg-type]
            config=config,
            max_new_tokens=REAL_COMPLETION_LEN,
            seed=seed + index,
            temperature=temperature,
        )
        samples.extend(Sample(index, tuple(row)) for row in rows)
    return samples


@pytest.fixture(scope="module")
def lm() -> tuple[Any, Any]:
    """Load SmolLM2-135M-Instruct once for the module."""
    from wmkit.text import load_lm

    return load_lm()


@pytest.mark.slow
class TestRealModel:
    """MET against SmolLM2-135M-Instruct (about 270 MB, from the HF cache)."""

    def test_same_model_is_not_rejected_but_alternatives_are(self, lm: tuple[Any, Any]) -> None:
        model, tokenizer = lm
        reference = real_samples(
            model, tokenizer, scheme="none", config=None, temperature=1.0, seed=1000
        )
        same = real_samples(
            model, tokenizer, scheme="none", config=None, temperature=1.0, seed=2000
        )
        colder = real_samples(
            model, tokenizer, scheme="none", config=None, temperature=0.7, seed=3000
        )
        marked = real_samples(
            model,
            tokenizer,
            scheme="kgw",
            config=KGWConfig(gamma=0.25, delta=2.5),
            temperature=1.0,
            seed=4000,
        )
        kwargs = {"completion_len": REAL_COMPLETION_LEN, "n_resamples": 199, "seed": 1}
        assert two_sample_test(reference, same, **kwargs).p_value > 0.01  # type: ignore[arg-type]
        assert two_sample_test(reference, colder, **kwargs).p_value <= 0.05  # type: ignore[arg-type]
        assert two_sample_test(reference, marked, **kwargs).p_value <= 0.05  # type: ignore[arg-type]
