"""Integration: modules of the four lots working together, offline and fast.

Covers the CLI end to end (in process and as a real `python -m wmkit` subprocess), the
registry -> classical watermarker -> attack -> extraction chain, text watermarking without a
language model (a table-driven next-token oracle), the Model Equality Test against the official
`model-equality-testing` package on independent toy data, the cached API client feeding the MET,
and an unlearning round trip (SCRUB -> report -> competition score).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as torch_f
from conftest import load_sample
from torch import Tensor, nn

from wmkit import cli
from wmkit.attacks import attack_seed, get_attack, get_suite, reseed_attack
from wmkit.audit import CachedCompletionClient, Sample, mmd_hamming, two_sample_test
from wmkit.eval import evaluate
from wmkit.io import load_image, save_image
from wmkit.messages import bits_to_hex, hex_to_bits, random_messages
from wmkit.methods import get_watermarker
from wmkit.metrics import DETECTION_NEG_LOG10_P, bit_accuracy, neg_log10_pvalue, psnr
from wmkit.text import (
    GumbelConfig,
    KGWConfig,
    KGWLogitsProcessor,
    delete_tokens,
    gumbel_detect,
    insert_tokens,
    kgw_detect,
    localize_segments,
    sample_sequences,
    substitute_tokens,
    token_scores,
)
from wmkit.unlearn import (
    ScrubConfig,
    evaluate_unlearning,
    finetune_retain,
    forget_quality_from_epsilons,
    neurips2023_score,
    scrub,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_METHODS = ["dct_qim", "spread_spectrum", "dwt_dct_svd"]
IMAGE_SIDE = 256
MIN_PSNR_DB = 38.0
MIN_CLEAN_BIT_ACC = 0.95
SUBPROCESS_TIMEOUT_S = 180


def _write_image(path: Path, name: str = "astronaut", side: int = IMAGE_SIDE) -> Path:
    save_image(load_sample(name, (side, side)), path)
    return path


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONWARNINGS"] = "ignore"
    return env


# ---------------------------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------------------------


class TestCliEndToEnd:
    """`wmkit` commands chained the way a user would, on the classical baselines."""

    def test_list_mentions_every_method_group_and_suite(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        for token in ("pixelseal", "dct_qim", "identity:", "valuemetric:", "compression:",
                      "geometric:", "combined:", "removal:", "quick:", "paper: 46", "removal: 8",
                      "all: 54 attacks"):
            assert token in out, token

    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_embed_extract_round_trip(
        self, method: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = _write_image(tmp_path / "in.png")
        marked = tmp_path / "marked.png"
        assert cli.main(["embed", "--method", method, "--in", str(source), "--out", str(marked),
                         "--seed", "7"]) == 0
        message = capsys.readouterr().out.strip()
        n_bits = get_watermarker(method).n_bits
        assert len(message) == n_bits // 4
        assert cli.main(["extract", "--method", method, "--in", str(marked),
                         "--message", message]) == 0
        out = capsys.readouterr().out
        assert f"bits: {message}" in out
        assert "bit_acc: 1.0000" in out
        assert "detected: True" in out
        # Embedding survives the PNG round trip with the contract's quality target.
        original, watermarked = load_image(source)[None], load_image(marked)[None]
        assert float(psnr(watermarked, original)[0]) >= MIN_PSNR_DB

    def test_extract_on_unmarked_image_is_not_detected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = _write_image(tmp_path / "clean.png")
        assert cli.main(["extract", "--method", "dct_qim", "--in", str(source),
                         "--message", "deadbeef"]) == 0
        assert "detected: False" in capsys.readouterr().out

    def test_embed_with_explicit_message_prints_it_back(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = _write_image(tmp_path / "in.png")
        assert cli.main(["embed", "--method", "dct_qim", "--in", str(source),
                         "--out", str(tmp_path / "o.png"), "--message", "cafebabe"]) == 0
        assert capsys.readouterr().out.strip() == "cafebabe"

    @pytest.mark.parametrize("attack_name", ["jpeg_50", "gaussian_blur_3", "hflip", "brightness_1.25"])
    def test_attack_command_then_extract_for_a_robust_method(
        self, attack_name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = _write_image(tmp_path / "in.png")
        marked, attacked = tmp_path / "m.png", tmp_path / "a.png"
        cli.main(["embed", "--method", "spread_spectrum", "--in", str(source), "--out",
                  str(marked), "--seed", "1"])
        message = capsys.readouterr().out.strip()
        assert cli.main(["attack", "--attack", attack_name, "--in", str(marked),
                         "--out", str(attacked)]) == 0
        capsys.readouterr()
        assert attacked.exists()
        assert cli.main(["extract", "--method", "spread_spectrum", "--in", str(attacked),
                         "--message", message]) == 0
        out = capsys.readouterr().out
        bit_acc = float(next(line for line in out.splitlines()
                             if line.startswith("bit_acc")).split(":")[1])
        # spread spectrum is only expected to keep a clear majority under these mild attacks
        if attack_name != "hflip":
            assert bit_acc >= 0.75, out

    @pytest.mark.parametrize(
        ("attack_name", "expected_shape"),
        [("crop_0.5", (128, 128)), ("resize_0.5", (128, 128)), ("rotate_10", (256, 256)),
         ("identity", (256, 256)), ("hflip", (256, 256))],
    )
    def test_attack_output_sizes_follow_the_contract(
        self, attack_name: str, expected_shape: tuple[int, int], tmp_path: Path
    ) -> None:
        source = _write_image(tmp_path / "in.png")
        out = tmp_path / "out.png"
        assert cli.main(["attack", "--attack", attack_name, "--in", str(source),
                         "--out", str(out)]) == 0
        assert tuple(load_image(out).shape[-2:]) == expected_shape

    def test_attack_command_is_deterministic_per_seed(self, tmp_path: Path) -> None:
        source = _write_image(tmp_path / "in.png")
        outputs = []
        for index, seed in enumerate((5, 5, 6)):
            out = tmp_path / f"o{index}.png"
            cli.main(["attack", "--attack", "crop_0.5", "--in", str(source), "--out", str(out),
                      "--seed", str(seed)])
            outputs.append(load_image(out))
        assert torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[0], outputs[2])

    def test_eval_command_writes_the_contract_csv(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        folder = tmp_path / "images"
        for name in ("a", "b", "c"):
            _write_image(folder / f"{name}.png", ["astronaut", "coffee", "chelsea"][ord(name) - 97],
                         side=128)
        csv_path = tmp_path / "res" / "eval.csv"
        assert cli.main(["eval", "--method", "dct_qim,dwt_dct_svd", "--images", str(folder),
                         "--suite", "quick", "--max-images", "2", "--seed", "0",
                         "--out", str(csv_path)]) == 0
        frame = pd.read_csv(csv_path)
        assert list(frame.columns) == [
            "method", "image_id", "is_watermarked", "attack", "attack_group", "n_bits", "bit_acc",
            "neg_log10_p", "detected", "psnr", "ssim", "lpips",
        ]
        n_attacks = len(get_suite("quick"))
        assert len(frame) == 2 * 2 * n_attacks * 2  # methods x images x attacks x (marked, clean)
        assert set(frame["method"]) == {"dct_qim", "dwt_dct_svd"}
        assert set(frame["image_id"]) == {"a.png", "b.png"}
        assert set(frame["is_watermarked"]) == {True, False}
        assert frame["lpips"].isna().all()
        assert not frame.drop(columns="lpips").isna().any().any()
        assert frame["is_watermarked"].dtype == bool and frame["detected"].dtype == bool
        out = capsys.readouterr().out
        assert "| method" in out  # markdown summary printed

    def test_error_paths_exit_nonzero_with_a_one_line_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = _write_image(tmp_path / "in.png")
        assert cli.main(["embed", "--method", "nope", "--in", str(source),
                         "--out", str(tmp_path / "o.png")]) == 1
        assert cli.main(["attack", "--attack", "nope", "--in", str(source),
                         "--out", str(tmp_path / "o.png")]) == 1
        assert cli.main(["embed", "--method", "dct_qim", "--in", str(tmp_path / "missing.png"),
                         "--out", str(tmp_path / "o.png")]) == 1
        assert cli.main(["eval", "--method", "dct_qim", "--images", str(tmp_path / "empty"),
                         "--out", str(tmp_path / "e.csv")]) == 1
        err = capsys.readouterr().err
        assert err.count("error:") == 4
        assert "Traceback" not in err
        with pytest.raises(SystemExit) as usage:
            cli.main(["eval", "--method", "dct_qim"])
        assert usage.value.code == 2

    def test_real_subprocess_python_dash_m(self, tmp_path: Path) -> None:
        """`python -m wmkit` as a user runs it, not only the in-process entry point."""
        source = _write_image(tmp_path / "in.png")
        run = lambda *args: subprocess.run(
            [sys.executable, "-m", "wmkit", *args], capture_output=True, text=True,
            env=_subprocess_env(), cwd=tmp_path, timeout=SUBPROCESS_TIMEOUT_S, check=False,
        )
        listing = run("list")
        assert listing.returncode == 0 and "Suites" in listing.stdout
        embedded = run("embed", "--method", "dct_qim", "--in", str(source),
                       "--out", str(tmp_path / "m.png"), "--seed", "2")
        assert embedded.returncode == 0, embedded.stderr
        message = embedded.stdout.strip().splitlines()[-1]
        extracted = run("extract", "--method", "dct_qim", "--in", str(tmp_path / "m.png"),
                        "--message", message)
        assert extracted.returncode == 0 and "bit_acc: 1.0000" in extracted.stdout
        failing = run("attack", "--attack", "nope", "--in", str(source), "--out", "x.png")
        assert failing.returncode == 1 and "unknown attack" in failing.stderr
        usage = run("embed")
        assert usage.returncode == 2


# ---------------------------------------------------------------------------------------------
# Registry -> watermarker -> metrics -> attacks
# ---------------------------------------------------------------------------------------------


class TestImageChain:
    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_registry_embed_metrics_contract(self, method: str) -> None:
        watermarker = get_watermarker(method)
        image = load_sample("astronaut", (IMAGE_SIDE, IMAGE_SIDE))[None]
        message = random_messages(1, watermarker.n_bits, torch.Generator().manual_seed(0))
        marked = watermarker.embed(image, message)
        assert marked.shape == image.shape and marked.dtype == torch.float32
        assert float(marked.min()) >= 0.0 and float(marked.max()) <= 1.0
        soft = watermarker.extract(marked)
        assert soft.shape == (1, watermarker.n_bits)
        assert float(soft.min()) >= 0.0 and float(soft.max()) <= 1.0
        accuracy = bit_accuracy(soft, message)
        assert float(accuracy[0]) >= MIN_CLEAN_BIT_ACC
        n_correct = torch.round(accuracy * watermarker.n_bits)
        assert float(neg_log10_pvalue(watermarker.n_bits, n_correct)[0]) > DETECTION_NEG_LOG10_P
        assert float(psnr(marked, image)[0]) >= MIN_PSNR_DB

    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_extract_accepts_a_different_size_after_a_crop_attack(self, method: str) -> None:
        watermarker = get_watermarker(method)
        image = load_sample("coffee", (IMAGE_SIDE, IMAGE_SIDE))[None]
        message = random_messages(1, watermarker.n_bits, torch.Generator().manual_seed(1))
        marked = watermarker.embed(image, message)
        spec = get_attack("crop_0.9")
        reseed_attack(spec, attack_seed(0, "coffee", spec.name))
        cropped = spec.fn(marked)
        assert cropped.shape[-1] < marked.shape[-1]
        assert watermarker.extract(cropped).shape == (1, watermarker.n_bits)

    def test_every_attack_but_regen_vae_runs_on_a_batch_and_stays_in_range(self) -> None:
        batch = torch.stack([load_sample("astronaut", (96, 96)), load_sample("coffee", (96, 96))])
        for spec in get_suite("all"):
            if spec.name == "regen_vae":  # downloads a VAE: covered by the slow suite
                continue
            reseed_attack(spec, 0)
            attacked = spec.fn(batch)
            assert attacked.ndim == 4 and attacked.shape[:2] == (2, 3), spec.name
            assert attacked.dtype == torch.float32, spec.name
            assert float(attacked.min()) >= -1e-6 and float(attacked.max()) <= 1.0 + 1e-6, spec.name
            assert torch.isfinite(attacked).all(), spec.name

    def test_message_hex_round_trip_for_every_native_size(self) -> None:
        for n_bits in (32, 96, 256, 1024):
            message = random_messages(1, n_bits, torch.Generator().manual_seed(n_bits))[0]
            assert torch.equal(hex_to_bits(bits_to_hex(message), n_bits), message)

    def test_evaluate_matches_a_manual_chain_for_one_row(self) -> None:
        """The harness row equals embed -> attack -> extract -> metrics done by hand."""
        watermarker = get_watermarker("dct_qim")
        image = load_sample("astronaut", (128, 128))
        spec = get_attack("jpeg_50")
        frame = evaluate(watermarker, [("img", image)], [spec], seed=4, include_clean=False)
        message = random_messages(1, watermarker.n_bits, torch.Generator().manual_seed(4))
        marked = watermarker.embed(image[None], message)
        reseed_attack(spec, attack_seed(4, "img", spec.name))
        accuracy = float(bit_accuracy(watermarker.extract(spec.fn(marked)), message)[0])
        assert frame.loc[0, "bit_acc"] == pytest.approx(accuracy)
        assert frame.loc[0, "psnr"] == pytest.approx(float(psnr(marked, image[None])[0]))


# ---------------------------------------------------------------------------------------------
# Text watermarking without an LM
# ---------------------------------------------------------------------------------------------

TEXT_VOCAB = 600
TEXT_LENGTH = 220
TEXT_LOGIT_SCALE = 2.0
TEXT_PROMPT = [1, 2, 3]


def _oracle(vocab: int = TEXT_VOCAB) -> tuple[object, Tensor]:
    table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(11)) * TEXT_LOGIT_SCALE

    def next_logits(prefix: Tensor) -> Tensor:
        return table[prefix[:, -1]]

    return next_logits, table


def _generate(scheme: str, config: object, seed: int = 0, n: int = 1) -> list[list[int]]:
    next_logits, _ = _oracle()
    return sample_sequences(
        next_logits,  # type: ignore[arg-type]
        TEXT_PROMPT, n_sequences=n, scheme=scheme,  # type: ignore[arg-type]
        config=config, max_new_tokens=TEXT_LENGTH, seed=seed, min_new_tokens=TEXT_LENGTH,
    )


class TestTextChain:
    def test_kgw_detects_watermarked_and_not_plain_text(self) -> None:
        config = KGWConfig()
        marked = _generate("kgw", config, seed=1)[0]
        plain = _generate("none", None, seed=1)[0]
        assert len(marked) == TEXT_LENGTH
        hit = kgw_detect(marked, TEXT_VOCAB, config)
        miss = kgw_detect(plain, TEXT_VOCAB, config)
        assert hit.detected and hit.p_value < 1e-3 and hit.n_green is not None
        assert hit.n_green / hit.n_scored > 0.5
        assert not miss.detected
        assert hit.neg_log10_p > miss.neg_log10_p

    def test_kgw_wrong_key_does_not_detect(self) -> None:
        marked = _generate("kgw", KGWConfig(), seed=2)[0]
        other = KGWConfig(hash_key=KGWConfig().hash_key + 1)
        assert not kgw_detect(marked, TEXT_VOCAB, other).detected

    def test_logits_processor_equals_sample_sequences_bias(self) -> None:
        """The HF LogitsProcessor adds delta to exactly the green list of the context."""
        config = KGWConfig()
        processor = KGWLogitsProcessor(TEXT_VOCAB, config)
        scores = torch.zeros(1, TEXT_VOCAB)
        biased = processor(torch.tensor([[5, 9]]), scores.clone())
        green = (biased - scores)[0]
        assert set(green.unique().tolist()) == {0.0, config.delta}
        assert int((green > 0).sum()) == round(config.gamma * TEXT_VOCAB)

    def test_gumbel_detects_and_is_deterministic_given_key(self) -> None:
        config = GumbelConfig(hash_key=123)
        first = _generate("gumbel", config, seed=0)[0]
        second = _generate("gumbel", config, seed=99)[0]
        assert first == second  # contract: depends on key and context only
        assert gumbel_detect(first, TEXT_VOCAB, config).detected
        assert not gumbel_detect(_generate("none", None, seed=3)[0], TEXT_VOCAB, config).detected

    def test_detection_survives_moderate_token_attacks_and_dies_under_heavy_ones(self) -> None:
        config = KGWConfig()
        marked = _generate("kgw", config, seed=5)[0]
        rng = np.random.default_rng(0)
        light = substitute_tokens(marked, 0.1, TEXT_VOCAB, rng)
        assert kgw_detect(light, TEXT_VOCAB, config).detected
        heavy = substitute_tokens(marked, 1.0, TEXT_VOCAB, np.random.default_rng(1))
        assert not kgw_detect(heavy, TEXT_VOCAB, config).detected
        assert len(delete_tokens(marked, 0.1, rng)) == len(marked) - round(0.1 * len(marked))
        assert len(insert_tokens(marked, 0.1, TEXT_VOCAB, rng)) == len(marked) + round(
            0.1 * len(marked)
        )

    def test_localize_finds_the_watermarked_middle_of_a_mixed_text(self) -> None:
        config = KGWConfig()
        marked = _generate("kgw", config, seed=6)[0]
        plain = _generate("none", None, seed=7)[0]
        mixed = plain[:100] + marked[:120] + plain[100:200]
        scores = token_scores(mixed, TEXT_VOCAB, config)
        assert scores.shape == (len(mixed),)
        segments = localize_segments(mixed, TEXT_VOCAB, config)
        assert segments, "no segment found"
        start, end, p_value = max(segments, key=lambda s: s[1] - s[0])
        overlap = max(0, min(end, 220) - max(start, 100))
        assert overlap / 120 >= 0.6, segments
        assert p_value < 1e-3


# ---------------------------------------------------------------------------------------------
# MET against the official package, on data independent from the unit tests
# ---------------------------------------------------------------------------------------------

MET_PROMPTS = 4
MET_VOCAB = 6
MET_LENGTH = 10


def _met_samples(
    n: int, rng: np.random.Generator, token_weights: np.ndarray, variable_length: bool
) -> list[Sample]:
    samples = []
    for _ in range(n):
        prompt = int(rng.integers(MET_PROMPTS))
        length = int(rng.integers(MET_LENGTH // 2, MET_LENGTH + 1)) if variable_length else MET_LENGTH
        tokens = rng.choice(MET_VOCAB, size=length, p=token_weights[prompt])
        samples.append(Sample(prompt, tuple(int(t) for t in tokens)))
    return samples


def _law(rng: np.random.Generator) -> np.ndarray:
    weights = rng.dirichlet(np.ones(MET_VOCAB) * 0.6, size=MET_PROMPTS)
    return np.asarray(weights)


def _official(samples: Sequence[Sample]) -> object:
    from model_equality_testing.distribution import CompletionSample

    padded = np.full((len(samples), MET_LENGTH), -1, dtype=np.int64)
    for row, sample in enumerate(samples):
        padded[row, : len(sample.completion_ids)] = sample.completion_ids
    prompts = np.array([s.prompt_id for s in samples])
    return CompletionSample(prompts=prompts, completions=padded, m=MET_PROMPTS)


class TestMetAgainstOfficial:
    @pytest.mark.parametrize("variable_length", [False, True])
    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_statistic_is_bit_for_bit_the_official_one(self, seed: int, variable_length: bool) -> None:
        pytest.importorskip("model_equality_testing")
        from model_equality_testing.algorithm import run_two_sample_test

        rng = np.random.default_rng(seed)
        samples_p = _met_samples(70, rng, _law(rng), variable_length)
        samples_q = _met_samples(90, rng, _law(rng), variable_length)
        _, official_stat = run_two_sample_test(
            _official(samples_p), _official(samples_q), pvalue_type="dummy",
            stat_type="mmd_hamming",
        )
        assert mmd_hamming(samples_p, samples_q, completion_len=MET_LENGTH) == pytest.approx(
            float(official_stat), abs=1e-12
        )

    def test_decisions_agree_under_h0_and_under_a_clear_h1(self) -> None:
        pytest.importorskip("model_equality_testing")
        from model_equality_testing.algorithm import run_two_sample_test

        rng = np.random.default_rng(21)
        law = _law(rng)
        other = _law(rng)
        cases = {
            f"h0-{index}": (_met_samples(150, rng, law, False), _met_samples(150, rng, law, False))
            for index in range(3)
        }
        cases["h1"] = (_met_samples(150, rng, law, False), _met_samples(150, rng, other, False))
        for name, (samples_p, samples_q) in cases.items():
            official_p, _ = run_two_sample_test(
                _official(samples_p), _official(samples_q), pvalue_type="permutation_pvalue",
                stat_type="mmd_hamming", b=300,
            )
            ours = two_sample_test(
                samples_p, samples_q, completion_len=MET_LENGTH, n_resamples=300,
                stratify_by_prompt=False,
            )
            assert ours.n_p == 150 and ours.n_q == 150
            if name == "h1":
                assert official_p < 0.02 and ours.p_value < 0.02
            # Monte Carlo error with 300 resamples is ~0.03 (1 sigma): 0.2 is a 5-sigma guard.
            assert abs(official_p - ours.p_value) < 0.2, (name, official_p, ours.p_value)

    def test_two_sample_test_is_seed_reproducible(self) -> None:
        rng = np.random.default_rng(3)
        law = _law(rng)
        samples_p, samples_q = _met_samples(40, rng, law, False), _met_samples(40, rng, law, False)
        first = two_sample_test(samples_p, samples_q, completion_len=MET_LENGTH, seed=5,
                                n_resamples=100)
        second = two_sample_test(samples_p, samples_q, completion_len=MET_LENGTH, seed=5,
                                 n_resamples=100)
        assert first == second
        assert 0.0 < first.p_value <= 1.0


class TestCachedClientFeedsMet:
    def test_cache_replays_without_new_requests_and_survives_a_new_process(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def fake_api(prompt: str) -> str:
            calls.append(prompt)
            return f"answer to {prompt} #{len(calls)}"

        cache = tmp_path / "cache" / "api.jsonl"
        client = CachedCompletionClient(fake_api, cache)
        first = client.sample("hello", 3)
        assert client.n_requests == 3 and len(calls) == 3
        assert client.sample("hello", 3) == first
        assert client.n_requests == 3 and client.n_cache_hits == 3
        reopened = CachedCompletionClient(fake_api, cache)
        assert reopened.sample("hello", 3) == first
        assert reopened.n_requests == 0 and len(calls) == 3
        lines = [json.loads(line) for line in cache.read_text().splitlines()]
        assert len(lines) == 3

    def test_rate_limit_is_respected_with_an_injected_clock(self, tmp_path: Path) -> None:
        now = [1000.0]
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            now[0] += seconds

        client = CachedCompletionClient(
            lambda prompt: "x", tmp_path / "c.jsonl", min_interval_s=900.0,
            clock=lambda: now[0], sleep=sleep,
        )
        client.sample("p", 3)
        assert len(slept) == 2 and all(s == pytest.approx(900.0) for s in slept)

    def test_api_failure_is_not_cached(self, tmp_path: Path) -> None:
        from wmkit.audit import CompletionRequestError

        state = {"fail": True}

        def flaky(prompt: str) -> str:
            if state["fail"]:
                raise RuntimeError("503")
            return "ok"

        client = CachedCompletionClient(flaky, tmp_path / "f.jsonl")
        with pytest.raises(CompletionRequestError):
            client("p")
        state["fail"] = False
        assert client("p") == "ok" and client.n_cached == 1

    def test_cached_completions_can_be_tokenised_and_tested(self, tmp_path: Path) -> None:
        """API client -> toy tokeniser -> Sample -> two_sample_test, the audit workflow."""
        vocabulary = {word: index for index, word in enumerate(["a", "b", "c", "d", "e", "f", "g", "h"])}

        def api_for(weights: Sequence[float]) -> object:
            rng = np.random.default_rng(0)

            def call(prompt: str) -> str:
                words = rng.choice(list(vocabulary), size=MET_LENGTH, p=list(weights))
                return " ".join(words)

            return call

        uniform = np.ones(8) / 8
        skewed = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
        samples: dict[str, list[Sample]] = {}
        for name, weights in (("p", uniform), ("q", skewed)):
            client = CachedCompletionClient(api_for(weights), tmp_path / f"{name}.jsonl")  # type: ignore[arg-type]
            samples[name] = [
                Sample(prompt_id, tuple(vocabulary[w] for w in client(f"prompt {prompt_id}", rep).split()))
                for prompt_id in range(2) for rep in range(60)
            ]
        result = two_sample_test(samples["p"], samples["q"], completion_len=MET_LENGTH,
                                 n_resamples=200)
        assert result.p_value < 0.02


# ---------------------------------------------------------------------------------------------
# Unlearning round trip
# ---------------------------------------------------------------------------------------------

UNLEARN_CLASSES = 3
UNLEARN_DIM = 8
UNLEARN_PER_CLASS = 100


class _Loader:
    """Re-iterable batch loader."""

    def __init__(self, x: Tensor, y: Tensor, batch: int = 32, seed: int = 0) -> None:
        self.x, self.y, self.batch = x, y, batch
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):  # type: ignore[no-untyped-def]
        order = torch.randperm(len(self.y), generator=self.generator)
        for start in range(0, len(order), self.batch):
            index = order[start : start + self.batch]
            yield self.x[index], self.y[index]


def _blobs(per_class: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centres = torch.zeros(UNLEARN_CLASSES, UNLEARN_DIM)
    for k in range(UNLEARN_CLASSES):
        centres[k, k] = 2.5
    labels = torch.arange(UNLEARN_CLASSES).repeat_interleave(per_class)
    return centres[labels] + torch.randn(len(labels), UNLEARN_DIM, generator=generator), labels


class TestUnlearnRoundTrip:
    def test_scrub_report_and_competition_score(self) -> None:
        torch.manual_seed(0)
        x, y = _blobs(UNLEARN_PER_CLASS, 1)
        test_x, test_y = _blobs(UNLEARN_PER_CLASS, 2)
        forget = y == 0
        retain_loader = _Loader(x[~forget], y[~forget], seed=3)
        forget_loader = _Loader(x[forget], y[forget], seed=4)
        test_loader = _Loader(test_x, test_y, batch=64, seed=5)
        model = nn.Sequential(nn.Linear(UNLEARN_DIM, 32), nn.ReLU(), nn.Linear(32, UNLEARN_CLASSES))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(150):
            optimizer.zero_grad()
            torch_f.cross_entropy(model(x), y).backward()
            optimizer.step()
        model.eval()

        original = evaluate_unlearning(model, retain_loader, forget_loader, test_loader, "cpu")
        assert original.forget_acc > 0.8 and original.retain_acc > 0.8

        unlearned, history = scrub(
            copy.deepcopy(model), copy.deepcopy(model), retain_loader, forget_loader,
            ScrubConfig(lr=5e-2, msteps=2, epochs=4, alpha=0.5, gamma=1.0, kd_T=2.0, wd=0.0,
                        lr_decay_epochs=(100,)),
            "cpu",
        )
        assert isinstance(history, list) and history and isinstance(history[0], dict)
        report = evaluate_unlearning(unlearned, retain_loader, forget_loader, test_loader, "cpu")
        assert 0.0 <= report.mia_auc <= 1.0
        assert report.forget_acc < original.forget_acc  # something was actually forgotten
        assert report.retain_acc > 0.7

        baseline, _ = finetune_retain(model, retain_loader, forget_loader, device="cpu")
        assert isinstance(baseline, nn.Module)

        score = neurips2023_score(
            forget_quality_from_epsilons([0.2, 0.7, 1.2]),
            report.retain_acc, original.retain_acc, report.test_acc, original.test_acc,
        )
        assert np.isfinite(score) and score >= 0.0


# ---------------------------------------------------------------------------------------------
# Adversarial helpers on a classical method (no weights)
# ---------------------------------------------------------------------------------------------


class TestAdversarialOnClassical:
    def test_average_pattern_attack_degrades_a_fixed_pattern_watermark(self) -> None:
        from wmkit.adversarial import estimate_pattern, subtract_pattern

        watermarker = get_watermarker("spread_spectrum")
        message = random_messages(1, watermarker.n_bits, torch.Generator().manual_seed(2))
        names = ["astronaut", "coffee", "chelsea", "rocket"]
        clean = torch.stack([load_sample(n, (128, 128)) for n in names])
        marked = watermarker.embed(clean, message.expand(len(names), -1))
        accuracy = bit_accuracy(watermarker.extract(marked), message.expand(len(names), -1))
        assert float(accuracy.min()) >= MIN_CLEAN_BIT_ACC
        pattern = estimate_pattern(marked)
        assert pattern.shape == marked.shape[1:]
        attacked = subtract_pattern(marked, pattern)
        assert attacked.shape == marked.shape
        after = bit_accuracy(watermarker.extract(attacked), message.expand(len(names), -1))
        assert float(after.mean()) <= float(accuracy.mean())
