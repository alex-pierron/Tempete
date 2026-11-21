"""Integration: the merged package still speaks the frozen contract (docs/CONTRACT-wmkit.md).

The expectations below are written out literally from the contract, never derived from the
code under test, so a rename, a reordering or a signature drift in any of the four lots
(imgcore, seal, textmet, unlearn) fails here. Everything is fast and offline.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
import torch
from conftest import load_sample

from wmkit import attacks as attacks_module
from wmkit import methods as methods_module
from wmkit.attacks import SUITES, get_attack, get_suite, list_attacks
from wmkit.eval import evaluate, summarize, summarize_by_attack
from wmkit.types import ATTACK_GROUPS, Watermarker

PIPELINE_IMAGE_SIDE = 128

CONTRACT_ATTACKS_BY_GROUP: dict[str, list[str]] = {
    "identity": ["identity"],
    "valuemetric": [
        "brightness_0.5", "brightness_0.75", "brightness_1.25", "brightness_1.5", "brightness_2",
        "contrast_0.5", "contrast_0.75", "contrast_1.5", "contrast_2",
        "saturation_0.5", "saturation_1.5",
        "hue_-0.1", "hue_0.1", "hue_0.25",
        "grayscale",
        "gaussian_blur_3", "gaussian_blur_9", "gaussian_blur_17",
        "gaussian_noise_0.02", "gaussian_noise_0.05",
    ],
    "compression": [
        "jpeg_40", "jpeg_50", "jpeg_60", "jpeg_70", "jpeg_80", "jpeg_90", "webp_50", "webp_80",
    ],
    "geometric": [
        "crop_0.33", "crop_0.5", "crop_0.71", "crop_0.9",
        "resize_0.5", "resize_0.75",
        "rotate_5", "rotate_10", "rotate_30", "rotate_90",
        "hflip",
        "perspective_0.1", "perspective_0.3", "perspective_0.5",
    ],
    "combined": [
        "combo_jpeg40_crop0.71_bright0.5",
        "combo_jpeg60_crop0.71_bright0.5",
        "combo_jpeg80_crop0.71_bright0.5",
    ],
    "removal": [
        "median_3", "median_5", "nlmeans_5", "nlmeans_10", "jpeg_10",
        "downup_0.25", "downup_0.5", "regen_vae",
    ],
}
CONTRACT_ATTACK_COUNT = 54
CONTRACT_QUICK_SUITE = [
    "identity", "brightness_1.5", "contrast_0.5", "hue_0.1", "gaussian_blur_9", "jpeg_50",
    "crop_0.71", "resize_0.5", "rotate_10", "hflip", "combo_jpeg40_crop0.71_bright0.5",
]
CONTRACT_PAPER_SIZE = 46
CONTRACT_REMOVAL_SIZE = 8
CONTRACT_RESULT_COLUMNS = [
    "method", "image_id", "is_watermarked", "attack", "attack_group", "n_bits", "bit_acc",
    "neg_log10_p", "detected", "psnr", "ssim", "lpips",
]
CONTRACT_METHODS = [
    "pixelseal", "videoseal", "videoseal0", "chunkyseal", "dct_qim", "spread_spectrum",
    "dwt_dct_svd", "minipixelseal",
]
CLASSICAL_METHODS = ["dct_qim", "spread_spectrum", "dwt_dct_svd"]
CLASSICAL_NATIVE_BITS = 32


def _parameter_names(function: Callable[..., Any]) -> list[str]:
    return list(inspect.signature(function).parameters)


def _defaults(function: Callable[..., Any]) -> dict[str, Any]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _keyword_only(function: Callable[..., Any]) -> list[str]:
    return [
        name for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]


class TestAttackCatalogue:
    """Section 1.1: exactly 54 attacks with the contract's names, groups and suites."""

    def test_exactly_the_54_contract_names(self) -> None:
        expected = [name for names in CONTRACT_ATTACKS_BY_GROUP.values() for name in names]
        assert len(expected) == CONTRACT_ATTACK_COUNT
        assert len(set(expected)) == CONTRACT_ATTACK_COUNT
        assert sorted(list_attacks()) == sorted(expected)

    @pytest.mark.parametrize("group", ATTACK_GROUPS)
    def test_group_membership(self, group: str) -> None:
        assert sorted(list_attacks(group)) == sorted(CONTRACT_ATTACKS_BY_GROUP[group])  # type: ignore[arg-type]
        for name in CONTRACT_ATTACKS_BY_GROUP[group]:
            assert get_attack(name).group == group
            assert get_attack(name).name == name

    def test_group_constant_matches_the_contract(self) -> None:
        assert tuple(CONTRACT_ATTACKS_BY_GROUP) == ATTACK_GROUPS

    def test_unknown_attack_is_a_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_attack("not_an_attack")

    def test_suites(self) -> None:
        assert set(SUITES) == {"quick", "paper", "removal", "all"}
        assert SUITES["quick"] == CONTRACT_QUICK_SUITE
        assert len(SUITES["paper"]) == CONTRACT_PAPER_SIZE
        assert sorted(SUITES["paper"]) == sorted(
            name for group, names in CONTRACT_ATTACKS_BY_GROUP.items() if group != "removal"
            for name in names
        )
        assert sorted(SUITES["removal"]) == sorted(CONTRACT_ATTACKS_BY_GROUP["removal"])
        assert len(SUITES["removal"]) == CONTRACT_REMOVAL_SIZE
        assert len(SUITES["all"]) == CONTRACT_ATTACK_COUNT
        assert [spec.name for spec in get_suite("quick")] == CONTRACT_QUICK_SUITE

    def test_attack_signatures(self) -> None:
        assert _parameter_names(attacks_module.get_attack) == ["name"]
        assert _parameter_names(attacks_module.list_attacks) == ["group"]
        assert _defaults(attacks_module.list_attacks) == {"group": None}
        assert _parameter_names(attacks_module.get_suite) == ["name"]
        assert _parameter_names(attacks_module.straight_through) == ["fn"]

    def test_straight_through_has_identity_gradient(self) -> None:
        wrapped = attacks_module.straight_through(lambda x: (x * 255).round() / 255)
        x = torch.rand(1, 3, 8, 8, requires_grad=True)
        wrapped(x).sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.ones_like(x))


class TestPublicSignatures:
    """Parameter names, order and defaults of every public entry point named by the contract."""

    def test_get_watermarker(self) -> None:
        assert _parameter_names(methods_module.get_watermarker)[:3] == ["name", "n_bits", "device"]
        assert _defaults(methods_module.get_watermarker) == {"n_bits": None, "device": "cpu"}

    def test_registry_keys(self) -> None:
        assert sorted(methods_module.METHODS) == sorted(CONTRACT_METHODS)
        assert sorted(methods_module.list_methods()) == sorted(CONTRACT_METHODS)

    def test_load_seal(self) -> None:
        from wmkit.seal import load_seal

        assert _parameter_names(load_seal) == [
            "n_bits", "device", "card", "scaling_w", "lowres_attenuation",
        ]
        assert _defaults(load_seal) == {
            "n_bits": None, "device": "cpu", "card": "pixelseal", "scaling_w": None,
            "lowres_attenuation": True,
        }

    def test_seal_watermarker_methods(self) -> None:
        from wmkit.seal import SealWatermarker

        for method in ("embed", "extract", "extract_logits", "embed_video", "extract_video"):
            assert callable(getattr(SealWatermarker, method))
        assert _parameter_names(SealWatermarker.embed) == ["self", "images", "messages"]
        assert _parameter_names(SealWatermarker.extract) == ["self", "images"]
        assert _parameter_names(SealWatermarker.embed_video) == ["self", "frames", "message"]

    def test_evaluate(self) -> None:
        assert _parameter_names(evaluate) == [
            "watermarker", "images", "attacks", "seed", "device", "with_lpips", "include_clean",
        ]
        assert _keyword_only(evaluate) == ["seed", "device", "with_lpips", "include_clean"]
        assert _defaults(evaluate) == {
            "seed": 0, "device": "cpu", "with_lpips": False, "include_clean": True,
        }

    def test_two_sample_test(self) -> None:
        from wmkit.audit.met import mmd_hamming, power_curve, two_sample_test

        # The contract fixes these names and defaults; extra keyword-only options are additive.
        assert _parameter_names(two_sample_test)[:2] == ["samples_p", "samples_q"]
        contract_keywords = {
            "completion_len": 50, "pvalue": "permutation", "n_resamples": 1000,
            "reference_pool": None, "seed": 0,
        }
        defaults = _defaults(two_sample_test)
        for name, value in contract_keywords.items():
            assert defaults[name] == value, name
        assert _defaults(mmd_hamming) == {"completion_len": 50}
        assert _parameter_names(power_curve)[:4] == [
            "sample_p_fn", "sample_q_fn", "budgets", "n_trials",
        ]
        assert _defaults(power_curve)["alpha"] == 0.05

    def test_kgw_detect_and_text_dataclasses(self) -> None:
        from wmkit.text.kgw import TEXT_DETECTION_P, KGWConfig, TextDetection, kgw_detect

        assert _parameter_names(kgw_detect)[:4] == [
            "token_ids", "vocab_size", "config", "dedup_ngrams",
        ]
        assert _defaults(kgw_detect) == {"dedup_ngrams": True}
        config = KGWConfig()
        assert (config.gamma, config.delta, config.context_width, config.hash_key,
                config.seeding) == (0.25, 2.0, 1, 15485863, "lefthash")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.gamma = 0.5  # type: ignore[misc]
        assert TEXT_DETECTION_P == 1e-3
        fields = list(TextDetection.__dataclass_fields__)
        assert fields[:6] == ["n_scored", "n_green", "score", "z_score", "p_value", "detected"]

    def test_gumbel_localize_attacks_generate_signatures(self) -> None:
        from wmkit.text import (
            GumbelConfig,
            delete_tokens,
            generate,
            gumbel_detect,
            insert_tokens,
            localize_segments,
            substitute_tokens,
            token_scores,
        )

        assert list(GumbelConfig.__dataclass_fields__) == ["hash_key", "context_width"]
        # A default for hash_key is an additive relaxation of the contract (hash_key, context_width=1).
        assert _defaults(GumbelConfig.__init__)["context_width"] == 1
        assert _parameter_names(gumbel_detect)[:3] == ["token_ids", "vocab_size", "config"]
        assert _parameter_names(token_scores)[:3] == ["token_ids", "vocab_size", "config"]
        assert _parameter_names(localize_segments)[:3] == ["token_ids", "vocab_size", "config"]
        assert _defaults(localize_segments)["window"] == 32
        assert _defaults(localize_segments)["p_threshold"] == 1e-3
        assert _parameter_names(substitute_tokens) == ["ids", "rate", "vocab_size", "rng"]
        assert _parameter_names(delete_tokens) == ["ids", "rate", "rng"]
        assert _parameter_names(insert_tokens) == ["ids", "rate", "vocab_size", "rng"]
        assert _parameter_names(generate)[:3] == ["model", "tokenizer", "prompt"]
        assert _keyword_only(generate)[:4] == ["scheme", "config", "max_new_tokens", "seed"]

    def test_cached_completion_client_signature(self) -> None:
        from wmkit.audit.api import CachedCompletionClient

        names = _parameter_names(CachedCompletionClient.__init__)
        assert names[:4] == ["self", "fn", "cache_path", "min_interval_s"]
        assert _defaults(CachedCompletionClient.__init__)["min_interval_s"] == 0.0

    def test_scrub_and_unlearn_signatures(self) -> None:
        from wmkit import unlearn

        assert _parameter_names(unlearn.scrub)[:6] == [
            "student", "teacher", "retain_loader", "forget_loader", "config", "device",
        ]
        assert "rewind_val_loader" in _keyword_only(unlearn.scrub)
        assert _defaults(unlearn.scrub)["rewind_val_loader"] is None
        for baseline in (unlearn.finetune_retain, unlearn.negative_gradient,
                         unlearn.neggrad_plus, unlearn.random_labels):
            assert _parameter_names(baseline)[:3] == ["model", "retain_loader", "forget_loader"]
        assert _parameter_names(unlearn.evaluate_unlearning)[:5] == [
            "model", "retain_loader", "forget_loader", "test_loader", "device",
        ]
        assert list(unlearn.UnlearnReport.__dataclass_fields__)[:4] == [
            "retain_acc", "forget_acc", "test_acc", "mia_auc",
        ]
        assert _parameter_names(unlearn.neurips2023_score) == [
            "forget_quality", "ra_u", "ra_r", "ta_u", "ta_r",
        ]
        assert _parameter_names(unlearn.forget_quality_from_epsilons) == ["eps"]

    def test_scrub_config_values_of_the_contract(self) -> None:
        from wmkit.unlearn import ScrubConfig

        small, large = ScrubConfig(), ScrubConfig.large()
        assert (small.lr, small.msteps, small.epochs, small.alpha, small.gamma, small.kd_T,
                small.wd) == (5e-4, 2, 3, 0.001, 0.99, 4.0, 5e-4)
        assert (large.msteps, large.epochs, large.alpha, large.gamma, large.kd_T,
                large.wd) == (3, 10, 0.5, 1.0, 2.0, 0.1)

    def test_core_helper_signatures(self) -> None:
        from wmkit import io, messages, metrics

        assert _parameter_names(io.list_images) == ["directory", "max_images"]
        assert _parameter_names(messages.random_messages) == ["batch", "n_bits", "generator"]
        assert _parameter_names(messages.hex_to_bits) == ["hex_str", "n_bits"]
        assert _parameter_names(metrics.neg_log10_pvalue) == ["n_bits", "n_correct"]
        assert _parameter_names(metrics.tpr_at_fpr) == ["scores_pos", "scores_neg", "fpr"]
        assert metrics.DETECTION_NEG_LOG10_P == 6.0

    def test_adversarial_and_video_signatures(self) -> None:
        from wmkit import adversarial, video

        assert _parameter_names(adversarial.pgd_remove)[:2] == ["extract_logits", "images"]
        assert _defaults(adversarial.pgd_remove) == {
            "eps": 4 / 255, "step": 1 / 255, "steps": 50, "target_bits": None,
        }
        assert _parameter_names(adversarial.pgd_forge)[:3] == [
            "extract_logits", "images", "target_message",
        ]
        assert _defaults(adversarial.estimate_pattern) == {"denoise": "gaussian"}
        assert _defaults(video.write_video) == {"codec": "h264", "crf": 23}
        assert _parameter_names(video.h264_attack) == ["frames", "crf"]


@pytest.fixture(scope="module")
def classical_images() -> list[tuple[str, torch.Tensor]]:
    size = (PIPELINE_IMAGE_SIDE, PIPELINE_IMAGE_SIDE)
    return [("astronaut", load_sample("astronaut", size)), ("coffee", load_sample("coffee", size))]


class TestEvaluateContract:
    """`evaluate` output on every classical baseline: columns, ordering, row counts, NaN policy."""

    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_frame_layout(
        self, method: str, classical_images: list[tuple[str, torch.Tensor]]
    ) -> None:
        watermarker = methods_module.get_watermarker(method)
        assert isinstance(watermarker, Watermarker)
        assert watermarker.n_bits == CLASSICAL_NATIVE_BITS
        quick = get_suite("quick")
        frame = evaluate(watermarker, classical_images, quick, seed=1)

        assert list(frame.columns) == CONTRACT_RESULT_COLUMNS
        assert len(frame) == len(classical_images) * len(quick) * 2
        # One clean twin per (image, attack), same keys as the watermarked row.
        keys = ["image_id", "attack"]
        watermarked = frame[frame["is_watermarked"]][keys].sort_values(keys).reset_index(drop=True)
        clean = frame[~frame["is_watermarked"]][keys].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(watermarked, clean)
        assert frame["method"].eq(method).all()
        assert frame["n_bits"].eq(CLASSICAL_NATIVE_BITS).all()
        # NaN policy: lpips only, and only because with_lpips=False.
        assert frame["lpips"].isna().all()
        assert not frame.drop(columns=["lpips"]).isna().any().any()
        # Attack group column agrees with the registry.
        for attack_name, group in frame[["attack", "attack_group"]].drop_duplicates().itertuples(
            index=False
        ):
            assert get_attack(attack_name).group == group
        # Value ranges.
        assert frame["bit_acc"].between(0.0, 1.0).all()
        assert (frame["neg_log10_p"] >= 0.0).all()
        assert frame["detected"].dtype == bool and frame["is_watermarked"].dtype == bool

    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_identity_row_and_clean_false_positive_rate(
        self, method: str, classical_images: list[tuple[str, torch.Tensor]]
    ) -> None:
        watermarker = methods_module.get_watermarker(method)
        frame = evaluate(watermarker, classical_images, get_suite("quick"), seed=1)
        identity = frame[(frame["attack"] == "identity") & frame["is_watermarked"]]
        assert (identity["bit_acc"] >= 0.95).all(), identity[["image_id", "bit_acc"]]
        assert identity["detected"].all()
        clean = frame[~frame["is_watermarked"]]
        assert not clean["detected"].any(), "false positive on a clean image"

    def test_summaries_use_contract_group_keys(
        self, classical_images: list[tuple[str, torch.Tensor]]
    ) -> None:
        watermarker = methods_module.get_watermarker("dct_qim")
        frame = evaluate(watermarker, classical_images, get_suite("quick"), seed=0)
        by_group = summarize(frame)
        by_attack = summarize_by_attack(frame)
        assert list(by_group.columns[:3]) == ["method", "is_watermarked", "attack_group"]
        assert list(by_attack.columns[:3]) == ["method", "is_watermarked", "attack"]
        assert len(by_attack) == 2 * len(get_suite("quick"))
        assert set(by_group["attack_group"]) == {
            "identity", "valuemetric", "compression", "geometric", "combined",
        }

    def test_same_seed_is_reproducible_and_different_seed_changes_messages(
        self, classical_images: list[tuple[str, torch.Tensor]]
    ) -> None:
        watermarker = methods_module.get_watermarker("spread_spectrum")
        suite = get_suite("quick")
        first = evaluate(watermarker, classical_images, suite, seed=3)
        second = evaluate(watermarker, classical_images, suite, seed=3)
        pd.testing.assert_frame_equal(first, second)

    def test_include_clean_false_drops_the_twins(
        self, classical_images: list[tuple[str, torch.Tensor]]
    ) -> None:
        watermarker = methods_module.get_watermarker("dct_qim")
        frame = evaluate(
            watermarker, classical_images[:1], get_suite("quick")[:3], include_clean=False
        )
        assert len(frame) == 3
        assert frame["is_watermarked"].all()


class TestRegistryResolves:
    """Every registered method that is delivered resolves to a Protocol-conformant object."""

    @pytest.mark.parametrize("method", CLASSICAL_METHODS)
    def test_classical_resolves(self, method: str) -> None:
        watermarker = methods_module.get_watermarker(method, n_bits=16)
        assert watermarker.n_bits == 16
        assert watermarker.name == method

    def test_unknown_method_names_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="pixelseal"):
            methods_module.get_watermarker("nope")

    def test_minipixelseal_is_deferred_and_fails_with_import_error(self) -> None:
        """Contract 2.4: the entry exists but is not implemented (lot 3 postponed)."""
        with pytest.raises(ImportError):
            methods_module.get_watermarker("minipixelseal")
