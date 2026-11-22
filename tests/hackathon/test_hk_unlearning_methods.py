"""Tests of the configs, training helpers and the three unlearning methods (tiny ViT, CPU)."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.common.errors import ConfigError
from hackathon.unlearning.configs import (
    PRESETS,
    UnlearnConfig,
    build_config,
    coerce_value,
    config_to_dict,
    load_config,
    with_overrides,
)
from hackathon.unlearning.methods import (
    UnlearnError,
    gold_soft_unlearn,
    model_classes,
    scrub_unlearn,
    unlearn,
    unlearn_with_history,
)
from hackathon.unlearning.proxy_eval import evaluate_proxy
from hackathon.unlearning.training import (
    epoch_batches,
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    prepare_student,
    steps_per_epoch,
    temper_probabilities,
)

TINY_RUN = {
    "batch_size": 4, "epochs": 2, "lr": 1e-3, "max_steps_per_epoch": 3, "target_views": 2,
    "scrub_msteps": 1, "warmup_fraction": 0.0,
}  # fmt: skip


def _config(method: str, **extra: object) -> UnlearnConfig:
    return build_config({**TINY_RUN, "method": method, **extra})


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


# ----- configs -------------------------------------------------------------------------------


def test_default_config_is_valid_and_presets_build() -> None:
    assert UnlearnConfig().method == "gold_soft"
    for name in PRESETS:
        assert load_config(name).epochs >= 1


def test_smoke_preset_is_small() -> None:
    smoke = load_config("smoke")
    assert smoke.forget_subset is not None and smoke.forget_subset <= 128
    assert smoke.max_steps_per_epoch is not None and smoke.epochs == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"method": "gradient_ascent"},
        {"batch_size": 0},
        {"lr": 0.0},
        {"warmup_fraction": 1.0},
        {"kd_temperature": -1.0},
        {"epochs": 1, "scrub_msteps": 2},
    ],
)
def test_invalid_configs_raise_config_error(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        build_config(bad)


def test_overrides_are_typed_from_strings() -> None:
    config = load_config("smoke", {"lr": "2e-5", "amp": "false", "device": "none", "epochs": "2",
                                    "scrub_msteps": "1"})  # fmt: skip
    assert config.lr == 2e-5 and config.amp is False and config.device is None
    assert config.epochs == 2
    assert coerce_value("forget_subset", "none") is None
    assert coerce_value("forget_subset", "12") == 12


def test_unknown_field_and_bad_number_are_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown config field"):
        build_config({"learning_rate": 1})
    with pytest.raises(ConfigError, match="Cannot parse"):
        coerce_value("epochs", "many")
    with pytest.raises(ConfigError, match="Cannot parse"):
        coerce_value("amp", "maybe")


def test_load_config_from_json_file_and_errors(tmp_path: Path) -> None:
    good = tmp_path / "c.json"
    good.write_text(json.dumps({"method": "scrub", "epochs": 2, "scrub_msteps": 2}), "utf-8")
    assert load_config(str(good)).method == "scrub"
    arr = tmp_path / "a.json"
    arr.write_text("[1]", "utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(str(arr))
    with pytest.raises(ConfigError, match="neither a preset"):
        load_config(str(tmp_path / "nope.json"))
    broken = tmp_path / "b.json"
    broken.write_text("{", "utf-8")
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(str(broken))


def test_config_round_trip_through_dict() -> None:
    config = _config("scrub")
    assert build_config(config_to_dict(config)) == config
    assert with_overrides(config, epochs=2).epochs == 2


# ----- training helpers ------------------------------------------------------------------------


def test_learning_rate_schedule_warms_up_then_decays_to_zero() -> None:
    total, peak = 100, 1e-3
    rates = [learning_rate_at(s, total, peak, 0.1) for s in range(total)]
    assert rates[0] < rates[9] <= peak
    assert math.isclose(rates[10], peak)
    assert all(a >= b for a, b in itertools.pairwise(rates[10:]))
    assert rates[-1] < peak * 1e-2
    assert learning_rate_at(0, 10, peak, 0.0) == peak


def test_steps_per_epoch_and_batches() -> None:
    config = _config("gold_soft")
    assert steps_per_epoch(10, config) == 3  # capped by max_steps_per_epoch
    assert steps_per_epoch(5, with_overrides(config, max_steps_per_epoch=None)) == 2
    assert steps_per_epoch(1, with_overrides(config, max_steps_per_epoch=None)) == 1
    generator = torch.Generator().manual_seed(0)
    batches = list(epoch_batches(10, 4, 5, generator))
    assert len(batches) == 5 and all(1 <= len(b) <= 4 for b in batches)
    first_pass = torch.cat(batches[:3]).sort().values
    assert first_pass.tolist() == list(range(10))


def test_temper_probabilities_flattens_and_keeps_the_ranking() -> None:
    p = torch.softmax(torch.tensor([[6.0, 3.0, 0.0]]), dim=1)
    assert torch.allclose(temper_probabilities(p, 1.0), p, atol=1e-6)
    flat = temper_probabilities(p, 3.0)
    assert float(flat.max()) < float(p.max())
    assert flat.argsort().tolist() == p.argsort().tolist()
    assert math.isclose(float(flat.sum()), 1.0, rel_tol=1e-6)


def test_kd_from_teacher_is_zero_for_equal_logits_and_scales_with_t_squared() -> None:
    z = torch.randn(4, 10)
    assert float(kd_from_teacher(z, z, 2.0)) < 1e-6
    other = torch.randn(4, 10)
    low = float(kd_from_teacher(z, other, 1.0))
    assert low > 0.0
    target = torch.softmax(z / 2.0, dim=1)
    assert math.isclose(float(kl_to_student(target, other, 2.0)),
                        float(kd_from_teacher(z, other, 2.0)), rel_tol=1e-6)  # fmt: skip


def test_prepare_student_freezes_early_blocks_and_keeps_head_trainable() -> None:
    model = tiny_model()
    student = prepare_student(model, torch.device("cpu"), _config("gold_soft", trainable_blocks=1))
    trainable = {n for n, p in student.named_parameters() if p.requires_grad}
    assert not any(n.startswith("blocks.0.") for n in trainable)
    assert any(n.startswith("blocks.1.") for n in trainable)
    assert {"head.weight", "norm.weight"} <= trainable
    assert not any(n in trainable for n in ("patch_embed.proj.weight", "cls_token", "pos_embed"))
    assert all(p.requires_grad for p in model.parameters())  # the original is untouched


# ----- methods ---------------------------------------------------------------------------------


def test_identity_returns_an_equal_but_distinct_model() -> None:
    model = tiny_model()
    out = unlearn(model, random_images(4), _config("identity"))
    assert out is not model
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                 out.state_dict().values(), strict=True))  # fmt: skip


@pytest.mark.parametrize("method", ["gold_soft", "scrub"])
def test_training_methods_change_weights_keep_architecture_and_spare_the_input(
    method: str,
) -> None:
    model = tiny_model()
    before = _state(model)
    forget, retain = random_images(8), random_images(6, seed=50)
    out, history = unlearn_with_history(model, forget, _config(method), retain)
    after = out.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert len(history) == 2 and not out.training


def test_gold_soft_is_reproducible_for_a_seed() -> None:
    forget, retain = random_images(8), random_images(6, seed=50)
    first = unlearn(tiny_model(), forget, _config("gold_soft"), retain)
    second = unlearn(tiny_model(), forget, _config("gold_soft"), retain)
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(),
                                                 second.state_dict().values(), strict=True))  # fmt: skip
    third = unlearn(tiny_model(), forget, _config("gold_soft", seed=5), retain)
    assert any(not torch.equal(a, b) for a, b in zip(first.state_dict().values(),
                                                     third.state_dict().values(), strict=True))  # fmt: skip


def test_gold_soft_moves_forget_outputs_toward_softer_targets() -> None:
    original = tiny_model()
    forget, retain = random_images(8), random_images(6, seed=50)
    config = _config("gold_soft", epochs=4, max_steps_per_epoch=2, lr=3e-3,
                     target_temperature=2.0)  # fmt: skip
    trained = unlearn(original, forget, config, retain)
    report = evaluate_proxy(original, trained, forget, retain, variants=("flip",))
    assert report.forget_kl_mean > 0.0  # it moved on the forget images


def test_scrub_max_step_is_capped_so_the_divergence_stays_finite() -> None:
    config = _config("scrub", scrub_kl_cap=0.05, lr=5e-3, epochs=3, scrub_msteps=3)
    out, history = scrub_unlearn(tiny_model(), random_images(8), config, random_images(6, 50))
    assert all(math.isfinite(row["max_kl"]) for row in history)
    assert all(torch.isfinite(v).all() for v in out.state_dict().values())


def test_subset_is_applied_and_empty_forget_is_rejected() -> None:
    config = _config("gold_soft", forget_subset=4, epochs=1, max_steps_per_epoch=None)
    _, history = unlearn_with_history(tiny_model(), random_images(12), config, random_images(4, 9))
    assert len(history) == 1
    with pytest.raises(UnlearnError, match="empty"):
        unlearn(tiny_model(), random_images(0), _config("gold_soft"), random_images(2))


def test_methods_need_a_retain_source() -> None:
    with pytest.raises(ConfigError, match="retain-proxy"):
        gold_soft_unlearn(tiny_model(), random_images(4), _config("gold_soft"), None)
    with pytest.raises(UnlearnError, match="empty"):
        gold_soft_unlearn(tiny_model(), random_images(4), _config("gold_soft"), random_images(0))


def test_retain_images_are_loaded_from_proxy_root(tmp_path: Path) -> None:
    from PIL import Image

    (tmp_path / "kodak").mkdir()
    for index in range(3):
        Image.new("RGB", (240, 240), (index * 40, 90, 200)).save(
            tmp_path / "kodak" / f"{index}.png"
        )
    config = _config("scrub", proxy_root=str(tmp_path), retain_count=3, epochs=1, scrub_msteps=1)
    out = unlearn(tiny_model(), random_images(4), config)
    assert out is not None


def test_model_classes_reads_the_head_and_rejects_headless_models() -> None:
    assert model_classes(tiny_model()) == 100
    with pytest.raises(UnlearnError, match="head"):
        model_classes(torch.nn.Linear(2, 2))
