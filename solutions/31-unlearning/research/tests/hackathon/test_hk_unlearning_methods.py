"""Tests of the configs, training helpers and the unlearning methods (tiny ViT, CPU)."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.common.errors import ConfigError
from hackathon.unlearning import methods
from hackathon.unlearning.configs import (
    PRESETS,
    UnlearnConfig,
    build_config,
    coerce_value,
    config_to_dict,
    load_config,
    with_overrides,
)
from hackathon.unlearning.data import stratified_split
from hackathon.unlearning.methods import (
    CV_SEED_OFFSET,
    SOFT_CROP_AREA,
    UnlearnError,
    _soft_copies,
    anchor_fold_indices,
    band_ascent_unlearn,
    band_hold_loss,
    band_loss,
    band_margins,
    band_targets,
    calibrated_target_logits,
    dumb_unlearn,
    gold_soft_unlearn,
    level_band_loss,
    loss_margin,
    model_classes,
    scrub_unlearn,
    trust_region_unlearn,
    trust_target_logits,
    unlearn,
    unlearn_with_history,
)
from hackathon.unlearning.model import predict
from hackathon.unlearning.proxy_eval import evaluate_proxy
from hackathon.unlearning.synthetic import save_synthetic
from hackathon.unlearning.training import (
    epoch_batches,
    kd_from_teacher,
    kl_to_student,
    learning_rate_at,
    prepare_student,
    steps_per_epoch,
    stratified_batches,
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
        {"kl_radius": 0.0},
        {"holdout_fraction": 1.0},
        {"trust_target": "gold"},
    ],
)
def test_invalid_configs_raise_config_error(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        build_config(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"trust_target": "calibrated", "trust_lambda": 1.5},
        {"trust_lambda": -0.1},
        {"trust_hedge": 1.0},
        {"trust_hedge": -0.1},
        {"radius_fraction": 0.0},
        {"radius_fraction": 1.0},
        {"altered_fraction": 1.0},
        {"altered_fraction": -0.5},
    ],
)
def test_invalid_v3_fields_raise_config_error(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        build_config(bad)


def test_v3_fields_accept_their_bounds_and_parse_from_strings() -> None:
    config = load_config("smoke", {"trust_target": "calibrated", "trust_lambda": "1",
                                    "trust_hedge": "0", "radius_fraction": "0.9",
                                    "altered_fraction": "0"})  # fmt: skip
    assert (config.trust_lambda, config.trust_hedge, config.radius_fraction) == (1.0, 0.0, 0.9)
    assert UnlearnConfig().radius_fraction is None and UnlearnConfig().altered_fraction == 0.0
    assert coerce_value("radius_fraction", "none") is None


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


def test_stratified_split_holds_out_each_class_share() -> None:
    labels = torch.tensor([0] * 10 + [1] * 20 + [2] * 5 + [3])
    kept, held = stratified_split(labels, 0.2, seed=3)
    assert sorted(torch.cat([kept, held]).tolist()) == list(range(len(labels)))
    assert len(held) == 7  # floor(36 * 0.2)
    for label, share in ((0, 2), (1, 4), (2, 1)):
        assert abs(int((labels[held] == label).sum()) - share) <= 1
    assert torch.equal(held, stratified_split(labels, 0.2, seed=3)[1])


def test_stratified_batches_span_the_classes_and_wrap_around() -> None:
    labels = torch.arange(4).repeat_interleave(8)  # 4 classes x 8 images
    batches = list(stratified_batches(labels, 8, 6, torch.Generator().manual_seed(0)))
    assert len(batches) == 6
    for batch in batches:
        assert torch.bincount(labels[batch], minlength=4).tolist() == [2, 2, 2, 2]
    assert sorted(torch.cat(batches[:4]).tolist()) == list(range(32))  # one full pass first


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


TRUST_TINY = {"target_temperature": 20.0, "kl_radius": 0.005}
"""Under the tiny model's target KL (0.006); measured only at the last of its 6 steps."""


@pytest.mark.parametrize("method", ["gold_soft", "scrub", "trust_region"])
def test_training_methods_change_weights_keep_architecture_and_spare_the_input(
    method: str,
) -> None:
    model = tiny_model()
    before = _state(model)
    forget, retain = random_images(8), random_images(6, seed=50)
    extra = TRUST_TINY if method == "trust_region" else {}
    out, history = unlearn_with_history(model, forget, _config(method, **extra), retain)
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


def test_trust_region_stops_at_the_radius() -> None:
    original = tiny_model()
    forget = random_images(8)
    probe = _config("trust_region", target_temperature=20.0, kl_radius=1e9)
    with pytest.raises(ConfigError, match="raise target_temperature"):
        trust_region_unlearn(original, forget, probe, None)
    config = _config(
        "trust_region", target_temperature=20.0, kl_radius=1e-7, epochs=8, max_steps_per_epoch=3
    )
    out, history = trust_region_unlearn(original, forget, config, None)
    last = history[-1]
    assert last["radius_reached"] == 1.0 and last["steps"] == 10.0  # first check, not 24 steps
    assert last["forget_kl_estimate"] >= 1e-7 and last["forget_kl"] >= 1e-7 and not out.training
    report = evaluate_proxy(original, out, forget, random_images(2, seed=9), variants=())
    assert report.groups[0].kl_orig_sub > 0.0
    every_step = with_overrides(config, radius_check_every=1)
    _, history = trust_region_unlearn(original, forget, every_step, None)
    assert history[-1]["steps"] == 1.0, "checked after every step: stops at the first one"


TRUST_VIEWS = {**TRUST_TINY, "trust_target": "views"}


def test_trust_target_views_are_distributions_drawn_from_the_seed_and_v1_is_unchanged() -> None:
    model, forget = tiny_model(), random_images(8)
    logits = predict(model, forget, 4)
    views = trust_target_logits(model, forget, logits, _config("trust_region", **TRUST_VIEWS))
    assert views.shape == logits.shape
    assert torch.allclose(views.exp().sum(dim=1), torch.ones(8), atol=1e-6)
    again = trust_target_logits(model, forget, logits, _config("trust_region", **TRUST_VIEWS))
    reseeded = _config("trust_region", **TRUST_VIEWS, seed=5)
    assert torch.equal(views, again)
    assert not torch.equal(views, trust_target_logits(model, forget, logits, reseeded))
    tempered = trust_target_logits(model, forget, logits, _config("trust_region", **TRUST_TINY))
    assert torch.equal(tempered, logits / TRUST_TINY["target_temperature"])  # V1 target


def test_trust_region_views_rejects_a_target_inside_the_radius() -> None:
    # The tiny model is near-uniform: at T = 1 its views barely move it, so the target KL is ~0.
    config = _config("trust_region", trust_target="views", target_temperature=1.0, kl_radius=0.5)
    with pytest.raises(ConfigError, match="views target .* lower target_view_min_area"):
        trust_region_unlearn(tiny_model(), random_images(8), config, None)


def test_trust_region_views_trains_and_stops_at_the_radius() -> None:
    original, forget = tiny_model(), random_images(8)
    before = _state(original)
    stop = {"kl_radius": 1e-7, "epochs": 8, "max_steps_per_epoch": 3}
    views = _config("trust_region", **{**TRUST_VIEWS, **stop})
    out, history = unlearn_with_history(original, forget, views)
    v1 = with_overrides(views, trust_target="tempered")
    _, tempered = trust_region_unlearn(original, forget, v1, None)
    last = history[-1]
    assert last["radius_reached"] == 1.0 and last["steps"] == 10.0 and not out.training
    assert last["target_kl"] > 1e-7 and last["target_kl"] != tempered[-1]["target_kl"]
    assert 0.0 <= last["target_agreement"] <= 1.0
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in out.state_dict().items())


def _calibrated_reference(
    original: torch.Tensor, views: list[torch.Tensor], lam: float, hedge: float, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(p_dir, l*) recomputed from the formulas of the decision, float64."""
    labels = original.argmax(dim=1)
    p_hat = torch.stack([torch.softmax(v.double() / 2.0, dim=1) for v in views]).mean(dim=0)
    l_bar = torch.stack([torch_f.cross_entropy(v.double(), labels, reduction="none")
                         for v in views]).mean(dim=0)  # fmt: skip
    p_dir = (1 - hedge) * p_hat + hedge * torch.softmax(original.double() / tau, dim=1)
    l_mix = torch_f.cross_entropy(2.0 * p_dir.log(), labels, reduction="none")
    return p_dir, l_mix + lam * (l_bar - l_mix)


@pytest.mark.parametrize(("lam", "hedge"), [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (1.0, 0.5)])
def test_calibrated_target_hits_l_star_and_keeps_the_wrong_class_ratios(
    lam: float, hedge: float
) -> None:
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(100, (16,), generator=generator)
    original = torch.randn(16, 100, generator=generator)
    original[torch.arange(16), labels] += 15.0  # memorised-looking: margin ~12 logits
    views = [original + 4.0 * torch.randn(16, 100, generator=generator) for _ in range(4)]
    config = _config("trust_region", trust_target="calibrated", target_views=4,
                     trust_lambda=lam, trust_hedge=hedge, target_temperature=6.0)  # fmt: skip
    logits = calibrated_target_logits(original, views, config)
    p_dir, l_star = _calibrated_reference(original, views, lam, hedge, 6.0)
    assert logits.dtype == original.dtype and logits.shape == original.shape
    loss = torch_f.cross_entropy(logits.double(), labels, reduction="none")
    assert torch.allclose(loss, l_star, atol=1e-5, rtol=0.0)  # CE at T = 1 is l*
    wrong = torch.ones_like(p_dir, dtype=torch.bool)
    wrong[torch.arange(16), labels] = False
    tempered = torch.softmax(logits.double() / 2.0, dim=1)[wrong].view(16, 99)
    direction = p_dir[wrong].view(16, 99)
    ratios = tempered / tempered.sum(dim=1, keepdim=True)
    assert torch.allclose(ratios, direction / direction.sum(dim=1, keepdim=True), rtol=1e-4)
    for temperature in (1.0, 2.0):
        rows = torch.softmax(logits / temperature, dim=1).sum(dim=1)
        assert torch.allclose(rows, torch.ones(16), atol=1e-6)


TRUST_CALIBRATED = {**TRUST_TINY, "trust_target": "calibrated", "radius_fraction": 0.5}


def test_trust_target_calibrated_is_drawn_from_the_seed() -> None:
    model, forget = tiny_model(), random_images(8)
    logits = predict(model, forget, 4)
    config = _config("trust_region", **TRUST_CALIBRATED)
    first = trust_target_logits(model, forget, logits, config)
    assert torch.equal(first, trust_target_logits(model, forget, logits, config))
    reseeded = with_overrides(config, seed=5)
    assert not torch.equal(first, trust_target_logits(model, forget, logits, reseeded))


def test_radius_fraction_sets_the_radius_from_the_target_and_ignores_kl_radius() -> None:
    config = _config("trust_region", **{**TRUST_CALIBRATED, "kl_radius": 1e9},
                     epochs=8, max_steps_per_epoch=3)  # fmt: skip
    out, history = trust_region_unlearn(tiny_model(), random_images(8), config, None)
    last = history[-1]
    assert last["kl_radius_effective"] == pytest.approx(0.5 * last["target_kl"])
    assert last["radius_reached"] == float(last["forget_kl"] >= last["kl_radius_effective"])
    assert last["target_loss_mean"] > 0.0 and last["target_loss_std"] >= 0.0
    assert not out.training


def test_soft_copies_replace_the_share_by_flips_and_mild_crops() -> None:
    images = random_images(8)
    out = _soft_copies(images, 0.5, torch.Generator().manual_seed(0))
    changed = [i for i in range(8) if not torch.equal(out[i], images[i])]
    assert len(changed) == 4 and out.shape == images.shape and out.dtype == torch.uint8
    flipped = [i for i in changed if torch.equal(out[i], images[i].flip(dims=(1,)))]
    assert len(flipped) == 2  # the other half are crops of SOFT_CROP_AREA
    assert SOFT_CROP_AREA == (0.70, 1.00)
    assert torch.equal(out, _soft_copies(images, 0.5, torch.Generator().manual_seed(0)))


def test_trust_region_v3_with_altered_copies_trains_reproducibly() -> None:
    original, forget = tiny_model(), random_images(8)
    before = _state(original)
    config = _config("trust_region", **TRUST_CALIBRATED, altered_fraction=0.5)
    out, history = unlearn_with_history(original, forget, config)
    again = unlearn(original, forget, config)
    exact_only = unlearn(original, forget, with_overrides(config, altered_fraction=0.0))
    after = out.state_dict()
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert all(torch.equal(after[k], v) for k, v in again.state_dict().items())
    assert any(not torch.equal(after[k], v) for k, v in exact_only.state_dict().items())
    assert len(history) == 2 and history[-1]["kl_radius_effective"] > 0.0


def test_anchor_folds_partition_the_set_and_span_the_labels() -> None:
    labels = torch.arange(30) % 7
    folds = anchor_fold_indices(labels, 3, seed=0)
    merged = torch.cat(folds)
    assert sorted(merged.tolist()) == list(range(30))  # disjoint and complete
    for label in range(7):
        counts = [int((labels[fold] == label).sum()) for fold in folds]
        assert max(counts) - min(counts) <= 1  # stratified to within one image
    again = anchor_fold_indices(labels, 3, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(folds, again, strict=True))
    assert not all(
        torch.equal(a, b) for a, b in zip(folds, anchor_fold_indices(labels, 3, 1), strict=True)
    )


def test_anchor_config_is_validated() -> None:
    for folds in (1, -2):
        with pytest.raises(ConfigError, match="anchor_folds"):
            _config("trust_region", anchor_folds=folds)
    with pytest.raises(ConfigError, match="anchor_weight"):
        _config("trust_region", anchor_weight=0.0)
    with pytest.raises(ConfigError, match="anchor_views"):
        _config("trust_region", anchor_views=1.5)


def test_radius_fraction_is_taken_on_the_probe_in_v4_and_on_the_set_in_v1_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(methods, "RADIUS_CHECK_IMAGES", 4)
    original, forget = tiny_model(), random_images(8)
    config = _config("trust_region", **TRUST_CALIBRATED, epochs=1, max_steps_per_epoch=1)
    logits = predict(original, forget, 4)
    target = trust_target_logits(original, forget, logits, config)
    probe = torch.randperm(8, generator=torch.Generator().manual_seed(config.seed))[:4]
    probe_kl = kl_to_student(torch.softmax(logits[probe], dim=1), target[probe], 1.0)
    set_kl = kl_to_student(torch.softmax(logits, dim=1), target, 1.0)
    _, v3 = trust_region_unlearn(original, forget, config, None)
    dual = with_overrides(config, radius_dual_lr=0.5)
    _, v4 = trust_region_unlearn(original, forget, dual, None)
    assert v3[-1]["kl_radius_effective"] == pytest.approx(0.5 * float(set_kl), rel=1e-5)
    assert v4[-1]["kl_radius_effective"] == pytest.approx(0.5 * float(probe_kl), rel=1e-5)


def test_trust_region_v4_forgets_fold_after_fold_with_an_anchor() -> None:
    original, forget = tiny_model(), random_images(12)
    before = _state(original)
    config = _config("trust_region", **TRUST_CALIBRATED, altered_fraction=0.5, anchor_folds=3,
                     anchor_views=0.5, epochs=3, radius_check_every=1)  # fmt: skip
    out, history = unlearn_with_history(original, forget, config)
    again, _ = unlearn_with_history(original, forget, config)
    unanchored = unlearn(original, forget, with_overrides(config, anchor_folds=0))
    exact_anchor = unlearn(original, forget, with_overrides(config, anchor_views=0.0))
    after, last = out.state_dict(), history[-1]
    assert sorted({row["phase"] for row in history}) == [1.0, 2.0, 3.0]
    # never-pushed folds exist in phases 1-2, already forgotten ones in phases 2-3
    assert {k for k in last if k.startswith(("pending_kl_p", "forgotten_kl_p"))} == {
        "pending_kl_p1", "pending_kl_p2", "forgotten_kl_p2", "forgotten_kl_p3"}  # fmt: skip
    assert last["pending_kl"] == last["pending_kl_p2"] >= 0.0
    assert all(f"radius_p{k}" in last and f"reached_p{k}" in last for k in (1, 2, 3))
    assert last["radius_reached"] == min(last[f"reached_p{k}"] for k in (1, 2, 3))
    assert last["radius_ratio"] == pytest.approx(last["forget_kl"] / last["kl_radius_effective"])
    steps = [row["steps"] for row in history]
    assert steps == sorted(steps) and history[-1]["steps"] > history[0]["steps"]  # cumulative
    assert history[-1]["forget_kl"] > 0.0 and not out.training
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert all(torch.equal(after[k], v) for k, v in again.state_dict().items())
    assert any(not torch.equal(after[k], v) for k, v in unanchored.state_dict().items())
    assert any(not torch.equal(after[k], v) for k, v in exact_anchor.state_dict().items())


def test_v4_anchor_holds_the_folds_never_pushed() -> None:
    original, forget = tiny_model(), random_images(12)
    config = _config("trust_region", **TRUST_CALIBRATED, anchor_folds=2, epochs=2,
                     radius_check_every=1, lr=3e-3)  # fmt: skip
    _, held = trust_region_unlearn(
        original, forget, with_overrides(config, anchor_weight=100.0), None
    )
    _, loose = trust_region_unlearn(
        original, forget, with_overrides(config, anchor_weight=1e-9), None
    )
    assert held[-1]["pending_kl"] < loose[-1]["pending_kl"]
    with pytest.raises(ConfigError, match="exceeds the 12 forget images"):
        trust_region_unlearn(original, forget, with_overrides(config, anchor_folds=13), None)


def test_radius_dual_runs_the_whole_schedule_with_the_radius_as_a_constraint() -> None:
    original, forget = tiny_model(), random_images(12)
    stop = _config("trust_region", **TRUST_CALIBRATED, epochs=3, radius_check_every=1)
    _, stopped = trust_region_unlearn(original, forget, stop, None)
    out, history = trust_region_unlearn(
        original, forget, with_overrides(stop, radius_dual_lr=1.0), None
    )
    assert stopped[-1]["steps"] < 9.0, "without the dual mode the radius stops the run early"
    assert history[-1]["steps"] == 9.0 and len(history) == 3  # every epoch runs
    assert all(row["dual_multiplier"] >= 0.0 for row in history)
    assert any(row["dual_multiplier"] > 0.0 for row in history)  # the constraint became active
    assert all(row["dual_penalty"] >= 0.0 for row in history) and "radius_ratio" in history[-1]
    assert "dual_multiplier" not in stopped[-1] and not out.training
    with pytest.raises(ConfigError, match="radius_dual_lr"):
        _config("trust_region", radius_dual_lr=-1.0)


@pytest.mark.parametrize("method", ["scrub", "gold_soft"])
def test_forget_views_replace_the_external_retain_images(method: str) -> None:
    original, forget = tiny_model(), random_images(8)
    config = _config(method, retain_source="forget_views", epochs=1, scrub_msteps=1)
    alone = unlearn(original, forget, config)  # no retain_images, no proxy_root: still runs
    ignored = unlearn(original, forget, config, random_images(4, seed=9))
    state = alone.state_dict()
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    assert all(torch.equal(state[k], v) for k, v in ignored.state_dict().items())
    with pytest.raises(ConfigError, match="retain_source"):
        _config(method, retain_source="coco")


def _synthetic_file(path: Path, kept_b: bool = True) -> Path:
    """A gated synthetic file: 6 images, pools A A B B C A, the second one a thermometer."""
    save_synthetic(
        path,
        {
            "images": random_images(6, seed=5),
            "labels": torch.arange(6),
            "pool": torch.tensor([0, 0, 1, 1, 2, 0], dtype=torch.uint8),
            "thermometer": torch.tensor([False, True, False, False, False, False]),
            "kept": torch.tensor([True, True, kept_b, kept_b, True, True]),
        },
        {},
    )
    return path


def test_synthetic_anchor_feeds_the_scrub_min_steps_from_a_gated_file(tmp_path: Path) -> None:
    original, forget = tiny_model(), random_images(8)
    path = _synthetic_file(tmp_path / "synthetic_gated.pt")
    config = _config(
        "scrub", retain_source="synthetic", synthetic_path=str(path), epochs=1, scrub_msteps=1
    )
    both = unlearn(original, forget, config)
    ignored = unlearn(original, forget, config, random_images(4, seed=9))  # retain_images unused
    only_b = unlearn(original, forget, with_overrides(config, synthetic_pools="B"))
    state = both.state_dict()
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    assert all(torch.equal(state[k], v) for k, v in ignored.state_dict().items())
    assert not all(torch.equal(state[k], v) for k, v in only_b.state_dict().items())
    with pytest.raises(ConfigError, match="synthetic_path"):
        _config("scrub", retain_source="synthetic")
    with pytest.raises(ConfigError, match="synthetic_pools"):
        _config("scrub", synthetic_pools="AC")
    empty = _synthetic_file(tmp_path / "no_b.pt", kept_b=False)
    with pytest.raises(UnlearnError, match="no kept anchor image"):
        unlearn(
            original, forget, with_overrides(config, synthetic_path=str(empty), synthetic_pools="B")
        )


# ----- band_ascent (V5) ----------------------------------------------------------------------

BAND_TINY = {"band_levels": 2, "band_floor": 10.0, "cv_check_every": 1}
"""The tiny model is unsure everywhere (loss ~4.6, 100 classes): a floor of 10 keeps the held
images inside their band, so the probe's hard stop stays off."""


def test_band_targets_push_the_highest_view_losses_inside_a_band() -> None:
    view_loss = torch.tensor([0.1, 2.0, 0.5, 3.0, 1.0, 0.2, 0.3, 0.4])
    config = _config("band_ascent", band_release=0.25, band_tolerance=2.0, band_floor=0.05)
    pushed, target, ceiling = band_targets(view_loss, config, 0.5)
    assert pushed.tolist() == [False, True, False, True, False, False, False, False]
    assert target.tolist() == pytest.approx([0, 1.0, 0, 1.5, 0, 0, 0, 0])
    assert ceiling.tolist() == pytest.approx([0.05, 2.05, 0.05, 3.05, 0.05, 0.05, 0.05, 0.05])
    _, _, open_ceiling = band_targets(view_loss, with_overrides(config, band_floor=None), 0.5)
    assert open_ceiling[pushed].tolist() == pytest.approx([2.0, 3.0])
    assert torch.isinf(open_ceiling[~pushed]).all()
    everyone, _, _ = band_targets(view_loss, with_overrides(config, band_release=1.0), 1.0)
    assert everyone.all()


def test_band_loss_lifts_below_the_target_and_pulls_back_above_the_ceiling() -> None:
    losses = torch.tensor([0.1, 1.5, 5.0, 0.0, 3.0], requires_grad=True)
    pushed = torch.tensor([True, True, True, False, False])
    target = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    ceiling = torch.tensor([2.0, 2.0, 2.0, 0.5, math.inf])
    loss = band_loss(losses, pushed, target, ceiling, 50.0)
    loss.backward()
    assert losses.grad is not None and math.isfinite(float(loss))
    grad = losses.grad.tolist()  # d loss / d L_i = (s(beta (L - h)) - pushed s(beta (t - L))) / n
    assert grad[0] == pytest.approx(-0.2, abs=1e-6)  # under its target: ascent
    assert grad[1] == pytest.approx(0.0, abs=1e-6)  # inside its band: no force
    assert grad[2] == pytest.approx(0.2, abs=1e-6)  # above its ceiling: pulled back
    assert grad[3] == pytest.approx(0.0, abs=1e-6) and grad[4] == 0.0  # held, under / no ceiling
    extreme = band_loss(torch.tensor([1e4, 0.0]), torch.tensor([True, False]),
                        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.05]), 10.0)  # fmt: skip
    assert math.isfinite(float(extreme))


@pytest.mark.parametrize(
    "bad",
    [{"band_release": 0.0}, {"band_release": 1.5}, {"band_tolerance": 0.5}, {"band_floor": -1.0},
     {"band_levels": 0}, {"band_sharpness": 0.0}, {"cv_folds": 1}, {"cv_folds": -2},
     {"cv_max_drift": 0.0}, {"cv_min_agreement": 1.5}, {"cv_scale_exponent": -1.0},
     {"cv_check_every": 0}],
)  # fmt: skip
def test_invalid_band_fields_raise_config_error(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        _config("band_ascent", **bad)


def test_band_fields_parse_from_strings() -> None:
    config = build_config({"method": "band_ascent", "band_floor": "none", "cv_folds": "0",
                           "band_release": "1"})  # fmt: skip
    assert config.band_floor is None and config.cv_folds == 0 and config.band_release == 1.0


def test_cv_folds_partition_the_set_apart_from_the_v4_folds() -> None:
    labels = torch.arange(40) % 7
    folds = anchor_fold_indices(labels, 5, seed=0, offset=CV_SEED_OFFSET)
    assert sorted(torch.cat(folds).tolist()) == list(range(40))
    assert [len(fold) for fold in folds] == [8] * 5
    assert any(not torch.equal(a, b) for a, b in zip(folds, anchor_fold_indices(labels, 5, 0),
                                                     strict=True))  # fmt: skip


def test_band_ascent_trains_reproducibly_spares_the_input_and_reports_its_choice() -> None:
    original, forget = tiny_model(), random_images(15)
    before = _state(original)
    config = _config("band_ascent", **BAND_TINY, cv_folds=3, cv_max_drift=1e9,
                     cv_min_agreement=0.0)  # fmt: skip
    out, history = unlearn_with_history(original, forget, config)
    again, _ = unlearn_with_history(original, forget, config)
    after, last = out.state_dict(), history[-1]
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert all(torch.equal(after[k], v) for k, v in again.state_dict().items())
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    assert not out.training
    assert [(row["phase"], row["level"]) for row in history] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert all(0.0 <= row["witness_agreement"] <= 1.0 for row in history[:2])
    assert last["cv_stop_level"] == 2.0 and last["band_levels_done"] == 2.0
    assert last["hard_stop"] == 0.0 and last["cv_scale"] == pytest.approx(1.5**2)
    assert last["pushed_count"] == 4.0 and last["forget_kl"] > 0.0  # round(0.25 * 15)
    assert last["steps"] > history[-2]["steps"]  # cumulative over the levels


def test_band_ascent_without_cross_validation_climbs_every_level() -> None:
    config = _config("band_ascent", **BAND_TINY, cv_folds=0, cv_max_drift=1e9)
    _, history = band_ascent_unlearn(tiny_model(), random_images(8), config, None)
    assert [row["level"] for row in history] == [1.0, 2.0]
    assert {row["phase"] for row in history} == {1.0} and "cv_stop_level" not in history[-1]


def test_band_ascent_keeps_the_original_when_no_level_passes_the_cross_validation() -> None:
    original, forget = tiny_model(), random_images(12)
    config = _config("band_ascent", **BAND_TINY, cv_folds=3, cv_max_drift=1e-12)
    out, history = band_ascent_unlearn(original, forget, config, None)
    last = history[-1]
    assert last["cv_stop_level"] == 0.0 and last["cv_hard_stop"] == 1.0
    assert last["phase"] == 1.0 and last["band_levels_done"] == 0.0
    assert all(torch.equal(v, out.state_dict()[k]) for k, v in original.state_dict().items())


def test_band_ascent_hard_stop_restores_the_last_level_completed() -> None:
    original, forget = tiny_model(), random_images(12)
    # every loss (~4.6) sits above a zero ceiling: the probe stops the run at its first check
    config = _config("band_ascent", **{**BAND_TINY, "band_floor": 0.0}, cv_folds=0)
    out, history = band_ascent_unlearn(original, forget, config, None)
    assert len(history) == 1 and history[0]["steps"] == 1.0
    assert history[-1]["hard_stop"] == 1.0 and history[-1]["band_levels_done"] == 0.0
    assert all(torch.equal(v, out.state_dict()[k]) for k, v in original.state_dict().items())
    with pytest.raises(ConfigError, match="exceeds the 12 forget images"):
        band_ascent_unlearn(original, forget, with_overrides(config, cv_folds=13), None)


# ----- band_ascent V5.1 options --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [{"band_space": "logit"}, {"band_hold_kl": 0.0}, {"band_hold_kl": -1.0},
            {"cv_min_retain": 0.0}, {"cv_min_retain": 1.5}],
)  # fmt: skip
def test_invalid_v51_fields_raise_config_error(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        _config("band_ascent", **bad)


def test_v51_fields_default_to_v5_and_parse_from_strings() -> None:
    default = _config("band_ascent")
    assert (default.band_space, default.band_hold_kl, default.cv_min_retain) == ("loss", None, None)
    config = build_config({"method": "band_ascent", "band_space": "margin",
                           "band_hold_kl": "0.128", "cv_min_retain": "0.8"})  # fmt: skip
    assert (config.band_space, config.band_hold_kl, config.cv_min_retain) == ("margin", 0.128, 0.8)


def test_margin_bands_are_the_loss_bands_with_an_order_one_force() -> None:
    torch.manual_seed(0)
    logits, labels = 3.0 * torch.randn(64, 10), torch.arange(64) % 10
    margins = band_margins(logits, labels)
    assert torch.allclose(torch_f.softplus(-margins),
                          torch_f.cross_entropy(logits, labels, reduction="none"), atol=1e-5)  # fmt: skip
    losses, bounds = torch.linspace(0.01, 5.0, 50), torch.tensor([0.05, 0.3, 1.0, 2.5])
    inside_loss = losses[:, None] <= bounds[None, :]
    inside_margin = loss_margin(losses)[:, None] >= loss_margin(bounds)[None, :]
    assert torch.equal(inside_loss, inside_margin)  # L <= h  <=>  m >= nu: the same band
    assert torch.isinf(loss_margin(torch.tensor([math.inf]))).all()
    assert torch.isfinite(loss_margin(torch.tensor([0.0]))).all()
    sure = torch.zeros(1, 10)
    sure[0, 0] = 12.0  # a memorised image: loss ~ 1e-4
    label, pushed = torch.tensor([0]), torch.tensor([True])
    target, ceiling = torch.tensor([0.5]), torch.tensor([2.0])
    for space, expected in (("loss", lambda g: g < 1e-3), ("margin", lambda g: g > 0.5)):
        z = sure.clone().requires_grad_(True)
        if space == "margin":
            band_loss(-band_margins(z, label), pushed, -loss_margin(target), -loss_margin(ceiling),
                      10.0).backward()  # fmt: skip
        else:
            band_loss(torch_f.cross_entropy(z, label, reduction="none"), pushed, target, ceiling,
                      10.0).backward()  # fmt: skip
        assert z.grad is not None and expected(float(z.grad.norm())), space


def test_band_hold_loss_is_flat_at_the_original_and_pulls_past_its_tolerance() -> None:
    torch.manual_seed(0)
    original = 4.0 * torch.randn(6, 10)
    held = torch.tensor([True, True, True, False, False, False])
    at_origin = original.clone().requires_grad_(True)
    band_hold_loss(original, at_origin, held, 0.1).backward()
    assert at_origin.grad is not None and float(at_origin.grad.abs().max()) < 1e-6
    drifted = original.clone()
    drifted[torch.arange(6), original.argmax(dim=1)] -= 30.0  # every image far past the tolerance
    moved = drifted.requires_grad_(True)
    band_hold_loss(original, moved, held, 0.1).backward()
    assert moved.grad is not None
    assert (moved.grad[:3].norm(dim=1) > 1e-3).all()  # held: pulled back
    assert float(moved.grad[3:].abs().max()) == 0.0  # pushed: never held


def test_band_ascent_v51_trains_reproducibly_and_reports_its_retain() -> None:
    original, forget = tiny_model(), random_images(15)
    config = _config("band_ascent", **BAND_TINY, cv_folds=3, band_space="margin",
                     band_hold_kl=5.0, cv_min_retain=1e-9)  # fmt: skip
    out, history = band_ascent_unlearn(original, forget, config, None)
    again, _ = band_ascent_unlearn(original, forget, config, None)
    after, last = out.state_dict(), history[-1]
    assert all(torch.equal(after[k], v) for k, v in again.state_dict().items())
    assert any(not torch.equal(v, after[k]) for k, v in original.state_dict().items())
    assert all(torch.isfinite(v).all() for v in after.values() if v.is_floating_point())
    for row in history[:2]:  # the cross-validation sub-run: the score's retain of the fold
        assert row["witness_retain"] == pytest.approx(
            row["witness_agreement"] / (1.0 + row["witness_kl_scaled"])
        )
    kept = int(last["cv_stop_level"])  # the tiny model flips its argmax easily: maybe level 1 only
    assert kept >= 1 and last["cv_retain"] == history[kept - 1]["witness_retain"]
    assert 0.0 <= last["pushed_in_band"] <= 1.0 and last["held_kl"] >= 0.0


def test_cv_min_retain_replaces_both_thresholds() -> None:
    original, forget = tiny_model(), random_images(12)
    # a small rate moves the fold by ~1e-11 and keeps its argmax: the V5 thresholds reject it
    strict = _config("band_ascent", **BAND_TINY, cv_folds=3, cv_max_drift=1e-12,
                     cv_min_agreement=1.0, lr=1e-5)  # fmt: skip
    _, history = band_ascent_unlearn(original, forget, strict, None)
    assert history[-1]["cv_stop_level"] == 0.0
    _, history = band_ascent_unlearn(original, forget, with_overrides(strict, cv_min_retain=0.5),
                                     None)  # fmt: skip
    assert history[-1]["cv_stop_level"] == 2.0  # the score's retain (~1) replaces them both
    out, history = band_ascent_unlearn(original, forget, with_overrides(strict, cv_min_retain=1.0),
                                       None)  # fmt: skip
    assert history[-1]["cv_stop_level"] == 0.0  # any drift breaks a retain of 1
    assert all(torch.equal(v, out.state_dict()[k]) for k, v in original.state_dict().items())


# ----- dumb (V_Dumb) -------------------------------------------------------------------------


def test_dumb_raises_the_forget_loss_reproducibly_and_spares_the_input() -> None:
    original, forget = tiny_model(), random_images(8)
    before = _state(original)
    start = torch_f.cross_entropy(predict(original, forget), predict(original, forget).argmax(1))
    config = _config("dumb", dumb_target_loss=50.0)  # above every reachable loss: always pushed
    out, history = unlearn_with_history(original, forget, config)
    again, _ = dumb_unlearn(original, forget, config, None)
    last = history[-1]
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in out.state_dict().items())
    assert not out.training and len(history) == 2 and last["steps"] == 4.0  # 2 x ceil(8 / 4)
    assert last["forget_loss_mean"] > float(start) and last["forget_kl"] > 0.0
    assert last["reached"] == 0.0 and last["retain_loss"] >= 0.0


def test_dumb_stops_pushing_images_already_above_the_target() -> None:
    original, forget = tiny_model(), random_images(8)
    # a 100-class tiny model sits at CE ~4.3 > 1e-3: the hinge is flat, only the anchor acts
    _, history = dumb_unlearn(original, forget, _config("dumb", dumb_target_loss=1e-3), None)
    assert all(row["forget_loss"] == 0.0 for row in history) and history[-1]["reached"] == 1.0
    assert history[-1]["forget_kl"] < 1e-3
    with pytest.raises(ConfigError, match="dumb_target_loss"):
        _config("dumb", dumb_target_loss=0.0)


# ----- V_Dumb2 (scrub_target_loss) --------------------------------------------------------


def test_scrub_target_loss_defaults_to_v4_is_validated_and_parses() -> None:
    assert _config("scrub").scrub_target_loss is None
    config = build_config({"method": "scrub", "scrub_target_loss": "0.45",
                           "band_tolerance": "inf"})  # fmt: skip
    assert config.scrub_target_loss == 0.45 and math.isinf(config.band_tolerance)
    for bad in (0.0, -0.1):
        with pytest.raises(ConfigError, match="scrub_target_loss"):
            _config("scrub", scrub_target_loss=bad)


def test_level_band_loss_lifts_with_an_order_one_force_stops_in_band_and_pulls_back() -> None:
    config = _config("scrub", scrub_target_loss=0.45, band_tolerance=2.0, band_sharpness=10.0)
    label = torch.tensor([0])
    grads = {}
    for name, top in (("memorised", 12.0), ("in_band", 2.29), ("past", -1.0)):
        z = torch.zeros(1, 10)
        z[0, 0] = top  # CE on label 0: ~6e-5, ~0.65 (mid-band of [0.45, 0.9] in margin), ~3.2
        z.requires_grad_(True)
        level_band_loss(z, label, config).backward()
        assert z.grad is not None
        grads[name] = z.grad[0, 0].item()  # d loss / d z_y: > 0 lowers the label logit
    assert grads["memorised"] > 0.5  # order 1, not ~1e-4 like the loss-space ascent
    assert abs(grads["in_band"]) < 0.05
    assert grads["past"] < -0.5  # pulled back: the label logit goes up
    free = with_overrides(config, band_tolerance=math.inf)
    z = torch.zeros(1, 10)
    z[0, 0] = -1.0
    z.requires_grad_(True)
    level_band_loss(z, label, free).backward()
    assert z.grad is not None and float(z.grad.abs().max()) < 1e-6  # no ceiling: no pull


def test_v_dumb2_trains_reproducibly_spares_the_input_and_reports_its_band() -> None:
    original, forget = tiny_model(), random_images(8)
    before = _state(original)
    start = torch_f.cross_entropy(predict(original, forget), predict(original, forget).argmax(1))
    # above every reachable loss of the 100-class tiny model: every image is lifted
    config = _config("scrub", retain_source="forget_views", scrub_target_loss=50.0)
    out, history = unlearn_with_history(original, forget, config)
    again, _ = scrub_unlearn(original, forget, config, None)
    last = history[-1]
    assert all(torch.equal(before[k], v) for k, v in original.state_dict().items())
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in out.state_dict().items())
    assert not out.training and len(history) == 2 and history[0]["band_loss"] > 0.0
    assert last["forget_loss_mean"] > float(start) and last["forget_kl"] > 0.0
    assert last["in_band"] == 0.0 and last["forget_loss_std"] >= 0.0
    assert "band_loss" not in scrub_unlearn(original, forget, _config(
        "scrub", retain_source="forget_views"), None)[1][0]  # V4 history unchanged  # fmt: skip


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
