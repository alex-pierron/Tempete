"""Tests of team A3's pool mode: claims, deadline-aware draws, generation pool, logits export."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hackathon.unlearning.research import a3_export
from hackathon.unlearning.research.a3_generate import (
    DEADLINE_MARGIN_SECONDS,
    DONE_SUFFIX,
    EXIT_TERMINATED,
    FIT_PEAK_ACCURACY,
    LOSS_TARGET_MEAN,
    LOSS_TARGET_STD,
    REFERENCE_CARD,
    REFERENCE_SHARP,
    _terminate,
    calibration_cost,
    candidate_name,
    estimate_seconds,
    fit_public_score,
    grid_points,
    is_done,
    loss_distance_to_gold,
    pick_next,
    reference_distance,
    select_to_save,
    try_claim,
)
from hackathon.unlearning.research.a3_pool import (
    build_all_pool,
    build_conform2_pool,
    build_conform3_pool,
    build_conform_pool,
    build_diag_pool,
    build_filler_pool,
    build_g2_pool,
    build_g3_pool,
    build_g4_pool,
    build_g5_pool,
    build_shape_pool,
    build_sibling_pool,
)
from hackathon.unlearning.research.a3_spec import SpecFile, VariantSpec

WORKER = "gpu0"
PLENTY_OF_TIME = 10_000.0


def _spec(*names: str, steps: int = 219) -> SpecFile:
    return SpecFile(
        generation="t", variants=tuple(VariantSpec(name=name, steps=steps) for name in names)
    )


def test_grid_points_are_alpha_major_and_names_are_stable() -> None:
    variant = VariantSpec(
        name="v", alphas=(1.0, 1.1), post_deltas=(0.0, 2.0), head_scales=(1.0, 1.25)
    )
    assert grid_points(variant) == [
        (1.0, 0.0, 1.0), (1.0, 0.0, 1.25), (1.0, 2.0, 1.0), (1.0, 2.0, 1.25),
        (1.1, 0.0, 1.0), (1.1, 0.0, 1.25), (1.1, 2.0, 1.0), (1.1, 2.0, 1.25),
    ]  # fmt: skip
    assert candidate_name(variant, 1.1, 2.0) == "v_a1.10_p2"
    assert candidate_name(variant, 1.0, 0.0) == "v_a1.00_p0"
    assert candidate_name(variant, 1.1, 2.0, 1.25) == "v_a1.10_p2_s125"


def test_estimate_grows_with_steps_grid_and_saved_files() -> None:
    small = VariantSpec(name="s", steps=219, alphas=(1.0,), post_deltas=(0.0,), save_top=0)
    long_run = VariantSpec(name="l", steps=2190, alphas=(1.0,), post_deltas=(0.0,), save_top=0)
    wide = VariantSpec(name="w", steps=219, alphas=(1.0, 1.1), post_deltas=(0.0, 2.0), save_top=0)
    saving = VariantSpec(name="v", steps=219, alphas=(1.0,), post_deltas=(0.0,), save_top=2)
    assert estimate_seconds(long_run) > estimate_seconds(small)
    assert estimate_seconds(wide) > estimate_seconds(small)
    assert estimate_seconds(saving) > estimate_seconds(small)


def test_claim_is_exclusive(tmp_path: Path) -> None:
    claims = tmp_path / "claims"
    assert try_claim(claims, "a", WORKER)
    assert not try_claim(claims, "a", "gpu1")
    assert try_claim(claims, "b", "gpu1")


def test_concurrent_workers_never_share_a_variant(tmp_path: Path) -> None:
    spec = _spec(*[f"v{index}" for index in range(12)])
    runs, claims = tmp_path / "runs", tmp_path / "claims"
    runs.mkdir()
    won: dict[str, list[str]] = {f"gpu{i}": [] for i in range(4)}

    def work(worker: str) -> None:
        while (variant := pick_next(spec, runs, claims, worker, PLENTY_OF_TIME)) is not None:
            won[worker].append(variant.name)

    threads = [threading.Thread(target=work, args=(worker,)) for worker in won]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    everything = [name for names in won.values() for name in names]
    assert sorted(everything) == sorted(f"v{index}" for index in range(12))


def test_pick_next_follows_spec_order_and_skips_done(tmp_path: Path) -> None:
    runs, claims = tmp_path / "runs", tmp_path / "claims"
    runs.mkdir()
    (runs / f"a3_first{DONE_SUFFIX}").write_text("done\n", encoding="utf-8")
    spec = _spec("first", "second", "third")
    assert is_done(runs, "first")
    picked = pick_next(spec, runs, claims, WORKER, PLENTY_OF_TIME)
    assert picked is not None and picked.name == "second"
    picked = pick_next(spec, runs, claims, WORKER, PLENTY_OF_TIME)
    assert picked is not None and picked.name == "third"
    assert pick_next(spec, runs, claims, WORKER, PLENTY_OF_TIME) is None


def test_pick_next_skips_a_variant_that_would_not_finish_before_the_deadline(
    tmp_path: Path,
) -> None:
    runs, claims = tmp_path / "runs", tmp_path / "claims"
    runs.mkdir()
    long_variant = VariantSpec(name="long", steps=2190)
    short_variant = VariantSpec(name="short", steps=219)
    spec = SpecFile(generation="t", variants=(long_variant, short_variant))
    seconds_left = estimate_seconds(short_variant) + DEADLINE_MARGIN_SECONDS + 1.0
    assert estimate_seconds(long_variant) > seconds_left - DEADLINE_MARGIN_SECONDS
    picked = pick_next(spec, runs, claims, WORKER, seconds_left)
    assert picked is not None and picked.name == "short"
    assert pick_next(spec, runs, claims, WORKER, seconds_left) is None


def test_g2_pool_is_valid_ordered_and_covers_each_axis_once() -> None:
    pool = build_g2_pool()
    names = [variant.name for variant in pool]
    assert "g2_fb1_cal22" in names and len(set(names)) == len(names)
    assert 20 <= len(pool) <= 50
    center = next(v for v in pool if v.name == "g2_fb1_cal22")
    for variant in (v for v in pool if v is not center):
        # One axis at a time: a variant differs from the centre in a short list of fields.
        changed = [
            field
            for field in ("steps", "lr", "retain_weight", "target_temperature", "delta_full",
                          "delta_partial", "anchor_view_fraction", "dual_weight", "removed_flatten", "cal_wrong_fraction", "fallback_temperature", "sibling_mass", "seed")
            if getattr(variant, field) != getattr(center, field)
        ]  # fmt: skip
        assert 1 <= len(changed) <= 2, (variant.name, changed)


class _TinyViT(nn.Module):
    """Stands for the timm ViT: same three entry points used by the export."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(3, 100)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return torch.as_tensor(self.pool(features).flatten(1))


def test_export_writes_the_team_c_format_and_is_idempotent(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8, generator=generator)
    views = a3_export.build_views(images)
    assert set(views) == set(a3_export.VIEWS_EXPORTED)
    indices = torch.arange(4)
    model = _TinyViT()
    path = a3_export.export_model(model, "cand", tmp_path, indices, views)
    with np.load(path) as payload:
        assert payload["subset_indices"].tolist() == [0, 1, 2, 3]
        for view in a3_export.VIEWS_EXPORTED:
            assert payload[f"logits_{view}"].shape == (4, 100)
            assert payload[f"feats_{view}"].shape == (4, 3)
            assert payload[f"feats_{view}"].dtype == np.float16
    modified = path.stat().st_mtime_ns
    assert a3_export.export_model(model, "cand", tmp_path, indices, views) == path
    assert path.stat().st_mtime_ns == modified
    assert not list(tmp_path.glob("*.partial.npz"))


def test_export_views_are_deterministic() -> None:
    generator = torch.Generator().manual_seed(1)
    images = torch.randint(0, 255, (3, 224, 224, 3), dtype=torch.uint8, generator=generator)
    first, second = a3_export.build_views(images), a3_export.build_views(images)
    for view in a3_export.VIEWS_EXPORTED:
        assert torch.equal(first[view], second[view])


def test_parse_pairs_and_shards() -> None:
    pairs = a3_export.parse_pairs(["a=/x/a.pt", "b=/x/b.pt", "c=/x/c.pt"])
    assert [name for name, _ in pairs] == ["a", "b", "c"]
    assert [name for name, _ in a3_export.shard(pairs, 0, 2)] == ["a", "c"]
    assert [name for name, _ in a3_export.shard(pairs, 1, 2)] == ["b"]
    with pytest.raises(a3_export.ExportError):
        a3_export.parse_pairs(["no-separator"])
    with pytest.raises(a3_export.ExportError):
        a3_export.shard(pairs, 2, 2)


def _entry(name: str, distance: float, acc_other: float) -> dict[str, object]:
    """A card at the reference, moved by `distance` in the loss mean and given an accuracy."""
    card = dict(REFERENCE_CARD)
    card["loss_mean"] += distance
    card["pseudo_acc_other"] = acc_other
    card["retain_flips"] = 0.3
    return {"name": name, "card": card}


def test_fit_public_score_rewards_depth_and_peaks_in_accuracy() -> None:
    assert fit_public_score(7.0, 0.75) > fit_public_score(4.0, 0.75) > fit_public_score(1.0, 0.75)
    peak = FIT_PEAK_ACCURACY
    assert 0.72 < peak < 0.75
    assert fit_public_score(7.0, peak) > fit_public_score(7.0, peak + 0.1)
    assert fit_public_score(7.0, peak) > fit_public_score(7.0, peak - 0.1)
    # The two points that broke the first (linear) fit: 0.63 must not beat 0.78.
    assert fit_public_score(7.25, 0.63) < fit_public_score(6.2, 0.777)
    # It reproduces c_f9d8 (0.1091) within the stated error.
    assert 0.08 < fit_public_score(7.15, 0.777) < 0.15


def test_fit_is_flat_then_penalised_beyond_the_plateau_of_the_removal_loss() -> None:
    assert fit_public_score(6.0, 0.75) < fit_public_score(8.5, 0.75)
    assert fit_public_score(11.6, 0.75) < fit_public_score(8.5, 0.75)
    # Team C: a loss of 11.6 is "too deep" (their model: 0.05), 8.2 is the target.
    assert fit_public_score(11.6, 0.785) < 0.5 * fit_public_score(8.2, 0.785)


def test_loss_distance_is_zero_at_the_gold_and_grows_with_either_error() -> None:
    assert loss_distance_to_gold(LOSS_TARGET_MEAN, LOSS_TARGET_STD) == 0.0
    # The two failures of the trained students: a3_g2 p2p5 (mean 3.64, std 3.52) against c_f9d7 (2.70, 3.05).
    assert loss_distance_to_gold(3.64, 3.52) > 4 * loss_distance_to_gold(2.70, 3.05)
    assert loss_distance_to_gold(1.0, 3.0) == pytest.approx(loss_distance_to_gold(3.9, 3.0))


def test_select_to_save_takes_the_main_reference_then_the_sharper_probe() -> None:
    sharp = {**REFERENCE_SHARP, "retain_flips": 0.3}
    cards: list[dict[str, object]] = [
        _entry("near_twin", 0.2, 0.76),
        {"name": "sharp", "card": sharp},
        _entry("main", 0.0, 0.777),
        _entry("far", 0.9, 0.84),
    ]
    assert [e["name"] for e in select_to_save(cards, 1)] == ["main"]
    assert [e["name"] for e in select_to_save(cards, 2)] == ["main", "sharp"]
    assert [e["name"] for e in select_to_save(cards, 3)] == ["main", "sharp", "near_twin"]
    assert len(select_to_save(cards, 10)) == 4


def test_select_to_save_penalises_a_kl_on_the_other_classes_above_the_limit() -> None:
    clean = _entry("clean", 0.3, 0.777)
    leaky = _entry("leaky", 0.0, 0.777)
    leaky["card"]["kl_others_t2"] = 3.0  # type: ignore[index]
    assert [e["name"] for e in select_to_save([leaky, clean], 1)] == ["clean"]


def test_select_to_save_without_a_distant_point_follows_the_distance_and_handles_empty() -> None:
    cards = [_entry("a", 0.5, 0.80), _entry("b", 0.9, 0.81)]
    assert [e["name"] for e in select_to_save(cards, 2)] == ["a", "b"]
    assert select_to_save(cards, 0) == []
    assert select_to_save([], 2) == []


def test_sibling_pool_and_the_combined_order() -> None:
    siblings = build_sibling_pool()
    assert [v.sibling_mass for v in siblings] == [0.7, 0.5, 0.85]
    assert all(v.fallback_temperature > 0 for v in siblings)
    combined = [v.name for v in build_all_pool()]
    assert len(set(combined)) == len(combined)
    assert combined.index("g3_scale_check") < combined.index("g3_sib70")
    first_sibling = combined.index("g3_sib70")
    assert first_sibling < combined.index("g3_cls0.5")  # ahead of the rest of g3
    assert first_sibling < combined.index("g2_rw16")  # ahead of the pending g2 axes


def test_diag_pool_holds_out_subset_b_writes_no_file_and_ignores_the_batch_scale() -> None:
    diag = build_diag_pool()
    assert len(diag) >= 5 and len({v.name for v in diag}) == len(diag)
    assert all(v.holdout and v.no_batch_scale and v.save_top == 0 for v in diag)
    assert all(v.dual_weight == 0.0 for v in diag)  # holdout is not implemented for the dual loss
    phase1 = next(v for v in diag if v.name == "g3_h_gsrepro")
    assert (phase1.steps, phase1.lr, phase1.retain_weight) == (219, 2e-5, 1.0)
    assert phase1.cal_wrong_fraction == 0.0 and phase1.fallback_temperature == 0.0
    combined = [v.name for v in build_all_pool()]
    assert combined.index("g3_h_gsrepro") < combined.index("g3_sib70")


def test_calibration_cost_rejects_a_confident_student_that_matches_mean_and_std() -> None:
    gold_like = calibration_cost(2.5, 3.0, 0.9)
    confident = calibration_cost(2.5, 3.0, 0.3)  # same mean and std, loss 0.3 on the other images
    assert confident == pytest.approx(gold_like + 0.6)
    assert gold_like < calibration_cost(3.64, 3.52, 1.05)  # the p2p5 failure


def test_g4_pool_refines_the_sibling_base_one_factor_at_a_time() -> None:
    g4 = build_g4_pool()
    assert len({v.name for v in g4}) == len(g4) >= 10
    reference = next(v for v in build_sibling_pool() if v.name == "g3_sib70")
    for variant in g4:
        assert variant.sibling_mass > 0 and variant.fallback_temperature > 0
        changed = [
            field
            for field in ("sibling_mass", "fallback_temperature", "target_temperature", "view_count",
                          "cal_class_aware", "cal_wrong_fraction", "seed", "dual_weight", "delta_partial")
            if getattr(variant, field) != getattr(reference, field)
        ]  # fmt: skip
        assert len(changed) <= 1, (variant.name, changed)
    combined = [v.name for v in build_all_pool()]
    assert combined.index("g4_sib100") < combined.index("g2_rw16")


def test_g3_pool_is_valid_distinct_and_uses_the_new_dimensions() -> None:
    g3 = build_g3_pool()
    names = [variant.name for variant in g3]
    assert len(set(names)) == len(names) and not set(names) & {v.name for v in build_g2_pool()}
    assert any(v.cal_class_aware > 0 for v in g3)
    assert any(v.view_count > 3 for v in g3)
    assert any(v.fallback_temperature > 0 for v in g3)
    assert all(v.cal_wrong_fraction > 0 for v in g3)


def test_filler_pool_is_valid_unique_and_distinct_from_the_main_pool() -> None:
    filler = build_filler_pool()
    names = [variant.name for variant in filler]
    assert len(set(names)) == len(names) >= 20
    assert not set(names) & {variant.name for variant in build_g2_pool()}
    assert all(variant.cal_wrong_fraction > 0.0 for variant in filler)
    assert all(variant.save_top == 1 for variant in filler)  # one inode per variant at most


def test_sigterm_handler_exits_with_the_conventional_status() -> None:
    with pytest.raises(SystemExit) as caught:
        _terminate(15, None)
    assert caught.value.code == EXIT_TERMINATED == 143


def test_export_leaves_no_partial_file_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = torch.Generator().manual_seed(2)
    images = torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8, generator=generator)
    views = a3_export.build_views(images)

    def interrupted(path: Path, **arrays: np.ndarray) -> None:
        path.write_bytes(b"half")
        raise SystemExit(143)

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(SystemExit):
        a3_export.export_model(_TinyViT(), "cand", tmp_path, torch.arange(2), views)
    assert list(tmp_path.iterdir()) == []


def test_g5_pool_holds_out_subset_b_on_the_sibling_base_and_writes_no_file() -> None:
    g5 = build_g5_pool()
    assert len({v.name for v in g5}) == len(g5) >= 8
    assert all(v.holdout and v.no_batch_scale and v.save_top == 0 for v in g5)
    assert all(v.sibling_mass == 0.7 and v.dual_weight == 0.0 for v in g5)
    assert min(v.steps for v in g5) < 219  # shorter than one epoch is covered
    combined = [v.name for v in build_all_pool()]
    assert combined.index(g5[0].name) < combined.index("g4_sib100")


def test_shape_pool_sharpens_the_calibrated_law_and_runs_first() -> None:
    shape = build_shape_pool()
    assert len({v.name for v in shape}) == len(shape) >= 6
    sharp = [v for v in shape if not v.name.startswith("g5s_gs")]
    assert all(v.cal_right_low > 0.55 and v.cal_wrong_top_share > 0.5 for v in sharp)
    assert all(not v.holdout for v in shape)
    repro = next(v for v in shape if v.name == "g5s_gs_repro")
    assert (repro.steps, repro.lr, repro.retain_weight, repro.retain_batch_size) == (
        219,
        2e-5,
        1.0,
        8,
    )
    assert repro.sibling_mass == 0 and repro.fallback_temperature == 0 and repro.no_batch_scale
    combined = [v.name for v in build_all_pool()]
    assert combined.index(shape[0].name) < combined.index(build_g5_pool()[0].name)


def test_selection_charges_a_kl_above_the_limit_on_the_partially_forgotten_classes() -> None:
    within = _entry("within", 0.30, 0.76)
    within["card"]["kl_others_t2"] = 1.3  # type: ignore[index]
    above = _entry("above", 0.10, 0.76)  # closer loss, but over the 1.4 limit
    above["card"]["kl_others_t2"] = 2.0  # type: ignore[index]
    assert [e["name"] for e in select_to_save([above, within], 1)] == ["within"]
    above["card"]["kl_others_t2"] = 1.4  # type: ignore[index]
    assert [e["name"] for e in select_to_save([above, within], 1)] == ["above"]


def test_every_variant_respects_the_official_rule_by_default() -> None:
    """No external image in any anchor unless a variant explicitly asks for it (none does)."""
    everything = build_all_pool()
    assert not any(v.external_anchor for v in everything)
    conform = build_conform_pool()
    assert [v.name for v in build_all_pool()][: len(conform)] == [v.name for v in conform]
    for variant in conform:
        assert variant.save_raw and variant.no_batch_scale
        assert (variant.steps, variant.lr, variant.retain_batch_size) == (219, 2e-5, 8)
        assert variant.target_temperature == 2.0 and variant.delta_full == 0.0
        assert variant.cal_wrong_fraction == 0.0 and variant.sibling_mass == 0.0
    assert [v.name for v in conform if v.anchor_skip_removed] == [
        "g5c_skip_rw1", "g5c_skip_rw4", "g5c_skip_rw025"
    ]  # fmt: skip
    assert any(v.retain_weight == 1.0 for v in conform)  # the phase 1 value is in the bracket
    assert len({v.name for v in conform}) == len(conform)


def test_reference_distance_is_zero_at_the_reference_and_counts_each_feature() -> None:
    assert reference_distance(dict(REFERENCE_CARD)) == 0.0
    sharper = dict(REFERENCE_CARD)
    sharper["entropy_other"] -= 0.2
    assert reference_distance(sharper) == pytest.approx(0.2 * 1.5)
    diffuse = dict(
        REFERENCE_CARD
    )  # a diffuse student: no confident-wrong images, few confident-right
    diffuse.update({"ptrue_lt01": 0.04, "ptrue_gt09": 0.15, "entropy_other": 1.7})
    assert reference_distance(diffuse) > 1.5  # the failure the shape was added to catch
    assert reference_distance({}) == 0.0  # missing fields count as the reference


def test_scale_logits_equals_a_head_scale_and_composes_with_the_bias_shift() -> None:
    import torch

    from hackathon.unlearning.research.a3_generate import (
        GroupLogits,
        scale_head_state,
        scale_logits,
        shift_removed,
    )

    generator = torch.Generator().manual_seed(0)
    features = torch.randn(7, 5, generator=generator)
    state = {
        "head.weight": torch.randn(10, 5, generator=generator),
        "head.bias": torch.randn(10, generator=generator),
    }
    removed = (2, 7)
    edited = {k: v.clone() for k, v in state.items()}
    edited["head.bias"][list(removed)] -= 3.0
    edited = scale_head_state(edited, 1.25)
    via_state = features @ edited["head.weight"].T + edited["head.bias"]
    base = features @ state["head.weight"].T + state["head.bias"]
    group = GroupLogits(exact=base, altered={}, retain=base, views=None)
    via_logits = scale_logits(shift_removed(group, removed, 3.0), 1.25).exact
    assert torch.allclose(via_state, via_logits, atol=1e-5)


def test_conform2_round_skips_the_removed_class_images_and_has_unique_names() -> None:
    round_two = build_conform2_pool()
    names = [v.name for v in round_two]
    assert len(set(names)) == len(names)
    assert not set(names) & {v.name for v in build_conform_pool()}
    assert all(v.anchor_skip_removed and not v.external_anchor and v.save_raw for v in round_two)
    assert all(v.no_batch_scale for v in round_two)
    everything = [v.name for v in build_all_pool()]
    assert everything[len(build_conform_pool()) :][: len(names)] == names
    assert len(set(everything)) == len(everything)


def test_conform3_round_widens_the_alpha_grid_and_repeats_seeds() -> None:
    round_three = build_conform3_pool()
    names = [v.name for v in round_three]
    assert len(set(names)) == len(names)
    assert not set(names) & {v.name for v in build_conform2_pool() + build_conform_pool()}
    assert all(max(v.alphas) == 1.5 and v.anchor_skip_removed and v.save_raw for v in round_three)
    seeds = {v.name: v.seed for v in round_three if "seed" in v.name}
    assert set(seeds.values()) == {1, 2}
    everything = [v.name for v in build_all_pool()]
    start = len(build_conform_pool()) + len(build_conform2_pool())
    assert everything[start : start + len(names)] == names
