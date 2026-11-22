"""Tests of team A3's class-removed targets, proxy gold and variant spec file (pure, fast)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hackathon.unlearning.research.a3_spec import (
    DEFAULT_REMOVED_CLASSES,
    SpecError,
    SpecFile,
    VariantSpec,
    load_spec_file,
)
from hackathon.unlearning.research.a3_targets import (
    A3TargetError,
    apply_class_offsets,
    build_removal_targets,
    class_aware_view_scale,
    class_counts,
    fallback_targets,
    finish_targets,
    fully_removed_classes,
    present_siblings,
    removal_offsets,
    removed_class_gold_logits,
    removed_mask,
    sibling_targets,
)
from hackathon.unlearning.research.trainer import RunSpec, build_target

CLASSES = 12
IMAGES = 40
VIEWS = 3
SEEDS = (0, 1, 2, 3, 4)


def _view_probs(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(IMAGES, VIEWS, CLASSES, generator=generator) * 4.0
    return torch.softmax(logits, dim=2)


def test_class_counts_and_fully_removed_classes() -> None:
    labels = torch.tensor([0] * 130 + [1] * 129 + [2] * 50 + [5] * 3)
    counts = class_counts(labels, class_count=8)
    assert counts.tolist() == [130, 129, 50, 0, 0, 3, 0, 0]
    assert fully_removed_classes(counts) == (0, 1)
    assert fully_removed_classes(counts, tolerance=0) == (0,)


def test_class_counts_rejects_empty_and_out_of_range_labels() -> None:
    with pytest.raises(A3TargetError):
        class_counts(torch.empty(0, dtype=torch.long))
    with pytest.raises(A3TargetError):
        class_counts(torch.tensor([0, 100]))
    with pytest.raises(A3TargetError):
        class_counts(torch.tensor([-1, 3]))


def test_removal_offsets_full_partial_and_absent_classes() -> None:
    counts = torch.tensor([130, 65, 0, 129])
    offsets = removal_offsets(counts, removed_classes=(0, 3), delta_full=6.0, delta_partial=2.0)
    assert offsets.tolist() == pytest.approx([6.0, 1.0, 0.0, 6.0])


def test_removal_offsets_reject_negative_magnitudes_and_bad_classes() -> None:
    counts = torch.tensor([130, 10])
    with pytest.raises(A3TargetError):
        removal_offsets(counts, (0,), delta_full=-1.0, delta_partial=0.0)
    with pytest.raises(A3TargetError):
        removal_offsets(counts, (0,), delta_full=1.0, delta_partial=-0.1)
    with pytest.raises(A3TargetError):
        removal_offsets(counts, (2,), delta_full=1.0, delta_partial=0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_offsets_keep_distributions_and_the_ranking_of_untouched_classes(seed: int) -> None:
    probs = _view_probs(seed).mean(dim=1)
    offsets = torch.zeros(CLASSES)
    offsets[[2, 7]] = 5.0
    shifted = apply_class_offsets(probs, offsets)
    assert torch.allclose(shifted.sum(dim=1), torch.ones(IMAGES), atol=1e-6)
    assert bool((shifted >= 0).all())
    kept = [index for index in range(CLASSES) if index not in (2, 7)]
    # Ratios between untouched classes are preserved: the "second choice" does not change.
    assert torch.allclose(
        shifted[:, kept] / shifted[:, kept[0] : kept[0] + 1],
        probs[:, kept] / probs[:, kept[0] : kept[0] + 1],
        rtol=1e-4,
    )
    # A removed class loses exactly exp(-delta) of its odds against an untouched class.
    ratio = (shifted[:, 2] / shifted[:, kept[0]]) / (probs[:, 2] / probs[:, kept[0]])
    assert torch.allclose(
        ratio, torch.full((IMAGES,), float(torch.exp(torch.tensor(-5.0)))), rtol=1e-4
    )


def test_zero_offsets_are_the_identity() -> None:
    probs = _view_probs(0).mean(dim=1)
    assert torch.allclose(apply_class_offsets(probs, torch.zeros(CLASSES)), probs, atol=1e-6)


def test_apply_class_offsets_rejects_a_class_count_mismatch() -> None:
    with pytest.raises(A3TargetError):
        apply_class_offsets(torch.ones(3, CLASSES) / CLASSES, torch.zeros(CLASSES + 1))


def test_targets_without_offsets_equal_the_gold_soft_targets_of_the_trainer() -> None:
    """Our builder must stay on the path that scored 0.0457: only the offsets are new."""
    views = _view_probs(1)
    ours = build_removal_targets(views, temperature=2.0, offsets=torch.zeros(CLASSES))
    spec = RunSpec(name="reference", method="gold_soft", target_temperature=2.0)
    reference = build_target(
        torch.nn.Identity(),
        torch.zeros(IMAGES, 3),
        spec,
        torch.Generator(),
        torch.device("cpu"),
        views=views.transpose(0, 1),
    )
    assert torch.allclose(ours, reference, atol=1e-6)


def test_removed_class_probability_drops_on_images_of_that_class() -> None:
    views = torch.full((4, VIEWS, CLASSES), 0.01)
    views[:, :, 3] = 0.89
    views = views / views.sum(dim=2, keepdim=True)
    offsets = torch.zeros(CLASSES)
    offsets[3] = 6.0
    plain = build_removal_targets(views, 2.0, torch.zeros(CLASSES))
    removed = build_removal_targets(views, 2.0, offsets)
    assert float(removed[:, 3].max()) < 0.05 * float(plain[:, 3].min())
    assert torch.allclose(removed.sum(dim=1), torch.ones(4), atol=1e-6)


def test_build_removal_targets_rejects_bad_shape_and_temperature() -> None:
    with pytest.raises(A3TargetError):
        build_removal_targets(torch.ones(4, CLASSES), 2.0, torch.zeros(CLASSES))
    with pytest.raises(A3TargetError):
        build_removal_targets(_view_probs(0), 0.0, torch.zeros(CLASSES))


@pytest.mark.parametrize("seed", SEEDS)
def test_proxy_gold_never_predicts_a_removed_class(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(IMAGES, CLASSES, generator=generator) * 5.0
    removed = (1, 4, 9)
    gold = removed_class_gold_logits(logits, removed, margin=10.0)
    assert not bool(removed_mask(gold.argmax(dim=1), removed).any())
    untouched = [index for index in range(CLASSES) if index not in removed]
    assert torch.equal(gold[:, untouched], logits[:, untouched])
    best_other = gold[:, untouched].amax(dim=1)
    assert torch.allclose(gold[:, 4], best_other - 10.0)


def test_proxy_gold_rejects_a_negative_margin() -> None:
    with pytest.raises(A3TargetError):
        removed_class_gold_logits(torch.zeros(2, CLASSES), (0,), margin=-1.0)


def test_removed_mask_selects_the_listed_labels() -> None:
    mask = removed_mask(torch.tensor([0, 1, 2, 3]), (1, 3))
    assert mask.tolist() == [False, True, False, True]


def test_variant_defaults_are_valid_and_frozen() -> None:
    variant = VariantSpec(name="a3_default")
    assert variant.removed_classes == DEFAULT_REMOVED_CLASSES
    assert len(variant.removed_classes) == 9
    with pytest.raises(AttributeError):
        variant.lr = 1.0  # type: ignore[misc]


def test_the_committed_spec_file_is_valid() -> None:
    """The file rsynced to the cluster must pass the same check as the job's first step."""
    path = Path(__file__).resolve().parents[2] / "jureca" / "a3_spec.json"
    spec = load_spec_file(path)
    assert len(spec.variants) >= 1


@pytest.mark.parametrize(
    "changes",
    [
        {"alphas": ()},
        {"alphas": (1.6,)},
        {"alphas": (0.0,)},
        {"post_deltas": ()},
        {"post_deltas": (-1.0,)},
        {"removed_classes": (100,)},
        {"steps": 0},
        {"lr": 0.0},
        {"name": "bad name"},
        {"anchor_view_fraction": 1.5},
        {"cal_wrong_fraction": 0.9},
        {"cal_right_low": 1.2},
        {"cal_wrong_high": 0.9},
        {"cal_wrong_top_share": 1.5},
        {"sibling_mass": 0.7},
        {"sibling_mass": 1.5, "fallback_temperature": 1.0},
        {"fallback_temperature": -1.0},
        {"cal_class_aware": 0.5},
        {"view_count": 0},
        {"view_area": 0.95},
        {"removed_flatten": 0.5},
        {"unknown_field": 1},
    ],
)
def test_variant_spec_rejects_unusable_values(changes: dict[str, object]) -> None:
    with pytest.raises(SpecError):
        VariantSpec.from_mapping({"name": "a3_x", **changes})


def test_variant_from_mapping_turns_json_lists_into_tuples() -> None:
    variant = VariantSpec.from_mapping({"name": "a", "alphas": [1.0, 1.2], "post_deltas": [0, 3]})
    assert variant.alphas == (1.0, 1.2)
    assert variant.post_deltas == (0, 3)
    with pytest.raises(SpecError):
        VariantSpec.from_mapping({"alphas": [1.0]})  # no name


def test_spec_file_needs_unique_names_and_a_bounded_pool() -> None:
    variants = [VariantSpec(name=f"v{index}") for index in range(4)]
    assert len(SpecFile(generation="g1", variants=tuple(variants)).variants) == 4
    with pytest.raises(SpecError):
        SpecFile(generation="g1", variants=(variants[0], variants[0]))
    with pytest.raises(SpecError):
        SpecFile(generation="g1", variants=())
    with pytest.raises(SpecError):
        SpecFile(generation="g1", variants=tuple(VariantSpec(name=f"w{i}") for i in range(201)))
    with pytest.raises(SpecError):
        SpecFile(generation="bad tag", variants=(variants[0],))


def test_load_spec_file_reads_valid_json_and_reports_faults(tmp_path: Path) -> None:
    good = tmp_path / "spec.json"
    good.write_text(
        json.dumps({"generation": "g1", "variants": [{"name": "a", "delta_full": 4.0}]}),
        encoding="utf-8",
    )
    assert load_spec_file(good).variants[0].delta_full == 4.0
    with pytest.raises(SpecError):
        load_spec_file(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec_file(broken)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"generation": "g1", "variants": []}), encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec_file(invalid)


def test_flattening_raises_the_entropy_of_the_selected_rows_only() -> None:
    views = _view_probs(3)
    offsets = torch.zeros(CLASSES)
    rows = torch.zeros(IMAGES, dtype=torch.bool)
    rows[:10] = True
    plain = build_removal_targets(views, 2.0, offsets)
    flat = build_removal_targets(views, 2.0, offsets, flatten_rows=rows, flatten=2.0)

    def entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * p.clamp_min(1e-12).log()).sum(dim=1)

    assert bool((entropy(flat[:10]) > entropy(plain[:10])).all())
    assert torch.equal(flat[10:], plain[10:])
    assert torch.allclose(flat.sum(dim=1), torch.ones(IMAGES), atol=1e-6)
    # Ranking of the classes is kept: flattening is a temperature, not a re-labelling.
    assert torch.equal(flat[:10].argmax(dim=1), plain[:10].argmax(dim=1))


def test_flatten_of_one_or_no_rows_changes_nothing_and_bad_inputs_raise() -> None:
    views = _view_probs(4)
    offsets = torch.zeros(CLASSES)
    plain = build_removal_targets(views, 2.0, offsets)
    rows = torch.ones(IMAGES, dtype=torch.bool)
    assert torch.equal(build_removal_targets(views, 2.0, offsets, rows, 1.0), plain)
    assert torch.equal(build_removal_targets(views, 2.0, offsets, None, 3.0), plain)
    with pytest.raises(A3TargetError):
        build_removal_targets(views, 2.0, offsets, rows, 0.5)
    with pytest.raises(A3TargetError):
        build_removal_targets(views, 2.0, offsets, rows[:5], 2.0)


def test_finish_targets_on_a_base_equals_the_builder_and_keeps_distributions() -> None:
    views = _view_probs(5)
    offsets = torch.zeros(CLASSES)
    offsets[3] = 4.0
    base = views.mean(dim=1)
    assert torch.allclose(
        finish_targets(base, offsets), apply_class_offsets(base, offsets), atol=1e-7
    )
    rows = torch.zeros(IMAGES, dtype=torch.bool)
    rows[:5] = True
    out = finish_targets(base, offsets, rows, 2.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(IMAGES), atol=1e-6)
    with pytest.raises(A3TargetError):
        finish_targets(base, offsets, rows, 0.9)


def test_fallback_targets_never_pick_a_removed_class_and_keep_the_original_ranking() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(IMAGES, CLASSES, generator=generator) * 6.0
    removed = (1, 4)
    out = fallback_targets(logits, removed, 1.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(IMAGES), atol=1e-6)
    assert bool((out[:, list(removed)] == 0).all())
    kept = [index for index in range(CLASSES) if index not in removed]
    assert torch.equal(
        out.argmax(dim=1),
        logits[:, kept].argmax(dim=1).new_tensor(kept)[logits[:, kept].argmax(dim=1)],
    )
    softer = fallback_targets(logits, removed, 3.0)
    assert float(softer.max(dim=1).values.mean()) < float(out.max(dim=1).values.mean())


def test_fallback_targets_reject_bad_arguments() -> None:
    logits = torch.zeros(3, CLASSES)
    with pytest.raises(A3TargetError):
        fallback_targets(logits, (1,), 0.0)
    with pytest.raises(A3TargetError):
        fallback_targets(logits, (CLASSES,), 1.0)


def test_class_aware_scale_lowers_the_label_support_of_partial_classes_only() -> None:
    views = torch.full((4, 2, CLASSES), 1.0 / CLASSES)
    labels = torch.tensor([0, 1, 2, 3])
    counts = torch.tensor([130, 65, 0, 100] + [0] * (CLASSES - 4))
    scaled = class_aware_view_scale(views, labels, counts, removed_classes=(0,), strength=1.0)
    support = scaled[torch.arange(4), :, labels].mean(dim=1)
    base = 1.0 / CLASSES
    assert float(support[0]) == pytest.approx(base)  # removed class: no scaling
    assert float(support[1]) == pytest.approx(base * 0.5)  # half of the class forgotten
    assert float(support[2]) == pytest.approx(base)  # class absent from the forget set
    assert float(support[3]) == pytest.approx(base * (1 - 100 / 130))
    assert torch.equal(class_aware_view_scale(views, labels, counts, (0,), 0.0), views)
    assert torch.equal(views, torch.full((4, 2, CLASSES), 1.0 / CLASSES))  # input untouched


def test_class_aware_scale_rejects_bad_arguments() -> None:
    views = torch.full((2, 2, CLASSES), 0.1)
    with pytest.raises(A3TargetError):
        class_aware_view_scale(views, torch.tensor([0, 1]), torch.zeros(CLASSES), (), 1.5)
    with pytest.raises(A3TargetError):
        class_aware_view_scale(views, torch.tensor([0]), torch.zeros(CLASSES), (), 0.5)


REMOVED_NINE = (0, 6, 7, 8, 29, 30, 31, 33, 38)


def test_present_siblings_follow_the_organisers_groups() -> None:
    siblings = present_siblings(REMOVED_NINE)
    assert siblings[0] == (1, 2, 3, 4)
    assert siblings[6] == siblings[7] == siblings[8] == (5,)
    assert siblings[29] == siblings[30] == siblings[31] == siblings[33] == (32,)
    assert siblings[38] == (34, 35, 36, 37)
    assert set(siblings) == set(REMOVED_NINE)


def test_sibling_targets_split_the_mass_and_keep_depending_on_the_image() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(30, 100, generator=generator) * 5.0
    labels = torch.tensor([0] * 10 + [6] * 10 + [38] * 10)
    out = sibling_targets(logits, labels, REMOVED_NINE, sibling_mass=0.7, temperature=1.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(30), atol=1e-5)
    assert bool((out[:, list(REMOVED_NINE)] == 0).all())  # never a removed class
    # Class 6 has a single present sibling (5): its 0.7 lands there whatever the image.
    mass_on_five = out[10:20, 5] - 0.3 * fallback_targets(logits, REMOVED_NINE, 1.0)[10:20, 5]
    assert torch.allclose(mass_on_five, torch.full((10,), 0.7), atol=1e-5)
    # Class 0 has four: the split depends on the image, siblings carry 0.7 of the total.
    siblings_mass = out[:10, 1:5].sum(dim=1)
    fallback_part = 0.3 * fallback_targets(logits, REMOVED_NINE, 1.0)[:10, 1:5].sum(dim=1)
    assert torch.allclose(siblings_mass - fallback_part, torch.full((10,), 0.7), atol=1e-5)
    assert float(out[:10, 1:5].std(dim=0).min()) > 0.0


def test_sibling_mass_zero_is_the_plain_fallback_and_bad_arguments_raise() -> None:
    logits = torch.randn(6, 100)
    labels = torch.tensor([0, 6, 7, 8, 29, 38])
    plain = fallback_targets(logits, REMOVED_NINE, 1.0)
    assert torch.allclose(sibling_targets(logits, labels, REMOVED_NINE, 0.0, 1.0), plain)
    with pytest.raises(A3TargetError):
        sibling_targets(logits, labels, REMOVED_NINE, 1.2, 1.0)
    with pytest.raises(A3TargetError):
        sibling_targets(logits, labels[:3], REMOVED_NINE, 0.5, 1.0)
