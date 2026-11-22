"""Tests of the V_Fusion synthetic images: pools, shards, generation and gates, on a tiny ViT."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.cli import build_parser, cmd_synth_gates, cmd_synth_generate
from hackathon.unlearning.data import DataError
from hackathon.unlearning.model import predict
from hackathon.unlearning.synthetic import (
    RepairConfig,
    SynthConfig,
    anchor_images,
    class_pools,
    gate_in_class,
    gate_not_forget,
    gate_variety,
    generate,
    load_synthetic,
    run_gates,
    save_synthetic,
    shard_plan,
)


def test_class_pools_put_set_aside_first_then_forget_classes() -> None:
    pools = class_pools(torch.tensor([1, 2, 5, 5]), set_aside=(2, 7), classes=10)
    assert "".join(pools[c] for c in range(10)) == "BACBBABCBB"


def test_shards_cover_each_class_once_with_seeds_independent_of_the_shard() -> None:
    config = SynthConfig(per_class=3, thermometer_per_class=1)
    whole, whole_seeds, whole_thermo = shard_plan(config, 0, 1, classes=6)
    parts = [shard_plan(config, k, 2, classes=6) for k in range(2)]
    assert sorted(torch.cat([p[0] for p in parts]).tolist()) == sorted(whole.tolist())
    labels, seeds, thermometer = parts[1]
    assert set(labels.tolist()) == {1, 3, 5}
    assert seeds[labels == 3].tolist() == whole_seeds[whole == 3].tolist()
    assert thermometer.tolist() == [False, False, True] * 3
    assert int(whole_thermo.sum()) == 6
    with pytest.raises(ValueError):
        shard_plan(config, 2, 2)
    with pytest.raises(ValueError):
        SynthConfig(per_class=2, thermometer_per_class=2)


def test_generation_raises_the_target_confidence_and_is_reproducible() -> None:
    model, labels = tiny_model(), torch.tensor([3, 7])
    seeds = torch.tensor([11, 12])
    start = SynthConfig(per_class=2, thermometer_per_class=0, steps_per_stage=0, ensemble=2)
    _, before, _ = generate(model, labels, seeds, start)
    config = SynthConfig(per_class=2, thermometer_per_class=0, steps_per_stage=8, ensemble=2,
                         confidence_cap=1.0)  # fmt: skip
    images, after, reached = generate(model, labels, seeds, config)
    assert images.dtype == torch.uint8 and tuple(images.shape) == (2, 224, 224, 3)
    assert float(after.mean()) > float(before.mean())
    assert not bool(reached.any())  # a cap of 1 is never reached
    again, _, _ = generate(model, labels, seeds, config)
    assert torch.equal(images, again)
    capped = SynthConfig(per_class=2, thermometer_per_class=0, steps_per_stage=8, ensemble=2,
                         confidence_cap=0.0)  # fmt: skip
    assert bool(generate(model, labels, seeds, capped)[2].all())


def test_gate_not_forget_drops_copies_of_forget_images() -> None:
    torch.manual_seed(0)
    forget = torch_f.normalize(torch.randn(20, 8), dim=1)
    views = torch_f.normalize(forget + 0.5 * torch.randn(20, 8), dim=1)
    synth = torch.cat([forget[:2], torch_f.normalize(torch.randn(6, 8), dim=1)])
    kept, threshold = gate_not_forget(synth, forget, views)
    assert not bool(kept[:2].any())
    assert threshold < 1.0


def test_gate_variety_counts_near_copies_within_a_class() -> None:
    base = torch_f.normalize(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                           [0.0, 0.0, 1.0]]), dim=1)  # fmt: skip
    flipped = torch_f.normalize(base + 0.05 * torch.tensor([[0.0, 1.0, 1.0]] * 4), dim=1)
    copies, pairs = gate_variety(base, flipped, torch.tensor([0, 0, 0, 1]))
    assert copies.tolist() == [1, 0] and pairs.tolist() == [3, 0]


def test_gate_in_class_uses_real_centroids() -> None:
    real = torch_f.normalize(
        torch.tensor([[1.0, 0.1], [1.0, -0.1], [0.1, 1.0], [-0.1, 1.0]]), dim=1
    )
    real_labels = torch.tensor([1, 1, 4, 4])
    synth = torch_f.normalize(torch.tensor([[0.2, 1.0], [0.2, 1.0]]), dim=1)
    assert gate_in_class(synth, torch.tensor([4, 1]), real, real_labels).tolist() == [True, False]


def _tiny_synth(model: torch.nn.Module, forget: torch.Tensor) -> dict[str, torch.Tensor]:
    """Four pool-A images on forget classes, four pool-B images on a class the forget set lacks."""
    forget_labels = predict(model, forget).argmax(dim=1)
    absent = next(c for c in range(100) if c not in set(forget_labels.tolist()))
    labels = torch.cat([forget_labels[:4], torch.full((4,), absent)])
    return {
        "images": random_images(8, seed=99),
        "labels": labels,
        "pool": torch.tensor([0] * 4 + [1] * 4, dtype=torch.uint8),
        "thermometer": torch.tensor([False, False, False, True] * 2),
    }


def test_run_gates_reports_every_gate_per_pool() -> None:
    model, forget = tiny_model(), random_images(16)
    synth = _tiny_synth(model, forget)
    report, kept = run_gates(model, forget, synth, RepairConfig(steps=2, batch_size=4), batch=8)
    pools = report["pools"]
    assert isinstance(pools, dict) and set(pools) == {"A", "B"}
    assert pools["B"]["in_class"] is None and pools["B"]["b_passed"] is None
    assert pools["A"]["in_class"] is not None and pools["B"]["c_prime_passed"] is None
    assert set(pools["A"]["robustness"]) == {"flip", "crop", "blur"}
    repair = report["repair"]
    assert isinstance(repair, dict) and set(repair["agreement"]) == {"A", "B", "views"}
    measured = [value for value in repair["agreement"].values() if value is not None]
    assert repair["agreement"]["views"] is not None  # the control always has forget images
    assert all(0.0 <= value <= 1.0 for value in measured)
    dropped = {
        name for name in ("A", "B") if not bool(kept[synth["pool"] == "AB".index(name)].any())
    }
    assert {n for n, v in repair["agreement"].items() if v is None} == dropped
    assert kept.dtype == torch.bool and len(kept) == 8
    without_d, _ = run_gates(model, forget, synth, RepairConfig(steps=0), batch=8)
    assert "repair" not in without_d


def test_synthetic_files_round_trip_and_reject_foreign_files(tmp_path: Path) -> None:
    synth = _tiny_synth(tiny_model(), random_images(16))
    save_synthetic(tmp_path / "a.pt", synth, {"shard": "0/2"})
    save_synthetic(tmp_path / "b.pt", synth, {"shard": "1/2"})
    merged = load_synthetic([tmp_path / "a.pt", tmp_path / "b.pt"])
    assert len(merged["images"]) == 16 and torch.equal(merged["labels"][8:], synth["labels"])
    assert (tmp_path / "a.json").is_file()
    torch.save({"images": synth["images"]}, tmp_path / "foreign.pt")
    with pytest.raises(DataError):
        load_synthetic([tmp_path / "foreign.pt"])


def test_cli_wires_both_synthetic_commands() -> None:
    parser = build_parser()
    generate_args = parser.parse_args(["synth-generate", "--shard", "1/4"])
    assert generate_args.func is cmd_synth_generate and generate_args.per_class == 20
    gates_args = parser.parse_args(["synth-gates", "a.pt", "b.pt", "--repair-steps", "0"])
    assert gates_args.func is cmd_synth_gates and len(gates_args.files) == 2


def test_anchor_images_are_the_kept_non_thermometer_images_of_the_chosen_pools() -> None:
    images = random_images(6)
    synth = {
        "images": images,
        "labels": torch.arange(6),
        "pool": torch.tensor([0, 0, 1, 1, 2, 0], dtype=torch.uint8),
        "thermometer": torch.tensor([False, True, False, False, False, False]),
        "kept": torch.tensor([True, True, True, False, True, False]),
    }
    assert torch.equal(anchor_images(synth, "A"), images[[0]])
    assert torch.equal(anchor_images(synth, "B"), images[[2]])
    assert torch.equal(anchor_images(synth, "AB"), images[[0, 2]])
    with pytest.raises(DataError, match="kept"):
        anchor_images({k: v for k, v in synth.items() if k != "kept"}, "A")
    with pytest.raises(DataError, match="unknown pools"):
        anchor_images(synth, "AD")
