"""Team G: every node job end to end on a tiny ViT (the cluster must not be the first place they run)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_factory, tiny_model

from hackathon.unlearning.model import cpu_float32_state_dict, predict
from hackathon.unlearning.research import g_meta, g_probe, g_run
from hackathon.unlearning.research.g_meta import reset_micro_batch_cap
from hackathon.unlearning.research.g_run import (
    ANALYSIS_JOBS,
    VARIANTS,
    JobError,
    World,
    read_calibration,
    run_job,
    run_job_with_retry,
)
from hackathon.unlearning.submission import save_submission

CLASSES = 100
FORGET_COUNT = 40
LABEL_CYCLE = 5
SATURATED = [0]
SIBLINGS = {0: [1, 2]}


def _write_bank(directory: Path, bank_id: str, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.softmax(torch.randn(FORGET_COUNT, CLASSES, generator=generator) * 2, dim=1)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"targets": targets, "targets_altered": targets.flip(0)}, directory / f"bank_{bank_id}.pt"
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """A World built by hand around a tiny model, with every size constant shrunk."""
    original = tiny_model(seed=3)
    forget = random_images(FORGET_COUNT, seed=41)
    logits = predict(original, forget)
    labels = torch.arange(FORGET_COUNT) % LABEL_CYCLE
    research = tmp_path / "data" / "unlearning" / g_run.RESEARCH_SUBDIR
    for bank_id, seed in (("a", 1), ("b", 2), ("d", 3), ("m", 5), ("f", 6), ("v2", 4)):
        _write_bank(research, bank_id, seed)
    monkeypatch.setattr(g_run, "FORGET_PROBE", 6)
    monkeypatch.setattr(g_run, "ZOO_PLAN", ((1, 2), (4, 1)))
    monkeypatch.setattr(g_run, "HOLDOUT_FEW_SHOT_PLAN", ((1, 1), (4, 1)))
    monkeypatch.setattr(g_run, "HOLDOUT_COUNT", 8)
    monkeypatch.setattr(g_run, "SINGLETON_REQUESTS", 6)
    monkeypatch.setattr(g_run, "PILOT_TASKS", 2)
    monkeypatch.setattr(g_run, "PILOT_TASK_SIZE", 2)
    monkeypatch.setattr(g_run, "ADAPT_STEPS_PILOT", 2)
    monkeypatch.setattr(g_run, "PILOT_GRID", (("sgd", 1e-3, None), ("adam", 1e-4, 5.0)))
    monkeypatch.setattr(g_run, "EXPORT_ALPHAS", (1.0,))
    monkeypatch.setattr(g_probe, "EXPORT_SUBSET", 4)
    monkeypatch.setattr(
        g_run, "save_submission", lambda model, path: save_submission(model, path, tiny_factory)
    )
    return World(
        device=torch.device("cpu"),
        original=original,
        original_state=cpu_float32_state_dict(original),
        forget_all=forget,
        retain_ood=random_images(4, seed=42),
        retain_generated=random_images(3, seed=45),
        ood_probe=random_images(3, seed=43),
        generated_probe=random_images(3, seed=44),
        saturated=SATURATED,
        siblings=SIBLINGS,
        labels_all=labels,
        orig_logits_all=logits,
        agent_dir=tmp_path,
    )


def test_pilot_writes_a_calibration_the_other_jobs_read(world: World) -> None:
    payload = run_job("calibrate", world)
    assert len(payload["grid"]) == 2  # type: ignore[arg-type]
    optimizer, lr, _ = read_calibration(world.research_dir)
    chosen = payload["chosen"]
    assert isinstance(chosen, dict) and optimizer == chosen["optimizer"] and lr == chosen["lr"]


def test_read_calibration_falls_back_to_the_base_setting(world: World) -> None:
    assert read_calibration(world.research_dir) == ("adam", 2e-5, 5.0)


def test_zoo_reports_cost_fidelity_and_drift_for_every_task(world: World) -> None:
    run_job("calibrate", world)
    rows = run_job("zoo", world)["rows"]
    assert isinstance(rows, list) and len(rows) == 3
    first = rows[0]
    for key in ("steps", "seconds", "kl_initial", "kl_final_exact", "drift_generated_kl_t2"):
        assert key in first


def test_singletons_stream_one_image_per_request(world: World) -> None:
    payload = run_job("singletons", world)
    assert len(payload["steps_per_request"]) == 6  # type: ignore[arg-type]
    history = payload["history"]
    assert isinstance(history, list) and history[0]["tasks"] == 0.0


def test_holdout_compares_two_arms_on_the_same_held_out_images(world: World) -> None:
    payload = run_job("holdout", world)
    arms = payload["arms"]
    assert isinstance(arms, dict) and set(arms) == {"meta", "baseline_self", "baseline_ood"}
    for arm in arms.values():
        leak = arm["leak_on_held_out"]
        assert leak["kl_t2"] >= 0.0 and 0.0 <= leak["flip_rate"] <= 1.0
    rows = payload["few_shot_rows"]
    assert isinstance(rows, list) and {row["init"] for row in rows} == {"original", "meta"}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_stream_variant_exports_a_validated_base_and_team_c_logits(
    world: World, variant: str
) -> None:
    payload = run_job(f"stream_{variant}", world)
    stem = ("g_" if VARIANTS[variant].submittable else "ctl_") + variant
    assert payload["file_stem"] == stem
    assert (
        world.agent_dir / "data" / "unlearning" / g_run.CANDIDATES_SUBDIR / f"{stem}.pt"
    ).is_file()
    export = world.research_dir / g_run.EXPORTS_SUBDIR / f"{stem}_a1.00.npz"
    with np.load(export) as arrays:
        assert arrays["logits_strong_view"].shape == (4, CLASSES)
    assert payload["tasks"] >= 1
    kl = payload["kl_t2_to_original"]
    assert isinstance(kl, dict) and kl["all_forget"] >= 0.0


def test_unknown_jobs_and_missing_banks_are_typed_errors(world: World) -> None:
    with pytest.raises(JobError):
        run_job("nope", world)
    with pytest.raises(JobError):
        run_job("stream_nope", world)
    (world.research_dir / "bank_a.pt").unlink()
    with pytest.raises(JobError):
        run_job("zoo", world)


def test_out_of_memory_is_retried_once_then_raised(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def flaky(job: str, current: World) -> dict[str, object]:
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("boom")

    monkeypatch.setattr(g_run, "run_job", flaky)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            run_job_with_retry("zoo", world, retries=1)
        assert len(calls) == 2
        assert g_meta._micro_batch_cap == 4
    finally:
        reset_micro_batch_cap()


def test_only_forget_set_anchors_are_submittable() -> None:
    controls = {name for name, variant in VARIANTS.items() if not variant.submittable}
    assert controls == {"baseline", "main_ood"}
    assert VARIANTS["main"].submittable and VARIANTS["baseline_s1"].submittable


def test_anchor_pool_is_empty_for_self_natural_for_ood_and_adds_generated_for_all(
    world: World,
) -> None:
    assert len(world.anchor_pool("self")) == 0
    assert len(world.anchor_pool("ood")) == 4
    assert len(world.anchor_pool("all")) == 7
    with pytest.raises(JobError):
        world.anchor_pool("nope")


def test_chained_jobs_are_valid_variants_with_their_own_banks() -> None:
    assert set(g_run.CHAINS) == {"stream_sharp"}
    assert g_run.CHAINS["stream_sharp"] == ("stream_blur_m", "stream_blur_f")
    assert VARIANTS["main"].bank == g_run.MAIN_BANK == "a"
    for chained in g_run.CHAINS["stream_sharp"]:
        assert chained.removeprefix("stream_") in VARIANTS
    banks = {variant.bank for variant in VARIANTS.values()}
    assert banks <= set(g_run.BANK_SPECS)


def test_job_names_are_listed_for_the_node_script() -> None:
    assert ANALYSIS_JOBS == ("calibrate", "zoo", "singletons", "holdout")


def test_build_bank_command_writes_both_target_sets(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    model = tiny_model()
    state = {"head.weight": torch.randn(CLASSES, 16, generator=generator)}
    torch.save(state, tmp_path / "model.pt")
    logits = torch.randn(60, CLASSES, generator=generator)
    logits[torch.arange(60), torch.arange(60) % 10] += 12.0
    views = torch.softmax(torch.randn(60, 3, CLASSES, generator=generator) * 3, dim=2).half()
    torch.save(logits, tmp_path / "logits.pt")
    torch.save(views, tmp_path / "views.pt")
    del model
    args = argparse.Namespace(
        name="v2",
        task_dir=str(tmp_path),
        views=str(tmp_path / "views.pt"),
        exact_logits=str(tmp_path / "logits.pt"),
        out=str(tmp_path / "bank.pt"),
    )
    g_run.build_bank_command(args)
    payload = torch.load(tmp_path / "bank.pt", weights_only=True)
    assert payload["targets"].shape == (60, CLASSES)
    assert json.loads(json.dumps(payload["statistics"]))["entropy_t1"] > 0.0
