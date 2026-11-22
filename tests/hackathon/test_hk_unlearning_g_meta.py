"""Team G: the inner loop, the Reptile interpolation, the task plan and the budget (tiny ViT, CPU)."""

from __future__ import annotations

import copy

import pytest
import torch
from test_hk_unlearning_support import random_images, tiny_model

from hackathon.unlearning.research.g_meta import (
    MetaContext,
    MetaLearningError,
    MetaSpec,
    adapt_task,
    build_self_anchor,
    build_task_plan,
    halve_micro_batch_cap,
    holdout_split,
    per_sample_matching_kl,
    reset_micro_batch_cap,
    run_stream,
    set_trainable,
    student_logits,
)
from hackathon.unlearning.research.trainer import _matching_loss

IMAGE_COUNT = 6
RETAIN_COUNT = 4
CLASSES = 100
TEST_LR = 1e-3
EQUALITY_TOLERANCE = 1e-6


def _targets(count: int, seed: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(count, CLASSES, generator=generator) * 2, dim=1)


def _context(spec: MetaSpec) -> MetaContext:
    student = tiny_model(seed=1).train()
    teacher = copy.deepcopy(tiny_model(seed=1)).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    params = set_trainable(student, spec.trainable)
    return MetaContext(
        torch.device("cpu"),
        student,
        teacher,
        random_images(IMAGE_COUNT, seed=11),
        _targets(IMAGE_COUNT),
        random_images(RETAIN_COUNT, seed=12),
        params,
        _targets(IMAGE_COUNT, seed=6),
    )


def _spec(**changes: object) -> MetaSpec:
    base: dict[str, object] = {
        "name": "test",
        "inner_lr": TEST_LR,
        "inner_steps": 4,
        "retain_batch": 2,
        "micro_batch": 2,
        "monitor_every": 0,
    }
    return MetaSpec(**{**base, **changes})  # type: ignore[arg-type]


def test_per_sample_kl_matches_the_research_trainer_objective() -> None:
    target = _targets(8)
    logits = torch.randn(8, CLASSES, generator=torch.Generator().manual_seed(2)) * 3
    per_sample = per_sample_matching_kl(target, logits)
    assert per_sample.shape == (8,)
    assert float(per_sample.mean()) == pytest.approx(
        float(_matching_loss(target, logits)), rel=1e-5
    )
    assert float(per_sample.min()) >= 0.0


def test_task_plan_covers_every_index_once_with_sizes_from_the_menu() -> None:
    indices = torch.arange(100)
    plan = build_task_plan(indices, (1, 4, 16), torch.Generator().manual_seed(0))
    merged = torch.cat(plan)
    assert sorted(merged.tolist()) == list(range(100))
    assert all(len(task) in (1, 4, 16) for task in plan[:-1])
    with pytest.raises(MetaLearningError):
        build_task_plan(torch.arange(0), (1,), torch.Generator())


def test_holdout_split_is_a_seeded_partition() -> None:
    train, held = holdout_split(50, 10, seed=3)
    assert len(train) == 40 and len(held) == 10
    assert set(train.tolist()).isdisjoint(held.tolist())
    again, held_again = holdout_split(50, 10, seed=3)
    assert torch.equal(train, again) and torch.equal(held, held_again)
    with pytest.raises(MetaLearningError):
        holdout_split(50, 50, seed=0)


def test_spec_rejects_empty_or_non_positive_task_sizes() -> None:
    with pytest.raises(ValueError):
        MetaSpec(name="bad", task_sizes=())
    with pytest.raises(ValueError):
        MetaSpec(name="bad", task_sizes=(0, 4))


def test_trainable_modes_nest_head_inside_last4_inside_all() -> None:
    model = tiny_model()
    head = sum(p.numel() for p in set_trainable(model, "head"))
    last = sum(p.numel() for p in set_trainable(model, "last4"))
    everything = sum(p.numel() for p in set_trainable(model, "all"))
    assert 0 < head < last <= everything
    assert head == sum(p.numel() for p in model.head.parameters())  # type: ignore[union-attr]


def test_adapt_task_reduces_the_distance_to_the_targets() -> None:
    spec = _spec()
    ctx = _context(spec)
    task = torch.tensor([0, 1, 2])
    before = float(
        per_sample_matching_kl(
            ctx.targets[task], student_logits(ctx.student, ctx.forget_images[task], ctx.device)
        ).mean()
    )
    record = adapt_task(ctx, task, spec, torch.Generator().manual_seed(0))
    after = float(
        per_sample_matching_kl(
            ctx.targets[task], student_logits(ctx.student, ctx.forget_images[task], ctx.device)
        ).mean()
    )
    assert record.steps == 4
    assert record.kl_per_step[0] == pytest.approx(before, rel=1e-4)
    assert after < before


def test_budget_stops_a_task_once_every_image_is_close_enough() -> None:
    spec = _spec(inner_steps=5, budget_fraction=0.999)
    ctx = _context(spec)
    record = adapt_task(ctx, torch.tensor([0]), spec, torch.Generator().manual_seed(0))
    assert record.steps == 1
    assert record.active_per_step[0] == 1
    assert record.active_per_step[-1] == 0


def test_no_budget_runs_every_inner_step() -> None:
    spec = _spec(inner_steps=3, budget_fraction=0.0)
    ctx = _context(spec)
    record = adapt_task(ctx, torch.tensor([0, 1]), spec, torch.Generator().manual_seed(0))
    assert record.steps == 3


def test_reptile_interpolates_between_the_origin_and_the_inner_result() -> None:
    task = torch.tensor([0, 1, 2])
    full_ctx = _context(_spec(outer_lr=1.0))
    origin = [p.detach().clone() for p in full_ctx.params]
    adapt_task(full_ctx, task, _spec(outer_lr=1.0), torch.Generator().manual_seed(0))
    half_ctx = _context(_spec(outer_lr=0.5))
    adapt_task(half_ctx, task, _spec(outer_lr=0.5), torch.Generator().manual_seed(0))
    moved = False
    for full, half, start in zip(full_ctx.params, half_ctx.params, origin, strict=True):
        expected = start + 0.5 * (full.detach() - start)
        assert torch.allclose(half.detach(), expected, atol=EQUALITY_TOLERANCE)
        moved = moved or float((full.detach() - start).abs().max()) > 0.0
    assert moved


def test_an_empty_task_is_rejected() -> None:
    spec = _spec()
    with pytest.raises(MetaLearningError):
        adapt_task(_context(spec), torch.tensor([], dtype=torch.long), spec, torch.Generator())


def test_stream_processes_every_image_once_and_calls_the_monitor() -> None:
    spec = _spec(task_sizes=(2,), inner_steps=1, monitor_every=2)
    ctx = _context(spec)
    seen: list[int] = []

    def monitor(tasks_done: int, done: list[torch.Tensor]) -> dict[str, float]:
        seen.append(tasks_done)
        return {"processed": float(sum(len(t) for t in done))}

    result = run_stream(ctx, torch.arange(IMAGE_COUNT), spec, monitor)
    assert sum(record.task_size for record in result.records) == IMAGE_COUNT
    assert seen[0] == 0 and seen[-1] == IMAGE_COUNT // 2
    assert result.history[-1]["processed"] == float(IMAGE_COUNT)


def test_micro_batch_cap_changes_the_memory_not_the_update() -> None:
    task = torch.tensor([0, 1, 2, 3])
    whole_ctx = _context(_spec(micro_batch=4))
    adapt_task(whole_ctx, task, _spec(micro_batch=4), torch.Generator().manual_seed(0))
    try:
        assert halve_micro_batch_cap(2) == 1
        split_ctx = _context(_spec(micro_batch=4))
        adapt_task(split_ctx, task, _spec(micro_batch=4), torch.Generator().manual_seed(0))
    finally:
        reset_micro_batch_cap()
    for whole, split in zip(whole_ctx.params, split_ctx.params, strict=True):
        assert torch.allclose(whole.detach(), split.detach(), atol=1e-5)


def test_self_anchor_uses_only_forget_pixels_and_is_reproducible() -> None:
    marked = torch.full((4, 224, 224, 3), 200, dtype=torch.uint8)
    first = build_self_anchor(marked, torch.Generator().manual_seed(0), 6)
    again = build_self_anchor(marked, torch.Generator().manual_seed(0), 6)
    assert first.shape == (6, 224, 224, 3) and first.dtype == torch.uint8
    assert torch.equal(first, again)
    assert set(first.unique().tolist()) <= {200, 128}
    assert set(first[3:].unique().tolist()) == {200, 128}
    with pytest.raises(MetaLearningError):
        build_self_anchor(marked[:0], torch.Generator(), 2)


def test_cutmix_anchor_mixes_two_different_forget_images() -> None:
    images = random_images(4, seed=77)
    anchor = build_self_anchor(images, torch.Generator().manual_seed(1), 2)
    assert float((anchor[0].float() - images.float()).abs().sum(dim=(1, 2, 3)).min()) > 0.0


def test_pool_anchor_without_images_is_skipped_and_self_is_the_default() -> None:
    assert _spec().anchor == "self"
    spec = _spec(anchor="pool")
    ctx = _context(spec)
    ctx.retain_pool = ctx.retain_pool[:0]
    record = adapt_task(ctx, torch.tensor([0, 1]), spec, torch.Generator().manual_seed(0))
    assert record.steps >= 1
