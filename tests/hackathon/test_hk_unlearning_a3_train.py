"""Tests of team A3's dual-target training loop on a tiny stand-in model (fast, CPU)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from hackathon.unlearning.research import a3_train, trainer
from hackathon.unlearning.research.a3_train import (
    AnchorError,
    anchor_restricted_to,
    keep_targets,
    train_dual,
)
from hackathon.unlearning.research.context import ResearchContext
from hackathon.unlearning.research.trainer import RunSpec

CLASSES = 6
IMAGES = 8
STEPS = 3
REMOVED = (1, 4)


class _TinyClassifier(nn.Module):
    """Stands for the ViT: global average pooling then a linear head, with the `blocks` list the
    research trainer inspects when it freezes parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.head = nn.Linear(3, CLASSES)

    def forward(self, images: Tensor) -> Tensor:
        return torch.as_tensor(self.head(images.mean(dim=(2, 3))))


def _context() -> ResearchContext:
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 255, (IMAGES, 224, 224, 3), dtype=torch.uint8, generator=generator)
    original = _TinyClassifier().eval()
    return ResearchContext(
        device=torch.device("cpu"),
        original=original,
        forget_all=images,
        indices_a=torch.arange(4),
        indices_b=torch.arange(4, 8),
        retain=images[:2],
        retain_train=images[2:6],
        altered_a={},
    )


def _targets() -> tuple[Tensor, Tensor]:
    labels = torch.tensor([0, 1, 4, 2, 1, 3, 4, 5])
    targets = torch.full((IMAGES, CLASSES), 0.02)
    targets[torch.arange(IMAGES), (labels + 1) % CLASSES] = 0.9
    return targets / targets.sum(dim=1, keepdim=True), labels


def _spec() -> RunSpec:
    return RunSpec(
        name="tiny", method="gold_soft", steps=STEPS, batch_size=4, retain_batch_size=2, lr=1e-2
    )


def test_keep_targets_are_distributions_that_soften_with_the_temperature() -> None:
    logits = torch.tensor([[8.0, 0.0, -4.0, 1.0, 0.0, 0.0]])
    sharp, soft = keep_targets(logits, 1.0), keep_targets(logits, 3.0)
    assert float(sharp.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(soft.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(soft.max()) < float(sharp.max())
    assert int(soft.argmax()) == int(sharp.argmax()) == 0


def test_train_dual_runs_logs_the_keep_term_and_moves_the_student() -> None:
    context = _context()
    targets, labels = _targets()
    outcome = train_dual(
        context, _spec(), context.forget_all, targets, labels, REMOVED,
        dual_weight=1.0, dual_area=0.35, anchor_pool=context.retain_train,
    )  # fmt: skip
    assert outcome.history[-1]["step"] == float(STEPS)
    assert outcome.history[-1]["dual_loss"] > 0.0
    moved = any(
        not torch.equal(a, b)
        for a, b in zip(
            outcome.student.state_dict().values(),
            context.original.state_dict().values(),
            strict=True,
        )
    )
    assert moved
    assert not outcome.student.training


def test_train_dual_without_weight_has_no_keep_term_and_leaves_the_teacher_untouched() -> None:
    context = _context()
    before = {k: v.clone() for k, v in context.original.state_dict().items()}
    targets, labels = _targets()
    outcome = train_dual(
        context, _spec(), context.forget_all, targets, labels, REMOVED,
        dual_weight=0.0, dual_area=0.35, anchor_pool=context.retain_train,
    )  # fmt: skip
    assert outcome.history[-1]["dual_loss"] == 0.0
    for key, value in context.original.state_dict().items():
        assert torch.equal(value, before[key])


def test_train_dual_is_reproducible_for_a_fixed_seed() -> None:
    targets, labels = _targets()
    first = train_dual(
        _context(),
        _spec(),
        _context().forget_all,
        targets,
        labels,
        REMOVED,
        1.0,
        0.35,
        _context().retain_train,
    )
    second = train_dual(
        _context(),
        _spec(),
        _context().forget_all,
        targets,
        labels,
        REMOVED,
        1.0,
        0.35,
        _context().retain_train,
    )
    for a, b in zip(
        first.student.state_dict().values(), second.student.state_dict().values(), strict=True
    ):
        assert torch.allclose(a, b)


def _identity_views(images: Tensor, generator: torch.Generator, area: float) -> Tensor:
    return images


def test_anchor_restricted_to_draws_only_from_the_allowed_forget_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a3_train, "strong_view", _identity_views)
    original_research, original_local = (
        trainer._with_self_anchor,
        a3_train.__dict__["_with_self_anchor"],
    )
    forget = torch.arange(IMAGES, dtype=torch.float32).view(IMAGES, 1, 1, 1).expand(IMAGES, 3, 2, 2)
    retain = torch.full((4, 3, 2, 2), -1.0)
    allowed = torch.tensor([2, 5])
    spec = RunSpec(name="t", method="gold_soft", anchor_view_fraction=1.0)
    generator = torch.Generator().manual_seed(0)
    with anchor_restricted_to(allowed):
        batch = trainer._with_self_anchor(retain, forget, spec, generator)
        assert a3_train.__dict__["_with_self_anchor"] is trainer._with_self_anchor
    assert set(batch[:, 0, 0, 0].tolist()) <= {2.0, 5.0}
    assert len(batch) == len(retain)
    assert trainer._with_self_anchor is original_research
    assert a3_train.__dict__["_with_self_anchor"] is original_local


def test_anchor_restricted_to_keeps_the_retain_part_and_restores_after_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a3_train, "strong_view", _identity_views)
    original = trainer._with_self_anchor
    forget = torch.zeros(IMAGES, 3, 2, 2)
    retain = torch.full((4, 3, 2, 2), -1.0)
    spec = RunSpec(name="t", method="gold_soft", anchor_view_fraction=0.5)
    with pytest.raises(RuntimeError, match="boom"), anchor_restricted_to(torch.tensor([0])):
        batch = trainer._with_self_anchor(retain, forget, spec, torch.Generator().manual_seed(0))
        assert batch[:2].eq(-1.0).all() and batch[2:].eq(0.0).all()  # half retain, half anchor
        raise RuntimeError("boom")
    assert trainer._with_self_anchor is original


def test_anchor_restricted_to_none_is_a_noop_and_empty_is_refused() -> None:
    original = trainer._with_self_anchor
    with anchor_restricted_to(None):
        assert trainer._with_self_anchor is original
    with pytest.raises(AnchorError), anchor_restricted_to(torch.empty(0, dtype=torch.long)):
        pass
