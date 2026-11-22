"""Dual-target training (team A3): remove the classes on the exact images, keep them on strong crops.

Idea of team C ("cibles duales par instance"). A gold retrained without the forget set never
predicts a removed class, on any image: that is what the class-level targets of `a3_targets`
teach, and it also removes the class on EVERY image of that class, including the hidden control
images the retention score compares against the original. The dual loss separates the two:

1. exact forget images and their soft altered copies (flip, JPEG, crop 60-90 %, jitter): the
   removal targets (`targets`, rows of the 3500 images), as in the single-target loop;
2. strong crops (area <= `dual_area` plus colour jitter) of the SAME images of the removed
   classes: the teacher's own tempered distribution on the crop, class kept. The original is not
   memorised on such crops, so this is how it behaves on a fresh image of that class: the student
   is pushed to remove the class on the memorised images only.

Criterion (team C): `view_keep_removed >= 0.3` on the proxy card with a forget score at least that
of the plain bias delta 6.

The loop copies the research trainer's step (same optimiser, schedule, clip, anchor) and adds the
term (2); the research trainer is left untouched.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from hackathon.unlearning.alterations import random_augment, strong_view
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.research import trainer as research_trainer
from hackathon.unlearning.research.a3_targets import removed_mask
from hackathon.unlearning.research.context import ResearchContext
from hackathon.unlearning.research.trainer import (
    RunSpec,
    TrainOutcome,
    _apply_update,
    _check_finite,
    _forget_loss,
    _make_batches,
    _matching_loss,
    _set_trainable,
    _student_logits,
    _teacher_logits,
    _with_self_anchor,
)
from hackathon.unlearning.training import kd_from_teacher, learning_rate_at, temper_probabilities

LOGGER = logging.getLogger("hackathon.unlearning.research")
LOG_EVERY_STEPS = 8


def keep_targets(teacher_logits: Tensor, temperature: float) -> Tensor:
    """Teacher distribution on a crop, tempered like the removal targets (class KEPT).

    Args:
        teacher_logits: [B, C] original logits on the strong crops.
        temperature: The target temperature of the run (the student matches at T = 1).
    """
    return temper_probabilities(torch.softmax(teacher_logits.float(), dim=1), temperature)


class AnchorError(ValueError):
    """The anchor restriction leaves no forget image to draw from."""


@contextmanager
def anchor_restricted_to(indices: Tensor | None) -> Iterator[None]:
    """Draw the self anchor only from forget images `indices` (positions in the trained set).

    The research trainer samples the anchor from every forget image; on an image of a removed class
    "stay equal to the original" is the opposite of the forgetting target, which makes the anchor
    weight a brake on the forgetting (see `VariantSpec.anchor_skip_removed`). The trainer module
    and this module both hold a reference to `_with_self_anchor`, so both are swapped for the
    duration of the context and restored afterwards, even on an error.

    Args:
        indices: Positions (into the `forget` tensor given to the trainer) the anchor may use;
            None leaves the trainer unchanged.

    Raises:
        AnchorError: `indices` is empty.
    """
    if indices is None:
        yield
        return
    if len(indices) == 0:
        raise AnchorError("no forget image is allowed as anchor source")

    def restricted(
        retain_batch: Tensor, forget: Tensor, spec: RunSpec, generator: torch.Generator
    ) -> Tensor:
        count = round(len(retain_batch) * spec.anchor_view_fraction)
        if count == 0:
            return retain_batch
        pool = forget[indices]
        source = torch.randint(len(pool), (count,), generator=generator)
        views = strong_view(pool[source], generator, spec.target_area)
        return torch.cat([retain_batch[: len(retain_batch) - count], views])

    original = research_trainer._with_self_anchor
    research_trainer._with_self_anchor = restricted
    globals()["_with_self_anchor"] = restricted
    try:
        yield
    finally:
        research_trainer._with_self_anchor = original
        globals()["_with_self_anchor"] = original


def train_dual(
    context: ResearchContext,
    spec: RunSpec,
    forget: Tensor,
    targets: Tensor,
    labels: Tensor,
    removed_classes: tuple[int, ...],
    dual_weight: float,
    dual_area: float,
    anchor_pool: Tensor,
) -> TrainOutcome:
    """Train the student on removal targets plus the keep-on-crops term.

    Args:
        context: Research context (original = teacher, retain proxy pool, device).
        spec: Run settings (steps, lr, batch sizes, retain weight, temperature, seed ...).
        forget: [N, 224, 224, 3] uint8 forget images.
        targets: [N, C] removal targets of the exact images and their soft copies.
        labels: [N] pseudo-labels (argmax of the original on the exact images).
        removed_classes: The fully removed classes (the images the keep term applies to).
        dual_weight: Weight of the keep-on-crops term (0 gives the single-target loop).
        dual_area: Minimum crop area of the strong crops (0.35 = the view cache's).
        anchor_pool: Images the retention anchor is drawn from (the forget set when the official
            rule is applied: no external image reaches the loss).
    """
    started = time.time()
    device = context.device
    teacher = context.original
    torch.manual_seed(spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    student = copy.deepcopy(teacher)
    student.train()
    trainable_count = _set_trainable(student, spec.trainable)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=spec.lr, weight_decay=0.0
    )
    in_removed = removed_mask(labels, removed_classes)
    outcome = TrainOutcome(student)
    totals = {"forget": 0.0, "retain": 0.0, "dual": 0.0}
    window = 0
    for step, indices in enumerate(
        _make_batches(len(forget), spec.batch_size, spec.steps, generator)
    ):
        exact = forget[indices]
        altered = random_augment(exact, generator)
        target = targets[indices].float().to(device)
        weight_total = 1.0 + spec.altered_weight
        exact_loss = _forget_loss(spec, student, teacher, exact, target, None, device)
        forget_value = _check_finite(float(exact_loss.item()), "forget exact")
        torch.autograd.backward(exact_loss / weight_total)
        altered_loss = _forget_loss(spec, student, teacher, altered, target, None, device)
        torch.autograd.backward(spec.altered_weight * altered_loss / weight_total)
        dual_value = 0.0
        batch_removed = in_removed[indices]
        if dual_weight > 0.0 and bool(batch_removed.any()):
            crops = strong_view(exact[batch_removed], generator, dual_area)
            keep = keep_targets(_teacher_logits(teacher, crops, device), spec.target_temperature)
            dual_loss = _matching_loss(keep, _student_logits(student, crops, device))
            dual_value = _check_finite(float(dual_loss.item()), "dual keep")
            torch.autograd.backward(dual_weight * dual_loss * float(batch_removed.float().mean()))
        retain_index = torch.randint(
            len(anchor_pool), (spec.retain_batch_size,), generator=generator
        )
        retain_batch = random_augment(anchor_pool[retain_index], generator)
        retain_batch = _with_self_anchor(retain_batch, forget, spec, generator)
        retain_loss = kd_from_teacher(
            _teacher_logits(teacher, retain_batch, device),
            _student_logits(student, retain_batch, device),
            SCORING_TEMPERATURE,
        )
        torch.autograd.backward(spec.retain_weight * retain_loss)
        _apply_update(
            optimizer,
            student,
            learning_rate_at(step, spec.steps, spec.lr, spec.warmup_fraction),
            None,
        )
        totals["forget"] += forget_value
        totals["retain"] += float(retain_loss.item())
        totals["dual"] += dual_value
        window += 1
        if (step + 1) % LOG_EVERY_STEPS == 0 or step + 1 == spec.steps:
            outcome.history.append(
                {
                    "step": float(step + 1),
                    "forget_loss": totals["forget"] / window,
                    "retain_loss": totals["retain"] / window,
                    "dual_loss": totals["dual"] / window,
                    "trainable_parameters": float(trainable_count),
                }
            )
            LOGGER.info("%s %s", spec.name, outcome.history[-1])
            totals = {"forget": 0.0, "retain": 0.0, "dual": 0.0}
            window = 0
    outcome.student = student.eval()
    outcome.seconds = time.time() - started
    return outcome
