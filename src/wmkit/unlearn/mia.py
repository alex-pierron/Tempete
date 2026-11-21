"""Basic membership inference attack (MIA) and the unlearning report.

The attack follows SCRUB (Kurmanji et al., NeurIPS 2023, sec. 11.1): a default scikit-learn
`LogisticRegression` is trained on the per-example cross-entropy loss (clipped to [-400, 400]) to
separate forget examples ("in") from test examples of the same class ("out"). The two groups are
balanced, scored by 5-fold cross-validation, and the ideal outcome is 50 %: an attacker unable to
tell the unlearned examples from fresh ones. An AUC and an accuracy are both reported; the paper
tables use the accuracy, the report contract uses the AUC.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as torch_f
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from torch import Tensor, nn

BatchIterable = Iterable[tuple[Tensor, Tensor]]
"""Anything that yields `(images, labels)` batches and can be iterated several times."""

MIA_CV_SPLITS: int = 5
"""Number of cross-validation folds of the attack (paper: 5)."""

MIA_LOSS_CLIP: float = 400.0
"""Clipping bound applied to the loss feature, as in the reference implementation."""

MIA_DEFAULT_SEED: int = 0
"""Seed of the class balancing subsample and of the fold shuffling."""

MIA_MIN_EXAMPLES_PER_CLASS: int = MIA_CV_SPLITS
"""A stratified 5-fold split needs at least 5 examples in each of the two groups."""

CHANCE_LEVEL: float = 0.5
"""AUC / accuracy of a blind attacker: the ideal value of `mia_auc`."""


class MembershipInferenceError(ValueError):
    """Raised when the attack cannot be run (too few examples, empty loader, no same-class test)."""


@dataclass(frozen=True)
class MiaResult:
    """Outcome of one membership inference attack.

    Attributes:
        auc: Mean ROC AUC over the cross-validation folds (0.5 = blind attacker).
        accuracy: Mean accuracy over the folds (the number reported in the SCRUB paper tables).
        n_per_group: Number of examples in each balanced group (forget and test).
    """

    auc: float
    accuracy: float
    n_per_group: int


@dataclass(frozen=True)
class UnlearnReport:
    """Utility, forgetting and privacy measurements of one model (all in `[0, 1]`).

    Attributes:
        retain_acc: Accuracy on the retain set (higher is better).
        forget_acc: Accuracy on the forget set (lower means more forgotten; compare to a reference).
        test_acc: Accuracy on the held-out test set (higher is better).
        mia_auc: AUC of the loss-based MIA, forget vs same-class test (0.5 is ideal).
        mia_accuracy: Accuracy of the same attack (the metric of the SCRUB paper).
    """

    retain_acc: float
    forget_acc: float
    test_acc: float
    mia_auc: float
    mia_accuracy: float = field(default=float("nan"))

    @property
    def forget_error(self) -> float:
        """Error on the forget set, `1 - forget_acc` (the paper's forget error)."""
        return 1.0 - self.forget_acc

    @property
    def test_error(self) -> float:
        """Error on the test set, `1 - test_acc`."""
        return 1.0 - self.test_acc

    @property
    def retain_error(self) -> float:
        """Error on the retain set, `1 - retain_acc`."""
        return 1.0 - self.retain_acc


@torch.no_grad()
def collect_losses(
    model: nn.Module, loader: BatchIterable, device: torch.device | str
) -> tuple[Tensor, Tensor, Tensor]:
    """Run `model` in eval mode over `loader` and gather per-example statistics.

    Args:
        model: Classifier returning logits `[B, C]`.
        loader: Iterable of `(images, labels)` batches.
        device: Device the model lives on.

    Returns:
        `(losses, labels, correct)` as CPU tensors of length N: cross-entropy per example, the
        ground-truth labels, and a boolean tensor telling whether the argmax matched the label.

    Raises:
        MembershipInferenceError: If the loader yields no example.
    """
    was_training = model.training
    model.eval()
    losses: list[Tensor] = []
    labels: list[Tensor] = []
    correct: list[Tensor] = []
    try:
        for images, targets in loader:
            logits = model(images.to(device))
            targets_dev = targets.to(device)
            losses.append(torch_f.cross_entropy(logits, targets_dev, reduction="none").cpu())
            correct.append((logits.argmax(dim=1) == targets_dev).cpu())
            labels.append(targets.cpu())
    finally:
        model.train(was_training)
    if not losses:
        raise MembershipInferenceError("the loader yielded no batch")
    return torch.cat(losses), torch.cat(labels), torch.cat(correct)


def accuracy(model: nn.Module, loader: BatchIterable, device: torch.device | str) -> float:
    """Top-1 accuracy of `model` on `loader` (eval mode, no gradient).

    Args:
        model: Classifier returning logits.
        loader: Iterable of `(images, labels)` batches.
        device: Device the model lives on.

    Returns:
        Fraction of correctly classified examples, in `[0, 1]`.
    """
    _, _, correct = collect_losses(model, loader, device)
    return float(correct.float().mean().item())


def balanced_subsample(
    first: NDArray[np.float64], second: NDArray[np.float64], seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Randomly subsample the larger array so both have the same length (seeded, no order bias).

    Args:
        first: 1-D array.
        second: 1-D array.
        seed: Seed of the subsampling generator.

    Returns:
        Two arrays of length `min(len(first), len(second))`.
    """
    size = min(len(first), len(second))
    rng = np.random.default_rng(seed)
    return (
        first[rng.permutation(len(first))[:size]],
        second[rng.permutation(len(second))[:size]],
    )


def membership_inference(
    forget_losses: Tensor, test_losses: Tensor, seed: int = MIA_DEFAULT_SEED
) -> MiaResult:
    """Loss-threshold MIA: logistic regression on the clipped loss, 5-fold cross-validated.

    Forget examples are the positive class ("in"), test examples the negative class ("out").

    Args:
        forget_losses: Per-example loss of the forget examples, shape `[N_f]`.
        test_losses: Per-example loss of same-class test examples, shape `[N_t]`.
        seed: Seed of the class balancing and of the fold shuffling.

    Returns:
        The cross-validated AUC and accuracy, plus the size of each balanced group.

    Raises:
        MembershipInferenceError: If either group has fewer than `MIA_MIN_EXAMPLES_PER_CLASS`
            examples or contains a non-finite loss.
    """
    forget = forget_losses.double().numpy()
    test = test_losses.double().numpy()
    if min(len(forget), len(test)) < MIA_MIN_EXAMPLES_PER_CLASS:
        raise MembershipInferenceError(
            f"need >= {MIA_MIN_EXAMPLES_PER_CLASS} examples per group, "
            f"got forget={len(forget)}, test={len(test)}"
        )
    if not (np.isfinite(forget).all() and np.isfinite(test).all()):
        raise MembershipInferenceError("non-finite loss: the model diverged")
    forget, test = balanced_subsample(forget, test, seed)
    features = np.clip(np.concatenate([forget, test]), -MIA_LOSS_CLIP, MIA_LOSS_CLIP)[:, None]
    membership = np.concatenate([np.ones(len(forget)), np.zeros(len(test))])
    folds = StratifiedKFold(n_splits=MIA_CV_SPLITS, shuffle=True, random_state=seed)
    scores = cross_validate(
        LogisticRegression(), features, membership, cv=folds, scoring=("roc_auc", "accuracy")
    )
    return MiaResult(
        auc=float(np.mean(scores["test_roc_auc"])),
        accuracy=float(np.mean(scores["test_accuracy"])),
        n_per_group=len(forget),
    )


def evaluate_unlearning(
    model: nn.Module,
    retain_loader: BatchIterable,
    forget_loader: BatchIterable,
    test_loader: BatchIterable,
    device: torch.device | str,
    *,
    seed: int = MIA_DEFAULT_SEED,
) -> UnlearnReport:
    """Measure retain / forget / test accuracy and the loss-based MIA of `model` in one call.

    The MIA compares forget examples with test examples *of the same classes as the forget set*
    (SCRUB sec. 11.1): `test_loader` is filtered on the labels present in `forget_loader`, so the
    attacker cannot win simply by recognising a class. `test_acc` still uses the full `test_loader`.

    Args:
        model: Classifier returning logits.
        retain_loader: Training examples that were kept.
        forget_loader: Training examples to be unlearned.
        test_loader: Held-out examples never used for training or rewinding.
        device: Device the model lives on.
        seed: Seed of the MIA balancing and fold shuffling.

    Returns:
        An `UnlearnReport` with the three accuracies and the MIA AUC (and accuracy).

    Raises:
        MembershipInferenceError: If the loaders are empty or the same-class test subset is too
            small for the attack.
    """
    _, _, retain_correct = collect_losses(model, retain_loader, device)
    forget_losses, forget_labels, forget_correct = collect_losses(model, forget_loader, device)
    test_losses, test_labels, test_correct = collect_losses(model, test_loader, device)
    same_class = torch.isin(test_labels, torch.unique(forget_labels))
    attack = membership_inference(forget_losses, test_losses[same_class], seed)
    return UnlearnReport(
        retain_acc=float(retain_correct.float().mean().item()),
        forget_acc=float(forget_correct.float().mean().item()),
        test_acc=float(test_correct.float().mean().item()),
        mia_auc=attack.auc,
        mia_accuracy=attack.accuracy,
    )
