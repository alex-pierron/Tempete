"""Tests of `wmkit.unlearn`: score formulas, SCRUB, baselines, MIA, splits (all on toy data).

The fast tests use a tiny MLP on synthetic Gaussian blobs (seconds, no CIFAR). The CIFAR run is
marked `slow`.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from wmkit.unlearn import (
    BaselineConfig,
    BaselineError,
    CifarArrays,
    MembershipInferenceError,
    ScoreError,
    ScrubConfig,
    ScrubError,
    TensorBatchLoader,
    build_loaders,
    estimate_epsilon,
    evaluate_unlearning,
    finetune_retain,
    forget_quality_from_epsilons,
    make_split,
    membership_inference,
    negative_gradient,
    neggrad_plus,
    neurips2023_score,
    random_labels,
    scrub,
)
from wmkit.unlearn.baselines import random_wrong_labels
from wmkit.unlearn.data import UnlearnDataError, load_or_train_original
from wmkit.unlearn.scrub import kl_teacher_student

TOY_CLASSES = 3
TOY_DIM = 8
TOY_PER_CLASS = 120
TOY_BATCH = 32
TOY_PRETRAIN_STEPS = 150
TOY_SEED = 0
CPU = "cpu"


# --------------------------------------------------------------------------------------------
# Competition score: values computed by hand
# --------------------------------------------------------------------------------------------


def test_forget_quality_hand_values() -> None:
    """F = mean 2 / 2**floor(eps / 0.5): bins of width 0.5, term 2 at eps < 0.5."""
    assert forget_quality_from_epsilons([0.0]) == 2.0
    assert forget_quality_from_epsilons([0.49]) == 2.0
    assert forget_quality_from_epsilons([0.5]) == 1.0
    assert forget_quality_from_epsilons([0.99]) == 1.0
    assert forget_quality_from_epsilons([1.0]) == 0.5
    assert forget_quality_from_epsilons([2.0]) == 2.0 / 2**4
    # (2 + 1 + 0.5 + 0.25) / 4
    assert forget_quality_from_epsilons([0.0, 0.5, 1.0, 1.5]) == pytest.approx(0.9375)


def test_forget_quality_edge_cases() -> None:
    """Negative epsilon is clamped to bin 0, +inf gives a zero term, NaN and empty are errors."""
    assert forget_quality_from_epsilons([-3.0]) == 2.0
    assert forget_quality_from_epsilons([math.inf]) == 0.0
    assert forget_quality_from_epsilons([0.0, math.inf]) == 1.0
    assert forget_quality_from_epsilons([1e6]) == 0.0
    with pytest.raises(ScoreError):
        forget_quality_from_epsilons([])
    with pytest.raises(ScoreError):
        forget_quality_from_epsilons([0.1, math.nan])


def test_neurips2023_score_hand_values() -> None:
    """score = F * RA_u / RA_r * TA_u / TA_r."""
    assert neurips2023_score(1.0, 0.9, 0.9, 0.8, 0.8) == pytest.approx(1.0)
    # 0.5 * (0.45 / 0.9) * (0.4 / 0.8) = 0.5 * 0.5 * 0.5
    assert neurips2023_score(0.5, 0.45, 0.9, 0.4, 0.8) == pytest.approx(0.125)
    # 2 * (0.99 / 0.9) * (0.88 / 0.8) = 2 * 1.1 * 1.1
    assert neurips2023_score(2.0, 0.99, 0.9, 0.88, 0.8) == pytest.approx(2.42)
    assert neurips2023_score(0.0, 0.9, 0.9, 0.8, 0.8) == 0.0


def test_neurips2023_score_rejects_invalid_inputs() -> None:
    """A zero reference accuracy or a negative / NaN input is a typed error."""
    with pytest.raises(ScoreError):
        neurips2023_score(1.0, 0.9, 0.0, 0.8, 0.8)
    with pytest.raises(ScoreError):
        neurips2023_score(1.0, 0.9, 0.9, 0.8, 0.0)
    with pytest.raises(ScoreError):
        neurips2023_score(-0.1, 0.9, 0.9, 0.8, 0.8)
    with pytest.raises(ScoreError):
        neurips2023_score(1.0, math.nan, 0.9, 0.8, 0.8)


def test_end_to_end_score_from_epsilons() -> None:
    """Chaining the two functions: F([0, 1.0]) = (2 + 0.5) / 2 = 1.25, then the ratios."""
    quality = forget_quality_from_epsilons([0.0, 1.0])
    assert quality == pytest.approx(1.25)
    assert neurips2023_score(quality, 0.8, 1.0, 0.5, 1.0) == pytest.approx(1.25 * 0.8 * 0.5)


def test_estimate_epsilon() -> None:
    """eps_hat = max(log((1-d-FPR)/FNR), log((1-d-FNR)/FPR)), clamped at 0."""
    assert estimate_epsilon(0.25, 0.25) == pytest.approx(math.log(3.0))
    assert estimate_epsilon(0.1, 0.4) == pytest.approx(
        max(math.log(0.9 / 0.4), math.log(0.6 / 0.1))
    )
    assert estimate_epsilon(0.5, 0.5) == 0.0
    assert estimate_epsilon(0.6, 0.6) == 0.0  # worse than chance: no evidence, clamped
    assert estimate_epsilon(0.0, 0.3) == math.inf
    assert estimate_epsilon(0.2, 0.2, delta=0.1) == pytest.approx(math.log(0.7 / 0.2))
    with pytest.raises(ScoreError):
        estimate_epsilon(1.5, 0.1)
    with pytest.raises(ScoreError):
        estimate_epsilon(0.1, 0.1, delta=1.0)


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def test_scrub_config_defaults_and_large_preset() -> None:
    """Defaults and the `large()` preset carry the numbers of docs/CONTRACT-wmkit.md 2.3."""
    default = ScrubConfig()
    assert (default.lr, default.msteps, default.epochs) == (5e-4, 2, 3)
    assert (default.alpha, default.gamma, default.kd_T, default.wd) == (0.001, 0.99, 4.0, 5e-4)
    large = ScrubConfig.large()
    assert (large.msteps, large.epochs) == (3, 10)
    assert (large.alpha, large.gamma, large.kd_T, large.wd) == (0.5, 1.0, 2.0, 0.1)


def test_scrub_config_validation() -> None:
    """Pydantic rejects msteps > epochs, non-positive lr and unknown fields."""
    with pytest.raises(ValueError):
        ScrubConfig(msteps=5, epochs=3)
    with pytest.raises(ValueError):
        ScrubConfig(lr=0.0)
    with pytest.raises(ValueError):
        ScrubConfig(unknown_field=1)  # type: ignore[call-arg]


def test_kl_teacher_student_formula() -> None:
    """d = T^2 * KL(softmax(z_t / T) || softmax(z_s / T)), batch mean; zero iff logits agree."""
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])
    student = torch.tensor([[0.5, 0.5, 0.5], [3.0, 1.0, 0.0]])
    temperature = 2.0
    p = torch.softmax(teacher / temperature, dim=1)
    q = torch.softmax(student / temperature, dim=1)
    expected = (p * (p.log() - q.log())).sum(dim=1).mean() * temperature**2
    assert kl_teacher_student(student, teacher, temperature).item() == pytest.approx(
        expected.item(), rel=1e-5
    )
    assert kl_teacher_student(teacher, teacher, temperature).item() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------------------------
# Toy world: blobs + tiny MLP
# --------------------------------------------------------------------------------------------


def _blobs(per_class: int, seed: int) -> tuple[Tensor, Tensor]:
    """Gaussian blobs: class k centred on a one-hot-ish axis, overlapping enough to be imperfect."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.zeros(TOY_CLASSES, TOY_DIM)
    for k in range(TOY_CLASSES):
        centres[k, k] = 2.5
    labels = torch.arange(TOY_CLASSES).repeat_interleave(per_class)
    points = centres[labels] + torch.randn(len(labels), TOY_DIM, generator=generator)
    return points, labels


class _ArrayLoader:
    """Re-iterable list-of-batches loader over float tensors (toy stand-in for DataLoader)."""

    def __init__(self, x: Tensor, y: Tensor, batch: int = TOY_BATCH, seed: int = 0) -> None:
        self.x, self.y, self.batch, self.seed = x, y, batch, seed
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):  # type: ignore[no-untyped-def]
        order = torch.randperm(len(self.y), generator=self.generator)
        for start in range(0, len(order), self.batch):
            index = order[start : start + self.batch]
            yield self.x[index], self.y[index]


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(TOY_DIM, 32), nn.ReLU(), nn.Linear(32, TOY_CLASSES))


class ToyWorld:
    """Pretrained toy model plus retain / forget (class 0) / val / test loaders."""

    def __init__(self) -> None:
        torch.manual_seed(TOY_SEED)
        x, y = _blobs(TOY_PER_CLASS, seed=1)
        test_x, test_y = _blobs(TOY_PER_CLASS, seed=2)
        forget_mask = y == 0
        self.retain = _ArrayLoader(x[~forget_mask], y[~forget_mask], seed=3)
        self.forget = _ArrayLoader(x[forget_mask], y[forget_mask], seed=4)
        val_mask = (test_y == 0) & (torch.arange(len(test_y)) % 2 == 0)
        test_eval = ~val_mask
        self.val = _ArrayLoader(test_x[val_mask], test_y[val_mask], batch=64, seed=5)
        self.test = _ArrayLoader(test_x[test_eval], test_y[test_eval], batch=64, seed=6)
        self.model = _mlp()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        for _ in range(TOY_PRETRAIN_STEPS):  # full-batch steps: well under a second
            optimizer.zero_grad()
            torch.autograd.backward(torch_f.cross_entropy(self.model(x), y))
            optimizer.step()
        self.model.eval()


@pytest.fixture(scope="module")
def toy() -> ToyWorld:
    return ToyWorld()


def _error(model: nn.Module, loader: _ArrayLoader) -> float:
    wrong = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            wrong += int((model(xb).argmax(1) != yb).sum())
            total += len(yb)
    return wrong / total


def _toy_scrub_config() -> ScrubConfig:
    return ScrubConfig(
        lr=5e-2, msteps=2, epochs=4, alpha=0.5, gamma=1.0, kd_T=2.0, wd=0.0, lr_decay_epochs=(100,)
    )


def test_scrub_forgets_class_keeps_utility_and_does_not_mutate_inputs(toy: ToyWorld) -> None:
    """SCRUB raises the forget error well above the original, keeps retain accuracy, leaves inputs alone."""
    teacher = copy.deepcopy(toy.model)
    reference_state = copy.deepcopy(toy.model.state_dict())
    student = copy.deepcopy(toy.model)
    unlearned, history = scrub(student, teacher, toy.retain, toy.forget, _toy_scrub_config(), CPU)

    assert unlearned is not student
    for key, value in reference_state.items():
        assert torch.equal(student.state_dict()[key], value), "the input student was modified"
        assert torch.equal(teacher.state_dict()[key], value), "the teacher was modified"
    assert all(p.requires_grad for p in teacher.parameters()), (
        "caller's teacher was frozen in place"
    )

    original_forget_error = _error(toy.model, toy.forget)
    assert _error(unlearned, toy.forget) > original_forget_error + 0.4
    assert _error(unlearned, toy.retain) < 0.35
    assert len(history) == 4
    assert [row["epoch"] for row in history] == [1.0, 2.0, 3.0, 4.0]
    assert not math.isnan(history[0]["max_kl"]) and not math.isnan(history[1]["max_kl"])
    assert math.isnan(history[2]["max_kl"]), "epochs after msteps have no max-step"
    assert "val_error" not in history[0]


def test_scrub_plus_r_rewinds_to_checkpoint_closest_to_reference(toy: ToyWorld) -> None:
    """SCRUB+R returns the epoch whose forget error is nearest the validation error, and says so."""
    student = copy.deepcopy(toy.model)
    unlearned, history = scrub(
        student, toy.model, toy.retain, toy.forget, _toy_scrub_config(), CPU,
        rewind_val_loader=toy.val,
    )  # fmt: skip
    summary = history[-1]
    epoch_rows = history[:-1]
    assert len(epoch_rows) == 4
    target = epoch_rows[-1]["val_error"]
    assert summary["rewind_target_error"] == pytest.approx(target)
    best_gap = min(abs(row["forget_error"] - target) for row in epoch_rows)
    assert abs(summary["rewind_forget_error"] - target) == pytest.approx(best_gap)
    chosen = epoch_rows[int(summary["rewind_epoch"]) - 1]
    assert chosen["forget_error"] == pytest.approx(summary["rewind_forget_error"])
    # the returned weights really are the chosen checkpoint: their forget error matches it
    assert _error(unlearned, toy.forget) == pytest.approx(chosen["forget_error"])


def test_scrub_empty_loader_and_divergence_raise(toy: ToyWorld) -> None:
    """Empty loaders and a non-finite loss are typed errors, not silent no-ops."""
    empty = _ArrayLoader(torch.zeros(0, TOY_DIM), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ScrubError):
        scrub(toy.model, toy.model, toy.retain, empty, _toy_scrub_config(), CPU)
    exploding = ScrubConfig(lr=1e9, msteps=2, epochs=3, alpha=0.5, gamma=1.0, kd_T=2.0, wd=0.0)
    with pytest.raises(ScrubError):
        scrub(toy.model, toy.model, toy.retain, toy.forget, exploding, CPU)


def test_baselines_run_and_do_not_mutate_input(toy: ToyWorld) -> None:
    """Every baseline returns a new model, leaves the input intact, and reports one row per epoch."""
    before = copy.deepcopy(toy.model.state_dict())
    runs = {
        "finetune_retain": finetune_retain,
        "negative_gradient": negative_gradient,
        "neggrad_plus": neggrad_plus,
        "random_labels": random_labels,
    }
    for name, method in runs.items():
        config = BaselineConfig(lr=1e-2, epochs=2)
        unlearned, history = method(toy.model, toy.retain, toy.forget, config, CPU)  # type: ignore[operator]
        assert unlearned is not toy.model, name
        assert len(history) == 2, name
        for key, value in before.items():
            assert torch.equal(toy.model.state_dict()[key], value), f"{name} mutated the input"
        changed = any(not torch.equal(before[k], v) for k, v in unlearned.state_dict().items())
        assert changed, f"{name} did not change the weights"


def test_negative_gradient_raises_forget_error(toy: ToyWorld) -> None:
    """Ascending the forget cross-entropy makes the forget class less accurate than the original."""
    unlearned, _ = negative_gradient(
        toy.model, toy.retain, toy.forget, BaselineConfig(lr=1e-2, epochs=5), CPU
    )
    assert _error(unlearned, toy.forget) > _error(toy.model, toy.forget) + 0.3


def test_random_labels_never_returns_the_true_label() -> None:
    """The random relabelling is uniform over the *other* classes."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(TOY_CLASSES).repeat(500)
    wrong = random_wrong_labels(labels, TOY_CLASSES, generator)
    assert bool((wrong != labels).all())
    assert int(wrong.min()) >= 0 and int(wrong.max()) < TOY_CLASSES
    counts = torch.bincount(wrong[labels == 0], minlength=TOY_CLASSES)
    assert counts[0] == 0 and counts[1] > 150 and counts[2] > 150
    with pytest.raises(BaselineError):
        random_wrong_labels(labels, 1, generator)


# --------------------------------------------------------------------------------------------
# MIA and report
# --------------------------------------------------------------------------------------------


def test_mia_separable_losses_give_high_auc() -> None:
    """Forget losses far below test losses: the attacker wins (AUC near 1)."""
    generator = torch.Generator().manual_seed(0)
    forget = torch.rand(200, generator=generator) * 0.1
    test = 1.0 + torch.rand(200, generator=generator)
    result = membership_inference(forget, test, seed=0)
    assert result.auc > 0.99 and result.accuracy > 0.97
    assert result.n_per_group == 200


def test_mia_identical_distributions_give_chance() -> None:
    """Same loss distribution: AUC and accuracy stay near the 0.5 ideal."""
    generator = torch.Generator().manual_seed(1)
    forget = torch.rand(400, generator=generator)
    test = torch.rand(400, generator=generator)
    result = membership_inference(forget, test, seed=0)
    assert 0.42 < result.auc < 0.58
    assert 0.42 < result.accuracy < 0.58


def test_mia_balances_groups_and_validates() -> None:
    """The larger group is subsampled to the smaller one; tiny or non-finite inputs raise."""
    generator = torch.Generator().manual_seed(2)
    result = membership_inference(torch.rand(30, generator=generator), torch.rand(500), seed=0)
    assert result.n_per_group == 30
    with pytest.raises(MembershipInferenceError):
        membership_inference(torch.rand(3), torch.rand(100), seed=0)
    with pytest.raises(MembershipInferenceError):
        membership_inference(torch.tensor([math.nan] * 10), torch.rand(10), seed=0)


def test_evaluate_unlearning_report_matches_manual_numbers(toy: ToyWorld) -> None:
    """Accuracies equal 1 - error computed by hand; the MIA uses same-class test examples only."""
    report = evaluate_unlearning(toy.model, toy.retain, toy.forget, toy.test, CPU)
    assert report.retain_acc == pytest.approx(1.0 - _error(toy.model, toy.retain))
    assert report.forget_acc == pytest.approx(1.0 - _error(toy.model, toy.forget))
    assert report.test_acc == pytest.approx(1.0 - _error(toy.model, toy.test))
    assert 0.0 <= report.mia_auc <= 1.0
    assert report.forget_error == pytest.approx(1.0 - report.forget_acc)
    assert not toy.model.training, "the model eval/train mode must be restored"


def test_scrub_moves_mia_toward_chance_on_toy(toy: ToyWorld) -> None:
    """After SCRUB the forget examples are as wrong as unseen class-0 examples: AUC drops to chance.

    The toy original has a small train/test gap on class 0, so the original AUC is only mildly
    above 0.5; the assertion is on the unlearned model being close to chance, not on the gap.
    """
    unlearned, _ = scrub(
        copy.deepcopy(toy.model), toy.model, toy.retain, toy.forget, _toy_scrub_config(), CPU
    )
    report = evaluate_unlearning(unlearned, toy.retain, toy.forget, toy.test, CPU)
    assert report.mia_auc < 0.7


# --------------------------------------------------------------------------------------------
# Data splits and loader
# --------------------------------------------------------------------------------------------

SYNTHETIC_PER_CLASS_TRAIN = 40
SYNTHETIC_PER_CLASS_TEST = 20


def _synthetic_cifar() -> CifarArrays:
    generator = torch.Generator().manual_seed(0)
    train_y = torch.arange(10).repeat_interleave(SYNTHETIC_PER_CLASS_TRAIN)
    test_y = torch.arange(10).repeat_interleave(SYNTHETIC_PER_CLASS_TEST)
    shape = (3, 32, 32)
    return CifarArrays(
        train_x=torch.randint(
            0, 256, (len(train_y), *shape), generator=generator, dtype=torch.uint8
        ),
        train_y=train_y,
        test_x=torch.randint(0, 256, (len(test_y), *shape), generator=generator, dtype=torch.uint8),
        test_y=test_y,
    )


def test_class_split_is_a_partition_and_validation_is_disjoint_from_test() -> None:
    """Class forgetting: forget = whole class, retain = rest; val and eval test never overlap."""
    arrays = _synthetic_cifar()
    split = make_split(arrays, scenario="class", forget_class=5)
    assert len(split.forget_idx) == SYNTHETIC_PER_CLASS_TRAIN
    assert bool((arrays.train_y[split.forget_idx] == 5).all())
    assert len(split.retain_idx) == len(arrays.train_y) - SYNTHETIC_PER_CLASS_TRAIN
    assert not set(split.forget_idx.tolist()) & set(split.retain_idx.tolist())
    assert not set(split.val_idx.tolist()) & set(split.test_idx.tolist())
    assert len(split.val_idx) + len(split.test_idx) == len(arrays.test_y)
    assert bool((arrays.test_y[split.val_idx] == 5).all())
    assert len(split.val_idx) == SYNTHETIC_PER_CLASS_TEST // 2


def test_selective_split_is_deterministic_and_validated() -> None:
    """Selective forgetting draws n examples of the class, same seed same set, bad input raises."""
    arrays = _synthetic_cifar()
    first = make_split(arrays, scenario="selective", forget_class=5, n_forget=10, seed=3)
    again = make_split(arrays, scenario="selective", forget_class=5, n_forget=10, seed=3)
    other = make_split(arrays, scenario="selective", forget_class=5, n_forget=10, seed=4)
    assert len(first.forget_idx) == 10
    assert torch.equal(first.forget_idx, again.forget_idx)
    assert not torch.equal(first.forget_idx, other.forget_idx)
    assert len(first.retain_idx) == len(arrays.train_y) - 10
    with pytest.raises(UnlearnDataError):
        make_split(arrays, scenario="selective", forget_class=5, n_forget=0)
    with pytest.raises(UnlearnDataError):
        make_split(arrays, scenario="selective", forget_class=5, n_forget=SYNTHETIC_PER_CLASS_TRAIN)
    with pytest.raises(UnlearnDataError):
        make_split(arrays, scenario="nope")
    with pytest.raises(UnlearnDataError):
        make_split(arrays, scenario="class", forget_class=10)


def test_tensor_batch_loader_shapes_normalisation_and_reproducibility() -> None:
    """Batches are normalised float32, the last one is short, and equal seeds give equal orders."""
    arrays = _synthetic_cifar()
    loader = TensorBatchLoader(arrays.train_x[:50], arrays.train_y[:50], 16, shuffle=True, seed=1)
    batches = list(loader)
    assert len(loader) == len(batches) == 4
    assert batches[0][0].shape == (16, 3, 32, 32) and batches[0][0].dtype == torch.float32
    assert batches[-1][0].shape[0] == 50 - 3 * 16
    assert sum(len(b[1]) for b in batches) == 50
    twin = TensorBatchLoader(arrays.train_x[:50], arrays.train_y[:50], 16, shuffle=True, seed=1)
    assert torch.equal(next(iter(twin))[1], batches[0][1])
    second_pass = next(iter(loader))[1]
    assert not torch.equal(second_pass, batches[0][1]), "each epoch must reshuffle"
    value = float(batches[0][0].mean())
    assert abs(value) < 0.2, "uniform pixels in [0, 255] normalise to roughly zero mean"


def test_build_loaders_use_the_papers_batch_sizes() -> None:
    """Class scenario: forget 512 / retain 128; selective: 16 / 64 (SCRUB paper, sec. 9)."""
    arrays = _synthetic_cifar()
    loaders = build_loaders(arrays, make_split(arrays, scenario="class", forget_class=5))
    assert len(next(iter(loaders.retain))[1]) == 128
    assert len(next(iter(loaders.forget))[1]) == SYNTHETIC_PER_CLASS_TRAIN  # < 512: one short batch
    selective = build_loaders(
        arrays, make_split(arrays, scenario="selective", forget_class=5, n_forget=20)
    )
    assert len(next(iter(selective.retain))[1]) == 64
    assert len(next(iter(selective.forget))[1]) == 16


def test_original_model_is_trained_once_then_loaded_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call loads the cached weights (same predictions, `cached=True`, no retraining)."""
    monkeypatch.setenv("WMKIT_WEIGHTS_DIR", str(tmp_path))
    # A tiny CNN instead of ResNet-18 keeps this test at a fraction of a second.
    monkeypatch.setattr(
        "wmkit.unlearn.data.resnet18_cifar",
        lambda num_classes=10: nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, num_classes),
        ),
    )
    arrays = _synthetic_cifar()
    first_model, first = load_or_train_original(arrays, CPU, epochs=1)
    second_model, second = load_or_train_original(arrays, CPU, epochs=1)
    assert not first.cached and second.cached
    assert second.seconds == pytest.approx(first.seconds)
    probe = arrays.test_x[:8].float() / 255.0
    first_model.eval()
    second_model.eval()
    with torch.no_grad():
        assert torch.allclose(first_model(probe), second_model(probe), atol=1e-5)
    assert len(list((tmp_path / "unlearn").glob("*.pt"))) == 1
    assert not list((tmp_path / "unlearn").glob("*.tmp*")), "no temporary file may remain"


# --------------------------------------------------------------------------------------------
# CIFAR-10 end to end (slow: downloads data, trains a ResNet-18 on a subset)
# --------------------------------------------------------------------------------------------

# A small subset (100/class) with enough epochs lets the ResNet fully over-fit (train_acc 1.0
# in ~55 s), which is the premise of this test; 500/class x 8 epochs plateaued near 0.70.
SLOW_SUBSET_PER_CLASS = 100
SLOW_ORIGINAL_EPOCHS = 30
# Whole-class forgetting needs the stronger "large" preset: the default (small) config only
# drops forget accuracy ~0.12 and leaves the MIA high, whereas large() drives forget accuracy
# to 0 and the MIA AUC to ~0.48 (chance). This is the config to reach for on a class-level
# unlearning task; the small default suits selective (few-sample) forgetting.
SLOW_SCRUB_CONFIG = ScrubConfig.large()
# Forgetting the whole class removes its test images from the overall test score, so a drop of
# ~0.08-0.10 in test accuracy is expected and correct here (retain accuracy stays ~1.0).
SLOW_TEST_ACC_TOLERANCE = 0.12


@pytest.mark.slow
def test_cifar_scrub_class_forgetting_on_a_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ResNet-18 on 5k CIFAR images: SCRUB forgets the class and keeps the test accuracy."""
    from wmkit.config import get_device
    from wmkit.unlearn import load_cifar10, seed_everything

    monkeypatch.setenv("WMKIT_WEIGHTS_DIR", str(tmp_path))
    device = get_device()
    arrays = load_cifar10(subset_per_class=SLOW_SUBSET_PER_CLASS)
    original, summary = load_or_train_original(arrays, device, epochs=SLOW_ORIGINAL_EPOCHS)
    assert summary.train_acc > 0.9, "the original must over-fit its training set"

    split = make_split(arrays, scenario="class", forget_class=5)
    loaders = build_loaders(arrays, split, device=device)
    seed_everything(0)
    before = evaluate_unlearning(original, loaders.retain, loaders.forget, loaders.test, device)
    unlearned, _ = scrub(
        copy.deepcopy(original), original, loaders.retain, loaders.forget, SLOW_SCRUB_CONFIG,
        device, rewind_val_loader=loaders.val,
    )  # fmt: skip
    after = evaluate_unlearning(unlearned, loaders.retain, loaders.forget, loaders.test, device)

    assert before.forget_acc > 0.8
    assert after.forget_acc < before.forget_acc - 0.3
    # Utility: overall test accuracy must not collapse (it dips by the forgotten class alone).
    assert after.test_acc > before.test_acc - SLOW_TEST_ACC_TOLERANCE
    assert after.mia_auc <= before.mia_auc + 0.02
