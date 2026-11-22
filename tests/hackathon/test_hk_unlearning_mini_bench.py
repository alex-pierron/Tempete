"""Tests of the miniature benchmark: exact score formulas, alterations, tiny end-to-end run, viz."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hackathon.unlearning.research import mini_bench as bench
from hackathon.unlearning.research import mini_core as core
from hackathon.unlearning.research import mini_methods as methods
from hackathon.unlearning.research import mini_score as score
from hackathon.unlearning.research import viz

CLASSES = 4
COUNT = 24
SEED = 3


def _logits(seed: int, count: int = COUNT, scale: float = 3.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, CLASSES)) * scale


def _sets(logits: np.ndarray) -> dict[str, np.ndarray]:
    return {name: logits for name in bench.score_set_names()}


def _context(
    orig: np.ndarray, golds: list[np.ndarray], labels: np.ndarray, threshold: float = 0.0
) -> score.ScoringContext:
    return score.ScoringContext(
        orig=_sets(orig),
        golds=[_sets(g) for g in golds],
        forget_labels=labels,
        control_labels=labels,
        validation_labels=labels,
        utility_threshold=threshold,
    )


def test_cross_entropy_and_kl_match_hand_values() -> None:
    logits = np.array([[2.0, 0.0]])
    assert score.cross_entropy(logits, np.array([0]))[0] == pytest.approx(math.log1p(math.exp(-2)))
    reference = score.softmax(np.array([[2.0, 0.0]]), 2.0)
    other = np.array([[0.0, 0.0]])
    p = 1 / (1 + math.exp(-1.0))
    expected = p * math.log(p / 0.5) + (1 - p) * math.log((1 - p) / 0.5)
    assert score.per_sample_kl(reference, other)[0] == pytest.approx(expected)
    assert score.per_sample_kl(reference, np.array([[2.0, 0.0]]))[0] == pytest.approx(
        0.0, abs=1e-12
    )


def test_submission_equal_to_original_scores_zero() -> None:
    orig, gold = _logits(1), _logits(2)
    labels = gold.argmax(axis=1)
    result = score.official_score(_sets(orig), _context(orig, [gold], labels))
    assert result.forget_score == pytest.approx(0.0)
    assert result.final_score == pytest.approx(0.0)
    assert result.retain_score == pytest.approx(1.0)


def test_submission_equal_to_gold_scores_one_on_forget() -> None:
    orig, gold = _logits(1), _logits(2)
    labels = (gold.argmax(axis=1) + 1) % CLASSES  # the gold is wrong on purpose: spread in losses
    result = score.official_score(_sets(gold), _context(orig, [gold], labels))
    assert result.mean_std_score == pytest.approx(1.0)
    assert result.prediction_score == pytest.approx(1.0)
    assert result.forget_score == pytest.approx(1.0)


def test_utility_gate_zeroes_everything() -> None:
    orig, gold = _logits(1), _logits(2)
    labels = gold.argmax(axis=1)
    context = _context(orig, [gold], labels, threshold=1.01)
    result = score.official_score(_sets(gold), context)
    assert not result.gate_passed
    assert result.final_score == 0.0
    assert result.forget_score == 0.0
    assert result.retain_score == 0.0


def test_worse_than_original_is_clipped_to_zero_not_negative() -> None:
    orig, gold = _logits(1), _logits(2)
    labels = gold.argmax(axis=1)
    worse = orig * 4.0 + 5.0
    result = score.official_score(_sets(worse), _context(orig, [gold], labels))
    assert result.mean_std_score >= 0.0
    assert result.prediction_score >= 0.0
    assert 0.0 <= result.final_score <= 1.0


def test_gold_reference_averages_models_and_rejects_empty() -> None:
    labels = np.zeros(COUNT, dtype=np.int64)
    reference = score.build_gold_reference([_logits(1), _logits(2)], labels)
    expected = np.mean([score.cross_entropy(_logits(s), labels).mean() for s in (1, 2)])
    assert reference.loss_mean == pytest.approx(expected)
    assert reference.probabilities.shape == (COUNT, CLASSES)
    assert np.allclose(reference.probabilities.sum(axis=1), 1.0)
    with pytest.raises(score.ScoreError):
        score.build_gold_reference([], labels)


def test_rank_helpers_handle_ties_and_constants() -> None:
    values = np.array([3.0, 1.0, 3.0, 2.0])
    assert list(score.average_ranks(values)) == [3.5, 1.0, 3.5, 2.0]
    assert score.spearman(np.arange(6.0), np.arange(6.0) ** 2) == pytest.approx(1.0)
    assert math.isnan(score.spearman(np.ones(6), np.arange(6.0)))
    assert score.top_k_overlap(np.arange(6.0), np.arange(6.0), 3) == 1.0
    low, high = score.bootstrap_spearman(np.arange(10.0), np.arange(10.0) + 0.1)
    assert low <= 1.0 and high <= 1.0001


def test_alterations_keep_shape_range_and_are_deterministic() -> None:
    images = torch.rand(
        12, 3, core.IMAGE_SIZE, core.IMAGE_SIZE, generator=torch.Generator().manual_seed(0)
    )
    for family in (core.KNOWN_FAMILY, core.UNKNOWN_FAMILY):
        first = core.apply_family(images, family, SEED)
        second = core.apply_family(images, family, SEED)
        assert first.shape == images.shape
        assert torch.equal(first, second)
        assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
        assert not torch.equal(first, images)
    view = core.strong_view(images, torch.Generator().manual_seed(1), 0.95)
    assert view.shape == images.shape


def test_method_config_rejects_unknown_method_and_bad_area() -> None:
    with pytest.raises(methods.MiniUnlearnError):
        methods.MiniUnlearnConfig("nope", "x")
    with pytest.raises(methods.MiniUnlearnError):
        methods.MiniUnlearnConfig(methods.METHOD_GOLD_SOFT, "x", view_min_area=0.0)


def test_default_candidates_have_unique_labels_and_cover_the_kit_methods() -> None:
    candidates = bench.default_candidates()
    methods_present = {c.method for c in candidates}
    assert {
        methods.METHOD_IDENTITY,
        methods.METHOD_SCRUB,
        methods.METHOD_GOLD_SOFT,
    } <= methods_present
    assert len({c.label for c in candidates}) == len(candidates)


def _synthetic_cifar(count_train: int = 700, count_test: int = 200) -> core.Cifar10:
    """Learnable toy data: each class has its own mean colour plus noise (32 px, 10 classes)."""
    generator = torch.Generator().manual_seed(0)
    means = torch.rand(core.NUM_CLASSES, 3, 1, 1, generator=generator)

    def make(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(core.NUM_CLASSES, (count,), generator=generator)
        noise = 0.25 * torch.randn(count, 3, 32, 32, generator=generator)
        return (means[labels] + noise).clamp(0, 1), labels

    train_x, train_y = make(count_train)
    test_x, test_y = make(count_test)
    return core.Cifar10(train_x, train_y, test_x, test_y)


def test_make_world_is_disjoint_and_deterministic() -> None:
    data = _synthetic_cifar()
    spec = core.WorldSpec(seed=1, retain_size=300, forget_size=50, control_size=60,
                          validation_size=60, member_control_size=40)  # fmt: skip
    first, second = core.make_world(data, spec), core.make_world(data, spec)
    assert torch.equal(first.forget_x, second.forget_x)
    assert len(first.retain_x) == 300 and len(first.forget_x) == 50
    with pytest.raises(core.MiniBenchError):
        core.make_world(data, core.WorldSpec(seed=0, retain_size=10_000))


def test_end_to_end_tiny_world_runs_scores_and_analyses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(score, "BOOTSTRAP_RESAMPLES", 20)  # the bootstrap is not under test here
    data = _synthetic_cifar()
    world = core.make_world(
        data,
        core.WorldSpec(seed=0, retain_size=300, forget_size=60, control_size=80,
                       validation_size=80, member_control_size=60),
    )  # fmt: skip
    ood = torch.rand(40, 3, 32, 32, generator=torch.Generator().manual_seed(5))
    cfg = methods.MiniUnlearnConfig
    candidates = (
        cfg(methods.METHOD_IDENTITY, "identity"),
        cfg(methods.METHOD_GOLD_SOFT, "gs_a", steps=4, batch_size=16),
        cfg(methods.METHOD_GOLD_SOFT, "gs_b", steps=4, batch_size=16, target_temperature=3.0),
        cfg(
            methods.METHOD_GOLD_SOFT,
            "gs_fixed",
            steps=3,
            batch_size=16,
            target_views=1,
            fixed_targets=True,
        ),
        cfg(methods.METHOD_SCRUB, "scrub", steps=2, batch_size=16),
        cfg(methods.METHOD_GRAD_ASCENT, "ga", steps=2, batch_size=16),
        cfg(methods.METHOD_UNIFORM, "uni", steps=2, batch_size=16),
        cfg(methods.METHOD_FINETUNE_RETAIN, "ft", steps=2, batch_size=16),
        cfg(methods.METHOD_WEIGHT_NOISE, "noise", noise_scale=0.05),
    )
    spec = bench.BenchSpec(
        world_seeds=(0,),
        train=core.TrainSpec(epochs=2, batch_size=64, width=8),
        cache_dir=tmp_path / "cache",
        candidates=candidates,
    )
    result = bench.run_world(world, ood, spec, torch.device("cpu"))
    labels = [str(c["label"]) for c in result.candidates]
    assert labels == [c.label for c in candidates] + [bench.HELD_OUT_GOLD_LABEL]
    identity = next(c for c in result.candidates if c["label"] == "identity")
    scores = identity["scores"]
    assert isinstance(scores, dict)
    assert scores["main"]["final_score"] == pytest.approx(0.0, abs=1e-9)
    for candidate in result.candidates:
        scores = candidate["scores"]
        assert isinstance(scores, dict)
        assert set(scores) == set(bench.SCORE_VARIANTS)
        assert 0.0 <= float(scores["main"]["final_score"]) <= 1.0
    analysis = bench.analyse([result])
    assert set(analysis["pooled"]) == set(bench.PROXY_ORIENTATION)  # type: ignore[arg-type]
    # the trained models are cached: a second run reuses them and gives identical scores
    again = bench.run_world(world, ood, spec, torch.device("cpu"))
    assert [c["scores"]["main"]["final_score"] for c in again.candidates][:1] == [0.0]  # type: ignore[index]


def test_viz_functions_return_savable_figures(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    series = {"a": rng.random(50), "b": rng.random(50) * 2}
    figures = [
        viz.plot_ecdf(series, "x"),
        viz.plot_ecdf(series, "x", log_x=True),
        viz.plot_histograms(series, "x"),
        viz.plot_boxes(series, "y"),
        viz.plot_ranked_bars(["a", "b", "c"], np.array([1.0, 3.0, 2.0]), "v", highlight=["b"]),
        viz.plot_scatter(
            series["a"],
            series["b"],
            "x",
            "y",
            labels=[str(i) for i in range(50)],
            annotate_top=3,
            diagonal=True,
        ),
        viz.plot_heatmap(rng.random((3, 4)), list("abc"), list("wxyz"), vmin=0, vmax=1),
        viz.plot_transition_matrix(np.diag(np.arange(1.0, 6.0))),
        viz.plot_lines(np.arange(5.0), {"s": rng.random(5)}, "x", "y"),
        viz.plot_grouped_bars(
            ["u", "v"], {"p": np.array([1.0, 2.0]), "q": np.array([2.0, 1.0])}, "y"
        ),
    ]
    for index, figure in enumerate(figures):
        path = viz.save_figure(figure, tmp_path / f"f{index}.png")
        assert path.stat().st_size > 1000


def test_viz_rejects_empty_and_too_many_series() -> None:
    with pytest.raises(viz.VizError):
        viz.plot_ecdf({}, "x")
    with pytest.raises(viz.VizError):
        viz.plot_ecdf({str(i): np.ones(3) for i in range(9)}, "x")


def test_image_grid_figure_and_validation(tmp_path: Path) -> None:
    images = [np.full((8, 8, 3), 40 * i, dtype=np.uint8) for i in range(5)]
    figure = viz.plot_image_grid(images, [f"i{i}" for i in range(5)], 3)
    assert viz.save_figure(figure, tmp_path / "grid.png").stat().st_size > 500
    with pytest.raises(viz.VizError):
        viz.plot_image_grid(images, ["only one"], 3)
