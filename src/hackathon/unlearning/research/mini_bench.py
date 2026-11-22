"""Miniature benchmark: the REAL official unlearning score, at a scale a laptop can train.

Question answered: "do the local proxies of the kit predict the real score?". The real gold models
are not available, so the protocol is reproduced on CIFAR-10 with small CNNs:

1. a *world* = a draw of (retain, forget, control, validation) splits and of every model's seed;
2. `orig` is trained on retain + forget until it memorises the forget images (confidence ~ 1);
3. `golds` (several seeds) are trained on retain only: the reference behaviour;
4. every candidate method unlearns from `orig`, seeing only orig, the forget images and an
   out-of-distribution proxy (the kit's situation), then is scored with the exact formulas of the
   subject (`mini_score.official_score`) AND with the kit's proxy readings, on the same logits;
5. rank correlations proxy <-> true score are computed per world and pooled.

Run (heavy: trains ~10 CNNs per world):

    HACKATHON_DATA_DIR=<data> PYTHONPATH=src .venv/bin/python \\
        -m hackathon.unlearning.research.mini_bench --worlds 0 1 --out <metrics_mini.json>

Everything trained is cached (state dicts of ~0.3 MB, atomic writes), so a rerun resumes where it
stopped and never retrains an identical model.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from hackathon.common.errors import HackathonError
from hackathon.unlearning.research import mini_core as core
from hackathon.unlearning.research import mini_methods as methods
from hackathon.unlearning.research import mini_score as score

LOGGER = logging.getLogger("hackathon.unlearning.mini_bench")

FloatArray = NDArray[np.float64]

GOLD_REFERENCE_COUNT = 3
"""Gold models averaged as the reference (the subject averages 'several')."""
ORIG_SEED_OFFSET = 1000
GOLD_SEED_OFFSET = 2000
HELD_OUT_GOLD_SEED_OFFSET = 3000
ALTERED_SEED_OFFSET = 100
UNKNOWN_ALTERED_SEED_OFFSET = 200
PROXY_ALTERATION_SEED_OFFSET = 300
VIEW_SEED_OFFSET = 400
UTILITY_FRACTION_OF_ORIG = 0.85
"""Mini utility gate = this fraction of the original's validation accuracy. The real gate is an
absolute 70 % on a model that is probably in the high 80s: 0.85 x ~0.85 is the same ratio."""
OOD_PROXY_COUNT = 624
TORCH_THREADS = 2
"""CPU threads of torch: the machine is shared by 7 agents, each process taking every core slows all."""
"""All the kit's local proxy images (500 COCO + 100 DIV2K + 24 Kodak)."""
SURROGATE_AREAS = (0.2, 0.35, 0.5, 0.7, 0.9)
SURROGATE_TEMPERATURES = (1.0, 2.0, 3.0, 5.0)
SURROGATE_VIEW_COUNT = 3
DEFAULT_SURROGATE_AREA = 0.5
DEFAULT_SURROGATE_TEMPERATURE = 2.0
"""A-priori surrogate used as a proxy (NOT tuned on the gold: that would need the gold)."""
HELD_OUT_GOLD_LABEL = "gold_heldout"
IDENTITY_LABEL = "identity"
KIT_PROXY_ALTERATIONS = ("flip", "crop", "jitter", "jpeg")
VARIANT_MAIN = "main"
VARIANT_UNKNOWN_ALTERED = "unknown_altered"
VARIANT_MEMBER_CONTROL = "member_control"
SCORE_VARIANTS = (VARIANT_MAIN, VARIANT_UNKNOWN_ALTERED, VARIANT_MEMBER_CONTROL)
EXCLUDED_FROM_RANKING = (IDENTITY_LABEL, HELD_OUT_GOLD_LABEL)
"""Candidates that are references, not methods: kept in the tables, dropped from correlations."""


class BenchError(HackathonError):
    """The benchmark cannot run (missing data) or its inputs are inconsistent."""


@dataclass(frozen=True)
class BenchSpec:
    """Everything that defines a benchmark run."""

    world_seeds: tuple[int, ...] = (0, 1)
    train: core.TrainSpec = field(default_factory=core.TrainSpec)
    cifar_root: Path = Path("data/cifar-10-batches-py")
    proxy_root: Path = Path("data")
    cache_dir: Path | None = None
    candidates: tuple[methods.MiniUnlearnConfig, ...] = ()
    tag: str = ""
    retain_size: int = core.WorldSpec.retain_size
    forget_size: int = core.WorldSpec.forget_size


def default_candidates() -> tuple[methods.MiniUnlearnConfig, ...]:
    """27 candidates: the kit's three methods and families of variants, spread in strength."""
    cfg = methods.MiniUnlearnConfig
    gs, sc = methods.METHOD_GOLD_SOFT, methods.METHOD_SCRUB
    out: list[methods.MiniUnlearnConfig] = [cfg(methods.METHOD_IDENTITY, IDENTITY_LABEL)]
    for steps in (30, 90, 270):
        out.append(cfg(gs, f"gold_soft_default_s{steps}", steps=steps))
    for temperature in (1.0, 1.5, 3.0, 5.0):
        out.append(cfg(gs, f"gold_soft_T{temperature:g}", steps=90, target_temperature=temperature))
    for area in (0.2, 0.6, 0.9):
        out.append(cfg(gs, f"gold_soft_area{area:g}", steps=90, view_min_area=area))
    out.append(cfg(gs, "gold_soft_noviews_T3", steps=90, target_views=0, target_temperature=3.0))
    out.append(cfg(gs, "gold_soft_fixed1", steps=90, target_views=1, fixed_targets=True))
    out.append(cfg(gs, "gold_soft_fixed3", steps=90, target_views=3, fixed_targets=True))
    out.append(cfg(gs, "gold_soft_lr3x", steps=90, lr=1e-3))
    out.append(cfg(gs, "gold_soft_noanchor", steps=90, retain_weight=0.0))
    for cap in (0.5, 2.0, 8.0):
        out.append(cfg(sc, f"scrub_cap{cap:g}", steps=30, scrub_kl_cap=cap))
    out.append(cfg(sc, "scrub_lr3x", steps=30, lr=1e-3))
    for steps in (5, 15):
        out.append(cfg(methods.METHOD_GRAD_ASCENT, f"grad_ascent_s{steps}", steps=steps))
    for steps in (30, 90):
        out.append(cfg(methods.METHOD_UNIFORM, f"uniform_s{steps}", steps=steps))
    out.append(cfg(methods.METHOD_FINETUNE_RETAIN, "finetune_retain_s90", steps=90))
    for scale in (0.02, 0.05, 0.1):
        out.append(cfg(methods.METHOD_WEIGHT_NOISE, f"weight_noise_{scale:g}", noise_scale=scale))
    labels = [c.label for c in out]
    if len(set(labels)) != len(labels):
        raise BenchError("duplicate candidate labels")
    return tuple(out)


def lowlr_candidates() -> tuple[methods.MiniUnlearnConfig, ...]:
    """Second pass: the same families at 3x to 30x smaller learning rates (gentler movement).

    The first pass (lr 3e-4 to 1e-3) over-shot: every method lost ~30 % of its retain score. This
    pass explores the regime the kit's conservative defaults aim at, with the identity as anchor.
    """
    cfg = methods.MiniUnlearnConfig
    gs, sc = methods.METHOD_GOLD_SOFT, methods.METHOD_SCRUB
    out: list[methods.MiniUnlearnConfig] = [cfg(methods.METHOD_IDENTITY, IDENTITY_LABEL)]
    for lr in (3e-5, 1e-4):
        for steps in (90, 270):
            out.append(cfg(gs, f"gs_lr{lr:g}_a0.6_s{steps}", steps=steps, lr=lr, view_min_area=0.6))
    out.append(
        cfg(gs, "gs_lr1e-4_a0.6_T1.5", steps=90, lr=1e-4, view_min_area=0.6, target_temperature=1.5)
    )
    out.append(cfg(gs, "gs_lr1e-4_a0.9", steps=90, lr=1e-4, view_min_area=0.9))
    out.append(
        cfg(
            gs,
            "gs_lr1e-4_fixed1_a0.6",
            steps=90,
            lr=1e-4,
            view_min_area=0.6,
            target_views=1,
            fixed_targets=True,
        )
    )
    out.append(
        cfg(
            gs,
            "gs_lr1e-4_fixed3_a0.6",
            steps=90,
            lr=1e-4,
            view_min_area=0.6,
            target_views=3,
            fixed_targets=True,
        )
    )
    out.append(
        cfg(gs, "gs_lr1e-4_noanchor", steps=90, lr=1e-4, view_min_area=0.6, retain_weight=0.0)
    )
    for cap in (0.5, 2.0):
        out.append(cfg(sc, f"scrub_lr1e-4_cap{cap:g}", steps=30, lr=1e-4, scrub_kl_cap=cap))
    out.append(cfg(sc, "scrub_lr3e-5_cap2", steps=30, lr=3e-5, scrub_kl_cap=2.0))
    for steps in (30, 90):
        out.append(cfg(methods.METHOD_UNIFORM, f"uniform_lr1e-4_s{steps}", steps=steps, lr=1e-4))
    for steps in (5, 15):
        out.append(
            cfg(methods.METHOD_GRAD_ASCENT, f"grad_ascent_lr3e-5_s{steps}", steps=steps, lr=3e-5)
        )
    out.append(cfg(methods.METHOD_GRAD_ASCENT, "grad_ascent_lr1e-4_s5", steps=5, lr=1e-4))
    return tuple(out)


# ------------------------------------------------------------------------------------------
# One world
# ------------------------------------------------------------------------------------------


def build_eval_sets(world: core.World, ood: Tensor) -> dict[str, Tensor]:
    """Every image set any score or proxy needs, computed once per world (deterministic)."""
    seed = world.spec.seed
    sets: dict[str, Tensor] = {
        score.EXACT_SET: world.forget_x,
        score.ALTERED_SET: core.apply_family(
            world.forget_x, core.KNOWN_FAMILY, seed + ALTERED_SEED_OFFSET
        ),
        "forget_altered_unknown": core.apply_family(
            world.forget_x, core.UNKNOWN_FAMILY, seed + UNKNOWN_ALTERED_SEED_OFFSET
        ),
        score.CONTROL_SET: world.control_x,
        "control_member": world.member_control_x,
        score.VALIDATION_SET: world.validation_x,
        score.KIT_RETAIN_GROUP: ood,
    }
    for offset, name in enumerate(KIT_PROXY_ALTERATIONS):
        generator = torch.Generator().manual_seed(seed + PROXY_ALTERATION_SEED_OFFSET + offset)
        sets[f"forget_{name}"] = core.ALTERATIONS[name](world.forget_x, generator)
    for area_index, area in enumerate(SURROGATE_AREAS):
        for view in range(SURROGATE_VIEW_COUNT):
            generator = torch.Generator().manual_seed(
                seed + VIEW_SEED_OFFSET + 10 * area_index + view
            )
            sets[view_set_name(area, view)] = core.strong_view(world.forget_x, generator, area)
    return sets


def view_set_name(area: float, view: int) -> str:
    """Name of the strong-view set of crop area `area` (view index `view`)."""
    return f"view_{area:g}_{view}"


def _scoring_contexts(
    world: core.World,
    orig: Mapping[str, FloatArray],
    golds: Sequence[Mapping[str, FloatArray]],
    threshold: float,
) -> dict[str, tuple[score.ScoringContext, dict[str, str]]]:
    """One (context, set-renaming) per score variant (main / unknown alterations / member control)."""
    forget_y = world.forget_y.numpy()
    control_y = world.control_y.numpy()
    validation_y = world.validation_y.numpy()

    def context(
        control_labels: NDArray[np.int64], rename: dict[str, str]
    ) -> tuple[score.ScoringContext, dict[str, str]]:
        def remap(sets: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
            return {name: sets[rename.get(name, name)] for name in score_set_names()}

        return (
            score.ScoringContext(
                orig=remap(orig),
                golds=[remap(g) for g in golds],
                forget_labels=forget_y,
                control_labels=control_labels,
                validation_labels=validation_y,
                utility_threshold=threshold,
            ),
            rename,
        )

    return {
        VARIANT_MAIN: context(control_y, {}),
        VARIANT_UNKNOWN_ALTERED: context(control_y, {score.ALTERED_SET: "forget_altered_unknown"}),
        VARIANT_MEMBER_CONTROL: context(
            world.member_control_y.numpy(), {score.CONTROL_SET: "control_member"}
        ),
    }


def score_set_names() -> tuple[str, ...]:
    """The four set names `official_score` reads."""
    return (score.EXACT_SET, score.ALTERED_SET, score.CONTROL_SET, score.VALIDATION_SET)


def _remap_sub(sub: Mapping[str, FloatArray], rename: Mapping[str, str]) -> dict[str, FloatArray]:
    return {name: sub[rename.get(name, name)] for name in score_set_names()}


def gold_loss_structure(
    golds: Sequence[Mapping[str, FloatArray]], labels: NDArray[np.int64]
) -> dict[str, float]:
    """Mean-of-per-model loss std (what the official score uses) vs the std of the ENSEMBLE loss.

    The subject averages each gold model's own loss mean / std. A method that matches the
    AVERAGED prediction (like `gold_soft`) reproduces the ensemble's loss, whose std is smaller
    (averaging smooths out each model's idiosyncratic errors): the ratio measures that gap.
    """
    exact = [g[score.EXACT_SET] for g in golds]
    per_model = [score.cross_entropy(z, labels) for z in exact]
    mean_prob = np.mean([score.softmax(z) for z in exact], axis=0)
    ensemble_loss = -np.log(np.clip(mean_prob[np.arange(len(labels)), labels], 1e-12, None))
    return {
        "per_model_loss_mean": float(np.mean([loss.mean() for loss in per_model])),
        "per_model_loss_std": float(np.mean([loss.std() for loss in per_model])),
        "ensemble_loss_mean": float(ensemble_loss.mean()),
        "ensemble_loss_std": float(ensemble_loss.std()),
    }


def surrogate_grid(
    orig: Mapping[str, FloatArray], gold_probabilities: FloatArray
) -> list[dict[str, float]]:
    """How well does the tempered original-on-strong-views predict the gold? (area x temperature)

    For every cell: mean KL(p_gold || p_surrogate) divided by mean KL(p_gold || p_orig) (below 1:
    the surrogate is closer to the gold than the original is; 0: perfect), plus the loss-statistics
    distance ratio. This is the premise of `gold_soft`, measured against real gold models.
    """
    exact = orig[score.EXACT_SET]
    kl_orig = float(score.per_sample_kl(gold_probabilities, exact).mean())
    rows: list[dict[str, float]] = []
    for area in SURROGATE_AREAS:
        views = [orig[view_set_name(area, view)] for view in range(SURROGATE_VIEW_COUNT)]
        for temperature in SURROGATE_TEMPERATURES:
            reference = score.surrogate_reference(views, exact.argmax(axis=1), temperature)
            target_logits = np.log(np.clip(reference.probabilities, 1e-12, None))
            kl_sur = float(score.per_sample_kl(gold_probabilities, target_logits).mean())
            rows.append(
                {
                    "area": area,
                    "temperature": temperature,
                    "kl_ratio": kl_sur / kl_orig if kl_orig > 0 else float("nan"),
                    "mean_confidence": float(
                        np.mean([score.softmax(z, temperature).max(axis=1).mean() for z in views])
                    ),
                }
            )
    return rows


@dataclass
class WorldResult:
    """All numbers of one world (JSON friendly through `to_dict`)."""

    seed: int
    diagnostics: dict[str, object]
    candidates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Plain dict for JSON."""
        return asdict(self)


def _model_stats(logits: Mapping[str, FloatArray], world: core.World) -> dict[str, float]:
    forget_y = world.forget_y.numpy()
    exact = logits[score.EXACT_SET]
    loss = score.cross_entropy(exact, forget_y)
    return {
        "forget_accuracy": score.accuracy(exact, forget_y),
        "forget_confidence": float(score.softmax(exact).max(axis=1).mean()),
        "forget_loss_mean": float(loss.mean()),
        "forget_loss_std": float(loss.std()),
        "control_accuracy": score.accuracy(logits[score.CONTROL_SET], world.control_y.numpy()),
        "validation_accuracy": score.accuracy(
            logits[score.VALIDATION_SET], world.validation_y.numpy()
        ),
        "retain_member_accuracy": score.accuracy(
            logits["control_member"], world.member_control_y.numpy()
        ),
    }


def run_world(world: core.World, ood: Tensor, spec: BenchSpec, device: torch.device) -> WorldResult:
    """Train orig / golds, run every candidate, score and proxy it."""
    seed = world.spec.seed
    cache = spec.cache_dir
    fingerprint = spec.train.fingerprint()

    def cache_path(kind: str, index: int) -> Path | None:
        if cache is None:
            return None
        return cache / f"w{seed}_{kind}{index}_{fingerprint}_{world.spec.retain_size}.pt"

    started = time.time()
    all_x = torch.cat([world.retain_x, world.forget_x])
    all_y = torch.cat([world.retain_y, world.forget_y])
    orig_model = core.load_or_train(
        cache_path("orig", 0), all_x, all_y, spec.train, seed + ORIG_SEED_OFFSET, device
    )
    gold_models = [
        core.load_or_train(
            cache_path("gold", index),
            world.retain_x,
            world.retain_y,
            spec.train,
            seed + GOLD_SEED_OFFSET + index,
            device,
        )
        for index in range(GOLD_REFERENCE_COUNT + 1)
    ]
    train_seconds = time.time() - started
    LOGGER.info("world %d: %d models ready in %.0fs", seed, 1 + len(gold_models), train_seconds)

    sets = build_eval_sets(world, ood)
    orig = core.logits_on_sets(orig_model, sets, device)
    golds_full = [core.logits_on_sets(model, sets, device) for model in gold_models]
    reference_golds, held_out_gold = golds_full[:GOLD_REFERENCE_COUNT], golds_full[-1]
    LOGGER.info("world %d orig stats %s", seed, _model_stats(orig, world))
    LOGGER.info("world %d gold0 stats %s", seed, _model_stats(golds_full[0], world))

    orig_val = score.accuracy(orig[score.VALIDATION_SET], world.validation_y.numpy())
    threshold = UTILITY_FRACTION_OF_ORIG * orig_val
    contexts = _scoring_contexts(world, orig, reference_golds, threshold)
    pseudo = orig[score.EXACT_SET].argmax(axis=1).astype(np.int64)
    gold_probs = np.mean(
        [score.softmax(g[score.EXACT_SET], score.SCORING_TEMPERATURE) for g in reference_golds],
        axis=0,
    )
    default_views = [
        orig[view_set_name(DEFAULT_SURROGATE_AREA, view)] for view in range(SURROGATE_VIEW_COUNT)
    ]
    surrogate = score.surrogate_reference(default_views, pseudo, DEFAULT_SURROGATE_TEMPERATURE)

    def evaluate(label: str, method: str, config: dict[str, object], sub: Mapping[str, FloatArray],
                 seconds: float) -> dict[str, object]:  # fmt: skip
        scores: dict[str, dict[str, float | bool]] = {}
        for variant, (context, rename) in contexts.items():
            scores[variant] = score.official_score(_remap_sub(sub, rename), context).to_dict()
        proxies = score.proxy_readings(orig, sub, pseudo)
        surrogate_forget = score.surrogate_forget_score(
            _remap_sub(sub, {}), _remap_sub(orig, {}), pseudo, surrogate
        )
        proxies["surrogate_forget"] = surrogate_forget
        proxies["surrogate_final"] = surrogate_forget / (1.0 + proxies["retain_proxy_kl"])
        return {
            "label": label,
            "method": method,
            "config": config,
            "seconds": seconds,
            "scores": scores,
            "proxies": proxies,
            "stats": _model_stats(sub, world),
        }

    candidates: list[dict[str, object]] = []
    inputs = methods.UnlearnInputs(
        orig=orig_model,
        forget_x=world.forget_x,
        forget_pseudo_y=torch.from_numpy(pseudo),
        proxy_x=ood,
        device=device,
    )
    for config in spec.candidates:
        config = replace(config, seed=seed)
        run_started = time.time()
        model = methods.unlearn_mini(inputs, config)
        seconds = time.time() - run_started
        sub = core.logits_on_sets(model, sets, device)
        candidates.append(evaluate(config.label, config.method, asdict(config), sub, seconds))
        final = candidates[-1]["scores"]
        assert isinstance(final, dict)
        LOGGER.info(
            "world %d %-26s final %.3f forget %.3f retain %.3f (%.0fs)", seed, config.label,
            final[VARIANT_MAIN]["final_score"], final[VARIANT_MAIN]["forget_score"],
            final[VARIANT_MAIN]["retain_score"], seconds,
        )  # fmt: skip
    candidates.append(
        evaluate(HELD_OUT_GOLD_LABEL, "gold", {}, held_out_gold, train_seconds / len(golds_full))
    )

    diagnostics: dict[str, object] = {
        "orig": _model_stats(orig, world),
        "gold_mean_of_stats": {
            key: float(np.mean([_model_stats(g, world)[key] for g in reference_golds]))
            for key in _model_stats(orig, world)
        },
        "utility_threshold": threshold,
        "gold_loss_structure": gold_loss_structure(reference_golds, world.forget_y.numpy()),
        "train_seconds_total": train_seconds,
        "surrogate_grid": surrogate_grid(orig, gold_probs),
        "sizes": {
            "retain": len(world.retain_x),
            "forget": len(world.forget_x),
            "control": len(world.control_x),
            "validation": len(world.validation_x),
            "ood_proxy": len(ood),
        },
    }
    return WorldResult(seed=seed, diagnostics=diagnostics, candidates=candidates)


# ------------------------------------------------------------------------------------------
# Analysis across candidates and worlds
# ------------------------------------------------------------------------------------------

TARGET_KEYS = ("final_score", "forget_score", "retain_score")
PROXY_ORIENTATION = {
    "forget_kl_mean": "higher = moved more",
    "retain_proxy_kl": "lower = drifted less",
    "movement_ratio": "higher = moved on forget not on retain",
    "forget_loss_shift": "higher = moved more",
    "forget_disagreement": "higher = moved more",
    "kit_composite": "higher = better",
    "surrogate_forget": "higher = closer to the surrogate gold",
    "surrogate_final": "higher = better",
}


def _column(rows: Sequence[Mapping[str, object]], variant: str, key: str) -> FloatArray:
    out: list[float] = []
    for row in rows:
        scores = row["scores"]
        assert isinstance(scores, dict)
        out.append(float(scores[variant][key]))
    return np.asarray(out, dtype=np.float64)


def _proxy_column(rows: Sequence[Mapping[str, object]], name: str) -> FloatArray:
    out: list[float] = []
    for row in rows:
        proxies = row["proxies"]
        assert isinstance(proxies, dict)
        out.append(float(proxies[name]))
    return np.asarray(out, dtype=np.float64)


def rank_candidates(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Rows without the reference candidates (identity, held-out gold)."""
    return [row for row in rows if row["label"] not in EXCLUDED_FROM_RANKING]


def correlation_table(
    rows: Sequence[Mapping[str, object]], variant: str = VARIANT_MAIN
) -> dict[str, dict[str, dict[str, float]]]:
    """Spearman (with bootstrap 95 % interval) of every proxy against every true target.

    Returns: {proxy: {target: {"spearman", "ci_low", "ci_high", "pearson", "n"}}}.
    """
    methods_only = rank_candidates(rows)
    table: dict[str, dict[str, dict[str, float]]] = {}
    for proxy in PROXY_ORIENTATION:
        x = _proxy_column(methods_only, proxy)
        x = np.where(np.isfinite(x), x, np.nan)
        table[proxy] = {}
        for target in TARGET_KEYS:
            y = _column(methods_only, variant, target)
            low, high = score.bootstrap_spearman(x, y)
            table[proxy][target] = {
                "spearman": score.spearman(x, y),
                "ci_low": low,
                "ci_high": high,
                "pearson": score.pearson(x, y),
                "n": float(len(x)),
            }
    return table


def top_k_table(
    rows: Sequence[Mapping[str, object]], k: int, variant: str = VARIANT_MAIN
) -> dict[str, float]:
    """Overlap of the k best candidates by true final score and by each proxy (higher = better)."""
    methods_only = rank_candidates(rows)
    truth = _column(methods_only, variant, "final_score")
    out: dict[str, float] = {}
    for proxy in PROXY_ORIENTATION:
        x = _proxy_column(methods_only, proxy)
        lower_is_better = proxy == "retain_proxy_kl"
        out[proxy] = score.top_k_overlap(x, truth, k, x_higher_is_better=not lower_is_better)
    return out


def analyse(results: Sequence[WorldResult]) -> dict[str, object]:
    """Correlations per world and pooled, family-restricted, variants, rank stability, summary."""
    per_world = {str(r.seed): correlation_table(r.candidates) for r in results}
    pooled_rows = [row for r in results for row in r.candidates]
    pooled = correlation_table(pooled_rows)
    gold_soft_rows = [row for row in pooled_rows if str(row["method"]) == methods.METHOD_GOLD_SOFT]
    summary: dict[str, object] = {
        "per_world": per_world,
        "pooled": pooled,
        "gold_soft_family_pooled": correlation_table(gold_soft_rows),
        "gate_passed_pooled": correlation_table(
            [row for row in pooled_rows if row["scores"]["main"]["gate_passed"]]  # type: ignore[index]
        ),
        "sensitivity_unknown_altered": correlation_table(pooled_rows, VARIANT_UNKNOWN_ALTERED),
        "sensitivity_member_control": correlation_table(pooled_rows, VARIANT_MEMBER_CONTROL),
        "top5_overlap_pooled": top_k_table(
            [r for r in pooled_rows if r["label"] not in EXCLUDED_FROM_RANKING], 5
        ),
    }
    labels = [row["label"] for row in rank_candidates(results[0].candidates)]
    if len(results) >= 2:
        vectors = []
        for result in results:
            by_label = {str(row["label"]): row for row in result.candidates}
            vectors.append(
                np.asarray(
                    [
                        _column([by_label[str(lab)]], VARIANT_MAIN, "final_score")[0]
                        for lab in labels
                    ]
                )
            )
        summary["world_to_world_spearman_final"] = score.spearman(vectors[0], vectors[1])
    return summary


# ------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------


def run_bench(spec: BenchSpec, ood: Tensor | None = None) -> list[WorldResult]:
    """Run every world of `spec` (one after the other, each cached on disk when possible)."""
    torch.set_num_threads(TORCH_THREADS)
    device = core.select_device()
    LOGGER.info("device %s", device)
    data = core.load_cifar10(spec.cifar_root)
    proxy = ood if ood is not None else core.load_ood_proxy(spec.proxy_root, OOD_PROXY_COUNT, 0)
    results: list[WorldResult] = []
    for world_seed in spec.world_seeds:
        world = core.make_world(
            data,
            core.WorldSpec(
                seed=world_seed, retain_size=spec.retain_size, forget_size=spec.forget_size
            ),
        )
        results.append(run_world(world, proxy, spec, device))
        if spec.cache_dir is not None:
            spec.cache_dir.mkdir(parents=True, exist_ok=True)
            (spec.cache_dir / f"world_{spec.tag}{world_seed}.json").write_text(
                json.dumps(results[-1].to_dict(), default=float)
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the benchmark and write `metrics_mini.json` (raw results + analysis)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=core.TrainSpec().epochs)
    parser.add_argument("--width", type=int, default=core.TrainSpec().width)
    parser.add_argument("--retain-size", type=int, default=core.WorldSpec.retain_size)
    parser.add_argument("--forget-size", type=int, default=core.WorldSpec.forget_size)
    parser.add_argument("--quick", action="store_true", help="keep 1 candidate per family")
    parser.add_argument("--candidate-set", choices=("default", "lowlr"), default="default")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    candidates = lowlr_candidates() if args.candidate_set == "lowlr" else default_candidates()
    if args.quick:
        first_of_method: dict[str, methods.MiniUnlearnConfig] = {}
        for candidate in candidates:
            first_of_method.setdefault(candidate.method, candidate)
        candidates = tuple(first_of_method.values())
    spec = BenchSpec(
        world_seeds=tuple(args.worlds),
        train=replace(core.TrainSpec(), epochs=args.epochs, width=args.width),
        retain_size=args.retain_size,
        forget_size=args.forget_size,
        cifar_root=args.cifar_root,
        proxy_root=args.proxy_root,
        cache_dir=args.cache_dir,
        candidates=candidates,
        tag="lowlr_" if args.candidate_set == "lowlr" else "",
    )
    results = run_bench(spec)
    payload = {
        "spec": {
            "worlds": list(spec.world_seeds),
            "train": asdict(spec.train),
            "gold_reference_count": GOLD_REFERENCE_COUNT,
            "utility_fraction_of_orig": UTILITY_FRACTION_OF_ORIG,
        },
        "worlds": [r.to_dict() for r in results],
        "analysis": analyse(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=float))
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
