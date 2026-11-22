"""Variant pools of team A3's generations (the content of `jureca/a3_spec.json`).

A pool is ordered: the four workers of a job take variants in this order, so the first ones are
those whose answer is worth the most. Axes are varied ONE AT A TIME around `CENTER`, so each
variant answers one question (the leaderboard and the proxy cards then read as a sensitivity
table), plus two extra seeds of the centre to bound the seed-to-seed spread.

Usage:
    python -m hackathon.unlearning.research.a3_pool --generation g2 --out jureca/a3_spec.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from hackathon.unlearning.research.a3_spec import EPOCH_STEPS, SpecFile, VariantSpec

EPOCHS_SHORT = 3
EPOCHS_MEDIUM = 6
EPOCHS_LONG = 10
GRID_ALPHAS = (1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4)
GRID_ALPHAS_WIDE = (1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5)
"""Round 3: the best conform2 points sat on the 1.4 edge, so the grid reaches `MAX_ALPHA`."""
GRID_HEAD_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
GRID_POST_DELTAS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
CENTER = VariantSpec(
    name="center",
    note="centre of g2: 6 epochs, lr 5e-5, T 3, removal 8 nats, anchor weight 4, 22 % wrong, fallback T 1",
    steps=EPOCHS_MEDIUM * EPOCH_STEPS,
    lr=5e-5,
    retain_weight=4.0,
    target_temperature=3.0,
    delta_full=8.0,
    cal_wrong_fraction=0.22,
    fallback_temperature=1.0,
    alphas=GRID_ALPHAS,
    post_deltas=GRID_POST_DELTAS,
    head_scales=GRID_HEAD_SCALES,
    save_top=1,
)
"""g1 (3 epochs, lr 2e-5) left the student at a removal loss of 1.8 against a 4.8 target: the
centre trains longer and harder, and the grid then trims the extrapolation and the bias."""


PRIORITY_ORDER = (
    "g2_fb1_cal22",
    "g2_fb2_cal22",
    "g2_fb1",
    "g2_fb1_cal22_T2",
    "g2_fbcal30",
    "g2_fbcal15",
    "g2_d10",
    "g2_rw8",
    "g2_fb1_cal30_T2",
    "g2_e10_lr5e-05",
    "g2_flat2",
    "g2_dual_w1",
    "g2_fbcal40",
    "g2_e6_lr1e-04",
    "g2_d12",
    "g2_rw16",
    "g2_flat3",
    "g2_T4",
    "g2_p2",
    "g2_dual_w3",
)
"""Order of the g2 pool. The public scores say the score peaks at an accuracy of about 0.74 on the
non-removed images (0.1114 at 0.78, 0.0916 at 0.63, 0.064 at 0.84) with a removal loss of 7 to 9
and flat in delta from 4 to 8: the variants that land there with fewer OOD class changes come
first (calibrated shares 22 and 15 %, which put the student near 0.78-0.85 before extrapolation),
then depth, anchor weight and the more aggressive shares."""


def _variant(name: str, note: str, **changes: object) -> VariantSpec:
    return replace(CENTER, name=name, note=note, **changes)  # type: ignore[arg-type]


def build_g2_pool() -> tuple[VariantSpec, ...]:
    """Generation g2: sensitivity of the centre along training, retention, T, delta, prior, anchor."""
    # The centre is "g2_fb1_cal22" (calibrated 22 % wrong share + team C's original-fallback targets,
    # T 1): g2 showed it keeps the fallback class (agreement 0.65-0.67 against 0.38-0.43 without it).
    # Every pending axis below changes ONE thing of that centre.
    pool: list[VariantSpec] = [_variant("g2_fb1_cal22", CENTER.note)]
    # Axis A: how far the student goes (steps x lr).
    for epochs, lr in (
        (EPOCHS_SHORT, 5e-5),
        (EPOCHS_MEDIUM, 2e-5),
        (EPOCHS_LONG, 5e-5),
        (EPOCHS_MEDIUM, 1e-4),
    ):
        pool.append(
            _variant(
                f"g2_e{epochs}_lr{lr:.0e}",
                f"{epochs} epochs at lr {lr:.0e}",
                steps=epochs * EPOCH_STEPS,
                lr=lr,
            )
        )
    # Axis B: weight of the retention anchor (drift against forget).
    for weight in (1.0, 2.0, 8.0, 16.0):
        pool.append(_variant(f"g2_rw{weight:g}", f"anchor weight {weight:g}", retain_weight=weight))
    # Axis C: temperature of the targets (gold confidence).
    for temperature in (2.0, 2.5, 3.5, 4.0, 5.0):
        pool.append(
            _variant(f"g2_T{temperature:g}", f"T {temperature:g}", target_temperature=temperature)
        )
    # Axis D: removal depth in the targets.
    for delta in (4.0, 6.0, 10.0, 12.0):
        pool.append(_variant(f"g2_d{delta:g}", f"removal {delta:g} nats", delta_full=delta))
    # Axis E: graded prior shift on the 36 partially forgotten classes.
    for partial in (1.0, 2.0, 4.0):
        pool.append(
            _variant(f"g2_p{partial:g}", f"graded shift {partial:g}", delta_partial=partial)
        )
    # Axis F: a small share of the anchor on strong views of forget images.
    for fraction in (0.1, 0.25):
        pool.append(
            _variant(
                f"g2_anchor{fraction:g}", f"view anchor {fraction:g}", anchor_view_fraction=fraction
            )
        )
    # Seeds of the centre: spread of the proxy and of the candidates at equal settings.
    for seed in (1, 2):
        pool.append(_variant(f"g2_fb1_cal22_s{seed}", f"centre, seed {seed}", seed=seed))
    # Axis I: explicit wrong-image share on the non-removed images (gold test accuracy ~0.7-0.8).
    for fraction in (0.15, 0.30, 0.40):
        pool.append(
            _variant(
                f"g2_fbcal{round(fraction * 100)}",
                f"{fraction:.0%} of the images made wrong (fallback centre)",
                cal_wrong_fraction=fraction,
            )
        )
    # Axis K: team C's "original fallback" targets for the images of the 9 classes.
    pool.append(
        _variant(
            "g2_fb2_cal22", "original fallback T 2, 22 % wrong elsewhere", fallback_temperature=2.0
        )
    )
    pool.append(
        _variant("g2_fb1", "original fallback T 1, plain targets elsewhere", cal_wrong_fraction=0.0)
    )
    # Axis J: the same wrong shares at T 2 (g1: the T 2 student keeps more accuracy after extrapolation).
    for fraction in (0.22, 0.30):
        pool.append(
            _variant(
                f"g2_fb1_cal{round(fraction * 100)}_T2",
                f"{fraction:.0%} wrong, T 2",
                cal_wrong_fraction=fraction,
                target_temperature=2.0,
            )
        )
    # Axis H: flatter targets on the images of the removed classes (the gold hesitates there).
    for flatten in (1.5, 2.0, 3.0):
        pool.append(
            _variant(
                f"g2_flat{flatten:g}",
                f"removed-class targets flattened x{flatten:g}",
                removed_flatten=flatten,
            )
        )
    # Axis G: team C's dual targets (class removed on exact images, kept on strong crops).
    for weight in (1.0, 3.0):
        pool.append(
            _variant(f"g2_dual_w{weight:g}", f"keep-on-crops weight {weight:g}", dual_weight=weight)
        )
    by_name = {variant.name: variant for variant in pool}
    ordered = [by_name[name] for name in PRIORITY_ORDER if name in by_name]
    ordered += [variant for variant in pool if variant.name not in PRIORITY_ORDER]
    pool = ordered
    return tuple(pool)


def build_diag_pool() -> tuple[VariantSpec, ...]:
    """Diagnostic of the systematic penalty of the trained students (public 0.064 against 0.111).

    Two A3 students scored 0.064 where gold_soft + a head-bias scored 0.111 at the SAME proxy values
    (removal loss, accuracy on the other images). The hypothesis tested here: the hidden control
    set holds in-distribution images the original memorised, and a longer or harder training than
    the phase 1 recipe moves the model on them (a "leak"), which the out-of-distribution anchor does
    not see. Each variant trains WITHOUT subset B (256 images) and reports the drift on it (`b_*`
    card fields: unseen, memorised, in-distribution). The recipes differ one factor at a time from
    the phase 1 recipe, at their own batch size, and write no file.
    """
    common: dict[str, Any] = {
        "save_top": 0,
        "holdout": True,
        "no_batch_scale": True,
        "alphas": (1.0, 1.1, 1.2, 1.3),
        "post_deltas": (0.0, 2.0, 4.0, 6.0, 8.0),
    }
    phase1 = replace(
        CENTER,
        name="g3_h_gsrepro",
        note="phase 1 recipe: plain gold_soft, 1 epoch, lr 2e-5, anchor weight 1 (public 0.0457 at alpha 1.1)",
        steps=EPOCH_STEPS, lr=2e-5, retain_weight=1.0, retain_batch_size=8,
        target_temperature=2.0, delta_full=0.0, cal_wrong_fraction=0.0, fallback_temperature=0.0,
        **common,
    )  # fmt: skip
    return (
        phase1,
        replace(CENTER, name="g3_h_mine", note="g2 centre", **common),
        replace(CENTER, name="g3_h_mine_rw1", note="g2 centre, anchor weight 1", retain_weight=1.0, **common),
        replace(CENTER, name="g3_h_mine_e1", note="g2 centre, 1 epoch at lr 2e-5",
                steps=EPOCH_STEPS, lr=2e-5, **common),
        replace(CENTER, name="g3_h_mine_e3", note="g2 centre, 3 epochs at lr 2e-5",
                steps=3 * EPOCH_STEPS, lr=2e-5, **common),
        replace(phase1, name="g3_h_g1t3", note="g1 T3 student recipe: plain, T 3, removal 6, 3 epochs, anchor 4",
                steps=3 * EPOCH_STEPS, retain_weight=4.0, retain_batch_size=16,
                target_temperature=3.0, delta_full=6.0),
    )  # fmt: skip


def build_g4_pool() -> tuple[VariantSpec, ...]:
    """Generation g4: refine the sibling targets, calibrated, one factor at a time around `g3_sib70`.

    The server confirmed the class groups (team C's routing of 2 logit units to the present siblings:
    0.1129, the best score so far) and the loss calibration (mean 2.25-2.7, std ~3.0). What is left
    is the per-image predicted distribution (the factor prediction x retain, constant under every
    post-hoc edit): the mass on the siblings, the temperature of the fallback part, the sharpness of
    the targets on the other classes, the ensemble of views, and the share of images made wrong.
    """
    base = replace(CENTER, sibling_mass=0.7, save_top=1)
    return (
        replace(base, name="g4_sib100", note="all mass on the present siblings", sibling_mass=1.0),
        replace(base, name="g4_sib70_fb2", note="sibling 0.7, fallback part at T 2",
                fallback_temperature=2.0),
        replace(base, name="g4_sib70_T2", note="sibling 0.7, targets at T 2", target_temperature=2.0),
        replace(base, name="g4_sib70_T4", note="sibling 0.7, targets at T 4", target_temperature=4.0),
        replace(base, name="g4_sib70_views8", note="sibling 0.7, 8 views averaged", view_count=8),
        replace(base, name="g4_sib70_cls1", note="sibling 0.7, class-aware wrong share",
                cal_class_aware=1.0),
        replace(base, name="g4_sib70_cal15", note="sibling 0.7, 15 % wrong", cal_wrong_fraction=0.15),
        replace(base, name="g4_sib70_cal30", note="sibling 0.7, 30 % wrong", cal_wrong_fraction=0.30),
        replace(base, name="g4_sib70_s1", note="sibling 0.7, seed 1", seed=1),
        replace(base, name="g4_sib70_s2", note="sibling 0.7, seed 2", seed=2),
        replace(base, name="g4_sib70_dual1", note="sibling 0.7 + keep-on-crops", dual_weight=1.0),
        replace(base, name="g4_sib70_p2", note="sibling 0.7 + graded prior 2", delta_partial=2.0),
    )  # fmt: skip


def build_conform_pool() -> tuple[VariantSpec, ...]:
    """The official-rule base: the phase 1 gold_soft recipe with NO external image anywhere.

    The task statement: "The unlearning method has to work from the original model and the forget
    set alone". `p2_gs_alpha1.10` (and every candidate of team C built on it) was trained with a
    retention anchor on COCO, DIV2K and Kodak images, a grey zone. These variants repeat the recipe
    of `p2_gs_alpha1.10` EXACTLY (219 steps at batch 16, lr 2e-5, T 2, the same targets) but the
    anchor is made only of strong crops of forget images with the original as teacher. The raw state
    (alpha 1, no bias) is saved so that team C replays its recipe on it (alpha 1.1, bias, sibling
    routing, head scale 1.25). Three anchor weights bracket the phase 1 value (1), then the same
    with the removed-class images left out of the anchor, then two lighter weights.
    """
    base = replace(
        CENTER, steps=EPOCH_STEPS, lr=2e-5, retain_batch_size=8, target_temperature=2.0,
        delta_full=0.0, cal_wrong_fraction=0.0, fallback_temperature=0.0, no_batch_scale=True,
        external_anchor=False, save_raw=True, save_top=1,
    )  # fmt: skip
    return (
        replace(base, name="g5c_conform_gs", note="phase 1 recipe, forget-derived anchor only",
                retain_weight=1.0),
        replace(base, name="g5c_conform_gs_rw4", note="same, anchor weight 4", retain_weight=4.0),
        replace(base, name="g5c_conform_gs_rw025", note="same, anchor weight 0.25",
                retain_weight=0.25),
        # Measured on the first three: weight 1 under-forgets (raw state at 92 % accuracy, alpha 1.3
        # needed), 4 forgets nothing, 0.25 matches the reference card at alpha 1.1. The anchor on a
        # removed-class image asks "equal to the original", the opposite of the forgetting target:
        # these variants draw the anchor from the other forget images only.
        replace(base, name="g5c_skip_rw1", note="anchor weight 1, removed-class images excluded",
                retain_weight=1.0, anchor_skip_removed=True),
        replace(base, name="g5c_skip_rw4", note="anchor weight 4, removed-class images excluded",
                retain_weight=4.0, anchor_skip_removed=True),
        replace(base, name="g5c_skip_rw025", note="anchor weight 0.25, removed-class images excluded",
                retain_weight=0.25, anchor_skip_removed=True),
        replace(base, name="g5c_conform_gs_rw05", note="same as the first, anchor weight 0.5",
                retain_weight=0.5),
        replace(base, name="g5c_conform_gs_rw01", note="same as the first, anchor weight 0.1",
                retain_weight=0.1),
    )  # fmt: skip


def build_conform2_pool() -> tuple[VariantSpec, ...]:
    """Second round of the official-rule base: the anchor skips the removed-class images.

    Round 1 (`build_conform_pool`) measured that the anchor weight is a brake on the forgetting:
    weight 1 under-forgets, 4 forgets nothing, 0.25 matches the reference card. These variants draw
    the anchor from the forget images of the OTHER classes only (`anchor_skip_removed`), so the weight
    can protect retention without fighting the removal, then vary one lever at a time around the
    weight 1 skip variant: sibling targets, target temperature, steps, learning rate.
    """
    base = replace(
        CENTER, steps=EPOCH_STEPS, lr=2e-5, retain_batch_size=8, target_temperature=2.0,
        delta_full=0.0, cal_wrong_fraction=0.0, fallback_temperature=0.0, no_batch_scale=True,
        external_anchor=False, save_raw=True, save_top=1, anchor_skip_removed=True,
        retain_weight=1.0,
    )  # fmt: skip
    return (
        replace(base, name="g5c_skip_rw05", note="skip anchor, weight 0.5", retain_weight=0.5),
        replace(base, name="g5c_skip_rw2", note="skip anchor, weight 2", retain_weight=2.0),
        replace(base, name="g5c_skip_rw0", note="no anchor at all (distillation on the forget set)",
                retain_weight=0.0),
        replace(base, name="g5c_skip_sib70_rw05", note="skip anchor 0.5 + sibling targets 0.7",
                retain_weight=0.5, fallback_temperature=1.0, sibling_mass=0.7),
        replace(base, name="g5c_skip_sib70_rw1", note="skip anchor 1 + sibling targets 0.7",
                fallback_temperature=1.0, sibling_mass=0.7),
        replace(base, name="g5c_skip_sib70_rw2", note="skip anchor 2 + sibling targets 0.7",
                retain_weight=2.0, fallback_temperature=1.0, sibling_mass=0.7),
        replace(base, name="g5c_skip_T15", note="skip anchor 1, targets at T 1.5",
                target_temperature=1.5),
        replace(base, name="g5c_skip_T3", note="skip anchor 1, targets at T 3", target_temperature=3.0),
        replace(base, name="g5c_skip_s2", note="skip anchor 1, twice the steps", steps=2 * EPOCH_STEPS),
        replace(base, name="g5c_skip_s2_rw2", note="skip anchor 2, twice the steps",
                steps=2 * EPOCH_STEPS, retain_weight=2.0),
        replace(base, name="g5c_skip_lr4", note="skip anchor 1, lr 4e-5", lr=4e-5),
        replace(base, name="g5c_skip_lr4_rw2", note="skip anchor 2, lr 4e-5", lr=4e-5,
                retain_weight=2.0),
    )  # fmt: skip


def build_conform3_pool() -> tuple[VariantSpec, ...]:
    """Third round: refine the skip-anchor family around its best measured points, and seeds.

    Round 2 (alphas up to 1.4) put the best retention at equal forgetting at the grid edge for the
    heavier anchors (weight 2) and for the softer targets (T 1.5), and the weight-1 variant on the
    reference card of the best public family. This round widens the alpha grid to 1.5, combines the
    two levers (weight and target temperature) and repeats the weight-1 and weight-2 recipes on
    two more seeds: a single run is a sample, and a file to submit must not rest on one seed.
    """
    base = replace(
        CENTER, steps=EPOCH_STEPS, lr=2e-5, retain_batch_size=8, target_temperature=2.0,
        delta_full=0.0, cal_wrong_fraction=0.0, fallback_temperature=0.0, no_batch_scale=True,
        external_anchor=False, save_raw=True, save_top=1, anchor_skip_removed=True,
        retain_weight=1.0, alphas=GRID_ALPHAS_WIDE,
    )  # fmt: skip
    return (
        replace(base, name="g5c_k_rw2_T15", note="skip anchor 2, targets T 1.5", retain_weight=2.0,
                target_temperature=1.5),
        replace(base, name="g5c_k_rw15_T15", note="skip anchor 1.5, targets T 1.5",
                retain_weight=1.5, target_temperature=1.5),
        replace(base, name="g5c_k_rw1_T15", note="skip anchor 1, targets T 1.5",
                target_temperature=1.5),
        replace(base, name="g5c_k_rw2_T1", note="skip anchor 2, targets T 1", retain_weight=2.0,
                target_temperature=1.0),
        replace(base, name="g5c_k_rw3", note="skip anchor 3", retain_weight=3.0),
        replace(base, name="g5c_k_sib70_rw2_T15", note="skip anchor 2 + sibling 0.7 + T 1.5",
                retain_weight=2.0, target_temperature=1.5, fallback_temperature=1.0,
                sibling_mass=0.7),
        replace(base, name="g5c_k_rw1_seed1", note="skip anchor 1, seed 1", seed=1),
        replace(base, name="g5c_k_rw1_seed2", note="skip anchor 1, seed 2", seed=2),
        replace(base, name="g5c_k_rw2_seed1", note="skip anchor 2, seed 1", retain_weight=2.0,
                seed=1),
        replace(base, name="g5c_k_rw2_seed2", note="skip anchor 2, seed 2", retain_weight=2.0,
                seed=2),
    )  # fmt: skip


def build_shape_pool() -> tuple[VariantSpec, ...]:
    """Make the students BIMODAL like gold_soft x1.1 (the best public family).

    Measured on the same 1000 images: p(true class) on the non-removed images is below 0.1 for 11 %
    of them and above 0.9 for 25 % with gold_soft x1.1 (and 21 % / 12 % at x1.25), but for only
    4 % / 13-17 % with every A3 student: the targets' "right" images sit in a diffuse middle
    (label mass 0.55-0.97) and the student does not even reach the "wrong" ones at alpha 1. Server
    scores: 0.1114 against 0.0856 (sib85) and 0.0644 at equal mean and standard deviation of the
    loss. These variants sharpen the calibrated law (confident right, clearly wrong), fit it harder
    (more steps, a higher lr, a lighter anchor) and keep the sibling targets.
    """
    base = replace(CENTER, sibling_mass=0.7, cal_wrong_fraction=0.20, cal_right_low=0.85,
                   cal_wrong_high=0.05, cal_wrong_top_share=0.9)  # fmt: skip
    gs = replace(
        CENTER, steps=EPOCH_STEPS, lr=2e-5, retain_weight=1.0, retain_batch_size=8,
        target_temperature=2.0, delta_full=0.0, cal_wrong_fraction=0.0, fallback_temperature=0.0,
        no_batch_scale=True, save_top=1,
    )  # fmt: skip
    # The targets themselves are NOT bimodal (3 % of the non-removed images below 0.1 at T 2); gold_soft
    # gets its 11 % by extrapolation of a 1-epoch student. The first variants keep that recipe exactly
    # (batch 16, 219 steps, lr 2e-5, anchor 1) and only change the targets of the removed-class images.
    return (
        replace(gs, name="g5s_gs_repro", note="phase 1 recipe as it was (control: shape, loss, file)"),
        replace(gs, name="g5s_gs_sib70", note="phase 1 recipe + sibling targets 0.7 on the removed classes",
                fallback_temperature=1.0, sibling_mass=0.7),
        replace(gs, name="g5s_gs_fb1", note="phase 1 recipe + original fallback on the removed classes",
                fallback_temperature=1.0),
        replace(gs, name="g5s_gs_sib70_T15", note="same, targets at T 1.5",
                fallback_temperature=1.0, sibling_mass=0.7, target_temperature=1.5),
        replace(gs, name="g5s_gs_sib70_T1", note="same, targets at T 1",
                fallback_temperature=1.0, sibling_mass=0.7, target_temperature=1.0),
        replace(base, name="g5s_sharp", note="sharp calibrated law, centre recipe"),
        replace(base, name="g5s_sharp_hard", note="sharp law, 2190 steps at lr 1e-4, anchor 1",
                steps=10 * EPOCH_STEPS, lr=1e-4, retain_weight=1.0),
        replace(base, name="g5s_sharp_e6_rw1", note="sharp law, anchor 1", retain_weight=1.0),
        replace(base, name="g5s_sharp_w30", note="sharp law, 30 % wrong", cal_wrong_fraction=0.30),
        replace(base, name="g5s_sharp_w15", note="sharp law, 15 % wrong", cal_wrong_fraction=0.15),
        replace(base, name="g5s_sharp_T2", note="sharp law, T 2", target_temperature=2.0),
        replace(base, name="g5s_sharp_hard_w30", note="sharp, hard, 30 % wrong", steps=10 * EPOCH_STEPS,
                lr=1e-4, retain_weight=1.0, cal_wrong_fraction=0.30),
        replace(base, name="g5s_vsharp", note="very sharp: right >= 0.93, wrong <= 0.02",
                cal_right_low=0.93, cal_wrong_high=0.02, cal_wrong_top_share=1.0),
    )  # fmt: skip


def build_g5_pool() -> tuple[VariantSpec, ...]:
    """Generation g5: the RETAIN lever, measured on images the student never saw.

    Team C: the public score is about 0.116 x the mean/std sub-score, and the factor prediction x
    retain (about 0.12) is the same for every post-hoc edit, so it is the ceiling. `retain` compares
    the original with the student on a hidden control set; if that set holds images the original
    memorised, its KL is what the diagnostic measured on subset B (never trained on): 1.6 for one
    epoch at lr 2e-5 against 2.3 for the phase 1 recipe and 2.6 for six epochs. A retain factor of
    1 / (1 + KL) moves by about 25 % between those. These variants train WITHOUT subset B on the
    sibling base, with gentler and shorter recipes, and the winner is retrained on all 3500 images.
    """
    common: dict[str, Any] = {
        "holdout": True,
        "no_batch_scale": True,
        "save_top": 0,
        "alphas": (1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5),
        "post_deltas": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    }
    base = replace(CENTER, sibling_mass=0.7, **common)
    pool: list[VariantSpec] = []
    for steps, lr in ((EPOCH_STEPS // 2, 2e-5), (EPOCH_STEPS, 2e-5), (2 * EPOCH_STEPS, 2e-5),
                      (EPOCH_STEPS, 1e-5), (EPOCH_STEPS, 4e-5), (2 * EPOCH_STEPS, 1e-5)):  # fmt: skip
        name = f"g5_h_s{steps}_lr{lr:.0e}"
        pool.append(replace(base, name=name, note=name, steps=steps, lr=lr))
    for weight in (1.0, 16.0):
        pool.append(replace(base, name=f"g5_h_e1_rw{weight:g}", note=f"1 epoch, anchor weight {weight:g}",
                            steps=EPOCH_STEPS, lr=2e-5, retain_weight=weight))  # fmt: skip
    for temperature in (2.0, 4.0):
        pool.append(replace(base, name=f"g5_h_e1_T{temperature:g}", note=f"1 epoch, T {temperature:g}",
                            steps=EPOCH_STEPS, lr=2e-5, target_temperature=temperature))  # fmt: skip
    return tuple(pool)


def build_scale_check() -> tuple[VariantSpec, ...]:
    """The g2 centre again under another name: compares a scaled-batch task with the g2 baseline.

    A task that runs at batch scale k (memory lever) trains with a bigger batch and a scaled lr; to
    know whether the cards stay comparable, this variant has the SAME settings as `g2_fb1_cal22`
    (done at batch 16) and runs at whatever scale the task uses.
    """
    return (_variant("g3_scale_check", "g2_fb1_cal22 settings, run at the task's batch scale"),)


def build_sibling_pool() -> tuple[VariantSpec, ...]:
    """Team C's sibling targets for the images of the 9 removed classes, around the fallback centre.

    The organisers' class groups overlap the forget set: a gold that never saw class 6, 7 or 8 has
    only class 5 left in its group, so it puts such images on their present siblings, not on the
    "first remaining class" of the original (which is a present sibling in only 38 % of the cases).
    The target keeps depending on the image (spread by the original's logits restricted to the
    siblings, then mixed with the original fallback), because the real gold's loss on these images
    is heterogeneous (team C).
    """
    return tuple(
        _variant(
            f"g3_sib{round(mass * 100)}",
            f"sibling mass {mass:.2f} + original fallback T 1 on the removed classes",
            sibling_mass=mass,
        )
        for mass in (0.7, 0.5, 0.85)
    )


def build_g3_pool() -> tuple[VariantSpec, ...]:
    """Generation g3: NEW dimensions around the calibrated centre, beyond alpha / delta tuning.

    The public fit puts the alpha / delta / accuracy family at a ceiling near 0.14, so g3 changes what
    the targets say, one thing at a time: the class-aware wrong share (a gold is less accurate on the
    classes it saw few images of), the ensemble size and crop area of the averaged views (the gold is
    an average of models), the original-fallback targets of team C on the removed classes, and their
    combination with the dual loss and the graded prior.
    """
    base = replace(CENTER, cal_wrong_fraction=0.22)
    pool = [
        replace(base, name="g3_cls0.5", note="class-aware wrong share 0.5", cal_class_aware=0.5),
        replace(base, name="g3_cls1", note="class-aware wrong share 1.0", cal_class_aware=1.0),
        replace(base, name="g3_views8", note="8 strong views averaged", view_count=8),
        replace(
            base, name="g3_views8_a0.25", note="8 views, area 0.25", view_count=8, view_area=0.25
        ),
        replace(base, name="g3_views8_a0.5", note="8 views, area 0.5", view_count=8, view_area=0.5),
        replace(
            base,
            name="g3_fb1_views8",
            note="original fallback T 1 + 8 views",
            fallback_temperature=1.0,
            view_count=8,
        ),
        replace(
            base,
            name="g3_fb1_dual1",
            note="original fallback T 1 + keep-on-crops",
            fallback_temperature=1.0,
            dual_weight=1.0,
        ),
        replace(
            base,
            name="g3_fb1_p2",
            note="original fallback T 1 + graded prior 2",
            fallback_temperature=1.0,
            delta_partial=2.0,
        ),
        replace(
            base,
            name="g3_fb1_cls1",
            note="original fallback T 1 + class-aware 1.0",
            fallback_temperature=1.0,
            cal_class_aware=1.0,
        ),
    ]
    return tuple(pool)


FILLER_CAL_FRACTIONS = (0.18, 0.26, 0.34, 0.45)
FILLER_DELTAS = (8.0, 10.0)
FILLER_TEMPERATURES = (2.5, 3.5)


def build_filler_pool() -> tuple[VariantSpec, ...]:
    """Low-priority pool for the interruptible filler task: the calibrated-accuracy region.

    The public fit says the score falls with the accuracy on the non-removed images and rises with
    the removal depth, so the filler crosses the share of images made wrong, the removal depth and
    the temperature around the centre, then varies training length and seeds at one setting.
    Each variant that finishes before the SIGTERM of a real task is kept.
    """
    base = replace(CENTER, save_top=1, alphas=(1.0, 1.1), post_deltas=(0.0, 2.0, 4.0))
    pool: list[VariantSpec] = []
    for fraction in FILLER_CAL_FRACTIONS:
        for delta in FILLER_DELTAS:
            for temperature in FILLER_TEMPERATURES:
                name = f"f1_cal{round(fraction * 100)}_d{delta:g}_T{temperature:g}"
                pool.append(
                    replace(
                        base,
                        name=name,
                        note=name,
                        cal_wrong_fraction=fraction,
                        delta_full=delta,
                        target_temperature=temperature,
                    )
                )
    pivot = replace(base, cal_wrong_fraction=0.30, delta_full=10.0)
    for epochs in (EPOCHS_SHORT, EPOCHS_LONG):
        pool.append(replace(pivot, name=f"f1_e{epochs}", note=f"pivot, {epochs} epochs",
                            steps=epochs * EPOCH_STEPS))  # fmt: skip
    pool.append(replace(pivot, name="f1_flat2", note="pivot, flattened x2", removed_flatten=2.0))
    for seed in (1, 2):
        pool.append(replace(pivot, name=f"f1_seed{seed}", note=f"pivot, seed {seed}", seed=seed))
    return tuple(pool)


SIBLINGS_AFTER = 5
"""Position of the sibling variants in the combined list: right behind the first g2 variants
(already done when this was written), ahead of the remaining g2 axes."""


def build_all_pool() -> tuple[VariantSpec, ...]:
    """The spec of a job: the official-rule base first, then shape, g5, diag, siblings, g4, g2, g3.

    The conforming base (`build_conform_pool`) leads: team C replays its post-hoc recipe on those
    raw states, so they must exist as early as possible in the first task that runs.
    """
    g2 = build_g2_pool()
    return (
        *build_conform_pool(),
        *build_conform2_pool(),
        *build_conform3_pool(),
        *g2[:SIBLINGS_AFTER],
        *build_shape_pool(),
        *build_g5_pool(),
        *build_diag_pool(),
        *build_scale_check(),
        *build_sibling_pool(),
        *build_g4_pool(),
        *g2[SIBLINGS_AFTER:],
        *build_g3_pool(),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write the pool of a generation as a spec file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="g2")
    parser.add_argument(
        "--pool",
        choices=(
            "g2",
            "g3",
            "g4",
            "g5",
            "shape",
            "conform",
            "conform2",
            "sib",
            "diag",
            "all",
            "filler",
        ),
        default="all",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    builders = {
        "g2": build_g2_pool,
        "g3": build_g3_pool,
        "sib": build_sibling_pool,
        "g4": build_g4_pool,
        "g5": build_g5_pool,
        "shape": build_shape_pool,
        "conform": build_conform_pool,
        "conform2": build_conform2_pool,
        "conform3": build_conform3_pool,
        "diag": build_diag_pool,
        "all": build_all_pool,
        "filler": build_filler_pool,
    }
    variants = builders[args.pool]()
    spec = SpecFile(generation=args.generation, variants=variants)
    payload = {
        "generation": spec.generation,
        "variants": [
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(variant).items()
            }
            for variant in spec.variants
        ],
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(spec.variants)} variants written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
