"""Named experiment groups of team A (one per research track). See the report for the verdicts.

All exploration runs train on subset A (256 forget images) with batch 8 and the cosine schedule of
`learning_rate_at`; `CENTER` is the reference point every sweep varies ONE factor around.
"""

from __future__ import annotations

from dataclasses import replace

from hackathon.unlearning.research.trainer import RunSpec

CENTER = RunSpec(
    name="gs_c24", method="gold_soft", steps=24, batch_size=8, retain_batch_size=8, lr=3e-5,
    target_temperature=2.0, target_area=0.35, target_views=3, eval_every=8, cache_views=True,
)  # fmt: skip
"""gold_soft at the kit's default targets: 24 steps of batch 8 (subset A seen 0.75 times), lr 3e-5.

Teacher views are cached once per image (`cache_views`) so a step costs three student passes.
`gs_center` (first run, 32 steps, lr 5e-5, views resampled each step) is kept as a reference.
"""


def _variant(name: str, **changes: object) -> RunSpec:
    return replace(CENTER, name=name, eval_every=0, **changes)  # type: ignore[arg-type]


SWEEP_GOLD_SOFT = [
    CENTER,
    _variant("gs_temp1.0", target_temperature=1.0),
    _variant("gs_temp3.0", target_temperature=3.0),
    _variant("gs_temp4.0", target_temperature=4.0),
    _variant("gs_area0.5", target_area=0.5),
    _variant("gs_area0.2", target_area=0.2),
    _variant("gs_lr1e-5", lr=1e-5),
    _variant("gs_lr1e-4", lr=1e-4, alphas=(0.25, 0.5, 0.75)),
    _variant("gs_retain4", retain_weight=4.0),
    _variant("gs_noaltered", altered_weight=0.0),
]

SCRUB_REFERENCE = [RunSpec(name="scrub_lr3e-5", method="scrub", steps=24, batch_size=8, lr=3e-5)]

RESTRICTED = [
    _variant(f"rs_{mode}", trainable=mode, lr=lr)
    for mode, lr in (("head", 3e-4), ("last2", 3e-5), ("layerscale_head", 3e-4), ("salient", 3e-5))
]

FALLBACK = [
    _variant(f"fb_{kind}", method="fallback", fallback_kind=kind, target_views=4, target_area=0.25)
    for kind in ("rank2", "view_mode")
]

NPO = [
    RunSpec(name="npo_lr1e-5", method="npo", steps=24, batch_size=8, retain_batch_size=8, lr=1e-5),
    RunSpec(name="npo_lr3e-5", method="npo", steps=24, batch_size=8, retain_batch_size=8, lr=3e-5),
    RunSpec(
        name="npo_floor0.6_lr3e-5", method="npo", steps=24, batch_size=8, retain_batch_size=8,
        lr=3e-5, npo_p_floor=0.6,
    ),
]  # fmt: skip

SMOKE = [_variant("smoke_gs", steps=4)]


def _pick(pool: list[RunSpec], *names: str) -> list[RunSpec]:
    by_name = {spec.name: spec for spec in pool}
    return [by_name[name] for name in names]


ALL_POOL = [*SWEEP_GOLD_SOFT, *SCRUB_REFERENCE, *RESTRICTED, *FALLBACK, *NPO]
PRIORITY = _pick(
    ALL_POOL,
    "gs_c24",
    "scrub_lr3e-5",
    "npo_lr3e-5",
    "gs_temp3.0",
    "gs_temp1.0",
    "rs_head",
    "rs_last2",
    "fb_rank2",
    "gs_lr1e-4",
    "gs_area0.5",
    "npo_floor0.6_lr3e-5",
    "rs_layerscale_head",
    "rs_salient",
    "gs_lr1e-5",
    "fb_view_mode",
    "gs_retain4",
    "gs_area0.2",
    "npo_lr1e-5",
    "gs_temp4.0",
    "gs_noaltered",
)
"""Run order of the time-boxed exploration: one-factor tracks first, refinements last."""

CANDIDATE_BASE = RunSpec(
    name="cand_gold_soft", method="gold_soft", steps=219, batch_size=16, retain_batch_size=8,
    lr=2e-5, target_temperature=2.0, target_area=0.35, target_views=3, train_on="all",
    cache_views=True, alphas=(0.5, 0.75, 1.0),
)  # fmt: skip
"""One epoch over the 3500 forget images; blended at three strengths = three candidate files."""

PHASE2_CANDIDATES_DIR = "candidates/phase2"
PHASE2_WARM_START = "candidates/cand_gold_soft_alpha1.00.pt"
"""Phase 1 gold_soft state (alpha 1), relative to the task dir: warm start of the calibrated run."""

CAL_WARM = RunSpec(
    name="p2_cal_warm", method="calibrated", steps=150, batch_size=16, retain_batch_size=8,
    lr=2e-5, retain_weight=4.0, target_temperature=2.0, target_area=0.35, target_views=3,
    train_on="all", cache_views=True, alphas=(0.85, 1.0, 1.25), cal_wrong_top_share=0.0,
    init_from=PHASE2_WARM_START,
    note="calibrated targets (78 % label accuracy, loss mean 0.88), warm start from gold_soft",
)  # fmt: skip
"""Phase 2: explicit test-like targets (accuracy ~78 %, loss mean ~0.9) on all 3500 images.

`cal_wrong_top_share=0.0` is the variant that was launched first (before the guarantee existed).
"""

CAL_COLD = replace(
    CAL_WARM, name="p2_cal_cold", init_from="", steps=219, lr=3e-5, cal_wrong_top_share=0.5,
    note="same targets, trained from the original",
)
"""Same targets from the original weights (no inheritance of the gold_soft drift)."""

CAL_SELF = replace(
    CAL_COLD, name="p2_cal_self", steps=219, lr=3e-5, retain_batch_size=16,
    anchor_view_fraction=0.5, alphas=(0.9, 1.0, 1.15),
    note="calibrated targets, half of the anchor on strong views of the forget images",
)  # fmt: skip
"""Cold start; anchor = half OOD retain proxy, half strong views of forget (in-distribution)."""

REMOVED_CLASSES = (0, 6, 7, 8, 29, 30, 31, 33, 38)
"""HYPOTHESIS (r-unl-c): the 9 classes holding 129-130 forget images are probably entirely forgotten."""

MASK_TRAIN = replace(
    CANDIDATE_BASE, name="p2_mask_train", steps=219, lr=3e-5, retain_weight=4.0,
    alphas=(0.9, 1.0, 1.1), mask_classes=REMOVED_CLASSES, mask_floor=1e-4,
    note="gold_soft targets with the 9 removed classes at 1e-4 on every image",
)  # fmt: skip

PHASE2_JURECA = [
    replace(MASK_TRAIN, name="j_mask_e1", note="1 epoch, lr 3e-5, T 2, floor 1e-4"),
    replace(
        MASK_TRAIN, name="j_mask_e3", steps=3 * 219, lr=2e-5,
        note="3 epochs, lr 2e-5 (more movement, gentler steps)",
    ),
    replace(
        MASK_TRAIN, name="j_mask_T3_e2", steps=2 * 219, target_temperature=3.0,
        note="2 epochs, softer targets (T 3)",
    ),
    replace(
        MASK_TRAIN, name="j_mask_f1e3_e2", steps=2 * 219, mask_floor=1e-3,
        note="2 epochs, milder removal (floor 1e-3, pseudo-label loss ~6.9)",
    ),
]  # fmt: skip
"""One job on 4 GPUs: the four class-removed targets variants, one per GPU (JURECA phase 2)."""

EPOCH_STEPS = 219
"""Optimiser steps of one pass over the 3500 images at batch 16."""

PHASE2_FULL = [
    replace(
        CANDIDATE_BASE, name="full_gs_e3_rw4", steps=3 * EPOCH_STEPS, lr=1.5e-5, retain_weight=4.0,
        alphas=(0.75, 1.0, 1.25, 1.5), note="gold_soft, 3 epochs, anchor x4",
    ),
    replace(
        CANDIDATE_BASE, name="full_gs_e3_rw4_T3", steps=3 * EPOCH_STEPS, lr=1.5e-5,
        retain_weight=4.0, target_temperature=3.0, alphas=(0.75, 1.0, 1.25, 1.5),
        note="softer targets (T = 3)",
    ),
    replace(
        CAL_COLD, name="full_cal_e2", steps=2 * EPOCH_STEPS, lr=2e-5, alphas=(0.85, 1.0, 1.25),
        note="calibrated targets, 2 epochs, from the original",
    ),
    replace(
        CAL_COLD, name="full_cal_e2_wrong30", steps=2 * EPOCH_STEPS, lr=2e-5,
        cal_wrong_fraction=0.30, alphas=(0.85, 1.0, 1.25), note="30 % of wrong targets",
    ),
]
"""Full-scale runs prepared for JURECA (one A100): not run locally, see the report for the times."""

GROUPS: dict[str, list[RunSpec]] = {
    "phase2": [CAL_WARM, CAL_COLD],
    "phase2_self": [CAL_SELF],
    "phase2_mask": [MASK_TRAIN],
    "phase2_jureca": PHASE2_JURECA,
    "phase2_full": PHASE2_FULL,
    "priority": PRIORITY,
    "candidates": [CANDIDATE_BASE],
    "smoke": SMOKE,
    "sweep_gold_soft": SWEEP_GOLD_SOFT,
    "scrub_reference": SCRUB_REFERENCE,
    "restricted": RESTRICTED,
    "fallback": FALLBACK,
    "npo": NPO,
}
