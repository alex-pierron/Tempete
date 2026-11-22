"""Step 0 of band_ascent (V5): what the bench's gold stand-ins say before any training.

Three questions, per stand-in (sXX crops, mXX masks; the masks are the family the V5 selection,
drawn from crops, is not built on), for a few release shares phi:

* mass: share of the stand-in's mean loss (`loss_share`) and of the prediction-score
  denominator KL(p_gold || p_orig) at T = 2 (`pred_share`) carried by the images band_ascent would
  push, against the best possible choice of the same size (`loss_share_oracle`). What the V5
  leaves on the table by not moving the other images;
* selection: does the pushed set R_phi pick the images the stand-in gets wrong (argmax of its
  view mixture != pseudo-label)? `precision` against `base_rate`, and `recall`;
* direction: the prediction score reached at free logits when the student puts exactly the
  stand-in's mass on the pseudo-label and shapes the wrong classes like the original on the exact
  image (`runner_ups`, what the ascent does to first order) or like the mixture of the method's
  own views (`views`). With p_y matched, KL(q || p) = (1 - q_y) KL(q~ || p~) on the wrong classes.

V5.1 (rapport section "V5.1") adds, still without training:

* encoder: the original's class token after each block and at the head input, captured by hooks
  during the forward already run on the exact images, and a k-NN vote cross-fitted on the V5
  cross-validation folds (an image's neighbours come from the other folds, rapport eq. v51knn):
  its loss on the pseudo-label is a selection score (`selection_by.knn_<block>`), its wrong
  classes a direction (`direction.<key>.knn_<block>`); `encoder.best` names the block of highest
  mean selection precision on the mask stand-ins, the rule fixed before any run;
* defects: the starvation of the loss ascent at the original (`force_*`), the push of the held
  images below their ceiling (`held_over_pushed`), and the drift at T = 2 an image held at the
  loss ceiling band_floor is allowed (`floor_kl_*`).

V_Dumb2 (rapport section "V_Dumb2") adds `level_ceiling`: the bench's forget on the exact images
when every image sits exactly in the band [t, band_tolerance * t] that its max-step aims, at free
logits (the pseudo-label logit alone moved, nothing else drifts), for a few levels t.

No training, no file written: `diagnose` prints the report and writes it as JSON.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.bench import (
    AUDIT,
    AUDIT_VIEW,
    EXACT,
    FAMILY_PREFIXES,
    gold_view_logits,
    score_from_logits,
)
from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.methods import (
    CV_SEED_OFFSET,
    anchor_fold_indices,
    band_margins,
    loss_margin,
    view_logits,
)
from hackathon.unlearning.model import model_device, predict
from hackathon.unlearning.proxy_eval import tempered_kl

RELEASES = (0.10, 0.25, 0.50)
"""Release shares phi examined (keys phi10, phi25, phi50)."""
FLOOR = 1e-12
PRE_LOGITS = -1
"""Block key of the head input: the class token after the final norm, what the head sees."""
KNN_NEIGHBOURS = 10
KNN_TEMPERATURE = 0.1
"""k and tau of the vote, fixed a priori: sweeping them on the stand-ins would tune in-sample."""
MASK_PREFIX = "m"
"""Keys of the mask stand-ins, the family the selections (drawn from crops) are not built on."""
LEVELS = (0.10, 0.20, 0.31, 0.45, 0.63, 1.00)
"""V_Dumb2 levels t of `level_ceiling`; 0.31 / 0.45 / 0.63: the original's mean loss on one strong
view of area 0.60 / 0.50 / 0.45 (measurements/v3_targets.json)."""


def _top(values: Tensor, share: float) -> Tensor:
    mask = torch.zeros(len(values), dtype=torch.bool)
    mask[torch.topk(values, max(1, round(share * len(values)))).indices] = True
    return mask


def _wrong_classes(probs: Tensor, labels: Tensor) -> Tensor:
    """The distribution restricted to the classes other than the label, renormalised."""
    out = probs.scatter(1, labels[:, None], 0.0)
    return out / out.sum(dim=1, keepdim=True).clamp_min(FLOOR)


def _kl(p: Tensor, q: Tensor) -> Tensor:
    """Per-row KL(p || q) (rows of probabilities, zeros of p contribute nothing)."""
    return (p * (p.clamp_min(FLOOR).log() - q.clamp_min(FLOOR).log())).sum(dim=1)


def _view_loss(views: Sequence[Tensor], labels: Tensor) -> Tensor:
    """Mean cross-entropy of the views on the labels (the V5 selection score l_bar)."""
    return torch.stack(
        [torch_f.cross_entropy(v.double().cpu(), labels, reduction="none") for v in views]
    ).mean(dim=0)


@contextmanager
def capture_cls(model: nn.Module, blocks: Sequence[int]) -> Iterator[dict[int, list[Tensor]]]:
    """Inside the context, every forward of `model` appends, per key of `blocks`, the class token
    [B, D] (float32, CPU) after `model.blocks[key]`, or the head input for PRE_LOGITS. Read-only:
    the hooks return nothing, so the logits are those of a plain forward."""
    store: dict[int, list[Tensor]] = {block: [] for block in blocks}
    head, layers = cast(nn.Module, model.head), cast(nn.ModuleList, model.blocks)
    handles = []
    for block in blocks:
        parts = store[block]
        if block == PRE_LOGITS:
            handles.append(
                head.register_forward_pre_hook(
                    lambda _m, args, parts=parts: parts.append(args[0].detach().float().cpu())
                )
            )
        else:
            handles.append(
                layers[block].register_forward_hook(
                    lambda _m, _a, out, parts=parts: parts.append(out[:, 0].detach().float().cpu())
                )
            )
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def knn_vote(
    features: Tensor,
    probs: Tensor,
    exclude: Tensor,
    k: int = KNN_NEIGHBOURS,
    tau: float = KNN_TEMPERATURE,
) -> Tensor:
    """[N, C] similarity-weighted mean of the neighbours' `probs` over the k nearest (cosine of
    centred features) outside `exclude` [N, N] (rapport eq. v51knn)."""
    centred = torch_f.normalize(features.double() - features.double().mean(dim=0), dim=1)
    similarity = (centred @ centred.T).masked_fill(exclude, -torch.inf)
    allowed = int((~exclude).sum(dim=1).min())
    top = similarity.topk(min(k, allowed), dim=1)
    weights = torch.softmax(top.values / tau, dim=1)
    return (weights[..., None] * probs.double()[top.indices]).sum(dim=1)


def encoder_signals(
    features: Mapping[int, Tensor], original_logits: Tensor, folds: Sequence[Tensor]
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, float]]:
    """(scores, shapes, agreement) of the cross-fitted k-NN vote at each block: the vote's loss on
    the pseudo-label (selection score), its wrong classes at T = 2 (direction) and how often its
    argmax is the pseudo-label (near 1: the block is collapsed on the memorised classes)."""
    original = original_logits.double().cpu()
    labels = original.argmax(dim=1)
    fold_of = torch.empty(len(original), dtype=torch.long)
    for index, fold in enumerate(folds):
        fold_of[fold] = index
    exclude = fold_of[:, None] == fold_of[None, :]  # the image itself and its own fold
    probs = torch.softmax(original, dim=1)
    tempered = torch.softmax(original / SCORING_TEMPERATURE, dim=1)
    scores: dict[str, Tensor] = {}
    shapes: dict[str, Tensor] = {}
    agreement: dict[str, float] = {}
    for block, block_features in features.items():
        name = "knn_pre_logits" if block == PRE_LOGITS else f"knn_b{block}"
        vote = knn_vote(block_features, probs, exclude)
        scores[name] = -vote.gather(1, labels[:, None]).squeeze(1).clamp_min(FLOOR).log()
        shapes[name] = _wrong_classes(knn_vote(block_features, tempered, exclude), labels)
        agreement[name] = float((vote.argmax(dim=1) == labels).double().mean())
    return scores, shapes, agreement


def band_defects(
    original_logits: Tensor, view_loss: Tensor, config: UnlearnConfig
) -> dict[str, float]:
    """The three V5 defects of the rapport (section V5.1), read on the original at the first step:
    share of the ascent force carried by the top 10 % of the pushed images and its participation
    ratio (1: even, ~0: starved), the push of the held images below their ceiling relative to the
    ascent, and the drift at T = 2 of an image whose label logit alone drops to the loss band_floor."""
    original = original_logits.double().cpu()
    labels = original.argmax(dim=1)
    loss = torch_f.cross_entropy(original, labels, reduction="none")
    force = (torch.softmax(original, dim=1) - torch_f.one_hot(labels, original.shape[1])).norm(
        dim=1
    )
    pushed = _top(view_loss, config.band_release)
    lift = force[pushed]
    floor = 0.0 if config.band_floor is None else config.band_floor
    held_weight = torch.sigmoid(config.band_sharpness * (loss - floor))  # s(beta (L - h)), V5
    drop = (band_margins(original, labels) - loss_margin(torch.tensor(floor).double())).clamp_min(0)
    lowered = original.scatter_add(1, labels[:, None], -drop[:, None])
    floor_kl = tempered_kl(original, lowered)
    return {
        "force_top10_share": float(lift[_top(lift, 0.10)].sum() / lift.sum().clamp_min(FLOOR)),
        "force_participation": float(
            lift.sum() ** 2 / (len(lift) * (lift**2).sum()).clamp_min(FLOOR)
        ),
        "held_over_pushed": float(
            (held_weight * force)[~pushed].sum() / lift.sum().clamp_min(FLOOR)
        )
        if config.band_floor is not None
        else 0.0,
        "floor_kl_mean": float(floor_kl.mean()),
        "floor_kl_median": float(floor_kl.median()),
    }


def step0_report(
    original_logits: Tensor,
    method_views: Sequence[Tensor],
    gold: Mapping[str, Sequence[Tensor]],
    releases: Sequence[float] = RELEASES,
    scores: Mapping[str, Tensor] | None = None,
    shapes: Mapping[str, Tensor] | None = None,
) -> dict[str, dict[str, object]]:
    """Step-0 report from logits (pure function, float64 on the CPU). `scores` (higher = released
    first) are judged like the V5 selection under `selection_by`; `shapes` (wrong-class laws)
    join the directions."""
    original = original_logits.double().cpu()
    labels = original.argmax(dim=1)
    views = [view.double().cpu() for view in method_views]
    view_loss = _view_loss(views, labels)
    orig_probs = torch.softmax(original / SCORING_TEMPERATURE, dim=1)
    view_mix = torch.stack([torch.softmax(v / SCORING_TEMPERATURE, dim=1) for v in views]).mean(0)
    shapes = {
        "runner_ups": _wrong_classes(orig_probs, labels),
        "views": _wrong_classes(view_mix, labels),
        **{name: shape.double().cpu() for name, shape in (shapes or {}).items()},
    }
    extra = {name: score.double().cpu() for name, score in (scores or {}).items()}
    report: dict[str, dict[str, object]] = {"mass": {}, "selection": {}, "direction": {}}
    if extra:
        report["selection_by"] = {name: {} for name in extra}
    for key, logits in gold.items():
        stand_in = [view.double().cpu() for view in logits]
        gold_loss = torch.stack(
            [torch_f.cross_entropy(v, labels, reduction="none") for v in stand_in]
        ).mean(dim=0)
        gold_probs = torch.stack(
            [torch.softmax(v / SCORING_TEMPERATURE, dim=1) for v in stand_in]
        ).mean(dim=0)
        denominator = _kl(gold_probs, orig_probs)
        errors = gold_probs.argmax(dim=1) != labels
        mass: dict[str, dict[str, float]] = {}
        selection: dict[str, dict[str, float]] = {}
        for share in releases:
            name = f"phi{round(share * 100)}"
            chosen = _top(view_loss, share)
            mass[name] = {
                "loss_share": float(gold_loss[chosen].sum() / gold_loss.sum()),
                "loss_share_oracle": float(
                    gold_loss[_top(gold_loss, share)].sum() / gold_loss.sum()
                ),
                "pred_share": float(denominator[chosen].sum() / denominator.sum().clamp_min(FLOOR)),
            }
            selection[name] = {
                "precision": float(errors[chosen].double().mean()),
                "recall": float(errors[chosen].sum() / max(1, int(errors.sum()))),
                "base_rate": float(errors.double().mean()),
            }
        true_mass = gold_probs.gather(1, labels[:, None]).squeeze(1)
        target_shape = _wrong_classes(gold_probs, labels)
        reference = float(denominator.mean())
        report["mass"][key] = mass
        report["selection"][key] = selection
        for name, score in extra.items():
            by_share: dict[str, dict[str, float]] = {}
            for share in releases:
                chosen = _top(score, share)
                by_share[f"phi{round(share * 100)}"] = {
                    "precision": float(errors[chosen].double().mean()),
                    "recall": float(errors[chosen].sum() / max(1, int(errors.sum()))),
                    "loss_share": float(gold_loss[chosen].sum() / gold_loss.sum()),
                    "pred_share": float(
                        denominator[chosen].sum() / denominator.sum().clamp_min(FLOOR)
                    ),
                }
            report["selection_by"][name][key] = by_share  # type: ignore[index]
        left = {
            n: float(((1 - true_mass) * _kl(target_shape, s)).mean()) for n, s in shapes.items()
        }
        report["direction"][key] = {
            name: 1.0 - min(1.0, value / max(reference, FLOOR)) for name, value in left.items()
        }
    return report


def level_ceiling(
    original_logits: Tensor,
    gold: Mapping[str, Sequence[Tensor]],
    tolerance: float,
    levels: Sequence[float] = LEVELS,
) -> dict[str, dict[str, float]]:
    """Per level t (key t045 for 0.45): the bench's forget on the exact images, S_ms x S_pred,
    under each stand-in and the worst of each family (min_crop, min_mask), when each image's
    margin on its pseudo-label is clamped into the band [m(tolerance * t), m(t)] by that logit
    alone. The wrong classes keep the original's shape (what the ascent does to first order) and
    the audit fold does not move (retain 1): an upper bound of what reaching the band gives."""
    original = original_logits.double().cpu()
    labels = original.argmax(dim=1)
    margins = band_margins(original, labels)
    rows = torch.arange(len(original))
    still = {EXACT: original, AUDIT: original[:1], AUDIT_VIEW: original[:1]}
    report: dict[str, dict[str, float]] = {}
    for level in levels:
        bounds = torch.tensor([tolerance * level, level], dtype=torch.float64)
        low, high = loss_margin(bounds).tolist()  # loss <= tolerance * t, loss >= t
        moved = original.clone()
        moved[rows, labels] += margins.clamp(min=low, max=high) - margins
        forget = cast(dict[str, float], score_from_logits(still, {**still, EXACT: moved},
                                                          gold)["forget"])  # fmt: skip
        row = dict(forget)
        for family, prefix in FAMILY_PREFIXES.items():
            row[f"min_{family}"] = min(v for k, v in forget.items() if k.startswith(prefix))
        report[f"t{round(level * 100):03d}"] = row
    return report


def best_encoder_block(report: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """The rule fixed before any run (rapport section V5.1): the k-NN block of highest mean
    selection precision at phi = 0.25 on the mask stand-ins, against the V5 selection, and on how
    many masks its direction beats the original's runner-ups."""
    masks = [key for key in report["selection"] if key.startswith(MASK_PREFIX)]
    by = report.get("selection_by", {})
    if not masks or not by:
        return {}

    def mask_precision(rows: Mapping[str, Any]) -> float:
        return float(sum(rows[key]["phi25"]["precision"] for key in masks)) / len(masks)

    best = max(by, key=lambda name: mask_precision(by[name]))
    direction = report["direction"]
    wins = sum(direction[key][best] > direction[key]["runner_ups"] for key in masks)
    return {
        "best": best,
        "best_mask_precision": mask_precision(by[best]),
        "views_mask_precision": mask_precision(report["selection"]),
        "best_direction_wins": float(wins),
        "masks": float(len(masks)),
    }


def run_step0(
    original: nn.Module, trained: Tensor, config: UnlearnConfig, batch_size: int
) -> dict[str, dict[str, object]]:
    """Step-0 report on the trained images: the method's views come from `config` (seed,
    target_views, target_view_min_area), the gold stand-ins are the bench's. The class tokens are
    read by hooks during the forward on the exact images; the k-NN folds are those of the V5
    cross-validation (seed, cv_folds), so no image of the run votes for itself."""
    device = model_device(original)
    trained = trained.to(device)
    blocks = [*range(len(original.blocks)), PRE_LOGITS]  # type: ignore[arg-type]
    with capture_cls(original, blocks) as store:
        original_logits = predict(original, trained, batch_size)
    features = {block: torch.cat(parts) for block, parts in store.items()}
    views = [v.cpu() for v in view_logits(original, trained, config, device)]
    gold = gold_view_logits(original, trained, batch_size)
    labels = original_logits.argmax(dim=1)
    folds = anchor_fold_indices(labels, max(config.cv_folds, 2), config.seed, CV_SEED_OFFSET)
    scores, shapes, agreement = encoder_signals(features, original_logits, folds)
    report = step0_report(original_logits, views, gold, scores=scores, shapes=shapes)
    report["encoder"] = {"knn_agreement": agreement, **best_encoder_block(report)}
    report["defects"] = dict(band_defects(original_logits, _view_loss(views, labels), config))
    report["level_ceiling"] = dict(level_ceiling(original_logits, gold, config.band_tolerance))
    return report


def format_step0(report: Mapping[str, Mapping[str, object]]) -> str:
    """One line per stand-in at phi = 0.25: mass, selection precision, direction scores."""
    lines = ["step 0 (V5), phi = 0.25: loss share (oracle) | precision (base) | direction"]
    for key, mass in report["mass"].items():
        m = mass["phi25"]  # type: ignore[index]
        s = report["selection"][key]["phi25"]  # type: ignore[index]
        d = report["direction"][key]
        lines.append(
            f"{key}: {m['loss_share']:.3f} ({m['loss_share_oracle']:.3f}) | "
            f"{s['precision']:.3f} ({s['base_rate']:.3f}) | "
            f"runner-ups {d['runner_ups']:.3f}, views {d['views']:.3f}"  # type: ignore[index]
        )
    encoder, defects = report.get("encoder", {}), report.get("defects", {})
    if "best" in encoder:
        lines.append(
            f"encoder (V5.1): best {encoder['best']} | mask precision "
            f"{encoder['best_mask_precision']:.3f} (views {encoder['views_mask_precision']:.3f}) | "
            f"direction beats runner-ups on {encoder['best_direction_wins']:.0f}/"
            f"{encoder['masks']:.0f} masks"
        )
    if defects:
        lines.append(
            f"defects (V5.1): force top10 {defects['force_top10_share']:.2f}, participation "
            f"{defects['force_participation']:.3f}, held/pushed {defects['held_over_pushed']:.2f} | "
            f"KL T=2 at the floor: mean {defects['floor_kl_mean']:.3f}, "
            f"median {defects['floor_kl_median']:.3f}"
        )
    for name, row in report.get("level_ceiling", {}).items():
        lines.append(
            f"level ceiling (V_Dumb2) {name}: forget on exact images, worst crop "
            f"{row['min_crop']:.3f}, worst mask {row['min_mask']:.3f}"  # type: ignore[index]
        )
    return "\n".join(lines)
