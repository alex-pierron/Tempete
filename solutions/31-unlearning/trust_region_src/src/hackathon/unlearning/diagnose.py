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

No training, no file written: `diagnose` prints the report and writes it as JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.bench import gold_view_logits
from hackathon.unlearning.configs import UnlearnConfig
from hackathon.unlearning.constants import SCORING_TEMPERATURE
from hackathon.unlearning.methods import view_logits
from hackathon.unlearning.model import model_device, predict

RELEASES = (0.10, 0.25, 0.50)
"""Release shares phi examined (keys phi10, phi25, phi50)."""
FLOOR = 1e-12


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


def step0_report(
    original_logits: Tensor,
    method_views: Sequence[Tensor],
    gold: Mapping[str, Sequence[Tensor]],
    releases: Sequence[float] = RELEASES,
) -> dict[str, dict[str, object]]:
    """Step-0 report from logits (pure function, float64 on the CPU)."""
    original = original_logits.double().cpu()
    labels = original.argmax(dim=1)
    views = [view.double().cpu() for view in method_views]
    view_loss = torch.stack([torch_f.cross_entropy(v, labels, reduction="none") for v in views])
    view_loss = view_loss.mean(dim=0)
    orig_probs = torch.softmax(original / SCORING_TEMPERATURE, dim=1)
    view_mix = torch.stack([torch.softmax(v / SCORING_TEMPERATURE, dim=1) for v in views]).mean(0)
    shapes = {
        "runner_ups": _wrong_classes(orig_probs, labels),
        "views": _wrong_classes(view_mix, labels),
    }
    report: dict[str, dict[str, object]] = {"mass": {}, "selection": {}, "direction": {}}
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
        left = {
            n: float(((1 - true_mass) * _kl(target_shape, s)).mean()) for n, s in shapes.items()
        }
        report["direction"][key] = {
            name: 1.0 - min(1.0, value / max(reference, FLOOR)) for name, value in left.items()
        }
    return report


def run_step0(
    original: nn.Module, trained: Tensor, config: UnlearnConfig, batch_size: int
) -> dict[str, dict[str, object]]:
    """Step-0 report on the trained images: the method's views come from `config` (seed,
    target_views, target_view_min_area), the gold stand-ins are the bench's."""
    device = model_device(original)
    trained = trained.to(device)
    original_logits = predict(original, trained, batch_size)
    views = [v.cpu() for v in view_logits(original, trained, config, device)]
    gold = gold_view_logits(original, trained, batch_size)
    return step0_report(original_logits, views, gold)


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
    return "\n".join(lines)
