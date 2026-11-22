"""Local PROXY evaluation of an unlearned model (the real score needs data we do not have).

What the server has and we do NOT: the gold models (their loss statistics and averaged
distributions), the true labels, the hidden altered copies, the control and validation sets.
Therefore this module cannot compute the score. It measures, against the ORIGINAL model:

* pseudo-labels = argmax of the original model on each exact forget image. The original model
  was trained on these images, so its argmax is almost surely the true label, but it is
  also exactly the class a gold model is LEAST bound to reproduce. Treat pseudo-label metrics as
  "how far did we move away from the original", not "how close are we to gold".
* loss mean/std of orig and sub on the pseudo-labels, and the shift |dmean| + |dstd|;
* KL(orig || sub) at temperature 2 and the argmax agreement, on the exact forget images and on
  simulated altered copies (flip, crop, jitter, JPEG);
* retain drift = KL(orig || sub) on natural images that are NOT from the task distribution
  (COCO / DIV2K / Kodak resized to 224). It bounds the collateral damage, but cannot give the
  control accuracy, and out-of-distribution inputs are where fine-tuning drifts differently.

Reading guide: a good method shows a clearly non-zero movement on forget (and the same on the
altered copies) with a retain drift near zero. Zero movement = identity baseline. Everything
else (is the movement in the RIGHT direction?) cannot be told locally.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from hackathon.unlearning.alterations import ALTERATIONS, apply_alteration
from hackathon.unlearning.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    SCORING_TEMPERATURE,
)
from hackathon.unlearning.model import predict

FORGET_EXACT_GROUP = "forget_exact"
FORGET_GROUP_PREFIX = "forget_"
RETAIN_GROUP = "retain_proxy"
FINGERPRINT_BYTES = 16
"""Hex characters kept from the cache fingerprint hash."""
TABLE_RULE_CHAR = "-"


@dataclass(frozen=True)
class GroupMetrics:
    """Proxy metrics of one image group (exact forget, one alteration, or retain proxy)."""

    name: str
    count: int
    loss_mean_orig: float
    loss_std_orig: float
    loss_mean_sub: float
    loss_std_sub: float
    loss_shift: float
    kl_orig_sub: float
    pseudo_acc_orig: float
    pseudo_acc_sub: float
    argmax_agreement: float
    confidence_orig: float
    confidence_sub: float


@dataclass(frozen=True)
class ProxyReport:
    """All groups plus the summary numbers used to compare methods."""

    groups: tuple[GroupMetrics, ...]
    pseudo_label_confidence_mean: float
    pseudo_label_confidence_min: float
    forget_kl_mean: float
    retain_kl: float
    movement_ratio: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form."""
        return {
            "groups": [asdict(group) for group in self.groups],
            "pseudo_label_confidence_mean": self.pseudo_label_confidence_mean,
            "pseudo_label_confidence_min": self.pseudo_label_confidence_min,
            "forget_kl_mean": self.forget_kl_mean,
            "retain_kl": self.retain_kl,
            "movement_ratio": self.movement_ratio,
        }


def tempered_kl(
    reference_logits: Tensor, other_logits: Tensor, temperature: float = SCORING_TEMPERATURE
) -> Tensor:
    """Per-sample KL(softmax(ref / T) || softmax(other / T)), shape [N] (float64 for stability)."""
    log_p = torch_f.log_softmax(reference_logits.double() / temperature, dim=1)
    log_q = torch_f.log_softmax(other_logits.double() / temperature, dim=1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=1)


def compute_group_metrics(
    name: str, orig_logits: Tensor, sub_logits: Tensor, pseudo_labels: Tensor
) -> GroupMetrics:
    """Proxy metrics of one group from precomputed logits [N, C] and pseudo-labels [N]."""
    loss_orig = torch_f.cross_entropy(orig_logits.double(), pseudo_labels, reduction="none")
    loss_sub = torch_f.cross_entropy(sub_logits.double(), pseudo_labels, reduction="none")
    mean_orig, std_orig = float(loss_orig.mean()), float(loss_orig.std(unbiased=False))
    mean_sub, std_sub = float(loss_sub.mean()), float(loss_sub.std(unbiased=False))
    orig_argmax = orig_logits.argmax(dim=1)
    sub_argmax = sub_logits.argmax(dim=1)
    return GroupMetrics(
        name=name,
        count=len(orig_logits),
        loss_mean_orig=mean_orig,
        loss_std_orig=std_orig,
        loss_mean_sub=mean_sub,
        loss_std_sub=std_sub,
        loss_shift=abs(mean_sub - mean_orig) + abs(std_sub - std_orig),
        kl_orig_sub=float(tempered_kl(orig_logits, sub_logits).mean()),
        pseudo_acc_orig=float((orig_argmax == pseudo_labels).double().mean()),
        pseudo_acc_sub=float((sub_argmax == pseudo_labels).double().mean()),
        argmax_agreement=float((orig_argmax == sub_argmax).double().mean()),
        confidence_orig=float(torch_f.softmax(orig_logits.double(), dim=1).amax(dim=1).mean()),
        confidence_sub=float(torch_f.softmax(sub_logits.double(), dim=1).amax(dim=1).mean()),
    )


def _iter_groups(
    forget: Tensor, retain: Tensor, variants: Sequence[str], seed: int
) -> Iterator[tuple[str, Tensor]]:
    """Yield (name, images) lazily so only one altered copy of the forget set lives in memory."""
    yield FORGET_EXACT_GROUP, forget
    for offset, variant in enumerate(variants, start=1):
        yield FORGET_GROUP_PREFIX + variant, apply_alteration(variant, forget, seed + offset)
    yield RETAIN_GROUP, retain


def cache_fingerprint(
    forget: Tensor, retain: Tensor, variants: Sequence[str], seed: int, device: torch.device
) -> str:
    """Stable id of everything the cached ORIGINAL logits depend on (data, variants, seed, device)."""
    digest = hashlib.sha256()
    for tensor in (forget[:1], retain[:1]):
        digest.update(tensor.numpy().tobytes())
    digest.update(f"{len(forget)}|{len(retain)}|{'-'.join(variants)}|{seed}|{device.type}".encode())
    return digest.hexdigest()[:FINGERPRINT_BYTES]


def load_logits_cache(path: Path, fingerprint: str) -> dict[str, Tensor]:
    """Cached original logits when the file exists and matches `fingerprint`, else empty."""
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return {}
    logits = payload.get("logits")
    return dict(logits) if isinstance(logits, dict) else {}


def save_logits_cache(path: Path, fingerprint: str, logits: Mapping[str, Tensor]) -> None:
    """Persist original logits (small: N x 100 floats per group) next to their fingerprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "logits": dict(logits)}, path)


def _movement_ratio(forget_kl: float, retain_kl: float) -> float:
    """forget KL / retain KL: inf when only forget moved, nan when nothing moved (identity)."""
    if retain_kl > 0:
        return forget_kl / retain_kl
    return float("inf") if forget_kl > 0 else float("nan")


def evaluate_proxy(
    original: nn.Module,
    submission: nn.Module,
    forget: Tensor,
    retain: Tensor,
    *,
    variants: Sequence[str] = tuple(ALTERATIONS),
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    original_logits_cache: dict[str, Tensor] | None = None,
) -> ProxyReport:
    """Compute the proxy report of `submission` against `original`.

    Args:
        original: The provided model (reference and pseudo-label source).
        submission: The unlearned model.
        forget: uint8 [N, 224, 224, 3] forget images (or a subset).
        retain: uint8 [M, 224, 224, 3] out-of-distribution proxy images.
        variants: Alteration names applied to the forget images.
        seed: Seed of the alterations.
        batch_size: Forward batch size.
        original_logits_cache: Mutable dict reused (and filled) across calls, so comparing several
            methods runs the original model once per group.

    Returns:
        A ProxyReport; the first group is always the exact forget images.
    """
    cache = original_logits_cache if original_logits_cache is not None else {}
    groups: list[GroupMetrics] = []
    pseudo_forget: Tensor | None = None
    confidence_mean = confidence_min = float("nan")
    for name, images in _iter_groups(forget, retain, variants, seed):
        if name not in cache:
            cache[name] = predict(original, images, batch_size)
        orig_logits = cache[name]
        sub_logits = predict(submission, images, batch_size)
        if name == FORGET_EXACT_GROUP:
            pseudo_forget = orig_logits.argmax(dim=1)
            top_probability = torch_f.softmax(orig_logits.double(), dim=1).amax(dim=1)
            confidence_mean, confidence_min = (
                float(top_probability.mean()),
                float(top_probability.min()),
            )
        labels = orig_logits.argmax(dim=1) if name == RETAIN_GROUP else pseudo_forget
        assert labels is not None  # the exact group is yielded first
        groups.append(compute_group_metrics(name, orig_logits, sub_logits, labels))
    forget_groups = [g for g in groups if g.name != RETAIN_GROUP]
    retain_group = groups[-1]
    forget_kl = sum(g.kl_orig_sub for g in forget_groups) / len(forget_groups)
    return ProxyReport(
        groups=tuple(groups),
        pseudo_label_confidence_mean=confidence_mean,
        pseudo_label_confidence_min=confidence_min,
        forget_kl_mean=forget_kl,
        retain_kl=retain_group.kl_orig_sub,
        movement_ratio=_movement_ratio(forget_kl, retain_group.kl_orig_sub),
    )


TABLE_HEADERS = (
    "group", "n", "L_orig", "L_sub", "shift", "KL(o||s)", "acc_o", "acc_s", "agree", "conf_o",
    "conf_s",
)  # fmt: skip


def format_report(report: ProxyReport, title: str = "") -> str:
    """Plain-text table of a ProxyReport (L = pseudo-label cross-entropy as mean+/-std)."""
    rows: list[tuple[str, ...]] = [TABLE_HEADERS]
    for g in report.groups:
        rows.append(
            (
                g.name,
                str(g.count),
                f"{g.loss_mean_orig:.3f}+/-{g.loss_std_orig:.3f}",
                f"{g.loss_mean_sub:.3f}+/-{g.loss_std_sub:.3f}",
                f"{g.loss_shift:.3f}",
                f"{g.kl_orig_sub:.4f}",
                f"{g.pseudo_acc_orig:.3f}",
                f"{g.pseudo_acc_sub:.3f}",
                f"{g.argmax_agreement:.3f}",
                f"{g.confidence_orig:.3f}",
                f"{g.confidence_sub:.3f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(TABLE_HEADERS))]
    lines = [title] if title else []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            lines.append(TABLE_RULE_CHAR * (sum(widths) + 2 * (len(widths) - 1)))
    lines.append(
        f"pseudo-label confidence of the original on forget: mean {report.pseudo_label_confidence_mean:.3f}"
        f", min {report.pseudo_label_confidence_min:.3f}"
    )
    lines.append(
        f"forget KL (mean over groups) {report.forget_kl_mean:.4f} | retain-proxy KL "
        f"{report.retain_kl:.4f} | movement ratio {report.movement_ratio:.2f}"
    )
    lines.append(
        "PROXY ONLY: no gold, no true labels, retain images out of distribution (see docs)."
    )
    return "\n".join(lines)
