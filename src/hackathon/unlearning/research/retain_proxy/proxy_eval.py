"""Retain-side drift of ANY candidate on in-distribution generated images (a control-set stand-in).

The server's retain score is `clip(acc / acc_orig, 0, 1) / (1 + KL(orig || sub))` at T = 2 on a
hidden control set of the training distribution. The forget images are a poor stand-in for it (the
original has memorised them), so team members measured KL on forget images of the retained classes
and got 1.4 to 2.1. The generated proxy images are unseen by the original AND in distribution: the
KL here is the honest estimate of the retain term.

The held-out split is the one `finetune.ProxyData.hold_out` uses (seeded permutation, first
`fraction`), so fine-tuning variants never train on the images a candidate is scored on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from hackathon.common.errors import HackathonError
from hackathon.unlearning.model import build_model, load_state_dict_file, predict
from hackathon.unlearning.research.retain_proxy.finetune import (
    HoldoutStatistics,
    holdout_statistics,
    softmax_np,
)

DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_SEED = 0
PREDICT_BATCH = 256


class ProxyEvalError(HackathonError):
    """The dataset or a candidate cannot be evaluated."""


def holdout_indices(count: int, fraction: float, seed: int) -> Tensor:
    """Rows of the held-out split: first `int(count * fraction)` of a seeded permutation."""
    held = int(count * fraction)
    if held == 0:
        raise ProxyEvalError(f"A holdout of {fraction:.2f} of {count} images is empty.")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return order[:held]


def load_holdout(dataset_path: Path, fraction: float, seed: int) -> tuple[Tensor, np.ndarray]:
    """(uint8 images [n, 224, 224, 3], labels [n]) of the held-out proxy images."""
    if not dataset_path.is_file():
        raise ProxyEvalError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path) as archive:
        images, labels = archive["images"], archive["labels"]
    rows = holdout_indices(len(labels), fraction, seed).numpy()
    return torch.from_numpy(images[rows]), labels[rows]


def evaluate_candidates(
    original_path: Path,
    candidates: Mapping[str, Path],
    dataset_path: Path,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    model_factory: Callable[[], nn.Module] = build_model,
    min_original_confidence: float = 0.0,
) -> dict[str, HoldoutStatistics]:
    """KL(orig || candidate) at T = 2, accuracy, confidence and the retain estimate per candidate.

    `min_original_confidence` keeps only the images the original answers at least that surely:
    the generated images come from another generator, and the surest ones are the closest to the
    real training distribution (a second, less pessimistic estimate).
    """
    images, labels = load_holdout(dataset_path, fraction, seed)

    def logits_of(path: Path) -> np.ndarray:
        model = model_factory()
        model.load_state_dict(load_state_dict_file(path), strict=True)
        model = model.to(device).eval()
        with torch.no_grad():
            return predict(model, images, PREDICT_BATCH).numpy()

    original_logits = logits_of(original_path)
    keep = softmax_np(original_logits).max(axis=1) >= min_original_confidence
    if not keep.any():
        raise ProxyEvalError(f"No holdout image reaches confidence {min_original_confidence}.")
    return {
        name: holdout_statistics(logits_of(path)[keep], original_logits[keep], labels[keep])
        for name, path in candidates.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `--original model.pt --dataset v1_proxy.npz --out stats.json NAME=PATH ...`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-original-confidence", type=float, default=0.0)
    parser.add_argument("pairs", nargs="+", metavar="NAME=PATH")
    args = parser.parse_args(argv)
    candidates: dict[str, Path] = {}
    for pair in args.pairs:
        name, separator, path = pair.partition("=")
        if not separator or not path:
            raise ProxyEvalError(f"Expected NAME=PATH, got {pair!r}.")
        candidates[name] = Path(path)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    stats = evaluate_candidates(
        args.original,
        candidates,
        args.dataset,
        fraction=args.fraction,
        seed=args.seed,
        device=device,
        min_original_confidence=args.min_original_confidence,
    )
    payload = {name: json.loads(value.model_dump_json()) for name, value in stats.items()}
    args.out.write_text(json.dumps(payload, indent=1))
    for name, value in payload.items():
        print(name, {key: round(number, 4) for key, number in value.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
