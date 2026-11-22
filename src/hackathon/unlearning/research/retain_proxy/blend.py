"""Candidates `original + alpha * (candidate - original)` from a fine-tuned model (team D).

The same blend the team already uses for `gold_soft` (alpha 1.0 is the candidate itself, above 1
extrapolates past it). Output goes through `save_submission`, i.e. the official validation
(float32, same keys and shapes, no NaN/Inf, size limit) before the file appears under its name.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from hackathon.unlearning.model import build_model, load_state_dict_file
from hackathon.unlearning.research.retain_proxy.blending import BlendError, blend_state_dicts
from hackathon.unlearning.submission import save_submission
from hackathon.unlearning.validate import ModelFactory, ValidationReport


def write_blend(
    original_path: Path,
    candidate_path: Path,
    alpha: float,
    out_path: Path,
    model_factory: ModelFactory = build_model,
) -> ValidationReport:
    """Blend two checkpoints on disk and save the validated result at `out_path`."""
    blended = blend_state_dicts(
        load_state_dict_file(original_path), load_state_dict_file(candidate_path), alpha
    )
    model: nn.Module = model_factory()
    model.load_state_dict(blended, strict=True)
    return save_submission(model, out_path, model_factory)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write one validated blend per `--alpha`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    for alpha in args.alpha:
        out = args.out_dir / f"{args.name}_a{alpha:.2f}.pt"
        report = write_blend(args.original, args.candidate, alpha, out)
        print(f"{out} ({report.size_bytes} bytes, {report.parameter_count} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["BlendError", "blend_state_dicts", "main", "write_blend"]
