"""Build one post-hoc candidate file from a base state (team A3): extrapolation then bias shift.

`theta = theta_original + alpha * (theta_base - theta_original)`, then `head.bias[removed] -= delta`
(exactly the family `a3_surface` scores in closed form). Pure state-dict arithmetic: no forward
pass, a few seconds of CPU; the file is validated by the official checker before it appears.

Usage:
    python -m hackathon.unlearning.research.a3_posthoc --base <state.pt> --alpha 1.4 --delta 4 \
        --out <name.pt>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hackathon.common.config import load_settings
from hackathon.unlearning.constants import MODEL_FILENAME, default_task_dir
from hackathon.unlearning.model import build_model, load_model
from hackathon.unlearning.research.a3_spec import DEFAULT_REMOVED_CLASSES
from hackathon.unlearning.research.phase2 import offset_head_bias
from hackathon.unlearning.research.runner import blend_state
from hackathon.unlearning.submission import save_submission

LOGGER = logging.getLogger("hackathon.unlearning.research")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="state dict at alpha 1")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="subtracted from head.bias of the removed classes; NEGATIVE adds back (alpha 1 = edit an existing file)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    task_dir = default_task_dir(load_settings(env_file=None).data_dir)
    original = load_model(task_dir / MODEL_FILENAME, torch.device("cpu"))
    original_state = {k: v.detach().clone() for k, v in original.state_dict().items()}
    base_state = torch.load(args.base, map_location="cpu", weights_only=True)
    # alpha 1 edits an existing file as it is (no float round trip through the original).
    state = base_state if args.alpha == 1.0 else blend_state(original_state, base_state, args.alpha)
    if args.delta != 0:
        state = offset_head_bias(state, DEFAULT_REMOVED_CLASSES, args.delta)
    model = build_model()
    model.load_state_dict(state)
    report = save_submission(model, args.out)
    LOGGER.info("%s written and validated (%d bytes)", args.out, report.size_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
