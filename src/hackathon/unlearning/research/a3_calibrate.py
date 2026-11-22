"""Proxy calibration (team A3): score existing candidate files with the A3 proxy card.

The A3 proxy (`a3_generate.proxy_card`: team C's H_C gold x OOD retention) is only useful if it
orders candidates like the server does. Candidates that already have a PUBLIC score (phase 2
gold_soft family, team C's bias probes) are evaluated here, one JSON line per file, so the proxy
can be compared with the server on points it was not fitted on. Files without a public score yet
are evaluated too: their proxy is a blind prediction to check when the score arrives.

Usage:
    python -m hackathon.unlearning.research.a3_calibrate --files a.pt b.pt --out calib.jsonl \
        --proxy-root <folder with coco_val_subset ...>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import torch
from torch import nn

from hackathon.common.config import load_settings
from hackathon.unlearning.constants import default_task_dir
from hackathon.unlearning.model import select_device
from hackathon.unlearning.research.a3_generate import proxy_card
from hackathon.unlearning.research.a3_spec import DEFAULT_REMOVED_CLASSES
from hackathon.unlearning.research.context import build_context
from hackathon.unlearning.research.runner import RESEARCH_SUBDIR

LOGGER = logging.getLogger("hackathon.unlearning.research")
DEFAULT_THREADS = 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; appends one JSON line per candidate file to `--out`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    settings = load_settings(env_file=None)
    task_dir = default_task_dir(settings.data_dir)
    device = select_device(None)
    context = build_context(task_dir, args.proxy_root, device, task_dir / RESEARCH_SUBDIR)
    scratch: nn.Module = copy.deepcopy(context.original).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        state = torch.load(path, map_location="cpu", weights_only=True)
        scratch.load_state_dict(state)
        card = proxy_card(context, scratch, DEFAULT_REMOVED_CLASSES)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": path.stem, "card": card}) + "\n")
        LOGGER.info("%s proxy_final_mean_tau %.4f", path.stem, card["proxy_final_mean_tau"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
