"""Post-hoc surface alpha x delta of existing candidate states (team A3), without training.

For a base state (a trained student, the gold_soft state of phase 2 ...) the family
`theta_original + alpha * (theta_base - theta_original)` followed by `head.bias[removed] -= delta`
is the whole space team C probes on the leaderboard. A bias shift moves only the logits of the
removed classes, by exactly `delta`, so ONE forward pass per alpha gives the proxy card of every
delta in closed form (`a3_generate.shift_removed`). This module evaluates that surface for several
bases and writes one JSON line per (base, alpha, delta), with the public-score fit
(`a3_generate.fit_public_score`) next to the raw measurements.

Usage:
    python -m hackathon.unlearning.research.a3_surface --bases gs=<state.pt> g1=<state.pt> \
        --alphas 1.0 1.2 1.4 --deltas 0 4 8 12 --out surface.jsonl --proxy-root <folder>
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
from hackathon.common.errors import HackathonError
from hackathon.unlearning.constants import default_task_dir
from hackathon.unlearning.model import select_device
from hackathon.unlearning.research.a3_generate import (
    build_view_probe,
    card_from_logits,
    collect_logits,
    shift_removed,
)
from hackathon.unlearning.research.a3_spec import DEFAULT_REMOVED_CLASSES
from hackathon.unlearning.research.context import build_context
from hackathon.unlearning.research.runner import RESEARCH_SUBDIR, blend_state

LOGGER = logging.getLogger("hackathon.unlearning.research")
DEFAULT_THREADS = 2
DEFAULT_ALPHAS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
DEFAULT_DELTAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
KEPT_KEYS = (
    "loss_removed_mean",
    "loss_other_mean",
    "loss_std",
    "pseudo_acc_removed",
    "pseudo_acc_other",
    "predicted_removed_share",
    "retain_kl_t2",
    "retain_flips",
    "view_kl_t2",
    "view_keep_removed",
    "view_keep_other",
    "fit_public",
    "forget_tau2",
    "forget_tau3",
    "forget_tau4",
    "proxy_final_mean_tau",
)
"""Card fields written to the surface file (the full card is not needed to choose a probe)."""


class SurfaceError(HackathonError):
    """A base argument could not be used."""


def parse_bases(arguments: list[str]) -> list[tuple[str, Path]]:
    """`NAME=PATH` arguments to (name, path).

    Raises:
        SurfaceError: malformed argument or missing file.
    """
    bases: list[tuple[str, Path]] = []
    for argument in arguments:
        name, separator, raw = argument.partition("=")
        path = Path(raw)
        if not separator or not name or not path.is_file():
            raise SurfaceError(f"expected NAME=EXISTING_FILE, got '{argument}'")
        bases.append((name, path))
    return bases


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", nargs="+", required=True, help="NAME=PATH (state at alpha 1)")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    bases = parse_bases(args.bases)
    removed = DEFAULT_REMOVED_CLASSES
    task_dir = default_task_dir(load_settings(env_file=None).data_dir)
    context = build_context(
        task_dir, args.proxy_root, select_device(None), task_dir / RESEARCH_SUBDIR
    )
    probe = build_view_probe(context, removed)
    original_state = {k: v.detach().cpu().clone() for k, v in context.original.state_dict().items()}
    scratch: nn.Module = copy.deepcopy(context.original).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, path in bases:
        base_state = torch.load(path, map_location="cpu", weights_only=True)
        for alpha in args.alphas:
            scratch.load_state_dict(blend_state(original_state, base_state, alpha))
            logits = collect_logits(context, scratch, probe)
            for delta in args.deltas:
                card = card_from_logits(
                    context, shift_removed(logits, removed, delta), removed, probe
                )
                row = {
                    "base": name,
                    "alpha": alpha,
                    "delta": delta,
                    **{key: card[key] for key in KEPT_KEYS if key in card},
                }
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            LOGGER.info("%s alpha %.2f done", name, alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
