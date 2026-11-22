"""Build the evaluation context once (fills the on-disk cache) and print the reference numbers."""

from __future__ import annotations

import json
import logging
import time

import torch

from hackathon.common.config import load_settings
from hackathon.unlearning.constants import default_task_dir
from hackathon.unlearning.model import select_device
from hackathon.unlearning.research.context import (
    LOGGER,
    build_context,
    evaluate,
    hypothesis_statistics,
)

RESEARCH_SUBDIR = "research_a"


def main() -> None:
    """Build the context, evaluate the identity model, print timing and hypothesis statistics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = load_settings()
    task_dir = default_task_dir(settings.data_dir)
    device = select_device(None)
    started = time.time()
    context = build_context(task_dir, settings.data_dir.parent, device, task_dir / RESEARCH_SUBDIR)
    LOGGER.info("context built in %.1f s on %s", time.time() - started, device)
    started = time.time()
    result = evaluate(context, context.original, full=True)
    LOGGER.info("full evaluation of the identity in %.1f s", time.time() - started)
    print(json.dumps({"identity": result.metrics}, indent=1)[:3000])
    print(json.dumps(hypothesis_statistics(context), indent=1))
    print("labels histogram max count", torch.bincount(context.labels_a).max().item())


if __name__ == "__main__":
    main()
