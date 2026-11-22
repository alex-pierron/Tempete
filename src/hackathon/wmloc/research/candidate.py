"""Build the submission candidate of team A (never submitted from here).

Fits the regime switch and the two mask rules of each method on ALL train images, predicts the
1 020 test images from the cached decoder outputs and writes a file validated by the official
validator of `hackathon.wmloc.submission`.

Usage (kit venv):
    python -m hackathon.wmloc.research.candidate names --out-dir /tmp/names
    python -m hackathon.wmloc.research.candidate build --metrics <metrics.json> --out <file.pt>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hackathon.wmloc.constants import GRID_SIZE, KNOWN_METHODS
from hackathon.wmloc.research.crossval import load_features, load_method_data
from hackathon.wmloc.research.data import SPLIT_TEST, SPLIT_TRAIN, Record, load_records
from hackathon.wmloc.research.errors import ResearchError
from hackathon.wmloc.research.extract import PASS_GLOBAL, cache_path
from hackathon.wmloc.research.maps import DEFAULT_SIDES
from hackathon.wmloc.research.rules import RuleError, SwitchRule
from hackathon.wmloc.research.run_cv import full_rules, rect_rules
from hackathon.wmloc.research.scoring import agreement, global_logits
from hackathon.wmloc.submission import save_submission


class CandidateError(ResearchError):
    """The candidate cannot be built (missing cache, unknown rule)."""


def fit_switch(method: str) -> SwitchRule:
    """Switch fitted on every train image of `method`."""
    data = load_method_data(method)
    switch = SwitchRule()
    switch.fit(data.agreements, data.is_full)
    return switch


def test_agreement(record: Record) -> float:
    """Whole-image bit agreement of one test image.

    Raises:
        CandidateError: the global decode is not cached.
    """
    if not cache_path(SPLIT_TEST, record.method, record.name, PASS_GLOBAL).exists():
        raise CandidateError(f"missing global cache for test image {record.name}")
    return float(agreement(global_logits(SPLIT_TEST, record.method, record.name), record.message))


def write_predicted_rect_names(out_dir: Path) -> dict[str, int]:
    """For each method, list the test images the switch sends to the rectangle regime."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for method in KNOWN_METHODS:
        switch = fit_switch(method)
        names = [
            r.name
            for r in load_records(SPLIT_TEST)
            if r.method == method and not switch.is_full(test_agreement(r))
        ]
        (out_dir / f"test_rect_{method}.txt").write_text("\n".join(names) + "\n")
        counts[method] = len(names)
    return counts


def build_candidate(metrics_path: Path, out_path: Path) -> dict[str, Any]:
    """Predict every test image with the best rules named in `metrics_path`; write + validate."""
    chosen = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_records = load_records(SPLIT_TEST)
    masks = np.zeros((len(test_records), GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    routed: dict[str, dict[str, int]] = {}
    for method in KNOWN_METHODS:
        data = load_method_data(method)
        switch = SwitchRule()
        switch.fit(data.agreements, data.is_full)
        full_name = chosen[method]["best_full_rule"]
        rect_name = chosen[method]["best_rect_rule"]
        full_factories, rect_factories = full_rules(data), rect_rules(data)
        if full_name not in full_factories or rect_name not in rect_factories:
            raise CandidateError(f"{method}: rules {full_name}/{rect_name} unavailable")
        full_rule, rect_rule = full_factories[full_name](), rect_factories[rect_name]()
        full_rule.fit(
            [f for f, full in zip(data.features, data.is_full, strict=True) if full],
            data.truth[data.is_full],
        )
        rect_rule.fit(
            [f for f, full in zip(data.features, data.is_full, strict=True) if not full],
            data.truth[~data.is_full],
        )
        fallback_name = (
            "posmap_z/area_matched" if "posmap_z/area_matched" in rect_factories else "zeros"
        )
        fallback_rule = rect_factories[fallback_name]()
        fallback_rule.fit(
            [f for f, full in zip(data.features, data.is_full, strict=True) if not full],
            data.truth[~data.is_full],
        )
        counts = {"full": 0, "rect": 0, "rect_fallback": 0}
        for index, record in enumerate(test_records):
            if record.method != method:
                continue
            feature = load_features(SPLIT_TEST, record, DEFAULT_SIDES, use_scale_space=True)
            go_full = switch.is_full(feature.agreement)
            if go_full:
                masks[index] = full_rule.predict(feature)
                counts["full"] += 1
                continue
            try:
                masks[index] = rect_rule.predict(feature)
                counts["rect"] += 1
            except RuleError:
                # The rule's feature (scale-space probe) was not extracted for this image.
                masks[index] = fallback_rule.predict(feature)
                counts["rect_fallback"] += 1
        routed[method] = counts
    names = [r.name for r in test_records]
    report = save_submission(out_path, names, torch.from_numpy(masks), expected_names=set(names))
    return {
        "path": str(out_path),
        "n_images": len(names),
        "bytes": report.n_bytes,
        "watermarked_fraction": report.watermarked_fraction,
        "routed": routed,
        "train_images_used": SPLIT_TRAIN,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Team A submission candidate (never submits).")
    commands = parser.add_subparsers(dest="command", required=True)
    names = commands.add_parser("names")
    names.add_argument("--out-dir", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--metrics", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "names":
        print(json.dumps(write_predicted_rect_names(args.out_dir)))
    else:
        print(json.dumps(build_candidate(args.metrics, args.out), indent=1))


if __name__ == "__main__":
    main()
