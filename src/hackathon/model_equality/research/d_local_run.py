"""Score stored ref replies under the local base model and its fp8 simulations, then apply the frozen criteria.

Usage (repo root, venv of the repo; heavy, 2 threads)::

    python -m hackathon.model_equality.research.d_local_run score --cells 120 \
        --stores artifacts/model_equality/reference_answers.jsonl ... --out /tmp/local_scores.json
    python -m hackathon.model_equality.research.d_local_run analyse --scores /tmp/local_scores.json \
        --truth data/hackathon/hf/model_equality/practice_answers.csv --out /tmp/local_results.json

The scoring function is `d_local.reply_log_probability` (chat template, `add_special_tokens=False`,
float32 on CPU): reuse it unchanged for the large ep families.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.model_equality.research.d_eval import (
    Roles,
    auc_curve_for_tensor,
    build_design,
    center_by_cell,
    roles_from_models,
)
from hackathon.model_equality.research.d_local import (
    DEFAULT_MODEL,
    FP8_VARIANTS,
    VARIANT_BASE,
    build_variants,
    load_tokenizer,
    reply_log_probability,
)
from hackathon.model_equality.research.dataset import load_truth
from hackathon.model_equality.store import load_many

PLAIN = "ref-01"
FP8 = "ref-02"
LORA_WEAK, LORA_MID, LORA_STRONG = "ref-09", "ref-04", "ref-07"
DISGUISES = ("ref-03", "ref-05", "ref-06", "ref-08")
FP8_T_THRESHOLD = 3.0
DISGUISE_T_LIMIT = 3.0
LORA_T_THRESHOLDS = {LORA_WEAK: -2.0, LORA_MID: -3.0, LORA_STRONG: -3.0}
AUC_SIZES = (5, 10, 20, 40)
AUC_TARGET_N = 20
AUC_TARGET = 0.8
TOKENS_T_TARGET = 3.0
REPLICATES = 300
DEFAULT_ENDPOINTS = tuple(f"ref-{i:02d}" for i in range(1, 10))


def score_command(args: argparse.Namespace) -> int:
    """Score every (endpoint, cell) under every variant and write the arrays."""
    answers = load_many([Path(p) for p in args.stores])
    design = build_design(answers, tuple(args.endpoints))
    cells = list(range(args.first, min(args.last if args.last is not None else args.cells, len(design.cells))))
    tokenizer = load_tokenizer(args.model)
    models = build_variants(args.model, (VARIANT_BASE, *args.variants))
    result: dict[str, Any] = {"model": args.model, "cells": [list(design.cells[c]) for c in cells], "endpoints": {}}
    started = time.time()
    wanted = set(args.score_endpoints) if args.score_endpoints else set(design.endpoints)
    for endpoint in design.endpoints:
        if endpoint not in wanted:
            continue
        rows: dict[str, list[float]] = {name: [] for name in models}
        tokens: list[int] = []
        for c in cells:
            prompt, reply = design.prompts[design.cells[c][0]], design.responses[endpoint][c]
            count = 0
            for name, model in models.items():
                total, count = reply_log_probability(model, tokenizer, prompt, reply)
                rows[name].append(total)
            tokens.append(count)
        result["endpoints"][endpoint] = {"tokens": tokens, **rows}
        print(f"{endpoint} scored in {time.time() - started:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    return 0


def paired_t(differences: np.ndarray) -> float:
    """One-sample t of paired differences (0 with no spread)."""
    spread = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return 0.0 if spread == 0.0 else float(differences.mean() / (spread / math.sqrt(len(differences))))


def per_token(sums: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Per-cell mean per-token value (cells without a token count as 0)."""
    return np.where(tokens > 0, sums / np.maximum(tokens, 1), 0.0)


def analyse(scores: dict[str, Any], truth_path: Path) -> dict[str, Any]:
    """Apply the four frozen criteria to scored replies."""
    endpoints = list(scores["endpoints"])
    data = {e: {k: np.asarray(v, dtype=np.float64) for k, v in scores["endpoints"][e].items()} for e in endpoints}
    variants = [v for v in FP8_VARIANTS if v in data[PLAIN]]
    out: dict[str, Any] = {"cells": len(scores["cells"]), "mean_base_ll_per_token": {
        e: float(data[e][VARIANT_BASE].sum() / data[e]["tokens"].sum()) for e in endpoints}}
    # Criterion 1: fp8 likelihood ratio
    fp8: dict[str, Any] = {}
    for variant in variants:
        lam = {e: per_token(data[e][variant] - data[e][VARIANT_BASE], data[e]["tokens"]) for e in endpoints}
        t_fp8 = paired_t(lam[FP8] - lam[PLAIN])
        t_disguise = {e: paired_t(lam[e] - lam[PLAIN]) for e in DISGUISES if e in lam}
        fp8[variant] = {
            "mean_lambda_per_token": {e: float((data[e][variant] - data[e][VARIANT_BASE]).sum() / data[e]["tokens"].sum())
                                      for e in endpoints},
            "t_fp8_vs_plain": t_fp8, "t_disguises": t_disguise,
            "seen": t_fp8 >= FP8_T_THRESHOLD, "confounded_by": [e for e, t in t_disguise.items() if abs(t) >= DISGUISE_T_LIMIT]}
    best = max(variants, key=lambda v: fp8[v]["t_fp8_vs_plain"]) if variants else None
    out["fp8"] = fp8
    out["best_variant"] = best
    # Criterion 2: LoRA base log-likelihood
    base_ll = {e: per_token(data[e][VARIANT_BASE], data[e]["tokens"]) for e in endpoints}
    out["lora"] = {e: {"t_vs_plain": paired_t(base_ll[e] - base_ll[PLAIN]),
                       "seen": paired_t(base_ll[e] - base_ll[PLAIN]) <= LORA_T_THRESHOLDS[e]}
                   for e in (LORA_WEAK, LORA_MID, LORA_STRONG) if e in base_ll}
    out["base_ll_t_disguises"] = {e: paired_t(base_ll[e] - base_ll[PLAIN]) for e in DISGUISES if e in base_ll}
    # Criterion 3: AUC
    truth = load_truth(truth_path)
    roles = roles_from_models(truth.model)
    design_endpoints = tuple(endpoints)
    auc: dict[str, Any] = {}
    if best is not None:
        lam_tensor = np.stack([per_token(data[e][best] - data[e][VARIANT_BASE], data[e]["tokens"]) for e in endpoints])[None]
        auc["fp8_lambda"] = _auc_json(lam_tensor, design_endpoints, roles)
    ll_tensor = np.stack([base_ll[e] for e in endpoints])[None]
    auc["base_ll"] = _auc_json(ll_tensor, design_endpoints, roles)
    out["auc"] = auc
    if best is not None:
        reached = auc["fp8_lambda"][str(AUC_TARGET_N)]["fp8"] >= AUC_TARGET
        out["auc_fp8_n20_target_reached"] = bool(reached)
    # Criterion 4: tokens needed
    if best is not None:
        lam = {e: data[e][best] - data[e][VARIANT_BASE] for e in endpoints}
        diff = lam[FP8] - lam[PLAIN]
        tokens_per_cell = float((data[FP8]["tokens"].mean() + data[PLAIN]["tokens"].mean()) / 2)
        mu = float(diff.mean())
        sigma = float(diff.std(ddof=1))
        out["tokens_needed_for_t3"] = None if mu <= 0 else (TOKENS_T_TARGET * sigma / mu) ** 2 * tokens_per_cell
        out["cells_needed_for_t3"] = None if mu <= 0 else (TOKENS_T_TARGET * sigma / mu) ** 2
    return out


def _auc_json(tensor: np.ndarray, endpoints: Sequence[str], roles: Roles) -> dict[str, dict[str, float]]:
    """AUC at the frozen sizes of the distance on one statistic tensor (1, endpoints, cells)."""
    sizes = tuple(n for n in AUC_SIZES if n <= tensor.shape[2])
    curve = auc_curve_for_tensor(center_by_cell(tensor), endpoints, roles, np.ones(1), sizes, REPLICATES)
    return {str(n): {"fp8": c["fp8"].mean, "lora_weak": c["lora_weak"].mean} for n, c in curve.items()}


def analyse_command(args: argparse.Namespace) -> int:
    """Read the scores, apply the criteria, write and print the result."""
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    result = analyse(scores, Path(args.truth))
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1)[:6000])
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--stores", nargs="+", required=True)
    score.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    score.add_argument("--cells", type=int, default=120)
    score.add_argument("--first", type=int, default=0)
    score.add_argument("--score-endpoints", nargs="+", default=None,
                       help="score only these endpoints (the cells are still those common to --endpoints)")
    score.add_argument("--last", type=int, default=None)
    score.add_argument("--model", default=DEFAULT_MODEL)
    score.add_argument("--variants", nargs="+", default=list(FP8_VARIANTS))
    score.add_argument("--out", required=True)
    analyse_parser = sub.add_parser("analyse")
    analyse_parser.add_argument("--scores", required=True)
    analyse_parser.add_argument("--truth", required=True)
    analyse_parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to `score` or `analyse`."""
    args = parse_args(argv)
    return score_command(args) if args.command == "score" else analyse_command(args)


if __name__ == "__main__":
    sys.exit(main())
