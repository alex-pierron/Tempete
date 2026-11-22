"""Estimate the ref server's DEFAULT sampling temperature from its stored answers.

Idea (team D showed ref-08, temperature 1.0, is far less likely under the base model than the
plain ref-01: t = -4.0 on the per-token log-likelihood). The mean per-token log-likelihood of text
SAMPLED at temperature T from a model is a decreasing function of T. Sample the exact base model
(SmolLM2-135M-Instruct) at a grid of temperatures on the same prompts, build the curve
`LL(T)`, and read the temperature at which it reaches the log-likelihood of the stored ref-01
answers. ref-08 (known temperature 1.0) is the check: its estimate should come out near 1.0.

Assumptions, stated: pure temperature sampling (the ref API ignores `top_p` / `top_k`, but the
server may set them internally; an effective temperature absorbs a fixed top-p), same chat
template, same 256-token limit. The estimate is a bootstrap over prompts. Run on the cluster (GPU),
never on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from hackathon.model_equality.zoo.config import MAX_NEW_TOKENS, ZOO_SEED
from hackathon.model_equality.zoo.prompts import load_bundle
from hackathon.model_equality.zoo.sampling import chat_text, generate_texts
from hackathon.model_equality.zoo.scoring import score_replies

FloatArray = npt.NDArray[np.float64]
TEMPERATURE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
GENERATIONS_PER_PROMPT = 4
BOOTSTRAP_REPLICATES = 300
CONFIDENCE_PERCENTILES: tuple[float, float] = (5.0, 95.0)
PLAIN_REF = "ref-01"
TEMPERATURE_REF = "ref-08"
ROUNDING_STEP = 0.05
CALIBRATION_BATCH = 400
ID_PREFIX = "h-arith"
"""Only the holdout arithmetic prompts: the ref answers exist for them."""


@dataclass(frozen=True)
class CellStats:
    """Per-prompt sums of one answer set: `prompt id -> (sum of log-probabilities, tokens)`."""

    per_prompt: dict[str, tuple[float, int]]

    def mean(self, prompts: Sequence[str]) -> float:
        """Token-weighted mean log-likelihood over the listed prompts (with repetition)."""
        total = sum(self.per_prompt[p][0] for p in prompts if p in self.per_prompt)
        tokens = sum(self.per_prompt[p][1] for p in prompts if p in self.per_prompt)
        return total / tokens if tokens else float("nan")


def aggregate(prompt_ids: Sequence[str], scores: Sequence[tuple[float, int]]) -> CellStats:
    """Sum `(log-prob, tokens)` pairs per prompt id."""
    sums: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for prompt_id, (total, tokens) in zip(prompt_ids, scores, strict=True):
        previous = sums[prompt_id]
        sums[prompt_id] = (previous[0] + total, previous[1] + tokens)
    return CellStats(dict(sums))


def invert_curve(temperatures: Sequence[float], values: Sequence[float], target: float
                 ) -> tuple[float, bool]:
    """Temperature at which a decreasing curve reaches `target`; `(value, out_of_range)`.

    Outside the grid the nearest end is returned and flagged. A curve that is not strictly
    decreasing is made monotone by a running minimum first (sampling noise on a flat part).
    """
    temps = np.asarray(temperatures, dtype=np.float64)
    curve = np.minimum.accumulate(np.asarray(values, dtype=np.float64))
    if target >= curve[0]:
        return float(temps[0]), target > curve[0]
    if target <= curve[-1]:
        return float(temps[-1]), target < curve[-1]
    unique_curve, positions = np.unique(curve, return_index=True)
    return float(np.interp(target, unique_curve, temps[positions])), False


def round_to_step(value: float, step: float = ROUNDING_STEP) -> float:
    """Round to the nearest multiple of `step` (the recommended default temperature)."""
    return round(round(value / step) * step, 10)


def estimate_with_interval(
    prompts: Sequence[str], grid: dict[float, CellStats], target_stats: CellStats,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Point estimate and bootstrap interval (over prompts) of the temperature of `target_stats`."""
    temperatures = sorted(grid)

    def estimate(sample: Sequence[str]) -> tuple[float, bool]:
        curve = [grid[t].mean(sample) for t in temperatures]
        return invert_curve(temperatures, curve, target_stats.mean(sample))

    point, out_of_range = estimate(prompts)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [prompts[i] for i in rng.integers(0, len(prompts), size=len(prompts))]
        draws.append(estimate(sample)[0])
    low, high = np.percentile(draws, CONFIDENCE_PERCENTILES)
    return {"temperature": point, "interval_90": [float(low), float(high)],
            "out_of_range": out_of_range, "target_mean_ll": target_stats.mean(prompts)}


def load_ref_answers(path: Path) -> dict[str, list[tuple[str, str]]]:
    """`ref endpoint -> [(prompt id, response)]` for the holdout arithmetic prompts."""
    answers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["id"]).startswith(ID_PREFIX):
            answers[str(row["endpoint"])].append((str(row["id"]), str(row["response"])))
    return dict(answers)


def calibrate(
    model: Any, tokenizer: Any, bundle_path: Path, ref_path: Path, device: torch.device,
) -> dict[str, object]:
    """Build the LL(T) curve and place every ref endpoint on it."""
    bundle = load_bundle(bundle_path)
    items = [p for p in bundle.arithmetic if p.prompt_id.startswith(ID_PREFIX)]
    if not items:
        raise ValueError(f"the bundle has no {ID_PREFIX} prompts")
    prompt_ids = [p.prompt_id for p in items]
    texts = [chat_text(tokenizer, p.text) for p in items]
    users = {p.prompt_id: p.text for p in items}
    grid: dict[float, CellStats] = {}
    for temperature in TEMPERATURE_GRID:
        started = time.time()
        repeated = [(pid, t) for pid, t in zip(prompt_ids, texts, strict=True)
                    for _ in range(GENERATIONS_PER_PROMPT)]
        result = generate_texts(
            model, tokenizer, [t for _, t in repeated], temperature, False, CALIBRATION_BATCH,
            ZOO_SEED + int(temperature * 100), device, max_new_tokens=MAX_NEW_TOKENS)
        pairs = [(users[pid], text) for (pid, _), text in zip(repeated, result.texts, strict=True)]
        scores = score_replies(model, tokenizer, pairs, CALIBRATION_BATCH, device)
        grid[temperature] = aggregate([pid for pid, _ in repeated], scores)
        print(f"T={temperature}: LL {grid[temperature].mean(prompt_ids):.3f} "
              f"({time.time() - started:.0f}s)", flush=True)
    rng = np.random.default_rng(ZOO_SEED)
    reference: dict[str, object] = {}
    for endpoint, answers in sorted(load_ref_answers(ref_path).items()):
        pairs = [(users[pid], text) for pid, text in answers if pid in users]
        scores = score_replies(model, tokenizer, pairs, CALIBRATION_BATCH, device)
        stats = aggregate([pid for pid, _ in answers if pid in users], scores)
        reference[endpoint] = estimate_with_interval(prompt_ids, grid, stats, rng)
        print(f"{endpoint}: effective T {reference[endpoint]}", flush=True)
    plain = reference.get(PLAIN_REF)
    recommended = (round_to_step(float(plain["temperature"]))
                   if isinstance(plain, dict) else None)
    return {
        "grid": [{"temperature": t, "mean_ll": grid[t].mean(prompt_ids),
                  "tokens": sum(v[1] for v in grid[t].per_prompt.values())}
                 for t in sorted(grid)],
        "reference": reference, "recommended_default_temperature": recommended,
        "plain_endpoint": PLAIN_REF, "check_endpoint": TEMPERATURE_REF,
        "assumptions": "pure temperature sampling, SmolLM2 chat template, 256 tokens",
        "generations_per_prompt": GENERATIONS_PER_PROMPT, "prompts": len(prompt_ids),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: needs a GPU model folder (flat), the bundle and the ref holdout answers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--ref-answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    model = model.to(device).eval()
    payload = calibrate(model, tokenizer, Path(args.bundle), Path(args.ref_answers), device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(out)
    print("recommended default temperature:", payload["recommended_default_temperature"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
