"""What do more long answers buy? Measured on the ref endpoints (same sampling regime as the ep).

The ref are 9 endpoints of one small model: 5 copies with disguises (the "S" model: plain, system
prompt, persona, watermark, temperature 1.0), fp8, and three LoRA strengths. For the prompts whose
answers are long (more than `LONG_WORDS` words on average) every endpoint has several draws. A
"unit" is one (prompt, draw). With N units per endpoint, each endpoint gets a style score: the mean
absolute robust z of five answer features (words, lines, bold marks, digit share, word length)
against the pool of the other endpoints. The AUC separates variants from copies of S.

This is a measurement on 16 prompts of one small model, so it prices the NEXT long prompts of the
ep only roughly: the ep models are larger and different. It is nevertheless the only place where
the model identity is known.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
LONG_WORDS = 60
REF_TRUTH: dict[str, str] = {
    "ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S", "ref-06": "S",
    "ref-07": "T4", "ref-08": "S", "ref-09": "T2",
}
STRONG = ("ref-04", "ref-07")
WEAK = ("ref-02", "ref-09")
COPIES = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
MAD_FLOOR = 1e-6
MAD_TO_SIGMA = 1.4826
STORE_FILES: tuple[str, ...] = (
    "reference_answers.jsonl", "reference_answers_b.jsonl", "reference_answers_m1.jsonl",
)


def answer_features(text: str) -> list[float]:
    """Five cheap style features of one answer."""
    words = text.split()
    count = max(len(words), 1)
    return [
        float(np.log1p(len(words))),
        float(np.log1p(text.count("\n"))),
        float(text.count("**")),
        sum(ch.isdigit() for ch in text) / max(len(text), 1),
        sum(len(w) for w in words) / count,
    ]


def load_long_units(store_dir: Path) -> dict[str, list[list[float]]]:
    """`endpoint -> feature vectors of its long-prompt answers`, units aligned across endpoints."""
    by_unit: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    lengths: dict[str, list[int]] = defaultdict(list)
    for name in STORE_FILES:
        path = store_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_unit[(row["id"], int(row["repeat"]))][row["endpoint"]] = row["response"]
                lengths[row["id"]].append(len(row["response"].split()))
    long_ids = {pid for pid, values in lengths.items() if np.mean(values) > LONG_WORDS}
    units = [u for u, v in sorted(by_unit.items()) if u[0] in long_ids and len(v) == len(REF_TRUTH)]
    return {
        endpoint: [answer_features(by_unit[u][endpoint]) for u in units] for endpoint in REF_TRUTH
    }


def style_scores(features: dict[str, FloatArray]) -> dict[str, float]:
    """Mean absolute robust z of each endpoint's mean features against the pool of the others."""
    endpoints = sorted(features)
    means = np.stack([features[e].mean(axis=0) for e in endpoints])
    out: dict[str, float] = {}
    for position, endpoint in enumerate(endpoints):
        others = np.delete(means, position, axis=0)
        centre = np.median(others, axis=0)
        spread = np.maximum(MAD_TO_SIGMA * np.median(np.abs(others - centre), axis=0), MAD_FLOOR)
        out[endpoint] = float(np.mean(np.abs(means[position] - centre) / spread))
    return out


def auc(positive: list[float], negative: list[float]) -> float:
    """Probability that a positive outscores a negative (ties count half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return float(wins / (len(positive) * len(negative)))


def power_curve(
    units: dict[str, list[list[float]]], sizes: tuple[int, ...], rng: np.random.Generator,
    draws: int = 200,
) -> dict[int, dict[str, float]]:
    """AUC (strong, weak variants vs the S copies) of the style score for N units per endpoint."""
    total = len(next(iter(units.values())))
    arrays = {e: np.asarray(v, dtype=np.float64) for e, v in units.items()}
    out: dict[int, dict[str, float]] = {}
    for size in sizes:
        if size > total:
            continue
        strong, weak = [], []
        for _ in range(draws):
            chosen = rng.choice(total, size=size, replace=False)
            scores = style_scores({e: a[chosen] for e, a in arrays.items()})
            strong.append(auc([scores[e] for e in STRONG], [scores[e] for e in COPIES]))
            weak.append(auc([scores[e] for e in WEAK], [scores[e] for e in COPIES]))
        out[size] = {
            "strong": float(np.mean(strong)), "strong_sd": float(np.std(strong)),
            "weak": float(np.mean(weak)), "weak_sd": float(np.std(weak)),
        }
    return out
