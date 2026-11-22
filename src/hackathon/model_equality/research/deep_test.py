"""Depth instead of breadth: few prompts, many draws, on the free ref endpoints.

Question: with D draws of the same 3 short prompts per endpoint, does an answer-distribution test
tell the variants (fp8, LoRA 0.3, 0.7, 1.5) from the disguised copies of the base model? The
many-prompts design never saw fp8 and LoRA 0.3; this is the untested alternative.

For each endpoint and prompt, the D answers are compared with the pooled answers of the copies
(the other four copies when the endpoint is itself a copy, so the copies' own scores are the null):

* `len`    two-sample Mann-Whitney z on log-length;
* `ends`   two-proportion z on "ends cleanly";
* `first`  chi-square of the first-word distribution, turned into a z.

The endpoint statistic is the sum over prompts and features of (z^2 - 1). AUC of variants against
copies is measured over random sub-samples of D draws.

Run: `python -m hackathon.model_equality.research.deep_test --out DIR`
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.stats import mannwhitneyu

from hackathon.model_equality.features import first_word, tokenize_words
from hackathon.model_equality.prompts.phases import REF_DEEP_PROMPT_IDS
from hackathon.model_equality.research.model_level import STRONG_INDICES, WEAK_INDICES, auc
from hackathon.model_equality.research.small_n_eval import (
    DISGUISED_COPY_INDICES,
    RANDOM_SEED,
    REF_ENDPOINTS,
    load_ref_answers,
)
from hackathon.model_equality.smalln import SENTENCE_ENDS
from hackathon.model_equality.store import Answer

FloatArray = npt.NDArray[np.float64]
DRAW_SIZES: tuple[int, ...] = (5, 10, 20, 30)
RESAMPLES = 60
MIN_PROPORTION_VARIANCE = 1e-9
MAX_FIRST_WORDS = 8


@dataclass(frozen=True)
class DeepRow:
    """Detection at one draw count."""

    draws: int
    auc_strong: float
    auc_strong_sd: float
    auc_weak: float
    auc_weak_sd: float
    auc_all: float


def _log_len(text: str) -> float:
    return math.log1p(len(tokenize_words(text)))


def _ends(text: str) -> float:
    return float(text.strip().endswith(SENTENCE_ENDS))


def _mann_whitney_z(first: list[float], second: list[float]) -> float:
    """Signed z of the Mann-Whitney U (0 when a sample is empty or all values tie)."""
    if not first or not second or len(set(first) | set(second)) < 2:
        return 0.0
    u = float(mannwhitneyu(first, second, alternative="two-sided").statistic)
    mean = len(first) * len(second) / 2.0
    sd = math.sqrt(len(first) * len(second) * (len(first) + len(second) + 1) / 12.0)
    return (u - mean) / sd


def _proportion_z(first: list[float], second: list[float]) -> float:
    """Two-proportion z with a pooled variance (0 when both are constant and equal)."""
    p1, p2 = float(np.mean(first)), float(np.mean(second))
    pooled = (sum(first) + sum(second)) / (len(first) + len(second))
    variance = pooled * (1 - pooled) * (1 / len(first) + 1 / len(second))
    return 0.0 if variance < MIN_PROPORTION_VARIANCE else (p1 - p2) / math.sqrt(variance)


def _first_word_z(first: list[str], second: list[str]) -> float:
    """Chi-square of the first-word counts, as (chi2 - df) / sqrt(2 df)."""
    counts = Counter(first) + Counter(second)
    words = [w for w, _ in counts.most_common(MAX_FIRST_WORDS)]
    df = len(words) - 1
    if df < 1:
        return 0.0
    n1, n2 = len(first), len(second)
    chi2 = 0.0
    for word in words:
        c1, c2 = first.count(word), second.count(word)
        expected_1 = (c1 + c2) * n1 / (n1 + n2)
        expected_2 = (c1 + c2) * n2 / (n1 + n2)
        chi2 += (c1 - expected_1) ** 2 / expected_1 + (c2 - expected_2) ** 2 / expected_2
    return (chi2 - df) / math.sqrt(2.0 * df)


def endpoint_statistic(
    texts: dict[tuple[int, str], list[str]], target: int, pool: list[int],
) -> float:
    """Sum over prompts and features of (z^2 - 1) of `target` against the pooled endpoints."""
    total = 0.0
    for prompt_id in REF_DEEP_PROMPT_IDS:
        mine = texts[(target, prompt_id)]
        others = [t for e in pool for t in texts[(e, prompt_id)]]
        features = (
            _mann_whitney_z([_log_len(t) for t in mine], [_log_len(t) for t in others]),
            _proportion_z([_ends(t) for t in mine], [_ends(t) for t in others]),
            _first_word_z([first_word(t) for t in mine], [first_word(t) for t in others]),
        )
        total += sum(z * z - 1.0 for z in features)
    return total


def _collect(answers: list[Answer]) -> dict[tuple[int, str], list[str]]:
    """Responses per (endpoint index, prompt id), draw order."""
    index = {e: i for i, e in enumerate(REF_ENDPOINTS)}
    table: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for a in answers:
        if a.endpoint in index and a.prompt_id in REF_DEEP_PROMPT_IDS:
            table.setdefault((index[a.endpoint], a.prompt_id), []).append((a.repeat, a.response))
    return {key: [text for _, text in sorted(rows)] for key, rows in table.items()}


def available_draws(texts: dict[tuple[int, str], list[str]]) -> int:
    """Draws available on every endpoint and every deep prompt."""
    return min(len(texts.get((e, p), [])) for e in range(len(REF_ENDPOINTS))
               for p in REF_DEEP_PROMPT_IDS)


def evaluate(texts: dict[tuple[int, str], list[str]]) -> list[DeepRow]:
    """AUC of variants against copies for each draw count that the data supports."""
    rng = np.random.default_rng(RANDOM_SEED)
    copies = list(DISGUISED_COPY_INDICES)
    limit = available_draws(texts)
    rows: list[DeepRow] = []
    for draws in DRAW_SIZES:
        if draws > limit:
            continue
        strong: list[float] = []
        weak: list[float] = []
        every: list[float] = []
        for _ in range(RESAMPLES):
            sample = {
                key: [texts[key][i] for i in rng.choice(len(texts[key]), size=draws, replace=False)]
                for key in texts}
            scores = [endpoint_statistic(sample, e, [c for c in copies if c != e])
                      for e in range(len(REF_ENDPOINTS))]
            strong.append(auc([scores[i] for i in STRONG_INDICES], [scores[i] for i in copies]))
            weak.append(auc([scores[i] for i in WEAK_INDICES], [scores[i] for i in copies]))
            every.append(auc([scores[i] for i in (*STRONG_INDICES, *WEAK_INDICES)],
                             [scores[i] for i in copies]))
        rows.append(DeepRow(draws, float(np.mean(strong)), float(np.std(strong)),
                            float(np.mean(weak)), float(np.std(weak)), float(np.mean(every))))
    return rows


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    texts = _collect(load_ref_answers(args.store_dir))
    rows = evaluate(texts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "deep_test.json").write_text(json.dumps(
        {"prompts": list(REF_DEEP_PROMPT_IDS), "draws_available": available_draws(texts),
         "rows": [asdict(r) for r in rows]}, indent=1), encoding="utf-8")
    print(f"draws available per endpoint and prompt: {available_draws(texts)}")
    for r in rows:
        print(f"D={r.draws:2d} AUC strong {r.auc_strong:.2f}+-{r.auc_strong_sd:.2f} "
              f"weak {r.auc_weak:.2f}+-{r.auc_weak_sd:.2f} all {r.auc_all:.2f}")


if __name__ == "__main__":
    main()
