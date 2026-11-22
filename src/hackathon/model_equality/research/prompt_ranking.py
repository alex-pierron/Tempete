"""Which prompts to spend the next ep answers on: single-prompt power against the variants.

For every prompt and every draw available on all 9 ref endpoints, the endpoint-level scores of
`model_level` are computed on that ONE prompt, and the AUC of the four variants (fp8, LoRA 0.3,
0.7, 1.5) against the five disguised copies is measured. A prompt is good when that AUC is high
on every draw, and the ranking is only trusted when it is stable: the report gives the AUC on
draw 0 (selection) and on the other draws (held out), and their Spearman correlation across
prompts.

Run: `python -m hackathon.model_equality.research.prompt_ranking --out DIR`
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS
from hackathon.model_equality.research.model_level import (
    STRONG_INDICES,
    WEAK_INDICES,
    auc,
    endpoint_scores,
)
from hackathon.model_equality.research.small_n_eval import (
    DISGUISED_COPY_INDICES,
    REF_ENDPOINTS,
    load_ref_answers,
)
from hackathon.model_equality.smalln import SmallNData, build_small_n_data

STATISTIC = "styler"
MIN_DRAWS = 2
TOP = 25
VARIANTS = (*STRONG_INDICES, *WEAK_INDICES)


@dataclass(frozen=True)
class PromptRank:
    """Single-prompt power of one prompt."""

    prompt_id: str
    category: str
    draws: int
    auc_draw0: float
    auc_other_draws: float
    auc_all_mean: float
    auc_strong: float
    auc_weak: float
    in_e1: bool


def _spearman(first: list[float], second: list[float]) -> float:
    """Spearman rank correlation (average ranks via argsort of argsort; ties are rare here)."""
    a = np.argsort(np.argsort(first)).astype(np.float64)
    b = np.argsort(np.argsort(second)).astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def rank_prompts(data: SmallNData) -> tuple[list[PromptRank], float]:
    """Rank the prompts that have at least `MIN_DRAWS` complete draws; also the split-draw rho."""
    copies = list(DISGUISED_COPY_INDICES)
    by_prompt: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, (prompt_id, repeat) in enumerate(data.units):
        by_prompt[prompt_id].append((repeat, position))
    ranks: list[PromptRank] = []
    for prompt_id, draws in by_prompt.items():
        if len(draws) < MIN_DRAWS:
            continue
        per_draw: dict[int, tuple[float, float, float]] = {}
        for repeat, position in draws:
            values = endpoint_scores(data, [position])[STATISTIC]
            per_draw[repeat] = (
                auc([values[i] for i in VARIANTS], [values[i] for i in copies]),
                auc([values[i] for i in STRONG_INDICES], [values[i] for i in copies]),
                auc([values[i] for i in WEAK_INDICES], [values[i] for i in copies]))
        first = min(per_draw)
        others = [v[0] for r, v in per_draw.items() if r != first]
        ranks.append(PromptRank(
            prompt_id=prompt_id, category=prompt_id.rsplit("-", 1)[0], draws=len(per_draw),
            auc_draw0=per_draw[first][0], auc_other_draws=float(np.mean(others)),
            auc_all_mean=float(np.mean([v[0] for v in per_draw.values()])),
            auc_strong=float(np.mean([v[1] for v in per_draw.values()])),
            auc_weak=float(np.mean([v[2] for v in per_draw.values()])),
            in_e1=prompt_id in E1_PROMPT_IDS))
    ranks.sort(key=lambda r: -r.auc_all_mean)
    rho = _spearman([r.auc_draw0 for r in ranks], [r.auc_other_draws for r in ranks])
    return ranks, rho


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build_small_n_data(load_ref_answers(args.store_dir), list(REF_ENDPOINTS))
    ranks, rho = rank_prompts(data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prompt_ranking.json").write_text(json.dumps(
        {"statistic": STATISTIC, "split_draw_spearman": rho, "prompts": len(ranks),
         "ranking": [asdict(r) for r in ranks]}, indent=1), encoding="utf-8")
    print(f"{len(ranks)} prompts, split-draw Spearman {rho:+.2f}")
    for r in ranks[:TOP]:
        print(f"{r.prompt_id:18s} draws {r.draws} AUC all {r.auc_all_mean:.2f} (draw0 {r.auc_draw0:.2f}, "
              f"other {r.auc_other_draws:.2f}) strong {r.auc_strong:.2f} weak {r.auc_weak:.2f}"
              f"{'  [E1]' if r.in_e1 else ''}")


if __name__ == "__main__":
    main()
