"""Endpoint-level statistics that isolate strong variants (LoRA) at small N, measured on the ref.

Pairwise clustering (`small_n_eval`) dilutes a variant that only differs on a few prompts. Here
each endpoint gets one score against the rest of the pool, and the one-class rule "the base model
is the bulk, outliers are their own models" gives the labels.

Scores (one answer per prompt, N prompts, higher = more unlike the other endpoints):

* `pair`    mean over the other endpoints of the `ham+len` distance (the clustering statistic).
* `lenz`    mean over prompts of the signed log-length gap to the other endpoints' median, divided
            by their median absolute deviation (floored), clipped: a LoRA rambles where the base
            answers in a word.
* `lenzr`   `lenz` re-expressed as a robust z against the pool of endpoints itself (median and
            MAD), so the cut does not depend on how many endpoints are variants: the rule that
            transfers from 9 ref to 33 ep.
* `ramble`  share of prompts where the answer is more than twice as long as the others' median.
* `ham`     mean over the other endpoints of the character-Hamming dissimilarity alone.

Measured on the 9 ref (strong variants: LoRA 0.7 and 1.5; weak: fp8 and LoRA 0.3; negatives: the
5 disguised copies of the base): AUC of strong versus copies, AUC of weak versus copies, and the
model MCC of the one-class rule with a threshold fitted on a disjoint half of the prompts.
Also measured: choosing the N prompts on one draw and scoring them on another draw, to price
"better prompts" against "more answers".
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.small_n_eval import (
    DISGUISED_COPY_INDICES,
    METHOD_WEIGHTS,
    RANDOM_SEED,
    REF_ENDPOINTS,
    load_ref_answers,
    ref_truth,
)
from hackathon.model_equality.scoring import confusion
from hackathon.model_equality.smalln import (
    SmallNData,
    build_small_n_data,
    cluster_at,
    pair_distance,
)

FloatArray = npt.NDArray[np.float64]
STRONG_INDICES = (3, 6)
WEAK_INDICES = (1, 8)
RAMBLE_FACTOR = 2.0
MAD_FLOOR = 0.25
Z_CLIP = 5.0
SUBSETS = 300
SPLITS = 3
SIZES: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 40)
STATISTICS: tuple[str, ...] = ("pair", "lenz", "lenzr", "lenendsr", "styler", "ramble", "ham")
ENDS_SPREAD_FLOOR = 0.5
PUNCT_SPREAD_FLOOR = 0.03
ROBUST_SPREAD_FLOOR = 0.2
MAD_TO_SIGMA = 1.4826
SELECTION_SIZES: tuple[int, ...] = (5, 10)
CLUSTERED = "lenz+cluster"
FLAGGED_TAU = 0.89
"""Cut between flagged endpoints (the `small_n_eval` model cut at N = 5): two flagged endpoints
stay one model only when their pair distance is below it."""
ALL_STATISTICS: tuple[str, ...] = (*STATISTICS, CLUSTERED)


@dataclass(frozen=True)
class StatRow:
    """Detection quality of one statistic at one N."""

    statistic: str
    n_prompts: int
    auc_strong: float
    auc_strong_sd: float
    auc_weak: float
    mcc_one_class: float
    mcc_one_class_sd: float
    mcc_p10: float
    mcc_p90: float
    threshold: float


def endpoint_scores(data: SmallNData, positions: Sequence[int]) -> dict[str, FloatArray]:
    """Every statistic of every endpoint on the given units."""
    index = list(positions)
    log_length = data.log_length[index]
    size = log_length.shape[1]
    scores: dict[str, FloatArray] = {}
    weights = METHOD_WEIGHTS["ham+len"]
    distance = pair_distance(data, index, weights)
    ham = pair_distance(data, index, {"ham": 1.0})
    off_diagonal = ~np.eye(size, dtype=bool)
    scores["pair"] = np.array([distance[a][off_diagonal[a]].mean() for a in range(size)])
    scores["ham"] = np.array([ham[a][off_diagonal[a]].mean() for a in range(size)])
    gaps = np.zeros(size)
    rambles = np.zeros(size)
    for a in range(size):
        others = np.delete(log_length, a, axis=1)
        median = np.median(others, axis=1)
        spread = np.maximum(np.median(np.abs(others - median[:, None]), axis=1), MAD_FLOOR)
        deviation = log_length[:, a] - median
        gaps[a] = np.clip(deviation / spread, -Z_CLIP, Z_CLIP).mean()
        rambles[a] = float((deviation > np.log(RAMBLE_FACTOR)).mean())
    scores["lenz"] = gaps
    ends_gap = _relative_gap(data.ends_cleanly[index], ENDS_SPREAD_FLOOR)
    punct_gap = _relative_gap(data.punctuation[index], PUNCT_SPREAD_FLOOR)
    scores["lenendsr"] = _robust(np.maximum(gaps, 0.0) + np.abs(ends_gap))
    scores["styler"] = _robust(np.maximum(gaps, 0.0) + np.abs(ends_gap) + np.abs(punct_gap))
    centre = float(np.median(gaps))
    spread = max(MAD_TO_SIGMA * float(np.median(np.abs(gaps - centre))), ROBUST_SPREAD_FLOOR)
    scores["lenzr"] = (gaps - centre) / spread
    scores["ramble"] = rambles
    return scores


def _relative_gap(values: FloatArray, floor: float) -> FloatArray:
    """Per endpoint: mean over prompts of the clipped gap to the others' median, in spreads."""
    size = values.shape[1]
    out = np.zeros(size)
    for a in range(size):
        others = np.delete(values, a, axis=1)
        median = np.median(others, axis=1)
        spread = np.maximum(np.median(np.abs(others - median[:, None])), floor)
        out[a] = np.clip((values[:, a] - median) / spread, -Z_CLIP, Z_CLIP).mean()
    return out


def _robust(scores: FloatArray) -> FloatArray:
    """Robust z of a score against the pool of endpoints (median and floored MAD)."""
    centre = float(np.median(scores))
    spread = max(MAD_TO_SIGMA * float(np.median(np.abs(scores - centre))), ROBUST_SPREAD_FLOOR)
    return np.asarray((scores - centre) / spread, dtype=np.float64)


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Probability that a positive scores above a negative (ties count half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return float(wins / (len(positive) * len(negative)))


def one_class_mcc(scores: FloatArray, threshold: float) -> float:
    """Model MCC of "endpoints below the threshold form the base model, the rest are alone"."""
    pairs, truth = ref_truth()
    flagged = scores > threshold
    guess = [bool((not flagged[i]) and (not flagged[j])) for i, j in pairs]
    return confusion(truth, guess).mcc


def flagged_labels(
    scores: FloatArray, distance: FloatArray, threshold: float, tau: float = FLAGGED_TAU,
) -> npt.NDArray[np.int64]:
    """Base model = unflagged endpoints (label 0); flagged ones cluster among themselves."""
    labels = np.zeros(len(scores), dtype=np.int64)
    flagged = np.flatnonzero(scores > threshold)
    if len(flagged) == 1:
        labels[flagged] = 1
    elif len(flagged) > 1:
        sub = distance[np.ix_(flagged, flagged)]
        labels[flagged] = cluster_at(sub, tau) + 1
    return labels


def clustered_mcc(labels: npt.NDArray[np.int64]) -> float:
    """Model MCC of a labelling against the ref truth."""
    pairs, truth = ref_truth()
    return confusion(truth, [bool(labels[i] == labels[j]) for i, j in pairs]).mcc


def _random_sets(
    prompts: Sequence[str], by_prompt: dict[str, list[int]], size: int, count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Random sets of `size` prompts with one random draw each (positions in the data)."""
    out: list[list[int]] = []
    for _ in range(count):
        chosen = rng.choice(len(prompts), size=size, replace=False)
        out.append([int(rng.choice(by_prompt[prompts[c]])) for c in chosen])
    return out


def evaluate(data: SmallNData) -> list[StatRow]:
    """AUC and model MCC of every statistic at every N (threshold fitted on other prompts)."""
    rng = np.random.default_rng(RANDOM_SEED)
    prompts = sorted({unit[0] for unit in data.units})
    by_prompt = {p: [k for k, u in enumerate(data.units) if u[0] == p] for p in prompts}
    copies = list(DISGUISED_COPY_INDICES)
    weights = METHOD_WEIGHTS["ham+len"]
    rows: list[StatRow] = []
    for size in SIZES:
        per_stat: dict[str, dict[str, list[float]]] = {
            s: {"strong": [], "weak": [], "mcc": [], "tau": []} for s in ALL_STATISTICS}
        for _ in range(SPLITS):
            order = rng.permutation(len(prompts))
            half = len(prompts) // 2
            cal = [prompts[i] for i in order[:half]]
            tst = [prompts[i] for i in order[half:2 * half]]
            if size > half:
                continue
            cal_sets = _random_sets(cal, by_prompt, size, SUBSETS // 3, rng)
            tst_sets = _random_sets(tst, by_prompt, size, SUBSETS // 3, rng)
            cal_scores = [endpoint_scores(data, s) for s in cal_sets]
            tst_scores = [endpoint_scores(data, s) for s in tst_sets]
            cal_dist = [pair_distance(data, s, weights) for s in cal_sets]
            tst_dist = [pair_distance(data, s, weights) for s in tst_sets]
            for stat in ALL_STATISTICS:
                base = "lenz" if stat == CLUSTERED else stat
                grid = np.quantile(np.concatenate([s[base] for s in cal_scores]),
                                   np.linspace(0.3, 0.95, 28))

                def mcc_at(
                    scores: FloatArray, dist: FloatArray, threshold: float, stat: str = stat
                ) -> float:
                    if stat == CLUSTERED:
                        return clustered_mcc(flagged_labels(scores, dist, threshold))
                    return one_class_mcc(scores, threshold)

                curve = [np.mean([mcc_at(s[base], d, float(t))
                                  for s, d in zip(cal_scores, cal_dist, strict=True)])
                         for t in grid]
                best = float(grid[int(np.argmax(curve))])
                per_stat[stat]["tau"].append(best)
                for s, d in zip(tst_scores, tst_dist, strict=True):
                    values = s[base]
                    per_stat[stat]["strong"].append(
                        auc([values[i] for i in STRONG_INDICES], [values[i] for i in copies]))
                    per_stat[stat]["weak"].append(
                        auc([values[i] for i in WEAK_INDICES], [values[i] for i in copies]))
                    per_stat[stat]["mcc"].append(mcc_at(values, d, best))
        for stat in ALL_STATISTICS:
            if not per_stat[stat]["strong"]:
                continue
            mcc = per_stat[stat]["mcc"]
            rows.append(StatRow(
                statistic=stat, n_prompts=size,
                auc_strong=float(np.mean(per_stat[stat]["strong"])),
                auc_strong_sd=float(np.std(per_stat[stat]["strong"])),
                auc_weak=float(np.mean(per_stat[stat]["weak"])),
                mcc_one_class=float(np.mean(mcc)), mcc_one_class_sd=float(np.std(mcc)),
                mcc_p10=float(np.percentile(mcc, 10)), mcc_p90=float(np.percentile(mcc, 90)),
                threshold=float(np.mean(per_stat[stat]["tau"]))))
    return rows


def selection_gain(data: SmallNData) -> list[dict[str, float]]:
    """Choose the N prompts on draw 0, score them on draw 1: top-N versus random-N (statistic `lenz`).

    Same prompts, independent draws: the selection effect without the prompt shift. The measure
    is the AUC of strong variants versus copies on draw 1.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    copies = list(DISGUISED_COPY_INDICES)
    draw0 = {u[0]: k for k, u in enumerate(data.units) if u[1] == 0}
    draw1 = {u[0]: k for k, u in enumerate(data.units) if u[1] == 1}
    prompts = sorted(set(draw0) & set(draw1))
    quality: dict[str, float] = {}
    for prompt in prompts:
        values = endpoint_scores(data, [draw0[prompt]])["pair"]
        quality[prompt] = float(np.mean([values[i] for i in STRONG_INDICES])
                                - np.mean([values[i] for i in copies]))
    ranking = sorted(prompts, key=lambda p: -quality[p])
    results: list[dict[str, float]] = []
    for size in SELECTION_SIZES:
        top = endpoint_scores(data, [draw1[p] for p in ranking[:size]])["lenz"]
        top_auc = auc([top[i] for i in STRONG_INDICES], [top[i] for i in copies])
        random_aucs = []
        for chosen in (rng.choice(len(prompts), size=size, replace=False) for _ in range(SUBSETS)):
            values = endpoint_scores(data, [draw1[prompts[c]] for c in chosen])["lenz"]
            random_aucs.append(auc([values[i] for i in STRONG_INDICES], [values[i] for i in copies]))
        results.append({"n_prompts": float(size), "auc_top_selected": top_auc,
                        "auc_random": float(np.mean(random_aucs)),
                        "auc_random_sd": float(np.std(random_aucs))})
    return results


def main() -> None:
    """CLI: evaluate on the ref stores and write `model_level.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = build_small_n_data(load_ref_answers(args.store_dir), list(REF_ENDPOINTS))
    rows = evaluate(data)
    gains = selection_gain(data)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"prompts": len({u[0] for u in data.units}), "units": len(data.units),
               "rows": [asdict(r) for r in rows], "selection_gain": gains}
    (args.out / "model_level.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r.statistic:7s} N={r.n_prompts:2d} AUC strong {r.auc_strong:.2f}+-{r.auc_strong_sd:.2f} "
              f"weak {r.auc_weak:.2f}  one-class MCC {r.mcc_one_class:+.2f}+-{r.mcc_one_class_sd:.2f}")
    for g in gains:
        print("selection", g)


if __name__ == "__main__":
    main()
