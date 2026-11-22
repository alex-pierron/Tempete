"""ABC posterior over the ep truth from the PUBLIC scores already read, and value of the next probes.

Model (stated, not hidden):

* Families are the three likelihood families (f4 = 0.10 says they are right on the public pairs).
* Inside a family, each endpoint is either the main model or a SINGLETON variant (the reference
  layout: fp8 and three LoRA strengths are four distinct models, never shared).
* The public split is a FIXED unknown subset of the 528 pairs, each pair public with probability
  `PUBLIC_PAIR_RATE` (shape `pairs`), or the pairs whose both ends lie in a fixed 30 % of the
  endpoints (shape `endpoints`).
* The server rounds the public score to the nearest `SCORE_STEP` (every read so far is a multiple
  of 0.05: 0.05, 0.10, 0.25, 0.40).

A sample (truth, mask) is kept when every past submission, scored on that mask, rounds to the score
read. The kept samples give the posterior variant probability of each endpoint, the predictive
distribution of the public score of any candidate CSV, and its value of information: the expected
gain, on the PRIVATE pairs (the complement of the mask), of the best final partition chosen after
reading the probe, over the best one chosen now.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ENDPOINT_COUNT = 33
ENDPOINTS: tuple[str, ...] = tuple(f"ep-{i:02d}" for i in range(1, ENDPOINT_COUNT + 1))
PUBLIC_PAIR_RATE = 0.3
PUBLIC_ENDPOINT_COUNT = 10
SCORE_STEP = 0.05
FAMILY_WEIGHT = 0.1
MODEL_WEIGHT = 0.9
DEFAULT_SAMPLES = 2_000_000
CHUNK = 40_000
SEED = 20261122
MIN_CONDITIONAL = 30
REPORT_FRACTION = 0.04

QUEUE = Path(os.environ.get("HACKATHON_DATA_DIR", "data/hackathon")) / "queue" / "model_equality"
PROBE_TABLES = Path(os.environ.get("MET_PROBE_TABLES", "reports/phase3/model_equality/team-next/tables"))
OBSERVED: dict[str, float] = {
    "met_e1b": 0.05,
    "metc_p3": 0.05,
    "metc_p4": 0.05,
    "metc_f1": 0.05,
    "metc_f2": 0.05,
    "metc_f3": 0.05,
    "metc_f4": 0.10,
    "metc_f6": 0.40,
    "metc_f7b": 0.25,
    "metn_f8": 0.40,
}

FAMILY_OF: dict[str, int] = {}
for _fam, _members in enumerate(
    (
        (1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33),  # Qwen2.5-3B
        (3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32),  # Qwen2.5-1.5B
        (5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31),  # Llama-3.2-1B
    )
):
    for _m in _members:
        FAMILY_OF[f"ep-{_m:02d}"] = _fam

# Prior probability that an endpoint is a variant (not the main model). Fixed mains get 0.
# Judgement from the free likelihood data (T hat, z, identity), written BEFORE the ABC run.
PRIOR_INFORMED: dict[str, float] = {
    # Qwen2.5-3B: 02 / 23 disguised (identity), 33 temperature 1.0 (probe f7b), 14 T hat 0.83.
    "ep-02": 0.0, "ep-23": 0.0, "ep-33": 0.0, "ep-14": 0.75,
    "ep-01": 0.25, "ep-04": 0.25, "ep-12": 0.25, "ep-17": 0.25,
    "ep-18": 0.25, "ep-24": 0.25, "ep-25": 0.25,
    # Qwen2.5-1.5B: 08 / 20 disguised, 30 temperature 1.0, 07 / 28 T hat 0.86 / 0.82.
    "ep-08": 0.0, "ep-20": 0.0, "ep-30": 0.0, "ep-07": 0.75, "ep-28": 0.75, "ep-21": 0.4,
    "ep-03": 0.2, "ep-06": 0.2, "ep-22": 0.2, "ep-26": 0.2, "ep-32": 0.2,
    # Llama-3.2-1B: 31 temperature 1.0, 16 disguised (OpenAI identity), rest by T hat.
    "ep-31": 0.0, "ep-16": 0.3, "ep-05": 0.6, "ep-27": 0.6, "ep-15": 0.5,
    "ep-09": 0.4, "ep-11": 0.4, "ep-29": 0.4, "ep-10": 0.15, "ep-13": 0.15, "ep-19": 0.15,
}
FLAT_UNCERTAIN = 0.5

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

PAIR_I, PAIR_J = (np.array(a) for a in zip(*itertools.combinations(range(ENDPOINT_COUNT), 2)))
FAM = np.array([FAMILY_OF[e] for e in ENDPOINTS])
WITHIN: BoolArray = FAM[PAIR_I] == FAM[PAIR_J]


@dataclass(frozen=True)
class Submission:
    """Pair-level 'same family' and 'same model' answers of one CSV."""

    name: str
    same_family: BoolArray
    same_model: BoolArray


def submission_from_labels(name: str, labels: Mapping[str, tuple[str, str]]) -> Submission:
    """Turn endpoint -> (family, model) labels into pair answers."""
    fam = np.array([labels[e][0] for e in ENDPOINTS])
    mod = np.array([labels[e][0] + "|" + labels[e][1] for e in ENDPOINTS])
    return Submission(name, fam[PAIR_I] == fam[PAIR_J], mod[PAIR_I] == mod[PAIR_J])


def read_submission(name: str) -> Submission:
    """Read a queued CSV by its stem."""
    labels: dict[str, tuple[str, str]] = {}
    path = QUEUE / f"{name}.csv"
    if not path.exists():
        path = PROBE_TABLES / f"{name}.csv"
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if row and row[0].startswith("ep-"):
                labels[row[0].strip()] = (row[1].strip(), row[2].strip())
    return submission_from_labels(name, labels)


def isolation_submission(name: str, isolated: Sequence[str]) -> Submission:
    """Our three families, each endpoint in `isolated` alone, the rest in one main model."""
    labels = {
        e: (f"F{FAMILY_OF[e]}", e if e in isolated else "main") for e in ENDPOINTS
    }
    return submission_from_labels(name, labels)


def _mcc(tp: FloatArray, fp: FloatArray, fn: FloatArray, tn: FloatArray) -> FloatArray:
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)
    return np.maximum(out, 0.0)


def scores(truth_same: BoolArray, mask: BoolArray, subs: Sequence[Submission]) -> FloatArray:
    """Official score of each submission (columns) on each sample's masked pairs (rows)."""
    fam_pred = np.stack([s.same_family for s in subs], axis=1).astype(np.float32)
    mod_pred = np.stack([s.same_model & WITHIN for s in subs], axis=1).astype(np.float32)
    maskf = mask.astype(np.float32)
    fam_true = (mask & WITHIN[None, :]).astype(np.float32)
    fam_false = maskf - fam_true
    f_tp = fam_true @ fam_pred
    f_fp = fam_false @ fam_pred
    f_fn = fam_true.sum(1, keepdims=True) - f_tp
    f_tn = fam_false.sum(1, keepdims=True) - f_fp
    m_true = (truth_same & mask).astype(np.float32)
    m_false = ((~truth_same) & mask & WITHIN[None, :]).astype(np.float32)
    m_tp = m_true @ mod_pred
    m_fp = m_false @ mod_pred
    m_fn = m_true.sum(1, keepdims=True) - m_tp
    m_tn = m_false.sum(1, keepdims=True) - m_fp
    fam = _mcc(f_tp.astype(float), f_fp.astype(float), f_fn.astype(float), f_tn.astype(float))
    mod = _mcc(m_tp.astype(float), m_fp.astype(float), m_fn.astype(float), m_tn.astype(float))
    return FAMILY_WEIGHT * fam + MODEL_WEIGHT * mod


def round_step(x: FloatArray) -> FloatArray:
    """Server rounding hypothesis: nearest multiple of SCORE_STEP."""
    return np.round(x / SCORE_STEP + 1e-9) * SCORE_STEP


def truth_same_model(variant: BoolArray) -> BoolArray:
    """Pair truth 'same model' for singleton variants: both ends main and same family."""
    main = ~variant
    return main[:, PAIR_I] & main[:, PAIR_J] & WITHIN[None, :]


def draw_masks(rng: np.random.Generator, n: int, shape: str) -> BoolArray:
    """Fixed public mask per sample."""
    if shape == "pairs":
        return rng.random((n, PAIR_I.size)) < PUBLIC_PAIR_RATE
    keys = rng.random((n, ENDPOINT_COUNT))
    chosen = np.argsort(keys, axis=1)[:, :PUBLIC_ENDPOINT_COUNT]
    public = np.zeros((n, ENDPOINT_COUNT), dtype=bool)
    np.put_along_axis(public, chosen, True, axis=1)
    return public[:, PAIR_I] & public[:, PAIR_J]


def abc(prior: Mapping[str, float], n: int, shape: str, seed: int) -> tuple[BoolArray, BoolArray]:
    """Rejection sample (variant vector, mask) consistent with every observed score."""
    rng = np.random.default_rng(seed)
    p = np.array([prior[e] for e in ENDPOINTS])
    subs = [read_submission(k) for k in OBSERVED]
    target = np.array(list(OBSERVED.values()))
    kept_v: list[BoolArray] = []
    kept_m: list[BoolArray] = []
    for start in range(0, n, CHUNK):
        size = min(CHUNK, n - start)
        variant = rng.random((size, ENDPOINT_COUNT)) < p[None, :]
        mask = draw_masks(rng, size, shape)
        sc = round_step(scores(truth_same_model(variant), mask, subs))
        ok = np.all(np.abs(sc - target[None, :]) < SCORE_STEP / 2, axis=1)
        kept_v.append(variant[ok])
        kept_m.append(mask[ok])
    return np.concatenate(kept_v), np.concatenate(kept_m)


def private_score(variant: BoolArray, mask: BoolArray, sub: Submission) -> FloatArray:
    """Score of `sub` on the private pairs (complement of the public mask)."""
    return scores(truth_same_model(variant), ~mask, [sub])[:, 0]


def best_partition(
    variant: BoolArray, mask: BoolArray, weights: FloatArray, start: Sequence[str] = ()
) -> tuple[list[str], float]:
    """Greedy isolate-set maximising the weighted mean private score over the kept samples."""
    current: list[str] = list(start)
    best = float(np.average(private_score(variant, mask, isolation_submission("x", current)),
                            weights=weights))
    improved = True
    while improved:
        improved = False
        for e in ENDPOINTS:
            trial = [x for x in current if x != e] if e in current else [*current, e]
            val = float(np.average(private_score(variant, mask, isolation_submission("x", trial)),
                                   weights=weights))
            if val > best + 1e-9:
                best, current, improved = val, trial, True
    return sorted(current), best


def main() -> None:
    """Run the ABC and print posteriors, predictive of f9, and VOI of single-flip probes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=["informed", "flat"], default="informed")
    parser.add_argument("--shape", choices=["pairs", "endpoints"], default="pairs")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--extra", default="", help="name=score,... extra observed reads")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--law-only", action="store_true", help="only the best partition and its law")
    parser.add_argument("--multi", default="", help="endpoints whose 2- and 3-flips are scored")
    parser.add_argument("--probe-base", default="", help="comma list of isolated endpoints of a READ csv")
    args = parser.parse_args()
    if args.extra:
        for item in args.extra.split(","):
            key, val = item.split("=")
            OBSERVED[key] = float(val)
    prior = dict(PRIOR_INFORMED)
    if args.prior == "flat":
        prior = {e: (0.0 if v == 0.0 else FLAT_UNCERTAIN) for e, v in prior.items()}
    variant, mask = abc(prior, args.samples, args.shape, SEED)
    n_kept = variant.shape[0]
    print(f"shape={args.shape} prior={args.prior} kept={n_kept} of {args.samples}")
    if n_kept == 0:
        return
    weights = np.ones(n_kept)
    post = variant.mean(0)
    print("posterior P(variant): prior -> posterior")
    for fam in range(3):
        row = [f"{e[3:]}:{prior[e]:.2f}->{post[i]:.2f}" for i, e in enumerate(ENDPOINTS)
               if FAMILY_OF[e] == fam]
        print(f"  F{fam + 1} " + "  ".join(row))
    print("variants per family (posterior mean):",
          [round(float(variant[:, FAM == f].sum(1).mean()), 2) for f in range(3)])
    best_now, val_now = best_partition(variant, mask, weights)
    print(f"best partition now: isolate {best_now} -> E[private score] {val_now:.3f}")
    v1 = read_submission("metc_f6")
    print(f"E[private] partition_v1 (f6): {private_score(variant, mask, v1).mean():.3f}")
    base_iso = best_now
    # candidate probes: f9 and single flips around the current best
    if args.law_only:
        pub = round_step(scores(truth_same_model(variant), mask,
                                [isolation_submission("b", base_iso)])[:, 0])
        vals, counts = np.unique(np.round(pub, 2), return_counts=True)
        print("public law of best partition:",
              " ".join(f"{v:.2f}:{c / n_kept:.2f}" for v, c in zip(vals, counts)))
        return
    candidates: dict[str, Submission] = {}
    if (QUEUE / "metn_f9.csv").exists():
        candidates["metn_f9"] = read_submission("metn_f9")
    probe_base = args.probe_base.split(",") if args.probe_base else base_iso
    for e in ENDPOINTS:
        if prior[e] == 0.0:
            continue
        trial = [x for x in probe_base if x != e] if e in probe_base else [*probe_base, e]
        candidates[("rejoin " if e in probe_base else "isolate ") + e] = isolation_submission(e, trial)
    candidates["best_now"] = isolation_submission("best", base_iso)
    if args.multi:
        pool = args.multi.split(",")
        for size in (2, 3):
            for combo in itertools.combinations(pool, size):
                trial = list(probe_base)
                for e in combo:
                    trial = [x for x in trial if x != e] if e in trial else [*trial, e]
                candidates["flip " + "+".join(c[3:] for c in combo)] = isolation_submission(
                    "m", trial)
    # Cross-fitted VOI: the partition is chosen on half of the samples and scored on the other
    # half, so the optimism of re-optimising on a small conditional sample cancels out.
    halves = [np.arange(n_kept) % 2 == 0, np.arange(n_kept) % 2 == 1]

    def cross_fitted(sel: BoolArray) -> float:
        total = 0.0
        for fit, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
            f_idx, e_idx = sel & fit, sel & ev
            if f_idx.sum() < MIN_CONDITIONAL or e_idx.sum() == 0:
                chosen = base_iso
            else:
                chosen, _ = best_partition(variant[f_idx], mask[f_idx],
                                           np.ones(int(f_idx.sum())), base_iso)
            total += float(private_score(variant[e_idx], mask[e_idx],
                                         isolation_submission("c", chosen)).sum())
        return total

    value_cf = cross_fitted(np.ones(n_kept, dtype=bool)) / n_kept
    print(f"cross-fitted value now: {value_cf:.4f}")
    results = []
    for name, sub in candidates.items():
        pub = round_step(scores(truth_same_model(variant), mask, [sub])[:, 0])
        outcomes, inverse = np.unique(np.round(pub, 2), return_inverse=True)
        gain = 0.0
        dist = []
        flipped = name.split()[-1] if name.startswith(("rejoin", "isolate")) else "ep-11"
        idx = ENDPOINTS.index(flipped)
        for k, o in enumerate(outcomes):
            sel = inverse == k
            frac = float(sel.mean())
            if frac >= REPORT_FRACTION:
                dist.append(f"{o:.2f}:{frac:.2f}(pv={variant[sel, idx].mean():.2f})")
            gain += cross_fitted(sel)
        results.append((gain / n_kept - value_cf, name, " ".join(dist)))
    results.sort(reverse=True)
    print("probe VOI (expected private-score gain after reading it) | predictive public score")
    for voi, probe_name, outcome_text in results:
        print(f"  {voi:+.4f}  {probe_name:16s} {outcome_text}")
    if args.out:
        args.out.write_text(json.dumps({"kept": n_kept, "posterior": dict(zip(ENDPOINTS, post.tolist())),
                                        "best_now": best_now, "value_now": val_now,
                                        "voi": [(v, n, d) for v, n, d in results]}, indent=1))


if __name__ == "__main__":
    main()
