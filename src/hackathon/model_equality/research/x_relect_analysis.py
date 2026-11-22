"""Mac-side reading of the x-met-relect containers: profile likelihood per hypothesis, Wald LLR, flags.

Input: the `.npz` written by `x_relect_gpu.py` (per-row sums at 13 temperatures, hits, pmax).
For every endpoint and hypothesis (template fixed by the identification, quantisation variant):

* `LL_v = max_T sum_rows logp_T` (grid maximum refined by a parabola through its neighbours): the
  temperature is a nuisance parameter fitted separately for each hypothesis;
* `llr_v = LL_v - LL_base` (nats) and its row-level z (`sum / (sd * sqrt(n))`), a check against a
  large LLR produced by a few outlier rows (habillage) rather than by every row;
* `z_cal`: calibration of the best hypothesis at its fitted temperature, `sum(logp + H) / sqrt(sum var)`,
  ~N(0, 1) when that hypothesis (at that T) wrote the text: the analytic version of "answers drawn
  locally from the best hypothesis" (its expectation along the observed prefixes);
* `z_top`: top-1 residual, `(hits - sum pmax_T) / sqrt(N p (1 - p))`, a moment that the fitted
  temperature does not absorb.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

STATS_PER_TEMPERATURE = 4
COL_HITS = 0
OFFSET_LOGP, OFFSET_ENT, OFFSET_VAR, OFFSET_PMAX = 0, 1, 2, 3
BASE = "base"
WALD_THRESHOLD_NATS = math.log(99.0)
"""Odds of 99 to 1: a hypothesis is classified when it beats every other one by ln 99 = 4.595 nats."""
NPZ_SEP = "__"

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Profile:
    """Profile log-likelihood of one (endpoint, hypothesis)."""

    ll: float
    temperature: float
    index: int


def column(index_t: int, offset: int) -> int:
    """Column of statistic `offset` at temperature number `index_t`."""
    return 1 + STATS_PER_TEMPERATURE * index_t + offset


def profile(rows: FloatArray, temperatures: Sequence[float]) -> Profile:
    """Maximum over T of the summed log-likelihood, refined by a parabola on the grid neighbours."""
    curve = np.array([rows[:, column(i, OFFSET_LOGP)].sum() for i in range(len(temperatures))])
    best = int(np.argmax(curve))
    if 0 < best < len(curve) - 1:
        left, mid, right = curve[best - 1], curve[best], curve[best + 1]
        denominator = left - 2 * mid + right
        if denominator < 0:
            shift = 0.5 * (left - right) / denominator
            step = temperatures[best + 1] - temperatures[best]
            return Profile(float(mid - 0.25 * (left - right) * shift), float(temperatures[best] + shift * step), best)
    return Profile(float(curve[best]), float(temperatures[best]), best)


def load(path: Path) -> tuple[dict[str, object], dict[str, dict[str, FloatArray]], dict[str, dict[str, npt.NDArray[np.str_]]]]:
    """`(meta, {tag: {variant: rows x stats}}, {tag: {field: array}})` of one container."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        scores: dict[str, dict[str, FloatArray]] = {}
        keys: dict[str, dict[str, npt.NDArray[np.str_]]] = {}
        for name in data.files:
            parts = name.split(NPZ_SEP)
            if len(parts) != 3 or parts[2].startswith("paired"):
                continue
            tag = NPZ_SEP.join(parts[:2])
            if parts[2] in ("endpoint", "prompt_id", "repeat", "n_tokens"):
                keys.setdefault(tag, {})[parts[2]] = data[name]
            else:
                scores.setdefault(tag, {})[parts[2]] = data[name].astype(np.float64)
    return meta, scores, keys


def analyse(path: Path, out_csv: Path | None, prompt_filter: set[str] | None = None) -> list[dict[str, object]]:
    """Per-endpoint table of the hypotheses (printed, and written to `out_csv`)."""
    meta, scores, keys = load(path)
    grid = meta["temperatures"]
    if not isinstance(grid, list):
        raise ValueError(f"{path}: meta.temperatures is not a list")
    temperatures = [float(t) for t in grid]
    table: list[dict[str, object]] = []
    for tag, by_variant in scores.items():
        endpoints = keys[tag]["endpoint"].astype(str)
        prompts = keys[tag]["prompt_id"].astype(str)
        n_tokens = keys[tag]["n_tokens"].astype(np.float64)
        variants = [v for v in by_variant if by_variant[v].shape[0] == len(endpoints)]
        for endpoint in sorted(set(endpoints)):
            mask = endpoints == endpoint
            if prompt_filter is not None:
                mask &= np.isin(prompts, list(prompt_filter))
            if not mask.any() or BASE not in by_variant:
                continue
            n = float(n_tokens[mask].sum())
            profiles = {v: profile(by_variant[v][mask], temperatures) for v in variants}
            base = profiles[BASE]
            row: dict[str, object] = {
                "tag": tag, "endpoint": endpoint, "rows": int(mask.sum()), "n_tokens": int(n),
                "t_base": round(base.temperature, 3),
            }
            base_rows = by_variant[BASE][mask][:, column(base.index, OFFSET_LOGP)]
            for variant in variants:
                if variant == BASE:
                    continue
                prof = profiles[variant]
                per_row = by_variant[variant][mask][:, column(prof.index, OFFSET_LOGP)] - base_rows
                spread = float(per_row.std(ddof=1)) if per_row.size > 1 else float("nan")
                z = float(per_row.sum() / (spread * math.sqrt(per_row.size))) if spread > 0 else float("nan")
                row[f"llr_{variant}"] = round(prof.ll - base.ll, 2)
                row[f"z_{variant}"] = round(z, 2)
                row[f"t_{variant}"] = round(prof.temperature, 3)
            best = max(profiles, key=lambda v: profiles[v].ll)
            others = [profiles[v].ll for v in profiles if v != best]
            margin = profiles[best].ll - max(others) if others else float("inf")
            stats = by_variant[best][mask]
            i = profiles[best].index
            z_cal = stats[:, column(i, OFFSET_LOGP)].sum() + stats[:, column(i, OFFSET_ENT)].sum()
            z_cal /= math.sqrt(max(stats[:, column(i, OFFSET_VAR)].sum(), 1e-9))
            hits = stats[:, COL_HITS].sum()
            # Expected top-1 mass at the FITTED temperature, interpolated between grid points: the grid
            # point alone (step 0.1) made the residual jump for endpoints near x.x5 (ep-03, ep-08).
            pmax_curve = np.array([stats[:, column(j, OFFSET_PMAX)].sum() for j in range(len(temperatures))])
            expected = float(np.interp(profiles[best].temperature, temperatures, pmax_curve))
            p_bar = expected / n if n else 0.0
            z_top = (hits - expected) / math.sqrt(max(n * p_bar * (1 - p_bar), 1e-9))
            row.update({
                "best": best, "margin_nats": round(margin, 2),
                "classified": bool(margin >= WALD_THRESHOLD_NATS), "z_cal": round(float(z_cal), 2),
                "top1_obs": round(hits / n, 4) if n else 0.0, "top1_exp": round(p_bar, 4),
                "z_top": round(float(z_top), 2),
            })
            table.append(row)
    if out_csv is not None and table:
        fields: list[str] = []
        for row in table:
            fields += [k for k in row if k not in fields]
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    return table


PAIRED_OBS, PAIRED_MEAN_B, PAIRED_VAR_B, PAIRED_MEAN_F, PAIRED_VAR_F, PAIRED_N = range(6)


def analyse_paired(path: Path, variant: str = "w_tensor") -> list[dict[str, object]]:
    """Exact Wald reading: observed fp8-vs-base LLR against its moments under each hypothesis.

    `rho = (obs - E_b) / (E_f - E_b)` is 0 for text written by the base and 1 for text written by the
    simulated fp8 model; the real fp8 reference (ref-02) gives the realism of the simulator.
    """
    with np.load(path, allow_pickle=False) as data:
        suffix = f"{NPZ_SEP}paired_{variant}"
        tags = sorted({name[: -len(suffix)] for name in data.files if name.endswith(suffix)})
        table: list[dict[str, object]] = []
        for tag in tags:
            stats = data[tag + suffix].astype(np.float64)
            endpoints = data[f"{tag}{NPZ_SEP}paired_endpoint"].astype(str)
            for endpoint in sorted(set(endpoints)):
                s = stats[endpoints == endpoint].sum(axis=0)
                gap = s[PAIRED_MEAN_F] - s[PAIRED_MEAN_B]
                var_b, var_f = max(s[PAIRED_VAR_B], 1e-12), max(s[PAIRED_VAR_F], 1e-12)
                z_b = (s[PAIRED_OBS] - s[PAIRED_MEAN_B]) / math.sqrt(var_b)
                z_f = (s[PAIRED_OBS] - s[PAIRED_MEAN_F]) / math.sqrt(var_f)
                log_odds = (z_b**2 - z_f**2) / 2 + 0.5 * math.log(var_b / var_f)
                table.append({
                    "tag": tag, "endpoint": endpoint, "n_tokens": int(s[PAIRED_N]),
                    "obs": round(float(s[PAIRED_OBS]), 2), "mean_b": round(float(s[PAIRED_MEAN_B]), 2),
                    "sd_b": round(math.sqrt(var_b), 2), "mean_f": round(float(s[PAIRED_MEAN_F]), 2),
                    "sd_f": round(math.sqrt(var_f), 2),
                    "rho": round(float((s[PAIRED_OBS] - s[PAIRED_MEAN_B]) / gap), 3) if gap else float("nan"),
                    "rho_se": round(float(math.sqrt(var_b) / gap), 3) if gap else float("nan"),
                    "z_b": round(float(z_b), 2), "z_f": round(float(z_f), 2),
                    "log_odds_exact": round(float(log_odds), 2),
                })
    return table


def main(argv: Sequence[str] | None = None) -> int:
    """Print the hypothesis table of a container."""
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--paired-out", type=Path, default=None)
    args = parser.parse_args(argv)
    paired = analyse_paired(args.npz)
    if paired:
        print("endpoint   N      obs     E_b   sd_b     E_f   sd_f    rho  rho_se   z_b    z_f  log_odds")
        for p in paired:
            print(f"{p['endpoint']:8s} {p['n_tokens']:6d} {p['obs']:+7.2f} {p['mean_b']:+7.2f} {p['sd_b']:6.2f} "
                  f"{p['mean_f']:+7.2f} {p['sd_f']:6.2f} {p['rho']:+6.3f} {p['rho_se']:6.3f} {p['z_b']:+6.2f} "
                  f"{p['z_f']:+6.2f} {p['log_odds_exact']:+8.2f}")
        if args.paired_out is not None:
            with args.paired_out.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
                writer.writeheader()
                writer.writerows(paired)
    table = analyse(args.npz, args.out)
    variants = sorted({k[4:] for row in table for k in row if k.startswith("llr_")})
    head = f"{'endpoint':8s} {'N':>6s} {'T_b':>5s} " + " ".join(f"{v[:10]:>17s}" for v in variants)
    print(head + "   best        margin  z_cal  top1o  top1e  z_top")
    for row in table:
        cells = " ".join(f"{row.get('llr_' + v, float('nan')):+8.2f}({row.get('z_' + v, float('nan')):+5.1f})  " for v in variants)
        print(f"{row['endpoint']:8s} {row['n_tokens']:6d} {row['t_base']:5.2f} {cells} {row['best']:12s} "
              f"{row['margin_nats']:6.2f} {row['z_cal']:6.2f} {row['top1_obs']:.4f} {row['top1_exp']:.4f} {row['z_top']:6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
