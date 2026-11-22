"""Round-level reading of the paired fp8-vs-base moments (x2-met-relect, Model Equality task 33).

`x_relect_analysis.analyse_paired` sums the exact Wald moments over every answer of an endpoint.
This module splits the same sums by question subset, so that a round of targeted questions (Q10:
rare-language stories, Q9: under-trained tokens) can be read on its own and checked against the
earlier answers:

* `rho = (obs - E_b) / (E_f - E_b)` is 0 for text written by the base and 1 for text written by
  the simulated per-tensor fp8 copy; a real fp8 sits at `family null + FP8_RHO_SHIFT` (ref-02,
  ep-15 and ep-22 put the shift between 0.38 and 0.62);
* the Wald log likelihood ratio fp8-vs-main is the Gaussian log-ratio of the two locations, with
  the endpoint's sampling variance plus the habillage spread of the base refs (rule v2);
* a genuine fp8 shows the same rho on the old and on the new questions; a habillage bias does not
  have to (it is question dependent), hence the per-subset columns.

Usage: `x2_relect_round.py <npz> --round <roundK_prompts.json> --ref-paired <paired_refs.csv>
--out <round_table.csv> [--detail ep-22,ep-25]`.
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
from statistics import median, pstdev

import numpy as np
import numpy.typing as npt

PAIRED_OBS, PAIRED_MEAN_B, PAIRED_VAR_B, PAIRED_MEAN_F, PAIRED_VAR_F, PAIRED_N = range(6)
PAIRED_VARIANT = "w_tensor"
NPZ_SEP = "__"
WALD_THRESHOLD_NATS = math.log(99.0)
"""99 to 1, the plan's sequential stopping rule (ln 99 = 4.595 nats)."""
FP8_RHO_SHIFT = 0.45
"""Same constant as rule v2: real fp8 sits this far above its family's null rho."""
BASE_REFS = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
"""Same-model refs (plain, watermark, T=1...) whose rho spread measures the habillage noise."""
MIN_GAP_NATS = 1e-9
EVERY_ENDPOINT = "*"

FloatArray = npt.NDArray[np.float64]


class RoundReadError(ValueError):
    """Raised when the container or the round file cannot be read as expected."""


@dataclass(frozen=True)
class RhoEstimate:
    """Paired fp8-vs-base location of one subset of answers."""

    rows: int
    n_tokens: int
    gap: float
    rho: float
    rho_se: float


@dataclass(frozen=True)
class EndpointPaired:
    """Per-row paired statistics of one endpoint (rows x 6) with their question ids."""

    family: str
    stats: FloatArray
    prompt_ids: npt.NDArray[np.str_]


def rho_of(stats: FloatArray) -> RhoEstimate:
    """Location of a subset of rows: summed moments, `rho` and its sampling standard error.

    Args:
        stats: `rows x 6` paired statistics (obs, E_b, Var_b, E_f, Var_f, n_tokens).

    Returns:
        The estimate; `rho` and `rho_se` are NaN when the subset is empty or the gap is not positive
        (E_f <= E_b cannot happen for a proper divergence, but an empty subset can).
    """
    if stats.shape[0] == 0:
        return RhoEstimate(0, 0, 0.0, float("nan"), float("nan"))
    total = stats.sum(axis=0)
    gap = float(total[PAIRED_MEAN_F] - total[PAIRED_MEAN_B])
    if gap <= MIN_GAP_NATS:
        return RhoEstimate(int(stats.shape[0]), int(total[PAIRED_N]), gap, float("nan"), float("nan"))
    rho = float(total[PAIRED_OBS] - total[PAIRED_MEAN_B]) / gap
    rho_se = math.sqrt(max(float(total[PAIRED_VAR_B]), 0.0)) / gap
    return RhoEstimate(int(stats.shape[0]), int(total[PAIRED_N]), gap, rho, rho_se)


def wald_log_lr(estimate: RhoEstimate, null: float, null_spread: float, shift: float = FP8_RHO_SHIFT) -> float:
    """Gaussian log LR of `rho` under fp8 (null + shift) against the family's main model (null).

    Returns NaN for an estimate without a finite rho (empty subset).
    """
    if not math.isfinite(estimate.rho):
        return float("nan")
    variance = estimate.rho_se**2 + null_spread**2
    return ((estimate.rho - null) ** 2 - (estimate.rho - null - shift) ** 2) / (2.0 * variance)


def verdict(log_lr: float) -> str:
    """`fp8` / `not_fp8` at 99 to 1, `undecided` in between (or without data)."""
    if not math.isfinite(log_lr):
        return "undecided"
    if log_lr >= WALD_THRESHOLD_NATS:
        return "fp8"
    if log_lr <= -WALD_THRESHOLD_NATS:
        return "not_fp8"
    return "undecided"


def load_paired(path: Path, variant: str = PAIRED_VARIANT) -> dict[str, EndpointPaired]:
    """Per-endpoint paired rows of a `x_relect_gpu.py` container (endpoint answers only)."""
    suffix = f"{NPZ_SEP}paired_{variant}"
    out: dict[str, EndpointPaired] = {}
    with np.load(path, allow_pickle=False) as data:
        tags = sorted({name[: -len(suffix)] for name in data.files if name.endswith(suffix)})
        if not tags:
            raise RoundReadError(f"{path}: no '{suffix}' arrays")
        for tag in tags:
            stats = data[tag + suffix].astype(np.float64)
            endpoints = data[f"{tag}{NPZ_SEP}paired_endpoint"].astype(str)
            prompts = data[f"{tag}{NPZ_SEP}paired_prompt_id"].astype(str)
            family = tag.split(NPZ_SEP)[0]
            for endpoint in sorted(set(endpoints)):
                if not endpoint.startswith("ep-"):
                    continue
                mask = endpoints == endpoint
                out[endpoint] = EndpointPaired(family, stats[mask], prompts[mask])
    return out


def round_keys(path: Path) -> set[tuple[str, str]]:
    """`(endpoint, question id)` pairs asked by a round file (format of round<k>_prompts.json)."""
    document = json.loads(path.read_text(encoding="utf-8"))
    prompts = document.get("prompts")
    if not isinstance(prompts, list):
        raise RoundReadError(f"{path}: no 'prompts' list")
    keys: set[tuple[str, str]] = set()
    for prompt in prompts:
        endpoints = prompt.get("endpoints")
        # `endpoints: null` means the question went to every endpoint (round 1 format).
        for endpoint in [EVERY_ENDPOINT] if endpoints is None else endpoints:
            keys.add((str(endpoint), str(prompt["id"])))
    return keys


def in_round(keys: set[tuple[str, str]], endpoint: str, prompt_id: str) -> bool:
    """True when the round asked `prompt_id` to `endpoint` (directly or to every endpoint)."""
    return (endpoint, prompt_id) in keys or (EVERY_ENDPOINT, prompt_id) in keys


def ref_null_spread(path: Path) -> float:
    """Population spread of rho over the same-model refs of a paired CSV (habillage noise)."""
    with path.open(encoding="utf-8") as handle:
        rows = {r["endpoint"]: float(r["rho"]) for r in csv.DictReader(handle)}
    values = [rows[ref] for ref in BASE_REFS if ref in rows]
    if len(values) < 2:
        raise RoundReadError(f"{path}: fewer than two base refs")
    return pstdev(values)


def build_table(
    paired: dict[str, EndpointPaired], new_keys: set[tuple[str, str]], null_spread: float,
    shift: float = FP8_RHO_SHIFT,
) -> list[dict[str, str]]:
    """One row per endpoint: rho on all / old / new answers and the 99:1 fp8 verdicts."""
    by_family: dict[str, list[float]] = {}
    for item in paired.values():
        estimate = rho_of(item.stats)
        if math.isfinite(estimate.rho):
            by_family.setdefault(item.family, []).append(estimate.rho)
    family_null = {family: median(values) for family, values in by_family.items()}
    table: list[dict[str, str]] = []
    for endpoint in sorted(paired):
        item = paired[endpoint]
        is_new = np.array([in_round(new_keys, endpoint, str(p)) for p in item.prompt_ids], dtype=bool)
        every, old, new = rho_of(item.stats), rho_of(item.stats[~is_new]), rho_of(item.stats[is_new])
        null = family_null[item.family]
        lr_all = wald_log_lr(every, null, null_spread, shift)
        lr_new = wald_log_lr(new, null, null_spread, shift)
        table.append({
            "endpoint": endpoint, "family": item.family, "family_null": f"{null:+.3f}",
            "rows": str(every.rows), "n_tokens": str(every.n_tokens),
            "rho_all": f"{every.rho:+.3f}", "se_all": f"{every.rho_se:.3f}",
            "rho_old": f"{old.rho:+.3f}", "se_old": f"{old.rho_se:.3f}",
            "rows_new": str(new.rows), "gap_new": f"{new.gap:.1f}",
            "rho_new": f"{new.rho:+.3f}", "se_new": f"{new.rho_se:.3f}",
            "log_lr_fp8_all": f"{lr_all:+.2f}", "log_lr_fp8_new": f"{lr_new:+.2f}",
            "verdict_all": verdict(lr_all), "verdict_new": verdict(lr_new),
        })
    return table


def detail_rows(item: EndpointPaired) -> list[str]:
    """Per-question lines (summed over repeats): gap, rho and z under the base."""
    lines: list[str] = []
    for prompt in sorted(set(item.prompt_ids)):
        estimate = rho_of(item.stats[item.prompt_ids == prompt])
        variance_b = float(item.stats[item.prompt_ids == prompt][:, PAIRED_VAR_B].sum())
        z_b = (estimate.rho * estimate.gap) / math.sqrt(max(variance_b, 1e-12))
        lines.append(f"  {prompt:22s} rows {estimate.rows} N {estimate.n_tokens:5d} gap {estimate.gap:7.2f} "
                     f"rho {estimate.rho:+7.3f} z_b {z_b:+6.2f}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print and write the round table of a container."""
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--round", type=Path, required=True, help="round<k>_prompts.json of the new answers")
    parser.add_argument("--ref-paired", type=Path, required=True, help="paired CSV holding the ref rows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detail", default="", help="comma-separated endpoints to print per question")
    parser.add_argument("--shift", type=float, default=FP8_RHO_SHIFT, help="fp8 location above the family null")
    args = parser.parse_args(argv)
    paired = load_paired(args.npz)
    spread = ref_null_spread(args.ref_paired)
    table = build_table(paired, round_keys(args.round), spread, args.shift)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"null spread (base refs) {spread:.3f}; threshold {WALD_THRESHOLD_NATS:.2f} nats; shift {args.shift}")
    for row in table:
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    for endpoint in [e for e in args.detail.split(",") if e]:
        if endpoint in paired:
            print(f"{endpoint} ({paired[endpoint].family}):")
            print("\n".join(detail_rows(paired[endpoint])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
