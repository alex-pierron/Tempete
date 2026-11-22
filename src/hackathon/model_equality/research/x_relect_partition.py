"""Frozen classification rule of x-met-relect, applied to the hypothesis table -> official CSV.

Per family (base checkpoint known with high confidence), for each endpoint:

1. fp8 test (calibrated on ref-02, the known fp8 reference): the per-tensor fp8 LLR against the base
   at fitted temperatures sits at the family level `mu_base` (median of the family, ~ -0.006 nat per
   token) for a base-weights endpoint and about `FP8_SHIFT_PER_TOKEN` higher for a real fp8 one
   (ref-02 vs the other refs: +0.0087 nat per token; ep-15 vs the Llama family: +0.0087 too). With
   the endpoint's own standard error (row-level), the Wald odds fp8:base are
   `exp((z_base^2 - z_fp8^2) / 2)`; >= 99 is classified fp8, >= 1 is a candidate.
2. other-weights test (calibrated on the LoRA refs 09 / 04 / 07): top-1 residual per token
   `top1_obs - top1_exp <= TOP1_LORA_PER_TOKEN`, or a fitted temperature in the LoRA band
   (`T_LORA_LOW <= T < T_ONE_LOW`) without a disguised identity (persona refs sit there too, and
   they share the base model).
3. Everything else joins the family's main model.

Each fp8 or other-weights endpoint is a singleton model (two LoRA endpoints are only merged on
evidence, none yet).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import median

FP8_SHIFT_PER_TOKEN = 0.0087
WALD_ODDS = 99.0
TOP1_LORA_PER_TOKEN = -0.012
T_LORA_LOW = 0.80
T_ONE_LOW = 0.95
FAMILY_LABEL = {"llama3.2-1b": "F3", "qwen2.5-1.5b": "F2", "qwen2.5-3b": "F1"}
DISGUISED: frozenset[str] = frozenset({"ep-02", "ep-08", "ep-20", "ep-23", "ep-16"})
"""A1 / A2 identity unique in the family (persona refs share the base model)."""


def classify(table_path: Path, out_csv: Path, notes_csv: Path) -> list[dict[str, str]]:
    """Apply the rule; write the official CSV (endpoint,family,model) and a notes CSV."""
    rows = [r for r in csv.DictReader(table_path.open(encoding="utf-8")) if r["endpoint"].startswith("ep-")]
    by_family: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_family.setdefault(row["tag"].split("__")[0], []).append(row)
    official: list[dict[str, str]] = []
    notes: list[dict[str, str]] = []
    for family, members in by_family.items():
        per_token = {m["endpoint"]: float(m["llr_w_tensor"]) / float(m["n_tokens"]) for m in members}
        mu_base = median(per_token.values())
        label = FAMILY_LABEL[family]
        for m in members:
            endpoint, n = m["endpoint"], float(m["n_tokens"])
            llr, z = float(m["llr_w_tensor"]), float(m["z_w_tensor"])
            se = abs(llr / z) if z not in (0.0,) and not math.isnan(z) and llr != 0 else 10.0
            z_base = (llr - mu_base * n) / se
            z_fp8 = (llr - (mu_base + FP8_SHIFT_PER_TOKEN) * n) / se
            log_odds = (z_base**2 - z_fp8**2) / 2
            residual = float(m["top1_obs"]) - float(m["top1_exp"])
            temperature = float(m["t_base"])
            reasons: list[str] = []
            if log_odds >= math.log(WALD_ODDS):
                kind = "fp8"
            elif log_odds >= 0 and per_token[endpoint] == max(per_token.values()):
                kind = "fp8_candidate"
            else:
                kind = "main"
            if kind == "main":
                if residual <= TOP1_LORA_PER_TOKEN:
                    kind = "other_weights"
                    reasons.append("top1_residual")
                if T_LORA_LOW <= temperature < T_ONE_LOW and endpoint not in DISGUISED:
                    kind = "other_weights"
                    reasons.append("temperature_band")
            model = f"{label}-main" if kind == "main" else f"{label}-{endpoint}"
            official.append({"endpoint": endpoint, "family": label, "model": model})
            notes.append({
                "endpoint": endpoint, "family": family, "kind": kind, "log_odds_fp8": f"{log_odds:.2f}",
                "llr_w_tensor_per_token": f"{per_token[endpoint]:+.5f}", "family_median": f"{mu_base:+.5f}",
                "top1_residual": f"{residual:+.4f}", "t_base": f"{temperature:.3f}",
                "reasons": "|".join(reasons), "n_tokens": m["n_tokens"],
            })
    official.sort(key=lambda r: r["endpoint"])
    notes.sort(key=lambda r: r["endpoint"])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "family", "model"])
        writer.writeheader()
        writer.writerows(official)
    with notes_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(notes[0]))
        writer.writeheader()
        writer.writerows(notes)
    return notes


def main(argv: Sequence[str] | None = None) -> int:
    """`x_relect_partition.py table.csv out.csv notes.csv`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("notes", type=Path)
    args = parser.parse_args(argv)
    for note in classify(args.table, args.out, args.notes):
        print(" ".join(f"{k}={v}" for k, v in note.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
