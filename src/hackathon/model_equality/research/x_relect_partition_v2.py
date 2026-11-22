"""Rule v2 of x-met-relect: posterior probability that an endpoint is NOT its family's main model.

Two independent pieces of evidence, both calibrated on the 9 ref endpoints (SmolLM2-360M, ground
truth known: plain, watermark, system, persona, T=1 share the model; fp8 and three LoRA strengths
are separate models):

1. fp8 (exact Wald): `rho = (obs - E_b) / (E_f - E_b)` from the paired moments, where E_b / E_f are
   the expected fp8-vs-base LLR under each hypothesis along the observed prefixes. Base refs give
   the null spread of rho (habillage adds variance beyond the sampling one), ref-02 the realism of
   the simulator (real fp8 sits at rho_ref02, not 1). log LR = Gaussian log-ratio with the
   endpoint's own sampling variance plus the null spread.
2. other weights (LoRA): top-1 residual per token `top1_obs - top1_exp` at the fitted temperature
   (LoRA refs -0.010 to -0.018, habillage refs -0.004 to +0.018), Gaussian log LR with a sampling
   sd inflated for token dependence; plus the fitted-temperature band 0.80-0.95 outside a disguised
   identity (LoRA 0.7 / 1.5 refs at 0.82 / 0.95, persona 0.86 but disguised).

Priors: one fp8 and three LoRA endpoints per family of 11 (the ref design). An endpoint is isolated
only when P(not main) > 0.5 (MCC on pairs: isolating below 0.5 costs in expectation).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

PRIOR_FP8 = 1.0 / 11.0
PRIOR_LORA = 3.0 / 11.0
RESIDUAL_BASE = 0.0024  # plain, watermark, T=1, fp8 refs at the interpolated T
RESIDUAL_LORA = -0.0050  # mean of the LoRA 0.3, 0.7, 1.5 refs
DEPENDENCE_INFLATION = 1.5
T_BAND = (0.80, 0.95)
T_BAND_LOG_LR = math.log(3.0)
ISOLATE_ABOVE = 0.5
BASE_REFS = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
FP8_REF = "ref-02"
FP8_RHO_SHIFT = 0.45
"""Real fp8 sits this far above its family's null rho: ref-02 +0.39 over the base refs, ep-15 +0.60 over
the Llama median; 0.45 leans to the conservative side of the two."""
FAMILY_LABEL = {"llama3.2-1b": "F3", "qwen2.5-1.5b": "F2", "qwen2.5-3b": "F1"}
DISGUISED: frozenset[str] = frozenset({"ep-02", "ep-08", "ep-20", "ep-23", "ep-16"})


def logistic(x: float) -> float:
    """1 / (1 + exp(-x)), safe for large |x|."""
    return 1.0 / (1.0 + math.exp(-x)) if x > -50 else 0.0


def main() -> int:
    """`x_relect_partition_v2.py table.csv paired.csv out.csv notes.csv`."""
    table_path, paired_path, out_path, notes_path = (Path(p) for p in sys.argv[1:5])
    table = {r["endpoint"]: r for r in csv.DictReader(table_path.open(encoding="utf-8"))}
    paired = {r["endpoint"]: r for r in csv.DictReader(paired_path.open(encoding="utf-8"))}
    rho_null = [float(paired[e]["rho"]) for e in BASE_REFS if e in paired]
    null_mean, null_spread = mean(rho_null), pstdev(rho_null)
    rho_fp8 = float(paired[FP8_REF]["rho"])
    print(f"ref calibration: base rho {[round(x, 3) for x in rho_null]} mean {null_mean:.3f} spread {null_spread:.3f}; "
          f"ref-02 rho {rho_fp8:.3f}")
    for ref in ("ref-02", "ref-04", "ref-07", "ref-09"):
        if ref in paired:
            print(f"  {ref} rho {float(paired[ref]['rho']):+.3f} se {float(paired[ref]['rho_se']):.3f}")
    family_rho: dict[str, list[float]] = {}
    for endpoint, row in table.items():
        if endpoint.startswith("ep-"):
            family_rho.setdefault(row["tag"].split("__")[0], []).append(float(paired[endpoint]["rho"]))
    family_null = {family: sorted(values)[len(values) // 2] for family, values in family_rho.items()}
    print("family null rho (median):", {k: round(v, 3) for k, v in family_null.items()})
    official, notes = [], []
    for endpoint in sorted(e for e in table if e.startswith("ep-")):
        row, pair = table[endpoint], paired[endpoint]
        family = row["tag"].split("__")[0]
        rho, rho_se = float(pair["rho"]), float(pair["rho_se"])
        variance = rho_se**2 + null_spread**2
        null = family_null[family]
        lr_fp8 = ((rho - null) ** 2 - (rho - (null + FP8_RHO_SHIFT)) ** 2) / (2 * variance)
        n = float(row["n_tokens"])
        residual = float(row["top1_obs"]) - float(row["top1_exp"])
        p_bar = float(row["top1_exp"])
        sd = DEPENDENCE_INFLATION * math.sqrt(p_bar * (1 - p_bar) / n)
        lr_lora = ((residual - RESIDUAL_BASE) ** 2 - (residual - RESIDUAL_LORA) ** 2) / (2 * sd**2)
        temperature = float(row["t_base"])
        if T_BAND[0] <= temperature < T_BAND[1] and endpoint not in DISGUISED:
            lr_lora += T_BAND_LOG_LR
        # Three exclusive hypotheses: main, fp8, LoRA (likelihoods relative to main).
        weight_main = 1.0 - PRIOR_FP8 - PRIOR_LORA
        weight_fp8 = PRIOR_FP8 * math.exp(min(lr_fp8, 50.0))
        weight_lora = PRIOR_LORA * math.exp(min(lr_lora, 50.0))
        total = weight_main + weight_fp8 + weight_lora
        p_fp8, p_lora = weight_fp8 / total, weight_lora / total
        p_not_main = p_fp8 + p_lora
        isolate = p_not_main > ISOLATE_ABOVE
        label = FAMILY_LABEL[family]
        official.append({"endpoint": endpoint, "family": label,
                         "model": f"{label}-{endpoint}" if isolate else f"{label}-main"})
        notes.append({"endpoint": endpoint, "family": family, "rho": f"{rho:+.3f}", "rho_se": f"{rho_se:.3f}",
                      "log_lr_fp8": f"{lr_fp8:+.2f}", "residual": f"{residual:+.4f}", "log_lr_lora": f"{lr_lora:+.2f}",
                      "t_base": f"{temperature:.3f}", "p_fp8": f"{p_fp8:.2f}", "p_lora": f"{p_lora:.2f}",
                      "p_not_main": f"{p_not_main:.2f}", "isolated": str(isolate), "n_tokens": row["n_tokens"]})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "family", "model"])
        writer.writeheader()
        writer.writerows(official)
    with notes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(notes[0]))
        writer.writeheader()
        writer.writerows(notes)
    for note in notes:
        print(" ".join(f"{k}={v}" for k, v in note.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
