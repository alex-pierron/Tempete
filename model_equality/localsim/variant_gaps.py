"""How far each variant moves from plain, on the exact first-token probabilities of the simulator.

    python model_equality/localsim/variant_gaps.py RUN_DIR

Per family and variant: median distance to plain (total variation, 0 = identical, 1 = nothing in common),
number of questions where the favourite first token is the same as plain, and where the top-3 order is the same.
"""
import csv
import json
import sys
from pathlib import Path
from statistics import median

run = Path(sys.argv[1])
info = {r["endpoint"]: r for r in csv.DictReader((run / "endpoints.csv").open())}
ft = {(r["endpoint"], r["id"]): r["top"] for r in map(json.loads, (run / "firsttok.jsonl").open())}
fams = {}
for ep, r in info.items():
    fams.setdefault(r["family"], {})[r["kind"]] = ep
qs = sorted({q for _, q in ft})


def tv(a, b):
    a, b = dict(a), dict(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


ORDER = ["quantized (fp8)", "LoRA at strength 0.3", "LoRA at strength 0.7", "LoRA at strength 1.5",
         "sampling at temperature 1.0", "watermark", "system prompt", "persona"]
for fam, k in sorted(fams.items()):
    if any((k["plain"], q) not in ft for q in qs):
        continue
    print(f"\n{fam} ({len(qs)} questions)")
    print(f"  {'variant':30s} {'median gap':>10s} {'same top-1':>11s} {'same top-3':>11s}")
    for kind in ORDER:
        ep = k[kind]
        gaps = [tv(ft[(k['plain'], q)], ft[(ep, q)]) for q in qs]
        top1 = sum(ft[(k["plain"], q)][0][0] == ft[(ep, q)][0][0] for q in qs)
        top3 = sum([t for t, _ in ft[(k["plain"], q)][:3]] == [t for t, _ in ft[(ep, q)][:3]] for q in qs)
        print(f"  {kind:30s} {median(gaps):10.2f} {top1:8d}/{len(qs)} {top3:8d}/{len(qs)}")
