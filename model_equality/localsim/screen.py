"""Screen candidate questions on the simulator's exact first-token probabilities.

    python model_equality/localsim/screen.py RUN_DIR [top=15]

Under the structural rule (only undisguised endpoints are compared with each other), a good question must
  - move fp8 (and weak LoRA) away from plain,
  - NOT move the temperature-1.0 twin away from plain (it is the same model).
Per family and question: d_fp8 = distance(plain, fp8), d_lora = distance(plain, LoRA 0.3), d_temp = distance(plain,
temperature 1.0), all total variation on the first token. score = d_fp8 - d_temp.
Prints the mean score per category, the best questions, and how much the families agree on which questions are good.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

run = Path(sys.argv[1])
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
info = {r["endpoint"]: r for r in csv.DictReader((run / "endpoints.csv").open())}
ft = {(r["endpoint"], r["id"]): dict(r["top"]) for r in map(json.loads, (run / "firsttok.jsonl").open())}
fams = defaultdict(dict)
for ep, r in info.items():
    fams[r["family"]][r["kind"]] = ep
qs = sorted({q for _, q in ft})


def tv_raw(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def temper(p, factor):
    """Re-temper a distribution: p^factor renormalised (factor = T_old / T_new), on the tokens we have."""
    q = {k: v ** factor for k, v in p.items() if v > 0}
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.43, 1.6, 1.8, 2.0]


def tv(a, b):
    """Distance after the best temperature alignment of a onto b (a temperature-only change gives ~0)."""
    return min(tv_raw(temper(a, f), b) for f in FACTORS) if ADJUST else tv_raw(a, b)


ADJUST = "--raw" not in sys.argv


rows = defaultdict(dict)  # q -> fam -> (d_fp8, d_lora, d_temp, flip)
for fam, k in fams.items():
    for q in qs:
        p = ft.get((k["plain"], q))
        if p is None:
            continue
        f8, lo, te = ft[(k["quantized (fp8)"], q)], ft[(k["LoRA at strength 0.3"], q)], ft[(k["sampling at temperature 1.0"], q)]
        flip = max(p, key=p.get) != max(f8, key=f8.get)
        rows[q][fam] = (tv(p, f8), tv(p, lo), tv(p, te), flip)

famlist = sorted(fams)
print(f"{len(qs)} questions, families {famlist}\n")
print("Mean per category (d_fp8 / d_lora0.3 / d_temp / score / fp8 flips the favourite token):")
cats = defaultdict(list)
for q, per in rows.items():
    cats[q.split("-")[1]].extend(per.values())
for c, vals in sorted(cats.items(), key=lambda kv: -mean(v[0] - v[2] for v in kv[1])):
    print(f"  {c:9s} d_fp8 {mean(v[0] for v in vals):.3f}  d_lora {mean(v[1] for v in vals):.3f}  "
          f"d_temp {mean(v[2] for v in vals):.3f}  score {mean(v[0] - v[2] for v in vals):+.3f}  "
          f"flips {sum(v[3] for v in vals)}/{len(vals)}")

score = {q: {f: v[0] - v[2] for f, v in per.items()} for q, per in rows.items()}
print(f"\nBest {top_n} questions by mean score over families:")
for q in sorted(score, key=lambda q: -mean(score[q].values()))[:top_n]:
    per = rows[q]
    detail = "  ".join(f"{f}: fp8 {per[f][0]:.2f} temp {per[f][2]:.2f}{' FLIP' if per[f][3] else ''}" for f in famlist
                       if f in per)
    print(f"  {mean(score[q].values()):+.3f}  {q}  | {detail}")

if len(famlist) > 1:
    from itertools import combinations
    print("\nDo families agree on good questions? (overlap of each family's top 30)")
    tops = {f: set(sorted(score, key=lambda q: -score[q].get(f, -9))[:30]) for f in famlist}
    for a, b in combinations(famlist, 2):
        print(f"  {a} vs {b}: {len(tops[a] & tops[b])}/30 in common")
