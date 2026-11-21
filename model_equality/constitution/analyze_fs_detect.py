"""Few-shot neutralisation and disguise detection, on a simulator run of prompts_fs_detect.jsonl.

    python constitution/analyze_fs_detect.py RUN_DIR

1. Neutralisation: for each variant, distance to plain on the 10 answer targets (first token, after the best
   temperature alignment), without few-shot (NF) and with the ~500-token few-shot prefix (FS). Few-shot helps if the
   disguises (system prompt, personas, watermark) get closer to plain while fp8 and LoRA 0.3 stay apart.
   Also: on how many targets the favourite answer equals plain's.
2. Detection: share of "yes" to the yes/no questions, share of non-"none" summaries, and share of answers citing the
   family's creator to the identity question, per variant. Disguised = system prompt and both personas.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

run = Path(sys.argv[1])
info = {r["endpoint"]: r for r in csv.DictReader((run / "endpoints.csv").open())}
ft = {(r["endpoint"], r["id"]): dict(r["top"]) for r in map(json.loads, (run / "firsttok.jsonl").open())}
ans = defaultdict(list)
for r in map(json.loads, (run / "answers.jsonl").open()):
    ans[(r["endpoint"], r["id"])].append(r["response"])
fams = defaultdict(dict)
for ep, r in info.items():
    fams[r["family"]][r["kind"]] = ep
ORDER = ["sampling at temperature 1.0", "watermark", "system prompt", "persona", "persona (discreet)",
         "quantized (fp8)", "LoRA at strength 0.3", "LoRA at strength 0.7"]
SHORT = {"sampling at temperature 1.0": "temp 1.0", "watermark": "watermark", "system prompt": "sys prompt",
         "persona": "persona pirate", "persona (discreet)": "persona discreet", "quantized (fp8)": "fp8",
         "LoRA at strength 0.3": "LoRA 0.3", "LoRA at strength 0.7": "LoRA 0.7"}
TARGETS = ["arith1", "arith2", "arith3", "count1", "count2", "seq1", "seq2", "fact1", "fact2", "pick1"]
FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.43, 1.6, 1.8, 2.0]


def tv_raw(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def temper(p, f):
    q = {k: v ** f for k, v in p.items() if v > 0}
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


def tv(a, b):
    return min(tv_raw(temper(a, f), b) for f in FACTORS)


def first_word(t):
    w = re.findall(r"[A-Za-z0-9]+", t)
    return w[0].lower() if w else "none"


def mode(eps_id):
    c = Counter(first_word(t) for t in ans.get(eps_id, []))
    return c.most_common(1)[0][0] if c else None


print("1. NEUTRALISATION: distance to plain on 10 targets (first token, temperature-aligned), "
      "and targets with the same favourite answer as plain")
for fam, k in sorted(fams.items()):
    p = k["plain"]
    print(f"\n  {fam}")
    print(f"    {'variant':17s} {'gap NF':>7s} {'gap FS':>7s}   {'same answer NF':>14s} {'same answer FS':>14s}")
    for kind in ORDER:
        x = k[kind]
        g = {m: mean(tv(ft[(p, f'{m}-{t}')], ft[(x, f'{m}-{t}')]) for t in TARGETS) for m in ("NF", "FS")}
        same = {m: sum(mode((p, f"{m}-{t}")) == mode((x, f"{m}-{t}")) for t in TARGETS) for m in ("NF", "FS")}
        print(f"    {SHORT[kind]:17s} {g['NF']:7.2f} {g['FS']:7.2f}   {same['NF']:11d}/10 {same['FS']:11d}/10")

CREATOR = {"qwen05": r"alibaba|qwen", "llama1b": r"\bmeta|llama", "smol360": r"hugging ?face|smol"}
print("\n2. DETECTION: share of answers per variant (5 draws). Disguised rows are marked *")
cols = [("D1-instr", "yes"), ("D2-char", "yes"), ("D3-name", "yes"), ("D4-summary", "not none"),
        ("A1-identity", "cites creator")]
for fam, k in sorted(fams.items()):
    print(f"\n  {fam}")
    print("    " + f"{'variant':19s}" + "".join(f"{c[0] + ' ' + c[1]:>24s}" for c in cols))
    for kind in ["plain"] + ORDER:
        x = k[kind]
        cells = []
        for q, what in cols:
            rs = ans.get((x, q), [])
            if what == "yes":
                v = sum(first_word(t) == "yes" for t in rs)
            elif what == "not none":
                v = sum(not re.match(r"\W*none\b", t.strip().lower()) for t in rs)
            else:
                v = sum(bool(re.search(CREATOR.get(fam, "$^"), t.lower())) for t in rs)
            cells.append(f"{v}/{len(rs)}")
        star = "*" if kind in ("system prompt", "persona", "persona (discreet)") else " "
        print(f"   {star}{SHORT.get(kind, kind):19s}" + "".join(f"{c:>24s}" for c in cells))
