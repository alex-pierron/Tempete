"""Which questions separate families (different base models), whatever the variant (disguise, fp8, LoRA)?

    python constitution/analyze_family.py RUN_DIR [draws=1] [PROMPTS.jsonl]

Kinds: pick (same option), number (closeness), prefix (memorisation, common prefix), digits, code / text (3-grams).

Similarity of two answers, by question kind:
  pick, number : 1 if the extracted value is the same, else 0
  digits       : length of the common prefix of the digit strings / 50
  text, code   : cosine of character 3-gram counts (style and wording)
Distance between two endpoints on a question = 1 - mean similarity over their first `draws` answers (all pairs).
For each question: AUC = probability that a random same-family pair is closer than a random different-family pair
(1 = perfect, 0.5 = useless). Then the same on the mean distance over all questions, the best family MCC over
thresholds, and the worst cases (closest different-family pair, farthest same-family pair).
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

run = Path(sys.argv[1])
draws = int(sys.argv[2]) if len(sys.argv) > 2 else 1
info = {r["endpoint"]: r for r in csv.DictReader((run / "endpoints.csv").open())}
info = {ep: r for ep, r in info.items() if r["kind"] != "LoRA at strength 1.5"}
PROMPTS = sys.argv[3] if len(sys.argv) > 3 else "model_equality/constitution/prompts_family.jsonl"
meta = {r["id"]: r for r in map(json.loads, open(PROMPTS))}
ans = defaultdict(list)
for r in sorted(map(json.loads, (run / "answers.jsonl").open()), key=lambda r: r["repeat"]):
    if r["endpoint"] in info:
        ans[(r["endpoint"], r["id"])].append(r["response"])


def grams(t):
    t = re.sub(r"\s+", " ", t.lower())
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0


def value(t, kind, opts=None):
    if kind == "pick":
        low = t.lower()
        hits = [(m.start(), o) for o in opts for m in [re.search(r"\b" + re.escape(o.lower()) + r"\b", low)] if m]
        return min(hits)[1] if hits else "none"
    m = re.search(r"\d[\d,]*", t)
    return m.group().replace(",", "") if m else "none"


def number(t):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", t)
    try:
        return float(m.group().replace(",", "")) if m else None
    except ValueError:
        return None


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def sim(a, b, q):
    kind = meta[q]["kind"]
    if kind == "number":  # closeness: 1 if equal, 0 when 10% apart or more (1240 vs 1250 -> 0.92)
        x, y = number(a), number(b)
        if x is None or y is None:
            return float(x is None and y is None)
        scale = max(abs(x), abs(y), 1e-9)
        return max(0.0, 1 - abs(x - y) / (0.1 * scale))
    if kind == "prefix":  # memorisation: length of the common word prefix (digits: common digit prefix)
        if "digits" in meta[q]["prompt"]:
            da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
        else:
            da, db = words(a), words(b)
        n = 0
        while n < min(len(da), len(db), 30) and da[n] == db[n]:
            n += 1
        return n / 30
    if kind in ("pick",):
        return float(value(a, kind, meta[q].get("options")) == value(b, kind, meta[q].get("options")))
    if kind == "digits":
        da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
        n = 0
        while n < min(len(da), len(db), 50) and da[n] == db[n]:
            n += 1
        return n / 50
    return cos(grams(a), grams(b))


def dist(e1, e2, q):
    a, b = ans[(e1, q)][:draws], ans[(e2, q)][:draws]
    return 1 - mean(sim(x, y, q) for x in a for y in b) if a and b else None


def auc(pairs):
    same = [d for d, s in pairs if s]
    diff = [d for d, s in pairs if not s]
    return mean((x < y) + 0.5 * (x == y) for x in same for y in diff)


eps = sorted(info)
pairs = list(combinations(eps, 2))
fam = {ep: info[ep]["family"] for ep in eps}
print(f"{len(eps)} endpoints ({sorted(set(fam.values()))}), {len(pairs)} pairs, {draws} draw(s) per question\n")
per_q = {}
print("AUC per question (1 = separates families perfectly, 0.5 = useless):")
for q in meta:
    ds = [(dist(a, b, q), fam[a] == fam[b]) for a, b in pairs]
    ds = [(d, s) for d, s in ds if d is not None]
    per_q[q] = {(a, b): d for (a, b), (d, _) in zip(pairs, ds)}
    print(f"  {q:20s} {auc(ds):.3f}")

total = {p: mean(per_q[q][p] for q in meta if p in per_q[q]) for p in pairs}
print(f"\nAll questions together: AUC {auc([(total[p], fam[p[0]] == fam[p[1]]) for p in pairs]):.3f}")
best = (-1, None)
for thr in sorted(set(total.values())):
    tp = sum(total[p] <= thr and fam[p[0]] == fam[p[1]] for p in pairs)
    fp = sum(total[p] <= thr and fam[p[0]] != fam[p[1]] for p in pairs)
    fn = sum(total[p] > thr and fam[p[0]] == fam[p[1]] for p in pairs)
    tn = len(pairs) - tp - fp - fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = (tp * tn - fp * fn) / den if den else 0
    if m > best[0]:
        best = (m, thr)
print(f"Best family MCC over thresholds: {best[0]:.3f} (threshold {best[1]:.3f})")
kind = {ep: info[ep]["kind"] for ep in eps}
same = sorted((total[p], p) for p in pairs if fam[p[0]] == fam[p[1]])
diff = sorted((total[p], p) for p in pairs if fam[p[0]] != fam[p[1]])
print("\nFarthest same-family pairs:")
for d, (a, b) in same[-4:]:
    print(f"  {d:.3f}  {fam[a]}: {kind[a]}  <->  {kind[b]}")
print("Closest different-family pairs:")
for d, (a, b) in diff[:4]:
    print(f"  {d:.3f}  {fam[a]} {kind[a]}  <->  {fam[b]} {kind[b]}")
