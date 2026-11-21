"""Rank the C questions by how well they separate fp8 from plain while ignoring the disguises.

    python constitution/analyze_c.py ANSWERS.jsonl TRUTH.csv [FIRSTTOK.jsonl] [PROMPTS.jsonl]

TRUTH.csv has endpoint,family,model,kind (practice_answers.csv or the simulator's endpoints.csv).
For each family and question:
  - parsed answers: a number, or the first word, depending on the question's "parse" field;
  - fp8 gap      = total variation distance between the parsed answers of plain and of fp8 (0 = same, 1 = disjoint);
  - disguise gap = largest distance between plain and each disguise of the same model (system prompt, persona,
                   watermark, temperature 1.0) -- what we must NOT react to;
  - margin       = fp8 gap - disguise gap (> 0 means the question sees fp8 more than the disguises);
  - with the simulator's first-token probabilities, the same two distances computed exactly on the first token.
Writes <answers dir>/report_C.md and prints it.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISGUISES = ["system prompt", "persona", "watermark", "sampling at temperature 1.0"]


def parse(text, kind):
    t = text.strip()
    if kind in ("number", "digits"):
        m = re.search(r"-?\d[\d,]*", t)
        return m.group().replace(",", "") if m else "none"
    words = re.findall(r"[a-zA-Z]+", t.lower())
    return words[0] if words else "none"


def tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def tv_dict(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def fmt(x):
    return "-" if x != x else f"{x:.2f}"


def main():
    answers, truth = Path(sys.argv[1]), Path(sys.argv[2])
    firsttok = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    prompts = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "prompts_v0_C_fp8.jsonl"
    meta = {r["id"]: r for r in map(json.loads, prompts.open()) if r}
    info = {r["endpoint"]: r for r in csv.DictReader(truth.open())}
    for r in info.values():
        r.setdefault("family", "ref")
        if not r.get("family"):
            r["family"] = "ref"

    dist = defaultdict(Counter)  # (endpoint, qid) -> parsed answers
    for r in map(json.loads, answers.open()):
        if r["id"] in meta and r["endpoint"] in info:
            dist[(r["endpoint"], r["id"])][parse(r["response"], meta[r["id"]].get("parse", "word"))] += 1
    first = {}
    if firsttok and firsttok.exists():
        for r in map(json.loads, firsttok.open()):
            first[(r["endpoint"], r["id"])] = {t.strip().lower(): p for t, p in r["top"]}

    by_family = defaultdict(dict)  # family -> kind -> endpoint
    for ep, r in info.items():
        by_family[r["family"]][r["kind"]] = ep

    out = ["# Block C: which questions separate fp8 from plain\n",
           "Distances are total variation (0 = same answers, 1 = no answer in common). "
           "margin = fp8 gap - largest disguise gap. 'tok' columns use the exact first-token probabilities "
           "(simulator only).\n"]
    ranking = defaultdict(list)
    for fam, kinds in sorted(by_family.items()):
        plain, fp8 = kinds.get("plain"), kinds.get("quantized (fp8)")
        if not plain or not fp8:
            continue
        out.append(f"## Family {fam} (plain = {plain}, fp8 = {fp8})\n")
        out.append("| question | plain answers | fp8 answers | fp8 gap | disguise gap (worst) | margin | "
                   "tok fp8 gap | tok disguise gap |")
        out.append("|---|---|---|---|---|---|---|---|")
        for q in meta:
            a, b = dist.get((plain, q), Counter()), dist.get((fp8, q), Counter())
            if not a or not b:
                continue
            gap = tv(a, b)
            dg = [(tv(a, dist[(kinds[d], q)]), d) for d in DISGUISES if d in kinds and dist.get((kinds[d], q))]
            worst = max(dg) if dg else (float("nan"), "-")
            margin = gap - worst[0] if dg else float("nan")
            tg = tdg = float("nan")
            if (plain, q) in first and (fp8, q) in first:
                tg = tv_dict(first[(plain, q)], first[(fp8, q)])
                tds = [tv_dict(first[(plain, q)], first[(kinds[d], q)]) for d in DISGUISES
                       if d in kinds and (kinds[d], q) in first]
                tdg = max(tds) if tds else float("nan")
            show = lambda c: ", ".join(f"{k}x{v}" for k, v in c.most_common(3))  # noqa: E731
            out.append(f"| {q} | {show(a)} | {show(b)} | {fmt(gap)} | {fmt(worst[0])} ({worst[1]}) | "
                       f"{fmt(margin)} | {fmt(tg)} | {fmt(tdg)} |")
            ranking[q].append(margin if margin == margin else gap)
        out.append("")
    out.append("## Ranking over families (mean margin, or mean fp8 gap when no disguise is available)\n")
    for q, vals in sorted(ranking.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        out.append(f"- {q}: {sum(vals) / len(vals):.2f}  (per family: {', '.join(fmt(v) for v in vals)})")
    report = "\n".join(out)
    (answers.parent / "report_C.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
