"""Build an anonymised dossier for an LLM judge, and score the judge's family grouping.

    python constitution/judge_dossier.py build RUN_DIR A|B OUT_DIR [seed=0]
    python constitution/judge_dossier.py score OUT_DIR JUDGE.json

build: one answer (first draw) per question and endpoint, endpoints renamed X01..Xnn in random order, variant
       LoRA 1.5 excluded. Dossier A uses the combined World Cup + cutoff question, dossier B the two separate ones.
       Answers to the harmful prompts are cut to 300 characters: the judge only needs how the model refuses.
       Writes OUT_DIR/dossier.md (for the judge) and OUT_DIR/key.csv (the truth, never shown to the judge).
score: JUDGE.json = {"X01": "group label", ...}; prints the family MCC, the MCC when Llama 1B and 3B count as one
       family, and the composition of each judge group.
"""
import csv
import json
import math
import random
import re
import sys
from itertools import combinations
from pathlib import Path

COMMON = ["J1-code", "J2-math", "J5-languages", "R1-bomb", "R2-selfharm"]
SETS = {"A": ["J1-code", "J2-math", "J3A-worldcup-cutoff", "J5-languages", "R1-bomb", "R2-selfharm"],
        "B": ["J1-code", "J2-math", "J3B-worldcup", "J3B-cutoff", "J5-languages", "R1-bomb", "R2-selfharm"]}
CUT = {"R1-bomb": 300, "R2-selfharm": 300}


def build(run, ab, out, seed=0):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    info = {r["endpoint"]: r for r in csv.DictReader((run / "endpoints.csv").open())
            if r["kind"] != "LoRA at strength 1.5"}
    prompts = {r["id"]: r["prompt"] for r in map(json.loads, open("model_equality/constitution/prompts_judge_v3.jsonl"))}
    first = {}
    for r in map(json.loads, (run / "answers.jsonl").open()):
        if r["repeat"] == 0 and r["endpoint"] in info:
            first[(r["endpoint"], r["id"])] = r["response"]
    eps = sorted(info)
    random.Random(seed).shuffle(eps)
    names = {ep: f"X{i + 1:02d}" for i, ep in enumerate(eps)}
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon", "endpoint", "family", "kind"])
        for ep in eps:
            w.writerow([names[ep], ep, info[ep]["family"], info[ep]["kind"]])
    lines = [f"# Dossier ({len(eps)} anonymous chatbot endpoints)\n",
             "Each endpoint received the same questions; one answer per question is shown.\n",
             "## Questions\n"]
    lines += [f"- **{q}**: {prompts[q]}" for q in SETS[ab]]
    for ep in eps:
        lines.append(f"\n## Endpoint {names[ep]}\n")
        for q in SETS[ab]:
            t = first.get((ep, q), "(no answer)").strip()
            if q in CUT:
                refuses = re.search(r"can.?t|cannot|won.?t|unable|not able|sorry|not provide|illegal|"
                                    r"hotline|helpline|crisis|support|reach out|988|samaritans", t, re.I)
                if not refuses:  # the model went along: keep only its opening, never the content
                    t = t[:80] + " [rest removed: the model did not refuse]"
                elif len(t) > CUT[q]:
                    t = t[:CUT[q]] + " [...]"
            lines.append(f"### {q}\n```text\n{t}\n```")
    (out / "dossier.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'dossier.md'} ({len(eps)} endpoints, {len(SETS[ab])} questions, "
          f"{sum(len(x) for x in lines) // 4} tokens approx.)")


def score(out, judge):
    key = {r["anon"]: r for r in csv.DictReader((Path(out) / "key.csv").open())}
    lab = json.load(open(judge))
    fam = {a: key[a]["family"] for a in key}
    merged = {a: ("llama" if f.startswith("llama") else f) for a, f in fam.items()}

    def mcc(truth):
        tp = fp = fn = tn = 0
        for a, b in combinations(sorted(key), 2):
            s, t = lab.get(a) == lab.get(b), truth[a] == truth[b]
            tp += s and t; fp += s and not t; fn += (not s) and t; tn += (not s) and (not t)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / den if den else 0.0

    print(f"family MCC {mcc(fam):.3f} | with Llama 1B = 3B counted as one family {mcc(merged):.3f}")
    groups = {}
    for a, g in lab.items():
        groups.setdefault(g, []).append(a)
    for g, members in sorted(groups.items()):
        comp = {}
        for a in members:
            comp[fam[a]] = comp.get(fam[a], 0) + 1
        odd = [f"{a} ({key[a]['family']}/{key[a]['kind']})" for a in members]
        print(f"  {g}: {comp}")
    missing = [a for a in key if a not in lab]
    if missing:
        print("  not labelled:", missing)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        score(sys.argv[2], sys.argv[3])
