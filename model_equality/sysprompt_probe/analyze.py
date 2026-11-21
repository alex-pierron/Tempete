"""Analyse the system-prompt probes (answers written by run.py).

For every endpoint it reports:
  1. extraction: the recovered text per probe, whether it is stable across repeats (a real system prompt comes back
     the same; an invented one changes), whether the model says NONE, and a short hash of the normalised text;
  2. groups: endpoints whose stable extracted text has the same hash (same system prompt);
  3. neutralisation: for the endpoints with a known disguise, is the answer closer to the plain endpoint when the
     prompt starts with "ignore all previous instructions..."?

    python sysprompt_probe/analyze.py [answers.jsonl] [practice_answers.csv]

Writes sysprompt_probe/report.md and prints it. Pure standard library.
"""
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
STABLE = 0.8  # mean pairwise similarity across repeats above which an extraction counts as stable


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_kinds(path):
    if not Path(path).exists():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["endpoint"]: r.get("kind", r.get("model", "")) for r in csv.DictReader(f)}


def clean(text):
    """Strip code fences, quotes and the ORIGINAL: marker, keep the recovered text only."""
    t = text.strip()
    if "ORIGINAL:" in t:
        t = t.split("ORIGINAL:", 1)[1]
    t = re.sub(r"```[a-zA-Z]*", "", t).replace('"""', "")
    return t.strip()


def normalise(text):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def short_hash(text):
    return hashlib.sha256(normalise(text).encode()).hexdigest()[:10]


def is_none(text):
    return normalise(text) in {"none", ""}


def sim(a, b):
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def stability(texts):
    pairs = list(combinations(texts, 2))
    return mean(sim(a, b) for a, b in pairs) if pairs else float("nan")


def main():
    answers_path = sys.argv[1] if len(sys.argv) > 1 else HERE / "answers.jsonl"
    kinds = load_kinds(sys.argv[2] if len(sys.argv) > 2 else HERE.parent / "practice_answers.csv")
    rows = load(answers_path)
    meta = {r["id"]: r for r in load(HERE / "prompts.jsonl")}
    by = defaultdict(list)  # (endpoint, prompt id) -> responses
    for r in rows:
        by[(r["endpoint"], r["id"])].append(r["response"])
    endpoints = sorted({r["endpoint"] for r in rows})
    out = []
    w = out.append

    # 1. Extraction
    w("# System-prompt probes: report\n")
    w(f"Answers: `{answers_path}` ({len(rows)} responses, {len(endpoints)} endpoints).\n")
    w("## 1. Extraction per endpoint\n")
    w("Stable = mean similarity between repeats >= %.1f. NONE = the model says it has no instructions.\n" % STABLE)
    best = {}  # endpoint -> (hash, text) of the most stable non-NONE extraction
    for ep in endpoints:
        w(f"### {ep} ({kinds.get(ep, '?')})\n")
        w("| probe | stable | similarity | NONE | hash | first recovered text |")
        w("|---|---|---|---|---|---|")
        candidates = []
        for pid, m in meta.items():
            if m["kind"] != "extract" or (ep, pid) not in by:
                continue
            texts = [clean(t) for t in by[(ep, pid)]]
            s = stability(texts)
            nones = sum(is_none(t) for t in texts)
            h = short_hash(texts[0])
            snippet = texts[0].replace("\n", " ").replace("|", "/")[:120]
            w(f"| {pid} | {'yes' if s >= STABLE else 'no'} | {s:.2f} | {nones}/{len(texts)} | `{h}` | {snippet} |")
            if s >= STABLE and nones == 0 and pid != "x5-identity":
                candidates.append((s, h, texts[0]))
        # A text invented by the model can be stable across repeats of one probe; a real system prompt is recovered
        # by different probes. Keep the endpoint only if two different probes agree.
        agreeing = [c for c in candidates if any(sim(c[2], d[2]) >= STABLE for d in candidates if d is not c)]
        if agreeing:
            s, h, t = max(agreeing)
            best[ep] = (h, t)
        elif candidates:
            w(f"\nOne probe only gave a stable text (not confirmed by another probe): {candidates[0][2][:120]!r}")
        w("")

    # 2. Groups of endpoints sharing the same recovered system prompt
    w("## 2. Endpoints sharing the same recovered system prompt\n")
    groups = defaultdict(list)
    for ep, (h, _) in best.items():
        groups[h].append(ep)
    if not groups:
        w("No stable extraction on any endpoint.\n")
    for h, eps in groups.items():
        w(f"- `{h}`: {', '.join(eps)}. Text: {best[eps[0]][1][:200]!r}")
    no_prompt = [ep for ep in endpoints if ep not in best]
    w(f"- no stable extraction (no system prompt, or extraction failed): {', '.join(no_prompt) or '-'}\n")

    # 3. Neutralisation: similarity to the plain endpoint, with and without the override prefix
    plain = next((ep for ep, k in kinds.items() if k == "plain" and ep in endpoints), None)
    w("## 3. Neutralisation\n")
    if plain is None:
        w("No endpoint labelled 'plain' in the practice answers: skipped.\n")
    else:
        w(f"Mean similarity to the plain endpoint {plain} (all repeat pairs). "
          "The override works if the second column is higher than the first for the disguised endpoints.\n")
        tasks = sorted({m["task"] for m in meta.values() if "task" in m})
        w("| endpoint | kind | " + " | ".join(f"{t} base | {t} override" for t in tasks) + " |")
        w("|---|---|" + "---|---|" * len(tasks))
        for ep in endpoints:
            if ep == plain:
                continue
            cells = []
            for t in tasks:
                for kind in ("neutral-base", "neutral-override"):
                    pid = next(p for p, m in meta.items() if m.get("task") == t and m["kind"] == kind)
                    a, b = by.get((ep, pid), []), by.get((plain, pid), [])
                    cells.append(f"{mean(sim(x, y) for x in a for y in b):.2f}" if a and b else "-")
            w(f"| {ep} | {kinds.get(ep, '?')} | " + " | ".join(cells) + " |")
        w("")

    report = "\n".join(out)
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
