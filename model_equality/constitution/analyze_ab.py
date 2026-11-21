"""Analyse blocks A and B of the constitution on the reference endpoints.

    python constitution/analyze_ab.py [answers file]

For every question and endpoint: the normalised answer of each draw, the most frequent one (mode) and how many draws
gave it. Then, per question, whether all SmolLM variants agree (they are all the same family) and whether the
disguises of model S agree with each other.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CREATORS = {"hugging face": "HF", "huggingface": "HF", "smol": "HF", "meta": "Meta", "llama": "Meta",
            "alibaba": "Qwen", "qwen": "Qwen", "allen": "AI2", "olmo": "AI2", "mistral": "Mistral",
            "google": "Google", "gemma": "Google", "microsoft": "MS", "phi": "MS", "openai": "OpenAI"}


def creator(text):
    t = text.lower()
    found = sorted({v for k, v in CREATORS.items() if re.search(r"\b" + re.escape(k), t)})
    return "+".join(found) if found else "none"


def pick(text, options):
    t = text.lower()
    hits = [(m.start(), o) for o in options for m in [re.search(r"\b" + re.escape(o.lower()) + r"\b", t)] if m]
    return min(hits)[1] if hits else "none"


def first_int(text):
    m = re.search(r"\d[\d,\. ]*\d|\d", text)
    return re.sub(r"[^\d]", "", m.group()) if m else "none"


def code_traits(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    name = re.search(r"def\s+(\w+)\s*\(", code)
    if "Counter" in code:
        method = "Counter"
    elif "defaultdict" in code:
        method = "defaultdict"
    elif ".get(" in code:
        method = "dict.get"
    elif re.search(r"\bin\s+\w+\s*:", code) or " not in " in code:
        method = "if-in"
    else:
        method = "other"
    return {
        "name": name.group(1) if name else "none",
        "method": method,
        "lower": ".lower()" in code,
        "punct": bool(re.search(r"\bre\.|translate\(|string\.punctuation|strip\(", code)),
        "docstring": '"""' in code or "'''" in code,
        "hints": bool(re.search(r"def\s+\w+\s*\([^)]*:\s*\w+|->", code)),
        "example": "print(" in code or "__main__" in code,
    }


def summarise(traits):
    keys = ["name", "method", "lower", "punct", "docstring", "hints", "example"]
    return "/".join(str(traits[k])[0:12] if not isinstance(traits[k], bool) else ("Y" if traits[k] else "n")
                    for k in keys)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "answers_prompts_v0_AB.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    meta = {r["id"]: r for r in (json.loads(line) for line in (HERE / "prompts_v0_AB.jsonl").open() if line.strip())}
    import csv
    kinds = {r["endpoint"]: r["kind"] for r in csv.DictReader((ROOT / "practice_answers.csv").open())}
    models = {r["endpoint"]: r["model"] for r in csv.DictReader((ROOT / "practice_answers.csv").open())}

    parsed = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["endpoint"], r["id"], r["repeat"])):
        q, resp = r["id"], r["response"]
        if q.startswith("A1"):
            v = creator(resp)
        elif "options" in meta[q]:
            v = pick(resp, meta[q]["options"])
        elif q.startswith("B3"):
            v = summarise(code_traits(resp))
        else:
            v = first_int(resp)
        parsed[(r["endpoint"], q)].append(v)

    endpoints = sorted({r["endpoint"] for r in rows})
    out = ["# Blocks A and B on the references\n",
           "Cell = most frequent normalised answer (draws giving it / draws). Code traits: "
           "name/method/lower/punctuation/docstring/type hints/example.\n"]
    out.append("| endpoint | variant | model | " + " | ".join(meta) + " |")
    out.append("|---|---|---|" + "---|" * len(meta))
    modes = {}
    for ep in endpoints:
        cells = []
        for q in meta:
            vals = parsed.get((ep, q), [])
            if not vals:
                cells.append("-")
                continue
            v, k = Counter(vals).most_common(1)[0]
            modes[(ep, q)] = v
            cells.append(f"{v} ({k}/{len(vals)})")
        out.append(f"| {ep} | {kinds.get(ep, '?')} | {models.get(ep, '?')} | " + " | ".join(cells) + " |")

    out.append("\n## Per question\n")
    s_eps = [ep for ep in endpoints if models.get(ep) == "S"]
    for q in meta:
        all_modes = Counter(modes[(ep, q)] for ep in endpoints if (ep, q) in modes)
        s_modes = Counter(modes[(ep, q)] for ep in s_eps if (ep, q) in modes)
        draws = Counter(v for ep in endpoints for v in parsed.get((ep, q), []))
        out.append(f"- **{q}**: modes over the 9 references {dict(all_modes)}; over the 5 disguises of S "
                   f"{dict(s_modes)}; all draws {dict(draws.most_common(6))}")
    report = "\n".join(out)
    (HERE / "report_v0_AB.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
