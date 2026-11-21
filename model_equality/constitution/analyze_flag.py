"""Does the identity question flag the endpoints with a system prompt or a persona?

    python constitution/analyze_flag.py ANSWERS.jsonl TRUTH.csv [draws=3]

Per family, the reference creator is the one cited most often over all endpoints of the family. An endpoint is
flagged as disguised when it cites that creator in at most 1/3 of its first `draws` answers.
Disguised = kind "system prompt" or "persona". Reports, per question and family, who is flagged and the errors.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

CREATORS = {"hugging face": "HF", "huggingface": "HF", "smol": "HF", "meta": "Meta", "llama": "Meta",
            "alibaba": "Alibaba", "qwen": "Alibaba", "allen": "AI2", "olmo": "AI2", "mistral": "Mistral",
            "google": "Google", "gemma": "Google", "microsoft": "MS", "phi": "MS", "openai": "OpenAI",
            "anthropic": "Anthropic", "deepseek": "DeepSeek"}
DISGUISED = {"system prompt", "persona"}


def creators(text):
    t = text.lower()
    return {v for k, v in CREATORS.items() if re.search(r"\b" + re.escape(k), t)}


answers, truth = sys.argv[1], sys.argv[2]
draws = int(sys.argv[3]) if len(sys.argv) > 3 else 3
info = {r["endpoint"]: r for r in csv.DictReader(open(truth))}
resp = defaultdict(list)  # (question, endpoint) -> list of creator sets, in repeat order
for r in sorted(map(json.loads, open(answers)), key=lambda r: r["repeat"]):
    if r["endpoint"] in info:
        resp[(r["id"], r["endpoint"])].append(creators(r["response"]))

for q in sorted({q for q, _ in resp}):
    print(f"\n=== {q} (first {draws} draws per endpoint)")
    total_err = 0
    for fam in sorted({r["family"] for r in info.values()}):
        eps = [ep for ep, r in info.items() if r["family"] == fam]
        count = Counter(c for ep in eps for s in resp[(q, ep)][:draws] for c in s)
        ref = count.most_common(1)[0][0] if count else None
        print(f"  family {fam}: reference creator = {ref}  (all mentions {dict(count)})")
        for ep in sorted(eps, key=lambda e: info[e]["kind"]):
            got = resp[(q, ep)][:draws]
            k = sum(ref in s for s in got)
            flagged = k <= len(got) / 3
            truth_dis = info[ep]["kind"] in DISGUISED
            err = flagged != truth_dis
            total_err += err
            print(f"    {info[ep]['kind']:30s} cites {ref}: {k}/{len(got)}  -> {'FLAG' if flagged else 'ok  '}"
                  f"{'   <-- WRONG' if err else ''}")
    print(f"  errors: {total_err}")
