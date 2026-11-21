"""Combine several judges' family groupings by majority vote on pairs.

    python constitution/ensemble_vote.py KEY_DIR OUT.json judge1.json judge2.json [judge3.json ...]

Two endpoints are linked when at least half of the judges (strictly more than half when the count is even... see
MIN_SHARE) put them in the same family; families are the connected groups of links. Writes the result in the
judges' JSON format and scores it with judge_dossier.py score.
"""
import json
import subprocess
import sys
from itertools import combinations

key_dir, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
judges = [json.load(open(f)) for f in files]
eps = sorted(judges[0])
MIN_SHARE = 0.5  # a pair is linked when more than this share of judges group it
parent = {e: e for e in eps}


def find(e):
    while parent[e] != e:
        parent[e] = parent[parent[e]]
        e = parent[e]
    return e


for a, b in combinations(eps, 2):
    share = sum(j[a] == j[b] for j in judges) / len(judges)
    if share > MIN_SHARE:
        parent[find(a)] = find(b)
roots = sorted({find(e) for e in eps})
label = {e: f"G{roots.index(find(e)) + 1}" for e in eps}
json.dump(label, open(out, "w"), indent=1)
print(f"{len(judges)} judges -> {len(roots)} families")
subprocess.run([sys.executable, "model_equality/constitution/judge_dossier.py", "score", key_dir, out])
