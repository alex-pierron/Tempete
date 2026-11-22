"""Rank plain-copy candidates by closeness to the watermark gauge, on the SAME prompts (tours B and C only).

Usage: plain_rank.py <relect.npz> [prompt id prefixes, default W8-,W9-,WC-]
Light (numpy on the container), run under nice. For each family: t_hat (base, profile max) with its
standard error from the profile curvature, z_cal and z_top on the prompt subset shared by every endpoint
of the comparison; distance = |dt|/0.01 + |dz_cal|/2 + |dz_top| (A8 rule) and the t-gap in SE units.
"""
import sys, math
# Run from the repository root with PYTHONPATH=src.
import numpy as np
from pathlib import Path
from hackathon.model_equality.research.x_relect_analysis import load, analyse
GAUGE = {'qwen2.5-3b': 'ep-04', 'qwen2.5-1.5b': 'ep-06', 'llama3.2-1b': 'ep-10'}
CANDS = {'qwen2.5-3b': ['ep-01', 'ep-12', 'ep-17', 'ep-18', 'ep-24', 'ep-25'],
         'qwen2.5-1.5b': ['ep-03', 'ep-21', 'ep-26', 'ep-32', 'ep-22'],
         'llama3.2-1b': ['ep-05', 'ep-09', 'ep-11', 'ep-13', 'ep-19']}
npz = Path(sys.argv[1]); prefixes = tuple((sys.argv[2] if len(sys.argv) > 2 else 'W8-,W9-,WC-').split(','))
meta, scores, keys = load(npz)
ids = set()
for tag in keys:
    ids |= {p for p in keys[tag]['prompt_id'].astype(str) if p.startswith(prefixes)}
for fam, g in GAUGE.items():
    # prompts answered by the gauge (the shared subset)
    shared = set()
    for tag in keys:
        e = keys[tag]['endpoint'].astype(str); p = keys[tag]['prompt_id'].astype(str)
        shared |= {x for x, y in zip(p, e) if y == g and x in ids}
    rows = {r['endpoint']: r for r in analyse(npz, None, prompt_filter=shared) if r['endpoint'] in [g] + CANDS[fam]}
    if g not in rows: print(fam, 'gauge missing'); continue
    G = rows[g]
    print(f"== {fam} gauge {g} rows {G['rows']} tok {G['n_tokens']} t {G['t_base']} zcal {G['z_cal']} ztop {G['z_top']} (shared prompts {len(shared)})")
    out = []
    for e in CANDS[fam]:
        if e not in rows: continue
        r = rows[e]
        d = abs(r['t_base'] - G['t_base']) / 0.01 + abs(r['z_cal'] - G['z_cal']) / 2 + abs(r['z_top'] - G['z_top'])
        out.append((d, e, r))
    for d, e, r in sorted(out):
        print(f"  {e} dist {d:5.2f}  rows {r['rows']:3d} tok {r['n_tokens']:6d} t {r['t_base']:.3f} zcal {r['z_cal']:+6.2f} ztop {r['z_top']:+6.2f} best {r['best']}")
