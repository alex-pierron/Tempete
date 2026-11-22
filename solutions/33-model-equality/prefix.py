"""Repeat-determinism fingerprint: identical rate and shared-prefix fraction between repeats of the same prompt."""
import json, collections, csv, itertools, os, sys
import numpy as np
# MET_META_DIR holds endpoints.csv (endpoint,family) and the answer stores answers_ref.jsonl / answers_ep.jsonl
# (raw API answers: not published in this repository).
M = os.environ.get('MET_META_DIR', 'data/hackathon/met_meta').rstrip('/') + '/'
fam={r['endpoint']:r['family'] for r in csv.DictReader(open(M+'endpoints.csv'))}
def pl(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n
for fn in ['answers_ref.jsonl','answers_ep.jsonl']:
    rows=[json.loads(l) for l in open(M+fn)]
    by=collections.defaultdict(list)
    for r in rows: by[(r['endpoint'],r['id'])].append(r['response'])
    eps=sorted({r['endpoint'] for r in rows}, key=lambda e:(fam.get(e,'smol'),e))
    # prompts repeated on ALL endpoints of the family (fair comparison)
    famof=lambda e: fam.get(e,'smol')
    for f in sorted({famof(e) for e in eps}):
        fe=[e for e in eps if famof(e)==f]
        pids=sorted({k[1] for k in by if k[0] in fe})
        common=[p for p in pids if all(len(by[(e,p)])>=2 for e in fe)]
        print(f'== {fn} {f} common repeated prompts: {len(common)}')
        stats=[]
        for e in fe:
            ident=[];pref=[]
            for p in common:
                v=by[(e,p)]
                for a,b in itertools.combinations(v,2):
                    ident.append(a==b); pref.append(pl(a,b)/max(1,min(len(a),len(b))))
            # all repeated prompts for this endpoint
            ia=[];pa=[]
            for (ee,p),v in by.items():
                if ee!=e: continue
                for a,b in itertools.combinations(v,2):
                    ia.append(a==b); pa.append(pl(a,b)/max(1,min(len(a),len(b))))
            stats.append((e,np.mean(ident),np.mean(pref),len(ident),np.mean(ia),np.mean(pa),len(ia)))
        pm=np.array([s[2] for s in stats]); mu,sd=np.median(pm),np.std(pm)
        for e,i,p,n,ia,pa,na in sorted(stats,key=lambda s:-s[2]):
            print(f'  {e} common: ident {i:.2f} prefix {p:.2f} (n={n}, z={(p-mu)/sd:+.1f}) | all: ident {ia:.2f} prefix {pa:.2f} (n={na})')
