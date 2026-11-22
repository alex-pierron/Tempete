"""Public-read law of candidate partitions under hypothesised truths (30% pair subsample, rounded to 0.05)."""
import itertools, numpy as np, sys
RNG_SEED = 20261122
PUBLIC_FRACTION = 0.30
N_DRAWS = 4000
FAM = {'F1':['01','02','04','12','14','17','18','23','24','25','33'],
       'F2':['03','06','07','08','20','21','22','26','28','30','32'],
       'F3':['05','09','10','11','13','15','16','19','27','29','31']}
pairs=[(f,a,b) for f,eps in FAM.items() for a,b in itertools.combinations(eps,2)]
def lab(main):  # main: dict fam->set
    return {(f,e):('M' if e in main[f] else e) for f,eps in FAM.items() for e in eps}
def vec(main): L=lab(main); return np.array([L[(f,a)]==L[(f,b)] for f,a,b in pairs])
def mcc(t,p,m):
    t=t[m];p=p[m]; tp=np.sum(t&p,axis=-1); tn=np.sum(~t&~p,axis=-1); fp=np.sum(~t&p,axis=-1); fn=np.sum(t&~p,axis=-1)
    d=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); return np.where(d>0,(tp*tn-fp*fn)/np.where(d>0,d,1),0)
rng=np.random.default_rng(RNG_SEED)
masks=rng.random((N_DRAWS,len(pairs)))<PUBLIC_FRACTION
def score(t,p):
    out=[]
    for m in masks: out.append(0.1+0.9*mcc(t,p,m))
    return np.array(out)
H5={'F1':{'02','23','33'},'F2':{'08','20','30'},'F3':{'16','31'}}
def add(base,**kw):
    return {f:base[f]|set(kw.get(f,[])) for f in base}
C={'h5':H5,'wm040610':add(H5,F1=['04'],F2=['06'],F3=['10']),'m1019':add(H5,F3=['10','19']),
   'wm+21+19':add(H5,F1=['04'],F2=['06','21'],F3=['10','19']),'wm+12':add(H5,F1=['04','12'],F2=['06'],F3=['10']),
   'wm-04':add(H5,F2=['06'],F3=['10'])}
TRUTHS={'T_wm_ok(5/fam: +12,+21,+19 +29)':add(H5,F1=['04','12'],F2=['06','21'],F3=['10','19','29']),
        'T_wm_ok(other unknowns 17,26,11+09)':add(H5,F1=['04','17'],F2=['06','26'],F3=['10','11','09']),
        'T_wm_wrong(12,21,19,29 + 17,26,11)':add(H5,F1=['12','17'],F2=['21','26'],F3=['19','29','11']),
        'T_04wrong(12,17 ; 06,21 ; 10,19,29)':add(H5,F1=['12','17'],F2=['06','21'],F3=['10','19','29'])}
for tn,T in TRUTHS.items():
    t=vec(T); print('==',tn)
    for cn,Cm in C.items():
        s=score(t,vec(Cm)); full=0.1+0.9*mcc(t,vec(Cm),np.ones(len(pairs),bool))
        r=np.round(s/0.05)*0.05
        vals,cnt=np.unique(np.round(r,2),return_counts=True)
        print(f'  {cn:10s} full {full:.3f} pub mean {s.mean():.3f} law '+' '.join(f'{v:.2f}:{c/len(r):.0%}' for v,c in zip(vals,cnt) if c/len(r)>=0.03))
