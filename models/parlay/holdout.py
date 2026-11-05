"""Search parlay stacks on the early half, test on the late half.
Book proxy = Gaussian copula with ALL-GAME hit rates and our latent correlations (a book that prices correlation but ignores the condition and the tail shape).
Ours = copula with the condition's hit rates from the TRAIN half. Edge = test hit rate / book proxy."""
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from copula import joint_gauss
import conds
T=json.load(open('parlay_corr.json'))
KEEP={'NFL':['all','fav 7+','dog 7+','total <=40','total 50+','wind 15+','dome/closed'],
      'NBA':['all','fav 10+','dog 10+','fav 6-9.5','total bottom 20%','total top 20%','back-to-back'],
      'MLB':['all','vs top-20% K starter','vs bottom-20% K starter','total 10+','total <=7.5','home','temp <=55'],
      'NHL':['all','big favourite (gf-ga exp top 15%)','big underdog','low expected goals (bottom 20%)','high expected goals (top 20%)','back-to-back']}
rows=[]
ONLY=sys.argv[1:] or list(conds.DATA)
for sp,(W,legs,C,half) in conds.DATA.items():
    if sp not in ONLY: continue
    idx={n:i for i,n in enumerate(T[sp]['legs'])}; R=np.array(T[sp]['R'])
    legs=[l for l in legs if l in idx and not l.startswith('OPP_G_') ]
    X=W[legs].to_numpy(float); ok=~np.isnan(X)
    L={}
    for d,s in (('U',-1),('O',1)):
        for j,l in enumerate(legs): L[(l,d)]=np.where(ok[:,j],(s*X[:,j]>0).astype(float),np.nan)
    keys=list(L); A=np.column_stack([L[k] for k in keys]); kidx={k:i for i,k in enumerate(keys)}
    tr=~half; te=half
    for cn in KEEP[sp]:
        cm=np.ones(len(W),bool) if cn=='all' else C[cn].to_numpy()
        Atr=A[tr&cm]; Ate=A[te&cm]; Aall=A[tr]
        if len(Atr)<200: continue
        pall=np.nanmean(Aall,0); pc=np.nanmean(Atr,0)
        def Rs(ks): ii=[idx[k[0]] for k in ks]; sg=np.array([1 if k[1]=='O' else -1 for k in ks]); return R[np.ix_(ii,ii)]*np.outer(sg,sg)
        def emp(M,ks):
            B=M[:,[kidx[k] for k in ks]]; m=~np.isnan(B).any(1); return (B[m].prod(1).mean() if m.sum() else np.nan), int(m.sum())
        seen=set()
        for seed in keys:
            ks=[seed]
            for step in range(3):
                cand=[]
                for k in keys:
                    if k[0] in [x[0] for x in ks]: continue
                    if any(abs(R[idx[k[0]],idx[x[0]]])>=0.85 for x in ks): continue   # same quantity twice (e.g. pts and PRA)
                    pj,n=emp(Atr,ks+[k])
                    if n<200 or pj*n<30: continue
                    cand.append((pj/np.prod(pc[[kidx[x] for x in ks+[k]]]),k,pj))
                if not cand: break
                cand.sort(reverse=True); best=None
                for lf,k,pj in cand[:6]:
                    kk=ks+[k]; bk=joint_gauss(pall[[kidx[x] for x in kk]],Rs(kk)); r=pj/bk
                    if best is None or r>best[0]: best=(r,k)
                ks=ks+[best[1]]
                if len(ks)>=3:
                    key=(cn,frozenset(ks))
                    if key in seen: continue
                    seen.add(key)
                    ptr,ntr=emp(Atr,ks); pte,nte=emp(Ate,ks)
                    pa=pall[[kidx[x] for x in ks]]; book=joint_gauss(pa,Rs(ks)); ours=joint_gauss(pc[[kidx[x] for x in ks]],Rs(ks))
                    rows.append(dict(sport=sp,condition=cn,k=len(ks),legs=' + '.join(f'{a} {"under" if d=="U" else "over"}' for a,d in ks),n_under=sum(d=='U' for _,d in ks),
                                     train=ptr,test=pte,n_test=nte,book=book,ours=ours,train_edge=ptr/book,test_edge=pte/book,test_vs_ours=pte/ours))
    print(sp,'done',len(rows),flush=True)
D=pd.DataFrame(rows).drop_duplicates(['sport','condition','legs']); D.to_csv('holdout_stacks'+('_'+'_'.join(ONLY) if sys.argv[1:] else '')+'.csv',index=False)
