"""Are joint unders more likely than a symmetric (Gaussian) copula says? Lower-tail vs upper-tail dependence, all sports."""
import pandas as pd, numpy as np, json, sys, itertools, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from copula import joint_gauss
T=json.load(open('parlay_corr.json')); rng=np.random.default_rng(7)
SPEC={'NFL':'nfl_wide.parquet','NBA':'nba_wide.parquet','NHL':'nhl_wide.parquet','MLB':'mlb_wide.parquet'}
rows=[]
for sp,f in SPEC.items():
    W=pd.read_parquet(f); legs=T[sp]['legs']; R=np.array(T[sp]['R']); idx={n:i for i,n in enumerate(legs)}
    ok=[l for l in legs if not l.endswith('_td') and 'GOALIE_ga' not in l]   # td legs only have overs
    # candidate stacks: random sets among legs that are positively correlated with each other (the stacks people build)
    pos=[(a,b) for a,b in itertools.combinations(ok,2) if R[idx[a],idx[b]]>0.15 and R[idx[a],idx[b]]<0.9]
    for k in (2,3,4):
        tried=0
        for _ in range(4000):
            if tried>=150: break
            a,b=pos[rng.integers(len(pos))]; s=[a,b]
            while len(s)<k:
                c=ok[rng.integers(len(ok))]
                if c in s or any(R[idx[c],idx[x]]<0.15 or R[idx[c],idx[x]]>=0.9 for x in s): 
                    if rng.random()<0.995: continue
                    else: break
                s.append(c)
            if len(s)<k: continue
            X=W[s].dropna()
            if len(X)<800: continue
            tried+=1
            Rs=R[np.ix_([idx[x] for x in s],[idx[x] for x in s])]
            for d in ('over','under'):
                H=(X>0) if d=='over' else (X<0); p=H.mean().values; emp=H.all(axis=1).mean()
                cop=joint_gauss(p,Rs); rows.append(dict(sport=sp,k=k,dir=d,legs=' + '.join(s),emp=emp,cop=cop,ind=p.prod(),n=len(X),ratio=emp/cop))
D=pd.DataFrame(rows); D.to_csv('under_vs_over.csv',index=False)
S=D.groupby(['sport','k','dir']).agg(stacks=('ratio','size'),emp_over_copula=('ratio','mean'),emp_over_indep=('emp',lambda x:np.nan)).reset_index()
S['emp_over_indep']=D.groupby(['sport','k','dir']).apply(lambda x:(x.emp/x.ind).mean()).values
S['mean_hit_rate']=D.groupby(['sport','k','dir']).emp.mean().values
pd.set_option('display.width',200); print(S.round(3).to_string(index=False))
# paired: same legs, under ratio minus over ratio
P=D.pivot_table(index=['sport','k','legs'],columns='dir',values='ratio').reset_index(); P['diff']=P.under-P.over
print(P.groupby(['sport','k'])['diff'].agg(['mean','std','count']).assign(t=lambda x:x['mean']/x['std']*np.sqrt(x['count'])).round(3))
