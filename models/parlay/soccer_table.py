"""Soccer match-market latent correlations by the favourite's no-vig win chance (football-data via models/soccer)."""
import pandas as pd, numpy as np, json, sys, itertools, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from copula import tetra
exec(open('soc_pairs.py').read().split("m['bk']=")[0])
legs=['F_win','D_win','draw','over25','btts','fav_cs','F_2plus','HT_fav_lead','corn_over','cards_over','fav_corn_more']
edges=[0,.45,.55,.65,.75,1.01]; res=[]
for lo,hi in zip(edges[:-1],edges[1:]):
    x=m[(m.pF>=lo)&(m.pF<hi)]; R=np.eye(len(legs))
    for i,j in itertools.combinations(range(len(legs)),2):
        A=x[[legs[i],legs[j]]].dropna().astype(float); a,b=A.iloc[:,0],A.iloc[:,1]
        if min(a.mean(),b.mean())<.03: continue
        R[i,j]=R[j,i]=round(tetra(a.mean(),b.mean(),(a*b).mean()),3)
    res.append(dict(lo=lo,hi=hi,n=len(x),R=R.tolist(),base_rate=[round(float(x[l].astype(float).mean()),3) for l in legs]))
T=json.load(open('parlay_corr.json')); T['SOCCER']=dict(legs=legs,by_fav_prob=res,note='F_=favourite, D_=underdog; corners/cards over = above the all-match median (10 corners, 4 card points)')
json.dump(T,open('parlay_corr.json','w'),separators=(',',':')); print('ok', [r['n'] for r in res])
