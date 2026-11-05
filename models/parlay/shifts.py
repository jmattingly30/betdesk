"""Condition shifts to our projections: select on early half, measure what survives in the late half; then fit on all data."""
import pandas as pd, numpy as np, json
from scipy.stats import norm
import conds
rows=[]
for sp,(W,legs,C,half) in conds.DATA.items():
    for cn,cm in C.items():
        cm=cm.to_numpy()
        if cm.sum()<150: continue
        for l in legs:
            v=W[l].to_numpy(float); ok=~np.isnan(v); o=(v>0)
            r={}
            for nm,h in (('a',~half),('b',half),('all',np.ones(len(v),bool))):
                m=ok&h; oc=o[m&cm].mean(); oe=o[m&~cm].mean(); oa=o[m].mean(); n=(m&cm).sum(); ne=(m&~cm).sum()
                se=np.sqrt(oc*(1-oc)/n+oe*(1-oe)/ne)
                r[nm]=(oc-oe,(oc-oe)/se,norm.ppf(np.clip(oc,.01,.99))-norm.ppf(np.clip(oa,.01,.99)),n)
            rows.append(dict(sport=sp,condition=cn,leg=l,sh_a=r['a'][0],z_a=r['a'][1],sh_b=r['b'][0],sh_all=r['all'][0],z_all=r['all'][1],dz=r['all'][2],n=r['all'][3],stable=np.sign(r['a'][0])==np.sign(r['b'][0])))
R=pd.DataFrame(rows); R.to_csv('shifts_all.csv',index=False)
sel=R[(R.z_a.abs()>=3)&~R.leg.str.startswith('OPP_')]
print('selected on early half:',len(sel),' same sign late:',(np.sign(sel.sh_a)==np.sign(sel.sh_b)).mean().round(3),' late/early slope:',round(float((sel.sh_a*sel.sh_b).sum()/(sel.sh_a**2).sum()),3))
for sp,x in sel.groupby('sport'): print(sp,len(x),'slope',round(float((x.sh_a*x.sh_b).sum()/(x.sh_a**2).sum()),2))
