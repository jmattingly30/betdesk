import pandas as pd, numpy as np
c=pd.read_parquet('/tmp/cbb_frame.parquet'); o=pd.read_parquet('/tmp/f_cbb_open.parquet')
fit=c[c.season<=2019]
# Model A tail check: mean actual by |ours| bucket (fav perspective), fit window
fit=fit.assign(s=np.sign(fit.ours).replace(0,1)); fit=fit.assign(ao=fit.ours.abs(), aa=fit.act*fit.s)
kn=np.array([0,5,10,15,20,25,30,35])
print('fit 2013-19: by |ours| bucket mean ours / mean actual (fav perspective)')
cb=pd.cut(fit.ao,list(kn)+[80],right=False)
t=fit.groupby(cb,observed=True).agg(n=('ao','size'),ours=('ao','mean'),act=('aa','mean')); print(t.round(2))
# piecewise linear stretch: act = ao + f(ao); f fit as mean(aa-ao) per bucket centred at bucket mean ours
xs=t.ours.values; ys=(t.act-t.ours).values
# only stretch the tail (>=20); below that keep 0 (no evidence + keeps the core model untouched)
ys=np.where(xs>=20,ys,0.0); print('tail knots',list(zip(xs.round(1),ys.round(2))))
def stretch(v): a=np.abs(v); return v+np.sign(v)*np.interp(a,xs,ys)
def ats(e,r):
    s=np.sign(e); rr=np.sign(r); k=(rr!=0)&(s!=0); n=int(k.sum()); w=(s[k]==rr[k]).mean() if n else np.nan; return f"n={n:4d} {w:.3f}"
for name,d,L in [('CLOSE 2020+',c[c.line_margin.notna()&(c.season>=2020)],'line_margin'),('DK OPEN 2026',o,'line_open')]:
    d=d.copy(); d['o2']=stretch(d.ours.values)
    big=d[L].abs()>=20
    print('\n==',name)
    for lab,col in [('v2',  'ours'),('v2+tail','o2')]:
        e=d[col]-d[L]; r=d.act-d[L]
        mae=(d[col]-d.act).abs(); dog=np.sign(e)!=np.sign(d[L])
        print(f" {lab:8s} MAE all {mae.mean():.3f}  MAE line20+ {mae[big].mean():.3f} (line {(d[L]-d.act).abs()[big].mean():.3f}) | gap3+ all {ats(e[e.abs()>=3],r[e.abs()>=3])} gap4+ {ats(e[e.abs()>=4],r[e.abs()>=4])} | line20+ gap3+ {ats(e[big&(e.abs()>=3)],r[big&(e.abs()>=3)])} dogs gap3+ share {dog[e.abs()>=3].mean():.2f} | under20 gap3+ {ats(e[~big&(e.abs()>=3)],r[~big&(e.abs()>=3)])}")
