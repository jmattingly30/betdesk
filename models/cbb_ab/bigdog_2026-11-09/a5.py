import pandas as pd, numpy as np
c=pd.read_parquet('/tmp/cbb_frame.parquet'); o=pd.read_parquet('/tmp/f_cbb_open.parquet')
f=c[(c.season<=2019)&c.line_margin.notna()].copy(); s=np.sign(f.line_margin).replace(0,1)
f['al']=f.line_margin.abs(); f['eg']=(f.ours-f.line_margin)*s
kn=[0,5,10,15,20,25,30,40]; cb=pd.cut(f.al,kn+[80],right=False)
t=f.groupby(cb,observed=True).agg(x=('al','mean'),y=('eg','mean'),n=('al','size')); print(t.round(2))
xs,ys=t.x.values,t.y.values
ys=np.where(xs<20,0,ys)  # only correct the tail
print('curve',list(zip(xs.round(1),ys.round(2))))
def adj(L): return np.sign(L)*np.interp(np.abs(L),xs,ys)
def W(e,r):
    s=np.sign(e); rr=np.sign(r); k=(rr!=0)&(s!=0); n=int(k.sum()); return f"n={n:4d} {((s[k]==rr[k]).mean() if n else 0):.3f}"
for name,d,L,cl in [('CLOSE 2020-26',c[c.line_margin.notna()&(c.season>=2020)].copy(),'line_margin',None),('DK OPEN 2026',o.copy(),'line_open','line_close_dk')]:
    print('\n==',name)
    for lab,e in [('raw gap',d.ours-d[L]),('tail-adjusted',d.ours-d[L]-adj(d[L]))]:
        r=d.act-d[L]; dog=np.sign(e)!=np.sign(d[L])
        line=f" {lab:14s}"
        for g in (3,4):
            k=e.abs()>=g; extra='' if cl is None else f" CLV {(np.sign(e)*(d[cl]-d[L]))[k].mean():+.2f}"
            line+=f" | gap{g}+ {W(e[k],r[k])}{extra} dogshare {dog[k].mean():.2f} big20dogs {int((k&dog&(d[L].abs()>=20)).sum())}"
        print(line)
