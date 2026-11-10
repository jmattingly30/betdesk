import pandas as pd, numpy as np
c=pd.read_parquet('/tmp/cbb_frame.parquet'); o=pd.read_parquet('/tmp/f_cbb_open.parquet')
def w(e,r):
    s=np.sign(e); rr=np.sign(r); k=(rr!=0)&(s!=0); n=int(k.sum()); return (n, round((s[k]==rr[k]).mean(),3) if n else None)
o['dog']=np.sign(o.edge)!=np.sign(o.line_open)
for lo,hi in [(0,20),(20,99),(25,99)]:
    z=o[(o.line_open.abs()>=lo)&(o.line_open.abs()<hi)&(o.edge.abs()>=3)]
    for dg in (True,False):
        zz=z[z.dog==dg]; print(f'open26 line {lo}-{hi} {"dog" if dg else "fav"} gap3+ n={len(zz)} ATS={w(zz.edge,zz.res)} CLV pts={zz.clv.mean():.2f}  close-ATS (vs DK close)={w(zz.edge, zz.act-zz.line_close_dk)}')
# what does the gap look like *against the DK close* for those opener big-dog picks: did gap persist?
z=o[(o.line_open.abs()>=20)&(o.edge.abs()>=3)&o.dog]
print('big dog opener picks: mean |gap| at open', z.edge.abs().mean().round(2), 'gap vs DK close', (np.sign(z.edge)*(z.ours-z.line_close_dk)).mean().round(2))
# season by season, close, line 20+ dog gap3+
c=c[c.line_margin.notna()].copy(); c['e']=c.ours-c.line_margin; c['r']=c.act-c.line_margin; c['dog']=np.sign(c.e)!=np.sign(c.line_margin)
z=c[(c.line_margin.abs()>=20)&(c.e.abs()>=3)&c.dog]
print('close 20+ dog gap3+ by season', {s:w(g.e,g.r) for s,g in z.groupby('season')})
print('close 20+ dog gap3+ Nov-Dec only', w(z[z.mon.isin([11,12])].e, z[z.mon.isin([11,12])].r))
