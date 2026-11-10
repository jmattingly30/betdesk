import pandas as pd, numpy as np
c=pd.read_parquet('/tmp/cbb_frame.parquet'); o=pd.read_parquet('/tmp/f_cbb_open.parquet')
def ats(z):
    s=np.sign(z.edge); r=np.sign(z.res); k=(r!=0)&(s!=0); n=int(k.sum())
    if n==0: return '-'
    w=(s[k]==r[k]).mean(); return f"n={n:4d} win={w:.3f} ±{np.sqrt(w*(1-w)/n):.3f}"
bins=[0,3.5,7.5,12.5,17.5,22.5,30,60]; lab=['0-3','3.5-7','7.5-12','12.5-17','17.5-22','22.5-30','30+']
for name,d,L in [('CLOSE 2013-26',c[c.line_margin.notna()].copy(),'line_margin'),('CLOSE 2020+',c[c.line_margin.notna()&(c.season>=2020)].copy(),'line_margin'),('DK OPEN 2026',o.copy(),'line_open')]:
    d['edge']=d.ours-d[L]; d['res']=d.act-d[L]; d['fs']=d[L].abs(); d['b']=pd.cut(d.fs,bins,labels=lab,right=False)
    d['dog']=np.sign(d.edge)!=np.sign(d[L])
    print('\n==',name, 'n',len(d))
    print('calibration (fav perspective): mean fav line, ours, actual; mean edge toward dog; % games gap3+ ; share of gap3+ that are dog')
    for b,z in d.groupby('b',observed=True):
        s=np.sign(z[L]).replace(0,1)
        print(f" {b:8s} n={len(z):5d} line={ (z[L]*s).mean():6.2f} ours={(z.ours*s).mean():6.2f} act={(z.act*s).mean():6.2f}  act-line={(z.res*s).mean():+.2f} ours-line={(z.edge*s).mean():+.2f}  gap3+ {(z.edge.abs()>=3).mean():.2f} dogshare {z[z.edge.abs()>=3].dog.mean():.2f}")
    print('ATS gap3+ by bucket, dog side vs fav side')
    for b,z in d.groupby('b',observed=True):
        g=z[z.edge.abs()>=3]; print(f" {b:8s} DOG {ats(g[g.dog])} | FAV {ats(g[~g.dog])}")
    for lo in (15,20,25):
        g=d[(d.fs>=lo)&(d.edge.abs()>=3)&d.dog]; print(f" line {lo}+ dog gap3+: {ats(g)}; gap4+: {ats(g[g.edge.abs()>=4])}")
    if name.startswith('DK'):
        for lo in (12.5,20):
          g=d[(d.fs>=lo)&(d.edge.abs()>=3)]; print(' by month', lo, {m:ats(z) for m,z in g.groupby(g.mon)})
