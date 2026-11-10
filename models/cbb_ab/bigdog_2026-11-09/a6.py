import pandas as pd, numpy as np, json
c=pd.read_parquet('/tmp/cbb_frame.parquet'); o=pd.read_parquet('/tmp/f_cbb_open.parquet')
def curve(f):
    s=np.sign(f.line_margin).replace(0,1); al=f.line_margin.abs(); eg=(f.ours-f.line_margin)*s
    cb=pd.cut(al,[0,5,10,15,20,25,30,40,80],right=False); t=pd.DataFrame({'x':al,'y':eg}).groupby(cb,observed=True).mean()
    xs,ys=t.x.values,np.where(t.x.values<20,0,t.y.values); return xs,ys
def W(e,r,clv=None):
    s=np.sign(e); rr=np.sign(r); k=(rr!=0)&(s!=0); n=int(k.sum())
    out=f"n={n:3d} win {((s[k]==rr[k]).mean() if n else 0):.3f}"
    return out+('' if clv is None else f" CLV {clv[k].mean():+.2f}")
C=c[c.line_margin.notna()]
for fitlab,F in [('fit 2013-19',C[C.season<=2019]),('fit 2013-23',C[C.season<=2023])]:
    xs,ys=curve(F); print('\n',fitlab,'curve',list(zip(xs.round(1),ys.round(2))))
    d=o.copy(); L=d.line_open; e=d.ours-L; ea=e-np.sign(L)*np.interp(L.abs(),xs,ys); r=d.act-L; clv=np.sign(e)*(d.line_close_dk-L)
    dog=np.sign(e)!=np.sign(L); big=L.abs()>=20
    for g in (3,4):
        keep=ea.abs()>=g; raw=e.abs()>=g
        print(f'  gap{g}+: raw {W(e[raw],r[raw],clv[raw])} | adjusted {W(ea[keep],r[keep],clv[keep])} | DROPPED by adjust {W(e[raw&~keep],r[raw&~keep],clv[raw&~keep])}')
d=o.copy(); L=d.line_open; e=d.ours-L; r=d.act-L; clv=np.sign(e)*(d.line_close_dk-L); dog=np.sign(e)!=np.sign(L)
for g in (3,4):
    k=(e.abs()>=g)
    print(f'cap-at-lean rule gap{g}+: keep (not big dog) {W(e[k&~(dog&(L.abs()>=20))],r[k&~(dog&(L.abs()>=20))],clv[k&~(dog&(L.abs()>=20))])} | big dogs 20+ capped {W(e[k&dog&(L.abs()>=20)],r[k&dog&(L.abs()>=20)],clv[k&dog&(L.abs()>=20)])}')
