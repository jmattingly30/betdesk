import pandas as pd, numpy as np, itertools, sys
def legs_from(W, cols, only_over=()):
    L={}
    for c in cols:
        v=W[c].to_numpy(float)
        L[c+' OVER']=np.where(np.isnan(v),np.nan,(v>0).astype(float))
        if not any(c.endswith(o) for o in only_over): L[c+' UNDER']=np.where(np.isnan(v),np.nan,(v<0).astype(float))
    return pd.DataFrame(L)
def entity(name):
    base=name.rsplit(' ',1)[0]; parts=base.split('_')
    if parts[0]=='OPP': return 'OPP_'+parts[1]
    return parts[0]
def kind(a,b):
    ea,eb=entity(a),entity(b)
    if ea==eb: return 'same player/side, two stats'
    ga=ea.startswith('G') or ea.startswith('OPP_G'); gb=eb.startswith('G') or eb.startswith('OPP_G')
    if ga and gb: return 'game x game'
    if ga or gb: return 'player x game/team'
    oa,ob=ea.startswith('OPP_'),eb.startswith('OPP_')
    return 'opponents (cross-team)' if oa!=ob else 'teammates'
def pair_table(Lt,Lv,minn=150):
    cols=list(Lt.columns); rows=[]
    At=Lt.to_numpy(); Av=Lv.to_numpy()
    for i,j in itertools.combinations(range(len(cols)),2):
        a,b=cols[i],cols[j]
        if a.rsplit(' ',1)[0]==b.rsplit(' ',1)[0]: continue
        r={}
        for lab,A in [('tr',At),('te',Av)]:
            x,y=A[:,i],A[:,j]; m=~np.isnan(x)&~np.isnan(y); x,y=x[m],y[m]
            if m.sum()<minn: r=None; break
            pa,pb,pab=x.mean(),y.mean(),(x*y).mean(); r[lab+'_lift']=pab/(pa*pb); r[lab+'_joint']=pab; r[lab+'_n']=int(m.sum()); r[lab+'_pa']=pa; r[lab+'_pb']=pb
        if r: rows.append(dict(a=a,b=b,kind=kind(a,b),**r))
    return pd.DataFrame(rows)
def joint(L,legs):
    A=L[legs].to_numpy(); m=~np.isnan(A).any(1); A=A[m]
    if len(A)==0: return np.nan,np.nan,0
    pj=A.prod(1).mean(); pi=A.mean(0).prod(); return pj,pj/pi,len(A)
def greedy(Lt,Lv,seed,maxlegs=5,minhits=40,pool=None):
    legs=list(seed); out=[]
    pool=pool or list(Lt.columns)
    while len(legs)<maxlegs:
        best=None
        for c in pool:
            if c in legs or any(c.rsplit(' ',1)[0]==l.rsplit(' ',1)[0] for l in legs): continue
            pj,lf,n=joint(Lt,legs+[c])
            if n==0 or pj*n<minhits: continue
            if best is None or lf>best[1]: best=(c,lf)
        if best is None: break
        legs.append(best[0]); tj,tl,tn=joint(Lt,legs); vj,vl,vn=joint(Lv,legs)
        out.append(dict(legs=' + '.join(legs),k=len(legs),train_joint=tj,train_lift=tl,test_joint=vj,test_lift=vl,test_n=vn))
    return out
