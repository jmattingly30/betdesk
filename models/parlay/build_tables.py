"""Build latent (tetrachoric) correlation tables between every parlay leg type, per sport, for the parlay pricer.
Inputs: the wide team-game tables made by nfl_wide.py, nba_wide.py, nhl_mlb_wide.py (residuals = actual minus our
projection; a leg 'hits' when the residual is > 0, i.e. the over of a line set at our projection).
Output: parlay_corr.json {sport: {legs: [...], R: [[...]], n: [[...]], note}} plus game-level spread x total tables."""
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from copula import tetra
out={}
SPEC={'NFL':('nfl_wide.parquet',lambda c:c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3','G','OPP') and c!='OPP_G_cover'),
      'NBA':('nba_wide.parquet',lambda c:(c[:1]=='P' or c.startswith('G_') or c.startswith('OPP_')) and not c.endswith('_min')),
      'NHL':('nhl_wide.parquet',lambda c:(c[:1]=='S' or c.startswith('G_') or c.startswith('GOALIE') or c.startswith('OPP_')) and c!='season'),
      'MLB':('mlb_wide.parquet',lambda c:(c[:1]=='B' or c.startswith('SP_') or c.startswith('G_') or c.startswith('OPP_')))}
for sp,(f,keep) in SPEC.items():
    W=pd.read_parquet(f); cols=[c for c in W.columns if keep(c) and pd.api.types.is_numeric_dtype(W[c])]
    X=(W[cols]>0).astype(float).where(W[cols].notna()); k=len(cols); R=np.eye(k); N=np.zeros((k,k),int)
    A=X.to_numpy()
    for i in range(k):
        for j in range(i+1,k):
            m=~np.isnan(A[:,i])&~np.isnan(A[:,j]); n=int(m.sum()); N[i,j]=N[j,i]=n
            if n<200: R[i,j]=R[j,i]=0; continue
            x,y=A[m,i],A[m,j]; pa,pb,pab=x.mean(),y.mean(),(x*y).mean()
            if min(pa,pb)<0.03 or max(pa,pb)>0.97: continue
            R[i,j]=R[j,i]=round(tetra(pa,pb,pab),3)
    out[sp]=dict(legs=cols,R=R.round(3).tolist(),n=N.tolist(),base_rate=[round(float(v),3) for v in np.nanmean(A,0)])
    print(sp,k,'legs')
# game-level spread x total latent correlation by spread/total ratio (favourite cover x over), all sports with lines
G=pd.read_parquet('game_pairs.parquet'); gl={}
for spn,d in G.groupby('sport'):
    edges=[0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,2]; rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        x=d[(d.ratio>=lo)&(d.ratio<hi)]
        if len(x)<250: continue
        F,O=(x.F>0).astype(float),(x.O>0).astype(float); rows.append(dict(lo=lo,hi=hi,n=len(x),rho=round(tetra(F.mean(),O.mean(),(F*O).mean()),3),fav_over=round((F*O).mean(),3),dog_under=round(((1-F)*(1-O)).mean(),3)))
    gl[spn]=rows
out['game_spread_total']=gl
out['note']=('Latent correlations between leg types (tetrachoric, on over-our-projection indicators). Leg names: role_stat from '
  'one team\'s view; OPP_ = the opponent. Under = flip the sign of that leg\'s correlations. Validated: pairwise Gaussian copula '
  'with these tables reproduced held-out NFL 3-5 leg stack hit rates within about 3% (copula_test). Built 2026-10-17.')
json.dump(out,open('parlay_corr.json','w'),separators=(',',':'))
print({k:len(v) for k,v in gl.items()})
