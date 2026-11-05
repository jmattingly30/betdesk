import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from lift import *
d=pd.read_parquet('/mnt/project-files/sports-analytics/models/nba-player/data/test_predictions.parquet')
d=d[(d.playoff==0)&(d.min_proj>=12)].copy()
d['played']=d.played.fillna(0)
d=d[d.played==1]
d['rk']=d.groupby(['game_id','team']).mu_pts.rank(ascending=False,method='first'); d=d[d.rk<=5]
for s in ['pts','reb','ast','fg3m','stl','blk','tov','pra']: d['r_'+s]=d[s]-d['mu_'+s]
d['r_min']=d['min']-d.min_proj
T=d.groupby(['game_id','team']).agg(season=('season','first'),opp=('opp','first'),ts=('team_score','first'),os_=('opponent_team_score','first'),tl=('total_line','first'),sp=('team_spread','first'),imp=('team_implied','first')).reset_index()
T['G_cover']=(T.ts-T.os_)-T.sp; T['G_team_total']=T.ts-T.imp; T['G_over']=(T.ts+T.os_)-T.tl   # sp = expected margin (positive = favourite)
for k in range(1,6):
    x=d[d.rk==k][['game_id','team']+[f'r_{s}' for s in ['pts','reb','ast','fg3m','stl','blk','tov','min']]]
    x.columns=['game_id','team']+[f'P{k}_{s}' for s in ['pts','reb','ast','3pm','stl','blk','tov','min']]
    T=T.merge(x,on=['game_id','team'],how='left')
own=[c for c in T.columns if c[:1]=='P' or c in ('G_team_total',)]
O=T[['game_id','team']+own].copy(); O.columns=['game_id','opp']+['OPP_'+c for c in own]
W=T.merge(O,on=['game_id','opp'])
cols=[c for c in W.columns if (c[:1]=='P' or c.startswith('G_') or c.startswith('OPP_')) and not c.endswith('_min')]
L=legs_from(W,cols)
dates=d.groupby('game_id').date.first(); W['date']=W.game_id.map(dates)
tr=(W.date<W.date.quantile(.5)).to_numpy()   # first half of the sample vs second half
Lt,Lv=L[tr].reset_index(drop=True),L[~tr].reset_index(drop=True)
P=pair_table(Lt,Lv,minn=300); P['stable']=np.sign(P.tr_lift-1)==np.sign(P.te_lift-1); P['avg']=(P.tr_lift+P.te_lift)/2; P['dev']=(P.avg-1).abs()
P=P[~P.a.str.startswith('OPP')]
P.to_csv('nba_pair_lifts.csv',index=False); pd.set_option('display.width',250)
for k,g in P[P.stable].groupby('kind'):
    print(f'\n=== {k}'); print(g.sort_values('dev',ascending=False).head(12)[['a','b','tr_lift','te_lift','te_joint','te_n']].round(3).to_string(index=False))
W.to_parquet('nba_wide.parquet')
