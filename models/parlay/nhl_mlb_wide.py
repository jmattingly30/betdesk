import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from lift import *
M='/mnt/project-files/sports-analytics/models'; pd.set_option('display.width',250)
def report(W,cols,trmask,name,minn=300):
    L=legs_from(W,cols); Lt,Lv=L[trmask].reset_index(drop=True),L[~trmask].reset_index(drop=True)
    P=pair_table(Lt,Lv,minn=minn); P['stable']=np.sign(P.tr_lift-1)==np.sign(P.te_lift-1); P['avg']=(P.tr_lift+P.te_lift)/2; P['dev']=(P.avg-1).abs()
    P=P[~P.a.str.startswith('OPP')]; P.to_csv(f'{name}_pair_lifts.csv',index=False)
    for k,g in P[P.stable].groupby('kind'):
        print(f'\n=== {name} {k}'); print(g.sort_values('dev',ascending=False).head(10)[['a','b','tr_lift','te_lift','te_joint','te_n']].round(3).to_string(index=False))
# ---- NHL
h=pd.read_parquet(f'{M}/nhl-player/data/player_games.parquet'); h=h[(h.playoff==0)&(h.n_prev>=10)]
sk=h[h.is_g==0].copy()
for s in ['points','sog','goals','assists','blk','hits']: sk['r_'+s]=sk[s]-sk[s+'_e5']
sk['rk']=sk.groupby(['game_id','team']).points_e5.rank(ascending=False,method='first'); sk=sk[sk.rk<=6]
T=sk.groupby(['game_id','team']).agg(season=('season','first'),opp=('opp','first'),gf_e=('gf_e','first'),ga_e=('ga_e','first'),sf_e=('sf_e','first')).reset_index()
gf=h[h.is_g==0].groupby(['game_id','team']).goals.sum().rename('gf').reset_index(); T=T.merge(gf,on=['game_id','team'])
sf=h[h.is_g==0].groupby(['game_id','team']).sog.sum().rename('sf').reset_index(); T=T.merge(sf,on=['game_id','team'])
T['G_team_goals']=T.gf-T.gf_e; T['G_team_shots']=T.sf-T.sf_e
for k in range(1,7):
    x=sk[sk.rk==k][['game_id','team']+[f'r_{s}' for s in ['points','sog','goals','assists','blk','hits']]]; x.columns=['game_id','team']+[f'S{k}_{s}' for s in ['pts','sog','goals','ast','blk','hits']]
    T=T.merge(x,on=['game_id','team'],how='left')
g=h[(h.is_g==1)&(h.started_g==1)][['game_id','team','saves','saves_l10','goals_against']].copy(); g['GOALIE_saves']=g.saves-g.saves_l10; g['GOALIE_ga']=g.goals_against-2.9
T=T.merge(g[['game_id','team','GOALIE_saves','GOALIE_ga']],on=['game_id','team'],how='left')
own=[c for c in T.columns if c[:1]=='S' or c.startswith('G_') or c.startswith('GOALIE')]
O=T[['game_id','team']+own].copy(); O.columns=['game_id','opp']+['OPP_'+c for c in own]; W=T.merge(O,on=['game_id','opp'])
W['G_total_goals']=W.G_team_goals+W.OPP_G_team_goals
cols=[c for c in W.columns if c[:1]=='S' or c.startswith('G_') or c.startswith('GOALIE') or c.startswith('OPP_')]
cols=[c for c in cols if c not in ('S1','season') and not c.startswith('season')]
W.to_parquet('nhl_wide.parquet'); report(W,cols,(W.season<=W.season.median()).to_numpy(),'nhl')
# ---- MLB
b=pd.read_parquet(f'{M}/mlb-player/data/batter_games.parquet'); b=b[b.pa>=3].copy()
ph=b.b_1B*b.pf_1B+b.b_2B*b.pf_2B+b.b_3B*b.pf_3B+b.b_HR*b.pf_HR; ptb=b.b_1B*b.pf_1B+2*b.b_2B*b.pf_2B+3*b.b_3B*b.pf_3B+4*b.b_HR*b.pf_HR
b['r_hits']=b.hits-b.pa*ph; b['r_tb']=b.tb-b.pa*ptb; b['r_k']=b.K-b.pa*b.b_K*b.pf_K; b['r_bb']=b.BB-b.pa*b.b_BB*b.pf_BB
T=b[['gid','team','pteam','home','season']].drop_duplicates(['gid','team'])
for sl in range(1,10):
    x=b[b.slot==sl][['gid','team','r_hits','r_tb','r_k','r_bb']]; x.columns=['gid','team',f'B{sl}_hits',f'B{sl}_tb',f'B{sl}_k',f'B{sl}_bb']; T=T.merge(x,on=['gid','team'],how='left')
s=pd.read_parquet(f'{M}/mlb-player/data/starter_games.parquet')
s['team']=np.where(s.home==1,s.gid.str[:3],None)
s['SP_k']=s.K-s.k_l5; s['SP_outs']=(s.OUT+s.K)-(s.OUT+s.K).groupby(s.pitcher).transform(lambda v:v.shift().rolling(5,min_periods=2).mean())
s['SP_hits']=(s['1B']+s['2B']+s['3B']+s.HR)-(s.bf*(s.p_1B+s.p_2B+s.p_3B+s.p_HR)); s['SP_bb']=s.BB-s.bf*s.p_BB
g=pd.read_parquet(f'{M}/team-strength/data/mlb_games_model.parquet')[['gid','hscore','vscore','total_close','ph_close','hometeam','visteam']]
s=s.merge(g[['gid','hometeam','visteam']],on='gid'); s['team']=np.where(s.home==1,s.hometeam,s.visteam)
T=T.merge(s[['gid','team','SP_k','SP_outs','SP_hits','SP_bb']],on=['gid','team'],how='left')   # this team's starter (pitching vs opponent)
T=T.merge(g,on='gid',how='left'); T=T[T.total_close.notna()]
T['G_over']=(T.hscore+T.vscore)-T.total_close; T['G_win']=np.where(T.team==T.hometeam,(T.hscore>T.vscore)-T.ph_close,(T.vscore>T.hscore)-(1-T.ph_close))
T['G_team_runs']=np.where(T.team==T.hometeam,T.hscore,T.vscore)-T.total_close/2
own=[c for c in T.columns if c[:1] in ('B',) or c.startswith('SP_') or c in ('G_team_runs',)]
O=T[['gid','team']+own].copy(); O.columns=['gid','pteam']+['OPP_'+c for c in own]; W=T.merge(O,on=['gid','pteam'])
cols=[c for c in W.columns if (c[:1]=='B' or c.startswith('SP_') or c.startswith('G_') or c.startswith('OPP_'))]
W.to_parquet('mlb_wide.parquet'); report(W,cols,(W.season<=2022).to_numpy(),'mlb',minn=500)
