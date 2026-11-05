import pandas as pd, numpy as np
from scipy.stats import rankdata, norm
d=pd.read_parquet('/mnt/project-files/sports-analytics/models/nba-player/data/test_predictions.parquet')
d=d[(d.played==1)&(d.min_proj>=15)&(d.playoff==0)].copy()
for s in ['pts','reb','ast','fg3m','pra','stl','blk','tov']: d['r_'+s]=d[s]-d['mu_'+s]
d['r_min']=d['min']-d.min_proj
d['tot']=d.team_score+d.opponent_team_score; d['r_tot']=d.tot-d.total_line; d['r_team']=d.team_score-d.team_implied
d['r_mar']=(d.team_score-d.opponent_team_score)+d.team_spread
print('spread sign check',np.corrcoef(d.team_spread,d.team_score-d.opponent_team_score)[0,1])
def ns(x): return norm.ppf((rankdata(x)-0.5)/len(x))
rows=[]
def add(lab,a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    rows.append((lab,round(np.corrcoef(ns(a),ns(b))[0,1],3),len(a),round(np.corrcoef(a>0,b>0)[0,1],3)))
d['rk']=d.groupby(['game_id','team']).mu_pts.rank(ascending=False,method='first')
for st in ['pts','reb','ast','pra','fg3m']:
    w=d.pivot_table(index=['game_id','team'],columns='rk',values='r_'+st); add(f'teammates #1 x #2 scorer: {st}',w[1],w[2]); add(f'teammates #1 x #3: {st}',w[1],w[3])
    w2=d.pivot_table(index=['game_id','team'],columns='rk',values='r_'+st)
w=d.pivot_table(index=['game_id','team'],columns='rk',values='r_min'); add('teammates #1 x #2 minutes',w[1],w[2])
wa=d.pivot_table(index=['game_id','team'],columns='rk',values='r_ast'); wp=d.pivot_table(index=['game_id','team'],columns='rk',values='r_pts')
add('#1 scorer assists x #2 scorer points',wa[1],wp[2]); add('#2 assists x #1 points',wa[2],wp[1])
x=d[d.rk==1]
for st in ['pts','reb','ast','pra','fg3m']:
    add(f'top scorer {st} x game total',x['r_'+st],x.r_tot); add(f'top scorer {st} x own team pts above implied',x['r_'+st],x.r_team); add(f'top scorer {st} x own team cover',x['r_'+st],x.r_mar)
y=d[d.rk>=4]; add('bench/role (#4+) pts x own team cover',y.r_pts,y.r_mar); add('role (#4+) minutes x own team cover',y.r_min,y.r_mar); add('top scorer minutes x own team cover',x.r_min,x.r_mar)
add('same player pts x reb',d.r_pts,d.r_reb); add('same player pts x ast',d.r_pts,d.r_ast); add('same player reb x ast',d.r_reb,d.r_ast); add('same player pts x 3PM',d.r_pts,d.r_fg3m); add('same player minutes x pts',d.r_min,d.r_pts)
o=x.merge(x[['game_id','team','r_pts','r_reb']].rename(columns={'team':'opp','r_pts':'o_pts','r_reb':'o_reb'}),on=['game_id','opp']); o=o[o.team<o.opp]
add('top scorer pts x OPPOSING top scorer pts',o.r_pts,o.o_pts); add('top scorer reb x opposing top scorer reb',o.r_reb,o.o_reb)
c=d[d.pos_grp.astype(str).str.contains('C|big',case=False)] if 'pos_grp' in d else d
w=d.pivot_table(index=['game_id','team'],columns='rk',values='r_reb')
R=pd.DataFrame(rows,columns=['pair','rho_normal','n','phi_binary']); print(R.to_string(index=False)); R.to_csv('nba_pairs.csv',index=False)
