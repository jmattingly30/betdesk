import pandas as pd, numpy as np
from scipy.stats import rankdata, norm
d=pd.read_parquet('/mnt/project-files/sports-analytics/models/nfl-player/data/backtest_predictions.parquet')
d=d[(d.season_type=='REG') if 'season_type' in d else True].copy()
for s in ['pass_yds','rec_yds','rush_yds','rec','carries','targets','cmp','pass_td']: d['r_'+s]=d[s]-d[s+'_hat']
d['r_td']=d.td-d.td_hat; d['tdany']=(d.td>0).astype(float)
d['tot']=d.home_score+d.away_score; d['r_tot']=d.tot-d.total_line
d['r_team']=d.team_score-d.implied   # team points above implied team total
d['opp_score']=d.tot-d.team_score; d['r_mar']=(d.team_score-d.opp_score)-d.spread   # spread = expected team margin? check sign
print('check spread sign corr(spread, margin)',np.corrcoef(d.spread,d.team_score-d.opp_score)[0,1])
def ns(x): return norm.ppf((rankdata(x)-0.5)/len(x))
def rho(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    if len(a)<200: return np.nan,len(a),np.nan
    nr=np.corrcoef(ns(a),ns(b))[0,1]; ph=np.corrcoef(a>0,b>0)[0,1]; return round(nr,3),len(a),round(ph,3)
qb=d[d.position=='QB'].sort_values('pass_att_hat',ascending=False).drop_duplicates(['game_id','posteam'])
rc=d[d.position.isin(['WR','TE','RB'])&(d.targets_hat>=2)].copy(); rc['rk']=rc.groupby(['game_id','posteam']).rec_yds_hat.rank(ascending=False,method='first')
rb=d[d.position=='RB'].sort_values('carries_hat',ascending=False).drop_duplicates(['game_id','posteam'])
rows=[]
def add(lab,a,b): r=rho(np.asarray(a,float),np.asarray(b,float)); rows.append((lab,)+r)
m=rc.merge(qb[['game_id','posteam','r_pass_yds','r_cmp','r_pass_td']],on=['game_id','posteam'],suffixes=('_own',''))
for k,lab in [(1,'top receiver'),(2,'2nd receiver'),(3,'3rd receiver')]:
    x=m[m.rk==k]; add(f'QB pass yds x his {lab} rec yds',x.r_pass_yds,x.r_rec_yds); add(f'QB completions x his {lab} receptions',x.r_cmp,x.r_rec)
x=m[m.rk==1]; add('QB pass TDs x top receiver anytime TD',x.r_pass_td,x.tdany-x.td_hat.clip(0,5).map(lambda v:1-np.exp(-v)))
w=rc.pivot_table(index=['game_id','posteam'],columns='rk',values='r_rec_yds'); add('top receiver x 2nd receiver rec yds (same team)',w[1],w[2]); add('top x 3rd receiver',w[1],w[3])
wr=rc.pivot_table(index=['game_id','posteam'],columns='rk',values='r_rec'); add('top x 2nd receiver receptions',wr[1],wr[2])
mm=rb.merge(qb[['game_id','posteam','r_pass_yds']],on=['game_id','posteam'],suffixes=('_own','')); add('lead RB rush yds x own QB pass yds',mm.r_rush_yds,mm.r_pass_yds)
for lab,x,cols in [('QB',qb,['r_pass_yds']),('lead RB',rb,['r_rush_yds','r_carries']),('top receiver',rc[rc.rk==1],['r_rec_yds'])]:
    for c in cols:
        add(f'{lab} {c[2:]} x game total (pts minus closing total)',x[c],x.r_tot); add(f'{lab} {c[2:]} x own team points above implied',x[c],x.r_team); add(f'{lab} {c[2:]} x own team margin vs spread (cover)',x[c],x.r_mar)
add('lead RB anytime TD x own team points above implied',rb.tdany-rb.td_hat.map(lambda v:1-np.exp(-v)),rb.r_team)
add('top receiver anytime TD x game total',rc[rc.rk==1].tdany,rc[rc.rk==1].r_tot)
o=qb.merge(qb[['game_id','posteam','r_pass_yds']].rename(columns={'posteam':'opp','r_pass_yds':'o_pass'}),on=['game_id','opp']); o=o[o.posteam<o.opp]
add('QB pass yds x OPPOSING QB pass yds',o.r_pass_yds,o.o_pass)
oo=rc[rc.rk==1].merge(rc[rc.rk==1][['game_id','posteam','r_rec_yds']].rename(columns={'posteam':'opp','r_rec_yds':'o'}),on=['game_id','opp']); oo=oo[oo.posteam<oo.opp]; add('top receiver x OPPOSING top receiver rec yds',oo.r_rec_yds,oo.o)
own=d[d.position.isin(['WR','TE'])&(d.targets_hat>=3)]; add('same player receptions x rec yds',own.r_rec,own.r_rec_yds)
own=d[(d.position=='RB')&(d.carries_hat>=8)]; add('same RB carries x rush yds',own.r_carries,own.r_rush_yds); add('same RB rush yds x rec yds',own.r_rush_yds,own.r_rec_yds)
add('QB pass yds x QB pass TDs',qb.r_pass_yds,qb.r_pass_td)
g=qb.drop_duplicates('game_id'); add('team margin vs spread x game total (team perspective, any team)',g.r_mar,g.r_tot)
R=pd.DataFrame(rows,columns=['pair','rho_normal','n','phi_binary']); print(R.to_string(index=False)); R.to_csv('nfl_pairs.csv',index=False)
