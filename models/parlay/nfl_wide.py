import pandas as pd, numpy as np, itertools
M='/mnt/project-files/sports-analytics/models'
f=pd.read_parquet(f'{M}/nfl-player/data/features.parquet')
f=f[(f.season.between(2016,2025))&(f.games_prior>=3)&(f.snap_l>=0.3) if 'snap_l' in f else True].copy()
f=f[f.season_type=='REG'] if 'season_type' in f else f
f['td']=f.rec_td+f.rush_td
# projections available pre-game in features
P={'rec':'exp_rec','rec_yds':'exp_rec_yds','rush_yds':'exp_rush_yds','cmp':'exp_cmp','pass_yds':'exp_pass_yds','pass_td':'exp_pass_td'}
legs={}
qb=f[f.position=='QB'].sort_values('pass_att',ascending=False).drop_duplicates(['game_id','posteam'])   # actual starter (who threw most)
rb=f[f.position=='RB'].copy(); rb['k']=rb.groupby(['game_id','posteam']).exp_rush_yds.rank(ascending=False,method='first'); rb=rb[rb.k==1]
rc=f[f.position.isin(['WR','TE'])].copy(); rc['k']=rc.groupby(['game_id','posteam']).exp_rec_yds.rank(ascending=False,method='first')
T=qb[['game_id','posteam','season','home_team','away_team','spread_line','total_line','home_score','away_score']].copy()
def put(src,pref,stats):
    global T
    x=src[['game_id','posteam']+[s for s in stats]+[P[s] for s in stats if s in P]].copy()
    for s in stats:
        if s in P: x[f'{pref}_{s}']=x[s]-x[P[s]]
        elif s=='td': x[f'{pref}_td']=(x.td>0).astype(float)-0.5   # anytime TD (only 'over' meaningful)
        elif s=='ints': x[f'{pref}_ints']=x.ints-0.75
        elif s=='carries': x[f'{pref}_carries']=x.carries-x.groupby('posteam').carries.transform('median')
    T=T.merge(x[['game_id','posteam']+[f'{pref}_{s}' for s in stats]],on=['game_id','posteam'],how='left')
put(qb,'QB',['pass_yds','cmp','pass_td','ints','rush_yds'])
put(rb,'RB1',['rush_yds','rec','rec_yds','td'])
put(rc[rc.k==1],'WR1',['rec','rec_yds','td']); put(rc[rc.k==2],'WR2',['rec','rec_yds','td']); put(rc[rc.k==3],'REC3',['rec','rec_yds'])
T['home']=(T.posteam==T.home_team).astype(int)
T['pts']=np.where(T.home==1,T.home_score,T.away_score); T['opp_pts']=np.where(T.home==1,T.away_score,T.home_score)
T['exp_margin']=np.where(T.home==1,T.spread_line,-T.spread_line); T['G_cover']=(T.pts-T.opp_pts)-T.exp_margin
T['G_team_total']=T.pts-(T.total_line+T.exp_margin)/2; T['G_over']=(T.pts+T.opp_pts)-T.total_line
T['fav']=(T.exp_margin>0).astype(int)
own=[c for c in T.columns if c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3') or c in ('G_cover','G_team_total')]
O=T[['game_id','posteam']+own].copy(); O.columns=['game_id','opp_team']+['OPP_'+c for c in own]
T['opp_team']=np.where(T.home==1,T.away_team,T.home_team)
W=T.merge(O,on=['game_id','opp_team'],how='inner')
W.to_parquet('nfl_wide.parquet'); print(W.shape)
