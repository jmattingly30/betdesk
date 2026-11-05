import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'.')
from copula import joint_gauss
T=json.load(open('parlay_corr.json'))
def stack(W,legs,mask,sp,lab):
    idx={n:i for i,n in enumerate(T[sp]['legs'])}; R=np.array(T[sp]['R'])[np.ix_([idx[l] for l in legs],[idx[l] for l in legs])]
    out=[]
    for nm,m in [('all',np.ones(len(W),bool)),(lab,mask),('not',~mask)]:
        X=W.loc[m,legs].dropna(); H=(X<0); p=H.mean().values; emp=H.all(axis=1).mean(); cop=joint_gauss(p,R)
        out.append(f'{nm}: hit {emp:.3f} (copula {cop:.3f}, x{emp/cop:.2f}) n={len(X)}')
    print(f'{sp} UNDER {" + ".join(legs)}\n    '+' | '.join(out))
# NBA
W=pd.read_parquet('nba_wide.parquet')
stack(W,['P1_pts','P2_pts','P3_pts'],(W.sp.abs()>=10).to_numpy(),'NBA','spread 10+')
stack(W,['P1_pts','P2_pts','P3_pts'],(W.sp>=8).to_numpy(),'NBA','big underdog 8+')
stack(W,['P1_pts','P2_pts','P3_pts'],(W.sp<=-8).to_numpy(),'NBA','big favourite 8+') if False else None
stack(W,['P1_pts','P1_reb','P1_ast'],(W.sp<=-10).to_numpy(),'NBA','fav by 10+ (star sits 4th Q)')
stack(W,['P1_pts','P2_pts','OPP_P1_pts','OPP_P2_pts'],(W.tl<=W.tl.quantile(.2)).to_numpy(),'NBA','lowest-20% total')
stack(W,['P1_3pm','P2_3pm','P3_3pm'],(W.tl<=W.tl.quantile(.2)).to_numpy(),'NBA','lowest-20% total')
# NFL with weather/total
N=pd.read_parquet('nfl_wide.parquet'); g=pd.read_csv('/tmp/claude-0/-home-claude/6b9b0bca-7d05-59fa-8d76-b5f28a8f31de/scratchpad/dl_week/games.csv')[['game_id','wind','roof','temp']]
N=N.merge(g,on='game_id',how='left'); out=N.roof.isin(['outdoors','open'])
stack(N,['QB_pass_yds','WR1_rec_yds','OPP_QB_pass_yds'],(out&(N.wind>=12)).to_numpy(),'NFL','wind 12+ mph')
stack(N,['QB_pass_yds','WR1_rec_yds','WR2_rec_yds'],(out&(N.wind>=12)).to_numpy(),'NFL','wind 12+ mph')
stack(N,['QB_pass_yds','OPP_QB_pass_yds','G_over'],(N.total_line<=N.total_line.quantile(.2)).to_numpy(),'NFL','lowest-20% total')
stack(N,['QB_pass_yds','QB_cmp','WR1_rec'],(N.exp_margin>=7).to_numpy(),'NFL','fav by 7+ (runs late)')
stack(N,['RB1_rush_yds','QB_pass_yds'],(N.exp_margin<=-7).to_numpy(),'NFL','dog by 7+')
# MLB: cold games, and lineup vs high-K opposing starter
M=pd.read_parquet('mlb_wide.parquet'); s=pd.read_parquet('/mnt/project-files/sports-analytics/models/mlb-player/data/starter_games.parquet')[['gid','home','k_szn','temp']]
M['home_t']=(M.team==M.hometeam).astype(int); opp=s.rename(columns={'home':'sp_home'}); M=M.merge(opp[['gid','sp_home','k_szn']],left_on=['gid'],right_on=['gid'],how='left'); M=M[M.sp_home!=M.home_t]   # opposing starter's K rate
M=M.merge(s.drop_duplicates('gid')[['gid','temp']],on='gid',how='left')
stack(M,['B1_tb','B2_tb','B3_tb','B4_tb'],(M.k_szn>=M.k_szn.quantile(.8)).to_numpy(),'MLB','facing top-20% K starter')
stack(M,['B1_tb','B2_tb','B3_tb','G_over'],(M.temp<=55).to_numpy(),'MLB','temp <= 55F')
stack(M,['B1_hits','B2_hits','B3_hits'],(M.k_szn>=M.k_szn.quantile(.8)).to_numpy(),'MLB','facing top-20% K starter')
# NHL
H=pd.read_parquet('nhl_wide.parquet')
stack(H,['S1_pts','S2_pts','OPP_S1_pts','OPP_S2_pts'],(H.gf_e+H.ga_e<=(H.gf_e+H.ga_e).quantile(.2)).to_numpy(),'NHL','lowest-20% expected goals')
stack(H,['S1_sog','S2_sog','S3_sog'],(H.sf_e<=H.sf_e.quantile(.2)).to_numpy(),'NHL','lowest-20% expected shots')
# single-leg skew: share of games under the projection (the mean)
print('\nshare of games UNDER our projection (projection = mean; skewed stats sit under the mean more than half the time)')
for sp,W_ ,cols in [('NBA',W,['P1_pts','P3_pts','P5_pts','P1_3pm','P1_ast','P1_reb','P1_stl','P1_blk']),('NFL',N,['QB_pass_yds','WR1_rec_yds','WR2_rec_yds','REC3_rec_yds','RB1_rush_yds','QB_rush_yds']),('NHL',H,['S1_pts','S1_sog','S3_goals']),('MLB',M,['B3_tb','B3_hits','B7_tb'])]:
    print(' ',sp,{c:round(float((W_[c].dropna()<0).mean()),3) for c in cols})
