import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6b9b0bca-7d05-59fa-8d76-b5f28a8f31de/scratchpad'; M='/mnt/project-files/sports-analytics/models'
DATA={}
def DATA_add(sp,W,legs,C,half): DATA[sp]=(W.reset_index(drop=True),legs,{k:pd.Series(np.asarray(v.fillna(False) if hasattr(v,'fillna') else v,dtype=bool)) for k,v in C.items()},np.asarray(half))
out=[]
# ---------- NFL
N=pd.read_parquet('nfl_wide.parquet'); g=pd.read_csv(f'{S}/dl_week/games.csv')
N=N.merge(g[['game_id','wind','temp','roof','surface','div_game','weekday','gametime','home_rest','away_rest','week']],on='game_id',how='left')
outd=N.roof.isin(['outdoors','open']); rest=np.where(N.home==1,N.home_rest,N.away_rest); orest=np.where(N.home==1,N.away_rest,N.home_rest)
hr=pd.to_numeric(N.gametime.str[:2],errors='coerce')
C={'wind 10-14':outd&N.wind.between(10,14),'wind 15+':outd&(N.wind>=15),'cold <=35F':outd&(N.temp<=35),'hot 85F+':outd&(N.temp>=85),'dome/closed':~outd,
   'grass':N.surface.astype(str).str.contains('grass'),'divisional':N.div_game==1,'Thursday':N.weekday=='Thursday','night 8pm+':hr>=20,
   'short rest vs normal':(rest<=5)&(orest>=6),'off bye':rest>=13,'weeks 1-3':N.week<=3,'weeks 15+':N.week>=15,
   'total <=40':N.total_line<=40,'total 50+':N.total_line>=50,'fav 7+':N.exp_margin>=7,'dog 7+':N.exp_margin<=-7,'home':N.home==1}
legs=[c for c in N.columns if c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3','G','OPP') and not c.endswith('_td') and c!='OPP_G_cover']
DATA_add('NFL',N,legs,{k:pd.Series(v) for k,v in C.items()},(N.season>=2021).to_numpy())
# ---------- NBA
B=pd.read_parquet('nba_wide.parquet'); tp=pd.read_parquet(f'{M}/nba-player/data/test_predictions.parquet')[['game_id','team','rest','is_home']].drop_duplicates(['game_id','team'])
B=B.merge(tp,on=['game_id','team'],how='left'); B['month']=pd.to_datetime(B.date).dt.month
C={'dog 10+':B.sp<=-10,'dog 6-9.5':B.sp.between(-9.5,-6),'fav 6-9.5':B.sp.between(6,9.5),'fav 10+':B.sp>=10,'total top 20%':B.tl>=B.tl.quantile(.8),'total bottom 20%':B.tl<=B.tl.quantile(.2),
   'back-to-back':B.rest==1,'3+ days rest':B.rest>=3,'home':B.is_home==1,'Oct-Nov':B.month.isin([10,11]),'Mar-Apr':B.month.isin([3,4])}
legs=[c for c in B.columns if (c[:1]=='P' or c.startswith('G_') or c.startswith('OPP_')) and not c.endswith('_min')]
DATA_add('NBA',B,legs,C,(B.date>=B.date.median()).to_numpy())
# ---------- MLB
Mw=pd.read_parquet('mlb_wide.parquet'); gm=pd.read_parquet(f'{M}/team-strength/data/mlb_games_model.parquet')[['gid','temp','wind_out','wind_in','day_game','pf','ump_factor']]
Mw=Mw.merge(gm,on='gid',how='left'); st=pd.read_parquet(f'{M}/mlb-player/data/starter_games.parquet')[['gid','home','k_szn','days_rest']]
Mw['home_t']=(Mw.team==Mw.hometeam).astype(int); o=st.rename(columns={'home':'sph','k_szn':'opp_sp_k','days_rest':'opp_sp_rest'}); Mw=Mw.merge(o,on='gid',how='left'); Mw=Mw[Mw.sph!=Mw.home_t]
C={'temp <=55':Mw.temp<=55,'temp 85+':Mw.temp>=85,'wind out 10+':Mw.wind_out>=10,'wind in 10+':Mw.wind_in>=10,'day game':Mw.day_game==1,'hitter park (pf top 20%)':Mw.pf>=Mw.pf.quantile(.8),
   'pitcher park (pf bottom 20%)':Mw.pf<=Mw.pf.quantile(.2),'big-zone ump (top 20%)':Mw.ump_factor<=Mw.ump_factor.quantile(.2),'small-zone ump':Mw.ump_factor>=Mw.ump_factor.quantile(.8),
   'vs top-20% K starter':Mw.opp_sp_k>=Mw.opp_sp_k.quantile(.8),'vs bottom-20% K starter':Mw.opp_sp_k<=Mw.opp_sp_k.quantile(.2),'total 10+':Mw.total_close>=10,'total <=7.5':Mw.total_close<=7.5,'home':Mw.home_t==1}
legs=[c for c in Mw.columns if (c[:1]=='B' or c.startswith('SP_') or c.startswith('G_') or c.startswith('OPP_')) and c not in ('OPP_SP_k',)]
legs=[c for c in legs if pd.api.types.is_numeric_dtype(Mw[c])]
DATA_add('MLB',Mw.reset_index(drop=True),legs,{k:v.reset_index(drop=True) for k,v in C.items()},(Mw.season>=2023).to_numpy())
# ---------- NHL
H=pd.read_parquet('nhl_wide.parquet'); pg=pd.read_parquet(f'{M}/nhl-player/data/player_games.parquet')[['game_id','team','rest','home','o_rest']].drop_duplicates(['game_id','team'])
H=H.merge(pg,on=['game_id','team'],how='left'); te=H.gf_e+H.ga_e
C={'back-to-back':H.rest<=1,'opp back-to-back':H.o_rest<=1,'home':H.home==1,'high expected goals (top 20%)':te>=te.quantile(.8),'low expected goals (bottom 20%)':te<=te.quantile(.2),
   'big favourite (gf-ga exp top 15%)':(H.gf_e-H.ga_e)>=(H.gf_e-H.ga_e).quantile(.85),'big underdog':(H.gf_e-H.ga_e)<=(H.gf_e-H.ga_e).quantile(.15)}
legs=[c for c in H.columns if (c[:1]=='S' or c.startswith('G_') or c.startswith('GOALIE') or c.startswith('OPP_')) and pd.api.types.is_numeric_dtype(H[c]) and c not in ('season',)]
DATA_add('NHL',H,legs,C,(H.season>H.season.median()).to_numpy())
