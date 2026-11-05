import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
m=pd.read_parquet('/mnt/project-files/sports-analytics/models/soccer/data/matches.parquet')
m=m[(m.OddHome>1)&(m.OddDraw>1)&(m.OddAway>1)].dropna(subset=['FTHome','FTAway']).copy()
iv=1/m[['OddHome','OddDraw','OddAway']]; s=iv.sum(axis=1); m['pH'],m['pA']=iv.OddHome/s,iv.OddAway/s
m['fav_home']=m.pH>=m.pA; m['pF']=np.maximum(m.pH,m.pA)
m['F_win']=np.where(m.fav_home,m.FTResult=='H',m.FTResult=='A')
m['D_win']=np.where(m.fav_home,m.FTResult=='A',m.FTResult=='H'); m['draw']=m.FTResult=='D'
m['over25']=(m.FTHome+m.FTAway)>2.5; m['btts']=(m.FTHome>0)&(m.FTAway>0)
m['fav_cs']=np.where(m.fav_home,m.FTAway==0,m.FTHome==0)
m['F_2plus']=np.where(m.fav_home,m.FTHome>=2,m.FTAway>=2)
m['corn']=(m.HomeCorners+m.AwayCorners); m['cards']=(m.HomeYellow+m.AwayYellow+2*(m.HomeRed+m.AwayRed))
m['corn_over']=m.corn>m.corn.median(); m['cards_over']=m.cards>m.cards.median()
m['fav_corn_more']=np.where(m.fav_home,m.HomeCorners>m.AwayCorners,m.AwayCorners>m.HomeCorners)
m['HT_fav_lead']=np.where(m.fav_home,m.HTHome>m.HTAway,m.HTAway>m.HTHome)
m['bk']=pd.cut(m.pF,[0,.45,.55,.65,.75,1])
pairs=[('F_win','over25'),('F_win','btts'),('F_win','fav_cs'),('D_win','over25'),('draw','over25'),('draw','btts'),('F_win','corn_over'),('F_win','cards_over'),('D_win','cards_over'),
       ('over25','corn_over'),('over25','cards_over'),('btts','cards_over'),('F_win','fav_corn_more'),('F_2plus','fav_cs'),('HT_fav_lead','F_win'),('HT_fav_lead','over25'),('F_win','over25','btts'),('F_win','fav_cs','corn_over')]
rows=[]
for bk,x in m.groupby('bk',observed=True):
    for pr in pairs:
        A=x[list(pr)].dropna().astype(float); pj=A.prod(axis=1).mean(); pi=A.mean().prod(); rows.append(dict(fav_prob=str(bk),legs=' + '.join(pr),n=len(A),joint=pj,lift=pj/pi))
R=pd.DataFrame(rows); P=R.pivot_table(index='legs',columns='fav_prob',values='lift'); pd.set_option('display.width',250); print(P.round(3).to_string())
J=R.pivot_table(index='legs',columns='fav_prob',values='joint'); print('\njoint hit rates'); print(J.round(3).to_string())
# stability check: halves
m['half']=m.MatchDate.astype(str)<'2016'
for pr in [('F_win','over25'),('D_win','over25'),('draw','btts'),('F_win','fav_cs')]:
    for h,x in m[m.pF>=.65].groupby('half'):
        A=x[list(pr)].astype(float); print(pr,'fav>=65% pre-2016' if h else 'fav>=65% 2016+', round(A.prod(axis=1).mean()/A.mean().prod(),3))
