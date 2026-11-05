import pandas as pd, numpy as np, json
S='/tmp/claude-0/-home-claude/6b9b0bca-7d05-59fa-8d76-b5f28a8f31de/scratchpad'; M='/mnt/project-files/sports-analytics/models'
def frame(name, margin, total, line, tline, season):
    # line = expected HOME margin (positive = home favoured); tline = closing total
    d=pd.DataFrame({'margin':margin,'total':total,'line':line,'tl':tline,'season':season}).dropna()
    d=d[(d.line!=0)]
    fav_ats=np.where(d.line>0,d.margin-d.line,-(d.margin-d.line)); ou=d.total-d.tl
    d['F']=np.sign(fav_ats); d['O']=np.sign(ou); d=d[(d.F!=0)&(d.O!=0)].copy()
    d['ratio']=d.line.abs()/d.tl; d['sport']=name; return d
fr=[]
g=pd.read_csv(f'{S}/dl_week/games.csv'); g=g[g.result.notna()&(g.season<=2025)]
fr.append(frame('NFL',g.result,g.home_score+g.away_score,g.spread_line,g.total_line,g.season))
c=pd.read_parquet(f'{M}/team-strength/data/cfb_games.parquet'); print('cfb corr line/margin',np.corrcoef(c.dropna(subset=['line_close','margin']).line_close,c.dropna(subset=['line_close','margin']).margin)[0,1])
fr.append(frame('CFB',c.margin,c.total,c.line_close,c.total_close,c.season))
b=pd.read_parquet(f'{S}/hunt/cbb_clean.parquet'); fr.append(frame('CBB',b.margin,b.total,b.line,b.total_line,b.season))
n=pd.read_parquet(f'{S}/hunt/nba_sched.parquet'); fr.append(frame('NBA',n.margin,n.tot,-n.home_close_spread,n.close_over_under,n.season))
D=pd.concat(fr)
out={}
print('\nFavourite-cover x over joint rates (standard 2-leg -110 parlay breaks even at 27.6%)')
for sp,d in D.groupby('sport'):
    qs=d.ratio.quantile([0,.25,.5,.75,.9,1]).values
    print(f'\n{sp} n={len(d)}  P(fav cover)={(d.F>0).mean():.3f} P(over)={(d.O>0).mean():.3f}  corr(F,O)={np.corrcoef(d.F,d.O)[0,1]:+.3f}')
    d['bk']=pd.cut(d.ratio,np.unique(qs),include_lowest=True)
    rows=[]
    for bk,x in d.groupby('bk',observed=True):
        r=dict(bucket=str(bk),n=len(x),FO=((x.F>0)&(x.O>0)).mean(),FU=((x.F>0)&(x.O<0)).mean(),DO=((x.F<0)&(x.O>0)).mean(),DU=((x.F<0)&(x.O<0)).mean(),rho=np.corrcoef(x.F,x.O)[0,1])
        rows.append(r)
    R=pd.DataFrame(rows); print(R.round(3).to_string(index=False))
    # halves stability for top-ratio bucket
    top=d[d.ratio>=d.ratio.quantile(.9)]; med=top.season.median()
    for lab,x in [('early',top[top.season<=med]),('late',top[top.season>med])]:
        print(f'   top-10% ratio {lab}: FO {((x.F>0)&(x.O>0)).mean():.3f} DU {((x.F<0)&(x.O<0)).mean():.3f} n={len(x)}')
    out[sp]=R.to_dict('records')
D.to_parquet('game_pairs.parquet')
