import pandas as pd, numpy as np, sys; sys.path.insert(0,'.')
from lift import *
W=pd.read_parquet('nfl_wide.parquet')
cols=[c for c in W.columns if c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3','G','OPP') and c!='OPP_G_cover']
L=legs_from(W,cols,only_over=('_td',)); tr=(W.season<=2021).to_numpy()
Lt,Lv=L[tr].reset_index(drop=True),L[~tr].reset_index(drop=True)
# hand-built narrative stacks
H={'shootout':['G_over OVER','QB_pass_yds OVER','OPP_QB_pass_yds OVER','WR1_rec_yds OVER','OPP_WR1_rec_yds OVER'],
   'one-sided blowout (team wins big, runs clock)':['G_cover OVER','RB1_rush_yds OVER','OPP_QB_ints OVER','QB_pass_yds UNDER'],
   'dog garbage time (trailing team passes)':['G_cover UNDER','QB_cmp OVER','WR1_rec OVER','OPP_RB1_rush_yds OVER'],
   'defensive slog':['G_over UNDER','QB_pass_yds UNDER','OPP_QB_pass_yds UNDER','RB1_rec_yds UNDER','OPP_WR1_rec_yds UNDER'],
   'QB full stack own team':['QB_pass_yds OVER','QB_pass_td OVER','WR1_rec_yds OVER','WR2_rec_yds OVER','G_team_total OVER'],
   'QB up, RB down (pass-first script)':['QB_pass_yds OVER','RB1_rush_yds UNDER','RB1_rec OVER','G_cover UNDER'],
   'mobile-QB game, no turnovers':['QB_rush_yds OVER','QB_ints UNDER','OPP_RB1_td OVER'],
   'same receiver ladder':['WR1_rec OVER','WR1_rec_yds OVER','WR1_td OVER'],
   'RB workhorse ladder':['RB1_rush_yds OVER','RB1_td OVER','G_cover OVER','G_team_total OVER']}
rows=[]
for nm,legs in H.items():
    for k in range(2,len(legs)+1):
        tj,tl,tn=joint(Lt,legs[:k]); vj,vl,vn=joint(Lv,legs[:k])
        rows.append(dict(stack=nm,k=k,last_leg=legs[k-1],train_joint=tj,train_lift=tl,test_joint=vj,test_lift=vl,test_n=vn,fair_odds_test=('+%d'%round(100*(1/vj-1)) if vj and vj<0.5 else '%d'%round(-100*vj/(1-vj)))))
H=pd.DataFrame(rows); pd.set_option('display.width',250); print(H.round(3).to_string(index=False)); H.to_csv('nfl_stacks_named.csv',index=False)
# greedy search from every leg as seed (train), report test
G=[]
for seed in Lt.columns:
    G+= greedy(Lt,Lv,[seed],maxlegs=4,minhits=60)
G=pd.DataFrame(G).drop_duplicates('legs'); G=G[G.k>=3]
G['legs_sorted']=G.legs.map(lambda s:' + '.join(sorted(s.split(' + ')))); G=G.drop_duplicates('legs_sorted')
print('\nGreedy-found stacks (searched on 2016-21, tested 2022-25), top by TEST lift with test hits >= 40')
G=G[G.test_joint*G.test_n>=40].sort_values('test_lift',ascending=False)
print(G.head(25)[['legs','k','train_lift','test_lift','test_joint','test_n']].round(3).to_string(index=False)); G.to_csv('nfl_stacks_greedy.csv',index=False)
print('\nshrinkage: median test/train lift ratio',round((G.test_lift/G.train_lift).median(),3))
