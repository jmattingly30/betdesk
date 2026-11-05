import pandas as pd, numpy as np, sys; sys.path.insert(0,'.')
from lift import legs_from, joint
from copula import tetra, joint_gauss
W=pd.read_parquet('nfl_wide.parquet')
cols=[c for c in W.columns if c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3','G','OPP') and c!='OPP_G_cover']
L=legs_from(W,cols,only_over=('_td',)); tr=(W.season<=2021).to_numpy(); Lt,Lv=L[tr].reset_index(drop=True),L[~tr].reset_index(drop=True)
G=pd.read_csv('nfl_stacks_greedy.csv'); N=pd.read_csv('nfl_stacks_named.csv')
stacks=[s.split(' + ') for s in G.legs.head(60)]
H={'shootout':['G_over OVER','QB_pass_yds OVER','OPP_QB_pass_yds OVER','WR1_rec_yds OVER','OPP_WR1_rec_yds OVER'],
   'defensive slog':['G_over UNDER','QB_pass_yds UNDER','OPP_QB_pass_yds UNDER','RB1_rec_yds UNDER','OPP_WR1_rec_yds UNDER'],
   'QB full stack':['QB_pass_yds OVER','QB_pass_td OVER','WR1_rec_yds OVER','WR2_rec_yds OVER','G_team_total OVER'],
   'RB workhorse':['RB1_rush_yds OVER','RB1_td OVER','G_cover OVER','G_team_total OVER']}
for v in H.values():
    for k in range(3,len(v)+1): stacks.append(v[:k])
rows=[]
for legs in stacks:
    A=Lt[legs].dropna(); p=A.mean().values
    R=np.eye(len(legs))
    for i in range(len(legs)):
        for j in range(i+1,len(legs)):
            R[i,j]=R[j,i]=tetra(p[i],p[j],(A.iloc[:,i]*A.iloc[:,j]).mean())
    B=Lv[legs].dropna(); pv=B.mean().values
    emp=B.prod(1).mean(); cop=joint_gauss(pv,R); ind=pv.prod()
    rows.append(dict(legs=' + '.join(legs),k=len(legs),test_emp=emp,pairwise_copula=cop,independent=ind,emp_over_copula=emp/cop,n=len(B)))
R=pd.DataFrame(rows).drop_duplicates('legs')
pd.set_option('display.width',250); print(R.groupby('k')[['emp_over_copula']].describe().round(3))
print(R.sort_values('emp_over_copula',ascending=False).head(15).round(3).to_string(index=False))
print('\nmean ratio by k: emp/indep',R.groupby('k').apply(lambda x:(x.test_emp/x.independent).mean()).round(3).to_dict())
R.to_csv('copula_vs_empirical.csv',index=False)
