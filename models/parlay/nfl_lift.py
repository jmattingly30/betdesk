import pandas as pd, numpy as np, sys; sys.path.insert(0,'.')
from lift import *
W=pd.read_parquet('nfl_wide.parquet')
cols=[c for c in W.columns if c.split('_')[0] in ('QB','RB1','WR1','WR2','REC3','G','OPP')]
L=legs_from(W,cols,only_over=('_td',))
tr=(W.season<=2021).to_numpy(); te=~tr
Lt,Lv=L[tr].reset_index(drop=True),L[te].reset_index(drop=True)
P=pair_table(Lt,Lv); P['stable']=(np.sign(P.tr_lift-1)==np.sign(P.te_lift-1))
P.to_csv('nfl_pair_lifts.csv',index=False)
pd.set_option('display.width',250)
for k,g in P.groupby('kind'):
    g=g[g.stable].copy(); g['avg']=(g.tr_lift+g.te_lift)/2
    print(f'\n=== {k}: top lifts (both periods agree)'); print(g.sort_values('avg',ascending=False).head(10)[['a','b','tr_lift','te_lift','te_joint','te_n']].round(3).to_string(index=False))
    print('   most NEGATIVE (books may price as independent or positive):'); print(g.sort_values('avg').head(4)[['a','b','tr_lift','te_lift','te_joint']].round(3).to_string(index=False))
