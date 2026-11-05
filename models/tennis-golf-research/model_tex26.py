"""2026 live model on the tennisexplorer stream (all levels with scores): walk-forward by month, graded against the
match's own closing price (tex odds). Blend weights are fit on earlier months only."""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
from model26 import walk, walk_ym, lg, ll
from te_vs_mkt import fair
FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'gelo', 'gelo_x_new']
out = []
V = os.environ.get('V', '')
for tour in os.environ.get('TOURS', 'atp,wta').split(','):
    F = pd.read_parquet(f'../data/featx26{V}_{tour}.parquet')
    G, p, lr = walk_ym(F, FULL) if os.environ.get('WALK_YM') == '1' else walk(F, FULL)
    if os.environ.get('WALK_YM') == '1': G['mon'] = np.where(G.start.dt.year == 2026, G.mon, 0)
    G['p'] = p; G = G[G.p.notna()]
    ok = (G.oa > 1) & (G.ob > 1) & (1 / G.oa + 1 / G.ob < 1.15)
    G['close'] = np.where(ok, fair(G.oa.fillna(2).values, G.ob.fillna(2).values), np.nan)
    G.to_parquet(f'../data/predx26{V}_{tour}.parquet')
    for c, z in G[G.close.notna()].groupby('cat'):
        z = z.copy(); z['bl'] = np.nan
        for m in range(4, 11):
            tr, te = z[z.mon < m], z.mon == m
            if te.sum() == 0 or len(tr) < 300: continue
            X = np.c_[lg(tr.close), lg(tr.p)]
            l2 = LogisticRegression(C=10).fit(np.r_[X, -X], np.r_[tr.y, 1 - tr.y])
            z.loc[te, 'bl'] = l2.predict_proba(np.c_[lg(z.close[te]), lg(z.p[te])])[:, 1]
        k = z.bl.notna()
        out.append(dict(tour=tour, cat=c, n=int(k.sum()), model=ll(z.p[k], z.y[k]), close=ll(z.close[k], z.y[k]), blend_close=ll(z.bl[k], z.y[k])))
    print(tour, 'coef', dict(zip(FULL, lr.coef_[0].round(3))))
print(pd.DataFrame(out).round(4).to_string())
