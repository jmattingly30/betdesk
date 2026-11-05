"""Which model form closes the gap to the market? walk-forward by month (train on all earlier rows incl. 2025), graded Apr-Oct 2026."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings; warnings.filterwarnings('ignore')
from model26 import feats, lg
tour = sys.argv[1]
F = pd.read_parquet(f'../data/featx26h_{tour}.parquet'); F = F[F.y.notna()].copy()
F['ym'] = F.start.dt.strftime('%Y-%m')
X = feats(F)
lv = {'ITF': 0, 'Challenger': 1, 'Tour': 2}; X['lvl'] = F.cat.map(lv).fillna(1)
for c in ['elo', 'elos', 'gelo']: X[c + '_lvl'] = X[c] * X.lvl
ok = (F.oa > 1) & (F.ob > 1) & (1 / F.oa + 1 / F.ob < 1.15)
from te_vs_mkt import fair
close = pd.Series(np.where(ok, fair(F.oa.fillna(2).values, F.ob.fillna(2).values), np.nan), index=F.index)
BASE = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'gelo', 'gelo_x_new']
V = {'base': BASE, '+form': BASE + ['tr', 'tr180', 'tm', 'quick'], '+level': BASE + ['tr', 'tr180', 'elo_lvl', 'elos_lvl', 'gelo_lvl']}
res = {}
months = [m for m in sorted(F.ym.unique()) if '2026-04' <= m <= '2026-10']
for name, cols in list(V.items()) + [('boost', V['+level'] + ['lvl'])]:
    p = pd.Series(np.nan, index=F.index)
    for m in months:
        tr, te = (F.ym < m).values, (F.ym == m).values
        if name == 'boost':
            Xa = pd.concat([X.loc[tr, cols], -X.loc[tr, cols].assign(lvl=X.lvl[tr])]); ya = np.r_[F.y[tr], 1 - F.y[tr]]
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200).fit(Xa, ya)
            p[te] = (mdl.predict_proba(X.loc[te, cols])[:, 1] + 1 - mdl.predict_proba((-X.loc[te, cols]).assign(lvl=X.lvl[te]))[:, 1]) / 2
        else:
            Xa = pd.concat([X.loc[tr, cols], -X.loc[tr, cols]]); ya = np.r_[F.y[tr], 1 - F.y[tr]]
            mdl = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000).fit(Xa, ya)
            p[te] = mdl.predict_proba(X.loc[te, cols])[:, 1]
    res[name] = p
k = res['base'].notna() & close.notna()
for c in ['ITF', 'Challenger', 'Tour']:
    s = k & (F.cat == c)
    if s.sum() < 300: continue
    y = F.y[s]
    line = f'{c:10s} n {s.sum():5d} close pick {np.mean((close[s] > .5) == (y == 1)):.3f} LL {-np.mean(y*np.log(close[s].clip(.01,.99))+(1-y)*np.log(1-close[s].clip(.01,.99))):.4f} |'
    for n, p in res.items():
        q = p[s].clip(.01, .99)
        line += f' {n} {np.mean((q > .5) == (y == 1)):.3f}/{-np.mean(y*np.log(q)+(1-y)*np.log(1-q)):.4f}'
    print(line)
